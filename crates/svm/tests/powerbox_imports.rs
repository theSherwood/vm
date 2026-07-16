//! **Phase 2 — name-based dynamic import binding.** A hand-written IR module (no C, no `svm-llvm`)
//! declares arbitrarily-named capability imports and is bound to a host-provided registry **by name**
//! (wasm-style linking), then run through the differential wrapper. This exercises decision #2: the
//! host offers capabilities under arbitrary names/interfaces/counts, `instantiate_with_imports`
//! matches each `call.import "<name>"` to a [`svm_run::HostCap`], and the powerbox stash delivers the
//! granted handles in import order — the fixed §3e powerbox is just one preset over this.
//!
//! Gated `#![cfg(unix)]` like the other JIT differential suites.
#![cfg(unix)]

use svm_run::{
    instantiate, instantiate_with_imports, Backend, HostCap, Imports, Outcome, RunConfig, Value,
};

/// Two **arbitrary-named** host-function imports — `add_seven` and `triple` — each its own handle but
/// the same nominal interface (`HOST_FN`), distinguished object-capability-style by which handle the
/// guest holds. The entry loads import 0's handle from stash slot 0 and import 1's from slot 1 (the
/// `slot i ↔ import i` contract), threads `5 → add_seven → triple`, and returns `(5+7)*3 = 36`.
const SRC: &str = "\
memory 15
export \"entry\" 0
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const 0
  v2 = i32.load v1
  v3 = i64.const 5
  v4 = call.import \"add_seven\" (i64) -> (i64) v2 (v3)
  v5 = i64.const 4
  v6 = i32.load v5
  v7 = call.import \"triple\" (i64) -> (i64) v6 (v4)
  return v7
}
";

#[test]
fn arbitrary_named_host_capabilities_bind_and_run() {
    let module = svm_text::parse_module(SRC).expect("frontend IR parses");
    // The text parser builds the import table from `call.import` names, in first-occurrence order.
    assert_eq!(
        module
            .imports
            .iter()
            .map(|i| i.name.as_str())
            .collect::<Vec<_>>(),
        ["add_seven", "triple"],
        "two arbitrarily-named imports, in declaration order"
    );

    // Prepend the powerbox `_start` for exactly the 2 granted handles (not the fixed 3–8 powerbox).
    let with_start = svm_ir::synth_powerbox_start_for_imports(module, 0, false)
        .expect("prepend _start (2 handles)");

    // The host offers each name a host-defined capability with arbitrary semantics. Distinct grants ⇒
    // distinct handles, so the guest reaches the right closure purely by which handle it holds.
    let imports = Imports::new()
        .provide(
            "add_seven",
            HostCap::host_fn(0, || Box::new(|_op, args, _mem| Ok(vec![args[0] + 7]))),
        )
        .provide(
            "triple",
            HostCap::host_fn(0, || Box::new(|_op, args, _mem| Ok(vec![args[0] * 3]))),
        );

    // Bound by name; runs interp + JIT under identical capabilities (interp == jit asserted inside).
    let instance = instantiate_with_imports(with_start, imports).expect("instantiate by name");
    let run = instance.call("_start", &[]).expect("run via the wrapper");
    assert_eq!(
        run.outcome,
        Outcome::Returned(vec![Value::I64(36)]),
        "(5 + 7) * 3, threaded through two name-bound host capabilities"
    );
}

/// The **wasm two-level import**: a module of named bindings backed by **one shared instance**
/// ([`Imports::provide_module`]). The guest imports `kv.set` and `kv.get` — the `"{module}.{name}"`
/// encoding `svm-wasm` emits for `(import "kv" "set" …)` — and wasm semantics require both fields to
/// reach the *same* provider instance. The provider is a stateful cell (op 0 stores, op 1 loads):
/// the guest stores 41 through the `kv.set` handle (stash slot 0), loads through the `kv.get` handle
/// (slot 1), adds the handle difference (must be 0 — one grant, one handle) and 1, and returns 42.
/// A per-name (non-shared) grant would read a fresh cell (0) and return 1 — so the value itself
/// proves the instance is shared. All three backends.
const MODULE_SRC: &str = "\
memory 15
export \"entry\" 0
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const 0
  v2 = i32.load v1
  v3 = i64.const 41
  v4 = call.import \"kv.set\" (i64) -> (i64) v2 (v3)
  v5 = i64.const 4
  v6 = i32.load v5
  v7 = i64.const 0
  v8 = call.import \"kv.get\" (i64) -> (i64) v6 (v7)
  v9 = i32.sub v2 v6
  v10 = i64.extend_i32_s v9
  v11 = i64.add v8 v10
  v12 = i64.const 1
  v13 = i64.add v11 v12
  return v13
}
";

#[test]
fn module_bindings_share_one_instance() {
    for backend in [Backend::TreeWalk, Backend::Bytecode, Backend::Jit] {
        let module = svm_text::parse_module(MODULE_SRC).expect("parse");
        let with_start = svm_ir::synth_powerbox_start_for_imports(module, 0, false).expect("synth");
        // One provider, two fields: `set` is op 0, `get` is op 1 — one closure, one cell.
        let imports = Imports::new().provide_module(
            "kv",
            HostCap::host_fn(0, || {
                let mut cell = 0i64;
                Box::new(move |op, args, _mem| {
                    Ok(vec![match op {
                        0 => {
                            cell = args[0];
                            0
                        }
                        _ => cell,
                    }])
                })
            }),
            &[("set", 0), ("get", 1)],
        );
        let instance = instantiate_with_imports(with_start, imports).expect("instantiate");
        let run = instance
            .run(backend, &RunConfig::default())
            .unwrap_or_else(|e| panic!("run on {backend:?}: {e}"));
        assert_eq!(
            run.outcome,
            Outcome::Returned(vec![Value::I64(42)]),
            "{backend:?}: set(41) through one field, get() through the other — shared instance, \
             identical handles"
        );
    }
}

/// An imported name with no entry in the registry is fail-closed at instantiation — wasm-style
/// "unsatisfied import", surfaced before anything runs.
#[test]
fn unbound_name_fails_closed() {
    let module = svm_text::parse_module(SRC).expect("parse");
    let with_start = svm_ir::synth_powerbox_start_for_imports(module, 0, false).expect("synth");
    // Provide only one of the two imports.
    let imports = Imports::new().provide(
        "add_seven",
        HostCap::host_fn(0, || Box::new(|_op, args, _mem| Ok(vec![args[0]]))),
    );
    let err = match instantiate_with_imports(with_start, imports) {
        Ok(_) => panic!("instantiation must fail closed when `triple` is unbound"),
        Err(e) => e,
    };
    assert!(
        err.contains("triple"),
        "the error must name the unbound import, got: {err}"
    );
}

/// The standard powerbox names work through the registry too — proving the fixed §3e powerbox is just
/// one preset over the same name-based mechanism. Here a single `write` import emits a RO string.
#[test]
fn standard_names_are_just_a_preset() {
    let src = "\
memory 15
data ro 16384 \"hi via registry\\n\"
export \"entry\" 0
func (i64) -> (i32) {
block0(v0: i64):
  v1 = i64.const 0
  v2 = i32.load v1
  v3 = i64.const 16384
  v4 = i64.const 16
  v5 = call.import \"write\" (i64, i64) -> (i64) v2 (v3, v4)
  v6 = i32.const 0
  return v6
}
";
    let module = svm_text::parse_module(src).expect("parse");
    let with_start =
        svm_ir::synth_powerbox_start_for_imports(module, 0, false).expect("synth (1 handle)");
    let imports = Imports::new().provide("write", HostCap::stdout());
    let instance = instantiate_with_imports(with_start, imports).expect("instantiate");
    let run = instance.call("_start", &[]).expect("run");
    assert_eq!(run.stdout, b"hi via registry\n");
    assert_eq!(run.outcome, Outcome::Returned(vec![Value::I32(0)]));
}

// --- F7: runtime name → handle resolution (the guest's `cap.self.resolve`) ------------------------
//
// `cap.self.resolve <name_ptr> <name_len> -> i32` resolves a capability **name** to the handle it was
// granted (`-errno` on miss). It confers no authority — it only re-finds a handle the guest already
// holds — and routes through the generic capability seam (op 2 over the reserved `CAP_SELF_TYPE_ID`),
// so it works identically on the tree-walker, bytecode engine, and JIT.

/// The guest resolves the name `"write"` (which it imported) to its handle **at runtime** — never
/// reading the stash slot — then uses that resolved handle to emit a string. Proves the resolved
/// handle is the real, working capability, on the tree-walker, bytecode engine, and JIT.
const RESOLVE_SRC: &str = "\
memory 15
data ro 16384 \"via resolve\\n\"
data ro 17000 \"write\"
export \"entry\" 0
func (i64) -> (i32) {
block0(v0: i64):
  v1 = i64.const 17000
  v2 = i64.const 5
  v3 = cap.self.resolve v1 v2
  v4 = i64.const 16384
  v5 = i64.const 12
  v6 = call.import \"write\" (i64, i64) -> (i64) v3 (v4, v5)
  v7 = i32.const 0
  return v7
}
";

#[test]
fn resolve_capability_by_name_at_runtime() {
    for backend in [Backend::TreeWalk, Backend::Bytecode, Backend::Jit] {
        let module = svm_text::parse_module(RESOLVE_SRC).expect("parse");
        let with_start = svm_ir::synth_powerbox_start_for_imports(module, 0, false).expect("synth");
        let imports = Imports::new().provide("write", HostCap::stdout());
        let instance = instantiate_with_imports(with_start, imports).expect("instantiate");
        let run = instance
            .run(backend, &RunConfig::default())
            .unwrap_or_else(|e| panic!("run on {backend:?}: {e}"));
        assert_eq!(
            run.stdout, b"via resolve\n",
            "the name-resolved write handle works on {backend:?}"
        );
        assert_eq!(run.outcome, Outcome::Returned(vec![Value::I32(0)]));
    }
}

/// An unknown name resolves to `-EINVAL` (-22) — fail-closed, the new untrusted-name surface never
/// traps or invents a handle. (The directory is empty here; a bad name fails regardless.)
const RESOLVE_BOGUS_SRC: &str = "\
memory 15
data ro 17000 \"nope\"
export \"entry\" 0
func (i64) -> (i32) {
block0(v0: i64):
  v1 = i64.const 17000
  v2 = i64.const 4
  v3 = cap.self.resolve v1 v2
  return v3
}
";

#[test]
fn resolve_unknown_name_is_fail_closed() {
    let module = svm_text::parse_module(RESOLVE_BOGUS_SRC).expect("parse");
    let with_start =
        svm_ir::synth_powerbox_start_for_imports(module, 0, false).expect("synth (0 handles)");
    let instance = instantiate_with_imports(with_start, Imports::new()).expect("instantiate");
    let run = instance.call("_start", &[]).expect("run");
    assert_eq!(
        run.outcome,
        Outcome::Returned(vec![Value::I32(-22)]),
        "an unknown capability name resolves to -EINVAL"
    );
}

/// The fixed §3e powerbox registers **canonical** names (no named imports): the guest resolves `"exit"`
/// and checks it equals the handle `_start` stashed at slot 2 — proving canonical registration and that
/// resolve returns the very handle the stash holds.
const CANON_SRC: &str = "\
memory 15
data ro 16384 \"exit\"
export \"entry\" 0
func (i64) -> (i32) {
block0(v0: i64):
  v1 = i64.const 16384
  v2 = i64.const 4
  v3 = cap.self.resolve v1 v2
  v4 = i64.const 8
  v5 = i32.load v4
  v6 = i32.sub v3 v5
  return v6
}
";

#[test]
fn canonical_powerbox_names_resolve_to_stash_handles() {
    let module = svm_text::parse_module(CANON_SRC).expect("parse");
    let with_start = svm_ir::synth_powerbox_start(module, 0, 3, false).expect("synth (3 handles)");
    let instance = instantiate(with_start).expect("instantiate");
    let run = instance.call("_start", &[]).expect("run");
    assert_eq!(
        run.outcome,
        Outcome::Returned(vec![Value::I32(0)]),
        "resolve(\"exit\") returns the same handle _start stashed at slot 2"
    );
}

// --- F9: capability labels (the guest's `cap.self.label`, reverse of resolve) ----------------------
//
// `cap.self.label <handle> <buf_ptr> <buf_cap> -> i32` writes the handle's human-readable label into
// the window and returns its full length (0 if unlabeled). A guest enumerating its handles
// (`cap.self.count`/`get`) can name each one — for diagnostics / discovery. Cosmetic and
// authority-neutral; routes through the generic seam (op 3 over `CAP_SELF_TYPE_ID`), so all three
// backends agree.

/// The guest reads the label of its `write` handle (`"write"`, its import name) into a scratch buffer
/// and streams it back out — proving `cap.self.label` returns the registered name and the byte-write
/// lands, on the tree-walker, bytecode engine, and JIT.
const LABEL_SRC: &str = "\
memory 15
export \"entry\" 0
func (i64) -> (i32) {
block0(v0: i64):
  v1 = i64.const 0
  v2 = i32.load v1
  v3 = i64.const 2048
  v4 = i64.const 64
  v5 = cap.self.label v2 v3 v4
  v6 = i64.extend_i32_s v5
  v7 = call.import \"write\" (i64, i64) -> (i64) v2 (v3, v6)
  v8 = i32.const 0
  return v8
}
";

#[test]
fn label_a_handle_to_its_registered_name() {
    for backend in [Backend::TreeWalk, Backend::Bytecode, Backend::Jit] {
        let module = svm_text::parse_module(LABEL_SRC).expect("parse");
        let with_start = svm_ir::synth_powerbox_start_for_imports(module, 0, false).expect("synth");
        let imports = Imports::new().provide("write", HostCap::stdout());
        let instance = instantiate_with_imports(with_start, imports).expect("instantiate");
        let run = instance
            .run(backend, &RunConfig::default())
            .unwrap_or_else(|e| panic!("run on {backend:?}: {e}"));
        assert_eq!(
            run.stdout, b"write",
            "cap.self.label wrote the handle's registered name on {backend:?}"
        );
    }
}

/// When the label doesn't fit, `cap.self.label` writes nothing and returns the **full** length, so the
/// guest can retry with a buffer that size. Here `buf_cap = 2 < len(\"write\") = 5`, so the entry
/// returns 5 (and stdout stays empty — nothing was written).
const LABEL_SMALL_SRC: &str = "\
memory 15
export \"entry\" 0
func (i64) -> (i32) {
block0(v0: i64):
  v1 = i64.const 0
  v2 = i32.load v1
  v3 = i64.const 2048
  v4 = i64.const 2
  v5 = cap.self.label v2 v3 v4
  v6 = i64.const 0
  v7 = call.import \"write\" (i64, i64) -> (i64) v2 (v3, v6)
  return v5
}
";

#[test]
fn label_too_small_buffer_returns_full_length() {
    let module = svm_text::parse_module(LABEL_SMALL_SRC).expect("parse");
    let with_start = svm_ir::synth_powerbox_start_for_imports(module, 0, false).expect("synth");
    let imports = Imports::new().provide("write", HostCap::stdout());
    let instance = instantiate_with_imports(with_start, imports).expect("instantiate");
    let run = instance.call("_start", &[]).expect("run");
    assert_eq!(
        run.outcome,
        Outcome::Returned(vec![Value::I32(5)]),
        "an undersized buffer yields the full label length (\"write\" = 5), writing nothing"
    );
    assert!(
        run.stdout.is_empty(),
        "nothing is written when it doesn't fit"
    );
}
