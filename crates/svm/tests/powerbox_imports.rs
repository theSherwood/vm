//! **Phase 2 — name-based dynamic import binding.** A hand-written IR module (no C, no `svm-llvm`)
//! declares arbitrarily-named capability imports and is bound to a host-provided registry **by name**
//! (wasm-style linking), then run through the differential wrapper. This exercises decision #2: the
//! host offers capabilities under arbitrary names/interfaces/counts, `instantiate_with_imports`
//! matches each `call.import "<name>"` to a [`svm_run::HostCap`] and binds slot `i` ↔ import `i` at
//! instantiation — the fixed §3e powerbox is just one preset over this.
//!
//! Gated `#![cfg(unix)]` like the other JIT differential suites.
#![cfg(unix)]

use svm_run::{
    instantiate, instantiate_with_imports, Backend, HostCap, Imports, Outcome, RunConfig, Value,
};

/// Two **arbitrary-named** host-function imports — `add_seven` and `triple` — each its own handle but
/// the same nominal interface (`HOST_FN`), distinguished object-capability-style by which slot the
/// call dispatches through (the handle operands are vestigial dummies; the manifest binds slot `i` ↔
/// import `i` at instantiation). The paramless `_start` threads `5 → add_seven → triple`, and
/// returns `(5+7)*3 = 36`.
const SRC: &str = "\
memory 15
export \"_start\" 0
func () -> (i64) {
block0():
  v0 = i32.const 0
  v1 = i64.const 5
  v2 = call.import \"add_seven\" (i64) -> (i64) v0 (v1)
  v3 = i32.const 0
  v4 = call.import \"triple\" (i64) -> (i64) v3 (v2)
  return v4
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

    // The host offers each name a host-defined capability with arbitrary semantics. Distinct grants ⇒
    // distinct slots, so the guest reaches the right closure purely by which slot it dispatches
    // through (slot `i` ↔ import `i`, bound at instantiation).
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
    let instance = instantiate_with_imports(module, imports).expect("instantiate by name");
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
/// the guest stores 41 through the `kv.set` slot, loads through the `kv.get` slot, adds 1, and
/// returns 42. A per-name (non-shared) grant would read a fresh cell (0) and return 1 — so the value
/// itself proves the instance is shared. All three backends.
const MODULE_SRC: &str = "\
memory 15
export \"_start\" 0
func () -> (i64) {
block0():
  v0 = i32.const 0
  v1 = i64.const 41
  v2 = call.import \"kv.set\" (i64) -> (i64) v0 (v1)
  v3 = i32.const 0
  v4 = i64.const 0
  v5 = call.import \"kv.get\" (i64) -> (i64) v3 (v4)
  v6 = i64.const 1
  v7 = i64.add v5 v6
  return v7
}
";

#[test]
fn module_bindings_share_one_instance() {
    for backend in [Backend::TreeWalk, Backend::Bytecode, Backend::Jit] {
        let module = svm_text::parse_module(MODULE_SRC).expect("parse");
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
        let instance = instantiate_with_imports(module, imports).expect("instantiate");
        let run = instance
            .run(backend, &RunConfig::default())
            .unwrap_or_else(|e| panic!("run on {backend:?}: {e}"));
        assert_eq!(
            run.outcome,
            Outcome::Returned(vec![Value::I64(42)]),
            "{backend:?}: set(41) through one field, get() through the other — shared instance"
        );
    }
}

/// An imported name with no entry in the registry is fail-closed at instantiation — wasm-style
/// "unsatisfied import", surfaced before anything runs.
#[test]
fn unbound_name_fails_closed() {
    let module = svm_text::parse_module(SRC).expect("parse");
    // Provide only one of the two imports.
    let imports = Imports::new().provide(
        "add_seven",
        HostCap::host_fn(0, || Box::new(|_op, args, _mem| Ok(vec![args[0]]))),
    );
    let err = match instantiate_with_imports(module, imports) {
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
export \"_start\" 0
func () -> (i32) {
block0():
  v0 = i32.const 0
  v1 = i64.const 16384
  v2 = i64.const 16
  v3 = call.import \"write\" (i64, i64) -> (i64) v0 (v1, v2)
  v4 = i32.const 0
  return v4
}
";
    let module = svm_text::parse_module(src).expect("parse");
    let imports = Imports::new().provide("write", HostCap::stdout());
    let instance = instantiate_with_imports(module, imports).expect("instantiate");
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

/// The guest resolves the name `"write"` (which it imported) to its handle **at runtime**, then
/// threads that resolved handle through the `write` call site (whose slot dispatch does the work).
/// Proves resolve returns a live handle for a name-bound grant, on the tree-walker, bytecode
/// engine, and JIT.
const RESOLVE_SRC: &str = "\
memory 15
data ro 16384 \"via resolve\\n\"
data ro 17000 \"write\"
export \"_start\" 0
func () -> (i32) {
block0():
  v0 = i64.const 17000
  v1 = i64.const 5
  v2 = cap.self.resolve v0 v1
  v3 = i64.const 16384
  v4 = i64.const 12
  v5 = call.import \"write\" (i64, i64) -> (i64) v2 (v3, v4)
  v6 = i32.const 0
  return v6
}
";

#[test]
fn resolve_capability_by_name_at_runtime() {
    for backend in [Backend::TreeWalk, Backend::Bytecode, Backend::Jit] {
        let module = svm_text::parse_module(RESOLVE_SRC).expect("parse");
        let imports = Imports::new().provide("write", HostCap::stdout());
        let instance = instantiate_with_imports(module, imports).expect("instantiate");
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
export \"_start\" 0
func () -> (i32) {
block0():
  v0 = i64.const 17000
  v1 = i64.const 4
  v2 = cap.self.resolve v0 v1
  return v2
}
";

#[test]
fn resolve_unknown_name_is_fail_closed() {
    let module = svm_text::parse_module(RESOLVE_BOGUS_SRC).expect("parse");
    let instance = instantiate_with_imports(module, Imports::new()).expect("instantiate");
    let run = instance.call("_start", &[]).expect("run");
    assert_eq!(
        run.outcome,
        Outcome::Returned(vec![Value::I32(-22)]),
        "an unknown capability name resolves to -EINVAL"
    );
}

/// The fixed §3e powerbox registers **canonical** names (no named imports needed): the guest
/// resolves `"exit"` at runtime and invokes the resolved handle through the generic `cap.call`
/// seam — proving canonical registration and that resolve returns the real, working capability.
const CANON_SRC: &str = "\
memory 15
data ro 16384 \"exit\"
export \"_start\" 0
func () -> (i32) {
block0():
  v0 = i64.const 16384
  v1 = i64.const 4
  v2 = cap.self.resolve v0 v1
  v3 = i32.const 7
  cap.call 1 0 (i32) -> () v2(v3)
  unreachable
}
";

#[test]
fn canonical_powerbox_names_resolve_to_working_handles() {
    let module = svm_text::parse_module(CANON_SRC).expect("parse");
    let instance = instantiate(module).expect("instantiate");
    let run = instance.call("_start", &[]).expect("run");
    assert_eq!(
        run.outcome,
        Outcome::Exited(7),
        "resolve(\"exit\") returns the fixed powerbox's live Exit handle"
    );
}

// --- F9: capability labels (the guest's `cap.self.label`, reverse of resolve) ----------------------
//
// `cap.self.label <handle> <buf_ptr> <buf_cap> -> i32` writes the handle's human-readable label into
// the window and returns its full length (0 if unlabeled). A guest enumerating its handles
// (`cap.self.count`/`get`) can name each one — for diagnostics / discovery. Cosmetic and
// authority-neutral; routes through the generic seam (op 3 over `CAP_SELF_TYPE_ID`), so all three
// backends agree.

/// The guest resolves its `write` import's handle by name, reads its label (`"write"`, the import
/// name) into a scratch buffer, and streams it back out — proving `cap.self.label` returns the
/// registered name and the byte-write lands, on the tree-walker, bytecode engine, and JIT.
const LABEL_SRC: &str = "\
memory 15
data ro 16384 \"write\"
export \"_start\" 0
func () -> (i32) {
block0():
  v0 = i64.const 16384
  v1 = i64.const 5
  v2 = cap.self.resolve v0 v1
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
        let imports = Imports::new().provide("write", HostCap::stdout());
        let instance = instantiate_with_imports(module, imports).expect("instantiate");
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
data ro 16384 \"write\"
export \"_start\" 0
func () -> (i32) {
block0():
  v0 = i64.const 16384
  v1 = i64.const 5
  v2 = cap.self.resolve v0 v1
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
    let imports = Imports::new().provide("write", HostCap::stdout());
    let instance = instantiate_with_imports(module, imports).expect("instantiate");
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
