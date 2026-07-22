//! IMPORTS.md **phase 1** — executable `call.import` over the powerbox prefix, with **no module
//! rewriting**. The load-bearing assertions:
//!
//! 1. `instantiate_with_imports` keeps the manifest and the `call.import` instructions intact —
//!    the instantiated module is byte-identical to the verified one (content-addressable across
//!    instantiations; contrast the legacy `resolve_imports` rewrite, which lowered every
//!    `call.import` to `cap.call` and cleared the import section).
//! 2. The module runs **identically on all three backends** (tree-walker, bytecode engine, JIT)
//!    through the one shared `CAP_IMPORT_TYPE_ID` dispatch translation.
//! 3. The verifier's manifest checks are fail-closed (out-of-range index, sig mismatch,
//!    duplicate names) — the directed spec legs live in `spec_verify.rs`; here we pin the
//!    end-to-end errors an embedder sees.

use svm_ir::Inst;
use svm_run::{instantiate_with_imports, Backend, HostCap, Imports, Outcome, RunConfig};
use svm_text::parse_module;

/// A paramless `_start` reaching `write` (stdout `Stream`) and `exit` purely through **named
/// imports** — the same guest shape as `powerbox_named.rs`, but instantiated through the
/// phase-1 no-rewrite path: the `call.import`s below execute as-is, against the run's
/// instantiation-time binding table. The handle operands are vestigial placeholders (`i32.const
/// 0`, IMPORTS.md §2.5) — under the legacy path the resolver patched them; here they are ignored.
const NAMED_START: &str = "memory 17\n\
func () -> () {\n\
block 0 () {\n\
  vo = i32.const 111\n\
  va = i64.const 16\n\
  i32.store8 va vo\n\
  vk = i32.const 107\n\
  vb = i64.const 17\n\
  i32.store8 vb vk\n\
  vph = i32.const 0\n\
  vbuf = i64.const 16\n\
  vlen = i64.const 2\n\
  vn = call.sym \"write\" (i64, i64) -> (i64) vph (vbuf, vlen)\n\
  vph2 = i32.const 0\n\
  vcode = i32.const 5\n\
  call.sym \"exit\" (i32) -> () vph2 (vcode)\n\
  unreachable\n\
  }\n\
}\n\
export 0 func \"_start\" 0\n";

fn registry() -> Imports {
    Imports::new()
        .provide("write", HostCap::stdout())
        .provide("exit", HostCap::exit())
}

/// The headline: instantiation does NOT rewrite. The manifest survives, every `call.import`
/// survives, and no `cap.call` is manufactured.
#[test]
fn no_rewrite_manifest_and_call_imports_survive_instantiation() {
    let m = parse_module(NAMED_START).expect("parse");
    assert_eq!(m.imports.len(), 2, "text form interned two named imports");
    let inst = instantiate_with_imports(m.clone(), registry()).expect("instantiate");
    assert_eq!(
        inst.module(),
        &m,
        "phase-1 instantiation must keep the module byte-identical (no rewrite)"
    );
    let insts: Vec<&Inst> = inst.module().funcs[0].blocks[0].insts.iter().collect();
    // The name-inline text form is the symbolic spelling (v8: `call.sym`); it survives
    // instantiation unrewritten exactly as an indexed `call.import` would.
    assert!(
        insts.iter().any(|i| matches!(i, Inst::CallSym { .. })),
        "symbolic call instructions survive"
    );
    assert!(
        !insts.iter().any(|i| matches!(i, Inst::CapCall { .. })),
        "no cap.call was manufactured"
    );
}

/// The same guest runs identically on all three backends via the shared import-dispatch
/// translation, writing through the bound stdout `Stream` and exiting through the bound `Exit`.
#[test]
fn call_import_runs_identically_on_all_three_backends() {
    let m = parse_module(NAMED_START).expect("parse");
    let inst = instantiate_with_imports(m, registry()).expect("instantiate");
    let mut runs = Vec::new();
    for backend in [Backend::TreeWalk, Backend::Bytecode, Backend::Jit] {
        let r = inst
            .run(backend, &RunConfig::default())
            .unwrap_or_else(|e| panic!("{backend:?}: {e}"));
        assert_eq!(r.stdout, b"ok", "{backend:?} stdout");
        assert_eq!(r.outcome, Outcome::Exited(5), "{backend:?} outcome");
        runs.push((r.outcome, r.stdout, r.stderr));
    }
    assert!(
        runs.windows(2).all(|w| w[0] == w[1]),
        "backends must agree exactly"
    );
    // And the default differential entry (interp == jit) over the same instance.
    let diff = inst.call("_start", &[]).expect("run_diff");
    assert_eq!(diff.stdout, b"ok");
    assert_eq!(diff.outcome, Outcome::Exited(5));
}

/// An unbound import name fails instantiation closed — before anything runs.
#[test]
fn unbound_import_name_fails_instantiation() {
    let m = parse_module(NAMED_START).expect("parse");
    let Err(err) = instantiate_with_imports(m, Imports::new().provide("write", HostCap::stdout()))
    else {
        panic!("missing `exit` binding must fail");
    };
    assert!(
        err.contains("exit"),
        "error names the unbound import: {err}"
    );
}

/// A `call.import` in a module with no manifest is rejected by the verifier (out-of-range =
/// the legacy pre-manifest shape), surfaced through instantiation.
#[test]
fn out_of_range_import_fails_verification() {
    let mut m = parse_module(NAMED_START).expect("parse");
    m.imports.clear(); // manifest gone; call.imports now dangle
    let Err(err) = instantiate_with_imports(m, registry()) else {
        panic!("dangling call.import must fail verify");
    };
    assert!(
        err.contains("UnresolvedImport"),
        "error is the verifier's out-of-range check: {err}"
    );
}

/// A call-site sig disagreeing with the manifest declaration is fail-closed.
#[test]
fn sig_mismatch_fails_verification() {
    let mut m = parse_module(NAMED_START).expect("parse");
    // Flip the declared sig of import 0 ("write") so the call site no longer matches:
    // repoint the import's shape at a fresh, different Func type entry (§3.5 — signatures
    // live in the type section).
    let bad = m.intern_func_type(svm_ir::FuncType {
        params: vec![],
        results: vec![],
    });
    m.imports[0].shape = svm_ir::ImportShape::Func(bad);
    let Err(err) = instantiate_with_imports(m, registry()) else {
        panic!("sig mismatch must fail verify");
    };
    assert!(
        err.contains("ImportSigMismatch"),
        "error is the verifier's manifest sig check: {err}"
    );
}

/// Mem-hooks are exclusive with a manifest-carrying instance (the hook grant would occupy
/// import slot 0 — IMPORTS.md §2.1's slot-layout rule).
#[test]
fn mem_hooks_refused_on_manifest_instance() {
    let m = parse_module(NAMED_START).expect("parse");
    let inst = instantiate_with_imports(m, registry()).expect("instantiate");
    let Err(err) = inst.with_mem_hooks(|| Box::new(|_| Ok(()))) else {
        panic!("hooks + manifest must be refused");
    };
    assert!(
        err.contains("slot 0"),
        "error explains the collision: {err}"
    );
}
