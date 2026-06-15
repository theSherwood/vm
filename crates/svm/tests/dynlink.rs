//! In-window dynamic linking, milestone 0: **compile-time (static) linking of a function symbol.**
//!
//! A unit `caller` references another unit `add` by *name* (`call.import "add"`). The loader resolves
//! the name to `add`'s function index and `svm_ir::resolve_imports_with` rewrites the `CallImport`
//! into a **direct `call`** — exactly what a static linker does (symbol → concrete call). By the time
//! the verifier and both backends see the module, it's an ordinary closed module; "linking" was a
//! source-to-source rewrite, above the TCB, re-verified like everything else. (Dynamic, separately-
//! compiled linking — `call_indirect` through a `Jit.install` slot — is the next milestone.)

use svm_interp::Value;
use svm_ir::{Resolved, ResolvedCap};

/// Two "units" in one module: `add(a,b)=a+b` at index 0, and `caller(a,b)` (index 1) that calls
/// `add` **by name**. The dummy `v2` is the (unused) capability-handle operand `call.import` carries;
/// resolving to a `Func` drops it.
const TWO_UNITS: &str = "\
func (i32, i32) -> (i32) {
block0(v0: i32, v1: i32):
  v2 = i32.add v0 v1
  return v2
}

func (i32, i32) -> (i32) {
block0(v0: i32, v1: i32):
  v2 = i32.const 0
  v3 = call.import \"add\" (i32, i32) -> (i32) v2 (v0, v1)
  return v3
}
";

/// Resolve + verify, then run `caller` (entry 1) on interp + JIT with `args`; assert they agree and
/// return the i32 result.
fn link_and_run(resolver: impl FnMut(&str) -> Option<Resolved>, args: &[i32]) -> i64 {
    let m = svm_text::parse_module(TWO_UNITS).expect("parse");
    assert_eq!(m.imports.len(), 1, "one named import: \"add\"");
    // The compile-time link step: rewrite call.import "add" → a direct call to add's index.
    let linked = svm_ir::resolve_imports_with(&m, resolver).expect("resolve");
    assert!(linked.imports.is_empty(), "imports lowered away");
    // No CallImport survives; it became a direct Call.
    assert!(
        linked.funcs[1].blocks[0]
            .insts
            .iter()
            .all(|i| !matches!(i, svm_ir::Inst::CallImport { .. })),
        "the import must be lowered to a direct call"
    );
    svm_verify::verify_module(&linked).expect("verify linked module");

    let ivals: Vec<Value> = args.iter().map(|&x| Value::I32(x)).collect();
    let mut fuel = 10_000_000u64;
    let interp = svm_interp::run(&linked, 1, &ivals, &mut fuel).expect("interp run");
    let jargs: Vec<i64> = args.iter().map(|&x| x as i64).collect();
    let jit = match svm_jit::compile_and_run(&linked, 1, &jargs).expect("jit compile") {
        svm_jit::JitOutcome::Returned(v) => v,
        other => panic!("jit did not return: {other:?}"),
    };
    let iv = match interp[0] {
        Value::I32(x) => x as i64,
        other => panic!("unexpected interp value {other:?}"),
    };
    assert_eq!(iv as u32 as u64, jit[0] as u32 as u64, "interp != jit");
    iv
}

/// The core: `caller` reaches `add` purely by name, resolved at link time to a direct call.
#[test]
fn caller_links_to_add_by_name() {
    assert_eq!(
        link_and_run(|n| (n == "add").then_some(Resolved::Func(0)), &[3, 4]),
        7
    );
    assert_eq!(
        link_and_run(|n| (n == "add").then_some(Resolved::Func(0)), &[100, -1]),
        99
    );
}

/// An **unresolved** symbol is fail-closed (the loader can't find `add`).
#[test]
fn unresolved_symbol_fails_closed() {
    let m = svm_text::parse_module(TWO_UNITS).expect("parse");
    let err = svm_ir::resolve_imports_with(&m, |_| None).expect_err("must fail closed");
    assert_eq!(err, svm_ir::ImportError::Unresolved("add".into()));
}

/// A **signature mismatch** can't produce a type-unsafe call: linking feeds the re-verifier, never
/// bypasses it. `sym` is declared `(i32,i32)->i32` but resolved to a `(i64)->i64` function, so the
/// rewritten direct call has the wrong arg count/types — and `verify_module` rejects the linked
/// module. (This is the link-time symbol-signature check, enforced by re-verification, not trust.)
#[test]
fn signature_mismatch_is_caught_by_reverify() {
    let src = "\
func (i64) -> (i64) {
block0(v0: i64):
  return v0
}

func (i32, i32) -> (i32) {
block0(v0: i32, v1: i32):
  v2 = i32.const 0
  v3 = call.import \"sym\" (i32, i32) -> (i32) v2 (v0, v1)
  return v3
}
";
    let m = svm_text::parse_module(src).expect("parse");
    let linked = svm_ir::resolve_imports_with(&m, |_| Some(Resolved::Func(0))).expect("resolve");
    assert!(
        svm_verify::verify_module(&linked).is_err(),
        "a signature-mismatched link must be rejected by re-verification"
    );
}

/// The generalized pass still does the §7 capability case (`Resolved::Cap`) — a sanity check that the
/// `resolve_imports` (cap-only) path is unchanged by delegating through `resolve_imports_with`.
#[test]
fn capability_resolution_still_works() {
    let src = "\
func (i32) -> (i32) {
block0(v0: i32):
  v1 = i32.const 0
  v2 = call.import \"write\" (i32) -> (i32) v0 (v1)
  return v2
}
";
    let m = svm_text::parse_module(src).expect("parse");
    let linked = svm_ir::resolve_imports_with(&m, |_| {
        Some(Resolved::Cap(ResolvedCap { type_id: 0, op: 1 }))
    })
    .expect("resolve");
    // The import lowered to a cap.call (not a direct call).
    assert!(linked.funcs[0].blocks[0].insts.iter().any(|i| matches!(
        i,
        svm_ir::Inst::CapCall {
            type_id: 0,
            op: 1,
            ..
        }
    )));
}
