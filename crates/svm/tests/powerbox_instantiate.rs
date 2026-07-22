//! The **frontend-independent powerbox** path: a hand-written IR module (no C, no `svm-llvm`) with
//! a paramless `_start` (func 0) and a `write` manifest import, instantiated and run through
//! [`svm_run::instantiate`] / [`svm_run::Instance::call`]. This is the `run_c_full` experience —
//! verify, grant the powerbox, bind the manifest slots, run on the interpreter **and** the JIT
//! under identical capabilities, assert interp == jit — but driven entirely from an IR module a
//! frontend (e.g. JACL's codegen) emits and links itself.
//!
//! Gated `#![cfg(unix)]` like the other JIT differential suites (`svm-jit`'s window/guard page is
//! unix-only).
#![cfg(unix)]

use svm_run::{instantiate, Outcome, Value};

/// A frontend's IR, hand-written in text form: a paramless `_start` (func 0, exported by name —
/// the phase-4 powerbox entry marker) whose `call.sym "write"` emits a read-only string
/// segment, then returns 0. The handle operand is a vestigial dummy (`i32.const 0`); the runtime
/// binds the `write` slot at instantiation — exactly what a non-C frontend that targets SVM-IR +
/// manifest capability imports produces.
const HELLO: &str = "\
memory 15
data ro 16384 \"hello, powerbox\\n\"
export 0 func \"_start\" 0
func () -> (i32) {
block 0 () {
  v0 = i32.const 0
  v1 = i64.const 16384
  v2 = i64.const 16
  v3 = call.sym \"write\" (i64, i64) -> (i64) v0 (v1, v2)
  v4 = i32.const 0
  return v4
  }
}
";

#[test]
fn handwritten_ir_runs_through_the_powerbox_wrapper() {
    // 1. A frontend emits IR (here, parsed from text) with a named `write` import and a named
    //    export `_start` (funcidx 0) — first-class imports *and* exports, like wasm.
    let module = svm_text::parse_module(HELLO).expect("frontend IR parses");
    assert_eq!(module.imports.len(), 1, "one named import: \"write\"");
    assert_eq!(module.imports[0].name, "write");
    assert!(
        module.funcs[0].params.is_empty(),
        "function 0 is the **paramless** powerbox entry — its imports bind to slots, not params"
    );
    assert_eq!(
        module.resolve_export("_start"),
        Some(0),
        "the `_start` export is the powerbox-entry marker"
    );

    // 2. The thin wrapper: validate the manifest, verify, grant the powerbox + bind the `write`
    //    slot, run interp + JIT.
    let instance = instantiate(module).expect("instantiate");
    let run = instance.call("_start", &[]).expect("run via the wrapper");

    // 4. Acceptance: the expected stdout bytes, produced identically on both backends (interp == jit
    //    is asserted *inside* the wrapper, so reaching here at all proves agreement).
    assert_eq!(run.stdout, b"hello, powerbox\n");
    assert_eq!(
        run.outcome,
        Outcome::Returned(vec![Value::I32(0)]),
        "_start returns 0"
    );
}

/// A module with a **non-`_start`** named export that is a pure kernel — `square(x) = x*x` at funcidx
/// **1** (func 0 is an unused stub, so the export's funcidx is non-zero) — to exercise
/// [`svm_run::Instance::call`] on a non-`_start` export: it resolves the name to its funcidx and runs
/// it as a **bare kernel** (args in, results out, interp == jit), with no powerbox capabilities.
const KERNEL: &str = "\
memory 15
export 0 func \"square\" 1
func (i64) -> (i32) {
block 0 (v0: i64) {
  v1 = i32.const 0
  return v1
  }
}
func (i32) -> (i32) {
block 0 (v0: i32) {
  v1 = i32.mul v0 v0
  return v1
  }
}
";

/// F3: `Instance::call("<non-_start>", args)` resolves a named export to its (non-zero) funcidx and
/// runs it as a pure kernel — returning results, with **no** capabilities granted. The decision: a
/// non-`_start` export gets no powerbox caps run-once (the stash is empty without `_start`); a
/// cap-using export is reached through a reactor `Session` instead. Here we pin the pure-kernel path.
#[test]
fn non_start_export_runs_as_bare_kernel() {
    let module = svm_text::parse_module(KERNEL).expect("parse");
    assert_eq!(
        module.resolve_export("square"),
        Some(1),
        "the kernel export lives at a non-zero funcidx"
    );
    let instance = instantiate(module).expect("instantiate");
    // Call the named kernel with an arg; results come back, interp == jit asserted inside the wrapper.
    let run = instance
        .call("square", &[Value::I32(7)])
        .expect("call a non-_start kernel export by name");
    assert_eq!(
        run.outcome,
        Outcome::Returned(vec![Value::I32(49)]),
        "square(7) via Instance::call on a non-zero export funcidx"
    );
    assert!(
        run.stdout.is_empty() && run.stderr.is_empty(),
        "a bare-kernel export produces no powerbox output (no caps granted)"
    );
}

/// The wrapper rejects caller args to the powerbox entry (the capabilities are slot-bound at
/// instantiation, not supplied by the caller) and an unknown export name — fail-closed, like the
/// rest of the load path.
#[test]
fn wrapper_guards_misuse() {
    let module = svm_text::parse_module(HELLO).expect("parse");
    let instance = instantiate(module).expect("instantiate");
    assert!(
        instance.call("_start", &[Value::I32(7)]).is_err(),
        "the powerbox entry takes no caller args"
    );
    assert!(
        instance.call("does_not_exist", &[]).is_err(),
        "unknown export is fail-closed"
    );
}

/// An **unbound** capability import fails closed at instantiation (the reference host policy has no
/// binding for it) — a required import a manifest module declares must be bindable before it runs.
#[test]
fn unbound_import_fails_closed() {
    let src = "\
memory 15
export 0 func \"_start\" 0
func () -> (i32) {
block 0 () {
  v0 = i32.const 0
  v1 = i64.const 0
  v2 = i64.const 0
  v3 = call.sym \"no_such_cap\" (i64, i64) -> (i64) v0 (v1, v2)
  v4 = i32.const 0
  return v4
  }
}
";
    let module = svm_text::parse_module(src).expect("parse");
    assert!(
        instantiate(module).is_err(),
        "an import with no host binding must fail closed at instantiate"
    );
}

/// A first-class export pointing past the end of `funcs` is rejected by the verifier (the gate
/// `instantiate` runs) — a dangling name is fail-closed, never a silent out-of-range dispatch.
#[test]
fn dangling_export_fails_verification() {
    let src = "\
memory 15
export 0 func \"ghost\" 9
func (i64) -> (i32) {
block 0 (v0: i64) {
  v1 = i32.const 0
  return v1
  }
}
";
    let module = svm_text::parse_module(src).expect("parse");
    assert_eq!(module.resolve_export("ghost"), Some(9));
    assert!(
        instantiate(module).is_err(),
        "an export funcidx past the functions must fail verification"
    );
}
