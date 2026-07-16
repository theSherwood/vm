//! The **frontend-independent powerbox** path: a hand-written IR module (no C, no `svm-llvm`) that
//! imports `write` and exports an entry is given the powerbox bootstrap via
//! [`svm_ir::synth_powerbox_start`], then instantiated and run through [`svm_run::instantiate`] /
//! [`svm_run::Instance::call`]. This is the `run_c_full` experience — verify, grant the powerbox,
//! run on the interpreter **and** the JIT under identical capabilities, assert interp == jit — but
//! driven entirely from an IR module a frontend (e.g. JACL's codegen) emits and links itself, with
//! no access to the C on-ramp's internal `synth_start`.
//!
//! Gated `#![cfg(unix)]` like the other JIT differential suites (`svm-jit`'s window/guard page is
//! unix-only).
#![cfg(unix)]

use svm_run::{instantiate, Outcome, Value};

/// A frontend's IR, hand-written in text form: one entry `(i64 sp) -> (i32)` that loads the stashed
/// `stdout` handle (window offset 0 — the powerbox stash `_start` writes), then `call.import "write"`
/// to emit a read-only string segment, and returns 0. No `_start`, no globals on page 0 — exactly
/// what a non-C frontend that targets SVM-IR + named capability imports produces.
///
/// Layout: the `"hello, powerbox\n"` literal is a *read-only* data segment at `POWERBOX_STACK_PAGE`
/// (16384) — page 1, isolated from the writable handle-stash on page 0, so `_start`'s handle stores
/// don't fault on a read-only page (the D40 page isolation the C path also relies on).
const HELLO: &str = "\
memory 15
data ro 16384 \"hello, powerbox\\n\"
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

#[test]
fn handwritten_ir_runs_through_the_powerbox_wrapper() {
    // 1. A frontend emits IR (here, parsed from text) with a named `write` import and a named
    //    export "entry" (funcidx 0) — first-class imports *and* exports, like wasm.
    let module = svm_text::parse_module(HELLO).expect("frontend IR parses");
    assert_eq!(module.imports.len(), 1, "one named import: \"write\"");
    assert_eq!(module.imports[0].name, "write");
    assert_eq!(
        module.resolve_export("entry"),
        Some(0),
        "frontend export by name"
    );

    // 2. Generalized synth_start: prepend the powerbox `_start` (stdout/stdin/exit — 3 handles), no
    //    heap. The entry is funcidx 0 before the prepend; it becomes funcidx 1 after.
    let with_start =
        svm_ir::synth_powerbox_start(module, 0, 3, false).expect("prepend powerbox _start");
    assert!(
        with_start.funcs[0].params.is_empty(),
        "function 0 is the **paramless** powerbox _start — it resolves its 3 handles by name (S15 c)"
    );
    // The frontend export shifted with the prepend, and `_start` is now a first-class export too —
    // both reachable by name, no funcidx bookkeeping for the embedder.
    assert_eq!(with_start.resolve_export("_start"), Some(0));
    assert_eq!(
        with_start.resolve_export("entry"),
        Some(1),
        "the frontend export shifted +1 past the prepended _start"
    );

    // 3. The thin wrapper: resolve the `write` import, verify, grant the powerbox, run interp + JIT.
    let instance = instantiate(with_start).expect("instantiate");
    let run = instance.call("_start", &[]).expect("run via the wrapper");

    // 4. Acceptance: the expected stdout bytes, produced identically on both backends (interp == jit
    //    is asserted *inside* the wrapper, so reaching here at all proves agreement).
    assert_eq!(run.stdout, b"hello, powerbox\n");
    assert_eq!(
        run.outcome,
        Outcome::Returned(vec![Value::I32(0)]),
        "the entry returns 0, propagated through _start"
    );
}

/// A module with a **non-`_start`** named export that is a pure kernel — `square(x) = x*x` at funcidx
/// **1** (func 0 is an unused stub, so the export's funcidx is non-zero) — to exercise
/// [`svm_run::Instance::call`] on a non-`_start` export: it resolves the name to its funcidx and runs
/// it as a **bare kernel** (args in, results out, interp == jit), with no powerbox capabilities.
const KERNEL: &str = "\
memory 15
export \"square\" 1
func (i64) -> (i32) {
block0(v0: i64):
  v1 = i32.const 0
  return v1
}
func (i32) -> (i32) {
block0(v0: i32):
  v1 = i32.mul v0 v0
  return v1
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

/// The wrapper rejects caller args to the powerbox entry (the handles are auto-granted, not supplied
/// by the caller) and an unknown export name — fail-closed, like the rest of the load path.
#[test]
fn wrapper_guards_misuse() {
    let module = svm_text::parse_module(HELLO).expect("parse");
    let with_start = svm_ir::synth_powerbox_start(module, 0, 3, false).expect("synth");
    let instance = instantiate(with_start).expect("instantiate");
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
/// binding for it) — resolution is mandatory before a module can run.
#[test]
fn unbound_import_fails_closed() {
    let src = "\
memory 15
func (i64) -> (i32) {
block0(v0: i64):
  v1 = i64.const 0
  v2 = i32.load v1
  v3 = i64.const 0
  v4 = i64.const 0
  v5 = call.import \"no_such_cap\" (i64, i64) -> (i64) v2 (v3, v4)
  v6 = i32.const 0
  return v6
}
";
    let module = svm_text::parse_module(src).expect("parse");
    let with_start = svm_ir::synth_powerbox_start(module, 0, 3, false).expect("synth");
    assert!(
        instantiate(with_start).is_err(),
        "an import with no host binding must fail closed at instantiate"
    );
}

/// A first-class export pointing past the end of `funcs` is rejected by the verifier (the gate
/// `instantiate` runs) — a dangling name is fail-closed, never a silent out-of-range dispatch.
#[test]
fn dangling_export_fails_verification() {
    let src = "\
memory 15
export \"ghost\" 9
func (i64) -> (i32) {
block0(v0: i64):
  v1 = i32.const 0
  return v1
}
";
    let module = svm_text::parse_module(src).expect("parse");
    assert_eq!(module.resolve_export("ghost"), Some(9));
    assert!(
        instantiate(module).is_err(),
        "an export funcidx past the functions must fail verification"
    );
}
