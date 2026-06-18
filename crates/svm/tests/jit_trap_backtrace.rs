//! §5 W3 — the **JIT trap-time backtrace**. When a JIT'd guest traps, the engine walks the
//! frame-pointer chain + symbolizes it into a source backtrace exposed by
//! [`CompiledModule::last_trap_backtrace`]. Always on — no debugger, no attach — and the `-g` debug
//! locs the module already carries make it `file:line`. Two capture paths:
//!   - **Stage 1, memory faults** — the §5 detect-and-kill guard handler (`trap_shim.c`) walks the
//!     chain in the SIGSEGV/SIGBUS handler, while the stack is intact (the overrun-the-window case
//!     `escape_oracle` exercises).
//!   - **Stage 2, explicit-check traps** (div-by-zero, …) — the lowering calls a capture helper at
//!     the trap site *before* it stores the kind and unwinds, since there is no signal there.
//!
//! Unix-only: both captures live in `trap_shim.c`; the Windows VEH path is a follow-up, so there it
//! degrades to an empty backtrace rather than a wrong one.

#![cfg(unix)]

use svm_ir::DEFAULT_RESERVED_LOG2;
use svm_jit::{CompiledModule, JitOutcome, Quota, TrapKind, INERT_CAP_THUNK};
use svm_text::parse_module;

/// Compile `src` to a runnable module (mirrors `jit_srcloc`), keeping the `CompiledModule` so the
/// test can read `last_trap_backtrace` after a run.
fn compile(src: &str) -> CompiledModule {
    let m = parse_module(src).expect("parse");
    svm_verify::verify_module(&m).expect("verify");
    CompiledModule::compile(
        &m,
        0,
        INERT_CAP_THUNK,
        core::ptr::null_mut(),
        DEFAULT_RESERVED_LOG2,
        None,
        None,
        None,
        None,
        Quota::default(),
        0,
    )
    .expect("jit compiles")
}

/// A single-frame fault: an 8-byte store at 65532 in a 64 KiB (`memory 16`) window writes
/// `[65532, 65540)`, overrunning the top of the window into the guard page — a clean detect-and-kill
/// `MemoryFault`. The store is block0 instruction index 2, mapped to source line 10.
const STORE_OOB_DBG: &str = "\
memory 16
func (i32) -> (i32) {
block0(v0: i32):
  v1 = i64.const 65532
  v2 = i64.const 0
  i64.store v1 v2
  v3 = i32.const 0
  return v3
}
debug.file 0 \"fault.c\"
debug.loc 0 0 2 0 10 3
";

/// Function 0 calls function 1, which does the overrunning store — so the trap backtrace must walk
/// the frame-pointer chain across the two guest frames. The call's result feeds a later `add`, so
/// the call is **not** in tail position (no `return_call` frame reuse) and func0's frame is live on
/// the stack when func1 faults. The call is func0 block0 instruction 0 (source line 20); the store
/// is func1 block0 instruction 2 (source line 10).
const CALL_THEN_FAULT_DBG: &str = "\
memory 16
func (i32) -> (i32) {
block0(v0: i32):
  v1 = call 1 (v0)
  v2 = i32.const 1
  v3 = i32.add v1 v2
  return v3
}
func (i32) -> (i32) {
block0(v0: i32):
  v1 = i64.const 65532
  v2 = i64.const 0
  i64.store v1 v2
  v3 = i32.const 0
  return v3
}
debug.file 0 \"fault.c\"
debug.loc 0 0 0 0 20 5
debug.loc 1 0 2 0 10 3
";

#[test]
fn trap_backtrace_names_the_faulting_store() {
    let mut cm = compile(STORE_OOB_DBG);
    let (outcome, _) = cm.run(&[0], None, None, None).expect("run");
    assert!(
        matches!(outcome, JitOutcome::Trapped(TrapKind::MemoryFault)),
        "the overrunning store must be caught as a MemoryFault, got {outcome:?}"
    );
    let bt = cm.last_trap_backtrace();
    assert_eq!(
        bt.len(),
        1,
        "exactly the one faulting guest frame, got {bt:?}"
    );
    assert_eq!(bt[0].func, 0, "the faulting frame is function 0");
    assert_eq!(bt[0].line, 10, "symbolized to the store's source line");
    assert_eq!(bt[0].file, "fault.c", "and its source file");
}

#[test]
fn trap_backtrace_walks_the_caller_chain() {
    let mut cm = compile(CALL_THEN_FAULT_DBG);
    let (outcome, _) = cm.run(&[0], None, None, None).expect("run");
    assert!(
        matches!(outcome, JitOutcome::Trapped(TrapKind::MemoryFault)),
        "the overrunning store in the callee must trap MemoryFault, got {outcome:?}"
    );
    let bt = cm.last_trap_backtrace();
    assert_eq!(
        bt.len(),
        2,
        "innermost callee (func1) + its caller (func0), got {bt:?}"
    );
    assert_eq!(
        (bt[0].func, bt[0].line),
        (1, 10),
        "innermost frame = function 1 at the store line"
    );
    assert_eq!(bt[1].func, 0, "caller frame = function 0");
    assert_eq!(
        bt[1].line, 20,
        "caller symbolized to the call-site line (ret-1 lands in the call)"
    );
}

/// A single-frame **explicit-check** trap: `v0 / 0` traps `DivByZero` via the lowered zero-divisor
/// check (no signal — the lowering stores the kind and returns). The div is block0 instruction 1
/// (the `const 0` is 0), mapped to source line 30.
const DIV_BY_ZERO_DBG: &str = "\
func (i32) -> (i32) {
block0(v0: i32):
  v1 = i32.const 0
  v2 = i32.div_s v0 v1
  return v2
}
debug.file 0 \"div.c\"
debug.loc 0 0 1 0 30 3
";

/// Function 0 calls function 1, which divides by zero — the explicit-trap capture must walk the
/// frame-pointer chain across both frames (call not in tail position, as in the memory-fault case).
/// The call is func0 instruction 0 (line 40); the div is func1 instruction 1 (line 30).
const CALL_THEN_DIV_DBG: &str = "\
func (i32) -> (i32) {
block0(v0: i32):
  v1 = call 1 (v0)
  v2 = i32.const 1
  v3 = i32.add v1 v2
  return v3
}
func (i32) -> (i32) {
block0(v0: i32):
  v1 = i32.const 0
  v2 = i32.div_s v0 v1
  return v2
}
debug.file 0 \"div.c\"
debug.loc 0 0 0 0 40 5
debug.loc 1 0 1 0 30 3
";

#[test]
fn explicit_trap_backtrace_names_the_div() {
    let mut cm = compile(DIV_BY_ZERO_DBG);
    let (outcome, _) = cm.run(&[7], None, None, None).expect("run");
    assert!(
        matches!(outcome, JitOutcome::Trapped(TrapKind::DivByZero)),
        "v0/0 must trap DivByZero, got {outcome:?}"
    );
    let bt = cm.last_trap_backtrace();
    assert_eq!(bt.len(), 1, "the one faulting guest frame, got {bt:?}");
    assert_eq!(bt[0].func, 0, "the trapping frame is function 0");
    assert_eq!(bt[0].line, 30, "symbolized to the div's source line");
}

#[test]
fn explicit_trap_backtrace_walks_the_caller_chain() {
    let mut cm = compile(CALL_THEN_DIV_DBG);
    let (outcome, _) = cm.run(&[7], None, None, None).expect("run");
    assert!(
        matches!(outcome, JitOutcome::Trapped(TrapKind::DivByZero)),
        "the callee's div-by-zero must trap DivByZero, got {outcome:?}"
    );
    let bt = cm.last_trap_backtrace();
    assert_eq!(
        bt.len(),
        2,
        "innermost callee (func1) + its caller (func0), got {bt:?}"
    );
    assert_eq!(
        (bt[0].func, bt[0].line),
        (1, 30),
        "innermost frame = function 1 at the div line"
    );
    assert_eq!(bt[1].func, 0, "caller frame = function 0");
    assert_eq!(
        bt[1].line, 40,
        "caller symbolized to the call-site line (ret-1 lands in the call)"
    );
}

#[test]
fn no_trap_leaves_an_empty_backtrace() {
    // A run that returns normally must not leave a stale backtrace behind.
    let mut cm = compile(STORE_OOB_DBG);
    // First trap, then a clean run of a different module: a fresh compile is the clean run here.
    let mut clean = compile(
        "\
func (i32) -> (i32) {
block0(v0: i32):
  return v0
}
",
    );
    let (out, _) = clean.run(&[7], None, None, None).expect("run");
    assert_eq!(out, JitOutcome::Returned(vec![7]));
    assert!(
        clean.last_trap_backtrace().is_empty(),
        "a non-trapping run has no trap backtrace"
    );
    // And the trapping module does populate one (sanity vs the empty case above).
    let _ = cm.run(&[0], None, None, None).expect("run");
    assert!(
        !cm.last_trap_backtrace().is_empty(),
        "the trapping module has a backtrace"
    );
}
