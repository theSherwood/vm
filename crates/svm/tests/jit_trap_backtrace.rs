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

use svm_interp::{Inspector, Stop, Value};
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
block 0 (v0: i32) {
  v1 = i64.const 65532
  v2 = i64.const 0
  i64.store v1 v2
  v3 = i32.const 0
  return v3
  }
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
block 0 (v0: i32) {
  v1 = call 1 (v0)
  v2 = i32.const 1
  v3 = i32.add v1 v2
  return v3
  }
}
func (i32) -> (i32) {
block 0 (v0: i32) {
  v1 = i64.const 65532
  v2 = i64.const 0
  i64.store v1 v2
  v3 = i32.const 0
  return v3
  }
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
block 0 (v0: i32) {
  v1 = i32.const 0
  v2 = i32.div_s v0 v1
  return v2
  }
}
debug.file 0 \"div.c\"
debug.fname 0 \"divzero\"
debug.loc 0 0 1 0 30 3
";

/// Function 0 calls function 1, which divides by zero — the explicit-trap capture must walk the
/// frame-pointer chain across both frames (call not in tail position, as in the memory-fault case).
/// The call is func0 instruction 0 (line 40); the div is func1 instruction 1 (line 30).
const CALL_THEN_DIV_DBG: &str = "\
func (i32) -> (i32) {
block 0 (v0: i32) {
  v1 = call 1 (v0)
  v2 = i32.const 1
  v3 = i32.add v1 v2
  return v3
  }
}
func (i32) -> (i32) {
block 0 (v0: i32) {
  v1 = i32.const 0
  v2 = i32.div_s v0 v1
  return v2
  }
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
    assert_eq!(
        bt[0].func_name.as_deref(),
        Some("divzero"),
        "the frame carries the -g function name"
    );
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

/// A **spawned-vCPU** trap (§5 W3 Stage 3): the root spawns function 1 on its own OS thread and
/// joins it; the worker divides by zero, trapping `DivByZero` *off the run thread*. Its capture lives
/// in the worker's thread-local, so the worker hands it to the domain for the run thread to surface.
/// The div is func1 block0 instruction 2, mapped to line 50.
const SPAWN_THEN_DIV_DBG: &str = "\
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = thread.spawn 1 v0 v0
  v2 = thread.join v1
  return v2
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  v2 = i32.const 5
  v3 = i32.const 0
  v4 = i32.div_s v2 v3
  v5 = i64.const 0
  return v5
  }
}
debug.file 0 \"thr.c\"
debug.loc 1 0 2 0 50 3
";

#[test]
fn trap_backtrace_attributes_a_spawned_vcpu_trap() {
    if !svm_jit::fiber_supported() {
        return; // no OS-thread vCPU runtime here — thread.spawn is inert, nothing to attribute
    }
    let mut cm = compile(SPAWN_THEN_DIV_DBG);
    let (outcome, _) = cm.run(&[0], None, None, None).expect("run");
    assert!(
        matches!(outcome, JitOutcome::Trapped(TrapKind::DivByZero)),
        "the spawned worker's div-by-zero must propagate as DivByZero, got {outcome:?}"
    );
    let bt = cm.last_trap_backtrace();
    assert!(
        bt.iter().any(|f| f.func == 1 && f.line == 50),
        "the trap that originated on the spawned vCPU is symbolized to func1's div line, got {bt:?}"
    );
}

/// The **differential oracle** the W3 scope was premised on: the JIT's trap-time backtrace must match
/// the *interpreter's* call stack at the same trap. The interpreter reifies its stack as `Vec<Frame>`
/// and leaves it intact on a trap (it returns `Err` without unwinding), so an `Inspector` driven to
/// the trap exposes it via `backtrace()` — no interpreter changes needed. Compares `(func, file,
/// line)` per frame, innermost first (both engines order that way). Single-threaded guests only (the
/// `Inspector` is single-threaded); the spawned-vCPU case is covered separately.
fn assert_jit_backtrace_matches_interp(src: &str, arg: i32) {
    let m = parse_module(src).expect("parse");
    svm_verify::verify_module(&m).expect("verify");

    // Interpreter oracle: drive to the trap, read its trap-time call stack.
    let mut insp = Inspector::attach(&m, 0, &[Value::I32(arg)], u64::MAX);
    let stop = insp.run_until_stop();
    assert!(
        matches!(stop, Stop::Finished(Err(_))),
        "the guest must trap on the interpreter, got {stop:?}"
    );
    let interp: Vec<(u32, String, u32)> = insp
        .backtrace()
        .iter()
        .map(|f| {
            let s = f
                .source
                .as_ref()
                .expect("each guest frame carries a source loc under -g");
            (f.pc.func, s.file.clone(), s.line)
        })
        .collect();

    // JIT under test.
    let mut cm = compile(src);
    cm.run(&[arg as i64], None, None, None).expect("run");
    let jit: Vec<(u32, String, u32)> = cm
        .last_trap_backtrace()
        .iter()
        .map(|f| (f.func, f.file.clone(), f.line))
        .collect();

    assert_eq!(
        jit, interp,
        "JIT trap backtrace must match the interpreter oracle for:\n{src}"
    );
}

#[test]
fn jit_trap_backtrace_matches_the_interpreter_oracle() {
    // Both trap families (memory fault + explicit check), single- and multi-frame.
    assert_jit_backtrace_matches_interp(STORE_OOB_DBG, 0);
    assert_jit_backtrace_matches_interp(CALL_THEN_FAULT_DBG, 0);
    assert_jit_backtrace_matches_interp(DIV_BY_ZERO_DBG, 7);
    assert_jit_backtrace_matches_interp(CALL_THEN_DIV_DBG, 7);
}

#[test]
fn no_trap_leaves_an_empty_backtrace() {
    // A run that returns normally must not leave a stale backtrace behind.
    let mut cm = compile(STORE_OOB_DBG);
    // First trap, then a clean run of a different module: a fresh compile is the clean run here.
    let mut clean = compile(
        "\
func (i32) -> (i32) {
block 0 (v0: i32) {
  return v0
  }
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
