//! §5 W3 / §23-D57 — **per-fiber trap attribution** on the JIT. When a JIT'd guest traps, the engine
//! already symbolizes a source backtrace ([`CompiledModule::last_trap_backtrace`]); this also records
//! *which fiber* was running at the trap instant ([`CompiledModule::last_trap_fiber`]). The fiber
//! identity is captured at the trap (the fiber runtime publishes the running fiber into the trap-capture
//! TLS across the resume seam), so it is correct even under §23-D57 work-stealing migration, where a
//! fiber may resume on a different vCPU thread than it last suspended on — the thread no longer
//! identifies the fiber, but the published handle does.
//!
//! Unix-only (the trap capture lives in `trap_shim.c`/`trap_capture.c`; the Windows VEH path reads the
//! same current-fiber TLS but isn't exercised here).

#![cfg(any(all(unix, target_arch = "x86_64"), all(unix, target_arch = "aarch64"),))]

use svm_ir::DEFAULT_RESERVED_LOG2;
use svm_jit::{CompiledModule, JitOutcome, Quota, TrapKind, INERT_CAP_THUNK};
use svm_text::parse_module;

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

/// Root creates a fiber (`cont.new` → the first fiber: slot 0, generation 0 → **handle 0**) and resumes
/// it; the fiber divides by zero. The run traps `DivByZero`, the backtrace names the fiber's div line,
/// and the trap is attributed to **fiber 0** — not the root.
const FIBER_DIV0: &str = "func () -> (i32, i64) {\n\
    block0():\n\
    \x20 v0 = ref.func 1\n\
    \x20 v1 = i64.const 4096\n\
    \x20 v2 = cont.new v0 v1\n\
    \x20 v3 = i64.const 7\n\
    \x20 v4, v5 = cont.resume v2 v3\n\
    \x20 return v4 v5\n\
    }\n\
    func (i64, i64) -> (i64) {\n\
    block0(v0: i64, v1: i64):\n\
    \x20 v2 = i64.const 0\n\
    \x20 v3 = i64.div_s v1 v2\n\
    \x20 return v3\n\
    }\n\
    debug.file 0 \"fib.c\"\n\
    debug.fname 1 \"divz\"\n\
    debug.loc 1 0 1 0 30 3\n";

#[test]
fn trap_in_a_resumed_fiber_is_attributed_to_that_fiber() {
    let mut cm = compile(FIBER_DIV0);
    let (outcome, _) = cm.run(&[], None, None, None).expect("run");
    assert!(
        matches!(outcome, JitOutcome::Trapped(TrapKind::DivByZero)),
        "the fiber's div-by-zero must trap DivByZero, got {outcome:?}"
    );
    assert_eq!(
        cm.last_trap_fiber(),
        Some(0),
        "the trap originated in the first fiber (handle 0), not the root"
    );
    // The backtrace still names where: the fiber's div line.
    let bt = cm.last_trap_backtrace();
    assert!(
        bt.iter().any(|f| f.line == 30),
        "backtrace should name the fiber's div line (30): {bt:?}"
    );
}

/// A trap in the **root** computation (no fiber running) is attributed to the root sentinel `-1`.
const ROOT_DIV0: &str = "func () -> (i64) {\n\
    block0():\n\
    \x20 v0 = i64.const 5\n\
    \x20 v1 = i64.const 0\n\
    \x20 v2 = i64.div_s v0 v1\n\
    \x20 return v2\n\
    }\n\
    debug.file 0 \"root.c\"\n\
    debug.fname 0 \"root\"\n\
    debug.loc 0 0 2 0 7 3\n";

#[test]
fn trap_in_the_root_is_attributed_to_no_fiber() {
    let mut cm = compile(ROOT_DIV0);
    let (outcome, _) = cm.run(&[], None, None, None).expect("run");
    assert!(matches!(outcome, JitOutcome::Trapped(TrapKind::DivByZero)));
    assert_eq!(
        cm.last_trap_fiber(),
        Some(-1),
        "a root trap is not attributed to any fiber"
    );
}

/// After the fiber traps and is attributed, a subsequent **clean** run clears the attribution — the
/// fiber handle is per-run, never stale.
const CLEAN: &str = "func () -> (i64) {\n\
    block0():\n\
    \x20 v0 = i64.const 42\n\
    \x20 return v0\n\
    }\n";

#[test]
fn a_clean_run_clears_the_trap_fiber() {
    let mut cm = compile(CLEAN);
    let (outcome, _) = cm.run(&[], None, None, None).expect("run");
    assert!(matches!(outcome, JitOutcome::Returned(_)));
    assert_eq!(
        cm.last_trap_fiber(),
        None,
        "a clean run leaves no trap-fiber attribution"
    );
}
