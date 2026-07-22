//! §5 W3 / §23-D57 — **multi-vCPU trap-origin attribution**. A trap on a `thread.spawn`ed worker
//! vCPU propagates to its `thread.join`er as a bare `Err(Trap)` — the parent re-traps with *its* frames
//! at the join — so the interpreter's run outcome would name the *join site*, not where the guest
//! actually trapped. The JIT (W3 Stage 3) hands the worker's capture to the run-scoped `Domain` and
//! reports the origin; the interpreter now matches via a run-shared **first-wins trap-origin cell**
//! (`Sched::trap_origin`), so `run_traced` names the origin too.
//!
//! This is an **interp↔JIT differential**: both engines must name the same innermost frame (the
//! worker's faulting line) and the same trapping fiber (`-1` — a spawned worker runs its own root
//! computation, no fiber). Unix-only (the JIT trap capture is unix here).

#![cfg(any(all(unix, target_arch = "x86_64"), all(unix, target_arch = "aarch64")))]

use svm_interp::{func_name, run_traced, source_loc, IrPc};
use svm_ir::DEFAULT_RESERVED_LOG2;
use svm_jit::{CompiledModule, JitOutcome, Quota, TrapKind, INERT_CAP_THUNK};
use svm_text::parse_module;

/// Root spawns worker `func1(sp, arg)` and joins it; the worker divides by zero. The join propagates
/// the trap to the root — but the **origin** is the worker's div at line 70, not the root's join at
/// line 30.
const WORKER_DIV0: &str = "func () -> (i64) {\n\
    block 0 () {\n\
    \x20 v0 = i64.const 4096\n\
    \x20 v1 = i64.const 0\n\
    \x20 v2 = thread.spawn 1 v0 v1\n\
    \x20 v3 = thread.join v2\n\
    \x20 return v3\n\
      }\n\
    }\n\
    func (i64, i64) -> (i64) {\n\
    block 0 (v0: i64, v1: i64) {\n\
    \x20 v2 = i64.const 0\n\
    \x20 v3 = i64.div_s v1 v2\n\
    \x20 return v3\n\
      }\n\
    }\n\
    debug.file 0 \"t.c\"\n\
    debug.fname 0 \"root\"\n\
    debug.fname 1 \"worker\"\n\
    debug.loc 0 0 1 0 30 3\n\
    debug.loc 1 0 1 0 70 3\n";

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

/// The innermost frame of an interpreter backtrace as `(func index, name, line)`.
fn interp_innermost(m: &svm_ir::Module, bt: &[IrPc]) -> Option<(u32, String, u32)> {
    bt.first().map(|&pc| {
        let s = source_loc(m, pc).expect("frame resolves to source under -g");
        let name = func_name(m, pc.func)
            .map(str::to_owned)
            .unwrap_or_else(|| format!("fn{}", pc.func));
        (pc.func, name, s.line)
    })
}

#[test]
fn interp_names_the_worker_origin_not_the_join_site() {
    let m = parse_module(WORKER_DIV0).expect("parse");
    svm_verify::verify_module(&m).expect("verify");
    let mut fuel = 10_000_000u64;
    let (res, bt, fiber) = run_traced(&m, 0, &[], &mut fuel);
    assert!(
        matches!(res, Err(svm_interp::Trap::DivByZero)),
        "got {res:?}"
    );
    assert_eq!(
        interp_innermost(&m, &bt),
        Some((1, "worker".into(), 70)),
        "the origin is the worker's div (func 1, line 70), not the root's join (line 30)"
    );
    assert_eq!(
        fiber,
        Some(-1),
        "a spawned worker runs its own root computation — no fiber"
    );
}

#[test]
fn interp_and_jit_agree_on_the_worker_trap_origin() {
    // Interpreter (the oracle).
    let m = parse_module(WORKER_DIV0).expect("parse");
    svm_verify::verify_module(&m).expect("verify");
    let mut fuel = 10_000_000u64;
    let (_res, bt, interp_fiber) = run_traced(&m, 0, &[], &mut fuel);
    let interp = interp_innermost(&m, &bt);

    // JIT.
    let mut cm = compile(WORKER_DIV0);
    let (outcome, _) = cm.run(&[], None, None, None).expect("run");
    assert!(matches!(outcome, JitOutcome::Trapped(TrapKind::DivByZero)));
    let jit = cm.last_trap_backtrace().first().map(|f| {
        (
            f.func,
            f.func_name
                .clone()
                .unwrap_or_else(|| format!("fn{}", f.func)),
            f.line,
        )
    });

    assert_eq!(
        interp, jit,
        "interp and JIT disagree on the worker trap origin\ninterp={interp:?} jit={jit:?}"
    );
    assert_eq!(
        interp_fiber,
        cm.last_trap_fiber(),
        "interp and JIT disagree on the trapping fiber"
    );
}
