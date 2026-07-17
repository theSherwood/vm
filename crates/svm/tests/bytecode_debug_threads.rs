//! Multithreaded debugging on the **bytecode** engine (DEBUGGING.md Milestone B, bytecode side): the
//! `ScheduledDebugRun` cooperative debug scheduler drives a `thread.spawn` guest with breakpoints that
//! fire **per-thread**, `stopped_task`/`select_task`/`threads` for per-thread inspection, and single-
//! step — parity-checked against the tree-walker `Inspector::attach_scheduled` oracle (the reference
//! multithreaded debug engine). This is the bytecode counterpart of `debug_threads.rs`.

use svm_interp::bytecode::{ScheduledDebugRun, SchedStop};
use svm_interp::{bytecode, run, Inspector, IrPc, Stop, Trap, Value};
use svm_text::parse_module;

// Two threads each do a non-atomic load/add/store on mem[0]. func 0 is the root (spawns + joins);
// func 1 is the worker: block0 inst0 = `vaddr = const 0`; inst1 = `vc = load vaddr`; inst2 = `vn =
// add vc varg`; inst3 = `store vaddr vn`; inst4 = `vz = const 0`. Same guest as `debug_threads.rs`.
const RACY_COUNTER: &str = r#"
memory 16
func () -> (i64) {
block0():
  vsp = i64.const 0
  va = i64.const 1
  vh0 = thread.spawn 1 vsp va
  vh1 = thread.spawn 1 vsp va
  vj0 = thread.join vh0
  vj1 = thread.join vh1
  vaddr = i64.const 0
  vr = i64.load vaddr
  return vr
}
func (i64, i64) -> (i64) {
block0(vsp: i64, varg: i64):
  vaddr = i64.const 0
  vc = i64.load vaddr
  vn = i64.add vc varg
  i64.store vaddr vn
  vz = i64.const 0
  return vz
}
"#;

// A *determinate* (interleaving-invariant) sibling of RACY_COUNTER: two workers each `atomic.rmw.add
// 1`, so the counter is exactly 2 on every schedule — the basis for a cross-engine result-equality
// check (RACY_COUNTER's non-atomic store makes the real M:N executor legitimately non-deterministic).
const ATOMIC_TWO: &str = r#"
memory 16
func () -> (i64) {
block0():
  vsp = i64.const 0
  va = i64.const 1
  vh0 = thread.spawn 1 vsp va
  vh1 = thread.spawn 1 vsp va
  vj0 = thread.join vh0
  vj1 = thread.join vh1
  vaddr = i64.const 0
  vr = i64.atomic.load vaddr
  return vr
}
func (i64, i64) -> (i64) {
block0(vsp: i64, varg: i64):
  vaddr = i64.const 0
  vrmw = i64.atomic.rmw.add vaddr varg
  vz = i64.const 0
  return vz
}
"#;

// The `vc = load` in the worker (func 1, block 0, inst 1) — where each spawned thread pauses.
fn worker_load_bp() -> IrPc {
    IrPc {
        module: 0,
        func: 1,
        block: 0,
        inst: 1,
    }
}

/// Run a `ScheduledDebugRun` to completion (no breakpoints), returning the root's result.
fn drive_to_end(sd: &mut ScheduledDebugRun, fuel: &mut u64) -> Result<Vec<Value>, Trap> {
    loop {
        match sd.run_until_stop(fuel) {
            SchedStop::Finished(r) => return r,
            SchedStop::Break(_) => continue,
            other => panic!("unexpected stop: {other:?}"),
        }
    }
}

/// A breakpoint in the worker fires **once per spawned thread**, and the debugger reports *which*
/// distinct vCPU is stopped, with that thread's backtrace live. Mirrors the tree-walker's
/// `breakpoint_fires_in_each_spawned_thread`.
#[test]
fn bytecode_breakpoint_fires_in_each_spawned_thread() {
    let m = parse_module(RACY_COUNTER).expect("parse");
    let mut sd = ScheduledDebugRun::new(&m, 0, &[]).expect("bytecode multithreaded debug session");
    sd.set_breakpoints(vec![worker_load_bp()]);
    let mut fuel = 50_000_000u64;

    // The first worker thread reaches the load.
    match sd.run_until_stop(&mut fuel) {
        SchedStop::Break(pc) => assert_eq!(pc, worker_load_bp()),
        other => panic!("expected Break, got {other:?}"),
    }
    let t1 = sd.stopped_task().expect("a thread is stopped");
    assert_eq!(
        sd.frame_pc(0),
        Some(worker_load_bp()),
        "innermost frame is at the breakpoint"
    );

    // Continue → the *other* worker thread hits the same breakpoint (a different vCPU).
    match sd.run_until_stop(&mut fuel) {
        SchedStop::Break(pc) => assert_eq!(pc, worker_load_bp()),
        other => panic!("expected Break, got {other:?}"),
    }
    let t2 = sd.stopped_task().expect("a thread is stopped");
    assert_ne!(t1, t2, "the two workers are distinct threads");

    // No more breakpoint hits (each worker runs the load once); the guest finishes ok.
    match sd.run_until_stop(&mut fuel) {
        SchedStop::Finished(r) => assert!(r.is_ok(), "guest finished: {r:?}"),
        other => panic!("expected Finished, got {other:?}"),
    }
}

/// The bytecode debug scheduler's result matches the tree-walker `attach_scheduled` oracle **and** the
/// production M:N executor — on a determinate (atomic) program, so any correct schedule yields the same
/// answer (always 2).
#[test]
fn bytecode_scheduled_run_matches_the_oracle() {
    let m = parse_module(ATOMIC_TWO).expect("parse");

    let mut sd = ScheduledDebugRun::new(&m, 0, &[]).unwrap();
    let mut fuel = 50_000_000u64;
    let bc = drive_to_end(&mut sd, &mut fuel);
    assert_eq!(bc, Ok(vec![Value::I64(2)]), "two atomic increments → 2");

    // Tree-walker scheduled debugger (empty schedule = deterministic default order).
    let mut insp = Inspector::attach_scheduled(&m, 0, &[], 50_000_000, vec![]);
    let tw = loop {
        match insp.run_until_stop() {
            Stop::Finished(r) => break r,
            Stop::Break { .. } => continue,
            Stop::Blocked => panic!("unexpected block"),
        }
    };
    assert_eq!(bc, tw, "bytecode scheduled ≡ tree-walker scheduled");

    // And the production M:N executor (real worker threads).
    let mut f = 50_000_000u64;
    assert_eq!(bc, run(&m, 0, &[], &mut f), "bytecode scheduled ≡ the M:N executor");
}

/// While stopped at one worker's breakpoint, `select_task` focuses another live thread and reads *its*
/// stack, then switches back — the multithreaded inspection surface.
#[test]
fn bytecode_select_task_inspects_another_thread_while_stopped() {
    let m = parse_module(RACY_COUNTER).expect("parse");
    let mut sd = ScheduledDebugRun::new(&m, 0, &[]).unwrap();
    sd.set_breakpoints(vec![worker_load_bp()]);
    let mut fuel = 50_000_000u64;

    assert!(matches!(sd.run_until_stop(&mut fuel), SchedStop::Break(pc) if pc == worker_load_bp()));
    let stopped = sd.stopped_task().unwrap();

    // Both workers plus the (join-blocked) root are live threads.
    let live = sd.threads();
    assert!(live.contains(&0), "root is live (blocked on join)");
    assert!(live.len() >= 3, "root + two workers live: {live:?}");

    // Focus the *other* worker — the one not yet run — and read its own top frame (its entry op).
    let other = *live
        .iter()
        .find(|&&t| t != 0 && t != stopped)
        .expect("a second worker is live");
    assert!(sd.select_task(other), "select the other worker");
    assert_eq!(
        sd.frame_pc(0),
        Some(IrPc {
            module: 0,
            func: 1,
            block: 0,
            inst: 0
        }),
        "the other worker sits at its own entry op, not the stopped thread's pc"
    );

    // Switch focus back to the stopped thread: it is still at the breakpoint.
    assert!(sd.select_task(stopped));
    assert_eq!(sd.frame_pc(0), Some(worker_load_bp()));
}

/// Single-stepping the stopped thread advances it one op (the load → the add), same thread, while the
/// schedule stays fixed and the logical clock ticks.
#[test]
fn bytecode_stepping_a_stopped_thread_advances_one_op() {
    let m = parse_module(RACY_COUNTER).expect("parse");
    let mut sd = ScheduledDebugRun::new(&m, 0, &[]).unwrap();
    sd.set_breakpoints(vec![worker_load_bp()]);
    let mut fuel = 50_000_000u64;

    assert!(matches!(sd.run_until_stop(&mut fuel), SchedStop::Break(pc) if pc == worker_load_bp()));
    let who = sd.stopped_task().unwrap();
    let t0 = sd.turn();

    match sd.step(&mut fuel) {
        SchedStop::Break(pc) => {
            assert_eq!(pc.inst, 2, "stepped from the load (inst 1) to the add (inst 2)");
            assert_eq!(sd.stopped_task(), Some(who), "still the same thread");
            assert!(sd.turn() > t0, "the logical clock ticked");
        }
        other => panic!("expected Break, got {other:?}"),
    }
}

/// `module_spawns_threads` routes the DAP backend: true for a `thread.spawn` guest (→ the scheduled
/// debugger), false for a spawn-free one (→ the reverse/watch-capable single-vCPU `DebugRun`).
#[test]
fn module_spawns_threads_detects_the_multithreaded_case() {
    let racy = parse_module(RACY_COUNTER).unwrap();
    assert!(bytecode::module_spawns_threads(&racy));
    let seq = parse_module("func () -> (i64) {\nblock0():\n  a = i64.const 7\n  return a\n}").unwrap();
    assert!(!bytecode::module_spawns_threads(&seq));
}
