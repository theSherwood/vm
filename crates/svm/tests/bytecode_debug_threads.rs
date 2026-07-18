//! Multithreaded debugging on the **bytecode** engine (DEBUGGING.md Milestone B, bytecode side): the
//! `ScheduledDebugRun` cooperative debug scheduler drives a `thread.spawn` guest with breakpoints that
//! fire **per-thread**, `stopped_task`/`select_task`/`threads` for per-thread inspection, and single-
//! step — parity-checked against the tree-walker `Inspector::attach_scheduled` oracle (the reference
//! multithreaded debug engine). This is the bytecode counterpart of `debug_threads.rs`.

use svm_interp::bytecode::{SchedBreak, SchedStop, ScheduledDebugRun};
use svm_interp::{bytecode, run, Inspector, IrPc, Stop, Trap, Value, WatchKind};
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
            SchedStop::Break { .. } => continue,
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
        SchedStop::Break { pc, .. } => assert_eq!(pc, worker_load_bp()),
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
        SchedStop::Break { pc, .. } => assert_eq!(pc, worker_load_bp()),
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
    assert_eq!(
        bc,
        run(&m, 0, &[], &mut f),
        "bytecode scheduled ≡ the M:N executor"
    );
}

/// While stopped at one worker's breakpoint, `select_task` focuses another live thread and reads *its*
/// stack, then switches back — the multithreaded inspection surface.
#[test]
fn bytecode_select_task_inspects_another_thread_while_stopped() {
    let m = parse_module(RACY_COUNTER).expect("parse");
    let mut sd = ScheduledDebugRun::new(&m, 0, &[]).unwrap();
    sd.set_breakpoints(vec![worker_load_bp()]);
    let mut fuel = 50_000_000u64;

    assert!(
        matches!(sd.run_until_stop(&mut fuel), SchedStop::Break { pc, .. } if pc == worker_load_bp())
    );
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

    assert!(
        matches!(sd.run_until_stop(&mut fuel), SchedStop::Break { pc, .. } if pc == worker_load_bp())
    );
    let who = sd.stopped_task().unwrap();
    let t0 = sd.turn();

    match sd.step(&mut fuel) {
        SchedStop::Break { pc, .. } => {
            assert_eq!(
                pc.inst, 2,
                "stepped from the load (inst 1) to the add (inst 2)"
            );
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
    let seq =
        parse_module("func () -> (i64) {\nblock0():\n  a = i64.const 7\n  return a\n}").unwrap();
    assert!(!bytecode::module_spawns_threads(&seq));
}

/// A **cross-thread write watchpoint** fires *before* each worker's store to the watched range, in
/// whichever thread runs the store, reporting the confined address + write — the multithreaded
/// counterpart of `cross_thread_watchpoints.rs`.
#[test]
fn bytecode_cross_thread_write_watchpoint_fires_per_worker() {
    let m = parse_module(RACY_COUNTER).unwrap();
    let mut sd = ScheduledDebugRun::new(&m, 0, &[]).unwrap();
    sd.set_watchpoints(vec![(0, 8, WatchKind::Write)]);
    let mut fuel = 50_000_000u64;

    // The worker's `i64.store vaddr vn` (func 1, block 0, inst 3) writes [0, 8).
    let store = IrPc {
        module: 0,
        func: 1,
        block: 0,
        inst: 3,
    };
    let mut hitters = Vec::new();
    for _ in 0..2 {
        match sd.run_until_stop(&mut fuel) {
            SchedStop::Break { pc, reason } => {
                assert_eq!(pc, store, "stops before the store that writes the range");
                assert_eq!(
                    reason,
                    SchedBreak::Watchpoint {
                        addr: 0,
                        write: true
                    },
                    "reports the confined address + write"
                );
                hitters.push(sd.stopped_task().unwrap());
            }
            other => panic!("expected a watchpoint stop, got {other:?}"),
        }
    }
    assert_ne!(
        hitters[0], hitters[1],
        "each worker's store trips the watch — distinct threads"
    );
    // No further writes to the range; the guest finishes.
    assert!(matches!(
        sd.run_until_stop(&mut fuel),
        SchedStop::Finished(_)
    ));
}

// A worker that calls a helper — for exercising step-over / step-into / step-out across a call frame
// on a spawned thread (while the root is blocked in `join`). func 1 worker: inst0 addr, inst1 load,
// inst2 = `call 2` (the call), inst3 store, inst4 const. func 2 helper: inst0 add, then return.
const WORKER_CALLS: &str = r#"
memory 16
func () -> (i64) {
block0():
  sp = i64.const 0
  one = i64.const 1
  h0 = thread.spawn 1 sp one
  j0 = thread.join h0
  addr = i64.const 0
  r = i64.load addr
  return r
}
func (i64, i64) -> (i64) {
block0(sp: i64, inc: i64):
  addr = i64.const 0
  cur = i64.load addr
  nxt = call 2(cur, inc)
  i64.store addr nxt
  z = i64.const 0
  return z
}
func (i64, i64) -> (i64) {
block0(a: i64, b: i64):
  s = i64.add a b
  return s
}
"#;

// The worker's `call 2` op (func 1, block 0, inst 2) — the step-over / step-into target.
fn worker_call_bp() -> IrPc {
    IrPc {
        module: 0,
        func: 1,
        block: 0,
        inst: 2,
    }
}

/// **Step-over** a call on a spawned thread runs the callee to completion and lands at the next op in
/// the *same* frame; **step-into** descends into the callee. Both drive the stopped thread across a
/// call while the root stays parked in `join`.
#[test]
fn bytecode_step_over_and_into_a_call_on_a_worker() {
    let m = parse_module(WORKER_CALLS).unwrap();

    // Step-over: from the call, land at the store (func 1, inst 3) — same thread, same depth.
    let mut sd = ScheduledDebugRun::new(&m, 0, &[]).unwrap();
    sd.set_breakpoints(vec![worker_call_bp()]);
    let mut fuel = 50_000_000u64;
    assert!(
        matches!(sd.run_until_stop(&mut fuel), SchedStop::Break { pc, .. } if pc == worker_call_bp())
    );
    let who = sd.stopped_task().unwrap();
    match sd.step_over(&mut fuel) {
        SchedStop::Break { pc, reason } => {
            assert_eq!(
                pc,
                IrPc {
                    module: 0,
                    func: 1,
                    block: 0,
                    inst: 3
                },
                "step-over landed at the op after the call, same frame"
            );
            assert_eq!(reason, SchedBreak::Step);
            assert_eq!(sd.stopped_task(), Some(who), "still the same worker");
        }
        other => panic!("expected a step stop, got {other:?}"),
    }

    // Step-into: from the call, descend into the helper (func 2, block 0, inst 0).
    let mut sd = ScheduledDebugRun::new(&m, 0, &[]).unwrap();
    sd.set_breakpoints(vec![worker_call_bp()]);
    let mut fuel = 50_000_000u64;
    assert!(
        matches!(sd.run_until_stop(&mut fuel), SchedStop::Break { pc, .. } if pc == worker_call_bp())
    );
    assert!(matches!(
        sd.step(&mut fuel),
        SchedStop::Break {
            pc: IrPc {
                func: 2,
                block: 0,
                inst: 0,
                ..
            },
            ..
        }
    ));
}

/// **Step-out** from inside the callee returns to the caller's next op — one call depth shallower — on
/// the spawned thread.
#[test]
fn bytecode_step_out_of_a_callee_on_a_worker() {
    let m = parse_module(WORKER_CALLS).unwrap();
    let mut sd = ScheduledDebugRun::new(&m, 0, &[]).unwrap();
    sd.set_breakpoints(vec![worker_call_bp()]);
    let mut fuel = 50_000_000u64;
    assert!(
        matches!(sd.run_until_stop(&mut fuel), SchedStop::Break { pc, .. } if pc == worker_call_bp())
    );

    // Into the helper, then out — back to the store after the call (func 1, inst 3).
    assert!(matches!(
        sd.step(&mut fuel),
        SchedStop::Break {
            pc: IrPc { func: 2, .. },
            ..
        }
    ));
    match sd.step_out(&mut fuel) {
        SchedStop::Break { pc, .. } => assert_eq!(
            pc,
            IrPc {
                module: 0,
                func: 1,
                block: 0,
                inst: 3
            },
            "step-out returned to the caller's next op"
        ),
        other => panic!("expected a step stop, got {other:?}"),
    }
}

/// Reverse debugging rests on **deterministic replay**: a fresh session ticked to a global `turn`
/// reproduces the exact scheduler position (same about-to-run thread, same pc) a forward run reached at
/// that turn. This is what `BytecodeBackend::seek` relies on for `stepBack`/`reverseContinue`.
#[test]
fn bytecode_scheduled_tick_replays_deterministically() {
    let m = parse_module(RACY_COUNTER).unwrap();

    // Forward to the first worker breakpoint; record (turn, thread, pc).
    let mut a = ScheduledDebugRun::new(&m, 0, &[]).unwrap();
    a.set_breakpoints(vec![worker_load_bp()]);
    let mut fuel = 50_000_000u64;
    assert!(matches!(
        a.run_until_stop(&mut fuel),
        SchedStop::Break { .. }
    ));
    let turn = a.op_turn();
    let who = a.stopped_task();
    let pc = a.frame_pc(0);
    assert_eq!(pc, Some(worker_load_bp()));

    // A fresh run raw-ticked to that same global turn lands at the identical position.
    let mut b = ScheduledDebugRun::new(&m, 0, &[]).unwrap();
    let mut f2 = 50_000_000u64;
    while b.op_turn() < turn && b.tick(&mut f2) {}
    b.locate();
    assert_eq!(b.op_turn(), turn, "replayed to the same global turn");
    assert_eq!(b.stopped_task(), who, "same about-to-run thread");
    assert_eq!(b.frame_pc(0), pc, "same pc");
}

// A futex handoff (from `bytecode_threads.rs`): the root seeds mem[8], spawns a worker, sets a flag +
// `atomic.notify`s mem[0], joins; the worker `atomic.wait`s on mem[0] then reads mem[8] → 987654. The
// worker parks on the wait until the root's notify wakes it — exercising `memory.wait`/`notify` under
// the debug scheduler.
const FUTEX_HANDOFF: &str = r#"
memory 16
func () -> (i64) {
block0():
  v0 = i64.const 8
  v1 = i64.const 987654
  i64.atomic.store.release v0 v1
  v2 = i64.const 0
  v3 = thread.spawn 1 v2 v2
  v4 = i64.const 0
  v5 = i32.const 1
  i32.atomic.store.release v4 v5
  v6 = i64.const 0
  v7 = i32.const 1
  v8 = atomic.notify v6 v7
  v9 = thread.join v3
  return v9
}
func (i64, i64) -> (i64) {
block0(vsp: i64, v0: i64):
  v1 = i64.const 0
  v2 = i32.const 0
  v3 = i64.const 1000000000
  v4 = i32.atomic.wait v1 v2 v3
  v5 = i64.const 8
  v6 = i64.atomic.load.acquire v5
  return v6
}
"#;

/// `memory.wait`/`notify` drive under the debug scheduler: the worker parks on the wait, the root's
/// notify wakes it, and the run completes with the handed-off value — matching the tree-walker oracle
/// and the production M:N executor (was `SchedStop::Declined` before this slice).
#[test]
fn bytecode_scheduled_wait_notify_completes() {
    let m = parse_module(FUTEX_HANDOFF).unwrap();
    let mut sd = ScheduledDebugRun::new(&m, 0, &[]).unwrap();
    let mut fuel = 50_000_000u64;
    let bc = drive_to_end(&mut sd, &mut fuel);
    assert_eq!(bc, Ok(vec![Value::I64(987654)]), "the futex handoff value");

    let mut insp = Inspector::attach_scheduled(&m, 0, &[], 50_000_000, vec![]);
    let tw = loop {
        match insp.run_until_stop() {
            Stop::Finished(r) => break r,
            Stop::Break { .. } => continue,
            Stop::Blocked => panic!("unexpected block"),
        }
    };
    assert_eq!(bc, tw, "scheduled ≡ tree-walker across wait/notify");
    let mut f = 50_000_000u64;
    assert_eq!(bc, run(&m, 0, &[], &mut f), "scheduled ≡ the M:N executor");
}

/// A breakpoint placed *after* the worker's `atomic.wait` fires — proving the worker actually parked and
/// was woken by the root's `notify` under the debugger (the wait didn't spuriously fall through).
#[test]
fn bytecode_breakpoint_after_a_wait_fires_once_woken() {
    let m = parse_module(FUTEX_HANDOFF).unwrap();
    let mut sd = ScheduledDebugRun::new(&m, 0, &[]).unwrap();
    // func 1, block 0, inst 5 = `v6 = i64.atomic.load.acquire v5` — the op after the wait (inst 3) and
    // the const (inst 4).
    let after_wait = IrPc {
        module: 0,
        func: 1,
        block: 0,
        inst: 5,
    };
    sd.set_breakpoints(vec![after_wait]);
    let mut fuel = 50_000_000u64;
    match sd.run_until_stop(&mut fuel) {
        SchedStop::Break { pc, .. } => {
            assert_eq!(pc, after_wait, "the woken worker reached the load")
        }
        other => panic!("expected the worker to wake and hit the breakpoint, got {other:?}"),
    }
    // The stopped thread is the spawned worker (task 1), not the root.
    assert_eq!(sd.stopped_task(), Some(1));
    // Continue → the guest finishes with the handed-off value.
    match sd.run_until_stop(&mut fuel) {
        SchedStop::Finished(r) => assert_eq!(r, Ok(vec![Value::I64(987654)])),
        other => panic!("expected Finished, got {other:?}"),
    }
}
