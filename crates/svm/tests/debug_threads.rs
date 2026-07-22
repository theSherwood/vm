//! Multithreaded interpreter debugging (DEBUGGING.md Milestone B): drive a `thread.spawn` guest
//! under a fixed, reproducible schedule on one OS thread, with breakpoints that fire per-thread —
//! and replay a failing interleaving (a W7 `find_schedule` witness) under the debugger.

use svm_interp::{find_schedule, Inspector, IrPc, Stop, Value};
use svm_text::parse_module;

// Two threads each do a non-atomic load/add/store on mem[0] (the lost-update race). func 0 is the
// root (spawns + joins); func 1 is the worker:
//   block0(vsp, varg): vaddr=const 0 [0]; vc=load vaddr [1]; vn=add vc varg [2];
//                      store vaddr vn [3]; vz=const 0 [4]; return vz
const RACY_COUNTER: &str = r#"
memory 16
func () -> (i64) {
block 0 () {
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
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vaddr = i64.const 0
  vc = i64.load vaddr
  vn = i64.add vc varg
  i64.store vaddr vn
  vz = i64.const 0
  return vz
  }
}
"#;

/// A breakpoint in the worker fires once per spawned thread, and the debugger reports *which*
/// thread (a distinct vCPU) is stopped, with that thread's backtrace live.
#[test]
fn breakpoint_fires_in_each_spawned_thread() {
    let m = parse_module(RACY_COUNTER).expect("parse");
    // Empty schedule ⇒ the deterministic default order; reproducible, no real OS threads.
    let mut insp = Inspector::attach_scheduled(&m, 0, &[], 50_000_000, vec![]);
    let bp = IrPc {
        module: 0,
        func: 1,
        block: 0,
        inst: 1,
    }; // the `vc = load` in the worker
    insp.set_breakpoint(bp);

    // The first worker thread reaches the load.
    assert!(matches!(insp.run_until_stop(), Stop::Break { pc, .. } if pc == bp));
    let t1 = insp.stopped_task().expect("a thread is stopped");
    let bt = insp.backtrace();
    assert_eq!(
        bt.first().map(|f| f.pc),
        Some(bp),
        "innermost frame is at the breakpoint"
    );

    // Continue → the *other* worker thread hits the same breakpoint (a different vCPU).
    assert!(matches!(insp.run_until_stop(), Stop::Break { pc, .. } if pc == bp));
    let t2 = insp.stopped_task().expect("a thread is stopped");
    assert_ne!(t1, t2, "the two workers are distinct threads");

    // No more breakpoint hits (each worker runs the load once); the guest finishes.
    match insp.run_until_stop() {
        Stop::Finished(r) => assert!(r.is_ok(), "guest finished: {r:?}"),
        other => panic!("expected Finished, got {other:?}"),
    }
}

/// The headline Milestone B capability: take a *failing* interleaving found by the model checker
/// (W7 `find_schedule`) and replay it **under the debugger**, deterministically, on one thread —
/// reproducing the lost update (1 instead of 2).
#[test]
fn debugger_replays_a_race_witness() {
    let m = parse_module(RACY_COUNTER).expect("parse");
    let raced = Ok(vec![Value::I64(1)]);
    let w = find_schedule(&m, 0, &[], 50_000_000, 200_000, |o| *o == raced)
        .expect("a lost-update schedule exists");

    // Drive that exact interleaving under the Inspector (no breakpoints ⇒ run to completion).
    let mut insp = Inspector::attach_scheduled(&m, 0, &[], 50_000_000, w.plan.clone());
    loop {
        match insp.run_until_stop() {
            Stop::Finished(r) => {
                assert_eq!(r, raced, "the debugger reproduced the failing interleaving");
                break;
            }
            Stop::Break { .. } => continue,
            Stop::Blocked => panic!("unexpected block"),
        }
    }
}

/// Single-stepping the stopped thread advances it one op at a time (logical clock ticks), while the
/// schedule stays fixed.
#[test]
fn stepping_a_stopped_thread_advances_one_op() {
    let m = parse_module(RACY_COUNTER).expect("parse");
    let mut insp = Inspector::attach_scheduled(&m, 0, &[], 50_000_000, vec![]);
    let bp = IrPc {
        module: 0,
        func: 1,
        block: 0,
        inst: 1,
    };
    insp.set_breakpoint(bp);
    assert!(matches!(insp.run_until_stop(), Stop::Break { pc, .. } if pc == bp));

    let who = insp.stopped_task().unwrap();
    let c0 = insp.clock();
    // Step one op of this thread: the inner pc advances to inst 2 (the add), same thread.
    match insp.step() {
        Stop::Break { pc, .. } => {
            assert_eq!(pc.inst, 2, "stepped from the load to the add");
            assert_eq!(insp.stopped_task(), Some(who), "still the same thread");
            assert_eq!(insp.clock(), c0 + 1, "one logical tick elapsed");
        }
        other => panic!("expected a step stop, got {other:?}"),
    }
}

/// `select_task` (DEBUGGING.md Milestone B `select_fiber`): while stopped at a breakpoint in one
/// thread, inspect *any other* live thread — its own stack, pc, and logical clock — then switch
/// back. Focus resets when the guest next moves.
#[test]
fn select_task_inspects_another_thread_while_stopped() {
    let m = parse_module(RACY_COUNTER).expect("parse");
    let mut insp = Inspector::attach_scheduled(&m, 0, &[], 50_000_000, vec![]);
    let bp = IrPc {
        module: 0,
        func: 1,
        block: 0,
        inst: 1,
    }; // the worker's `load`
    insp.set_breakpoint(bp);
    assert!(matches!(insp.run_until_stop(), Stop::Break { pc, .. } if pc == bp));

    let stopped = insp.stopped_task().unwrap();
    // By default inspection targets the stopped thread, paused before the load (inst 1).
    assert_eq!(insp.focused_task(), Some(stopped));
    assert_eq!(insp.backtrace().first().map(|f| f.pc.inst), Some(1));

    // More than one thread is live (the two workers + the join-parked root).
    let threads = insp.threads();
    assert!(threads.contains(&stopped));
    assert!(threads.len() >= 2, "several threads are live: {threads:?}");

    // Focus another live thread and inspect IT: a distinct, live stack at a different pc.
    let other = *threads.iter().find(|&&t| t != stopped).unwrap();
    assert!(insp.select_task(other));
    assert_eq!(insp.focused_task(), Some(other));
    let other_bt = insp.backtrace();
    assert!(!other_bt.is_empty(), "the other thread has a live stack");
    assert_ne!(
        other_bt.first().map(|f| f.pc),
        Some(bp),
        "the other thread is not sitting at the stopped thread's pc"
    );

    // Switch focus back to the stopped thread → its view (at the breakpoint) returns.
    assert!(insp.select_task(stopped));
    assert_eq!(insp.backtrace().first().map(|f| f.pc), Some(bp));

    // Selecting a non-existent thread fails and leaves the focus unchanged.
    assert!(!insp.select_task(99_999));
    assert_eq!(insp.focused_task(), Some(stopped));

    // Resuming resets the focus to whatever thread next stops.
    if let Stop::Break { .. } = insp.run_until_stop() {
        assert_eq!(
            insp.focused_task(),
            insp.stopped_task(),
            "focus defaults back to the newly-stopped thread"
        );
    }
}

// --- W1 scheduled-mode time-travel: seek to a global scheduler turn (DEBUGGING.md W1) -----------

/// `seek(t)` in scheduled mode re-executes the pinned plan for `t` global scheduler turns and lands
/// at a reproducible global snapshot — every thread inspectable — and resuming from turn 0
/// reproduces the exact interleaving. The coordinate is `turn()`, not a per-thread clock.
#[test]
fn scheduled_seek_time_travels_to_a_global_turn() {
    let m = parse_module(RACY_COUNTER).expect("parse");
    // A witness plan pins the (racy) interleaving so the whole run is deterministic.
    let raced = Ok(vec![Value::I64(1)]);
    let plan = find_schedule(&m, 0, &[], 50_000_000, 200_000, |o| *o == raced)
        .expect("a lost-update schedule exists")
        .plan;

    let mut insp = Inspector::attach_scheduled(&m, 0, &[], 50_000_000, plan.clone());

    // Run to completion under the plan → the raced outcome; note how many turns it took.
    run_to_finished(&mut insp, &raced);
    let total = insp.turn();
    assert!(total > 2, "the run spans several scheduler turns: {total}");

    // Focused-thread snapshot (pc + live SSA per frame) — what time-travel must reproduce.
    fn snap(i: &Inspector) -> (Vec<u64>, Vec<(IrPc, Vec<Value>)>) {
        let bt = i
            .backtrace()
            .iter()
            .map(|f| (f.pc, f.vals.clone()))
            .collect();
        (i.threads(), bt)
    }

    // Seek to a middle turn: a global snapshot, not finished, with live threads.
    let mid = total / 2;
    assert!(matches!(insp.seek(mid), Stop::Break { .. }));
    assert_eq!(insp.turn(), mid);
    assert!(insp.result().is_none(), "mid-run is not finished");
    assert!(
        !insp.threads().is_empty(),
        "threads are live at the snapshot"
    );
    let at_mid = snap(&insp);

    // Time-travel is reproducible: seeking the same turn again restores the identical global state.
    assert!(matches!(insp.seek(mid), Stop::Break { .. }));
    assert_eq!(
        snap(&insp),
        at_mid,
        "seek(mid) reproduces the exact global snapshot"
    );

    // Seek to the start, then resume forward: the pinned plan reproduces the raced outcome.
    assert!(matches!(insp.seek(0), Stop::Break { .. }));
    assert_eq!(insp.turn(), 0);
    run_to_finished(&mut insp, &raced);

    // step_back walks the global turn down by one.
    insp.seek(mid);
    insp.step_back();
    assert_eq!(insp.turn(), mid - 1, "step_back decrements the global turn");
    // ...and seeking past the end lands on the finished result.
    assert!(matches!(insp.seek(total), Stop::Finished(ref r) if *r == raced));
}

fn run_to_finished(insp: &mut Inspector, want: &Result<Vec<Value>, svm_interp::Trap>) {
    loop {
        match insp.run_until_stop() {
            Stop::Finished(r) => {
                assert_eq!(&r, want);
                return;
            }
            Stop::Break { .. } => continue,
            Stop::Blocked => panic!("unexpected block"),
        }
    }
}

// --- W1 SchedTape: capture a (fuzzed) interleaving as a portable, replayable artifact -----------

fn run_outcome(insp: &mut Inspector) -> Result<Vec<Value>, svm_interp::Trap> {
    loop {
        match insp.run_until_stop() {
            Stop::Finished(r) => return r,
            Stop::Break { .. } => continue,
            Stop::Blocked => panic!("unexpected block"),
        }
    }
}

/// Schedule fuzzing (`attach_scheduled_seeded`) explores random interleavings — surfacing both the
/// correct total and the lost-update race — and each run's `sched_tape()` is a portable artifact
/// that `attach_scheduled` replays to the *exact* same outcome and interleaving.
#[test]
fn sched_tape_captures_a_seeded_run_and_replays_it() {
    let m = parse_module(RACY_COUNTER).expect("parse");
    let mut saw_race = false;
    let mut saw_ok = false;
    for seed in 0..64u64 {
        let mut insp = Inspector::attach_scheduled_seeded(&m, 0, &[], 50_000_000, seed);
        let outcome = run_outcome(&mut insp);
        match &outcome {
            Ok(v) if *v == vec![Value::I64(1)] => saw_race = true,
            Ok(v) if *v == vec![Value::I64(2)] => saw_ok = true,
            other => panic!("unexpected outcome {other:?}"),
        }

        // The interleaving that actually ran, captured as a plan.
        let tape = insp.sched_tape();
        assert!(
            !tape.is_empty(),
            "a multithreaded run records scheduling decisions"
        );

        // Replay the captured tape deterministically (no seed) → identical outcome and interleaving.
        let mut replay = Inspector::attach_scheduled(&m, 0, &[], 50_000_000, tape.clone());
        assert_eq!(
            run_outcome(&mut replay),
            outcome,
            "captured SchedTape replays to the same outcome (seed {seed})"
        );
        assert_eq!(
            replay.sched_tape(),
            tape,
            "replay follows the recorded schedule exactly"
        );
    }
    assert!(saw_race, "fuzzing surfaced the lost-update race (1)");
    assert!(saw_ok, "fuzzing surfaced the correct total (2)");
}

/// A seeded run reproduces under `seek` (same seed ⇒ same random interleaving) and under tape replay.
#[test]
fn seek_reproduces_a_seeded_run() {
    let m = parse_module(RACY_COUNTER).expect("parse");
    let mut insp = Inspector::attach_scheduled_seeded(&m, 0, &[], 50_000_000, 7);
    let outcome = run_outcome(&mut insp);
    let tape = insp.sched_tape();

    // seek(0) rebuilds with the same seed → the same random schedule replays on resume.
    assert!(matches!(insp.seek(0), Stop::Break { .. }));
    assert_eq!(
        run_outcome(&mut insp),
        outcome,
        "seek replays the seeded run's outcome"
    );
    assert_eq!(
        insp.sched_tape(),
        tape,
        "...with the identical interleaving"
    );
}
