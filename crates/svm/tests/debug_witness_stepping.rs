//! Debug stops **during a fixed `find_schedule` witness-plan replay** (DEBUGGING.md Milestone B —
//! "stepping that crosses a scheduler decision point"). The scheduled debugger replays an exact
//! interleaving (a model-checker witness) on one OS thread; a breakpoint or watchpoint can now pause
//! *within* that replay and resume without desyncing the plan.
//!
//! Before the fix, a stop at a budget-exhausted visible op ran that op inside the *previous* turn
//! (the debug seam preempted the turn-boundary yield), collapsing two one-visible-op turns into one
//! and leaving the plan's next `TaskId` no longer runnable. The fix yields at the turn boundary
//! *before* the debug seam, so a stop fires at the start of its own turn — keeping the replay aligned.
//!
//! This is the headline these stops enable: **catch a lost-update data race in the act** by watching
//! the contended address under the exact failing interleaving.

use svm_interp::{find_schedule, Inspector, IrPc, Stop, StopReason, Value, WatchKind};
use svm_text::parse_module;

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

/// Find the lost-update interleaving (result `1`) once, shared by the tests below.
fn lost_update_witness() -> (svm_ir::Module, Vec<u64>) {
    let m = parse_module(RACY_COUNTER).expect("parse");
    let raced = Ok(vec![Value::I64(1)]);
    let w = find_schedule(&m, 0, &[], 50_000_000, 200_000, |o| *o == raced)
        .expect("a lost-update schedule exists");
    (m, w.plan)
}

/// A **breakpoint** at the worker's store, hit during the witness replay, then stepped through — the
/// replay stays aligned (no plan desync) and still reproduces the lost update (`1`). Each worker's
/// store is caught exactly once.
#[test]
fn breakpoint_during_witness_replay_stays_aligned() {
    let (m, plan) = lost_update_witness();
    let mut insp = Inspector::attach_scheduled(&m, 0, &[], 50_000_000, plan);
    let store = IrPc {
        module: 0,
        func: 1,
        block: 0,
        inst: 3,
    };
    insp.set_breakpoint(store);

    let mut store_threads = Vec::new();
    loop {
        match insp.run_until_stop() {
            Stop::Break { pc, .. } => {
                assert_eq!(pc, store, "the only breakpoint is the store");
                store_threads.push(insp.stopped_task().expect("a thread is stopped"));
                insp.step(); // step off the store and keep replaying
            }
            Stop::Finished(r) => {
                assert_eq!(
                    r,
                    Ok(vec![Value::I64(1)]),
                    "the witness still reproduces the lost update"
                );
                break;
            }
            Stop::Blocked => panic!("unexpected block"),
        }
    }
    store_threads.sort();
    assert_eq!(
        store_threads,
        vec![1, 2],
        "both workers' stores were caught, each once"
    );
}

/// **The headline:** watch the contended address under the lost-update witness. The write-watch fires
/// before each worker's store; reading the value each is about to write shows **both threads are
/// about to write `1`** — each from the same stale read — so the second clobbers the first and the
/// counter ends at `1`. The race, caught in the act, under the exact failing interleaving.
#[test]
fn watchpoint_catches_the_lost_update_race_under_the_witness() {
    let (m, plan) = lost_update_witness();
    let mut insp = Inspector::attach_scheduled(&m, 0, &[], 50_000_000, plan);
    insp.set_watchpoint(0, 8, WatchKind::Write);

    let mut about_to_write = Vec::new();
    loop {
        match insp.run_until_stop() {
            Stop::Break { reason, .. } => {
                assert!(matches!(reason, StopReason::Watchpoint { write: true, .. }));
                let task = insp.stopped_task().expect("a thread is stopped");
                // `vn` (the add result being stored) is SSA value 4 in the worker block.
                let v = insp
                    .read_ir_value(0, 4)
                    .expect("the store's value operand is live");
                about_to_write.push((task, v));
                insp.step(); // apply the store, continue the replay
            }
            Stop::Finished(r) => {
                assert_eq!(
                    r,
                    Ok(vec![Value::I64(1)]),
                    "the lost update (1) is reproduced"
                );
                break;
            }
            Stop::Blocked => panic!("unexpected block"),
        }
    }
    assert_eq!(about_to_write.len(), 2, "both workers' stores were watched");
    for (_t, v) in &about_to_write {
        assert_eq!(
            *v,
            Value::I64(1),
            "each worker is about to write its stale increment (1)"
        );
    }
    let mut tasks: Vec<u64> = about_to_write.iter().map(|(t, _)| *t).collect();
    tasks.sort();
    assert_eq!(
        tasks,
        vec![1, 2],
        "the two racing writers are distinct threads"
    );
}
