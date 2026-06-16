//! W7 concurrency-debugging surfacing (DEBUGGING.md): turn the built DPOR model checker into a
//! debugging tool — find a failing interleaving as a *replayable witness*, then reproduce it
//! deterministically (the W7 → W1 bridge).

use svm_interp::{explore_all, find_schedule, replay_schedule, Value};
use svm_text::parse_module;

// Two threads do a non-atomic load/add/store on mem[0]. The serial result is 2; the interleaving
// where both load 0 before either stores loses an update and yields 1.
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

#[test]
fn find_schedule_returns_a_replayable_witness_for_the_race() {
    let m = parse_module(RACY_COUNTER).expect("parse");
    let raced = Ok(vec![Value::I64(1)]); // the lost-update outcome
    let w = find_schedule(&m, 0, &[], 50_000_000, 200_000, |o| *o == raced)
        .expect("a racy schedule exists and DPOR finds it");
    assert_eq!(w.outcome, raced);
    assert!(!w.plan.is_empty(), "witness carries a concrete schedule");

    // The witness reproduces the raced outcome deterministically — every time.
    for _ in 0..5 {
        assert_eq!(replay_schedule(&m, 0, &[], 50_000_000, &w.plan), raced);
    }
}

#[test]
fn find_schedule_is_none_when_no_interleaving_matches() {
    let m = parse_module(RACY_COUNTER).expect("parse");
    // No schedule yields 99; the (complete) search returns None.
    assert!(find_schedule(&m, 0, &[], 50_000_000, 200_000, |o| *o
        == Ok(vec![Value::I64(99)]))
    .is_none());
}

#[test]
fn witness_for_the_correct_outcome_replays_and_matches_explore_all() {
    let m = parse_module(RACY_COUNTER).expect("parse");
    let report = explore_all(&m, 0, &[], 50_000_000, 200_000);
    assert!(report.complete);
    let correct = Ok(vec![Value::I64(2)]);
    assert!(
        report.outcomes.contains(&correct),
        "the serial total is reachable"
    );

    let w = find_schedule(&m, 0, &[], 50_000_000, 200_000, |o| *o == correct).unwrap();
    assert_eq!(replay_schedule(&m, 0, &[], 50_000_000, &w.plan), correct);
}
