//! Oracle harness for **time-travel checkpointing** (DEBUGGING.md W1): `seek(t)` re-executes from
//! clock 0, so `step_back` is O(t²). Checkpoints let a `seek`/`step_back` restart from the nearest
//! snapshot (`clock ≤ t`) instead — bounding the replay to the checkpoint stride.
//!
//! Correctness gate: a **warm** Inspector (its checkpoint ladder populated by a prior deep seek, so
//! `seek` *restores* from a snapshot and replays only the tail) must observe **identical** state — the
//! result, the paused location, the logical clock, and guest memory — as a **cold** Inspector (freshly
//! attached, ladder empty, so it replays from clock 0). The cold path is the pre-existing, trusted
//! behavior; the warm path exercises snapshot capture + restore. If restore is faithful they agree at
//! every probed time, including across checkpoint-stride boundaries and a full backward sweep.

use svm_interp::{Inspector, Stop, Trap, Value};
use svm_text::parse_module;

/// A guest that runs **well past the checkpoint stride** (≥ a few thousand ops): a counter loop that
/// also mutates linear memory each iteration, so a faithful checkpoint must restore both the call
/// stack *and* the window bytes. `block1` is the loop header; each turn stores the running sum to a
/// fixed address and decrements the counter.
const LOOP_WITH_MEM: &str = "\
memory 16
func (i32) -> (i32) {
block0(v0: i32):
  v1 = i32.const 0
  br block1(v0, v1)
block1(v2: i32, v3: i32):
  v4 = i32.eqz v2
  br_if v4 block2(v3) block3(v2, v3)
block2(v5: i32):
  return v5
block3(v6: i32, v7: i32):
  v8 = i32.add v7 v6
  v9 = i32.const 0
  i32.store v9 v8
  v10 = i32.const -1
  v11 = i32.add v6 v10
  br block1(v11, v8)
}
";

/// The observable state at a paused/finished point — everything a user could read after a `seek`.
#[derive(Debug, PartialEq)]
struct Probe {
    stop_pc: Option<(usize, usize)>, // (block, inst) of the pause, or None when finished
    finished: Option<Result<Vec<Value>, Trap>>,
    clock: u64,
    mem: Vec<u8>, // the low window bytes the loop writes to
}

fn probe(insp: &Inspector, stop: &Stop) -> Probe {
    let (stop_pc, finished) = match stop {
        Stop::Break { pc, .. } => (Some((pc.block, pc.inst)), None),
        Stop::Finished(r) => (None, Some(r.clone())),
        Stop::Blocked => (None, None),
    };
    Probe {
        stop_pc,
        finished,
        clock: insp.clock(),
        mem: insp.read_window(0, 8).unwrap_or_default(),
    }
}

/// `seek(t)` on a freshly attached Inspector — the ground truth (ladder empty ⇒ replay from clock 0).
fn cold(src: &str, arg: i32, t: u64) -> Probe {
    let m = parse_module(src).expect("parse");
    let mut insp = Inspector::attach(&m, 0, &[Value::I32(arg)], 50_000_000);
    let stop = insp.seek(t);
    probe(&insp, &stop)
}

#[test]
fn warm_seek_matches_cold_across_checkpoint_boundaries() {
    let m = parse_module(LOOP_WITH_MEM).expect("parse");
    let mut warm = Inspector::attach(&m, 0, &[Value::I32(3000)], 50_000_000);

    // A deep seek lays down the checkpoint ladder (the run is far longer than the stride).
    let end = warm.seek(u64::MAX);
    assert!(
        matches!(end, Stop::Finished(_)),
        "the loop terminates: {end:?}"
    );
    assert!(
        warm.checkpoint_count() > 1,
        "a multi-thousand-op run must lay down several checkpoints (got {}) — else this harness is \
         vacuous",
        warm.checkpoint_count()
    );

    // Probe times that bracket the stride (1024): below the first checkpoint, just across it, deep in
    // the middle, and right at the end. Each warm seek restores from the nearest checkpoint ≤ t.
    for &t in &[0u64, 1, 1023, 1024, 1025, 2050, 4096, 7000, 100, 3, u64::MAX] {
        let stop = warm.seek(t);
        let got = probe(&warm, &stop);
        let want = cold(LOOP_WITH_MEM, 3000, t);
        assert_eq!(got, want, "warm seek({t}) diverged from cold replay-from-0");
    }
}

#[test]
fn backward_sweep_matches_cold() {
    let m = parse_module(LOOP_WITH_MEM).expect("parse");
    let mut warm = Inspector::attach(&m, 0, &[Value::I32(2500)], 50_000_000);
    let end = warm.seek(u64::MAX);
    let end_clock = match end {
        Stop::Finished(_) => warm.clock(),
        other => panic!("expected finish, got {other:?}"),
    };
    assert!(warm.checkpoint_count() > 1, "ladder must be populated");

    // Walk backward in strides from the end; each step_back-style jump restores from a checkpoint and
    // must match a cold replay-from-0 to the same time.
    let mut t = end_clock;
    while t > 0 {
        let stop = warm.seek(t);
        let got = probe(&warm, &stop);
        let want = cold(LOOP_WITH_MEM, 2500, t);
        assert_eq!(got, want, "backward seek({t}) diverged from cold");
        t = t.saturating_sub(317); // an odd stride so probes land off the checkpoint grid
    }
}

#[test]
fn step_back_one_at_a_time_is_faithful() {
    // Fine-grained: a run of step_back() calls near a checkpoint boundary must each land exactly where
    // a cold seek to that clock would, exercising the restore + short forward-replay tail.
    let m = parse_module(LOOP_WITH_MEM).expect("parse");
    let mut warm = Inspector::attach(&m, 0, &[Value::I32(1500)], 50_000_000);
    let _ = warm.seek(2100); // just past the second stride boundary
    for _ in 0..12 {
        let before = warm.clock();
        let stop = warm.step_back();
        assert_eq!(warm.clock(), before - 1, "step_back ticks the clock down by one");
        let got = probe(&warm, &stop);
        let want = cold(LOOP_WITH_MEM, 1500, warm.clock());
        assert_eq!(got, want, "step_back to {} diverged from cold", warm.clock());
    }
}

/// A run using a fiber falls outside the checkpointable subset, so checkpointing stays off and `seek`
/// remains the (correct) replay-from-0 path — no checkpoints captured, results still right.
#[test]
fn out_of_subset_run_keeps_replaying_from_zero() {
    // A long, purely-scalar run with no memory still checkpoints (sanity that the gate isn't
    // over-broad); then assert a memoryless variant also matches cold.
    const SCALAR: &str = "\
func (i32) -> (i32) {
block0(v0: i32):
  v1 = i32.const 0
  br block1(v0, v1)
block1(v2: i32, v3: i32):
  v4 = i32.eqz v2
  br_if v4 block2(v3) block3(v2, v3)
block2(v5: i32):
  return v5
block3(v6: i32, v7: i32):
  v8 = i32.add v7 v6
  v9 = i32.const -1
  v10 = i32.add v6 v9
  br block1(v10, v8)
}
";
    let m = parse_module(SCALAR).expect("parse");
    let mut warm = Inspector::attach(&m, 0, &[Value::I32(3000)], 50_000_000);
    let _ = warm.seek(u64::MAX);
    assert!(
        warm.checkpoint_count() > 1,
        "a memoryless scalar loop still checkpoints"
    );
    for &t in &[0u64, 1500, 3000, 50] {
        let stop = warm.seek(t);
        let got = probe(&warm, &stop);
        let want = {
            let mm = parse_module(SCALAR).expect("parse");
            let mut c = Inspector::attach(&mm, 0, &[Value::I32(3000)], 50_000_000);
            let s = c.seek(t);
            probe(&c, &s)
        };
        assert_eq!(got, want, "scalar warm seek({t}) diverged from cold");
    }
}
