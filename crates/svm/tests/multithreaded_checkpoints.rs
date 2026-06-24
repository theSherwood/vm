//! Incremental **forward** time-travel seek for a multithreaded (scheduled) run (DEBUGGING.md W1).
//! A scheduled `seek(t)` used to rebuild the whole multi-vCPU run from turn 0 every time (O(t)); a
//! forward seek now *continues the live run* to turn `t` instead (O(t − current turn)). The replay is
//! deterministic (fixed schedule + cap tape + seed), so the landed state must be **identical** to a
//! rebuild — this asserts that across a forward sweep, a backward jump (which still rebuilds), and a
//! forward seek after the backward jump.

use svm_interp::{Inspector, Stop, Trap, Value};
use svm_text::parse_module;

/// Two workers each increment a shared counter; the default schedule (empty plan = smallest-runnable
/// thread each turn) is deterministic, so the per-turn state is reproducible.
const COUNTER: &str = r#"
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

/// A turn's observable state — what a debugger would show after `seek`: the stop (finished result or
/// the focused thread's program point), the global turn, and the shared counter bytes.
#[derive(Debug, PartialEq)]
struct Snap {
    finished: Option<Result<Vec<Value>, Trap>>,
    pc: Option<(usize, usize)>,
    turn: u64,
    mem: Vec<u8>,
}

fn snap(insp: &Inspector, s: &Stop) -> Snap {
    let (finished, pc) = match s {
        Stop::Finished(r) => (Some(r.clone()), None),
        Stop::Break { pc, .. } => (None, Some((pc.block, pc.inst))),
        Stop::Blocked => (None, None),
    };
    Snap {
        finished,
        pc,
        turn: insp.turn(),
        mem: insp.read_window(0, 8).unwrap_or_default(),
    }
}

fn attach() -> Inspector {
    let m = parse_module(COUNTER).expect("parse");
    Inspector::attach_scheduled(&m, 0, &[], 50_000_000, Vec::new())
}

/// `seek(t)` on a freshly attached inspector — the ground truth (always rebuilds from turn 0).
fn cold(t: u64) -> Snap {
    let mut insp = attach();
    let s = insp.seek(t);
    snap(&insp, &s)
}

#[test]
fn forward_seek_matches_a_rebuild() {
    let mut warm = attach();
    // Ascending targets ⇒ each `seek` continues the live run forward (the incremental path), past the
    // end (`u64::MAX`) so the finish is exercised too.
    for &t in &[1u64, 2, 3, 4, 5, 6, 8, 12, 20, u64::MAX] {
        let s = warm.seek(t);
        assert_eq!(
            snap(&warm, &s),
            cold(t),
            "forward (incremental) seek({t}) diverged from a cold rebuild"
        );
    }
}

#[test]
fn backward_then_forward_still_matches() {
    let mut warm = attach();
    // Forward to a mid turn, jump *backward* (this rebuilds), then forward again (incremental from the
    // rebuilt state) — each landing must still match a cold rebuild.
    for &t in &[10u64, 3, 7, 2, 9, u64::MAX] {
        let s = warm.seek(t);
        assert_eq!(
            snap(&warm, &s),
            cold(t),
            "seek({t}) after a backward jump diverged from a cold rebuild"
        );
    }
}

#[test]
fn seeking_the_same_turn_is_idempotent() {
    let mut warm = attach();
    let a = warm.seek(5);
    let b = warm.seek(5); // t == current turn: the incremental path runs zero turns
    assert_eq!(snap(&warm, &a), snap(&warm, &b));
    assert_eq!(snap(&warm, &b), cold(5));
}
