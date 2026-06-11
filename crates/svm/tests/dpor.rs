//! Differential proof that the DPOR model checker ([`explore_all`]) is **sound**: on every program
//! here it reports the *same set of terminal outcomes* as the unreduced enumerator
//! ([`explore_all_bruteforce`], which explores every ordering of every visible op), while running
//! **no more** schedules — and strictly **fewer** when the program has independent operations to
//! commute. The brute-force enumerator is the oracle; matching it across the racy programs below
//! (whose outcome *multiplicity* directly reflects interleaving coverage — a lost update, a store-
//! buffering read) means DPOR is not silently pruning reachable states.

use svm_interp::{explore_all, explore_all_bruteforce, Exhaustive, Trap, Value};
use svm_text::parse_module;
use svm_verify::verify_module;

const FUEL: u64 = 50_000_000;
const MAX: u64 = 20_000_000;

fn module(src: &str) -> svm_ir::Module {
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");
    m
}

/// `a` and `b` hold the same multiset of outcomes (order-independent; `Value` isn't `Hash`/`Ord`, so
/// compare by mutual containment over the de-duplicated `outcomes` vectors).
fn same_outcomes(a: &[Result<Vec<Value>, Trap>], b: &[Result<Vec<Value>, Trap>]) -> bool {
    a.len() == b.len() && a.iter().all(|x| b.contains(x)) && b.iter().all(|x| a.contains(x))
}

/// Run both checkers on `src` and assert DPOR is sound (same outcome set, both complete, DPOR ≤ brute).
/// Returns `(dpor, brute)` so a caller can additionally assert the *strict* reduction.
fn check(src: &str) -> (Exhaustive, Exhaustive) {
    let m = module(src);
    let dpor = explore_all(&m, 0, &[], FUEL, MAX);
    let brute = explore_all_bruteforce(&m, 0, &[], FUEL, MAX);
    assert!(
        dpor.complete,
        "DPOR truncated at {} schedules",
        dpor.schedules
    );
    assert!(
        brute.complete,
        "brute force truncated at {} schedules",
        brute.schedules
    );
    assert!(
        same_outcomes(&dpor.outcomes, &brute.outcomes),
        "DPOR outcomes {:?} != brute-force outcomes {:?}",
        dpor.outcomes,
        brute.outcomes
    );
    assert!(
        dpor.schedules <= brute.schedules,
        "DPOR ran more schedules ({}) than brute force ({})",
        dpor.schedules,
        brute.schedules
    );
    (dpor, brute)
}

/// Two threads each `atomic.rmw.add 1` to `mem[0]`; the total is 2 on **every** interleaving. The RMW
/// is atomic, so the two adds conflict (same word, both write) — DPOR explores both orders but the
/// outcome is invariant.
const ATOMIC_COUNTER: &str = r#"
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

/// Deliberately racy: each thread does a *non-atomic* load/add/store on `mem[0]`. The serial result is
/// 2, but the interleaving where both load 0 before either stores loses an update → 1. DPOR must report
/// **both** {1, 2}, exactly like brute force.
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

/// **Store buffering** — the classic two-variable shape. Thread 1: `X=1; r=load Y`; thread 2:
/// `Y=1; r=load X`; each returns its read. `main` encodes the pair as `2*r1 + r2`. Under the
/// interpreter's sequential consistency the reachable pairs are (0,1), (1,0), (1,1) — never (0,0) —
/// so the outcome set is exactly {1, 2, 3}. Conflicts span *two* objects (X and Y), exercising DPOR's
/// per-object race detection; DPOR must reproduce all three.
const STORE_BUFFER: &str = r#"
memory 16
func () -> (i64) {
block0():
  vsp = i64.const 0
  va = i64.const 0
  vh0 = thread.spawn 1 vsp va
  vh1 = thread.spawn 2 vsp va
  vj0 = thread.join vh0
  vj1 = thread.join vh1
  v2 = i64.const 2
  vt = i64.mul vj0 v2
  vres = i64.add vt vj1
  return vres
}
func (i64, i64) -> (i64) {
block0(vsp: i64, varg: i64):
  vx = i64.const 0
  v1 = i64.const 1
  i64.store vx v1
  vy = i64.const 8
  vr = i64.load vy
  return vr
}
func (i64, i64) -> (i64) {
block0(vsp: i64, varg: i64):
  vy = i64.const 8
  v1 = i64.const 1
  i64.store vy v1
  vx = i64.const 0
  vr = i64.load vx
  return vr
}
"#;

/// Two threads writing to **disjoint** addresses (child A → 0/8/16, child B → 32/40/48). Every pair of
/// stores is independent, so brute force explores all `C(6,3)`-plus orderings while DPOR collapses them
/// to a single representative. The outcome is invariant (0). This is the non-vacuous reduction witness.
const INDEPENDENT_STORES: &str = r#"
memory 16
func () -> (i64) {
block0():
  vsp = i64.const 0
  va = i64.const 0
  vh0 = thread.spawn 1 vsp va
  vh1 = thread.spawn 2 vsp va
  vj0 = thread.join vh0
  vj1 = thread.join vh1
  vz = i64.const 0
  return vz
}
func (i64, i64) -> (i64) {
block0(vsp: i64, varg: i64):
  v1 = i64.const 1
  va0 = i64.const 0
  i64.store va0 v1
  va8 = i64.const 8
  i64.store va8 v1
  va16 = i64.const 16
  i64.store va16 v1
  vz = i64.const 0
  return vz
}
func (i64, i64) -> (i64) {
block0(vsp: i64, varg: i64):
  v1 = i64.const 2
  va0 = i64.const 32
  i64.store va0 v1
  va8 = i64.const 40
  i64.store va8 v1
  va16 = i64.const 48
  i64.store va16 v1
  vz = i64.const 0
  return vz
}
"#;

#[test]
fn dpor_matches_bruteforce_atomic_counter() {
    let (dpor, _) = check(ATOMIC_COUNTER);
    assert_eq!(dpor.outcomes, vec![Ok(vec![Value::I64(2)])]);
}

#[test]
fn dpor_matches_bruteforce_racy_counter() {
    let (dpor, _) = check(RACY_COUNTER);
    assert!(
        dpor.outcomes.contains(&Ok(vec![Value::I64(1)]))
            && dpor.outcomes.contains(&Ok(vec![Value::I64(2)])),
        "DPOR missed a racy-counter interleaving: {:?}",
        dpor.outcomes
    );
}

#[test]
fn dpor_matches_bruteforce_store_buffer() {
    let (dpor, _) = check(STORE_BUFFER);
    let mut got: Vec<i64> = dpor
        .outcomes
        .iter()
        .map(|o| match o {
            Ok(v) => match v[0] {
                Value::I64(x) => x,
                _ => panic!("unexpected value {v:?}"),
            },
            Err(e) => panic!("unexpected trap {e:?}"),
        })
        .collect();
    got.sort_unstable();
    assert_eq!(got, vec![1, 2, 3], "store-buffering outcome set wrong");
}

/// The reduction is real: on the all-independent program DPOR runs *strictly* (here, dramatically)
/// fewer schedules than the unreduced enumeration, while reaching the same single outcome.
#[test]
fn dpor_reduces_independent_stores() {
    let (dpor, brute) = check(INDEPENDENT_STORES);
    assert_eq!(dpor.outcomes, vec![Ok(vec![Value::I64(0)])]);
    assert!(
        dpor.schedules < brute.schedules,
        "expected DPOR to prune independent reorderings (dpor={}, brute={})",
        dpor.schedules,
        brute.schedules
    );
    // The disjoint stores should collapse to a handful of schedules, not the full enumeration.
    assert!(
        dpor.schedules * 4 < brute.schedules,
        "DPOR reduction weaker than expected (dpor={}, brute={})",
        dpor.schedules,
        brute.schedules
    );
}
