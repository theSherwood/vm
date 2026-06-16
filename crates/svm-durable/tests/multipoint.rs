//! Multiple resume points in one function (DURABILITY.md §9, the `br_table` arms).
//!
//! A function with several may-suspend ops can freeze at *any* of them, so each becomes a
//! resume point with its own `br_table` arm. The "freeze from the start" driver only ever
//! lands on the first point, so to exercise an **interior** arm we simulate a host-
//! requested freeze *at that point*: a hand-written guest stores `UNWINDING` into the
//! state word just before the chosen suspend op (the durable region is guest-addressable
//! in Phase 1 — that's the R9 hazard, used here deliberately). The poll after that op then
//! unwinds with the matching resume id.
//!
//! Because operations *after* the freeze point are genuinely re-performed on thaw (the
//! host clock is not in the artifact — D-scope), the oracle seeds the thaw host's clock to
//! the value the freeze host's clock had reached. A correct reload of the freeze-point
//! result still reproduces the uninterrupted run; a re-issue would consume a clock tick
//! and shift the result. The `Clock` advances by one per call, so this is exact.

use svm_durable::{
    init_durable_window, read_state, transform_module_assume_confined, write_state, STATE_NORMAL,
    STATE_REWINDING,
};
use svm_interp::{run_capture_reserved_with_host, Host, Value};
use svm_ir::{Memory, Module};

const SIZE_LOG2: u8 = 18;
const WINDOW: usize = 1 << SIZE_LOG2;

fn instrument(src: &str) -> Module {
    let mut m = svm_text::parse_module(src).expect("parse");
    m.memory = Some(Memory {
        size_log2: SIZE_LOG2,
    });
    // These guests deliberately store to the state word to simulate a host-requested freeze
    // at an interior point, which the R9 check would otherwise reject — so use the
    // confinement-assuming entry. (A real freeze trigger comes from the host, not the guest.)
    let inst = transform_module_assume_confined(&m).expect("transform");
    svm_verify::verify_module(&inst).expect("instrumented IR must verify");
    inst
}

/// Run the entry (func 0) against `window` with the clock seeded to `clock_ns`. Returns
/// the result, the final window image, and the host's final `clock_ns` (how far the
/// monotonic clock advanced — used to seed the continuation host on thaw).
fn run(inst: &Module, clock_ns: i64, window: &[u8]) -> (Vec<Value>, Vec<u8>, i64) {
    let mut host = Host::new();
    host.clock_ns = clock_ns;
    let clk = host.grant_clock();
    let mut fuel = 1_000_000u64;
    let (r, win) = run_capture_reserved_with_host(
        inst,
        0,
        &[Value::I32(clk)],
        &mut fuel,
        window,
        SIZE_LOG2,
        &mut host,
    );
    (r.expect("runs to completion"), win, host.clock_ns)
}

/// `oracle` is the unmodified guest (no self-flip store); `freezable` is the same guest
/// with a `store UNWINDING` inserted just before the suspend point we want to freeze at.
/// They compute the same result (the store only touches the state word). Assert: thaw of
/// the frozen `freezable` artifact, on a clock-continuation host, equals the oracle.
fn assert_resume_at_point(oracle: &str, freezable: &str, expected: i64) {
    let oracle = instrument(oracle);
    let freezable = instrument(freezable);

    // Baseline: the uninterrupted oracle run, clock from 42.
    let (baseline, _, _) = run(&oracle, 42, &init_durable_window(WINDOW));
    assert_eq!(baseline, vec![Value::I64(expected)], "oracle result");

    // Freeze: run the self-flipping variant from a NORMAL window; it flips to UNWINDING at
    // the chosen point and unwinds there. Record how far the clock advanced.
    let (_, snapshot, clock_after) = run(&freezable, 42, &init_durable_window(WINDOW));
    assert_ne!(
        read_state(&snapshot),
        STATE_NORMAL,
        "the guest flipped the state word and unwound (did not complete normally)"
    );

    // Thaw: restore the artifact, set REWINDING, and continue the clock from where freeze
    // left off (D-scope: the host clock is not in the artifact).
    let mut win = snapshot.clone();
    write_state(&mut win, STATE_REWINDING);
    let (thawed, final_win, _) = run(&freezable, clock_after, &win);
    assert_eq!(
        thawed, baseline,
        "thaw at the frozen resume point equals the oracle"
    );
    assert_eq!(read_state(&final_win), STATE_NORMAL, "thaw ends NORMAL");
}

// Two leaf cap.calls. Oracle: v2 = clock(42), v3 = clock(43), v4 = v2 + v3 = 85.
const ORACLE_2: &str = r#"
func (i32) -> (i64) {
block0(v0: i32):
  v1 = i32.const 0
  v2 = cap.call 2 0 (i32) -> (i64) v0 (v1)
  v3 = cap.call 2 0 (i32) -> (i64) v0 (v1)
  v4 = i64.add v2 v3
  return v4
}
"#;

// Freeze at point 0: flip UNWINDING before the *first* cap.call. (`v_a`/`v_u` are the
// state-word address and the UNWINDING constant.)
const FREEZE_AT_0: &str = r#"
func (i32) -> (i64) {
block0(v0: i32):
  v1 = i32.const 0
  v_a = i64.const 0
  v_u = i32.const 1
  i32.store v_a v_u
  v2 = cap.call 2 0 (i32) -> (i64) v0 (v1)
  v3 = cap.call 2 0 (i32) -> (i64) v0 (v1)
  v4 = i64.add v2 v3
  return v4
}
"#;

// Freeze at point 1: the first cap.call runs NORMAL, then flip UNWINDING before the second.
const FREEZE_AT_1: &str = r#"
func (i32) -> (i64) {
block0(v0: i32):
  v1 = i32.const 0
  v2 = cap.call 2 0 (i32) -> (i64) v0 (v1)
  v_a = i64.const 0
  v_u = i32.const 1
  i32.store v_a v_u
  v3 = cap.call 2 0 (i32) -> (i64) v0 (v1)
  v4 = i64.add v2 v3
  return v4
}
"#;

#[test]
fn resume_at_first_of_two_points() {
    // Freeze at point 0 (resume id 1, ARM_0): v2=42 reloaded; v3 re-performed at clock 43.
    assert_resume_at_point(ORACLE_2, FREEZE_AT_0, 85);
}

#[test]
fn resume_at_second_of_two_points() {
    // Freeze at point 1 (resume id 2, ARM_1): v2=42 already done & spilled; v3=43 reloaded.
    // Exercises the interior br_table arm — the case the single-point transform never hit.
    assert_resume_at_point(ORACLE_2, FREEZE_AT_1, 85);
}

// Three leaf cap.calls, distinct weights so a misrouted arm shifts the total.
// Oracle: clock 42,43,44 → 42 + 43*10 + 44*100 = 4872.
const ORACLE_3: &str = r#"
func (i32) -> (i64) {
block0(v0: i32):
  v1 = i32.const 0
  v2 = cap.call 2 0 (i32) -> (i64) v0 (v1)
  v3 = cap.call 2 0 (i32) -> (i64) v0 (v1)
  v4 = cap.call 2 0 (i32) -> (i64) v0 (v1)
  v5 = i64.const 10
  v6 = i64.const 100
  v7 = i64.mul v3 v5
  v8 = i64.mul v4 v6
  v9 = i64.add v2 v7
  v10 = i64.add v9 v8
  return v10
}
"#;

// Freeze at point 2 (the third cap.call): resume id 3, ARM_2.
const FREEZE_AT_2: &str = r#"
func (i32) -> (i64) {
block0(v0: i32):
  v1 = i32.const 0
  v2 = cap.call 2 0 (i32) -> (i64) v0 (v1)
  v3 = cap.call 2 0 (i32) -> (i64) v0 (v1)
  v_a = i64.const 0
  v_u = i32.const 1
  i32.store v_a v_u
  v4 = cap.call 2 0 (i32) -> (i64) v0 (v1)
  v5 = i64.const 10
  v6 = i64.const 100
  v7 = i64.mul v3 v5
  v8 = i64.mul v4 v6
  v9 = i64.add v2 v7
  v10 = i64.add v9 v8
  return v10
}
"#;

#[test]
fn resume_at_third_of_three_points() {
    assert_resume_at_point(ORACLE_3, FREEZE_AT_2, 4872);
}

// An *interior propagated* resume point: func 0 (A) calls B then C (both leaves). Freeze
// at the second call (point 1) ⇒ A's ARM_1 takes the re-issue path: B already completed
// (its result spilled & reloaded, B not re-run), C is re-issued and rewinds its own frame.
// Oracle: B = clock(42)+1000 = 1042; C = clock(43)+1 = 44; A = 1086.
const ORACLE_PROP: &str = r#"
func (i32) -> (i64) {
block0(v0: i32):
  v1 = call 1 (v0)
  v2 = call 2 (v0)
  v3 = i64.add v1 v2
  return v3
}
func (i32) -> (i64) {
block0(v0: i32):
  v1 = i32.const 0
  v2 = cap.call 2 0 (i32) -> (i64) v0 (v1)
  v3 = i64.const 1000
  v4 = i64.add v2 v3
  return v4
}
func (i32) -> (i64) {
block0(v0: i32):
  v1 = i32.const 0
  v2 = cap.call 2 0 (i32) -> (i64) v0 (v1)
  v3 = i64.const 1
  v4 = i64.add v2 v3
  return v4
}
"#;

// Same, but flip UNWINDING after the first call returns, so A freezes at the call to C.
const FREEZE_PROP_AT_1: &str = r#"
func (i32) -> (i64) {
block0(v0: i32):
  v1 = call 1 (v0)
  v_a = i64.const 0
  v_u = i32.const 1
  i32.store v_a v_u
  v2 = call 2 (v0)
  v3 = i64.add v1 v2
  return v3
}
func (i32) -> (i64) {
block0(v0: i32):
  v1 = i32.const 0
  v2 = cap.call 2 0 (i32) -> (i64) v0 (v1)
  v3 = i64.const 1000
  v4 = i64.add v2 v3
  return v4
}
func (i32) -> (i64) {
block0(v0: i32):
  v1 = i32.const 0
  v2 = cap.call 2 0 (i32) -> (i64) v0 (v1)
  v3 = i64.const 1
  v4 = i64.add v2 v3
  return v4
}
"#;

#[test]
fn resume_at_interior_propagated_point() {
    assert_resume_at_point(ORACLE_PROP, FREEZE_PROP_AT_1, 1086);
}
