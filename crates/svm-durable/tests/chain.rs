//! Multi-frame call-chain freeze/thaw on the real interpreter (DURABILITY.md §12.7, R8).
//!
//! A call chain `A → B → … → leaf cap.call` stacks one shadow frame per suspended
//! activation. On thaw the chain rewinds outside-in: each non-deepest frame reloads its
//! pre-call live set and **re-issues its call** (leaving the state word `REWINDING`), and
//! only the innermost leaf reloads the host-produced `cap.call` result and flips the
//! state back to `NORMAL`. The property is the same as the single-frame case — thaw on a
//! *fresh* host equals the uninterrupted run — but it now exercises the "re-issue vs.
//! continue" branch that the single-frame transform never hit.

use svm_durable::{
    init_durable_window, read_state, read_thaw_state, transform_module, write_state,
    begin_thaw, STATE_NORMAL, STATE_UNWINDING,
};
use svm_interp::{run_capture_reserved_with_host, Host, Value};
use svm_ir::{Memory, Module};

const SIZE_LOG2: u8 = 18; // 256 KiB window — ample room for a handful of stacked frames
const WINDOW: usize = 1 << SIZE_LOG2;

fn instrument(src: &str) -> Module {
    let mut m = svm_text::parse_module(src).expect("parse");
    m.memory = Some(Memory {
        size_log2: SIZE_LOG2,
    });
    let inst = transform_module(&m).expect("transform");
    svm_verify::verify_module(&inst).expect("instrumented IR must verify");
    inst
}

/// Run the entry (func 0) against `window` with the clock seeded to `clock_ns`.
fn run(
    inst: &Module,
    clock_ns: i64,
    window: &[u8],
) -> (Result<Vec<Value>, svm_interp::Trap>, Vec<u8>) {
    let mut host = Host::new();
    host.clock_ns = clock_ns;
    let clk = host.grant_clock();
    let mut fuel = 1_000_000u64;
    run_capture_reserved_with_host(
        inst,
        0,
        &[Value::I32(clk)],
        &mut fuel,
        window,
        SIZE_LOG2,
        &mut host,
    )
}

/// Drive the full property for `inst`: baseline (clock 42) vs. freeze→thaw on a fresh
/// host (clock 0). A correct thaw reloads the saved `42` at the leaf and reproduces the
/// baseline; a re-issue bug at the leaf — or a missing re-issue at a propagated frame —
/// would diverge. Returns the agreed result for the caller to pin.
fn assert_roundtrips(inst: &Module) -> Vec<Value> {
    let (baseline, _) = run(inst, 42, &init_durable_window(WINDOW));
    let baseline = baseline.expect("baseline runs to completion");

    // Freeze: same initial conditions, state UNWINDING.
    let mut win = init_durable_window(WINDOW);
    write_state(&mut win, STATE_UNWINDING);
    let (frozen, snapshot) = run(inst, 42, &win);
    assert!(frozen.is_ok(), "freeze returns a placeholder, not a trap");
    assert_eq!(
        read_state(&snapshot),
        STATE_UNWINDING,
        "artifact is still UNWINDING (the whole chain unwound, none completed)"
    );

    // Thaw on a fresh host (clock now 0): reload, do not re-issue the cap.call.
    let mut win = snapshot.clone();
    begin_thaw(&mut win, 0); // §12.8 stage 1: thaw the root (ctx 0) per-context
    let (thawed, final_win) = run(inst, 0, &win);
    assert_eq!(
        thawed,
        Ok(baseline.clone()),
        "thaw equals the uninterrupted run"
    );
    assert_eq!(
        read_thaw_state(&final_win, 0),
        STATE_NORMAL,
        "the deepest frame flipped the state back to NORMAL exactly once"
    );
    baseline
}

// Dead values across a `cap.call`: `v2`/`v3` are computed before the call but never used
// after it, while `v1` is. The minimal live-set must spill only `v1` (+ the call result)
// and drop `v2`/`v3`, yet the thaw must still reproduce the run. Baseline: 42 + 10 = 52.
const DEAD_VALUES: &str = r#"
func (i32) -> (i64) {
block0(v0: i32):
  v1 = i64.const 10
  v2 = i64.const 20
  v3 = i64.const 30
  v4 = i32.const 0
  v5 = cap.call 2 0 (i32) -> (i64) v0 (v4)
  v6 = i64.add v5 v1
  return v6
}
"#;

#[test]
fn dead_values_across_cap_call_are_dropped() {
    let inst = instrument(DEAD_VALUES);
    assert_eq!(
        assert_roundtrips(&inst),
        vec![Value::I64(52)],
        "42 + 10; dead consts dropped"
    );
}

// A → B(leaf). A adds 1000 to B's result; B adds 100 to the clock. Baseline: 1142.
const TWO_LEVEL: &str = r#"
func (i32) -> (i64) {
block0(v0: i32):
  v1 = call 1 (v0)
  v2 = i64.const 1000
  v3 = i64.add v1 v2
  return v3
}
func (i32) -> (i64) {
block0(v0: i32):
  v1 = i32.const 0
  v2 = cap.call 2 0 (i32) -> (i64) v0 (v1)
  v3 = i64.const 100
  v4 = i64.add v2 v3
  return v4
}
"#;

#[test]
fn two_level_chain_round_trips() {
    let inst = instrument(TWO_LEVEL);
    assert_eq!(
        assert_roundtrips(&inst),
        vec![Value::I64(1142)],
        "1000 + (100 + 42)"
    );
}

// A → B → C(leaf), each adding a distinct constant, so a dropped or doubly-applied frame
// would shift the total. Baseline: 42 + 1 + 20 + 300 = 363.
const THREE_LEVEL: &str = r#"
func (i32) -> (i64) {
block0(v0: i32):
  v1 = call 1 (v0)
  v2 = i64.const 300
  v3 = i64.add v1 v2
  return v3
}
func (i32) -> (i64) {
block0(v0: i32):
  v1 = call 2 (v0)
  v2 = i64.const 20
  v3 = i64.add v1 v2
  return v3
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
fn three_level_chain_round_trips() {
    let inst = instrument(THREE_LEVEL);
    assert_eq!(
        assert_roundtrips(&inst),
        vec![Value::I64(363)],
        "42 + 1 + 20 + 300"
    );
}

// A propagated frame with a non-empty pre-call live set: A computes a value *before* the
// call and uses it *after*, so that value must be spilled and reloaded across the freeze
// (the re-issue path must not clobber it). Baseline: leaf=42+5=47; A = 47 + (7*2) = 61.
const LIVE_ACROSS_CALL: &str = r#"
func (i32) -> (i64) {
block0(v0: i32):
  v1 = i64.const 7
  v2 = call 1 (v0)
  v3 = i64.add v1 v1
  v4 = i64.add v2 v3
  return v4
}
func (i32) -> (i64) {
block0(v0: i32):
  v1 = i32.const 0
  v2 = cap.call 2 0 (i32) -> (i64) v0 (v1)
  v3 = i64.const 5
  v4 = i64.add v2 v3
  return v4
}
"#;

#[test]
fn live_value_across_propagated_call_survives() {
    let inst = instrument(LIVE_ACROSS_CALL);
    assert_eq!(
        assert_roundtrips(&inst),
        vec![Value::I64(61)],
        "(42+5) + (7+7)"
    );
}
