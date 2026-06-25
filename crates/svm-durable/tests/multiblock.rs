//! Multi-block CFGs: a suspend op in a non-entry block, across branches and loops
//! (DURABILITY.md §9 — the last Phase-1 transform extension).
//!
//! Values are block-local and cross blocks only as branch arguments, so the live set at a
//! resume point is exactly its block's params + the locals before the op — all spilled and
//! reloaded on thaw, with the segment tail recomputing the rest. The transform splits each
//! original block at its suspend ops and remaps every branch target to the target block's
//! first segment. These tests freeze a frozen domain whose in-flight `cap.call` lives in a
//! *non-entry* block (so the carried block params, not the function params, must survive),
//! and resume across a conditional join and around a loop back-edge.
//!
//! As in `multipoint.rs`, the thaw host's clock continues from where freeze left off
//! (D-scope), so a loop body that re-reads the clock after the resume matches the oracle
//! while the frozen reading is reloaded.

use svm_durable::{
    init_durable_window, read_state, read_thaw_state, transform_module, write_state, begin_thaw, STATE_NORMAL,
    STATE_UNWINDING,
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
    let inst = transform_module(&m).expect("transform");
    svm_verify::verify_module(&inst).expect("instrumented IR must verify");
    inst
}

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

/// Baseline (clock 42) vs. freeze-from-start → thaw on a clock-continuation host.
fn assert_roundtrips(src: &str, expected: i64) {
    let inst = instrument(src);

    let (baseline, _, _) = run(&inst, 42, &init_durable_window(WINDOW));
    assert_eq!(baseline, vec![Value::I64(expected)], "uninterrupted run");

    let mut win = init_durable_window(WINDOW);
    write_state(&mut win, STATE_UNWINDING);
    let (_, snapshot, clock_after) = run(&inst, 42, &win);
    assert_eq!(
        read_state(&snapshot),
        STATE_UNWINDING,
        "froze, did not complete"
    );

    let mut win = snapshot.clone();
    begin_thaw(&mut win, 0);
    let (thawed, final_win, _) = run(&inst, clock_after, &win);
    assert_eq!(
        thawed, baseline,
        "thaw of a non-entry-block freeze equals the oracle"
    );
    assert_eq!(read_thaw_state(&final_win, 0), STATE_NORMAL, "thaw ends NORMAL");
}

// A conditional whose taken arm holds the cap.call: the suspend point is in block1, a
// *non-entry* block, so block1's param (the handle, carried as a branch arg) must be
// spilled and reloaded — it cannot be recovered from the function entry. Baseline: 142.
const COND: &str = r#"
func (i32) -> (i64) {
block0(v0: i32):
  v1 = i32.const 1
  br_if v1 block1(v0) block2(v0)
block1(v0: i32):
  v2 = i32.const 0
  v3 = cap.call 2 0 (i32) -> (i64) v0 (v2)
  v4 = i64.const 100
  v5 = i64.add v3 v4
  return v5
block2(v0: i32):
  v6 = i64.const 999
  return v6
}
"#;

#[test]
fn suspend_in_conditional_branch_round_trips() {
    assert_roundtrips(COND, 142);
}

// A loop that reads the clock each iteration and accumulates. The cap.call is in the loop
// header (block1), reached via a back-edge; the loop-carried accumulator and counter flow
// as block params and must survive the freeze. Clock 42,43,44 over three iterations ⇒
// 42 + 43 + 44 = 129. Freeze lands on iteration 0; the continuation clock makes iterations
// 1 and 2 reproduce the oracle while iteration 0's reading is reloaded.
const LOOP: &str = r#"
func (i32) -> (i64) {
block0(v0: i32):
  v1 = i64.const 0
  v2 = i64.const 0
  br block1(v0, v1, v2)
block1(v0: i32, v1: i64, v2: i64):
  v3 = i32.const 0
  v4 = cap.call 2 0 (i32) -> (i64) v0 (v3)
  v5 = i64.add v1 v4
  v6 = i64.const 1
  v7 = i64.add v2 v6
  v8 = i64.const 3
  v9 = i64.lt_s v7 v8
  br_if v9 block1(v0, v5, v7) block2(v5)
block2(v10: i64):
  return v10
}
"#;

#[test]
fn suspend_in_loop_body_round_trips() {
    assert_roundtrips(LOOP, 129);
}
