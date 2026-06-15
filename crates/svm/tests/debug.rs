//! Interpreter-rooted debugger (DEBUGGING.md W2/W8, Milestone A slice 1): breakpoints, single
//! stepping, IR-value / window inspection, and a backtrace across a call — driven through the
//! host-side `Inspector`. These pin the S4 per-op seam + S5 driver model end-to-end.

use svm_interp::{Inspector, IrPc, Stop, StopReason, Value};
use svm_text::parse_module;

// sum = 1 + 2 + ... + N via a back-edge loop with block parameters (same shape as pipeline.rs).
const LOOP_SUM: &str = r#"
func (i32) -> (i32) {
block0(v0: i32):
  v1 = i32.const 0
  br block1(v0, v1)
block1(v2: i32, v3: i32):
  v4 = i32.add v3 v2
  v5 = i32.const -1
  v6 = i32.add v2 v5
  br_if v6 block1(v6, v4) block2(v4)
block2(v7: i32):
  return v7
}
"#;

// A callee + caller, to exercise a multi-frame backtrace.
const CALLER: &str = r#"
func (i32) -> (i32) {
block0(v0: i32):
  v1 = i32.const 7
  v2 = call 1 (v0)
  return v2
}
func (i32) -> (i32) {
block0(v0: i32):
  v1 = i32.const 10
  v2 = i32.add v0 v1
  return v2
}
"#;

fn finished_ok(stop: Stop) -> Vec<Value> {
    match stop {
        Stop::Finished(Ok(vals)) => vals,
        other => panic!("expected Finished(Ok), got {other:?}"),
    }
}

#[test]
fn runs_to_completion_with_no_breakpoints() {
    // An attached-but-unconstrained run matches a plain `run` (S7: debug off-path is transparent).
    let m = parse_module(LOOP_SUM).expect("parse");
    let mut insp = Inspector::attach(&m, 0, &[Value::I32(5)], 1_000_000);
    let out = finished_ok(insp.run_until_stop());
    assert_eq!(out, vec![Value::I32(15)]); // 1+2+3+4+5
    assert_eq!(insp.result(), Some(&Ok(vec![Value::I32(15)])));
    assert!(insp.clock() > 0, "executed ops should advance logical time");
}

#[test]
fn breakpoint_stops_at_the_loop_body_each_iteration() {
    let m = parse_module(LOOP_SUM).expect("parse");
    let mut insp = Inspector::attach(&m, 0, &[Value::I32(3)], 1_000_000);
    // Break at the accumulate op (block1, inst 0: `v4 = i32.add v3 v2`).
    let bp = IrPc { module: 0, func: 0, block: 1, inst: 0 };
    insp.set_breakpoint(bp);

    // N = 3 ⇒ the loop body runs three times, so we hit the breakpoint three times.
    let mut accs = Vec::new();
    for _ in 0..3 {
        match insp.run_until_stop() {
            Stop::Break { reason: StopReason::Breakpoint, pc } => {
                assert_eq!(pc, bp);
                // v2 (counter) and v3 (running accumulator) are this block's params: vals[0], vals[1].
                let v2 = insp.read_ir_value(0, 0).expect("v2");
                let v3 = insp.read_ir_value(0, 1).expect("v3");
                accs.push((v2, v3));
            }
            other => panic!("expected breakpoint, got {other:?}"),
        }
    }
    // Counter walks 3,2,1; accumulator walks 0,3,5.
    assert_eq!(
        accs,
        vec![
            (Value::I32(3), Value::I32(0)),
            (Value::I32(2), Value::I32(3)),
            (Value::I32(1), Value::I32(5)),
        ]
    );
    // After the third hit, removing the breakpoint lets it finish: 1+2+3 = 6.
    assert!(insp.clear_breakpoint(bp));
    let out = finished_ok(insp.run_until_stop());
    assert_eq!(out, vec![Value::I32(6)]);
}

#[test]
fn single_step_advances_exactly_one_op_and_ticks_the_clock() {
    let m = parse_module(LOOP_SUM).expect("parse");
    let mut insp = Inspector::attach(&m, 0, &[Value::I32(2)], 1_000_000);

    // block0 has a single instruction (`v1 = i32.const 0`); the `br` is the block *terminator*,
    // not an `insts` op, so it isn't a hookable step point. Stepping one op therefore runs the
    // const and the branch, landing before block1's first op. The clock counts non-terminator ops.
    let before = insp.clock();
    match insp.step() {
        Stop::Break { reason: StopReason::Step, pc } => {
            assert_eq!(pc, IrPc { module: 0, func: 0, block: 1, inst: 0 });
        }
        other => panic!("expected step stop, got {other:?}"),
    }
    assert_eq!(insp.clock(), before + 1, "exactly one (non-terminator) op executed");
    // In block1 the frame's values are its params v2 (counter) and v3 (accumulator) = N, 0.
    assert_eq!(insp.read_ir_value(0, 1), Some(Value::I32(0)));

    // A handful more steps stay single-op; the clock advances monotonically.
    let mut last = insp.clock();
    for _ in 0..4 {
        if let Stop::Break { .. } = insp.step() {
            assert_eq!(insp.clock(), last + 1);
            last = insp.clock();
        } else {
            break;
        }
    }
}

#[test]
fn backtrace_shows_both_frames_inside_the_callee() {
    let m = parse_module(CALLER).expect("parse");
    let mut insp = Inspector::attach(&m, 0, &[Value::I32(5)], 1_000_000);
    // Break on the callee's add (func 1, block0, inst1: `v2 = i32.add v0 v1`).
    let bp = IrPc { module: 0, func: 1, block: 0, inst: 1 };
    insp.set_breakpoint(bp);
    match insp.run_until_stop() {
        Stop::Break { pc, .. } => assert_eq!(pc, bp),
        other => panic!("expected callee breakpoint, got {other:?}"),
    }
    let bt = insp.backtrace();
    assert_eq!(bt.len(), 2, "callee + caller frames");
    // Innermost frame is the callee (func 1); the caller (func 0) is paused at its `call` op.
    assert_eq!(bt[0].pc.func, 1);
    assert_eq!(bt[1].pc.func, 0);
    // The callee's arg arrived as v0 = 5; v1 = 10 was just produced.
    assert_eq!(insp.read_ir_value(0, 0), Some(Value::I32(5)));
    assert_eq!(insp.read_ir_value(0, 1), Some(Value::I32(10)));

    // Let it finish: callee returns 5 + 10 = 15.
    assert!(insp.clear_breakpoint(bp));
    assert_eq!(finished_ok(insp.run_until_stop()), vec![Value::I32(15)]);
}

#[test]
fn reads_window_memory_a_store_left_behind() {
    // Store a known i64 at offset 0, then return — and read it back via the Inspector.
    const MAGIC: u64 = 0x1122334455667788;
    let src = r#"
memory 17
func () -> (i32) {
block0():
  v0 = i64.const 0
  v1 = i64.const 1234605616436508552
  i64.store v0 v1
  v2 = i32.const 0
  return v2
}
"#;
    let m = parse_module(src).expect("parse");
    let mut insp = Inspector::attach(&m, 0, &[], 1_000_000);
    let _ = finished_ok(insp.run_until_stop());
    let bytes = insp.read_window(0, 8).expect("read window");
    assert_eq!(bytes, MAGIC.to_le_bytes());
}
