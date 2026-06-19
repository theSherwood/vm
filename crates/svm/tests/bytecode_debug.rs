//! Equality harness for the bytecode engine's **debug seam** (INTERP_PERF.md Slice 1c-3): the
//! `pc → {block, inst}` reverse map that lets the bytecode engine report tree-walker-identical
//! [`IrPc`] locations for stepping and breakpoints.
//!
//! The tree-walker's debug seam (`run_inner`'s `before_op`) stops only at **instructions**, never
//! terminators, and its logical clock counts one tick per instruction — so driving the
//! [`Inspector`] with `seek(0), seek(1), …` yields the exact sequence of instruction locations the
//! guest executes. The bytecode engine's [`bytecode::ir_trace`] single-steps (`budget = 1`) and
//! records the `IrPc` of each instruction op (skipping terminator ops via the [`Program::src`] map).
//! This harness asserts the two sequences are **identical** op-for-op — and the results match — so a
//! breakpoint/step at any `IrPc` lands at the same program point on both backends.

use svm_interp::{bytecode, Inspector, IrPc, Stop, Trap, Value};
use svm_text::parse_module;

/// The instruction-location trace the tree-walker executes, via the `Inspector`'s logical-time seek:
/// `seek(t)` pauses *before* the instruction at clock `t` (terminators don't tick), so iterating `t`
/// upward enumerates every executed instruction's `IrPc`, then `Finished` gives the result.
fn tw_trace(src: &str, args: &[Value], fuel: u64) -> (Vec<IrPc>, Result<Vec<Value>, Trap>) {
    let m = parse_module(src).expect("parse");
    let mut insp = Inspector::attach(&m, 0, args, fuel);
    let mut trace = Vec::new();
    let mut t = 0u64;
    loop {
        match insp.seek(t) {
            Stop::Break { pc, .. } => {
                trace.push(pc);
                t += 1;
            }
            Stop::Finished(r) => return (trace, r),
            Stop::Blocked => panic!("single-threaded debug-scope program must not block"),
        }
    }
}

fn check(src: &str, args: &[Value]) {
    let (tw_pcs, tw_res) = tw_trace(src, args, 5_000_000);

    let m = parse_module(src).expect("parse");
    let mut fuel = 5_000_000u64;
    let (bc_pcs, bc_res) =
        bytecode::ir_trace(&m, 0, args, &mut fuel).expect("bytecode engine must drive the module");

    assert_eq!(
        tw_pcs, bc_pcs,
        "debug location trace: tree-walker != bytecode\n{src}"
    );
    assert_eq!(
        tw_res, bc_res,
        "debug run result: tree-walker != bytecode\n{src}"
    );
}

/// Straight-line block: every instruction is a distinct step in `block0`, then the terminator
/// (skipped). Pins the basic `(block, inst)` numbering.
const STRAIGHT: &str = r#"func (i32) -> (i32) {
block0(v0: i32):
  v1 = i32.const 10
  v2 = i32.add v0 v1
  v3 = i32.const 2
  v4 = i32.mul v2 v3
  return v4
}
"#;

#[test]
fn straight_line_locations() {
    check(STRAIGHT, &[Value::I32(5)]);
}

/// A branch: the trace must cross block boundaries (the terminator is silent, the next location is
/// the target block's first instruction) and pick the taken arm.
const BRANCH: &str = r#"func (i32) -> (i32) {
block0(v0: i32):
  v1 = i32.eqz v0
  br_if v1 block1(v0) block2(v0)
block1(v2: i32):
  v3 = i32.const 100
  return v3
block2(v4: i32):
  v5 = i32.const 200
  v6 = i32.add v4 v5
  return v6
}
"#;

#[test]
fn branch_locations() {
    check(BRANCH, &[Value::I32(0)]); // takes block1
    check(BRANCH, &[Value::I32(7)]); // takes block2
}

/// A loop: the same block's instructions recur, so the location trace revisits the same `IrPc`s once
/// per iteration — the clearest test that stepping granularity matches across back-edges.
const LOOP: &str = r#"func (i32) -> (i32) {
block0(v0: i32):
  v1 = i32.const 0
  br block1(v0, v1)
block1(v2: i32, v3: i32):
  v4 = i32.eqz v2
  br_if v4 block2(v3) block3(v2, v3)
block2(v5: i32):
  return v5
block3(v6: i32, v7: i32):
  v8 = i32.const -1
  v9 = i32.add v6 v8
  v10 = i32.add v7 v6
  br block1(v9, v10)
}
"#;

#[test]
fn loop_locations() {
    check(LOOP, &[Value::I32(5)]); // sum 5+4+3+2+1 = 15
}

/// A direct call: the trace must descend into the callee (its `func` index in the `IrPc`) and return
/// to the instruction after the call — exercising the cross-frame location reporting.
const CALL: &str = r#"func (i32) -> (i32) {
block0(v0: i32):
  v1 = call 1(v0)
  v2 = i32.const 1
  v3 = i32.add v1 v2
  return v3
}
func (i32) -> (i32) {
block0(v0: i32):
  v1 = i32.const 3
  v2 = i32.mul v0 v1
  return v2
}
"#;

#[test]
fn call_locations() {
    check(CALL, &[Value::I32(4)]); // 4*3 + 1 = 13
}

/// A trapping run: the faulting instruction's location is recorded on both engines, then the trap is
/// the result — stepping observes the same final program point.
const TRAP: &str = r#"func (i32) -> (i32) {
block0(v0: i32):
  v1 = i32.const 0
  v2 = i32.div_s v0 v1
  return v2
}
"#;

#[test]
fn trap_locations() {
    check(TRAP, &[Value::I32(10)]);
}
