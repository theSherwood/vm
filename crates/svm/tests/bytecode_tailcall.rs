//! Equality harness for the bytecode engine's **tail calls** (INTERP_PERF.md Slice 1c-5e/3):
//! `return_call` / `return_call_indirect`. The generator never emits them, so these are hand-authored
//! (adapted from `jit_diff`'s tail-call cases) and checked bit-identical to the tree-walker `run`.
//! Tail calls reuse the current activation window, so deep tail recursion must not grow state.

use svm_interp::{bytecode, run, Trap, Value};
use svm_text::parse_module;

fn check(src: &str, args: &[Value], want: Result<i32, Trap>) {
    let m = parse_module(src).expect("parse");
    let mut f_tw = 10_000_000u64;
    let tw = run(&m, 0, args, &mut f_tw);
    let mut f_bc = 10_000_000u64;
    let bc = bytecode::compile_and_run(&m, 0, args, &mut f_bc)
        .expect("bytecode engine must support tail calls (Slice 1c-5e/3)");
    assert_eq!(tw, bc, "tail call: tree-walker != bytecode\n{src}");
    let got = bc.map(|v| match v.first() {
        Some(Value::I32(x)) => *x,
        other => panic!("unexpected result {other:?}"),
    });
    assert_eq!(got, want, "tail call result\n{src}");
}

/// Tail-recursive factorial accumulator `f(n, acc) = n==0 ? acc : f(n-1, acc*n)` via `return_call` —
/// must run in O(1) state (window reuse). `f(5, 1) = 120`.
const TAIL_RECURSION: &str = r#"
func (i32, i32) -> (i32) {
block 0 (v0: i32, v1: i32) {
  v2 = i32.eqz v0
  br_if v2 1(v1) 2(v0, v1)
}
block 1 (v3: i32) {
  return v3
}
block 2 (v4: i32, v5: i32) {
  v6 = i32.mul v5 v4
  v7 = i32.const -1
  v8 = i32.add v4 v7
  return_call 0(v8, v6)
  }
}
"#;

/// `return_call_indirect` through the natural table: `f(idx, x)` tail-calls func `idx`. idx 1 = `+10`,
/// idx 2 = `*2`; idx 0 selects func 0 (the wrong signature) → `IndirectCallType` trap.
const TAIL_INDIRECT: &str = r#"
func (i32, i32) -> (i32) {
block 0 (v0: i32, v1: i32) {
  return_call_indirect (i32) -> (i32) v0 (v1)
  }
}
func (i32) -> (i32) {
block 0 (v0: i32) {
  v1 = i32.const 10
  v2 = i32.add v0 v1
  return v2
  }
}
func (i32) -> (i32) {
block 0 (v0: i32) {
  v1 = i32.const 2
  v2 = i32.mul v0 v1
  return v2
  }
}
"#;

#[test]
fn tail_recursion() {
    check(TAIL_RECURSION, &[Value::I32(5), Value::I32(1)], Ok(120));
    check(TAIL_RECURSION, &[Value::I32(0), Value::I32(7)], Ok(7));
}

/// Deep tail recursion must not exhaust state — window reuse keeps it O(1). `f(100000, 1)` overflows
/// i32 but completes (the point is it doesn't grow the stack); just assert both engines agree.
#[test]
fn tail_recursion_deep() {
    let m = parse_module(TAIL_RECURSION).expect("parse");
    let args = [Value::I32(100_000), Value::I32(1)];
    let mut f_tw = 50_000_000u64;
    let tw = run(&m, 0, &args, &mut f_tw);
    let mut f_bc = 50_000_000u64;
    let bc = bytecode::compile_and_run(&m, 0, &args, &mut f_bc).expect("supported");
    assert_eq!(tw, bc, "deep tail recursion: tree-walker != bytecode");
}

#[test]
fn tail_call_indirect() {
    check(TAIL_INDIRECT, &[Value::I32(1), Value::I32(5)], Ok(15));
    check(TAIL_INDIRECT, &[Value::I32(2), Value::I32(5)], Ok(10));
    check(
        TAIL_INDIRECT,
        &[Value::I32(0), Value::I32(5)],
        Err(Trap::IndirectCallType),
    );
}
