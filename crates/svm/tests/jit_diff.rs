//! Differential testing: the Cranelift JIT vs the reference interpreter (`DESIGN.md`
//! §18, invariants I1/I4). For every supported program and input, the JIT's result
//! must equal the interpreter oracle's. This is the methodology the whole
//! escape-freedom argument leans on, so it is wired up alongside the very first JIT
//! slice and grows with the lowering.

use svm_interp::{run, Value};
use svm_ir::ValType;
use svm_jit::{compile_and_run, JitError};
use svm_text::parse_module;
use svm_verify::verify_module;

/// Marshal a `Value` into its `i64` calling-convention slot (§ JIT calling convention).
fn to_slot(v: Value) -> i64 {
    match v {
        Value::I32(x) => x as i64,
        Value::I64(x) => x,
        other => panic!("integer slice only: {other:?}"),
    }
}

/// Decode a result slot back to a `Value` given the declared result type.
fn from_slot(t: ValType, s: i64) -> Value {
    match t {
        ValType::I32 => Value::I32(s as i32),
        ValType::I64 => Value::I64(s),
        other => panic!("integer slice only: {other:?}"),
    }
}

/// Assert the JIT and the interpreter agree on `src(args)` for every input row.
fn assert_jit_matches_interp(src: &str, inputs: &[Vec<Value>]) {
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");
    let results_ty = m.funcs[0].results.clone();

    for args in inputs {
        let mut fuel = 10_000_000u64;
        let want = run(&m, 0, args, &mut fuel).expect("interp ok");

        let slots: Vec<i64> = args.iter().copied().map(to_slot).collect();
        let got_slots = match compile_and_run(&m, 0, &slots) {
            Ok(s) => s,
            Err(JitError::Unsupported(_)) => return, // not in this slice yet — skip
            Err(e) => panic!("JIT failed: {e:?}"),
        };
        let got: Vec<Value> = results_ty
            .iter()
            .zip(got_slots)
            .map(|(t, s)| from_slot(*t, s))
            .collect();

        assert_eq!(
            want, got,
            "interp/JIT disagree on {src:?} for args {args:?}"
        );
    }
}

fn i32s(xs: &[i32]) -> Vec<Value> {
    xs.iter().map(|x| Value::I32(*x)).collect()
}

#[test]
fn jit_matches_interp_add() {
    let src = r#"
func (i32, i32) -> (i32) {
block0(v0: i32, v1: i32):
  v2 = i32.add v0 v1
  return v2
}
"#;
    assert_jit_matches_interp(
        src,
        &[
            i32s(&[2, 3]),
            i32s(&[-1, 1]),
            i32s(&[i32::MAX, 1]), // wraps to i32::MIN — must match the interp
            i32s(&[-5, -7]),
        ],
    );
}

#[test]
fn jit_matches_interp_arith_with_select() {
    // (v0 < v1) ? 100 : (v0 - v1)^2 — sub, mul, lt_s, select, const.
    let src = r#"
func (i32, i32) -> (i32) {
block0(v0: i32, v1: i32):
  v2 = i32.sub v0 v1
  v3 = i32.mul v2 v2
  v4 = i32.lt_s v0 v1
  v5 = i32.const 100
  v6 = select v4 v5 v3
  return v6
}
"#;
    assert_jit_matches_interp(
        src,
        &[
            i32s(&[1, 9]),
            i32s(&[9, 1]),
            i32s(&[5, 5]),
            i32s(&[-3, -10]),
        ],
    );
}

#[test]
fn jit_matches_interp_bitwise_and_shifts() {
    // Exercise and/or/xor/shl/shr_u/shr_s/rotl, incl. shift-count masking semantics.
    let src = r#"
func (i32, i32) -> (i32) {
block0(v0: i32, v1: i32):
  v2 = i32.and v0 v1
  v3 = i32.or v0 v1
  v4 = i32.xor v2 v3
  v5 = i32.shl v4 v1
  v6 = i32.shr_u v5 v1
  v7 = i32.shr_s v6 v1
  v8 = i32.rotl v7 v1
  return v8
}
"#;
    assert_jit_matches_interp(
        src,
        &[
            i32s(&[0x1234_5678u32 as i32, 3]),
            i32s(&[-1, 31]),
            i32s(&[-1, 33]), // shift count must be masked mod 32
            i32s(&[0xFF00FF00u32 as i32, 7]),
        ],
    );
}

#[test]
fn jit_matches_interp_comparisons_and_eqz() {
    let src = r#"
func (i32, i32) -> (i32) {
block0(v0: i32, v1: i32):
  v2 = i32.lt_u v0 v1
  v3 = i32.ge_s v0 v1
  v4 = i32.eqz v2
  v5 = i32.add v3 v4
  return v5
}
"#;
    assert_jit_matches_interp(
        src,
        &[i32s(&[1, 2]), i32s(&[-1, 1]), i32s(&[5, 5]), i32s(&[0, 0])],
    );
}

#[test]
fn jit_matches_interp_i64_ops() {
    let src = r#"
func (i64, i64) -> (i64) {
block0(v0: i64, v1: i64):
  v2 = i64.mul v0 v1
  v3 = i64.sub v2 v1
  return v3
}
"#;
    assert_jit_matches_interp(
        src,
        &[
            vec![Value::I64(1_000_000), Value::I64(1_000_000)],
            vec![Value::I64(-3), Value::I64(7)],
            vec![Value::I64(i64::MAX), Value::I64(2)], // overflow wraps
        ],
    );
}

#[test]
fn jit_matches_interp_loop_with_back_edge() {
    // sum = 1 + 2 + ... + n via a back-edge loop with block parameters — exercises
    // br / br_if and multi-block SSA lowering.
    let src = r#"
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
    let inputs: Vec<Vec<Value>> = [1, 2, 5, 10, 100, 1000]
        .iter()
        .map(|n| i32s(&[*n]))
        .collect();
    assert_jit_matches_interp(src, &inputs);
}

#[test]
fn jit_matches_interp_mem_store_load_roundtrip() {
    // Store an i64 at the given address, read it back — exercises the §4 masking
    // lowering (mem_base + ((addr+offset) & mask)) against the interpreter.
    let src = r#"
memory 16

func (i64, i64) -> (i64) {
block0(v0: i64, v1: i64):
  i64.store v0 v1
  v2 = i64.load v0
  return v2
}
"#;
    assert_jit_matches_interp(
        src,
        &[
            vec![Value::I64(0), Value::I64(0x0123_4567_89AB_CDEF)],
            vec![Value::I64(64), Value::I64(-1)],
            vec![Value::I64(65528), Value::I64(42)], // last aligned 8-byte slot (2^16-8)
        ],
    );
}

#[test]
fn jit_matches_interp_mem_narrow_store_load() {
    // store8 keeps the low byte; load8_u zero-extends, load8_s sign-extends.
    let src = r#"
memory 16

func (i64, i32) -> (i32) {
block0(v0: i64, v1: i32):
  i32.store8 v0 v1
  v2 = i32.load8_u v0
  v3 = i32.load8_s v0
  v4 = i32.add v2 v3
  return v4
}
"#;
    assert_jit_matches_interp(
        src,
        &[
            vec![Value::I64(0), Value::I32(0x1FF)], // truncates to 0xFF
            vec![Value::I64(10), Value::I32(0x80)], // 128: u=128, s=-128
            vec![Value::I64(7), Value::I32(0x41)],  // 'A'
        ],
    );
}

#[test]
fn jit_matches_interp_mem_masking_aliases_out_of_window() {
    // I1: an out-of-window address must alias back via the mask, identically in the
    // JIT and the interpreter. Store at offset 8, read at (2^16 + 8) — same cell.
    let src = r#"
memory 16

func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const 8
  i64.store v1 v0
  v2 = i64.const 65544
  v3 = i64.load v2
  return v3
}
"#;
    assert_jit_matches_interp(src, &[vec![Value::I64(0xDEAD_BEEF)], vec![Value::I64(-99)]]);
}

#[test]
fn jit_matches_interp_no_args_const() {
    let src = r#"
func () -> (i32) {
block0():
  v0 = i32.const 42
  return v0
}
"#;
    assert_jit_matches_interp(src, &[vec![]]);
}
