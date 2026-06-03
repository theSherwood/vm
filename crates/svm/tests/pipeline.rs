//! End-to-end pipeline + differential tests for the Phase-1 slice.
//!
//! Proves the loop closes: `text -> binary -> verify -> interp`, that binary and
//! text encodings round-trip to identical IR, and that interpreting the text form
//! and the decoded binary form agree (the differential property the JIT will later
//! be held to against this same interpreter, §18).

use svm::{assemble, load};
use svm_encode::{decode_module, encode_module};
use svm_interp::{run, Trap, Value};
use svm_ir::{BinOp, Inst, IntTy};
use svm_text::{parse_module, print_module};
use svm_verify::{verify_module, VerifyError};

const ADD: &str = r#"
func (i32, i32) -> (i32) {
block0(v0: i32, v1: i32):
  v2 = i32.add v0 v1
  return v2
}
"#;

const CONST42: &str = r#"
func () -> (i32) {
block0():
  v0 = i32.const 42
  return v0
}
"#;

// sum = 1 + 2 + ... + N  (N >= 1), via a back-edge loop with block parameters.
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

// (v0 < v1) ? 100 : (v0 - v1)^2  — exercises sub/mul/lt_s/select/const.
const ARITH: &str = r#"
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

// sign-extend i32 -> i64, then add a large i64 constant.
const CONV: &str = r#"
func (i32) -> (i64) {
block0(v0: i32):
  v1 = i64.extend_i32_s v0
  v2 = i64.const 1000000000000
  v3 = i64.add v1 v2
  return v3
}
"#;

const DIV: &str = r#"
func (i32, i32) -> (i32) {
block0(v0: i32, v1: i32):
  v2 = i32.div_s v0 v1
  return v2
}
"#;

// br_table: idx selects 10/20/30, else default 99.
const BRTABLE: &str = r#"
func (i32) -> (i32) {
block0(v0: i32):
  br_table v0 [block1(), block2(), block3()] block4()
block1():
  v1 = i32.const 10
  return v1
block2():
  v2 = i32.const 20
  return v2
block3():
  v3 = i32.const 30
  return v3
block4():
  v4 = i32.const 99
  return v4
}
"#;

const CORPUS: &[&str] = &[ADD, CONST42, LOOP_SUM, ARITH, CONV, DIV, BRTABLE];

#[test]
fn corpus_parses_and_verifies() {
    for src in CORPUS {
        let m = parse_module(src).expect("parse");
        verify_module(&m).expect("verify");
    }
}

#[test]
fn binary_roundtrip_is_identity() {
    for src in CORPUS {
        let m = parse_module(src).unwrap();
        let m2 = decode_module(&encode_module(&m)).expect("decode");
        assert_eq!(m, m2, "binary round-trip changed the IR");
    }
}

#[test]
fn text_roundtrip_is_identity() {
    for src in CORPUS {
        let m = parse_module(src).unwrap();
        let m2 = parse_module(&print_module(&m)).expect("reparse printed");
        assert_eq!(m, m2, "text round-trip changed the IR");
    }
}

#[test]
fn text_and_binary_execution_agree() {
    // Differential: interpreting the parsed text and the decoded binary must match.
    let m1 = parse_module(LOOP_SUM).unwrap();
    let m2 = decode_module(&encode_module(&m1)).unwrap();
    for n in [1, 2, 5, 10, 100] {
        let mut f1 = 100_000u64;
        let mut f2 = 100_000u64;
        let r1 = run(&m1, 0, &[Value::I32(n)], &mut f1);
        let r2 = run(&m2, 0, &[Value::I32(n)], &mut f2);
        assert_eq!(r1, r2, "text/binary disagree for n={n}");
    }
}

#[test]
fn add_computes_sum() {
    let bytes = assemble(ADD).unwrap();
    let m = load(&bytes).unwrap();
    let mut fuel = 100u64;
    let r = run(&m, 0, &[Value::I32(2), Value::I32(3)], &mut fuel).unwrap();
    assert_eq!(r, vec![Value::I32(5)]);
}

#[test]
fn const_returns_42() {
    let r = svm::run_text(CONST42, 0, &[], 100).unwrap();
    assert_eq!(r, vec![Value::I32(42)]);
}

#[test]
fn loop_sum_matches_closed_form() {
    let m = load(&assemble(LOOP_SUM).unwrap()).unwrap();
    for n in 1..=100i32 {
        let mut fuel = 1_000_000u64;
        let r = run(&m, 0, &[Value::I32(n)], &mut fuel).unwrap();
        let expected = n * (n + 1) / 2; // 1 + 2 + ... + n
        assert_eq!(r, vec![Value::I32(expected)], "wrong sum for n={n}");
    }
}

#[test]
fn add_wraps_two_complement() {
    // i32.add wraps (§3b): INT_MAX + 1 == INT_MIN.
    let m = load(&assemble(ADD).unwrap()).unwrap();
    let mut fuel = 100u64;
    let r = run(&m, 0, &[Value::I32(i32::MAX), Value::I32(1)], &mut fuel).unwrap();
    assert_eq!(r, vec![Value::I32(i32::MIN)]);
}

// ---- the verifier must reject ill-typed / ill-formed modules (fail-closed) ----

#[test]
fn verifier_rejects_type_mismatch() {
    // i32.add applied to an i64 parameter.
    let m = parse_module(
        r#"
func (i64) -> (i32) {
block0(v0: i64):
  v1 = i32.add v0 v0
  return v1
}
"#,
    )
    .unwrap();
    assert!(matches!(
        verify_module(&m),
        Err(VerifyError::TypeMismatch { .. })
    ));
}

#[test]
fn verifier_rejects_forward_value_reference() {
    // Hand-build a module that names a value index not yet defined (the text parser
    // would refuse a forward name, so we construct the IR directly).
    use svm_ir::{Block, Func, Module, Terminator, ValType};
    let m = Module {
        funcs: vec![Func {
            params: vec![],
            results: vec![ValType::I32],
            blocks: vec![Block {
                params: vec![],
                insts: vec![Inst::IntBin {
                    ty: IntTy::I32,
                    op: BinOp::Add,
                    a: 0,
                    b: 1,
                }], // no values defined yet
                term: Terminator::Return(vec![0]),
            }],
        }],
    };
    assert!(matches!(
        verify_module(&m),
        Err(VerifyError::ValueOutOfRange { .. })
    ));
}

#[test]
fn verifier_rejects_bad_branch_target() {
    use svm_ir::{Block, Func, Module, Terminator, ValType};
    let m = Module {
        funcs: vec![Func {
            params: vec![],
            results: vec![ValType::I32],
            blocks: vec![Block {
                params: vec![],
                insts: vec![Inst::ConstI32(1)],
                term: Terminator::Br {
                    target: 7, // does not exist
                    args: vec![],
                },
            }],
        }],
    };
    assert!(matches!(
        verify_module(&m),
        Err(VerifyError::BlockOutOfRange { .. })
    ));
}

#[test]
fn verifier_rejects_entry_param_mismatch() {
    use svm_ir::{Block, Func, Module, Terminator, ValType};
    let m = Module {
        funcs: vec![Func {
            params: vec![ValType::I32],
            results: vec![],
            blocks: vec![Block {
                params: vec![], // entry params must equal func params
                insts: vec![],
                term: Terminator::Return(vec![]),
            }],
        }],
    };
    assert!(matches!(
        verify_module(&m),
        Err(VerifyError::EntryParamsMismatch { .. })
    ));
}

// ---- expanded instruction set: results + traps ----

fn run1(src: &str, args: &[Value]) -> Result<Vec<Value>, Trap> {
    let m = load(&assemble(src).unwrap()).unwrap();
    let mut fuel = 100_000u64;
    run(&m, 0, args, &mut fuel)
}

#[test]
fn arith_select_results() {
    assert_eq!(
        run1(ARITH, &[Value::I32(3), Value::I32(5)]),
        Ok(vec![Value::I32(100)]) // 3 < 5 -> 100
    );
    assert_eq!(
        run1(ARITH, &[Value::I32(5), Value::I32(3)]),
        Ok(vec![Value::I32(4)]) // (5-3)^2 = 4
    );
}

#[test]
fn conversion_results() {
    assert_eq!(
        run1(CONV, &[Value::I32(5)]),
        Ok(vec![Value::I64(1_000_000_000_005)])
    );
    // sign extension: -1 i32 -> -1 i64, + 1e12 = 999999999999
    assert_eq!(
        run1(CONV, &[Value::I32(-1)]),
        Ok(vec![Value::I64(999_999_999_999)])
    );
}

#[test]
fn div_traps() {
    assert_eq!(
        run1(DIV, &[Value::I32(6), Value::I32(3)]),
        Ok(vec![Value::I32(2)])
    );
    assert_eq!(
        run1(DIV, &[Value::I32(7), Value::I32(0)]),
        Err(Trap::DivByZero)
    );
    assert_eq!(
        run1(DIV, &[Value::I32(i32::MIN), Value::I32(-1)]),
        Err(Trap::IntOverflow)
    );
}

#[test]
fn br_table_dispatch() {
    for (idx, want) in [(0, 10), (1, 20), (2, 30), (3, 99), (7, 99)] {
        assert_eq!(
            run1(BRTABLE, &[Value::I32(idx)]),
            Ok(vec![Value::I32(want)]),
            "br_table idx={idx}"
        );
    }
}

#[test]
fn shifts_take_amount_mod_bitwidth() {
    // i32.shl by 33 == shl by 1 (amount mod 32).
    let src = r#"
func (i32, i32) -> (i32) {
block0(v0: i32, v1: i32):
  v2 = i32.shl v0 v1
  return v2
}
"#;
    assert_eq!(
        run1(src, &[Value::I32(1), Value::I32(33)]),
        Ok(vec![Value::I32(2)])
    );
}

#[test]
fn verifier_rejects_select_type_mismatch() {
    // select of an i32 and an i64 — operands must share a type.
    let m = parse_module(
        r#"
func (i32, i64) -> (i32) {
block0(v0: i32, v1: i64):
  v2 = i32.const 1
  v3 = select v2 v0 v1
  return v3
}
"#,
    )
    .unwrap();
    assert!(matches!(
        verify_module(&m),
        Err(VerifyError::TypeMismatch { .. })
    ));
}

// ---- floats ----

// area = pi * r * r  (f64), then floor it to an i32 via trunc_sat.
const CIRCLE: &str = r#"
func (f64) -> (i32) {
block0(v0: f64):
  v1 = f64.const 3.14159265358979
  v2 = f64.mul v0 v0
  v3 = f64.mul v1 v2
  v4 = f64.floor v3
  v5 = i32.trunc_sat_f64_s v4
  return v5
}
"#;

// round-trip an i32 through f32 and back; also exercises convert + sqrt.
const FSQRT: &str = r#"
func (i32) -> (f32) {
block0(v0: i32):
  v1 = f32.convert_i32_s v0
  v2 = f32.sqrt v1
  return v2
}
"#;

#[test]
fn float_corpus_roundtrips() {
    for src in [CIRCLE, FSQRT] {
        let m = parse_module(src).expect("parse");
        verify_module(&m).expect("verify");
        // binary round-trip
        assert_eq!(m, decode_module(&encode_module(&m)).unwrap());
        // text round-trip
        assert_eq!(m, parse_module(&print_module(&m)).unwrap());
    }
}

#[test]
fn float_arithmetic_results() {
    // r = 2.0 -> area = pi*4 = 12.566.. -> floor 12.
    assert_eq!(run1(CIRCLE, &[Value::F64(2.0)]), Ok(vec![Value::I32(12)]));
    // sqrt(16) = 4.0
    assert_eq!(run1(FSQRT, &[Value::I32(16)]), Ok(vec![Value::F32(4.0)]));
    // sqrt(2) ~ 1.4142135
    match run1(FSQRT, &[Value::I32(2)]) {
        Ok(v) => match v[..] {
            [Value::F32(x)] => assert!((x - 2.0f32.sqrt()).abs() < 1e-6),
            _ => panic!("wrong result shape"),
        },
        e => panic!("unexpected {e:?}"),
    }
}

#[test]
fn float_const_bits_roundtrip() {
    // f32.const printed and reparsed must preserve bits exactly.
    let src = "func () -> (f32) {\nblock0():\n  v0 = f32.const 1.5\n  return v0\n}\n";
    let m = parse_module(src).unwrap();
    assert_eq!(m, parse_module(&print_module(&m)).unwrap());
    assert_eq!(run1(src, &[]), Ok(vec![Value::F32(1.5)]));
}

#[test]
fn reinterpret_preserves_bits() {
    // f32.reinterpret_i32 then i32.reinterpret_f32 is identity on the bit pattern.
    let src = r#"
func (i32) -> (i32) {
block0(v0: i32):
  v1 = f32.reinterpret_i32 v0
  v2 = i32.reinterpret_f32 v1
  return v2
}
"#;
    assert_eq!(
        run1(src, &[Value::I32(0x4048_f5c3u32 as i32)]),
        Ok(vec![Value::I32(0x4048_f5c3u32 as i32)])
    );
}
