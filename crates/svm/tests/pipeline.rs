//! End-to-end pipeline + differential tests for the Phase-1 slice.
//!
//! Proves the loop closes: `text -> binary -> verify -> interp`, that binary and
//! text encodings round-trip to identical IR, and that interpreting the text form
//! and the decoded binary form agree (the differential property the JIT will later
//! be held to against this same interpreter, §18).

use svm::{assemble, load};
use svm_encode::{decode_module, encode_module};
use svm_interp::{run, Value};
use svm_ir::Inst;
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

const CORPUS: &[&str] = &[ADD, CONST42, LOOP_SUM];

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
                insts: vec![Inst::I32Add(0, 1)], // no values defined yet
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
                insts: vec![Inst::I32Const(1)],
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
