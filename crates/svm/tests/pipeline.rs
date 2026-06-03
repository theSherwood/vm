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
        memory: None,
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
        memory: None,
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
        memory: None,
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

// ---- linear memory + confinement masking (I1) ----

// store a value then load it back at the same address.
const MEM_ROUNDTRIP: &str = r#"
memory 16

func (i64, i64) -> (i64) {
block0(v0: i64, v1: i64):
  i64.store v0 v1
  v2 = i64.load v0
  return v2
}
"#;

// narrow store/load: store low byte, load it back zero- and sign-extended.
const MEM_NARROW: &str = r#"
memory 16

func (i64, i32) -> (i32) {
block0(v0: i64, v1: i32):
  i32.store8 v0 v1
  v2 = i32.load8_u v0
  return v2
}
"#;

#[test]
fn memory_corpus_roundtrips() {
    for src in [MEM_ROUNDTRIP, MEM_NARROW] {
        let m = parse_module(src).expect("parse");
        verify_module(&m).expect("verify");
        assert_eq!(m, decode_module(&encode_module(&m)).unwrap(), "binary");
        assert_eq!(m, parse_module(&print_module(&m)).unwrap(), "text");
    }
}

#[test]
fn store_then_load_roundtrips() {
    assert_eq!(
        run1(
            MEM_ROUNDTRIP,
            &[Value::I64(128), Value::I64(0x0123_4567_89ab_cdef)]
        ),
        Ok(vec![Value::I64(0x0123_4567_89ab_cdef)])
    );
}

#[test]
fn narrow_store_load_truncates_and_extends() {
    // store8 of 0x1ff keeps only 0xff; load8_u zero-extends -> 255.
    assert_eq!(
        run1(MEM_NARROW, &[Value::I64(8), Value::I32(0x1ff)]),
        Ok(vec![Value::I32(255)])
    );
    // load8_s of 0x80 sign-extends -> -128.
    let signed = r#"
memory 16
func (i64, i32) -> (i32) {
block0(v0: i64, v1: i32):
  i32.store8 v0 v1
  v2 = i32.load8_s v0
  return v2
}
"#;
    assert_eq!(
        run1(signed, &[Value::I64(8), Value::I32(0x80)]),
        Ok(vec![Value::I32(-128)])
    );
}

#[test]
fn confinement_masks_out_of_window_address() {
    // The window is 2^16 bytes. A store at offset (2^16 + 8) must alias offset 8
    // after masking, so a load at 8 observes it. This is invariant I1: every access
    // is masked into [0, size).
    let src = r#"
memory 16
func (i64, i64, i64) -> (i64) {
block0(v0: i64, v1: i64, v2: i64):
  i64.store v0 v2
  v3 = i64.load v1
  return v3
}
"#;
    let big = 65536 + 8; // 2^16 + 8 aliases 8
    assert_eq!(
        run1(
            src,
            &[Value::I64(big), Value::I64(8), Value::I64(0xdead_beef)]
        ),
        Ok(vec![Value::I64(0xdead_beef)])
    );
}

#[test]
fn access_crossing_window_top_faults() {
    // size = 2^16; an 8-byte load whose masked base is size-4 crosses the top and
    // must fault against the guard region.
    let src = r#"
memory 16
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.load v0
  return v1
}
"#;
    assert_eq!(run1(src, &[Value::I64(65536 - 4)]), Err(Trap::MemoryFault));
    // an in-window aligned access at the same window is fine.
    assert_eq!(run1(src, &[Value::I64(65536 - 8)]), Ok(vec![Value::I64(0)]));
}

#[test]
fn offset_immediate_folds_into_effective_address() {
    // store at base=0 offset=16, load at base=16 offset=0 -> same address.
    let src = r#"
memory 16
func (i64) -> (i32) {
block0(v0: i64):
  v1 = i32.const 777
  i32.store v0 v1 offset=16
  v2 = i32.load v0 offset=16
  return v2
}
"#;
    assert_eq!(run1(src, &[Value::I64(0)]), Ok(vec![Value::I32(777)]));
}

#[test]
fn verifier_rejects_memory_op_without_memory() {
    // load with no `memory` declaration -> rejected.
    let m = parse_module(
        "func (i64) -> (i64) {\nblock0(v0: i64):\n  v1 = i64.load v0\n  return v1\n}\n",
    )
    .unwrap();
    assert!(matches!(
        verify_module(&m),
        Err(VerifyError::MemoryNotDeclared { .. })
    ));
}

// ---- direct calls ----

// func0 returns its arg + arg; func1 calls func0 and adds 1.
const CALL_SIMPLE: &str = r#"
func (i32) -> (i32) {
block0(v0: i32):
  v1 = i32.add v0 v0
  return v1
}

func (i32) -> (i32) {
block0(v0: i32):
  v1 = call 0(v0)
  v2 = i32.const 1
  v3 = i32.add v1 v2
  return v3
}
"#;

// recursive factorial: func0(n) = n <= 1 ? 1 : n * func0(n-1).
const FACT: &str = r#"
func (i32) -> (i32) {
block0(v0: i32):
  v1 = i32.const 1
  v2 = i32.le_s v0 v1
  br_if v2 block1() block2(v0)
block1():
  v3 = i32.const 1
  return v3
block2(v4: i32):
  v5 = i32.const 1
  v6 = i32.sub v4 v5
  v7 = call 0(v6)
  v8 = i32.mul v4 v7
  return v8
}
"#;

// func0 returns two values (quotient, remainder); func1 sums them.
const CALL_MULTI: &str = r#"
func (i32, i32) -> (i32, i32) {
block0(v0: i32, v1: i32):
  v2 = i32.div_s v0 v1
  v3 = i32.rem_s v0 v1
  return v2, v3
}

func (i32, i32) -> (i32) {
block0(v0: i32, v1: i32):
  v2, v3 = call 0(v0, v1)
  v4 = i32.add v2 v3
  return v4
}
"#;

#[test]
fn call_corpus_roundtrips() {
    for src in [CALL_SIMPLE, FACT, CALL_MULTI] {
        let m = parse_module(src).expect("parse");
        verify_module(&m).expect("verify");
        assert_eq!(m, decode_module(&encode_module(&m)).unwrap(), "binary");
        assert_eq!(m, parse_module(&print_module(&m)).unwrap(), "text");
    }
}

#[test]
fn direct_call_computes() {
    // func1(5) = (5+5) + 1 = 11
    assert_eq!(
        run1at(CALL_SIMPLE, 1, &[Value::I32(5)]),
        Ok(vec![Value::I32(11)])
    );
}

#[test]
fn recursive_factorial() {
    for (n, want) in [(0, 1), (1, 1), (5, 120), (7, 5040)] {
        assert_eq!(
            run1at(FACT, 0, &[Value::I32(n)]),
            Ok(vec![Value::I32(want)]),
            "fact({n})"
        );
    }
}

#[test]
fn multi_result_call() {
    // 17 / 5 = 3 rem 2; sum = 5.
    assert_eq!(
        run1at(CALL_MULTI, 1, &[Value::I32(17), Value::I32(5)]),
        Ok(vec![Value::I32(5)])
    );
}

#[test]
fn verifier_rejects_call_to_missing_function() {
    // Hand-built: the text parser can't bind results for an unknown callee arity, so
    // we construct the IR directly to exercise the verifier's range check.
    use svm_ir::{Block, Func, Module, Terminator, ValType};
    let m = Module {
        funcs: vec![Func {
            params: vec![ValType::I32],
            results: vec![ValType::I32],
            blocks: vec![Block {
                params: vec![ValType::I32],
                insts: vec![Inst::Call {
                    func: 9, // does not exist
                    args: vec![0],
                }],
                term: Terminator::Return(vec![0]),
            }],
        }],
        memory: None,
    };
    assert!(matches!(
        verify_module(&m),
        Err(VerifyError::CallFuncOutOfRange { .. })
    ));
}

#[test]
fn verifier_rejects_call_arg_type_mismatch() {
    // call passes an i64 where the callee wants i32.
    let m = parse_module(
        r#"
func (i32) -> (i32) {
block0(v0: i32):
  return v0
}

func (i64) -> (i32) {
block0(v0: i64):
  v1 = call 0(v0)
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
fn unbounded_recursion_traps_not_overflows() {
    // func0 calls itself unconditionally -> must hit the depth bound and trap as
    // StackOverflow (never crash the host stack), well within fuel.
    let src = r#"
func (i32) -> (i32) {
block0(v0: i32):
  v1 = call 0(v0)
  return v1
}
"#;
    let m = load(&assemble(src).unwrap()).unwrap();
    let mut fuel = 10_000_000u64;
    assert_eq!(
        run(&m, 0, &[Value::I32(0)], &mut fuel),
        Err(Trap::StackOverflow)
    );
}

/// Run a specific function index (the corpus helpers default to func 0).
fn run1at(src: &str, func: u32, args: &[Value]) -> Result<Vec<Value>, Trap> {
    let m = load(&assemble(src).unwrap()).unwrap();
    let mut fuel = 1_000_000u64;
    run(&m, func, args, &mut fuel)
}
