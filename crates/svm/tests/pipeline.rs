//! End-to-end pipeline + differential tests for the Phase-1 slice.
//!
//! Proves the loop closes: `text -> binary -> verify -> interp`, that binary and
//! text encodings round-trip to identical IR, and that interpreting the text form
//! and the decoded binary form agree (the differential property the JIT will later
//! be held to against this same interpreter, §18).

use svm::{assemble, load};
use svm_encode::{decode_module, encode_module};
use svm_interp::{run, run_with_host, Host, StreamRole, Trap, Value};
use svm_ir::{BinOp, Inst, IntTy};
use svm_text::{parse_module, print_module};
use svm_verify::{verify_module, VerifyError};

const ADD: &str = r#"
func (i32, i32) -> (i32) {
block 0 (v0: i32, v1: i32) {
  v2 = i32.add v0 v1
  return v2
  }
}
"#;

const CONST42: &str = r#"
func () -> (i32) {
block 0 () {
  v0 = i32.const 42
  return v0
  }
}
"#;

// sum = 1 + 2 + ... + N  (N >= 1), via a back-edge loop with block parameters.
const LOOP_SUM: &str = r#"
func (i32) -> (i32) {
block 0 (v0: i32) {
  v1 = i32.const 0
  br 1(v0, v1)
}
block 1 (v2: i32, v3: i32) {
  v4 = i32.add v3 v2
  v5 = i32.const -1
  v6 = i32.add v2 v5
  br_if v6 1(v6, v4) 2(v4)
}
block 2 (v7: i32) {
  return v7
  }
}
"#;

// (v0 < v1) ? 100 : (v0 - v1)^2  — exercises sub/mul/lt_s/select/const.
const ARITH: &str = r#"
func (i32, i32) -> (i32) {
block 0 (v0: i32, v1: i32) {
  v2 = i32.sub v0 v1
  v3 = i32.mul v2 v2
  v4 = i32.lt_s v0 v1
  v5 = i32.const 100
  v6 = select v4 v5 v3
  return v6
  }
}
"#;

// sign-extend i32 -> i64, then add a large i64 constant.
const CONV: &str = r#"
func (i32) -> (i64) {
block 0 (v0: i32) {
  v1 = i64.extend_i32_s v0
  v2 = i64.const 1000000000000
  v3 = i64.add v1 v2
  return v3
  }
}
"#;

const DIV: &str = r#"
func (i32, i32) -> (i32) {
block 0 (v0: i32, v1: i32) {
  v2 = i32.div_s v0 v1
  return v2
  }
}
"#;

// br_table: idx selects 10/20/30, else default 99.
const BRTABLE: &str = r#"
func (i32) -> (i32) {
block 0 (v0: i32) {
  br_table v0 [1(), 2(), 3()] 4()
}
block 1 () {
  v1 = i32.const 10
  return v1
}
block 2 () {
  v2 = i32.const 20
  return v2
}
block 3 () {
  v3 = i32.const 30
  return v3
}
block 4 () {
  v4 = i32.const 99
  return v4
  }
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
fn data_section_parses_roundtrips_and_verifies() {
    // `data [ro] <offset> "<bytes>"` (§3a / D40): RO + RW segments, incl. non-UTF-8 bytes.
    let src = "data ro 16 \"hi\\x00\\xff\"\ndata 32 \"abc\"\nmemory 8\n\
               func () -> (i32) {\nblock 0 () {\n  v0 = i32.const 0\n  return v0\n  }\n}\n";
    let m = parse_module(src).expect("parse");
    assert_eq!(
        m.data,
        vec![
            svm_ir::Data {
                offset: 16,
                readonly: true,
                bytes: vec![b'h', b'i', 0, 0xff]
            },
            svm_ir::Data {
                offset: 32,
                readonly: false,
                bytes: vec![b'a', b'b', b'c']
            },
        ]
    );
    verify_module(&m).expect("verify");
    // text and binary round-trips preserve the segments exactly.
    assert_eq!(parse_module(&print_module(&m)).expect("reparse"), m);
    assert_eq!(decode_module(&encode_module(&m)).expect("decode"), m);
}

#[test]
fn atomics_parse_roundtrip_and_verify() {
    // §12 atomics across both widths, every RMW op, and the `offset=` memarg: parse, verify, then
    // assert the text and binary serializations round-trip to the identical IR.
    let src = "memory 16\n\
        func (i64) -> (i64) {\n\
        block 0 (v0: i64) {\n\
        \x20 v1 = i32.atomic.load v0\n\
        \x20 v2 = i64.atomic.load v0 offset=8\n\
        \x20 i32.atomic.store v0 v1\n\
        \x20 i64.atomic.store v0 v2\n\
        \x20 v3 = i32.atomic.rmw.add v0 v1\n\
        \x20 v4 = i32.atomic.rmw.sub v0 v1\n\
        \x20 v5 = i32.atomic.rmw.and v0 v1\n\
        \x20 v6 = i32.atomic.rmw.or v0 v1\n\
        \x20 v7 = i32.atomic.rmw.xor v0 v1\n\
        \x20 v8 = i32.atomic.rmw.xchg v0 v1\n\
        \x20 v9 = i64.atomic.rmw.add v0 v2\n\
        \x20 v10 = i32.atomic.cmpxchg v0 v1 v3\n\
        \x20 v11 = i64.atomic.cmpxchg v0 v2 v9 offset=16\n\
        \x20 return v2\n\
          }\n\
        }\n";
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");
    assert_eq!(
        parse_module(&print_module(&m)).expect("reparse printed"),
        m,
        "atomics text round-trip changed the IR"
    );
    assert_eq!(
        decode_module(&encode_module(&m)).expect("decode"),
        m,
        "atomics binary round-trip changed the IR"
    );
}

#[test]
fn fibers_parse_roundtrip_and_verify() {
    // §12 stack switching: `cont.new` (funcref, sp -> handle), `cont.resume` (handle, i64 ->
    // status, value), and `suspend` (i64 -> i64). Parse, verify, then assert text and
    // binary serializations round-trip to the identical IR. Func 1 is a fiber body
    // `(i64 sp, i64 arg) -> (i64)` that suspends once then returns; func 0 drives it.
    let src = "func (i64) -> (i64) {\n\
        block 0 (v0: i64) {\n\
        \x20 v1 = ref.func 1\n\
        \x20 v2 = i64.const 4096\n\
        \x20 v3 = cont.new v1 v2\n\
        \x20 v4 = i64.const 10\n\
        \x20 v5, v6 = cont.resume v3 v4\n\
        \x20 v7, v8 = cont.resume v3 v6\n\
        \x20 return v8\n\
          }\n\
        }\n\
        func (i64, i64) -> (i64) {\n\
        block 0 (v0: i64, v1: i64) {\n\
        \x20 v2 = suspend v1\n\
        \x20 return v2\n\
          }\n\
        }\n";
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");
    assert_eq!(
        parse_module(&print_module(&m)).expect("reparse printed"),
        m,
        "fibers text round-trip changed the IR"
    );
    assert_eq!(
        decode_module(&encode_module(&m)).expect("decode"),
        m,
        "fibers binary round-trip changed the IR"
    );
}

#[test]
fn verify_rejects_out_of_window_data() {
    // window = 2^3 = 8 bytes; a 4-byte segment at offset 6 overruns `[0, 8)`.
    let src = "data 6 \"abcd\"\nmemory 3\n\
               func () -> (i32) {\nblock 0 () {\n  v0 = i32.const 0\n  return v0\n  }\n}\n";
    let m = parse_module(src).expect("parse");
    assert!(matches!(
        verify_module(&m),
        Err(VerifyError::DataOutOfWindow { seg: 0 })
    ));
}

#[test]
fn verify_rejects_data_without_memory() {
    let src = "data 0 \"x\"\n\
               func () -> (i32) {\nblock 0 () {\n  v0 = i32.const 0\n  return v0\n  }\n}\n";
    let m = parse_module(src).expect("parse");
    assert!(matches!(
        verify_module(&m),
        Err(VerifyError::DataWithoutMemory { seg: 0 })
    ));
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

// ---- §12 fibers: stack switching on the interpreter ----

#[test]
fn fiber_suspend_then_resume_threads_values() {
    // Func 1 is a fiber `(i64 sp, i64 arg) -> (i64)`: it `suspend`s its arg (yielding it to
    // the resumer), then on the next resume adds 100 to the delivered value and returns it.
    // The root drives it: resume(10) -> (SUSPENDED, 10); resume(7) -> (RETURNED, 107).
    let src = "func () -> (i32, i64, i32, i64) {\n\
        block 0 () {\n\
        \x20 v0 = ref.func 1\n\
        \x20 v1 = i64.const 4096\n\
        \x20 v2 = cont.new v0 v1\n\
        \x20 v3 = i64.const 10\n\
        \x20 v4, v5 = cont.resume v2 v3\n\
        \x20 v6 = i64.const 7\n\
        \x20 v7, v8 = cont.resume v2 v6\n\
        \x20 return v4 v5 v7 v8\n\
          }\n\
        }\n\
        func (i64, i64) -> (i64) {\n\
        block 0 (v0: i64, v1: i64) {\n\
        \x20 v2 = suspend v1\n\
        \x20 v3 = i64.const 100\n\
        \x20 v4 = i64.add v2 v3\n\
        \x20 return v4\n\
          }\n\
        }\n";
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");
    let mut fuel = 1_000_000u64;
    let r = run(&m, 0, &[], &mut fuel).unwrap();
    // (status=SUSPENDED=0, yielded=10), (status=RETURNED=1, returned=107).
    assert_eq!(
        r,
        vec![
            Value::I32(0),
            Value::I64(10),
            Value::I32(1),
            Value::I64(107)
        ]
    );
}

#[test]
fn fiber_generator_loop_sums_a_sequence() {
    // A generator fiber yields 1, 2, 3 (three `suspend`s) then returns 4. The root loops
    // resuming it, accumulating every delivered value until the status is RETURNED — a
    // workout for repeated resume/suspend with the fiber handle threaded as a block param.
    let src = "func () -> (i64) {\n\
        block 0 () {\n\
        \x20 v0 = ref.func 1\n\
        \x20 v1 = i64.const 4096\n\
        \x20 v2 = cont.new v0 v1\n\
        \x20 v3 = i64.const 0\n\
        \x20 br 1(v2, v3)\n\
        }\n\
        block 1 (v4: i64, v5: i64) {\n\
        \x20 v6 = i64.const 0\n\
        \x20 v7, v8 = cont.resume v4 v6\n\
        \x20 v9 = i64.add v5 v8\n\
        \x20 br_if v7 2(v9) 1(v4, v9)\n\
        }\n\
        block 2 (v10: i64) {\n\
        \x20 return v10\n\
          }\n\
        }\n\
        func (i64, i64) -> (i64) {\n\
        block 0 (v0: i64, v1: i64) {\n\
        \x20 v2 = i64.const 1\n\
        \x20 v3 = suspend v2\n\
        \x20 v4 = i64.const 2\n\
        \x20 v5 = suspend v4\n\
        \x20 v6 = i64.const 3\n\
        \x20 v7 = suspend v6\n\
        \x20 v8 = i64.const 4\n\
        \x20 return v8\n\
          }\n\
        }\n";
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");
    let mut fuel = 1_000_000u64;
    let r = run(&m, 0, &[], &mut fuel).unwrap();
    assert_eq!(r, vec![Value::I64(1 + 2 + 3 + 4)]);
}

#[test]
fn fiber_nested_resume_chain() {
    // A three-level resume chain: root resumes fiber A (func 1), which itself resumes
    // fiber B (func 2). B suspends -> control returns to A's resume site; A suspends ->
    // control returns to root. Then unwinding the other way to completion. Exercises a
    // resume chain deeper than one and `suspend` returning to the correct resumer.
    let src = "func () -> (i64, i64) {\n\
        block 0 () {\n\
        \x20 v0 = ref.func 1\n\
        \x20 v1 = i64.const 4096\n\
        \x20 v2 = cont.new v0 v1\n\
        \x20 v3 = i64.const 0\n\
        \x20 v4, v5 = cont.resume v2 v3\n\
        \x20 v6, v7 = cont.resume v2 v3\n\
        \x20 return v5 v7\n\
          }\n\
        }\n\
        func (i64, i64) -> (i64) {\n\
        block 0 (v0: i64, v1: i64) {\n\
        \x20 v2 = ref.func 2\n\
        \x20 v3 = i64.const 8192\n\
        \x20 v4 = cont.new v2 v3\n\
        \x20 v5 = i64.const 0\n\
        \x20 v6, v7 = cont.resume v4 v5\n\
        \x20 v8 = suspend v7\n\
        \x20 v9, v10 = cont.resume v4 v5\n\
        \x20 return v10\n\
          }\n\
        }\n\
        func (i64, i64) -> (i64) {\n\
        block 0 (v0: i64, v1: i64) {\n\
        \x20 v2 = i64.const 11\n\
        \x20 v3 = suspend v2\n\
        \x20 v4 = i64.const 22\n\
        \x20 return v4\n\
          }\n\
        }\n";
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");
    let mut fuel = 1_000_000u64;
    let r = run(&m, 0, &[], &mut fuel).unwrap();
    // First root resume: A resumes B (B yields 11), A suspends yielding 11 -> root sees 11.
    // Second root resume: A resumes B again (B returns 22), A returns 22 -> root sees 22.
    assert_eq!(r, vec![Value::I64(11), Value::I64(22)]);
}

#[test]
fn fiber_resume_after_return_traps() {
    // Func 1 returns immediately (no suspend): the first resume yields (RETURNED, arg).
    // Resuming the now-`Done` fiber a second time is inert -> `FiberFault`.
    let src = "func () -> (i64) {\n\
        block 0 () {\n\
        \x20 v0 = ref.func 1\n\
        \x20 v1 = i64.const 4096\n\
        \x20 v2 = cont.new v0 v1\n\
        \x20 v3 = i64.const 1\n\
        \x20 v4, v5 = cont.resume v2 v3\n\
        \x20 v6, v7 = cont.resume v2 v3\n\
        \x20 return v7\n\
          }\n\
        }\n\
        func (i64, i64) -> (i64) {\n\
        block 0 (v0: i64, v1: i64) {\n\
        \x20 return v1\n\
          }\n\
        }\n";
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");
    let mut fuel = 1_000_000u64;
    assert_eq!(run(&m, 0, &[], &mut fuel), Err(Trap::FiberFault));
}

#[test]
fn fiber_suspend_at_root_traps() {
    // `suspend` with no resumer (the root computation) traps rather than escaping.
    let src = "func () -> (i64) {\n\
        block 0 () {\n\
        \x20 v0 = i64.const 5\n\
        \x20 v1 = suspend v0\n\
        \x20 return v1\n\
          }\n\
        }\n";
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");
    let mut fuel = 1_000_000u64;
    assert_eq!(run(&m, 0, &[], &mut fuel), Err(Trap::FiberFault));
}

#[test]
fn fiber_forged_handle_is_inert() {
    // A forged handle (no fiber created) is masked into the table and resolves to the
    // running root, which is in the resume chain -> inert (`FiberFault`), never an escape.
    let src = "func () -> (i64) {\n\
        block 0 () {\n\
        \x20 v0 = i64.const 999\n\
        \x20 v1 = i64.const 0\n\
        \x20 v2, v3 = cont.resume v0 v1\n\
        \x20 return v3\n\
          }\n\
        }\n";
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");
    let mut fuel = 1_000_000u64;
    assert_eq!(run(&m, 0, &[], &mut fuel), Err(Trap::FiberFault));
}

// ---- the verifier must reject ill-typed / ill-formed modules (fail-closed) ----

#[test]
fn verifier_rejects_type_mismatch() {
    // i32.add applied to an i64 parameter.
    let m = parse_module(
        r#"
func (i64) -> (i32) {
block 0 (v0: i64) {
  v1 = i32.add v0 v0
  return v1
  }
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
        data: Vec::new(),
        imports: Vec::new(),
        exports: Vec::new(),
        impl_exports: vec![],
        types: vec![],
        debug_info: None,
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
        data: Vec::new(),
        imports: Vec::new(),
        exports: Vec::new(),
        impl_exports: vec![],
        types: vec![],
        debug_info: None,
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
        data: Vec::new(),
        imports: Vec::new(),
        exports: Vec::new(),
        impl_exports: vec![],
        types: vec![],
        debug_info: None,
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

const REM_S: &str = r#"
func (i32, i32) -> (i32) {
block 0 (v0: i32, v1: i32) {
  v2 = i32.rem_s v0 v1
  return v2
  }
}
"#;

#[test]
fn rem_s_overflow_is_zero_not_a_trap() {
    // Design choice (§3b): trap only when there is no representable result. `INT_MIN %
    // -1 == 0` is representable (only the *quotient* overflows, not the remainder), so
    // `rem_s` returns 0 — unlike `div_s`, whose quotient `+2^31` does not fit and traps.
    // Regression for an over-trapping bug the JIT differential harness surfaced.
    assert_eq!(
        run1(REM_S, &[Value::I32(i32::MIN), Value::I32(-1)]),
        Ok(vec![Value::I32(0)])
    );
    assert_eq!(
        run1(REM_S, &[Value::I32(7), Value::I32(3)]),
        Ok(vec![Value::I32(1)])
    );
    assert_eq!(
        run1(REM_S, &[Value::I32(7), Value::I32(0)]),
        Err(Trap::DivByZero)
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
block 0 (v0: i32, v1: i32) {
  v2 = i32.shl v0 v1
  return v2
  }
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
block 0 (v0: i32, v1: i64) {
  v2 = i32.const 1
  v3 = select v2 v0 v1
  return v3
  }
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
block 0 (v0: f64) {
  v1 = f64.const 3.14159265358979
  v2 = f64.mul v0 v0
  v3 = f64.mul v1 v2
  v4 = f64.floor v3
  v5 = i32.trunc_sat_f64_s v4
  return v5
  }
}
"#;

// round-trip an i32 through f32 and back; also exercises convert + sqrt.
const FSQRT: &str = r#"
func (i32) -> (f32) {
block 0 (v0: i32) {
  v1 = f32.convert_i32_s v0
  v2 = f32.sqrt v1
  return v2
  }
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
    let src = "func () -> (f32) {\nblock 0 () {\n  v0 = f32.const 1.5\n  return v0\n  }\n}\n";
    let m = parse_module(src).unwrap();
    assert_eq!(m, parse_module(&print_module(&m)).unwrap());
    assert_eq!(run1(src, &[]), Ok(vec![Value::F32(1.5)]));
}

#[test]
fn reinterpret_preserves_bits() {
    // f32.reinterpret_i32 then i32.reinterpret_f32 is identity on the bit pattern.
    let src = r#"
func (i32) -> (i32) {
block 0 (v0: i32) {
  v1 = f32.reinterpret_i32 v0
  v2 = i32.reinterpret_f32 v1
  return v2
  }
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
block 0 (v0: i64, v1: i64) {
  i64.store v0 v1
  v2 = i64.load v0
  return v2
  }
}
"#;

// narrow store/load: store low byte, load it back zero- and sign-extended.
const MEM_NARROW: &str = r#"
memory 16

func (i64, i32) -> (i32) {
block 0 (v0: i64, v1: i32) {
  i32.store8 v0 v1
  v2 = i32.load8_u v0
  return v2
  }
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
block 0 (v0: i64, v1: i32) {
  i32.store8 v0 v1
  v2 = i32.load8_s v0
  return v2
  }
}
"#;
    assert_eq!(
        run1(signed, &[Value::I64(8), Value::I32(0x80)]),
        Ok(vec![Value::I32(-128)])
    );
}

#[test]
fn confinement_faults_out_of_window_address() {
    // The window is 2^16 bytes *backed*, inside a large reserved range (the §4 default policy).
    // A store at offset (2^16 + 8) lands in the reserved-but-unmapped tail, so it **faults**
    // (detect-and-kill, §4/§5) rather than wrapping in — the deliberate decoupled-model
    // behaviour (invariant I1: confined to `reserved`, faults outside the backed `mapped`).
    let src = r#"
memory 16
func (i64, i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64, v2: i64) {
  i64.store v0 v2
  v3 = i64.load v1
  return v3
  }
}
"#;
    let big = 65536 + 8; // 2^16 + 8 — in the unmapped tail
    assert_eq!(
        run1(
            src,
            &[Value::I64(big), Value::I64(8), Value::I64(0xdead_beef)]
        ),
        Err(Trap::MemoryFault)
    );
    // An in-window access is unaffected: store at 8, load it back.
    assert_eq!(
        run1(
            src,
            &[Value::I64(8), Value::I64(8), Value::I64(0xdead_beef)]
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
block 0 (v0: i64) {
  v1 = i64.load v0
  return v1
  }
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
block 0 (v0: i64) {
  v1 = i32.const 777
  i32.store v0 v1 offset=16
  v2 = i32.load v0 offset=16
  return v2
  }
}
"#;
    assert_eq!(run1(src, &[Value::I64(0)]), Ok(vec![Value::I32(777)]));
}

#[test]
fn verifier_rejects_memory_op_without_memory() {
    // load with no `memory` declaration -> rejected.
    let m = parse_module(
        "func (i64) -> (i64) {\nblock 0 (v0: i64) {\n  v1 = i64.load v0\n  return v1\n  }\n}\n",
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
block 0 (v0: i32) {
  v1 = i32.add v0 v0
  return v1
  }
}

func (i32) -> (i32) {
block 0 (v0: i32) {
  v1 = call 0(v0)
  v2 = i32.const 1
  v3 = i32.add v1 v2
  return v3
  }
}
"#;

// recursive factorial: func0(n) = n <= 1 ? 1 : n * func0(n-1).
const FACT: &str = r#"
func (i32) -> (i32) {
block 0 (v0: i32) {
  v1 = i32.const 1
  v2 = i32.le_s v0 v1
  br_if v2 1() 2(v0)
}
block 1 () {
  v3 = i32.const 1
  return v3
}
block 2 (v4: i32) {
  v5 = i32.const 1
  v6 = i32.sub v4 v5
  v7 = call 0(v6)
  v8 = i32.mul v4 v7
  return v8
  }
}
"#;

// func0 returns two values (quotient, remainder); func1 sums them.
const CALL_MULTI: &str = r#"
func (i32, i32) -> (i32, i32) {
block 0 (v0: i32, v1: i32) {
  v2 = i32.div_s v0 v1
  v3 = i32.rem_s v0 v1
  return v2, v3
  }
}

func (i32, i32) -> (i32) {
block 0 (v0: i32, v1: i32) {
  v2, v3 = call 0(v0, v1)
  v4 = i32.add v2 v3
  return v4
  }
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
        data: Vec::new(),
        imports: Vec::new(),
        exports: Vec::new(),
        impl_exports: vec![],
        types: vec![],
        debug_info: None,
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
block 0 (v0: i32) {
  return v0
  }
}

func (i64) -> (i32) {
block 0 (v0: i64) {
  v1 = call 0(v0)
  return v1
  }
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
block 0 (v0: i32) {
  v1 = call 0(v0)
  return v1
  }
}
"#;
    let m = load(&assemble(src).unwrap()).unwrap();
    let mut fuel = 10_000_000u64;
    assert_eq!(
        run(&m, 0, &[Value::I32(0)], &mut fuel),
        Err(Trap::StackOverflow)
    );
}

#[test]
fn mutual_recursion_traps_not_overflows() {
    // func0 -> func1 -> func0 -> ... unbounded. The explicit-stack interpreter must keep a
    // call stack that alternates *between functions* and hit the depth bound as a clean
    // StackOverflow (never host-stack recursion), well within fuel.
    let src = r#"
func (i32) -> (i32) {
block 0 (v0: i32) {
  v1 = call 1(v0)
  return v1
  }
}

func (i32) -> (i32) {
block 0 (v0: i32) {
  v1 = call 0(v0)
  return v1
  }
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

// ---- indirect calls (function table, §3c) ----

// func0/func1 are (i32,i32)->(i32) implementations; func2 dispatches to one of
// them via call_indirect on a funcref built with ref.func.
const INDIRECT: &str = r#"
func (i32, i32) -> (i32) {
block 0 (v0: i32, v1: i32) {
  v2 = i32.add v0 v1
  return v2
  }
}

func (i32, i32) -> (i32) {
block 0 (v0: i32, v1: i32) {
  v2 = i32.sub v0 v1
  return v2
  }
}

func (i32, i32, i32) -> (i32) {
block 0 (v0: i32, v1: i32, v2: i32) {
  v3 = call_indirect (i32, i32) -> (i32) v0 (v1, v2)
  return v3
  }
}
"#;

#[test]
fn indirect_corpus_roundtrips() {
    let m = parse_module(INDIRECT).expect("parse");
    verify_module(&m).expect("verify");
    assert_eq!(m, decode_module(&encode_module(&m)).unwrap(), "binary");
    assert_eq!(m, parse_module(&print_module(&m)).unwrap(), "text");
}

#[test]
fn indirect_call_dispatches_by_index() {
    // func2(idx, 10, 3): idx 0 -> add = 13, idx 1 -> sub = 7.
    assert_eq!(
        run1at(INDIRECT, 2, &[Value::I32(0), Value::I32(10), Value::I32(3)]),
        Ok(vec![Value::I32(13)])
    );
    assert_eq!(
        run1at(INDIRECT, 2, &[Value::I32(1), Value::I32(10), Value::I32(3)]),
        Ok(vec![Value::I32(7)])
    );
}

#[test]
fn indirect_call_type_mismatch_traps() {
    // func0 here is ()->() ; dispatching it through an (i32,i32)->(i32) site must
    // trap on the signature check rather than misinterpret the call.
    let src = r#"
func () -> () {
block 0 () {
  return
  }
}

func (i32) -> (i32) {
block 0 (v0: i32) {
  v1 = i32.const 5
  v2 = call_indirect (i32, i32) -> (i32) v0 (v1, v1)
  return v2
  }
}
"#;
    // index 0 selects the ()->() function -> signature mismatch.
    assert_eq!(
        run1at(src, 1, &[Value::I32(0)]),
        Err(Trap::IndirectCallType)
    );
}

#[test]
fn indirect_call_index_is_masked_then_type_checked() {
    // 3 functions -> table padded to 4 (pow2), mask 3. An out-of-range index is
    // masked back in: idx 4 masks to 0 (the add fn), idx 5 masks to 1 (sub).
    assert_eq!(
        run1at(INDIRECT, 2, &[Value::I32(4), Value::I32(10), Value::I32(3)]),
        Ok(vec![Value::I32(13)]) // 4 & 3 == 0 -> add
    );
    assert_eq!(
        run1at(INDIRECT, 2, &[Value::I32(5), Value::I32(10), Value::I32(3)]),
        Ok(vec![Value::I32(7)]) // 5 & 3 == 1 -> sub
    );
    // idx 2 masks to 2 (the dispatcher itself), whose signature differs from the
    // call site -> trap on the type check rather than recurse.
    assert_eq!(
        run1at(INDIRECT, 2, &[Value::I32(2), Value::I32(10), Value::I32(3)]),
        Err(Trap::IndirectCallType)
    );
}

#[test]
fn ref_func_then_indirect_call() {
    // Build a funcref with ref.func, then call through it.
    let src = r#"
func (i32) -> (i32) {
block 0 (v0: i32) {
  v1 = i32.const 100
  v2 = i32.add v0 v1
  return v2
  }
}

func (i32) -> (i32) {
block 0 (v0: i32) {
  v1 = ref.func 0
  v2 = call_indirect (i32) -> (i32) v1 (v0)
  return v2
  }
}
"#;
    assert_eq!(run1at(src, 1, &[Value::I32(7)]), Ok(vec![Value::I32(107)]));
}

// ---- completed integer ops + unreachable ----

// exercises clz/ctz/popcnt/rotl/rotr/extend8_s and an unreachable arm.
const BITOPS: &str = r#"
func (i32) -> (i32) {
block 0 (v0: i32) {
  v1 = i32.clz v0
  v2 = i32.ctz v0
  v3 = i32.add v1 v2
  v4 = i32.popcnt v0
  v5 = i32.add v3 v4
  return v5
  }
}
"#;

#[test]
fn bitops_roundtrip_and_compute() {
    let m = parse_module(BITOPS).expect("parse");
    verify_module(&m).expect("verify");
    assert_eq!(m, decode_module(&encode_module(&m)).unwrap(), "binary");
    assert_eq!(m, parse_module(&print_module(&m)).unwrap(), "text");

    // v0 = 0x00010000 (bit 16 set): clz=15, ctz=16, popcnt=1 -> 32.
    assert_eq!(
        run1at(BITOPS, 0, &[Value::I32(0x0001_0000)]),
        Ok(vec![Value::I32(32)])
    );
    // v0 = 0: clz=32, ctz=32, popcnt=0 -> 64.
    assert_eq!(
        run1at(BITOPS, 0, &[Value::I32(0)]),
        Ok(vec![Value::I32(64)])
    );
}

#[test]
fn rotate_ops() {
    let src = r#"
func (i32, i32) -> (i32) {
block 0 (v0: i32, v1: i32) {
  v2 = i32.rotl v0 v1
  return v2
  }
}
"#;
    // rotl(0x12345678, 8) = 0x34567812
    assert_eq!(
        run1at(src, 0, &[Value::I32(0x1234_5678), Value::I32(8)]),
        Ok(vec![Value::I32(0x3456_7812u32 as i32)])
    );
    // rotation amount is mod 32: rotl by 40 == rotl by 8.
    assert_eq!(
        run1at(src, 0, &[Value::I32(0x1234_5678), Value::I32(40)]),
        Ok(vec![Value::I32(0x3456_7812u32 as i32)])
    );
}

#[test]
fn extend8_s_sign_extends() {
    let src = r#"
func (i32) -> (i32) {
block 0 (v0: i32) {
  v1 = i32.extend8_s v0
  return v1
  }
}
"#;
    assert_eq!(
        run1at(src, 0, &[Value::I32(0xff)]),
        Ok(vec![Value::I32(-1)])
    );
    assert_eq!(
        run1at(src, 0, &[Value::I32(0x7f)]),
        Ok(vec![Value::I32(127)])
    );
}

#[test]
fn unreachable_traps() {
    let src = r#"
func (i32) -> (i32) {
block 0 (v0: i32) {
  br_if v0 1() 2()
}
block 1 () {
  unreachable
}
block 2 () {
  v1 = i32.const 7
  return v1
  }
}
"#;
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");
    assert_eq!(m, decode_module(&encode_module(&m)).unwrap());
    assert_eq!(m, parse_module(&print_module(&m)).unwrap());
    // cond != 0 -> block1 -> unreachable trap; cond == 0 -> block2 -> 7.
    assert_eq!(run1at(src, 0, &[Value::I32(1)]), Err(Trap::Unreachable));
    assert_eq!(run1at(src, 0, &[Value::I32(0)]), Ok(vec![Value::I32(7)]));
}

// ---- tail calls (return_call / return_call_indirect) ----

// Tail-recursive sum: sum(n, acc) = n==0 ? acc : sum(n-1, acc+n), via return_call.
// Values flow between blocks only through block parameters (block-local SSA).
const TAILSUM: &str = r#"
func (i32, i32) -> (i32) {
block 0 (v0: i32, v1: i32) {
  v2 = i32.eqz v0
  br_if v2 1(v1) 2(v0, v1)
}
block 1 (v3: i32) {
  return v3
}
block 2 (v4: i32, v5: i32) {
  v6 = i32.const -1
  v7 = i32.add v4 v6
  v8 = i32.add v5 v4
  return_call 0(v7, v8)
  }
}
"#;

/// Reference sum with the same wrapping i32 arithmetic the interpreter uses.
fn wrapping_sum(n: i32) -> i32 {
    let mut acc = 0i32;
    let mut k = n;
    while k != 0 {
        acc = acc.wrapping_add(k);
        k -= 1;
    }
    acc
}

#[test]
fn tailcall_roundtrips_and_computes() {
    let m = parse_module(TAILSUM).expect("parse");
    verify_module(&m).expect("verify");
    assert_eq!(m, decode_module(&encode_module(&m)).unwrap(), "binary");
    assert_eq!(m, parse_module(&print_module(&m)).unwrap(), "text");

    // sum(10, 0) = 55
    assert_eq!(
        run1at(TAILSUM, 0, &[Value::I32(10), Value::I32(0)]),
        Ok(vec![Value::I32(55)])
    );
}

#[test]
fn deep_tail_recursion_is_constant_host_stack() {
    // 100_000 tail calls — far beyond MAX_CALL_DEPTH (256). A non-tail
    // implementation would StackOverflow; tail calls must run in O(1) host stack.
    let m = load(&assemble(TAILSUM).unwrap()).unwrap();
    let mut fuel = 100_000_000u64;
    let n = 100_000i32;
    let r = run(&m, 0, &[Value::I32(n), Value::I32(0)], &mut fuel).unwrap();
    assert_eq!(r, vec![Value::I32(wrapping_sum(n))]);
}

#[test]
fn return_call_indirect_tail_dispatches() {
    // The tail-sum body, but tail-calling *indirectly* through table index 0.
    let src = r#"
func (i32, i32) -> (i32) {
block 0 (v0: i32, v1: i32) {
  v2 = i32.eqz v0
  br_if v2 1(v1) 2(v0, v1)
}
block 1 (v3: i32) {
  return v3
}
block 2 (v4: i32, v5: i32) {
  v6 = i32.const -1
  v7 = i32.add v4 v6
  v8 = i32.add v5 v4
  v9 = i32.const 0
  return_call_indirect (i32, i32) -> (i32) v9 (v7, v8)
  }
}
"#;
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");
    assert_eq!(m, decode_module(&encode_module(&m)).unwrap());
    assert_eq!(m, parse_module(&print_module(&m)).unwrap());
    assert_eq!(
        run1at(src, 0, &[Value::I32(5), Value::I32(0)]),
        Ok(vec![Value::I32(15)])
    );
}

#[test]
fn verifier_rejects_tail_call_result_mismatch() {
    // callee returns i64 but the caller's result is i32 -> tail results must match.
    let m = parse_module(
        r#"
func (i32) -> (i64) {
block 0 (v0: i32) {
  v1 = i64.const 0
  return v1
  }
}

func (i32) -> (i32) {
block 0 (v0: i32) {
  return_call 0(v0)
  }
}
"#,
    )
    .unwrap();
    assert!(matches!(
        verify_module(&m),
        Err(VerifyError::ResultCountMismatch { .. })
    ));
}

// ---- trapping float->int conversions + ptr ops ----

const TRUNC_TRAP: &str = r#"
func (f64) -> (i32) {
block 0 (v0: f64) {
  v1 = i32.trunc_f64_s v0
  return v1
  }
}
"#;

#[test]
fn trapping_trunc_roundtrips_and_traps() {
    let m = parse_module(TRUNC_TRAP).expect("parse");
    verify_module(&m).expect("verify");
    assert_eq!(m, decode_module(&encode_module(&m)).unwrap(), "binary");
    assert_eq!(m, parse_module(&print_module(&m)).unwrap(), "text");

    // In range -> truncates toward zero.
    assert_eq!(
        run1at(TRUNC_TRAP, 0, &[Value::F64(3.9)]),
        Ok(vec![Value::I32(3)])
    );
    assert_eq!(
        run1at(TRUNC_TRAP, 0, &[Value::F64(-3.9)]),
        Ok(vec![Value::I32(-3)])
    );
    // i32::MIN is exactly representable and in range.
    assert_eq!(
        run1at(TRUNC_TRAP, 0, &[Value::F64(-2147483648.0)]),
        Ok(vec![Value::I32(i32::MIN)])
    );
    // NaN traps.
    assert_eq!(
        run1at(TRUNC_TRAP, 0, &[Value::F64(f64::NAN)]),
        Err(Trap::BadConversion)
    );
    // Out of range (2^31) traps; saturating would have clamped instead.
    assert_eq!(
        run1at(TRUNC_TRAP, 0, &[Value::F64(2147483648.0)]),
        Err(Trap::BadConversion)
    );
    assert_eq!(
        run1at(TRUNC_TRAP, 0, &[Value::F64(f64::INFINITY)]),
        Err(Trap::BadConversion)
    );
}

#[test]
fn trapping_vs_saturating_trunc_differ_out_of_range() {
    // Same input, two ops: trunc_sat clamps, trunc traps.
    let sat = r#"
func (f32) -> (i32) {
block 0 (v0: f32) {
  v1 = i32.trunc_sat_f32_s v0
  return v1
  }
}
"#;
    let trap = r#"
func (f32) -> (i32) {
block 0 (v0: f32) {
  v1 = i32.trunc_f32_s v0
  return v1
  }
}
"#;
    assert_eq!(
        run1at(sat, 0, &[Value::F32(1e30)]),
        Ok(vec![Value::I32(i32::MAX)])
    );
    assert_eq!(
        run1at(trap, 0, &[Value::F32(1e30)]),
        Err(Trap::BadConversion)
    );
}

#[test]
fn ptr_ops_roundtrip_and_compute() {
    // base + offset via ptr.add, bracketed by from_int/to_int provenance casts.
    let src = r#"
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  v2 = ptr.from_int v0
  v3 = ptr.add v2 v1
  v4 = ptr.to_int v3
  return v4
  }
}
"#;
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");
    assert_eq!(m, decode_module(&encode_module(&m)).unwrap(), "binary");
    assert_eq!(m, parse_module(&print_module(&m)).unwrap(), "text");
    // off-CHERI these are i64 identity / wrapping add: 1000 + 24 = 1024.
    assert_eq!(
        run1at(src, 0, &[Value::I64(1000), Value::I64(24)]),
        Ok(vec![Value::I64(1024)])
    );
}

// ---- verifier fail-closed coverage for the newer ops (escape-TCB rejection paths) ----

/// Every program here is ill-typed/ill-formed in one specific way; the verifier
/// must reject each (it is the TCB contract that a bad module never verifies).
#[test]
fn verifier_rejects_newer_op_violations() {
    let cases: &[(&str, &str)] = &[
        // call_indirect index operand must be i32, not i64.
        (
            "indirect index not i32",
            r#"
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = call_indirect (i64) -> (i64) v0 (v0)
  return v1
  }
}
"#,
        ),
        // ptr.add operands must be i64.
        (
            "ptr.add on i32",
            r#"
func (i32) -> (i64) {
block 0 (v0: i32) {
  v1 = ptr.add v0 v0
  return v1
  }
}
"#,
        ),
        // trapping trunc operand must be a float, not an integer.
        (
            "trunc on i32",
            r#"
func (i32) -> (i32) {
block 0 (v0: i32) {
  v1 = i32.trunc_f64_s v0
  return v1
  }
}
"#,
        ),
        // a store needs the address to be i64.
        (
            "store address not i64",
            r#"
memory 16
func (i32, i32) -> (i32) {
block 0 (v0: i32, v1: i32) {
  i32.store v0 v1
  return v1
  }
}
"#,
        ),
    ];
    for (name, src) in cases {
        let m = parse_module(src).unwrap_or_else(|e| panic!("{name}: parse failed: {e}"));
        assert!(
            verify_module(&m).is_err(),
            "{name}: verifier accepted an ill-typed module"
        );
    }
}

#[test]
fn verifier_rejects_oversized_memory() {
    // A window of 1 << 64 is not representable -> rejected.
    let m = parse_module("memory 64\nfunc () -> () {\nblock 0 () {\n  return\n  }\n}\n").unwrap();
    assert!(matches!(
        verify_module(&m),
        Err(VerifyError::MemorySizeTooLarge { .. })
    ));
}

#[test]
fn verifier_rejects_ref_func_out_of_range() {
    let m = parse_module(
        "func (i32) -> (i32) {\nblock 0 (v0: i32) {\n  v1 = ref.func 9\n  return v0\n  }\n}\n",
    )
    .unwrap();
    assert!(matches!(
        verify_module(&m),
        Err(VerifyError::CallFuncOutOfRange { .. })
    ));
}

// ---- capabilities: cap.call, the host-owned handle table, the mock powerbox (§3c/§3e) ----

// Store "Hi" into the window, then write(ptr=0, len=2) to the Stream handle (v0).
const CAP_WRITE: &str = r#"
memory 16

func (i32) -> (i64) {
block 0 (v0: i32) {
  v1 = i64.const 0
  v2 = i32.const 72
  i32.store8 v1 v2
  v3 = i64.const 1
  v4 = i32.const 105
  i32.store8 v3 v4
  v5 = i64.const 0
  v6 = i64.const 2
  v7 = cap.call 0 1 (i64, i64) -> (i64) v0 (v5, v6)
  return v7
  }
}
"#;

#[test]
fn cap_stream_write_captures_output() {
    let m = parse_module(CAP_WRITE).expect("parse");
    verify_module(&m).expect("verify");
    // cap.call round-trips through both formats.
    assert_eq!(m, decode_module(&encode_module(&m)).unwrap(), "binary");
    assert_eq!(m, parse_module(&print_module(&m)).unwrap(), "text");

    let mut host = Host::new();
    let stdout = host.grant_stream(StreamRole::Out);
    let mut fuel = 10_000u64;
    let r = run_with_host(&m, 0, &[Value::I32(stdout)], &mut fuel, &mut host).unwrap();
    assert_eq!(r, vec![Value::I64(2)], "write returns the byte count");
    assert_eq!(host.stdout, b"Hi", "host captured the written bytes");
}

// read(ptr=0, len=4) from the Stream handle, then load the first byte back.
const CAP_READ: &str = r#"
memory 16

func (i32) -> (i32) {
block 0 (v0: i32) {
  v1 = i64.const 0
  v2 = i64.const 4
  v3 = cap.call 0 0 (i64, i64) -> (i64) v0 (v1, v2)
  v4 = i32.load8_u v1
  return v4
  }
}
"#;

#[test]
fn cap_stream_read_fills_memory() {
    let m = parse_module(CAP_READ).unwrap();
    verify_module(&m).unwrap();
    let mut host = Host::new();
    host.stdin = b"ABCD".to_vec();
    let stdin = host.grant_stream(StreamRole::In);
    let mut fuel = 10_000u64;
    let r = run_with_host(&m, 0, &[Value::I32(stdin)], &mut fuel, &mut host).unwrap();
    assert_eq!(
        r,
        vec![Value::I32(65)],
        "first byte 'A' read into the window"
    );
}

// exit(code) is noreturn; the frontend emits `unreachable` after it.
const CAP_EXIT: &str = r#"
func (i32) -> () {
block 0 (v0: i32) {
  v1 = i32.const 7
  cap.call 1 0 (i32) -> () v0 (v1)
  unreachable
  }
}
"#;

#[test]
fn cap_exit_traps_with_code() {
    let m = parse_module(CAP_EXIT).unwrap();
    verify_module(&m).unwrap();
    let mut host = Host::new();
    let exit = host.grant_exit();
    let mut fuel = 1000u64;
    let r = run_with_host(&m, 0, &[Value::I32(exit)], &mut fuel, &mut host);
    assert_eq!(r, Err(Trap::Exit(7)));
}

const CAP_CLOCK: &str = r#"
func (i32) -> (i64) {
block 0 (v0: i32) {
  v1 = i32.const 0
  v2 = cap.call 2 0 (i32) -> (i64) v0 (v1)
  return v2
  }
}
"#;

#[test]
fn cap_clock_is_deterministic_and_monotonic() {
    let m = parse_module(CAP_CLOCK).unwrap();
    verify_module(&m).unwrap();
    let mut host = Host::new();
    let clk = host.grant_clock();
    let mut fuel = 1000u64;
    let a = run_with_host(&m, 0, &[Value::I32(clk)], &mut fuel, &mut host).unwrap();
    let mut fuel = 1000u64;
    let b = run_with_host(&m, 0, &[Value::I32(clk)], &mut fuel, &mut host).unwrap();
    assert_eq!(a, vec![Value::I64(0)]);
    assert_eq!(b, vec![Value::I64(1)], "clock advances deterministically");
}

#[test]
fn cap_forged_wrong_type_and_closed_handles_are_inert() {
    // The CAP_CLOCK program calls type_id=2 (Clock). Each bad handle must be inert
    // (Trap::CapFault) — never an escape, never a panic.
    let m = parse_module(CAP_CLOCK).unwrap();
    verify_module(&m).unwrap();

    // (a) A forged index into an empty table.
    let mut host = Host::new();
    let mut fuel = 1000u64;
    assert_eq!(
        run_with_host(&m, 0, &[Value::I32(0x7fff)], &mut fuel, &mut host),
        Err(Trap::CapFault),
    );

    // (b) A real handle of the WRONG interface type (a Stream invoked as a Clock).
    let mut host = Host::new();
    let stream = host.grant_stream(StreamRole::Out);
    let mut fuel = 1000u64;
    assert_eq!(
        run_with_host(&m, 0, &[Value::I32(stream)], &mut fuel, &mut host),
        Err(Trap::CapFault),
    );

    // (c) A closed handle — dead generation. Since I41 (graceful revocation) a once-valid
    // revoked handle completes with the probeable `-EBADF` errno (the same value the §3.6
    // revocation-unpark delivers) instead of trapping — still inert, still no escape; the
    // trap stays reserved for the forgery in (a) and the type confusion in (b).
    let mut host = Host::new();
    let clk = host.grant_clock();
    host.close(clk);
    let mut fuel = 1000u64;
    assert_eq!(
        run_with_host(&m, 0, &[Value::I32(clk)], &mut fuel, &mut host),
        Ok(vec![Value::I64(-9)]),
    );
}

#[test]
fn cap_reusing_a_closed_slot_does_not_alias_old_handle() {
    // Close a handle, then grant another (reusing the slot). The old handle value must
    // still be inert — the per-slot generation advanced (ABA-safe). Since I41 the stale
    // handle answers the revocation errno (it was once valid) rather than trapping; the
    // point pinned here is that it must NEVER resolve to the new grant — a successful
    // clock read (`0`) through the old handle would be the aliasing escape.
    let m = parse_module(CAP_CLOCK).unwrap();
    verify_module(&m).unwrap();
    let mut host = Host::new();
    let old = host.grant_clock();
    host.close(old);
    let _new = host.grant_clock(); // reuses the freed slot, bumps its generation
    let mut fuel = 1000u64;
    assert_eq!(
        run_with_host(&m, 0, &[Value::I32(old)], &mut fuel, &mut host),
        Ok(vec![Value::I64(-9)]),
        "stale handle must not alias the new grant in the same slot",
    );
}

// A buffer that runs past the window: the §7 trampoline reports -EFAULT (recoverable),
// NOT a trap — and writes nothing.
const CAP_WRITE_OOB: &str = r#"
memory 16

func (i32) -> (i64) {
block 0 (v0: i32) {
  v1 = i64.const 0
  v2 = i64.const 100000
  v3 = cap.call 0 1 (i64, i64) -> (i64) v0 (v1, v2)
  return v3
  }
}
"#;

#[test]
fn cap_buffer_out_of_range_is_efault_not_trap() {
    let m = parse_module(CAP_WRITE_OOB).unwrap();
    verify_module(&m).unwrap();
    let mut host = Host::new();
    let stdout = host.grant_stream(StreamRole::Out);
    let mut fuel = 10_000u64;
    let r = run_with_host(&m, 0, &[Value::I32(stdout)], &mut fuel, &mut host).unwrap();
    assert_eq!(r, vec![Value::I64(-14)], "-EFAULT, a recoverable guest bug");
    assert!(host.stdout.is_empty(), "nothing written on EFAULT");
}

#[test]
fn verifier_rejects_cap_call_non_i32_handle() {
    // The handle operand must be the i32 index; an i64 handle is rejected.
    let m = parse_module(
        "func (i64) -> (i64) {\nblock 0 (v0: i64) {\n  v1 = cap.call 2 0 () -> (i64) v0 ()\n  return v1\n  }\n}\n",
    )
    .unwrap();
    assert!(matches!(
        verify_module(&m),
        Err(VerifyError::TypeMismatch { .. })
    ));
}
