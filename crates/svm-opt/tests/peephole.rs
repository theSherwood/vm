//! Spec for the arithmetic peephole folds (OPT.md Phase 2). Each asserts behavior is preserved on
//! the reference interpreter and that the targeted op folds away.

use svm_interp::{Trap, Value};
use svm_ir::{BinOp, Block, CmpOp, Func, Inst, IntTy, Memory, Module, Terminator, ValType};
use svm_opt::optimize_module;
use svm_verify::verify_module;

fn bin(op: BinOp, a: u32, b: u32) -> Inst {
    Inst::IntBin {
        ty: IntTy::I32,
        op,
        a,
        b,
    }
}

fn module(f: Func) -> Module {
    Module {
        funcs: vec![f],
        memory: Some(Memory { size_log2: 16 }),
        data: vec![],
        imports: vec![],
        exports: vec![],
        debug_info: None,
    }
}

fn run(m: &Module, args: &[Value]) -> Result<Vec<Value>, Trap> {
    let mut fuel = 1_000_000u64;
    svm_interp::run(m, 0, args, &mut fuel)
}

fn count<F: Fn(&Inst) -> bool>(m: &Module, pred: F) -> usize {
    m.funcs
        .iter()
        .flat_map(|f| f.blocks.iter())
        .flat_map(|b| b.insts.iter())
        .filter(|i| pred(i))
        .count()
}

/// `cmp(op, a, a)` for a single value `a` folds to the constant self-comparison result.
fn self_cmp_module(op: CmpOp) -> Module {
    module(Func {
        params: vec![ValType::I32],
        results: vec![ValType::I32],
        blocks: vec![Block {
            params: vec![ValType::I32], // a
            insts: vec![Inst::IntCmp {
                ty: IntTy::I32,
                op,
                a: 0,
                b: 0,
            }],
            term: Terminator::Return(vec![1]),
        }],
    })
}

#[test]
fn integer_self_comparison_folds_to_a_constant() {
    // Expected result of comparing a value with itself, for every integer compare op.
    let cases = [
        (CmpOp::Eq, 1),
        (CmpOp::Ne, 0),
        (CmpOp::LtS, 0),
        (CmpOp::LtU, 0),
        (CmpOp::LeS, 1),
        (CmpOp::LeU, 1),
        (CmpOp::GtS, 0),
        (CmpOp::GtU, 0),
        (CmpOp::GeS, 1),
        (CmpOp::GeU, 1),
    ];
    for (op, expected) in cases {
        let m = self_cmp_module(op);
        verify_module(&m).expect("verifies");
        let opt = optimize_module(&m);
        verify_module(&opt).expect("re-verifies");
        // The compare folded to a constant — no IntCmp remains.
        assert_eq!(
            count(&opt, |i| matches!(i, Inst::IntCmp { .. })),
            0,
            "self-compare {op:?} should fold"
        );
        // ...and to the right constant, for several input values (must not depend on `a`).
        for a in [0i32, 1, -1, i32::MIN, i32::MAX] {
            assert_eq!(run(&m, &[Value::I32(a)]), run(&opt, &[Value::I32(a)]));
            assert_eq!(run(&opt, &[Value::I32(a)]), Ok(vec![Value::I32(expected)]));
        }
    }
}

// ---- constant reassociation ----

/// Build `fn(x: i32) -> i32` computing a chain of `x OP c1 OP c2 ...` and return (module, expected
/// number of surviving IntBin ops after optimization, the closed-form result fn).
fn reassoc_module(op: BinOp, consts: &[i32]) -> Module {
    // v0 = x; then for each c: const c, then acc = acc OP c.
    let mut insts = Vec::new();
    let mut acc = 0u32; // v0 = x
    let mut next = 1u32;
    for &c in consts {
        insts.push(Inst::ConstI32(c));
        let cidx = next;
        next += 1;
        insts.push(bin(op, acc, cidx));
        acc = next;
        next += 1;
    }
    module(Func {
        params: vec![ValType::I32],
        results: vec![ValType::I32],
        blocks: vec![Block {
            params: vec![ValType::I32],
            insts,
            term: Terminator::Return(vec![acc]),
        }],
    })
}

#[test]
fn constant_chain_reassociates_to_a_single_op() {
    // (((x + 1) + 2) + 3) → x + 6 : three adds collapse to one.
    for (op, consts) in [
        (BinOp::Add, &[1, 2, 3][..]),
        (BinOp::Mul, &[2, 3, 5][..]),
        (BinOp::And, &[0xF0, 0x3C][..]),
        (BinOp::Or, &[1, 2, 4][..]),
        (BinOp::Xor, &[5, 6, 7][..]),
    ] {
        let m = reassoc_module(op, consts);
        verify_module(&m).expect("verifies");
        let opt = optimize_module(&m);
        verify_module(&opt).expect("re-verifies");
        // The whole constant chain folds to `x OP <combined>` — exactly one IntBin remains.
        assert_eq!(
            count(&opt, |i| matches!(i, Inst::IntBin { .. })),
            1,
            "chain of {op:?} over {consts:?} should collapse to one op"
        );
        for x in [0i32, 1, -1, 123, i32::MIN, i32::MAX] {
            assert_eq!(run(&m, &[Value::I32(x)]), run(&opt, &[Value::I32(x)]));
        }
    }
}

#[test]
fn reassociation_exposes_cse() {
    // (x + 4) + 4  and  (x + 6) + 2  both become x + 8, then CSE folds them to one add.
    // b0(x): a = (x+4)+4 ; b = (x+6)+2 ; return a + b   → 2*(x+8), a single `x+8` shared.
    let m = module(Func {
        params: vec![ValType::I32],
        results: vec![ValType::I32],
        blocks: vec![Block {
            params: vec![ValType::I32], // v0 = x
            insts: vec![
                Inst::ConstI32(4),     // v1
                bin(BinOp::Add, 0, 1), // v2 = x+4
                Inst::ConstI32(4),     // v3
                bin(BinOp::Add, 2, 3), // v4 = (x+4)+4
                Inst::ConstI32(6),     // v5
                bin(BinOp::Add, 0, 5), // v6 = x+6
                Inst::ConstI32(2),     // v7
                bin(BinOp::Add, 6, 7), // v8 = (x+6)+2
                bin(BinOp::Add, 4, 8), // v9 = a + b
            ],
            term: Terminator::Return(vec![9]),
        }],
    });
    verify_module(&m).expect("verifies");
    let opt = optimize_module(&m);
    verify_module(&opt).expect("re-verifies");
    for x in [0i32, 7, -20, i32::MAX] {
        assert_eq!(run(&m, &[Value::I32(x)]), run(&opt, &[Value::I32(x)]));
    }
    // Both branches reassociate to `x + 8`; CSE shares it, leaving `x+8` and the final sum.
    assert_eq!(
        count(&opt, |i| matches!(i, Inst::IntBin { .. })),
        2,
        "the two x+8 recomputations should share one op after reassociation + CSE"
    );
}
