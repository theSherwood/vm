//! Spec for the arithmetic peephole folds (OPT.md Phase 2). Each asserts behavior is preserved on
//! the reference interpreter and that the targeted op folds away.

use svm_interp::{Trap, Value};
use svm_ir::{Block, CmpOp, Func, Inst, IntTy, Memory, Module, Terminator, ValType};
use svm_opt::optimize_module;
use svm_verify::verify_module;

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
