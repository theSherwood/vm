//! Spec for the branch & select simplifications (OPT.md Phase 2): a `br_if`/`br_table` whose targets
//! all coincide is unconditional (the condition dies), and a `select` with equal arms is a copy.
//! These compound with SCCP/GVN, whose output often contains such degenerate branches/selects.
//! Asserts behavior is preserved and the degenerate construct (plus its now-dead selector) is gone.

use svm_interp::{Trap, Value};
use svm_ir::{BinOp, Block, Func, Inst, IntTy, Memory, Module, Terminator, ValType};
use svm_opt::optimize_module;
use svm_verify::verify_module;

fn module(f: Func) -> Module {
    Module {
        funcs: vec![f],
        memory: Some(Memory { size_log2: 16 }),
        data: vec![],
        imports: vec![],
        exports: vec![],
        impl_exports: vec![],
        debug_info: None,
    }
}

fn run(m: &Module, args: &[Value]) -> Result<Vec<Value>, Trap> {
    let mut fuel = 1_000_000u64;
    svm_interp::run(m, 0, args, &mut fuel)
}

fn check_equiv(orig: &Module, argsets: &[Vec<Value>]) -> Module {
    verify_module(orig).expect("original verifies");
    let opt = optimize_module(orig);
    verify_module(&opt).expect("optimized re-verifies");
    for args in argsets {
        assert_eq!(
            run(orig, args),
            run(&opt, args),
            "behavioral divergence on args {args:?}"
        );
    }
    opt
}

fn count_insts<F: Fn(&Inst) -> bool>(m: &Module, pred: F) -> usize {
    m.funcs
        .iter()
        .flat_map(|f| f.blocks.iter())
        .flat_map(|b| b.insts.iter())
        .filter(|i| pred(i))
        .count()
}

fn count_terms<F: Fn(&Terminator) -> bool>(m: &Module, pred: F) -> usize {
    m.funcs
        .iter()
        .flat_map(|f| f.blocks.iter())
        .filter(|b| pred(&b.term))
        .count()
}

#[test]
fn coincident_br_if_becomes_unconditional_and_condition_dies() {
    // b0(a): c = a + 1 ; if c { b1(a) } else { b1(a) }   — both arms identical.
    // b1(x): return x.  Simplifies to `br b1(a)`; then `c` is dead, and b1 merges into b0.
    let f = Func {
        params: vec![ValType::I32],
        results: vec![ValType::I32],
        blocks: vec![
            Block {
                params: vec![ValType::I32], // a
                insts: vec![
                    Inst::ConstI32(1),
                    Inst::IntBin {
                        ty: IntTy::I32,
                        op: BinOp::Add,
                        a: 0,
                        b: 1,
                    }, // c = a + 1
                ],
                term: Terminator::BrIf {
                    cond: 2,
                    then_blk: 1,
                    then_args: vec![0],
                    else_blk: 1,
                    else_args: vec![0],
                },
            },
            Block {
                params: vec![ValType::I32], // x
                insts: vec![],
                term: Terminator::Return(vec![0]),
            },
        ],
    };
    let m = module(f);
    let opt = check_equiv(
        &m,
        &[
            vec![Value::I32(0)],
            vec![Value::I32(5)],
            vec![Value::I32(-1)],
        ],
    );
    assert_eq!(
        count_terms(&opt, |t| matches!(t, Terminator::BrIf { .. })),
        0,
        "the coincident br_if should become unconditional"
    );
    assert_eq!(
        count_insts(&opt, |i| matches!(i, Inst::IntBin { .. })),
        0,
        "the now-dead condition should be eliminated"
    );
    assert_eq!(run(&opt, &[Value::I32(42)]), Ok(vec![Value::I32(42)]));
}

#[test]
fn coincident_br_table_becomes_unconditional() {
    // b0(a, idx): br_table idx [b1(a), b1(a)] default b1(a)  — every edge identical.
    let f = Func {
        params: vec![ValType::I32, ValType::I32],
        results: vec![ValType::I32],
        blocks: vec![
            Block {
                params: vec![ValType::I32, ValType::I32], // a, idx
                insts: vec![],
                term: Terminator::BrTable {
                    idx: 1,
                    targets: vec![(1, vec![0]), (1, vec![0])],
                    default: (1, vec![0]),
                },
            },
            Block {
                params: vec![ValType::I32],
                insts: vec![],
                term: Terminator::Return(vec![0]),
            },
        ],
    };
    let m = module(f);
    let opt = check_equiv(
        &m,
        &[
            vec![Value::I32(7), Value::I32(0)],
            vec![Value::I32(7), Value::I32(1)],
            vec![Value::I32(7), Value::I32(99)],
        ],
    );
    assert_eq!(
        count_terms(&opt, |t| matches!(t, Terminator::BrTable { .. })),
        0,
        "the all-coincident br_table should become unconditional"
    );
    assert_eq!(
        run(&opt, &[Value::I32(7), Value::I32(3)]),
        Ok(vec![Value::I32(7)])
    );
}

#[test]
fn select_with_equal_arms_folds_to_a_copy() {
    // s = select(cond, a, a) → a, regardless of cond; the select disappears.
    let f = Func {
        params: vec![ValType::I32, ValType::I32], // cond, a
        results: vec![ValType::I32],
        blocks: vec![Block {
            params: vec![ValType::I32, ValType::I32],
            insts: vec![Inst::Select {
                cond: 0,
                a: 1,
                b: 1,
            }], // v2 = select(cond, a, a)
            term: Terminator::Return(vec![2]),
        }],
    };
    let m = module(f);
    assert_eq!(count_insts(&m, |i| matches!(i, Inst::Select { .. })), 1);
    let opt = check_equiv(
        &m,
        &[
            vec![Value::I32(0), Value::I32(9)],
            vec![Value::I32(1), Value::I32(-4)],
        ],
    );
    assert_eq!(
        count_insts(&opt, |i| matches!(i, Inst::Select { .. })),
        0,
        "select with equal arms should fold away"
    );
    assert_eq!(
        run(&opt, &[Value::I32(1), Value::I32(77)]),
        Ok(vec![Value::I32(77)])
    );
}
