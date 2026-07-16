//! Spec for jump threading (OPT.md Phase 2). A branch that computes a flag and, through an empty
//! forwarder block, immediately re-tests that flag — the *correlated-branch* pattern SCCP cannot
//! catch, since the forwarder's selector parameter is a different constant on each incoming edge —
//! must (1) still produce identical results on the reference interpreter, and (2) have the empty
//! forwarder eliminated (the edge threaded straight to the resolved target).

use svm_interp::{Trap, Value};
use svm_ir::{BinOp, Block, CmpOp, Func, Inst, IntTy, Module, Terminator, ValType};
use svm_opt::optimize_module;
use svm_verify::verify_module;

fn module(f: Func) -> Module {
    Module {
        funcs: vec![f],
        memory: None,
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

/// A block that is empty (no instructions) and ends in a conditional branch on one of its own
/// parameters — the forwarder jump threading targets.
fn is_empty_conditional_forwarder(b: &Block) -> bool {
    b.insts.is_empty() && matches!(b.term, Terminator::BrIf { .. } | Terminator::BrTable { .. })
}

/// `f(x)`:
///   b0(x):  z = (x == 5); flag = z ? 1 : 0; br_if z, b1(x, 1), b1(x, 0)
///   b1(x', flag): br_if flag, b2(x'), b3(x')          // empty forwarder; SCCP sees flag = ⊥
///   b2(y): return y + 100
///   b3(y): return y + 200
/// So x == 5 → 105, otherwise x + 200. The flag passed into b1 is a *distinct* constant on each
/// edge, so SCCP's meet is not constant — only threading resolves b1's branch per edge.
fn correlated_branch() -> Module {
    module(Func {
        params: vec![ValType::I32], // x = v0
        results: vec![ValType::I32],
        blocks: vec![
            Block {
                params: vec![ValType::I32], // x = v0
                insts: vec![
                    Inst::ConstI32(5), // v1
                    Inst::IntCmp {
                        ty: IntTy::I32,
                        op: CmpOp::Eq,
                        a: 0,
                        b: 1,
                    }, // v2 = (x == 5)
                    Inst::ConstI32(1), // v3
                    Inst::ConstI32(0), // v4
                ],
                term: Terminator::BrIf {
                    cond: 2,
                    then_blk: 1,
                    then_args: vec![0, 3], // (x, flag=1)
                    else_blk: 1,
                    else_args: vec![0, 4], // (x, flag=0)
                },
            },
            Block {
                params: vec![ValType::I32, ValType::I32], // x' = v0, flag = v1
                insts: vec![],
                term: Terminator::BrIf {
                    cond: 1,
                    then_blk: 2,
                    then_args: vec![0],
                    else_blk: 3,
                    else_args: vec![0],
                },
            },
            Block {
                params: vec![ValType::I32], // y = v0
                insts: vec![
                    Inst::ConstI32(100), // v1
                    Inst::IntBin {
                        ty: IntTy::I32,
                        op: BinOp::Add,
                        a: 0,
                        b: 1,
                    }, // v2
                ],
                term: Terminator::Return(vec![2]),
            },
            Block {
                params: vec![ValType::I32], // y = v0
                insts: vec![
                    Inst::ConstI32(200), // v1
                    Inst::IntBin {
                        ty: IntTy::I32,
                        op: BinOp::Add,
                        a: 0,
                        b: 1,
                    }, // v2
                ],
                term: Terminator::Return(vec![2]),
            },
        ],
    })
}

#[test]
fn correlated_branch_is_threaded_through_the_forwarder() {
    let m = correlated_branch();
    verify_module(&m).expect("verifies");
    assert!(
        m.funcs[0].blocks.iter().any(is_empty_conditional_forwarder),
        "precondition: the module starts with an empty conditional forwarder (b1)"
    );

    let opt = optimize_module(&m);
    verify_module(&opt).expect("optimized (threaded) re-verifies");

    // Behavior is preserved on the reference interpreter across both branch outcomes.
    for x in [-3i32, 0, 4, 5, 6, 100] {
        let args = [Value::I32(x)];
        assert_eq!(run(&m, &args), run(&opt, &args), "divergence at x={x}");
    }

    // The forwarder is gone: threading redirected b0's edges straight to b2 / b3, leaving b1 with no
    // predecessors, and the cleanup pruned it.
    assert!(
        !opt.funcs[0]
            .blocks
            .iter()
            .any(is_empty_conditional_forwarder),
        "jump threading should eliminate the empty conditional forwarder"
    );
    assert!(
        opt.funcs[0].blocks.len() < m.funcs[0].blocks.len(),
        "threading + prune should drop the forwarder block"
    );
}
