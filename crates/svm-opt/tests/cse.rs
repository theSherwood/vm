//! Spec for intra-block CSE (OPT.md Phase 2). Asserts (1) `optimize_module` preserves behavior on
//! the reference interpreter and (2) a redundant *pure* expression is deduped — while an impure one
//! (a load: memory may change, it may trap) is left untouched, which is the load-bearing safety line.

use svm_interp::{Trap, Value};
use svm_ir::{
    BinOp, Block, Func, Inst, IntTy, LoadOp, Memory, Module, StoreOp, Terminator, ValType,
};
use svm_opt::optimize_module;
use svm_verify::verify_module;

fn module(f: Func) -> Module {
    Module {
        interfaces: vec![],
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

fn count<F: Fn(&Inst) -> bool>(m: &Module, pred: F) -> usize {
    m.funcs
        .iter()
        .flat_map(|f| f.blocks.iter())
        .flat_map(|b| b.insts.iter())
        .filter(|i| pred(i))
        .count()
}

fn add(a: u32, b: u32) -> Inst {
    Inst::IntBin {
        ty: IntTy::I32,
        op: BinOp::Add,
        a,
        b,
    }
}

fn argsets() -> Vec<Vec<Value>> {
    vec![
        vec![Value::I32(0), Value::I32(0)],
        vec![Value::I32(3), Value::I32(4)],
        vec![Value::I32(-5), Value::I32(9)],
        vec![Value::I32(i32::MAX), Value::I32(1)], // wrapping add
    ]
}

#[test]
fn redundant_pure_expression_is_deduped() {
    // v2 = a + b ; v3 = a + b (redundant) ; v4 = v2 + v3 ; return v4   → 2*(a+b)
    let f = Func {
        params: vec![ValType::I32, ValType::I32],
        results: vec![ValType::I32],
        blocks: vec![Block {
            params: vec![ValType::I32, ValType::I32], // v0=a, v1=b
            insts: vec![add(0, 1), add(0, 1), add(2, 3)],
            term: Terminator::Return(vec![4]),
        }],
    };
    let m = module(f);
    assert_eq!(count(&m, |i| matches!(i, Inst::IntBin { .. })), 3);
    let opt = check_equiv(&m, &argsets());
    // The duplicate `a + b` is gone: 3 adds collapse to 2 (the shared `a+b`, then `+ itself`).
    assert_eq!(count(&opt, |i| matches!(i, Inst::IntBin { .. })), 2);
}

#[test]
fn equal_expressions_from_equal_subexpressions_are_deduped() {
    // v2 = a+b ; v3 = a+b ; v4 = v2+a ; v5 = v3+a ; return v4 + v5.
    // After canonicalizing v3→v2, v5 = v2+a matches v4 → deduped, so the result is (v2+a)*2.
    let f = Func {
        params: vec![ValType::I32, ValType::I32],
        results: vec![ValType::I32],
        blocks: vec![Block {
            params: vec![ValType::I32, ValType::I32], // v0=a, v1=b
            insts: vec![
                add(0, 1), // v2 = a+b
                add(0, 1), // v3 = a+b   (dup of v2)
                add(2, 0), // v4 = v2+a
                add(3, 0), // v5 = v3+a  (dup of v4 after canonicalization)
                add(4, 5), // v6 = v4+v5
            ],
            term: Terminator::Return(vec![6]),
        }],
    };
    let m = module(f);
    assert_eq!(count(&m, |i| matches!(i, Inst::IntBin { .. })), 5);
    let opt = check_equiv(&m, &argsets());
    // v2 (a+b), v4 (v2+a), v6 (v4+v4) survive — the two duplicates fold away.
    assert_eq!(count(&opt, |i| matches!(i, Inst::IntBin { .. })), 3);
}

#[test]
fn loads_across_a_may_alias_store_are_not_merged() {
    // Two loads of the same address with an intervening store to a **runtime** address must both
    // survive: a load is impure (memory may change between them, and it may trap), so CSE never treats
    // them as one value, and the Phase-4 memory pass clobbers on a store it can't prove disjoint (the
    // store address is a parameter, so never provably distinct from the load address). With no
    // intervening write the memory pass *does* forward such loads — see `tests/memopt.rs`; this pins
    // the conservative boundary that keeps that sound.
    let f = Func {
        params: vec![ValType::I32, ValType::I64], // v0 = value, v1 = store address (runtime)
        results: vec![ValType::I32],
        blocks: vec![Block {
            params: vec![ValType::I32, ValType::I64],
            insts: vec![
                Inst::ConstI64(0), // v2 = load address (const 0)
                Inst::Load {
                    op: LoadOp::I32,
                    addr: 2,
                    offset: 0,
                    align: 2,
                }, // v3 = mem[0]  (x)
                Inst::Store {
                    op: StoreOp::I32,
                    addr: 1,
                    value: 0,
                    offset: 0,
                    align: 2,
                }, // mem[b] = value — may alias mem[0]
                Inst::Load {
                    op: LoadOp::I32,
                    addr: 2,
                    offset: 0,
                    align: 2,
                }, // v4 = mem[0]  (y) — must NOT merge with x
                add(3, 4),         // v5 = x + y
            ],
            term: Terminator::Return(vec![5]),
        }],
    };
    let m = module(f);
    let opt = check_equiv(
        &m,
        &[
            vec![Value::I32(5), Value::I64(0)], // b == 0: store overwrites mem[0]
            vec![Value::I32(5), Value::I64(8)], // b != 0: disjoint
            vec![Value::I32(7), Value::I64(0)],
        ],
    );
    assert_eq!(
        count(&opt, |i| matches!(i, Inst::Load { .. })),
        2,
        "a may-alias store between identical loads must block merging them"
    );
    // Aliasing case: x = 0 (init), then mem[0] = 5, so y = 5 → 5.
    assert_eq!(
        run(&opt, &[Value::I32(5), Value::I64(0)]),
        Ok(vec![Value::I32(5)])
    );
}
