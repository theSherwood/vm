//! Differential spec for `MemCopy`/`MemFill` handling in the specializer — the feature that lets a
//! real interpreter's `TValue` struct moves (`llvm.memcpy`) fold. A source cell (a dynamic 8-byte
//! value + a constant 1-byte tag, the `TValue` shape) is copied within a rename region and read
//! back; the residual must equal the interpreter. Also covers the residual (no-rename) path.

use svm_interp::{Trap, Value};
use svm_ir::{
    BinOp, Block, Func, Inst, IntTy, LoadOp, Memory, Module, StoreOp, Terminator, ValType,
};
use svm_peval::{specialize_with_config, SpecArg, SpecConfig};
use svm_verify::verify_module;

fn run(m: &Module, args: &[Value]) -> Result<Vec<Value>, Trap> {
    let mut fuel = 1_000_000u64;
    svm_interp::run(m, 0, args, &mut fuel)
}

/// `f(x) = { S.val = x; S.tag = 7; memcpy(D, S, 16); return D.val + D.tag }` — a TValue-shaped copy.
/// S at 64, D at 128 (both constant addresses).
fn tvalue_copy_module() -> Module {
    let entry = Func {
        params: vec![ValType::I64],
        results: vec![ValType::I64],
        blocks: vec![Block {
            params: vec![ValType::I64], // x = v0
            insts: vec![
                Inst::ConstI64(64),  // v1: S
                Inst::ConstI64(128), // v2: D
                Inst::ConstI64(7),   // v3: tag
                Inst::Store {
                    op: StoreOp::I64,
                    addr: 1,
                    value: 0,
                    offset: 0,
                }, // S.val = x
                Inst::Store {
                    op: StoreOp::I64_8,
                    addr: 1,
                    value: 3,
                    offset: 8,
                }, // S.tag = 7
                Inst::ConstI64(16),  // v4: len
                Inst::MemCopy {
                    dst: 2,
                    src: 1,
                    len: 4,
                },
                Inst::Load {
                    op: LoadOp::I64,
                    addr: 2,
                    offset: 0,
                }, // v5: D.val
                Inst::Load {
                    op: LoadOp::I64_8U,
                    addr: 2,
                    offset: 8,
                }, // v6: D.tag
                Inst::IntBin {
                    ty: IntTy::I64,
                    op: BinOp::Add,
                    a: 5,
                    b: 6,
                }, // v7
            ],
            term: Terminator::Return(vec![7]),
        }],
    };
    Module {
        funcs: vec![entry],
        memory: Some(Memory { size_log2: 16 }),
        data: vec![],
        imports: vec![],
        exports: vec![],
        data_exports: vec![],
        data_ptrs: vec![],
        data_funcrefs: vec![],
        impl_exports: vec![],
        types: vec![],
        debug_info: None,
    }
}

#[test]
fn memcopy_in_rename_region_folds_and_matches() {
    let m = tvalue_copy_module();
    verify_module(&m).expect("source verifies");

    // Rename [64, 144) covers both S and D: the copy is modeled as abstract cell moves, so no
    // MemCopy (and no residual store/load of the region) survives.
    let cfg = SpecConfig {
        rename: Some((64, 144)),
        rename_is_private: true,
        ..SpecConfig::default()
    };
    let residual = specialize_with_config(&m, 0, &[SpecArg::Dynamic], &cfg).expect("specializes");
    verify_module(&residual).expect("residual verifies");
    let has_bulk = residual
        .funcs
        .iter()
        .flat_map(|f| &f.blocks)
        .flat_map(|b| &b.insts)
        .any(|i| matches!(i, Inst::MemCopy { .. } | Inst::MemFill { .. }));
    assert!(
        !has_bulk,
        "the copy should be modeled abstractly, not left as a residual MemCopy"
    );

    for x in [0i64, 1, -5, 1000, i64::MAX] {
        assert_eq!(
            run(&residual, &[Value::I64(x)]),
            run(&m, &[Value::I64(x)]),
            "diverged at x={x}"
        );
        assert_eq!(
            run(&residual, &[Value::I64(x)]),
            Ok(vec![Value::I64(x.wrapping_add(7))])
        );
    }
}

#[test]
fn memcopy_without_rename_stays_residual_and_matches() {
    let m = tvalue_copy_module();
    // No rename region: the copy is emitted as a residual MemCopy (memory stays faithful).
    let residual = specialize_with_config(&m, 0, &[SpecArg::Dynamic], &SpecConfig::default())
        .expect("specializes");
    verify_module(&residual).expect("residual verifies");
    assert!(
        residual
            .funcs
            .iter()
            .flat_map(|f| &f.blocks)
            .flat_map(|b| &b.insts)
            .any(|i| matches!(i, Inst::MemCopy { .. })),
        "a residual MemCopy should survive when the region is not renamed"
    );
    for x in [0i64, 42, -1, 999] {
        assert_eq!(
            run(&residual, &[Value::I64(x)]),
            run(&m, &[Value::I64(x)]),
            "diverged at x={x}"
        );
    }
}
