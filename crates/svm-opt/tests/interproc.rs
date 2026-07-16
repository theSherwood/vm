//! Spec for the interprocedural passes (OPT.md Phase 3). Slice 1: dead-function elimination must
//! (1) drop functions no reachable code can call, (2) renumber survivors and remap every static
//! funcidx reference + export target, (3) keep the reachable functions behaving identically on the
//! reference interpreter, and (4) conservatively leave a module untouched while indirect funcref
//! dispatch (`call_indirect`) is present, since a funcref equals its funcidx (identity table).

use svm_interp::{Trap, Value};
use svm_ir::{BinOp, Block, Export, Func, FuncType, Inst, IntTy, Module, Terminator, ValType};
use svm_opt::interproc::dead_func_elim;
use svm_opt::optimize_module;
use svm_verify::verify_module;

fn run(m: &Module, func: u32, args: &[Value]) -> Result<Vec<Value>, Trap> {
    let mut fuel = 1_000_000u64;
    svm_interp::run(m, func, args, &mut fuel)
}

fn add(a: u32, b: u32) -> Inst {
    Inst::IntBin {
        ty: IntTy::I32,
        op: BinOp::Add,
        a,
        b,
    }
}

/// `f(a) = a + k`, a one-block leaf function.
fn add_const(k: i32) -> Func {
    Func {
        params: vec![ValType::I32],
        results: vec![ValType::I32],
        blocks: vec![Block {
            params: vec![ValType::I32],
            insts: vec![Inst::ConstI32(k), add(0, 1)],
            term: Terminator::Return(vec![2]),
        }],
    }
}

/// Entry (func 0) that returns `helper(a)` via a direct call to `callee`.
fn entry_calling(callee: u32) -> Func {
    Func {
        params: vec![ValType::I32],
        results: vec![ValType::I32],
        blocks: vec![Block {
            params: vec![ValType::I32],
            insts: vec![Inst::Call {
                func: callee,
                args: vec![0],
            }],
            term: Terminator::Return(vec![1]),
        }],
    }
}

#[test]
fn drops_uncalled_function_and_renumbers() {
    // func 0: entry, returns helper(a) = call func 2.
    // func 1: DEAD — nobody calls it, not exported.
    // func 2: the live helper (a + 1), between the dead func and the exported one so removal shifts
    //         its index (2 -> 1), exercising the funcidx + export remap.
    // func 3: exported "pub_helper" (a * 2 via a+a), a root, kept though nothing calls it.
    let pub_helper = Func {
        params: vec![ValType::I32],
        results: vec![ValType::I32],
        blocks: vec![Block {
            params: vec![ValType::I32],
            insts: vec![add(0, 0)], // a + a = a*2
            term: Terminator::Return(vec![1]),
        }],
    };
    let m = Module {
        funcs: vec![
            entry_calling(2), // 0
            add_const(999),   // 1 (dead)
            add_const(1),     // 2 (live helper)
            pub_helper,       // 3 (exported)
        ],
        exports: vec![Export {
            name: "pub_helper".into(),
            func: 3,
        }],
        ..Default::default()
    };
    verify_module(&m).expect("input verifies");

    let opt = dead_func_elim(&m);
    verify_module(&opt).expect("DFE output re-verifies");

    // The dead function is gone; the three reachable ones remain.
    assert_eq!(
        opt.funcs.len(),
        3,
        "the uncalled function should be dropped"
    );

    // The export name still resolves — to its *renumbered* index (3 -> 2).
    let ph = opt
        .resolve_export("pub_helper")
        .expect("export name survives");
    assert_eq!(ph, 2, "export target renumbered with the survivors");

    // Behavior preserved on the interpreter: the entry (which called the shifted helper) and the
    // exported function both still compute the right thing.
    for a in [-5i32, 0, 7, 100] {
        assert_eq!(
            run(&m, 0, &[Value::I32(a)]),
            run(&opt, 0, &[Value::I32(a)]),
            "entry divergence at a={a}"
        );
        assert_eq!(run(&opt, 0, &[Value::I32(a)]), Ok(vec![Value::I32(a + 1)]));
        // exported helper via its (remapped) index.
        assert_eq!(run(&opt, ph, &[Value::I32(a)]), Ok(vec![Value::I32(a * 2)]));
    }

    // The full pipeline drops it too (DFE runs at the end of optimize_module).
    assert_eq!(optimize_module(&m).funcs.len(), 3);
}

#[test]
fn keeps_all_functions_when_indirect_dispatch_present() {
    // func 0: entry — call_indirect(ref.func(2), a). func 1 is otherwise dead, but an indirect
    // dispatch could target any function (funcref == funcidx), so DFE must keep everything.
    let sig = FuncType {
        params: vec![ValType::I32],
        results: vec![ValType::I32],
    };
    let entry = Func {
        params: vec![ValType::I32],
        results: vec![ValType::I32],
        blocks: vec![Block {
            params: vec![ValType::I32],
            insts: vec![
                Inst::RefFunc { func: 2 }, // v1 = funcref(2)
                Inst::CallIndirect {
                    ty: sig,
                    idx: 1,
                    args: vec![0],
                }, // v2 = (*funcref)(a)
            ],
            term: Terminator::Return(vec![2]),
        }],
    };
    let m = Module {
        funcs: vec![
            entry,          // 0
            add_const(999), // 1 (uncalled, but must be kept — indirect dispatch present)
            add_const(1),   // 2 (indirect target)
        ],
        ..Default::default()
    };
    verify_module(&m).expect("input verifies");

    let opt = dead_func_elim(&m);
    assert_eq!(
        opt.funcs.len(),
        3,
        "DFE must be a no-op while call_indirect is present (funcref == funcidx)"
    );
    // And it is behavior-preserving (trivially — identity).
    for a in [0i32, 3, 42] {
        assert_eq!(
            run(&m, 0, &[Value::I32(a)]),
            run(&opt, 0, &[Value::I32(a)]),
            "divergence at a={a}"
        );
        assert_eq!(run(&opt, 0, &[Value::I32(a)]), Ok(vec![Value::I32(a + 1)]));
    }
}
