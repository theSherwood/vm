//! Spec for the interprocedural passes (OPT.md Phase 3). Slice 1: dead-function elimination must
//! (1) drop functions no reachable code can call, (2) renumber survivors and remap every static
//! funcidx reference + export target, (3) keep the reachable functions behaving identically on the
//! reference interpreter, and (4) conservatively leave a module untouched while indirect funcref
//! dispatch (`call_indirect`) is present, since a funcref equals its funcidx (identity table).

use svm_interp::{Trap, Value};
use svm_ir::{
    BinOp, Block, CmpOp, Export, Func, FuncType, Inst, IntTy, Module, Terminator, ValType,
};
use svm_opt::interproc::{const_prop, dead_func_elim, devirtualize, inline_calls};
use svm_opt::{optimize_module, optimize_module_with, OptConfig};
use svm_verify::verify_module;

fn n_call_indirect(m: &Module) -> usize {
    m.funcs
        .iter()
        .flat_map(|f| &f.blocks)
        .flat_map(|b| &b.insts)
        .filter(|i| matches!(i, Inst::CallIndirect { .. }))
        .count()
}

fn mul(a: u32, b: u32) -> Inst {
    Inst::IntBin {
        ty: IntTy::I32,
        op: BinOp::Mul,
        a,
        b,
    }
}

fn n_calls(m: &Module) -> usize {
    m.funcs
        .iter()
        .flat_map(|f| &f.blocks)
        .flat_map(|b| &b.insts)
        .filter(|i| matches!(i, Inst::Call { .. }))
        .count()
}

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
fn impl_export_ops_are_dfe_roots() {
    // An interface offer's op functions (IMPORTS.md §3.2) are entered from another domain via
    // wiring — invisible to intra-module reachability, so DFE must treat them as roots and remap
    // their funcidxs with the survivors.
    let m = Module {
        funcs: vec![
            entry_calling(2), // 0: entry
            add_const(999),   // 1: DEAD — uncalled, unoffered
            add_const(1),     // 2: live helper
            add_const(7),     // 3: only referenced by the offer's op list
        ],
        types: vec![
            svm_ir::TypeEntry::Func(FuncType {
                params: vec![ValType::I32],
                results: vec![ValType::I32],
            }),
            svm_ir::TypeEntry::Interface(vec![svm_ir::IfaceOp {
                name: "add".into(),
                ty: 0,
            }]),
        ],
        impl_exports: vec![svm_ir::ImplExport {
            name: "adder".into(),
            interface: 1,
            ops: vec![3],
        }],
        ..Default::default()
    };
    verify_module(&m).expect("input verifies");

    let opt = dead_func_elim(&m);
    verify_module(&opt).expect("DFE output re-verifies");
    assert_eq!(opt.funcs.len(), 3, "only the unrooted function drops");
    let offer = opt.resolve_impl_export("adder").expect("offer survives");
    assert_eq!(
        offer.ops,
        vec![2],
        "op funcidx renumbered with the survivors"
    );
    for a in [-5i32, 0, 7] {
        assert_eq!(
            run(&opt, offer.ops[0], &[Value::I32(a)]),
            Ok(vec![Value::I32(a + 7)]),
            "offered op behavior preserved at a={a}"
        );
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

    // The full pipeline is even stronger: it also *inlines* the single-block `a+1` helper into the
    // entry, so that leaf becomes dead too — leaving just the entry and the exported helper (2 funcs).
    let opt = optimize_module(&m);
    verify_module(&opt).expect("optimized re-verifies");
    assert_eq!(
        opt.funcs.len(),
        2,
        "inline + DFE remove both the dead func and the inlined leaf"
    );
    let ph2 = opt.resolve_export("pub_helper").expect("export survives");
    for a in [-5i32, 0, 7, 100] {
        assert_eq!(run(&m, 0, &[Value::I32(a)]), run(&opt, 0, &[Value::I32(a)]));
        assert_eq!(
            run(&opt, ph2, &[Value::I32(a)]),
            Ok(vec![Value::I32(a * 2)])
        );
    }
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

/// `helper(a, b) = a*3 + b*5 + 7`, a single-block leaf.
fn affine_helper() -> Func {
    Func {
        params: vec![ValType::I32, ValType::I32],
        results: vec![ValType::I32],
        blocks: vec![Block {
            params: vec![ValType::I32, ValType::I32],
            insts: vec![
                Inst::ConstI32(3),
                mul(0, 2), // a*3
                Inst::ConstI32(5),
                mul(1, 4), // b*5
                add(3, 5), // a*3 + b*5
                Inst::ConstI32(7),
                add(6, 7), // + 7
            ],
            term: Terminator::Return(vec![8]),
        }],
    }
}

#[test]
fn inlines_leaf_helper_then_dfe_removes_it() {
    // func 0: entry(a,b) = helper(a,b) via a direct call. func 1: the leaf helper.
    let entry = Func {
        params: vec![ValType::I32, ValType::I32],
        results: vec![ValType::I32],
        blocks: vec![Block {
            params: vec![ValType::I32, ValType::I32],
            insts: vec![Inst::Call {
                func: 1,
                args: vec![0, 1],
            }],
            term: Terminator::Return(vec![2]),
        }],
    };
    let m = Module {
        funcs: vec![entry, affine_helper()],
        ..Default::default()
    };
    verify_module(&m).expect("input verifies");

    // Inlining alone: the call is spliced away (helper still present, now uncalled).
    let inl = inline_calls(&m);
    verify_module(&inl).expect("inlined re-verifies");
    assert_eq!(n_calls(&inl), 0, "the direct call should be inlined away");
    assert_eq!(
        inl.funcs.len(),
        2,
        "inlining does not itself remove the callee"
    );
    for (a, b) in [(-3i32, 4i32), (0, 0), (7, 11), (100, -2)] {
        assert_eq!(
            run(&m, 0, &[Value::I32(a), Value::I32(b)]),
            run(&inl, 0, &[Value::I32(a), Value::I32(b)]),
            "divergence at ({a},{b})"
        );
    }

    // Full pipeline: inline → fold → DFE collapses to a single self-contained function.
    let opt = optimize_module(&m);
    verify_module(&opt).expect("optimized re-verifies");
    assert_eq!(
        opt.funcs.len(),
        1,
        "the inlined leaf is DCE'd as a function"
    );
    assert_eq!(n_calls(&opt), 0);
    for (a, b) in [(-3i32, 4i32), (0, 0), (7, 11), (100, -2)] {
        let args = [Value::I32(a), Value::I32(b)];
        assert_eq!(run(&m, 0, &args), run(&opt, 0, &args));
        assert_eq!(run(&opt, 0, &args), Ok(vec![Value::I32(a * 3 + b * 5 + 7)]));
    }
}

#[test]
fn inlines_with_live_code_after_the_call() {
    // entry(a) = inc(a) * 2, so a value flows *through* the call site and code runs after it — the
    // renumbering across the splice must keep it correct.
    let inc = Func {
        params: vec![ValType::I32],
        results: vec![ValType::I32],
        blocks: vec![Block {
            params: vec![ValType::I32],
            insts: vec![Inst::ConstI32(1), add(0, 1)],
            term: Terminator::Return(vec![2]),
        }],
    };
    let entry = Func {
        params: vec![ValType::I32],
        results: vec![ValType::I32],
        blocks: vec![Block {
            params: vec![ValType::I32],
            insts: vec![
                Inst::Call {
                    func: 1,
                    args: vec![0],
                }, // v1 = inc(a)
                Inst::ConstI32(2),
                mul(1, 2), // v3 = v1 * 2
            ],
            term: Terminator::Return(vec![3]),
        }],
    };
    let m = Module {
        funcs: vec![entry, inc],
        ..Default::default()
    };
    verify_module(&m).expect("input verifies");

    let inl = inline_calls(&m);
    verify_module(&inl).expect("inlined re-verifies");
    assert_eq!(n_calls(&inl), 0);
    for a in [-5i32, 0, 3, 21] {
        assert_eq!(
            run(&m, 0, &[Value::I32(a)]),
            run(&inl, 0, &[Value::I32(a)]),
            "divergence at a={a}"
        );
        assert_eq!(
            run(&inl, 0, &[Value::I32(a)]),
            Ok(vec![Value::I32((a + 1) * 2)])
        );
    }
}

fn cmp(op: CmpOp, a: u32, b: u32) -> Inst {
    Inst::IntCmp {
        ty: IntTy::I32,
        op,
        a,
        b,
    }
}

/// `abs(x)` as a three-block callee: `b0` tests `x < 0` and branches; `b1` returns `0 - x`; `b2`
/// returns `x`. Two return points joining at the inlined continuation.
fn abs_callee() -> Func {
    Func {
        params: vec![ValType::I32],
        results: vec![ValType::I32],
        blocks: vec![
            Block {
                params: vec![ValType::I32],                            // x = v0
                insts: vec![Inst::ConstI32(0), cmp(CmpOp::LtS, 0, 1)], // v1=0, v2 = x<0
                term: Terminator::BrIf {
                    cond: 2,
                    then_blk: 1,
                    then_args: vec![0],
                    else_blk: 2,
                    else_args: vec![0],
                },
            },
            Block {
                params: vec![ValType::I32], // x = v0
                insts: vec![
                    Inst::ConstI32(0),
                    Inst::IntBin {
                        // v1=0, v2 = 0 - x
                        ty: IntTy::I32,
                        op: BinOp::Sub,
                        a: 1,
                        b: 0,
                    },
                ],
                term: Terminator::Return(vec![2]),
            },
            Block {
                params: vec![ValType::I32], // x = v0
                insts: vec![],
                term: Terminator::Return(vec![0]),
            },
        ],
    }
}

#[test]
fn inlines_a_multiblock_callee_threading_a_captured_value() {
    // entry(a): k = 10; t = abs(a); return t + k
    // `k` is defined before the call and used after it, so inlining abs (three blocks, two returns)
    // must thread `k` through the callee's CFG to the continuation where `t + k` lives.
    let entry = Func {
        params: vec![ValType::I32],
        results: vec![ValType::I32],
        blocks: vec![Block {
            params: vec![ValType::I32], // a = v0
            insts: vec![
                Inst::ConstI32(10), // v1 = k
                Inst::Call {
                    func: 1,
                    args: vec![0],
                }, // v2 = abs(a)
                add(2, 1),          // v3 = t + k
            ],
            term: Terminator::Return(vec![3]),
        }],
    };
    let m = Module {
        funcs: vec![entry, abs_callee()],
        ..Default::default()
    };
    verify_module(&m).expect("input verifies");

    let inl = inline_calls(&m);
    verify_module(&inl).expect("inlined module re-verifies");
    assert_eq!(n_calls(&inl), 0, "the multi-block callee is inlined away");
    assert!(
        inl.funcs[0].blocks.len() > 1,
        "inlining a multi-block callee splits the caller into a CFG"
    );
    for a in [0i32, 4, -4, 40, -40, 1000, -1000] {
        assert_eq!(
            run(&m, 0, &[Value::I32(a)]),
            run(&inl, 0, &[Value::I32(a)]),
            "divergence at a={a}"
        );
        assert_eq!(
            run(&inl, 0, &[Value::I32(a)]),
            Ok(vec![Value::I32(a.wrapping_abs().wrapping_add(10))]),
            "abs(a)+10 at a={a}"
        );
    }
}

#[test]
fn multiblock_inline_through_the_full_pipeline() {
    // The whole optimizer must inline the multi-block callee, DFE the now-dead function, and preserve
    // behavior — with the output re-verified (guards the threaded block params + CFG splice).
    let entry = Func {
        params: vec![ValType::I32],
        results: vec![ValType::I32],
        blocks: vec![Block {
            params: vec![ValType::I32],
            insts: vec![Inst::Call {
                func: 1,
                args: vec![0],
            }],
            term: Terminator::Return(vec![1]),
        }],
    };
    let m = Module {
        funcs: vec![entry, abs_callee()],
        ..Default::default()
    };
    let opt = optimize_module(&m);
    verify_module(&opt).expect("optimized re-verifies");
    assert_eq!(n_calls(&opt), 0, "callee inlined");
    assert_eq!(opt.funcs.len(), 1, "the inlined-away callee is DFE'd");
    for a in [0i32, 7, -7, 123, -123] {
        assert_eq!(
            run(&m, 0, &[Value::I32(a)]),
            run(&opt, 0, &[Value::I32(a)]),
            "divergence at a={a}"
        );
    }
}

#[test]
fn inlines_a_callee_with_a_loop_threading_a_captured_value_around_the_back_edge() {
    // callee sum(n) = n + (n-1) + ... + 1, a counted loop (a back edge inside the callee).
    //   b0(n): br b1(n, 0)
    //   b1(i, acc): brif i!=0 -> b2(i, acc) else b3(acc)
    //   b2(i, acc): br b1(i-1, acc+i)
    //   b3(acc): return acc
    let sum = Func {
        params: vec![ValType::I32],
        results: vec![ValType::I32],
        blocks: vec![
            Block {
                params: vec![ValType::I32],     // n = v0
                insts: vec![Inst::ConstI32(0)], // v1 = 0
                term: Terminator::Br {
                    target: 1,
                    args: vec![0, 1],
                },
            },
            Block {
                params: vec![ValType::I32, ValType::I32], // i=v0, acc=v1
                insts: vec![Inst::ConstI32(0), cmp(CmpOp::Ne, 0, 2)], // v2=0, v3 = i!=0
                term: Terminator::BrIf {
                    cond: 3,
                    then_blk: 2,
                    then_args: vec![0, 1],
                    else_blk: 3,
                    else_args: vec![1],
                },
            },
            Block {
                params: vec![ValType::I32, ValType::I32], // i=v0, acc=v1
                insts: vec![
                    Inst::ConstI32(1), // v2 = 1
                    Inst::IntBin {
                        ty: IntTy::I32,
                        op: BinOp::Sub,
                        a: 0,
                        b: 2,
                    }, // v3 = i-1
                    add(1, 0),         // v4 = acc + i
                ],
                term: Terminator::Br {
                    target: 1,
                    args: vec![3, 4],
                },
            },
            Block {
                params: vec![ValType::I32], // acc = v0
                insts: vec![],
                term: Terminator::Return(vec![0]),
            },
        ],
    };
    // entry(a): k = 100; t = sum(a); return t + k  — `k` threads through the loop callee.
    let entry = Func {
        params: vec![ValType::I32],
        results: vec![ValType::I32],
        blocks: vec![Block {
            params: vec![ValType::I32],
            insts: vec![
                Inst::ConstI32(100),
                Inst::Call {
                    func: 1,
                    args: vec![0],
                },
                add(2, 1),
            ],
            term: Terminator::Return(vec![3]),
        }],
    };
    let m = Module {
        funcs: vec![entry, sum],
        ..Default::default()
    };
    verify_module(&m).expect("input verifies");
    let inl = inline_calls(&m);
    verify_module(&inl).expect("inlined module re-verifies");
    assert_eq!(n_calls(&inl), 0, "the loop callee is inlined away");
    // Also drive it through the whole optimizer (which re-runs the cleanup fixpoint over the CFG).
    let opt = optimize_module(&m);
    verify_module(&opt).expect("optimized re-verifies");
    for a in [0i32, 1, 2, 5, 10] {
        let want = (1..=a).sum::<i32>() + 100;
        for candidate in [&inl, &opt] {
            assert_eq!(
                run(candidate, 0, &[Value::I32(a)]),
                Ok(vec![Value::I32(want)]),
                "sum(1..={a})+100"
            );
        }
    }
}

#[test]
fn devirtualizes_constant_funcref_then_inlines_and_dfes() {
    // entry(a): call_indirect(ref.func(1), a) — a constant funcref whose target signature matches, so
    // it devirtualizes to a direct call to func 1, which then inlines; func 2 is dead. After the full
    // pipeline only the entry remains, computing a+1 with no calls at all.
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
                Inst::RefFunc { func: 1 }, // v1 = funcref(1)
                Inst::CallIndirect {
                    ty: sig.clone(),
                    idx: 1,
                    args: vec![0],
                }, // v2
            ],
            term: Terminator::Return(vec![2]),
        }],
    };
    let m = Module {
        funcs: vec![entry, add_const(1), add_const(100)],
        ..Default::default()
    };
    verify_module(&m).expect("input verifies");

    // Devirtualization alone rewrites the indirect call to a direct one (no renumbering).
    let dv = devirtualize(&m);
    verify_module(&dv).expect("devirtualized re-verifies");
    assert_eq!(
        n_call_indirect(&dv),
        0,
        "constant funcref call should devirtualize"
    );
    assert_eq!(n_calls(&dv), 1, "it becomes a direct call to func 1");
    for a in [-2i32, 0, 9] {
        assert_eq!(
            run(&m, 0, &[Value::I32(a)]),
            run(&dv, 0, &[Value::I32(a)]),
            "divergence at a={a}"
        );
    }

    // Full pipeline: devirt -> inline -> DFE collapses to a single function, no calls of either kind.
    let opt = optimize_module(&m);
    verify_module(&opt).expect("optimized re-verifies");
    assert_eq!(
        opt.funcs.len(),
        1,
        "the devirtualized+inlined targets are DCE'd"
    );
    assert_eq!(n_call_indirect(&opt), 0);
    assert_eq!(n_calls(&opt), 0);
    for a in [-2i32, 0, 9, 50] {
        assert_eq!(run(&m, 0, &[Value::I32(a)]), run(&opt, 0, &[Value::I32(a)]));
        assert_eq!(run(&opt, 0, &[Value::I32(a)]), Ok(vec![Value::I32(a + 1)]));
    }
}

#[test]
fn does_not_devirtualize_on_signature_mismatch() {
    // entry(a): call_indirect(ref.func(1), [a]) but the call's declared ty is (i32)->i32 while func 1
    // is (i32,i32)->i32 — a runtime signature mismatch that must *trap*. Devirtualizing to a direct
    // call would run the wrong function instead, so the indirect call must be left untouched.
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
                Inst::RefFunc { func: 1 }, // funcref to the 2-arg function
                Inst::CallIndirect {
                    ty: sig,
                    idx: 1,
                    args: vec![0],
                },
            ],
            term: Terminator::Return(vec![2]),
        }],
    };
    let two_arg = Func {
        params: vec![ValType::I32, ValType::I32],
        results: vec![ValType::I32],
        blocks: vec![Block {
            params: vec![ValType::I32, ValType::I32],
            insts: vec![add(0, 1)],
            term: Terminator::Return(vec![2]),
        }],
    };
    let m = Module {
        funcs: vec![entry, two_arg],
        ..Default::default()
    };
    verify_module(&m).expect("input verifies");

    let dv = devirtualize(&m);
    assert_eq!(
        n_call_indirect(&dv),
        1,
        "a signature mismatch must not be devirtualized (it must still trap)"
    );
    // The runtime signature check traps in both — identically.
    for a in [0i32, 5] {
        let r0 = run(&m, 0, &[Value::I32(a)]);
        let r1 = run(&dv, 0, &[Value::I32(a)]);
        assert!(r0.is_err(), "mismatched call_indirect should trap");
        assert_eq!(r0, r1, "trap preserved at a={a}");
    }
}

// ---- interprocedural constant propagation (const_prop) ----

fn n_brif(m: &Module) -> usize {
    m.funcs
        .iter()
        .flat_map(|f| &f.blocks)
        .filter(|b| matches!(b.term, Terminator::BrIf { .. }))
        .count()
}

/// `sel(flag, v) = flag ? v : -v`, a two-arm helper whose branch tests `flag`.
fn sel_helper() -> Func {
    Func {
        params: vec![ValType::I32, ValType::I32], // flag=v0, v=v1
        results: vec![ValType::I32],
        blocks: vec![
            Block {
                params: vec![ValType::I32, ValType::I32],
                insts: vec![Inst::ConstI32(0), cmp(CmpOp::Ne, 0, 2)], // v3 = flag != 0
                term: Terminator::BrIf {
                    cond: 3,
                    then_blk: 1,
                    then_args: vec![1],
                    else_blk: 2,
                    else_args: vec![1],
                },
            },
            Block {
                params: vec![ValType::I32], // v
                insts: vec![],
                term: Terminator::Return(vec![0]),
            },
            Block {
                params: vec![ValType::I32], // v
                insts: vec![
                    Inst::ConstI32(0),
                    Inst::IntBin {
                        ty: IntTy::I32,
                        op: BinOp::Sub,
                        a: 1,
                        b: 0,
                    },
                ], // -v
                term: Terminator::Return(vec![2]),
            },
        ],
    }
}

/// const_prop-only config (plus SCCP to fold the propagated constant), no inline/dfe — so the callee
/// survives and the effect of the substitution is observable on it.
fn cp_only() -> OptConfig {
    OptConfig {
        const_prop: true,
        sccp: true,
        ..OptConfig::none()
    }
}

#[test]
fn const_arg_folds_a_branch_in_the_callee() {
    // entry(x): sel(1, x) + sel(1, x)  — both sites pass flag=1, so const_prop makes `flag` constant in
    // `sel`, SCCP folds `flag != 0` to true, and the branch (and the negate arm) disappear.
    let entry = Func {
        params: vec![ValType::I32],
        results: vec![ValType::I32],
        blocks: vec![Block {
            params: vec![ValType::I32], // x
            insts: vec![
                Inst::ConstI32(1),
                Inst::Call {
                    func: 1,
                    args: vec![1, 0],
                }, // sel(1, x) -> v2
                Inst::ConstI32(1),
                Inst::Call {
                    func: 1,
                    args: vec![3, 0],
                }, // sel(1, x) -> v4
                add(2, 4),
            ],
            term: Terminator::Return(vec![5]),
        }],
    };
    let m = Module {
        funcs: vec![entry, sel_helper()],
        ..Default::default()
    };
    verify_module(&m).expect("input verifies");
    assert_eq!(n_brif(&m), 1, "precondition: the helper branches on flag");

    let opt = optimize_module_with(&m, &cp_only());
    verify_module(&opt).expect("re-verifies");
    assert_eq!(
        n_brif(&opt),
        0,
        "const_prop + sccp should fold the helper's branch away"
    );
    for x in [0i32, 3, -5, 100] {
        assert_eq!(run(&m, 0, &[Value::I32(x)]), run(&opt, 0, &[Value::I32(x)]));
        assert_eq!(run(&opt, 0, &[Value::I32(x)]), Ok(vec![Value::I32(2 * x)]));
        // sel(1,x)=x, twice
    }
}

#[test]
fn a_parameter_with_differing_constants_is_not_specialized() {
    // Two sites pass flag=1 and flag=0 — no single constant, so `sel` must keep its branch and behave
    // correctly for both.
    let entry = Func {
        params: vec![ValType::I32],
        results: vec![ValType::I32],
        blocks: vec![Block {
            params: vec![ValType::I32], // x
            insts: vec![
                Inst::ConstI32(1),
                Inst::Call {
                    func: 1,
                    args: vec![1, 0],
                }, // sel(1, x) = x
                Inst::ConstI32(0),
                Inst::Call {
                    func: 1,
                    args: vec![3, 0],
                }, // sel(0, x) = -x
                add(2, 4),
            ],
            term: Terminator::Return(vec![5]),
        }],
    };
    let m = Module {
        funcs: vec![entry, sel_helper()],
        ..Default::default()
    };
    verify_module(&m).expect("input verifies");
    let opt = optimize_module_with(&m, &cp_only());
    verify_module(&opt).expect("re-verifies");
    assert_eq!(
        n_brif(&opt),
        1,
        "a parameter that isn't a single constant across all sites must not be specialized"
    );
    for x in [0i32, 3, -5, 100] {
        assert_eq!(run(&m, 0, &[Value::I32(x)]), run(&opt, 0, &[Value::I32(x)]));
        assert_eq!(run(&opt, 0, &[Value::I32(x)]), Ok(vec![Value::I32(0)])); // x + (-x)
    }
}

#[test]
fn an_exported_function_is_not_specialized() {
    // `sel` is exported, so the host may call it with any `flag` — the single internal call passing
    // flag=1 must not license specializing it. Its branch stays.
    let entry = Func {
        params: vec![ValType::I32],
        results: vec![ValType::I32],
        blocks: vec![Block {
            params: vec![ValType::I32],
            insts: vec![
                Inst::ConstI32(1),
                Inst::Call {
                    func: 1,
                    args: vec![1, 0],
                },
            ],
            term: Terminator::Return(vec![2]),
        }],
    };
    let m = Module {
        funcs: vec![entry, sel_helper()],
        exports: vec![Export {
            name: "sel".into(),
            func: 1,
        }],
        ..Default::default()
    };
    verify_module(&m).expect("input verifies");
    // const_prop alone must leave the exported callee's branch in place.
    let cp = const_prop(&m);
    verify_module(&cp).expect("re-verifies");
    assert_eq!(
        n_brif(&cp),
        1,
        "an exported function must not be specialized to an internal call's constant"
    );
    // still exercised via the interpreter through the sel export
    let ph = cp.resolve_export("sel").unwrap();
    for f in [0i32, 1] {
        for v in [4i32, -9] {
            assert_eq!(
                run(&m, ph, &[Value::I32(f), Value::I32(v)]),
                run(&cp, ph, &[Value::I32(f), Value::I32(v)]),
            );
        }
    }
}

#[test]
fn const_prop_bails_on_an_unresolvable_indirect_index() {
    // A `call_indirect` on a **runtime** funcref (here the entry's own parameter `fp`) could reach any
    // function with arguments const_prop can't see, so the whole pass must bail — `sel`, whose one
    // direct caller passes flag=1, keeps its branch (an indirect call could pass flag=0).
    let sig = FuncType {
        params: vec![ValType::I32, ValType::I32],
        results: vec![ValType::I32],
    };
    let entry = Func {
        params: vec![ValType::I32, ValType::I32], // fp (runtime funcref), x
        results: vec![ValType::I32],
        blocks: vec![Block {
            params: vec![ValType::I32, ValType::I32],
            insts: vec![
                Inst::ConstI32(1),
                Inst::Call {
                    func: 1,
                    args: vec![2, 1],
                }, // sel(1, x) — a direct call passing a constant flag
                Inst::CallIndirect {
                    ty: sig,
                    idx: 0,
                    args: vec![2, 1],
                }, // (*fp)(1, x) — fp is unknown, so the pass can't see who this calls
                add(3, 4),
            ],
            term: Terminator::Return(vec![5]),
        }],
    };
    let m = Module {
        funcs: vec![entry, sel_helper()],
        ..Default::default()
    };
    verify_module(&m).expect("input verifies");
    let cp = const_prop(&m);
    verify_module(&cp).expect("re-verifies");
    assert_eq!(
        n_brif(&cp),
        1,
        "an unresolvable indirect index forces const_prop to leave every function alone"
    );
    assert_eq!(cp, m, "const_prop is the identity here");
}

#[test]
fn const_funcref_argument_devirtualizes_through_the_callee() {
    // entry(x): apply(g, x)  where `apply(fp, a) = call_indirect fp (a)` and `g` (func 2) is `a+1`.
    // The funcref is a constant `ConstI32(2)` (funcref == funcidx), so const_prop's fixpoint resolves
    // `apply`'s dispatch to `g`: it propagates the constant into `fp`, the re-run of devirt turns the
    // now-constant `call_indirect` into a direct call to `g`, which inlines — leaving the entry
    // computing a+1 with no indirect (and ultimately no) call.
    let sig = FuncType {
        params: vec![ValType::I32],
        results: vec![ValType::I32],
    };
    let entry = Func {
        params: vec![ValType::I32], // x
        results: vec![ValType::I32],
        blocks: vec![Block {
            params: vec![ValType::I32],
            insts: vec![
                Inst::ConstI32(2), // funcref g (== funcidx 2)
                Inst::Call {
                    func: 1,
                    args: vec![1, 0],
                }, // apply(g, x)
            ],
            term: Terminator::Return(vec![2]),
        }],
    };
    let apply = Func {
        params: vec![ValType::I32, ValType::I32], // fp=v0, a=v1
        results: vec![ValType::I32],
        blocks: vec![Block {
            params: vec![ValType::I32, ValType::I32],
            insts: vec![Inst::CallIndirect {
                ty: sig,
                idx: 0,
                args: vec![1],
            }], // (*fp)(a)
            term: Terminator::Return(vec![2]),
        }],
    };
    let g = Func {
        params: vec![ValType::I32],
        results: vec![ValType::I32],
        blocks: vec![Block {
            params: vec![ValType::I32],
            insts: vec![Inst::ConstI32(1), add(0, 1)], // a + 1
            term: Terminator::Return(vec![2]),
        }],
    };
    let m = Module {
        funcs: vec![entry, apply, g],
        ..Default::default()
    };
    verify_module(&m).expect("input verifies");
    assert_eq!(
        n_call_indirect(&m),
        1,
        "precondition: apply dispatches indirectly"
    );

    let opt = optimize_module(&m);
    verify_module(&opt).expect("re-verifies");
    assert_eq!(
        n_call_indirect(&opt),
        0,
        "const_prop resolves the funcref so devirt turns the indirect call direct"
    );
    assert_eq!(n_calls(&opt), 0, "the resolved call then inlines");
    for x in [0i32, 5, -3, 41] {
        assert_eq!(run(&m, 0, &[Value::I32(x)]), run(&opt, 0, &[Value::I32(x)]));
        assert_eq!(run(&opt, 0, &[Value::I32(x)]), Ok(vec![Value::I32(x + 1)]));
    }
}

#[test]
fn a_non_uniform_funcref_argument_is_not_devirtualized() {
    // `apply` is called with two *different* constant funcrefs (g at one site, h at another), so its
    // `fp` is not a single constant — the dispatch must stay indirect, and behavior must be preserved.
    let sig = FuncType {
        params: vec![ValType::I32],
        results: vec![ValType::I32],
    };
    let apply = Func {
        params: vec![ValType::I32, ValType::I32], // fp, a
        results: vec![ValType::I32],
        blocks: vec![Block {
            params: vec![ValType::I32, ValType::I32],
            insts: vec![Inst::CallIndirect {
                ty: sig.clone(),
                idx: 0,
                args: vec![1],
            }],
            term: Terminator::Return(vec![2]),
        }],
    };
    let plus1 = Func {
        params: vec![ValType::I32],
        results: vec![ValType::I32],
        blocks: vec![Block {
            params: vec![ValType::I32],
            insts: vec![Inst::ConstI32(1), add(0, 1)],
            term: Terminator::Return(vec![2]),
        }],
    };
    let times2 = Func {
        params: vec![ValType::I32],
        results: vec![ValType::I32],
        blocks: vec![Block {
            params: vec![ValType::I32],
            insts: vec![add(0, 0)],
            term: Terminator::Return(vec![1]),
        }],
    };
    // entry(sel, x): if sel { apply(plus1, x) } else { apply(times2, x) }
    let entry = Func {
        params: vec![ValType::I32, ValType::I32], // sel, x
        results: vec![ValType::I32],
        blocks: vec![
            Block {
                params: vec![ValType::I32, ValType::I32],
                insts: vec![],
                term: Terminator::BrIf {
                    cond: 0,
                    then_blk: 1,
                    then_args: vec![1],
                    else_blk: 2,
                    else_args: vec![1],
                },
            },
            Block {
                params: vec![ValType::I32], // x
                insts: vec![
                    Inst::ConstI32(2), // funcref plus1 (funcidx 2)
                    Inst::Call {
                        func: 1,
                        args: vec![1, 0],
                    },
                ],
                term: Terminator::Return(vec![2]),
            },
            Block {
                params: vec![ValType::I32], // x
                insts: vec![
                    Inst::ConstI32(3), // funcref times2 (funcidx 3)
                    Inst::Call {
                        func: 1,
                        args: vec![1, 0],
                    },
                ],
                term: Terminator::Return(vec![2]),
            },
        ],
    };
    let m = Module {
        funcs: vec![entry, apply, plus1, times2],
        ..Default::default()
    };
    verify_module(&m).expect("input verifies");
    let opt = optimize_module(&m);
    verify_module(&opt).expect("re-verifies");
    assert!(
        n_call_indirect(&opt) >= 1,
        "a callee dispatched on two different funcrefs must keep its indirect call"
    );
    for sel in [0i32, 1] {
        for x in [0i32, 5, -4] {
            assert_eq!(
                run(&m, 0, &[Value::I32(sel), Value::I32(x)]),
                run(&opt, 0, &[Value::I32(sel), Value::I32(x)]),
            );
        }
    }
}

#[test]
fn an_indirect_call_with_a_runtime_arg_blocks_specialization() {
    // Soundness guard for the fixpoint: `g` is called *directly* with flag=1 and *indirectly* (through a
    // constant funcref) with a runtime flag. If const_prop counted only the direct call it would wrongly
    // fix flag=1; it must join in the indirect call's runtime argument, leaving `g`'s branch intact.
    // g(flag) = flag ? 10 : 20   (func 1)
    let g = Func {
        params: vec![ValType::I32],
        results: vec![ValType::I32],
        blocks: vec![
            Block {
                params: vec![ValType::I32],
                insts: vec![Inst::ConstI32(0), cmp(CmpOp::Ne, 0, 1)], // flag != 0
                term: Terminator::BrIf {
                    cond: 2,
                    then_blk: 1,
                    then_args: vec![],
                    else_blk: 2,
                    else_args: vec![],
                },
            },
            Block {
                params: vec![],
                insts: vec![Inst::ConstI32(10)],
                term: Terminator::Return(vec![0]),
            },
            Block {
                params: vec![],
                insts: vec![Inst::ConstI32(20)],
                term: Terminator::Return(vec![0]),
            },
        ],
    };
    let sig = FuncType {
        params: vec![ValType::I32],
        results: vec![ValType::I32],
    };
    // entry(r): t1 = g(1); t2 = call_indirect(funcref g, r); return t1 + t2
    let entry = Func {
        params: vec![ValType::I32], // r (runtime)
        results: vec![ValType::I32],
        blocks: vec![Block {
            params: vec![ValType::I32],
            insts: vec![
                Inst::ConstI32(1),
                Inst::Call {
                    func: 1,
                    args: vec![1],
                }, // g(1) -> v2
                Inst::ConstI32(1), // funcref g (funcidx 1)
                Inst::CallIndirect {
                    ty: sig,
                    idx: 3,
                    args: vec![0],
                }, // g(r) -> v4
                add(2, 4),
            ],
            term: Terminator::Return(vec![5]),
        }],
    };
    let m = Module {
        funcs: vec![entry, g],
        ..Default::default()
    };
    verify_module(&m).expect("input verifies");
    let opt = optimize_module(&m);
    verify_module(&opt).expect("re-verifies");
    for r in [0i32, 1, 2, -5] {
        let want = 10 + if r != 0 { 10 } else { 20 };
        assert_eq!(run(&m, 0, &[Value::I32(r)]), run(&opt, 0, &[Value::I32(r)]));
        assert_eq!(
            run(&opt, 0, &[Value::I32(r)]),
            Ok(vec![Value::I32(want)]),
            "g must stay branch-sensitive at r={r}"
        );
    }
}

#[test]
fn a_block_parameter_is_not_read_as_a_function_parameter() {
    // Regression: the fixpoint must not confuse a *block* parameter (a phi in a non-entry block) with a
    // *function* parameter. `h(p)` is always called with p=1, so h.p is constant — but in a non-entry
    // block h calls `k(q)` where `q` is a runtime phi (5 or 7). If the analysis read q as h's param it
    // would fix k's argument to 1 and mis-specialize k. k must stay branch-sensitive.
    let entry = Func {
        params: vec![],
        results: vec![ValType::I32],
        blocks: vec![Block {
            params: vec![],
            insts: vec![
                Inst::ConstI32(1),
                Inst::Call {
                    func: 1,
                    args: vec![0],
                }, // h(1)
            ],
            term: Terminator::Return(vec![1]),
        }],
    };
    let h = Func {
        params: vec![ValType::I32], // p
        results: vec![ValType::I32],
        blocks: vec![
            Block {
                params: vec![ValType::I32], // p = v0
                insts: vec![Inst::ConstI32(5), Inst::ConstI32(7)],
                term: Terminator::BrIf {
                    cond: 0,
                    then_blk: 1,
                    then_args: vec![1], // q = 5
                    else_blk: 1,
                    else_args: vec![2], // q = 7
                },
            },
            Block {
                params: vec![ValType::I32], // q = v0 (a phi: 5 or 7)
                insts: vec![Inst::Call {
                    func: 2,
                    args: vec![0],
                }], // k(q)
                term: Terminator::Return(vec![1]),
            },
        ],
    };
    let k = Func {
        params: vec![ValType::I32], // v
        results: vec![ValType::I32],
        blocks: vec![
            Block {
                params: vec![ValType::I32],
                insts: vec![Inst::ConstI32(5), cmp(CmpOp::Eq, 0, 1)], // v == 5
                term: Terminator::BrIf {
                    cond: 2,
                    then_blk: 1,
                    then_args: vec![],
                    else_blk: 2,
                    else_args: vec![],
                },
            },
            Block {
                params: vec![],
                insts: vec![Inst::ConstI32(100)],
                term: Terminator::Return(vec![0]),
            },
            Block {
                params: vec![],
                insts: vec![Inst::ConstI32(200)],
                term: Terminator::Return(vec![0]),
            },
        ],
    };
    let m = Module {
        funcs: vec![entry, h, k],
        ..Default::default()
    };
    verify_module(&m).expect("input verifies");
    // h(1): p=1 -> q=5 -> k(5) -> 100.
    assert_eq!(run(&m, 0, &[]), Ok(vec![Value::I32(100)]));
    let opt = optimize_module(&m);
    verify_module(&opt).expect("re-verifies");
    assert_eq!(
        run(&opt, 0, &[]),
        Ok(vec![Value::I32(100)]),
        "k must not be mis-specialized from a block parameter read as a function parameter"
    );
}
