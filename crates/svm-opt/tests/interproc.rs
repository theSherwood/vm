//! Spec for the interprocedural passes (OPT.md Phase 3). Slice 1: dead-function elimination must
//! (1) drop functions no reachable code can call, (2) renumber survivors and remap every static
//! funcidx reference + export target, (3) keep the reachable functions behaving identically on the
//! reference interpreter, and (4) conservatively leave a module untouched while indirect funcref
//! dispatch (`call_indirect`) is present, since a funcref equals its funcidx (identity table).

use svm_interp::{Trap, Value};
use svm_ir::{
    BinOp, Block, CmpOp, Export, Func, FuncType, Inst, IntTy, Module, Terminator, ValType,
};
use svm_opt::interproc::{dead_func_elim, devirtualize, inline_calls};
use svm_opt::optimize_module;
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
