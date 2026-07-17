//! Spec for global value numbering / CSE with cross-block threading (OPT.md Phase 2). These are the
//! redundancies `merge_blocks` + `local_cse` cannot reach — recomputations at **multi-predecessor
//! joins**, where GVN's value-number congruence proves the join's fresh parameters equal the original
//! operands. Asserts (1) `optimize_module` preserves behavior on the reference interpreter, (2) the
//! threaded output re-verifies, and (3) GVN actually fired (the redundant computation is gone).

use svm_interp::{Trap, Value};
use svm_ir::{BinOp, Block, Func, Inst, IntTy, LoadOp, Memory, Module, Terminator, ValType};
use svm_opt::{optimize_module, optimize_module_with, OptConfig};
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

fn check_equiv(orig: &Module, argsets: &[Vec<Value>]) -> Module {
    verify_module(orig).expect("original verifies");
    let opt = optimize_module(orig);
    verify_module(&opt).expect("optimized (threaded) re-verifies");
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
        vec![Value::I32(1), Value::I32(0)], // cond true
        vec![Value::I32(0), Value::I32(9)], // cond false
        vec![Value::I32(7), Value::I32(-3)],
        vec![Value::I32(i32::MAX), Value::I32(1)], // wrapping
    ]
}

#[test]
fn diamond_join_redundancy_eliminated() {
    // b0(a,b): v2 = a+b ; if a { b1(a,b) } else { b2(a,b) }.  b1/b2 forward to b3.
    // b3(x,y): w = x+y  — congruent to a+b, but x,y are fresh params so only GVN can prove it.
    // b3 has two predecessors, so merge_blocks cannot fuse it.
    let f = Func {
        params: vec![ValType::I32, ValType::I32],
        results: vec![ValType::I32],
        blocks: vec![
            Block {
                params: vec![ValType::I32, ValType::I32], // a, b
                insts: vec![add(0, 1)],                   // v2 = a + b
                term: Terminator::BrIf {
                    cond: 0,
                    then_blk: 1,
                    then_args: vec![0, 1],
                    else_blk: 2,
                    else_args: vec![0, 1],
                },
            },
            Block {
                params: vec![ValType::I32, ValType::I32],
                insts: vec![],
                term: Terminator::Br {
                    target: 3,
                    args: vec![0, 1],
                },
            },
            Block {
                params: vec![ValType::I32, ValType::I32],
                insts: vec![],
                term: Terminator::Br {
                    target: 3,
                    args: vec![0, 1],
                },
            },
            Block {
                params: vec![ValType::I32, ValType::I32], // x, y
                insts: vec![add(0, 1)],                   // w = x + y  (== a + b)
                term: Terminator::Return(vec![2]),
            },
        ],
    };
    let m = module(f);
    assert_eq!(count(&m, |i| matches!(i, Inst::IntBin { .. })), 2);
    let opt = check_equiv(&m, &argsets());
    // GVN proved w == a+b and threaded the b0 value to b3; the recomputation is gone.
    assert_eq!(
        count(&opt, |i| matches!(i, Inst::IntBin { .. })),
        1,
        "the join recomputation should be eliminated by GVN"
    );
}

#[test]
fn join_recomputation_of_a_derived_expression() {
    // A two-level expression recomputed at a join: b0 computes (a+b) then (a+b)+a; the join recomputes
    // the same, threaded across the diamond. Behavior must be identical and the duplicates gone.
    let f = Func {
        params: vec![ValType::I32, ValType::I32],
        results: vec![ValType::I32],
        blocks: vec![
            Block {
                params: vec![ValType::I32, ValType::I32], // a, b
                insts: vec![add(0, 1), add(2, 0)],        // v2=a+b, v3=(a+b)+a
                term: Terminator::BrIf {
                    cond: 1,
                    then_blk: 1,
                    then_args: vec![0, 1],
                    else_blk: 2,
                    else_args: vec![0, 1],
                },
            },
            Block {
                params: vec![ValType::I32, ValType::I32],
                insts: vec![],
                term: Terminator::Br {
                    target: 3,
                    args: vec![0, 1],
                },
            },
            Block {
                params: vec![ValType::I32, ValType::I32],
                insts: vec![],
                term: Terminator::Br {
                    target: 3,
                    args: vec![0, 1],
                },
            },
            Block {
                params: vec![ValType::I32, ValType::I32], // x, y
                insts: vec![add(0, 1), add(2, 0)],        // x+y, (x+y)+x  (== a+b, (a+b)+a)
                term: Terminator::Return(vec![3]),
            },
        ],
    };
    let m = module(f);
    assert_eq!(count(&m, |i| matches!(i, Inst::IntBin { .. })), 4);
    let opt = check_equiv(&m, &argsets());
    // Both recomputations in the join collapse to the b0 values.
    assert_eq!(
        count(&opt, |i| matches!(i, Inst::IntBin { .. })),
        2,
        "both join recomputations should be eliminated"
    );
}

#[test]
fn impure_loads_at_a_join_are_not_deduped_by_gvn() {
    // GVN's safety line: a load has a unique value number (it may trap / read changing memory), so GVN
    // never treats a join recomputation of a load as redundant, even when the address is congruent.
    // (The *cross-block load* pass — OPT.md Phase 4 — does soundly remove such a reload; that this
    // diamond is one is asserted at the end via the full pipeline.)
    let ld = |addr: u32| Inst::Load {
        op: LoadOp::I32,
        addr,
        offset: 0,
        align: 2,
    };
    let f = Func {
        params: vec![ValType::I32, ValType::I64], // cond, addr
        results: vec![ValType::I32],
        blocks: vec![
            Block {
                params: vec![ValType::I32, ValType::I64],
                insts: vec![ld(1)], // v2 = load[addr]
                term: Terminator::BrIf {
                    cond: 0,
                    then_blk: 1,
                    then_args: vec![1],
                    else_blk: 2,
                    else_args: vec![1],
                },
            },
            Block {
                params: vec![ValType::I64],
                insts: vec![],
                term: Terminator::Br {
                    target: 3,
                    args: vec![0],
                },
            },
            Block {
                params: vec![ValType::I64],
                insts: vec![],
                term: Terminator::Br {
                    target: 3,
                    args: vec![0],
                },
            },
            Block {
                params: vec![ValType::I64], // addr'
                insts: vec![ld(0)],         // load[addr'] — must NOT be deduped
                term: Terminator::Return(vec![1]),
            },
        ],
    };
    let m = module(f);
    let args = [
        vec![Value::I32(1), Value::I64(0)],
        vec![Value::I32(0), Value::I64(0)],
    ];
    // GVN alone (cross-block load elimination off) leaves both loads — its safety line.
    let gvn_only = optimize_module_with(
        &m,
        &OptConfig {
            gvn: true,
            load_elim: false,
            ..OptConfig::none()
        },
    );
    verify_module(&gvn_only).expect("gvn-only re-verifies");
    assert_eq!(
        count(&gvn_only, |i| matches!(i, Inst::Load { .. })),
        2,
        "GVN itself must not dedup impure loads"
    );
    // The full pipeline's cross-block load pass soundly removes the dominated join reload.
    let opt = check_equiv(&m, &args);
    assert_eq!(
        count(&opt, |i| matches!(i, Inst::Load { .. })),
        1,
        "load_elim removes the join reload the diamond makes redundant"
    );
}

// ---- randomized differential: branchy DAGs stress GVN congruence + threading ----

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn upto(&mut self, n: u32) -> u32 {
        (self.next() % n as u64) as u32
    }
}

/// A random **forward-only** CFG (a DAG — block `i` only branches to blocks `> i`, the last returns),
/// so it always terminates and there is no fuel divergence to confuse the differential. Every block
/// has `P` `i32` parameters and passes `P` `i32` args on each edge, so types trivially check.
/// Diamonds (two blocks targeting the same later block) arise naturally, exercising congruence and
/// the threading lowering. Instructions are pure, non-trapping `i32` arithmetic.
fn random_dag(rng: &mut Lcg, nparams: u32, nblocks: u32) -> Func {
    let p = nparams.max(1);
    let ops = [
        BinOp::Add,
        BinOp::Sub,
        BinOp::Mul,
        BinOp::And,
        BinOp::Or,
        BinOp::Xor,
        BinOp::Shl,
    ];
    let mut blocks = Vec::new();
    for b in 0..nblocks {
        let params = vec![ValType::I32; p as usize];
        let mut slots = p;
        let mut insts = Vec::new();
        for _ in 0..rng.upto(4) {
            let a = rng.upto(slots);
            let b2 = rng.upto(slots);
            let op = ops[rng.upto(ops.len() as u32) as usize];
            insts.push(Inst::IntBin {
                ty: IntTy::I32,
                op,
                a,
                b: b2,
            });
            slots += 1;
        }
        let arg = |rng: &mut Lcg| rng.upto(slots);
        let mkargs = |rng: &mut Lcg| (0..p).map(|_| arg(rng)).collect::<Vec<_>>();
        let term = if b + 1 >= nblocks {
            Terminator::Return(vec![arg(rng)])
        } else {
            let remaining = nblocks - (b + 1);
            match rng.upto(3) {
                0 => Terminator::Br {
                    target: b + 1 + rng.upto(remaining),
                    args: mkargs(rng),
                },
                _ => {
                    let t = b + 1 + rng.upto(remaining);
                    let e = b + 1 + rng.upto(remaining);
                    Terminator::BrIf {
                        cond: arg(rng),
                        then_blk: t,
                        then_args: mkargs(rng),
                        else_blk: e,
                        else_args: mkargs(rng),
                    }
                }
            }
        };
        blocks.push(Block {
            params,
            insts,
            term,
        });
    }
    Func {
        params: vec![ValType::I32; p as usize],
        results: vec![ValType::I32],
        blocks,
    }
}

#[test]
fn randomized_branchy_dags_preserve_behavior() {
    let mut rng = Lcg(0xc0ff_ee12_3456_789a);
    let mut fired = 0usize;
    for _ in 0..600 {
        let nparams = 1 + rng.upto(3);
        let nblocks = 2 + rng.upto(6);
        let f = random_dag(&mut rng, nparams, nblocks);
        let m = module(f);
        if verify_module(&m).is_err() {
            continue; // skip any that don't verify (should be rare/none by construction)
        }
        let before = count(&m, |i| matches!(i, Inst::IntBin { .. }));
        let args: Vec<Value> = (0..nparams)
            .map(|k| Value::I32((rng.next() as i32) ^ k as i32))
            .collect();
        let opt = optimize_module(&m);
        verify_module(&opt).expect("optimized (threaded) module must re-verify");
        assert_eq!(
            run(&m, &args),
            run(&opt, &args),
            "GVN/optimize changed behavior on a random DAG"
        );
        if count(&opt, |i| matches!(i, Inst::IntBin { .. })) < before {
            fired += 1;
        }
    }
    // Sanity: across 600 random branchy modules, the optimizer eliminated arithmetic in a good share
    // (confirms the pipeline — including GVN — is actually doing work, not vacuously passing).
    assert!(
        fired > 50,
        "expected the optimizer to fire on many DAGs, only {fired}"
    );
}
