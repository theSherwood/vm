//! Round-trip spec for the internal conventional-SSA form (OPT.md Phase 1c). The load-bearing
//! invariant: `from_ssa(to_ssa(f)) == f` exactly (a pure renaming and its inverse). Covered on
//! hand-built shapes (params, `br_table`, loops, a multi-result `call`), through the reference
//! interpreter, and over a randomized structural generator.

use svm_interp::{Trap, Value as IValue};
use svm_ir::{
    BinOp, Block, Func, Inst, IntTy, LoadOp, Memory, Module, StoreOp, Terminator, ValType,
};
use svm_opt::ssa::{from_ssa, to_ssa, Def};

fn fn_results(m: &Module) -> Vec<usize> {
    m.funcs.iter().map(|f| f.results.len()).collect()
}

/// Assert the round-trip is the identity for every function of a module.
fn assert_roundtrip(m: &Module) {
    let fr = fn_results(m);
    for f in &m.funcs {
        let back = from_ssa(&to_ssa(f, &fr));
        assert_eq!(&back, f, "round-trip must be the identity");
    }
}

fn one_func_module(f: Func) -> Module {
    Module {
        funcs: vec![f],
        memory: Some(Memory { size_log2: 16 }),
        data: vec![],
        imports: vec![],
        exports: vec![],
        debug_info: None,
    }
}

#[test]
fn straight_line_identity() {
    // (i32 a) -> i32 : return a + (a + 7)
    let f = Func {
        params: vec![ValType::I32],
        results: vec![ValType::I32],
        blocks: vec![Block {
            params: vec![ValType::I32], // v0 = a
            insts: vec![
                Inst::ConstI32(7), // v1
                Inst::IntBin {
                    ty: IntTy::I32,
                    op: BinOp::Add,
                    a: 0,
                    b: 1,
                }, // v2 = a + 7
                Inst::IntBin {
                    ty: IntTy::I32,
                    op: BinOp::Add,
                    a: 0,
                    b: 2,
                }, // v3 = a + v2
            ],
            term: Terminator::Return(vec![3]),
        }],
    };
    assert_roundtrip(&one_func_module(f));
}

#[test]
fn block_params_and_brif_identity() {
    // Two blocks, cross-block dataflow via a block parameter (the phi).
    let f = Func {
        params: vec![ValType::I32],
        results: vec![ValType::I32],
        blocks: vec![
            Block {
                params: vec![ValType::I32], // v0 = a
                insts: vec![Inst::ConstI32(0)], // v1
                term: Terminator::BrIf {
                    cond: 0,
                    then_blk: 1,
                    then_args: vec![0], // pass a
                    else_blk: 1,
                    else_args: vec![1], // pass 0
                },
            },
            Block {
                params: vec![ValType::I32], // v0 = the phi
                insts: vec![],
                term: Terminator::Return(vec![0]),
            },
        ],
    };
    assert_roundtrip(&one_func_module(f));
}

#[test]
fn br_table_and_loop_identity() {
    // A loop with a back edge carrying an accumulator through a block parameter, plus a br_table.
    let f = Func {
        params: vec![ValType::I32],
        results: vec![ValType::I32],
        blocks: vec![
            Block {
                params: vec![ValType::I32], // v0 = n
                insts: vec![Inst::ConstI32(0)], // v1 = acc0
                term: Terminator::Br {
                    target: 1,
                    args: vec![0, 1],
                },
            },
            Block {
                params: vec![ValType::I32, ValType::I32], // v0 = i, v1 = acc
                insts: vec![
                    Inst::IntBin {
                        ty: IntTy::I32,
                        op: BinOp::Add,
                        a: 1,
                        b: 0,
                    }, // v2 = acc + i
                    Inst::ConstI32(1), // v3
                    Inst::IntBin {
                        ty: IntTy::I32,
                        op: BinOp::Sub,
                        a: 0,
                        b: 3,
                    }, // v4 = i - 1
                ],
                term: Terminator::BrTable {
                    idx: 0,
                    targets: vec![(2, vec![2])],       // idx 0 -> exit with acc
                    default: (1, vec![4, 2]),          // else loop with (i-1, acc)
                },
            },
            Block {
                params: vec![ValType::I32], // v0 = result
                insts: vec![],
                term: Terminator::Return(vec![0]),
            },
        ],
    };
    assert_roundtrip(&one_func_module(f));
}

#[test]
fn multi_result_call_identity_and_defs() {
    // func0 calls func1 (which returns 0 values here → a Call that appends nothing), then a second
    // func returning one value, to exercise result-arity-driven global id assignment.
    let callee = Func {
        params: vec![ValType::I32],
        results: vec![ValType::I32],
        blocks: vec![Block {
            params: vec![ValType::I32],
            insts: vec![],
            term: Terminator::Return(vec![0]),
        }],
    };
    let caller = Func {
        params: vec![ValType::I32],
        results: vec![ValType::I32],
        blocks: vec![Block {
            params: vec![ValType::I32], // v0 = a
            insts: vec![
                Inst::Call {
                    func: 1,
                    args: vec![0],
                }, // v1 = callee(a)
                Inst::IntBin {
                    ty: IntTy::I32,
                    op: BinOp::Add,
                    a: 0,
                    b: 1,
                }, // v2 = a + v1
            ],
            term: Terminator::Return(vec![2]),
        }],
    };
    let m = Module {
        funcs: vec![caller, callee],
        memory: Some(Memory { size_log2: 16 }),
        data: vec![],
        imports: vec![],
        exports: vec![],
        debug_info: None,
    };
    assert_roundtrip(&m);

    // The Call's single result is a global value defined at (block 0, inst 0, result 0).
    let ssa = to_ssa(&m.funcs[0], &fn_results(&m));
    let call_result = ssa.values[0][1]; // slot 1 = first inst result (slot 0 is the param)
    assert_eq!(
        ssa.defs[call_result as usize],
        Def::Result {
            block: 0,
            inst: 0,
            result: 0
        }
    );
    assert_eq!(
        ssa.defs[ssa.values[0][0] as usize],
        Def::Param { block: 0, param: 0 }
    );
}

fn run(m: &Module, args: &[IValue]) -> Result<Vec<IValue>, Trap> {
    let mut fuel = 1_000_000u64;
    svm_interp::run(m, 0, args, &mut fuel)
}

#[test]
fn roundtrip_preserves_behavior_through_interp() {
    // A memory-touching function so the behavioral check exercises more than arithmetic: store a,
    // load it back, add. Round-tripping must not change the observable result.
    let f = Func {
        params: vec![ValType::I32],
        results: vec![ValType::I32],
        blocks: vec![Block {
            params: vec![ValType::I32], // v0 = a
            insts: vec![
                Inst::ConstI32(0), // v1 = addr
                Inst::Store {
                    op: StoreOp::I32,
                    addr: 1,
                    value: 0,
                    offset: 0,
                    align: 2,
                },
                Inst::Load {
                    op: LoadOp::I32,
                    addr: 1,
                    offset: 0,
                    align: 2,
                }, // v2 = *addr
                Inst::IntBin {
                    ty: IntTy::I32,
                    op: BinOp::Add,
                    a: 0,
                    b: 2,
                }, // v3 = a + a
            ],
            term: Terminator::Return(vec![3]),
        }],
    };
    let m = one_func_module(f);
    let fr = fn_results(&m);
    let back = from_ssa(&to_ssa(&m.funcs[0], &fr));
    let m2 = one_func_module(back);
    for a in [0i32, 1, -5, 1000, i32::MIN] {
        assert_eq!(run(&m, &[IValue::I32(a)]), run(&m2, &[IValue::I32(a)]));
    }
}

// ---- randomized structural round-trip ----

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        // SplitMix64-ish: deterministic, no external deps.
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn upto(&mut self, n: u32) -> u32 {
        if n == 0 {
            0
        } else {
            (self.next() % n as u64) as u32
        }
    }
}

/// Build a structurally-valid block-local function: operands reference only earlier local slots,
/// terminator targets are in range, and edge args are valid local indices. Types are irrelevant to
/// the round-trip (a pure renaming), so this deliberately does not type-check — it stresses the
/// index bookkeeping across many block/param/terminator shapes.
fn random_func(rng: &mut Lcg) -> Func {
    let nblocks = 1 + rng.upto(5); // 1..=5 blocks
    let mut blocks = Vec::new();
    for _ in 0..nblocks {
        let nparams = rng.upto(3); // 0..=2 params
        let params = vec![ValType::I32; nparams as usize];
        // Always seed one constant so every block has ≥1 value slot — a real verified module never
        // names a value that doesn't exist, so operands/args below can always reference slot 0.
        let mut insts = vec![Inst::ConstI32(rng.next() as i32)];
        let mut slots = nparams + 1; // live local-index count
        let ninsts = rng.upto(6);
        for _ in 0..ninsts {
            let inst = if slots == 0 || rng.upto(2) == 0 {
                Inst::ConstI32(rng.next() as i32)
            } else {
                let a = rng.upto(slots);
                let b = rng.upto(slots);
                Inst::IntBin {
                    ty: IntTy::I32,
                    op: BinOp::Add,
                    a,
                    b,
                }
            };
            insts.push(inst);
            slots += 1; // both ops above append exactly one result
        }
        // A valid local index for edge args (fall back to 0 when the block is empty of slots).
        let mut arg = |rng: &mut Lcg| if slots == 0 { 0 } else { rng.upto(slots) };
        let term = match rng.upto(4) {
            0 => Terminator::Return(if slots == 0 { vec![] } else { vec![arg(rng)] }),
            1 => Terminator::Br {
                target: rng.upto(nblocks),
                args: vec![arg(rng), arg(rng)],
            },
            2 => Terminator::BrIf {
                cond: arg(rng),
                then_blk: rng.upto(nblocks),
                then_args: vec![arg(rng)],
                else_blk: rng.upto(nblocks),
                else_args: vec![arg(rng)],
            },
            _ => Terminator::BrTable {
                idx: arg(rng),
                targets: vec![
                    (rng.upto(nblocks), vec![arg(rng)]),
                    (rng.upto(nblocks), vec![]),
                ],
                default: (rng.upto(nblocks), vec![arg(rng)]),
            },
        };
        blocks.push(Block {
            params,
            insts,
            term,
        });
    }
    Func {
        params: vec![],
        results: vec![],
        blocks,
    }
}

#[test]
fn randomized_structural_roundtrip_is_identity() {
    let mut rng = Lcg(0x1234_5678_9abc_def0);
    for _ in 0..5000 {
        let f = random_func(&mut rng);
        let back = from_ssa(&to_ssa(&f, &[]));
        assert_eq!(back, f, "randomized round-trip diverged");
    }
}
