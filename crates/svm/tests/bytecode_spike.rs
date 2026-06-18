//! Phase-1 ROI spike for the bytecode-dispatch rewrite (see INTERP_PERF.md).
//!
//! A *self-contained, throwaway* measurement: it compiles the `alu` recurrence kernel into a flat,
//! operand-resolved bytecode over a function-wide register file (each SSA value gets a global slot;
//! branches copy edge args into the target block's param slots; a single instruction pointer drives
//! a tight dispatch loop), then checks it agrees with the reference interpreter and times both.
//!
//! The question it answers: under our constraints (`#![forbid(unsafe_code)]` → bounds checks stay;
//! stable Rust → no computed-goto), how much faster than the IR-walking interpreter is a flat
//! resolved bytecode? If the win here is large, the full (seam-preserving) rewrite is worth it; if
//! it's marginal, we've learned that cheaply. The executor only covers the handful of ops the ALU
//! kernel uses (no calls/memory/fibers) — enough to measure the dispatch+layout ceiling.
//!
//! Run: cargo test -p svm --release --test bytecode_spike -- --nocapture --ignored

use std::hint::black_box;
use std::time::Instant;

use svm_interp::Value;
use svm_ir::{BinOp, CmpOp, Func, Inst, IntTy, Module, Terminator};

const ALU: &str = r#"
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const 0
  v2 = i64.const 0
  br block1(v0, v1, v2)
block1(v3: i64, v4: i64, v5: i64):
  v6 = i64.lt_s v5 v3
  br_if v6 block2(v3, v4, v5) block3(v4)
block2(v7: i64, v8: i64, v9: i64):
  v10 = i64.const 6364136223846793005
  v11 = i64.mul v8 v10
  v12 = i64.const 1442695040888963407
  v13 = i64.add v11 v12
  v14 = i64.add v13 v9
  v15 = i64.const 1
  v16 = i64.add v9 v15
  br block1(v7, v14, v16)
block3(v17: i64):
  return v17
}
"#;

// ---- Flat resolved bytecode (ALU subset) ------------------------------------------------------

#[derive(Debug)]
enum Op {
    Const {
        dst: u32,
        c: i64,
    },
    Add {
        dst: u32,
        a: u32,
        b: u32,
    },
    Mul {
        dst: u32,
        a: u32,
        b: u32,
    },
    LtS {
        dst: u32,
        a: u32,
        b: u32,
    },
    // Branch: parallel-copy edge args into the target block's param slots, then jump.
    Br {
        copies: Vec<(u32, u32)>,
        target: u32,
    },
    BrIf {
        cond: u32,
        then_copies: Vec<(u32, u32)>,
        then_pc: u32,
        else_copies: Vec<(u32, u32)>,
        else_pc: u32,
    },
    Ret {
        src: u32,
    },
    // Direct call: gather arg slots, run `callee`, write its single result to `dst`.
    Call {
        callee: u32,
        args: Vec<u32>,
        dst: u32,
    },
}

struct Program {
    ops: Vec<Op>,
    nslots: u32,
}

/// Lower an all-i64 (straight-line + branches + direct calls) function to the flat bytecode.
/// `arities[f]` is each function's result count (for `Call` slot sizing). Panics on any op outside
/// the supported subset — this is a measurement spike, not the real compiler.
fn compile(f: &Func, arities: &[usize]) -> Program {
    // Global slot assignment: each block's params then its value-producing insts, in order.
    let mut base = Vec::with_capacity(f.blocks.len());
    let mut nslots = 0u32;
    for b in &f.blocks {
        base.push(nslots);
        nslots += b.params.len() as u32;
        for inst in &b.insts {
            nslots += inst.result_count(arities) as u32;
        }
    }
    // First op index of each block (its entry pc), for branch targets.
    let mut block_pc = vec![0u32; f.blocks.len()];
    let mut ops: Vec<Op> = Vec::new();
    for (bi, b) in f.blocks.iter().enumerate() {
        block_pc[bi] = ops.len() as u32;
        let mut local = b.params.len() as u32; // next result's local index
        let g = |local_idx: u32| base[bi] + local_idx; // operand: block-local -> global slot
        for inst in &b.insts {
            let dst = base[bi] + local;
            local += inst.result_count(arities) as u32;
            match inst {
                Inst::ConstI64(c) => ops.push(Op::Const { dst, c: *c }),
                Inst::Call { func, args } => ops.push(Op::Call {
                    callee: *func,
                    args: args.iter().map(|a| g(*a)).collect(),
                    dst,
                }),
                Inst::IntBin {
                    ty: IntTy::I64,
                    op,
                    a,
                    b,
                } => {
                    let (a, b) = (g(*a), g(*b));
                    ops.push(match op {
                        BinOp::Add => Op::Add { dst, a, b },
                        BinOp::Mul => Op::Mul { dst, a, b },
                        other => panic!("spike: unsupported binop {other:?}"),
                    });
                }
                Inst::IntCmp {
                    ty: IntTy::I64,
                    op: CmpOp::LtS,
                    a,
                    b,
                } => ops.push(Op::LtS {
                    dst,
                    a: g(*a),
                    b: g(*b),
                }),
                other => panic!("spike: unsupported inst {other:?}"),
            }
        }
        // Terminator → copies + jump. Edge args are block-local in the *source* block; targets'
        // params are the first `nparams` slots of the target block.
        let edge = |bidx: usize, args: &[u32]| -> (Vec<(u32, u32)>, u32) {
            let copies = args
                .iter()
                .enumerate()
                .map(|(i, a)| (g(*a), base[bidx] + i as u32))
                .collect();
            (copies, bidx as u32) // target block index; patched to pc below
        };
        match &b.term {
            Terminator::Br { target, args } => {
                let (copies, t) = edge(*target as usize, args);
                ops.push(Op::Br { copies, target: t });
            }
            Terminator::BrIf {
                cond,
                then_blk,
                then_args,
                else_blk,
                else_args,
            } => {
                let (then_copies, tt) = edge(*then_blk as usize, then_args);
                let (else_copies, et) = edge(*else_blk as usize, else_args);
                ops.push(Op::BrIf {
                    cond: g(*cond),
                    then_copies,
                    then_pc: tt,
                    else_copies,
                    else_pc: et,
                });
            }
            Terminator::Return(vs) => ops.push(Op::Ret { src: g(vs[0]) }),
            other => panic!("spike: unsupported terminator {other:?}"),
        }
    }
    // Patch branch targets from block index to entry pc.
    for op in &mut ops {
        match op {
            Op::Br { target, .. } => *target = block_pc[*target as usize],
            Op::BrIf {
                then_pc, else_pc, ..
            } => {
                *then_pc = block_pc[*then_pc as usize];
                *else_pc = block_pc[*else_pc as usize];
            }
            _ => {}
        }
    }
    Program { ops, nslots }
}

#[inline]
fn apply(regs: &mut [i64], copies: &[(u32, u32)]) {
    // Edge src/dst slot sets are disjoint across distinct blocks (the ALU kernel has no self-loop),
    // so a sequential copy is parallel-safe here.
    for &(s, d) in copies {
        regs[d as usize] = regs[s as usize];
    }
}

fn run_program(p: &Program, arg: i64, fuel: &mut u64) -> i64 {
    let mut regs = vec![0i64; p.nslots as usize];
    regs[0] = arg; // block0 param v0
    let mut pc = 0usize;
    loop {
        *fuel = fuel.checked_sub(1).expect("fuel"); // keep the mandatory per-op seam, for fairness
        match &p.ops[pc] {
            Op::Const { dst, c } => {
                regs[*dst as usize] = *c;
                pc += 1;
            }
            Op::Add { dst, a, b } => {
                regs[*dst as usize] = regs[*a as usize].wrapping_add(regs[*b as usize]);
                pc += 1;
            }
            Op::Mul { dst, a, b } => {
                regs[*dst as usize] = regs[*a as usize].wrapping_mul(regs[*b as usize]);
                pc += 1;
            }
            Op::LtS { dst, a, b } => {
                regs[*dst as usize] = (regs[*a as usize] < regs[*b as usize]) as i64;
                pc += 1;
            }
            Op::Br { copies, target } => {
                apply(&mut regs, copies);
                pc = *target as usize;
            }
            Op::BrIf {
                cond,
                then_copies,
                then_pc,
                else_copies,
                else_pc,
            } => {
                if regs[*cond as usize] != 0 {
                    apply(&mut regs, then_copies);
                    pc = *then_pc as usize;
                } else {
                    apply(&mut regs, else_copies);
                    pc = *else_pc as usize;
                }
            }
            Op::Ret { src } => return regs[*src as usize],
            Op::Call { .. } => unreachable!("single-program ALU executor has no calls"),
        }
    }
}

/// Register-window executor for multi-function modules: one big `regs` file, each call activation
/// occupying `[base, base + nslots)`; a `Call` opens the next window (copying args into the callee's
/// param slots) with no per-call allocation, and `Ret` writes the result back into the caller's
/// window and restores it. Measures the flat model's *call* path.
fn run_module(progs: &[Program], entry: usize, arg: i64, fuel: &mut u64) -> i64 {
    let mut regs = vec![0i64; progs[entry].nslots as usize];
    regs[0] = arg; // entry block0 param v0
                   // Current activation:
    let mut cur = entry;
    let mut base = 0usize;
    let mut pc = 0usize;
    // Resume stack of suspended callers: (prog, base, pc, absolute result slot in caller window).
    let mut stack: Vec<(usize, usize, usize, usize)> = Vec::new();
    loop {
        *fuel = fuel.checked_sub(1).expect("fuel");
        match &progs[cur].ops[pc] {
            Op::Const { dst, c } => {
                regs[base + *dst as usize] = *c;
                pc += 1;
            }
            Op::Add { dst, a, b } => {
                regs[base + *dst as usize] =
                    regs[base + *a as usize].wrapping_add(regs[base + *b as usize]);
                pc += 1;
            }
            Op::Mul { dst, a, b } => {
                regs[base + *dst as usize] =
                    regs[base + *a as usize].wrapping_mul(regs[base + *b as usize]);
                pc += 1;
            }
            Op::LtS { dst, a, b } => {
                regs[base + *dst as usize] =
                    (regs[base + *a as usize] < regs[base + *b as usize]) as i64;
                pc += 1;
            }
            Op::Br { copies, target } => {
                for &(s, d) in copies {
                    regs[base + d as usize] = regs[base + s as usize];
                }
                pc = *target as usize;
            }
            Op::BrIf {
                cond,
                then_copies,
                then_pc,
                else_copies,
                else_pc,
            } => {
                let (copies, next) = if regs[base + *cond as usize] != 0 {
                    (then_copies, *then_pc)
                } else {
                    (else_copies, *else_pc)
                };
                for &(s, d) in copies {
                    regs[base + d as usize] = regs[base + s as usize];
                }
                pc = next as usize;
            }
            Op::Call { callee, args, dst } => {
                let callee = *callee as usize;
                let new_base = base + progs[cur].nslots as usize;
                let need = new_base + progs[callee].nslots as usize;
                if regs.len() < need {
                    regs.resize(need, 0);
                }
                for (i, a) in args.iter().enumerate() {
                    regs[new_base + i] = regs[base + *a as usize];
                }
                stack.push((cur, base, pc + 1, base + *dst as usize));
                cur = callee;
                base = new_base;
                pc = 0;
            }
            Op::Ret { src } => {
                let result = regs[base + *src as usize];
                match stack.pop() {
                    None => return result,
                    Some((cprog, cbase, cpc, ret_abs)) => {
                        regs[ret_abs] = result;
                        cur = cprog;
                        base = cbase;
                        pc = cpc;
                    }
                }
            }
        }
    }
}

fn interp(m: &Module, n: i64) -> i64 {
    let mut fuel = u64::MAX;
    let v = svm_interp::run(m, 0, &[Value::I64(n)], &mut fuel).expect("interp");
    match v[0] {
        Value::I64(x) => x,
        o => panic!("{o:?}"),
    }
}

fn per_call(it: u32, mut f: impl FnMut()) -> f64 {
    for _ in 0..(it / 4).max(1) {
        f();
    }
    let t = Instant::now();
    for _ in 0..it {
        f();
    }
    t.elapsed().as_secs_f64() / it as f64
}

fn ns_per_iter(reps: u32, big: i64, small: i64, mut call: impl FnMut(i64) -> i64) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let b = per_call(10, || {
            black_box(call(big));
        });
        let s = per_call(10, || {
            black_box(call(small));
        });
        best = best.min((b - s) / (big - small) as f64 * 1e9);
    }
    best
}

#[test]
#[ignore = "ROI spike; run explicitly with --nocapture --ignored"]
fn bytecode_spike_alu() {
    let m = svm::text::parse_module(ALU).expect("parse");
    svm::verify::verify_module(&m).expect("verify");
    let arities: Vec<usize> = m.funcs.iter().map(|f| f.results.len()).collect();
    let prog = compile(&m.funcs[0], &arities);

    // Correctness: the flat executor must agree with the reference interpreter.
    for n in [0i64, 1, 7, 1000, 123_456] {
        let mut fuel = u64::MAX;
        assert_eq!(run_program(&prog, n, &mut fuel), interp(&m, n), "n={n}");
    }

    let i = ns_per_iter(5, 200_000, 1_000, |n| interp(&m, n));
    let b = ns_per_iter(5, 2_000_000, 1_000, |n| {
        let mut fuel = u64::MAX;
        run_program(&prog, n, &mut fuel)
    });
    println!("\nbytecode spike (alu recurrence, ns/iter):");
    println!("  tree-walk interp : {i:>9.3}");
    println!("  flat bytecode    : {b:>9.3}   ({:.2}x faster)", i / b);
}

// acc += leaf(acc, i) per iteration, leaf(a,b) = a + b — a direct call + return each step.
const CALL: &str = r#"
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const 0
  v2 = i64.const 0
  br block1(v0, v1, v2)
block1(v3: i64, v4: i64, v5: i64):
  v6 = i64.lt_s v5 v3
  br_if v6 block2(v3, v4, v5) block3(v4)
block2(v7: i64, v8: i64, v9: i64):
  v10 = call 1 (v8, v9)
  v11 = i64.const 1
  v12 = i64.add v9 v11
  br block1(v7, v10, v12)
block3(v13: i64):
  return v13
}
func (i64, i64) -> (i64) {
block0(v0: i64, v1: i64):
  v2 = i64.add v0 v1
  return v2
}
"#;

#[test]
#[ignore = "ROI spike; run explicitly with --nocapture --ignored"]
fn bytecode_spike_call() {
    let m = svm::text::parse_module(CALL).expect("parse");
    svm::verify::verify_module(&m).expect("verify");
    let arities: Vec<usize> = m.funcs.iter().map(|f| f.results.len()).collect();
    let progs: Vec<Program> = m.funcs.iter().map(|f| compile(f, &arities)).collect();

    for n in [0i64, 1, 7, 1000, 123_456] {
        let mut fuel = u64::MAX;
        assert_eq!(run_module(&progs, 0, n, &mut fuel), interp(&m, n), "n={n}");
    }

    let i = ns_per_iter(5, 200_000, 1_000, |n| interp(&m, n));
    let b = ns_per_iter(5, 2_000_000, 1_000, |n| {
        let mut fuel = u64::MAX;
        run_module(&progs, 0, n, &mut fuel)
    });
    println!("\nbytecode spike (call/return loop, ns/iter):");
    println!("  tree-walk interp : {i:>9.3}");
    println!("  flat bytecode    : {b:>9.3}   ({:.2}x faster)", i / b);
}
