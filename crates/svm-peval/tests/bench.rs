//! ROI benchmark for the Futamura projection. A register-machine bytecode interpreter runs a
//! program with a **runtime-controlled loop** (sum 1..=N, N = input), so specialization cannot
//! unroll it — the residual is a genuinely *compiled* native loop, while the interpreter pays
//! full bytecode-dispatch overhead every iteration. We time four configurations on the same
//! workload and print the speedups:
//!
//!   1. interp(interpreter)  — reference interpreter executing the bytecode interpreter (the
//!      classic "interpreted interpreter"; the slowest, and the honest baseline).
//!   2. interp(residual)     — reference interpreter executing the specialized program.
//!   3. jit(interpreter)     — native-compiled bytecode interpreter (dispatch in native code).
//!   4. jit(residual)        — native-compiled specialized program (the Futamura payoff).
//!
//! Run with:  cargo test -p svm-peval --test bench -- --ignored --nocapture

use std::hint::black_box;
use std::time::{Duration, Instant};

use svm_interp::Value;
use svm_ir::{
    BinOp, Block, CmpOp, Data, Func, Inst, IntTy, LoadOp, Memory, Module, StoreOp, Terminator,
    ValType, DEFAULT_RESERVED_LOG2,
};
use svm_jit::{CompiledModule, JitOutcome, Quota, INERT_CAP_THUNK};
use svm_peval::{
    optimize_module, specialize, specialize_with, specialize_with_config, SpecArg, SpecConfig,
};
use svm_verify::verify_module;

/// Static size of a module: total blocks, total instructions (terminators excluded), and the
/// encoded `.svmb` byte length — the three things specialization is meant to shrink (or, for a
/// runtime loop, trade dispatch for a tight compiled body).
#[derive(Clone, Copy)]
struct Sizes {
    blocks: usize,
    insts: usize,
    bytes: usize,
}

fn sizes(m: &Module) -> Sizes {
    Sizes {
        blocks: m.funcs.iter().map(|f| f.blocks.len()).sum(),
        insts: m
            .funcs
            .iter()
            .flat_map(|f| &f.blocks)
            .map(|b| b.insts.len())
            .sum(),
        bytes: svm_encode::encode_module(m).len(),
    }
}

// Register-machine bytecode, 9 bytes/instruction (opcode + little-endian i64 immediate).
// Two i64 registers: `acc` and `i`. State also carries the runtime `input`.
const HALT: u8 = 0; //          return acc
const SETACC: u8 = 1; // imm    acc = imm
const SETI_INPUT: u8 = 2; //    i = input
const ADD_I: u8 = 3; //         acc = acc + i
const DEC_I: u8 = 4; //         i = i - 1
const JNZ: u8 = 5; // imm       if i != 0 then pc = imm

fn encode_program(program: &[(u8, i64)]) -> Vec<u8> {
    let mut bytes = Vec::new();
    for &(op, imm) in program {
        bytes.push(op);
        bytes.extend_from_slice(&imm.to_le_bytes());
    }
    bytes
}

/// `interp(input: i64) -> i64`: a register machine with a real dispatch loop over a program in a
/// readonly data segment. State threaded through the header is `(acc, i, pc, input)`.
fn build_interpreter(program: &[(u8, i64)]) -> Module {
    let t = || ValType::I64;
    let add = |a, b| Inst::IntBin {
        ty: IntTy::I64,
        op: BinOp::Add,
        a,
        b,
    };
    let load = |op, addr, offset| Inst::Load {
        op,
        addr,
        offset,
        align: 0,
    };

    // 0 — entry(input): acc = 0, i = 0, pc = 0.
    let entry = Block {
        params: vec![t()],                                                    // 0: input
        insts: vec![Inst::ConstI64(0), Inst::ConstI64(0), Inst::ConstI64(0)], // 1: acc, 2: i, 3: pc
        term: Terminator::Br {
            target: 1,
            args: vec![1, 2, 3, 0],
        }, // header(acc, i, pc, input)
    };
    // 1 — header(acc, i, pc, input): decode + dispatch. Op-blocks get (acc, i, pc, imm, input).
    let header = Block {
        params: vec![t(), t(), t(), t()], // 0: acc, 1: i, 2: pc, 3: input
        insts: vec![
            Inst::ConstI64(0),          // 4: base
            add(4, 2),                  // 5: addr = base + pc
            load(LoadOp::I32_8U, 5, 0), // 6: op
            load(LoadOp::I64, 5, 1),    // 7: imm
        ],
        term: Terminator::BrTable {
            idx: 6,
            targets: vec![
                (2, vec![0]),             // HALT       -> halt(acc)
                (3, vec![0, 1, 2, 7, 3]), // SETACC
                (4, vec![0, 1, 2, 7, 3]), // SETI_INPUT
                (5, vec![0, 1, 2, 7, 3]), // ADD_I
                (6, vec![0, 1, 2, 7, 3]), // DEC_I
                (7, vec![0, 1, 2, 7, 3]), // JNZ
            ],
            default: (2, vec![0]),
        },
    };
    // 2 — halt(acc).
    let halt = Block {
        params: vec![t()],
        insts: vec![],
        term: Terminator::Return(vec![0]),
    };
    // Op-block params are always (acc, i, pc, imm, input) = indices 0,1,2,3,4.
    // 3 — setacc: acc = imm; pc += 9.
    let setacc = Block {
        params: vec![t(), t(), t(), t(), t()],
        insts: vec![Inst::ConstI64(9), add(2, 5)], // 5: 9, 6: npc
        term: Terminator::Br {
            target: 1,
            args: vec![3, 1, 6, 4],
        }, // (imm, i, npc, input)
    };
    // 4 — seti_input: i = input; pc += 9.
    let seti_input = Block {
        params: vec![t(), t(), t(), t(), t()],
        insts: vec![Inst::ConstI64(9), add(2, 5)],
        term: Terminator::Br {
            target: 1,
            args: vec![0, 4, 6, 4],
        }, // (acc, input, npc, input)
    };
    // 5 — add_i: acc = acc + i; pc += 9.
    let add_i = Block {
        params: vec![t(), t(), t(), t(), t()],
        insts: vec![add(0, 1), Inst::ConstI64(9), add(2, 6)], // 5: nacc, 6: 9, 7: npc
        term: Terminator::Br {
            target: 1,
            args: vec![5, 1, 7, 4],
        },
    };
    // 6 — dec_i: i = i - 1; pc += 9.
    let dec_i = Block {
        params: vec![t(), t(), t(), t(), t()],
        insts: vec![
            Inst::ConstI64(1),
            Inst::IntBin {
                ty: IntTy::I64,
                op: BinOp::Sub,
                a: 1,
                b: 5,
            }, // 6: ni
            Inst::ConstI64(9),
            add(2, 7), // 8: npc
        ],
        term: Terminator::Br {
            target: 1,
            args: vec![0, 6, 8, 4],
        },
    };
    // 7 — jnz: if i != 0 then pc = imm else pc += 9.
    let jnz = Block {
        params: vec![t(), t(), t(), t(), t()],
        insts: vec![
            Inst::ConstI64(0),
            Inst::IntCmp {
                ty: IntTy::I64,
                op: CmpOp::Ne,
                a: 1,
                b: 5,
            }, // 6: i != 0
            Inst::ConstI64(9),
            add(2, 7), // 8: npc_fallthrough
        ],
        term: Terminator::BrIf {
            cond: 6,
            then_blk: 1,
            then_args: vec![0, 1, 3, 4], // header(acc, i, imm, input)  -- jump
            else_blk: 1,
            else_args: vec![0, 1, 8, 4], // header(acc, i, npc, input)  -- fall through
        },
    };

    Module {
        funcs: vec![Func {
            params: vec![t()],
            results: vec![t()],
            blocks: vec![entry, header, halt, setacc, seti_input, add_i, dec_i, jnz],
        }],
        memory: Some(Memory { size_log2: 16 }),
        data: vec![Data {
            offset: 0,
            readonly: true,
            bytes: encode_program(program),
        }],
        ..Default::default()
    }
}

// sum 1..=N : acc=0; i=input; do { acc+=i; i-=1 } while i!=0; return acc.
fn sum_program() -> [(u8, i64); 6] {
    [
        (SETACC, 0),
        (SETI_INPUT, 0),
        (ADD_I, 0), // pc=18: loop body
        (DEC_I, 0),
        (JNZ, 18), // back to pc=18 while i!=0
        (HALT, 0),
    ]
}

fn interp_run(m: &Module, input: i64) -> i64 {
    let mut fuel = u64::MAX;
    match svm_interp::run(m, 0, &[Value::I64(input)], &mut fuel) {
        Ok(v) => match v.as_slice() {
            [Value::I64(x)] => *x,
            o => panic!("bad interp result {o:?}"),
        },
        Err(t) => panic!("interp trapped: {t:?}"),
    }
}

fn jit_run(m: &Module, input: i64) -> i64 {
    match svm_jit::compile_and_run(m, 0, &[input]) {
        Ok(JitOutcome::Returned(v)) => match v.as_slice() {
            [x] => *x,
            o => panic!("bad jit result {o:?}"),
        },
        o => panic!("bad jit outcome {o:?}"),
    }
}

fn best_of(reps: usize, mut f: impl FnMut() -> i64) -> Duration {
    f(); // warm up
    let mut best = Duration::MAX;
    for _ in 0..reps {
        let start = Instant::now();
        let r = f();
        best = best.min(start.elapsed());
        black_box(r);
    }
    best
}

#[test]
#[ignore = "perf benchmark — run with --ignored --nocapture"]
fn roi_futamura_loop() {
    let n: i64 = 2_000_000; // loop trip count (runtime input)
    let expect = n.wrapping_mul(n + 1) / 2; // sum 1..=N
    let reps = 5;

    let interp = build_interpreter(&sum_program());
    verify_module(&interp).expect("interpreter verifies");
    let residual = specialize(&interp, 0, &[SpecArg::Dynamic]).expect("specializes");
    verify_module(&residual).expect("residual verifies");

    // Correctness across all four configs before timing.
    for cfg in [
        ("interp(interpreter)", interp_run(&interp, n)),
        ("interp(residual)", interp_run(&residual, n)),
        ("jit(interpreter)", jit_run(&interp, n)),
        ("jit(residual)", jit_run(&residual, n)),
    ] {
        assert_eq!(cfg.1, expect, "{} produced wrong result", cfg.0);
    }

    let t_interp_interp = best_of(reps, || interp_run(&interp, n));
    let t_interp_resid = best_of(reps, || interp_run(&residual, n));
    let t_jit_interp = best_of(reps, || jit_run(&interp, n));
    let t_jit_resid = best_of(reps, || jit_run(&residual, n));

    let ms = |d: Duration| d.as_secs_f64() * 1e3;
    let base = ms(t_interp_interp);
    println!("\n=== Futamura ROI: sum 1..={n} (loop runs {n} times) ===");
    let (is, rs, os) = (
        sizes(&interp),
        sizes(&residual),
        sizes(&optimize_module(&residual)),
    );
    println!(
        "size  interpreter: {} blocks, {} insts, {} bytes",
        is.blocks, is.insts, is.bytes
    );
    println!(
        "size  residual:    {} blocks, {} insts, {} bytes",
        rs.blocks, rs.insts, rs.bytes
    );
    println!(
        "size  optimized:   {} blocks, {} insts, {} bytes  ({:.0}% of interpreter bytes)",
        os.blocks,
        os.insts,
        os.bytes,
        100.0 * os.bytes as f64 / is.bytes as f64
    );
    println!(
        "{:<22} {:>10} {:>10}",
        "configuration", "time(ms)", "speedup"
    );
    for (name, d) in [
        ("interp(interpreter)", t_interp_interp),
        ("interp(residual)", t_interp_resid),
        ("jit(interpreter)", t_jit_interp),
        ("jit(residual)", t_jit_resid),
    ] {
        println!("{:<22} {:>10.3} {:>9.1}x", name, ms(d), base / ms(d));
    }
    println!(
        "\nspecialization win, interpreter backend: {:.1}x",
        ms(t_interp_interp) / ms(t_interp_resid)
    );
    println!(
        "specialization win, JIT backend:         {:.1}x",
        ms(t_jit_interp) / ms(t_jit_resid)
    );
    println!(
        "end-to-end (interp(interp) -> jit(residual)): {:.1}x\n",
        ms(t_interp_interp) / ms(t_jit_resid)
    );
}

// ===========================================================================================
// Size corpus: report program-size gains (blocks / insts / encoded bytes) across a few interpreter
// *shapes* and program *workloads*, for interpreter vs residual vs optimized residual. Unlike the
// timing test this is cheap and assertion-backed, so it doubles as a size-regression guard. Print
// the table with:  cargo test -p svm-peval --test bench size_corpus -- --nocapture
// ===========================================================================================

// A second interpreter shape — a stack machine whose operand stack lives in the window and is
// renamed entirely out of the residual (Stage 2). 9 bytes/instruction.
const S_HALT: u8 = 0; //      pop and return top of stack
const S_PUSH: u8 = 1; // imm  push imm
const S_PUSHIN: u8 = 2; //    push the runtime input
const S_ADD: u8 = 3; //       pop b, pop a, push a + b
const S_MUL: u8 = 4; //       pop b, pop a, push a * b
const STACK_LO: u64 = 32768;
const STACK_HI: u64 = 32768 + 512;

fn build_stack_interpreter(program: &[(u8, i64)]) -> Module {
    let t = || ValType::I64;
    let load = |op, addr, offset| Inst::Load {
        op,
        addr,
        offset,
        align: 0,
    };
    let store = |addr, value| Inst::Store {
        op: StoreOp::I64,
        addr,
        value,
        offset: 0,
        align: 0,
    };
    let bin = |op, a, b| Inst::IntBin {
        ty: IntTy::I64,
        op,
        a,
        b,
    };

    let entry = Block {
        params: vec![t()],                                               // 0: input
        insts: vec![Inst::ConstI64(0), Inst::ConstI64(STACK_LO as i64)], // 1: pc, 2: sp
        term: Terminator::Br {
            target: 1,
            args: vec![1, 2, 0],
        },
    };
    let header = Block {
        params: vec![t(), t(), t()], // 0: pc, 1: sp, 2: input
        insts: vec![
            Inst::ConstI64(0),
            bin(BinOp::Add, 3, 0),      // 4: addr = base + pc
            load(LoadOp::I32_8U, 4, 0), // 5: op
            load(LoadOp::I64, 4, 1),    // 6: imm
        ],
        term: Terminator::BrTable {
            idx: 5,
            targets: vec![
                (2, vec![1]),          // HALT   -> halt(sp)
                (3, vec![0, 1, 6, 2]), // PUSH
                (4, vec![0, 1, 6, 2]), // PUSHIN
                (5, vec![0, 1, 6, 2]), // ADD
                (6, vec![0, 1, 6, 2]), // MUL
            ],
            default: (2, vec![1]),
        },
    };
    let halt = Block {
        params: vec![t()], // 0: sp
        insts: vec![
            Inst::ConstI64(8),
            bin(BinOp::Sub, 0, 1),   // 2: sp - 8
            load(LoadOp::I64, 2, 0), // 3: top
        ],
        term: Terminator::Return(vec![3]),
    };
    let push_body = |value_idx: u32| Block {
        params: vec![t(), t(), t(), t()], // 0: pc, 1: sp, 2: imm, 3: input
        insts: vec![
            store(1, value_idx),
            Inst::ConstI64(8),
            bin(BinOp::Add, 1, 4), // 5: nsp
            Inst::ConstI64(9),
            bin(BinOp::Add, 0, 6), // 7: npc
        ],
        term: Terminator::Br {
            target: 1,
            args: vec![7, 5, 3],
        },
    };
    let binop_body = |op: BinOp| Block {
        params: vec![t(), t(), t(), t()],
        insts: vec![
            Inst::ConstI64(8),
            bin(BinOp::Sub, 1, 4),   // 5: sp1
            load(LoadOp::I64, 5, 0), // 6: b
            bin(BinOp::Sub, 5, 4),   // 7: sp2
            load(LoadOp::I64, 7, 0), // 8: a
            bin(op, 8, 6),           // 9: r
            store(7, 9),
            Inst::ConstI64(9),
            bin(BinOp::Add, 0, 10), // 11: npc
        ],
        term: Terminator::Br {
            target: 1,
            args: vec![11, 5, 3],
        },
    };

    Module {
        funcs: vec![Func {
            params: vec![t()],
            results: vec![t()],
            blocks: vec![
                entry,
                header,
                halt,
                push_body(2), // PUSH imm
                push_body(3), // PUSHIN
                binop_body(BinOp::Add),
                binop_body(BinOp::Mul),
            ],
        }],
        memory: Some(Memory { size_log2: 16 }),
        data: vec![Data {
            offset: 0,
            readonly: true,
            bytes: encode_program(program),
        }],
        ..Default::default()
    }
}

fn has_dispatch(m: &Module) -> bool {
    m.funcs.iter().flat_map(|f| &f.blocks).any(|b| {
        matches!(b.term, Terminator::BrTable { .. })
            || b.insts.iter().any(|i| matches!(i, Inst::Load { .. }))
    })
}

#[test]
fn size_corpus() {
    // Each entry: (label, interpreter, residual). The residual is specialized against the program
    // with the runtime input left dynamic.
    let reg = |prog: &[(u8, i64)]| {
        let interp = build_interpreter(prog);
        verify_module(&interp).expect("interp verifies");
        let residual = specialize(&interp, 0, &[SpecArg::Dynamic]).expect("specializes");
        (interp, residual)
    };

    // A register-machine straight-line program (acc = 5 + input) and a constant one (acc = 7).
    let straight = [(SETACC, 5), (SETI_INPUT, 0), (ADD_I, 0), (HALT, 0)];
    let konst = [(SETACC, 7), (HALT, 0)];

    let (sl_i, sl_r) = reg(&straight);
    let (k_i, k_r) = reg(&konst);
    let (loop_i, loop_r) = reg(&sum_program());

    // Stack machine ((input + 5) * 3) with operand-stack renaming.
    let stack_prog = [
        (S_PUSHIN, 0),
        (S_PUSH, 5),
        (S_ADD, 0),
        (S_PUSH, 3),
        (S_MUL, 0),
        (S_HALT, 0),
    ];
    let stack_i = build_stack_interpreter(&stack_prog);
    verify_module(&stack_i).expect("stack interp verifies");
    let stack_r = specialize_with(&stack_i, 0, &[SpecArg::Dynamic], Some((STACK_LO, STACK_HI)))
        .expect("specializes with renaming");

    let corpus = [
        ("regmachine: constant (acc=7)", &k_i, &k_r),
        ("regmachine: straight-line (5+in)", &sl_i, &sl_r),
        ("regmachine: sum-loop (runtime loop)", &loop_i, &loop_r),
        ("stackmachine: (in+5)*3 [renamed]", &stack_i, &stack_r),
    ];

    println!(
        "\n=== specialization size corpus (i=interpreter, r=residual, o=optimized) ===\n{:<36} {:>12} {:>12} {:>16} {:>7}",
        "shape", "blocks i/r/o", "insts i/r/o", "bytes i/r/o", "bytes"
    );
    for (name, interp, residual) in corpus {
        verify_module(residual).expect("residual verifies");
        let opt = optimize_module(residual);
        verify_module(&opt).expect("optimized residual verifies");

        let (i, r, o) = (sizes(interp), sizes(residual), sizes(&opt));
        println!(
            "{:<36} {:>12} {:>12} {:>16} {:>6.0}%",
            name,
            format!("{}/{}/{}", i.blocks, r.blocks, o.blocks),
            format!("{}/{}/{}", i.insts, r.insts, o.insts),
            format!("{}/{}/{}", i.bytes, r.bytes, o.bytes),
            100.0 * o.bytes as f64 / i.bytes as f64,
        );

        // Every residual has the interpreter's dispatch (br_table + opcode/operand loads) folded
        // away — the defining property of the projection.
        assert!(
            !has_dispatch(residual),
            "{name}: dispatch (br_table / load) survived specialization"
        );
    }

    // The folding shapes collapse to a single block well under the interpreter's size; the loop
    // shape keeps a compiled loop but still drops the whole dispatch table.
    assert_eq!(optimize_module(&k_r).funcs[0].blocks.len(), 1);
    assert_eq!(optimize_module(&sl_r).funcs[0].blocks.len(), 1);
    assert_eq!(optimize_module(&stack_r).funcs[0].blocks.len(), 1);
    for (i, r) in [(&k_i, &k_r), (&sl_i, &sl_r), (&stack_i, &stack_r)] {
        assert!(
            sizes(&optimize_module(r)).bytes < sizes(i).bytes,
            "expected the residual to be smaller than the interpreter"
        );
    }
}

// ===========================================================================================
// Outlining + renaming TOGETHER (the PR-#82 capability): a stack-machine interpreter whose binary
// op dispatches to a *helper function*. The operand stack lives in a renamed region (Stage-2 SSA),
// and with `outline_calls` the helper becomes a shared residual function — so the renamed operand
// cells that are live across the call must thread across the residual call boundary (in as extra
// args, out as extra results). Before PR #82 this combination was rejected (outline required
// rename=None); this exercises it on an interpreter shape and reports what it buys.
//
//   cargo test -p svm-peval --test bench outline_rename -- --nocapture
// ===========================================================================================

const H_HALT: u8 = 0; //      pop and return top of stack
const H_PUSH: u8 = 1; // imm  push imm
const H_PUSHIN: u8 = 2; //    push the runtime input
const H_COMBINE: u8 = 3; //   pop b, pop a, push combine(a, b)  (a CALL to the helper)

const COMB_K1: i64 = 2654435761;
const COMB_K2: i64 = 40503;
/// The helper's pure kernel — chunky enough that inlining it at every call site visibly costs code.
fn combine_ref(a: i64, b: i64) -> i64 {
    a.wrapping_add(b)
        .wrapping_mul(COMB_K1)
        .wrapping_add(a)
        .wrapping_mul(COMB_K2)
}

/// A stack machine (operand stack in `[STACK_LO, STACK_HI)`, renamed to SSA) whose `COMBINE` opcode
/// calls a separate `combine(a, b)` helper (func 1) instead of inlining the arithmetic. Program is in
/// a readonly segment; `run(input)` returns the final top of stack.
fn build_stack_interpreter_calls(program: &[(u8, i64)]) -> Module {
    let t = || ValType::I64;
    let load = |op, addr, offset| Inst::Load {
        op,
        addr,
        offset,
        align: 0,
    };
    let store = |addr, value| Inst::Store {
        op: StoreOp::I64,
        addr,
        value,
        offset: 0,
        align: 0,
    };
    let bin = |op, a, b| Inst::IntBin {
        ty: IntTy::I64,
        op,
        a,
        b,
    };

    let entry = Block {
        params: vec![t()],                                               // 0: input
        insts: vec![Inst::ConstI64(0), Inst::ConstI64(STACK_LO as i64)], // 1: pc, 2: sp
        term: Terminator::Br {
            target: 1,
            args: vec![1, 2, 0],
        },
    };
    let header = Block {
        params: vec![t(), t(), t()], // 0: pc, 1: sp, 2: input
        insts: vec![
            Inst::ConstI64(0),
            bin(BinOp::Add, 3, 0),      // 4: addr = base + pc
            load(LoadOp::I32_8U, 4, 0), // 5: op
            load(LoadOp::I64, 4, 1),    // 6: imm
        ],
        term: Terminator::BrTable {
            idx: 5,
            targets: vec![
                (2, vec![1]),          // HALT    -> halt(sp)
                (3, vec![0, 1, 6, 2]), // PUSH
                (4, vec![0, 1, 6, 2]), // PUSHIN
                (5, vec![0, 1, 6, 2]), // COMBINE
            ],
            default: (2, vec![1]),
        },
    };
    let halt = Block {
        params: vec![t()], // 0: sp
        insts: vec![
            Inst::ConstI64(8),
            bin(BinOp::Sub, 0, 1),
            load(LoadOp::I64, 2, 0),
        ],
        term: Terminator::Return(vec![3]),
    };
    let push_body = |value_idx: u32| Block {
        params: vec![t(), t(), t(), t()], // 0: pc, 1: sp, 2: imm, 3: input
        insts: vec![
            store(1, value_idx),
            Inst::ConstI64(8),
            bin(BinOp::Add, 1, 4), // 5: nsp
            Inst::ConstI64(9),
            bin(BinOp::Add, 0, 6), // 7: npc
        ],
        term: Terminator::Br {
            target: 1,
            args: vec![7, 5, 3],
        },
    };
    // COMBINE: b = stk[sp-8]; a = stk[sp-16]; stk[sp-16] = combine(a, b); sp -= 8. The CALL happens
    // while cells below the two operands are still live in the renamed region — those must thread.
    let combine_body = Block {
        params: vec![t(), t(), t(), t()], // 0: pc, 1: sp, 2: imm, 3: input
        insts: vec![
            Inst::ConstI64(8),
            bin(BinOp::Sub, 1, 4),   // 5: sp1 = sp - 8
            load(LoadOp::I64, 5, 0), // 6: b = stk[sp1]
            bin(BinOp::Sub, 5, 4),   // 7: sp2 = sp1 - 8
            load(LoadOp::I64, 7, 0), // 8: a = stk[sp2]
            Inst::Call {
                func: 1,
                args: vec![8, 6],
            }, // 9: r = combine(a, b)
            store(7, 9),             // stk[sp2] = r
            Inst::ConstI64(9),
            bin(BinOp::Add, 0, 10), // 11: npc
        ],
        term: Terminator::Br {
            target: 1,
            args: vec![11, 5, 3],
        }, // sp = sp1 (net -8)
    };
    // combine(a, b) = ((a + b) * K1 + a) * K2 — a few ops, no memory.
    let combine = Func {
        params: vec![t(), t()], // 0: a, 1: b
        results: vec![t()],
        blocks: vec![Block {
            params: vec![t(), t()],
            insts: vec![
                bin(BinOp::Add, 0, 1), // 2: a+b
                Inst::ConstI64(COMB_K1),
                bin(BinOp::Mul, 2, 3), // 4: *K1
                bin(BinOp::Add, 4, 0), // 5: +a
                Inst::ConstI64(COMB_K2),
                bin(BinOp::Mul, 5, 6), // 7: *K2
            ],
            term: Terminator::Return(vec![7]),
        }],
    };

    Module {
        funcs: vec![
            Func {
                params: vec![t()],
                results: vec![t()],
                blocks: vec![
                    entry,
                    header,
                    halt,
                    push_body(2),
                    push_body(3),
                    combine_body,
                ],
            },
            combine,
        ],
        memory: Some(Memory { size_log2: 16 }),
        data: vec![Data {
            offset: 0,
            readonly: true,
            bytes: encode_program(program),
        }],
        ..Default::default()
    }
}

/// The reference result of running `program` over `input` (mirrors the interpreter's semantics).
fn run_stack_calls_ref(program: &[(u8, i64)], input: i64) -> i64 {
    let mut stk = Vec::new();
    let mut pc = 0;
    loop {
        let (op, imm) = program[pc];
        match op {
            H_HALT => return *stk.last().unwrap(),
            H_PUSH => stk.push(imm),
            H_PUSHIN => stk.push(input),
            H_COMBINE => {
                let b = stk.pop().unwrap();
                let a = stk.pop().unwrap();
                stk.push(combine_ref(a, b));
            }
            _ => unreachable!(),
        }
        pc += 1;
    }
}

// An accumulator machine with a small heap addressed by a *runtime* value (`acc`). When specialized
// with a rename region, a heap access whose address is dynamic can't be proved disjoint from the
// region, so the engine bails `Unsupported` — unless the caller promises the region is private
// (`rename_is_private`), which lets the access be emitted faithfully. The adversarial fuzz shape for
// the `Unsupported` path and the private-region contract.
const A_HALT: u8 = 0; //       return acc
const A_SETI: u8 = 1; // imm   acc = imm                (acc constant)
const A_ADDIN: u8 = 2; //      acc = acc + input        (acc becomes dynamic)
const A_ADDK: u8 = 3; // imm   acc = acc + imm
const A_STOREH: u8 = 4; //     heap[acc & 63] = acc     (dynamic addr when acc is dynamic)
const A_LOADH: u8 = 5; //      acc = heap[acc & 63]
const HEAP_LO: u64 = 4096; // disjoint from the program (at 0) and the rename region

fn build_heap_interpreter(program: &[(u8, i64)]) -> Module {
    let t = || ValType::I64;
    let bin = |op, a, b| Inst::IntBin {
        ty: IntTy::I64,
        op,
        a,
        b,
    };
    let load = |op, addr, offset| Inst::Load {
        op,
        addr,
        offset,
        align: 0,
    };
    // Address of heap[acc & 63] from the op-block's `acc` (param 0): (acc & 63) << 3 + HEAP_LO.
    let heap_addr = || {
        vec![
            Inst::ConstI64(63),
            bin(BinOp::And, 0, 4), // 5: slot = acc & 63
            Inst::ConstI64(3),
            bin(BinOp::Shl, 5, 6), // 7: off = slot << 3
            Inst::ConstI64(HEAP_LO as i64),
            bin(BinOp::Add, 7, 8), // 9: addr
        ]
    };

    let entry = Block {
        params: vec![t()],                                 // 0: input
        insts: vec![Inst::ConstI64(0), Inst::ConstI64(0)], // 1: acc, 2: pc
        term: Terminator::Br {
            target: 1,
            args: vec![1, 2, 0],
        },
    };
    let header = Block {
        params: vec![t(), t(), t()], // 0: acc, 1: pc, 2: input
        insts: vec![
            Inst::ConstI64(0),
            bin(BinOp::Add, 3, 1),      // 4: addr = base + pc
            load(LoadOp::I32_8U, 4, 0), // 5: op
            load(LoadOp::I64, 4, 1),    // 6: imm
        ],
        term: Terminator::BrTable {
            idx: 5,
            targets: vec![
                (2, vec![0]),          // HALT   -> halt(acc)
                (3, vec![0, 1, 6, 2]), // SETI
                (4, vec![0, 1, 6, 2]), // ADDIN
                (5, vec![0, 1, 6, 2]), // ADDK
                (6, vec![0, 1, 6, 2]), // STOREH
                (7, vec![0, 1, 6, 2]), // LOADH
            ],
            default: (2, vec![0]),
        },
    };
    let halt = Block {
        params: vec![t()],
        insts: vec![],
        term: Terminator::Return(vec![0]),
    };
    // Op blocks: params (acc, pc, imm, input) = 0,1,2,3.
    let seti = Block {
        params: vec![t(), t(), t(), t()],
        insts: vec![Inst::ConstI64(9), bin(BinOp::Add, 1, 4)], // 5: npc
        term: Terminator::Br {
            target: 1,
            args: vec![2, 5, 3],
        }, // acc=imm
    };
    let addin = Block {
        params: vec![t(), t(), t(), t()],
        insts: vec![
            bin(BinOp::Add, 0, 3), // 4: acc+input
            Inst::ConstI64(9),
            bin(BinOp::Add, 1, 5), // 6: npc
        ],
        term: Terminator::Br {
            target: 1,
            args: vec![4, 6, 3],
        },
    };
    let addk = Block {
        params: vec![t(), t(), t(), t()],
        insts: vec![
            bin(BinOp::Add, 0, 2), // 4: acc+imm
            Inst::ConstI64(9),
            bin(BinOp::Add, 1, 5), // 6: npc
        ],
        term: Terminator::Br {
            target: 1,
            args: vec![4, 6, 3],
        },
    };
    let storeh = {
        let mut insts = heap_addr(); // 4..9, addr at 9
        insts.push(Inst::Store {
            op: StoreOp::I64,
            addr: 9,
            value: 0,
            offset: 0,
            align: 0,
        });
        insts.push(Inst::ConstI64(9));
        insts.push(bin(BinOp::Add, 1, 10)); // 11: npc
        Block {
            params: vec![t(), t(), t(), t()],
            insts,
            term: Terminator::Br {
                target: 1,
                args: vec![0, 11, 3],
            }, // acc unchanged
        }
    };
    let loadh = {
        let mut insts = heap_addr(); // 4..9, addr at 9
        insts.push(load(LoadOp::I64, 9, 0)); // 10: nacc
        insts.push(Inst::ConstI64(9));
        insts.push(bin(BinOp::Add, 1, 11)); // 12: npc
        Block {
            params: vec![t(), t(), t(), t()],
            insts,
            term: Terminator::Br {
                target: 1,
                args: vec![10, 12, 3],
            },
        }
    };

    Module {
        funcs: vec![Func {
            params: vec![t()],
            results: vec![t()],
            blocks: vec![entry, header, halt, seti, addin, addk, storeh, loadh],
        }],
        memory: Some(Memory { size_log2: 16 }),
        data: vec![Data {
            offset: 0,
            readonly: true,
            bytes: encode_program(program),
        }],
        ..Default::default()
    }
}

/// Count surviving memory ops (loads + stores) across all functions.
fn memory_ops(m: &Module) -> usize {
    m.funcs
        .iter()
        .flat_map(|f| &f.blocks)
        .flat_map(|b| &b.insts)
        .filter(|i| matches!(i, Inst::Load { .. } | Inst::Store { .. }))
        .count()
}

#[test]
fn outline_rename_threads_operand_stack_through_helpers() {
    // A left fold: acc starts at 0; each step combines the running acc with combine(input, c_i). At the
    // inner COMBINE the acc cell is live below the two operands, so it must thread across the call.
    let cs = [3i64, 5, 7, 11, 13, 17];
    let mut prog = vec![(H_PUSH, 0)]; // acc = 0
    for &c in &cs {
        prog.push((H_PUSHIN, 0)); // input
        prog.push((H_PUSH, c)); // c_i
        prog.push((H_COMBINE, 0)); // t = combine(input, c_i)   -- acc live below
        prog.push((H_COMBINE, 0)); // acc = combine(acc, t)
    }
    prog.push((H_HALT, 0));

    let interp = build_stack_interpreter_calls(&prog);
    verify_module(&interp).expect("interpreter verifies");
    let region = Some((STACK_LO, STACK_HI));

    // (1) OLD fallback: rename only. Outlining was rejected with a region, so the helper inlines —
    // one function, its body duplicated at every COMBINE.
    let inline = specialize_with(&interp, 0, &[SpecArg::Dynamic], region).expect("inline+rename");
    // (2) Outline WITHOUT rename: the helper is shared, but the operand stack stays in *real memory*
    // (loads/stores survive) — the only way to outline this interpreter before #82.
    let outline_mem = specialize_with_config(
        &interp,
        0,
        &[SpecArg::Dynamic],
        &SpecConfig {
            outline_calls: true,
            ..SpecConfig::default()
        },
    )
    .expect("outline, no rename");
    // (3) NEW (#82): rename + outline together. Shared residual helper AND the operand stack stays in
    // SSA — the live renamed cells thread across each residual call.
    let outlined = specialize_with_config(
        &interp,
        0,
        &[SpecArg::Dynamic],
        &SpecConfig {
            rename: region,
            outline_calls: true,
            ..SpecConfig::default()
        },
    )
    .expect("outline+rename");
    for m in [&inline, &outline_mem, &outlined] {
        verify_module(m).expect("residual verifies");
    }

    // The payoff of #82: outlining no longer forces the operand stack into memory. (2) outlines but
    // leaves real memory traffic; (3) outlines AND keeps the stack in SSA — zero memory ops, even
    // across the call (cells crossed as args/results). The old fallback (1) is also SSA but can't
    // share the helper.
    assert!(
        memory_ops(&outline_mem) > 0,
        "outline-no-rename should keep stack in memory"
    );
    assert_eq!(
        memory_ops(&outlined),
        0,
        "outline+rename must keep the stack in SSA"
    );
    assert_eq!(memory_ops(&inline), 0, "inline+rename is SSA");
    // Dispatch folds away in every config.
    assert!(!has_dispatch(&inline) && !has_dispatch(&outlined));
    // (1) is a single inlined function; (2)/(3) share the outlined helper as separate functions.
    assert_eq!(
        inline.funcs.len(),
        1,
        "inline+rename should be a single function"
    );
    assert!(
        outlined.funcs.len() > 1,
        "outline+rename should emit a shared residual helper"
    );

    // Correctness: interpreter, all three residuals, and the Rust reference agree.
    for input in [0i64, 1, -3, 7, 1000, i64::MIN] {
        let want = run_stack_calls_ref(&prog, input);
        assert_eq!(
            interp_run(&interp, input),
            want,
            "interpreter wrong at {input}"
        );
        assert_eq!(
            jit_run(&inline, input),
            want,
            "inline+rename wrong at {input}"
        );
        assert_eq!(
            jit_run(&outline_mem, input),
            want,
            "outline-no-rename wrong at {input}"
        );
        assert_eq!(
            jit_run(&outlined, input),
            want,
            "outline+rename wrong at {input}"
        );
    }

    let row = |label: &str, m: &Module| {
        let o = optimize_module(m);
        let s = sizes(&o);
        println!(
            "{label:<22} {:>3} fns {:>4} blocks {:>4} insts {:>5} bytes {:>4} mem-ops",
            o.funcs.len(),
            s.blocks,
            s.insts,
            s.bytes,
            memory_ops(&o)
        );
    };
    println!(
        "\n=== outline + rename together (stack machine, {} COMBINE calls) ===",
        2 * cs.len()
    );
    println!(
        "interpreter:           {:>3} fns ... {:>4} bytes",
        interp.funcs.len(),
        sizes(&interp).bytes
    );
    row("inline+rename (old)", &inline);
    row("outline, no-rename", &outline_mem);
    row("outline+rename (#82)", &outlined);
    println!(
        "\n#82 buys: outlining WITHOUT spilling the operand stack — {} -> 0 memory ops vs outline-no-rename.",
        memory_ops(&optimize_module(&outline_mem))
    );
    // The function count ({} here) is dominated by *constant-argument specialization*, not dead cells:
    // combine(in, c_i) bakes each distinct c_i into its own residual helper (by design — the same
    // per-static-pattern specialization the `outlining_makes_*` tests assert), while the all-dynamic
    // combine(acc, t) calls already share one helper. Renamed region cells are *over-threaded* (the
    // whole region crosses each call, including cells redundant with the operands or dead above the
    // live stack), which inflates helper signatures but does not change the count.
    println!(
        "note: {} residual fns — fragmentation is constant-arg specialization, not dead cells; the\n      live cells are over-threaded (bigger signatures), a separate liveness cleanup.",
        optimize_module(&outlined).funcs.len()
    );
}

// ===========================================================================================
// Differential fuzzer: throw many *random guest programs* at the specializer and assert the
// partial-evaluation correctness property — the reference interpreter running the interpreter, the
// reference interpreter running the residual, and the JIT running the residual all agree:
//
//     interp(interpreter, in) == interp(residual, in) == jit(residual, in)
//
// This is the literal "throw more programs at it": it catches miscompiles the curated corpus misses,
// and the distribution of Budget / Unsupported / non-terminating outcomes maps the specializer's
// bail surface. Deterministic (fixed seed) so it doubles as a regression guard.
//
//   cargo test -p svm-peval --test bench fuzz -- --nocapture
// ===========================================================================================

/// SplitMix64 — a tiny deterministic PRNG (the workspace is dependency-free, so no `rand`).
struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }
    /// A small immediate in `[-9, 10]` — keeps results readable; magnitude doesn't change behavior.
    fn imm(&mut self) -> i64 {
        self.below(20) as i64 - 9
    }
}

/// A random register-machine program (SETACC / SETI_INPUT / ADD_I / DEC_I / JNZ, then HALT). JNZ
/// targets any instruction, so programs may loop — including statically (→ the specializer's Budget
/// bail) or input-dependently (→ a compiled residual loop). Termination is handled by the oracle.
fn gen_reg_program(rng: &mut Rng) -> Vec<(u8, i64)> {
    let body = 1 + rng.below(8) as usize;
    let total = body + 1; // + HALT
    let mut prog = Vec::with_capacity(total);
    for _ in 0..body {
        prog.push(match rng.below(5) {
            0 => (SETACC, rng.imm()),
            1 => (SETI_INPUT, 0),
            2 => (ADD_I, 0),
            3 => (DEC_I, 0),
            _ => (JNZ, 9 * rng.below(total as u64) as i64), // jump to a valid instruction boundary
        });
    }
    prog.push((HALT, 0));
    prog
}

/// A random, stack-valid program for the helper-calling stack machine (PUSH / PUSHIN / COMBINE, then
/// HALT). Generation tracks depth so COMBINE always has two operands and the stack never overflows
/// the renamed region; the program always terminates (no loops) and fully unrolls.
fn gen_stack_calls_program(rng: &mut Rng) -> Vec<(u8, i64)> {
    let steps = 2 + rng.below(10) as usize;
    let mut prog = Vec::new();
    let mut depth = 0u32;
    for _ in 0..steps {
        let can_combine = depth >= 2;
        let push_only = !(2..50).contains(&depth); // <2 has no operands; ≥50 keeps the 64-cell region
        match if push_only {
            rng.below(2)
        } else {
            rng.below(3)
        } {
            0 => {
                prog.push((H_PUSH, rng.imm()));
                depth += 1;
            }
            1 => {
                prog.push((H_PUSHIN, 0));
                depth += 1;
            }
            _ if can_combine => {
                prog.push((H_COMBINE, 0));
                depth -= 1;
            }
            _ => unreachable!(),
        }
    }
    if depth == 0 {
        prog.push((H_PUSHIN, 0));
    }
    prog.push((H_HALT, 0));
    prog
}

/// A random accumulator+heap program (SETI / ADDIN / ADDK / STOREH / LOADH, then HALT). A heap op
/// after an ADDIN reaches it with a *dynamic* `acc` ⇒ a dynamic heap address; otherwise the address
/// is constant. Always terminates (no loops).
fn gen_heap_program(rng: &mut Rng) -> Vec<(u8, i64)> {
    let body = 2 + rng.below(6) as usize;
    let mut prog = Vec::with_capacity(body + 1);
    for _ in 0..body {
        prog.push(match rng.below(5) {
            0 => (A_SETI, rng.imm()),
            1 => (A_ADDIN, 0),
            2 => (A_ADDK, rng.imm()),
            3 => (A_STOREH, 0),
            _ => (A_LOADH, 0),
        });
    }
    prog.push((A_HALT, 0));
    prog
}

/// Run `entry`(`input`) on the reference interpreter under a fuel cap. `None` ⇒ it did not finish
/// (a non-terminating program / fuel exhausted) — the only `Err` our trap-free op sets can produce.
fn interp_try(m: &Module, input: i64, fuel: u64) -> Option<i64> {
    let mut f = fuel;
    match svm_interp::run(m, 0, &[Value::I64(input)], &mut f) {
        Ok(v) => match v.as_slice() {
            [Value::I64(x)] => Some(*x),
            _ => None,
        },
        Err(_) => None,
    }
}

#[derive(Default)]
struct Tally {
    programs: usize,
    succeeded: usize,      // specialized and the oracle held on ≥1 terminating input
    budget: usize,         // SpecError::Budget (unbounded specialization)
    unsupported: usize,    // SpecError::Unsupported
    nonterminating: usize, // specialized, but no chosen input terminated (nothing to check)
    oracle_checks: usize,  // total interp==interp==jit comparisons that passed
    // One example program per bail category, for the report ("where it bails").
    ex_budget: Option<Vec<(u8, i64)>>,
    ex_unsupported: Option<Vec<(u8, i64)>>,
    ex_nonterm: Option<Vec<(u8, i64)>>,
}

impl Tally {
    fn merge(&mut self, o: Tally) {
        self.programs += o.programs;
        self.succeeded += o.succeeded;
        self.budget += o.budget;
        self.unsupported += o.unsupported;
        self.nonterminating += o.nonterminating;
        self.oracle_checks += o.oracle_checks;
        self.ex_budget = self.ex_budget.take().or(o.ex_budget);
        self.ex_unsupported = self.ex_unsupported.take().or(o.ex_unsupported);
        self.ex_nonterm = self.ex_nonterm.take().or(o.ex_nonterm);
    }
}

/// Specialize `interp` against each generated program and assert the PE oracle. `specialize` is the
/// per-shape entry (plain, or rename+outline). Returns the outcome tally.
fn fuzz_shape(
    label: &str,
    n: usize,
    seed: u64,
    mut gen: impl FnMut(&mut Rng) -> Vec<(u8, i64)>,
    build: impl Fn(&[(u8, i64)]) -> Module,
    specialize: impl Fn(&Module) -> Result<Module, svm_peval::SpecError>,
) -> Tally {
    const FUEL: u64 = 200_000;
    let inputs = [0i64, 1, 2, 5, 11]; // non-negative + small ⇒ most legitimate loops terminate fast
    let mut rng = Rng(seed);
    let mut t = Tally::default();
    for _ in 0..n {
        t.programs += 1;
        let prog = gen(&mut rng);
        let interp = build(&prog);
        verify_module(&interp).expect("generated interpreter must verify");

        let residual = match specialize(&interp) {
            Ok(r) => r,
            Err(svm_peval::SpecError::Budget) => {
                t.budget += 1;
                t.ex_budget.get_or_insert_with(|| prog.clone());
                continue;
            }
            Err(svm_peval::SpecError::Unsupported) => {
                t.unsupported += 1;
                t.ex_unsupported.get_or_insert_with(|| prog.clone());
                continue;
            }
            Err(e) => panic!("{label}: unexpected specialize error {e:?} on {prog:?}"),
        };
        // A residual that fails verification is always a bug, regardless of semantics.
        verify_module(&residual)
            .unwrap_or_else(|e| panic!("{label}: residual failed to verify ({e:?}) on {prog:?}"));

        let mut terminated = false;
        let mut jit_checked = false;
        for &input in &inputs {
            let Some(want) = interp_try(&interp, input, FUEL) else {
                continue; // interpreter didn't finish on this input; nothing to compare
            };
            // The residual must reproduce the interpreter exactly. A divergence here (different value,
            // or the residual fails to terminate where the interpreter did) is a real miscompile.
            let got = interp_try(&residual, input, FUEL);
            assert_eq!(
                got,
                Some(want),
                "{label}: interp(residual) diverged from interp(interpreter) at input {input} on {prog:?}"
            );
            t.oracle_checks += 1;
            terminated = true;
            // The residual is proven terminating (above), so the JIT can't hang. Check it once.
            if !jit_checked {
                assert_eq!(
                    jit_run(&residual, input),
                    want,
                    "{label}: jit(residual) diverged at input {input} on {prog:?}"
                );
                t.oracle_checks += 1;
                jit_checked = true;
            }
        }
        if terminated {
            t.succeeded += 1;
        } else {
            t.nonterminating += 1;
            t.ex_nonterm.get_or_insert_with(|| prog.clone());
        }
    }
    t
}

/// Print the merged outcome line for a shape, plus one example program per bail category — so the
/// report shows *where* the fuzzer hits the engine's limits, not just how often.
fn report_shape(label: &str, t: &Tally) {
    println!(
        "{label:<26} {:>5} programs | {:>5} ok, {:>3} budget, {:>3} unsup, {:>3} nonterm | {:>6} oracle checks",
        t.programs, t.succeeded, t.budget, t.unsupported, t.nonterminating, t.oracle_checks
    );
    if let Some(p) = &t.ex_budget {
        println!("    budget  e.g. {p:?}");
    }
    if let Some(p) = &t.ex_unsupported {
        println!("    unsup   e.g. {p:?}");
    }
    if let Some(p) = &t.ex_nonterm {
        println!("    nonterm e.g. {p:?}");
    }
}

/// Adversarial differential for the `Unsupported` path and the `rename_is_private` contract: the
/// accumulator+heap interpreter (heap addressed by a runtime value) specialized under a rename region
/// two ways. Non-private: a dynamic heap address can't be proved disjoint from the region ⇒
/// `Unsupported`. Private: the caller's promise lets the access through ⇒ a faithful residual. Asserts
/// the bail fail-closes, private is at least as permissive as non-private, and every residual is
/// correct (oracle holds). Returns (programs, non-private Unsupported, rescued-by-private, checks).
fn fuzz_heap(n: usize, seeds: &[u64]) -> (usize, usize, usize, usize) {
    const FUEL: u64 = 200_000;
    let region = Some((STACK_LO, STACK_HI)); // a (private) scratch range disjoint from the heap
    let inputs = [0i64, 1, 2, 7, -3, 100];
    let cfg = |private| SpecConfig {
        rename: region,
        rename_is_private: private,
        ..SpecConfig::default()
    };
    let (mut progs, mut np_unsup, mut rescues, mut checks) = (0usize, 0usize, 0usize, 0usize);
    for &seed in seeds {
        let mut rng = Rng(seed ^ 0x5151);
        for _ in 0..n {
            progs += 1;
            let prog = gen_heap_program(&mut rng);
            let interp = build_heap_interpreter(&prog);
            verify_module(&interp).expect("heap interp verifies");
            let np = specialize_with_config(&interp, 0, &[SpecArg::Dynamic], &cfg(false));
            let pv = specialize_with_config(&interp, 0, &[SpecArg::Dynamic], &cfg(true));
            // Heap programs always terminate; precompute the reference results.
            let wants: Vec<i64> = inputs
                .iter()
                .map(|&i| interp_try(&interp, i, FUEL).expect("heap interp terminates"))
                .collect();
            let icheck = |r: &Module, who: &str| {
                verify_module(r)
                    .unwrap_or_else(|e| panic!("{who} residual verify {e:?}: {prog:?}"));
                for (k, &input) in inputs.iter().enumerate() {
                    assert_eq!(
                        interp_try(r, input, FUEL),
                        Some(wants[k]),
                        "{who}: interp(residual) diverged at {input} on {prog:?}"
                    );
                }
            };
            // Private must be at least as permissive: it never bails where non-private succeeds.
            if np.is_ok() {
                assert!(
                    pv.is_ok(),
                    "private bailed where non-private succeeded: {prog:?}"
                );
            }
            match (np, pv) {
                (Ok(rn), Ok(rp)) => {
                    icheck(&rn, "np");
                    icheck(&rp, "pv");
                    assert_eq!(jit_run(&rp, inputs[0]), wants[0], "jit(pv): {prog:?}");
                    checks += 1;
                }
                (Err(svm_peval::SpecError::Unsupported), Ok(rp)) => {
                    np_unsup += 1;
                    rescues += 1;
                    icheck(&rp, "pv");
                    assert_eq!(jit_run(&rp, inputs[0]), wants[0], "jit(pv): {prog:?}");
                    checks += 1;
                }
                (
                    Err(svm_peval::SpecError::Unsupported),
                    Err(svm_peval::SpecError::Unsupported),
                ) => {
                    np_unsup += 1;
                }
                (a, b) => panic!("unexpected outcomes np={a:?} pv={b:?} on {prog:?}"),
            }
        }
    }
    (progs, np_unsup, rescues, checks)
}

fn run_fuzz(n: usize, seeds: &[u64], verify_bails: bool) {
    let region = Some((STACK_LO, STACK_HI));
    println!("\n=== differential fuzz: interp(interp) == interp(residual) == jit(residual) ===");

    // (1) Register machine, no rename: branches and loops exercise dispatch folding and the Budget
    // bail on statically unbounded loops. (2) Helper-calling stack machine, rename + outline: fuzzes
    // the PR-#82 path (region cells threaded across outlined calls). Each shape runs every seed; the
    // tallies merge so the bail surface is sampled across many program streams.
    let mut reg = Tally::default();
    let mut stk = Tally::default();
    for &seed in seeds {
        reg.merge(fuzz_shape(
            "regmachine (plain)",
            n,
            seed,
            gen_reg_program,
            build_interpreter,
            |m| specialize(m, 0, &[SpecArg::Dynamic]),
        ));
        stk.merge(fuzz_shape(
            "stackmachine+helpers (#82)",
            n,
            seed ^ 0x9999,
            gen_stack_calls_program,
            build_stack_interpreter_calls,
            move |m| {
                specialize_with_config(
                    m,
                    0,
                    &[SpecArg::Dynamic],
                    &SpecConfig {
                        rename: region,
                        outline_calls: true,
                        ..SpecConfig::default()
                    },
                )
            },
        ));
    }
    report_shape("regmachine (plain)", &reg);
    report_shape("stackmachine+helpers (#82)", &stk);

    // Confirm the bails are legitimate, not the engine giving up on tractable programs. A
    // "nonterminating" example must still not finish with 50x the fuel (genuinely an infinite loop,
    // not merely slow); a "budget" example must indeed be unspecializable (re-bail) — a static,
    // unbounded loop. (The oracle assertions inside fuzz_shape already prove no miscompiles.) The
    // high-fuel re-run is slow, so it is gated to the thorough run.
    if verify_bails {
        if let Some(p) = &reg.ex_nonterm {
            let m = build_interpreter(p);
            for input in [0i64, 1, 2, 5, 11] {
                assert!(
                    interp_try(&m, input, 10_000_000).is_none(),
                    "nonterm example actually terminated at {input}: {p:?}"
                );
            }
        }
        if let Some(p) = &reg.ex_budget {
            assert!(
                matches!(
                    specialize(&build_interpreter(p), 0, &[SpecArg::Dynamic]),
                    Err(svm_peval::SpecError::Budget)
                ),
                "budget example did not re-bail: {p:?}"
            );
        }
    }

    // (3) Adversarial: dynamic-address heap access under a rename region — the canonical Unsupported
    // case — fuzzed non-private (must bail) vs private (must rescue + stay correct).
    let (hp_progs, hp_unsup, hp_rescues, hp_checks) = fuzz_heap(n.min(150), seeds);
    println!(
        "heap dyn-addr (priv vs non)  {hp_progs:>5} programs | {hp_unsup} non-private Unsupported, {hp_rescues} rescued by private | {hp_checks} oracle checks"
    );

    assert!(
        reg.succeeded > 0 && reg.oracle_checks > 0,
        "register fuzz did no useful work"
    );
    assert!(
        stk.succeeded > 0 && stk.oracle_checks > 0,
        "stack fuzz did no useful work"
    );
    // The thorough run must actually exercise the Unsupported path and the private rescue.
    if verify_bails {
        assert!(
            hp_rescues > 0,
            "dynamic-address Unsupported path / private rescue was not exercised"
        );
    }
}

/// A fast, deterministic smoke that runs by default — a cheap regression guard on the PE oracle.
#[test]
fn fuzz_specialization_smoke() {
    run_fuzz(16, &[0x5EED], false);
}

/// The thorough run (hundreds of programs per shape; JIT-compiles each residual, so it's slow). Run
/// it to map the bail surface: `cargo test -p svm-peval --test bench fuzz_specialization_thorough --
/// --ignored --nocapture`.
#[test]
#[ignore = "thorough fuzz — run with --ignored --nocapture"]
fn fuzz_specialization_thorough() {
    run_fuzz(300, &[0x1111, 0x2222, 0x3333, 0x4444], true); // 4 seeds x 300 = 1200 programs/shape
}

// ===========================================================================================
// Gain spectrum: how specialization performs across a *range* of guest programs, from "folds to a
// constant" through loops whose per-iteration real work grows. It shows where the win is large and
// durable (overhead-bound: dispatch dwarfs the work) and where it shrinks (work-bound: the real
// per-iteration work dwarfs the dispatch we removed). JIT timing uses compile-once / run-many so the
// numbers are run time, not compile time.
//
//   cargo test -p svm-peval --test bench gain_spectrum -- --ignored --nocapture
// ===========================================================================================

/// A runtime-trip sum loop whose body adds `i` to `acc` `adds_per_iter` times before `i--; jnz`.
/// More adds per iteration ⇒ more real work relative to the (fixed) dispatch overhead. Result is
/// `adds_per_iter * sum(1..=input)`.
fn sum_loop_program(adds_per_iter: usize) -> Vec<(u8, i64)> {
    let mut p = vec![(SETACC, 0), (SETI_INPUT, 0)]; // acc=0, i=input; loop starts at pc=18
    let loop_start = 9 * p.len() as i64;
    for _ in 0..adds_per_iter {
        p.push((ADD_I, 0));
    }
    p.push((DEC_I, 0));
    p.push((JNZ, loop_start));
    p.push((HALT, 0));
    p
}

fn jit_compile(m: &Module) -> CompiledModule {
    CompiledModule::compile(
        m,
        0,
        INERT_CAP_THUNK,
        std::ptr::null_mut(),
        DEFAULT_RESERVED_LOG2,
        None,
        None,
        None,
        None,
        Quota::default(),
        0,
    )
    .expect("jit compile")
}

fn jit_call(cm: &mut CompiledModule, input: i64) -> i64 {
    match cm.run(&[input], None, None, None) {
        Ok((JitOutcome::Returned(v), _)) => v[0],
        o => panic!("jit outcome {o:?}"),
    }
}

#[test]
#[ignore = "benchmark — run with --ignored --nocapture"]
fn gain_spectrum() {
    // Each row: (label, interpreter, residual, runtime-loop?). The size win shows for all; the JIT
    // speedup is only meaningful where the program has a runtime loop (otherwise it folds away and run
    // time is dominated by the fixed per-call floor).
    let reg = |prog: &[(u8, i64)]| {
        let i = build_interpreter(prog);
        let r = optimize_module(&specialize(&i, 0, &[SpecArg::Dynamic]).expect("specializes"));
        (i, r)
    };
    let stk = |prog: &[(u8, i64)]| {
        let i = build_stack_interpreter(prog);
        let r = optimize_module(
            &specialize_with(&i, 0, &[SpecArg::Dynamic], Some((STACK_LO, STACK_HI)))
                .expect("specializes"),
        );
        (i, r)
    };

    let (const_i, const_r) = reg(&[(SETACC, 7), (HALT, 0)]);
    let (affine_i, affine_r) = reg(&[(SETACC, 5), (SETI_INPUT, 0), (ADD_I, 0), (HALT, 0)]);
    let (l1_i, l1_r) = reg(&sum_loop_program(1));
    let (l2_i, l2_r) = reg(&sum_loop_program(2));
    let (l4_i, l4_r) = reg(&sum_loop_program(4));
    let (l8_i, l8_r) = reg(&sum_loop_program(8));
    // Stack machine (operand stack renamed to SSA): (in+5)*3, and a longer chained expression.
    let (sx_i, sx_r) = stk(&[
        (S_PUSHIN, 0),
        (S_PUSH, 5),
        (S_ADD, 0),
        (S_PUSH, 3),
        (S_MUL, 0),
        (S_HALT, 0),
    ]);
    let mut big = vec![(S_PUSHIN, 0)];
    for k in 1..=8 {
        big.push((S_PUSH, k));
        big.push((S_ADD, 0));
        big.push((S_PUSHIN, 0));
        big.push((S_MUL, 0));
    }
    big.push((S_HALT, 0));
    let (bx_i, bx_r) = stk(&big);

    // (label, interp, residual, loop?)
    let rows: [(&str, &Module, &Module, bool); 8] = [
        ("reg: constant (acc=7)", &const_i, &const_r, false),
        ("reg: affine (in+5)", &affine_i, &affine_r, false),
        ("reg: loop body x1 (light)", &l1_i, &l1_r, true),
        ("reg: loop body x2", &l2_i, &l2_r, true),
        ("reg: loop body x4", &l4_i, &l4_r, true),
        ("reg: loop body x8 (heavy)", &l8_i, &l8_r, true),
        ("stack: (in+5)*3 [renamed]", &sx_i, &sx_r, false),
        ("stack: chained expr [renamed]", &bx_i, &bx_r, false),
    ];

    println!("\n=== gain spectrum (size + runtime-loop JIT speedup) ===");
    println!(
        "{:<32} {:>16} {:>7} {:>7} {:>10}",
        "program", "bytes i/r", "%", "folded", "jit x"
    );
    // These sum loops count i down from `input`, so they terminate only for input >= 1 (a floor of 1
    // is one iteration). The folding rows are fine at any input.
    let n: i64 = 2_000_000;
    const FLOOR: i64 = 1;
    for (label, interp, residual, is_loop) in rows {
        // Correctness: residual matches the interpreter on a few inputs.
        for input in [1i64, 2, 7, 50] {
            assert_eq!(
                jit_run(residual, input),
                interp_run(interp, input),
                "{label}: residual diverged at {input}"
            );
        }
        let (si, sr) = (sizes(interp), sizes(residual));
        let folded = if has_dispatch(residual) { "no" } else { "yes" };

        let speedup = if is_loop {
            // Compile once; subtract the 1-iteration floor so the number is per-loop compute.
            let mut ci = jit_compile(interp);
            let mut cr = jit_compile(residual);
            assert_eq!(jit_call(&mut ci, n), jit_call(&mut cr, n), "{label}");
            let fi = best(5, || {
                black_box(jit_call(&mut ci, FLOOR));
            });
            let fr = best(5, || {
                black_box(jit_call(&mut cr, FLOOR));
            });
            let ti = best(5, || {
                black_box(jit_call(&mut ci, n));
            }) - fi;
            let tr = best(5, || {
                black_box(jit_call(&mut cr, n));
            }) - fr;
            format!("{:.1}x", ti / tr)
        } else {
            "—".to_string()
        };
        println!(
            "{label:<32} {:>16} {:>6.0}% {:>7} {:>10}",
            format!("{}/{}", si.bytes, sr.bytes),
            100.0 * sr.bytes as f64 / si.bytes as f64,
            folded,
            speedup
        );
    }
    println!(
        "\nlight loop bodies are overhead-bound (dispatch dwarfs the work ⇒ big, durable speedup);\nheavier bodies are work-bound (real work dwarfs the removed dispatch ⇒ the speedup shrinks)."
    );
}

/// Best (min) wall time of `reps` runs of `f`, in seconds; warms up once.
fn best(reps: usize, mut f: impl FnMut()) -> f64 {
    f();
    let mut b = f64::INFINITY;
    for _ in 0..reps {
        let t = Instant::now();
        f();
        b = b.min(t.elapsed().as_secs_f64());
    }
    b
}
