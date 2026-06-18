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
    BinOp, Block, CmpOp, Data, Func, Inst, IntTy, LoadOp, Memory, Module, Terminator, ValType,
};
use svm_jit::JitOutcome;
use svm_peval::{specialize, SpecArg};
use svm_verify::verify_module;

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
    println!("residual blocks: {}", residual.funcs[0].blocks.len());
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
