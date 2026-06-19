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
    ValType,
};
use svm_jit::JitOutcome;
use svm_peval::{optimize_module, specialize, specialize_with, SpecArg};
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
