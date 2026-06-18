//! Stage-1 differential spec: the first Futamura projection on a toy accumulator interpreter.
//!
//! A tiny bytecode interpreter is built in IR with a real dispatch loop (a `br_table` over a
//! readonly program in "constant memory"). `specialize` is run against a fixed program, and the
//! residual is asserted to (1) re-verify, (2) be byte-identical to the interpreter on the
//! reference interpreter for every input, and (3) actually be *compiled* — no opcode loads and
//! no dispatch table remain. Plus direct tests of static/dynamic branch specialization.

use svm_interp::{Trap, Value};
use svm_ir::{
    BinOp, Block, CmpOp, Data, Func, Inst, IntTy, LoadOp, Memory, Module, Terminator, ValType,
};
use svm_peval::{optimize_module, specialize, SpecArg};
use svm_verify::verify_module;

fn run(m: &Module, args: &[Value]) -> Result<Vec<Value>, Trap> {
    let mut fuel = 10_000_000u64;
    svm_interp::run(m, 0, args, &mut fuel)
}

// ----- the toy accumulator ISA -----
//
// Each instruction is 9 bytes: a 1-byte opcode followed by a little-endian i64 immediate.
// State is a single i64 accumulator plus the i64 runtime input.
const HALT: u8 = 0; //          return acc
const SETI: u8 = 1; // imm      acc = imm
const ADDI: u8 = 2; // imm      acc = acc + imm
const MULI: u8 = 3; // imm      acc = acc * imm
const ADDIN: u8 = 4; //         acc = acc + input

fn encode(program: &[(u8, i64)]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(program.len() * 9);
    for &(op, imm) in program {
        bytes.push(op);
        bytes.extend_from_slice(&imm.to_le_bytes());
    }
    bytes
}

/// Build the interpreter as an IR module with the program in a readonly data segment.
/// `interp(input: i64) -> i64`.
fn build_interpreter(program: &[(u8, i64)]) -> Module {
    let i64t = || ValType::I64;
    let load = |op, addr, offset| Inst::Load {
        op,
        addr,
        offset,
        align: 0,
    };
    let add = |a, b| Inst::IntBin {
        ty: IntTy::I64,
        op: BinOp::Add,
        a,
        b,
    };

    // Block 0 — entry(input): acc = 0, pc = 0; jump to the header.
    let entry = Block {
        params: vec![i64t()],                              // 0: input
        insts: vec![Inst::ConstI64(0), Inst::ConstI64(0)], // 1: acc, 2: pc
        term: Terminator::Br {
            target: 1,
            args: vec![1, 2, 0], // header(acc, pc, input)
        },
    };

    // Block 1 — header(acc, pc, input): decode at program+pc and dispatch on the opcode.
    let header = Block {
        params: vec![i64t(), i64t(), i64t()], // 0: acc, 1: pc, 2: input
        insts: vec![
            Inst::ConstI64(0),          // 3: program base (offset 0)
            add(3, 1),                  // 4: addr = base + pc
            load(LoadOp::I32_8U, 4, 0), // 5: op   (i32)
            load(LoadOp::I64, 4, 1),    // 6: imm  (i64)
        ],
        term: Terminator::BrTable {
            idx: 5,
            targets: vec![
                (2, vec![0]),          // HALT  -> halt(acc)
                (3, vec![0, 1, 6, 2]), // SETI  -> set(acc, pc, imm, input)
                (4, vec![0, 1, 6, 2]), // ADDI  -> add(acc, pc, imm, input)
                (5, vec![0, 1, 6, 2]), // MULI  -> mul(acc, pc, imm, input)
                (6, vec![0, 1, 6, 2]), // ADDIN -> addin(acc, pc, imm, input)
            ],
            default: (2, vec![0]), // any other opcode halts
        },
    };

    // Block 2 — halt(acc): return acc.
    let halt = Block {
        params: vec![i64t()],
        insts: vec![],
        term: Terminator::Return(vec![0]),
    };

    // The three "pc += 9 then loop" bodies share a shape; only the accumulator update differs.
    let step_body = |acc_update: Vec<Inst>, nacc_idx: u32| Block {
        params: vec![i64t(), i64t(), i64t(), i64t()], // 0: acc, 1: pc, 2: imm, 3: input
        insts: {
            let mut v = acc_update; // computes the new accumulator at index `nacc_idx`
            v.push(Inst::ConstI64(9)); // pc step
            v.push(add(1, nacc_idx + 1)); // npc = pc + 9
            v
        },
        term: Terminator::Br {
            target: 1,
            args: vec![nacc_idx, nacc_idx + 2, 3], // header(nacc, npc, input)
        },
    };

    // Block 3 — set: nacc = imm (forward the immediate directly; no compute needed).
    let set = Block {
        params: vec![i64t(), i64t(), i64t(), i64t()],
        insts: vec![Inst::ConstI64(9), add(1, 4)], // 4: step, 5: npc
        term: Terminator::Br {
            target: 1,
            args: vec![2, 5, 3], // header(imm, npc, input)
        },
    };
    // Block 4 — add: nacc = acc + imm (index 4), then the shared step.
    let add_blk = step_body(vec![add(0, 2)], 4);
    // Block 5 — mul: nacc = acc * imm.
    let mul_blk = step_body(
        vec![Inst::IntBin {
            ty: IntTy::I64,
            op: BinOp::Mul,
            a: 0,
            b: 2,
        }],
        4,
    );
    // Block 6 — addin: nacc = acc + input.
    let addin_blk = step_body(vec![add(0, 3)], 4);

    Module {
        funcs: vec![Func {
            params: vec![i64t()],
            results: vec![i64t()],
            blocks: vec![entry, header, halt, set, add_blk, mul_blk, addin_blk],
        }],
        memory: Some(Memory { size_log2: 16 }),
        data: vec![Data {
            offset: 0,
            readonly: true,
            bytes: encode(program),
        }],
        ..Default::default()
    }
}

fn assert_no_dispatch_left(residual: &Module) {
    for block in &residual.funcs[0].blocks {
        assert!(
            !block.insts.iter().any(|i| matches!(i, Inst::Load { .. })),
            "residual still contains an opcode/operand load — dispatch not fully folded"
        );
        assert!(
            !matches!(block.term, Terminator::BrTable { .. }),
            "residual still contains a dispatch table"
        );
    }
}

#[test]
fn futamura_specializes_accumulator_program() {
    // acc = ((10 + 5) + input) * 3
    let program = [(SETI, 10), (ADDI, 5), (ADDIN, 0), (MULI, 3), (HALT, 0)];
    let interp = build_interpreter(&program);
    verify_module(&interp).expect("interpreter verifies");

    let residual = specialize(&interp, 0, &[SpecArg::Dynamic]).expect("specializes");
    verify_module(&residual).expect("residual re-verifies");
    let opt = optimize_module(&residual);
    verify_module(&opt).expect("optimized residual re-verifies");

    // The dispatch loop is gone: no opcode loads, no br_table.
    assert_no_dispatch_left(&residual);

    for input in [0i64, 1, 2, -5, 100, i64::MIN] {
        let expect = (15i64.wrapping_add(input)).wrapping_mul(3);
        let args = [Value::I64(input)];
        assert_eq!(run(&interp, &args), Ok(vec![Value::I64(expect)]));
        assert_eq!(
            run(&residual, &args),
            Ok(vec![Value::I64(expect)]),
            "residual diverged at input {input}"
        );
        assert_eq!(run(&opt, &args), Ok(vec![Value::I64(expect)]));
    }
}

#[test]
fn futamura_constant_program_folds_to_a_constant() {
    // A program that never touches the input: acc = 7 * 6 = 42, for any input.
    let program = [(SETI, 7), (MULI, 6), (HALT, 0)];
    let interp = build_interpreter(&program);
    let residual = specialize(&interp, 0, &[SpecArg::Dynamic]).expect("specializes");
    verify_module(&residual).expect("residual re-verifies");
    assert_no_dispatch_left(&residual);

    for input in [0i64, 7, -123, 9999] {
        assert_eq!(run(&interp, &[Value::I64(input)]), Ok(vec![Value::I64(42)]));
        assert_eq!(
            run(&residual, &[Value::I64(input)]),
            Ok(vec![Value::I64(42)])
        );
    }

    // The entire program computation has folded away: the optimized residual contains no
    // arithmetic and no loads — only constant moves and branches. (It is still a chain of
    // `br`-connected blocks because block *merging* — collapsing a block into its single
    // unconditional predecessor — is a later increment; the values are all constant.)
    let opt = optimize_module(&residual);
    verify_module(&opt).expect("optimized residual verifies");
    for block in &opt.funcs[0].blocks {
        assert!(!block
            .insts
            .iter()
            .any(|i| matches!(i, Inst::IntBin { .. } | Inst::Load { .. })));
    }
}

// ----- direct branch-specialization tests (no interpreter) -----

/// `g(x) = if x != 0 { x * 2 } else { 99 }`.
fn branchy() -> Module {
    Module {
        funcs: vec![Func {
            params: vec![ValType::I64],
            results: vec![ValType::I64],
            blocks: vec![
                Block {
                    params: vec![ValType::I64], // 0: x
                    insts: vec![
                        Inst::ConstI64(0), // 1
                        Inst::IntCmp {
                            ty: IntTy::I64,
                            op: CmpOp::Ne,
                            a: 0,
                            b: 1,
                        }, // 2: x != 0
                    ],
                    term: Terminator::BrIf {
                        cond: 2,
                        then_blk: 1,
                        then_args: vec![0],
                        else_blk: 2,
                        else_args: vec![],
                    },
                },
                Block {
                    params: vec![ValType::I64], // 0: y
                    insts: vec![
                        Inst::ConstI64(2), // 1
                        Inst::IntBin {
                            ty: IntTy::I64,
                            op: BinOp::Mul,
                            a: 0,
                            b: 1,
                        }, // 2
                    ],
                    term: Terminator::Return(vec![2]),
                },
                Block {
                    params: vec![],
                    insts: vec![Inst::ConstI64(99)], // 0
                    term: Terminator::Return(vec![0]),
                },
            ],
        }],
        ..Default::default()
    }
}

#[test]
fn dynamic_condition_keeps_a_residual_branch() {
    let g = branchy();
    verify_module(&g).expect("g verifies");
    let residual = specialize(&g, 0, &[SpecArg::Dynamic]).expect("specializes");
    verify_module(&residual).expect("residual verifies");

    // The data-dependent branch survives specialization.
    assert!(residual.funcs[0]
        .blocks
        .iter()
        .any(|b| matches!(b.term, Terminator::BrIf { .. })));

    for x in [0i64, 1, 5, -3, 1000] {
        let expect = if x != 0 { x.wrapping_mul(2) } else { 99 };
        assert_eq!(run(&g, &[Value::I64(x)]), Ok(vec![Value::I64(expect)]));
        assert_eq!(
            run(&residual, &[Value::I64(x)]),
            Ok(vec![Value::I64(expect)])
        );
    }
}

#[test]
fn static_condition_resolves_the_branch() {
    let g = branchy();

    // x = 5 (static) -> the taken side only: 5 * 2 = 10, no branch left, takes no parameters.
    let taken = specialize(&g, 0, &[SpecArg::ConstI64(5)]).expect("specializes");
    verify_module(&taken).expect("verifies");
    assert!(taken.funcs[0]
        .blocks
        .iter()
        .all(|b| !matches!(b.term, Terminator::BrIf { .. })));
    assert!(taken.funcs[0].params.is_empty());
    assert_eq!(run(&taken, &[]), Ok(vec![Value::I64(10)]));

    // x = 0 (static) -> the else side: 99.
    let other = specialize(&g, 0, &[SpecArg::ConstI64(0)]).expect("specializes");
    verify_module(&other).expect("verifies");
    assert_eq!(run(&other, &[]), Ok(vec![Value::I64(99)]));
}
