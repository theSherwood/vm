//! Stage-1 differential spec: the first Futamura projection on a toy accumulator interpreter.
//!
//! A tiny bytecode interpreter is built in IR with a real dispatch loop (a `br_table` over a
//! readonly program in "constant memory"). `specialize` is run against a fixed program, and the
//! residual is asserted to (1) re-verify, (2) be byte-identical to the interpreter on the
//! reference interpreter for every input, and (3) actually be *compiled* — no opcode loads and
//! no dispatch table remain. Plus direct tests of static/dynamic branch specialization.

use svm_interp::{Trap, Value};
use svm_ir::{
    BinOp, Block, CastOp, CmpOp, ConvOp, Data, FBinOp, FCmpOp, FToI, FUnOp, FloatTy, Func, IToF,
    Inst, IntTy, LoadOp, Memory, Module, StoreOp, Terminator, ValType,
};
use svm_peval::{
    optimize_module, specialize, specialize_with, specialize_with_config, SpecArg, SpecConfig,
};
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
    // After cleanup the whole straight-line program is a single block: input -> +15 -> *3.
    assert_eq!(opt.funcs[0].blocks.len(), 1);
    assert!(matches!(opt.funcs[0].blocks[0].term, Terminator::Return(_)));

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

    // After cleanup the whole thing collapses to a single block that returns the constant 42.
    let opt = optimize_module(&residual);
    verify_module(&opt).expect("optimized residual verifies");
    assert_eq!(opt.funcs[0].blocks.len(), 1);
    assert!(matches!(
        opt.funcs[0].blocks[0].insts.last(),
        Some(Inst::ConstI64(42))
    ));
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

// ===========================================================================================
// Stage 2 — value-stack renaming: a stack-machine interpreter whose operand stack lives in the
// window and is renamed entirely out of the residual.
// ===========================================================================================

// Stack-machine ISA: 9 bytes per instruction (1-byte opcode + little-endian i64 immediate).
const S_HALT: u8 = 0; //          pop and return the top of stack
const S_PUSH: u8 = 1; // imm      push imm
const S_PUSHIN: u8 = 2; //        push the runtime input
const S_ADD: u8 = 3; //           pop b, pop a, push a + b
const S_MUL: u8 = 4; //           pop b, pop a, push a * b

// The operand stack lives in a private, zero-initialized window range. It must sit in a
// different host page from the readonly program at offset 0 — RO protection is page-granular
// (host pages can be up to 16 KiB), so a stack sharing the program's page would fault on write.
const STACK_LO: u64 = 32768;
const STACK_HI: u64 = 32768 + 512; // 64 i64 slots

/// A stack-machine interpreter: `interp(input: i64) -> i64`. The operand stack is kept in the
/// window (8-byte slots based at `STACK_LO`, addressed by a stack pointer `sp`).
fn build_stack_interpreter(program: &[(u8, i64)]) -> Module {
    let i64t = || ValType::I64;
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

    // 0 — entry(input): pc = 0, sp = STACK_LO; jump to header.
    let entry = Block {
        params: vec![i64t()],                                            // 0: input
        insts: vec![Inst::ConstI64(0), Inst::ConstI64(STACK_LO as i64)], // 1: pc, 2: sp
        term: Terminator::Br {
            target: 1,
            args: vec![1, 2, 0], // header(pc, sp, input)
        },
    };

    // 1 — header(pc, sp, input): decode at program+pc, dispatch.
    let header = Block {
        params: vec![i64t(), i64t(), i64t()], // 0: pc, 1: sp, 2: input
        insts: vec![
            Inst::ConstI64(0),          // 3: program base
            bin(BinOp::Add, 3, 0),      // 4: addr = base + pc
            load(LoadOp::I32_8U, 4, 0), // 5: op
            load(LoadOp::I64, 4, 1),    // 6: imm
        ],
        term: Terminator::BrTable {
            idx: 5,
            targets: vec![
                (2, vec![1]),          // HALT   -> halt(sp)
                (3, vec![0, 1, 6, 2]), // PUSH   -> push(pc, sp, imm, input)
                (4, vec![0, 1, 6, 2]), // PUSHIN -> pushin(...)
                (5, vec![0, 1, 6, 2]), // ADD    -> add(...)
                (6, vec![0, 1, 6, 2]), // MUL    -> mul(...)
            ],
            default: (2, vec![1]),
        },
    };

    // 2 — halt(sp): pop the top slot and return it.
    let halt = Block {
        params: vec![i64t()], // 0: sp
        insts: vec![
            Inst::ConstI64(8),       // 1
            bin(BinOp::Sub, 0, 1),   // 2: sp - 8
            load(LoadOp::I64, 2, 0), // 3: top
        ],
        term: Terminator::Return(vec![3]),
    };

    // A "push then loop" body: store `value_idx` at [sp], sp += 8, pc += 9.
    let push_body = |value_idx: u32| Block {
        params: vec![i64t(), i64t(), i64t(), i64t()], // 0: pc, 1: sp, 2: imm, 3: input
        insts: vec![
            store(1, value_idx),   // [sp] = value
            Inst::ConstI64(8),     // 4
            bin(BinOp::Add, 1, 4), // 5: nsp = sp + 8
            Inst::ConstI64(9),     // 6
            bin(BinOp::Add, 0, 6), // 7: npc = pc + 9
        ],
        term: Terminator::Br {
            target: 1,
            args: vec![7, 5, 3], // header(npc, nsp, input)
        },
    };
    // 3 — push(imm): the pushed value is the immediate (index 2).
    let push = push_body(2);
    // 4 — pushin: the pushed value is the input (index 3).
    let pushin = push_body(3);

    // A binary-op body: pop b, pop a, push (a `op` b); nsp = sp - 8.
    let binop_body = |op: BinOp| Block {
        params: vec![i64t(), i64t(), i64t(), i64t()], // 0: pc, 1: sp, 2: imm, 3: input
        insts: vec![
            Inst::ConstI64(8),       // 4
            bin(BinOp::Sub, 1, 4),   // 5: sp1 = sp - 8
            load(LoadOp::I64, 5, 0), // 6: b = [sp1]
            bin(BinOp::Sub, 5, 4),   // 7: sp2 = sp1 - 8
            load(LoadOp::I64, 7, 0), // 8: a = [sp2]
            bin(op, 8, 6),           // 9: r = a op b
            store(7, 9),             // [sp2] = r
            Inst::ConstI64(9),       // 10
            bin(BinOp::Add, 0, 10),  // 11: npc = pc + 9
        ],
        term: Terminator::Br {
            target: 1,
            args: vec![11, 5, 3], // header(npc, nsp = sp1, input)
        },
    };
    // 5 — add, 6 — mul.
    let add = binop_body(BinOp::Add);
    let mul = binop_body(BinOp::Mul);

    Module {
        funcs: vec![Func {
            params: vec![i64t()],
            results: vec![i64t()],
            blocks: vec![entry, header, halt, push, pushin, add, mul],
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

fn assert_no_memory_ops(residual: &Module) {
    for block in &residual.funcs[0].blocks {
        assert!(
            !block
                .insts
                .iter()
                .any(|i| matches!(i, Inst::Load { .. } | Inst::Store { .. })),
            "residual still touches the window — the stack was not fully renamed"
        );
        assert!(!matches!(block.term, Terminator::BrTable { .. }));
    }
}

#[test]
fn renames_stack_machine_to_pure_ssa() {
    // ((input + 5) * 3) computed entirely on the in-memory operand stack.
    let program = [
        (S_PUSHIN, 0),
        (S_PUSH, 5),
        (S_ADD, 0),
        (S_PUSH, 3),
        (S_MUL, 0),
        (S_HALT, 0),
    ];
    let interp = build_stack_interpreter(&program);
    verify_module(&interp).expect("interpreter verifies");

    let residual = specialize_with(&interp, 0, &[SpecArg::Dynamic], Some((STACK_LO, STACK_HI)))
        .expect("specializes with stack renaming");
    verify_module(&residual).expect("residual re-verifies");
    let opt = optimize_module(&residual);
    verify_module(&opt).expect("optimized residual re-verifies");

    // The whole operand stack is gone: no loads, no stores, no dispatch.
    assert_no_memory_ops(&residual);
    // After cleanup it is a single straight-line block.
    assert_eq!(opt.funcs[0].blocks.len(), 1);

    for input in [0i64, 1, 2, -5, 100, i64::MAX] {
        let expect = (input.wrapping_add(5)).wrapping_mul(3);
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
fn renamed_cell_flows_across_a_dynamic_branch() {
    // h(x) = { [R] = x; if x != 0 { [R] * 2 } else { [R] + 100 } }. The renamed cell holds a
    // dynamic value that must flow into both branches as a block parameter.
    let region = (STACK_LO, STACK_LO + 8);
    let st = |addr, value| Inst::Store {
        op: StoreOp::I64,
        addr,
        value,
        offset: 0,
        align: 0,
    };
    let ld = |addr| Inst::Load {
        op: LoadOp::I64,
        addr,
        offset: 0,
        align: 0,
    };
    let h = Module {
        funcs: vec![Func {
            params: vec![ValType::I64],
            results: vec![ValType::I64],
            blocks: vec![
                Block {
                    params: vec![ValType::I64], // 0: x
                    insts: vec![
                        Inst::ConstI64(STACK_LO as i64), // 1: R
                        st(1, 0),                        // [R] = x
                        Inst::ConstI64(0),               // 2
                        Inst::IntCmp {
                            ty: IntTy::I64,
                            op: CmpOp::Ne,
                            a: 0,
                            b: 2,
                        }, // 3: x != 0
                    ],
                    term: Terminator::BrIf {
                        cond: 3,
                        then_blk: 1,
                        then_args: vec![],
                        else_blk: 2,
                        else_args: vec![],
                    },
                },
                Block {
                    params: vec![], // then
                    insts: vec![
                        Inst::ConstI64(STACK_LO as i64), // 0: R
                        ld(0),                           // 1: v = [R]
                        Inst::ConstI64(2),               // 2
                        Inst::IntBin {
                            ty: IntTy::I64,
                            op: BinOp::Mul,
                            a: 1,
                            b: 2,
                        }, // 3
                    ],
                    term: Terminator::Return(vec![3]),
                },
                Block {
                    params: vec![], // else
                    insts: vec![
                        Inst::ConstI64(STACK_LO as i64), // 0: R
                        ld(0),                           // 1: v = [R]
                        Inst::ConstI64(100),             // 2
                        Inst::IntBin {
                            ty: IntTy::I64,
                            op: BinOp::Add,
                            a: 1,
                            b: 2,
                        }, // 3
                    ],
                    term: Terminator::Return(vec![3]),
                },
            ],
        }],
        memory: Some(Memory { size_log2: 16 }),
        ..Default::default()
    };
    verify_module(&h).expect("h verifies");

    let residual = specialize_with(&h, 0, &[SpecArg::Dynamic], Some(region)).expect("specializes");
    verify_module(&residual).expect("residual verifies");
    assert_no_memory_ops(&residual);
    // The data-dependent branch survives; the cell became a value threaded into both arms.
    assert!(residual.funcs[0]
        .blocks
        .iter()
        .any(|b| matches!(b.term, Terminator::BrIf { .. })));

    for x in [0i64, 1, 5, -3, 1000] {
        let expect = if x != 0 {
            x.wrapping_mul(2)
        } else {
            x.wrapping_add(100)
        };
        assert_eq!(run(&h, &[Value::I64(x)]), Ok(vec![Value::I64(expect)]));
        assert_eq!(
            run(&residual, &[Value::I64(x)]),
            Ok(vec![Value::I64(expect)])
        );
    }
}

// ===========================================================================================
// Caller-declared constant memory: the program need not be in readonly memory — the caller can
// promise an arbitrary region (or supply overlay bytes) is constant at specialization time.
// ===========================================================================================

#[test]
fn specializes_program_in_mutable_memory_via_const_region() {
    // The same accumulator program, but in a *writable* data segment. The caller promises the
    // program bytes are constant via a const_region; specialization folds exactly as if readonly.
    let program = [(SETI, 10), (ADDI, 5), (ADDIN, 0), (MULI, 3), (HALT, 0)];
    let mut interp = build_interpreter(&program);
    interp.data[0].readonly = false; // an arbitrary mutable buffer now
    verify_module(&interp).expect("interpreter verifies");

    let len = (program.len() * 9) as u64;
    let cfg = SpecConfig {
        const_regions: vec![(0, len)],
        ..SpecConfig::default()
    };
    let residual =
        specialize_with_config(&interp, 0, &[SpecArg::Dynamic], &cfg).expect("specializes");
    verify_module(&residual).expect("residual re-verifies");
    assert_no_dispatch_left(&residual); // folded just like the readonly case

    for input in [0i64, 1, 2, -5, 100] {
        let expect = (15i64.wrapping_add(input)).wrapping_mul(3);
        assert_eq!(
            run(&interp, &[Value::I64(input)]),
            Ok(vec![Value::I64(expect)])
        );
        assert_eq!(
            run(&residual, &[Value::I64(input)]),
            Ok(vec![Value::I64(expect)])
        );
    }
}

#[test]
fn overlay_bytes_drive_folding() {
    // A function that loads an i64 from a writable window location. Without a promise the load
    // stays in the residual; with a matching const_overlay it folds to the constant.
    let value: i64 = 0x0102_0304_0506_0708;
    let m = Module {
        funcs: vec![Func {
            params: vec![],
            results: vec![ValType::I64],
            blocks: vec![Block {
                params: vec![],
                insts: vec![
                    Inst::ConstI64(16), // 0: addr
                    Inst::Load {
                        op: LoadOp::I64,
                        addr: 0,
                        offset: 0,
                        align: 0,
                    }, // 1
                ],
                term: Terminator::Return(vec![1]),
            }],
        }],
        memory: Some(Memory { size_log2: 16 }),
        // The bytes are actually present in the window (so the unspecialized run reads them), but
        // in a *writable* segment the engine won't fold on its own.
        data: vec![Data {
            offset: 16,
            readonly: false,
            bytes: value.to_le_bytes().to_vec(),
        }],
        ..Default::default()
    };
    verify_module(&m).expect("verifies");

    // No promise: the load is preserved (the engine won't fold writable memory).
    let plain = specialize(&m, 0, &[]).expect("specializes");
    assert!(plain.funcs[0]
        .blocks
        .iter()
        .any(|b| b.insts.iter().any(|i| matches!(i, Inst::Load { .. }))));
    assert_eq!(run(&plain, &[]), Ok(vec![Value::I64(value)]));

    // With an overlay promising those bytes, the load folds away to the constant.
    let cfg = SpecConfig {
        const_overlays: vec![(16, value.to_le_bytes().to_vec())],
        ..SpecConfig::default()
    };
    let folded = specialize_with_config(&m, 0, &[], &cfg).expect("specializes");
    assert!(!folded.funcs[0]
        .blocks
        .iter()
        .any(|b| b.insts.iter().any(|i| matches!(i, Inst::Load { .. }))));
    assert_eq!(run(&folded, &[]), Ok(vec![Value::I64(value)]));
}

// ===========================================================================================
// Widened coverage: a *float* bytecode interpreter. The float arithmetic isn't constant-folded
// (the engine tracks integer constants only), but it passes through to the residual faithfully,
// so dispatch is still eliminated — the residual is the compiled float computation.
// ===========================================================================================

// 9 bytes/instruction (opcode + ignored i64 immediate). State: an f64 accumulator `facc`.
const F_HALT: u8 = 0; //      return facc
const F_ADDSELF: u8 = 1; //   facc = facc + facc
const F_SQ: u8 = 2; //        facc = facc * facc

fn build_float_interpreter(program: &[(u8, i64)]) -> Module {
    let f64t = || ValType::F64;
    let i64t = || ValType::I64;
    let iadd = |a, b| Inst::IntBin {
        ty: IntTy::I64,
        op: BinOp::Add,
        a,
        b,
    };
    let fbin = |op, a, b| Inst::FBin {
        ty: FloatTy::F64,
        op,
        a,
        b,
    };

    // 0 — entry(finput): facc = finput, pc = 0.
    let entry = Block {
        params: vec![f64t()],           // 0: finput
        insts: vec![Inst::ConstI64(0)], // 1: pc
        term: Terminator::Br {
            target: 1,
            args: vec![0, 1, 0], // header(facc = finput, pc, finput)
        },
    };
    // 1 — header(facc, pc, finput): decode the opcode and dispatch.
    let header = Block {
        params: vec![f64t(), i64t(), f64t()], // 0: facc, 1: pc, 2: finput
        insts: vec![
            Inst::ConstI64(0), // 3: base
            iadd(3, 1),        // 4: addr = base + pc
            Inst::Load {
                op: LoadOp::I32_8U,
                addr: 4,
                offset: 0,
                align: 0,
            }, // 5: op
        ],
        term: Terminator::BrTable {
            idx: 5,
            targets: vec![
                (2, vec![0]),       // HALT     -> halt(facc)
                (3, vec![0, 1, 2]), // ADDSELF  -> addself(facc, pc, finput)
                (4, vec![0, 1, 2]), // SQ       -> sq(facc, pc, finput)
            ],
            default: (2, vec![0]),
        },
    };
    // 2 — halt(facc).
    let halt = Block {
        params: vec![f64t()],
        insts: vec![],
        term: Terminator::Return(vec![0]),
    };
    // A float op then pc += 9: params (facc, pc, finput).
    let fstep = |op: FBinOp| Block {
        params: vec![f64t(), i64t(), f64t()],
        insts: vec![
            fbin(op, 0, 0),    // 3: nfacc = facc `op` facc
            Inst::ConstI64(9), // 4
            iadd(1, 4),        // 5: npc = pc + 9
        ],
        term: Terminator::Br {
            target: 1,
            args: vec![3, 5, 2],
        },
    };
    let addself = fstep(FBinOp::Add);
    let sq = fstep(FBinOp::Mul);

    Module {
        funcs: vec![Func {
            params: vec![f64t()],
            results: vec![f64t()],
            blocks: vec![entry, header, halt, addself, sq],
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

fn run_f64(m: &Module, x: f64) -> f64 {
    let mut fuel = 10_000_000u64;
    match svm_interp::run(m, 0, &[Value::F64(x)], &mut fuel) {
        Ok(v) => match v.as_slice() {
            [Value::F64(r)] => *r,
            o => panic!("unexpected float result {o:?}"),
        },
        Err(t) => panic!("interp trapped: {t:?}"),
    }
}

#[test]
fn specializes_float_interpreter() {
    // facc = finput; ADDSELF -> 2*finput; SQ -> (2*finput)^2 = 4*finput^2.
    let program = [(F_ADDSELF, 0), (F_SQ, 0), (F_HALT, 0)];
    let interp = build_float_interpreter(&program);
    verify_module(&interp).expect("interpreter verifies");

    let residual = specialize(&interp, 0, &[SpecArg::Dynamic]).expect("specializes");
    verify_module(&residual).expect("residual re-verifies");

    // Dispatch is gone (no opcode loads, no br_table), but the float ops remain — the residual
    // *is* the compiled float computation.
    assert_no_dispatch_left(&residual);
    assert!(
        residual.funcs[0]
            .blocks
            .iter()
            .any(|b| b.insts.iter().any(|i| matches!(i, Inst::FBin { .. }))),
        "the residual should carry the float arithmetic"
    );

    for x in [0.0f64, 0.5, 1.5, 2.5, 3.0, -4.0] {
        let expect = (2.0 * x) * (2.0 * x);
        assert_eq!(run_f64(&interp, x), expect, "interp at {x}");
        assert_eq!(run_f64(&residual, x), expect, "residual at {x}");
        // The float residual also JIT-compiles to native (f64 passed/returned as raw bits).
        let jit = match svm_jit::compile_and_run(&residual, 0, &[x.to_bits() as i64]) {
            Ok(svm_jit::JitOutcome::Returned(v)) => f64::from_bits(v[0] as u64),
            o => panic!("unexpected jit outcome {o:?}"),
        };
        assert_eq!(jit, expect, "jit residual at {x}");
    }
}

#[test]
fn private_rename_allows_dynamic_heap_access() {
    // h(ptr) = 7 + *ptr, but the constant 7 is staged through a renamed operand-stack slot. The
    // stack store/load are renamed away; the *ptr load has a dynamic address and must survive.
    let region = (STACK_LO, STACK_LO + 8);
    let st = |addr, value| Inst::Store {
        op: StoreOp::I64,
        addr,
        value,
        offset: 0,
        align: 0,
    };
    let ld = |addr| Inst::Load {
        op: LoadOp::I64,
        addr,
        offset: 0,
        align: 0,
    };
    let h = Module {
        funcs: vec![Func {
            params: vec![ValType::I64], // 0: ptr
            results: vec![ValType::I64],
            blocks: vec![Block {
                params: vec![ValType::I64],
                insts: vec![
                    Inst::ConstI64(STACK_LO as i64), // 1: stack slot addr
                    Inst::ConstI64(7),               // 2
                    st(1, 2),                        // [slot] = 7   (renamed)
                    ld(1),                           // 3: v = [slot] (renamed -> 7)
                    ld(0),                           // 4: hv = *ptr  (dynamic addr -> residual)
                    Inst::IntBin {
                        ty: IntTy::I64,
                        op: BinOp::Add,
                        a: 3,
                        b: 4,
                    }, // 5: v + hv
                ],
                term: Terminator::Return(vec![5]),
            }],
        }],
        memory: Some(Memory { size_log2: 16 }),
        // The heap cell *ptr will point at: a writable word holding 100.
        data: vec![Data {
            offset: 4096,
            readonly: false,
            bytes: 100i64.to_le_bytes().to_vec(),
        }],
        ..Default::default()
    };
    verify_module(&h).expect("verifies");

    // Without the privacy promise, the dynamic-address load under an active rename region bails.
    let conservative = SpecConfig {
        rename: Some(region),
        ..SpecConfig::default()
    };
    assert_eq!(
        specialize_with_config(&h, 0, &[SpecArg::Dynamic], &conservative),
        Err(svm_peval::SpecError::Unsupported)
    );

    // With it, the heap access is emitted residually while the stack is still renamed away.
    let cfg = SpecConfig {
        rename: Some(region),
        rename_is_private: true,
        ..SpecConfig::default()
    };
    let residual = specialize_with_config(&h, 0, &[SpecArg::Dynamic], &cfg).expect("specializes");
    verify_module(&residual).expect("re-verifies");
    let n_load = residual.funcs[0]
        .blocks
        .iter()
        .flat_map(|b| &b.insts)
        .filter(|i| matches!(i, Inst::Load { .. }))
        .count();
    let n_store = residual.funcs[0]
        .blocks
        .iter()
        .flat_map(|b| &b.insts)
        .filter(|i| matches!(i, Inst::Store { .. }))
        .count();
    assert_eq!(n_load, 1, "the heap load survives");
    assert_eq!(n_store, 0, "the operand-stack store is renamed away");

    // ptr = 4096 -> *ptr = 100 -> 7 + 100 = 107.
    assert_eq!(run(&h, &[Value::I64(4096)]), Ok(vec![Value::I64(107)]));
    assert_eq!(
        run(&residual, &[Value::I64(4096)]),
        Ok(vec![Value::I64(107)])
    );
}

// ===========================================================================================
// Cross-function `call`: a direct call (and a `return_call` tail call) is inlined at the call
// site — the callee's CFG is traced in the caller's context, sharing the same abstract memory,
// so the call disappears and the callee's residual is spliced in. The callee must trace as a
// single straight-line path (its internal branches resolve statically; static recursion unrolls);
// a callee whose control flow stays dynamic returns `Unsupported`.
// ===========================================================================================

/// No direct or tail call survives in the residual — every call was inlined.
fn assert_no_calls(residual: &Module) {
    for block in &residual.funcs[0].blocks {
        assert!(
            !block.insts.iter().any(|i| matches!(i, Inst::Call { .. })),
            "residual still contains a direct call — not inlined"
        );
        assert!(
            !matches!(block.term, Terminator::ReturnCall { .. }),
            "residual still contains a tail call — not inlined"
        );
    }
}

fn imul(a: u32, b: u32) -> Inst {
    Inst::IntBin {
        ty: IntTy::I64,
        op: BinOp::Mul,
        a,
        b,
    }
}

#[test]
fn inlines_direct_call_leaf_helper() {
    // main(x) = square(square(x)) = x^4 ; square(a) = a * a. Both calls inline to plain muls.
    let m = Module {
        funcs: vec![
            // 0: main(x) -> i64
            Func {
                params: vec![ValType::I64],
                results: vec![ValType::I64],
                blocks: vec![Block {
                    params: vec![ValType::I64], // 0: x
                    insts: vec![
                        Inst::Call {
                            func: 1,
                            args: vec![0],
                        }, // 1: square(x)
                        Inst::Call {
                            func: 1,
                            args: vec![1],
                        }, // 2: square(square(x))
                    ],
                    term: Terminator::Return(vec![2]),
                }],
            },
            // 1: square(a) -> i64
            Func {
                params: vec![ValType::I64],
                results: vec![ValType::I64],
                blocks: vec![Block {
                    params: vec![ValType::I64], // 0: a
                    insts: vec![imul(0, 0)],    // 1: a*a
                    term: Terminator::Return(vec![1]),
                }],
            },
        ],
        ..Default::default()
    };
    verify_module(&m).expect("verifies");

    let residual = specialize(&m, 0, &[SpecArg::Dynamic]).expect("specializes");
    verify_module(&residual).expect("residual re-verifies");
    assert_no_calls(&residual);

    for x in [0i64, 1, 2, -3, 7, 1000, i64::MIN] {
        let s = x.wrapping_mul(x);
        let expect = s.wrapping_mul(s);
        assert_eq!(run(&m, &[Value::I64(x)]), Ok(vec![Value::I64(expect)]));
        assert_eq!(
            run(&residual, &[Value::I64(x)]),
            Ok(vec![Value::I64(expect)]),
            "residual diverged at x={x}"
        );
        // The inlined residual also JIT-compiles to native.
        let jit = match svm_jit::compile_and_run(&residual, 0, &[x]) {
            Ok(svm_jit::JitOutcome::Returned(v)) => v[0],
            o => panic!("unexpected jit outcome {o:?}"),
        };
        assert_eq!(jit, expect, "jit residual at x={x}");
    }
}

#[test]
fn inlines_static_recursion_unrolled() {
    // main(base) = pow(base, 3) ; pow(b, e) = if e == 0 { 1 } else { b * pow(b, e - 1) }.
    // The exponent is static, so the recursion and its base-case test resolve at every level and
    // the whole thing unrolls into base * base * base * 1 — no call, no branch.
    let m = Module {
        funcs: vec![
            // 0: main(base)
            Func {
                params: vec![ValType::I64],
                results: vec![ValType::I64],
                blocks: vec![Block {
                    params: vec![ValType::I64], // 0: base
                    insts: vec![
                        Inst::ConstI64(3), // 1: e = 3
                        Inst::Call {
                            func: 1,
                            args: vec![0, 1],
                        }, // 2: pow(base, 3)
                    ],
                    term: Terminator::Return(vec![2]),
                }],
            },
            // 1: pow(b, e)
            Func {
                params: vec![ValType::I64, ValType::I64],
                results: vec![ValType::I64],
                blocks: vec![
                    // block 0(b, e): branch on e == 0
                    Block {
                        params: vec![ValType::I64, ValType::I64], // 0: b, 1: e
                        insts: vec![
                            Inst::ConstI64(0), // 2
                            Inst::IntCmp {
                                ty: IntTy::I64,
                                op: CmpOp::Eq,
                                a: 1,
                                b: 2,
                            }, // 3: e == 0
                        ],
                        term: Terminator::BrIf {
                            cond: 3,
                            then_blk: 1,
                            then_args: vec![],
                            else_blk: 2,
                            else_args: vec![0, 1],
                        },
                    },
                    // block 1(): base case -> 1
                    Block {
                        params: vec![],
                        insts: vec![Inst::ConstI64(1)], // 0
                        term: Terminator::Return(vec![0]),
                    },
                    // block 2(b, e): b * pow(b, e - 1)
                    Block {
                        params: vec![ValType::I64, ValType::I64], // 0: b, 1: e
                        insts: vec![
                            Inst::ConstI64(1), // 2
                            Inst::IntBin {
                                ty: IntTy::I64,
                                op: BinOp::Sub,
                                a: 1,
                                b: 2,
                            }, // 3: e - 1
                            Inst::Call {
                                func: 1,
                                args: vec![0, 3],
                            }, // 4: pow(b, e - 1)
                            imul(0, 4),        // 5: b * rec
                        ],
                        term: Terminator::Return(vec![5]),
                    },
                ],
            },
        ],
        ..Default::default()
    };
    verify_module(&m).expect("verifies");

    let residual = specialize(&m, 0, &[SpecArg::Dynamic]).expect("specializes");
    verify_module(&residual).expect("residual re-verifies");
    assert_no_calls(&residual);
    // No data-dependent branch survives: the unrolled body is straight-line.
    assert!(residual.funcs[0]
        .blocks
        .iter()
        .all(|b| matches!(b.term, Terminator::Return(_))));

    for base in [0i64, 1, 2, -3, 5, 11] {
        let expect = base.wrapping_mul(base).wrapping_mul(base);
        assert_eq!(run(&m, &[Value::I64(base)]), Ok(vec![Value::I64(expect)]));
        assert_eq!(
            run(&residual, &[Value::I64(base)]),
            Ok(vec![Value::I64(expect)]),
            "residual diverged at base={base}"
        );
    }
}

#[test]
fn inlines_direct_tail_call() {
    // main(x) = return_call dbl(x) ; dbl(a) = a + a. The tail call becomes a plain return.
    let m = Module {
        funcs: vec![
            Func {
                params: vec![ValType::I64],
                results: vec![ValType::I64],
                blocks: vec![Block {
                    params: vec![ValType::I64], // 0: x
                    insts: vec![],
                    term: Terminator::ReturnCall {
                        func: 1,
                        args: vec![0],
                    },
                }],
            },
            Func {
                params: vec![ValType::I64],
                results: vec![ValType::I64],
                blocks: vec![Block {
                    params: vec![ValType::I64], // 0: a
                    insts: vec![Inst::IntBin {
                        ty: IntTy::I64,
                        op: BinOp::Add,
                        a: 0,
                        b: 0,
                    }], // 1: a + a
                    term: Terminator::Return(vec![1]),
                }],
            },
        ],
        ..Default::default()
    };
    verify_module(&m).expect("verifies");

    let residual = specialize(&m, 0, &[SpecArg::Dynamic]).expect("specializes");
    verify_module(&residual).expect("residual re-verifies");
    assert_no_calls(&residual);

    for x in [0i64, 3, -7, 1000] {
        let expect = x.wrapping_add(x);
        assert_eq!(run(&m, &[Value::I64(x)]), Ok(vec![Value::I64(expect)]));
        assert_eq!(
            run(&residual, &[Value::I64(x)]),
            Ok(vec![Value::I64(expect)])
        );
    }
}

#[test]
fn dynamic_branch_in_callee_inlines_as_cfg() {
    // main(x) = pick(x) ; pick(a) = if a != 0 { a * 2 } else { a + 1 }. With `a` dynamic the
    // callee's branch can't be traced straight-line, so the engine inlines pick's CFG as residual
    // blocks (the data-dependent branch survives). With a *static* argument the branch resolves and
    // the call folds straight-line.
    let m = Module {
        funcs: vec![
            Func {
                params: vec![ValType::I64],
                results: vec![ValType::I64],
                blocks: vec![Block {
                    params: vec![ValType::I64], // 0: x
                    insts: vec![Inst::Call {
                        func: 1,
                        args: vec![0],
                    }], // 1
                    term: Terminator::Return(vec![1]),
                }],
            },
            Func {
                params: vec![ValType::I64],
                results: vec![ValType::I64],
                blocks: vec![
                    Block {
                        params: vec![ValType::I64], // 0: a
                        insts: vec![
                            Inst::ConstI64(0), // 1
                            Inst::IntCmp {
                                ty: IntTy::I64,
                                op: CmpOp::Ne,
                                a: 0,
                                b: 1,
                            }, // 2: a != 0
                        ],
                        term: Terminator::BrIf {
                            cond: 2,
                            then_blk: 1,
                            then_args: vec![0],
                            else_blk: 2,
                            else_args: vec![0],
                        },
                    },
                    Block {
                        params: vec![ValType::I64],                 // 0: a
                        insts: vec![Inst::ConstI64(2), imul(0, 1)], // 1, 2: a * 2
                        term: Terminator::Return(vec![2]),
                    },
                    Block {
                        params: vec![ValType::I64], // 0: a
                        insts: vec![
                            Inst::ConstI64(1), // 1
                            Inst::IntBin {
                                ty: IntTy::I64,
                                op: BinOp::Add,
                                a: 0,
                                b: 1,
                            }, // 2: a + 1
                        ],
                        term: Terminator::Return(vec![2]),
                    },
                ],
            },
        ],
        ..Default::default()
    };
    verify_module(&m).expect("verifies");

    // Dynamic argument: pick's CFG is inlined as residual blocks; the call is gone but the
    // data-dependent branch survives, and the residual matches the interpreter for every input.
    let residual = specialize(&m, 0, &[SpecArg::Dynamic]).expect("inlines callee CFG");
    verify_module(&residual).expect("residual re-verifies");
    assert_no_calls(&residual);
    assert!(
        residual.funcs[0]
            .blocks
            .iter()
            .any(|b| matches!(b.term, Terminator::BrIf { .. })),
        "the callee's data-dependent branch should survive as a residual branch"
    );
    for x in [0i64, 1, 5, -3, 1000] {
        let expect = if x != 0 {
            x.wrapping_mul(2)
        } else {
            x.wrapping_add(1)
        };
        assert_eq!(run(&m, &[Value::I64(x)]), Ok(vec![Value::I64(expect)]));
        assert_eq!(
            run(&residual, &[Value::I64(x)]),
            Ok(vec![Value::I64(expect)]),
            "residual diverged at x={x}"
        );
    }

    // Static argument: a = 5 -> the `a != 0` test resolves true -> 5 * 2 = 10, no branch left.
    let folded = specialize(&m, 0, &[SpecArg::ConstI64(5)]).expect("static arg inlines");
    verify_module(&folded).expect("residual re-verifies");
    assert_no_calls(&folded);
    assert_eq!(run(&folded, &[]), Ok(vec![Value::I64(10)]));
}

// A call-threaded variant of the accumulator interpreter: the per-opcode arithmetic is factored
// into helper functions invoked by `call` from inside the dispatch loop. Specialization must fold
// the dispatch *and* inline the helpers, leaving the bare compiled program.
fn build_call_interpreter(program: &[(u8, i64)]) -> Module {
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

    // entry / header / halt / set are identical to `build_interpreter`.
    let entry = Block {
        params: vec![i64t()],
        insts: vec![Inst::ConstI64(0), Inst::ConstI64(0)], // 1: acc, 2: pc
        term: Terminator::Br {
            target: 1,
            args: vec![1, 2, 0],
        },
    };
    let header = Block {
        params: vec![i64t(), i64t(), i64t()], // 0: acc, 1: pc, 2: input
        insts: vec![
            Inst::ConstI64(0),          // 3: base
            add(3, 1),                  // 4: addr
            load(LoadOp::I32_8U, 4, 0), // 5: op
            load(LoadOp::I64, 4, 1),    // 6: imm
        ],
        term: Terminator::BrTable {
            idx: 5,
            targets: vec![
                (2, vec![0]),          // HALT  -> halt(acc)
                (3, vec![0, 1, 6, 2]), // SETI  -> set
                (4, vec![0, 1, 6, 2]), // ADDI  -> add
                (5, vec![0, 1, 6, 2]), // MULI  -> mul
                (6, vec![0, 1, 6, 2]), // ADDIN -> addin
            ],
            default: (2, vec![0]),
        },
    };
    let halt = Block {
        params: vec![i64t()],
        insts: vec![],
        term: Terminator::Return(vec![0]),
    };
    let set = Block {
        params: vec![i64t(), i64t(), i64t(), i64t()],
        insts: vec![Inst::ConstI64(9), add(1, 4)], // 4, 5: npc
        term: Terminator::Br {
            target: 1,
            args: vec![2, 5, 3],
        },
    };

    // The three arithmetic bodies route the accumulator update through a helper `call`.
    // params: 0: acc, 1: pc, 2: imm, 3: input. The call result (nacc) lands at index 4.
    let call_step = |callee: u32, arg: u32| Block {
        params: vec![i64t(), i64t(), i64t(), i64t()],
        insts: vec![
            Inst::Call {
                func: callee,
                args: vec![0, arg],
            }, // 4: nacc
            Inst::ConstI64(9), // 5
            add(1, 5),         // 6: npc
        ],
        term: Terminator::Br {
            target: 1,
            args: vec![4, 6, 3],
        },
    };
    let add_blk = call_step(1, 2); // ADDI:  f_add(acc, imm)
    let mul_blk = call_step(2, 2); // MULI:  f_mul(acc, imm)
    let addin_blk = call_step(1, 3); // ADDIN: f_add(acc, input)

    let helper = |op| Func {
        params: vec![i64t(), i64t()],
        results: vec![i64t()],
        blocks: vec![Block {
            params: vec![i64t(), i64t()], // 0: a, 1: b
            insts: vec![Inst::IntBin {
                ty: IntTy::I64,
                op,
                a: 0,
                b: 1,
            }], // 2
            term: Terminator::Return(vec![2]),
        }],
    };

    Module {
        funcs: vec![
            Func {
                params: vec![i64t()],
                results: vec![i64t()],
                blocks: vec![entry, header, halt, set, add_blk, mul_blk, addin_blk],
            },
            helper(BinOp::Add), // func 1
            helper(BinOp::Mul), // func 2
        ],
        memory: Some(Memory { size_log2: 16 }),
        data: vec![Data {
            offset: 0,
            readonly: true,
            bytes: encode(program),
        }],
        ..Default::default()
    }
}

#[test]
fn inlines_calls_in_specialized_dispatch_loop() {
    // acc = ((10 + 5) + input) * 3, with every arithmetic step issued as a helper `call`.
    let program = [(SETI, 10), (ADDI, 5), (ADDIN, 0), (MULI, 3), (HALT, 0)];
    let interp = build_call_interpreter(&program);
    verify_module(&interp).expect("interpreter verifies");

    let residual = specialize(&interp, 0, &[SpecArg::Dynamic]).expect("specializes");
    verify_module(&residual).expect("residual re-verifies");
    let opt = optimize_module(&residual);
    verify_module(&opt).expect("optimized residual re-verifies");

    // Both the dispatch table and the helper calls are gone.
    assert_no_dispatch_left(&residual);
    assert_no_calls(&residual);
    // After cleanup the whole program is a single straight-line block: input -> +15 -> *3.
    assert_eq!(opt.funcs[0].blocks.len(), 1);

    for input in [0i64, 1, 2, -5, 100, i64::MIN] {
        let expect = (15i64.wrapping_add(input)).wrapping_mul(3);
        assert_eq!(
            run(&interp, &[Value::I64(input)]),
            Ok(vec![Value::I64(expect)])
        );
        assert_eq!(
            run(&residual, &[Value::I64(input)]),
            Ok(vec![Value::I64(expect)]),
            "residual diverged at input={input}"
        );
        assert_eq!(
            run(&opt, &[Value::I64(input)]),
            Ok(vec![Value::I64(expect)])
        );
    }
}

// ===========================================================================================
// Stage 2 — narrow (i8/i16) renamed cells: sub-word stores/loads into the private region are
// renamed into SSA like full-width ones. A constant cell keeps its raw bytes and re-extends per
// the load op (sign/zero); a *dynamic* narrow access (which would need residual masking to read
// back) and any partial-width overlap stay refused.
// ===========================================================================================

#[test]
fn narrow_constant_cells_round_trip_with_extension() {
    // Store i8/i16/i32/i64 constants into the renamed region and read them back at matching widths
    // with every sign/zero extension. The interpreter is the oracle for the whole matrix.
    let region = (STACK_LO, STACK_LO + 64);
    let (a0, a1, a2, a3) = (
        STACK_LO as i64,
        (STACK_LO + 8) as i64,
        (STACK_LO + 16) as i64,
        (STACK_LO + 24) as i64,
    );
    let st = |op, addr, value| Inst::Store {
        op,
        addr,
        value,
        offset: 0,
        align: 0,
    };
    let ld = |op, addr| Inst::Load {
        op,
        addr,
        offset: 0,
        align: 0,
    };
    let ext = |a| Inst::Convert {
        op: ConvOp::ExtendI32S,
        a,
    };
    let add = |a, b| Inst::IntBin {
        ty: IntTy::I64,
        op: BinOp::Add,
        a,
        b,
    };

    let insts = vec![
        Inst::ConstI64(a0),                    // 0
        Inst::ConstI32(0x1FF),                 // 1: low byte 0xFF
        st(StoreOp::I32_8, 0, 1),              // [a0] = 0xFF
        Inst::ConstI64(a1),                    // 2
        Inst::ConstI32(0x12345),               // 3: low half 0x2345
        st(StoreOp::I32_16, 2, 3),             // [a1] = 0x2345
        Inst::ConstI64(a2),                    // 4
        Inst::ConstI32(-7),                    // 5
        st(StoreOp::I32, 4, 5),                // [a2] = -7
        Inst::ConstI64(a3),                    // 6
        Inst::ConstI64(0x1122_3344_5566_7788), // 7
        st(StoreOp::I64, 6, 7),                // [a3] = big
        ld(LoadOp::I32_8U, 0),                 // 8  -> 0xFF   (i32)
        ld(LoadOp::I32_8S, 0),                 // 9  -> -1     (i32)
        ld(LoadOp::I32_16U, 2),                // 10 -> 0x2345 (i32)
        ld(LoadOp::I64_16S, 2),                // 11 -> 0x2345 (i64)
        ld(LoadOp::I32, 4),                    // 12 -> -7     (i32)
        ld(LoadOp::I64, 6),                    // 13 -> big    (i64)
        ext(8),                                // 14
        ext(9),                                // 15
        ext(10),                               // 16
        ext(12),                               // 17
        add(14, 15),                           // 18
        add(18, 16),                           // 19
        add(19, 17),                           // 20
        add(20, 11),                           // 21
        add(21, 13),                           // 22
    ];
    let m = Module {
        funcs: vec![Func {
            params: vec![],
            results: vec![ValType::I64],
            blocks: vec![Block {
                params: vec![],
                insts,
                term: Terminator::Return(vec![22]),
            }],
        }],
        memory: Some(Memory { size_log2: 16 }),
        ..Default::default()
    };
    verify_module(&m).expect("verifies");

    let residual = specialize_with(&m, 0, &[], Some(region)).expect("specializes");
    verify_module(&residual).expect("residual re-verifies");
    assert_no_memory_ops(&residual); // every narrow cell renamed away
    let opt = optimize_module(&residual);
    verify_module(&opt).expect("optimized residual re-verifies");
    assert_eq!(opt.funcs[0].blocks.len(), 1);

    let expect = run(&m, &[]);
    assert!(expect.is_ok(), "interpreter ran");
    assert_eq!(run(&residual, &[]), expect, "residual matches interp");
    assert_eq!(run(&opt, &[]), expect);
}

#[test]
fn narrow_store_overwrites_overlapping_cell() {
    // A narrow store invalidates the wider cell it overlaps: write i32, then i8 at the same slot,
    // and the i8 load sees the new byte (matching the interpreter's byte-level memory).
    let region = (STACK_LO, STACK_LO + 8);
    let st = |op, addr, value| Inst::Store {
        op,
        addr,
        value,
        offset: 0,
        align: 0,
    };
    let ld = |op, addr| Inst::Load {
        op,
        addr,
        offset: 0,
        align: 0,
    };
    let m = Module {
        funcs: vec![Func {
            params: vec![],
            results: vec![ValType::I32],
            blocks: vec![Block {
                params: vec![],
                insts: vec![
                    Inst::ConstI64(STACK_LO as i64),       // 0
                    Inst::ConstI32(0x4433_2211u32 as i32), // 1
                    st(StoreOp::I32, 0, 1),                // [A] = 0x44332211 (cell A,4)
                    Inst::ConstI32(0xAB),                  // 2
                    st(StoreOp::I32_8, 0, 2),              // [A] = 0xAB  (invalidates A,4 -> A,1)
                    ld(LoadOp::I32_8U, 0),                 // 3 -> 0xAB
                ],
                term: Terminator::Return(vec![3]),
            }],
        }],
        memory: Some(Memory { size_log2: 16 }),
        ..Default::default()
    };
    verify_module(&m).expect("verifies");

    let residual = specialize_with(&m, 0, &[], Some(region)).expect("specializes");
    verify_module(&residual).expect("residual re-verifies");
    assert_no_memory_ops(&residual);
    assert_eq!(run(&residual, &[]), run(&m, &[]));
    assert_eq!(run(&m, &[]), Ok(vec![Value::I32(0xAB)]));
}

#[test]
fn narrow_dynamic_and_overlap_loads_are_unsupported() {
    let region = (STACK_LO, STACK_LO + 8);
    let st = |op, addr, value| Inst::Store {
        op,
        addr,
        value,
        offset: 0,
        align: 0,
    };
    let ld = |op, addr| Inst::Load {
        op,
        addr,
        offset: 0,
        align: 0,
    };

    // A narrow store of a *dynamic* value can't be renamed (reading it back needs residual masking).
    let dyn_narrow_store = Module {
        funcs: vec![Func {
            params: vec![ValType::I64],
            results: vec![ValType::I64],
            blocks: vec![Block {
                params: vec![ValType::I64], // 0: x
                insts: vec![
                    Inst::ConstI64(STACK_LO as i64), // 1
                    st(StoreOp::I64_8, 1, 0),        // [A] = low byte of x (dynamic)
                ],
                term: Terminator::Return(vec![0]),
            }],
        }],
        memory: Some(Memory { size_log2: 16 }),
        ..Default::default()
    };
    verify_module(&dyn_narrow_store).expect("verifies");
    assert_eq!(
        specialize_with(&dyn_narrow_store, 0, &[SpecArg::Dynamic], Some(region)),
        Err(svm_peval::SpecError::Unsupported)
    );

    // A full-width dynamic cell read back at a narrower width is a partial overlap — refused.
    let narrow_load_of_wide_cell = Module {
        funcs: vec![Func {
            params: vec![ValType::I64],
            results: vec![ValType::I64],
            blocks: vec![Block {
                params: vec![ValType::I64], // 0: x
                insts: vec![
                    Inst::ConstI64(STACK_LO as i64), // 1
                    st(StoreOp::I64, 1, 0),          // [A] = x (cell A,8 dynamic)
                    ld(LoadOp::I64_8U, 1),           // 2: low byte -> overlap, can't resolve
                ],
                term: Terminator::Return(vec![2]),
            }],
        }],
        memory: Some(Memory { size_log2: 16 }),
        ..Default::default()
    };
    verify_module(&narrow_load_of_wide_cell).expect("verifies");
    assert_eq!(
        specialize_with(
            &narrow_load_of_wide_cell,
            0,
            &[SpecArg::Dynamic],
            Some(region)
        ),
        Err(svm_peval::SpecError::Unsupported)
    );
}

// ===========================================================================================
// Dynamic-control-flow call inlining: when a callee's branch stays dynamic, its CFG is inlined as
// residual blocks. Caller values live across the call thread through the callee to the continuation;
// a dynamic loop inside the callee survives as a residual loop. The interpreter is the oracle.
// ===========================================================================================

#[test]
fn dynamic_cf_call_threads_caller_values() {
    // main(x) = (x + 100) + pick(x) ; pick(a) = if a != 0 { a * 2 } else { a + 1 }. The pre-call
    // value `x + 100` is live across the dynamic-CF call and must be threaded through pick's inlined
    // CFG to the continuation that adds it.
    let m = Module {
        funcs: vec![
            Func {
                params: vec![ValType::I64],
                results: vec![ValType::I64],
                blocks: vec![Block {
                    params: vec![ValType::I64], // 0: x
                    insts: vec![
                        Inst::ConstI64(100), // 1
                        Inst::IntBin {
                            ty: IntTy::I64,
                            op: BinOp::Add,
                            a: 0,
                            b: 1,
                        }, // 2: pre = x + 100
                        Inst::Call {
                            func: 1,
                            args: vec![0],
                        }, // 3: r = pick(x)
                        Inst::IntBin {
                            ty: IntTy::I64,
                            op: BinOp::Add,
                            a: 2,
                            b: 3,
                        }, // 4: pre + r
                    ],
                    term: Terminator::Return(vec![4]),
                }],
            },
            Func {
                params: vec![ValType::I64],
                results: vec![ValType::I64],
                blocks: vec![
                    Block {
                        params: vec![ValType::I64], // 0: a
                        insts: vec![
                            Inst::ConstI64(0), // 1
                            Inst::IntCmp {
                                ty: IntTy::I64,
                                op: CmpOp::Ne,
                                a: 0,
                                b: 1,
                            }, // 2: a != 0
                        ],
                        term: Terminator::BrIf {
                            cond: 2,
                            then_blk: 1,
                            then_args: vec![0],
                            else_blk: 2,
                            else_args: vec![0],
                        },
                    },
                    Block {
                        params: vec![ValType::I64],                 // 0: a
                        insts: vec![Inst::ConstI64(2), imul(0, 1)], // a * 2
                        term: Terminator::Return(vec![2]),
                    },
                    Block {
                        params: vec![ValType::I64], // 0: a
                        insts: vec![
                            Inst::ConstI64(1),
                            Inst::IntBin {
                                ty: IntTy::I64,
                                op: BinOp::Add,
                                a: 0,
                                b: 1,
                            },
                        ], // a + 1
                        term: Terminator::Return(vec![2]),
                    },
                ],
            },
        ],
        ..Default::default()
    };
    verify_module(&m).expect("verifies");

    let residual = specialize(&m, 0, &[SpecArg::Dynamic]).expect("inlines callee CFG");
    verify_module(&residual).expect("residual re-verifies");
    assert_no_calls(&residual);

    for x in [0i64, 1, 5, -3, 42, -1000] {
        let r = if x != 0 {
            x.wrapping_mul(2)
        } else {
            x.wrapping_add(1)
        };
        let expect = x.wrapping_add(100).wrapping_add(r);
        assert_eq!(run(&m, &[Value::I64(x)]), Ok(vec![Value::I64(expect)]));
        assert_eq!(
            run(&residual, &[Value::I64(x)]),
            Ok(vec![Value::I64(expect)]),
            "residual diverged at x={x}"
        );
        let jit = match svm_jit::compile_and_run(&residual, 0, &[x]) {
            Ok(svm_jit::JitOutcome::Returned(v)) => v[0],
            o => panic!("unexpected jit outcome {o:?}"),
        };
        assert_eq!(jit, expect, "jit residual at x={x}");
    }
}

#[test]
fn dynamic_loop_in_callee_survives_inlined() {
    // main(n) = sum(n) ; sum(n) = { acc = 0; while n != 0 { acc += n; n -= 1 } acc } — a callee with
    // a dynamic loop. Inlining it must reproduce the loop as a residual back-edge (memoization closes
    // it), not diverge.
    let m = Module {
        funcs: vec![
            Func {
                params: vec![ValType::I64],
                results: vec![ValType::I64],
                blocks: vec![Block {
                    params: vec![ValType::I64], // 0: n
                    insts: vec![Inst::Call {
                        func: 1,
                        args: vec![0],
                    }], // 1
                    term: Terminator::Return(vec![1]),
                }],
            },
            Func {
                params: vec![ValType::I64],
                results: vec![ValType::I64],
                blocks: vec![
                    // 0 — entry(n): acc = 0; loop.
                    Block {
                        params: vec![ValType::I64],     // 0: n
                        insts: vec![Inst::ConstI64(0)], // 1: acc
                        term: Terminator::Br {
                            target: 1,
                            args: vec![0, 1],
                        },
                    },
                    // 1 — header(n, acc): if n == 0 done else step.
                    Block {
                        params: vec![ValType::I64, ValType::I64], // 0: n, 1: acc
                        insts: vec![
                            Inst::ConstI64(0), // 2
                            Inst::Eqz {
                                ty: IntTy::I64,
                                a: 0,
                            }, // 3: n == 0
                        ],
                        term: Terminator::BrIf {
                            cond: 3,
                            then_blk: 2,
                            then_args: vec![1],
                            else_blk: 3,
                            else_args: vec![0, 1],
                        },
                    },
                    // 2 — done(acc): return acc.
                    Block {
                        params: vec![ValType::I64],
                        insts: vec![],
                        term: Terminator::Return(vec![0]),
                    },
                    // 3 — step(n, acc): acc += n; n -= 1; loop.
                    Block {
                        params: vec![ValType::I64, ValType::I64], // 0: n, 1: acc
                        insts: vec![
                            Inst::IntBin {
                                ty: IntTy::I64,
                                op: BinOp::Add,
                                a: 1,
                                b: 0,
                            }, // 2: acc + n
                            Inst::ConstI64(1), // 3
                            Inst::IntBin {
                                ty: IntTy::I64,
                                op: BinOp::Sub,
                                a: 0,
                                b: 3,
                            }, // 4: n - 1
                        ],
                        term: Terminator::Br {
                            target: 1,
                            args: vec![4, 2],
                        },
                    },
                ],
            },
        ],
        ..Default::default()
    };
    verify_module(&m).expect("verifies");

    let residual = specialize(&m, 0, &[SpecArg::Dynamic]).expect("inlines callee CFG");
    verify_module(&residual).expect("residual re-verifies");
    assert_no_calls(&residual);
    // The loop survived: some residual block branches back to an earlier one (a back-edge).
    assert!(
        residual.funcs[0].blocks.iter().enumerate().any(|(i, b)| {
            match &b.term {
                Terminator::Br { target, .. } => (*target as usize) <= i,
                Terminator::BrIf {
                    then_blk, else_blk, ..
                } => (*then_blk as usize) <= i || (*else_blk as usize) <= i,
                _ => false,
            }
        }),
        "the callee's dynamic loop should survive as a residual back-edge"
    );

    for n in [0i64, 1, 2, 5, 10, 100] {
        let expect: i64 = (1..=n).sum();
        assert_eq!(run(&m, &[Value::I64(n)]), Ok(vec![Value::I64(expect)]));
        assert_eq!(
            run(&residual, &[Value::I64(n)]),
            Ok(vec![Value::I64(expect)]),
            "residual diverged at n={n}"
        );
    }
}

#[test]
fn nested_dynamic_cf_calls_inline() {
    // main(x) = outer(x) ; outer(a) = if a > 0 { inner(a) + 1 } else { inner(a) - 1 } ;
    // inner(b) = if b != 0 { b * 2 } else { 7 }. Specializing main inlines outer's CFG, and from two
    // different call sites inside it inlines inner's CFG too — a three-deep frame stack.
    let cmp = |op, a, b| Inst::IntCmp {
        ty: IntTy::I64,
        op,
        a,
        b,
    };
    let m = Module {
        funcs: vec![
            // 0: main(x)
            Func {
                params: vec![ValType::I64],
                results: vec![ValType::I64],
                blocks: vec![Block {
                    params: vec![ValType::I64],
                    insts: vec![Inst::Call {
                        func: 1,
                        args: vec![0],
                    }],
                    term: Terminator::Return(vec![1]),
                }],
            },
            // 1: outer(a)
            Func {
                params: vec![ValType::I64],
                results: vec![ValType::I64],
                blocks: vec![
                    Block {
                        params: vec![ValType::I64],                            // 0: a
                        insts: vec![Inst::ConstI64(0), cmp(CmpOp::GtS, 0, 1)], // 1, 2: a > 0
                        term: Terminator::BrIf {
                            cond: 2,
                            then_blk: 1,
                            then_args: vec![0],
                            else_blk: 2,
                            else_args: vec![0],
                        },
                    },
                    Block {
                        params: vec![ValType::I64], // 0: a  (then: inner(a) + 1)
                        insts: vec![
                            Inst::Call {
                                func: 2,
                                args: vec![0],
                            }, // 1
                            Inst::ConstI64(1), // 2
                            Inst::IntBin {
                                ty: IntTy::I64,
                                op: BinOp::Add,
                                a: 1,
                                b: 2,
                            }, // 3
                        ],
                        term: Terminator::Return(vec![3]),
                    },
                    Block {
                        params: vec![ValType::I64], // 0: a  (else: inner(a) - 1)
                        insts: vec![
                            Inst::Call {
                                func: 2,
                                args: vec![0],
                            }, // 1
                            Inst::ConstI64(1), // 2
                            Inst::IntBin {
                                ty: IntTy::I64,
                                op: BinOp::Sub,
                                a: 1,
                                b: 2,
                            }, // 3
                        ],
                        term: Terminator::Return(vec![3]),
                    },
                ],
            },
            // 2: inner(b)
            Func {
                params: vec![ValType::I64],
                results: vec![ValType::I64],
                blocks: vec![
                    Block {
                        params: vec![ValType::I64],                           // 0: b
                        insts: vec![Inst::ConstI64(0), cmp(CmpOp::Ne, 0, 1)], // 1, 2: b != 0
                        term: Terminator::BrIf {
                            cond: 2,
                            then_blk: 1,
                            then_args: vec![0],
                            else_blk: 2,
                            else_args: vec![],
                        },
                    },
                    Block {
                        params: vec![ValType::I64],                 // 0: b
                        insts: vec![Inst::ConstI64(2), imul(0, 1)], // b * 2
                        term: Terminator::Return(vec![2]),
                    },
                    Block {
                        params: vec![],
                        insts: vec![Inst::ConstI64(7)],
                        term: Terminator::Return(vec![0]),
                    },
                ],
            },
        ],
        ..Default::default()
    };
    verify_module(&m).expect("verifies");

    let residual = specialize(&m, 0, &[SpecArg::Dynamic]).expect("inlines nested callee CFGs");
    verify_module(&residual).expect("residual re-verifies");
    assert_no_calls(&residual);

    for x in [0i64, 1, 5, -3, -1, 42] {
        let inner = |b: i64| if b != 0 { b.wrapping_mul(2) } else { 7 };
        let expect = if x > 0 {
            inner(x).wrapping_add(1)
        } else {
            inner(x).wrapping_sub(1)
        };
        assert_eq!(run(&m, &[Value::I64(x)]), Ok(vec![Value::I64(expect)]));
        assert_eq!(
            run(&residual, &[Value::I64(x)]),
            Ok(vec![Value::I64(expect)]),
            "residual diverged at x={x}"
        );
    }
}

// ===========================================================================================
// Scalar float constant folding: f32/f64 arithmetic, compares, fused multiply-add, float↔int
// conversions, and reinterpret/demote/promote casts fold to constants bit-for-bit the interpreter
// (NaN payloads, ±0, ±inf, ties-to-even). Results are reinterpreted to integers before return so the
// differential comparison is exact (NaN-safe). The interpreter is the oracle.
// ===========================================================================================

// Edge-case operand matrices: signed zeros, inf, NaN, subnormals, ties.
const F64S: [f64; 12] = [
    0.0,
    -0.0,
    1.0,
    -1.0,
    2.5,
    -3.5,
    0.1,
    1e308,
    f64::INFINITY,
    f64::NEG_INFINITY,
    f64::NAN,
    4.5, // a tie for round-to-even (nearest -> 4)
];
const F32S: [f32; 12] = [
    0.0,
    -0.0,
    1.0,
    -1.0,
    2.5,
    -3.5,
    0.1,
    1e38,
    f32::INFINITY,
    f32::NEG_INFINITY,
    f32::NAN,
    4.5,
];

fn ret(results: Vec<ValType>, insts: Vec<Inst>, ret_idx: u32) -> Module {
    Module {
        funcs: vec![Func {
            params: vec![],
            results,
            blocks: vec![Block {
                params: vec![],
                insts,
                term: Terminator::Return(vec![ret_idx]),
            }],
        }],
        memory: None,
        ..Default::default()
    }
}

fn assert_no_residual_float(r: &Module) {
    for b in &r.funcs[0].blocks {
        assert!(
            !b.insts.iter().any(|i| matches!(
                i,
                Inst::FBin { .. }
                    | Inst::FUn { .. }
                    | Inst::FCmp { .. }
                    | Inst::Fma { .. }
                    | Inst::FToISat { .. }
                    | Inst::FToITrap { .. }
                    | Inst::IToFConv { .. }
                    | Inst::Cast { .. }
            )),
            "a float op survived folding: {:?}",
            b.insts
        );
    }
}

/// Verify the module, specialize it (no args), check the residual fully folded away its float ops,
/// and assert the residual matches the interpreter exactly (results reinterpreted to ints upstream).
fn check_fold(m: &Module) {
    verify_module(m).expect("verifies");
    let r = specialize(m, 0, &[]).expect("specializes");
    verify_module(&r).expect("residual re-verifies");
    assert_no_residual_float(&r);
    assert_eq!(run(m, &[]), run(&r, &[]), "residual diverged from interp");
}

#[test]
fn folds_f64_binops() {
    for op in FBinOp::ALL {
        for &a in &F64S {
            for &b in &F64S {
                let m = ret(
                    vec![ValType::I64],
                    vec![
                        Inst::ConstF64(a.to_bits()),
                        Inst::ConstF64(b.to_bits()),
                        Inst::FBin {
                            ty: FloatTy::F64,
                            op,
                            a: 0,
                            b: 1,
                        },
                        Inst::Cast {
                            op: CastOp::ReinterpF64I64,
                            a: 2,
                        },
                    ],
                    3,
                );
                check_fold(&m);
            }
        }
    }
}

#[test]
fn folds_f32_binops() {
    for op in FBinOp::ALL {
        for &a in &F32S {
            for &b in &F32S {
                let m = ret(
                    vec![ValType::I32],
                    vec![
                        Inst::ConstF32(a.to_bits()),
                        Inst::ConstF32(b.to_bits()),
                        Inst::FBin {
                            ty: FloatTy::F32,
                            op,
                            a: 0,
                            b: 1,
                        },
                        Inst::Cast {
                            op: CastOp::ReinterpF32I32,
                            a: 2,
                        },
                    ],
                    3,
                );
                check_fold(&m);
            }
        }
    }
}

#[test]
fn folds_float_unops() {
    for op in FUnOp::ALL {
        for &a in &F64S {
            let m = ret(
                vec![ValType::I64],
                vec![
                    Inst::ConstF64(a.to_bits()),
                    Inst::FUn {
                        ty: FloatTy::F64,
                        op,
                        a: 0,
                    },
                    Inst::Cast {
                        op: CastOp::ReinterpF64I64,
                        a: 1,
                    },
                ],
                2,
            );
            check_fold(&m);
        }
        for &a in &F32S {
            let m = ret(
                vec![ValType::I32],
                vec![
                    Inst::ConstF32(a.to_bits()),
                    Inst::FUn {
                        ty: FloatTy::F32,
                        op,
                        a: 0,
                    },
                    Inst::Cast {
                        op: CastOp::ReinterpF32I32,
                        a: 1,
                    },
                ],
                2,
            );
            check_fold(&m);
        }
    }
}

#[test]
fn folds_float_compares() {
    // FCmp result is i32 0/1 directly — no reinterpret needed.
    for op in FCmpOp::ALL {
        for &a in &F64S {
            for &b in &F64S {
                let m = ret(
                    vec![ValType::I32],
                    vec![
                        Inst::ConstF64(a.to_bits()),
                        Inst::ConstF64(b.to_bits()),
                        Inst::FCmp {
                            ty: FloatTy::F64,
                            op,
                            a: 0,
                            b: 1,
                        },
                    ],
                    2,
                );
                check_fold(&m);
            }
        }
    }
}

#[test]
fn folds_fma() {
    for &a in &F64S {
        for &b in &[1.0f64, -2.0, 0.5, f64::NAN, f64::INFINITY] {
            for &c in &[0.0f64, 3.5, -1e9, f64::NAN] {
                let m = ret(
                    vec![ValType::I64],
                    vec![
                        Inst::ConstF64(a.to_bits()),
                        Inst::ConstF64(b.to_bits()),
                        Inst::ConstF64(c.to_bits()),
                        Inst::Fma {
                            ty: FloatTy::F64,
                            a: 0,
                            b: 1,
                            c: 2,
                        },
                        Inst::Cast {
                            op: CastOp::ReinterpF64I64,
                            a: 3,
                        },
                    ],
                    4,
                );
                check_fold(&m);
            }
        }
    }
}

#[test]
fn folds_int_float_conversions() {
    // Saturating float→int (total: NaN -> 0, out-of-range saturates).
    for op in FToI::ALL {
        let (from, to, _) = op.parts();
        let result_ty = match to {
            IntTy::I32 => ValType::I32,
            IntTy::I64 => ValType::I64,
        };
        for &f in &[0.0f64, 3.9, -3.9, 1e30, -1e30, f64::NAN, f64::INFINITY] {
            let konst = match from {
                FloatTy::F32 => Inst::ConstF32((f as f32).to_bits()),
                FloatTy::F64 => Inst::ConstF64(f.to_bits()),
            };
            check_fold(&ret(
                vec![result_ty],
                vec![konst, Inst::FToISat { op, a: 0 }],
                1,
            ));
        }
    }
    // Int→float.
    for op in IToF::ALL {
        for &i in &[0i64, 1, -1, 123_456_789, -987, i64::MAX, i64::MIN] {
            let konst = match op {
                IToF::I32F32S | IToF::I32F32U | IToF::I32F64S | IToF::I32F64U => {
                    Inst::ConstI32(i as i32)
                }
                _ => Inst::ConstI64(i),
            };
            // Result is f32 or f64; reinterpret to its int width for an exact compare.
            let (reinterp, res_ty) = match op {
                IToF::I32F32S | IToF::I32F32U | IToF::I64F32S | IToF::I64F32U => {
                    (CastOp::ReinterpF32I32, ValType::I32)
                }
                _ => (CastOp::ReinterpF64I64, ValType::I64),
            };
            check_fold(&ret(
                vec![res_ty],
                vec![
                    konst,
                    Inst::IToFConv { op, a: 0 },
                    Inst::Cast { op: reinterp, a: 1 },
                ],
                2,
            ));
        }
    }
}

#[test]
fn folds_demote_promote_and_reinterpret() {
    for &f in &F64S {
        // demote f64 -> f32 -> reinterpret to i32.
        check_fold(&ret(
            vec![ValType::I32],
            vec![
                Inst::ConstF64(f.to_bits()),
                Inst::Cast {
                    op: CastOp::Demote,
                    a: 0,
                },
                Inst::Cast {
                    op: CastOp::ReinterpF32I32,
                    a: 1,
                },
            ],
            2,
        ));
    }
    for &f in &F32S {
        // promote f32 -> f64 -> reinterpret to i64.
        check_fold(&ret(
            vec![ValType::I64],
            vec![
                Inst::ConstF32(f.to_bits()),
                Inst::Cast {
                    op: CastOp::Promote,
                    a: 0,
                },
                Inst::Cast {
                    op: CastOp::ReinterpF64I64,
                    a: 1,
                },
            ],
            2,
        ));
    }
    // Integer reinterpret both ways round-trips a bit pattern.
    check_fold(&ret(
        vec![ValType::I32],
        vec![
            Inst::ConstI32(0x3f80_0000u32 as i32), // 1.0f32 bits
            Inst::Cast {
                op: CastOp::ReinterpI32F32,
                a: 0,
            },
            Inst::Cast {
                op: CastOp::ReinterpF32I32,
                a: 1,
            },
        ],
        2,
    ));
}

#[test]
fn trapping_ftoi_folds_in_range_but_preserves_out_of_range_trap() {
    let trap_mod = |a: f64| {
        ret(
            vec![ValType::I32],
            vec![
                Inst::ConstF64(a.to_bits()),
                Inst::FToITrap {
                    op: FToI::F64I32S,
                    a: 0,
                },
            ],
            1,
        )
    };

    // In range: folds to the truncated constant, no trapping op left.
    let m = trap_mod(3.9);
    verify_module(&m).expect("verifies");
    let r = specialize(&m, 0, &[]).expect("specializes");
    verify_module(&r).expect("re-verifies");
    assert_no_residual_float(&r);
    assert_eq!(run(&m, &[]), Ok(vec![Value::I32(3)]));
    assert_eq!(run(&r, &[]), Ok(vec![Value::I32(3)]));

    // Out of range / NaN: NOT folded — the trapping op survives and traps identically.
    for &a in &[1e30f64, -1e30, f64::NAN, f64::INFINITY] {
        let m = trap_mod(a);
        verify_module(&m).expect("verifies");
        let r = specialize(&m, 0, &[]).expect("specializes");
        verify_module(&r).expect("re-verifies");
        assert!(
            r.funcs[0]
                .blocks
                .iter()
                .any(|b| b.insts.iter().any(|i| matches!(i, Inst::FToITrap { .. }))),
            "out-of-range trapping conversion must be preserved, not folded"
        );
        assert_eq!(run(&m, &[]), run(&r, &[]), "trap behavior diverged");
        assert!(run(&r, &[]).is_err(), "should trap at a={a}");
    }
}

#[test]
fn optimizer_folds_float_constants() {
    // The generic optimizer shares the fold helpers: (2.0 * 3.0) + 1.0 = 7.0 collapses to a const.
    let m = ret(
        vec![ValType::I64],
        vec![
            Inst::ConstF64(2.0f64.to_bits()),
            Inst::ConstF64(3.0f64.to_bits()),
            Inst::FBin {
                ty: FloatTy::F64,
                op: FBinOp::Mul,
                a: 0,
                b: 1,
            },
            Inst::ConstF64(1.0f64.to_bits()),
            Inst::FBin {
                ty: FloatTy::F64,
                op: FBinOp::Add,
                a: 2,
                b: 3,
            },
            Inst::Cast {
                op: CastOp::ReinterpF64I64,
                a: 4,
            },
        ],
        5,
    );
    verify_module(&m).expect("verifies");
    let opt = optimize_module(&m);
    verify_module(&opt).expect("optimized re-verifies");
    assert_no_residual_float(&opt);
    assert_eq!(
        run(&opt, &[]),
        Ok(vec![Value::I64((7.0f64).to_bits() as i64)])
    );
    assert_eq!(run(&m, &[]), run(&opt, &[]));
}
