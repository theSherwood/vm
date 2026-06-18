//! Ahead-of-time pipeline proof. `svm-peval` is a pure host-side `Module -> Module` transform, so
//! it is usable entirely as a build step: specialize an interpreter against a fixed program,
//! re-verify, serialize the residual to bytes (the shippable artifact), then later load it back
//! and run or JIT-compile it — with zero specialization cost at run time. This test exercises that
//! whole chain and asserts every stage agrees with the source interpreter:
//!
//!   specialize -> verify -> encode_module -> decode_module -> verify
//!              -> {interpret loaded residual, JIT-compile loaded residual}

use svm_encode::{decode_module, encode_module};
use svm_interp::Value;
use svm_ir::{BinOp, Block, Data, Func, Inst, IntTy, LoadOp, Memory, Module, Terminator, ValType};
use svm_jit::JitOutcome;
use svm_peval::{specialize, SpecArg};
use svm_verify::verify_module;

// A minimal accumulator bytecode (9 bytes/instruction: opcode + little-endian i64 immediate),
// the same shape as the Stage-1 demo: acc starts 0; HALT returns it.
const HALT: u8 = 0;
const SETI: u8 = 1; // acc = imm
const ADDI: u8 = 2; // acc += imm
const ADDIN: u8 = 3; // acc += input
const MULI: u8 = 4; // acc *= imm

fn encode_program(program: &[(u8, i64)]) -> Vec<u8> {
    let mut bytes = Vec::new();
    for &(op, imm) in program {
        bytes.push(op);
        bytes.extend_from_slice(&imm.to_le_bytes());
    }
    bytes
}

/// `interp(input: i64) -> i64` with the program in a readonly data segment and a `br_table`
/// dispatch loop — a real interpreter for the specializer to compile away.
fn build_interpreter(program: &[(u8, i64)]) -> Module {
    let i64t = || ValType::I64;
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

    let entry = Block {
        params: vec![i64t()],                              // 0: input
        insts: vec![Inst::ConstI64(0), Inst::ConstI64(0)], // 1: acc, 2: pc
        term: Terminator::Br {
            target: 1,
            args: vec![1, 2, 0],
        },
    };
    let header = Block {
        params: vec![i64t(), i64t(), i64t()], // acc, pc, input
        insts: vec![
            Inst::ConstI64(0),          // 3: base
            add(3, 1),                  // 4: addr = base + pc
            load(LoadOp::I32_8U, 4, 0), // 5: op
            load(LoadOp::I64, 4, 1),    // 6: imm
        ],
        term: Terminator::BrTable {
            idx: 5,
            targets: vec![
                (2, vec![0]),          // HALT  -> halt(acc)
                (3, vec![0, 1, 6, 2]), // SETI
                (4, vec![0, 1, 6, 2]), // ADDI
                (5, vec![0, 1, 6, 2]), // ADDIN
                (6, vec![0, 1, 6, 2]), // MULI
            ],
            default: (2, vec![0]),
        },
    };
    let halt = Block {
        params: vec![i64t()],
        insts: vec![],
        term: Terminator::Return(vec![0]),
    };
    // pc += 9 then loop; new accumulator is at `nacc_idx`.
    let step = |acc_update: Vec<Inst>, nacc_idx: u32| Block {
        params: vec![i64t(), i64t(), i64t(), i64t()], // acc, pc, imm, input
        insts: {
            let mut v = acc_update;
            v.push(Inst::ConstI64(9));
            v.push(add(1, nacc_idx + 1)); // npc = pc + 9
            v
        },
        term: Terminator::Br {
            target: 1,
            args: vec![nacc_idx, nacc_idx + 2, 3],
        },
    };
    let seti = Block {
        params: vec![i64t(), i64t(), i64t(), i64t()],
        insts: vec![Inst::ConstI64(9), add(1, 4)],
        term: Terminator::Br {
            target: 1,
            args: vec![2, 5, 3], // header(imm, npc, input)
        },
    };
    let addi = step(vec![add(0, 2)], 4); // acc + imm
    let addin = step(vec![add(0, 3)], 4); // acc + input
    let muli = step(
        vec![Inst::IntBin {
            ty: IntTy::I64,
            op: BinOp::Mul,
            a: 0,
            b: 2,
        }],
        4,
    );

    Module {
        funcs: vec![Func {
            params: vec![i64t()],
            results: vec![i64t()],
            blocks: vec![entry, header, halt, seti, addi, addin, muli],
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

fn interp_run(m: &Module, input: i64) -> i64 {
    let mut fuel = 10_000_000u64;
    match svm_interp::run(m, 0, &[Value::I64(input)], &mut fuel) {
        Ok(vals) => match vals.as_slice() {
            [Value::I64(v)] => *v,
            other => panic!("unexpected interpreter result: {other:?}"),
        },
        Err(t) => panic!("interpreter trapped: {t:?}"),
    }
}

fn jit_run(m: &Module, input: i64) -> i64 {
    match svm_jit::compile_and_run(m, 0, &[input]) {
        Ok(JitOutcome::Returned(vals)) => match vals.as_slice() {
            [v] => *v,
            other => panic!("unexpected jit result: {other:?}"),
        },
        other => panic!("unexpected jit outcome: {other:?}"),
    }
}

#[test]
fn aot_specialize_serialize_and_jit_roundtrip() {
    // acc = ((10 + 5) + input) * 3
    let program = [(SETI, 10), (ADDI, 5), (ADDIN, 0), (MULI, 3), (HALT, 0)];
    let interp = build_interpreter(&program);
    verify_module(&interp).expect("interpreter verifies");

    // --- the AOT build step (host-side, no runtime) ---
    let residual = specialize(&interp, 0, &[SpecArg::Dynamic]).expect("specializes");
    verify_module(&residual).expect("residual verifies");

    // Serialize the residual to a shippable artifact, then load it back.
    let artifact: Vec<u8> = encode_module(&residual);
    let loaded = decode_module(&artifact).expect("artifact decodes");
    assert_eq!(loaded, residual, "encode/decode round trip is byte-perfect");
    verify_module(&loaded).expect("loaded artifact re-verifies");

    // --- run time: the loaded residual matches the interpreter on both backends ---
    for input in [0i64, 1, 2, -5, 100, 12345, i64::MIN] {
        let expect = (15i64.wrapping_add(input)).wrapping_mul(3);
        assert_eq!(
            interp_run(&interp, input),
            expect,
            "source interp, input {input}"
        );
        assert_eq!(
            interp_run(&loaded, input),
            expect,
            "loaded residual (interpreter), input {input}"
        );
        assert_eq!(
            jit_run(&loaded, input),
            expect,
            "loaded residual (JIT), input {input}"
        );
    }
}
