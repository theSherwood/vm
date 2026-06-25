//! The **richer guest-side Futamura `Jit` probe** (PEVAL.md Milestone 3 follow-up). A `no_std`
//! powerbox guest that, entirely in-sandbox:
//!   1. builds a tiny **accumulator-machine interpreter** in svm-IR — `interp(a, b)` loops over a
//!      bytecode program in memory (a `br_table` dispatch over opcodes), folding `b`/immediates into
//!      an accumulator seeded with `a`;
//!   2. specializes it against a **fixed program** supplied as a `SpecConfig` const-overlay (the
//!      "program is constant" caller contract) — so the dispatch loop *unrolls* and the program reads
//!      *fold away*, leaving a straight-line residual `(a, b) -> result`;
//!   3. encodes the residual with `svm-encode` and submits it to the §22 `Jit` capability
//!      (`__vm_jit_compile`);
//!   4. invokes the Cranelift-compiled residual (`__vm_jit_invoke2`) over an input grid and checks it
//!      against a plain-Rust oracle.
//!
//! This is a genuine first Futamura projection performed *in-sandbox*: interpreter + program → a
//! compiled program, with the interpreter's decode/dispatch gone. (The sibling `peval_jit` demo shows
//! the simpler inline + constant-fold path.)
//!
//! The fixed program computes `((a*3) + b) * b + 7`. The residual must satisfy the `Jit.compile` gate:
//! entry `(i64,i64)->(i64)`, **no data segments** (the program rides a const-overlay, not a segment),
//! and `memory.size_log2` equal to the guest's own window (passed as `argv[1]`).
#![no_std]
#![no_main]
extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;
use core::alloc::{GlobalAlloc, Layout};

extern "C" {
    fn malloc(n: usize) -> *mut u8;
    fn free(p: *mut u8);
    fn write(fd: i32, buf: *const u8, n: isize) -> isize;
    fn __vm_jit_compile(blob: *const u8, len: i64) -> i64;
    fn __vm_jit_invoke2(code: i64, a: i64, b: i64) -> i64;
    fn __vm_jit_release(code: i64) -> i64;
}

struct G;
unsafe impl GlobalAlloc for G {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        malloc(l.size())
    }
    unsafe fn dealloc(&self, p: *mut u8, _l: Layout) {
        free(p)
    }
}
#[global_allocator]
static A: G = G;

#[panic_handler]
fn ph(_: &core::panic::PanicInfo) -> ! {
    loop {}
}

fn puts(s: &[u8]) {
    unsafe {
        write(1, s.as_ptr(), s.len() as isize);
    }
}
fn putdec(x: i64) {
    let mut buf = [0u8; 24];
    let mut i = 24usize;
    let neg = x < 0;
    let mut v = if neg {
        (x as i128).unsigned_abs() as u64
    } else {
        x as u64
    };
    if v == 0 {
        i -= 1;
        buf[i] = b'0';
    }
    while v > 0 {
        i -= 1;
        buf[i] = b'0' + (v % 10) as u8;
        v /= 10;
    }
    if neg {
        i -= 1;
        buf[i] = b'-';
    }
    unsafe {
        write(1, buf.as_ptr().add(i), (24 - i) as isize);
    }
}
unsafe fn parse_u8(mut p: *const u8) -> u8 {
    let mut v: u32 = 0;
    while *p != 0 {
        if (*p).is_ascii_digit() {
            v = v.wrapping_mul(10).wrapping_add((*p - b'0') as u32);
        }
        p = p.add(1);
    }
    v as u8
}

use svm_ir::{
    BinOp, Block, ConvOp, Func, Inst, IntTy, LoadOp, Memory, Module, Terminator, ValType,
};
use svm_peval::{SpecArg, SpecConfig};

// The bytecode opcodes (an accumulator machine over inputs `(a, b)`; `acc` starts at `a`).
const OP_ADDB: i64 = 0; // acc += b
const OP_MULB: i64 = 1; // acc *= b
const OP_ADDK: i64 = 2; // acc += k (immediate)
const OP_MULK: i64 = 3; // acc *= k
const OP_END: i64 = 4; // return acc

/// Where the program lives in the window during specialization (a const-overlay promise — the residual
/// never actually reads it, every load folds). Any in-window address works; keep it clear of page 0.
const PROG_BASE: u64 = 0x1_0000;

/// The fixed program: `((a*3) + b) * b + 7`. Each instruction is two little-endian i64 words
/// `[op, k]`, matching the interpreter's 16-byte stride.
fn program() -> &'static [(i64, i64)] {
    &[
        (OP_MULK, 3),
        (OP_ADDB, 0),
        (OP_MULB, 0),
        (OP_ADDK, 7),
        (OP_END, 0),
    ]
}

/// Serialize [`program`] to the const-overlay bytes (op then k, LE i64 each).
fn program_bytes() -> Vec<u8> {
    let mut b = Vec::new();
    for &(op, k) in program() {
        b.extend_from_slice(&op.to_le_bytes());
        b.extend_from_slice(&k.to_le_bytes());
    }
    b
}

/// The plain-Rust oracle — the same computation the residual must produce, wrapping like the IR.
fn oracle(a: i64, b: i64) -> i64 {
    let mut acc = a;
    for &(op, k) in program() {
        match op {
            OP_ADDB => acc = acc.wrapping_add(b),
            OP_MULB => acc = acc.wrapping_mul(b),
            OP_ADDK => acc = acc.wrapping_add(k),
            OP_MULK => acc = acc.wrapping_mul(k),
            _ => break,
        }
    }
    acc
}

/// Build the **generic interpreter** `interp(a, b) -> i64`: a `br_table`-dispatched accumulator loop
/// over the program at [`PROG_BASE`]. Blocks: 0 entry, 1 loop, 2 addb, 3 mulb, 4 addk, 5 mulk, 6 end.
fn build_interpreter(win_log2: u8) -> Module {
    let i64t = ValType::I64;
    let add = |a, b| Inst::IntBin {
        ty: IntTy::I64,
        op: BinOp::Add,
        a,
        b,
    };
    let mul = |a, b| Inst::IntBin {
        ty: IntTy::I64,
        op: BinOp::Mul,
        a,
        b,
    };

    // Block 0 — entry(a=v0, b=v1): seed acc=a, i=0, jump to the loop.
    let entry = Block {
        params: vec![i64t, i64t],
        insts: vec![Inst::ConstI64(0)], // v2 = 0 (the initial program counter)
        term: Terminator::Br {
            target: 1,
            args: vec![0, 1, 2], // loop(acc=a, b, i=0)
        },
    };

    // Block 1 — loop(acc=v0, b=v1, i=v2): decode the instruction at PROG_BASE + i*16 and dispatch.
    let loop_blk = Block {
        params: vec![i64t, i64t, i64t],
        insts: vec![
            Inst::ConstI64(PROG_BASE as i64), // v3
            Inst::ConstI64(16),               // v4
            mul(2, 4),                        // v5 = i*16
            add(3, 5),                        // v6 = PROG_BASE + i*16
            Inst::Load {
                op: LoadOp::I64,
                addr: 6,
                offset: 0,
                align: 0,
            }, // v7 = op (i64)
            Inst::Load {
                op: LoadOp::I64,
                addr: 6,
                offset: 8,
                align: 0,
            }, // v8 = k (i64)
            Inst::Convert {
                op: ConvOp::WrapI64,
                a: 7,
            }, // v9 = op as i32 (br_table index)
            Inst::ConstI64(1),                // v10
            add(2, 10),                       // v11 = i + 1
        ],
        term: Terminator::BrTable {
            idx: 9,
            // targets[op]: 0=addb, 1=mulb, 2=addk, 3=mulk. END (4) is out of range -> default.
            targets: vec![
                (2, vec![0, 1, 11]),    // addb(acc, b, i+1)
                (3, vec![0, 1, 11]),    // mulb(acc, b, i+1)
                (4, vec![0, 1, 11, 8]), // addk(acc, b, i+1, k)
                (5, vec![0, 1, 11, 8]), // mulk(acc, b, i+1, k)
            ],
            default: (6, vec![0]), // end(acc)
        },
    };

    // Blocks 2..5 — the ops. Each folds into acc, then loops back with the advanced counter.
    let addb = Block {
        params: vec![i64t, i64t, i64t],
        insts: vec![add(0, 1)], // v3 = acc + b
        term: Terminator::Br {
            target: 1,
            args: vec![3, 1, 2],
        },
    };
    let mulb = Block {
        params: vec![i64t, i64t, i64t],
        insts: vec![mul(0, 1)], // v3 = acc * b
        term: Terminator::Br {
            target: 1,
            args: vec![3, 1, 2],
        },
    };
    let addk = Block {
        params: vec![i64t, i64t, i64t, i64t],
        insts: vec![add(0, 3)], // v4 = acc + k
        term: Terminator::Br {
            target: 1,
            args: vec![4, 1, 2],
        },
    };
    let mulk = Block {
        params: vec![i64t, i64t, i64t, i64t],
        insts: vec![mul(0, 3)], // v4 = acc * k
        term: Terminator::Br {
            target: 1,
            args: vec![4, 1, 2],
        },
    };

    // Block 6 — end(acc=v0): return the accumulator.
    let end = Block {
        params: vec![i64t],
        insts: vec![],
        term: Terminator::Return(vec![0]),
    };

    let interp = Func {
        params: vec![i64t, i64t],
        results: vec![i64t],
        blocks: vec![entry, loop_blk, addb, mulb, addk, mulk, end],
    };
    Module {
        funcs: vec![interp],
        memory: Some(Memory {
            size_log2: win_log2,
        }),
        data: Vec::new(),
        imports: Vec::new(),
        exports: Vec::new(),
        debug_info: None,
    }
}

#[no_mangle]
pub extern "C" fn main(argc: i32, argv: *const *const u8) -> i32 {
    let win_log2 = if argc >= 2 {
        unsafe { parse_u8(*argv.add(1)) }
    } else {
        21
    };

    // 1–2. Build the interpreter and specialize it against the fixed program (a const-overlay), with
    // both inputs dynamic. The dispatch loop unrolls; the program reads fold; the residual is a
    // straight-line `(a, b) -> result`.
    let m = build_interpreter(win_log2);
    let config = SpecConfig {
        const_overlays: vec![(PROG_BASE, program_bytes())],
        ..SpecConfig::default()
    };
    let residual = match svm_peval::specialize_with_config(
        &m,
        0,
        &[SpecArg::Dynamic, SpecArg::Dynamic],
        &config,
    ) {
        Ok(r) => r,
        Err(_) => {
            puts(b"specialize failed\n");
            return 1;
        }
    };
    // Specialization unrolled the dispatch loop and folded every program read: the residual is a
    // straight-line `br`-chain with **no `br_table` (dispatch) and no `Load` (decode)** left — the
    // interpreter is gone, only the compiled program remains. (We do *not* run the generic optimizer
    // in-sandbox to merge the chain into one block: it would pull `String::clone` into the closure,
    // whose body lives in the precompiled `alloc` rlib, not our emitted bitcode. Block-count
    // minimization is a host-side concern; the fold itself is what matters.)
    let mut dispatch_left = false;
    let mut decode_left = false;
    for f in &residual.funcs {
        for b in &f.blocks {
            if matches!(b.term, Terminator::BrTable { .. }) {
                dispatch_left = true;
            }
            if b.insts.iter().any(|i| matches!(i, Inst::Load { .. })) {
                decode_left = true;
            }
        }
    }
    let blocks: usize = residual.funcs.iter().map(|f| f.blocks.len()).sum();
    puts(b"residual blocks: ");
    putdec(blocks as i64);
    if dispatch_left || decode_left {
        puts(b" (dispatch/decode NOT folded)\n");
    } else {
        puts(b" (dispatch + decode folded away)\n");
    }

    // 3. Encode + submit to the Jit capability.
    let blob = svm_encode::encode_module(&residual);
    puts(b"emitted ");
    putdec(blob.len() as i64);
    puts(b" bytes of residual IR\n");
    let code = unsafe { __vm_jit_compile(blob.as_ptr(), blob.len() as i64) };
    if code < 0 {
        puts(b"jit compile failed: ");
        putdec(code);
        puts(b"\n");
        return 1;
    }

    // 4. Invoke over a grid and check against the oracle.
    let mut bad: i64 = 0;
    let mut a = -4i64;
    while a <= 4 {
        let mut b = -4i64;
        while b <= 4 {
            if unsafe { __vm_jit_invoke2(code, a, b) } != oracle(a, b) {
                bad += 1;
            }
            b += 1;
        }
        a += 1;
    }
    puts(b"jit(5, 9) = ");
    putdec(unsafe { __vm_jit_invoke2(code, 5, 9) });
    puts(b" (oracle ");
    putdec(oracle(5, 9));
    puts(b")\n");
    unsafe {
        __vm_jit_release(code);
    }

    if bad != 0 {
        puts(b"MISMATCHES: ");
        putdec(bad);
        puts(b"\n");
        return 1;
    }
    puts(b"81 inputs agree: interpreter specialized to its program, JITed in-sandbox\n");
    0
}
