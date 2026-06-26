//! The guest-side Futamura `Jit` probe (DESIGN.md §20c capstone). A `no_std`/`panic=abort`
//! powerbox program that, **entirely in-sandbox**:
//!   1. builds a small two-function module (`entry(a,b)` calls `helper(a,b) = a*3 + b*5 + 7`),
//!   2. specializes it with `svm-peval` (inlining the call, folding the constants into one function),
//!   3. serializes the residual with `svm-encode` (the binary module blob),
//!   4. submits the blob to the §22 `Jit` capability (`__vm_jit_compile`),
//!   5. invokes the Cranelift-compiled residual (`__vm_jit_invoke2`) over a grid of inputs and
//!      checks each against a plain-Rust oracle.
//!
//! It prints the emitted blob size, a sample `jit(a,b)` vs oracle, and the mismatch count. The
//! `peval_jit.rs` test compiles this to svm-IR and runs it under `run_powerbox` (which grants the
//! `Jit` cap), asserting `0` mismatches.
//!
//! The residual must satisfy the `Jit.compile` gate: entry `(i64,i64)->(i64)`, **no data segments**,
//! and `memory.size_log2` **exactly** the guest's own window. The window size is layout-dependent, so
//! the host passes it as `argv[1]`; the guest builds the module's memory descriptor with it.
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
    // §22 Jit capability builtins (lowered by bare name by the on-ramp; `extern "C"` is unmangled).
    fn __vm_jit_compile(blob: *const u8, len: i64) -> i64;
    fn __vm_jit_invoke2(code: i64, a: i64, b: i64) -> i64;
    fn __vm_jit_release(code: i64) -> i64;
}

struct Guest;
unsafe impl GlobalAlloc for Guest {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        malloc(l.size())
    }
    unsafe fn dealloc(&self, p: *mut u8, _l: Layout) {
        free(p)
    }
}
#[global_allocator]
static A: Guest = Guest;

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
    // Negate into u64 space so i64::MIN is representable.
    let mut v = if neg { (x as i128).unsigned_abs() as u64 } else { x as u64 };
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

/// Parse a NUL-terminated decimal C string into a `u8` (the window `size_log2` from `argv[1]`).
unsafe fn parse_u8(mut p: *const u8) -> u8 {
    let mut v: u32 = 0;
    while *p != 0 {
        let c = *p;
        if c.is_ascii_digit() {
            v = v.wrapping_mul(10).wrapping_add((c - b'0') as u32);
        }
        p = p.add(1);
    }
    v as u8
}

use svm_ir::{BinOp, Block, Func, Inst, IntTy, Memory, Module, Terminator, ValType};

/// The program both the guest specializes and the oracle mirrors: `helper(a,b) = a*3 + b*5 + 7`,
/// reached through a direct `call` from `entry` (so specialization inlines it into one function and
/// folds the `3`/`5`/`7` in). `win_log2` is the guest's window size, so the residual's memory
/// descriptor matches the `Jit.compile` precondition.
fn build_module(win_log2: u8) -> Module {
    let i64x2 = || vec![ValType::I64, ValType::I64];
    // func0 — entry(a,b): call helper(a,b); return.
    let entry = Func {
        params: i64x2(),
        results: vec![ValType::I64],
        blocks: vec![Block {
            params: i64x2(),
            insts: vec![Inst::Call {
                func: 1,
                args: vec![0, 1],
            }],
            term: Terminator::Return(vec![2]),
        }],
    };
    // func1 — helper(a,b) = a*3 + b*5 + 7.
    let mul = |a, b| Inst::IntBin {
        ty: IntTy::I64,
        op: BinOp::Mul,
        a,
        b,
    };
    let add = |a, b| Inst::IntBin {
        ty: IntTy::I64,
        op: BinOp::Add,
        a,
        b,
    };
    let helper = Func {
        params: i64x2(),
        results: vec![ValType::I64],
        blocks: vec![Block {
            params: i64x2(),
            insts: vec![
                Inst::ConstI64(3), // v2
                mul(0, 2),         // v3 = a*3
                Inst::ConstI64(5), // v4
                mul(1, 4),         // v5 = b*5
                add(3, 5),         // v6 = a*3 + b*5
                Inst::ConstI64(7), // v7
                add(6, 7),         // v8 = a*3 + b*5 + 7
            ],
            term: Terminator::Return(vec![8]),
        }],
    };
    Module {
        funcs: vec![entry, helper],
        memory: Some(Memory {
            size_log2: win_log2,
        }),
        data: Vec::new(),
        imports: Vec::new(),
        exports: Vec::new(),
        debug_info: None,
    }
}

/// The plain-Rust oracle — the same arithmetic the residual must compute, wrapping like the IR.
fn oracle(a: i64, b: i64) -> i64 {
    a.wrapping_mul(3)
        .wrapping_add(b.wrapping_mul(5))
        .wrapping_add(7)
}

#[no_mangle]
pub extern "C" fn main(argc: i32, argv: *const *const u8) -> i32 {
    // The window size_log2 the host (which translated us) passes as argv[1]; default to the powerbox
    // minimum (21) if absent.
    let win_log2 = if argc >= 2 {
        unsafe { parse_u8(*argv.add(1)) }
    } else {
        21
    };

    // 1–2. Build + specialize (both inputs dynamic → residual is one (i64,i64)->i64 function).
    let m = build_module(win_log2);
    let residual = match svm_peval::specialize(&m, 0, &[svm_peval::SpecArg::Dynamic, svm_peval::SpecArg::Dynamic]) {
        Ok(r) => r,
        Err(_) => {
            puts(b"specialize failed\n");
            return 1;
        }
    };

    // 3. Serialize the residual to the binary module blob.
    let blob = svm_encode::encode_module(&residual);
    puts(b"emitted ");
    putdec(blob.len() as i64);
    puts(b" bytes of residual IR\n");

    // 4. Submit to the Jit capability.
    let code = unsafe { __vm_jit_compile(blob.as_ptr(), blob.len() as i64) };
    if code < 0 {
        puts(b"jit compile failed: ");
        putdec(code);
        puts(b"\n");
        return 1;
    }

    // 5. Invoke over a grid and check against the oracle.
    let mut bad: i64 = 0;
    let mut a = -4i64;
    while a <= 4 {
        let mut b = -4i64;
        while b <= 4 {
            let got = unsafe { __vm_jit_invoke2(code, a, b) };
            if got != oracle(a, b) {
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
    puts(b"81 inputs agree: guest-specialized, guest-encoded, host-verified, Cranelift-compiled\n");
    0
}
