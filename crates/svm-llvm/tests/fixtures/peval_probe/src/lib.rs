//! The in-sandbox `svm-peval` probe (DESIGN.md §20c). A `no_std`/`panic=abort` powerbox program
//! that builds a small module, calls `svm_peval::specialize`, and prints a summary of the residual
//! (`funcs`, total `blocks`, total `insts`) — one decimal per line. The `peval_in_sandbox.rs` test
//! compiles this to svm-IR and runs it, asserting the printed summary equals the **same** specialization
//! run host-side (a differential: in-sandbox specializer == host specializer).
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

fn putdec(mut x: u64) {
    let mut buf = [0u8; 24];
    let mut i = 24usize;
    if x == 0 {
        i -= 1;
        buf[i] = b'0';
    }
    while x > 0 {
        i -= 1;
        buf[i] = b'0' + (x % 10) as u8;
        x /= 10;
    }
    unsafe {
        write(1, buf.as_ptr().add(i), (24 - i) as isize);
        write(1, b"\n".as_ptr(), 1);
    }
}

use svm_ir::{Block, Func, IntTy, BinOp, Inst, Module, Terminator, ValType};

/// The module both this guest and the host oracle build: a single `() -> i32` whose body is the
/// constant product `21 * 2` — foldable, so a *correct* specializer collapses it (proving the
/// in-sandbox engine actually runs its folding logic, not just returns the input unchanged).
pub fn build_module() -> Module {
    let f = Func {
        params: vec![],
        results: vec![ValType::I32],
        blocks: vec![Block {
            params: vec![],
            insts: vec![
                Inst::ConstI32(21),
                Inst::ConstI32(2),
                Inst::IntBin { ty: IntTy::I32, op: BinOp::Mul, a: 0, b: 1 },
            ],
            term: Terminator::Return(vec![2]),
        }],
    };
    Module {
        funcs: vec![f],
        memory: None,
        data: Vec::new(),
        imports: Vec::new(),
        exports: Vec::new(),
        impl_exports: Vec::new(),
        debug_info: None,
    }
}

/// `(funcs, blocks, insts)` of a module — the summary printed/compared by the differential.
pub fn summarize(m: &Module) -> (u64, u64, u64) {
    let funcs = m.funcs.len() as u64;
    let blocks: u64 = m.funcs.iter().map(|f| f.blocks.len() as u64).sum();
    let insts: u64 = m
        .funcs
        .iter()
        .flat_map(|f| f.blocks.iter())
        .map(|b| b.insts.len() as u64)
        .sum();
    (funcs, blocks, insts)
}

#[no_mangle]
pub extern "C" fn main() -> i32 {
    let m = build_module();
    match svm_peval::specialize(&m, 0, &[]) {
        Ok(res) => {
            let (f, b, i) = summarize(&res);
            putdec(f);
            putdec(b);
            putdec(i);
            0
        }
        // Encode the error so a failure is visible rather than silent.
        Err(_) => {
            putdec(9_999_999);
            1
        }
    }
}
