// Shared no_std prelude prepended to every workload by the `rustbench` driver. It gives real heap
// data structures (Vec/BTreeMap/…) with a bump `#[global_allocator]` and ZERO libc surface — the
// property that lets a real Rust program compile cleanly to svm-jit (LP64 bitcode), Wasmtime
// (wasm32/wasm64), and native, with no shim assembly. Each workload exports `run(n) -> i64` and
// calls `reset_arena()` first so the bump allocator is fresh per call (runs are timed many times).
#![no_std]
#![allow(unused_imports, static_mut_refs, internal_features, dead_code)]
extern crate alloc;
use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use core::alloc::{GlobalAlloc, Layout};

// 32 MiB arena — bounds the svm guest window (larger risks "window too large" for the JIT) and is
// ample for these workloads, which reset it every `run`.
const ARENA: usize = 32 * 1024 * 1024;
static mut HEAP: [u8; ARENA] = [0; ARENA];
static mut OFF: usize = 0;

struct Bump;
unsafe impl GlobalAlloc for Bump {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let a = l.align();
        let s = (OFF + a - 1) & !(a - 1);
        if s + l.size() > ARENA {
            return core::ptr::null_mut();
        }
        OFF = s + l.size();
        (core::ptr::addr_of_mut!(HEAP) as *mut u8).add(s)
    }
    unsafe fn dealloc(&self, _: *mut u8, _: Layout) {}
}
#[global_allocator]
static GA: Bump = Bump;

#[panic_handler]
fn ph(_: &core::panic::PanicInfo) -> ! {
    loop {}
}

// The native-lane staticlib links precompiled `alloc` (built with unwind), which references the
// personality even under `panic=abort`; it is never called, so a stub satisfies the linker. Harmless
// (unused) on the svm/wasm lanes.
#[no_mangle]
pub extern "C" fn rust_eh_personality() {}

#[inline(always)]
fn reset_arena() {
    unsafe {
        OFF = 0;
    }
}

// A small deterministic PRNG (xorshift64) so every lane computes the identical checksum.
#[inline(always)]
fn xs(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}
