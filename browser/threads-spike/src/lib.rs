//! Cross-thread shared-memory atomics spike (no_std, no deps).
//!
//! All instances (the main thread + each worker) import the **same** shared linear memory, so a
//! 32-bit counter at a fixed high address `COUNTER` is one shared cell. The exports do `n`
//! increments of it two ways:
//!   * [`add_atomic`] — `AtomicI32::fetch_add` (a real wasm `i32.atomic.rmw.add`), and
//!   * [`add_plain`]  — a non-atomic read-modify-write,
//! so two workers running concurrently must yield exactly `2n` via the atomic path and *fewer* via
//! the racy plain path — proving the atomics are genuine hardware atomics over contended memory, not
//! an artifact of the build.
//!
//! The functions are deliberately tiny (loop over locals only) so they touch no shadow stack — the
//! per-instance `__stack_pointer` globals all start at the same address over the shared memory, so a
//! function that spilled locals to that stack would have two workers clobber each other. Keeping
//! them register/local-only sidesteps per-thread stack setup, which a real runtime would provide.

#![no_std]

use core::panic::PanicInfo;
use core::sync::atomic::{AtomicI32, Ordering};

#[panic_handler]
fn panic(_: &PanicInfo) -> ! {
    loop {}
}

/// A fixed linear-memory address (8 MiB) for the shared counter — well above this tiny module's
/// static data, so re-running data init in a worker never touches it. The host zeroes it once.
const COUNTER: usize = 8 * 1024 * 1024;

/// The counter's byte address (so the host can `Atomics`/`Int32Array` it directly too).
#[no_mangle]
pub extern "C" fn counter_addr() -> i32 {
    COUNTER as i32
}

fn cell(addr: i32) -> &'static AtomicI32 {
    // SAFETY (spike): `addr` is a host-provided, naturally-aligned address inside the shared memory;
    // an `AtomicI32` over it is the whole point — many threads touch this one cell concurrently.
    unsafe { &*(addr as usize as *const AtomicI32) }
}

/// `n` **atomic** increments of the cell at `addr` (real `i32.atomic.rmw.add`).
#[no_mangle]
pub extern "C" fn add_atomic(addr: i32, n: i32) {
    let c = cell(addr);
    let mut i = 0;
    while i < n {
        c.fetch_add(1, Ordering::SeqCst);
        i += 1;
    }
}

/// `n` **non-atomic** increments (plain read-modify-write) — racy under contention, to contrast.
#[no_mangle]
pub extern "C" fn add_plain(addr: i32, n: i32) {
    let p = addr as usize as *mut i32;
    let mut i = 0;
    while i < n {
        // SAFETY (spike): volatile RMW on the shared cell — intentionally *not* atomic.
        unsafe { p.write_volatile(p.read_volatile().wrapping_add(1)) };
        i += 1;
    }
}

/// Read the cell at `addr` (atomic load).
#[no_mangle]
pub extern "C" fn load(addr: i32) -> i32 {
    cell(addr).load(Ordering::SeqCst)
}

/// Store `v` to the cell at `addr` (atomic store) — the host zeroes the counter before a run.
#[no_mangle]
pub extern "C" fn store(addr: i32, v: i32) {
    cell(addr).store(v, Ordering::SeqCst);
}
