//! Native **stack switching** — the substrate for JIT green threads and fibers (§3d/§6/§12).
//!
//! The interpreter's continuation is data (a `Vec<Frame>`), so it can park and migrate between
//! workers in safe Rust. JITted code's continuation is a *native machine stack*; to suspend it you
//! **switch stacks** — save the callee-saved registers + stack pointer, load another context's — not
//! reify frames. That is exactly what §3d's two-stack model calls for: every fiber/green-thread owns
//! an out-of-band **control stack** the runtime allocates, and a switch is `~ns` (no syscall).
//!
//! This crate is the single home for that `unsafe`, kept tiny and auditable (the way `svm-mem` is for
//! memory). It is split into two seams so a port only touches one of them:
//!
//! - `switch` — the arch-specific register/stack swap (`jump`/`make`/`Transfer`): `boost.context`
//!   `fcontext`-style, one file per ABI (`switch_x86_64_sysv.rs`, `switch_aarch64.rs`,
//!   `switch_x86_64_windows.rs`).
//! - `stack` — the OS-specific guard-paged control stack (`stack_unix.rs` = `mmap`+`PROT_NONE`;
//!   `stack_windows.rs` = `VirtualAlloc`+`PAGE_NOACCESS`).
//! - [`Fiber`] / [`Yielder`] — an ABI-agnostic safe RAII *asymmetric coroutine* over those two: resume
//!   a fiber with a value, the body runs and [`Yielder::suspend`]s values back, RAII frees the stack,
//!   and a panic inside the body aborts (unwinding across a stack switch would be UB).
//!
//! Supported on **x86-64 unix** and **x86-64 Windows** today; other targets compile but
//! [`supported`] returns `false` and the primitives are absent (the JIT keeps bailing `Unsupported`
//! there). The aarch64 (macOS) port is staged behind the same `switch`/`stack` seams.

/// Whether real stack switching is available on this target.
///
/// Keep this in lockstep with the `switch`/`stack`/`fiber` module gates below and with the
/// `fiber_rt` cfg that `svm-jit` derives for the same target set.
pub const fn supported() -> bool {
    cfg!(any(
        all(unix, target_arch = "x86_64"),
        all(windows, target_arch = "x86_64"),
    ))
}

// Arch/OS-specific register/stack switch (the `unsafe` core: `jump`/`make`/`Transfer`).
#[cfg(all(unix, target_arch = "x86_64"))]
#[path = "switch_x86_64_sysv.rs"]
mod switch;
#[cfg(all(windows, target_arch = "x86_64"))]
#[path = "switch_x86_64_windows.rs"]
mod switch;

// OS-specific guard-paged control stack.
#[cfg(all(unix, target_arch = "x86_64"))]
#[path = "stack_unix.rs"]
mod stack;
#[cfg(all(windows, target_arch = "x86_64"))]
#[path = "stack_windows.rs"]
mod stack;

// Safe RAII asymmetric-coroutine wrapper (ABI-agnostic; built on `switch` + `stack`).
#[cfg(any(
    all(unix, target_arch = "x86_64"),
    all(windows, target_arch = "x86_64")
))]
mod fiber;

#[cfg(any(
    all(unix, target_arch = "x86_64"),
    all(windows, target_arch = "x86_64")
))]
pub use fiber::{Fiber, State, Yielder};

// Combined switch + stack integration tests, ABI-agnostic (they drive whichever `switch`/`stack` the
// target selected), so every supported target runs the same behavioral checks.
#[cfg(all(
    test,
    any(
        all(unix, target_arch = "x86_64"),
        all(windows, target_arch = "x86_64")
    )
))]
mod switch_tests {
    use crate::stack::Stack;
    use crate::switch::{jump, make, Transfer};

    /// A coroutine that keeps a running total: each resume adds the payload and yields the new total;
    /// a `u64::MAX` payload is the "finish" sentinel.
    extern "C" fn summer(mut t: Transfer) -> ! {
        let mut total: u64 = 0;
        loop {
            if t.data == u64::MAX {
                // Final switch back to the resumer; never resumed again.
                // SAFETY: `t.fctx` is the live resumer context.
                unsafe { jump(t.fctx, total) };
                unreachable!("resumed after finishing");
            }
            total = total.wrapping_add(t.data);
            // SAFETY: switch back to the resumer, who holds our stack alive.
            t = unsafe { jump(t.fctx, total) };
        }
    }

    #[test]
    fn switch_roundtrip_accumulates() {
        let stack = Stack::new(64 * 1024);
        let mut ctx = unsafe { make(&stack, summer) };
        for (payload, want) in [(5u64, 5u64), (10, 15), (100, 115), (1, 116)] {
            let t = unsafe { jump(ctx, payload) };
            ctx = t.fctx;
            assert_eq!(t.data, want, "after adding {payload}");
        }
        let t = unsafe { jump(ctx, u64::MAX) };
        assert_eq!(t.data, 116, "final total");
    }

    /// The coroutine reports the address of one of its locals; it must lie inside the fiber stack's
    /// usable range, proving execution really moved off the caller's stack.
    extern "C" fn report_sp(t: Transfer) -> ! {
        let local: u64 = 0xABCD;
        let addr = &local as *const u64 as u64;
        core::hint::black_box(local);
        unsafe { jump(t.fctx, addr) };
        unreachable!()
    }

    #[test]
    fn runs_on_the_fiber_stack() {
        let stack = Stack::new(64 * 1024);
        let (lo, hi) = stack.usable_range();
        let ctx = unsafe { make(&stack, report_sp) };
        let t = unsafe { jump(ctx, 0) };
        let addr = t.data as usize;
        assert!(
            (lo..hi).contains(&addr),
            "local at {addr:#x} not in fiber stack [{lo:#x}, {hi:#x})"
        );
    }

    /// Deep recursion on the fiber stack: a recursive Fibonacci proves the switched-to stack supports
    /// ordinary nested frames, not just a single leaf.
    extern "C" fn fibber(t: Transfer) -> ! {
        fn fib(n: u64) -> u64 {
            if n < 2 {
                n
            } else {
                fib(n - 1) + fib(n - 2)
            }
        }
        let r = fib(t.data);
        unsafe { jump(t.fctx, r) };
        unreachable!()
    }

    #[test]
    fn recursion_on_fiber_stack() {
        let stack = Stack::new(256 * 1024);
        let ctx = unsafe { make(&stack, fibber) };
        let t = unsafe { jump(ctx, 25) };
        assert_eq!(t.data, 75025); // fib(25)
    }

    /// Stress: many back-and-forth switches must stay consistent (no register/stack drift).
    #[test]
    fn many_switches_are_stable() {
        let stack = Stack::new(64 * 1024);
        let mut ctx = unsafe { make(&stack, summer) };
        let mut expect = 0u64;
        for i in 0..100_000u64 {
            expect = expect.wrapping_add(i | 1); // never u64::MAX
            let t = unsafe { jump(ctx, i | 1) };
            ctx = t.fctx;
            assert_eq!(t.data, expect);
        }
    }

    /// Two independent fibers interleaved through the same resumer keep separate state.
    #[test]
    fn two_fibers_independent() {
        let sa = Stack::new(64 * 1024);
        let sb = Stack::new(64 * 1024);
        let mut a = unsafe { make(&sa, summer) };
        let mut b = unsafe { make(&sb, summer) };
        let ta = unsafe { jump(a, 3) };
        a = ta.fctx;
        let tb = unsafe { jump(b, 7) };
        b = tb.fctx;
        let ta = unsafe { jump(a, 4) };
        a = ta.fctx;
        assert_eq!(ta.data, 7); // 3 + 4
        let tb = unsafe { jump(b, 70) };
        b = tb.fctx;
        assert_eq!(tb.data, 77); // 7 + 70
        let _ = (a, b);
    }
}
