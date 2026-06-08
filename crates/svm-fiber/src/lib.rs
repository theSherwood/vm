//! Native **stack switching** — the substrate for JIT green threads and fibers (§3d/§6/§12).
//!
//! The interpreter's continuation is data (a `Vec<Frame>`), so it can park and migrate between
//! workers in safe Rust. JITted code's continuation is a *native machine stack*; to suspend it you
//! **switch stacks** — save the callee-saved registers + stack pointer, load another context's — not
//! reify frames. That is exactly what §3d's two-stack model calls for: every fiber/green-thread owns
//! an out-of-band **control stack** the runtime allocates, and a switch is `~ns` (no syscall).
//!
//! This crate is the single home for that `unsafe`, kept tiny and auditable (the way `svm-mem` is for
//! memory). Two layers:
//!
//! - [`imp`] — the raw `boost.context`-style primitive: [`Stack`] (a guard-paged control stack),
//!   [`make`] (lay out a fresh stack), and [`jump`] (the register/stack swap, exchanging a [`Transfer`]).
//! - [`Fiber`] / [`Yielder`] — a safe RAII *asymmetric coroutine*: resume a fiber with a value, the
//!   fiber body does work and [`Yielder::suspend`]s values back, RAII frees the stack, and a panic
//!   inside the fiber aborts (unwinding across a stack switch would be UB) rather than corrupting.
//!
//! Supported today on **x86-64 unix** only; other targets compile but [`supported`] returns `false`
//! and the primitives are absent (the JIT keeps bailing `Unsupported` there).

/// Whether real stack switching is available on this target.
pub const fn supported() -> bool {
    cfg!(all(unix, target_arch = "x86_64"))
}

#[cfg(all(unix, target_arch = "x86_64"))]
mod imp;

#[cfg(all(unix, target_arch = "x86_64"))]
pub use imp::{jump, make, Stack, Transfer};

#[cfg(all(unix, target_arch = "x86_64"))]
mod fiber;

#[cfg(all(unix, target_arch = "x86_64"))]
pub use fiber::{Fiber, State, Yielder};
