//! Native **stack switching** — the substrate for JIT green threads and fibers (§3d/§6/§12).
//!
//! The interpreter's continuation is data (a `Vec<Frame>`), so it can park and migrate between
//! workers in safe Rust. JITted code's continuation is a *native machine stack*; to suspend it you
//! **switch stacks** — save the callee-saved registers + stack pointer, load another context's — not
//! reify frames. That is exactly what §3d's two-stack model calls for: every fiber/green-thread owns
//! an out-of-band **control stack** the runtime allocates, and a switch is `~ns` (no syscall).
//!
//! This crate is the single home for that `unsafe`, kept tiny and auditable (the way `svm-mem` is for
//! memory). It exposes a *symmetric* `boost.context`-style primitive:
//!
//! - [`Stack`] — a guard-paged, mmap'd region for one control stack (a `PROT_NONE` page catches
//!   overflow as a fault rather than silent corruption, §5).
//! - [`make`] — lay out a fresh stack so the first [`jump`] into it begins executing an entry
//!   function.
//! - [`jump`] — switch to a saved context, handing it a `u64` and receiving back the context we came
//!   *from* plus its `u64` (a [`Transfer`]).
//!
//! Higher layers (a safe `Fiber`/`Yielder` wrapper, then scheduler integration) build on this.
//!
//! Supported today on **x86-64 unix** only; other targets compile but [`supported`] returns `false`
//! and the primitive is absent (the JIT keeps bailing `Unsupported` there).

#![cfg_attr(not(test), no_std)]

/// Whether real stack switching is available on this target.
pub const fn supported() -> bool {
    cfg!(all(unix, target_arch = "x86_64"))
}

#[cfg(all(unix, target_arch = "x86_64"))]
mod imp;

#[cfg(all(unix, target_arch = "x86_64"))]
pub use imp::{jump, make, Stack, Transfer};
