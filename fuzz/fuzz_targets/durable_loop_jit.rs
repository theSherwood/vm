//! libFuzzer target: the cross-backend freeze/thaw property for **poll-free-loop** modules
//! (Phase-4 Slice A back-edge polls). Drives the loop generator (shared with the stable
//! `durable_jit::loop_freeze_thaw_cross_backend_over_generated_modules` test) and asserts the
//! reference interpreter and the Cranelift JIT agree on the loop-header poll — same NORMAL
//! result, byte-identical freeze artifact, and a portable thaw. A crash here is a backend
//! divergence in the back-edge-poll IR.
//!
//! Run: `cargo +nightly fuzz run durable_loop_jit`
#![no_main]

use libfuzzer_sys::fuzz_target;

#[path = "../../crates/svm/tests/support/durjit.rs"]
mod durjit;

fuzz_target!(|data: &[u8]| {
    let mut g = durjit::durgen::Gen::from_bytes(data);
    durjit::fuzz_loop_one_xbackend(&mut g);
});
