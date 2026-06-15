//! libFuzzer target: the cross-backend freeze/thaw property (DURABILITY.md §7/§12.6).
//!
//! Drives the in-scope durable-module generator (shared with the stable
//! `durable_jit` test) and asserts the reference interpreter and the Cranelift JIT agree
//! on the instrumented module — same NORMAL result, byte-identical freeze artifact, and a
//! portable thaw (interpreter-frozen artifact resumed on the JIT). A crash here is a
//! backend divergence in the transform's emitted IR.
//!
//! Run: `cargo +nightly fuzz run durable_jit`
#![no_main]

use libfuzzer_sys::fuzz_target;

#[path = "../../crates/svm/tests/support/durjit.rs"]
mod durjit;

fuzz_target!(|data: &[u8]| {
    let mut g = durjit::durgen::Gen::from_bytes(data);
    durjit::fuzz_one_xbackend(&mut g);
});
