//! libFuzzer target: generative interpreter-vs-JIT differential (`DESIGN.md` §18).
//!
//! The input bytes drive the structured generator (`crates/svm/tests/support/irgen.rs`,
//! shared with the stable `jit_fuzz` test): they synthesize a verifier-valid module, run
//! its entry on both the reference interpreter and the Cranelift JIT, and assert the two
//! agree on result and trap. A crash here is a JIT miscompile (or a generator bug).
//!
//! Run: `cargo +nightly fuzz run diff`
#![no_main]

use libfuzzer_sys::fuzz_target;

#[path = "../../crates/svm/tests/support/irgen.rs"]
mod irgen;

fuzz_target!(|data: &[u8]| {
    let mut g = irgen::Gen::from_bytes(data);
    irgen::fuzz_one(&mut g);
});
