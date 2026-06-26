//! libFuzzer target: the freeze/thaw equivalence property (DURABILITY.md §7/§12.6).
//!
//! The input bytes drive the in-scope durable-module generator
//! (`crates/svm-durable/tests/support/durgen.rs`, shared with the stable
//! `durable_fuzz` test): synthesize a verifier-valid in-scope module, then assert both
//! that instrumentation is inert in `NORMAL` state and that
//! freeze → serialize → restore → thaw equals the uninterrupted run. A crash here is a
//! transform bug (or a generator bug).
//!
//! Run: `cargo +nightly fuzz run durable`
#![no_main]

use libfuzzer_sys::fuzz_target;

#[path = "../../crates/svm-durable/tests/support/durgen.rs"]
mod durgen;

fuzz_target!(|data: &[u8]| {
    let mut g = durgen::Gen::from_bytes(data);
    durgen::fuzz_one(&mut g);
});
