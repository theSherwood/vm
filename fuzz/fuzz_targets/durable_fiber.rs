//! libFuzzer target: the §12.8 single-fiber freeze/thaw equivalence property (Phase 3.1).
//!
//! The input bytes drive the fiber'd-module generator
//! (`crates/svm-durable/tests/support/durgen.rs`, shared with the stable
//! `durable_fuzz::fiber_freeze_thaw_equivalence_over_generated_modules` test): synthesize a
//! verifier-valid root+fiber module, then assert that instrumentation is inert in `NORMAL` and
//! that freeze (driver flattens the parked fiber + exports its residue) → thaw (re-seed + re-enter
//! under `REWINDING`) equals the uninterrupted run. A crash here is a transform or runtime bug.
//!
//! Run: `cargo +nightly fuzz run durable_fiber`
#![no_main]

use libfuzzer_sys::fuzz_target;

#[path = "../../crates/svm-durable/tests/support/durgen.rs"]
mod durgen;

fuzz_target!(|data: &[u8]| {
    let mut g = durgen::Gen::from_bytes(data);
    durgen::fuzz_fiber_one(&mut g);
});
