//! libFuzzer target: the freeze/thaw equivalence property for **poll-free-loop** modules
//! (Phase-4 Slice A back-edge polls). Drives the loop generator (shared with the stable
//! `durable_fuzz::loop_freeze_thaw_equivalence_over_generated_modules` test): a loop header
//! (a back-edge target) ahead of the `cap.call`, so the inserted loop-header poll is the
//! freeze site. Asserts inert-in-NORMAL and freeze → thaw equals the uninterrupted run. A
//! crash here is a transform bug in the back-edge-poll path.
//!
//! Run: `cargo +nightly fuzz run durable_loop`
#![no_main]

use libfuzzer_sys::fuzz_target;

#[path = "../../crates/svm-durable/tests/support/durgen.rs"]
mod durgen;

fuzz_target!(|data: &[u8]| {
    let mut g = durgen::Gen::from_bytes(data);
    durgen::fuzz_loop_one(&mut g);
});
