//! libFuzzer target: verifier accept/reject differential (SPEC.md suite 2).
//!
//! The input bytes drive `irgen` to synthesize a verifier-valid module, then a random
//! structural mutation; `svm-verify` (the production TCB verifier) and `svm_spec::verify` (the
//! independent reference verifier) must agree on accept/reject. This is the coverage-guided
//! counterpart to `spec_verify`'s deterministic `irgen` sweep — a crash is a verifier
//! disagreement (an accept-direction bug). Driver + stable mirror:
//! `crates/svm/tests/support/specfuzz.rs`.
//!
//! Run: `cargo +nightly fuzz run spec_verify`
#![no_main]

use libfuzzer_sys::fuzz_target;

#[path = "../../crates/svm/tests/support/specfuzz.rs"]
mod specfuzz;

fuzz_target!(|data: &[u8]| {
    specfuzz::verify_one(data);
});
