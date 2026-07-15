//! libFuzzer target: per-op semantics vs the executable spec (SPEC.md), on all three backends.
//!
//! The input bytes pick a scalar/float spec row (`svm_spec::all_rows`) and feed it random
//! operand values; the tree-walk interpreter, the bytecode interpreter, and the Cranelift JIT
//! must all agree with the spec's reference `eval`. This is the coverage-guided counterpart to
//! the deterministic boundary lattice in `spec_vectors` — a crash is a backend diverging from
//! the spec definition. Driver + stable mirror: `crates/svm/tests/support/specfuzz.rs`.
//!
//! Run: `cargo +nightly fuzz run spec_ops`
#![no_main]

use libfuzzer_sys::fuzz_target;

#[path = "../../crates/svm/tests/support/specfuzz.rs"]
mod specfuzz;

fuzz_target!(|data: &[u8]| {
    specfuzz::ops_one(data);
});
