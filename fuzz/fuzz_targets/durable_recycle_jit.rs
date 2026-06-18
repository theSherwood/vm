//! libFuzzer target: the cross-backend recycled-fiber freeze/thaw property (recycling step 4).
//!
//! Drives the recycling-churn generator (shared with the stable `durable_jit`
//! ::recycled_fiber_freeze_thaw_cross_backend_over_generated_modules test) and asserts the reference
//! interpreter and the Cranelift JIT armed-freeze a recycled (generation > 0) parked fiber to a
//! byte-identical durable reserve + §12 artifact, and that the interp-frozen artifact thaws on the
//! JIT to the uninterrupted result. A crash here is a backend divergence in the recycling / mid-run
//! freeze-trigger path.
//!
//! Run: `cargo +nightly fuzz run durable_recycle_jit`
#![no_main]

use libfuzzer_sys::fuzz_target;

#[path = "../../crates/svm/tests/support/durjit.rs"]
mod durjit;

fuzz_target!(|data: &[u8]| {
    let mut g = durjit::durgen::Gen::from_bytes(data);
    durjit::fuzz_recycle_fiber_one_xbackend(&mut g);
});
