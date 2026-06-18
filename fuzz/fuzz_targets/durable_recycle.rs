//! libFuzzer target: the recycled-fiber freeze/thaw equivalence property (recycling step 4).
//!
//! The input bytes drive the recycling-churn generator
//! (`crates/svm-durable/tests/support/durgen.rs`, shared with the stable
//! `durable_fuzz::recycled_fiber_freeze_thaw_equivalence_over_generated_modules` test): synthesize a
//! module that recycles a slot 1..=3 times (so the real fiber lands at generation 1..=3), park it,
//! freeze *mid-run* via `arm_freeze_after`, and assert the residue carries the bumped generation and
//! the thaw reproduces the uninterrupted run. A crash here is a recycling / freeze-trigger bug.
//!
//! Run: `cargo +nightly fuzz run durable_recycle`
#![no_main]

use libfuzzer_sys::fuzz_target;

#[path = "../../crates/svm-durable/tests/support/durgen.rs"]
mod durgen;

fuzz_target!(|data: &[u8]| {
    let mut g = durgen::Gen::from_bytes(data);
    durgen::fuzz_recycle_fiber_one(&mut g);
});
