//! Stable (no-nightly) driver for the freeze/thaw equivalence property — the CI-run
//! counterpart of the libFuzzer target `fuzz/fuzz_targets/durable.rs`, mirroring how
//! `crates/svm/tests/jit_fuzz.rs` exercises `irgen` continuously. A failure here is a
//! transform bug (or a generator bug), surfaced without a fuzzing toolchain.

#[path = "support/durgen.rs"]
mod durgen;

#[test]
fn freeze_thaw_equivalence_over_generated_modules() {
    // A fast regression smoke (each module runs the property's 4 executions over a
    // 256 KiB window). The libFuzzer target `durable` does the heavy continuous run.
    for seed in 0..400u64 {
        let mut g = durgen::Gen::from_seed(seed);
        durgen::fuzz_one(&mut g);
    }
}

#[test]
fn fiber_freeze_thaw_equivalence_over_generated_modules() {
    // The §12.8 single-fiber freeze/thaw property over generated root+fiber modules (varying
    // suspend counts, live-across-suspend values, multi-point resume/suspend). The libFuzzer
    // target `durable_fiber` does the heavy continuous run.
    for seed in 0..400u64 {
        let mut g = durgen::Gen::from_seed(seed);
        durgen::fuzz_fiber_one(&mut g);
    }
}

#[test]
fn recycled_fiber_freeze_thaw_equivalence_over_generated_modules() {
    // Recycling step 4: churn modules that recycle a slot (1..=3 times → generation 1..=3) before
    // parking the real fiber, frozen *mid-run* via `arm_freeze_after`. The residue must carry the
    // bumped generation and the thaw must reproduce the uninterrupted run. The libFuzzer target
    // `durable_recycle` does the heavy continuous run.
    for seed in 0..400u64 {
        let mut g = durgen::Gen::from_seed(seed);
        durgen::fuzz_recycle_fiber_one(&mut g);
    }
}
