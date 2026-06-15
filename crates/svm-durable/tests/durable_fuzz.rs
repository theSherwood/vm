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
