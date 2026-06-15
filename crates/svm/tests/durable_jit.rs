//! Stable (no-nightly) driver for the cross-backend freeze/thaw property — the CI-run
//! counterpart of the libFuzzer target `fuzz/fuzz_targets/durable_jit.rs`, sitting beside
//! the interpreter-vs-JIT differential (`jit_fuzz.rs`). A failure here is a backend
//! divergence in the durable transform's emitted IR.

#[path = "support/durjit.rs"]
mod durjit;

#[test]
fn freeze_thaw_cross_backend_over_generated_modules() {
    // Each module JIT-compiles three times over a 256 KiB window, so the count is lower
    // than the interp-only smoke; the libFuzzer target `durable_jit` does the heavy run.
    for seed in 0..150u64 {
        let mut g = durjit::durgen::Gen::from_seed(seed);
        durjit::fuzz_one_xbackend(&mut g);
    }
}
