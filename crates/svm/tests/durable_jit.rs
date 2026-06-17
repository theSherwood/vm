//! Stable (no-nightly) driver for the cross-backend freeze/thaw property — the CI-run
//! counterpart of the libFuzzer target `fuzz/fuzz_targets/durable_jit.rs`, sitting beside
//! the interpreter-vs-JIT differential (`jit_fuzz.rs`). A failure here is a backend
//! divergence in the durable transform's emitted IR.

#[path = "support/durjit.rs"]
mod durjit;

#[test]
fn freeze_thaw_cross_backend_over_generated_modules() {
    // Each module JIT-compiles three times and commits a guest window each time; the libFuzzer
    // target `durable_jit` does the heavy run, so keep this a modest smoke. The low count also
    // bounds the cumulative JIT commit on memory-tight Windows CI (os error 1455 / commit limit).
    for seed in 0..64u64 {
        let mut g = durjit::durgen::Gen::from_seed(seed);
        durjit::fuzz_one_xbackend(&mut g);
    }
}
