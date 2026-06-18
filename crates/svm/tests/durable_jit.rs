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

#[test]
fn recycled_fiber_freeze_thaw_cross_backend_over_generated_modules() {
    // Recycling step 4, cross-backend: churn modules recycle a slot (generation 1..=3), park the
    // real fiber there, and are frozen *mid-run* via `arm_freeze_after`. The interp and JIT must
    // armed-freeze the recycled fiber to a byte-identical reserve + §12 artifact, and the
    // interp-frozen artifact must thaw on the JIT to the uninterrupted result. The libFuzzer target
    // `durable_recycle_jit` does the heavy run, so keep this a modest smoke (each seed compiles the
    // JIT twice — freeze + thaw — and commits a window each time).
    for seed in 0..64u64 {
        let mut g = durjit::durgen::Gen::from_seed(seed);
        durjit::fuzz_recycle_fiber_one_xbackend(&mut g);
    }
}
