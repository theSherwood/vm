//! Generative differential fuzzing of the JIT against the reference interpreter
//! (`DESIGN.md` §18). For thousands of seeds we synthesize a **verifier-valid** module
//! (see `support/irgen.rs`), run its entry on both backends, and assert they agree on the
//! result *and* on whether/why they trap. The interpreter is the spec; any divergence is
//! a JIT miscompile. This systematically explores the op × type × control-flow × memory
//! space the hand-written `jit_diff` cases cannot.
//!
//! Stable-toolchain (deterministic seeds, runs in CI). The libFuzzer `diff` target drives
//! the *same* generator (`fuzz_one`) from coverage-guided input for unbounded exploration.

#[path = "support/irgen.rs"]
mod irgen;

use irgen::{fuzz_one, Gen};

#[test]
fn jit_matches_interp_on_generated_modules() {
    for seed in 0..4000u64 {
        let mut g = Gen::from_seed(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1CE_F00D);
        fuzz_one(&mut g);
    }
}
