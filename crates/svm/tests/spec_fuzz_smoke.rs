//! Stable-toolchain smoke run of the spec fuzz drivers (SPEC.md) — the coverage-guided
//! targets live in `fuzz/` (nightly); this drives the *same* `specfuzz` drivers over a
//! deterministic seed range so they gate on stable CI and cannot silently rot, mirroring
//! how `fuzz_smoke` mirrors the escape-TCB targets and `jit_fuzz` mirrors `diff`.

#[path = "support/specfuzz.rs"]
mod specfuzz;

/// The op-semantics differential (interp / bytecode / JIT vs the spec `eval`) over seeded
/// pseudo-random inputs. Compiles a JIT module per seed, so keep the count modest.
#[test]
fn spec_ops_smoke() {
    for seed in 0..200u64 {
        specfuzz::ops_one(&seed.to_le_bytes());
    }
}

/// The verifier accept/reject differential (svm-verify vs the reference verifier) over
/// seeded `irgen` modules + a mutation each. Cheap (no JIT), so run more.
#[test]
fn spec_verify_smoke() {
    for seed in 0..2_000u64 {
        specfuzz::verify_one(&seed.to_le_bytes());
    }
}
