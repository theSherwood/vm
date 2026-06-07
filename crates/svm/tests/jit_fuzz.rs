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

/// Guard that the generator actually covers loops (back-edges), indirect calls, and cap.calls —
/// so the differential above is exercising them, not silently regressing to forward-only DAGs.
#[test]
fn generator_covers_loops_indirect_and_cap_calls() {
    use svm_ir::{Inst, Terminator};
    let (mut loops, mut indirect, mut cap) = (0u32, 0u32, 0u32);
    let (mut data, mut data_ro, mut mem_cap) = (0u32, 0u32, 0u32);
    let (mut atomics, mut fences) = (0u32, 0u32);
    for seed in 0..2000u64 {
        let mut g = Gen::from_seed(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x5EED_5EED);
        let m = irgen::gen_module(&mut g);
        data += m.data.iter().filter(|d| !d.bytes.is_empty()).count() as u32;
        data_ro += m
            .data
            .iter()
            .filter(|d| d.readonly && !d.bytes.is_empty())
            .count() as u32;
        for f in &m.funcs {
            for (bi, blk) in f.blocks.iter().enumerate() {
                let back = |t: u32| t as usize <= bi; // a back-edge re-enters this/an earlier block
                let has_back = match &blk.term {
                    Terminator::Br { target, .. } => back(*target),
                    Terminator::BrIf {
                        then_blk, else_blk, ..
                    } => back(*then_blk) || back(*else_blk),
                    Terminator::BrTable {
                        targets, default, ..
                    } => targets.iter().any(|(t, _)| back(*t)) || back(default.0),
                    _ => false,
                };
                loops += has_back as u32;
                indirect += blk
                    .insts
                    .iter()
                    .filter(|i| matches!(i, Inst::CallIndirect { .. }))
                    .count() as u32;
                cap += blk
                    .insts
                    .iter()
                    .filter(|i| matches!(i, Inst::CapCall { .. }))
                    .count() as u32;
                // type_id 3 = the Memory interface: a *valid* (granted-handle) cap.call, exercising
                // the success path, vs the forged-handle (CapFault) ones the other arm emits.
                mem_cap += blk
                    .insts
                    .iter()
                    .filter(|i| matches!(i, Inst::CapCall { type_id: 3, .. }))
                    .count() as u32;
                atomics += blk
                    .insts
                    .iter()
                    .filter(|i| {
                        matches!(
                            i,
                            Inst::AtomicLoad { .. }
                                | Inst::AtomicStore { .. }
                                | Inst::AtomicRmw { .. }
                                | Inst::AtomicCmpxchg { .. }
                        )
                    })
                    .count() as u32;
                fences += blk
                    .insts
                    .iter()
                    .filter(|i| matches!(i, Inst::AtomicFence { .. }))
                    .count() as u32;
            }
        }
    }
    assert!(atomics > 0, "generator produced no atomic ops");
    assert!(fences > 0, "generator produced no fences");
    assert!(loops > 0, "generator produced no loop back-edges");
    assert!(indirect > 0, "generator produced no call_indirect");
    assert!(cap > 0, "generator produced no cap.call");
    assert!(
        mem_cap > 0,
        "generator produced no valid Memory cap.call (success path)"
    );
    assert!(data > 0, "generator produced no (non-empty) data segments");
    assert!(data_ro > 0, "generator produced no read-only data segments");
}
