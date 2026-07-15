//! libFuzzer target: the optimizer's internal conventional-SSA round-trip is the identity
//! (`OPT.md` Phase 1c). The input bytes drive the structured generator (shared with the `diff`
//! target and the stable `jit_fuzz` test) to synthesize a verifier-valid module; for every function
//! `from_ssa(to_ssa(f))` must reproduce `f` **exactly**, and the round-tripped module must still
//! agree interp-vs-JIT. A crash here is a bug in the SSA convert/lower boundary before any pass has
//! even run over it.
//!
//! Run: `cargo +nightly fuzz run opt_ssa_roundtrip`
#![no_main]

use libfuzzer_sys::fuzz_target;

#[path = "../../crates/svm/tests/support/irgen.rs"]
mod irgen;

use svm_opt::ssa::{from_ssa, to_ssa};

fuzz_target!(|data: &[u8]| {
    let mut g = irgen::Gen::from_bytes(data);
    let m = irgen::gen_module(&mut g);
    if m.funcs.is_empty() {
        return;
    }
    let fr: Vec<usize> = m.funcs.iter().map(|f| f.results.len()).collect();

    // Round-trip every function through the internal SSA form; the lowering must be lossless.
    let mut rt = m.clone();
    for f in &mut rt.funcs {
        let back = from_ssa(&to_ssa(f, &fr));
        assert_eq!(&back, f, "SSA round-trip must be the identity");
        *f = back;
    }

    // Defense in depth: the round-tripped module must behave identically on both backends.
    let args = irgen::gen_args(&mut g, &m.funcs[0].params);
    irgen::run_differential(&rt, &args);
});
