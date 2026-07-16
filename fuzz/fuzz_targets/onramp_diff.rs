//! libFuzzer target: LLVM on-ramp source-oracle differential (`DESIGN.md` §18).
//!
//! The input bytes drive the structured generator (`crates/svm-llvm/tests/support/llgen.rs`,
//! shared with the stable `onramp_diff` test): it emits a well-defined, terminating `@run()` and
//! computes its result *as it emits* (the oracle). We translate through the LLVM on-ramp and
//! assert every backend that can execute it (tree-walker always; bytecode/JIT when they don't
//! fail-closed) returns the oracle. A crash is an **on-ramp translation miscompile** — the I23
//! class, where all backends agree with a mistranslated IR (so an interp-vs-JIT differential is
//! blind to it; the source-semantics oracle is not).
//!
//! Run: `cargo +nightly fuzz run onramp_diff`
#![no_main]

use libfuzzer_sys::fuzz_target;

#[path = "../../crates/svm-llvm/tests/support/llgen.rs"]
mod llgen;

fuzz_target!(|data: &[u8]| {
    let mut g = llgen::Gen::from_bytes(data);
    let (ll, oracle) = llgen::gen_program(&mut g);
    if let Err(reason) = llgen::check(&ll, oracle) {
        panic!("{reason}\n--- module ---\n{ll}");
    }
});
