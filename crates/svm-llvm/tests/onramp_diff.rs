//! On-ramp differential harness (stable/seeded — the CI gate; `fuzz/onramp_diff` is the
//! coverage-guided sibling). For each generated `@run()` (well-defined + terminating by
//! construction — see `support/llgen.rs`), translate it through the LLVM on-ramp and assert all
//! **three** svm backends (tree-walker, bytecode, JIT) return the generator's **oracle** value.
//!
//! The oracle is the source semantics (computed as the IR is emitted), not another backend — so
//! this catches the I23 class where every backend agrees with a *mistranslated* IR (a
//! const-GEP stride ignoring the source element type; a 2-lane vector min/max comparing the packed
//! word). An interp-vs-JIT differential is blind to those; this is not.
//!
//! Run more iterations locally: `SVM_ONRAMP_DIFF_ITERS=200000 cargo test -p svm-llvm onramp_diff`.

#[path = "support/llgen.rs"]
mod llgen;

#[test]
fn onramp_backends_agree_with_source_oracle() {
    let iters: u64 = std::env::var("SVM_ONRAMP_DIFF_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2000);
    let (mut translated, mut bc, mut jit) = (0u64, 0u64, 0u64);
    for seed in 0..iters {
        let mut g = llgen::Gen::from_seed(seed.wrapping_mul(0x9e3779b97f4a7c15) | 1);
        let (ll, oracle) = llgen::gen_program(&mut g);
        match llgen::check(&ll, oracle) {
            Ok(r) => {
                translated += r.translated as u64;
                bc += r.bc as u64;
                jit += r.jit as u64;
            }
            Err(reason) => panic!("seed {seed}: {reason}\n--- module ---\n{ll}"),
        }
    }
    eprintln!("onramp_diff: {iters} programs — {translated} translated, {bc} bytecode, {jit} JIT");
    // Guard against a vacuous pass: the generator must actually exercise all three backends on a
    // meaningful fraction (else a coverage/translation regression would hide here).
    assert!(
        translated * 2 >= iters,
        "only {translated}/{iters} translated"
    );
    assert!(bc * 4 >= iters, "bytecode ran on only {bc}/{iters}");
    assert!(jit * 4 >= iters, "JIT ran on only {jit}/{iters}");
}
