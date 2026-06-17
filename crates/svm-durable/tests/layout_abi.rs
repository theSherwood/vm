//! The durable shadow-stack layout is a contract between the **transform** (`svm-durable`,
//! tooling tier) and the **runtime** (`svm-interp`, TCB): the transform emits IR that
//! reads/writes the active shadow-SP at a fixed window offset, and the runtime keeps that word
//! pointing at the running context's per-fiber region (D-fiber-cont option A, DURABILITY.md
//! §12.8). `svm-interp` must not depend on the tooling-tier `svm-durable`, so the shared
//! constants are defined in both crates; this test fails CI if they ever drift apart.

#[test]
fn shadow_layout_constants_agree_across_crates() {
    assert_eq!(
        svm_durable::SHADOW_SP_OFF,
        svm_interp::SHADOW_SP_OFF,
        "active shadow-SP offset must match (the transform writes it; the runtime swaps it)"
    );
    assert_eq!(
        svm_durable::SHADOW_BASE,
        svm_interp::SHADOW_BASE,
        "context-0 shadow base must match"
    );
    assert_eq!(
        svm_durable::DURABLE_RESERVE,
        svm_interp::DURABLE_RESERVE,
        "durable reserve ceiling must match"
    );
    assert_eq!(
        svm_durable::STATE_OFF,
        svm_interp::STATE_OFF,
        "state-word offset must match (the freeze driver reads what the transform writes)"
    );
    assert_eq!(
        svm_durable::STATE_UNWINDING,
        svm_interp::STATE_UNWINDING,
        "the UNWINDING state value must match (the freeze-driver trigger)"
    );
}
