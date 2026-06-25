//! **The richer in-sandbox Futamura `Jit` demo (PEVAL.md Milestone 3 follow-up), as a test.**
//!
//! Builds the in-repo fixture `tests/fixtures/peval_futamura` — a `no_std` powerbox guest that, in
//! sandbox, builds a small accumulator-machine *interpreter* in svm-IR, **specializes it against a
//! fixed program** supplied as a `SpecConfig` const-overlay (so the dispatch loop unrolls and the
//! program folds away), encodes the residual, and `Jit.compile`s + invokes it. The genuine first
//! Futamura projection — interpreter + program → compiled program — performed in-sandbox.
//!
//! The host translates the guest, passes its on-ramp-assigned window `size_log2` as `argv[1]` (so the
//! residual satisfies the `Jit.compile` memory-match precondition), runs it under `run_powerbox`, and
//! asserts the guest reports `0` mismatches and that the residual collapsed to a single straight-line
//! block (no dispatch loop left). Auto-skips without `rustc +1.81.0` / `llvm-link-18` / `opt-18`.

mod common;

#[test]
fn peval_guest_specializes_interpreter_and_jits_in_sandbox() {
    let Some(bc) = common::build_fixture_bc("peval_futamura") else {
        return; // toolchain unavailable — skip
    };
    let t = svm_llvm::translate_bc_path(&bc).expect("translate the peval-futamura guest to svm-IR");
    assert!(
        svm_run::is_powerbox_entry(&t.module),
        "the guest must produce a powerbox entry"
    );
    let win_log2 = t
        .module
        .memory
        .expect("a heap-allocating guest has a window")
        .size_log2;
    eprintln!("guest window size_log2 = {win_log2}");

    let module = svm_run::resolve_capability_imports(t.module).expect("resolve capability imports");
    svm_verify::verify_module(&module).expect("verify the translated guest");

    let win_arg = win_log2.to_string();
    let argv: [&[u8]; 2] = [b"peval-futamura", win_arg.as_bytes()];
    let run = svm_run::run_powerbox_with_args_and_limits(
        &module,
        b"",
        &argv,
        &[],
        Some(std::time::Duration::from_secs(180)),
        Default::default(),
    )
    .expect("run the peval-futamura guest");

    let stdout = String::from_utf8_lossy(&run.stdout);
    eprintln!(
        "--- guest stdout ---\n{stdout}\n--- outcome: {:?} ---",
        run.outcome
    );
    assert!(
        stdout.contains("inputs agree"),
        "guest did not report success; stdout:\n{stdout}"
    );
    assert!(
        !stdout.contains("MISMATCH") && !stdout.contains("failed"),
        "guest reported a failure; stdout:\n{stdout}"
    );
    // Proof the interpreter actually specialized away (not merely that some residual ran): the
    // residual must contain no `br_table` (dispatch) and no `Load` (decode) — the loop unrolled and
    // every program read folded.
    assert!(
        stdout.contains("dispatch + decode folded away"),
        "expected the dispatch + decode to fold away; stdout:\n{stdout}"
    );
}
