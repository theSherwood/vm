//! **The end-to-end §22 `Jit` demo (DESIGN.md §20c capstone), as a test.**
//!
//! Builds the in-repo fixture `tests/fixtures/peval_jit` — a `no_std` powerbox guest that, entirely
//! in-sandbox, *specializes* a module with `svm-peval`, *encodes* the residual with `svm-encode`, and
//! submits it to the §22 `Jit` capability (`__vm_jit_compile` / `__vm_jit_invoke2`) to compile and
//! run it. The full guest-driven Futamura loop: **specialize → encode → Jit.compile → invoke**, with
//! no host involvement beyond the capability the powerbox already grants.
//!
//! The host translates the guest to svm-IR, reads the on-ramp-assigned window `size_log2` (the
//! `Jit.compile` memory-match precondition requires the residual to declare the *same* window), passes
//! it to the guest as `argv[1]`, runs it under `run_powerbox` (which grants the `Jit` cap), and asserts
//! the guest reports `0` mismatches against its own oracle.
//!
//! Auto-skips when the toolchain (`rustc +1.81.0`, `llvm-link-18`, `opt-18`) is unavailable.

mod common;

/// **Guest specializes a module and JITs the residual, all in-sandbox.** The guest builds
/// `entry(a,b) -> helper(a,b) = a*3 + b*5 + 7`, specializes it (inlining + folding), encodes the
/// residual, `__vm_jit_compile`s it, and `__vm_jit_invoke2`s it over an input grid against its own
/// oracle. We pass the on-ramp-assigned window size so the residual satisfies the `Jit.compile`
/// memory-match precondition.
#[test]
fn peval_guest_specializes_and_jits_in_sandbox() {
    let Some(bc) = common::build_fixture_bc("peval_jit") else {
        return; // toolchain unavailable — skip
    };
    let t = svm_llvm::translate_ll_path(&bc).expect("translate the peval-jit guest to svm-IR");
    assert!(
        svm_run::is_named_powerbox_entry(&t.module),
        "the guest must produce a powerbox entry"
    );
    // The window the on-ramp assigned this guest — the residual must declare the same one.
    let win_log2 = t
        .module
        .memory
        .expect("a heap-allocating guest has a window")
        .size_log2;
    eprintln!("guest window size_log2 = {win_log2}");

    let module = svm_run::resolve_capability_imports(t.module).expect("resolve capability imports");
    svm_verify::verify_module(&module).expect("verify the translated guest");

    let win_arg = win_log2.to_string();
    let argv: [&[u8]; 2] = [b"peval-jit", win_arg.as_bytes()];
    let run = svm_run::run_powerbox_with_args_and_limits(
        &module,
        b"",
        &argv,
        &[],
        Some(std::time::Duration::from_secs(180)),
        Default::default(),
    )
    .expect("run the peval-jit guest");

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
}
