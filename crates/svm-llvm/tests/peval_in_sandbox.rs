//! **The in-sandbox `svm-peval` pipeline (DESIGN.md §20c), folded into a test.**
//!
//! The manual probe — `rustc +1.81 --emit=llvm-bc` → `llvm-link-18` → `opt-18 internalize,globaldce`
//! → translate → verify → run — was a scratch-dir dance. This test runs it on the in-repo fixture
//! `tests/fixtures/peval_probe`: a `no_std` powerbox program that builds a small module and calls
//! `svm_peval::specialize` (the `default-features = false`, no-`libm` in-svm build). It then asserts
//! the residual summary the *in-sandbox* specializer prints equals the **same** specialization run
//! host-side — a differential: in-sandbox specializer == host specializer.
//!
//! This regression-proofs the *whole* `specialize` closure end-to-end in-sandbox (the ZST-layout fix
//! only added a unit-level test). It auto-skips when the toolchain (`rustc +1.81.0`, `llvm-link-18`,
//! `opt-18`) is unavailable — the same posture as the `rust_*` tests in `translate.rs`.

use svm_ir::{BinOp, Block, Func, Inst, IntTy, Module, Terminator, ValType};

mod common;

/// The module the fixture builds (kept in lockstep with `peval_probe`'s `build_module`): a single
/// `() -> i32` whose body is the constant product `21 * 2`. A correct specializer folds it.
fn oracle_module() -> Module {
    Module {
        funcs: vec![Func {
            params: vec![],
            results: vec![ValType::I32],
            blocks: vec![Block {
                params: vec![],
                insts: vec![
                    Inst::ConstI32(21),
                    Inst::ConstI32(2),
                    Inst::IntBin {
                        ty: IntTy::I32,
                        op: BinOp::Mul,
                        a: 0,
                        b: 1,
                    },
                ],
                term: Terminator::Return(vec![2]),
            }],
        }],
        memory: None,
        data: Vec::new(),
        imports: Vec::new(),
        exports: Vec::new(),
        debug_info: None,
    }
}

/// `funcs\nblocks\ninsts\n` for a module — the summary the fixture prints and the oracle compares.
fn summary_bytes(m: &Module) -> Vec<u8> {
    let funcs = m.funcs.len();
    let blocks: usize = m.funcs.iter().map(|f| f.blocks.len()).sum();
    let insts: usize = m
        .funcs
        .iter()
        .flat_map(|f| f.blocks.iter())
        .map(|b| b.insts.len())
        .sum();
    format!("{funcs}\n{blocks}\n{insts}\n").into_bytes()
}

/// **The in-sandbox specializer runs and agrees with the host specializer.** Build the probe to
/// svm-IR, run it, and assert its printed residual summary equals the same `specialize` run host-side.
/// The host (`svm-peval` default features, `libm` on) and guest (no `libm`) agree because this module
/// is integer-only — no float folds differ. The whole `specialize` closure (≈100 funcs spanning
/// `svm-peval` + `svm-ir` + `svm-verify` + `core`/`alloc`) translates, verifies, and executes.
#[test]
fn peval_specialize_runs_in_sandbox_and_matches_host() {
    let Some(bc) = common::build_fixture_bc("peval_probe") else {
        return; // toolchain unavailable — skip
    };
    let t =
        svm_llvm::translate_ll_path(&bc).expect("translate the in-sandbox specializer to svm-IR");
    assert!(
        svm_run::is_named_powerbox_entry(&t.module),
        "the probe must produce a powerbox entry"
    );
    let module = svm_run::resolve_capability_imports(t.module).expect("resolve capability imports");
    svm_verify::verify_module(&module).expect("verify the translated specializer");

    let run = svm_run::run_powerbox_with_deadline(
        &module,
        b"",
        Some(std::time::Duration::from_secs(120)),
    )
    .expect("run the in-sandbox specializer");

    // The host oracle: the identical module, specialized host-side, summarized the same way.
    let residual = svm_peval::specialize(&oracle_module(), 0, &[]).expect("host specialize");
    let expected = summary_bytes(&residual);

    assert_eq!(
        run.stdout,
        expected,
        "in-sandbox specializer summary {:?} != host summary {:?}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&expected),
    );
}
