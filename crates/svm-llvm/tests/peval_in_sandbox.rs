//! **The in-sandbox `svm-peval` pipeline (PEVAL.md Milestone 3), folded into a test.**
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

use std::path::{Path, PathBuf};
use std::process::Command;

use svm_ir::{BinOp, Block, Func, Inst, IntTy, Module, Terminator, ValType};

/// The directory of the in-repo fixture crate (`tests/fixtures/peval_probe`).
fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/peval_probe")
}

/// Is `cmd --version` runnable? Used to auto-skip when a pipeline tool is absent.
fn tool_ok(cmd: &str, version_arg: &str) -> bool {
    Command::new(cmd)
        .arg(version_arg)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Build the fixture crate to LLVM-18 bitcode with `rustc +1.81.0` (the on-ramp's pinned toolchain),
/// link every dependency's `.bc`, and `globaldce` down to the closure reachable from the powerbox
/// `main`/`malloc`/`free`. Returns the legalized `.bc` path, or `None` if a tool is unavailable.
///
/// Mirrors the manual probe exactly: `RUSTFLAGS=--emit=llvm-bc cargo +1.81.0 build --release
/// --ignore-rust-version` → `llvm-link-18 <deps>/*.bc` → `opt-18 internalize,globaldce`.
fn build_probe_bc() -> Option<PathBuf> {
    if !tool_ok("rustc", "--version")
        || !Command::new("rustc")
            .args(["+1.81.0", "--version"])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
        || !tool_ok("llvm-link-18", "--version")
        || !tool_ok("opt-18", "--version")
    {
        eprintln!(
            "note: skipping peval_in_sandbox (need `rustc +1.81.0`, `llvm-link-18`, `opt-18`)"
        );
        return None;
    }

    // A dedicated, out-of-tree target dir keeps the repo clean and isolates the bitcode artifacts.
    let work = std::env::temp_dir().join(format!("peval_in_sandbox_{}", std::process::id()));
    let target = work.join("target");
    std::fs::create_dir_all(&target).expect("create target dir");

    // Emit per-crate bitcode for the whole dependency closure. Building a `lib` crate-type means no
    // final executable link, so cargo exits cleanly even though `malloc`/`free`/`write` are undefined
    // (the on-ramp synthesizes them). We still tolerate a non-zero status and check for the `.bc`.
    let status = Command::new("cargo")
        .current_dir(fixture_dir())
        .env("RUSTFLAGS", "--emit=llvm-bc")
        .env("CARGO_TARGET_DIR", &target)
        .args(["+1.81.0", "build", "--release", "--ignore-rust-version"])
        .status()
        .expect("run cargo build for the probe fixture");
    if !status.success() {
        eprintln!("note: probe `cargo build` returned {status} (tolerated if .bc emitted)");
    }

    let deps = target.join("release/deps");
    let mut bcs: Vec<PathBuf> = std::fs::read_dir(&deps)
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|x| x == "bc").unwrap_or(false))
        .collect();
    bcs.sort();
    if bcs.is_empty() {
        eprintln!("note: skipping peval_in_sandbox (no .bc emitted — build failed before codegen)");
        return None;
    }

    let linked = work.join("linked.bc");
    let ok = Command::new("llvm-link-18")
        .args(&bcs)
        .arg("-o")
        .arg(&linked)
        .status()
        .expect("run llvm-link-18")
        .success();
    assert!(ok, "llvm-link-18 failed");

    let legalized = work.join("probe.bc");
    let ok = Command::new("opt-18")
        .args([
            "-passes=internalize,globaldce",
            "-internalize-public-api-list=main,malloc,free",
        ])
        .arg(&linked)
        .arg("-o")
        .arg(&legalized)
        .status()
        .expect("run opt-18")
        .success();
    assert!(ok, "opt-18 failed");
    Some(legalized)
}

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
    let Some(bc) = build_probe_bc() else {
        return; // toolchain unavailable — skip
    };
    let t =
        svm_llvm::translate_bc_path(&bc).expect("translate the in-sandbox specializer to svm-IR");
    assert!(
        svm_run::is_powerbox_entry(&t.module),
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
