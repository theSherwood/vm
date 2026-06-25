//! Shared harness for the `peval_*` on-ramp tests (`peval_in_sandbox.rs`, `peval_jit.rs`,
//! `peval_futamura.rs`). Each runs the manual probe — `rustc +1.81 --emit=llvm-bc` → `llvm-link-18`
//! → `opt-18 internalize,globaldce` → translate → verify → run — on an in-repo fixture crate under
//! `tests/fixtures/<name>`. The build half is identical across them, so it lives here.
//!
//! As a `tests/common/mod.rs` submodule it is **not** compiled as its own test binary; each test does
//! `mod common;` and calls [`build_fixture_bc`].

#![allow(dead_code)] // each test binary uses only the part it needs

use std::path::{Path, PathBuf};
use std::process::Command;

/// Is `cmd <args>` runnable and successful? Used to auto-skip when a pipeline tool is absent.
fn tool_ok(cmd: &str, args: &[&str]) -> bool {
    Command::new(cmd)
        .args(args)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// True when the on-ramp toolchain (`rustc +1.81.0`, `llvm-link-18`, `opt-18`) is available.
pub fn toolchain_present() -> bool {
    tool_ok("rustc", &["+1.81.0", "--version"])
        && tool_ok("llvm-link-18", &["--version"])
        && tool_ok("opt-18", &["--version"])
}

/// Build the in-repo fixture crate `tests/fixtures/<fixture>` to a single legalized LLVM-18 bitcode
/// blob, ready for [`svm_llvm::translate_bc_path`]. Returns `None` (skip) if the toolchain is absent
/// or no bitcode is emitted.
///
/// Mirrors the manual probe exactly: emit per-crate bitcode for the whole dependency closure
/// (`RUSTFLAGS=--emit=llvm-bc cargo +1.81.0 build --release`), `llvm-link-18` them, then
/// `opt-18 internalize,globaldce` down to the closure reachable from the powerbox `main`/`malloc`/
/// `free`. Building the fixture as a `lib` means no final executable link, so cargo exits cleanly even
/// though `malloc`/`free`/`write`/`__vm_jit_*` are undefined (the on-ramp synthesizes/lowers them); we
/// still tolerate a non-zero status and check for the `.bc`.
pub fn build_fixture_bc(fixture: &str) -> Option<PathBuf> {
    if !toolchain_present() {
        eprintln!("note: skipping {fixture} (need `rustc +1.81.0`, `llvm-link-18`, `opt-18`)");
        return None;
    }

    let fixture_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(fixture);
    // A dedicated, out-of-tree target dir keeps the repo clean and isolates the bitcode artifacts.
    let work = std::env::temp_dir().join(format!("{fixture}_{}", std::process::id()));
    let target = work.join("target");
    std::fs::create_dir_all(&target).expect("create target dir");

    let status = Command::new("cargo")
        .current_dir(&fixture_dir)
        .env("RUSTFLAGS", "--emit=llvm-bc")
        .env("CARGO_TARGET_DIR", &target)
        .args(["+1.81.0", "build", "--release", "--ignore-rust-version"])
        .status()
        .unwrap_or_else(|e| panic!("run cargo build for the {fixture} fixture: {e}"));
    if !status.success() {
        eprintln!("note: {fixture} `cargo build` returned {status} (tolerated if .bc emitted)");
    }

    let deps = target.join("release/deps");
    let mut bcs: Vec<PathBuf> = std::fs::read_dir(&deps)
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|x| x == "bc").unwrap_or(false))
        .collect();
    bcs.sort();
    if bcs.is_empty() {
        eprintln!("note: skipping {fixture} (no .bc emitted — build failed before codegen)");
        return None;
    }

    let linked = work.join("linked.bc");
    assert!(
        Command::new("llvm-link-18")
            .args(&bcs)
            .arg("-o")
            .arg(&linked)
            .status()
            .expect("run llvm-link-18")
            .success(),
        "llvm-link-18 failed"
    );

    let legalized = work.join("legalized.bc");
    assert!(
        Command::new("opt-18")
            .args([
                "-passes=internalize,globaldce",
                "-internalize-public-api-list=main,malloc,free",
            ])
            .arg(&linked)
            .arg("-o")
            .arg(&legalized)
            .status()
            .expect("run opt-18")
            .success(),
        "opt-18 failed"
    );
    Some(legalized)
}
