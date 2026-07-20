//! The **on-ramp powerbox** entry (`onramp_exec` / the wasm `svm_run_onramp` export): a `.svmb`
//! straight off `svm-llvm-translate` runs under the fixed §3e `VM_CAP_*` grant prefix the LLVM
//! on-ramp's synthesized `_start` expects — the seam that lets the browser run real C/C++ guests
//! (Lua, SQLite) the same way `svm-run` does natively.
//!
//! The fixture `fixtures/hello_onramp.svmb` is `crates/svm-run/demos/hello.c` compiled with stock
//! `clang -O2 -emit-llvm` and translated (`svm-llvm-translate hello.bc -o hello_onramp.svmb`). The
//! current on-ramp emits the **by-name** paramless `_start` (S15), whose manifest imports
//! (`write`/`exit`) bind to slot bindings at instantiation (`grant_onramp_caps`; IMPORTS.md phase 4
//! — the positional handle-args entry form is gone, and an import-bearing module without the
//! manifest entry shape fails closed). Larger guests (malloc → the memory cap, Lua, SQLite Phase A)
//! are verified out-of-tree via `cargo run --example run_onramp`.

use svm_browser::{onramp_exec, STATUS_OK, STATUS_UNSUPPORTED};

/// A pre-manifest **legacy positional** entry shape: func 0 takes its `write` handle as a
/// parameter (slot-order delivery) and is not exported as `_start`. Phase 4 deleted the
/// `resolve_imports` rewrite that used to accept this, so an import-bearing module without the
/// manifest entry shape (paramless func 0 exported `_start`) must **fail closed**.
const LEGACY_POSITIONAL: &str = r#"memory 16
func (i32) -> (i64) {
block0(v0: i32):
  v1 = i64.const 0
  v2 = i64.const 5
  v3 = call.import "write" (i64, i64) -> (i64) v0 (v1, v2)
  return v3
}
"#;

#[test]
fn legacy_positional_import_blob_is_rejected() {
    let m = svm_text::parse_module(LEGACY_POSITIONAL).expect("parse legacy module");
    assert!(
        !m.imports.is_empty(),
        "the fixture must actually carry an import manifest"
    );
    let out = onramp_exec(&m, b"");
    assert_eq!(
        out.status, STATUS_UNSUPPORTED,
        "an import-bearing module without the manifest entry shape fails closed (phase 4)"
    );
}

#[test]
fn hello_onramp_prints_through_the_powerbox() {
    let bytes = include_bytes!("fixtures/hello_onramp.svmb");
    let m = svm_encode::decode_module(bytes).expect("decode hello_onramp.svmb");
    let out = onramp_exec(&m, b"");
    assert_eq!(out.status, STATUS_OK, "on-ramp guest should run cleanly");
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "hello, sandbox!\n",
        "the guest's write(1, …) must reach the captured stdout stream",
    );
}
