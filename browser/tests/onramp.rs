//! The **on-ramp powerbox** entry (`onramp_exec` / the wasm `svm_run_onramp` export): a `.svmb`
//! straight off `svm-llvm-translate` runs under the fixed §3e `VM_CAP_*` grant prefix the LLVM
//! on-ramp's synthesized `_start` expects — the seam that lets the browser run real C/C++ guests
//! (Lua, SQLite) the same way `svm-run` does natively.
//!
//! The fixture `fixtures/hello_onramp.svmb` is `crates/svm-run/demos/hello.c` compiled with stock
//! `clang -O2 -emit-llvm` and translated (`svm-llvm-translate hello.bc -o hello_onramp.svmb`). It
//! imports `write`/`exit` (a 3-handle `_start`), so it exercises import-lowering + the grant order
//! without any clang/LLVM dependency at test time. Larger guests (malloc → the memory cap, Lua,
//! SQLite Phase A) are verified out-of-tree via `cargo run --example run_onramp`.

use svm_browser::{onramp_exec, STATUS_OK};

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
