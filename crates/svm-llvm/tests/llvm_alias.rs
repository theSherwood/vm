//! **On-ramp LLVM function-alias resolution (regression for gap #12).**
//!
//! Identical-code folding — LLVM's own pass *and* Rust's cross-crate dedup — collapses functions with
//! byte-identical bodies into one definition plus an `@x = alias <ty>, ptr @y`. An alias has no body,
//! so a `call` to it once looked like an undefined external (`Unsupported("call to external/undefined
//! function …")`); the on-ramp now registers each function alias under its aliasee's index. The
//! `peval_jit` end-to-end test exercises this, but only with the heavy `rustc +1.81`/`svm-peval`
//! toolchain. This test pins the behaviour directly with a hand-written `.ll`: `@entry` calls
//! `@aliasfn`, an alias of `@real`, and the result must flow through to `@real` on **both** backends.
//!
//! Gated only on `llvm-as-18` (far lighter than the on-ramp's `rustc +1.81` lane).

use std::path::PathBuf;
use svm_ir::ValType;
use svm_jit::JitOutcome;

/// Write `ll` to a temp `.ll` for the in-house textual reader (no `llvm-as` round-trip). Always
/// `Some` — kept `Option`-returning so the call sites read like the other harnesses.
fn assemble(name: &str, ll: &str) -> Option<PathBuf> {
    let dir = std::env::temp_dir();
    let llp = dir.join(format!("svm_alias_{}_{}.ll", std::process::id(), name));
    std::fs::write(&llp, ll).expect("write .ll");
    Some(llp)
}

/// A function alias (`@aliasfn` → `@real`) called by `@entry`. `@real(x) = x + 100`;
/// `@entry(x, y) = aliasfn(x) + y`. With alias resolution, `entry(5, 1) = (5+100) + 1 = 106`.
const ALIAS_LL: &str = r#"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @real(i32 %x) {
entry:
  %r = add i32 %x, 100
  ret i32 %r
}

@aliasfn = alias i32 (i32), ptr @real

define i32 @entry(i32 %x, i32 %y) {
entry:
  %c = call i32 @aliasfn(i32 %x)
  %r = add i32 %c, %y
  ret i32 %r
}
"#;

#[test]
fn on_ramp_resolves_function_alias() {
    let Some(bc) = assemble("alias", ALIAS_LL) else {
        return; // toolchain unavailable — skip
    };
    // Before the fix this is `Unsupported("call to external/undefined function `aliasfn`")`.
    let t = svm_llvm::translate_ll_path(&bc).expect("translate bitcode with a function alias");
    let module = t.module;
    svm_verify::verify_module(&module).expect("verify translated IR");

    // `@entry` has the distinct signature `(i64 sp, i32, i32) -> i32` (the on-ramp prepends `sp`).
    let entry = module
        .funcs
        .iter()
        .position(|f| {
            f.params == [ValType::I64, ValType::I32, ValType::I32] && f.results == [ValType::I32]
        })
        .expect("entry(i32,i32) present") as u32;

    let full = vec![
        svm_interp::Value::I64(t.entry_sp as i64),
        svm_interp::Value::I32(5),
        svm_interp::Value::I32(1),
    ];
    let mut fuel = 10_000_000u64;
    let interp = match svm_interp::run(&module, entry, &full, &mut fuel)
        .expect("interp run")
        .as_slice()
    {
        [svm_interp::Value::I32(x)] => *x,
        other => panic!("expected one i32, got {other:?}"),
    };
    let slots = vec![t.entry_sp as i64, 5, 1];
    let jit = match svm_jit::compile_and_run(&module, entry, &slots).expect("jit run") {
        JitOutcome::Returned(s) => s[0] as i32,
        other => panic!("unexpected JIT outcome {other:?}"),
    };
    assert_eq!(interp, jit, "interp {interp} vs JIT {jit}");
    // (5 + 100) + 1 — proving the call routed through the alias to `@real`.
    assert_eq!(interp, 106, "alias call did not resolve to @real");
}
