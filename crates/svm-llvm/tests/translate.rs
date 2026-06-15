//! Milestone-0 first light: compile C through *stock clang* to legalized bitcode, translate it
//! to SVM IR, **verify** it (the untrusted-frontend re-check, §2a), and run it on the reference
//! interpreter — asserting the observable result. This is the first rung of the chibicc-as-oracle
//! differential (LLVM.md §5): native-clang semantics flowing through the LLVM on-ramp.

use std::path::PathBuf;
use std::process::Command;

use svm_interp::Value;

/// Compile a C snippet to legalized LLVM bitcode with the pinned pipeline (LLVM.md §4):
/// `-O2` runs `mem2reg`/SROA (the §3a two-stack split for free); `-fno-*-vectorize` keeps SIMD
/// out of the MVP. Returns `None` (skip, don't fail) when `clang` is unavailable, matching the
/// `assert_demo_matches_cc` convention.
fn compile_to_bc(name: &str, src: &str) -> Option<PathBuf> {
    let dir = std::env::temp_dir();
    let c = dir.join(format!("svm_llvm_{}_{}.c", std::process::id(), name));
    let bc = dir.join(format!("svm_llvm_{}_{}.bc", std::process::id(), name));
    std::fs::write(&c, src).expect("write C source");
    let status = Command::new("clang")
        .args([
            "-O2",
            "-emit-llvm",
            "-c",
            "-fno-vectorize",
            "-fno-slp-vectorize",
        ])
        .arg(&c)
        .arg("-o")
        .arg(&bc)
        .status();
    match status {
        Ok(s) if s.success() => Some(bc),
        _ => {
            eprintln!("note: skipping {name} (clang unavailable)");
            None
        }
    }
}

/// Translate → verify → run on the interpreter, returning the result values.
fn run(name: &str, src: &str, args: &[Value]) -> Option<Vec<Value>> {
    let bc = compile_to_bc(name, src)?;
    let module = svm_llvm::translate_bc_path(&bc).expect("translate bitcode");
    svm_verify::verify_module(&module).expect("verify translated IR");
    let mut fuel = 1_000_000u64;
    Some(svm_interp::run(&module, 0, args, &mut fuel).expect("interp run"))
}

#[test]
fn returns_constant() {
    // `ret i32 42` — the absolute floor: a single block, no instructions, a constant return.
    let Some(out) = run("ret_const", "int main(void){ return 42; }", &[]) else {
        return;
    };
    assert_eq!(out, vec![Value::I32(42)]);
}

#[test]
fn integer_add_with_params() {
    // External `f` survives `-O2` (kept for linkage) as a single block: two i32 params, one
    // `add`, a `ret` — exercises param mapping, a real instruction walk, and constant-free flow.
    let Some(out) = run(
        "add",
        "int f(int a, int b){ return a + b; }",
        &[Value::I32(40), Value::I32(2)],
    ) else {
        return;
    };
    assert_eq!(out, vec![Value::I32(42)]);
}

#[test]
fn i64_arithmetic() {
    let Some(out) = run(
        "mul64",
        "long g(long a, long b){ return a * b - 1; }",
        &[Value::I64(7), Value::I64(6)],
    ) else {
        return;
    };
    assert_eq!(out, vec![Value::I64(41)]);
}

#[test]
fn unsupported_is_fail_closed() {
    // A float return is outside the Milestone-0 subset — it must be a clean `Unsupported`, never
    // a silent mis-translation (LLVM.md §2/§8, the fail-closed chokepoint).
    let Some(bc) = compile_to_bc("float", "float h(void){ return 1.5f; }") else {
        return;
    };
    match svm_llvm::translate_bc_path(&bc) {
        Err(svm_llvm::Error::Unsupported(_)) => {}
        other => panic!("expected Unsupported, got {other:?}"),
    }
}
