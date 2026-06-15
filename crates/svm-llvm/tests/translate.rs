//! Translator tests: compile C through *stock clang* to legalized bitcode, translate it to SVM
//! IR, **verify** it (the untrusted-frontend re-check, §2a), and run it on **both** the reference
//! interpreter and the Cranelift JIT — asserting they agree with each other and with the
//! hand-computed result. This is the chibicc-as-oracle differential (LLVM.md §5) plus the §18
//! interp↔JIT differential, applied to the LLVM on-ramp.

use std::path::PathBuf;
use std::process::Command;

use svm_interp::Value;
use svm_ir::ValType;
use svm_jit::JitOutcome;

/// Compile a C snippet to legalized LLVM bitcode with the pinned pipeline (LLVM.md §4):
/// `-O2` runs `mem2reg`/SROA (the §3a two-stack split for free); `-fno-*-vectorize` keeps SIMD
/// out of the MVP. Returns `None` (skip, don't fail) when `clang` is unavailable.
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

fn to_slot(v: &Value) -> i64 {
    match v {
        Value::I32(x) => *x as i64,
        Value::I64(x) => *x,
        other => panic!("unsupported arg {other:?}"),
    }
}

fn from_slot(t: ValType, s: i64) -> Value {
    match t {
        ValType::I32 => Value::I32(s as i32),
        ValType::I64 => Value::I64(s),
        other => panic!("unsupported result type {other:?}"),
    }
}

/// Translate `src` (one defined function, the unit under test at index 0), verify, and run on
/// **both** backends with `args`; assert they agree and equal `expect`. Returns silently if clang
/// is unavailable.
fn check(name: &str, src: &str, args: &[Value], expect: &[Value]) {
    let Some(bc) = compile_to_bc(name, src) else {
        return;
    };
    let module = svm_llvm::translate_bc_path(&bc).expect("translate bitcode");
    svm_verify::verify_module(&module).expect("verify translated IR");
    let results = module.funcs[0].results.clone();

    let mut fuel = 10_000_000u64;
    let interp = svm_interp::run(&module, 0, args, &mut fuel).expect("interp run");
    assert_eq!(interp, expect, "{name}: interp result");

    let slots: Vec<i64> = args.iter().map(to_slot).collect();
    let jit = match svm_jit::compile_and_run(&module, 0, &slots).expect("jit run") {
        JitOutcome::Returned(s) => s
            .iter()
            .zip(&results)
            .map(|(&v, &t)| from_slot(t, v))
            .collect::<Vec<_>>(),
        other => panic!("{name}: unexpected JIT outcome {other:?}"),
    };
    assert_eq!(jit, expect, "{name}: JIT result (interp said {interp:?})");
}

#[test]
fn returns_constant() {
    check(
        "ret_const",
        "int main(void){ return 42; }",
        &[],
        &[Value::I32(42)],
    );
}

#[test]
fn integer_add_with_params() {
    check(
        "add",
        "int f(int a, int b){ return a + b; }",
        &[Value::I32(40), Value::I32(2)],
        &[Value::I32(42)],
    );
}

#[test]
fn i64_arithmetic() {
    check(
        "mul64",
        "long g(long a, long b){ return a * b - 1; }",
        &[Value::I64(7), Value::I64(6)],
        &[Value::I64(41)],
    );
}

#[test]
fn icmp_select_zext() {
    // -O2 if-converts this to icmp + zext i1 + select (a single block) — exercises the
    // comparison, the i1→i32 zero-extend, and branchless select.
    let src = "int classify(int x){ if (x < 0) return -1; if (x == 0) return 0; return 1; }";
    check("classify_neg", src, &[Value::I32(-5)], &[Value::I32(-1)]);
    check("classify_zero", src, &[Value::I32(0)], &[Value::I32(0)]);
    check("classify_pos", src, &[Value::I32(7)], &[Value::I32(1)]);
}

#[test]
fn loop_with_phi_popcount() {
    // A data-dependent loop -O2 cannot close-form: a multi-block CFG with φ-nodes for the loop
    // variables — the SSA→block-argument conversion (the slice-A headline).
    let src = "int popcount(unsigned x){ int c = 0; while (x) { c += x & 1; x >>= 1; } return c; }";
    check("pop_ff", src, &[Value::I32(0xFF)], &[Value::I32(8)]);
    check("pop_0", src, &[Value::I32(0)], &[Value::I32(0)]);
    check("pop_mix", src, &[Value::I32(0x10101)], &[Value::I32(3)]);
}

#[test]
fn loop_with_branchy_body() {
    // Collatz: a loop whose body branches (if/else) — back-edges, a join with φ, and nested
    // control flow, all over i32.
    let src = "int collatz(int n){ int c = 0; while (n != 1) { if (n & 1) n = 3*n + 1; else n = n/2; c++; } return c; }";
    check("collatz_6", src, &[Value::I32(6)], &[Value::I32(8)]);
    check("collatz_1", src, &[Value::I32(1)], &[Value::I32(0)]);
    check("collatz_27", src, &[Value::I32(27)], &[Value::I32(111)]);
}

#[test]
fn shifts_and_divrem() {
    let src = "int mix(int a, int b){ return (a << 3) | (a >> 1) ^ (a / b) + (a % b); }";
    // a=29,b=5: (29<<3)=232, (29>>1)=14, 29/5=5, 29%5=4 -> 5+4=9; 14^9=7; 232|7=239
    check(
        "mix",
        src,
        &[Value::I32(29), Value::I32(5)],
        &[Value::I32(239)],
    );
}

#[test]
fn unsupported_is_fail_closed() {
    // A float return is outside slice A — it must be a clean `Unsupported`, never a silent
    // mis-translation (LLVM.md §2/§8, the fail-closed chokepoint).
    let Some(bc) = compile_to_bc("float", "float h(void){ return 1.5f; }") else {
        return;
    };
    match svm_llvm::translate_bc_path(&bc) {
        Err(svm_llvm::Error::Unsupported(_)) => {}
        other => panic!("expected Unsupported, got {other:?}"),
    }
}
