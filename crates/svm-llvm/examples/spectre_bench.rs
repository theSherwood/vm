//! Confinement-lowering microbench (JIT lane only) — built for the TRAP_CONFINEMENT.md
//! Spectre-v1 decision, where a temporary `SVM_CONFINE` env gate in `svm-jit` let one binary
//! A/B/C/D the candidate lowerings (mask / check / check+cmov / check+AND-clamp) on identical
//! LLVM-frontend IR; the measurements are recorded there and the gate is gone (check+clamp is
//! the one production lowering). Kept as a general per-access-cost probe: compiles a C kernel
//! file (`clang -O2 -emit-llvm` → `svm_llvm`), JIT-compiles each kernel once, and prints
//! `kernel,ns_per_iter` with an interp-vs-JIT correctness pin. Usage:
//!   cargo run --release --example spectre_bench -- <kernels.c> <sym>[:<large-n>][,<sym>...]

use std::hint::black_box;
use std::process::Command;
use std::time::Instant;

const SMALL: i32 = 1_000;
const LARGE: i32 = 2_000_000;
// `sym:LARGE` overrides the large-n count for heavyweight kernels (e.g. matmul, where one "iter"
// is a whole O(N^3) multiply).

fn main() {
    let mut args = std::env::args().skip(1);
    let src = args
        .next()
        .expect("usage: spectre_bench <kernels.c> <sym,sym,...>");
    let syms = args
        .next()
        .expect("usage: spectre_bench <kernels.c> <sym,sym,...>");
    let bc = std::env::temp_dir().join(format!("svm_spectre_bench_{}.ll", std::process::id()));
    let ok = Command::new("clang")
        .args(["-O2", "-emit-llvm", "-S"])
        .arg(&src)
        .arg("-o")
        .arg(&bc)
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    assert!(ok, "clang failed on {src}");
    let t = svm_llvm::translate_ll_path(&bc).expect("translate");
    let sp = t.entry_sp as i64;
    for spec in syms.split(',') {
        let (sym, large) = match spec.split_once(':') {
            Some((s, n)) => (s, n.parse::<i32>().expect("bad :LARGE")),
            None => (spec, LARGE),
        };
        let e = t
            .exports
            .iter()
            .find(|(name, _)| name == sym)
            .unwrap_or_else(|| panic!("kernel `{sym}` not exported"))
            .1;
        let mut cm = svm_jit::compile(&t.module, e).expect("jit compiles");
        // Correctness pin: the small-n JIT result must match the tree-walk oracle in every mode.
        let mut fuel = u64::MAX;
        let want = svm_interp::run(
            &t.module,
            e,
            &[svm_interp::Value::I64(sp), svm_interp::Value::I32(SMALL)],
            &mut fuel,
        );
        let (got, _mem) = cm
            .run(&[sp, SMALL as i64], None, None, None)
            .expect("jit runs");
        let got0 = match got {
            svm_jit::JitOutcome::Returned(vals) => vals[0],
            other => panic!("JIT did not return on {sym}: {other:?}"),
        };
        // Compare as i32: these kernels return i32, which the interp sign-extends and the JIT
        // zero-extends into the i64 return slot.
        let want0 = as_i64(want.expect("interp runs")[0]) as i32;
        let got0 = got0 as i32;
        assert_eq!(
            want0, got0,
            "MISCOMPILE on {sym}: interp={want0} jit={got0}"
        );
        let ns = per_iter(large, |n| {
            let r = cm.run(&[sp, n as i64], None, None, None).expect("jit runs");
            black_box(&r);
        });
        println!("{sym},{ns:.4}");
    }
    let _ = std::fs::remove_file(&bc);
}

fn as_i64(v: svm_interp::Value) -> i64 {
    match v {
        svm_interp::Value::I32(x) => x as i64,
        svm_interp::Value::I64(x) => x,
        other => panic!("unexpected {other:?}"),
    }
}

fn per_iter(large: i32, mut run_one: impl FnMut(i32)) -> f64 {
    let mut m = |n: i32| {
        run_one(n); // warm up
        let mut best = f64::MAX;
        for _ in 0..9 {
            let t = Instant::now();
            run_one(n);
            best = best.min(t.elapsed().as_nanos() as f64);
        }
        best
    };
    let small = (SMALL).min(large / 100).max(1);
    (m(large) - m(small)) / (large - small) as f64
}
