//! Cross-engine SVM benchmark driven by the **real LLVM frontend** (D54): compile the shared
//! `bench/cross-engine/kernels.c` with `clang -O2 -emit-llvm -fno-*-vectorize` (the on-ramp's
//! legalized subset), translate the bitcode to SVM IR via [`svm_llvm::translate_bc_path`], and time
//! each kernel on the three SVM engines — **tree-walker**, **bytecode**, **JIT**. So the SVM rows
//! reflect IR the toolchain actually produces (not hand-written IR), from the *same* C source the
//! native/wasm/js/python drivers use.
//!
//! Output is `engine,kernel,ns_per_iter` CSV (same format as `run.sh`), with per-iteration compute
//! isolated by large/small-`n` subtraction and taken as the min over reps. Run:
//!   cd crates/svm-llvm && cargo run --release --example cross_engine

use std::hint::black_box;
use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

use svm_interp::{bytecode, Value};

// (display name, exported C symbol). `fma` is `fma_k` in C (libm owns `fma`).
const KERNELS: &[(&str, &str)] = &[
    ("alu", "alu"),
    ("call", "call"),
    ("call_indirect", "call_indirect"),
    ("mem", "mem"),
    ("chase", "chase"),
    ("chase_rand", "chase_rand"),
    ("fnv", "fnv"),
    ("fma", "fma_k"),
    // vsum omitted: the on-ramp legalizes with -fno-vectorize (scalar MVP) and the opaque-pointer
    // barrier does not survive LLVM->SVM->Cranelift, so the reduction folds to a bogus ~0 ns.
];

// `svm_jit::compile_and_run` recompiles the module on every call, so the timed loop must be long
// enough that the run dominates compile jitter (frontend_bench uses the same large n for this reason).
const SMALL: i32 = 1_000;
const LARGE: i32 = 2_000_000;

fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    let kernels_c = root.join("bench/cross-engine/kernels.c");
    let bc = std::env::temp_dir().join(format!("svm_llvm_xe_{}.bc", std::process::id()));

    // The on-ramp's legalized subset (LLVM.md §4): O2 with vectorization off (the MVP is scalar).
    let ok = Command::new("clang")
        .args([
            "-O2",
            "-emit-llvm",
            "-c",
            "-fno-vectorize",
            "-fno-slp-vectorize",
        ])
        .arg(&kernels_c)
        .arg("-o")
        .arg(&bc)
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if !ok {
        eprintln!("note: clang unavailable or failed; skipping the LLVM-frontend SVM bench");
        return;
    }
    let t = svm_llvm::translate_bc_path(&bc).expect("translate kernels.c bitcode");
    let sp = t.entry_sp as i64;
    let idx = |sym: &str| -> u32 {
        t.exports
            .iter()
            .find(|(name, _)| name == sym)
            .unwrap_or_else(|| panic!("kernel `{sym}` not exported"))
            .1
    };

    for &(disp, sym) in KERNELS {
        let e = idx(sym);

        let tw = per_iter(|n| {
            let mut fuel = u64::MAX;
            let r = svm_interp::run(&t.module, e, &[Value::I64(sp), Value::I32(n)], &mut fuel);
            black_box(&r);
        });
        println!("svm-tree-walk,{disp},{tw:.4}");

        let bcn = per_iter(|n| {
            let mut fuel = u64::MAX;
            let r = bytecode::compile_and_run(
                &t.module,
                e,
                &[Value::I64(sp), Value::I32(n)],
                &mut fuel,
            )
            .expect("bytecode drives the frontend kernel");
            black_box(&r);
        });
        println!("svm-bytecode,{disp},{bcn:.4}");

        let jit = per_iter(|n| {
            let r = svm_jit::compile_and_run(&t.module, e, &[sp, n as i64]).expect("jit runs");
            black_box(&r);
        });
        println!("svm-jit,{disp},{jit:.4}");
    }
    let _ = std::fs::remove_file(&bc);
}

/// Per-iteration compute (ns), isolated by large/small-`n` subtraction, min over reps.
fn per_iter(run_one: impl Fn(i32)) -> f64 {
    let m = |n: i32| {
        run_one(n); // warm up (JIT compile, caches)
        let mut best = f64::MAX;
        for _ in 0..9 {
            let t = Instant::now();
            run_one(n);
            best = best.min(t.elapsed().as_nanos() as f64);
        }
        best
    };
    (m(LARGE) - m(SMALL)) / (LARGE - SMALL) as f64
}
