//! Cross-engine SVM benchmark driven by the **real LLVM frontend** (D54): compile the shared
//! `bench/cross-engine/kernels.c` with `clang -O2 -emit-llvm` (vectorization on — the on-ramp
//! legalizes `<N x T>` to v128), translate the bitcode to SVM IR via [`svm_llvm::translate_bc_path`],
//! and time
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
    ("alu", "alu"), // demonstrator: clang collapses the LCG (M^4); svm-jit doesn't (~8x)
    ("xorshift", "xorshift"), // representative scalar throughput (svm-jit ~= native)
    ("call", "call"),
    ("call_indirect", "call_indirect"),
    ("mem", "mem"),
    ("chase", "chase"),
    ("chase_rand", "chase_rand"),
    ("fnv", "fnv"),
    ("fma", "fma_k"),
    ("vadd", "vadd"), // vectorizable: the on-ramp emits v128, svm-jit lowers 128-bit SIMD
];

// Per-iteration compute is isolated by large/small-`n` subtraction. The JIT row compiles **once**
// (`svm_jit::compile` → reuse `CompiledModule::run`), so its timed loop carries no Cranelift codegen;
// the tree-walk + bytecode rows still re-drive their engine each call, but their per-iter cost dwarfs
// that fixed setup, which the subtraction cancels anyway.
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

    // Vectorization ON (plain -O2 → SSE-width <4 x i32> → one v128): the on-ramp legalizes to v128
    // (ISSUES.md I2) so `vadd` reaches svm-jit as real 128-bit SIMD. No -mavx2: a wider <8 x i32> *does*
    // legalize + lower now (it splits into two 128-bit chunks — I2/I11), but the chunks stay 128-bit, so
    // it buys no throughput over <4 x i32> while muddying the width comparison. 128-bit is the SVM
    // determinism width anyway (ISSUES.md I8); host-native width would be an opt-in non-deterministic
    // mode (see DESIGN.md §17).
    let ok = Command::new("clang")
        .args(["-O2", "-emit-llvm", "-c"])
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

        let jit = {
            // Compile **once** and time many `run`s (DESIGN.md §22 compile/run split). The one-shot
            // `compile_and_run` recompiles every call (~5–6 ms of Cranelift codegen), whose jitter
            // swamps a fast vectorized loop's signal even through the large/small subtraction (the two
            // min-over-reps compiles don't cancel exactly). Compiling once makes the JIT row honest —
            // the `vadd` 128-bit-SIMD number, not compile noise.
            let mut cm = svm_jit::compile(&t.module, e).expect("jit compiles");
            per_iter(|n| {
                let r = cm.run(&[sp, n as i64], None, None, None).expect("jit runs");
                black_box(&r);
            })
        };
        println!("svm-jit,{disp},{jit:.4}");
    }
    let _ = std::fs::remove_file(&bc);
}

/// Per-iteration compute (ns), isolated by large/small-`n` subtraction, min over reps.
fn per_iter(mut run_one: impl FnMut(i32)) -> f64 {
    let mut m = |n: i32| {
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
