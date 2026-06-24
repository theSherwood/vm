//! **Compile latency & engine break-even.** The cross-engine table reports *steady-state*
//! throughput — it sizes `n` huge precisely so `svm_jit::compile_and_run`'s per-call recompile
//! washes out. That hides a first-order JIT property: how long until the *first* result, and how
//! many iterations a workload must run before paying for a JIT (or even a bytecode) compile is worth
//! it versus just tree-walking.
//!
//! This driver measures, per kernel from the shared `bench/cross-engine/kernels.c` (via the real
//! LLVM frontend):
//!   * **translate latency** — one-time `svm_llvm::translate_bc_path` (LLVM-bitcode → SVM IR), shared
//!     by all three engines.
//!   * **cold cost** of each engine — the fixed per-`compile_and_run` cost (JIT codegen / bytecode
//!     compile / tree-walk frame setup), recovered as the `n → 0` intercept of `T(n) = cold + n·iter`.
//!   * **per-iter** — the steady-state slope (same number the cross-engine table reports).
//!
//! From `cold` + `iter` it then computes the **break-even iteration count** between engines: the `n`
//! where the *total* cost (compile once + run `n` times), `cold + n·iter`, of the faster-per-iter but
//! slower-to-compile engine overtakes the cheaper-to-start one. Run:
//!   cd crates/svm-llvm && cargo run --release --example compile_latency

use std::hint::black_box;
use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

use svm_interp::{bytecode, Value};

const KERNELS: &[(&str, &str)] = &[
    ("alu", "alu"),
    ("xorshift", "xorshift"),
    ("call", "call"),
    ("call_indirect", "call_indirect"),
    ("mem", "mem"),
    ("fnv", "fnv"),
    ("fma", "fma_k"),
    ("vadd", "vadd"),
];

const SMALL: i32 = 1_000;
const LARGE: i32 = 2_000_000;

/// Two-point linear fit `T(n) = cold + n·iter` over the min-of-reps timing of `run_one(n)`.
/// `iter` (ns/iteration) is the slope; `cold` (ns) is the `n → 0` intercept — the fixed
/// compile + setup cost paid once per `compile_and_run`.
fn fit(run_one: impl Fn(i32)) -> (f64, f64) {
    let m = |n: i32| {
        run_one(n); // warm up (the very first call also pays one-time global init / icache)
        let mut best = f64::MAX;
        for _ in 0..15 {
            let t = Instant::now();
            run_one(n);
            best = best.min(t.elapsed().as_nanos() as f64);
        }
        best
    };
    let (ts, tl) = (m(SMALL), m(LARGE));
    let iter = (tl - ts) / (LARGE - SMALL) as f64;
    let cold = ts - SMALL as f64 * iter;
    (cold.max(0.0), iter)
}

/// Break-even iteration count where the engine with the *cheaper per-iter* but *costlier cold*
/// overtakes the cheaper-to-start one: solve `cold_a + n·iter_a = cold_b + n·iter_b`. Returns
/// `None` when the costlier-cold engine is also slower per-iter (it never wins) or already wins at
/// `n = 0`. `(fast_cold, fast_iter)` must be the engine with the smaller `iter`.
fn break_even(slow_cold: f64, slow_iter: f64, fast_cold: f64, fast_iter: f64) -> Option<f64> {
    if fast_iter >= slow_iter {
        return None; // the "fast" engine isn't actually faster per-iter → no crossover
    }
    let n = (fast_cold - slow_cold) / (slow_iter - fast_iter);
    if n <= 0.0 {
        None // fast engine already cheaper at n = 0
    } else {
        Some(n)
    }
}

fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    let kernels_c = root.join("bench/cross-engine/kernels.c");
    let bc = std::env::temp_dir().join(format!("svm_cl_{}.bc", std::process::id()));

    let ok = Command::new("clang")
        .args(["-O2", "-emit-llvm", "-c"])
        .arg(&kernels_c)
        .arg("-o")
        .arg(&bc)
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if !ok {
        eprintln!("note: clang unavailable or failed; skipping compile-latency bench");
        return;
    }

    // One-time LLVM-frontend ingest (shared by every engine). Min over reps.
    let mut translate_ns = f64::MAX;
    for _ in 0..9 {
        let t = Instant::now();
        let tr = svm_llvm::translate_bc_path(&bc).expect("translate");
        translate_ns = translate_ns.min(t.elapsed().as_nanos() as f64);
        black_box(&tr);
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

    println!(
        "module translate (LLVM bitcode → SVM IR): {:.1} µs (one-time, shared by all engines)\n",
        translate_ns / 1e3
    );
    println!(
        "{:<14} {:>11} {:>11} {:>11} | {:>9} {:>9} {:>9}",
        "kernel", "jit cold", "bc cold", "tw cold", "jit/it", "bc/it", "tw/it"
    );
    println!(
        "{:<14} {:>11} {:>11} {:>11} | {:>9} {:>9} {:>9}",
        "", "(µs)", "(µs)", "(µs)", "(ns)", "(ns)", "(ns)"
    );

    let mut rows = Vec::new();
    for &(disp, sym) in KERNELS {
        let e = idx(sym);

        let (tw_cold, tw_iter) = fit(|n| {
            let mut fuel = u64::MAX;
            let r = svm_interp::run(&t.module, e, &[Value::I64(sp), Value::I32(n)], &mut fuel);
            black_box(&r);
        });
        let (bc_cold, bc_iter) = fit(|n| {
            let mut fuel = u64::MAX;
            let r = bytecode::compile_and_run(
                &t.module,
                e,
                &[Value::I64(sp), Value::I32(n)],
                &mut fuel,
            )
            .expect("bytecode");
            black_box(&r);
        });
        let (jit_cold, jit_iter) = fit(|n| {
            let r = svm_jit::compile_and_run(&t.module, e, &[sp, n as i64]).expect("jit");
            black_box(&r);
        });

        println!(
            "{disp:<14} {:>11.1} {:>11.1} {:>11.1} | {:>9.3} {:>9.3} {:>9.3}",
            jit_cold / 1e3,
            bc_cold / 1e3,
            tw_cold / 1e3,
            jit_iter,
            bc_iter,
            tw_iter
        );
        rows.push((disp, jit_cold, jit_iter, bc_cold, bc_iter, tw_cold, tw_iter));
    }

    // Break-even: with compile paid once, how many iterations until the faster-per-iter engine wins?
    println!(
        "\nbreak-even iterations (compile once + run n times: cold + n·iter crossover):\n{:<14} {:>14} {:>14} {:>14}",
        "kernel", "jit beats bc", "jit beats tw", "bc beats tw"
    );
    for &(disp, jc, ji, bc_c, bi, tc, ti) in &rows {
        let f = |x: Option<f64>| match x {
            Some(n) => format!("{:.0}", n.ceil()),
            None => "—".to_string(),
        };
        println!(
            "{disp:<14} {:>14} {:>14} {:>14}",
            f(break_even(bc_c, bi, jc, ji)),
            f(break_even(tc, ti, jc, ji)),
            f(break_even(tc, ti, bc_c, bi)),
        );
    }
    println!(
        "\n(“—” = the per-iter-faster engine never overtakes at any n, or already wins at n=0.\n cold costs are the n→0 intercept of T(n)=cold+n·iter; jit/bc recompile each call, so cold is\n the realistic per-invocation compile cost. The interpreters’ tiny cold is pure frame setup.)"
    );
    let _ = std::fs::remove_file(&bc);
}
