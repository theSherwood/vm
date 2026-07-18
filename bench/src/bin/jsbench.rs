//! **svm-peval on a real JS engine.** Does the partial evaluator / IR optimizer move the needle on
//! JavaScript-script perf? We run a fixed JS program on the QuickJS engine (the committed
//! `qjs_repl.svmb`, native-JIT tier via `run_powerbox`) in three configurations and time each:
//!
//!   1. **baseline**  — the module as translated.
//!   2. **optimized** — after `svm_opt::optimize_module` (svm-peval's generic IR→IR optimizer:
//!      const-fold / DCE / block-merge / devirtualize).
//!   3. **specialized** — an *attempt* at the first Futamura projection (`specialize_module`): fold
//!      the engine against its startup.
//!
//! **Finding (2026-07-18).** `svm_peval` does not help real-JS-via-QuickJS perf — it is a
//! small-clean-kernel tool, and QuickJS (1176 funcs / ~250k insts) is neither. `specialize(_start)`
//! **refuses in ~11 ms with `Unsupported`**: `_start` uses constructs outside the specializer's
//! symbolic-execution subset (the powerbox ABI, `cap.call`/`call_indirect`, a real allocator's memory
//! patterns), so it never reaches the point where a const JS region would matter. And `optimize_module`
//! churns for **>8 min without completing** on a module this large — not a usable lever. By contrast the
//! toy register-machine Futamura bench (`svm-peval/tests/bench.rs`) folds the dispatch away for a ~5×
//! JIT speedup. Partial-evaluating a *production* interpreter (runtime `malloc`/GC ⇒ pointer-dependent
//! values) has poor binding-time separation — the classic obstacle. Real QuickJS speed comes from the
//! execution tiers (tree-walker → native-JIT / wasm-JIT ≈6×), not peval. Use `JSBENCH_SPEC=1` to run
//! only the (fast) specialize attempt; `JSBENCH_OPTONLY=1` to time the (very slow) optimize pass.
//!
//! Run from `bench/`:  cargo run --release --bin jsbench [-- script.js]

use std::path::PathBuf;
use std::time::{Duration, Instant};

use svm_ir::Module;
use svm_run::{run_powerbox, specialize_module, SpecArg, SpecializeOpts};

fn sizes(m: &Module) -> (usize, usize, usize) {
    let blocks = m.funcs.iter().map(|f| f.blocks.len()).sum();
    let insts = m
        .funcs
        .iter()
        .flat_map(|f| &f.blocks)
        .map(|b| b.insts.len())
        .sum();
    let bytes = svm_encode::encode_module(m).len();
    (blocks, insts, bytes)
}

/// Min wall-clock over `reps` of a full `run_powerbox` (JIT compile + execute) on the script; also
/// returns the stdout of the first run so the configs can be checked identical.
fn time_run(m: &Module, stdin: &[u8], reps: u32) -> (Duration, Vec<u8>) {
    let mut best = Duration::MAX;
    let mut out = Vec::new();
    for r in 0..reps {
        let t = Instant::now();
        let run = run_powerbox(m, stdin).expect("run_powerbox");
        let dt = t.elapsed();
        if dt < best {
            best = dt;
        }
        if r == 0 {
            out = run.stdout;
        }
    }
    (best, out)
}

fn main() {
    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
    let asset = repo.join("browser/web/assets/qjs_repl.svmb");
    let bytes = std::fs::read(&asset).expect("read qjs_repl.svmb");
    let m0 = svm_encode::decode_module(&bytes).expect("decode");
    let module = svm_run::resolve_capability_imports(m0).expect("resolve caps");
    svm_verify::verify_module(&module).expect("verify base");

    // The JS workload: a compute-heavy loop so run-time dominates the fixed JIT-compile cost.
    let js = std::env::args()
        .nth(1)
        .map(|p| std::fs::read(p).expect("read script"))
        .unwrap_or_else(|| {
            b"let s=0; for(let i=0;i<200000;i++){ s=(s+i*i)%1000003; } print(s);".to_vec()
        });
    let reps = 3u32;
    // JSBENCH_OPTONLY: skip the (slow) timed JIT runs and only measure the peval passes themselves —
    // the optimize pass and the specialization attempt — on the full QuickJS module.
    let opt_only = std::env::var_os("JSBENCH_OPTONLY").is_some();

    let (b_blk, b_ins, b_by) = sizes(&module);
    println!(
        "QuickJS module: {b_blk} blocks, {b_ins} insts, {b_by} bytes ({} funcs)",
        module.funcs.len()
    );
    println!(
        "workload: {} bytes of JS, {reps} reps (min wall-clock, JIT compile+run)\n",
        js.len()
    );

    use std::io::Write;
    // JSBENCH_SPEC: run *only* the specialization attempt (the interesting Futamura question) and exit,
    // skipping the slow baseline runs and the even-slower optimize pass.
    if std::env::var_os("JSBENCH_SPEC").is_some() {
        let nparams = module.funcs[0].params.len();
        let opts = SpecializeOpts {
            func: 0,
            args: vec![SpecArg::Dynamic; nparams],
            const_regions: vec![],
            rename: None,
            rename_private: false,
            optimize: true,
            outline: false,
            selective_outline: false,
        };
        println!("specialize(_start) attempt (first Futamura projection):");
        std::io::stdout().flush().ok();
        let t = Instant::now();
        match specialize_module(&module, &opts) {
            Ok(res) => {
                let (s_blk, s_ins, s_by) = sizes(&res);
                println!(
                    "  residual: {s_blk} blocks, {s_ins} insts, {s_by} bytes  (in {:.0} ms)",
                    t.elapsed().as_secs_f64() * 1e3
                );
            }
            Err(e) => println!(
                "  specialization refused: {e}  (in {:.0} ms) — the expected outcome for a production interpreter",
                t.elapsed().as_secs_f64() * 1e3
            ),
        }
        return;
    }

    // 1) baseline
    let base_t = if opt_only {
        Duration::from_secs(0)
    } else {
        let (base_t, base_out) = time_run(&module, &js, reps);
        println!(
            "baseline    : {:>8.1} ms   out={:?}",
            base_t.as_secs_f64() * 1e3,
            String::from_utf8_lossy(&base_out).trim()
        );
        std::io::stdout().flush().ok();
        base_t
    };

    // 2) optimized
    let t = Instant::now();
    let opt = svm_opt::optimize_module(&module);
    let opt_pass_ms = t.elapsed().as_secs_f64() * 1e3;
    let (o_blk, o_ins, o_by) = sizes(&opt);
    println!(
        "  optimize pass: {opt_pass_ms:.0} ms; size {b_blk}->{o_blk} blocks ({:+.1}%), {b_ins}->{o_ins} insts ({:+.1}%), {b_by}->{o_by} bytes",
        (o_blk as f64 / b_blk as f64 - 1.0) * 100.0,
        (o_ins as f64 / b_ins as f64 - 1.0) * 100.0,
    );
    std::io::stdout().flush().ok();
    if !opt_only {
        svm_verify::verify_module(&opt).expect("verify optimized");
        let (opt_t, opt_out) = time_run(&opt, &js, reps);
        println!(
            "optimized   : {:>8.1} ms   out={:?}   ({:+.1}% vs baseline)",
            opt_t.as_secs_f64() * 1e3,
            String::from_utf8_lossy(&opt_out).trim(),
            (opt_t.as_secs_f64() / base_t.as_secs_f64() - 1.0) * 100.0,
        );
        std::io::stdout().flush().ok();
    }

    // 3) specialization attempt (the first Futamura projection on _start). All powerbox handles are
    // Dynamic (runtime capabilities); no const-region promise, since the JS is read from stdin.
    let nparams = module.funcs[0].params.len();
    let opts = SpecializeOpts {
        func: 0,
        args: vec![SpecArg::Dynamic; nparams],
        const_regions: vec![],
        rename: None,
        rename_private: false,
        optimize: true,
        outline: false,
        selective_outline: false,
    };
    println!("\nspecialize(_start) attempt (first Futamura projection):");
    let t = Instant::now();
    match specialize_module(&module, &opts) {
        Ok(res) => {
            let (s_blk, s_ins, s_by) = sizes(&res);
            println!(
                "  residual: {s_blk} blocks, {s_ins} insts, {s_by} bytes  (in {:.0} ms)",
                t.elapsed().as_secs_f64() * 1e3
            );
        }
        Err(e) => println!("  specialization refused: {e}  (in {:.0} ms) — the expected outcome for a production interpreter", t.elapsed().as_secs_f64() * 1e3),
    }
}
