//! **Confinement-cost perf harness** — svm-jit vs Wasmtime-w64 on confinement-dense kernels, the
//! reliable/tracked instrument for the "close the gap to Wasmtime" work (DESIGN.md §1a).
//!
//! Motivation: the embench differential (`crates/svm-llvm/examples/embench.rs`) shows the gap on
//! array/memory-dense kernels (matmult/edn/picojpeg), but it needs an external Embench checkout and
//! is single-pass in CI — too noisy to tell a real 1.1× from a 1.3×. This driver is **self-contained**
//! (kernels vendored under `bench/confine/kernels/`) and **best-of-N**, so it runs locally and in CI
//! and gives a stable svm-jit÷Wasmtime-w64 ratio per kernel.
//!
//! Both engines consume the **same C source at the same widths** (LP64): svm-jit via the production
//! LLVM on-ramp (`clang -O2 -emit-llvm` → `svm_llvm` → `svm_jit`), Wasmtime via `clang --target=wasm64`
//! (memory64, `long` = 64-bit) with its default Spectre-mitigation posture on. That is the honest
//! same-backend (both Cranelift), same-widths comparison — exactly the `wt/w64` column of embench.
//!
//! Each kernel exports `long run(long n)`; per-iteration compute is isolated by the large/small-`n`
//! subtraction `(t(large) − t(small)) / Δn`, taken as the **min over reps** (the noise floor). Results
//! are cross-checked (svm-jit and Wasmtime must agree on `run(small)`) before timing, so a miscompile
//! is never benchmarked.
//!
//! Run from `bench/`:
//!   cargo run --release --bin confine                    # human table
//!   cargo run --release --bin confine -- --csv           # machine-readable: kernel,svm_ns,wt64_ns,ratio
//!   cargo run --release --bin confine -- --save baseline # write bench/confine/baseline.txt
//!   cargo run --release --bin confine -- --check baseline --tol 0.15   # exit 1 if a ratio regressed >tol

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use wasmtime::{Config, Engine, Instance, Module, Store, Val};

const REPS: u32 = 25;

/// (name, small-n, large-n). `large` sized so the large run is a few ms; matmul is O(N³)/iter so it
/// needs a far smaller count than the light streaming kernels.
const KERNELS: &[(&str, i64, i64)] = &[
    ("matmul", 100, 20_000),
    ("matmul_eb", 100, 20_000),
    ("fir", 1_000, 12_000_000),
    ("bytes", 1_000, 12_000_000),
];

fn kernels_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("confine/kernels")
}

/// Compile `kernel.c` to LP64 bitcode and translate+JIT it; return a runner closure timing `run(n)`
/// on svm-jit (compiled once, reused) plus its `run(small)` result for the cross-check.
fn svm_lane(cfile: &Path, small: i64) -> Option<(impl FnMut(i64) -> i64, i64)> {
    let bc = std::env::temp_dir().join(format!("confine_{}.bc", std::process::id()));
    let ok = Command::new("clang")
        .args(["-O2", "-emit-llvm", "-c"])
        .arg(cfile)
        .arg("-o")
        .arg(&bc)
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if !ok {
        return None;
    }
    let t = svm_llvm::translate_bc_path(&bc).ok()?;
    let _ = std::fs::remove_file(&bc);
    let sp = t.entry_sp as i64;
    let e = t.exports.iter().find(|(n, _)| n == "run")?.1;
    let mut cm = svm_jit::compile(&t.module, e).ok()?;
    let mut runner = move |n: i64| -> i64 {
        match cm.run(&[sp, n], None, None, None).expect("svm-jit run") {
            (svm_jit::JitOutcome::Returned(v), _) => v[0],
            (other, _) => panic!("svm-jit did not return: {other:?}"),
        }
    };
    let want = runner(small);
    Some((runner, want))
}

/// Compile `kernel.c` to a wasm64 (memory64) module and instantiate it on Wasmtime (Cranelift,
/// default Spectre mitigation on); return a runner timing `run(n)` plus its `run(small)` result.
fn wt_lane(cfile: &Path, small: i64) -> Option<(impl FnMut(i64) -> i64, i64)> {
    let wasm = std::env::temp_dir().join(format!("confine_{}.64.wasm", std::process::id()));
    let ok = Command::new("clang")
        .args([
            "--target=wasm64",
            "-O2",
            "-mbulk-memory", // lower memcpy/memset to wasm memory.copy/fill (matches embench build)
            "-nostdlib",
            "-Wl,--no-entry",
            "-Wl,--export=run",
            "-Wl,--gc-sections",
        ])
        .arg(cfile)
        .arg("-o")
        .arg(&wasm)
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if !ok {
        return None;
    }
    let mut cfg = Config::new();
    cfg.wasm_memory64(true);
    let engine = Engine::new(&cfg).ok()?;
    let module = Module::from_file(&engine, &wasm).ok()?;
    let _ = std::fs::remove_file(&wasm);
    let mut store = Store::new(&engine, ());
    let inst = Instance::new(&mut store, &module, &[]).ok()?;
    let f = inst.get_func(&mut store, "run")?;
    let mut out = [Val::I64(0)];
    let mut run1 = move |n: i64| -> i64 {
        f.call(&mut store, &[Val::I64(n)], &mut out)
            .expect("wt run");
        match out[0] {
            Val::I64(x) => x,
            Val::I32(x) => x as i64,
            _ => panic!("unexpected wt return"),
        }
    };
    let want = run1(small);
    Some((run1, want))
}

/// Min-over-reps per-iteration ns, isolated by the large/small subtraction.
fn per_iter(small: i64, large: i64, mut run: impl FnMut(i64) -> i64) -> f64 {
    let mut best = |n: i64| -> f64 {
        run(n); // warm up
        let mut b = f64::MAX;
        for _ in 0..REPS {
            let t = Instant::now();
            let r = run(n);
            b = b.min(t.elapsed().as_nanos() as f64);
            std::hint::black_box(r);
        }
        b
    };
    (best(large) - best(small)) / (large - small) as f64
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let csv = args.iter().any(|a| a == "--csv");
    let save = args
        .iter()
        .position(|a| a == "--save")
        .map(|i| args[i + 1].clone());
    let check = args
        .iter()
        .position(|a| a == "--check")
        .map(|i| args[i + 1].clone());
    let tol: f64 = args
        .iter()
        .position(|a| a == "--tol")
        .map(|i| args[i + 1].parse().unwrap())
        .unwrap_or(0.15);

    let dir = kernels_dir();
    let mut rows: Vec<(String, f64, f64, f64)> = Vec::new(); // name, svm_ns, wt_ns, ratio

    if !csv && save.is_none() && check.is_none() {
        println!(
            "{:<10} {:>12} {:>12} {:>10}   correctness",
            "kernel", "svm-jit(ns)", "wt/w64(ns)", "svm÷wt64"
        );
    }
    for &(name, small, large) in KERNELS {
        let cfile = dir.join(format!("{name}.c"));
        let Some((svm_run, svm_want)) = svm_lane(&cfile, small) else {
            eprintln!("note: {name}: svm-jit lane unavailable (clang/on-ramp) — skipped");
            continue;
        };
        let Some((wt_run, wt_want)) = wt_lane(&cfile, small) else {
            eprintln!("note: {name}: wasmtime-w64 lane unavailable (clang wasm64) — skipped");
            continue;
        };
        // Widths differ only in the outer-loop counter; `run(small)` must agree or we're timing a
        // miscompile in one engine.
        assert_eq!(
            svm_want, wt_want,
            "{name}: MISCOMPILE — svm-jit={svm_want} wt64={wt_want} on run({small})"
        );
        let svm_ns = per_iter(small, large, svm_run);
        let wt_ns = per_iter(small, large, wt_run);
        let ratio = svm_ns / wt_ns;
        if !csv && save.is_none() && check.is_none() {
            println!("{name:<10} {svm_ns:>12.3} {wt_ns:>12.3} {ratio:>9.3}x   OK");
        }
        rows.push((name.to_string(), svm_ns, wt_ns, ratio));
    }

    if csv {
        for (n, s, w, r) in &rows {
            println!("{n},{s:.4},{w:.4},{r:.4}");
        }
    }
    if let Some(path) = save {
        let body: String = rows
            .iter()
            .map(|(n, _, _, r)| format!("{n} {r:.4}\n"))
            .collect();
        std::fs::write(&path, body).expect("write baseline");
        eprintln!("wrote {} kernel ratios to {path}", rows.len());
    }
    if let Some(path) = check {
        let base = std::fs::read_to_string(&path).expect("read baseline");
        let mut regressed = false;
        for line in base.lines() {
            let mut it = line.split_whitespace();
            let (Some(kn), Some(bv)) = (it.next(), it.next()) else {
                continue;
            };
            let bratio: f64 = bv.parse().unwrap();
            let Some((_, _, _, now)) = rows.iter().find(|(n, ..)| n == kn) else {
                eprintln!("note: {kn} not measured this run");
                continue;
            };
            let grew = now / bratio - 1.0;
            let flag = if grew > tol {
                regressed = true;
                "REGRESSED"
            } else {
                "ok"
            };
            eprintln!(
                "{kn:<10} baseline {bratio:.3}x  now {now:.3}x  ({:+.1}%) {flag}",
                grew * 100.0
            );
        }
        if regressed {
            eprintln!(
                "confine: a svm÷wt64 ratio grew by more than {:.0}%",
                tol * 100.0
            );
            std::process::exit(1);
        }
    }
}
