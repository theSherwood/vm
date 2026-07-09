//! **Confinement-cost perf harness** — a cross-engine table on confinement-dense kernels, the
//! reliable/tracked instrument for the "close the gap to Wasmtime" work (DESIGN.md §1a).
//!
//! Motivation: the embench differential (`crates/svm-llvm/examples/embench.rs`) shows the gap on
//! array/memory-dense kernels (matmult/edn/picojpeg), but it needs an external Embench checkout and
//! is single-pass in CI — too noisy to tell a real 1.1× from a 1.3×. This driver is **self-contained**
//! (kernels vendored under `bench/confine/kernels/`) and **best-of-N**, so it runs locally and in CI
//! and gives stable per-kernel ratios. The headline is **svm-jit ÷ Wasmtime-w64** (the tracked
//! baseline gate); the other columns mirror embench's cross-engine set.
//!
//! Columns (each engine as ×native, like embench):
//!   native   `clang -O2 -march=native` — the oracle, self-timed (per-iter ns via large/small subtraction)
//!   svm-jit  production LLVM on-ramp (`clang -O2 -emit-llvm` → `svm_llvm` → `svm_jit`), in-process
//!   svm-bc   the bytecode interpreter (`svm_interp::bytecode`), in-process — the same IR, tree-executed
//!   v8/w32   `clang --target=wasm32` on V8 (Node), self-timed via `bench/embench/wasm/run.mjs`
//!   v8/w64   `clang --target=wasm64` (memory64) on V8 (Node)
//!   wt/w32   `clang --target=wasm32` on in-process Wasmtime (Cranelift)
//!   wt/w64   `clang --target=wasm64` (memory64) on in-process Wasmtime — the same widths + backend as
//!            svm-jit, so **svm÷wt64** is the honest Cranelift-vs-Cranelift, same-LP64-widths comparison.
//!
//! Every lane isolates per-iteration compute by the large/small-`n` subtraction `(t(large) − t(small))
//! / Δn`, taken as the **min over reps** (the noise floor), and every engine's `run(small)` is
//! cross-checked against native before timing, so a miscompile is never benchmarked. The interpreter
//! is ~600× slower, so its `large` is **calibrated down** to a bounded per-run budget (its per-iter is
//! n-independent, so a smaller delta measures the same rate). Missing tools (no `node`, no wasm32
//! target) just leave that column blank.
//!
//! Run from `bench/`:
//!   cargo run --release --bin confine                    # human table
//!   cargo run --release --bin confine -- --csv           # machine-readable: kernel,svm_ns,wt64_ns,ratio
//!   cargo run --release --bin confine -- --save baseline # write bench/confine/baseline.txt
//!   cargo run --release --bin confine -- --check baseline --tol 0.15   # exit 1 if svm÷wt64 regressed >tol

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use svm_interp::{bytecode, Value};
use wasmtime::{Config, Engine, Instance, Module, Store, Val};

const REPS: u32 = 25;
/// The bytecode interpreter is ~600× native; cap each of its timed runs to roughly this many seconds
/// (its per-iter is n-independent, so the calibrated-down delta measures the same rate). Fewer reps
/// too — it is deterministic, so the min converges fast.
const BC_TARGET_S: f64 = 0.3;
const BC_REPS: u32 = 5;

/// (name, small-n, large-n). `large` sized so the large run is a few ms; matmul is O(N³)/iter so it
/// needs a far smaller count than the light streaming kernels.
const KERNELS: &[(&str, i64, i64)] = &[
    ("matmul", 100, 20_000),
    ("matmul_eb", 100, 20_000),
    ("fir", 1_000, 12_000_000),
    ("bytes", 1_000, 12_000_000),
];

fn confine_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("confine")
}
fn kernels_dir() -> PathBuf {
    confine_dir().join("kernels")
}

/// Min-over-reps per-iteration ns, isolated by the large/small subtraction.
fn per_iter(small: i64, large: i64, reps: u32, mut run: impl FnMut(i64) -> i64) -> f64 {
    let mut best = |n: i64| -> f64 {
        run(n); // warm up
        let mut b = f64::MAX;
        for _ in 0..reps {
            let t = Instant::now();
            let r = run(n);
            b = b.min(t.elapsed().as_nanos() as f64);
            std::hint::black_box(r);
        }
        b
    };
    (best(large) - best(small)) / (large - small) as f64
}

/// Native `clang -O2 -march=native` oracle: compile `kernel.c` + the self-timing driver, run it, and
/// parse `(per_iter_ns, run(small))`.
fn native_lane(cfile: &Path, small: i64, large: i64) -> Option<(f64, i64)> {
    let exe = std::env::temp_dir().join(format!("confine_{}.exe", std::process::id()));
    let ok = Command::new("clang")
        .args(["-O2", "-march=native"])
        .arg(cfile)
        .arg(confine_dir().join("native_main.c"))
        .args(["-lm", "-o"])
        .arg(&exe)
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if !ok {
        return None;
    }
    let out = Command::new(&exe)
        .args([small.to_string(), large.to_string(), REPS.to_string()])
        .output()
        .ok()?;
    let _ = std::fs::remove_file(&exe);
    if !out.status.success() {
        return None;
    }
    parse_ns_chk(&out.stdout)
}

/// Compile `kernel.c` to LP64 bitcode, translate, and JIT it (compiled once, reused). Returns a
/// `run(n)` runner plus `run(small)` — the caller times it. The two headline lanes (svm-jit, wt/w64)
/// return runners rather than self-timing so the caller can time them **back-to-back**, keeping the
/// tracked `svm÷wt64` ratio off the perturbation of the slow informational lanes (native/interp/v8).
fn svmjit_runner(cfile: &Path, small: i64) -> Option<(impl FnMut(i64) -> i64, i64)> {
    let (module, sp, e) = translate(cfile)?;
    let mut cm = svm_jit::compile(&module, e).ok()?;
    let mut runner = move |n: i64| -> i64 {
        match cm.run(&[sp, n], None, None, None).expect("svm-jit run") {
            (svm_jit::JitOutcome::Returned(v), _) => v[0],
            (other, _) => panic!("svm-jit did not return: {other:?}"),
        }
    };
    let want = runner(small);
    Some((runner, want))
}

/// Time `run(n)` on the bytecode interpreter (the same IR, tree-executed). `large` is calibrated down
/// to `BC_TARGET_S` (the interp's per-iter is n-independent), so this stays bounded despite being
/// ~600× native. Returns `(per_iter_ns, run(small))`.
fn svmbc_lane(cfile: &Path, small: i64, large: i64) -> Option<(f64, i64)> {
    let (module, sp, e) = translate(cfile)?;
    let runner = move |n: i64| -> i64 {
        let mut fuel = u64::MAX;
        match bytecode::compile_and_run(&module, e, &[Value::I64(sp), Value::I64(n)], &mut fuel)
            .expect("bytecode compiles")
            .expect("bytecode run")[0]
        {
            Value::I64(v) => v,
            Value::I32(v) => v as i64,
            other => panic!("bytecode did not return i64: {other:?}"),
        }
    };
    let want = runner(small);
    // Calibrate: measure the per-iter rate off a cheap probe, then pick the largest delta whose run
    // stays within the time budget (capped at the kernel's real `large`).
    let t0 = Instant::now();
    runner(small);
    let base = t0.elapsed().as_secs_f64();
    let probe = small + (large - small).min(small.max(1) * 4);
    let t1 = Instant::now();
    runner(probe);
    let dt = t1.elapsed().as_secs_f64();
    let per_extra = ((dt - base) / (probe - small).max(1) as f64).max(1e-12);
    let extra = (BC_TARGET_S / per_extra) as i64;
    let large_bc = (small + extra.max(1)).min(large);
    Some((per_iter(small, large_bc, BC_REPS, runner), want))
}

/// Build `kernel.c` to a wasm module at the given target (`wasm32`/`wasm64`), matching the embench
/// build's `-mbulk-memory` (so `memcpy`/`memset` lower to `memory.copy`/`fill`, not undefined libc
/// symbols). Returns the module path.
fn build_wasm(cfile: &Path, target: &str) -> Option<PathBuf> {
    let wasm = std::env::temp_dir().join(format!("confine_{}.{target}.wasm", std::process::id()));
    let ok = Command::new("clang")
        .args([
            &format!("--target={target}"),
            "-O2",
            "-mbulk-memory",
            "-nostdlib",
            "-Wl,--no-entry",
            "-Wl,--export=run",
            "-Wl,--gc-sections",
        ])
        .arg(cfile)
        .arg("-o")
        .arg(&wasm)
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    ok.then_some(wasm)
}

/// Instantiate a prebuilt wasm module on Wasmtime (Cranelift, default Spectre mitigation on). Returns
/// a `run(n)` runner plus `run(small)`; the caller times it.
fn wt_runner(wasm: &Path, w64: bool, small: i64) -> Option<(impl FnMut(i64) -> i64, i64)> {
    let mut cfg = Config::new();
    cfg.wasm_memory64(w64);
    let engine = Engine::new(&cfg).ok()?;
    let module = Module::from_file(&engine, wasm).ok()?;
    let mut store = Store::new(&engine, ());
    let inst = Instance::new(&mut store, &module, &[]).ok()?;
    let f = inst.get_func(&mut store, "run")?;
    let mut out = [Val::I64(0)];
    // wasm32 (`long` = i32) exports `run(i32)->i32`; wasm64 (memory64, LP64) exports `run(i64)->i64`.
    let mut runner = move |n: i64| -> i64 {
        let arg = if w64 { Val::I64(n) } else { Val::I32(n as i32) };
        f.call(&mut store, &[arg], &mut out).expect("wt run");
        match out[0] {
            Val::I64(x) => x,
            Val::I32(x) => x as i64,
            _ => panic!("unexpected wt return"),
        }
    };
    let want = runner(small);
    Some((runner, want))
}

/// Run a prebuilt wasm module on V8 (Node) via `bench/embench/wasm/run.mjs` (self-timed, same
/// subtraction methodology). `None` if `node` is missing/errors. Returns `(per_iter_ns, run(small))`.
fn v8_lane(run_mjs: &Path, wasm: &Path, small: i64, large: i64) -> Option<(f64, i64)> {
    let out = Command::new("node")
        .arg(run_mjs)
        .arg(wasm)
        .args([small.to_string(), large.to_string(), small.to_string()])
        .stderr(std::process::Stdio::null())
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    parse_ns_chk(&out.stdout)
}

/// Compile `kernel.c` to LP64 bitcode and translate to SVM IR; return `(module, entry_sp, run idx)`.
fn translate(cfile: &Path) -> Option<(svm_ir::Module, i64, u32)> {
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
    Some((t.module, sp, e))
}

fn parse_ns_chk(stdout: &[u8]) -> Option<(f64, i64)> {
    let s = String::from_utf8_lossy(stdout);
    let mut it = s.lines();
    let ns = it.next()?.trim().parse::<f64>().ok()?;
    let chk = it.next()?.trim().parse::<i64>().ok()?;
    Some((ns, chk))
}

/// One kernel's row: native ns + every engine's per-iter ns (`None` = lane unavailable).
struct Row {
    name: String,
    native: Option<f64>,
    svmjit: Option<f64>,
    svmbc: Option<f64>,
    v8_32: Option<f64>,
    v8_64: Option<f64>,
    wt_32: Option<f64>,
    wt_64: Option<f64>,
    svm_over_wt64: Option<f64>, // the tracked headline ratio
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
    let quiet = csv || save.is_some() || check.is_some();

    let dir = kernels_dir();
    let run_mjs = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("embench/wasm/run.mjs");
    let have_node = Command::new("node")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
        && run_mjs.exists();

    let mut rows: Vec<Row> = Vec::new();

    if !quiet {
        println!(
            "{:<10} {:>11} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>9}   correctness",
            "kernel",
            "native(ns)",
            "svm-jit",
            "svm-bc",
            "v8/w32",
            "v8/w64",
            "wt/w32",
            "wt/w64",
            "svm÷wt64"
        );
    }
    for &(name, small, large) in KERNELS {
        let cfile = dir.join(format!("{name}.c"));
        // Build/compile every artifact first (nothing timed yet). The svm-jit lane is mandatory (the
        // whole point); everything else is best-effort.
        let Some((mut svm_run, svm_want)) = svmjit_runner(&cfile, small) else {
            eprintln!("note: {name}: svm-jit lane unavailable (clang/on-ramp) — skipped");
            continue;
        };
        let wasm32 = build_wasm(&cfile, "wasm32");
        let wasm64 = build_wasm(&cfile, "wasm64");
        let wt64 = wasm64.as_deref().and_then(|w| wt_runner(w, true, small));

        // Headline timing block: svm-jit and wt/w64 **back-to-back**, so the tracked `svm÷wt64` ratio
        // is measured under one consistent machine state (not straddling the slow native/interp/v8
        // lanes). Everything below this is informational.
        let svm_ns = per_iter(small, large, REPS, &mut svm_run);
        let wt_64 = wt64.map(|(mut r, c)| (per_iter(small, large, REPS, &mut r), c));
        let svm_over_wt64 = wt_64.map(|(ns, _)| svm_ns / ns);

        // Informational lanes (timed after the headline; native subprocess + interp are slow).
        let native = native_lane(&cfile, small, large);
        let native_chk = native.map(|(_, c)| c);
        let svmbc = svmbc_lane(&cfile, small, large);
        let wt_32 = wasm32
            .as_deref()
            .and_then(|w| wt_runner(w, false, small))
            .map(|(mut r, c)| (per_iter(small, large, REPS, &mut r), c));
        let v8_32 = (have_node)
            .then(|| {
                wasm32
                    .as_deref()
                    .and_then(|w| v8_lane(&run_mjs, w, small, large))
            })
            .flatten();
        let v8_64 = (have_node)
            .then(|| {
                wasm64
                    .as_deref()
                    .and_then(|w| v8_lane(&run_mjs, w, small, large))
            })
            .flatten();
        for w in [wasm32, wasm64].into_iter().flatten() {
            let _ = std::fs::remove_file(w);
        }

        // Correctness: every available *64-bit* engine must agree with the oracle on `run(small)`.
        // Native is the oracle when present, else svm-jit. The w32 lanes run at 32-bit `long` (ILP32)
        // — a *different program* whose checksum legitimately differs when a kernel's result overflows
        // 32 bits (embench §header), so they are informational-only and never gate.
        let oracle = native_chk.unwrap_or(svm_want);
        let mut bad: Vec<String> = Vec::new();
        let mut chk = |label: &str, w: Option<i64>| {
            if let Some(v) = w {
                if v != oracle {
                    bad.push(format!("{label}={v}"));
                }
            }
        };
        chk("svm-jit", Some(svm_want));
        chk("svm-bc", svmbc.map(|(_, c)| c));
        chk("v8/w64", v8_64.map(|(_, c)| c));
        chk("wt/w64", wt_64.map(|(_, c)| c));
        assert!(
            bad.is_empty(),
            "{name}: MISCOMPILE — oracle(run({small}))={oracle}, disagree: {}",
            bad.join(" ")
        );

        let row = Row {
            name: name.to_string(),
            native: native.map(|(ns, _)| ns),
            svmjit: Some(svm_ns),
            svmbc: svmbc.map(|(ns, _)| ns),
            v8_32: v8_32.map(|(ns, _)| ns),
            v8_64: v8_64.map(|(ns, _)| ns),
            wt_32: wt_32.map(|(ns, _)| ns),
            wt_64: wt_64.map(|(ns, _)| ns),
            svm_over_wt64,
        };
        if !quiet {
            // Engine columns as ×native (like embench); "—" when the lane or native is unavailable.
            let rel = |x: Option<f64>| match (x, row.native) {
                (Some(v), Some(n)) => format!("{:.2}x", v / n),
                _ => "—".to_string(),
            };
            let nat = row.native.map_or("—".to_string(), |n| format!("{n:.1}"));
            let hl = row
                .svm_over_wt64
                .map_or("—".to_string(), |r| format!("{r:.3}x"));
            println!(
                "{:<10} {nat:>11} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {hl:>9}   OK",
                row.name,
                rel(row.svmjit),
                rel(row.svmbc),
                rel(row.v8_32),
                rel(row.v8_64),
                rel(row.wt_32),
                rel(row.wt_64),
            );
        }
        rows.push(row);
    }

    if csv {
        // kernel,native_ns,svmjit_ns,svmbc_ns,v8w32_ns,v8w64_ns,wtw32_ns,wtw64_ns,svm_over_wt64
        let f = |x: Option<f64>| x.map_or("".to_string(), |v| format!("{v:.4}"));
        println!("kernel,native,svmjit,svmbc,v8w32,v8w64,wtw32,wtw64,svm_over_wt64");
        for r in &rows {
            println!(
                "{},{},{},{},{},{},{},{},{}",
                r.name,
                f(r.native),
                f(r.svmjit),
                f(r.svmbc),
                f(r.v8_32),
                f(r.v8_64),
                f(r.wt_32),
                f(r.wt_64),
                f(r.svm_over_wt64),
            );
        }
    }
    if let Some(path) = save {
        let body: String = rows
            .iter()
            .filter_map(|r| r.svm_over_wt64.map(|x| format!("{} {x:.4}\n", r.name)))
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
            let Some(now) = rows
                .iter()
                .find(|r| r.name == kn)
                .and_then(|r| r.svm_over_wt64)
            else {
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
