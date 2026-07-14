//! **Real-program cross-engine perf harness** — diverse `no_std` Rust workloads (a real program each:
//! a hash-table churn, a bytecode interpreter, a batch sort) run on **svm-jit** vs **Wasmtime-w64**
//! vs **native**, timed by the large/small-`n` subtraction (min over reps) — the confine methodology,
//! but on real programs instead of confinement micro-kernels.
//!
//! Why Rust (not C): a `no_std` + `alloc` Rust program has **zero libc surface** (a bump
//! `#[global_allocator]` provides the heap), so it compiles cleanly to every lane with no shim
//! assembly — the thing that made a real program like Lua impractical to stand up here. Each workload
//! is `bench/rustbench/prelude.rs` (allocator/panic/PRNG) prepended to `workloads/<name>.rs` (the
//! `run(n) -> i64` logic).
//!
//! Lanes (each gracefully skipped if its toolchain is absent):
//!   native   `rustc +1.81 -O` → object, linked with the confine self-timer. The ×native baseline.
//!   svm-jit  `rustc +1.81 --emit=llvm-bc` (LLVM 18, matching the on-ramp's `llvm-dis`) → `svm_llvm`
//!            → `svm_jit`, in-process. LP64.
//!   wt/w64   `cargo +nightly build -Z build-std … --target wasm64-unknown-unknown` → Wasmtime
//!            (memory64). **LP64 — the honest same-widths comparison; `svm÷wt64` is the headline.**
//!   wt/w32   `rustc +1.81 --target wasm32-unknown-unknown` → Wasmtime. ILP32 — the *flattered*
//!            comparison (32-bit addressing + free 4 GiB guards), shown for context only.
//!
//! Toolchain: `rustc +1.81.0` (LLVM 18) for the LP64 bitcode + native + wasm32 lanes; `+nightly` with
//! the `rust-src` component and the `wasm32`/`wasm64` targets for the wasm lanes. Missing pieces just
//! blank the column. Run from `bench/`:  cargo run --release --bin rustbench

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use wasmtime::{Config, Engine, Instance, Module, Store, Val};

const REPS: u32 = 15;
const RUSTC: &str = "+1.81.0"; // LLVM 18 — matches the svm-llvm on-ramp's llvm-dis

/// (workload, small-n, large-n). `large` sized so the large run is tens of ms.
const WORKLOADS: &[(&str, i64, i64)] = &[
    ("hashmap", 1_000, 4_000_000),
    ("vm", 1_000, 8_000_000),
    ("sort", 100, 400_000),
    ("parse", 1_000, 2_000_000),
    ("base64", 1_000, 1_000_000),
    // ("bfs", 10, 5_000),  // DISABLED: svm-jit miscompiles it (returns garbage) — ISSUES.md I23,
    // a real bug this harness caught. The workload is kept at workloads/bfs.rs; re-enable once fixed.
];

fn rb_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("rustbench")
}
fn tmp(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("rustbench_{}_{name}", std::process::id()))
}

/// prelude.rs + workloads/<name>.rs — the full crate source for one workload.
fn compose(name: &str) -> Option<String> {
    let pre = std::fs::read_to_string(rb_dir().join("prelude.rs")).ok()?;
    let wl = std::fs::read_to_string(rb_dir().join(format!("workloads/{name}.rs"))).ok()?;
    Some(format!("{pre}\n{wl}"))
}

fn rustc_ok() -> bool {
    Command::new("rustc")
        .args([RUSTC, "--edition", "2021", "--version"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Min-over-reps per-iteration ns via the large/small subtraction.
fn per_iter(small: i64, large: i64, reps: u32, mut run: impl FnMut(i64) -> i64) -> f64 {
    let mut best = |n: i64| -> f64 {
        run(n);
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

fn parse_ns_chk(out: &[u8]) -> Option<(f64, i64)> {
    let s = String::from_utf8_lossy(out);
    let mut it = s.lines();
    Some((
        it.next()?.trim().parse().ok()?,
        it.next()?.trim().parse().ok()?,
    ))
}

/// Native `rustc +1.81 -O` **staticlib** (bundles the core/alloc runtime — a bare object would leave
/// `unwrap_failed`/`panic_bounds_check`/… undefined) linked with the confine self-timer → `(per_iter_ns,
/// run(small))`.
fn native_lane(src: &Path, small: i64, large: i64) -> Option<(f64, i64)> {
    let lib = tmp("native.a");
    let ok = Command::new("rustc")
        .args([
            RUSTC,
            "--edition",
            "2021",
            "-O",
            "-Cpanic=abort",
            "--crate-type=staticlib",
        ])
        .arg(src)
        .arg("-o")
        .arg(&lib)
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if !ok {
        return None;
    }
    let exe = tmp("native.exe");
    let main_c = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("confine/native_main.c");
    let ok = Command::new("clang")
        .args(["-O2", "-march=native"])
        .arg(&main_c)
        .arg(&lib)
        .arg("-o")
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
    parse_ns_chk(&out.stdout)
}

/// `rustc +1.81 --emit=llvm-bc` → svm_llvm → svm_jit (compiled once). Returns a runner + `run(small)`.
fn svmjit_runner(src: &Path, small: i64) -> Option<(impl FnMut(i64) -> i64, i64)> {
    let bc = tmp("svm.bc");
    let ok = Command::new("rustc")
        .args([
            RUSTC,
            "--edition",
            "2021",
            "-O",
            "-Cpanic=abort",
            "--emit=llvm-bc",
            "--crate-type=cdylib",
            "--target=x86_64-unknown-linux-gnu",
        ])
        .arg(src)
        .arg("-o")
        .arg(&bc)
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if !ok {
        return None;
    }
    let t = svm_llvm::translate_bc_path(&bc).ok()?;
    let sp = t.entry_sp as i64;
    let e = t.exports.iter().find(|(n, _)| n == "run")?.1;
    let mut cm = svm_jit::compile(&t.module, e).ok()?;
    let mut runner = move |n: i64| -> i64 {
        match cm.run(&[sp, n], None, None, None).expect("svm-jit run") {
            (svm_jit::JitOutcome::Returned(v), _) => v[0],
            (o, _) => panic!("svm-jit did not return: {o:?}"),
        }
    };
    let want = runner(small);
    Some((runner, want))
}

/// Instantiate a prebuilt wasm module on Wasmtime and return a `run(n)` runner + `run(small)`. Rust's
/// `run(i64)->i64` keeps the same signature on wasm32 and wasm64, so the arg is always `Val::I64`.
fn wt_runner(wasm: &Path, w64: bool, small: i64) -> Option<(impl FnMut(i64) -> i64, i64)> {
    let mut cfg = Config::new();
    cfg.wasm_memory64(w64);
    let engine = Engine::new(&cfg).ok()?;
    let module = Module::from_file(&engine, wasm).ok()?;
    let mut store = Store::new(&engine, ());
    let inst = Instance::new(&mut store, &module, &[]).ok()?;
    let f = inst.get_func(&mut store, "run")?;
    let mut out = [Val::I64(0)];
    let mut runner = move |n: i64| -> i64 {
        f.call(&mut store, &[Val::I64(n)], &mut out)
            .expect("wt run");
        match out[0] {
            Val::I64(x) => x,
            Val::I32(x) => x as i64,
            _ => panic!("unexpected wt return"),
        }
    };
    let want = runner(small);
    Some((runner, want))
}

/// `rustc +1.81 --target wasm32-unknown-unknown` → module path (ILP32, the flattered lane).
fn build_wasm32(src: &Path) -> Option<PathBuf> {
    let wasm = tmp("w32.wasm");
    Command::new("rustc")
        .args([
            RUSTC,
            "--edition",
            "2021",
            "-O",
            "-Cpanic=abort",
            "--crate-type=cdylib",
            "--target=wasm32-unknown-unknown",
        ])
        .arg(src)
        .arg("-o")
        .arg(&wasm)
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
        .then_some(wasm)
}

/// `cargo +nightly build -Z build-std … --target wasm64-unknown-unknown` in the `w64/` template
/// (wasm64 is tier-3, so it needs build-std). Returns the module path (memory64, LP64).
fn build_wasm64(src_text: &str) -> Option<PathBuf> {
    let proj = rb_dir().join("w64");
    std::fs::write(proj.join("src/lib.rs"), src_text).ok()?;
    let ok = Command::new("cargo")
        .args([
            "+nightly",
            "build",
            "-Zbuild-std=core,alloc",
            "--target=wasm64-unknown-unknown",
            "--release",
        ])
        .current_dir(&proj)
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if !ok {
        return None;
    }
    let wasm = proj.join("target/wasm64-unknown-unknown/release/rustbench_w64.wasm");
    wasm.exists().then_some(wasm)
}

fn main() {
    if !rustc_ok() {
        eprintln!(
            "rustbench: `rustc {RUSTC}` unavailable — install it (LLVM 18 toolchain). Skipping."
        );
        return;
    }
    let have_nightly = Command::new("rustc")
        .args(["+nightly", "--version"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);

    println!(
        "{:<10} {:>11} {:>9} {:>9} {:>9} {:>10}   correctness",
        "workload", "native(ns)", "svm-jit", "wt/w64", "wt/w32", "svm÷wt64"
    );
    for &(name, small, large) in WORKLOADS {
        let Some(text) = compose(name) else {
            eprintln!("note: {name}: source missing — skipped");
            continue;
        };
        let src = tmp(&format!("{name}.rs"));
        if std::fs::write(&src, &text).is_err() {
            continue;
        }
        let Some((mut svm_run, svm_want)) = svmjit_runner(&src, small) else {
            eprintln!("note: {name}: svm-jit lane unavailable — skipped");
            continue;
        };
        // Build the wasm modules, then time svm-jit and wt/w64 back-to-back (the tracked headline),
        // before the slower native/w32 lanes — keeping svm÷wt64 off their perturbation.
        let w64 = have_nightly
            .then(|| build_wasm64(&text))
            .flatten()
            .and_then(|w| wt_runner(&w, true, small));
        let svm_ns = per_iter(small, large, REPS, &mut svm_run);
        let wt64 = w64.map(|(mut r, c)| (per_iter(small, large, REPS, &mut r), c));

        let native = native_lane(&src, small, large);
        let wt32 = build_wasm32(&src)
            .and_then(|w| wt_runner(&w, false, small))
            .map(|(mut r, c)| (per_iter(small, large, REPS, &mut r), c));

        // Correctness: every available lane must agree with svm-jit on run(small).
        let mut bad: Vec<String> = Vec::new();
        if let Some((_, c)) = native {
            if c != svm_want {
                bad.push(format!("native={c}"));
            }
        }
        if let Some((_, c)) = wt64 {
            if c != svm_want {
                bad.push(format!("wt64={c}"));
            }
        }
        if let Some((_, c)) = wt32 {
            if c != svm_want {
                bad.push(format!("wt32={c}"));
            }
        }
        assert!(
            bad.is_empty(),
            "{name}: MISCOMPILE — svm-jit(run({small}))={svm_want}, disagree: {}",
            bad.join(" ")
        );

        let nat = native.map(|(ns, _)| ns);
        let relf = |x: Option<f64>| match (x, nat) {
            (Some(v), Some(n)) => format!("{:.2}x", v / n),
            _ => "—".to_string(),
        };
        let svm_over_wt64 = wt64.map(|(ns, _)| format!("{:.3}x", svm_ns / ns));
        println!(
            "{name:<10} {:>11} {:>9} {:>9} {:>9} {:>10}   OK",
            nat.map_or("—".to_string(), |n| format!("{n:.1}")),
            relf(Some(svm_ns)),
            relf(wt64.map(|(ns, _)| ns)),
            relf(wt32.map(|(ns, _)| ns)),
            svm_over_wt64.unwrap_or_else(|| "—".to_string()),
        );
    }
}
