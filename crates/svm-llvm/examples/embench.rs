//! **Embench-IoT — externally-comparable kernels.** The other drivers use our own kernels; this runs
//! the recognized **Embench-IoT** embedded suite through the real LLVM frontend across native + the
//! three SVM engines, for numbers comparable to published Embench results. Embench source is *not*
//! vendored (mixed per-benchmark licenses) — point it at a checkout:
//!
//!   curl -sSL https://github.com/embench/embench-iot/archive/refs/heads/master.tar.gz | tar xz -C /tmp
//!   EMBENCH=/tmp/embench-iot-master cargo run -p svm-llvm --release --example embench
//!
//! Each benchmark is wrapped by `bench/embench/wrapper.c` (which `#include`s the kernel `.c` and
//! exposes `long run(long n)` = `n` Embench iterations → `verify_benchmark` strict pass/fail). Native
//! is `clang -O2`; the SVM rows compile the same wrapper with `clang -O2 -emit-llvm`, translate, and
//! run on tree-walk/bytecode/JIT. Every engine's `verify` must equal native's (1 = matched Embench's
//! expected output) — so this is a benchmark *and* a whole-stack differential on real third-party code.
//!
//! **Cross-engine (svm-jit vs V8 vs Wasmtime).** The headline `svm-jit` ratio is reported alongside two
//! external JITs running the *same* kernel: each wrapper also compiles to a self-contained **wasm32**
//! module (`run` export, no imports — via the freestanding shim in `bench/embench/wasm/`, `-mbulk-memory`
//! so memcpy/memset are wasm instructions, `--gc-sections` to drop dead `printf`/libc), timed on **V8**
//! (Node, `bench/embench/wasm/run.mjs`) and **Wasmtime** (in-process Cranelift — the same backend
//! svm-jit uses — via `bench/cross-engine/wasmtime-rs`'s `embench_one` bin). Both are optional: a missing
//! `node`/runner just leaves that column blank. All engines' `verify` is still checked against native.

use std::hint::black_box;
use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

use svm_interp::{bytecode, Value};

struct Bench {
    /// display name
    name: &'static str,
    /// path under $EMBENCH to the `.c` defining `benchmark_body`/`initialise_benchmark`
    src: &'static str,
    /// extra library `.c` files (under $EMBENCH) the wrapper also `#include`s for *multi-translation-unit*
    /// kernels — a unity build, so the SVM side stays a single module (no llvm-link) and native/SVM
    /// compile identically. Empty for single-file kernels.
    extra: &'static [&'static str],
    /// needs the BEEBS rand/heap support file
    beebs: bool,
    /// trailing args spliced into the `benchmark_body(n, GSF ...)` call for kernels whose arity differs
    /// from the (lsf, gsf) norm (e.g. md5sum's third `len`). Empty ⇒ the plain two-arg call.
    tail: &'static str,
    /// native large-run iteration count, sized so the native large run is a few ms; the JIT reuses it
    /// (its per-call recompile washes out via subtraction).
    large: i64,
}

const fn b(name: &'static str, src: &'static str, beebs: bool, large: i64) -> Bench {
    Bench {
        name,
        src,
        extra: &[],
        beebs,
        tail: "",
        large,
    }
}

const BENCHES: &[Bench] = &[
    b(
        "matmult-int",
        "src/matmult-int/matmult-int.c",
        false,
        200_000,
    ),
    b("crc32", "src/crc32/crc_32.c", true, 20_000),
    b(
        "nettle-sha256",
        "src/nettle-sha256/nettle-sha256.c",
        false,
        20_000,
    ),
    b("edn", "src/edn/libedn.c", false, 50_000),
    b("ud", "src/ud/libud.c", false, 200_000),
    b("aha-mont64", "src/aha-mont64/mont64.c", false, 200_000),
    b("nsichneu", "src/nsichneu/libnsichneu.c", false, 50_000),
    b("depthconv", "src/depthconv/depthconv.c", false, 50_000),
    b("huffbench", "src/huffbench/libhuffbench.c", true, 5_000),
    b("nettle-aes", "src/nettle-aes/nettle-aes.c", false, 20_000),
    b("tarfind", "src/tarfind/tarfind.c", true, 50_000),
    b("wikisort", "src/wikisort/libwikisort.c", true, 2_000),
    b(
        "sglib-combined",
        "src/sglib-combined/combined.c",
        true,
        20_000,
    ),
    b("slre", "src/slre/libslre.c", true, 20_000),
    // md5sum: single file, but benchmark_body takes a third `len` arg — splice in MSG_SIZE (defined in
    // md5.c, in scope at the call site).
    Bench {
        name: "md5sum",
        src: "src/md5sum/md5.c",
        extra: &[],
        beebs: true, // uses the BEEBS heap (calloc_beebs)
        tail: ", MSG_SIZE",
        large: 5_000,
    },
    // Multi-TU kernels: the file with benchmark_body plus its library `.c` files, unity-built (see Bench.extra).
    Bench {
        name: "xgboost",
        src: "src/xgboost/testbench.c",
        extra: &["src/xgboost/xgboost.c"],
        beebs: false,
        tail: "",
        // ~1 ms/iter (a full GBDT inference) — 1000× the other kernels, so `large` is sized far smaller
        // to keep each run a few hundred ms (× 4 engines × ~10 reps). `large=20_000` was a ~20 s/run
        // mis-size that only barely fit the old native+svm timing.
        large: 400,
    },
    Bench {
        name: "qrduino",
        src: "src/qrduino/qrtest.c",
        extra: &["src/qrduino/qrencode.c", "src/qrduino/qrframe.c"],
        beebs: true, // uses the BEEBS heap (init_heap_beebs)
        tail: "",
        large: 5_000,
    },
    Bench {
        name: "picojpeg",
        src: "src/picojpeg/picojpeg_test.c",
        extra: &["src/picojpeg/libpicojpeg.c"],
        beebs: false,
        tail: "",
        large: 5_000,
    },
    // Still excluded (need on-ramp work, not just a BENCHES row):
    //  - `statemate`: defines a global `unsigned long time;` that collides with `<time.h>`'s `time()`
    //    in the native-oracle build (the wrapper includes time.h); the SVM side translates fine, but
    //    without a buildable native oracle the differential can't be honest. Needs a per-kernel rename.
];

const SMALL: i64 = 10;
const VERIFY_N: i64 = 1; // verify is n-independent (≥1); 1 rep is the cheapest correctness probe

fn main() {
    let Ok(embench) = std::env::var("EMBENCH") else {
        eprintln!("set EMBENCH=/path/to/embench-iot checkout (see this file's header). skipping.");
        return;
    };
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    let wrapper = root.join("bench/embench/wrapper.c");
    let support = format!("{embench}/support");
    let dir = std::env::temp_dir();

    // Cross-engine setup: each kernel also compiles to a self-contained wasm32 module (`run` export, no
    // imports — see bench/embench/wasm/) timed on V8 (Node) and on in-process Wasmtime (Cranelift, the
    // same backend svm-jit uses). Both are optional: a missing tool just leaves that column blank.
    let wasm_dir = root.join("bench/embench/wasm");
    let run_mjs = wasm_dir.join("run.mjs");
    let have_node = Command::new("node")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    let wt_manifest = root.join("bench/cross-engine/wasmtime-rs/Cargo.toml");
    let wt_bin = root.join("bench/cross-engine/wasmtime-rs/target/release/embench_one");
    if !wt_bin.exists() {
        // Build the in-process Wasmtime runner once (reuses the cached Wasmtime build under bench/).
        let _ = Command::new("cargo")
            .args([
                "build",
                "--release",
                "--bin",
                "embench_one",
                "--manifest-path",
            ])
            .arg(&wt_manifest)
            .status();
    }
    let have_wt = wt_bin.exists();

    println!(
        "{:<16} {:>11} {:>9} {:>9} {:>9}   correctness   (×native)",
        "benchmark", "native(ns)", "svm-jit", "v8", "wasmtime"
    );
    // Per-engine perf ratios vs native (geomean at the end). svm-jit always present; v8/wasmtime only
    // for kernels whose wasm built and whose runner is available.
    let mut jit_ratios = Vec::new();
    let mut v8_ratios = Vec::new();
    let mut wt_ratios = Vec::new();
    for bench in BENCHES {
        let &Bench {
            name,
            src: rel,
            extra,
            beebs,
            tail,
            large,
        } = bench;
        // Every translation unit (the benchmark .c plus any library .c) must be present.
        let missing = std::iter::once(rel)
            .chain(extra.iter().copied())
            .find(|p| !std::path::Path::new(&format!("{embench}/{p}")).exists());
        if let Some(p) = missing {
            println!("{name:<16} (skipped: {p} not found in $EMBENCH)");
            continue;
        }
        let bench_def = format!("-DBENCH_SRC=\"{embench}/{rel}\"");
        // Library TUs for multi-`.c` kernels, `#include`d by the wrapper as BENCH_EXTRA1/2 (unity build).
        let extra_defs: Vec<String> = extra
            .iter()
            .enumerate()
            .map(|(i, p)| format!("-DBENCH_EXTRA{}=\"{embench}/{p}\"", i + 1))
            .collect();
        let beebs_def = format!("-DBEEBS_SRC=\"{embench}/support/beebsc.c\"");
        let tail_def = format!("-DBENCH_TAIL_ARGS={tail}");
        let common = |c: &mut Command| {
            c.args(["-O2", "-DNDEBUG"])
                .arg(format!("-I{support}"))
                .arg(&bench_def)
                .args(&extra_defs);
            if !tail.is_empty() {
                c.arg(&tail_def);
            }
            if beebs {
                c.arg(&beebs_def);
            }
            // `aha-mont64`'s `mulul64`/`modul64` use `unsigned __int128`. I14 tier 3 now lowers i128
            // arithmetic/shifts/compares, so the kernel's *local* i128 ops translate — but its Montgomery
            // exponentiation loop may carry an i128 across iterations (a cross-block φ), which still
            // fail-closes (the i128 `agg` pair is same-block). Until that's confirmed against a real
            // checkout, route it through the `#ifdef __SIZEOF_INT128__` pure-64-bit fallback by undefining
            // the macro — applied to *both* native and SVM builds so the differential stays honest.
            if name == "aha-mont64" {
                c.arg("-U__SIZEOF_INT128__");
            }
        };

        // Native: compile + run → (per_iter ns, verify checksum).
        let exe = dir.join(format!("emb_{name}.exe"));
        let mut nc = Command::new("clang");
        common(&mut nc);
        nc.args(["-march=native", "-lm"])
            .arg(&wrapper)
            .arg("-o")
            .arg(&exe)
            .stderr(std::process::Stdio::null());
        if !nc.status().map(|s| s.success()).unwrap_or(false) {
            println!("{name:<16} (skipped: native compile failed)");
            continue;
        }
        let out = Command::new(&exe)
            .args([SMALL.to_string(), large.to_string(), VERIFY_N.to_string()])
            .output()
            .expect("run native");
        let s = String::from_utf8_lossy(&out.stdout);
        let mut it = s.lines();
        let (Some(nat_ns), Some(nat_chk)) = (
            it.next().and_then(|l| l.trim().parse::<f64>().ok()),
            it.next().and_then(|l| l.trim().parse::<i64>().ok()),
        ) else {
            println!("{name:<16} (skipped: native run produced no result)");
            continue;
        };

        // SVM: compile the wrapper to bitcode (no main), translate.
        let bc = dir.join(format!("emb_{name}.bc"));
        let mut sc = Command::new("clang");
        common(&mut sc);
        sc.args([
            "-emit-llvm",
            "-c",
            "-DSVM_BUILD",
            "-fno-builtin-memcmp",
            "-fno-builtin-bcmp",
        ])
        .arg(&wrapper)
        .arg("-o")
        .arg(&bc)
        .stderr(std::process::Stdio::null());
        if !sc.status().map(|s| s.success()).unwrap_or(false) {
            println!("{name:<16} (skipped: svm bitcode compile failed)");
            continue;
        }
        let t = match svm_llvm::translate_bc_path(&bc) {
            Ok(t) => t,
            Err(e) => {
                println!("{name:<16} (skipped: translate failed: {e:?})");
                continue;
            }
        };
        let sp = t.entry_sp as i64;
        let Some(e) = t.exports.iter().find(|(n, _)| n == "run").map(|x| x.1) else {
            println!("{name:<16} (skipped: `run` not exported)");
            continue;
        };

        // Correctness: every engine's verify at VERIFY_N must equal native's (1 = Embench-correct).
        let mut fuel = u64::MAX;
        let tw = as_i64(
            svm_interp::run(
                &t.module,
                e,
                &[Value::I64(sp), Value::I64(VERIFY_N)],
                &mut fuel,
            )
            .unwrap()[0],
        );
        let mut fuel = u64::MAX;
        let bcv = as_i64(
            bytecode::compile_and_run(
                &t.module,
                e,
                &[Value::I64(sp), Value::I64(VERIFY_N)],
                &mut fuel,
            )
            .expect("bytecode")
            .unwrap()[0],
        );
        let jitv = match svm_jit::compile_and_run(&t.module, e, &[sp, VERIFY_N]).unwrap() {
            svm_jit::JitOutcome::Returned(v) => v[0],
            o => panic!("jit: {o:?}"),
        };
        // wasm32 build (shared shim) for the V8 + Wasmtime rows. Same kernel flags as the native/SVM
        // builds via `common`; adds the freestanding-wasm shim + bulk-memory (memcpy/memset → wasm
        // instructions) and exports only `run` (so dead `printf`/`main`/libc references get DCE'd).
        let wasm = dir.join(format!("emb_{name}.wasm"));
        let mut wc = Command::new("clang");
        common(&mut wc);
        wc.args([
            "--target=wasm32",
            "-msimd128",
            "-mbulk-memory",
            "-DSVM_BUILD",
            "-fno-builtin-memcmp",
            "-fno-builtin-bcmp",
            "-fno-builtin-strlen",
            "-fno-builtin-strchr",
            "-fno-builtin-strcmp",
            "-nostdlib",
            "-Wl,--no-entry",
            "-Wl,--export=run",
            "-Wl,--gc-sections",
        ])
        .arg("-include")
        .arg(wasm_dir.join("defs.h"))
        .arg("-isystem")
        .arg(wasm_dir.join("include"))
        .arg(&wrapper)
        .arg("-o")
        .arg(&wasm)
        .stderr(std::process::Stdio::null());
        let wasm_ok = wc.status().map(|s| s.success()).unwrap_or(false);
        let v8 = if wasm_ok && have_node {
            time_wasm(Command::new("node").arg(&run_mjs), &wasm, large)
        } else {
            None
        };
        let wt = if wasm_ok && have_wt {
            time_wasm(&mut Command::new(&wt_bin), &wasm, large)
        } else {
            None
        };

        // Correctness: every engine that ran must match native's verify=1 (1 = Embench-correct). An
        // absent wasm engine (no tool / build) doesn't gate; one that ran and disagrees does. (`edn`'s
        // old I13 lane-arithmetic miscompile is fixed; this stays a guard against any future regression.)
        let wasm_bad = v8.is_some_and(|(_, c)| c != 1) || wt.is_some_and(|(_, c)| c != 1);
        let ok = nat_chk == 1 && tw == nat_chk && bcv == nat_chk && jitv == nat_chk && !wasm_bad;
        if !ok {
            println!(
                "{name:<16} {nat_ns:>11.1} {:>9} {:>9} {:>9}   MISCOMPILE nat={nat_chk} tw={tw} bc={bcv} jit={jitv} v8={} wt={}",
                "—",
                "—",
                "—",
                v8.map_or(-1, |(_, c)| c),
                wt.map_or(-1, |(_, c)| c),
            );
            continue;
        }

        // JIT per-iter timing (the comparability headline).
        let jit_ns = per_iter(large, |n| {
            black_box(svm_jit::compile_and_run(&t.module, e, &[sp, n]).unwrap());
        });
        jit_ratios.push((name, jit_ns / nat_ns));
        if let Some((ns, _)) = v8 {
            v8_ratios.push((name, ns / nat_ns));
        }
        if let Some((ns, _)) = wt {
            wt_ratios.push((name, ns / nat_ns));
        }
        let ratio = |x: Option<(f64, i64)>| {
            x.map_or_else(|| "—".to_string(), |(ns, _)| format!("{:.2}x", ns / nat_ns))
        };
        println!(
            "{name:<16} {nat_ns:>11.1} {:>9} {:>9} {:>9}   OK (verify=1)",
            format!("{:.2}x", jit_ns / nat_ns),
            ratio(v8),
            ratio(wt),
        );
    }
    let geomean =
        |rs: &[(&str, f64)]| (rs.iter().map(|(_, r)| r.ln()).sum::<f64>() / rs.len() as f64).exp();
    if !jit_ratios.is_empty() {
        println!("\nvs native `clang -O2`, geomean over the kernels each engine ran:");
        println!(
            "  svm-jit   {:.2}x   ({} kernels)",
            geomean(&jit_ratios),
            jit_ratios.len()
        );
        if !v8_ratios.is_empty() {
            println!(
                "  v8        {:.2}x   ({} kernels)",
                geomean(&v8_ratios),
                v8_ratios.len()
            );
        }
        if !wt_ratios.is_empty() {
            println!(
                "  wasmtime  {:.2}x   ({} kernels)",
                geomean(&wt_ratios),
                wt_ratios.len()
            );
        }
    }
}

/// Run a wasm timing runner (`<cmd> [prefix args…] <wasm> <small> <large> <verify_n>`) and parse its
/// two stdout lines into `(per_iter_ns, verify)`. `None` if the tool is missing, errors, or its output
/// doesn't parse — the caller then leaves that engine's column blank for the kernel.
fn time_wasm(cmd: &mut Command, wasm: &std::path::Path, large: i64) -> Option<(f64, i64)> {
    let out = cmd
        .arg(wasm)
        .arg(SMALL.to_string())
        .arg(large.to_string())
        .arg(VERIFY_N.to_string())
        .stderr(std::process::Stdio::null())
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout);
    let mut it = s.lines();
    let ns = it.next()?.trim().parse::<f64>().ok()?;
    let chk = it.next()?.trim().parse::<i64>().ok()?;
    Some((ns, chk))
}

fn per_iter(large: i64, run_one: impl Fn(i64)) -> f64 {
    let m = |n: i64| {
        run_one(n);
        let mut best = f64::MAX;
        for _ in 0..9 {
            let t = Instant::now();
            run_one(n);
            best = best.min(t.elapsed().as_nanos() as f64);
        }
        best
    };
    (m(large) - m(SMALL)) / (large - SMALL) as f64
}

fn as_i64(v: Value) -> i64 {
    match v {
        Value::I32(x) => x as i64,
        Value::I64(x) => x,
        other => panic!("unexpected {other:?}"),
    }
}
