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
        large: 20_000,
    },
    Bench {
        name: "qrduino",
        src: "src/qrduino/qrtest.c",
        extra: &["src/qrduino/qrencode.c", "src/qrduino/qrframe.c"],
        beebs: true, // uses the BEEBS heap (init_heap_beebs)
        tail: "",
        large: 5_000,
    },
    // Still excluded (need on-ramp work, not just a BENCHES row):
    //  - `statemate`: defines a global `unsigned long time;` that collides with `<time.h>`'s `time()`
    //    in the native-oracle build (the wrapper includes time.h); the SVM side translates fine, but
    //    without a buildable native oracle the differential can't be honest. Needs a per-kernel rename.
    //  - `picojpeg`: multi-TU plumbing works (it's a unity build like xgboost/qrduino), but the kernel
    //    needs two on-ramp features the translator lacks: `-O2` emits 8-wide vector lane conversions
    //    (`trunc/zext/sext <8 x i16>`), and even with vectorization disabled it uses an i16 funnel
    //    shift (`llvm.fshl.i16` — the on-ramp lowers only i32/i64). Add it back once those land.
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

    println!(
        "{:<16} {:>12} {:>12} {:>8}   correctness",
        "benchmark", "native(ns)", "svm-jit(ns)", "ratio"
    );
    let mut ratios = Vec::new();
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
        let ok = nat_chk == 1 && tw == nat_chk && bcv == nat_chk && jitv == nat_chk;

        // Backstop: a runtime miscompile is reported but **excluded from the perf geomean** — a wrong
        // answer has no meaningful speed. (`edn` now translates *and* verifies correct across all three
        // engines — the ISSUES.md I13 `<2 x i16>`/`<2 x i32>` lane-arithmetic miscompile is fixed — so it
        // no longer trips this; the branch remains as a guard against any future translate-but-miscompile
        // kernel.)
        if !ok {
            println!(
                "{name:<16} {nat_ns:>12.1} {:>12}   {:>7}   MISCOMPILE (excluded) nat={nat_chk} tw={tw} bc={bcv} jit={jitv} — see ISSUES.md I13",
                "—", "—"
            );
            continue;
        }

        // JIT per-iter timing (native + jit, the comparability headline).
        let jit_ns = per_iter(large, |n| {
            black_box(svm_jit::compile_and_run(&t.module, e, &[sp, n]).unwrap());
        });

        ratios.push((name, jit_ns / nat_ns));
        println!(
            "{name:<16} {nat_ns:>12.1} {jit_ns:>12.1} {:>7.2}x   OK (all engines = native, verify=1)",
            jit_ns / nat_ns,
        );
    }
    if !ratios.is_empty() {
        let geo = (ratios.iter().map(|(_, r)| r.ln()).sum::<f64>() / ratios.len() as f64).exp();
        let worst = ratios
            .iter()
            .cloned()
            .fold(("", 0f64), |a, b| if b.1 > a.1 { b } else { a });
        println!(
            "\nsvm-jit vs native over {} Embench-IoT kernels: geomean {geo:.2}x | worst {} {:.2}x",
            ratios.len(),
            worst.0,
            worst.1
        );
    }
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
