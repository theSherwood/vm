//! **Intra-guest scaling — one guest, many threads.** The companion to `parallel` (which runs many
//! *independent* guests): here a **single** guest spawns `W` worker threads via `thread.spawn` and
//! joins them, and we ask whether the runtime spreads them across cores — i.e. does the §12 scheduler
//! give one guest near-`W`× speedup? On the JIT, `thread.spawn` maps to a real OS thread per guest
//! thread (`os_thread_rt`), so this exercises actual parallel execution (the *interpreters*
//! model-check thread interleavings on one OS thread, so they don't apply). Driven through the
//! embedding runtime `run_powerbox` (the concurrent path takes the per-domain `Mutex<Host>`).
//! Runtime-only: **no libLLVM**.
//!
//! Each worker runs a serial `xorshift64*` loop (pure ALU). For width `W` we time the whole guest at
//! two per-worker iteration counts and take the slope `(T(LARGE) − T(SMALL))/Δn` — the wall time per
//! worker-iteration, with compile + spawn/join overhead subtracted out. If the workers run in
//! parallel that slope stays flat as `W` grows (each on its own core); if they serialized it would
//! grow ∝ `W`. **efficiency = slope(1) / slope(W)** (100% = perfect parallelism to that width).
//!
//! Run: cargo run -p svm-run --release --example thread_scaling

use std::time::Instant;

use svm_run::{run_powerbox, Outcome, Value};

const SMALL: i64 = 1_000_000;
const LARGE: i64 = 200_000_000;

// Worker = func 1: `(i64 sp, i64 n) -> i64`, xorshift64* `n` times (serial, pure registers).
const WORKER: &str = r#"
func (i64, i64) -> (i64) {
block 0 (vsp: i64, vn: i64) {
  vx = i64.const -7046029254386353131
  vi = i64.const 0
  br 1(vi, vx, vn)
}
block 1 (va: i64, vb: i64, vc: i64) {
  vs1 = i64.const 13
  vt1 = i64.shl vb vs1
  vb1 = i64.xor vb vt1
  vs2 = i64.const 7
  vt2 = i64.shr_u vb1 vs2
  vb2 = i64.xor vb1 vt2
  vs3 = i64.const 17
  vt3 = i64.shl vb2 vs3
  vb3 = i64.xor vb2 vt3
  vone = i64.const 1
  va1 = i64.add va vone
  vlt = i64.lt_s va1 vc
  br_if vlt 1(va1, vb3, vc) 2(vb3)
}
block 2 (vr: i64) {
  return vr
  }
}
"#;

/// A powerbox guest (entry func 0: `(i32,i32,i32)->(i32)`, ignoring the granted cap handles) that
/// spawns `w` copies of the worker — each on its own 64 KiB data-stack region — runs `n` iterations
/// apiece, joins all, and returns 0.
fn guest(w: usize, n: i64) -> String {
    let mut s = String::from(
        "memory 22\nfunc (i32, i32, i32) -> (i32) {\nblock 0 (v0: i32, v1: i32, v2: i32) {\n",
    );
    s.push_str(&format!("  vn = i64.const {n}\n"));
    for i in 0..w {
        let sp = (i as i64 + 1) * 0x10000; // 64 KiB apart, all inside the 4 MiB window
        s.push_str(&format!(
            "  vsp{i} = i64.const {sp}\n  vh{i} = thread.spawn 1 vsp{i} vn\n"
        ));
    }
    for i in 0..w {
        s.push_str(&format!("  vj{i} = thread.join vh{i}\n"));
    }
    s.push_str("  vrc = i32.const 0\n  return vrc\n  }\n}\n");
    s.push_str(WORKER);
    s
}

/// Min-of-reps wall time (ns) of the whole `w`-thread guest doing `n` iterations per worker.
fn wall(w: usize, n: i64) -> f64 {
    let m = svm_text::parse_module(&guest(w, n)).expect("parse");
    svm_verify::verify_module(&m).expect("verify");
    let mut best = f64::MAX;
    for _ in 0..4 {
        let t = Instant::now();
        let run = run_powerbox(&m, b"").expect("run_powerbox");
        let e = t.elapsed().as_nanos() as f64;
        assert_eq!(
            run.outcome,
            Outcome::Returned(vec![Value::I32(0)]),
            "guest failed"
        );
        best = best.min(e);
    }
    best
}

/// Slope = wall-ns per worker-iteration (compile + spawn/join subtracted via the two-point fit).
fn slope(w: usize) -> f64 {
    (wall(w, LARGE) - wall(w, SMALL)) / (LARGE - SMALL) as f64
}

fn main() {
    let cores = std::thread::available_parallelism().map_or(1, |c| c.get());
    let mut counts = vec![1usize];
    let mut w = 2;
    while w < cores {
        counts.push(w);
        w *= 2;
    }
    counts.push(cores);
    counts.push(cores * 2); // one oversubscribed point

    println!(
        "host: {cores} logical cores | one guest, W threads via thread.spawn (JIT/os_thread_rt)\n\
         efficiency = slope(1)/slope(W); speedup = W × efficiency; 100% = perfect parallelism\n"
    );
    let base = slope(1);
    println!(
        "{:>8} {:>16} {:>14} {:>12}",
        "threads", "agg Miter/s", "speedup", "efficiency"
    );
    for &w in &counts {
        let s = slope(w);
        let eff = base / s; // ≤ 1
        let agg = w as f64 / s * 1e9 / 1e6; // (w workers × iters/ns) → iters/s → Miter/s
        println!(
            "{w:>8} {:>16.1} {:>13.2}x {:>11.0}%",
            agg,
            w as f64 * eff,
            eff * 100.0
        );
    }
    println!(
        "\n(One guest's `thread.spawn` workers are scheduled onto real OS threads by the JIT runtime\n \
         (os_thread_rt), so a single sandbox uses multiple cores. Near-linear speedup to the core\n \
         count means the scheduler/host adds no serialization; the plateau past it is SMT/oversubscription.\n \
         The interpreters model-check thread interleavings on one OS thread, so they're excluded here.)"
    );
}
