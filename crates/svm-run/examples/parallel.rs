//! **Parallel scaling — many isolated guests at once.** The SVM's reason for being is hosting many
//! sandboxed guests; this measures whether *W* concurrent guests deliver ~*W*× throughput, and where
//! it saturates. Each OS thread runs its **own** guest (the JIT compiles a private `CompiledModule`,
//! the interpreters run the shared read-only `&Module`) — no shared mutable runtime state — so this
//! probes the runtime for hidden contention (global locks, allocator, false sharing) that would cap
//! multi-tenant scaling. Runtime-only: this binary links **no libLLVM**.
//!
//! Kernel: a serial `xorshift64*` loop (pure ALU, no memory) so the only thing that can limit scaling
//! is the runtime/host, not memory bandwidth. For each engine and thread count W: every worker
//! pre-builds its artifact, all wait on a barrier, then run `n` iterations together; throughput is the
//! aggregate iters/s over the barrier-to-join window (compile excluded), and **efficiency** is
//! `throughput(W) / (W · throughput(1))` — 100% = perfect linear scaling.
//!
//! Run: cargo run -p svm-run --release --example parallel

use std::ffi::c_void;
use std::sync::Barrier;
use std::time::Instant;

use svm_interp::{bytecode, Value};
use svm_ir::Module;

// xorshift64*: `x ^= x<<13; x ^= x>>7; x ^= x<<17` for `n` iterations, returning the final state.
// Serial dependency (unvectorizable, unfoldable), pure registers — a clean CPU-bound scaling probe.
const SRC: &str = r#"
func (i32) -> (i64) {
block 0 (v0: i32) {
  v1 = i64.const -7046029254386353131
  v2 = i32.const 0
  br 1(v2, v1, v0)
}
block 1 (v3: i32, v4: i64, v5: i32) {
  v6 = i64.const 13
  v7 = i64.shl v4 v6
  v8 = i64.xor v4 v7
  v9 = i64.const 7
  v10 = i64.shr_u v8 v9
  v11 = i64.xor v8 v10
  v12 = i64.const 17
  v13 = i64.shl v11 v12
  v14 = i64.xor v11 v13
  v15 = i32.const 1
  v16 = i32.add v3 v15
  v17 = i32.lt_s v16 v5
  br_if v17 1(v16, v14, v5) 2(v14)
}
block 2 (v18: i64) {
  return v18
  }
}
"#;

#[derive(Clone, Copy, PartialEq)]
enum Engine {
    TreeWalk,
    Bytecode,
    Jit,
}

fn jit_compile(m: &Module) -> svm_jit::CompiledModule {
    svm_jit::CompiledModule::compile(
        m,
        0,
        svm_jit::INERT_CAP_THUNK,
        std::ptr::null_mut::<c_void>(),
        28,
        None,
        None,
        None,
        None,
        svm_jit::Quota::default(),
        0,
    )
    .expect("jit compile")
}

/// Run one worker: pre-build the engine artifact, sync on the barrier, then execute `n` iterations.
fn worker(engine: Engine, m: &Module, n: i32, barrier: &Barrier) {
    match engine {
        Engine::Jit => {
            let mut cm = jit_compile(m); // compile excluded from the timed window
            barrier.wait();
            let r = cm.run(&[n as i64], None, None, None).expect("jit run");
            std::hint::black_box(&r);
        }
        Engine::Bytecode => {
            // The bytecode compile is ~µs (negligible vs the timed run at this n); it recompiles in
            // `compile_and_run`, so sync first, then run.
            barrier.wait();
            let mut fuel = u64::MAX;
            let r = bytecode::compile_and_run(m, 0, &[Value::I32(n)], &mut fuel);
            std::hint::black_box(&r);
        }
        Engine::TreeWalk => {
            barrier.wait();
            let mut fuel = u64::MAX;
            let r = svm_interp::run(m, 0, &[Value::I32(n)], &mut fuel);
            std::hint::black_box(&r);
        }
    }
}

/// Aggregate throughput (million iters/s) for `w` concurrent workers each doing `n` iterations.
fn throughput(engine: Engine, m: &Module, n: i32, w: usize) -> f64 {
    let barrier = Barrier::new(w + 1);
    let mut t0 = None;
    std::thread::scope(|s| {
        for _ in 0..w {
            s.spawn(|| worker(engine, m, n, &barrier));
        }
        barrier.wait(); // released once every worker has finished setup (compile)
        t0 = Some(Instant::now());
    }); // workers join here
    let secs = t0.unwrap().elapsed().as_secs_f64();
    (w as f64 * n as f64) / secs / 1e6
}

fn main() {
    let m = svm_text::parse_module(SRC).expect("parse");
    svm_verify::verify_module(&m).expect("verify");

    let cores = std::thread::available_parallelism().map_or(1, |c| c.get());
    let mut counts = Vec::new();
    let mut w = 1;
    while w < cores {
        counts.push(w);
        w *= 2;
    }
    counts.push(cores);
    if cores * 2 > *counts.last().unwrap() {
        counts.push(cores * 2); // one oversubscribed point
    }

    // Per-engine iteration count so a single-thread run is ~100-200 ms (interpreters are ~50x slower).
    println!(
        "host: {cores} logical cores | scaling = throughput(W) / (W · throughput(1)); 100% = linear\n"
    );
    for (engine, name, n) in [
        (Engine::Jit, "jit", 100_000_000i32),
        (Engine::Bytecode, "bytecode", 4_000_000),
        (Engine::TreeWalk, "tree-walk", 2_000_000),
    ] {
        println!("{name} (xorshift, n={n} per thread):");
        println!(
            "  {:>8} {:>16} {:>14} {:>12}",
            "threads", "Miter/s", "per-thread", "scaling"
        );
        let base = throughput(engine, &m, n, 1);
        for &w in &counts {
            let tput = throughput(engine, &m, n, w);
            let eff = tput / (w as f64 * base) * 100.0;
            println!(
                "  {w:>8} {tput:>16.1} {:>14.1} {eff:>11.0}%",
                tput / w as f64
            );
        }
        println!();
    }
    println!(
        "(Each thread runs an independent guest — the JIT a private CompiledModule, the interpreters\n \
         the shared read-only Module — with no shared mutable runtime state. Near-100% scaling to the\n \
         core count means the runtime adds no contention to multi-tenant execution; the drop past the\n \
         physical-core count is expected SMT/oversubscription, not a runtime bottleneck.)"
    );
}
