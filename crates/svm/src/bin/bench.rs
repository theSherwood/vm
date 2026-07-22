//! Dependency-free benchmark harness (`DESIGN.md` §18, AGENTS.md "benchmark early").
//!
//! Two jobs:
//!   1. **escape-TCB hot paths** — decode / verify throughput, watched over time for regressions.
//!   2. **interpreter A/B** — the *same* compute kernels run through the **tree-walker** (`run`) and
//!      the **bytecode engine** (`bytecode::compile_and_run`), so we can see where the bytecode
//!      engine stands and measure Phase-2 (memory-op specialization) work against a real baseline
//!      rather than guessing (INTERP_PERF.md "Benchmark first").
//!
//! Each compute kernel takes its **loop count `n`** as the entry argument, so per-iteration compute
//! is isolated by **subtraction** — `(time(large_n) − time(small_n)) / (large_n − small_n)` — which
//! cancels the fixed per-run cost each engine pays (the tree-walker's frame setup, the bytecode
//! engine's per-run *compile*), leaving steady-state op cost. Times are the **min** over repetitions
//! (robust to a noisy box). Uses only `std`.
//!
//! Run: `cargo run --release --bin svm-bench`

use std::time::Instant;

use svm::{encode, ir, verify};
use svm_interp::{bytecode, Value};

fn main() {
    // ---- escape-TCB hot paths (decode/verify) ----------------------------------------------
    let module = ir_from_text(ALU);
    let bytes = encode::encode_module(&module);
    println!(
        "module: {} funcs, {} encoded bytes\n",
        module.funcs.len(),
        bytes.len()
    );

    bench("decode", 200_000, || {
        let m = encode::decode_module(&bytes).expect("decode");
        std::hint::black_box(&m);
    });
    bench("verify", 200_000, || {
        let m = encode::decode_module(&bytes).unwrap();
        verify::verify_module(&m).expect("verify");
        std::hint::black_box(&m);
    });

    // ---- interpreter A/B: tree-walker vs bytecode, per-iteration compute --------------------
    println!("\ninterpreter A/B (ns per loop iteration, compute-isolated by subtraction):");
    println!(
        "{:>12}  {:>12}  {:>12}  {:>8}",
        "kernel", "tree-walker", "bytecode", "tw/bc"
    );
    // (name, source, small_n, large_n). Each kernel loops exactly `n` times.
    let kernels = [
        ("alu", ALU, 1_000, 201_000),
        ("call", CALL, 1_000, 201_000),
        ("call_indirect", CALL_INDIRECT, 1_000, 201_000),
        ("mem", MEM, 1_000, 201_000),
    ];
    for (name, src, small, large) in kernels {
        let m = ir_from_text(src);
        let tw = per_iter(&m, small, large, |m, n| {
            let mut fuel = u64::MAX;
            let r = svm_interp::run(m, 0, &[Value::I32(n)], &mut fuel);
            std::hint::black_box(&r);
        });
        let bc = per_iter(&m, small, large, |m, n| {
            let mut fuel = u64::MAX;
            let r = bytecode::compile_and_run(m, 0, &[Value::I32(n)], &mut fuel)
                .expect("bytecode engine drives the kernel");
            std::hint::black_box(&r);
        });
        println!("{name:>12}  {tw:>10.2}ns  {bc:>10.2}ns  {:>7.2}×", tw / bc);
    }
}

/// `acc += n; n -= 1` until zero — a pure scalar/branch recurrence (the ALU kernel).
const ALU: &str = r#"
func (i32) -> (i32) {
block 0 (v0: i32) {
  v1 = i32.const 0
  br 1(v0, v1)
}
block 1 (v2: i32, v3: i32) {
  v4 = i32.add v3 v2
  v5 = i32.const 1
  v6 = i32.sub v2 v5
  br_if v6 1(v6, v4) 2(v4)
}
block 2 (v7: i32) {
  return v7
  }
}
"#;

/// Each iteration calls a leaf `+1` function — the call/return kernel (window open/close cost).
const CALL: &str = r#"
func (i32) -> (i32) {
block 0 (v0: i32) {
  v1 = i32.const 0
  br 1(v0, v1)
}
block 1 (v2: i32, v3: i32) {
  v4 = call 1(v3)
  v5 = i32.const 1
  v6 = i32.sub v2 v5
  br_if v6 1(v6, v4) 2(v4)
}
block 2 (v7: i32) {
  return v7
  }
}
func (i32) -> (i32) {
block 0 (v0: i32) {
  v1 = i32.const 1
  v2 = i32.add v0 v1
  return v2
  }
}
"#;

/// Each iteration dispatches through the `call_indirect` table — mask + slot read + type-check.
const CALL_INDIRECT: &str = r#"
func (i32) -> (i32) {
block 0 (v0: i32) {
  v1 = i32.const 0
  br 1(v0, v1)
}
block 1 (v2: i32, v3: i32) {
  v4 = i32.const 1
  v5 = call_indirect (i32) -> (i32) v4 (v3)
  v6 = i32.const 1
  v7 = i32.sub v2 v6
  br_if v7 1(v7, v5) 2(v5)
}
block 2 (v8: i32) {
  return v8
  }
}
func (i32) -> (i32) {
block 0 (v0: i32) {
  v1 = i32.const 1
  v2 = i32.add v0 v1
  return v2
  }
}
"#;

/// Each iteration does one `i32.store` + one `i32.load` at a fixed address — the memory kernel that
/// Phase 2 (width-specialized load/store + inlined confinement) targets.
const MEM: &str = r#"memory 16
func (i32) -> (i32) {
block 0 (v0: i32) {
  v1 = i32.const 0
  br 1(v0, v1)
}
block 1 (v2: i32, v3: i32) {
  v4 = i64.const 0
  i32.store v4 v3
  v5 = i32.load v4
  v6 = i32.const 1
  v7 = i32.add v5 v6
  v8 = i32.const 1
  v9 = i32.sub v2 v8
  br_if v9 1(v9, v7) 2(v7)
}
block 2 (v10: i32) {
  return v10
  }
}
"#;

fn ir_from_text(src: &str) -> ir::Module {
    svm::text::parse_module(src).expect("corpus program must parse")
}

/// Per-iteration compute (ns) for `run_one(module, n)`, isolated by large/small-`n` subtraction and
/// taken as the min over repetitions (robust to a noisy box).
fn per_iter(m: &ir::Module, small: i32, large: i32, run_one: impl Fn(&ir::Module, i32)) -> f64 {
    let t_small = min_run(m, small, &run_one);
    let t_large = min_run(m, large, &run_one);
    (t_large - t_small) / (large - small) as f64
}

fn min_run(m: &ir::Module, n: i32, run_one: &impl Fn(&ir::Module, i32)) -> f64 {
    // Warm up, then take the fastest of several reps (compute is deterministic; min rejects noise).
    run_one(m, n);
    let reps = 25;
    let mut best = f64::MAX;
    for _ in 0..reps {
        let start = Instant::now();
        run_one(m, n);
        best = best.min(start.elapsed().as_nanos() as f64);
    }
    best
}

fn bench(name: &str, iters: u64, mut f: impl FnMut()) {
    for _ in 0..(iters / 10).max(1) {
        f();
    }
    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    let elapsed = start.elapsed();
    let per = elapsed.as_nanos() as f64 / iters as f64;
    println!("{name:>8}: {iters} iters in {elapsed:?}  ({per:.1} ns/iter)");
}
