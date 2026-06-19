//! Cross-engine micro-benchmark for the SVM backends — **tree-walker**, **bytecode engine**, and
//! **JIT** — over the same compute kernels, with per-iteration compute isolated by large/small-`n`
//! subtraction and taken as the min over repetitions (the methodology of `src/bin/bench.rs`). It is an
//! *example* (not the `svm-bench` binary) because the JIT lives in `svm-jit`, a dev-dependency.
//!
//! Output is machine-readable CSV on stdout — `engine,kernel,ns_per_iter` — so an external driver can
//! merge it with native / wasm / python numbers into one table. Run:
//!   cargo run --release --example megabench -p svm

use std::time::Instant;

use svm::{ir, text};
use svm_interp::{bytecode, Value};

fn main() {
    let kernels = [
        ("alu", ALU, 1_000i32, 201_000i32),
        ("call", CALL, 1_000, 201_000),
        ("call_indirect", CALL_INDIRECT, 1_000, 201_000),
        ("mem", MEM, 1_000, 201_000),
    ];
    for (name, src, small, large) in kernels {
        let m = text::parse_module(src).expect("kernel parses");

        let tw = per_iter(small, large, |n| {
            let mut fuel = u64::MAX;
            let r = svm_interp::run(&m, 0, &[Value::I32(n)], &mut fuel);
            std::hint::black_box(&r);
        });
        println!("svm-tree-walk,{name},{tw:.4}");

        let bc = per_iter(small, large, |n| {
            let mut fuel = u64::MAX;
            let r = bytecode::compile_and_run(&m, 0, &[Value::I32(n)], &mut fuel)
                .expect("bytecode drives the kernel");
            std::hint::black_box(&r);
        });
        println!("svm-bytecode,{name},{bc:.4}");

        let jit = per_iter(small, large, |n| {
            let r = svm_jit::compile_and_run(&m, 0, &[n as i64]).expect("jit compiles + runs");
            std::hint::black_box(&r);
        });
        println!("svm-jit,{name},{jit:.4}");
    }
}

/// Per-iteration compute (ns) for `run_one(n)`, isolated by large/small-`n` subtraction and taken as
/// the min over repetitions (compute is deterministic; min rejects scheduler/noise spikes).
fn per_iter(small: i32, large: i32, run_one: impl Fn(i32)) -> f64 {
    let t_small = min_run(small, &run_one);
    let t_large = min_run(large, &run_one);
    (t_large - t_small) / (large - small) as f64
}

fn min_run(n: i32, run_one: &impl Fn(i32)) -> f64 {
    run_one(n); // warm up (the JIT's compile, the caches)
    let reps = 25;
    let mut best = f64::MAX;
    for _ in 0..reps {
        let start = Instant::now();
        run_one(n);
        best = best.min(start.elapsed().as_nanos() as f64);
    }
    best
}

// The kernels mirror `src/bin/bench.rs` exactly (so the SVM numbers are comparable run-to-run), and
// the external C / wasm / python drivers replicate the *same* computation.

/// `acc += n; n -= 1` until zero — a pure scalar/branch recurrence (sum 1..n, i32).
const ALU: &str = r#"
func (i32) -> (i32) {
block0(v0: i32):
  v1 = i32.const 0
  br block1(v0, v1)
block1(v2: i32, v3: i32):
  v4 = i32.add v3 v2
  v5 = i32.const 1
  v6 = i32.sub v2 v5
  br_if v6 block1(v6, v4) block2(v4)
block2(v7: i32):
  return v7
}
"#;

/// Each iteration calls a leaf `+1` function — the call/return kernel (window open/close cost).
const CALL: &str = r#"
func (i32) -> (i32) {
block0(v0: i32):
  v1 = i32.const 0
  br block1(v0, v1)
block1(v2: i32, v3: i32):
  v4 = call 1(v3)
  v5 = i32.const 1
  v6 = i32.sub v2 v5
  br_if v6 block1(v6, v4) block2(v4)
block2(v7: i32):
  return v7
}
func (i32) -> (i32) {
block0(v0: i32):
  v1 = i32.const 1
  v2 = i32.add v0 v1
  return v2
}
"#;

/// Each iteration dispatches through the `call_indirect` table — mask + slot read + type-check.
const CALL_INDIRECT: &str = r#"
func (i32) -> (i32) {
block0(v0: i32):
  v1 = i32.const 0
  br block1(v0, v1)
block1(v2: i32, v3: i32):
  v4 = i32.const 1
  v5 = call_indirect (i32) -> (i32) v4 (v3)
  v6 = i32.const 1
  v7 = i32.sub v2 v6
  br_if v7 block1(v7, v5) block2(v5)
block2(v8: i32):
  return v8
}
func (i32) -> (i32) {
block0(v0: i32):
  v1 = i32.const 1
  v2 = i32.add v0 v1
  return v2
}
"#;

/// Each iteration does one `i32.store` + one `i32.load` at a fixed address — the memory kernel.
const MEM: &str = r#"memory 16
func (i32) -> (i32) {
block0(v0: i32):
  v1 = i32.const 0
  br block1(v0, v1)
block1(v2: i32, v3: i32):
  v4 = i64.const 0
  i32.store v4 v3
  v5 = i32.load v4
  v6 = i32.const 1
  v7 = i32.add v5 v6
  v8 = i32.const 1
  v9 = i32.sub v2 v8
  br_if v9 block1(v9, v7) block2(v7)
block2(v10: i32):
  return v10
}
"#;

// Keep `ir` referenced (the parser returns `ir::Module`) without an unused-import warning if the
// signature ever changes — a no-op the optimizer drops.
#[allow(dead_code)]
fn _ir_ref(_m: &ir::Module) {}
