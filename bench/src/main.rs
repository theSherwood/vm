//! SVM JIT vs. Wasmtime — the relative-performance harness (`DESIGN.md` §1a, AGENTS.md
//! "benchmark early, measured relative to wasm/Wasmtime").
//!
//! Both engines lower the *same* algorithm through **Cranelift**, so this is a
//! like-for-like check of the design's perf thesis (§1a):
//!   - **steady-state compute → ≈ parity** ("we share the backend; we cannot out-run it on
//!     a tight inner loop"). A ratio near 1.0× is the expected, healthy result.
//!   - **cold start → we should be faster** ("SSA on the wire: no SSA reconstruction from a
//!     stack machine"). Source bytes → first result for a trivial program.
//!
//! Each kernel is written once in our IR text and once in the equivalent WAT; we assert the
//! two engines agree on the result before timing (so we never benchmark a miscompile).
//!
//! Methodology (kept simple + dependency-light, like `crates/svm/src/bin/bench.rs`):
//!   - *compute* is isolated by **subtraction**: time the kernel at a large and a small
//!     iteration count and divide the difference by the iteration delta. For our engine each
//!     timed run recompiles, but compile cost is identical at both counts so it cancels; for
//!     Wasmtime the module is compiled once and only the call is timed. Either way the result
//!     is per-iteration steady-state compute.
//!   - *cold start* times the whole path source → first result (n=0, so the loop body never
//!     runs but the full function is still compiled).
//!
//! This is a watch-it-over-time regression harness, not a statistical benchmark. Run with:
//!   cargo run --release            # from bench/
//!   cargo run --release -- --csv   # machine-readable line per kernel

use std::hint::black_box;
use std::time::Instant;

use svm_jit::{compile_and_run, JitOutcome};
use wasmtime::{Engine, Instance, Module, Store, TypedFunc};

struct Kernel {
    name: &'static str,
    /// Our IR text: `func (i64 n) -> (i64)`, entry = function 0.
    ir: &'static str,
    /// Equivalent core wasm: `(func (export "run") (param i64) (result i64))`.
    wat: &'static str,
}

/// `(i64 n) -> i64`: an LCG-style recurrence over `n` iterations — a tight `i64` mul/add
/// inner loop, the "compute parity" case (no memory).
const ALU: Kernel = Kernel {
    name: "alu",
    ir: "\
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const 0
  v2 = i64.const 0
  br block1(v0, v1, v2)
block1(v3: i64, v4: i64, v5: i64):
  v6 = i64.lt_s v5 v3
  br_if v6 block2(v3, v4, v5) block3(v4)
block2(v7: i64, v8: i64, v9: i64):
  v10 = i64.const 6364136223846793005
  v11 = i64.mul v8 v10
  v12 = i64.const 1442695040888963407
  v13 = i64.add v11 v12
  v14 = i64.add v13 v9
  v15 = i64.const 1
  v16 = i64.add v9 v15
  br block1(v7, v14, v16)
block3(v17: i64):
  return v17
}
",
    wat: r#"
(module
  (func (export "run") (param $n i64) (result i64)
    (local $acc i64) (local $i i64)
    (block $done
      (loop $loop
        (br_if $done (i64.ge_s (local.get $i) (local.get $n)))
        (local.set $acc
          (i64.add
            (i64.add
              (i64.mul (local.get $acc) (i64.const 6364136223846793005))
              (i64.const 1442695040888963407))
            (local.get $i)))
        (local.set $i (i64.add (local.get $i) (i64.const 1)))
        (br $loop)))
    (local.get $acc)))
"#,
};

/// `(i64 n) -> i64`: store then load `i` through a windowed address each iteration, so the
/// memory path (our 64-bit mask vs wasm32's bounds check) is exercised. Result = Σ i.
const MEMSUM: Kernel = Kernel {
    name: "memsum",
    ir: "\
memory 16
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const 0
  v2 = i64.const 0
  br block1(v0, v1, v2)
block1(v3: i64, v4: i64, v5: i64):
  v6 = i64.lt_s v5 v3
  br_if v6 block2(v3, v4, v5) block3(v4)
block2(v7: i64, v8: i64, v9: i64):
  v10 = i64.const 1023
  v11 = i64.and v9 v10
  v12 = i64.const 8
  v13 = i64.mul v11 v12
  i64.store v13 v9
  v14 = i64.load v13
  v15 = i64.add v8 v14
  v16 = i64.const 1
  v17 = i64.add v9 v16
  br block1(v7, v15, v17)
block3(v18: i64):
  return v18
}
",
    wat: r#"
(module
  (memory 1)
  (func (export "run") (param $n i64) (result i64)
    (local $acc i64) (local $i i64) (local $addr i32)
    (block $done
      (loop $loop
        (br_if $done (i64.ge_s (local.get $i) (local.get $n)))
        (local.set $addr
          (i32.mul (i32.and (i32.wrap_i64 (local.get $i)) (i32.const 1023)) (i32.const 8)))
        (i64.store (local.get $addr) (local.get $i))
        (local.set $acc (i64.add (local.get $acc) (i64.load (local.get $addr))))
        (local.set $i (i64.add (local.get $i) (i64.const 1)))
        (br $loop)))
    (local.get $acc)))
"#,
};

const KERNELS: &[Kernel] = &[ALU, MEMSUM];

const N_SMALL: i64 = 1_000;
const N_BIG: i64 = 2_000_000;

/// Compile + run our IR entry once and return the single `i64` result.
fn svm_call(m: &svm_ir::Module, n: i64) -> i64 {
    match compile_and_run(m, 0, &[n]) {
        Ok(JitOutcome::Returned(s)) => s[0],
        other => panic!("svm jit produced {other:?}"),
    }
}

/// Average wall time per call of `f`, in seconds, after a short warm-up.
fn per_call(iters: u32, mut f: impl FnMut()) -> f64 {
    for _ in 0..(iters / 4).max(1) {
        f();
    }
    let t = Instant::now();
    for _ in 0..iters {
        f();
    }
    t.elapsed().as_secs_f64() / iters as f64
}

fn main() {
    let csv = std::env::args().any(|a| a == "--csv");
    let engine = Engine::default(); // Cranelift, default opt level

    if !csv {
        println!(
            "SVM JIT vs Wasmtime — both via Cranelift (expect compute ≈ 1.0×, cold-start < 1.0×)\n\
             N_big={N_BIG} N_small={N_SMALL}\n"
        );
        println!(
            "{:<8} {:>14} {:>14} {:>7} {:>13} {:>13} {:>7}",
            "kernel", "svm ns/it", "wmt ns/it", "ratio", "svm cold ms", "wmt cold ms", "ratio"
        );
    }

    for k in KERNELS {
        let m = svm_text::parse_module(k.ir).expect("parse our IR text");
        let wasm = wat::parse_str(k.wat).expect("assemble WAT");

        // Compile the wasm once; instantiate once; grab the typed entry.
        let module = Module::new(&engine, &wasm).expect("wasmtime compile");
        let mut store = Store::new(&engine, ());
        let instance = Instance::new(&mut store, &module, &[]).expect("instantiate");
        let run: TypedFunc<i64, i64> = instance
            .get_typed_func(&mut store, "run")
            .expect("entry `run`");

        // Never benchmark a miscompile: the two engines must agree on the result.
        let ours = svm_call(&m, N_SMALL);
        let theirs = run.call(&mut store, N_SMALL).expect("wasmtime call");
        assert_eq!(
            ours, theirs,
            "kernel `{}` disagrees: svm={ours} wasmtime={theirs}",
            k.name
        );

        // --- steady-state compute (per-iteration, compile cancelled by subtraction) ---
        let svm_big = per_call(25, || {
            black_box(svm_call(&m, N_BIG));
        });
        let svm_small = per_call(25, || {
            black_box(svm_call(&m, N_SMALL));
        });
        let svm_ns = (svm_big - svm_small) * 1e9 / (N_BIG - N_SMALL) as f64;

        let wmt_big = per_call(100, || {
            black_box(run.call(&mut store, N_BIG).unwrap());
        });
        let wmt_small = per_call(100, || {
            black_box(run.call(&mut store, N_SMALL).unwrap());
        });
        let wmt_ns = (wmt_big - wmt_small) * 1e9 / (N_BIG - N_SMALL) as f64;

        // --- cold start: source bytes → first result for a trivial (n=0) program ---
        let svm_cold = per_call(60, || {
            black_box(svm_call(&m, 0));
        }) * 1e3;
        let wmt_cold = per_call(60, || {
            let module = Module::new(&engine, &wasm).unwrap();
            let mut s = Store::new(&engine, ());
            let inst = Instance::new(&mut s, &module, &[]).unwrap();
            let f: TypedFunc<i64, i64> = inst.get_typed_func(&mut s, "run").unwrap();
            black_box(f.call(&mut s, 0).unwrap());
        }) * 1e3;

        if csv {
            println!(
                "{},{:.3},{:.3},{:.3},{:.4},{:.4}",
                k.name, svm_ns, wmt_ns, svm_ns / wmt_ns, svm_cold, wmt_cold
            );
        } else {
            println!(
                "{:<8} {:>14.3} {:>14.3} {:>6.2}× {:>13.4} {:>13.4} {:>6.2}×",
                k.name,
                svm_ns,
                wmt_ns,
                svm_ns / wmt_ns,
                svm_cold,
                wmt_cold,
                svm_cold / wmt_cold
            );
        }
    }
}
