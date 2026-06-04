//! SVM JIT vs. Wasmtime — the relative-performance harness (`DESIGN.md` §1a, AGENTS.md
//! "benchmark early, measured relative to wasm/Wasmtime").
//!
//! Both engines lower the *same* algorithm through **Cranelift**, so this is a
//! like-for-like check of the design's perf thesis (§1a):
//!   - **steady-state compute → ≈ parity** ("we share the backend; we cannot out-run it on
//!     a tight inner loop"). A ratio near 1.0× is the expected, healthy result.
//!   - **cold start → we should be faster** ("SSA on the wire: no SSA reconstruction from a
//!     stack machine"). Source bytes → first result for a trivial program.
//!   - **memory: faster than wasm64, ~wash-or-worse than wasm32** (§1a). Our 64-bit window
//!     masks the final address (one `AND`); wasm32 gets the zero-instruction large-guard
//!     trick (so it wins), while wasm64 must emit an explicit bounds check per access (so a
//!     mask beats it). The memory kernel is therefore timed against *both* wasm memory types.
//!
//! Each kernel is written once in our IR text and once (or twice) in equivalent WAT; we
//! assert all engines agree on the result before timing (so we never benchmark a miscompile).
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
use wasmtime::{Config, Engine, Instance, Module, Store, TypedFunc};

struct Kernel {
    name: &'static str,
    /// Our IR text: `func (i64 n) -> (i64)`, entry = function 0.
    ir: &'static str,
    /// Core wasm32 (`(memory N)`): `(func (export "run") (param i64) (result i64))`.
    wat32: &'static str,
    /// Equivalent wasm64 (`(memory i64 N)`), for kernels that touch memory — `None` for
    /// pure-compute kernels, where the memory type is irrelevant.
    wat64: Option<&'static str>,
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
    wat32: r#"
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
    wat64: None,
};

/// `(i64 n) -> i64`: store then load `i` through a windowed address each iteration, so the
/// memory path is exercised. Result = Σ i (independent of where it lands). Timed against
/// both wasm32 (i32 address + guard page) and wasm64 (i64 address + bounds check); we use a
/// 64-bit masked address, so the design expects wasm32 < us < wasm64.
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
    wat32: r#"
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
    wat64: Some(
        r#"
(module
  (memory i64 1)
  (func (export "run") (param $n i64) (result i64)
    (local $acc i64) (local $i i64) (local $addr i64)
    (block $done
      (loop $loop
        (br_if $done (i64.ge_s (local.get $i) (local.get $n)))
        (local.set $addr
          (i64.mul (i64.and (local.get $i) (i64.const 1023)) (i64.const 8)))
        (i64.store (local.get $addr) (local.get $i))
        (local.set $acc (i64.add (local.get $acc) (i64.load (local.get $addr))))
        (local.set $i (i64.add (local.get $i) (i64.const 1)))
        (br $loop)))
    (local.get $acc)))
"#,
    ),
};

const KERNELS: &[Kernel] = &[ALU, MEMSUM, SCATTER];

/// `(i64 n) -> i64`: like `memsum` but the store and the load go to **different, per-iter
/// varying** slots — write slot `(i·M1)&1023`, read slot `(i·M2)&1023` (M1,M2 odd, so each
/// is a bijection mod 1024 → scattered across all slots). This defeats the same-address
/// bounds-check CSE/prefetch that `memsum` allowed, so it's the harder, more realistic test
/// of "mask vs bounds check" — does our memory gap survive when accesses are varied?
const SCATTER: Kernel = Kernel {
    name: "scatter",
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
  v10 = i64.const 2654435761
  v11 = i64.mul v9 v10
  v12 = i64.const 1023
  v13 = i64.and v11 v12
  v14 = i64.const 8
  v15 = i64.mul v13 v14
  i64.store v15 v9
  v16 = i64.const 2246822519
  v17 = i64.mul v9 v16
  v18 = i64.and v17 v12
  v19 = i64.mul v18 v14
  v20 = i64.load v19
  v21 = i64.add v8 v20
  v22 = i64.const 1
  v23 = i64.add v9 v22
  br block1(v7, v21, v23)
block3(v24: i64):
  return v24
}
",
    wat32: r#"
(module
  (memory 1)
  (func (export "run") (param $n i64) (result i64)
    (local $acc i64) (local $i i64)
    (block $done
      (loop $loop
        (br_if $done (i64.ge_s (local.get $i) (local.get $n)))
        (i64.store
          (i32.mul (i32.and (i32.wrap_i64 (i64.mul (local.get $i) (i64.const 2654435761)))
                            (i32.const 1023)) (i32.const 8))
          (local.get $i))
        (local.set $acc
          (i64.add (local.get $acc)
            (i64.load
              (i32.mul (i32.and (i32.wrap_i64 (i64.mul (local.get $i) (i64.const 2246822519)))
                                (i32.const 1023)) (i32.const 8)))))
        (local.set $i (i64.add (local.get $i) (i64.const 1)))
        (br $loop)))
    (local.get $acc)))
"#,
    wat64: Some(
        r#"
(module
  (memory i64 1)
  (func (export "run") (param $n i64) (result i64)
    (local $acc i64) (local $i i64)
    (block $done
      (loop $loop
        (br_if $done (i64.ge_s (local.get $i) (local.get $n)))
        (i64.store
          (i64.mul (i64.and (i64.mul (local.get $i) (i64.const 2654435761))
                            (i64.const 1023)) (i64.const 8))
          (local.get $i))
        (local.set $acc
          (i64.add (local.get $acc)
            (i64.load
              (i64.mul (i64.and (i64.mul (local.get $i) (i64.const 2246822519))
                                (i64.const 1023)) (i64.const 8)))))
        (local.set $i (i64.add (local.get $i) (i64.const 1)))
        (br $loop)))
    (local.get $acc)))
"#,
    ),
};

const N_SMALL: i64 = 1_000;
const N_BIG: i64 = 2_000_000;

/// Compile + run our IR entry once and return the single `i64` result.
fn svm_call(m: &svm_ir::Module, n: i64) -> i64 {
    match compile_and_run(m, 0, &[n]) {
        Ok(JitOutcome::Returned(s)) => s[0],
        other => panic!("svm jit produced {other:?}"),
    }
}

/// Compile + instantiate a wasm module and return its `(i64) -> i64` entry, store and all.
fn wasm_entry(engine: &Engine, wasm: &[u8]) -> (Store<()>, TypedFunc<i64, i64>) {
    let module = Module::new(engine, wasm).expect("wasmtime compile");
    let mut store = Store::new(engine, ());
    let instance = Instance::new(&mut store, &module, &[]).expect("instantiate");
    let run = instance
        .get_typed_func(&mut store, "run")
        .expect("entry `run`");
    (store, run)
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

/// Per-iteration steady-state compute (ns) of a compiled wasm entry, via subtraction.
fn wasm_compute_ns(store: &mut Store<()>, run: &TypedFunc<i64, i64>) -> f64 {
    let big = per_call(100, || {
        black_box(run.call(&mut *store, N_BIG).unwrap());
    });
    let small = per_call(100, || {
        black_box(run.call(&mut *store, N_SMALL).unwrap());
    });
    (big - small) * 1e9 / (N_BIG - N_SMALL) as f64
}

fn main() {
    let csv = std::env::args().any(|a| a == "--csv");
    // Enable the memory64 proposal so `(memory i64 …)` modules compile; it does not change
    // how wasm32 modules are lowered, so the wasm32 numbers stay comparable.
    let mut config = Config::new();
    config.wasm_memory64(true);
    let engine = Engine::new(&config).expect("engine");

    if !csv {
        println!(
            "SVM JIT vs Wasmtime — both via Cranelift.  ratio = svm / wasm  (<1 = svm faster)\n\
             Expect: alu compute ≈1×; cold-start <1×.  Memory: wasm32 < svm always (guard\n\
             pages are free); svm < wasm64 once addresses *vary* (scatter) so Wasmtime can't\n\
             CSE the bounds check — memsum (same addr) lets it, so wasm64 looks ~tied there.\n\
             N_big={N_BIG} N_small={N_SMALL}\n"
        );
        println!(
            "{:<8} | {:>8} {:>8} {:>6} | {:>8} {:>6} | {:>8} {:>8} {:>6}",
            "kernel", "svm", "wasm32", "ratio", "wasm64", "ratio", "svm", "wasm32", "ratio"
        );
        println!(
            "{:<8} | {:>8} {:>8} {:>6} | {:>8} {:>6} | {:>8} {:>8} {:>6}",
            "", "ns/it", "ns/it", "", "ns/it", "", "cold ms", "cold ms", ""
        );
    }

    for k in KERNELS {
        let m = svm_text::parse_module(k.ir).expect("parse our IR text");
        let wasm32 = wat::parse_str(k.wat32).expect("assemble wasm32 WAT");
        let (mut s32, run32) = wasm_entry(&engine, &wasm32);

        // Cross-check every engine agrees before timing (never benchmark a miscompile).
        let ours = svm_call(&m, N_SMALL);
        assert_eq!(
            ours,
            run32.call(&mut s32, N_SMALL).unwrap(),
            "kernel `{}`: svm vs wasm32 disagree",
            k.name
        );

        // --- steady-state compute ---
        let svm_big = per_call(25, || {
            black_box(svm_call(&m, N_BIG));
        });
        let svm_small = per_call(25, || {
            black_box(svm_call(&m, N_SMALL));
        });
        let svm_ns = (svm_big - svm_small) * 1e9 / (N_BIG - N_SMALL) as f64;
        let w32_ns = wasm_compute_ns(&mut s32, &run32);

        let w64 = k.wat64.map(|wat| {
            let wasm64 = wat::parse_str(wat).expect("assemble wasm64 WAT");
            let (mut s64, run64) = wasm_entry(&engine, &wasm64);
            assert_eq!(
                ours,
                run64.call(&mut s64, N_SMALL).unwrap(),
                "kernel `{}`: svm vs wasm64 disagree",
                k.name
            );
            wasm_compute_ns(&mut s64, &run64)
        });

        // --- cold start: source bytes → first result for a trivial (n=0) program (wasm32) ---
        let svm_cold = per_call(60, || {
            black_box(svm_call(&m, 0));
        }) * 1e3;
        let wmt_cold = per_call(60, || {
            let (mut s, f) = wasm_entry(&engine, &wasm32);
            black_box(f.call(&mut s, 0).unwrap());
        }) * 1e3;

        if csv {
            let (w64s, r64) = match w64 {
                Some(v) => (format!("{v:.3}"), format!("{:.3}", svm_ns / v)),
                None => ("NA".into(), "NA".into()),
            };
            println!(
                "{},{:.3},{:.3},{:.3},{w64s},{r64},{:.4},{:.4},{:.3}",
                k.name,
                svm_ns,
                w32_ns,
                svm_ns / w32_ns,
                svm_cold,
                wmt_cold,
                svm_cold / wmt_cold
            );
        } else {
            let (w64s, r64) = match w64 {
                Some(v) => (format!("{v:.3}"), format!("{:.2}×", svm_ns / v)),
                None => ("—".into(), "—".into()),
            };
            println!(
                "{:<8} | {:>8.3} {:>8.3} {:>5.2}× | {:>8} {:>6} | {:>8.4} {:>8.4} {:>5.2}×",
                k.name,
                svm_ns,
                w32_ns,
                svm_ns / w32_ns,
                w64s,
                r64,
                svm_cold,
                wmt_cold,
                svm_cold / wmt_cold
            );
        }
    }
}
