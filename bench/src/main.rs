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
//! One kernel (`alu_c`) instead gets its IR from the **chibicc frontend** (the same recurrence
//! as `alu`, compiled from C), so its steady-state time tracks the **SSA-promotion win
//! end-to-end** — it should stay at ≈parity with `alu`; if a promotable loop body regressed to
//! memory it would drift toward the memory-bound path. It is skipped if the frontend can't build.
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
//!   cargo run --release                          # from bench/, human table
//!   cargo run --release -- --csv                 # machine-readable line per kernel
//!   cargo run --release -- --save-baseline FILE  # record the current ratios
//!   cargo run --release -- --check FILE           # rerun + flag any ratio regression
//!
//! **Over-time regression tracking (AGENTS.md "catch regressions one commit old").** The
//! absolute ns are machine-dependent, so the *tracked* quantity is the **ratio** (svm ÷
//! wasm) per kernel — far more portable across machines than the raw timings. `--save-baseline`
//! writes the three ratios per kernel (compute-vs-wasm32, compute-vs-wasm64, cold-vs-Wasmtime)
//! to a committed file; `--check` reruns and **exits non-zero** if any ratio has grown by more
//! than `--tol` (default 25%, i.e. svm got relatively slower) — that band absorbs runner noise
//! while still catching a real regression (e.g. losing mask-elision moved `scatter` ≈1.21→1.53,
//! +26%; losing SSA promotion would be far larger). `--check`/`--save-baseline` default to
//! `--reps 5` (best-of, to stabilise the comparison); plain/`--csv` use one pass for speed.

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

/// Compile + run a kernel's IR entry once and return the single `i64` result. `lead` is the
/// fixed leading args (e.g. the data-stack pointer chibicc threads as v0); `n` is appended.
fn svm_call(m: &svm_ir::Module, entry: u32, lead: &[i64], n: i64) -> i64 {
    let mut args: Vec<i64> = lead.to_vec();
    args.push(n);
    match compile_and_run(m, entry, &args) {
        Ok(JitOutcome::Returned(s)) => s[0],
        other => panic!("svm jit produced {other:?}"),
    }
}

/// A kernel resolved for this run: IR text — hand-written, or generated from C through the
/// chibicc frontend — plus how to invoke its entry. `svm_call` appends `n` after `lead_args`.
struct Resolved {
    name: String,
    ir: String,
    entry: u32,
    lead_args: Vec<i64>,
    wat32: String,
    wat64: Option<String>,
}

/// The hand-written [`KERNELS`] plus, when the C frontend is buildable, the chibicc-lowered
/// kernels (`alu_c`, `locals_c` — see [`alu_from_c`] / [`locals_from_c`]).
fn resolve_kernels() -> Vec<Resolved> {
    let mut v: Vec<Resolved> = KERNELS
        .iter()
        .map(|k| Resolved {
            name: k.name.to_string(),
            ir: k.ir.to_string(),
            entry: 0,
            lead_args: Vec::new(),
            wat32: k.wat32.to_string(),
            wat64: k.wat64.map(|w| w.to_string()),
        })
        .collect();
    for k in [alu_from_c(), locals_from_c()] {
        match k {
            Ok(r) => v.push(r),
            Err(e) => eprintln!("note: skipping a C-frontend kernel (frontend unavailable): {e}"),
        }
    }
    v
}

/// Compile `src` through the chibicc frontend and wrap its `run` as a kernel timed against
/// `wat32`. `run` is found by signature — the unique `(i64, i64) -> (i64)` function (`main`
/// returns i32, `_start` takes three i32s) — so this is robust against the frontend's function
/// ordering. `lead` is the args before `n`: `run` threads the data-stack pointer as v0, so it is
/// the initial SP (0 is safe here — the frame is tiny and self-contained). Returns `Err` (caller
/// skips the kernel) if the frontend can't be built/run.
fn c_kernel(name: &str, src: &str, lead: Vec<i64>, wat32: String) -> Result<Resolved, String> {
    let ir = compile_c_to_ir(src)?;
    let m = svm_text::parse_module(&ir).map_err(|e| format!("parse frontend IR: {e:?}"))?;
    let entry = m
        .funcs
        .iter()
        .position(|f| {
            f.params == [svm_ir::ValType::I64, svm_ir::ValType::I64]
                && f.results == [svm_ir::ValType::I64]
        })
        .ok_or("no `run(i64,i64)->i64` entry in frontend output")? as u32;
    Ok(Resolved {
        name: name.to_string(),
        ir,
        entry,
        lead_args: lead,
        wat32,
        wat64: None,
    })
}

/// The same LCG recurrence as `alu`, but lowered from C — so its steady-state time tracks the
/// **SSA-promotion win end-to-end**: if a promotable loop body regressed back to memory, `alu_c`
/// would drift toward the memory-bound path while the hand-written (already register-only) `alu`
/// would not. Reuses `alu`'s WAT as the oracle, since the algorithm is identical.
fn alu_from_c() -> Result<Resolved, String> {
    const SRC: &str = "long run(long n){\n  long acc = 0;\n  \
        for (long i = 0; i < n; i++)\n    \
        acc = acc * 6364136223846793005L + 1442695040888963407L + i;\n  \
        return acc;\n}\nint main(){ return (int)run(0); }\n";
    c_kernel("alu_c", SRC, vec![0], ALU.wat32.to_string())
}

/// A **data-SP–relative** memory loop from C: an address-taken `volatile` stack array, so each
/// iteration stores/loads through `sp + (i & 255)*8` — and `sp` is an *unbounded* i64 block
/// param, so the JIT cannot prove the address in-window and masks every access. This is exactly
/// the case the large-reserved-window / guard-when-bounded work (§4, §1a) targets: it should
/// move toward wasm32 parity once the SP base is provably bounded and the per-access mask elides.
/// `memsum`/`scatter` don't exercise it (their indices are already provably small ⇒ pre-elided).
fn locals_from_c() -> Result<Resolved, String> {
    const SRC: &str = "long run(long n){\n  volatile long a[256];\n  long acc = 0;\n  \
        for (long i = 0; i < n; i++) { a[i & 255] = i; acc += a[i & 255]; }\n  \
        return acc;\n}\nint main(){ return (int)run(0); }\n";
    // wasm32 oracle: the same store-then-load-and-sum through a windowed slot `(i&255)*8`.
    // Result is Σ i (the slot is overwritten then read back each iteration), matching the C run.
    const WAT32: &str = r#"
(module
  (memory 1)
  (func (export "run") (param $n i64) (result i64)
    (local $acc i64) (local $i i64) (local $addr i32)
    (block $done
      (loop $loop
        (br_if $done (i64.ge_s (local.get $i) (local.get $n)))
        (local.set $addr
          (i32.mul (i32.and (i32.wrap_i64 (local.get $i)) (i32.const 255)) (i32.const 8)))
        (i64.store (local.get $addr) (local.get $i))
        (local.set $acc (i64.add (local.get $acc) (i64.load (local.get $addr))))
        (local.set $i (i64.add (local.get $i) (i64.const 1)))
        (br $loop)))
    (local.get $acc)))
"#;
    c_kernel("locals_c", SRC, vec![0], WAT32.to_string())
}

/// Build the chibicc fork (idempotent `make`) and compile `src` to our text IR. Returns `Err`
/// (so the caller can skip the kernel) if the C toolchain / frontend is unavailable.
fn compile_c_to_ir(src: &str) -> Result<String, String> {
    use std::process::Command;
    // `CARGO_MANIFEST_DIR` is `<repo>/bench`; the frontend lives at `<repo>/frontend/chibicc`.
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .ok_or("no repo root above CARGO_MANIFEST_DIR")?;
    let dir = root.join("frontend/chibicc");
    let ok = Command::new("make")
        .arg("-s")
        .current_dir(&dir)
        .status()
        .map_err(|e| format!("run make: {e}"))?
        .success();
    if !ok {
        return Err("chibicc build failed".into());
    }
    let base = std::env::temp_dir().join(format!("svm_bench_{}", std::process::id()));
    let cfile = base.with_extension("c");
    let irfile = base.with_extension("svm");
    std::fs::write(&cfile, src).map_err(|e| format!("write temp C: {e}"))?;
    let ok = Command::new(dir.join("chibicc"))
        .args([
            "-cc1",
            "--emit-ir",
            "-cc1-input",
            cfile.to_str().unwrap(),
            "-cc1-output",
            irfile.to_str().unwrap(),
            cfile.to_str().unwrap(),
        ])
        .status()
        .map_err(|e| format!("run chibicc: {e}"))?
        .success();
    if !ok {
        return Err("chibicc --emit-ir failed".into());
    }
    std::fs::read_to_string(&irfile).map_err(|e| format!("read frontend IR: {e}"))
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

/// Raw per-iteration timings for one kernel (ns for compute, ms for cold start), each the
/// **min over `reps`** measurements (best observed per engine — the noise floor we compare).
struct Raw {
    svm_ns: f64,
    w32_ns: f64,
    w64_ns: Option<f64>,
    svm_cold: f64,
    wmt_cold: f64,
}

impl Raw {
    /// The three machine-portable ratios we track: compute vs wasm32, compute vs wasm64
    /// (when the kernel has a wasm64 form), cold start vs Wasmtime. Higher = svm slower.
    fn ratios(&self) -> (f64, Option<f64>, f64) {
        (
            self.svm_ns / self.w32_ns,
            self.w64_ns.map(|v| self.svm_ns / v),
            self.svm_cold / self.wmt_cold,
        )
    }
}

/// Time one kernel, taking the **best (min)** of `reps` passes per engine. Cross-checks every
/// engine agrees on the result first, so we never benchmark a miscompile.
fn measure(engine: &Engine, k: &Resolved, reps: u32) -> Raw {
    let m = svm_text::parse_module(&k.ir).expect("parse our IR text");
    let wasm32 = wat::parse_str(&k.wat32).expect("assemble wasm32 WAT");
    let wasm64 = k
        .wat64
        .as_deref()
        .map(|wat| wat::parse_str(wat).expect("assemble wasm64 WAT"));
    let svm = |n: i64| svm_call(&m, k.entry, &k.lead_args, n);

    // Cross-check every engine agrees before timing (never benchmark a miscompile).
    let ours = svm(N_SMALL);
    {
        let (mut s32, run32) = wasm_entry(engine, &wasm32);
        assert_eq!(
            ours,
            run32.call(&mut s32, N_SMALL).unwrap(),
            "kernel `{}`: svm vs wasm32 disagree",
            k.name
        );
        if let Some(w) = &wasm64 {
            let (mut s64, run64) = wasm_entry(engine, w);
            assert_eq!(
                ours,
                run64.call(&mut s64, N_SMALL).unwrap(),
                "kernel `{}`: svm vs wasm64 disagree",
                k.name
            );
        }
    }

    let mut raw = Raw {
        svm_ns: f64::INFINITY,
        w32_ns: f64::INFINITY,
        w64_ns: wasm64.as_ref().map(|_| f64::INFINITY),
        svm_cold: f64::INFINITY,
        wmt_cold: f64::INFINITY,
    };
    for _ in 0..reps.max(1) {
        // --- steady-state compute (subtraction isolates the loop body) ---
        let svm_big = per_call(25, || {
            black_box(svm(N_BIG));
        });
        let svm_small = per_call(25, || {
            black_box(svm(N_SMALL));
        });
        raw.svm_ns = raw
            .svm_ns
            .min((svm_big - svm_small) * 1e9 / (N_BIG - N_SMALL) as f64);

        let (mut s32, run32) = wasm_entry(engine, &wasm32);
        raw.w32_ns = raw.w32_ns.min(wasm_compute_ns(&mut s32, &run32));

        if let Some(w) = &wasm64 {
            let (mut s64, run64) = wasm_entry(engine, w);
            let v = wasm_compute_ns(&mut s64, &run64);
            raw.w64_ns = Some(raw.w64_ns.unwrap().min(v));
        }

        // --- cold start: source bytes → first result for a trivial (n=0) program (wasm32) ---
        let svm_cold = per_call(60, || {
            black_box(svm(0));
        }) * 1e3;
        raw.svm_cold = raw.svm_cold.min(svm_cold);
        let wmt_cold = per_call(60, || {
            let (mut s, f) = wasm_entry(engine, &wasm32);
            black_box(f.call(&mut s, 0).unwrap());
        }) * 1e3;
        raw.wmt_cold = raw.wmt_cold.min(wmt_cold);
    }
    raw
}

/// Value following `flag` in the arg list, if present (`--flag value`).
fn flag_value(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1).cloned())
}

/// Write the tracked ratios to a baseline file (`kernel,compute32,compute64,cold`; `NA` for a
/// kernel with no wasm64 form). The header documents what the numbers are + how to regenerate.
fn save_baseline(path: &str, results: &[(Resolved, Raw)]) {
    let mut out = String::from(
        "# svm-bench baseline — the tracked signal is the RATIO (svm / wasm), which is far more\n\
         # portable across machines than the absolute ns. `--check` flags any ratio that grew\n\
         # past the tolerance (svm got relatively slower). Regenerate after an intended change:\n\
         #   cargo run --release -- --save-baseline baseline.txt\n\
         # columns: kernel,compute_vs_wasm32,compute_vs_wasm64,cold_vs_wasmtime\n",
    );
    for (k, raw) in results {
        let (c32, c64, cold) = raw.ratios();
        let c64s = c64
            .map(|v| format!("{v:.3}"))
            .unwrap_or_else(|| "NA".into());
        out.push_str(&format!("{},{:.3},{c64s},{:.3}\n", k.name, c32, cold));
    }
    std::fs::write(path, out).unwrap_or_else(|e| panic!("write baseline `{path}`: {e}"));
    eprintln!("wrote baseline to {path}");
}

/// One tracked baseline row loaded from a file.
struct BaseRow {
    compute32: f64,
    compute64: Option<f64>,
    cold: f64,
}

/// Parse a baseline file written by [`save_baseline`] (comments/blank lines skipped).
fn load_baseline(path: &str) -> std::collections::HashMap<String, BaseRow> {
    let text =
        std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read baseline `{path}`: {e}"));
    let mut map = std::collections::HashMap::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        assert!(f.len() == 4, "baseline line `{line}`: want 4 fields");
        map.insert(
            f[0].to_string(),
            BaseRow {
                compute32: f[1].parse().expect("compute32"),
                compute64: (f[2] != "NA").then(|| f[2].parse().expect("compute64")),
                cold: f[3].parse().expect("cold"),
            },
        );
    }
    map
}

/// Rerun, compare each ratio to the baseline, print a table, and return `true` if **no** ratio
/// regressed past `tol` (a fractional increase, e.g. `0.25` = 25%). A missing baseline kernel
/// is reported but not a regression; an unexpectedly *improved* ratio just prints `ok`.
fn check_baseline(path: &str, results: &[(Resolved, Raw)], tol: f64) -> bool {
    let base = load_baseline(path);
    println!(
        "regression check vs {path}  (tol {:.0}%, ratio = svm/wasm, lower is better)\n",
        tol * 100.0
    );
    println!(
        "{:<8} {:<16} {:>9} {:>9} {:>8}  status",
        "kernel", "metric", "baseline", "current", "delta"
    );
    let mut ok = true;
    for (k, raw) in results {
        let Some(b) = base.get(k.name.as_str()) else {
            println!(
                "{:<8} {:<16} {:>9} {:>9} {:>8}  MISSING",
                k.name, "(all)", "-", "-", "-"
            );
            continue;
        };
        let (c32, c64, cold) = raw.ratios();
        let mut row = |metric: &str, baseline: f64, current: f64| {
            let delta = current / baseline - 1.0;
            let regressed = delta > tol;
            ok &= !regressed;
            println!(
                "{:<8} {:<16} {:>9.3} {:>9.3} {:>+7.1}%  {}",
                k.name,
                metric,
                baseline,
                current,
                delta * 100.0,
                if regressed { "REGRESSED" } else { "ok" }
            );
        };
        row("compute/wasm32", b.compute32, c32);
        if let (Some(bv), Some(cv)) = (b.compute64, c64) {
            row("compute/wasm64", bv, cv);
        }
        row("cold/wasmtime", b.cold, cold);
    }
    if ok {
        println!("\nOK - no ratio regressed past {:.0}%.", tol * 100.0);
    } else {
        println!(
            "\nREGRESSION - a ratio grew past {:.0}% (svm got relatively slower). If intended,\n\
             re-baseline with `--save-baseline {path}`.",
            tol * 100.0
        );
    }
    ok
}

fn print_table(results: &[(Resolved, Raw)]) {
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
    for (k, raw) in results {
        let (c32, c64, cold) = raw.ratios();
        let (w64s, r64) = match (raw.w64_ns, c64) {
            (Some(v), Some(r)) => (format!("{v:.3}"), format!("{r:.2}×")),
            _ => ("—".into(), "—".into()),
        };
        println!(
            "{:<8} | {:>8.3} {:>8.3} {:>5.2}× | {:>8} {:>6} | {:>8.4} {:>8.4} {:>5.2}×",
            k.name, raw.svm_ns, raw.w32_ns, c32, w64s, r64, raw.svm_cold, raw.wmt_cold, cold
        );
    }
}

fn print_csv(results: &[(Resolved, Raw)]) {
    for (k, raw) in results {
        let (c32, c64, cold) = raw.ratios();
        let (w64s, r64) = match (raw.w64_ns, c64) {
            (Some(v), Some(r)) => (format!("{v:.3}"), format!("{r:.3}")),
            _ => ("NA".into(), "NA".into()),
        };
        println!(
            "{},{:.3},{:.3},{:.3},{w64s},{r64},{:.4},{:.4},{:.3}",
            k.name, raw.svm_ns, raw.w32_ns, c32, raw.svm_cold, raw.wmt_cold, cold
        );
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let csv = args.iter().any(|a| a == "--csv");
    let save = flag_value(&args, "--save-baseline");
    let check = flag_value(&args, "--check");
    let tol = flag_value(&args, "--tol")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.25);
    // `--check`/`--save-baseline` take best-of-5 to stabilise the comparison; the live views
    // (table/csv) use a single fast pass. `--reps N` overrides.
    let reps = flag_value(&args, "--reps")
        .and_then(|s| s.parse().ok())
        .unwrap_or(if save.is_some() || check.is_some() {
            5
        } else {
            1
        });

    // Enable the memory64 proposal so `(memory i64 …)` modules compile; it does not change
    // how wasm32 modules are lowered, so the wasm32 numbers stay comparable.
    let mut config = Config::new();
    config.wasm_memory64(true);
    let engine = Engine::new(&config).expect("engine");

    let results: Vec<(Resolved, Raw)> = resolve_kernels()
        .into_iter()
        .map(|k| {
            let raw = measure(&engine, &k, reps);
            (k, raw)
        })
        .collect();

    if let Some(path) = save {
        save_baseline(&path, &results);
    } else if let Some(path) = check {
        if !check_baseline(&path, &results, tol) {
            std::process::exit(1);
        }
    } else if csv {
        print_csv(&results);
    } else {
        print_table(&results);
    }
}
