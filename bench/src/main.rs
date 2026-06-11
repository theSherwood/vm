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
//!   - **interface / host calls → the "around-compute" axis** (§1a, the strongest claimed
//!     win). `hostcall` times a scalar `cap.call` round-trip vs a Wasmtime imported function;
//!     `hostbuf` times a zero-copy `(ptr,len)` **borrow buffer** the host reads in place (§7)
//!     vs a (cached-memory) wasm import doing the same read. Honest current state: scalar
//!     cap.call is *slower* (a generic arg-packing thunk; the devirtualize-to-direct-call
//!     win, D45, is deferred), while the zero-copy buffer path is *faster* (the host gets the
//!     window base for free). The larger §1a claim — vs the component model's lift/lower
//!     marshalling, and async rings — is a heavier comparison, not attempted here.
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
//!   cargo run --release -- --from-wasm           # SVM IR transpiled from the WAT (same bytes as wasm)
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

use std::ffi::c_void;
use std::hint::black_box;
use std::time::Instant;

use std::sync::atomic::{AtomicBool, Ordering};
use svm_jit::{
    compile_and_run, compile_and_run_with_host, compile_and_run_with_host_fast, JitOutcome,
};

/// `--fast-cap`: drive `HostCall` kernels through the §9/D45 devirtualized fast path
/// ([`compile_and_run_with_host_fast`] + [`bench_fast_resolver`]) instead of the generic thunk, to
/// measure the register-to-register win.
static FAST_CAP: AtomicBool = AtomicBool::new(false);
use wasmtime::{Caller, Config, Engine, Instance, Linker, Memory, Module, Store, TypedFunc};

/// Wasmtime store state: the host-call buffer benchmark **caches the exported `Memory`** here
/// (populated after instantiation) so its host fn accesses linear memory without a per-call
/// `get_export("memory")` string lookup — the fair, perf-conscious wasm baseline. `None` for
/// compute kernels, which never touch host state.
type HostState = Option<Memory>;

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
// Host-call kernels do real boundary crossings per iteration (≫ a compute op), so they use a
// smaller iteration span — still large enough that the subtraction isolates per-call cost.
const N_HOST_SMALL: i64 = 1_000;
const N_HOST_BIG: i64 = 200_000;

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

/// A minimal capability host trampoline for the interface benchmark: op 0 is a scalar
/// round-trip (`x -> x+1`); op 1 sums a `(ptr, len)` **borrow buffer** read in place from the
/// window (the §7 zero-copy path). It does the least work that still forces the call, so the
/// timing isolates the boundary-crossing cost rather than the work.
///
/// # Safety
/// Honours the [`svm_jit::CapThunk`] contract: `args`/`results` are valid for their declared
/// lengths, and for op 1 the kernel passes in-window constants so `[ptr, ptr+len) ⊆ window`.
unsafe extern "C" fn bench_thunk(
    _ctx: *mut c_void,
    mem_base: *mut u8,
    _mem_size: u64,
    _mem_reserved: u64,
    _type_id: u32,
    op: u32,
    _handle: i32,
    args: *const i64,
    n_args: u64,
    results: *mut i64,
    n_results: u64,
    trap_out: *mut i64,
) {
    let a = std::slice::from_raw_parts(args, n_args as usize);
    let r: i64 = match op {
        // op 1: sum the borrow buffer in place (no copy) — the §7 zero-copy I/O path.
        1 => {
            let (ptr, len) = (a[0] as usize, a[1] as usize);
            let buf = std::slice::from_raw_parts(mem_base.add(ptr), len);
            buf.iter().map(|&b| b as i64).sum()
        }
        // op 0: scalar round-trip.
        _ => a[0].wrapping_add(1),
    };
    if n_results > 0 {
        *results = r;
    }
    *trap_out = 0;
}

/// The §9/D45 **devirtualized** counterparts of [`bench_thunk`]'s ops: register-to-register host fns
/// the JIT calls directly when `--fast-cap` is set. `op 0` (`x -> x+1`) and `op 1` (sum a `(ptr,len)`
/// borrow buffer) — identical results to the generic thunk, but no stack marshalling / runtime
/// dispatch. The ABI matches [`svm_jit::FastCapResolver`]: `(ctx, mem_base, mem_size, handle, trap_out,
/// args…)`.
unsafe extern "C" fn fast_op0(
    _ctx: *mut c_void,
    _mem_base: *mut u8,
    _mem_size: u64,
    _handle: i32,
    trap_out: *mut i64,
    a0: i64,
) -> i64 {
    *trap_out = 0;
    a0.wrapping_add(1)
}
unsafe extern "C" fn fast_op1(
    _ctx: *mut c_void,
    mem_base: *mut u8,
    _mem_size: u64,
    _handle: i32,
    trap_out: *mut i64,
    ptr: i64,
    len: i64,
) -> i64 {
    *trap_out = 0;
    let buf = std::slice::from_raw_parts(mem_base.add(ptr as usize), len as usize);
    buf.iter().map(|&b| b as i64).sum()
}
unsafe extern "C" fn bench_fast_resolver(
    _type_id: u32,
    op: u32,
    n_args: u32,
    n_res: u32,
) -> *const c_void {
    // Only claim an op when the IR arity matches the specialized fn's (else the generic path).
    match (op, n_args, n_res) {
        (0, 1, 1) => fast_op0 as *const c_void, // x -> x+1
        (1, 2, 1) => fast_op1 as *const c_void, // sum a (ptr,len) buffer
        _ => std::ptr::null(),
    }
}

/// Like [`svm_call`] but drives the cap.call trampoline ([`bench_thunk`]) — for `HostCall`
/// kernels. The context pointer is unused (the thunk is stateless), so it is null. With `--fast-cap`
/// the call instead takes the §9/D45 devirtualized fast path via [`bench_fast_resolver`].
fn svm_call_host(m: &svm_ir::Module, entry: u32, lead: &[i64], n: i64) -> i64 {
    let mut args: Vec<i64> = lead.to_vec();
    args.push(n);
    let out = if FAST_CAP.load(Ordering::Relaxed) {
        compile_and_run_with_host_fast(
            m,
            entry,
            &args,
            bench_thunk,
            std::ptr::null_mut(),
            bench_fast_resolver,
            svm_jit::Quota::default(),
        )
    } else {
        compile_and_run_with_host(m, entry, &args, bench_thunk, std::ptr::null_mut())
    };
    match out {
        Ok(JitOutcome::Returned(s)) => s[0],
        other => panic!("svm jit (host-call) produced {other:?}"),
    }
}

/// What a kernel measures. `Compute` kernels run an import-less wasm module and the no-cap
/// SVM JIT (per-iteration *compute*). `HostCall` kernels instead make one host crossing per
/// iteration — SVM `cap.call` through a trampoline thunk vs a Wasmtime **imported host
/// function** — so the subtraction isolates the *per-host-call* cost (§1a interface axis).
#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Compute,
    HostCall,
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
    mode: Mode,
}

/// The hand-written [`KERNELS`] plus, when the C frontend is buildable, the chibicc-lowered
/// kernels (`alu_c`, `locals_c` — see [`alu_from_c`] / [`locals_from_c`]).
///
/// With `from_wasm`, each kernel's SVM IR is replaced by transpiling its `wat32` through `svm-wasm`
/// (the same bytes Wasmtime runs) — the genuine apples-to-apples comparison. This now covers the
/// **`hostcall` / `hostbuf`** interface kernels too: their `host.op` / `host.sum` imports use the
/// host-ABI convention (`module` = capability type_id, `name` = op), so they transpile to the same
/// `cap.call` the hand-written IR used. Kernels the transpiler can't handle keep their hand-written
/// IR, with a note saying why.
///
/// **svm-wasm doesn't transpile (so these keep their hand-written IR under `--from-wasm`):**
/// **`memory.grow` / `memory.size`**, **passive** data/element segments, imports across multiple
/// capability interfaces, **SIMD (v128)**, and **reference types** beyond funcref tables. (Supported:
/// i32/i64/f32/f64 numeric + all conversions, locals, the full structured control set, linear memory,
/// direct + indirect calls, globals, active data/element segments, and **function imports / the host
/// ABI** — enough for every kernel here.)
fn resolve_kernels(from_wasm: bool) -> Vec<Resolved> {
    let mut v: Vec<Resolved> = KERNELS
        .iter()
        .map(|k| Resolved {
            name: k.name.to_string(),
            ir: k.ir.to_string(),
            entry: 0,
            lead_args: Vec::new(),
            wat32: k.wat32.to_string(),
            wat64: k.wat64.map(|w| w.to_string()),
            mode: Mode::Compute,
        })
        .collect();
    for k in [alu_from_c(), locals_from_c()] {
        match k {
            Ok(r) => v.push(r),
            Err(e) => eprintln!("note: skipping a C-frontend kernel (frontend unavailable): {e}"),
        }
    }
    // Interface (host-call) kernels — the §1a "around-compute" axis the harness was missing.
    v.push(hostcall_kernel());
    v.push(hostbuf_kernel());

    if from_wasm {
        for k in &mut v {
            match transpile_wat_to_ir(&k.wat32) {
                Ok((ir, entry)) => {
                    k.ir = ir;
                    k.entry = entry;
                    // A `HostCall` kernel's wasm imports a host function, so the transpiled entry takes
                    // the threaded capability handle as its leading param (the host-ABI convention). The
                    // stateless `bench_thunk` ignores the handle, so any value works — pass 0.
                    k.lead_args = if k.mode == Mode::HostCall {
                        vec![0]
                    } else {
                        Vec::new()
                    };
                }
                Err(e) => eprintln!(
                    "note: --from-wasm keeps `{}` hand-written (svm-wasm: {e})",
                    k.name
                ),
            }
        }
    }
    v
}

/// Transpile a WAT kernel through `svm-wasm` to SVM IR text + the `run` entry index. The IR is printed
/// and re-parsed by `measure`, so this also exercises the text round-trip.
fn transpile_wat_to_ir(wat32: &str) -> Result<(String, u32), String> {
    let wasm = wat::parse_str(wat32).map_err(|e| e.to_string())?;
    let t = svm_wasm::transpile(&wasm).map_err(|e| e.to_string())?;
    let entry = t
        .exports
        .iter()
        .find(|(n, _)| n == "run")
        .map(|(_, i)| *i)
        .ok_or_else(|| "transpiled module has no `run` export".to_string())?;
    Ok((svm_text::print_module(&t.module), entry))
}

/// Interface benchmark — **scalar host round-trip.** Each iteration makes one guest→host→guest
/// crossing: SVM `cap.call` (op 0) through the trampoline thunk vs a Wasmtime imported function
/// `host.op`, both `x -> x+1`. The subtraction isolates the per-call boundary cost. (Today SVM's
/// `cap.call` lowers to a *generic* indirect thunk that packs args into an array — the
/// devirtualize-to-direct-call optimization, D45, is deferred — so this is the honest baseline a
/// future inlining win will move.)
fn hostcall_kernel() -> Resolved {
    Resolved {
        name: "hostcall".into(),
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
  v10 = i32.const 0
  v11 = cap.call 0 0 (i64) -> (i64) v10(v9)
  v12 = i64.add v8 v11
  v13 = i64.const 1
  v14 = i64.add v9 v13
  br block1(v7, v12, v14)
block3(v15: i64):
  return v15
}
"
        .into(),
        entry: 0,
        lead_args: Vec::new(),
        wat32: r#"
(module
  (import "0" "0" (func $op (param i64) (result i64)))
  (func (export "run") (param $n i64) (result i64)
    (local $acc i64) (local $i i64)
    (block $done
      (loop $loop
        (br_if $done (i64.ge_s (local.get $i) (local.get $n)))
        (local.set $acc (i64.add (local.get $acc) (call $op (local.get $i))))
        (local.set $i (i64.add (local.get $i) (i64.const 1)))
        (br $loop)))
    (local.get $acc)))
"#
        .into(),
        wat64: None,
        mode: Mode::HostCall,
    }
}

/// Interface benchmark — **zero-copy borrow buffer (the strongest §1a claim).** Each iteration
/// hands the host a `(ptr, len)` buffer the host reads **in place** from the window (§7) and
/// sums — no marshalling, no copy-out. SVM `cap.call` (op 1) passes the window base to the thunk
/// directly; the Wasmtime import `host.sum` must fetch the exported `memory` and slice it. Both
/// are zero-copy in a *core* embedding (the larger §1a win is vs the component model's lift/lower,
/// not measured here), so this isolates the per-call buffer-access overhead. Buffer is 64 B of
/// zero-initialized window (the work is the read, not the value; the result is 0 on both).
fn hostbuf_kernel() -> Resolved {
    Resolved {
        name: "hostbuf".into(),
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
  v10 = i32.const 0
  v11 = i64.const 0
  v12 = i64.const 64
  v13 = cap.call 0 1 (i64, i64) -> (i64) v10(v11, v12)
  v14 = i64.add v8 v13
  v15 = i64.const 1
  v16 = i64.add v9 v15
  br block1(v7, v14, v16)
block3(v17: i64):
  return v17
}
"
        .into(),
        entry: 0,
        lead_args: Vec::new(),
        wat32: r#"
(module
  (import "0" "1" (func $sum (param i32 i32) (result i64)))
  (memory (export "memory") 1)
  (func (export "run") (param $n i64) (result i64)
    (local $acc i64) (local $i i64)
    (block $done
      (loop $loop
        (br_if $done (i64.ge_s (local.get $i) (local.get $n)))
        (local.set $acc (i64.add (local.get $acc) (call $sum (i32.const 0) (i32.const 64))))
        (local.set $i (i64.add (local.get $i) (i64.const 1)))
        (br $loop)))
    (local.get $acc)))
"#
        .into(),
        wat64: None,
        mode: Mode::HostCall,
    }
}

/// Compile `src` through the chibicc frontend and wrap its `run` as a kernel timed against
/// `wat32`. `run` is found by signature — the unique `(i64, i64) -> (i64)` function (`main`
/// returns i32, `_start` takes three i32s) — so this is robust against the frontend's function
/// ordering. `lead` is the args before `n`: `run` threads the data-stack pointer as v0, so it is
/// the initial SP (0 is safe here — the frame is tiny and self-contained). Returns `Err` (caller
/// skips the kernel) if the frontend can't be built/run.
fn c_kernel(
    name: &str,
    src: &str,
    lead: Vec<i64>,
    wat32: String,
    wat64: Option<String>,
) -> Result<Resolved, String> {
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
        wat64,
        mode: Mode::Compute,
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
    c_kernel("alu_c", SRC, vec![0], ALU.wat32.to_string(), None)
}

/// A **data-SP–relative** memory loop from C: an address-taken `volatile` stack array, so each
/// iteration stores/loads through `sp + (i & 255)*8` — and `sp` is an *unbounded* i64 block
/// param, so the JIT cannot prove the address in-window and masks every access. This is svm's
/// **weakest** kernel: unlike `memsum`/`scatter` (provably-small indices ⇒ mask pre-elided ⇒ svm
/// *beats* wasm64), here the mask can't be elided, so svm is slower than **both** wasm32 (~3.3×) and
/// wasm64 (~1.8×). Measured split: the mask is only ~1/3 of it (force-eliding drops it to ~2.2× wasm32
/// / ~1.2× wasm64); the rest is structural — the threaded-`sp` add + chibicc-generated IR + the
/// `volatile` memory-resident pattern, vs hand-written WAT over a pinned heap base. Closing the mask
/// third needs the verifier to prove the data-SP bounded (the §3d register-pinned-`sp` direction), not
/// 32-bit addressing (D50, rejected). Kept as a tracked metric so the mask path can't *regress
/// further*, with both a wasm32 and a (fair, 64-bit) wasm64 oracle.
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
    // wasm64 oracle (`(memory i64 1)`): the **fair 64-bit comparison** — like SVM, the address is a
    // 64-bit value, so Wasmtime emits an explicit bounds check per access (it can't lean on a 4 GiB
    // guard region the way wasm32 does). This is the apples-to-apples row for a 64-bit memory model.
    const WAT64: &str = r#"
(module
  (memory i64 1)
  (func (export "run") (param $n i64) (result i64)
    (local $acc i64) (local $i i64) (local $addr i64)
    (block $done
      (loop $loop
        (br_if $done (i64.ge_s (local.get $i) (local.get $n)))
        (local.set $addr
          (i64.mul (i64.and (local.get $i) (i64.const 255)) (i64.const 8)))
        (i64.store (local.get $addr) (local.get $i))
        (local.set $acc (i64.add (local.get $acc) (i64.load (local.get $addr))))
        (local.set $i (i64.add (local.get $i) (i64.const 1)))
        (br $loop)))
    (local.get $acc)))
"#;
    c_kernel(
        "locals_c",
        SRC,
        vec![0],
        WAT32.to_string(),
        Some(WAT64.to_string()),
    )
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
fn wasm_entry(engine: &Engine, wasm: &[u8]) -> (Store<HostState>, TypedFunc<i64, i64>) {
    let module = Module::new(engine, wasm).expect("wasmtime compile");
    let mut store = Store::new(engine, None);
    let instance = Instance::new(&mut store, &module, &[]).expect("instantiate");
    let run = instance
        .get_typed_func(&mut store, "run")
        .expect("entry `run`");
    (store, run)
}

/// Like [`wasm_entry`] but links the host imports the `HostCall` kernels need — the wasm
/// counterpart of [`bench_thunk`]: `host.op` (`x -> x+1`) and `host.sum` (sum a `(ptr, len)`
/// slice read in place from linear memory). The exported `Memory` is **cached in the store**
/// so `host.sum` skips the per-call export lookup — the perf-conscious wasm baseline, so the
/// comparison is like-for-like: both engines do the same zero-copy read, only the boundary
/// mechanism differs.
fn wasm_entry_host(engine: &Engine, wasm: &[u8]) -> (Store<HostState>, TypedFunc<i64, i64>) {
    let module = Module::new(engine, wasm).expect("wasmtime compile");
    let mut linker: Linker<HostState> = Linker::new(engine);
    // Imports use the svm-wasm host-ABI convention (module = capability type_id, name = op) so the
    // *same WAT* transpiles to `cap.call <type_id> <op>` under `--from-wasm`: "0"/"0" → op 0 (scalar
    // x+1), "0"/"1" → op 1 (sum a borrow buffer), matching `bench_thunk`'s op dispatch.
    linker
        .func_wrap("0", "0", |x: i64| -> i64 { x.wrapping_add(1) })
        .expect("define host op 0");
    linker
        .func_wrap(
            "0",
            "1",
            |caller: Caller<'_, HostState>, ptr: i32, len: i32| -> i64 {
                let mem = caller.data().expect("memory cached in store"); // Memory is Copy
                let data = mem.data(&caller);
                let (p, l) = (ptr as usize, len as usize);
                data[p..p + l].iter().map(|&b| b as i64).sum()
            },
        )
        .expect("define host op 1");
    let mut store = Store::new(engine, None);
    let instance = linker
        .instantiate(&mut store, &module)
        .expect("instantiate (host-call)");
    // Cache the exported memory (if any) so `host.sum` avoids a per-call export lookup.
    if let Some(mem) = instance.get_memory(&mut store, "memory") {
        *store.data_mut() = Some(mem);
    }
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

/// Per-iteration time (ns) of a compiled wasm entry, via subtraction over `[n_small, n_big]`
/// (steady-state compute for `Compute` kernels; per-host-call for `HostCall` kernels).
fn wasm_compute_ns(
    store: &mut Store<HostState>,
    run: &TypedFunc<i64, i64>,
    n_big: i64,
    n_small: i64,
) -> f64 {
    let big = per_call(100, || {
        black_box(run.call(&mut *store, n_big).unwrap());
    });
    let small = per_call(100, || {
        black_box(run.call(&mut *store, n_small).unwrap());
    });
    (big - small) * 1e9 / (n_big - n_small) as f64
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
    // `Compute` kernels time the inner loop; `HostCall` kernels make one host crossing per
    // iteration, so they use the no-cap vs cap-thunk SVM path, the import-linked wasm path, and
    // a smaller iteration span (a host call ≫ a compute op).
    let (n_big, n_small) = match k.mode {
        Mode::Compute => (N_BIG, N_SMALL),
        Mode::HostCall => (N_HOST_BIG, N_HOST_SMALL),
    };
    let svm = |n: i64| match k.mode {
        Mode::Compute => svm_call(&m, k.entry, &k.lead_args, n),
        Mode::HostCall => svm_call_host(&m, k.entry, &k.lead_args, n),
    };
    let inst = |wasm: &[u8]| match k.mode {
        Mode::Compute => wasm_entry(engine, wasm),
        Mode::HostCall => wasm_entry_host(engine, wasm),
    };

    // Cross-check every engine agrees before timing (never benchmark a miscompile).
    let ours = svm(n_small);
    {
        let (mut s32, run32) = inst(&wasm32);
        assert_eq!(
            ours,
            run32.call(&mut s32, n_small).unwrap(),
            "kernel `{}`: svm vs wasm32 disagree",
            k.name
        );
        if let Some(w) = &wasm64 {
            let (mut s64, run64) = inst(w);
            assert_eq!(
                ours,
                run64.call(&mut s64, n_small).unwrap(),
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
        // --- per-iteration time (subtraction isolates the loop body / the host call) ---
        let svm_big = per_call(25, || {
            black_box(svm(n_big));
        });
        let svm_small = per_call(25, || {
            black_box(svm(n_small));
        });
        raw.svm_ns = raw
            .svm_ns
            .min((svm_big - svm_small) * 1e9 / (n_big - n_small) as f64);

        let (mut s32, run32) = inst(&wasm32);
        raw.w32_ns = raw
            .w32_ns
            .min(wasm_compute_ns(&mut s32, &run32, n_big, n_small));

        if let Some(w) = &wasm64 {
            let (mut s64, run64) = inst(w);
            let v = wasm_compute_ns(&mut s64, &run64, n_big, n_small);
            raw.w64_ns = Some(raw.w64_ns.unwrap().min(v));
        }

        // --- cold start: source bytes → first result for a trivial (n=0) program (wasm32) ---
        let svm_cold = per_call(60, || {
            black_box(svm(0));
        }) * 1e3;
        raw.svm_cold = raw.svm_cold.min(svm_cold);
        let wmt_cold = per_call(60, || {
            let (mut s, f) = inst(&wasm32);
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
         Interface (host calls, §1a): `hostcall` (scalar cap.call vs a wasm import) is svm-\n\
         slower today — cap.call is a generic arg-packing thunk; devirtualization (D45) is\n\
         deferred. `hostbuf` (a zero-copy (ptr,len) borrow buffer the host reads in place)\n\
         is svm-faster even vs a cached-memory wasm import — the §7 win. Their `ns/it` and\n\
         `ratio` columns are per *host call* (N_big={N_HOST_BIG}), not per compute iteration.\n\
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
    // `--from-wasm`: get each compute kernel's SVM IR by transpiling its WAT (the same bytes Wasmtime
    // runs) instead of using the hand-written IR — the apples-to-apples comparison.
    let from_wasm = args.iter().any(|a| a == "--from-wasm");
    // `--fast-cap`: route HostCall kernels through the §9/D45 devirtualized fast path (vs the generic
    // thunk) so the two can be compared head-to-head.
    FAST_CAP.store(args.iter().any(|a| a == "--fast-cap"), Ordering::Relaxed);
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

    let results: Vec<(Resolved, Raw)> = resolve_kernels(from_wasm)
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
