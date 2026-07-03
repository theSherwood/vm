//! SVM **bytecode interpreter as a wasm guest** — the browser entry point (see `BROWSER.md`).
//!
//! Exports for a wasm host (browser / any runtime):
//!   * [`run_guest`] — a self-contained, no-import smoke probe (an embedded compute kernel), used by
//!     the wasm32 anchors in `run.mjs`.
//!   * [`svm_alloc`]/[`svm_dealloc`] — the host allocates a buffer in linear memory (no fixed cap),
//!     writes an **encoded SVM IR module** (the `svm-encode` binary form) into it, and frees it
//!     after the run.
//!   * [`svm_run`] — the production shape: `svm_run(ptr, len, arg)` decodes the module at
//!     `[ptr, len)`, runs function 0 on the **bytecode engine** with a **deny-all `Host`**
//!     (compute-only), and returns its first `i64` result. **Fail-closed:** a module the engine
//!     can't compile yields `STATUS_UNSUPPORTED` rather than any tree-walker fallback.
//!   * [`svm_run_pb`] — the **powerbox**: streams/clock/exit, I/O marshalled through allocations.
//!     `svm_run_live` (feature `live`) instead binds those to real host imports.
//!
//! Status of the last run is read separately via [`svm_status`] (a single `i64` return can't
//! disambiguate an error from a guest result of the same value).

// Every `#[no_mangle] extern "C"` export here is a wasm-host FFI boundary that, by construction,
// dereferences host-provided pointers (module bytes, the shared window, vCPU handles); each documents
// its host contract in a `SAFETY:` note. That is exactly the pattern `not_unsafe_ptr_arg_deref` warns
// about, so allow it crate-wide for these boundary functions.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::alloc::Layout;

use svm_interp::{bytecode, Host, StreamRole, Trap, Value};
#[cfg(feature = "live")]
use svm_interp::HostFn;

// ---- self-contained smoke probe (no host imports) --------------------------------------------

/// In-wasm roundtrip probe: parse → **encode** → **decode** → run, entirely inside the sandbox, so
/// the production `svm-encode` decode path (which `svm_run` relies on) is exercised on whatever
/// target this is built for — incl. wasm64 via `wasmtime --invoke run_roundtrip`. Returns the ALU
/// result for `arg = 1` (`1442695040888963407`), or `i64::MIN` on any failure.
#[no_mangle]
pub extern "C" fn run_roundtrip() -> i64 {
    let Ok(m) = svm_text::parse_module(ALU) else {
        return i64::MIN;
    };
    let bytes = svm_encode::encode_module(&m);
    let Ok(m2) = svm_encode::decode_module(&bytes) else {
        return i64::MIN;
    };
    let mut fuel = u64::MAX;
    match bytecode::compile_and_run(&m2, 0, &[Value::I64(1)], &mut fuel) {
        Some(Ok(vals)) => match vals.first() {
            Some(Value::I64(x)) => *x,
            _ => i64::MIN,
        },
        _ => i64::MIN,
    }
}

/// The §ROI-spike "alu" hash recurrence: loops `n` times mixing an LCG, returns the accumulator.
const ALU: &str = r#"
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
"#;

/// Parse the embedded guest, run it on the bytecode engine with arg `n`, return its i64 result.
/// `i64::MIN` is the in-band failure sentinel (parse/compile/trap).
#[no_mangle]
pub extern "C" fn run_guest(n: i64) -> i64 {
    let Ok(m) = svm_text::parse_module(ALU) else {
        return i64::MIN;
    };
    let mut fuel = u64::MAX;
    match bytecode::compile_and_run(&m, 0, &[Value::I64(n)], &mut fuel) {
        Some(Ok(vals)) => match vals.first() {
            Some(Value::I64(x)) => *x,
            _ => i64::MIN,
        },
        _ => i64::MIN,
    }
}

/// A self-contained **concurrency** smoke probe: 8 vCPUs each `atomic.rmw.add` a shared counter
/// 500× on the bytecode engine's cooperative `drive`, returning `4000` on every interleaving.
/// No host imports — usable via `wasmtime --invoke run_threads` to exercise the scheduler on wasm64.
const THREADS: &str = r#"
memory 16
func () -> (i64) {
block0():
  v0 = i64.const 0
  br block1(v0)
block1(v1: i64):
  v2 = i64.const 8
  v3 = i64.lt_u v1 v2
  br_if v3 block2(v1) block3()
block2(v4: i64):
  v5 = i64.const 500
  v6 = thread.spawn 1 v5 v5
  v7 = i64.const 4
  v8 = i64.mul v4 v7
  v9 = i64.const 16
  v10 = i64.add v9 v8
  i32.store v10 v6
  v11 = i64.const 1
  v12 = i64.add v4 v11
  br block1(v12)
block3():
  v13 = i64.const 0
  br block4(v13)
block4(v14: i64):
  v15 = i64.const 8
  v16 = i64.lt_u v14 v15
  br_if v16 block5(v14) block6()
block5(v17: i64):
  v18 = i64.const 4
  v19 = i64.mul v17 v18
  v20 = i64.const 16
  v21 = i64.add v20 v19
  v22 = i32.load v21
  v23 = thread.join v22
  v24 = i64.const 1
  v25 = i64.add v17 v24
  br block4(v25)
block6():
  v26 = i64.const 0
  v27 = i64.atomic.load v26
  return v27
}
func (i64, i64) -> (i64) {
block0(vsp: i64, v0: i64):
  br block1(v0)
block1(v1: i64):
  v2 = i64.const 0
  v3 = i64.eq v1 v2
  br_if v3 block2() block3(v1)
block3(v4: i64):
  v5 = i64.const 0
  v6 = i64.const 1
  v7 = i64.atomic.rmw.add v5 v6
  v8 = i64.const -1
  v9 = i64.add v4 v8
  br block1(v9)
block2():
  v10 = i64.const 0
  return v10
}
"#;

/// Run the embedded concurrency probe; returns `4000`, or `i64::MIN` on any failure.
#[no_mangle]
pub extern "C" fn run_threads() -> i64 {
    let Ok(m) = svm_text::parse_module(THREADS) else {
        return i64::MIN;
    };
    let mut fuel = u64::MAX;
    match bytecode::compile_and_run(&m, 0, &[], &mut fuel) {
        Some(Ok(vals)) => match vals.first() {
            Some(Value::I64(x)) => *x,
            _ => i64::MIN,
        },
        _ => i64::MIN,
    }
}

// ---- production entry: run an encoded guest module -------------------------------------------

/// `svm_run` completed and returned a guest `i64`.
pub const STATUS_OK: i32 = 0;
/// The bytes at the scratch buffer were not a well-formed encoded module.
pub const STATUS_DECODE_ERR: i32 = 1;
/// Fail-closed: the bytecode engine doesn't drive some op the module uses (no tree-walker fallback).
pub const STATUS_UNSUPPORTED: i32 = 2;
/// The guest trapped (masking/confinement violation, fuel exhaustion, explicit trap, …).
pub const STATUS_TRAP: i32 = 3;
/// The guest returned, but not a single `i64` (compute-only v1 only surfaces `i64`).
pub const STATUS_BAD_RESULT: i32 = 4;

/// Most recent status (a `STATUS_*` code), read via [`svm_status`] after any run entry.
static mut LAST_STATUS: i32 = STATUS_OK;

// ---- linear-memory allocator: the host manages I/O buffers of arbitrary size ------------------
//
// Replaces the old fixed scratch buffers. The host calls [`svm_alloc`] to reserve `len` bytes in
// *this module's* linear memory (the Rust allocator grows it as needed — no 1 MiB cap), writes the
// encoded module / stdin there, passes the `(ptr, len)` to a run entry, then [`svm_dealloc`]s it.
// Allocations are plain bytes (alignment 1), so `dealloc` only needs the same `len`.

/// Allocate `len` bytes (alignment 1) in linear memory; returns the pointer (null for `len == 0` or
/// on allocation failure). Pair every non-null result with a [`svm_dealloc`] of the same `len`.
#[no_mangle]
pub extern "C" fn svm_alloc(len: usize) -> *mut u8 {
    match Layout::from_size_align(len, 1) {
        Ok(layout) if len != 0 => unsafe { std::alloc::alloc(layout) },
        _ => core::ptr::null_mut(),
    }
}

/// Free a [`svm_alloc`]ation — `ptr`/`len` must match the original request. No-op for a null `ptr`
/// or `len == 0`. (Do **not** call this on the `svm_stdout_ptr`/`svm_stderr_ptr` buffers: those are
/// cdylib-managed, reclaimed on the next [`svm_run_pb`].)
#[no_mangle]
pub extern "C" fn svm_dealloc(ptr: *mut u8, len: usize) {
    if ptr.is_null() || len == 0 {
        return;
    }
    if let Ok(layout) = Layout::from_size_align(len, 1) {
        unsafe { std::alloc::dealloc(ptr, layout) };
    }
}

/// `1` on a 64-bit (`wasm64`/`memory64`) build, `0` on `wasm32` — so a host harness knows whether
/// the pointer/length ABI values are `i64` (BigInt) or `i32`.
#[no_mangle]
pub extern "C" fn svm_abi_is64() -> i32 {
    (core::mem::size_of::<usize>() == 8) as i32
}

/// Status of the most recent run entry (one of the `STATUS_*` codes).
#[no_mangle]
pub extern "C" fn svm_status() -> i32 {
    // SAFETY: single-threaded wasm; plain `i32` read.
    unsafe { LAST_STATUS }
}

/// Decode the `len` bytes at `ptr` as an SVM IR module, run function 0 on the bytecode engine with
/// `args` and a deny-all `Host`, and return its first `i64` result (`0` on any non-`OK` status —
/// read [`svm_status`] to disambiguate). Sets [`LAST_STATUS`]. Shared by [`svm_run`]/[`svm_run0`].
fn run_at(ptr: *const u8, len: usize, args: &[Value]) -> i64 {
    let set = |s: i32| unsafe { LAST_STATUS = s };
    // SAFETY: the host guarantees `[ptr, ptr+len)` is a live `svm_alloc`ation it just filled.
    let bytes = unsafe { core::slice::from_raw_parts(ptr, len) };
    let m = match svm_encode::decode_module(bytes) {
        Ok(m) => m,
        Err(_) => {
            set(STATUS_DECODE_ERR);
            return 0;
        }
    };
    let mut fuel = u64::MAX;
    let mut host = svm_interp::Host::new(); // deny-all powerbox (compute-only)
    match bytecode::compile_and_run_with_host(&m, 0, args, &mut fuel, &mut host) {
        None => {
            set(STATUS_UNSUPPORTED);
            0
        }
        Some(Err(_)) => {
            set(STATUS_TRAP);
            0
        }
        Some(Ok(vals)) => match vals.first() {
            Some(Value::I64(x)) => {
                set(STATUS_OK);
                *x
            }
            _ => {
                set(STATUS_BAD_RESULT);
                0
            }
        },
    }
}

/// Run the encoded module at `[ptr, ptr+len)` passing a single `i64` argument (the common shape).
#[no_mangle]
pub extern "C" fn svm_run(ptr: *const u8, len: usize, arg: i64) -> i64 {
    run_at(ptr, len, &[Value::I64(arg)])
}

/// Run the encoded module at `[ptr, ptr+len)` with **no** arguments — e.g. the `() -> (i64)` thread
/// kernels that spawn/join cooperatively on the engine's `drive`.
#[no_mangle]
pub extern "C" fn svm_run0(ptr: *const u8, len: usize) -> i64 {
    run_at(ptr, len, &[])
}

/// **Benchmark entry: run an arbitrary kernel function under the LLVM-frontend ABI.** Decode the
/// module at `[mod_ptr, mod_len)`, run function `func` on the bytecode engine with the frontend's
/// `(sp, n)` calling convention — `(sp, n)` for a ≥2-param entry, `(n)` for a 1-param one — under a
/// deny-all `Host`, and return its first result widened to `i64` (`0` on any non-`OK` status; read
/// [`svm_status`]). Each argument is coerced to its declared `ValType` so a 32-bit `n` param (the
/// `cross_engine` kernels) and a 64-bit one (the `embench` kernels, `long n`) both run correctly.
///
/// This is the seam the cross-engine benchmark uses to time the **bytecode engine running inside
/// wasm** (`crates/svm-llvm/examples/cross_engine.rs`'s `svm-bytecode-wasm` row, driven via
/// `browser/bench.mjs`) on the *same* LLVM-frontend IR the native `svm-bytecode` row runs — isolating
/// the cost of the wasm sandbox over the interpreter. `svm_run`/`svm_run0` only reach function 0 with
/// a fixed arity, so a dedicated entry is needed to drive a kernel exported at an arbitrary index.
#[no_mangle]
pub extern "C" fn svm_run_bench(
    mod_ptr: *const u8,
    mod_len: usize,
    func: u32,
    sp: i64,
    n: i64,
) -> i64 {
    let set = |s: i32| unsafe { LAST_STATUS = s };
    // SAFETY: the host guarantees `[mod_ptr, mod_len)` is a live `svm_alloc`ation it just filled.
    let bytes = unsafe { core::slice::from_raw_parts(mod_ptr, mod_len) };
    let m = match svm_encode::decode_module(bytes) {
        Ok(m) => m,
        Err(_) => {
            set(STATUS_DECODE_ERR);
            return 0;
        }
    };
    let Some(f) = m.funcs.get(func as usize) else {
        set(STATUS_UNSUPPORTED);
        return 0;
    };
    // Frontend ABI: the entry is `func(sp, n)`; a 1-param entry (e.g. a hand-written text kernel)
    // takes just `n`. Coerce each value to the declared param type (i32 vs i64 `n`); pad any extra
    // params with 0 of their type.
    let supplied: &[i64] = if f.params.len() >= 2 { &[sp, n] } else { &[n] };
    let args: Vec<Value> = f
        .params
        .iter()
        .enumerate()
        .map(|(i, ty)| {
            let raw = supplied.get(i).copied().unwrap_or(0);
            match ty {
                svm_ir::ValType::I32 => Value::I32(raw as i32),
                _ => Value::I64(raw),
            }
        })
        .collect();
    let mut fuel = u64::MAX;
    let mut host = Host::new(); // deny-all powerbox (compute-only)
    match bytecode::compile_and_run_with_host(&m, func, &args, &mut fuel, &mut host) {
        None => {
            set(STATUS_UNSUPPORTED);
            0
        }
        Some(Err(_)) => {
            set(STATUS_TRAP);
            0
        }
        Some(Ok(vals)) => match vals.first() {
            Some(Value::I64(x)) => {
                set(STATUS_OK);
                *x
            }
            Some(Value::I32(x)) => {
                set(STATUS_OK);
                *x as i64
            }
            _ => {
                set(STATUS_BAD_RESULT);
                0
            }
        },
    }
}

// ---- shared-memory window: run the engine over a caller-owned region of *this* linear memory ----
//
// THREADS.md step 4. `svm_run` runs over a window the engine backs internally; `svm_run_shared` runs
// over a window the **host** carves out of this module's linear memory (`[win_ptr, win_size)`, via
// `svm_alloc`). Built as a wasm threads module (shared memory + `+atomics`), that linear memory is
// the host's `SharedArrayBuffer`, so the window lives in shared memory — the substrate the parallel
// mode's per-vCPU Workers will all execute over. Today still cooperative (one thread); the only
// change from `svm_run` is *where the guest window lives*. Stateless (no `static mut`), so two
// Workers running it over **disjoint** windows don't race on engine ABI globals.

/// Decode the module at `[mod_ptr, mod_len)` and run function 0 over the guest window
/// `[win_ptr, win_ptr+win_size)` of this module's linear memory (a `Region::shared`; `win_size` must
/// cover the module's `memory` size). Returns the guest's `i64` result, or `i64::MIN` on
/// decode/unsupported/trap/non-`i64`. The host reads the guest's memory effects directly from the
/// window region afterward.
#[no_mangle]
pub extern "C" fn svm_run_shared(
    mod_ptr: *const u8,
    mod_len: usize,
    win_ptr: *mut u8,
    win_size: usize,
    arg: i64,
) -> i64 {
    // SAFETY: the host guarantees `[mod_ptr, mod_len)` is a live `svm_alloc`ation it just filled.
    let bytes = unsafe { core::slice::from_raw_parts(mod_ptr, mod_len) };
    let Ok(m) = svm_encode::decode_module(bytes) else {
        return i64::MIN;
    };
    // SAFETY: the host guarantees `[win_ptr, win_size)` is a live `svm_alloc`ed region of this linear
    // memory used solely as this guest window for the call. The `unsafe` borrow lives here in the
    // embedder; the engine stays `#![forbid(unsafe_code)]` and just takes the `Arc<Region>`.
    let back = std::sync::Arc::new(unsafe { svm_interp::Region::shared(win_ptr, win_size as u64) });
    let arity = m.funcs.first().map_or(0, |f| f.params.len());
    let args: &[Value] = if arity >= 1 { &[Value::I64(arg)] } else { &[] };
    let mut fuel = u64::MAX;
    match bytecode::compile_and_run_capture_over(&m, 0, args, &mut fuel, &[], back) {
        Some((Ok(vals), _)) => match vals.first() {
            Some(Value::I64(x)) => *x,
            Some(Value::I32(x)) => *x as i64,
            _ => i64::MIN,
        },
        _ => i64::MIN,
    }
}

// ==== THREADS.md step 4c-wasm — the host-orchestrated parallel driver =============================
//
// wasm32 has no `thread::spawn`, so one guest's `thread.spawn`ed vCPUs are distributed across **Web
// Workers** by the JS host: each Worker runs **one** vCPU via the engine's resumable `Vcpu` API
// (`svm_par_run` → an event the host services → deliver the result → run again) over the **one** shared
// linear-memory window. The host services the events with real cross-Worker primitives: `thread.spawn`
// → start a Worker, `thread.join` → `Atomics.wait` on the child's completion slot, `memory.wait`/
// `notify` → `Atomics.wait`/`notify` on the futex word — so this is genuinely parallel, the native
// `bytecode_vcpu_orchestration.rs` test being its differential oracle.
//
// `VcpuProgram` is compiled once and shared **read-only** across Workers by pointer (it is `Sync`, and
// under `--shared-memory` every Worker's instance sees the same linear memory, so a `Box::leak`ed
// program built by one Worker is valid in all). Each `Vcpu` is `'static` here: the program outlives the
// run (never freed), so the borrow is sound — the `unsafe` of asserting that lives in this embedder.

/// Allocate `len` bytes **16-aligned** (so windows / futex words / completion slots are naturally
/// aligned for `Atomics` / the engine's hardware atomics, which `svm_alloc`'s align-1 does not
/// guarantee). Leaked for the run (the parallel demo never frees; the process exits). Null on `len==0`.
#[no_mangle]
pub extern "C" fn svm_par_alloc(len: usize) -> *mut u8 {
    match Layout::from_size_align(len, 16) {
        Ok(layout) if len != 0 => unsafe { std::alloc::alloc_zeroed(layout) },
        _ => core::ptr::null_mut(),
    }
}

/// Event codes returned by [`svm_par_run`] — the host switches on these (operands via `svm_par_ev_*`).
pub const PAR_DONE: i32 = 0;
pub const PAR_TRAP: i32 = 1;
pub const PAR_SPAWN: i32 = 2;
pub const PAR_JOIN: i32 = 3;
pub const PAR_WAIT: i32 = 4;
pub const PAR_NOTIFY: i32 = 5;
pub const PAR_INSTANTIATE: i32 = 6;

/// A boxed resumable vCPU plus the operands of its last [`svm_par_run`] event (flattened to four
/// `i64`s the host reads via [`svm_par_ev_a`]–[`svm_par_ev_d`]).
pub struct ParVcpu {
    inner: bytecode::Vcpu<'static>,
    a: i64,
    b: i64,
    c: i64,
    d: i64,
}

fn first_i64(vals: &[Value]) -> i64 {
    match vals.first() {
        Some(Value::I64(x)) => *x,
        Some(Value::I32(x)) => *x as i64,
        _ => 0,
    }
}

/// Compile the module at `[mod_ptr, mod_len)` into a shareable [`bytecode::VcpuProgram`], returned as a
/// leaked pointer (lives for the run; shared read-only across Workers). Null on decode/unsupported.
#[no_mangle]
pub extern "C" fn svm_par_compile(mod_ptr: *const u8, mod_len: usize) -> *mut bytecode::VcpuProgram {
    // SAFETY: the host guarantees `[mod_ptr, mod_len)` is a live `svm_alloc`ation it just filled.
    let bytes = unsafe { core::slice::from_raw_parts(mod_ptr, mod_len) };
    let Ok(m) = svm_encode::decode_module(bytes) else {
        return core::ptr::null_mut();
    };
    match bytecode::VcpuProgram::compile(&m) {
        Some(p) => Box::into_raw(Box::new(p)),
        None => core::ptr::null_mut(),
    }
}

/// Borrow a `*mut VcpuProgram` as `&'static` (the program outlives the run). SAFETY: the host keeps it
/// alive for the whole run and never frees it before the last `Vcpu` over it.
unsafe fn prog_ref(prog: *mut bytecode::VcpuProgram) -> &'static bytecode::VcpuProgram {
    &*prog
}

// ---- §22 guest-JIT across Workers: a Rust-side shared powerbox (THREADS.md 4c-domain C2) ---------
// The powerbox (a `Host` with the `Jit` cap + the host-compiled unit) is built once and **leaked** into
// the shared linear memory; its pointer is published in a process-wide `static` which — under
// `--shared-memory` — lives in that shared memory, so every Worker's instance reads the same value
// (the same mechanism the `Box::leak`ed `VcpuProgram` uses, but a `static` instead of a JS-threaded
// pointer). A worker vCPU's `Jit.install`/`uninstall`/`invoke` is then serviced **inside**
// [`svm_par_run`] against this powerbox + the shared `Domain` — so the JS host services no new events
// (it never sees a JIT op, needs no new glue). During the run the powerbox is read-only (the unit is
// compiled at setup, before any spawn), so the concurrent `&Host` reads need no lock; the install/
// dispatch mutation lives in the `Domain`, which is already interior-mutable + thread-safe.

/// The shared §22 powerbox: a `Host` with the `Jit` cap granted + [`JIT_SERVICE`] host-compiled, plus
/// the handles the root guest receives as `(jit, code)`.
struct ParPowerbox {
    host: Host,
    jit: i32,
    code: i32,
}

/// The leaked [`ParPowerbox`] pointer (or `0`), shared across Workers via shared linear memory.
static PAR_PB: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// `2^4 = 16` dispatch-table slots — the `Jit` table reservation matched by [`svm_par_compile_jit`]
/// and the powerbox grant so guest `install` lands in range (mirrors [`jit_exec`]).
const PAR_JIT_TABLE_LOG2: u8 = 4;

/// Build the **shared powerbox** for a §22-JIT run: grant the `Jit` cap (16-slot table) on a fresh
/// `Host`, host-compile [`JIT_SERVICE`] into it, then leak it and publish the pointer for every Worker.
/// `guest`'s declared memory sizes the domain (the validator's memory-match precondition). Returns `1`
/// on success, `0` on decode / parse / compile failure. Call **once** (on the main thread) before the
/// run; the published pointer outlives it.
#[no_mangle]
pub extern "C" fn svm_par_powerbox(guest_ptr: *const u8, guest_len: usize) -> i32 {
    // SAFETY: the host guarantees `[guest_ptr, guest_len)` is a live allocation it just filled.
    let bytes = unsafe { core::slice::from_raw_parts(guest_ptr, guest_len) };
    let Ok(m) = svm_encode::decode_module(bytes) else {
        return 0;
    };
    let service = match svm_text::parse_module(JIT_SERVICE) {
        Ok(s) => svm_encode::encode_module(&s),
        Err(_) => return 0,
    };
    let mut host = Host::new();
    let jit = host.grant_jit_with_table(m.memory.map(|mc| mc.size_log2), PAR_JIT_TABLE_LOG2);
    host.set_jit_validator(browser_jit_validator);
    let code = match host.jit_compile(jit, &service) {
        Ok(Ok(c)) => c.handle,
        _ => return 0,
    };
    let pb = Box::into_raw(Box::new(ParPowerbox { host, jit, code }));
    PAR_PB.store(pb as usize, std::sync::atomic::Ordering::Release);
    // Last-published run recipe wins (a page runs several kinds back to back).
    PAR_INST.store(0, std::sync::atomic::Ordering::Release);
    PAR_IO.store(0, std::sync::atomic::Ordering::Release);
    1
}

/// Borrow the published powerbox (`None` until [`svm_par_powerbox`] ran). The pointer is published with
/// `Release`; this `Acquire` load pairs with it so the `Host` it built is visible to this Worker.
fn par_pb() -> Option<&'static ParPowerbox> {
    let p = PAR_PB.load(std::sync::atomic::Ordering::Acquire) as *const ParPowerbox;
    // SAFETY: once published the powerbox is leaked (never freed) and read-only for the run, so the
    // shared `&'static` is sound (concurrent `&self` reads only).
    unsafe { p.as_ref() }
}

/// Resolve a code-handle's unit funcs under authority `handle` against the powerbox (the `install` /
/// `invoke` service): a forged / cross-domain / wrong-type handle is an inert `CapFault` → trap.
fn par_resolve_unit(
    pb: &ParPowerbox,
    handle: i32,
    code: i32,
) -> Result<std::sync::Arc<[svm_ir::Func]>, Trap> {
    let domain = pb.host.resolve_jit_domain(handle)?;
    let (cd, cu) = pb.host.resolve_jit_code(code)?;
    if cd != domain {
        return Err(Trap::CapFault);
    }
    pb.host.jit_unit_funcs(cd, cu).ok_or(Trap::CapFault)
}

// ---- §14 instantiate across Workers (THREADS.md 4c-domain §14-D2) -------------------------------
// The §14 root powerbox lives **in the root vCPU** (unlike the §22 JIT powerbox, which the vCPU asks
// the host to resolve against): §14 resolves its `Instantiator` authority in-Vm during `resume`, so
// the grant must be in the vCPU's own `Host`. This static only carries the *recipe* — the authority
// range and the optional granted module — published once by the main thread so the root Worker can
// build its powerbox deterministically. Confined children never touch it: their attenuated powerbox
// is built inside `Vcpu::new_confined_child`, so no authority ever crosses JS (the `PAR_INSTANTIATE`
// event operands are inert integers).

/// The §14 run recipe: `Instantiator` authority over `[0, win_size)` + an optional `Module` grant.
struct ParInstCfg {
    win_size: u64,
    module: Option<svm_ir::Module>,
}

/// The leaked [`ParInstCfg`] pointer (or `0`), shared across Workers via shared linear memory.
static PAR_INST: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Publish the §14 run recipe: the root's `Instantiator` will span `[0, win_size)`; a non-empty
/// `[mod_ptr, mod_len)` is decoded as the **granted module** for `instantiate_module` (`0` len ⇒ no
/// grant). Returns `1`, or `0` on a bad module. Call once (on the main thread) before the run.
#[no_mangle]
pub extern "C" fn svm_par_powerbox_inst(win_size: u64, mod_ptr: *const u8, mod_len: usize) -> i32 {
    let module = if mod_len == 0 {
        None
    } else {
        // SAFETY: the host guarantees `[mod_ptr, mod_len)` is a live allocation it just filled.
        let bytes = unsafe { core::slice::from_raw_parts(mod_ptr, mod_len) };
        match svm_encode::decode_module(bytes) {
            Ok(m) => Some(m),
            Err(_) => return 0,
        }
    };
    let cfg = Box::into_raw(Box::new(ParInstCfg { win_size, module }));
    PAR_INST.store(cfg as usize, std::sync::atomic::Ordering::Release);
    // Last-published run recipe wins (a page runs several kinds back to back).
    PAR_PB.store(0, std::sync::atomic::Ordering::Release);
    PAR_IO.store(0, std::sync::atomic::Ordering::Release);
    1
}

/// Borrow the published §14 recipe (`None` until [`svm_par_powerbox_inst`] ran). Leaked + read-only,
/// as [`par_pb`].
fn par_inst() -> Option<&'static ParInstCfg> {
    let p = PAR_INST.load(std::sync::atomic::Ordering::Acquire) as *const ParInstCfg;
    // SAFETY: once published the recipe is leaked (never freed) and read-only for the run.
    unsafe { p.as_ref() }
}

// ---- 4d: host I/O across Workers — the run's shared powerbox ------------------------------------
// THREADS.md 4d: one `Mutex<Host>`, leaked into the shared linear memory (the same cross-Worker
// sharing as `PAR_PB`/`PAR_INST`), attached to **every** vCPU of the run
// ([`bytecode::Vcpu::with_shared_host`]) — so a worker vCPU's `cap.call` (host I/O) dispatches
// in-engine under the lock, `drive_parallel`'s 4c-host model, with no JS in the loop at all: the
// `Host` is fully virtual (stdout is an in-memory buffer the page reads back after the run).

/// The shared I/O powerbox: the `Mutex<Host>` every vCPU dispatches through, plus the handles the
/// root guest receives as its args.
struct ParIoCfg {
    host: std::sync::Mutex<Host>,
    /// The `Stream(Out)` handle (the root's single entry arg).
    out: i32,
}

/// The leaked [`ParIoCfg`] pointer (or `0`), shared across Workers via shared linear memory.
static PAR_IO: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Publish the run's **shared I/O powerbox**: a fresh `Host` granted a `Stream(Out)`, wrapped in the
/// `Mutex` every vCPU will dispatch `cap.call` through. The root is seeded with `[out_handle]`
/// (`svm_par_root`); read the accumulated stdout back after the run via [`svm_par_stdout_len`] +
/// [`svm_par_stdout_ptr`]. Call once (on the main thread) before the run; last-published run recipe
/// wins (the §22/§14 recipes are cleared, and vice versa).
#[no_mangle]
pub extern "C" fn svm_par_powerbox_io() -> i32 {
    let mut host = Host::new();
    let out = host.grant_stream(StreamRole::Out);
    let cfg = Box::into_raw(Box::new(ParIoCfg {
        host: std::sync::Mutex::new(host),
        out,
    }));
    PAR_IO.store(cfg as usize, std::sync::atomic::Ordering::Release);
    PAR_INST.store(0, std::sync::atomic::Ordering::Release);
    PAR_PB.store(0, std::sync::atomic::Ordering::Release);
    1
}

/// Clear every published run recipe — the next run is **plain** (compute-only, no powerbox). The
/// recipes are last-published-wins for back-to-back runs of *different* kinds; a plain run after a
/// powerbox run (the playground can run modes in any order) needs this explicit "none" publish, or
/// the stale recipe would seed the new root with args its entry doesn't take.
#[no_mangle]
pub extern "C" fn svm_par_powerbox_none() {
    PAR_PB.store(0, std::sync::atomic::Ordering::Release);
    PAR_INST.store(0, std::sync::atomic::Ordering::Release);
    PAR_IO.store(0, std::sync::atomic::Ordering::Release);
}

/// Borrow the published I/O powerbox (`None` until [`svm_par_powerbox_io`] ran). Leaked; interior
/// mutability is the `Mutex` (cross-Worker-safe on wasm atomics, like the `Domain`'s `ModuleSource`).
fn par_io() -> Option<&'static ParIoCfg> {
    let p = PAR_IO.load(std::sync::atomic::Ordering::Acquire) as *const ParIoCfg;
    // SAFETY: once published the powerbox is leaked (never freed); all access is via the `Mutex`.
    unsafe { p.as_ref() }
}

/// Live-vCPU counter across Workers — the browser path's anti-bomb **backstop** (the native drivers
/// give the spawner a clean `ThreadFault`; here a construction past the cap returns null and the JS
/// host fails the run — cruder, but it bounds Worker creation). Incremented by the `svm_par_*` vCPU
/// constructors, decremented by [`svm_par_free`].
static PAR_LIVE: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
/// Far above any legitimate fan-out (a tab with 256 live Workers is already pathological), far below
/// a Worker bomb's ambition.
const PAR_MAX_VCPUS: u32 = 256;

/// Admit one vCPU under the live cap (decrementing back out on refusal).
fn par_vcpu_admit() -> bool {
    use std::sync::atomic::Ordering;
    if PAR_LIVE.fetch_add(1, Ordering::AcqRel) >= PAR_MAX_VCPUS {
        PAR_LIVE.fetch_sub(1, Ordering::AcqRel);
        return false;
    }
    true
}

/// Un-admit a vCPU that failed to construct (the success path decrements via [`svm_par_free`]).
fn par_vcpu_retire() {
    PAR_LIVE.fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
}

/// Like [`svm_par_compile`], but reserve the `Jit` dispatch table (matching the powerbox grant) so a
/// guest `install` lands in range. Use this (not [`svm_par_compile`]) for a §22-JIT run.
#[no_mangle]
pub extern "C" fn svm_par_compile_jit(
    mod_ptr: *const u8,
    mod_len: usize,
) -> *mut bytecode::VcpuProgram {
    // SAFETY: the host guarantees `[mod_ptr, mod_len)` is a live `svm_alloc`ation it just filled.
    let bytes = unsafe { core::slice::from_raw_parts(mod_ptr, mod_len) };
    let Ok(m) = svm_encode::decode_module(bytes) else {
        return core::ptr::null_mut();
    };
    match bytecode::VcpuProgram::compile_with_jit_table(&m, PAR_JIT_TABLE_LOG2) {
        Some(p) => Box::into_raw(Box::new(p)),
        None => core::ptr::null_mut(),
    }
}

/// Build the **root** vCPU (function `func`) over the shared window `[win_ptr, win_size)`; it seeds +
/// data-initialises the window (the once). Returns a boxed [`ParVcpu`] pointer, null on a bad func.
#[no_mangle]
pub extern "C" fn svm_par_root(
    prog: *mut bytecode::VcpuProgram,
    win_ptr: *mut u8,
    win_size: usize,
    func: u32,
) -> *mut ParVcpu {
    if !par_vcpu_admit() {
        return core::ptr::null_mut();
    }
    // SAFETY: the host guarantees `[win_ptr, win_size)` is a live shared window for the run.
    let back = std::sync::Arc::new(unsafe { svm_interp::Region::shared(win_ptr, win_size as u64) });
    // A §14 run builds the root's **own** powerbox from the published recipe (`Instantiator` +
    // optional `Module` grant; §14 resolves authority in-Vm, so the grants must live in the vCPU's
    // host) and seeds the root with the handles. A §22-JIT run seeds `(jit, code)` from the shared
    // powerbox; a 4d I/O run attaches the shared `Mutex<Host>` and seeds `[out]`; a plain run gets
    // no args. Signatures unchanged either way — the JS host just calls the matching
    // `svm_par_powerbox*` first.
    if let Some(cfg) = par_inst() {
        let mut host = Host::new();
        let inst = host.grant_instantiator(0, cfg.win_size);
        let mut args = vec![Value::I32(inst)];
        if let Some(m) = &cfg.module {
            args.push(Value::I32(host.grant_module(m)));
        }
        // SAFETY: `prog` is a live program pointer the host keeps alive for the run.
        return match bytecode::Vcpu::new_root_with_powerbox(
            unsafe { prog_ref(prog) },
            func,
            &args,
            back,
            &[],
            host,
        ) {
            Ok(inner) => Box::into_raw(Box::new(ParVcpu {
                inner,
                a: 0,
                b: 0,
                c: 0,
                d: 0,
            })),
            Err(_) => {
                par_vcpu_retire();
                core::ptr::null_mut()
            }
        };
    }
    let (args, io): (Vec<Value>, Option<&'static ParIoCfg>) = match (par_io(), par_pb()) {
        (Some(io), _) => (vec![Value::I32(io.out)], Some(io)),
        (None, Some(pb)) => (vec![Value::I32(pb.jit), Value::I32(pb.code)], None),
        (None, None) => (Vec::new(), None),
    };
    // SAFETY: `prog` is a live program pointer the host keeps alive for the run.
    match bytecode::Vcpu::new_root(unsafe { prog_ref(prog) }, func, &args, back, &[]) {
        Ok(inner) => {
            let inner = match io {
                Some(io) => inner.with_shared_host(&io.host),
                None => inner,
            };
            Box::into_raw(Box::new(ParVcpu {
                inner,
                a: 0,
                b: 0,
                c: 0,
                d: 0,
            }))
        }
        Err(_) => {
            par_vcpu_retire();
            core::ptr::null_mut()
        }
    }
}

/// Build a `thread.spawn`ed **child** vCPU (`func(sp, arg)`) over the **same** shared window — it does
/// not re-seed (the window is already live). Called on the child's Worker. Null on a bad func.
#[no_mangle]
pub extern "C" fn svm_par_child(
    prog: *mut bytecode::VcpuProgram,
    win_ptr: *mut u8,
    win_size: usize,
    func: u32,
    sp: i64,
    arg: i64,
) -> *mut ParVcpu {
    if !par_vcpu_admit() {
        return core::ptr::null_mut();
    }
    // SAFETY: the host guarantees `[win_ptr, win_size)` is the same live shared window.
    let back = std::sync::Arc::new(unsafe { svm_interp::Region::shared(win_ptr, win_size as u64) });
    let args = [Value::I64(sp), Value::I64(arg)];
    // SAFETY: `prog` is a live program pointer the host keeps alive for the run.
    match bytecode::Vcpu::new_child(unsafe { prog_ref(prog) }, func, &args, back) {
        Ok(inner) => {
            // A 4d I/O run shares one powerbox across every vCPU (worker `cap.call` = host I/O).
            let inner = match par_io() {
                Some(io) => inner.with_shared_host(&io.host),
                None => inner,
            };
            Box::into_raw(Box::new(ParVcpu {
                inner,
                a: 0,
                b: 0,
                c: 0,
                d: 0,
            }))
        }
        Err(_) => {
            par_vcpu_retire();
            core::ptr::null_mut()
        }
    }
}

/// Build a §14 **confined executor child** vCPU (THREADS.md 4c-domain §14-D2) over the parent's carve
/// `[carve_ptr, carve_ptr + 2^size_log2)` — the operands of a [`PAR_INSTANTIATE`] event, shuttled
/// verbatim by the JS host (`carve_ptr` = the parent Worker's window pointer + the event's `carve`).
/// Per DESIGN.md §14 a sub-window is indistinguishable from a top-level window, so the carve region
/// simply *is* the child's window; the attenuated powerbox and the child's own dispatch table are
/// built in-engine ([`bytecode::Vcpu::new_confined_child`]) — no authority crosses JS. Called on the
/// child's Worker. Null on a bad module/entry.
#[no_mangle]
pub extern "C" fn svm_par_child_confined(
    prog: *mut bytecode::VcpuProgram,
    carve_ptr: *mut u8,
    size_log2: u32,
    module: u32,
    entry: u32,
    fuel: i64,
) -> *mut ParVcpu {
    if size_log2 >= 64 || !par_vcpu_admit() {
        return core::ptr::null_mut();
    }
    // SAFETY: the host guarantees the carve is inside the parent's live window (the engine validated
    // it before surfacing the event); aliasing views of the shared memory are the §13 data plane.
    let back = std::sync::Arc::new(unsafe {
        svm_interp::Region::shared(carve_ptr, 1u64 << size_log2)
    });
    // SAFETY: `prog` is a live program pointer the host keeps alive for the run.
    // (No shared-host attach: a §14 confined child's powerbox is its own attenuated one, built
    // in-engine — its capability set never includes the run's I/O grants.)
    match bytecode::Vcpu::new_confined_child(
        unsafe { prog_ref(prog) },
        module,
        entry,
        back,
        size_log2 as u8,
        fuel as u64,
    ) {
        Ok(inner) => Box::into_raw(Box::new(ParVcpu {
            inner,
            a: 0,
            b: 0,
            c: 0,
            d: 0,
        })),
        Err(_) => {
            par_vcpu_retire();
            core::ptr::null_mut()
        }
    }
}

/// Pointer / length of the accumulated stdout in the run's shared I/O powerbox (4d). Call `len`
/// **first** — it snapshots the buffer under the powerbox lock into a stable stash `ptr` then reads —
/// after the run completes (the root's `done`; a mid-run call sees a prefix). `0` when no
/// [`svm_par_powerbox_io`] was published.
#[no_mangle]
pub extern "C" fn svm_par_stdout_len() -> usize {
    let Some(io) = par_io() else { return 0 };
    let bytes = {
        let g = io.host.lock().unwrap_or_else(|e| e.into_inner());
        g.stdout.clone()
    };
    // SAFETY: the stash slot is only touched from the main thread (the JS host reads results after
    // the run), matching the `svm_run_pb` accessors' single-reader contract.
    unsafe { stash(&mut *core::ptr::addr_of_mut!(PAR_OUT), bytes) };
    unsafe { (*core::ptr::addr_of!(PAR_OUT)).1 }
}
#[no_mangle]
pub extern "C" fn svm_par_stdout_ptr() -> *const u8 {
    // SAFETY: as above — main-thread single-reader stash.
    unsafe { (*core::ptr::addr_of!(PAR_OUT)).0 }
}
/// The stashed 4d stdout snapshot (`svm_par_stdout_len` fills it; `_ptr` reads it).
static mut PAR_OUT: (*mut u8, usize) = (core::ptr::null_mut(), 0);

/// Advance the vCPU until it finishes, traps, or hits a host-serviced event; returns a `PAR_*` code.
/// The host reads operands via `svm_par_ev_a`–`d`, services the event, calls the matching `deliver`,
/// then calls `svm_par_run` again.
#[no_mangle]
pub extern "C" fn svm_par_run(v: *mut ParVcpu) -> i32 {
    // SAFETY: `v` is a live `ParVcpu` from `svm_par_root`/`svm_par_child`, owned by this Worker.
    let v = unsafe { &mut *v };
    // Loop so §22 JIT events (serviced in-Rust against the shared powerbox) never surface to the JS
    // host — it only ever sees the multi-vCPU events `spawn`/`join`/`wait`/`notify` (+ `done`/`trap`).
    loop {
        match v.inner.run() {
            bytecode::VcpuEvent::Done(vals) => {
                v.a = first_i64(&vals);
                return PAR_DONE;
            }
            bytecode::VcpuEvent::Trapped(_) => return PAR_TRAP,
            bytecode::VcpuEvent::Spawn { func, sp, arg } => {
                v.a = func as i64;
                v.b = sp;
                v.c = arg;
                return PAR_SPAWN;
            }
            bytecode::VcpuEvent::Join { handle } => {
                v.a = handle as i64;
                return PAR_JOIN;
            }
            bytecode::VcpuEvent::Wait {
                addr,
                expected,
                width,
                timeout,
            } => {
                v.a = addr as i64;
                v.b = expected as i64;
                v.c = width as i64;
                v.d = timeout as i64;
                return PAR_WAIT;
            }
            bytecode::VcpuEvent::Notify { addr, count } => {
                v.a = addr as i64;
                v.b = count as i64;
                return PAR_NOTIFY;
            }
            // §22 guest-JIT serviced in-Rust against the shared powerbox + `Domain` (THREADS.md
            // 4c-domain C2): resolve the unit (the powerbox holds authority) and deliver it; the vCPU
            // installs / invokes against the shared `Domain`, then we loop. Without a powerbox (a
            // non-JIT run) a JIT op is fail-closed, exactly as before this seam existed.
            bytecode::VcpuEvent::JitInstall { handle, code } => match par_pb() {
                Some(pb) => v.inner.deliver_jit_install(par_resolve_unit(pb, handle, code)),
                None => return PAR_TRAP,
            },
            bytecode::VcpuEvent::JitUninstall { handle, .. } => match par_pb() {
                Some(pb) => v
                    .inner
                    .deliver_jit_uninstall(pb.host.resolve_jit_domain(handle).map(|_| ())),
                None => return PAR_TRAP,
            },
            bytecode::VcpuEvent::JitInvoke {
                handle, code, ..
            } => match par_pb() {
                Some(pb) => v.inner.deliver_jit_invoke(par_resolve_unit(pb, handle, code)),
                None => return PAR_TRAP,
            },
            // §14 confined executor child (THREADS.md 4c-domain §14-D2): all authority-bearing work
            // already happened in-Vm — the operands are inert integers the JS host shuttles into a
            // new Worker running `svm_par_child_confined` over `[win + carve, +2^size_log2)`, joined
            // through the same completion-slot protocol as `PAR_SPAWN`.
            bytecode::VcpuEvent::Instantiate {
                module,
                entry,
                carve,
                size_log2,
                fuel,
            } => {
                v.a = ((module as i64) << 32) | entry as i64;
                v.b = carve as i64;
                v.c = size_log2 as i64;
                v.d = fuel as i64;
                return PAR_INSTANTIATE;
            }
        }
    }
}

macro_rules! par_ev_getter {
    ($name:ident, $field:ident) => {
        /// Read an operand of the last [`svm_par_run`] event.
        #[no_mangle]
        pub extern "C" fn $name(v: *mut ParVcpu) -> i64 {
            // SAFETY: `v` is a live `ParVcpu` owned by this Worker.
            unsafe { (*v).$field }
        }
    };
}
par_ev_getter!(svm_par_ev_a, a);
par_ev_getter!(svm_par_ev_b, b);
par_ev_getter!(svm_par_ev_c, c);
par_ev_getter!(svm_par_ev_d, d);

/// Deliver a `thread.spawn` handle (after `PAR_SPAWN`).
#[no_mangle]
pub extern "C" fn svm_par_deliver_handle(v: *mut ParVcpu, handle: i32) {
    // SAFETY: `v` is a live `ParVcpu` awaiting a delivery.
    unsafe { (*v).inner.deliver_handle(handle) };
}

/// Deliver a `memory.wait` code / `memory.notify` count (after `PAR_WAIT` / `PAR_NOTIFY`).
#[no_mangle]
pub extern "C" fn svm_par_deliver_code(v: *mut ParVcpu, code: i32) {
    // SAFETY: `v` is a live `ParVcpu` awaiting a delivery.
    unsafe { (*v).inner.deliver_code(code) };
}

/// Deliver a joined child's result (after `PAR_JOIN`): `val` is its first return value, or — if
/// `is_trap != 0` — the child trapped and the joiner traps on its next `svm_par_run`.
#[no_mangle]
pub extern "C" fn svm_par_deliver_join(v: *mut ParVcpu, val: i64, is_trap: i32) {
    // SAFETY: `v` is a live `ParVcpu` awaiting a delivery.
    let v = unsafe { &mut *v };
    if is_trap != 0 {
        v.inner.deliver_join(Err(Trap::ThreadFault));
    } else {
        v.inner.deliver_join(Ok(vec![Value::I64(val)]));
    }
}

/// Free a finished vCPU.
#[no_mangle]
pub extern "C" fn svm_par_free(v: *mut ParVcpu) {
    if !v.is_null() {
        // SAFETY: `v` came from `Box::into_raw` in `svm_par_root`/`svm_par_child` and is freed once.
        drop(unsafe { Box::from_raw(v) });
        par_vcpu_retire(); // the live-cap admit from this vCPU's constructor
    }
}

// ---- host powerbox: console + clock, marshalled through host-allocated memory ----------------
//
// Beyond compute-only: grant the guest a real capability set (stdin/stdout/stderr streams, a
// monotonic clock, and exit). The `Host` powerbox is already self-contained and **deterministic** —
// stream writes accumulate in `Host::stdout`/`stderr`, `read` draws from `Host::stdin`, and
// `Clock.now` is a strictly-increasing counter — so no wasm host *imports* are needed: I/O crosses
// the boundary the same way the module does, through `svm_alloc`ed memory. The host writes stdin to
// an allocation it passes in; the captured streams come back as cdylib-managed allocations the host
// reads (via the `*_ptr`/`*_len` exports) before the next call. The cdylib stays import-free.

/// The guest called `Exit.exit(code)` (a non-error trap); read the code via [`svm_exit_code`].
pub const STATUS_EXIT: i32 = 5;

/// Outcome of a [`powerbox_exec`] run: the status (a `STATUS_*` code), the `i64`-widened return value
/// (when `STATUS_OK`), the exit code (when `STATUS_EXIT`), and the bytes the guest wrote to its
/// stdout / stderr streams.
pub struct PbOutcome {
    pub status: i32,
    pub value: i64,
    pub exit_code: i32,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
}

/// The canonical names of the browser powerbox's capabilities, in grant order — the vocabulary a
/// powerbox guest resolves against via `cap.self.resolve` (F7) / labels via `cap.self.label` (F9). The
/// browser ABI grants `(stdout, stdin, exit, stderr, clock)` by arity (its set differs from `svm-run`'s
/// fixed §3e prefix after slot 3, since the capabilities differ), so the names follow that order.
const POWERBOX_CAP_NAMES: [&str; 5] = ["stdout", "stdin", "exit", "stderr", "clock"];

/// Run `m`'s function 0 under the **browser powerbox**, seeding `stdin` and capturing the streams.
///
/// Capabilities are granted by the entry's **arity** (so `hello.svm`'s 3-handle `(out, in, exit)`
/// shape works unchanged), in this order — the browser embedder's ABI:
///
/// | param # | capability        | `cap.call` type_id |
/// |---------|-------------------|--------------------|
/// | 1       | `Stream(Out)`     | 0 (op 1 = write)   |
/// | 2       | `Stream(In)`      | 0 (op 0 = read)    |
/// | 3       | `Exit`            | 1 (op 0 = exit)    |
/// | 4       | `Stream(Err)`     | 0 (op 1 = write)   |
/// | 5       | `Clock`           | 2 (op 0 = now)     |
///
/// Shared verbatim by the wasm [`svm_run_pb`] export and the native `gencorpus` ground truth, so the
/// differential compares the *same* logic on both builds.
pub fn powerbox_exec(m: &svm_ir::Module, stdin: &[u8]) -> PbOutcome {
    let arity = m.funcs.first().map_or(0, |f| f.params.len());
    let mut host = Host::new();
    host.stdin = stdin.to_vec();
    let mut slots: Vec<Value> = Vec::new();
    if arity >= 1 {
        slots.push(Value::I32(host.grant_stream(StreamRole::Out)));
    }
    if arity >= 2 {
        slots.push(Value::I32(host.grant_stream(StreamRole::In)));
    }
    if arity >= 3 {
        slots.push(Value::I32(host.grant_exit()));
    }
    if arity >= 4 {
        slots.push(Value::I32(host.grant_stream(StreamRole::Err)));
    }
    if arity >= 5 {
        slots.push(Value::I32(host.grant_clock()));
    }
    // §7 register each granted capability under its canonical name (F7/F9, PR #118) so a guest can
    // `cap.self.resolve` / `cap.self.label` it at runtime — mirroring `svm-run`'s powerbox so the
    // browser stays a faithful twin. Names parallel the grant order above; only the `arity` actually
    // granted are registered.
    for (name, slot) in POWERBOX_CAP_NAMES.iter().zip(&slots) {
        if let Value::I32(handle) = slot {
            host.register_cap_name(name, *handle);
        }
    }
    let mut fuel = u64::MAX;
    let (status, value, exit_code) =
        match bytecode::compile_and_run_with_host(m, 0, &slots, &mut fuel, &mut host) {
            None => (STATUS_UNSUPPORTED, 0, 0),
            Some(Err(Trap::Exit(code))) => (STATUS_EXIT, 0, code),
            Some(Err(_)) => (STATUS_TRAP, 0, 0),
            Some(Ok(vals)) => match vals.first() {
                Some(Value::I64(x)) => (STATUS_OK, *x, 0),
                Some(Value::I32(x)) => (STATUS_OK, *x as i64, 0),
                _ => (STATUS_BAD_RESULT, 0, 0),
            },
        };
    PbOutcome {
        status,
        value,
        exit_code,
        stdout: host.stdout,
        stderr: host.stderr,
    }
}

/// Outcome of a [`capture_exec`] run: the status, the `i64`-widened return value (when `STATUS_OK`),
/// and the **final window image** — the first `init.len()` bytes of the guest's memory after the run.
pub struct CapOutcome {
    pub status: i32,
    pub value: i64,
    pub snapshot: Vec<u8>,
}

/// Run `m`'s function 0 over a window seeded with `init` (deny-all `Host`), and capture the final
/// window image. This is the "host hands in a buffer, the guest transforms it in place, the host
/// reads it back" shape: [`bytecode::compile_and_run_capture`] snapshots the first `init.len()`
/// bytes of memory after the run. Shared verbatim by the wasm [`svm_run_capture`] export and the
/// native `gencorpus` ground truth, so the differential compares identical logic.
pub fn capture_exec(m: &svm_ir::Module, init: &[u8], arg: i64) -> CapOutcome {
    let mut fuel = u64::MAX;
    match bytecode::compile_and_run_capture(m, 0, &[Value::I64(arg)], &mut fuel, init) {
        None => CapOutcome {
            status: STATUS_UNSUPPORTED,
            value: 0,
            snapshot: Vec::new(),
        },
        Some((r, snapshot)) => {
            let (status, value) = match r {
                Err(_) => (STATUS_TRAP, 0),
                Ok(vals) => match vals.first() {
                    Some(Value::I64(x)) => (STATUS_OK, *x),
                    Some(Value::I32(x)) => (STATUS_OK, *x as i64),
                    _ => (STATUS_BAD_RESULT, 0),
                },
            };
            CapOutcome {
                status,
                value,
                snapshot,
            }
        }
    }
}

/// Run `m`'s function 0 with an `Instantiator` (iface 6) granted over `[0, 128 KiB)` — the §14
/// **nested-child** seam: function 0 may `instantiate`/`join` confined child domains over power-of-two
/// sub-windows of that range (a child runs on the cooperative executor, confined by masking to its
/// slice, joinable through the shared thread machinery). Returns `(status, i64-widened value)`.
/// Shared by the wasm [`svm_run_nested`] export and the native `gencorpus` ground truth.
pub fn instantiate_exec(m: &svm_ir::Module) -> (i32, i64) {
    let mut host = Host::new();
    let inst = host.grant_instantiator(0, 128 << 10);
    let mut fuel = 5_000_000u64;
    match bytecode::compile_and_run_with_host(m, 0, &[Value::I32(inst)], &mut fuel, &mut host) {
        None => (STATUS_UNSUPPORTED, 0),
        Some(Err(_)) => (STATUS_TRAP, 0),
        Some(Ok(vals)) => match vals.first() {
            Some(Value::I64(x)) => (STATUS_OK, *x),
            Some(Value::I32(x)) => (STATUS_OK, *x as i64),
            _ => (STATUS_BAD_RESULT, 0),
        },
    }
}

/// Captured stdout / stderr of the most recent [`svm_run_pb`], as cdylib-managed allocations
/// `(ptr, len)`. Each is a leaked boxed slice (exact length, alignment 1) freed when the next
/// [`svm_run_pb`] replaces it — so the host reads it via the `*_ptr`/`*_len` exports *before* the
/// next call and never frees it itself.
static mut OUT: (*mut u8, usize) = (core::ptr::null_mut(), 0);
static mut ERR: (*mut u8, usize) = (core::ptr::null_mut(), 0);
static mut EXIT_CODE: i32 = 0;
/// Captured final window image of the most recent [`svm_run_capture`] (same cdylib-managed lifetime
/// as `OUT`/`ERR`: valid until the next `svm_run_capture`).
static mut SNAP: (*mut u8, usize) = (core::ptr::null_mut(), 0);

/// Replace the capture in `slot` with `data`, freeing the previous allocation. Empty `data` stores
/// `(null, 0)`. The stored allocation is a boxed slice — exactly `len` bytes, alignment 1 — so it is
/// freed with the matching `Layout`.
fn stash(slot: &mut (*mut u8, usize), data: Vec<u8>) {
    let (old_ptr, old_len) = *slot;
    if !old_ptr.is_null() && old_len != 0 {
        if let Ok(layout) = Layout::from_size_align(old_len, 1) {
            unsafe { std::alloc::dealloc(old_ptr, layout) };
        }
    }
    *slot = if data.is_empty() {
        (core::ptr::null_mut(), 0)
    } else {
        let boxed = data.into_boxed_slice(); // shrink-to-fit: capacity == len, alignment 1
        let len = boxed.len();
        (Box::into_raw(boxed) as *mut u8, len)
    };
}

/// Decode the module at `[mod_ptr, mod_len)` and run function 0 under the **powerbox** (see
/// [`powerbox_exec`]): grant streams/clock/exit, seed stdin from `[stdin_ptr, stdin_len)` (a null /
/// zero-length range ⇒ empty stdin), capture the streams + exit code, and return the guest's `i64`
/// result (`0` on any non-`OK`/`EXIT` status). Read [`svm_status`] / [`svm_exit_code`] /
/// `svm_stdout_ptr`+`svm_stdout_len` / `svm_stderr_ptr`+`svm_stderr_len` afterward. Sets
/// [`LAST_STATUS`].
#[no_mangle]
pub extern "C" fn svm_run_pb(
    mod_ptr: *const u8,
    mod_len: usize,
    stdin_ptr: *const u8,
    stdin_len: usize,
) -> i64 {
    let set = |s: i32| unsafe { LAST_STATUS = s };
    // SAFETY: the host guarantees both ranges are live `svm_alloc`ations it just filled.
    let bytes = unsafe { core::slice::from_raw_parts(mod_ptr, mod_len) };
    let stdin: &[u8] = if stdin_ptr.is_null() || stdin_len == 0 {
        &[]
    } else {
        unsafe { core::slice::from_raw_parts(stdin_ptr, stdin_len) }
    };
    let m = match svm_encode::decode_module(bytes) {
        Ok(m) => m,
        Err(_) => {
            set(STATUS_DECODE_ERR);
            return 0;
        }
    };
    let out = powerbox_exec(&m, stdin);
    set(out.status);
    // SAFETY: single-threaded wasm; the capture slots are read back only via the export accessors.
    unsafe {
        stash(&mut *core::ptr::addr_of_mut!(OUT), out.stdout);
        stash(&mut *core::ptr::addr_of_mut!(ERR), out.stderr);
        EXIT_CODE = out.exit_code;
    }
    out.value
}

/// Pointer / length of the captured stdout from the most recent [`svm_run_pb`] (valid until the next
/// `svm_run_pb`; do not `svm_dealloc` it).
#[no_mangle]
pub extern "C" fn svm_stdout_ptr() -> *const u8 {
    unsafe { (*core::ptr::addr_of!(OUT)).0 }
}
#[no_mangle]
pub extern "C" fn svm_stdout_len() -> usize {
    unsafe { (*core::ptr::addr_of!(OUT)).1 }
}
/// Pointer / length of the captured stderr from the most recent [`svm_run_pb`] (same lifetime rule).
#[no_mangle]
pub extern "C" fn svm_stderr_ptr() -> *const u8 {
    unsafe { (*core::ptr::addr_of!(ERR)).0 }
}
#[no_mangle]
pub extern "C" fn svm_stderr_len() -> usize {
    unsafe { (*core::ptr::addr_of!(ERR)).1 }
}
/// Exit code from the most recent [`svm_run_pb`] (valid when [`svm_status`] is [`STATUS_EXIT`]).
#[no_mangle]
pub extern "C" fn svm_exit_code() -> i32 {
    unsafe { EXIT_CODE }
}

/// Decode the module at `[mod_ptr, mod_len)` and run function 0 (single `i64` `arg`, deny-all
/// `Host`) over a window **seeded** with `[init_ptr, init_len)`, then capture the final window image
/// (see [`capture_exec`]). Returns the guest's `i64` result; sets [`LAST_STATUS`]. The captured image
/// (the first `init_len` bytes of memory after the run) is read via [`svm_snapshot_ptr`] /
/// [`svm_snapshot_len`] and is cdylib-managed (valid until the next call; do not `svm_dealloc` it).
#[no_mangle]
pub extern "C" fn svm_run_capture(
    mod_ptr: *const u8,
    mod_len: usize,
    init_ptr: *const u8,
    init_len: usize,
    arg: i64,
) -> i64 {
    let set = |s: i32| unsafe { LAST_STATUS = s };
    // SAFETY: the host guarantees both ranges are live `svm_alloc`ations it just filled.
    let bytes = unsafe { core::slice::from_raw_parts(mod_ptr, mod_len) };
    let init: &[u8] = if init_ptr.is_null() || init_len == 0 {
        &[]
    } else {
        unsafe { core::slice::from_raw_parts(init_ptr, init_len) }
    };
    let m = match svm_encode::decode_module(bytes) {
        Ok(m) => m,
        Err(_) => {
            set(STATUS_DECODE_ERR);
            return 0;
        }
    };
    let out = capture_exec(&m, init, arg);
    set(out.status);
    // SAFETY: single-threaded wasm; the slot is read back only via the export accessors.
    unsafe { stash(&mut *core::ptr::addr_of_mut!(SNAP), out.snapshot) };
    out.value
}

/// Pointer / length of the captured final window image from the most recent [`svm_run_capture`].
#[no_mangle]
pub extern "C" fn svm_snapshot_ptr() -> *const u8 {
    unsafe { (*core::ptr::addr_of!(SNAP)).0 }
}
#[no_mangle]
pub extern "C" fn svm_snapshot_len() -> usize {
    unsafe { (*core::ptr::addr_of!(SNAP)).1 }
}

// ---- playground: in-browser SVM-text front end (parse → verify → encode) ------------------------

/// Compile the **SVM text** at `[src_ptr, src_len)` (UTF-8) into the `svm-encode` binary form the
/// `svm_run*` / `svm_par_*` entries consume: parse (`svm-text`) → verify (`svm-verify`) → encode.
/// Returns `1` and stashes the encoded module bytes, or `0` and stashes a UTF-8 error message
/// (which stage failed and why). Read the stash via [`svm_parse_ptr`] + [`svm_parse_len`] before
/// the next call — this is the playground's front end, so rejects must come back as *messages*,
/// not statuses.
#[no_mangle]
pub extern "C" fn svm_parse(src_ptr: *const u8, src_len: usize) -> i32 {
    let bytes: &[u8] = if src_ptr.is_null() || src_len == 0 {
        &[]
    } else {
        // SAFETY: the host guarantees `[src_ptr, src_len)` is a live allocation it just filled.
        unsafe { core::slice::from_raw_parts(src_ptr, src_len) }
    };
    // SAFETY: single-threaded main-thread use; the slot is read back only via the accessors below.
    let put = |ok: i32, data: Vec<u8>| -> i32 {
        unsafe { stash(&mut *core::ptr::addr_of_mut!(PARSE), data) };
        ok
    };
    let src = match core::str::from_utf8(bytes) {
        Ok(s) => s,
        Err(e) => return put(0, format!("source is not UTF-8: {e}").into_bytes()),
    };
    let m = match svm_text::parse_module(src) {
        Ok(m) => m,
        // `ParseError`'s Display already carries the "parse error: " prefix.
        Err(e) => return put(0, format!("{e}").into_bytes()),
    };
    if let Err(e) = svm_verify::verify_module(&m) {
        return put(0, format!("verify error: {e:?}").into_bytes());
    }
    put(1, svm_encode::encode_module(&m))
}

/// Pointer / length of the most recent [`svm_parse`] output (module bytes on `1`, error text on `0`).
#[no_mangle]
pub extern "C" fn svm_parse_ptr() -> *const u8 {
    unsafe { (*core::ptr::addr_of!(PARSE)).0 }
}
#[no_mangle]
pub extern "C" fn svm_parse_len() -> usize {
    unsafe { (*core::ptr::addr_of!(PARSE)).1 }
}
/// The stashed [`svm_parse`] output (same cdylib-managed lifetime as `OUT`/`ERR`).
static mut PARSE: (*mut u8, usize) = (core::ptr::null_mut(), 0);

/// Run `m`'s function 0 under a deterministic **3-cap powerbox** — `Stream(Out)` (type 0), `Exit`
/// (type 1), and a host-fn (type 13), granted in that order — so the §7 reflection ops
/// `cap.self.count` / `cap.self.get` see a fixed, known capability table. Passes `arg` only if the
/// entry takes one. Returns `(status, i64-widened value)`. Shared by [`svm_run_reflect`] and
/// `gencorpus`.
pub fn reflect_exec(m: &svm_ir::Module, arg: i64) -> (i32, i64) {
    let mut host = Host::new();
    let _ = host.grant_stream(StreamRole::Out); // handle 0, type_id 0
    let _ = host.grant_exit(); // handle 1, type_id 1
    let _ = host.grant_host_fn(Box::new(|_op, _args, _mem| Ok(vec![0]))); // handle 2, type_id 13
    let arity = m.funcs.first().map_or(0, |f| f.params.len());
    let args: Vec<Value> = if arity >= 1 {
        vec![Value::I32(arg as i32)]
    } else {
        Vec::new()
    };
    let mut fuel = 1_000_000u64;
    match bytecode::compile_and_run_with_host(m, 0, &args, &mut fuel, &mut host) {
        None => (STATUS_UNSUPPORTED, 0),
        Some(Err(_)) => (STATUS_TRAP, 0),
        Some(Ok(vals)) => match vals.first() {
            Some(Value::I64(x)) => (STATUS_OK, *x),
            Some(Value::I32(x)) => (STATUS_OK, *x as i64),
            _ => (STATUS_BAD_RESULT, 0),
        },
    }
}

// A minimal symbol-table wire form for `compile_linked` (the browser embedder's own, since the engine
// passes the bytes opaquely to the validator — both ends are ours). Each entry: `name_len: u8`,
// `name` bytes (UTF-8), `type_id: u32` LE, `op: u32` LE — a name → `Cap(type_id, op)` binding. Empty
// bytes ⇒ no bindings (the closed-blob `compile` op), so a unit with imports fails closed.

/// Build a `compile_linked` symbol table binding each `name` to a host capability `(type_id, op)`.
fn encode_symtab(entries: &[(&str, u32, u32)]) -> Vec<u8> {
    let mut out = Vec::new();
    for (name, type_id, op) in entries {
        out.push(name.len() as u8);
        out.extend_from_slice(name.as_bytes());
        out.extend_from_slice(&type_id.to_le_bytes());
        out.extend_from_slice(&op.to_le_bytes());
    }
    out
}

/// Decode an [`encode_symtab`] buffer; `None` (fail-closed) on any malformation.
fn decode_symtab(bytes: &[u8]) -> Option<Vec<(String, u32, u32)>> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        let len = *bytes.get(i)? as usize;
        i += 1;
        let name = core::str::from_utf8(bytes.get(i..i + len)?).ok()?.to_string();
        i += len;
        let type_id = u32::from_le_bytes(bytes.get(i..i + 4)?.try_into().ok()?);
        i += 4;
        let op = u32::from_le_bytes(bytes.get(i..i + 4)?.try_into().ok()?);
        i += 4;
        out.push((name, type_id, op));
    }
    Some(out)
}

/// The browser's [`svm_interp::JitValidator`] — the §22 security hinge for the guest-driven `Jit`
/// cap: decode the symbol table → `decode_module` (fail-closed) → resolve named imports against the
/// table → `verify_module` (the escape-freedom gate) → the memory-match precondition → reject data
/// segments and concurrency ops. A pure-Rust replica of `svm-run`'s canonical validator (own symtab
/// wire form), so it builds for wasm with no Cranelift dep.
fn browser_jit_validator(
    bytes: &[u8],
    mem_log2: Option<u8>,
    symtab: &[u8],
) -> Result<std::sync::Arc<[svm_ir::Func]>, i64> {
    const EINVAL: i64 = -22;
    let Some(table) = decode_symtab(symtab) else {
        return Err(EINVAL);
    };
    let Ok(m) = svm_encode::decode_module(bytes) else {
        return Err(EINVAL);
    };
    // Bind named imports to host caps via the table; an unresolved import ⇒ fail closed (re-verified).
    let resolve = |name: &str| {
        table.iter().find(|(n, _, _)| n == name).map(|(_, t, o)| {
            svm_ir::Resolved::Cap(svm_ir::ResolvedCap {
                type_id: *t,
                op: *o,
            })
        })
    };
    let Ok(m) = svm_ir::resolve_imports_with(&m, resolve) else {
        return Err(EINVAL);
    };
    if svm_verify::verify_module(&m).is_err() {
        return Err(EINVAL);
    }
    if m.memory.map(|mc| mc.size_log2) != mem_log2 {
        return Err(EINVAL); // declared memory must equal the parent window
    }
    if !m.data.is_empty() || m.funcs.is_empty() || m.funcs.iter().any(|f| f.uses_concurrency()) {
        return Err(EINVAL);
    }
    Ok(m.funcs.into())
}

/// A unit the guest-JIT path installs and calls: `service(a, b) = a*b + 100`. Host-compiled (the
/// bytecode entry builds memory from the module, so no in-guest blob seeding is needed).
const JIT_SERVICE: &str = r#"memory 16
func (i32, i32) -> (i32) {
block0(v0: i32, v1: i32):
  v2 = i32.mul v0 v1
  v3 = i32.const 100
  v4 = i32.add v2 v3
  return v4
}
"#;

/// Run `m`'s function 0 with a **`Jit`** cap (iface 11) and a host-compiled [`JIT_SERVICE`] unit:
/// the guest receives `(jit_handle, code_handle, a, b)`, `install`s the unit into its dispatch table
/// (op 3), then `call_indirect`s it — guest-driven code loading, **interpreted** (the bytecode engine
/// lowers the submitted unit to bytecode; no native backend). `a=6, b=7`. Returns `(status, value)`.
pub fn jit_exec(m: &svm_ir::Module) -> (i32, i64) {
    let service = match svm_text::parse_module(JIT_SERVICE) {
        Ok(s) => svm_encode::encode_module(&s),
        Err(_) => return (STATUS_BAD_RESULT, 0),
    };
    let mut host = Host::new();
    let jit = host.grant_jit_with_table(m.memory.map(|mc| mc.size_log2), 4); // 2^4 = 16-slot table
    host.set_jit_validator(browser_jit_validator);
    let code = match host.jit_compile(jit, &service) {
        Ok(Ok(c)) => c.handle,
        _ => return (STATUS_TRAP, 0),
    };
    let args = [Value::I32(jit), Value::I32(code), Value::I32(6), Value::I32(7)];
    let mut fuel = 50_000_000u64;
    match bytecode::compile_and_run_with_host(m, 0, &args, &mut fuel, &mut host) {
        None => (STATUS_UNSUPPORTED, 0),
        Some(Err(_)) => (STATUS_TRAP, 0),
        Some(Ok(vals)) => match vals.first() {
            Some(Value::I64(x)) => (STATUS_OK, *x),
            Some(Value::I32(x)) => (STATUS_OK, *x as i64),
            _ => (STATUS_BAD_RESULT, 0),
        },
    }
}

/// A separately-compiled unit with a **named import** `"clock"`, resolved by `compile_linked`'s
/// symbol table to a host capability — a plugin reaching a host service by name. `clock.now` first
/// reads `0`, so the unit returns `0 + 777 = 777` once linked. (Declares `memory 16` to satisfy the
/// memory-match precondition against the parent window.)
const DL_UNIT: &str = r#"memory 16
func (i32) -> (i64) {
block0(v0: i32):
  v1 = call.import "clock" () -> (i64) v0 ()
  v2 = i64.const 777
  v3 = i64.add v1 v2
  return v3
}
"#;

/// Run `m`'s function 0 with a `Jit` cap, a `Clock` cap, and a host-`compile_linked` [`DL_UNIT`]:
/// **dynamic linking** — the unit's named import `"clock"` is bound (via the symbol table) to the
/// `Clock` capability `(type_id 2, op 0)` before verify, lowering `call.import "clock"` to a real
/// `cap.call 2 0`. The guest receives `(jit, code, clock)`, installs the unit and `call_indirect`s it
/// passing the clock handle → `777`. With `link == false` the symbol table is empty, so the import is
/// unresolved and `compile_linked` fails closed (`STATUS_TRAP`). Returns `(status, value)`.
pub fn dynlink_exec(m: &svm_ir::Module, link: bool) -> (i32, i64) {
    let unit = match svm_text::parse_module(DL_UNIT) {
        Ok(u) => svm_encode::encode_module(&u),
        Err(_) => return (STATUS_BAD_RESULT, 0),
    };
    let mut host = Host::new();
    let jit = host.grant_jit_with_table(m.memory.map(|mc| mc.size_log2), 4);
    host.set_jit_validator(browser_jit_validator);
    let clock = host.grant_clock();
    // Bind "clock" → the Clock cap (iface 2, op 0) iff linking; otherwise an empty table (fail-closed).
    let symtab = if link {
        encode_symtab(&[("clock", 2, 0)])
    } else {
        Vec::new()
    };
    let code = match host.jit_compile_linked(jit, &unit, &symtab) {
        Ok(Ok(c)) => c.handle,
        _ => return (STATUS_TRAP, 0), // unresolved import ⇒ compile_linked fails closed
    };
    let args = [Value::I32(jit), Value::I32(code), Value::I32(clock)];
    let mut fuel = 50_000_000u64;
    match bytecode::compile_and_run_with_host(m, 0, &args, &mut fuel, &mut host) {
        None => (STATUS_UNSUPPORTED, 0),
        Some(Err(_)) => (STATUS_TRAP, 0),
        Some(Ok(vals)) => match vals.first() {
            Some(Value::I64(x)) => (STATUS_OK, *x),
            Some(Value::I32(x)) => (STATUS_OK, *x as i64),
            _ => (STATUS_BAD_RESULT, 0),
        },
    }
}

/// Decode the module at `[mod_ptr, mod_len)` and run function 0 with a dynamically-**linked** unit
/// (see [`dynlink_exec`]); `link != 0` binds the unit's `"clock"` import, `0` leaves it unresolved
/// (fail-closed). Returns the guest's `i64` result; sets [`LAST_STATUS`].
#[no_mangle]
pub extern "C" fn svm_run_dynlink(mod_ptr: *const u8, mod_len: usize, link: i32) -> i64 {
    let set = |s: i32| unsafe { LAST_STATUS = s };
    // SAFETY: the host guarantees `[mod_ptr, mod_len)` is a live `svm_alloc`ation it just filled.
    let bytes = unsafe { core::slice::from_raw_parts(mod_ptr, mod_len) };
    let m = match svm_encode::decode_module(bytes) {
        Ok(m) => m,
        Err(_) => {
            set(STATUS_DECODE_ERR);
            return 0;
        }
    };
    let (status, value) = dynlink_exec(&m, link != 0);
    set(status);
    value
}

/// Decode the module at `[mod_ptr, mod_len)` and run function 0 under the **guest-JIT** powerbox (see
/// [`jit_exec`]). Returns the guest's `i64` result; sets [`LAST_STATUS`].
#[no_mangle]
pub extern "C" fn svm_run_jit(mod_ptr: *const u8, mod_len: usize) -> i64 {
    let set = |s: i32| unsafe { LAST_STATUS = s };
    // SAFETY: the host guarantees `[mod_ptr, mod_len)` is a live `svm_alloc`ation it just filled.
    let bytes = unsafe { core::slice::from_raw_parts(mod_ptr, mod_len) };
    let m = match svm_encode::decode_module(bytes) {
        Ok(m) => m,
        Err(_) => {
            set(STATUS_DECODE_ERR);
            return 0;
        }
    };
    let (status, value) = jit_exec(&m);
    set(status);
    value
}

/// Run an **already-instrumented** (durability-transformed) module's function 0 over a durable
/// `window` (its low bytes carry the state word `NORMAL`/`UNWINDING`/`REWINDING` + the shadow region),
/// with a `Clock` cap seeded to `clock_v`. Single-vCPU / single-fiber freeze/thaw is *driven by the
/// transform's emitted IR* (DURABILITY.md §2) — the engine just runs it. Returns `(status, value,
/// final-window snapshot, clock_after)`. Shared by [`svm_run_durable`] and `gencorpus`.
pub fn durable_run(inst: &svm_ir::Module, window: &[u8], clock_v: i64) -> (i32, i64, Vec<u8>, i64) {
    let mut host = Host::new();
    host.set_durable(true);
    let clk = host.grant_clock();
    host.clock_ns = clock_v;
    let mut fuel = 1_000_000u64;
    match bytecode::compile_and_run_capture_reserved_with_host(
        inst,
        0,
        &[Value::I32(clk)],
        &mut fuel,
        window,
        17, // SIZE_LOG2 = 128 KiB ≥ the durable reserve
        &mut host,
    ) {
        None => (STATUS_UNSUPPORTED, 0, Vec::new(), host.clock_ns),
        Some((r, snap)) => {
            let (status, value) = match r {
                Err(_) => (STATUS_TRAP, 0),
                Ok(vals) => match vals.first() {
                    Some(Value::I64(x)) => (STATUS_OK, *x),
                    Some(Value::I32(x)) => (STATUS_OK, *x as i64),
                    _ => (STATUS_BAD_RESULT, 0),
                },
            };
            (status, value, snap, host.clock_ns)
        }
    }
}

/// Decode the **instrumented** module at `[mod_ptr, mod_len)`, run function 0 over the durable window
/// at `[init_ptr, init_len)` (the state word lives in those bytes) with the clock seeded to `clock`
/// (see [`durable_run`]). The final window image is captured to the snapshot slot
/// (`svm_snapshot_ptr`/`svm_snapshot_len`). Returns the guest's `i64` result; sets [`LAST_STATUS`].
#[no_mangle]
pub extern "C" fn svm_run_durable(
    mod_ptr: *const u8,
    mod_len: usize,
    init_ptr: *const u8,
    init_len: usize,
    clock: i64,
) -> i64 {
    let set = |s: i32| unsafe { LAST_STATUS = s };
    // SAFETY: the host guarantees both ranges are live `svm_alloc`ations it just filled.
    let bytes = unsafe { core::slice::from_raw_parts(mod_ptr, mod_len) };
    let window = unsafe { core::slice::from_raw_parts(init_ptr, init_len) };
    let m = match svm_encode::decode_module(bytes) {
        Ok(m) => m,
        Err(_) => {
            set(STATUS_DECODE_ERR);
            return 0;
        }
    };
    let (status, value, snap, _clk) = durable_run(&m, window, clock);
    set(status);
    // SAFETY: single-threaded wasm; read back only via the snapshot accessors.
    unsafe { stash(&mut *core::ptr::addr_of_mut!(SNAP), snap) };
    value
}

/// Run `m`'s function 0 with a host-granted **`SharedRegion`** (iface 4, 64 KiB) as its sole cap —
/// the §13 host-backed memory object a guest `map`s into its window (op 0), aliasing the same backing
/// at multiple offsets (the magic-ring-buffer primitive); op 2 `len`, op 3 `page_size`. Returns
/// `(status, i64-widened value)`. Shared by [`svm_run_region`] and `gencorpus`.
pub fn region_exec(m: &svm_ir::Module) -> (i32, i64) {
    let mut host = Host::new();
    let h = host.grant_shared_region(1 << 16); // 64 KiB, comfortably larger than any host page
    let mut fuel = 5_000_000u64;
    match bytecode::compile_and_run_with_host(m, 0, &[Value::I32(h)], &mut fuel, &mut host) {
        None => (STATUS_UNSUPPORTED, 0),
        Some(Err(_)) => (STATUS_TRAP, 0),
        Some(Ok(vals)) => match vals.first() {
            Some(Value::I64(x)) => (STATUS_OK, *x),
            Some(Value::I32(x)) => (STATUS_OK, *x as i64),
            _ => (STATUS_BAD_RESULT, 0),
        },
    }
}

/// Decode the module at `[mod_ptr, mod_len)` and run function 0 with a `SharedRegion` cap (see
/// [`region_exec`]). Returns the guest's `i64` result; sets [`LAST_STATUS`].
#[no_mangle]
pub extern "C" fn svm_run_region(mod_ptr: *const u8, mod_len: usize) -> i64 {
    let set = |s: i32| unsafe { LAST_STATUS = s };
    // SAFETY: the host guarantees `[mod_ptr, mod_len)` is a live `svm_alloc`ation it just filled.
    let bytes = unsafe { core::slice::from_raw_parts(mod_ptr, mod_len) };
    let m = match svm_encode::decode_module(bytes) {
        Ok(m) => m,
        Err(_) => {
            set(STATUS_DECODE_ERR);
            return 0;
        }
    };
    let (status, value) = region_exec(&m);
    set(status);
    value
}

/// Decode the module at `[mod_ptr, mod_len)` and run function 0 under a fixed 3-cap powerbox, so §7
/// reflection (`cap.self.count`/`get`) is deterministic (see [`reflect_exec`]). Returns the guest's
/// `i64` result; sets [`LAST_STATUS`].
#[no_mangle]
pub extern "C" fn svm_run_reflect(mod_ptr: *const u8, mod_len: usize, arg: i64) -> i64 {
    let set = |s: i32| unsafe { LAST_STATUS = s };
    // SAFETY: the host guarantees `[mod_ptr, mod_len)` is a live `svm_alloc`ation it just filled.
    let bytes = unsafe { core::slice::from_raw_parts(mod_ptr, mod_len) };
    let m = match svm_encode::decode_module(bytes) {
        Ok(m) => m,
        Err(_) => {
            set(STATUS_DECODE_ERR);
            return 0;
        }
    };
    let (status, value) = reflect_exec(&m, arg);
    set(status);
    value
}

/// Decode the module at `[mod_ptr, mod_len)` and run function 0 under the **nested-child** powerbox
/// (an `Instantiator` over `[0, 128 KiB)`; see [`instantiate_exec`]): function 0 may `instantiate`
/// confined child guests over sub-windows and `join` them. Returns the guest's `i64` result; sets
/// [`LAST_STATUS`].
#[no_mangle]
pub extern "C" fn svm_run_nested(mod_ptr: *const u8, mod_len: usize) -> i64 {
    let set = |s: i32| unsafe { LAST_STATUS = s };
    // SAFETY: the host guarantees `[mod_ptr, mod_len)` is a live `svm_alloc`ation it just filled.
    let bytes = unsafe { core::slice::from_raw_parts(mod_ptr, mod_len) };
    let m = match svm_encode::decode_module(bytes) {
        Ok(m) => m,
        Err(_) => {
            set(STATUS_DECODE_ERR);
            return 0;
        }
    };
    let (status, value) = instantiate_exec(&m);
    set(status);
    value
}

/// Self-contained powerbox probe (no host buffers, so usable via `wasmtime --invoke run_powerbox`):
/// run a greeting guest that writes 17 bytes to stdout, then an `exit(42)` guest, and return `17`
/// iff both the captured stdout length **and** the exit code are correct on this target — i.e. the
/// stream-write/capture and exit-trap paths work on wasm64. Returns `-1` on any mismatch.
#[no_mangle]
pub extern "C" fn run_powerbox() -> i64 {
    const HELLO: &str = r#"
memory 16
data 0 "hello, powerbox!\n"
func (i32, i32, i32) -> (i32) {
block0(v0: i32, v1: i32, v2: i32):
  v3 = i64.const 0
  v4 = i64.const 17
  v5 = cap.call 0 1 (i64, i64) -> (i64) v0(v3, v4)
  v6 = i32.const 0
  return v6
}
"#;
    const EXIT: &str = r#"
func (i32, i32, i32) -> (i32) {
block0(v0: i32, v1: i32, v2: i32):
  v3 = i32.const 42
  cap.call 1 0 (i32) -> () v2(v3)
  v4 = i32.const 0
  return v4
}
"#;
    let (Ok(hm), Ok(em)) = (svm_text::parse_module(HELLO), svm_text::parse_module(EXIT)) else {
        return -1;
    };
    let h = powerbox_exec(&hm, &[]);
    let e = powerbox_exec(&em, &[]);
    if h.status == STATUS_OK && h.stdout == b"hello, powerbox!\n" && e.status == STATUS_EXIT && e.exit_code == 42 {
        h.stdout.len() as i64
    } else {
        -1
    }
}

/// Self-contained capture probe (seeds its own window, so usable via `wasmtime --invoke run_capture`):
/// run an in-place "add `arg` to each i64 word" guest over a 16-word window whose word 0 is `1000`,
/// with `arg = 7`, and return word 0 of the **captured final image** — `1007` iff seeding, the
/// in-place writes, and the snapshot all work on this target. Returns `-1` on any mismatch.
#[no_mangle]
pub extern "C" fn run_capture() -> i64 {
    const ADDK: &str = r#"
memory 16
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const 0
  br block1(v0, v1)
block1(v2: i64, v3: i64):
  v4 = i64.const 128
  v5 = i64.lt_u v3 v4
  br_if v5 block2(v2, v3) block3()
block2(v6: i64, v7: i64):
  v8 = i64.load v7
  v9 = i64.add v8 v6
  i64.store v7 v9
  v10 = i64.const 8
  v11 = i64.add v7 v10
  br block1(v6, v11)
block3():
  v12 = i64.const 0
  v13 = i64.load v12
  return v13
}
"#;
    let Ok(m) = svm_text::parse_module(ADDK) else {
        return -1;
    };
    // Seed 16 i64 words: word 0 = 1000, the rest 0.
    let mut init = [0u8; 128];
    init[..8].copy_from_slice(&1000i64.to_le_bytes());
    let out = capture_exec(&m, &init, 7);
    if out.status != STATUS_OK || out.snapshot.len() != 128 {
        return -1;
    }
    // Word 0 of the captured image should be 1000 + 7 = 1007.
    i64::from_le_bytes(out.snapshot[..8].try_into().unwrap())
}

/// Self-contained nested-child probe (so usable via `wasmtime --invoke run_instantiate`): a parent
/// `instantiate`s a confined child in a 4 KiB sub-window at 64 KiB, the child writes a marker into
/// the shared backing and returns 42, the parent joins and reads the marker back — returning
/// `42 * 1000 + 123 = 42123` iff confined child execution + the shared data plane work on this
/// target. Returns `-1` on any mismatch.
#[no_mangle]
pub extern "C" fn run_instantiate() -> i64 {
    const SHARED: &str = r#"memory 17
func (i32) -> (i64) {
block0(v0: i32):
  v1 = i64.const 1
  v2 = i64.const 65536
  v3 = i64.const 12
  v4 = i64.const 0
  v5 = cap.call 6 0 (i64, i64, i64, i64) -> (i32) v0 (v1, v2, v3, v4)
  v6 = cap.call 6 1 (i32) -> (i64) v0 (v5)
  v7 = i64.const 65543
  v8 = i32.load8_u v7
  v9 = i64.extend_i32_u v8
  v10 = i64.const 1000
  v11 = i64.mul v6 v10
  v12 = i64.add v11 v9
  return v12
}
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const 7
  v2 = i32.const 123
  i32.store8 v1 v2
  v3 = i64.const 42
  return v3
}
"#;
    let Ok(m) = svm_text::parse_module(SHARED) else {
        return -1;
    };
    match instantiate_exec(&m) {
        (STATUS_OK, v) => v,
        _ => -1,
    }
}

/// Self-contained SIMD probe (`wasmtime --invoke run_simd`): splat 21 into an `i64x2`, add lanewise,
/// extract lane 0 → `42`. Returns `-1` on any mismatch.
#[no_mangle]
pub extern "C" fn run_simd() -> i64 {
    const S: &str = r#"
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64x2.splat v0
  v2 = i64x2.add v1 v1
  v3 = i64x2.extract_lane 0 v2
  return v3
}
"#;
    let Ok(m) = svm_text::parse_module(S) else {
        return -1;
    };
    let mut fuel = u64::MAX;
    match bytecode::compile_and_run(&m, 0, &[Value::I64(21)], &mut fuel) {
        Some(Ok(vals)) => match vals.first() {
            Some(Value::I64(x)) => *x,
            _ => -1,
        },
        _ => -1,
    }
}

/// Self-contained durability probe (`wasmtime --invoke run_durable`): instrument a single-fiber
/// program that reads the clock twice (each an unwind point), run it NORMAL over a fresh durable
/// window with the clock seeded to 1000 → `1000 + 1001 = 2001`. Proves the freeze/thaw transform's
/// emitted IR runs on this target. Returns `-1` on any mismatch.
#[no_mangle]
pub extern "C" fn run_durable() -> i64 {
    const SRC: &str = r#"memory 17
func (i32) -> (i64) {
block0(v0: i32):
  v1 = cap.call 2 0 () -> (i64) v0 ()
  v2 = cap.call 2 0 () -> (i64) v0 ()
  v3 = i64.add v1 v2
  return v3
}
"#;
    let Ok(m) = svm_text::parse_module(SRC) else {
        return -1;
    };
    let Ok(inst) = svm_durable::transform_module(&m) else {
        return -1;
    };
    let mut win = svm_durable::init_durable_window(1 << 17);
    svm_durable::write_state(&mut win, svm_durable::STATE_NORMAL);
    match durable_run(&inst, &win, 1000) {
        (STATUS_OK, v, _, _) => v,
        _ => -1,
    }
}

/// Self-contained dynamic-linking probe (`wasmtime --invoke run_dynlink`): a unit's named import
/// `"clock"` is resolved by `compile_linked`'s symbol table to the Clock cap; the guest installs and
/// calls it → `clock.now (0) + 777 = 777`. Returns `-1` on any mismatch.
#[no_mangle]
pub extern "C" fn run_dynlink() -> i64 {
    const G: &str = r#"memory 16
func (i32, i32, i32) -> (i64) {
block0(v0: i32, v1: i32, v2: i32):
  v3 = i64.extend_i32_u v1
  v4 = cap.call 11 3 (i64) -> (i64) v0 (v3)
  v5 = i32.wrap_i64 v4
  v6 = call_indirect (i32) -> (i64) v5 (v2)
  return v6
}
"#;
    let Ok(m) = svm_text::parse_module(G) else {
        return -1;
    };
    match dynlink_exec(&m, true) {
        (STATUS_OK, v) => v,
        _ => -1,
    }
}

/// Self-contained guest-JIT probe (`wasmtime --invoke run_jit`): a guest installs a host-compiled
/// unit (`a*b+100`) into its dispatch table and `call_indirect`s it with `(6, 7)` → `142`. Proves
/// guest-driven code loading (validated + interpreted, no native backend) works. `-1` on mismatch.
#[no_mangle]
pub extern "C" fn run_jit() -> i64 {
    const G: &str = r#"memory 16
func (i32, i32, i32, i32) -> (i32) {
block0(v0: i32, v1: i32, v2: i32, v3: i32):
  v4 = i64.extend_i32_u v1
  v5 = cap.call 11 3 (i64) -> (i64) v0 (v4)
  v6 = i32.wrap_i64 v5
  v7 = call_indirect (i32, i32) -> (i32) v6 (v2, v3)
  return v7
}
"#;
    let Ok(m) = svm_text::parse_module(G) else {
        return -1;
    };
    match jit_exec(&m) {
        (STATUS_OK, v) => v,
        _ => -1,
    }
}

/// Self-contained SharedRegion probe (`wasmtime --invoke run_region`): map a host region at two
/// window offsets, store a marker through one and load it through the other → `0x0123456789abcdef`
/// (`81985529216486895`) iff the mappings alias the same backing. Returns `-1` on any mismatch.
#[no_mangle]
pub extern "C" fn run_region() -> i64 {
    const R: &str = r#"memory 17
func (i32) -> (i64) {
block0(v0: i32):
  v1 = cap.call 4 3 () -> (i64) v0 ()
  v2 = i64.const 0
  v3 = i32.const 3
  v4 = cap.call 4 0 (i64, i64, i64, i32) -> (i64) v0 (v2, v2, v1, v3)
  v5 = cap.call 4 0 (i64, i64, i64, i32) -> (i64) v0 (v1, v2, v1, v3)
  v6 = i64.const 81985529216486895
  i64.store v2 v6
  v7 = i64.load v1
  return v7
}
"#;
    let Ok(m) = svm_text::parse_module(R) else {
        return -1;
    };
    match region_exec(&m) {
        (STATUS_OK, v) => v,
        _ => -1,
    }
}

/// Self-contained reflection probe (`wasmtime --invoke run_reflect`): under the fixed 3-cap powerbox,
/// `cap.self.count` reports `3`. Returns `-1` on any mismatch.
#[no_mangle]
pub extern "C" fn run_reflect() -> i64 {
    const R: &str = r#"
func () -> (i32) {
block0():
  v0 = cap.self.count
  return v0
}
"#;
    let Ok(m) = svm_text::parse_module(R) else {
        return -1;
    };
    match reflect_exec(&m, 0) {
        (STATUS_OK, v) => v,
        _ => -1,
    }
}

/// Self-contained GC-roots probe (`wasmtime --invoke run_gcroots`): a `gc.roots` scan over an
/// activation holding the in-range constants `{4096, 5000}` (one duplicated, one out of range)
/// returns the root count `2`. Returns `-1` on any mismatch.
#[no_mangle]
pub extern "C" fn run_gcroots() -> i64 {
    const G: &str = r#"memory 16
func () -> (i64) {
block0():
  va = i64.const 4096
  vb = i64.const 5000
  vc = i64.const 5000
  vd = i64.const 9000
  vlo = i64.const 4096
  vhi = i64.const 8192
  vmask = i64.const -1
  vbuf = i64.const 0
  vcap = i64.const 64
  vt = gc.roots vlo vhi vmask vbuf vcap
  return vt
}
"#;
    let Ok(m) = svm_text::parse_module(G) else {
        return -1;
    };
    let init = [0u8; 4096];
    match capture_exec(&m, &init, 0) {
        out if out.status == STATUS_OK => out.value,
        _ => -1,
    }
}

/// Self-contained tail-call probe (`wasmtime --invoke run_tailcall`): a tail-recursive factorial via
/// `return_call` (O(1) window reuse) returns `5! = 120`. Returns `-1` on any mismatch.
#[no_mangle]
pub extern "C" fn run_tailcall() -> i64 {
    const T: &str = r#"
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const 1
  return_call 1(v0, v1)
}
func (i64, i64) -> (i64) {
block0(v0: i64, v1: i64):
  v2 = i64.const 1
  v3 = i64.lt_s v0 v2
  br_if v3 block1(v1) block2(v0, v1)
block1(v4: i64):
  return v4
block2(v5: i64, v6: i64):
  v7 = i64.mul v6 v5
  v8 = i64.const -1
  v9 = i64.add v5 v8
  return_call 1(v9, v7)
}
"#;
    let Ok(m) = svm_text::parse_module(T) else {
        return -1;
    };
    let mut fuel = u64::MAX;
    match bytecode::compile_and_run(&m, 0, &[Value::I64(5)], &mut fuel) {
        Some(Ok(vals)) => match vals.first() {
            Some(Value::I64(x)) => *x,
            _ => -1,
        },
        _ => -1,
    }
}

/// Self-contained fiber probe (`wasmtime --invoke run_fiber`): a §12 continuation (`cont.new`/
/// `cont.resume`) runs to completion, resumed with 7 and returning `7 + 100`. Returns `107` iff
/// cooperative continuation switching works on this target, else `-1`.
#[no_mangle]
pub extern "C" fn run_fiber() -> i64 {
    const FIB: &str = r#"
func () -> (i64) {
block0():
  v0 = ref.func 1
  v1 = i64.const 0
  v2 = cont.new v0 v1
  v3 = i64.const 7
  v4, v5 = cont.resume v2 v3
  return v5
}
func (i64, i64) -> (i64) {
block0(vsp: i64, varg: i64):
  v0 = i64.const 100
  v1 = i64.add varg v0
  return v1
}
"#;
    let Ok(m) = svm_text::parse_module(FIB) else {
        return -1;
    };
    let mut fuel = u64::MAX;
    match bytecode::compile_and_run(&m, 0, &[], &mut fuel) {
        Some(Ok(vals)) => match vals.first() {
            Some(Value::I64(x)) => *x,
            _ => -1,
        },
        _ => -1,
    }
}

/// Self-contained coroutine probe (`wasmtime --invoke run_coroutine`): a §14 coroutine confined to a
/// sub-window is resumed three times, yielding 100, 210, then returning 1019. Returns
/// `100 + 210 + 1019 + RETURNED*1_000_000 = 1001329` iff `spawn_coroutine`/`resume`/`yield` work on
/// this target, else `-1`.
#[no_mangle]
pub extern "C" fn run_coroutine() -> i64 {
    const CORO: &str = r#"memory 17
func (i32) -> (i64) {
block0(v0: i32):
  v1 = i64.const 1
  v2 = i64.const 65536
  v3 = i64.const 16
  v4 = i64.const 0
  v5 = cap.call 6 2 (i64, i64, i64, i64) -> (i32) v0 (v1, v2, v3, v4)
  v6 = i64.const 0
  v7, v8 = cap.call 6 3 (i32, i64) -> (i32, i64) v0 (v5, v6)
  v9 = i64.const 10
  v10, v11 = cap.call 6 3 (i32, i64) -> (i32, i64) v0 (v5, v9)
  v12 = i64.const 20
  v13, v14 = cap.call 6 3 (i32, i64) -> (i32, i64) v0 (v5, v12)
  v15 = i64.add v8 v11
  v16 = i64.add v15 v14
  v17 = i64.extend_i32_s v13
  v18 = i64.const 1000000
  v19 = i64.mul v17 v18
  v20 = i64.add v16 v19
  return v20
}
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i32.wrap_i64 v0
  v2 = i64.const 0
  v3 = i32.const 7
  i32.store8 v2 v3
  v4 = i64.const 100
  v5 = cap.call 7 0 (i64) -> (i64) v1 (v4)
  v6 = i64.const 200
  v7 = i64.add v6 v5
  v8 = cap.call 7 0 (i64) -> (i64) v1 (v7)
  v9 = i64.const 999
  v10 = i64.add v9 v8
  return v10
}
"#;
    let Ok(m) = svm_text::parse_module(CORO) else {
        return -1;
    };
    match instantiate_exec(&m) {
        (STATUS_OK, v) => v,
        _ => -1,
    }
}

/// Self-contained scalar-float probe (`wasmtime --invoke run_float`): reinterpret the f64 bits of
/// `4.0`, take `sqrt(|·|)`, and return the result's i64 bits — `4611686018427387904` (the bits of
/// `2.0`) iff f64 reinterpret/abs/sqrt round-trip bit-exactly on this target, else `-1`.
#[no_mangle]
pub extern "C" fn run_float() -> i64 {
    const SQRT: &str = r#"
func (i64) -> (i64) {
block0(v0: i64):
  v1 = f64.reinterpret_i64 v0
  v2 = f64.abs v1
  v3 = f64.sqrt v2
  v4 = i64.reinterpret_f64 v3
  return v4
}
"#;
    let Ok(m) = svm_text::parse_module(SQRT) else {
        return -1;
    };
    let mut fuel = u64::MAX;
    // arg = bits(4.0) = 0x4010000000000000; sqrt(4.0) = 2.0 = bits 0x4000000000000000.
    let arg = 0x4010000000000000u64 as i64;
    match bytecode::compile_and_run(&m, 0, &[Value::I64(arg)], &mut fuel) {
        Some(Ok(vals)) => match vals.first() {
            Some(Value::I64(x)) => *x,
            _ => -1,
        },
        _ => -1,
    }
}

// ---- live host imports: bind capabilities to real host functions ----------------------------
//
// Everything above keeps the cdylib import-free by buffering I/O. This (feature-gated) entry instead
// bridges guest capabilities to **real wasm imports**, so a guest's writes reach the live host
// console *as they happen* and the clock reads real host time. The seam is `Host::grant_host_fn`
// (iface 13) — the designed extension point: a closure supplies the capability's semantics, here by
// calling out to the imported host function. The guest sees only a masked, type-checked handle.

#[cfg(feature = "live")]
pub mod live {
    use super::*;

    // The host functions the embedder must supply (module `svm_host`). `host_write` receives a
    // pointer into *this module's* linear memory (the bytes the guest wrote, copied out of its
    // window into a Rust buffer that lives on the wasm heap), so JS reads them as
    // `new Uint8Array(memory.buffer, ptr, len)`. `host_now_ns` returns real host time.
    #[link(wasm_import_module = "svm_host")]
    extern "C" {
        /// `host_write(stream, ptr, len)` — `stream` 0 = stdout, 1 = stderr.
        fn host_write(stream: i32, ptr: *const u8, len: usize);
        /// `host_now_ns() -> i64` — host wall/monotonic clock, nanoseconds.
        fn host_now_ns() -> i64;
    }

    const EFAULT: i64 = -14;
    const EINVAL: i64 = -22;

    /// Decode the module at `[mod_ptr, mod_len)` and run function 0 with a **host-backed** powerbox:
    /// `(console, clock)` capabilities (both iface `HOST_FN` = 13) bridged to the imports above.
    /// The guest calls `cap.call 13 1 (i64,i64,i64) -> (i64) v<console>(stream, ptr, len)` to write
    /// live, and `cap.call 13 0 () -> (i64) v<clock>()` to read the host clock. Returns the guest's
    /// `i64` result; sets [`LAST_STATUS`].
    #[no_mangle]
    pub extern "C" fn svm_run_live(mod_ptr: *const u8, mod_len: usize) -> i64 {
        // SAFETY: the host guarantees `[mod_ptr, mod_len)` is a live `svm_alloc`ation it just filled.
        let bytes = unsafe { core::slice::from_raw_parts(mod_ptr, mod_len) };
        let set = |s: i32| unsafe { LAST_STATUS = s };
        let m = match svm_encode::decode_module(bytes) {
            Ok(m) => m,
            Err(_) => {
                set(STATUS_DECODE_ERR);
                return 0;
            }
        };
        let mut host = Host::new();
        // console (param 1): op 1 = write(stream, ptr, len) → reads the guest window, forwards live.
        let console: HostFn = Box::new(|op, args, mem| {
            if op != 1 {
                return Ok(vec![EINVAL]);
            }
            let (Some(&stream), Some(&ptr), Some(&n)) =
                (args.first(), args.get(1), args.get(2))
            else {
                return Ok(vec![EINVAL]);
            };
            let Some(m) = mem else { return Ok(vec![EFAULT]) };
            match m.read_bytes(ptr as u64, n as u64) {
                // The copied bytes live on this module's wasm heap; hand their pointer to the host.
                Some(buf) => {
                    unsafe { host_write(stream as i32, buf.as_ptr(), buf.len()) };
                    Ok(vec![n])
                }
                None => Ok(vec![EFAULT]),
            }
        });
        // clock (param 2): op 0 = now() → real host time.
        let clock: HostFn = Box::new(|op, _args, _mem| {
            if op != 0 {
                return Ok(vec![EINVAL]);
            }
            Ok(vec![unsafe { host_now_ns() }])
        });
        let arity = m.funcs.first().map_or(0, |f| f.params.len());
        let mut slots: Vec<Value> = Vec::new();
        if arity >= 1 {
            slots.push(Value::I32(host.grant_host_fn(console)));
        }
        if arity >= 2 {
            slots.push(Value::I32(host.grant_host_fn(clock)));
        }
        // §7 register the live caps under canonical names (F7/F9, PR #118) so the guest can
        // `cap.self.resolve`/`label` them at runtime, matching the fixed-powerbox path.
        for (name, slot) in ["console", "clock"].iter().zip(&slots) {
            if let Value::I32(handle) = slot {
                host.register_cap_name(name, *handle);
            }
        }
        let mut fuel = u64::MAX;
        match bytecode::compile_and_run_with_host(&m, 0, &slots, &mut fuel, &mut host) {
            None => {
                set(STATUS_UNSUPPORTED);
                0
            }
            Some(Err(Trap::Exit(code))) => {
                set(STATUS_EXIT);
                unsafe { EXIT_CODE = code };
                0
            }
            Some(Err(_)) => {
                set(STATUS_TRAP);
                0
            }
            Some(Ok(vals)) => match vals.first() {
                Some(Value::I64(x)) => {
                    set(STATUS_OK);
                    *x
                }
                Some(Value::I32(x)) => {
                    set(STATUS_OK);
                    *x as i64
                }
                _ => {
                    set(STATUS_BAD_RESULT);
                    0
                }
            },
        }
    }
}
