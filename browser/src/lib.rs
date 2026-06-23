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
