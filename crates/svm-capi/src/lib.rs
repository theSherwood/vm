//! **`svm-capi` — the C ABI over the `svm-run` embedding surface** (POWERBOX.md Phase 5).
//!
//! A C program can: parse a module (text or binary IR), prepend the powerbox `_start`
//! ([`svm_ir::synth_powerbox_start`]), bind host capabilities **by name** (built-ins, or its own C
//! function pointers — the wasm-style import registry of Phase 2), instantiate, and run on any backend
//! under a uniform config (Phase 3) — then read back the outcome and captured stdout/stderr. It is the
//! same pipeline as the Rust `Instance` API, exposed through `extern "C"`.
//!
//! **Discipline (FFI safety):** every entry point catches panics at the boundary (a panic never
//! unwinds into C — it becomes a null/error return), reports failures through a thread-local message
//! ([`svm_last_error`]), and owns memory through explicit `*_free` calls. Handles are opaque pointers;
//! `instantiate*` **consume** the module/imports handles passed to them.
//!
//! **Host-capability callbacks** are compute-only in this slice: `(op, args) -> results`, no direct
//! guest-memory access (a guest reaches memory-backed I/O through the built-in `Stream` caps, whose
//! Rust implementations read/write the window). A C callback that needs the window is a follow-up
//! (it requires a bounds-checked `GuestMem` shim across the ABI).

use std::cell::RefCell;
use std::ffi::{c_char, c_void, CStr, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;
use std::time::Duration;

use svm_interp::{HostFn, Trap};
use svm_run::{
    instantiate, instantiate_with_imports, Backend, HostCap, Imports, Instance, Limits, Outcome,
    Run, RunConfig, Value,
};

// ----------------------------------------------------------------------------
// Status codes
// ----------------------------------------------------------------------------

/// Success.
pub const SVM_OK: i32 = 0;
/// A null handle was passed where a live one was required.
pub const SVM_ERR_NULL: i32 = 1;
/// A fallible operation failed; see [`svm_last_error`] for the message.
pub const SVM_ERR_FAILED: i32 = 2;
/// A panic was caught at the FFI boundary (a bug — please report); see [`svm_last_error`].
pub const SVM_ERR_PANIC: i32 = 3;

/// Backend selectors for [`svm_instance_run`] (mirror [`svm_run::Backend`]).
pub const SVM_BACKEND_TREEWALK: i32 = 0;
pub const SVM_BACKEND_BYTECODE: i32 = 1;
pub const SVM_BACKEND_JIT: i32 = 2;

/// `svm_run_outcome_kind` values.
pub const SVM_OUTCOME_RETURNED: i32 = 0;
pub const SVM_OUTCOME_EXITED: i32 = 1;

/// The max results a host-capability callback may return (the closure's scratch buffer size).
const SVM_MAX_RESULTS: usize = 16;

// ----------------------------------------------------------------------------
// Error reporting (thread-local; never panics across the boundary)
// ----------------------------------------------------------------------------

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

fn set_error(msg: impl Into<Vec<u8>>) {
    // Strip interior NULs so the message always survives as a C string.
    let bytes: Vec<u8> = msg.into().into_iter().filter(|&b| b != 0).collect();
    let c = CString::new(bytes).unwrap_or_default();
    LAST_ERROR.with(|e| *e.borrow_mut() = Some(c));
}

/// The last error message on this thread (set by a failed call), or `NULL` if none. Valid until the
/// next `svm-capi` call on the same thread; copy it if you need to keep it.
#[no_mangle]
pub extern "C" fn svm_last_error() -> *const c_char {
    LAST_ERROR.with(|e| e.borrow().as_ref().map_or(ptr::null(), |c| c.as_ptr()))
}

/// Run `f`, catching panics and `Err`s; on either, set the error and return a null pointer.
fn guard_ptr<T>(f: impl FnOnce() -> Result<*mut T, String>) -> *mut T {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(p)) => p,
        Ok(Err(e)) => {
            set_error(e);
            ptr::null_mut()
        }
        Err(_) => {
            set_error("panic caught at the svm-capi boundary");
            ptr::null_mut()
        }
    }
}

/// Run `f`, catching panics and `Err`s; map to a status code.
fn guard_status(f: impl FnOnce() -> Result<(), String>) -> i32 {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(())) => SVM_OK,
        Ok(Err(e)) => {
            set_error(e);
            SVM_ERR_FAILED
        }
        Err(_) => {
            set_error("panic caught at the svm-capi boundary");
            SVM_ERR_PANIC
        }
    }
}

// ----------------------------------------------------------------------------
// Opaque handles
// ----------------------------------------------------------------------------

/// An IR module (opaque). Built by `svm_module_parse_text` / `svm_module_decode`, consumed by
/// `svm_instantiate*`.
pub struct SvmModule(svm_ir::Module);
/// A name → capability registry (opaque). Built by `svm_imports_new`, consumed by
/// `svm_instantiate_with_imports`.
pub struct SvmImports(Imports);
/// A resolved, verified instance (opaque).
pub struct SvmInstance(Instance);
/// The result of a run (opaque): outcome + captured stdout/stderr.
pub struct SvmRun(Run);

// ----------------------------------------------------------------------------
// Module
// ----------------------------------------------------------------------------

/// Parse a module from **text IR** (a NUL-terminated UTF-8 string). Returns a module handle, or
/// `NULL` on a parse error (see [`svm_last_error`]).
///
/// # Safety
/// `ir` must be a valid NUL-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn svm_module_parse_text(ir: *const c_char) -> *mut SvmModule {
    guard_ptr(|| {
        if ir.is_null() {
            return Err("svm_module_parse_text: null ir".into());
        }
        let s = CStr::from_ptr(ir)
            .to_str()
            .map_err(|_| "ir is not valid UTF-8".to_string())?;
        let m = svm_text::parse_module(s).map_err(|e| format!("parse: {e:?}"))?;
        Ok(Box::into_raw(Box::new(SvmModule(m))))
    })
}

/// Parse a module from the **binary IR** encoding (`svm-encode`).
///
/// # Safety
/// `bytes` must point to `len` readable bytes.
#[no_mangle]
pub unsafe extern "C" fn svm_module_decode(bytes: *const u8, len: usize) -> *mut SvmModule {
    guard_ptr(|| {
        if bytes.is_null() && len != 0 {
            return Err("svm_module_decode: null bytes".into());
        }
        let slice = if len == 0 {
            &[][..]
        } else {
            std::slice::from_raw_parts(bytes, len)
        };
        let m = svm_encode::decode_module(slice).map_err(|e| format!("decode: {e:?}"))?;
        Ok(Box::into_raw(Box::new(SvmModule(m))))
    })
}

/// Prepend the powerbox `_start` (the bootstrap) to `m` in place, for `n_handles` granted capability
/// handles (stash slot `i` ↔ import `i`), seeding a guest heap if `seed_heap`. `entry` is the funcidx
/// of the program's entry (a `(i64 sp) -> ()`/`(T)` function) *before* the prepend.
///
/// # Safety
/// `m` must be a live module handle from this library.
#[no_mangle]
pub unsafe extern "C" fn svm_module_synth_powerbox_start(
    m: *mut SvmModule,
    entry: u32,
    n_handles: usize,
    seed_heap: bool,
) -> i32 {
    guard_status(|| {
        let m = m
            .as_mut()
            .ok_or("svm_module_synth_powerbox_start: null module")?;
        let module = std::mem::take(&mut m.0);
        let synthd = svm_ir::synth_powerbox_start(module, entry, n_handles, seed_heap)?;
        m.0 = synthd;
        Ok(())
    })
}

/// Free a module handle (only if it was *not* consumed by an `svm_instantiate*` call).
///
/// # Safety
/// `m` must be a live module handle from this library, or null.
#[no_mangle]
pub unsafe extern "C" fn svm_module_free(m: *mut SvmModule) {
    if !m.is_null() {
        drop(Box::from_raw(m));
    }
}

// ----------------------------------------------------------------------------
// Imports registry
// ----------------------------------------------------------------------------

/// A C host-capability callback: compute `n_results` (≤ buffer capacity) outputs from `n_args` inputs
/// for operation `op`. Return the number of results written (`>= 0`), or a negative value to **trap**
/// the capability call (fail-closed). `ctx` is the opaque pointer registered alongside the callback.
pub type SvmHostFn = extern "C" fn(
    ctx: *mut c_void,
    op: u32,
    args: *const i64,
    n_args: usize,
    results: *mut i64,
    results_cap: usize,
) -> i32;

/// A `Send`/`Sync` carrier for the callback's opaque `ctx` so the grant closure can cross threads (a
/// concurrent guest's workers may invoke the cap). The embedder is responsible for `ctx` being safe to
/// use from multiple threads when the guest is concurrent.
#[derive(Clone, Copy)]
struct CtxPtr(*mut c_void);
// SAFETY: opaque to us; thread-safety of the pointee is the embedder's contract (documented).
unsafe impl Send for CtxPtr {}
unsafe impl Sync for CtxPtr {}

/// Create an empty capability registry.
#[no_mangle]
pub extern "C" fn svm_imports_new() -> *mut SvmImports {
    guard_ptr(|| Ok(Box::into_raw(Box::new(SvmImports(Imports::new())))))
}

unsafe fn provide(imports: *mut SvmImports, name: *const c_char, cap: HostCap) -> i32 {
    guard_status(|| {
        let imports = imports.as_mut().ok_or("null imports")?;
        if name.is_null() {
            return Err("null name".into());
        }
        let name = CStr::from_ptr(name)
            .to_str()
            .map_err(|_| "name is not valid UTF-8".to_string())?;
        // `provide` takes `self` by value (builder); swap through a temporary.
        let reg = std::mem::take(&mut imports.0);
        imports.0 = reg.provide(name, cap);
        Ok(())
    })
}

/// Bind `name` to a writable `Stream` (stdout).
///
/// # Safety
/// `i` is a live registry handle and `name` a valid NUL-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn svm_imports_provide_stdout(
    i: *mut SvmImports,
    name: *const c_char,
) -> i32 {
    provide(i, name, HostCap::stdout())
}
/// Bind `name` to a readable `Stream` (stdin).
///
/// # Safety
/// `i` is a live registry handle and `name` a valid NUL-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn svm_imports_provide_stdin(i: *mut SvmImports, name: *const c_char) -> i32 {
    provide(i, name, HostCap::stdin())
}
/// Bind `name` to the `Exit` capability.
///
/// # Safety
/// `i` is a live registry handle and `name` a valid NUL-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn svm_imports_provide_exit(i: *mut SvmImports, name: *const c_char) -> i32 {
    provide(i, name, HostCap::exit())
}
/// Bind `name` to the `Clock` capability.
///
/// # Safety
/// `i` is a live registry handle and `name` a valid NUL-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn svm_imports_provide_clock(i: *mut SvmImports, name: *const c_char) -> i32 {
    provide(i, name, HostCap::clock())
}

/// Bind `name` to a **host-defined** capability implemented by the C callback `f` (operation `op`,
/// opaque `ctx`). The guest reaches it as `call.import "<name>"`.
///
/// # Safety
/// `i` is a live registry; `name` a valid C string; `f` a valid function pointer for the lifetime of
/// any instance built from this registry; `ctx` valid for that lifetime (and thread-safe if the guest
/// is concurrent).
#[no_mangle]
pub unsafe extern "C" fn svm_imports_provide_host_fn(
    i: *mut SvmImports,
    name: *const c_char,
    op: u32,
    f: SvmHostFn,
    ctx: *mut c_void,
) -> i32 {
    let ctx = CtxPtr(ctx);
    // `make` is called once per backend host; each builds a fresh `HostFn` that trampolines into `f`.
    let cap = HostCap::host_fn(op, move || -> HostFn {
        let ctx = ctx;
        Box::new(move |op, args, _mem| {
            // Force whole-`ctx` capture (the `Send`/`Sync` wrapper), not the disjoint `ctx.0` field
            // (a bare `*mut c_void`, which isn't `Send`) — Rust 2021 edition capture.
            let ctx = ctx;
            let mut buf = [0i64; SVM_MAX_RESULTS];
            let n = f(
                ctx.0,
                op,
                args.as_ptr(),
                args.len(),
                buf.as_mut_ptr(),
                buf.len(),
            );
            if n < 0 {
                return Err(Trap::CapFault);
            }
            let n = (n as usize).min(buf.len());
            Ok(buf[..n].to_vec())
        })
    });
    provide(i, name, cap)
}

/// Free a registry handle (only if it was *not* consumed by `svm_instantiate_with_imports`).
///
/// # Safety
/// `i` is a live registry handle from this library, or null.
#[no_mangle]
pub unsafe extern "C" fn svm_imports_free(i: *mut SvmImports) {
    if !i.is_null() {
        drop(Box::from_raw(i));
    }
}

// ----------------------------------------------------------------------------
// Instantiate (consume the module / imports)
// ----------------------------------------------------------------------------

/// Instantiate `m` under the fixed §3e powerbox (resolve imports via the reference host policy, then
/// verify). **Consumes `m`** (do not use or free it afterward). `NULL` on failure.
///
/// # Safety
/// `m` is a live module handle from this library.
#[no_mangle]
pub unsafe extern "C" fn svm_instantiate(m: *mut SvmModule) -> *mut SvmInstance {
    guard_ptr(|| {
        if m.is_null() {
            return Err("svm_instantiate: null module".into());
        }
        let module = Box::from_raw(m).0;
        let inst = instantiate(module)?;
        Ok(Box::into_raw(Box::new(SvmInstance(inst))))
    })
}

/// Instantiate `m` against the name-keyed registry `imports` (wasm-style binding). **Consumes both
/// `m` and `imports`** (do not use or free them afterward). `NULL` on failure (e.g. an unbound import).
///
/// # Safety
/// `m` and `imports` are live handles from this library.
#[no_mangle]
pub unsafe extern "C" fn svm_instantiate_with_imports(
    m: *mut SvmModule,
    imports: *mut SvmImports,
) -> *mut SvmInstance {
    guard_ptr(|| {
        if m.is_null() || imports.is_null() {
            return Err("svm_instantiate_with_imports: null module or imports".into());
        }
        let module = Box::from_raw(m).0;
        let reg = Box::from_raw(imports).0;
        let inst = instantiate_with_imports(module, reg)?;
        Ok(Box::into_raw(Box::new(SvmInstance(inst))))
    })
}

/// Free an instance handle.
///
/// # Safety
/// `i` is a live instance handle from this library, or null.
#[no_mangle]
pub unsafe extern "C" fn svm_instance_free(i: *mut SvmInstance) {
    if !i.is_null() {
        drop(Box::from_raw(i));
    }
}

// ----------------------------------------------------------------------------
// Run config (C-ABI mirror of `RunConfig`/`Limits`)
// ----------------------------------------------------------------------------

/// Run configuration. A `NULL` pointer means all defaults. `*_set` flags select whether the paired
/// field is applied (else the default is used); `max_fibers`/`max_vcpus` of `0` also mean "default".
#[repr(C)]
pub struct SvmRunConfig {
    /// Per-op fuel for the interpreters (applied iff `fuel_set`). Ignored by the JIT.
    pub fuel: u64,
    pub fuel_set: i32,
    /// JIT detect-and-kill deadline in milliseconds (applied iff `deadline_set`). Ignored by interps.
    pub deadline_ms: u64,
    pub deadline_set: i32,
    /// §15 spawn quota (`0` ⇒ default).
    pub max_fibers: usize,
    pub max_vcpus: usize,
    /// Guest stdin bytes (`NULL`/`0` ⇒ empty).
    pub stdin: *const u8,
    pub stdin_len: usize,
    /// Linear-memory window `size_log2` override (applied iff `memory_set`).
    pub memory_size_log2: u8,
    pub memory_set: i32,
}

/// Translate the C config (possibly null) into a Rust [`RunConfig`].
///
/// # Safety
/// `c` is null or points to a valid `SvmRunConfig`; its `stdin`/`stdin_len` describe a readable slice.
unsafe fn run_config(c: *const SvmRunConfig) -> RunConfig {
    let Some(c) = c.as_ref() else {
        return RunConfig::default();
    };
    let d = Limits::default();
    let stdin = if c.stdin.is_null() || c.stdin_len == 0 {
        Vec::new()
    } else {
        std::slice::from_raw_parts(c.stdin, c.stdin_len).to_vec()
    };
    RunConfig {
        limits: Limits {
            fuel: (c.fuel_set != 0).then_some(c.fuel),
            deadline: (c.deadline_set != 0).then(|| Duration::from_millis(c.deadline_ms)),
            max_fibers: if c.max_fibers != 0 {
                c.max_fibers
            } else {
                d.max_fibers
            },
            max_vcpus: if c.max_vcpus != 0 {
                c.max_vcpus
            } else {
                d.max_vcpus
            },
        },
        stdin,
        memory_size_log2: (c.memory_set != 0).then_some(c.memory_size_log2),
        // argv/env are not part of the current C config surface (the C tests run arg-less programs);
        // an `svm_run_config_set_args`-style setter is a later C-ABI follow-up.
        ..RunConfig::default()
    }
}

fn backend_of(b: i32) -> Result<Backend, String> {
    match b {
        SVM_BACKEND_TREEWALK => Ok(Backend::TreeWalk),
        SVM_BACKEND_BYTECODE => Ok(Backend::Bytecode),
        SVM_BACKEND_JIT => Ok(Backend::Jit),
        other => Err(format!("unknown backend selector {other}")),
    }
}

/// Run the powerbox entry on a single `backend` under `config` (null ⇒ defaults). Returns a run handle
/// (read with `svm_run_*`), or `NULL` on a trap / failure.
///
/// # Safety
/// `i` is a live instance handle; `config` is null or a valid `SvmRunConfig`.
#[no_mangle]
pub unsafe extern "C" fn svm_instance_run(
    i: *mut SvmInstance,
    backend: i32,
    config: *const SvmRunConfig,
) -> *mut SvmRun {
    guard_ptr(|| {
        let inst = i.as_ref().ok_or("svm_instance_run: null instance")?;
        let backend = backend_of(backend)?;
        let cfg = run_config(config);
        let run = inst.0.run(backend, &cfg)?;
        Ok(Box::into_raw(Box::new(SvmRun(run))))
    })
}

/// Run the powerbox entry on the tree-walker **and** the JIT under `config`, asserting they agree (the
/// interp == jit oracle). Returns a run handle, or `NULL` on divergence / trap / failure.
///
/// # Safety
/// `i` is a live instance handle; `config` is null or a valid `SvmRunConfig`.
#[no_mangle]
pub unsafe extern "C" fn svm_instance_run_diff(
    i: *mut SvmInstance,
    config: *const SvmRunConfig,
) -> *mut SvmRun {
    guard_ptr(|| {
        let inst = i.as_ref().ok_or("svm_instance_run_diff: null instance")?;
        let cfg = run_config(config);
        let run = inst.0.run_diff(&cfg)?;
        Ok(Box::into_raw(Box::new(SvmRun(run))))
    })
}

// ----------------------------------------------------------------------------
// Run results
// ----------------------------------------------------------------------------

fn value_slot(v: &Value) -> i64 {
    match v {
        Value::I32(x) => *x as i64,
        Value::I64(x) => *x,
        Value::F32(x) => x.to_bits() as i64,
        Value::F64(x) => x.to_bits() as i64,
        Value::Ref(x) => *x as i64,
        Value::V128(b) => i64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]),
    }
}

/// The captured stdout bytes (valid until `svm_run_free`); writes `*len`. Returns `NULL` (and `*len`
/// = 0) for a null handle.
///
/// # Safety
/// `r` is a live run handle (or null); `len` is a valid `size_t*` (or null).
#[no_mangle]
pub unsafe extern "C" fn svm_run_stdout(r: *const SvmRun, len: *mut usize) -> *const u8 {
    bytes_field(r, len, |run| &run.0.stdout)
}

/// The captured stderr bytes (valid until `svm_run_free`); writes `*len`.
///
/// # Safety
/// `r` is a live run handle (or null); `len` is a valid `size_t*` (or null).
#[no_mangle]
pub unsafe extern "C" fn svm_run_stderr(r: *const SvmRun, len: *mut usize) -> *const u8 {
    bytes_field(r, len, |run| &run.0.stderr)
}

unsafe fn bytes_field(
    r: *const SvmRun,
    len: *mut usize,
    pick: impl FnOnce(&SvmRun) -> &Vec<u8>,
) -> *const u8 {
    match r.as_ref() {
        Some(run) => {
            let v = pick(run);
            if !len.is_null() {
                *len = v.len();
            }
            v.as_ptr()
        }
        None => {
            if !len.is_null() {
                *len = 0;
            }
            ptr::null()
        }
    }
}

/// The outcome kind: [`SVM_OUTCOME_RETURNED`] or [`SVM_OUTCOME_EXITED`] (or `< 0` for a null handle).
///
/// # Safety
/// `r` is a live run handle from this library, or null.
#[no_mangle]
pub unsafe extern "C" fn svm_run_outcome_kind(r: *const SvmRun) -> i32 {
    match r.as_ref() {
        Some(run) => match run.0.outcome {
            Outcome::Returned(_) => SVM_OUTCOME_RETURNED,
            Outcome::Exited(_) => SVM_OUTCOME_EXITED,
        },
        None => -1,
    }
}

/// The exit code (valid when the outcome kind is [`SVM_OUTCOME_EXITED`]; else `0`).
///
/// # Safety
/// `r` is a live run handle from this library, or null.
#[no_mangle]
pub unsafe extern "C" fn svm_run_exit_code(r: *const SvmRun) -> i32 {
    match r.as_ref() {
        Some(run) => match run.0.outcome {
            Outcome::Exited(code) => code,
            Outcome::Returned(_) => 0,
        },
        None => 0,
    }
}

/// The number of returned result values (when the outcome kind is [`SVM_OUTCOME_RETURNED`]).
///
/// # Safety
/// `r` is a live run handle from this library, or null.
#[no_mangle]
pub unsafe extern "C" fn svm_run_result_count(r: *const SvmRun) -> usize {
    match r.as_ref() {
        Some(run) => match &run.0.outcome {
            Outcome::Returned(v) => v.len(),
            Outcome::Exited(_) => 0,
        },
        None => 0,
    }
}

/// The `idx`-th returned value as a raw `i64` slot (floats are bit-reinterpreted; `0` if out of range).
///
/// # Safety
/// `r` is a live run handle from this library, or null.
#[no_mangle]
pub unsafe extern "C" fn svm_run_result(r: *const SvmRun, idx: usize) -> i64 {
    match r.as_ref() {
        Some(run) => match &run.0.outcome {
            Outcome::Returned(v) => v.get(idx).map_or(0, value_slot),
            Outcome::Exited(_) => 0,
        },
        None => 0,
    }
}

/// Free a run handle.
///
/// # Safety
/// `r` is a live run handle from this library, or null.
#[no_mangle]
pub unsafe extern "C" fn svm_run_free(r: *mut SvmRun) {
    if !r.is_null() {
        drop(Box::from_raw(r));
    }
}

// ----------------------------------------------------------------------------
// Reactor sessions (Phase 6): instantiate once, call exports repeatedly with
// persistent window state.
// ----------------------------------------------------------------------------

/// A live, stateful reactor session (opaque) — the C view of [`svm_run::Session`]. Built by
/// `svm_instance_start`, freed by `svm_session_free`.
pub struct SvmSession(svm_run::Session);

/// Start a reactor session on `backend` under `config` (null ⇒ defaults): grant the powerbox once,
/// run the bootstrap, and keep the window + host live for repeated `svm_session_call_export` calls.
/// Does **not** consume `i` (the instance can start more sessions). Returns `NULL` on failure.
///
/// # Safety
/// `i` is a live instance handle; `config` is null or a valid `SvmRunConfig`.
#[no_mangle]
pub unsafe extern "C" fn svm_instance_start(
    i: *const SvmInstance,
    backend: i32,
    config: *const SvmRunConfig,
) -> *mut SvmSession {
    guard_ptr(|| {
        let inst = i.as_ref().ok_or("svm_instance_start: null instance")?;
        let backend = backend_of(backend)?;
        let cfg = run_config(config);
        let session = inst.0.start(backend, &cfg)?;
        Ok(Box::into_raw(Box::new(SvmSession(session))))
    })
}

/// Call exported function `name` with `n_args` `i64` arguments, writing up to `results_cap` `i64`
/// results into `results` and the actual count into `*n_results`. The window (globals, stash, BSS) and
/// capability handles persist from prior calls. Returns `SVM_OK`, or an error status (message in
/// `svm_last_error`). Arguments are passed as raw `i64` slots (interpreted as `i64` values — the
/// common case; floats can be passed by bit pattern).
///
/// # Safety
/// `s` is a live session; `name` a valid C string; `args`/`results` describe readable/writable
/// `n_args`/`results_cap` slots; `n_results` is a valid `size_t*` (or null).
#[no_mangle]
pub unsafe extern "C" fn svm_session_call_export(
    s: *mut SvmSession,
    name: *const c_char,
    args: *const i64,
    n_args: usize,
    results: *mut i64,
    results_cap: usize,
    n_results: *mut usize,
) -> i32 {
    guard_status(|| {
        let s = s.as_mut().ok_or("svm_session_call_export: null session")?;
        if name.is_null() {
            return Err("null export name".into());
        }
        let name = CStr::from_ptr(name)
            .to_str()
            .map_err(|_| "export name is not valid UTF-8".to_string())?;
        let arg_slots = if n_args == 0 {
            &[][..]
        } else {
            std::slice::from_raw_parts(args, n_args)
        };
        let vals: Vec<Value> = arg_slots.iter().map(|&x| Value::I64(x)).collect();
        let out = s.0.call_export(name, &vals)?;
        if !n_results.is_null() {
            *n_results = out.len();
        }
        if !results.is_null() && results_cap > 0 {
            let dst = std::slice::from_raw_parts_mut(results, results_cap);
            for (d, v) in dst.iter_mut().zip(&out) {
                *d = value_slot(v);
            }
        }
        Ok(())
    })
}

/// The session's captured stdout so far (valid until the next call / `svm_session_free`); writes `*len`.
///
/// # Safety
/// `s` is a live session (or null); `len` a valid `size_t*` (or null).
#[no_mangle]
pub unsafe extern "C" fn svm_session_stdout(s: *const SvmSession, len: *mut usize) -> *const u8 {
    match s.as_ref() {
        Some(sess) => {
            let out = sess.0.stdout();
            if !len.is_null() {
                *len = out.len();
            }
            out.as_ptr()
        }
        None => {
            if !len.is_null() {
                *len = 0;
            }
            ptr::null()
        }
    }
}

/// Free a session handle.
///
/// # Safety
/// `s` is a live session handle from this library, or null.
#[no_mangle]
pub unsafe extern "C" fn svm_session_free(s: *mut SvmSession) {
    if !s.is_null() {
        drop(Box::from_raw(s));
    }
}

#[cfg(test)]
mod abi_tests;
