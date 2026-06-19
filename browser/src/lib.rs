//! SVM **bytecode interpreter as a wasm guest** — the browser entry point (see `BROWSER.md`).
//!
//! Two exports for a wasm host (browser / any runtime):
//!   * [`run_guest`] — a self-contained, no-import smoke probe (an embedded compute kernel), used by
//!     the wasm32 anchors in `run.mjs`.
//!   * [`svm_run`] — the production shape: the host writes an **encoded SVM IR module** (the
//!     `svm-encode` binary form) into the scratch buffer at [`svm_buf`], then calls `svm_run(len,
//!     arg)`. We decode it, run function 0 on the **bytecode engine** with a **deny-all `Host`**
//!     (compute-only v1), and return its first `i64` result. **Fail-closed:** a module the engine
//!     can't compile yields `STATUS_UNSUPPORTED` rather than any tree-walker fallback.
//!
//! Status of the last [`svm_run`] is read separately via [`svm_status`] (a single `i64` return
//! can't disambiguate an error from a guest result of the same value).

use svm_interp::{bytecode, Host, StreamRole, Trap, Value};

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

/// Scratch buffer the host fills with the encoded guest module before calling [`svm_run`].
/// Fixed-capacity (v1); a real embedding would expose `alloc`/`dealloc` instead.
const BUF_CAP: usize = 1 << 20;
static mut BUF: [u8; BUF_CAP] = [0; BUF_CAP];
static mut LAST_STATUS: i32 = STATUS_OK;

/// Pointer to the scratch buffer (host writes `len` encoded bytes here, then calls [`svm_run`]).
#[no_mangle]
pub extern "C" fn svm_buf() -> *mut u8 {
    core::ptr::addr_of_mut!(BUF) as *mut u8
}

/// Capacity of the scratch buffer in bytes.
#[no_mangle]
pub extern "C" fn svm_buf_cap() -> usize {
    BUF_CAP
}

/// Status of the most recent [`svm_run`] (one of the `STATUS_*` codes).
#[no_mangle]
pub extern "C" fn svm_status() -> i32 {
    // SAFETY: single-threaded wasm; plain `i32` read.
    unsafe { LAST_STATUS }
}

/// Decode the `len` bytes at [`svm_buf`] as an SVM IR module, run function 0 on the bytecode engine
/// with `args` and a deny-all `Host`, and return its first `i64` result (`0` on any non-`OK` status
/// — read [`svm_status`] to disambiguate). Sets [`LAST_STATUS`].
fn run_buf(len: usize, args: &[Value]) -> i64 {
    // SAFETY: single-threaded wasm; `len` is bounded by the host to `<= svm_buf_cap()`.
    let bytes = unsafe { core::slice::from_raw_parts(core::ptr::addr_of!(BUF) as *const u8, len) };
    let set = |s: i32| unsafe { LAST_STATUS = s };

    let m = match svm_encode::decode_module(bytes) {
        Ok(m) => m,
        Err(_) => {
            set(STATUS_DECODE_ERR);
            return 0;
        }
    };
    let mut fuel = u64::MAX;
    let mut host = svm_interp::Host::new(); // deny-all powerbox (compute-only v1)
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

/// Run the encoded module at [`svm_buf`] passing a single `i64` argument (the common kernel shape).
#[no_mangle]
pub extern "C" fn svm_run(len: usize, arg: i64) -> i64 {
    run_buf(len, &[Value::I64(arg)])
}

/// Run the encoded module at [`svm_buf`] with **no** arguments — e.g. the `() -> (i64)` thread
/// kernels that spawn/join cooperatively on the engine's `drive`.
#[no_mangle]
pub extern "C" fn svm_run0(len: usize) -> i64 {
    run_buf(len, &[])
}

// ---- host powerbox: console + clock, all marshalled through buffers --------------------------
//
// Beyond compute-only: grant the guest a real capability set (stdin/stdout/stderr streams, a
// monotonic clock, and exit). The `Host` powerbox is already self-contained and **deterministic** —
// stream writes accumulate in `Host::stdout`/`stderr`, `read` draws from `Host::stdin`, and
// `Clock.now` is a strictly-increasing counter — so no wasm host *imports* are needed: I/O crosses
// the boundary the same way the module does, through fixed scratch buffers (`svm_stdin_buf` in,
// `svm_stdout_ptr`/`svm_stderr_ptr` out). The cdylib stays import-free; the embedder fills stdin
// before the call and reads the captured streams after.

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

/// Stdin scratch buffer (host writes `len` input bytes here, then sets the length via
/// [`svm_set_stdin_len`] before calling [`svm_run_pb`]).
static mut STDIN: [u8; BUF_CAP] = [0; BUF_CAP];
static mut STDIN_LEN: usize = 0;
/// Captured stdout / stderr of the most recent [`svm_run_pb`] (read via the `*_ptr`/`*_len`
/// exports). Fixed buffers (mirroring [`svm_buf`]) so the read-back ABI is a plain ptr+len and we
/// keep the `addr_of_mut!` pattern that avoids `&'static mut`.
static mut OUT: [u8; BUF_CAP] = [0; BUF_CAP];
static mut OUT_LEN: usize = 0;
static mut ERR: [u8; BUF_CAP] = [0; BUF_CAP];
static mut ERR_LEN: usize = 0;
static mut EXIT_CODE: i32 = 0;

/// Copy `src` (capped to `BUF_CAP`) into `dst`, returning the number of bytes written.
fn fill(dst: *mut u8, src: &[u8]) -> usize {
    let n = src.len().min(BUF_CAP);
    // SAFETY: `dst` is one of the fixed `BUF_CAP` capture buffers; `n <= BUF_CAP`.
    unsafe { core::ptr::copy_nonoverlapping(src.as_ptr(), dst, n) };
    n
}

/// Pointer to the stdin scratch buffer (host writes input bytes here).
#[no_mangle]
pub extern "C" fn svm_stdin_buf() -> *mut u8 {
    core::ptr::addr_of_mut!(STDIN) as *mut u8
}

/// Set how many bytes at [`svm_stdin_buf`] are valid stdin for the next [`svm_run_pb`].
#[no_mangle]
pub extern "C" fn svm_set_stdin_len(len: usize) {
    // SAFETY: single-threaded wasm; `len` is bounded by the host to `<= svm_buf_cap()`.
    unsafe { STDIN_LEN = len.min(BUF_CAP) }
}

/// Decode the `len` bytes at [`svm_buf`] and run function 0 under the **powerbox** (see
/// [`powerbox_exec`]): grant streams/clock/exit, seed stdin from [`svm_stdin_buf`], capture the
/// streams + exit code, and return the guest's `i64` result (`0` on any non-`OK`/`EXIT` status —
/// read [`svm_status`] / [`svm_exit_code`] / the `*_ptr` exports to disambiguate). Sets
/// [`LAST_STATUS`].
#[no_mangle]
pub extern "C" fn svm_run_pb(len: usize) -> i64 {
    // SAFETY: single-threaded wasm; `len`/`STDIN_LEN` are host-bounded to `<= svm_buf_cap()`.
    let bytes = unsafe { core::slice::from_raw_parts(core::ptr::addr_of!(BUF) as *const u8, len) };
    let stdin = unsafe {
        core::slice::from_raw_parts(core::ptr::addr_of!(STDIN) as *const u8, STDIN_LEN)
    };
    let set = |s: i32| unsafe { LAST_STATUS = s };
    let m = match svm_encode::decode_module(bytes) {
        Ok(m) => m,
        Err(_) => {
            set(STATUS_DECODE_ERR);
            return 0;
        }
    };
    let out = powerbox_exec(&m, stdin);
    set(out.status);
    let out_len = fill(core::ptr::addr_of_mut!(OUT) as *mut u8, &out.stdout);
    let err_len = fill(core::ptr::addr_of_mut!(ERR) as *mut u8, &out.stderr);
    // SAFETY: single-threaded wasm; read back only via the export accessors below.
    unsafe {
        OUT_LEN = out_len;
        ERR_LEN = err_len;
        EXIT_CODE = out.exit_code;
    }
    out.value
}

/// Pointer / length of the captured stdout from the most recent [`svm_run_pb`].
#[no_mangle]
pub extern "C" fn svm_stdout_ptr() -> *const u8 {
    core::ptr::addr_of!(OUT) as *const u8
}
#[no_mangle]
pub extern "C" fn svm_stdout_len() -> usize {
    unsafe { OUT_LEN }
}
/// Pointer / length of the captured stderr from the most recent [`svm_run_pb`].
#[no_mangle]
pub extern "C" fn svm_stderr_ptr() -> *const u8 {
    core::ptr::addr_of!(ERR) as *const u8
}
#[no_mangle]
pub extern "C" fn svm_stderr_len() -> usize {
    unsafe { ERR_LEN }
}
/// Exit code from the most recent [`svm_run_pb`] (valid when [`svm_status`] is [`STATUS_EXIT`]).
#[no_mangle]
pub extern "C" fn svm_exit_code() -> i32 {
    unsafe { EXIT_CODE }
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
