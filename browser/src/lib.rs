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

use svm_interp::{bytecode, Value};

// ---- self-contained smoke probe (no host imports) --------------------------------------------

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
/// with the single `i64` argument `arg` and a deny-all `Host`, and return its first `i64` result
/// (`0` on any non-`OK` status — read [`svm_status`] to disambiguate).
#[no_mangle]
pub extern "C" fn svm_run(len: usize, arg: i64) -> i64 {
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
    match bytecode::compile_and_run_with_host(&m, 0, &[Value::I64(arg)], &mut fuel, &mut host) {
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
