//! PROCESS.md S3 — §14 child lifecycle ops `poll` (9) / `detach` (10) / `kill` (12) on the **JIT**.
//!
//! A JIT child runs synchronously at `instantiate`, so by the time the parent acts the child is
//! already *done*: `poll` reports `1`/`2` (never `0`, the running state — interp-only until JIT async
//! children, S1c), and `kill`/`detach` of a finished child are harmless successes returning `0`.
//!
//! - `kill_detach_match_interp` is a **cross-backend differential**: `instantiate` → `kill` → `detach`
//!   returns `0` on both engines (no futex/loop, so it stays on the nesting compile path — a program
//!   mixing §14 nesting with §12 `atomic.wait` is a separate, unsupported combination on the JIT).
//! - `jit_poll_of_done_child_is_returned` pins the JIT `poll` value (`1`) for a completed child; its
//!   *cross-backend* value can't be asserted (the interp child may still be running at poll time —
//!   sync vs async), so poll's interp semantics live in `lifecycle_poll_detach.rs`.

use svm_interp::{run_capture_reserved_with_host, Host, Value};
use svm_jit::{compile_and_run_capture_reserved_with_host, JitOutcome};
use svm_text::parse_module;
use svm_verify::verify_module;

fn run_interp(src: &str) -> Result<Vec<Value>, svm_interp::Trap> {
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");
    let mut h = Host::new();
    let ih = h.grant_instantiator(0, 128 << 10);
    let mut fuel = 50_000_000u64;
    run_capture_reserved_with_host(
        &m,
        0,
        &[Value::I32(ih)],
        &mut fuel,
        &[0u8; 128 << 10],
        0,
        &mut h,
    )
    .0
}

fn run_jit(src: &str) -> JitOutcome {
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");
    let mut h = Host::new();
    let jh = h.grant_instantiator(0, 128 << 10);
    compile_and_run_capture_reserved_with_host(
        &m,
        0,
        &[jh as i64],
        &[0u8; 128 << 10],
        0,
        svm_run::cap_thunk,
        &mut h as *mut Host as *mut core::ffi::c_void,
    )
    .expect("jit")
    .0
}

/// `instantiate` a child (returns 7), then `kill` and `detach` it; return `kill_status*10 +
/// detach_status` = `0` (both are harmless successes on a finished child). No `poll` loop / futex, so
/// the result is backend-stable: `0` on the interpreter (kill flags the child, detach drops the claim)
/// and `0` on the JIT (the child already ran synchronously).
const KILL_DETACH: &str = "memory 17\n\
func (i32) -> (i64) {\n\
block0(v0: i32):\n\
  ventry = i64.const 1\n\
  voff = i64.const 0\n\
  vsl = i64.const 12\n\
  vq = i64.const 0\n\
  vch = cap.call 6 0 (i64, i64, i64, i64) -> (i32) v0 (ventry, voff, vsl, vq)\n\
  vk = cap.call 6 12 (i32) -> (i32) v0 (vch)\n\
  vd = cap.call 6 10 (i32) -> (i32) v0 (vch)\n\
  vten = i32.const 10\n\
  vkm = i32.mul vk vten\n\
  vsum = i32.add vkm vd\n\
  vr = i64.extend_i32_u vsum\n\
  return vr\n\
}\n\
func (i64) -> (i64) {\n\
block0(vci: i64):\n\
  v7 = i64.const 7\n\
  return v7\n\
}\n";

/// `instantiate` a child, `poll` it once, return the status. On the JIT the child is already done, so
/// `poll` is `1` (returned).
const POLL_DONE: &str = "memory 17\n\
func (i32) -> (i64) {\n\
block0(v0: i32):\n\
  ventry = i64.const 1\n\
  voff = i64.const 0\n\
  vsl = i64.const 12\n\
  vq = i64.const 0\n\
  vch = cap.call 6 0 (i64, i64, i64, i64) -> (i32) v0 (ventry, voff, vsl, vq)\n\
  vp = cap.call 6 9 (i32) -> (i32) v0 (vch)\n\
  vr = i64.extend_i32_u vp\n\
  return vr\n\
}\n\
func (i64) -> (i64) {\n\
block0(vci: i64):\n\
  v7 = i64.const 7\n\
  return v7\n\
}\n";

#[test]
fn kill_detach_match_interp() {
    let ir = run_interp(KILL_DETACH);
    let jo = run_jit(KILL_DETACH);
    assert_eq!(ir, Ok(vec![Value::I64(0)]), "interp: kill+detach both 0");
    assert!(
        matches!(jo, JitOutcome::Returned(ref s) if s == &[0]),
        "jit: kill+detach must match interp (0), got {jo:?}"
    );
}

#[test]
fn jit_poll_of_done_child_is_returned() {
    let jo = run_jit(POLL_DONE);
    // The JIT child ran synchronously at instantiate, so poll reports 1 (returned).
    assert!(
        matches!(jo, JitOutcome::Returned(ref s) if s == &[1]),
        "jit: poll of a finished child must be 1 (returned), got {jo:?}"
    );
}
