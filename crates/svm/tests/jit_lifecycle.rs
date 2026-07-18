//! PROCESS.md S3/S1c — §14 child lifecycle ops `poll` (9) / `detach` (10) / `kill` (12) on the **JIT**.
//!
//! A non-durable JIT child now runs on its **own OS thread** (S1c async children), so `poll` reports
//! the live state: `0` (running) while its thread is still executing, then `1` (returned) / `2`
//! (trapped) once it finishes; `kill`/`detach` of a child are harmless successes returning `0`. (The
//! child's thread is joined at run teardown either way.)
//!
//! - `kill_detach_match_interp` is a **cross-backend differential**: `instantiate` → `kill` → `detach`
//!   returns `0` on both engines (no futex/loop, so it stays on the nesting compile path — a program
//!   mixing §14 nesting with §12 `atomic.wait` is a separate, unsupported combination on the JIT).
//! - `jit_poll_reports_child_done` spins `poll` until the async child finishes and pins the terminal
//!   value (`1`, returned) — exercising the running→done transition the OS-thread child now goes
//!   through; poll's interp semantics live in `lifecycle_poll_detach.rs`.

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

/// `instantiate` a child (returns 7), then **spin `poll` until it finishes** and return the terminal
/// status. The async child runs on its own OS thread, so early polls may report `0` (running); once its
/// thread completes, `poll` reports `1` (returned) and the loop exits. Deterministic (the child always
/// finishes) and exercises the running→done transition of an OS-thread child.
const POLL_DONE: &str = "memory 17\n\
func (i32) -> (i64) {\n\
block0(v0: i32):\n\
  ventry = i64.const 1\n\
  voff = i64.const 0\n\
  vsl = i64.const 12\n\
  vq = i64.const 0\n\
  vch = cap.call 6 0 (i64, i64, i64, i64) -> (i32) v0 (ventry, voff, vsl, vq)\n\
  br block1(v0, vch)\n\
block1(bv0: i32, bvch: i32):\n\
  vp = cap.call 6 9 (i32) -> (i32) bv0 (bvch)\n\
  vzero = i32.const 0\n\
  vrun = i32.eq vp vzero\n\
  br_if vrun block1(bv0, bvch) block2(vp)\n\
block2(vpf: i32):\n\
  vr = i64.extend_i32_u vpf\n\
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
fn jit_poll_reports_child_done() {
    let jo = run_jit(POLL_DONE);
    // The async child runs on its own OS thread; the guest spins `poll` until it finishes, so the
    // terminal value is 1 (returned).
    assert!(
        matches!(jo, JitOutcome::Returned(ref s) if s == &[1]),
        "jit: poll must reach 1 (returned) once the async child finishes, got {jo:?}"
    );
}

/// **Concurrency proof** (S1c): the child runs a long bounded loop before returning `7`, so when the
/// parent — running *concurrently on its own thread* — `poll`s it immediately after `instantiate`, the
/// child is still running (`poll` = 0). The parent records that first poll, then spins `poll` to
/// completion (`1`) and returns `first*10 + final`. The async OS-thread executor yields `0*10 + 1 = 1`;
/// a **synchronous** `instantiate` (child fully run before it returns) would see the child already done
/// at the first poll → `1*10 + 1 = 11`. So `== 1` is a deterministic witness that the child executed
/// concurrently with the parent — the whole point of async children (the substrate for a pipeline).
const POLL_RUNNING: &str = "memory 17\n\
func (i32) -> (i64) {\n\
block0(v0: i32):\n\
  ventry = i64.const 1\n\
  voff = i64.const 0\n\
  vsl = i64.const 12\n\
  vq = i64.const 0\n\
  vch = cap.call 6 0 (i64, i64, i64, i64) -> (i32) v0 (ventry, voff, vsl, vq)\n\
  vfirst = cap.call 6 9 (i32) -> (i32) v0 (vch)\n\
  br block1(v0, vch, vfirst)\n\
block1(bv0: i32, bvch: i32, bfirst: i32):\n\
  vp = cap.call 6 9 (i32) -> (i32) bv0 (bvch)\n\
  vzero = i32.const 0\n\
  vrun = i32.eq vp vzero\n\
  br_if vrun block1(bv0, bvch, bfirst) block2(bfirst, vp)\n\
block2(bf: i32, vfin: i32):\n\
  vten = i32.const 10\n\
  vfm = i32.mul bf vten\n\
  vsum = i32.add vfm vfin\n\
  vr = i64.extend_i32_u vsum\n\
  return vr\n\
}\n\
func (i64) -> (i64) {\n\
block0(vci: i64):\n\
  vz = i64.const 0\n\
  br block1(vz)\n\
block1(i: i64):\n\
  vlim = i64.const 20000000\n\
  vlt = i64.lt_u i vlim\n\
  vinc = i64.const 1\n\
  vnext = i64.add i vinc\n\
  br_if vlt block1(vnext) block2()\n\
block2():\n\
  v7 = i64.const 7\n\
  return v7\n\
}\n";

#[test]
fn jit_poll_observes_a_concurrently_running_child() {
    let jo = run_jit(POLL_RUNNING);
    assert!(
        matches!(jo, JitOutcome::Returned(ref s) if s == &[1]),
        "jit: the first poll must see the child still running (0) — proof it runs concurrently on its \
         own thread (async); got {jo:?} (11 would mean the child ran synchronously before instantiate \
         returned)"
    );
}
