//! PROCESS.md S3 — §14 child **lifecycle** ops `poll` (op 9) and `detach` (op 10) on the interpreter.
//!
//! - `poll(child) -> 0 running | 1 returned | 2 trapped` — never parks; the reap probe a shell loops
//!   for `WNOHANG`/`SIGCHLD`. Non-destructive: a later `join` still delivers the result.
//! - `detach(child) -> 0` — drop the parent's join claim; the child runs to completion on its own
//!   (detach is not kill), the parent never blocks on it.
//!
//! (`kill` needs a per-child §5 interrupt on the M:N executor — a follow-up. Interp-first, like the
//! rest of the §14 substrate.)

use svm_interp::{run_capture_reserved_with_host, Host, Trap, Value};
use svm_text::parse_module;
use svm_verify::verify_module;

/// Run `src`'s entry 0 with an `Instantiator` over the whole 128 KiB window.
fn run(src: &str) -> Result<Vec<Value>, Trap> {
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");
    let mut host = Host::new();
    let ih = host.grant_instantiator(0, 128 << 10);
    let mut fuel = 50_000_000u64;
    run_capture_reserved_with_host(
        &m,
        0,
        &[Value::I32(ih)],
        &mut fuel,
        &[0u8; 128 << 10],
        0,
        &mut host,
    )
    .0
}

/// `poll` of a **blocked** child is `0` (running); `detach` lets the parent finish without joining.
///
/// func 0 (parent): spawn func 1 in a 4 KiB carve at offset 0; `poll` it (must be `0` — the child
/// spins until window byte 0 becomes 1, which is still 0, so it cannot have finished); `detach` it;
/// set byte 0 = 1 to release it; return the poll status. The child then runs to completion detached —
/// if `detach` blocked or the run waited wrong, this would hang.
const POLL_RUNNING_THEN_DETACH: &str = "memory 17\n\
func (i32) -> (i64) {\n\
block0(v0: i32):\n\
  ventry = i64.const 1\n\
  voff = i64.const 0\n\
  vsl = i64.const 12\n\
  vq = i64.const 0\n\
  vch = cap.call 6 0 (i64, i64, i64, i64) -> (i32) v0 (ventry, voff, vsl, vq)\n\
  vp = cap.call 6 9 (i32) -> (i32) v0 (vch)\n\
  vd = cap.call 6 10 (i32) -> (i32) v0 (vch)\n\
  vz = i64.const 0\n\
  vone = i32.const 1\n\
  i32.store8 vz vone\n\
  vp64 = i64.extend_i32_u vp\n\
  return vp64\n\
}\n\
func (i64) -> (i64) {\n\
block0(vci: i64):\n\
  br block1()\n\
block1():\n\
  vz = i64.const 0\n\
  vb = i32.load8_u vz\n\
  v1 = i32.const 1\n\
  veq = i32.eq vb v1\n\
  br_if veq block2() block1()\n\
block2():\n\
  v7 = i64.const 7\n\
  return v7\n\
}\n";

/// `poll` reaches `1` (returned) for a child that finishes. The parent polls in a loop, **yielding**
/// the worker between probes with a short `atomic.wait` on an anonymous byte (so a single-worker pool
/// schedules the child instead of spinning forever); once `poll != 0` it `join`s and returns the poll
/// status, which must be `1`.
const POLL_RETURNED: &str = "memory 17\n\
func (i32) -> (i64) {\n\
block0(v0: i32):\n\
  ventry = i64.const 1\n\
  voff = i64.const 0\n\
  vsl = i64.const 12\n\
  vq = i64.const 0\n\
  vch = cap.call 6 0 (i64, i64, i64, i64) -> (i32) v0 (ventry, voff, vsl, vq)\n\
  br block1(v0, vch)\n\
block1(v0a: i32, vcha: i32):\n\
  vp = cap.call 6 9 (i32) -> (i32) v0a (vcha)\n\
  vz32 = i32.const 0\n\
  vne = i32.ne vp vz32\n\
  br_if vne block2(v0a, vcha, vp) block3(v0a, vcha)\n\
block3(v0b: i32, vchb: i32):\n\
  v8192 = i64.const 8192\n\
  vexp = i32.const 0\n\
  vto = i64.const 100000\n\
  vy = i32.atomic.wait v8192 vexp vto\n\
  br block1(v0b, vchb)\n\
block2(v0c: i32, vchc: i32, vpf: i32):\n\
  vjr = cap.call 6 1 (i32) -> (i64) v0c (vchc)\n\
  vpf64 = i64.extend_i32_u vpf\n\
  return vpf64\n\
}\n\
func (i64) -> (i64) {\n\
block0(vci: i64):\n\
  v7 = i64.const 7\n\
  return v7\n\
}\n";

#[test]
fn poll_running_is_zero_and_detach_does_not_block() {
    // The parent must observe the still-blocked child as running (0), detach it, and finish — with
    // the detached child completing on its own (no hang).
    assert_eq!(run(POLL_RUNNING_THEN_DETACH), Ok(vec![Value::I64(0)]));
}

#[test]
fn poll_reaches_returned() {
    // The yielding reap loop must see the child transition to returned (1), then join it cleanly.
    assert_eq!(run(POLL_RETURNED), Ok(vec![Value::I64(1)]));
}
