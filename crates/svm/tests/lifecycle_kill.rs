//! PROCESS.md S3 — `Instantiator.kill` (op 12): a parent terminates a §14 child's subtree. The child
//! (and its `thread.spawn` descendants, which share the flag) polls a parent-set kill flag once per
//! op; when set it traps (`ThreadFault`, which `poll` reports as `2`).
//!
//! The test is **race-free and discriminating**, not a fuel-exhaustion false positive: the child spins
//! until a go-byte the parent sets *after* `kill`, then would return `7`. Because the kill flag is set
//! **before** the go-byte and is checked every op, a working kill traps the child before it can ever
//! observe the go-byte — so `poll` reports `2` (killed). If kill did nothing, the child would see the
//! go-byte and return `7` → `poll` `1`; the assertion `== 2` fails fast (no hang, no slow fuel trap).

use svm_interp::{run_capture_reserved_with_host, Host, Value};
use svm_text::parse_module;
use svm_verify::verify_module;

/// func 0 (parent): spawn func 1 in a 4 KiB carve at offset 0; `kill` it; set the go-byte (window 0)
/// to 1; then a yielding `poll` loop until the child is no longer running, `detach`, and return the
/// final poll status (must be `2` = trapped/killed).
///
/// func 1 (child): spin until window byte 0 is 1, then return 7 — but a live kill flag traps it first.
const SRC: &str = "memory 17\n\
func (i32) -> (i64) {\n\
block 0 (v0: i32) {\n\
  ventry = i64.const 1\n\
  voff = i64.const 0\n\
  vsl = i64.const 12\n\
  vq = i64.const 0\n\
  vch = cap.call 6 0 (i64, i64, i64, i64) -> (i32) v0 (ventry, voff, vsl, vq)\n\
  vk = cap.call 6 12 (i32) -> (i32) v0 (vch)\n\
  bz = i64.const 0\n\
  b1 = i32.const 1\n\
  i32.store8 bz b1\n\
  br 1(v0, vch)\n\
}\n\
block 1 (v0a: i32, vcha: i32) {\n\
  vp = cap.call 6 9 (i32) -> (i32) v0a (vcha)\n\
  vzero = i32.const 0\n\
  vne = i32.ne vp vzero\n\
  br_if vne 3(v0a, vcha, vp) 2(v0a, vcha)\n\
}\n\
block 2 (v0b: i32, vchb: i32) {\n\
  v8192 = i64.const 8192\n\
  vexp = i32.const 0\n\
  vto = i64.const 100000\n\
  vy = i32.atomic.wait v8192 vexp vto\n\
  br 1(v0b, vchb)\n\
}\n\
block 3 (v0c: i32, vchc: i32, vpf: i32) {\n\
  vd = cap.call 6 10 (i32) -> (i32) v0c (vchc)\n\
  vpf64 = i64.extend_i32_u vpf\n\
  return vpf64\n\
  }\n\
}\n\
func (i64) -> (i64) {\n\
block 0 (vci: i64) {\n\
  br 1()\n\
}\n\
block 1 () {\n\
  bz = i64.const 0\n\
  vb = i32.load8_u bz\n\
  vone = i32.const 1\n\
  veq = i32.eq vb vone\n\
  br_if veq 2() 1()\n\
}\n\
block 2 () {\n\
  v7 = i64.const 7\n\
  return v7\n\
  }\n\
}\n";

#[test]
fn kill_traps_the_child_before_it_can_return() {
    let m = parse_module(SRC).expect("parse");
    verify_module(&m).expect("verify");
    let mut host = Host::new();
    let ih = host.grant_instantiator(0, 128 << 10);
    let mut fuel = 50_000_000u64;
    let (res, _snap) = run_capture_reserved_with_host(
        &m,
        0,
        &[Value::I32(ih)],
        &mut fuel,
        &[0u8; 128 << 10],
        0,
        &mut host,
    );
    // 2 = poll saw the child trapped (killed). A broken kill would let the child return 7 → poll 1.
    assert_eq!(res, Ok(vec![Value::I64(2)]));
}
