//! §12 real-thread vCPUs (`thread.spawn` / `thread.join`): a guest program starts another vCPU on a
//! real OS thread, sharing one guest memory image. These run on the **interpreter** only (the JIT
//! reports the ops `Unsupported`, like fibers), so there is no differential/escape-oracle pairing —
//! the assertions pin the interpreter's behaviour directly. The shared-memory soundness underneath
//! is separately proven under ThreadSanitizer in `svm-mem`.

use svm_encode::{decode_module, encode_module};
use svm_interp::{run, run_capture, Trap, Value};
use svm_text::{parse_module, print_module};
use svm_verify::verify_module;

/// Parse + verify a text module (every program here is well-formed and must verify).
fn module(src: &str) -> svm_ir::Module {
    let m = parse_module(src).unwrap_or_else(|e| panic!("parse failed: {e:?}\n{src}"));
    verify_module(&m).unwrap_or_else(|e| panic!("verify failed: {e:?}\n{src}"));
    m
}

fn run_i64(src: &str) -> Result<i64, Trap> {
    let m = module(src);
    let mut fuel = 10_000_000u64;
    match run(&m, 0, &[], &mut fuel) {
        Ok(vals) => match vals.as_slice() {
            [Value::I64(v)] => Ok(*v),
            other => panic!("expected one i64 result, got {other:?}"),
        },
        Err(t) => Err(t),
    }
}

/// The headline: a spawned vCPU writes a sentinel into shared memory; the parent `thread.join`s it
/// (which orders the child's write before the read) and reads the *same* bytes back. If memory were
/// not shared across the OS threads, the parent would read 0.
#[test]
fn spawned_thread_shares_memory() {
    // func 1: the thread body `(i64 arg) -> (i64)` — store `arg` at mem[0], return `arg`.
    // func 0: spawn func 1 with 0xABCD, join, then atomically read mem[0] and return it.
    let src = r#"
memory 16
func () -> (i64) {
block0():
  v0 = i64.const 43981
  v1 = thread.spawn 1 v0
  v2 = thread.join v1
  v3 = i64.const 0
  v4 = i64.atomic.load v3
  return v4
}
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const 0
  i64.atomic.store v1 v0
  return v0
}
"#;
    assert_eq!(run_i64(src), Ok(43981));
}

/// `thread.join` yields the spawned vCPU's own return value (the `(i64) -> i64` result), distinct
/// from anything it wrote to memory.
#[test]
fn join_returns_thread_result() {
    // Thread body returns arg*3; parent returns the joined result directly (no memory involved).
    let src = r#"
memory 16
func () -> (i64) {
block0():
  v0 = i64.const 7
  v1 = thread.spawn 1 v0
  v2 = thread.join v1
  return v2
}
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const 3
  v2 = i64.mul v0 v1
  return v2
}
"#;
    assert_eq!(run_i64(src), Ok(21));
}

/// Many vCPUs each `fetch_add` a shared counter; after joining all of them the counter is the exact
/// sum — real cross-thread atomicity over the shared region, driven from guest code (cf. the
/// `svm-mem` TSan test, here through the whole spawn/join/atomic pipeline).
#[test]
fn concurrent_atomic_increments_sum_exactly() {
    // func 1: loop `arg` times doing i64.atomic.rmw.add of 1 at mem[0]; return 0.
    // func 0: spawn 4 workers each adding 1000, join all, read the counter.
    let src = r#"
memory 16
func () -> (i64) {
block0():
  v0 = i64.const 1000
  v1 = thread.spawn 1 v0
  v2 = thread.spawn 1 v0
  v3 = thread.spawn 1 v0
  v4 = thread.spawn 1 v0
  v5 = thread.join v1
  v6 = thread.join v2
  v7 = thread.join v3
  v8 = thread.join v4
  v9 = i64.const 0
  v10 = i64.atomic.load v9
  return v10
}
func (i64) -> (i64) {
block0(v0: i64):
  br block1(v0)
block1(v1: i64):
  v2 = i64.const 0
  v3 = i64.const 1
  v4 = i64.atomic.rmw.add v2 v3
  v5 = i64.const -1
  v6 = i64.add v1 v5
  v7 = i64.const 0
  v8 = i64.ne v6 v7
  br_if v8 block1(v6) block2()
block2():
  v9 = i64.const 0
  return v9
}
"#;
    assert_eq!(run_i64(src), Ok(4000));
}

/// A forged / never-spawned thread handle is **inert**: `thread.join` traps (`ThreadFault`) rather
/// than doing anything — the handle is masked + bounds/liveness-checked like a fiber/capability
/// handle (§3c), so it can never reach another vCPU's state.
#[test]
fn forged_join_handle_is_inert() {
    let src = r#"
memory 16
func () -> (i64) {
block0():
  v0 = i64.const 99
  v1 = thread.spawn 1 v0
  v2 = thread.join v1
  v3 = i32.const 1234567
  v4 = thread.join v3
  return v2
}
func (i64) -> (i64) {
block0(v0: i64):
  return v0
}
"#;
    assert_eq!(run_i64(src), Err(Trap::ThreadFault));
}

/// Joining the same handle twice is inert on the second join (the slot is consumed) — a re-join
/// can't observe or double-free another vCPU.
#[test]
fn double_join_is_inert() {
    let src = r#"
memory 16
func () -> (i64) {
block0():
  v0 = i64.const 5
  v1 = thread.spawn 1 v0
  v2 = thread.join v1
  v3 = thread.join v1
  return v2
}
func (i64) -> (i64) {
block0(v0: i64):
  return v0
}
"#;
    assert_eq!(run_i64(src), Err(Trap::ThreadFault));
}

/// A trap inside a spawned vCPU propagates out of the `thread.join` (a misaligned atomic in the
/// child — a `MemoryFault` — surfaces as the parent's result).
#[test]
fn child_trap_propagates_through_join() {
    // Thread body does a misaligned i64 atomic store (addr 1) ⇒ MemoryFault in the child.
    let src = r#"
memory 16
func () -> (i64) {
block0():
  v0 = i64.const 0
  v1 = thread.spawn 1 v0
  v2 = thread.join v1
  return v2
}
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const 1
  i64.atomic.store v1 v0
  return v0
}
"#;
    assert_eq!(run_i64(src), Err(Trap::MemoryFault));
}

/// The final-memory snapshot (`run_capture`) reflects a spawned vCPU's writes once it is joined —
/// the escape-oracle capture path sees shared-thread effects too.
#[test]
fn capture_reflects_thread_writes() {
    let src = r#"
memory 16
func () -> (i64) {
block0():
  v0 = i64.const 255
  v1 = thread.spawn 1 v0
  v2 = thread.join v1
  return v2
}
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const 8
  i64.atomic.store v1 v0
  return v0
}
"#;
    let m = module(src);
    let mut fuel = 10_000_000u64;
    let (res, snap) = run_capture(&m, 0, &[], &mut fuel, &[0u8; 64]);
    assert_eq!(res, Ok(vec![Value::I64(255)]));
    // The child stored the i64 255 at offset 8 (little-endian low byte 0xFF).
    assert_eq!(snap[8], 0xFF);
    assert_eq!(snap[9], 0x00);
}

/// The verifier rejects a `thread.spawn` whose target isn't the fixed thread entry type
/// `(i64) -> i64` — here func 1 is `(i32) -> i32`.
#[test]
fn verify_rejects_bad_thread_entry_signature() {
    let src = r#"
memory 16
func () -> (i64) {
block0():
  v0 = i64.const 1
  v1 = thread.spawn 1 v0
  v2 = thread.join v1
  return v2
}
func (i32) -> (i32) {
block0(v0: i32):
  return v0
}
"#;
    let m = parse_module(src).expect("parse");
    let err = verify_module(&m).expect_err("a non-(i64)->i64 thread entry must be rejected");
    assert!(
        matches!(err, svm_verify::VerifyError::ThreadEntrySignature { .. }),
        "unexpected verify error: {err:?}"
    );
}

/// `thread.spawn`/`thread.join` survive both round-trips (text print↔parse and binary
/// encode↔decode) unchanged — the new ops are wired through the whole pipeline, not just the interp.
#[test]
fn thread_ops_round_trip() {
    let src = r#"
memory 16
func () -> (i64) {
block0():
  v0 = i64.const 1
  v1 = thread.spawn 1 v0
  v2 = thread.join v1
  return v2
}
func (i64) -> (i64) {
block0(v0: i64):
  return v0
}
"#;
    let m = module(src);
    // Binary: encode → decode is identity.
    assert_eq!(decode_module(&encode_module(&m)).expect("decode"), m);
    // Text: print → parse is identity.
    assert_eq!(parse_module(&print_module(&m)).expect("reparse"), m);
}
