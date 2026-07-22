//! Differential + confinement tests for the bulk-memory ops `mem.copy`/`mem.move`/`mem.fill` (D62).
//!
//! The three engines — tree-walker (`svm_interp::run`), bytecode interp
//! (`svm_interp::bytecode::compile_and_run`), and the JIT (`svm_jit::compile_and_run`, which lowers
//! these to the platform `memcpy`/`memmove`/`memset` libcall behind a single span range-check) — must
//! agree on every case: the same returned value, and the same `MemoryFault` when a span escapes the
//! window `[0, reserved)`. That equivalence is the whole point of the op (§18 escape oracle).
#![cfg(any(
    all(unix, target_arch = "x86_64"),
    all(unix, target_arch = "aarch64"),
    all(windows, target_arch = "x86_64")
))]

use svm_interp::{bytecode, Value};
use svm_jit::{compile_and_run, JitOutcome};
use svm_text::parse_module;

/// Run `func 0` (no args) on all three engines; return `Ok(ret0)` if they agree on a returned value,
/// `Err(())` if they agree it traps. Panics if the engines disagree (a differential miscompile).
fn run_all(src: &str) -> Result<i64, ()> {
    let m = parse_module(src).expect("parse");
    svm_verify::verify_module(&m).expect("verify");

    let mut fuel = u64::MAX;
    let tw = svm_interp::run(&m, 0, &[], &mut fuel);
    let mut fuel = u64::MAX;
    let bc = bytecode::compile_and_run(&m, 0, &[], &mut fuel).expect("bytecode in-subset");
    let jit = compile_and_run(&m, 0, &[]).expect("jit compile/run");

    let tw = tw.map(|v| as_i64(v[0]));
    let bc = bc.map(|v| as_i64(v[0]));
    let jit = match jit {
        JitOutcome::Returned(v) => Ok(v[0]),
        JitOutcome::Trapped(_) => Err(()),
        other => panic!("unexpected jit outcome: {other:?}"),
    };
    // Normalize traps to `Err(())` (the trap *kind* is checked separately where it matters).
    let norm = |r: Result<i64, svm_interp::Trap>| r.map_err(|_| ());
    let (tw, bc) = (norm(tw), norm(bc));
    assert_eq!(tw, bc, "tree-walker vs bytecode disagree");
    assert_eq!(tw, jit, "interp vs jit disagree");
    tw
}

fn as_i64(v: Value) -> i64 {
    match v {
        Value::I64(x) => x,
        Value::I32(x) => x as i64,
        other => panic!("unexpected {other:?}"),
    }
}

/// `mem.fill` then `mem.copy`: fill `[0,16)` with 0xAB, copy it to `[100,116)`, read back an i64.
#[test]
fn fill_then_copy() {
    let src = "\
memory 16
func () -> (i64) {
block 0 () {
  v0 = i64.const 0
  v1 = i32.const 171
  v2 = i64.const 16
  mem.fill v0 v1 v2
  v3 = i64.const 100
  v4 = i64.const 0
  v5 = i64.const 16
  mem.copy v3 v4 v5
  v6 = i64.const 100
  v7 = i64.load v6
  return v7
  }
}
";
    // 0xABABABABABABABAB reinterpreted as i64.
    assert_eq!(run_all(src), Ok(0xABAB_ABAB_ABAB_ABABu64 as i64));
}

/// `mem.move` with **overlapping** spans (dst = src+8, forward overlap): must be overlap-safe (read
/// the whole source before writing). Fill `[0,24)` with 0x01, then move `[0,16)` to `[8,24)`.
#[test]
fn overlapping_move() {
    let src = "\
memory 16
func () -> (i64) {
block 0 () {
  v0 = i64.const 0
  v1 = i32.const 1
  v2 = i64.const 24
  mem.fill v0 v1 v2
  v3 = i64.const 8
  v4 = i64.const 0
  v5 = i64.const 16
  mem.move v3 v4 v5
  v6 = i64.const 16
  v7 = i64.load v6
  return v7
  }
}
";
    // Every byte in [0,24) was 0x01, so after the move [16,24) is still 0x0101010101010101.
    assert_eq!(run_all(src), Ok(0x0101_0101_0101_0101));
}

/// A zero-length copy of a **wild** (out-of-window) pointer is a no-op, never a fault — matching C
/// `memcpy(_,_,0)` and the JIT's `len != 0`-guarded check.
#[test]
fn zero_length_wild_pointer_is_noop() {
    let src = "\
memory 16
func () -> (i64) {
block 0 () {
  v0 = i64.const 999999999
  v1 = i64.const 888888888
  v2 = i64.const 0
  mem.copy v0 v1 v2
  v3 = i64.const 42
  return v3
  }
}
";
    assert_eq!(run_all(src), Ok(42));
}

/// A `mem.copy` whose destination span runs past the window `[0, 1<<16)` must fault `MemoryFault` —
/// on every engine (checked via `run_all` agreeing on the trap).
#[test]
fn copy_past_window_faults() {
    let src = "\
memory 16
func () -> (i64) {
block 0 () {
  v0 = i64.const 65528
  v1 = i64.const 0
  v2 = i64.const 32
  mem.copy v0 v1 v2
  v3 = i64.const 0
  return v3
  }
}
";
    assert_eq!(run_all(src), Err(()));
}

/// A `mem.fill` whose length exceeds the whole reservation must fault (guards the `len > reserved`
/// sub-check that stops the `reserved - len` subtraction from wrapping).
#[test]
fn fill_oversized_length_faults() {
    let src = "\
memory 16
func () -> (i64) {
block 0 () {
  v0 = i64.const 0
  v1 = i32.const 0
  v2 = i64.const 4294967296
  mem.fill v0 v1 v2
  v3 = i64.const 0
  return v3
  }
}
";
    assert_eq!(run_all(src), Err(()));
}

// --- ISSUES.md I21 regressions: bulk spans overrunning `mapped` inside the reservation ---
//
// `memory 16` backs `mapped = 64 KiB` while the JIT reserves a far larger window, so a span past
// 64 KiB lands in the `(mapped, reserved]` guard hole. Before the `probe_span` fix the JIT (1) lost
// the trap on a `dst == src` self-copy (libc short-circuits it) and (2) partial-wrote before the
// libcall faulted — both interp↔JIT divergences `run_all` panics on. Each case must now agree on a
// fault across all three engines.

/// I21(1): a `dst == src` self-copy whose span overruns `mapped` must fault (not silently return).
/// The pre-fix JIT returned here while both interpreters trapped.
#[test]
fn self_copy_overrunning_mapped_faults() {
    let src = "\
memory 16
func () -> (i64) {
block 0 () {
  v0 = i64.const 0
  v1 = i64.const 65537
  mem.copy v0 v0 v1
  v2 = i64.const 0
  return v2
  }
}
";
    assert_eq!(run_all(src), Err(()));
}

/// I21: same, for `mem.move` (also libc-short-circuited on `dst == src`).
#[test]
fn self_move_overrunning_mapped_faults() {
    let src = "\
memory 16
func () -> (i64) {
block 0 () {
  v0 = i64.const 0
  v1 = i64.const 65537
  mem.move v0 v0 v1
  v2 = i64.const 0
  return v2
  }
}
";
    assert_eq!(run_all(src), Err(()));
}

/// I21: a copy whose `src` span starts in-window but overruns `mapped` faults (distinct-pointer
/// case — guards the read side of the span probe).
#[test]
fn copy_src_overrunning_mapped_faults() {
    let src = "\
memory 16
func () -> (i64) {
block 0 () {
  v0 = i64.const 0
  v1 = i64.const 63000
  v2 = i64.const 4096
  mem.copy v0 v1 v2
  v3 = i64.const 0
  return v3
  }
}
";
    assert_eq!(run_all(src), Err(()));
}

/// I21: a `mem.fill` overrunning `mapped` (but within the reservation) faults up-front. The pre-fix
/// JIT faulted only after partial-writing the in-`mapped` prefix.
#[test]
fn fill_overrunning_mapped_faults() {
    let src = "\
memory 16
func () -> (i64) {
block 0 () {
  v0 = i64.const 63000
  v1 = i32.const 65
  v2 = i64.const 4096
  mem.fill v0 v1 v2
  v3 = i64.const 0
  return v3
  }
}
";
    assert_eq!(run_all(src), Err(()));
}
