//! Concurrent **property** tests (§12/§18) — the verification approach for multi-threaded guest code.
//!
//! The interpreter↔JIT differential oracle doesn't apply here: thread ops are interpreter-only, and a
//! threaded run is nondeterministic, so there's no single expected value to diff against. Instead each
//! program is written so that **one invariant must hold under every interleaving**, and we run it many
//! times on the real M:N executor (whose OS-thread scheduling supplies interleaving variety) — a
//! failure on any run is a real concurrency bug (a lost update, a lost wakeup, scheduler corruption).
//! These also run clean under ThreadSanitizer (`-Zsanitizer=thread`).
//!
//! Next iteration (planned): a *deterministic, seeded* scheduler so interleavings are reproducible and
//! can be enumerated (loom-style), turning "run many times" into systematic coverage.

use svm_interp::{run, Trap, Value};
use svm_text::parse_module;
use svm_verify::verify_module;

/// Parse + verify + run entry 0, expecting a single `i64`. Generous fuel (concurrent spin-loops burn
/// some). Each vCPU inherits this fuel, so it bounds every green thread.
fn run_i64(src: &str) -> Result<i64, Trap> {
    let m = parse_module(src).unwrap_or_else(|e| panic!("parse: {e:?}\n{src}"));
    verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}\n{src}"));
    let mut fuel = 50_000_000u64;
    match run(&m, 0, &[], &mut fuel) {
        Ok(vals) => match vals.as_slice() {
            [Value::I64(v)] => Ok(*v),
            other => panic!("expected one i64, got {other:?}"),
        },
        Err(t) => Err(t),
    }
}

/// Assert the program yields `want` on every one of `runs` independent executions — so an
/// interleaving-dependent bug (which only shows on some schedules) is very likely to surface.
fn assert_stable(src: &str, want: i64, runs: usize) {
    for r in 0..runs {
        assert_eq!(run_i64(src), Ok(want), "run #{r} diverged");
    }
}

/// **Mutual exclusion.** 8 worker vCPUs each take a `cmpxchg` spinlock, increment a *non-atomic*
/// counter 100×, and release — so the final count is 800 **iff** the lock truly serializes the
/// critical section. A broken lock (or a scheduler that double-runs / drops a vCPU) races the
/// non-atomic read-modify-write and loses updates, yielding < 800.
///
/// Layout: `mem[0]` i32 lock, `mem[8]` i64 counter, `mem[16+4i]` i32 handle of worker `i`.
#[test]
fn spinlock_serializes_nonatomic_counter() {
    let src = r#"
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
  v5 = i64.const 100
  v6 = thread.spawn 1 v5
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
  v26 = i64.const 8
  v27 = i64.load v26
  return v27
}
func (i64) -> (i64) {
block0(v0: i64):
  br block1(v0)
block1(v1: i64):
  v2 = i64.const 0
  v3 = i64.eq v1 v2
  br_if v3 block4() block2(v1)
block2(v4: i64):
  v5 = i64.const 0
  v6 = i32.const 0
  v7 = i32.const 1
  v8 = i32.atomic.cmpxchg v5 v6 v7
  v9 = i32.const 0
  v10 = i32.eq v8 v9
  br_if v10 block3(v4) block2(v4)
block3(v11: i64):
  v12 = i64.const 8
  v13 = i64.load v12
  v14 = i64.const 1
  v15 = i64.add v13 v14
  i64.store v12 v15
  v16 = i64.const 0
  v17 = i32.const 0
  i32.atomic.store v16 v17
  v18 = i64.const -1
  v19 = i64.add v11 v18
  br block1(v19)
block4():
  v20 = i64.const 0
  return v20
}
"#;
    assert_stable(src, 800, 50);
}

/// **Atomicity.** 8 workers each `atomic.rmw.add` a shared counter 500× — no lock, just the atomic.
/// The total must be exactly 4000 on every interleaving (a non-atomic RMW would lose updates).
#[test]
fn concurrent_atomic_rmw_never_loses_updates() {
    let src = r#"
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
  v6 = thread.spawn 1 v5
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
func (i64) -> (i64) {
block0(v0: i64):
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
    assert_stable(src, 4000, 50);
}

/// **Message passing (release/acquire handoff).** The parent writes a payload to `mem[8]`, then spawns
/// a child that spin-waits (acquire-load) on a flag at `mem[0]`; the parent sets the flag
/// (release-store) and joins. The child returns the payload it read — which must be the written value
/// on every interleaving (the flag orders the payload write before the child's read).
#[test]
fn release_acquire_message_passing() {
    let src = r#"
memory 16
func () -> (i64) {
block0():
  v0 = i64.const 8
  v1 = i64.const 12345
  i64.atomic.store.release v0 v1
  v2 = i64.const 0
  v3 = thread.spawn 1 v2
  v4 = i64.const 0
  v5 = i32.const 1
  i32.atomic.store.release v4 v5
  v6 = thread.join v3
  return v6
}
func (i64) -> (i64) {
block0(v0: i64):
  br block1()
block1():
  v1 = i64.const 0
  v2 = i32.atomic.load.acquire v1
  v3 = i32.const 0
  v4 = i32.eq v2 v3
  br_if v4 block1() block2()
block2():
  v5 = i64.const 8
  v6 = i64.atomic.load.acquire v5
  return v6
}
"#;
    assert_stable(src, 12345, 50);
}
