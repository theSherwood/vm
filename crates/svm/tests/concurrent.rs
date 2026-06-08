//! Concurrent **property** tests (§12/§18) — the verification approach for multi-threaded guest code.
//!
//! The interpreter↔JIT differential oracle doesn't apply here: thread ops are interpreter-only, and a
//! threaded run is nondeterministic, so there's no single expected value to diff against. Instead each
//! program is written so **one invariant must hold under every interleaving**, and we check it two ways:
//!
//! - **Stress** (`stress`): run on the real M:N executor many times; OS scheduling supplies
//!   interleaving variety. Catches lost updates / lost wakeups / scheduler corruption, and is
//!   ThreadSanitizer-clean.
//! - **Deterministic sweep** (`sweep`): run on [`svm_interp::run_scheduled`] — a single-threaded,
//!   seed-driven explorer — across many seeds. Each seed realizes one *reproducible* interleaving, so
//!   sweeping is systematic coverage and any failure is replayable from its seed.

use svm_interp::{run, run_scheduled, Trap, Value};
use svm_text::parse_module;
use svm_verify::verify_module;

fn module(src: &str) -> svm_ir::Module {
    let m = parse_module(src).unwrap_or_else(|e| panic!("parse: {e:?}\n{src}"));
    verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}\n{src}"));
    m
}

fn one_i64(vals: Result<Vec<Value>, Trap>) -> Result<i64, Trap> {
    match vals {
        Ok(v) => match v.as_slice() {
            [Value::I64(x)] => Ok(*x),
            other => panic!("expected one i64, got {other:?}"),
        },
        Err(t) => Err(t),
    }
}

/// Run on the **real M:N executor** `runs` times; the invariant must hold every time.
fn stress(src: &str, want: i64, runs: usize) {
    let m = module(src);
    for r in 0..runs {
        let mut fuel = 50_000_000u64;
        assert_eq!(
            one_i64(run(&m, 0, &[], &mut fuel)),
            Ok(want),
            "real-executor run #{r}"
        );
    }
}

/// Run on the **deterministic explorer** across `seeds`; the invariant must hold for every seed (and
/// each is reproducible from that seed).
fn sweep(src: &str, want: i64, seeds: u64) {
    let m = module(src);
    for seed in 0..seeds {
        assert_eq!(
            one_i64(run_scheduled(&m, 0, &[], 50_000_000, seed)),
            Ok(want),
            "explorer seed {seed}"
        );
    }
}

/// **Mutual exclusion.** 8 vCPUs each take a `cmpxchg` spinlock, increment a *non-atomic* counter
/// 100×, and release — final = 800 **iff** the lock truly serializes the critical section. A broken
/// lock (or a scheduler that double-runs / drops a vCPU) races the non-atomic read-modify-write and
/// loses updates. Layout: `mem[0]` i32 lock, `mem[8]` i64 counter, `mem[16+4i]` i32 handle of `i`.
const SPINLOCK: &str = r#"
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
  v26 = i64.const 8
  v27 = i64.load v26
  return v27
}
func (i64, i64) -> (i64) {
block0(vsp: i64, v0: i64):
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

/// **Atomicity.** 8 vCPUs each `atomic.rmw.add` a shared counter 500× — total must be exactly 4000 on
/// every interleaving (a non-atomic RMW would lose updates).
const ATOMIC_COUNTER: &str = r#"
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

/// **Futex handoff.** Producer writes a payload to `mem[8]`, spawns a consumer that `atomic.wait`s on
/// the flag at `mem[0]`, then sets the flag (release) and notifies. The consumer returns the payload
/// it reads — which is the written value on *every* interleaving: if it parked, the notify wakes it
/// after the write; if it checked late, `wait` returns not-equal and it reads the (already-written)
/// payload. Exercises wait/notify + the explorer's logical-clock timeout path.
const FUTEX_HANDOFF: &str = r#"
memory 16
func () -> (i64) {
block0():
  v0 = i64.const 8
  v1 = i64.const 987654
  i64.atomic.store.release v0 v1
  v2 = i64.const 0
  v3 = thread.spawn 1 v2 v2
  v4 = i64.const 0
  v5 = i32.const 1
  i32.atomic.store.release v4 v5
  v6 = i64.const 0
  v7 = i32.const 1
  v8 = atomic.notify v6 v7
  v9 = thread.join v3
  return v9
}
func (i64, i64) -> (i64) {
block0(vsp: i64, v0: i64):
  v1 = i64.const 0
  v2 = i32.const 0
  v3 = i64.const 1000000000
  v4 = i32.atomic.wait v1 v2 v3
  v5 = i64.const 8
  v6 = i64.atomic.load.acquire v5
  return v6
}
"#;

#[test]
fn spinlock_serializes_nonatomic_counter() {
    stress(SPINLOCK, 800, 30);
}

#[test]
fn concurrent_atomic_rmw_never_loses_updates() {
    stress(ATOMIC_COUNTER, 4000, 30);
}

// ---- deterministic explorer: systematic, reproducible interleaving coverage ----

#[test]
fn explorer_spinlock_all_seeds() {
    sweep(SPINLOCK, 800, 200);
}

#[test]
fn explorer_atomic_counter_all_seeds() {
    sweep(ATOMIC_COUNTER, 4000, 200);
}

#[test]
fn explorer_futex_handoff_all_seeds() {
    sweep(FUTEX_HANDOFF, 987654, 200);
}

/// The explorer is deterministic: a given seed always yields the same result (so a failure found at
/// seed S is replayable by re-running seed S).
#[test]
fn explorer_is_reproducible() {
    let m = module(ATOMIC_COUNTER);
    for seed in [1u64, 7, 42, 1000, 999_999] {
        let a = run_scheduled(&m, 0, &[], 50_000_000, seed);
        let b = run_scheduled(&m, 0, &[], 50_000_000, seed);
        assert_eq!(one_i64(a), one_i64(b), "seed {seed} not reproducible");
    }
}
