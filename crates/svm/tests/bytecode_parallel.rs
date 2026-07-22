//! THREADS.md step 4c — the **parallel** driver: one guest's `thread.spawn`ed vCPUs run on **separate
//! OS threads** (the native stand-in for per-vCPU wasm Workers) over **one** `Region::shared` window,
//! with genuine cross-core `atomic.*`. The cooperative `drive` over the same window is the
//! **deterministic oracle** — every parallel run must agree with it (result + final image).
//!
//! The `unsafe` (borrowing host memory via `Region::shared`) lives here in the embedder/test, not in
//! the `#![forbid(unsafe_code)]` engine, which just accepts the pre-built `Arc<Region>`.

use std::sync::Arc;
use svm_interp::{bytecode, Region, Value};
use svm_text::parse_module;

// 8 vCPUs each `atomic.rmw.add` a shared counter 500× → exactly 4000. Under the parallel driver these
// 8 adds-of-500 race on one cell across 8 real OS threads, so the exact total is genuine atomicity,
// not a single-thread interleaving. (Same kernel the cooperative oracle and the wasm Workers run.)
const THREADS: &str = r#"memory 16
func () -> (i64) {
block 0 () {
  v0 = i64.const 0
  br 1(v0)
}
block 1 (v1: i64) {
  v2 = i64.const 8
  v3 = i64.lt_u v1 v2
  br_if v3 2(v1) 3()
}
block 2 (v4: i64) {
  v5 = i64.const 500
  v6 = thread.spawn 1 v5 v5
  v7 = i64.const 4
  v8 = i64.mul v4 v7
  v9 = i64.const 16
  v10 = i64.add v9 v8
  i32.store v10 v6
  v11 = i64.const 1
  v12 = i64.add v4 v11
  br 1(v12)
}
block 3 () {
  v13 = i64.const 0
  br 4(v13)
}
block 4 (v14: i64) {
  v15 = i64.const 8
  v16 = i64.lt_u v14 v15
  br_if v16 5(v14) 6()
}
block 5 (v17: i64) {
  v18 = i64.const 4
  v19 = i64.mul v17 v18
  v20 = i64.const 16
  v21 = i64.add v20 v19
  v22 = i32.load v21
  v23 = thread.join v22
  v24 = i64.const 1
  v25 = i64.add v17 v24
  br 4(v25)
}
block 6 () {
  v26 = i64.const 0
  v27 = i64.atomic.load v26
  return v27
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, v0: i64) {
  br 1(v0)
}
block 1 (v1: i64) {
  v2 = i64.const 0
  v3 = i64.eq v1 v2
  br_if v3 3() 2(v1)
}
block 2 (v4: i64) {
  v5 = i64.const 0
  v6 = i64.const 1
  v7 = i64.atomic.rmw.add v5 v6
  v8 = i64.const -1
  v9 = i64.add v4 v8
  br 1(v9)
}
block 3 () {
  v10 = i64.const 0
  return v10
  }
}
"#;

// A child returns a per-spawn value (its `arg`); the root sums the joined results — exercises that
// `thread.join` delivers each child's **return value** to the joiner across threads, not just a count.
const JOIN_VALUES: &str = r#"memory 16
func () -> (i64) {
block 0 () {
  v0 = i64.const 0
  br 1(v0, v0)
}
block 1 (v1: i64, vacc: i64) {
  v2 = i64.const 4
  v3 = i64.lt_u v1 v2
  br_if v3 2(v1, vacc) 3(vacc)
}
block 2 (v4: i64, vacc2: i64) {
  v5 = i64.const 10
  v5b = i64.add v4 v5
  v6 = thread.spawn 1 v5b v5b
  v7 = i64.const 4
  v8 = i64.mul v4 v7
  v9 = i64.const 16
  v10 = i64.add v9 v8
  i32.store v10 v6
  v11 = i64.const 1
  v12 = i64.add v4 v11
  br 1(v12, vacc2)
}
block 3 (vacc3: i64) {
  v13 = i64.const 0
  br 4(v13, vacc3)
}
block 4 (v14: i64, vacc4: i64) {
  v15 = i64.const 4
  v16 = i64.lt_u v14 v15
  br_if v16 5(v14, vacc4) 6(vacc4)
}
block 5 (v17: i64, vacc5: i64) {
  v18 = i64.const 4
  v19 = i64.mul v17 v18
  v20 = i64.const 16
  v21 = i64.add v20 v19
  v22 = i32.load v21
  v23 = thread.join v22
  v24 = i64.add vacc5 v23
  v25 = i64.const 1
  v26 = i64.add v17 v25
  br 4(v26, v24)
}
block 6 (vacc6: i64) {
  return vacc6
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, v0: i64) {
  return v0
  }
}
"#;

// Futex handoff: the producer writes a payload at mem[8], spawns a consumer that `atomic.wait`s on
// the flag at mem[0], then sets the flag and `notify`s it. Under the parallel driver the consumer is a
// **real** OS thread that genuinely parks on the cross-thread futex (or takes the not-equal fast path
// if it loses the race to the flag store) — either way it returns the payload (987654) on every
// interleaving, so the result is interleaving-invariant and differential-tests the futex cleanly.
const FUTEX_HANDOFF: &str = r#"memory 16
func () -> (i64) {
block 0 () {
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
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, v0: i64) {
  v1 = i64.const 0
  v2 = i32.const 0
  v3 = i64.const 1000000000
  v4 = i32.atomic.wait v1 v2 v3
  v5 = i64.const 8
  v6 = i64.atomic.load.acquire v5
  return v6
  }
}
"#;

// A barrier-style fan-in: 8 workers each `atomic.rmw.add` a shared counter, and the last one to arrive
// (counter == 8) `notify`s the flag at mem[8]; the root parks on that flag via `atomic.wait` until
// released, then returns the counter (8). Exercises notify waking a genuinely-parked root across
// threads, with an interleaving-invariant result.
const BARRIER: &str = r#"memory 16
func () -> (i64) {
block 0 () {
  v0 = i64.const 0
  br 1(v0)
}
block 1 (v1: i64) {
  v2 = i64.const 8
  v3 = i64.lt_u v1 v2
  br_if v3 2(v1) 3()
}
block 2 (v4: i64) {
  v5 = i64.const 0
  v6 = thread.spawn 1 v5 v5
  v7 = i64.const 4
  v8 = i64.mul v4 v7
  v9 = i64.const 16
  v10 = i64.add v9 v8
  i32.store v10 v6
  v11 = i64.const 1
  v12 = i64.add v4 v11
  br 1(v12)
}
block 3 () {
  br 4()
}
block 4 () {
  v13 = i64.const 8
  v14 = i32.const 0
  v15 = i64.const 1000000000
  v16 = i32.atomic.wait v13 v14 v15
  v17 = i64.const 0
  v18 = i64.atomic.load v17
  v19 = i64.const 8
  v20 = i64.lt_u v18 v19
  br_if v20 4() 5(v18)
}
block 5 (v21: i64) {
  br 6(v21)
}
block 6 (v22: i64) {
  v23 = i64.const 0
  br 7(v22, v23)
}
block 7 (v24: i64, v25: i64) {
  v26 = i64.const 8
  v27 = i64.lt_u v25 v26
  br_if v27 8(v24, v25) 9(v24)
}
block 8 (v28: i64, v29: i64) {
  v30 = i64.const 4
  v31 = i64.mul v29 v30
  v32 = i64.const 16
  v33 = i64.add v32 v31
  v34 = i32.load v33
  v35 = thread.join v34
  v36 = i64.const 1
  v37 = i64.add v29 v36
  br 7(v28, v37)
}
block 9 (v38: i64) {
  return v38
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, v0: i64) {
  v1 = i64.const 0
  v2 = i64.const 1
  v3 = i64.atomic.rmw.add v1 v2
  v4 = i64.const 7
  v5 = i64.eq v3 v4
  br_if v5 1() 2()
}
block 1 () {
  v6 = i64.const 8
  v7 = i32.const 1
  i32.atomic.store v6 v7
  v8 = i64.const 8
  v9 = i32.const 100
  v10 = atomic.notify v8 v9
  br 2()
}
block 2 () {
  v11 = i64.const 0
  return v11
  }
}
"#;

/// An 8-aligned zeroed buffer + a `Region::shared` over it; caller frees via the returned layout.
fn shared_window(size: usize) -> (Arc<Region>, *mut u8, std::alloc::Layout) {
    let layout = std::alloc::Layout::from_size_align(size, 8).unwrap();
    // SAFETY: non-zero layout; the buffer is `size` valid 8-aligned bytes owned here, used only as
    // this window until freed by the caller after the region (and any borrows) are dropped.
    let base = unsafe { std::alloc::alloc_zeroed(layout) };
    assert!(!base.is_null());
    // SAFETY: `base` is `size` valid 8-aligned bytes, exclusively this window's, freed only after.
    let back = Arc::new(unsafe { Region::shared(base, size as u64) });
    (back, base, layout)
}

/// Run `src`'s function 0 in **parallel** over a fresh shared window, returning (result, final image).
fn run_parallel(src: &str) -> (Result<Vec<Value>, svm_interp::Trap>, Vec<u8>) {
    let m = parse_module(src).unwrap();
    let (back, base, layout) = shared_window(1 << 16);
    let mut f = u64::MAX;
    let cap =
        bytecode::compile_and_run_capture_over_parallel(&m, 0, &[], &mut f, &[], Arc::clone(&back))
            .unwrap();
    drop(back);
    // SAFETY: same layout; the region (and all borrows of `base`) are gone (the scope joined all vCPUs).
    unsafe { std::alloc::dealloc(base, layout) };
    cap
}

/// The cooperative oracle over its own engine backing, for differential comparison.
fn run_cooperative(src: &str) -> (Result<Vec<Value>, svm_interp::Trap>, Vec<u8>) {
    let m = parse_module(src).unwrap();
    let mut f = u64::MAX;
    bytecode::compile_and_run_capture(&m, 0, &[], &mut f, &[]).unwrap()
}

/// Real-race repeat count: many natively (flakiness shows up across repeats), but just a couple under
/// Miri, whose data-race/UB checker (not the count) is the point and which runs ~100× slower.
const fn reps(native: usize) -> usize {
    if cfg!(miri) {
        2
    } else {
        native
    }
}

/// 8 vCPUs on 8 real OS threads racing one counter through the shared window land on the exact total —
/// genuine parallel atomicity, byte-identical to the cooperative oracle, and stable across repeats.
#[test]
fn parallel_threads_match_cooperative_oracle() {
    let (want_r, want_snap) = run_cooperative(THREADS);
    assert_eq!(
        want_r,
        Ok(vec![Value::I64(4000)]),
        "oracle: 8 vCPUs × 500 atomic adds → 4000"
    );

    // Repeat: real races, so a wrong driver would be flaky — a stable exact match is the evidence.
    for i in 0..reps(50) {
        let (got_r, got_snap) = run_parallel(THREADS);
        assert_eq!(
            got_r, want_r,
            "parallel result != cooperative oracle (run {i})"
        );
        assert_eq!(
            got_snap, want_snap,
            "parallel final image != cooperative oracle (run {i})"
        );
    }
}

/// `thread.join` delivers each child's **return value** to the joiner across OS threads: the root sums
/// the four children's args (10+11+12+13 = 46), matching the cooperative oracle.
#[test]
fn parallel_join_delivers_child_return_values() {
    let (want_r, _) = run_cooperative(JOIN_VALUES);
    assert_eq!(
        want_r,
        Ok(vec![Value::I64(46)]),
        "oracle: sum of joined args"
    );

    for i in 0..reps(50) {
        let (got_r, _) = run_parallel(JOIN_VALUES);
        assert_eq!(got_r, want_r, "parallel join sum != oracle (run {i})");
    }
}

/// The cross-thread futex: a real OS thread parks on `memory.wait` and is released by the producer's
/// `notify` (or wins the not-equal fast path) — either way the handoff delivers 987654, matching the
/// cooperative oracle on every interleaving.
#[test]
fn parallel_futex_handoff_matches_oracle() {
    let (want_r, _) = run_cooperative(FUTEX_HANDOFF);
    assert_eq!(
        want_r,
        Ok(vec![Value::I64(987654)]),
        "oracle: futex handoff"
    );

    for i in 0..reps(100) {
        let (got_r, _) = run_parallel(FUTEX_HANDOFF);
        assert_eq!(got_r, want_r, "parallel futex handoff != oracle (run {i})");
    }
}

/// A barrier where the **root** genuinely parks on `memory.wait` until the last of 8 worker threads
/// `notify`s it — the wakeup must cross OS threads for the run to make progress. Returns 8, matching
/// the oracle; a stuck/lost-wakeup futex would hang or diverge.
#[test]
fn parallel_futex_barrier_matches_oracle() {
    let (want_r, _) = run_cooperative(BARRIER);
    assert_eq!(want_r, Ok(vec![Value::I64(8)]), "oracle: 8-worker barrier");

    for i in 0..reps(50) {
        let (got_r, _) = run_parallel(BARRIER);
        assert_eq!(got_r, want_r, "parallel barrier != oracle (run {i})");
    }
}
