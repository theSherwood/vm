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

// A child returns a per-spawn value (its `arg`); the root sums the joined results — exercises that
// `thread.join` delivers each child's **return value** to the joiner across threads, not just a count.
const JOIN_VALUES: &str = r#"memory 16
func () -> (i64) {
block0():
  v0 = i64.const 0
  br block1(v0, v0)
block1(v1: i64, vacc: i64):
  v2 = i64.const 4
  v3 = i64.lt_u v1 v2
  br_if v3 block2(v1, vacc) block3(vacc)
block2(v4: i64, vacc2: i64):
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
  br block1(v12, vacc2)
block3(vacc3: i64):
  v13 = i64.const 0
  br block4(v13, vacc3)
block4(v14: i64, vacc4: i64):
  v15 = i64.const 4
  v16 = i64.lt_u v14 v15
  br_if v16 block5(v14, vacc4) block6(vacc4)
block5(v17: i64, vacc5: i64):
  v18 = i64.const 4
  v19 = i64.mul v17 v18
  v20 = i64.const 16
  v21 = i64.add v20 v19
  v22 = i32.load v21
  v23 = thread.join v22
  v24 = i64.add vacc5 v23
  v25 = i64.const 1
  v26 = i64.add v17 v25
  br block4(v26, v24)
block6(vacc6: i64):
  return vacc6
}
func (i64, i64) -> (i64) {
block0(vsp: i64, v0: i64):
  return v0
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
    for i in 0..50 {
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

    for i in 0..50 {
        let (got_r, _) = run_parallel(JOIN_VALUES);
        assert_eq!(got_r, want_r, "parallel join sum != oracle (run {i})");
    }
}
