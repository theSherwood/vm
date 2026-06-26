//! THREADS.md step 3 — the substrate→engine bridge: the bytecode engine runs a guest over a
//! **caller-owned** shared-memory window (`Region::shared`) instead of an engine-`mmap`ped one. This
//! is the backing the parallel-wasm mode will run every per-vCPU Worker over; here it's exercised
//! **cooperatively**, so results + final image must be byte-identical to the engine's own backing,
//! and the guest's writes must land in the caller's buffer.
//!
//! The `unsafe` (borrowing host memory via `Region::shared`) lives in this embedder/test, not in the
//! `#![forbid(unsafe_code)]` engine, which just accepts the pre-built `Arc<Region>`.

use std::sync::Arc;
use svm_interp::{bytecode, Region, Value};
use svm_text::parse_module;

// Writes the accumulator to offset 8 each iteration, then returns the sum (touches linear memory).
const MEM: &str = r#"memory 16
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const 0
  v2 = i64.const 0
  br block1(v0, v1, v2)
block1(v3: i64, v4: i64, v5: i64):
  v6 = i64.lt_s v5 v3
  br_if v6 block2(v3, v4, v5) block3(v4)
block2(v7: i64, v8: i64, v9: i64):
  v10 = i64.const 8
  i64.store v10 v8
  v11 = i64.load v10
  v12 = i64.add v11 v9
  v13 = i64.const 1
  v14 = i64.add v9 v13
  br block1(v7, v12, v14)
block3(v15: i64):
  return v15
}
"#;

// 8 vCPUs each `atomic.rmw.add` a shared counter 500× → exactly 4000 (cooperative multi-vCPU +
// atomics + futex join, all over the one shared window).
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

/// A compute+memory guest over a caller-owned shared window matches the engine's own backing exactly,
/// and its store lands in the caller's buffer.
#[test]
fn engine_runs_over_caller_owned_shared_window() {
    let m = parse_module(MEM).unwrap();
    let init = vec![0u8; 64];

    let mut f = u64::MAX;
    let (want_r, want_snap) =
        bytecode::compile_and_run_capture(&m, 0, &[Value::I64(5)], &mut f, &init).unwrap();

    let (back, base, layout) = shared_window(1 << 16);
    let mut f2 = u64::MAX;
    let (got_r, got_snap) = bytecode::compile_and_run_capture_over(
        &m,
        0,
        &[Value::I64(5)],
        &mut f2,
        &init,
        Arc::clone(&back),
    )
    .unwrap();

    assert_eq!(got_r, want_r, "result over shared window != engine backing");
    assert_eq!(
        got_snap, want_snap,
        "final image over shared window != engine backing"
    );
    // The guest's store at offset 8 physically landed in the caller's buffer.
    let in_buf = back.read_word(8, 8);
    assert_eq!(
        in_buf,
        u64::from_le_bytes(got_snap[8..16].try_into().unwrap())
    );
    assert_ne!(
        in_buf, 0,
        "the guest write should be visible in the borrowed buffer"
    );

    drop(back);
    // SAFETY: same layout; the region (and all borrows of `base`) are gone.
    unsafe { std::alloc::dealloc(base, layout) };
}

/// Cooperative multi-vCPU concurrency (`thread.spawn`/`join` + atomics + futex) runs over the shared
/// window too — the exact configuration the parallel mode will later run on separate Workers.
#[test]
fn cooperative_threads_over_shared_window() {
    let m = parse_module(THREADS).unwrap();

    let mut f = u64::MAX;
    let want = bytecode::compile_and_run_capture(&m, 0, &[], &mut f, &[])
        .unwrap()
        .0;

    let (back, base, layout) = shared_window(1 << 16);
    let mut f2 = u64::MAX;
    let got = bytecode::compile_and_run_capture_over(&m, 0, &[], &mut f2, &[], Arc::clone(&back))
        .unwrap()
        .0;

    assert_eq!(
        got, want,
        "thread kernel over shared window != engine backing"
    );
    assert_eq!(
        got,
        Ok(vec![Value::I64(4000)]),
        "8 cooperative vCPUs over the shared window → 4000"
    );

    drop(back);
    // SAFETY: same layout; the region (and all borrows of `base`) are gone.
    unsafe { std::alloc::dealloc(base, layout) };
}
