//! THREADS.md step 4c — **Miri** verification of the parallel driver. The differential tests in
//! `svm/tests/bytecode_parallel.rs` prove the parallel driver agrees with the cooperative oracle; this
//! proves the genuinely-parallel machinery (the per-thread `fork_for_thread` views of one
//! `Region::shared` backing, the cross-thread `Futex` park/wake, the `thread.spawn`/`join` registry)
//! is **free of data races / UB / provenance errors** when the *real interpreter* drives concurrent
//! atomic + non-atomic accesses over the shared window — Miri's checker, not the iteration count, is
//! the point, so the kernels and repeats are small (Miri runs ~100× slower and lives in the `svm`
//! crate's heavyweight suite would drag in the Cranelift JIT it can't build).
//!
//! Run: `cargo +nightly miri test -p svm-interp --test parallel_miri`

use std::sync::Arc;
use svm_interp::{bytecode, Region, Value};
use svm_text::parse_module;

// 4 vCPUs each `atomic.rmw.add` a shared counter 50× → 200. Real cross-thread atomics on one cell.
const ATOMICS: &str = r#"memory 16
func () -> (i64) {
block0():
  v0 = i64.const 0
  br block1(v0)
block1(v1: i64):
  v2 = i64.const 4
  v3 = i64.lt_u v1 v2
  br_if v3 block2(v1) block3()
block2(v4: i64):
  v5 = i64.const 50
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
  v15 = i64.const 4
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

// Futex handoff (2 threads): the producer parks/wakes a consumer via `memory.wait`/`notify`.
const FUTEX: &str = r#"memory 16
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

fn run_parallel(src: &str) -> Result<Vec<Value>, svm_interp::Trap> {
    let m = parse_module(src).unwrap();
    let size = 1 << 16;
    let layout = std::alloc::Layout::from_size_align(size, 8).unwrap();
    // SAFETY: non-zero layout; `size` valid 8-aligned bytes owned here until freed below.
    let base = unsafe { std::alloc::alloc_zeroed(layout) };
    assert!(!base.is_null());
    // SAFETY: `base` is `size` valid 8-aligned bytes, exclusively this window's, freed only after.
    let back = Arc::new(unsafe { Region::shared(base, size as u64) });
    let mut f = u64::MAX;
    let r =
        bytecode::compile_and_run_capture_over_parallel(&m, 0, &[], &mut f, &[], Arc::clone(&back))
            .unwrap()
            .0;
    drop(back);
    // SAFETY: same layout; the region (and all borrows) are gone (the scope joined every vCPU).
    unsafe { std::alloc::dealloc(base, layout) };
    r
}

#[test]
fn parallel_atomics_race_free_under_miri() {
    assert_eq!(run_parallel(ATOMICS), Ok(vec![Value::I64(200)]));
}

#[test]
fn parallel_futex_race_free_under_miri() {
    assert_eq!(run_parallel(FUTEX), Ok(vec![Value::I64(987654)]));
}
