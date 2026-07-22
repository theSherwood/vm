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
use svm_interp::{bytecode, Host, Region, StreamRole, Value};
use svm_text::parse_module;

// 4 vCPUs each `atomic.rmw.add` a shared counter 50× → 200. Real cross-thread atomics on one cell.
const ATOMICS: &str = r#"memory 16
func () -> (i64) {
block 0 () {
  v0 = i64.const 0
  br 1(v0)
}
block 1 (v1: i64) {
  v2 = i64.const 4
  v3 = i64.lt_u v1 v2
  br_if v3 2(v1) 3()
}
block 2 (v4: i64) {
  v5 = i64.const 50
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
  v15 = i64.const 4
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

// Futex handoff (2 threads): the producer parks/wakes a consumer via `memory.wait`/`notify`.
const FUTEX: &str = r#"memory 16
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

// 2 worker vCPUs each write "hi\n" to stdout via `cap.call` (handle threaded through block args) + bump
// a shared counter — exercises the shared `Mutex<Host>` and per-cap.call locking across real threads.
const CAPS: &str = r#"memory 16
data 0 "hi\n"
func (i32) -> (i64) {
block 0 (v0: i32) {
  vh0 = i64.extend_i32_u v0
  v1 = i64.const 0
  br 1(v1, vh0)
}
block 1 (vi: i64, vhh: i64) {
  v2 = i64.const 2
  v3 = i64.lt_u vi v2
  br_if v3 2(vi, vhh) 3()
}
block 2 (vi2: i64, vhh2: i64) {
  vsp = i64.const 0
  vt = thread.spawn 1 vsp vhh2
  v4 = i64.const 4
  v5 = i64.mul vi2 v4
  v6 = i64.const 16
  v7 = i64.add v6 v5
  i32.store v7 vt
  v8 = i64.const 1
  v9 = i64.add vi2 v8
  br 1(v9, vhh2)
}
block 3 () {
  v10 = i64.const 0
  br 4(v10)
}
block 4 (vj: i64) {
  v11 = i64.const 2
  v12 = i64.lt_u vj v11
  br_if v12 5(vj) 6()
}
block 5 (vj2: i64) {
  v13 = i64.const 4
  v14 = i64.mul vj2 v13
  v15 = i64.const 16
  v16 = i64.add v15 v14
  v17 = i32.load v16
  v18 = thread.join v17
  v19 = i64.const 1
  v20 = i64.add vj2 v19
  br 4(v20)
}
block 6 () {
  v21 = i64.const 8
  v22 = i64.atomic.load v21
  return v22
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, vh: i64) {
  vhandle = i32.wrap_i64 vh
  vptr = i64.const 0
  vlen = i64.const 3
  vw = cap.call 0 1 (i64, i64) -> (i64) vhandle(vptr, vlen)
  v1 = i64.const 8
  v2 = i64.const 1
  v3 = i64.atomic.rmw.add v1 v2
  v4 = i64.const 0
  return v4
  }
}
"#;

#[test]
fn parallel_shared_host_capcall_race_free_under_miri() {
    let m = parse_module(CAPS).unwrap();
    let mut host = Host::new();
    let h = host.grant_stream(StreamRole::Out);
    let size = 1 << 16;
    let layout = std::alloc::Layout::from_size_align(size, 8).unwrap();
    // SAFETY: non-zero layout; `size` valid 8-aligned bytes owned here until freed below.
    let base = unsafe { std::alloc::alloc_zeroed(layout) };
    assert!(!base.is_null());
    // SAFETY: `base` is `size` valid 8-aligned bytes, exclusively this window's, freed only after.
    let back = Arc::new(unsafe { Region::shared(base, size as u64) });
    let mut f = u64::MAX;
    let r = bytecode::compile_and_run_capture_over_parallel_with_host(
        &m,
        0,
        &[Value::I32(h)],
        &mut f,
        &[],
        Arc::clone(&back),
        &mut host,
    )
    .unwrap()
    .0;
    drop(back);
    // SAFETY: same layout; the region (and all borrows) are gone (the scope joined every vCPU).
    unsafe { std::alloc::dealloc(base, layout) };
    assert_eq!(r, Ok(vec![Value::I64(2)]));
    assert_eq!(host.stdout, b"hi\n".repeat(2));
}
