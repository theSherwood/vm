//! THREADS.md 4c-host — spawned vCPUs of the **parallel** driver share the powerbox, so `cap.call`
//! (host I/O) works from worker vCPUs, serialized per call by the shared-host lock while
//! compute/atomics stay parallel. Differential-tested against the **cooperative** oracle (which already
//! shares one host across its vCPUs, deterministically).
//!
//! The kernel has 8 worker vCPUs each (a) write the **same** 5-byte line to stdout via `cap.call` and
//! (b) `atomic.rmw.add` a shared counter. Because every write is identical and each `cap.call` is
//! atomic under the lock, the stdout bytes are **schedule-independent** (`"tick\n"` × 8) — so even this
//! stateful capability is byte-identical to the oracle, while genuinely exercising concurrent
//! `cap.call` on the shared host. (Order-sensitive caps like distinct writes / `Clock.now` would race —
//! that is the documented opt-in non-determinism of the parallel mode.)
//!
//! The `unsafe` (borrowing host memory via `Region::shared`) lives in this embedder/test.

use std::sync::Arc;
use svm_interp::{bytecode, Host, Region, StreamRole, Value};
use svm_text::parse_module;

// func 0 (root, param = stdout handle): spawn 8 workers (passing the handle), join them, return the
// counter at mem[8]. func 1 (worker, args = sp, stdout handle): write "tick\n" then bump the counter.
const CAPS: &str = r#"memory 16
data 0 "tick\n"
func (i32) -> (i64) {
block 0 (v0: i32) {
  vh0 = i64.extend_i32_u v0
  v1 = i64.const 0
  br 1(v1, vh0)
}
block 1 (vi: i64, vhh: i64) {
  v2 = i64.const 8
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
  v11 = i64.const 8
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
  vlen = i64.const 5
  vw = cap.call 0 1 (i64, i64) -> (i64) vhandle(vptr, vlen)
  v1 = i64.const 8
  v2 = i64.const 1
  v3 = i64.atomic.rmw.add v1 v2
  v4 = i64.const 0
  return v4
  }
}
"#;

/// An 8-aligned zeroed buffer + a `Region::shared` over it; caller frees via the returned layout.
fn shared_window(size: usize) -> (Arc<Region>, *mut u8, std::alloc::Layout) {
    let layout = std::alloc::Layout::from_size_align(size, 8).unwrap();
    // SAFETY: non-zero layout; `size` valid 8-aligned bytes owned here, used only as this window.
    let base = unsafe { std::alloc::alloc_zeroed(layout) };
    assert!(!base.is_null());
    // SAFETY: `base` is `size` valid 8-aligned bytes, exclusively this window's, freed only after.
    let back = Arc::new(unsafe { Region::shared(base, size as u64) });
    (back, base, layout)
}

/// Cooperative oracle: one shared host across all vCPUs (deterministic). Returns (result, stdout).
fn run_cooperative() -> (Result<Vec<Value>, svm_interp::Trap>, Vec<u8>) {
    let m = parse_module(CAPS).unwrap();
    let mut host = Host::new();
    let h = host.grant_stream(StreamRole::Out);
    let mut f = u64::MAX;
    let r =
        bytecode::compile_and_run_with_host(&m, 0, &[Value::I32(h)], &mut f, &mut host).unwrap();
    (r, host.stdout)
}

/// Parallel: the same kernel over real OS threads sharing one powerbox. Returns (result, stdout).
fn run_parallel() -> (Result<Vec<Value>, svm_interp::Trap>, Vec<u8>) {
    let m = parse_module(CAPS).unwrap();
    let mut host = Host::new();
    let h = host.grant_stream(StreamRole::Out);
    let (back, base, layout) = shared_window(1 << 16);
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
    (r, host.stdout)
}

/// 8 worker vCPUs do host I/O (`cap.call` stdout write) + atomics on the **shared** powerbox, in
/// genuine parallel — result and the (identical-line, schedule-independent) stdout match the oracle.
#[test]
fn parallel_shared_host_capcall_matches_oracle() {
    let (want_r, want_out) = run_cooperative();
    assert_eq!(
        want_r,
        Ok(vec![Value::I64(8)]),
        "oracle: 8 workers → counter 8"
    );
    assert_eq!(want_out, b"tick\n".repeat(8), "oracle stdout: 8 lines");

    // Real races on the shared host — a wrong lock/sharing would corrupt stdout or the counter.
    for i in 0..50 {
        let (got_r, got_out) = run_parallel();
        assert_eq!(got_r, want_r, "parallel result != oracle (run {i})");
        assert_eq!(got_out, want_out, "parallel stdout != oracle (run {i})");
    }
}
