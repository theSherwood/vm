//! §9/§12 the **submit/complete ring** (io_uring-shaped), increment 1 — synchronous batched
//! `cap.call`s. The guest writes `n` 64-byte SQEs (each a *deferred `cap.call`*) into its window,
//! submits the whole batch with **one** `cap.call` on its `IoRing` handle (the boundary-crossing
//! amortization, §1a), and reaps `n` 32-byte CQEs. Because an SQE routes through the *same*
//! capability dispatch as a direct call, the JIT gets the ring for free (a generic `cap.call` through
//! the host thunk), so the whole thing is differentially tested interp↔JIT (the §18 oracle). The
//! result is identical to issuing the ops directly — that's the correctness claim.

use std::time::Duration;
use svm_interp::{run_capture_reserved_with_host, Host, Value, OFFLOAD_POOL_THREADS};
use svm_jit::{compile_and_run_capture_reserved_with_host, JitOutcome};
use svm_text::parse_module;
use svm_verify::verify_module;

/// Grant **both** an interp `Host` and a JIT `Host` an identical `(IoRing, Clock)` pair (granted in
/// the same order ⇒ matching handle encodings), run the entry over a fully-mapped 128 KiB window, and
/// return both results and final windows for byte-comparison.
fn both(src: &str) -> (Value, Vec<u8>, JitOutcome, Vec<u8>) {
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");
    let init = [0u8; 128 << 10];

    let mut hi = Host::new();
    let (iri, ici) = (hi.grant_io_ring(), hi.grant_clock());
    let mut hj = Host::new();
    let (irj, icj) = (hj.grant_io_ring(), hj.grant_clock());
    assert_eq!((iri, ici), (irj, icj), "grants must encode identically");

    let mut fuel = 1_000_000u64;
    let (ir, imem) = run_capture_reserved_with_host(
        &m,
        0,
        &[Value::I32(iri), Value::I32(ici)],
        &mut fuel,
        &init,
        0,
        &mut hi,
    );
    let (jo, jmem) = compile_and_run_capture_reserved_with_host(
        &m,
        0,
        &[irj as i64, icj as i64],
        &init,
        0,
        svm_run::cap_thunk,
        &mut hj as *mut Host as *mut core::ffi::c_void,
    )
    .expect("jit");
    let ival = ir.expect("interp ran ok").pop().expect("one result");
    (ival, imem, jo, jmem)
}

/// Build 8 SQEs (each a `Clock.now`, iface 2 / op 0, on the granted Clock handle) at window offset 0,
/// `submit` them through the ring (`cap.call 9 0`), then sum the 8 CQE result fields. The mock clock
/// returns a strictly-increasing counter (0,1,…,7), so the deferred batch must total `0+…+7 = 28` —
/// exactly what 8 direct `Clock.now` calls would yield. SQE = 64 B `{u32 type_id|u32 op|i32 handle|
/// u32 n_args|i64 args[4]|i64 user_data|i64 pad}`; CQE = 32 B `{i64 user_data|i64 result|i64 status|
/// i64 pad}`; the 8 CQEs go at offset 512 (just past the 8·64-byte SQ).
#[test]
fn ring_runs_a_batch_of_deferred_cap_calls() {
    let src = "memory 17
func (i32, i32) -> (i64) {
block0(v0: i32, v1: i32):
  v2 = i64.const 0
  br block1(v0, v1, v2)
block1(v3: i32, v4: i32, v5: i64):
  v6 = i64.const 64
  v7 = i64.mul v5 v6
  v8 = i32.const 2
  i32.store v7 v8
  v9 = i64.const 4
  v10 = i64.add v7 v9
  v11 = i32.const 0
  i32.store v10 v11
  v12 = i64.const 8
  v13 = i64.add v7 v12
  i32.store v13 v4
  v14 = i64.const 12
  v15 = i64.add v7 v14
  i32.store v15 v11
  v16 = i64.const 48
  v17 = i64.add v7 v16
  i64.store v17 v5
  v18 = i64.const 1
  v19 = i64.add v5 v18
  v20 = i64.const 8
  v21 = i64.lt_u v19 v20
  br_if v21 block1(v3, v4, v19) block2(v3)
block2(v22: i32):
  v23 = i64.const 0
  v24 = i64.const 8
  v25 = i64.const 512
  v26 = cap.call 9 0 (i64, i64, i64) -> (i64) v22 (v23, v24, v25)
  v27 = i64.const 0
  v28 = i64.const 0
  br block3(v27, v28)
block3(v29: i64, v30: i64):
  v31 = i64.const 32
  v32 = i64.mul v29 v31
  v33 = i64.const 512
  v34 = i64.add v33 v32
  v35 = i64.const 8
  v36 = i64.add v34 v35
  v37 = i64.load v36
  v38 = i64.add v30 v37
  v39 = i64.const 1
  v40 = i64.add v29 v39
  v41 = i64.const 8
  v42 = i64.lt_u v40 v41
  br_if v42 block3(v40, v38) block4(v38)
block4(v43: i64):
  return v43
}
";
    let (ival, imem, jo, jmem) = both(src);
    assert_eq!(
        ival,
        Value::I64(28),
        "interp: 8 batched Clock.now must total 0+1+...+7 = 28"
    );
    assert!(
        matches!(jo, JitOutcome::Returned(ref s) if s == &[28]),
        "jit: {jo:?}"
    );
    assert_eq!(
        imem, jmem,
        "interp/JIT windows diverge after the ring batch"
    );
}

/// The `completed` return value: `submit` reports how many SQEs it ran. Submit 5 (no-arg `Clock.now`)
/// and return the count — must be 5 on both backends.
#[test]
fn ring_reports_completed_count() {
    let src = "memory 17
func (i32, i32) -> (i64) {
block0(v0: i32, v1: i32):
  v2 = i64.const 0
  br block1(v0, v1, v2)
block1(v3: i32, v4: i32, v5: i64):
  v6 = i64.const 64
  v7 = i64.mul v5 v6
  v8 = i32.const 2
  i32.store v7 v8
  v9 = i64.const 8
  v10 = i64.add v7 v9
  i32.store v10 v4
  v11 = i64.const 1
  v12 = i64.add v5 v11
  v13 = i64.const 5
  v14 = i64.lt_u v12 v13
  br_if v14 block1(v3, v4, v12) block2(v3)
block2(v15: i32):
  v16 = i64.const 0
  v17 = i64.const 5
  v18 = i64.const 512
  v19 = cap.call 9 0 (i64, i64, i64) -> (i64) v15 (v16, v17, v18)
  return v19
}
";
    let (ival, _imem, jo, _jmem) = both(src);
    assert_eq!(ival, Value::I64(5), "interp: submit reports 5 completed");
    assert!(
        matches!(jo, JitOutcome::Returned(ref s) if s == &[5]),
        "jit: {jo:?}"
    );
}

// ----- increment 2: the bounded blocking-offload pool -----------------------------------------
//
// A `submit` batch of `Blocking` SQEs (iface 10, op 0) is handed to the host's K-thread offload pool
// and run *concurrently*, instead of serially on the guest's vCPU thread (§12 "0 blocked vCPU
// threads"). Because each `Blocking` result is a deterministic pure transform and the CQEs are still
// written on the submit thread in SQE order, the outcome is **identical to running every op inline** —
// so the whole thing stays differentially testable interp↔JIT (the §18 oracle), and a side counter
// (`max_active`) lets us *prove* the batch genuinely overlapped.

/// The deterministic transform the mock `Blocking` op applies (mirrors `AsyncState::mix` in
/// svm-interp). One Knuth LCG step — non-trivial, so a divergence would show in the CQE results.
fn mix(arg: i64) -> i64 {
    arg.wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407)
}

/// Entry `(i32 ioring, i32 blocking) -> (i64)`: build `n` 64-byte SQEs at window offset 0 — each a
/// `Blocking.work` (`type_id 10, op 0, n_args 1, args[0] = i, user_data = i`) — `submit` the batch on
/// the IoRing handle, then sum the `n` CQE result fields (the CQ sits at offset 512, past the ≤512 B
/// SQ for `n ≤ 8`). The sum must equal `Σ mix(i)` regardless of the order the pool ran them in.
fn batch_src(n: u64) -> String {
    format!(
        "memory 17
func (i32, i32) -> (i64) {{
block0(v0: i32, v1: i32):
  v2 = i64.const 0
  br block1(v0, v1, v2)
block1(v3: i32, v4: i32, v5: i64):
  v6 = i64.const 64
  v7 = i64.mul v5 v6
  v8 = i32.const 10
  i32.store v7 v8
  v9 = i64.const 4
  v10 = i64.add v7 v9
  v11 = i32.const 0
  i32.store v10 v11
  v12 = i64.const 8
  v13 = i64.add v7 v12
  i32.store v13 v4
  v14 = i64.const 12
  v15 = i64.add v7 v14
  v16 = i32.const 1
  i32.store v15 v16
  v17 = i64.const 16
  v18 = i64.add v7 v17
  i64.store v18 v5
  v19 = i64.const 48
  v20 = i64.add v7 v19
  i64.store v20 v5
  v21 = i64.const 1
  v22 = i64.add v5 v21
  v23 = i64.const {n}
  v24 = i64.lt_u v22 v23
  br_if v24 block1(v3, v4, v22) block2(v3)
block2(v25: i32):
  v26 = i64.const 0
  v27 = i64.const {n}
  v28 = i64.const 512
  v29 = cap.call 9 0 (i64, i64, i64) -> (i64) v25 (v26, v27, v28)
  v30 = i64.const 0
  v31 = i64.const 0
  br block3(v30, v31)
block3(v32: i64, v33: i64):
  v34 = i64.const 32
  v35 = i64.mul v32 v34
  v36 = i64.const 512
  v37 = i64.add v36 v35
  v38 = i64.const 8
  v39 = i64.add v37 v38
  v40 = i64.load v39
  v41 = i64.add v33 v40
  v42 = i64.const 1
  v43 = i64.add v32 v42
  v44 = i64.const {n}
  v45 = i64.lt_u v43 v44
  br_if v45 block3(v43, v41) block4(v41)
block4(v46: i64):
  return v46
}}
"
    )
}

/// Run `batch_src(n)` on **both** backends with a `(IoRing, Blocking)` grant pair (block duration +
/// optional rendezvous configurable), returning both results, both final windows, and both `Host`s
/// (so a test can read back each pool's realized `max_active`) plus the shared blocking handle.
#[allow(clippy::type_complexity)]
fn run_batch(
    n: u64,
    block_for: Duration,
    rendezvous: Option<usize>,
) -> (Value, Vec<u8>, JitOutcome, Vec<u8>, Host, Host, i32) {
    let src = batch_src(n);
    let m = parse_module(&src).expect("parse");
    verify_module(&m).expect("verify");
    let init = [0u8; 128 << 10];

    let mut hi = Host::new();
    let (iri, ibi) = (hi.grant_io_ring(), hi.grant_blocking(block_for, rendezvous));
    let mut hj = Host::new();
    let (irj, ibj) = (hj.grant_io_ring(), hj.grant_blocking(block_for, rendezvous));
    assert_eq!((iri, ibi), (irj, ibj), "grants must encode identically");

    let mut fuel = 5_000_000u64;
    let (ir, imem) = run_capture_reserved_with_host(
        &m,
        0,
        &[Value::I32(iri), Value::I32(ibi)],
        &mut fuel,
        &init,
        0,
        &mut hi,
    );
    let (jo, jmem) = compile_and_run_capture_reserved_with_host(
        &m,
        0,
        &[irj as i64, ibj as i64],
        &init,
        0,
        svm_run::cap_thunk,
        &mut hj as *mut Host as *mut core::ffi::c_void,
    )
    .expect("jit");
    let ival = ir.expect("interp ran ok").pop().expect("one result");
    (ival, imem, jo, jmem, hi, hj, ibi)
}

/// **Transparency:** an offloaded batch yields exactly what running every op inline in order would —
/// `Σ mix(i)` — identically on both backends, with byte-identical final windows. The whole point of
/// the pool is that overlapping the blocking ops changes *when* they run, never the result.
#[test]
fn offload_batch_matches_inline_on_both_backends() {
    let n = 6u64;
    let expected: i64 = (0..n as i64).map(mix).fold(0i64, |a, b| a.wrapping_add(b));
    let (ival, imem, jo, jmem, _hi, _hj, _h) = run_batch(n, Duration::ZERO, None);
    assert_eq!(
        ival,
        Value::I64(expected),
        "interp: offloaded batch must sum to Σ mix(i)"
    );
    assert!(
        matches!(jo, JitOutcome::Returned(ref s) if s == &[expected]),
        "jit: {jo:?} (want {expected})"
    );
    assert_eq!(
        imem, jmem,
        "interp/JIT windows diverge after the offloaded batch"
    );
}

/// **Overlap proof (deterministic):** submit exactly `K` blocking ops with a width-`K` rendezvous, so
/// every op must be simultaneously in-flight on the `K`-thread pool before any completes. The realized
/// peak concurrency (`max_active`) must therefore reach `K` on *both* backends' pools — i.e. the batch
/// genuinely ran on `K` overlapping host threads, not serially on the one vCPU thread. The rendezvous
/// makes this independent of sleep timing (no flakiness).
#[test]
fn offload_pool_overlaps_blocking_ops_on_k_threads() {
    let k = OFFLOAD_POOL_THREADS as u64;
    let (_ival, _imem, jo, _jmem, hi, hj, h) = run_batch(k, Duration::ZERO, Some(k as usize));
    assert!(matches!(jo, JitOutcome::Returned(_)), "jit: {jo:?}");
    let imax = hi
        .blocking_state(h)
        .expect("interp blocking state")
        .max_active();
    let jmax = hj
        .blocking_state(h)
        .expect("jit blocking state")
        .max_active();
    assert_eq!(
        imax, k as usize,
        "interp: the pool must overlap all {k} blocking ops (max_active)"
    );
    assert_eq!(
        jmax, k as usize,
        "jit: the pool must overlap all {k} blocking ops (max_active)"
    );
}

/// A `Blocking` op is *also* an ordinary synchronous `cap.call` (iface 10, op 0): called directly
/// (not via the ring) it runs inline on the caller's thread and returns `mix(arg)`, identically on
/// both backends — covering the degenerate single-op path the offload pool short-circuits.
#[test]
fn blocking_direct_cap_call_runs_inline() {
    let src = "memory 17
func (i32, i32) -> (i64) {
block0(v0: i32, v1: i32):
  v2 = i64.const 7
  v3 = cap.call 10 0 (i64) -> (i64) v1 (v2)
  return v3
}
";
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");
    let init = [0u8; 128 << 10];

    let mut hi = Host::new();
    let (iri, ibi) = (hi.grant_io_ring(), hi.grant_blocking(Duration::ZERO, None));
    let mut hj = Host::new();
    let (irj, ibj) = (hj.grant_io_ring(), hj.grant_blocking(Duration::ZERO, None));
    assert_eq!((iri, ibi), (irj, ibj));

    let mut fuel = 1_000_000u64;
    let (ir, _imem) = run_capture_reserved_with_host(
        &m,
        0,
        &[Value::I32(iri), Value::I32(ibi)],
        &mut fuel,
        &init,
        0,
        &mut hi,
    );
    let (jo, _jmem) = compile_and_run_capture_reserved_with_host(
        &m,
        0,
        &[irj as i64, ibj as i64],
        &init,
        0,
        svm_run::cap_thunk,
        &mut hj as *mut Host as *mut core::ffi::c_void,
    )
    .expect("jit");
    let want = mix(7);
    assert_eq!(
        ir.expect("ok").pop(),
        Some(Value::I64(want)),
        "interp inline"
    );
    assert!(
        matches!(jo, JitOutcome::Returned(ref s) if s == &[want]),
        "jit inline: {jo:?} (want {want})"
    );
}

/// A `Blocking` SQE carrying a **forged** handle is inert: it is never queued to the pool, lands as a
/// CQE with a non-zero (`CapFault`) status and a `0` result, and still counts toward `completed` — the
/// I2 "a forged handle is inert" check, on the offload path. `submit` returns `1`; the CQE status is
/// the `CapFault` code (6).
#[test]
fn offload_forged_blocking_handle_is_inert() {
    // One SQE: type_id 10, op 0, handle = 0x7FFFFFFF (never granted), n_args 1, args[0]=3, ud=99.
    let src = "memory 17
func (i32, i32) -> (i64) {
block0(v0: i32, v1: i32):
  v2 = i32.const 10
  v3 = i64.const 0
  i32.store v3 v2
  v4 = i32.const 0
  v5 = i64.const 4
  i32.store v5 v4
  v6 = i32.const 2147483647
  v7 = i64.const 8
  i32.store v7 v6
  v8 = i32.const 1
  v9 = i64.const 12
  i32.store v9 v8
  v10 = i64.const 3
  v11 = i64.const 16
  i64.store v11 v10
  v12 = i64.const 99
  v13 = i64.const 48
  i64.store v13 v12
  v14 = i64.const 0
  v15 = i64.const 1
  v16 = i64.const 512
  v17 = cap.call 9 0 (i64, i64, i64) -> (i64) v0 (v14, v15, v16)
  v18 = i64.const 528
  v19 = i64.load v18
  return v19
}
";
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");
    let init = [0u8; 128 << 10];

    let mut hi = Host::new();
    let (iri, ibi) = (hi.grant_io_ring(), hi.grant_blocking(Duration::ZERO, None));
    let mut hj = Host::new();
    let (irj, ibj) = (hj.grant_io_ring(), hj.grant_blocking(Duration::ZERO, None));
    assert_eq!((iri, ibi), (irj, ibj));

    let mut fuel = 1_000_000u64;
    let (ir, imem) = run_capture_reserved_with_host(
        &m,
        0,
        &[Value::I32(iri), Value::I32(ibi)],
        &mut fuel,
        &init,
        0,
        &mut hi,
    );
    let (jo, jmem) = compile_and_run_capture_reserved_with_host(
        &m,
        0,
        &[irj as i64, ibj as i64],
        &init,
        0,
        svm_run::cap_thunk,
        &mut hj as *mut Host as *mut core::ffi::c_void,
    )
    .expect("jit");
    // CQE status field (offset +16) holds the CapFault code (6) — a forged handle ran nothing.
    assert_eq!(
        ir.expect("ok").pop(),
        Some(Value::I64(6)),
        "interp: forged → CapFault status"
    );
    assert!(
        matches!(jo, JitOutcome::Returned(ref s) if s == &[6]),
        "jit: forged → CapFault status: {jo:?}"
    );
    assert_eq!(imem, jmem, "interp/JIT windows must agree on the inert CQE");
}
