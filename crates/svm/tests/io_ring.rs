//! §9/§12 the **submit/complete ring** (io_uring-shaped), increment 1 — synchronous batched
//! `cap.call`s. The guest writes `n` 64-byte SQEs (each a *deferred `cap.call`*) into its window,
//! submits the whole batch with **one** `cap.call` on its `IoRing` handle (the boundary-crossing
//! amortization, §1a), and reaps `n` 32-byte CQEs. Because an SQE routes through the *same*
//! capability dispatch as a direct call, the JIT gets the ring for free (a generic `cap.call` through
//! the host thunk), so the whole thing is differentially tested interp↔JIT (the §18 oracle). The
//! result is identical to issuing the ops directly — that's the correctness claim.

use svm_interp::{run_capture_reserved_with_host, Host, Value};
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
