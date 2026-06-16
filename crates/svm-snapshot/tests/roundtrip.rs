//! The snapshot codec end-to-end: freeze → **serialize** → restore → thaw ≡ uninterrupted run,
//! through the real §12 artifact (not a kept-in-memory window), plus the two §12.6 invariants
//! (canonical re-serialize; identity-gated refusal) and the non-durable freeze refusal.

use svm_durable::{
    init_durable_window, transform_module, write_state, STATE_REWINDING, STATE_UNWINDING,
};
use svm_interp::{run_capture_reserved_with_host, Host, Value};
use svm_ir::{Memory, Module};
use svm_snapshot::{freeze, restore, FreezeError, RestoreError};

const SIZE_LOG2: u8 = 18;
const WINDOW: usize = 1 << SIZE_LOG2;

// Calls Clock.now and *uses* the result after the call, so a thaw that re-issued the call
// (clock now 0) instead of reloading the saved 42 would be observable (100 vs 142).
const SRC: &str = r#"
func (i32) -> (i64) {
block0(v0: i32):
  v1 = i32.const 0
  v2 = cap.call 2 0 (i32) -> (i64) v0 (v1)
  v3 = i64.const 100
  v4 = i64.add v2 v3
  return v4
}
"#;

// A *different* instrumented module (adds 200), for the identity-gate test.
const SRC_OTHER: &str = r#"
func (i32) -> (i64) {
block0(v0: i32):
  v1 = i32.const 0
  v2 = cap.call 2 0 (i32) -> (i64) v0 (v1)
  v3 = i64.const 200
  v4 = i64.add v2 v3
  return v4
}
"#;

fn instrument(src: &str) -> Module {
    let mut m = svm_text::parse_module(src).expect("parse");
    m.memory = Some(Memory {
        size_log2: SIZE_LOG2,
    });
    let inst = transform_module(&m).expect("transform");
    svm_verify::verify_module(&inst).expect("verify");
    inst
}

#[test]
fn freeze_serialize_restore_thaw_through_the_codec() {
    let inst = instrument(SRC);

    // Baseline: uninterrupted run, clock at 42.
    let mut host = Host::new();
    host.clock_ns = 42;
    let clk = host.grant_clock();
    let mut fuel = 100_000u64;
    let (baseline, _) = run_capture_reserved_with_host(
        &inst,
        0,
        &[Value::I32(clk)],
        &mut fuel,
        &init_durable_window(WINDOW),
        SIZE_LOG2,
        &mut host,
    );
    assert_eq!(
        baseline,
        Ok(vec![Value::I64(142)]),
        "uninterrupted: 42 + 100"
    );

    // Freeze run: UNWINDING → the poll unwinds out, leaving shadow state in the window.
    let mut fhost = Host::new();
    fhost.clock_ns = 42;
    let clk = fhost.grant_clock();
    let mut win = init_durable_window(WINDOW);
    write_state(&mut win, STATE_UNWINDING);
    let mut fuel = 100_000u64;
    let (frozen, snapshot) = run_capture_reserved_with_host(
        &inst,
        0,
        &[Value::I32(clk)],
        &mut fuel,
        &win,
        SIZE_LOG2,
        &mut fhost,
    );
    assert_eq!(
        frozen,
        Ok(vec![Value::I64(0)]),
        "freeze returns a placeholder"
    );

    // Serialize the real artifact.
    let artifact = freeze(&inst, &snapshot, &fhost).expect("freeze");

    // Restore into a FRESH host (clock now 0 — D-scope: resources aren't in the artifact).
    let mut thost = Host::new();
    let window = restore(&artifact, &inst, &mut thost).expect("restore");

    // §12.6 invariant 1 — canonical: re-serializing the freshly-restored domain at the same
    // safepoint reproduces the artifact byte-for-byte.
    assert_eq!(
        freeze(&inst, &window, &thost).expect("re-freeze"),
        artifact,
        "re-serialize of a restored domain is byte-identical"
    );

    // Thaw: flip to REWINDING and re-enter; the stack rebuilds from the restored window. The
    // guest receives the clock as a handle argument; restore reinstated it at its original
    // slot/generation, so the same handle value (`(generation << 8) | slot`) still resolves.
    let mut win = window;
    write_state(&mut win, STATE_REWINDING);
    let caps = thost.capture_durable_handles().expect("durable");
    let clk = ((caps[0].generation << 8) | caps[0].slot) as i32;
    let mut fuel = 100_000u64;
    let (thawed, _) = run_capture_reserved_with_host(
        &inst,
        0,
        &[Value::I32(clk)],
        &mut fuel,
        &win,
        SIZE_LOG2,
        &mut thost,
    );
    assert_eq!(thawed, baseline, "thawed run equals the uninterrupted run");
    assert_eq!(
        thawed,
        Ok(vec![Value::I64(142)]),
        "saved cap result (42) reloaded, not re-issued (which would give 100)"
    );
}

#[test]
fn restore_refuses_a_mismatched_module() {
    // Freeze under SRC, then try to restore against a different instrumented module.
    let inst = instrument(SRC);
    let mut host = Host::new();
    host.grant_clock();
    let win = init_durable_window(WINDOW);
    let artifact = freeze(&inst, &win, &host).expect("freeze");

    let other = instrument(SRC_OTHER);
    let mut thost = Host::new();
    let err = restore(&artifact, &other, &mut thost).expect_err("digest mismatch must refuse");
    assert_eq!(err, RestoreError::ModuleMismatch, "R5 identity gate");
}

#[test]
fn freeze_refuses_a_non_durable_handle() {
    let inst = instrument(SRC);
    let mut host = Host::new();
    host.grant_clock();
    host.grant_io_ring(); // non-durable (carries out-of-line ring state)
    let win = init_durable_window(WINDOW);
    match freeze(&inst, &win, &host) {
        Err(FreezeError::NonDurableHandle(h)) => assert_eq!(h.slot, 1),
        other => panic!("expected NonDurableHandle refusal, got {other:?}"),
    }
}
