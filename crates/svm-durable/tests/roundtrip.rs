//! Phase-1 go/no-go: a frozen domain round-trips on the real interpreter.
//!
//! The property (DURABILITY.md §7, §12.6):
//!
//! > freeze → serialize window → restore → thaw → run-to-end  ≡  uninterrupted run
//!
//! We drive it with `svm_interp::run_capture_reserved_with_host`, which seeds the
//! window with bytes and returns the final window image — exactly the
//! serialize/restore primitive the snapshot format needs (the durable region rides
//! along in those bytes, DURABILITY.md §12.0).

use svm_durable::{
    begin_thaw, init_durable_window, read_state, transform_module, write_state, STATE_UNWINDING,
};
use svm_interp::{run_capture_reserved_with_host, Host, Value};
use svm_ir::Memory;

const SIZE_LOG2: u8 = 18; // 256 KiB window, fully mapped (reserved == mapped)
const WINDOW: usize = 1 << SIZE_LOG2;

// A guest that calls Clock.now (the may-suspend `cap.call`) and then *uses the result*
// after the call — so a buggy thaw that re-issued the call instead of reloading the
// saved value would be observable.
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

fn module() -> svm_ir::Module {
    let mut m = svm_text::parse_module(SRC).expect("parse");
    m.memory = Some(Memory {
        size_log2: SIZE_LOG2,
    });
    transform_module(&m).expect("transform")
}

#[test]
fn freeze_serialize_restore_thaw_equals_uninterrupted_run() {
    let inst = module();
    svm_verify::verify_module(&inst).expect("verify");

    // ---- Baseline: the uninterrupted run (the oracle). Clock seeded at 42. ----
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

    // ---- Freeze: seed UNWINDING; the poll after the call unwinds out to the host. ----
    let mut host = Host::new();
    host.clock_ns = 42; // same initial conditions as the baseline
    let clk = host.grant_clock();
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
        &mut host,
    );
    // The entry returned a placeholder while still UNWINDING — it froze, it did not finish.
    assert_eq!(
        frozen,
        Ok(vec![Value::I64(0)]),
        "freeze returns a placeholder"
    );
    assert_eq!(
        read_state(&snapshot),
        STATE_UNWINDING,
        "state word is still UNWINDING in the artifact (the stack unwound, not completed)"
    );

    // ---- Restore + thaw: a FRESH host (D-scope: host resources are not in the ----
    // artifact). Clock would now return 0 — so if thaw re-issued the call instead of
    // reloading the saved 42, the result would be 100, not 142.
    let mut win = snapshot.clone();
    begin_thaw(&mut win, 0);
    let mut host = Host::new(); // clock_ns defaults to 0
    let clk = host.grant_clock();
    let mut fuel = 100_000u64;
    let (thawed, _) = run_capture_reserved_with_host(
        &inst,
        0,
        &[Value::I32(clk)],
        &mut fuel,
        &win,
        SIZE_LOG2,
        &mut host,
    );

    assert_eq!(
        thawed, baseline,
        "thawed run is bytewise-equivalent to the uninterrupted run"
    );
    assert_eq!(
        thawed,
        Ok(vec![Value::I64(142)]),
        "the saved cap result (42) was reloaded, not re-issued (which would give 100)"
    );
}

#[test]
fn normal_run_of_instrumented_module_matches_unmodified_behavior() {
    // With the state word left NORMAL, the instrumentation is inert: the prologue and
    // poll fall straight through, so the instrumented module behaves like the original.
    let inst = module();
    let mut host = Host::new();
    host.clock_ns = 7;
    let clk = host.grant_clock();
    let mut fuel = 100_000u64;
    let (r, _) = run_capture_reserved_with_host(
        &inst,
        0,
        &[Value::I32(clk)],
        &mut fuel,
        &init_durable_window(WINDOW),
        SIZE_LOG2,
        &mut host,
    );
    assert_eq!(
        r,
        Ok(vec![Value::I64(107)]),
        "7 + 100, instrumentation inert"
    );
}
