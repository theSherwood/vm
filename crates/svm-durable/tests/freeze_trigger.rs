//! The **mid-run freeze trigger** (DURABILITY.md §12, "freeze after N safepoints"). The freeze
//! mechanism unwinds at the first poll that sees `UNWINDING`; the *before-start* harness sets that
//! word before the run, so it freezes at the very first safepoint. `arm_freeze_after` instead lets
//! the run make forward progress and promotes the word to `UNWINDING` at the N-th safepoint, so the
//! freeze lands *mid-run* — deterministically, in a single-threaded test. This pins where the freeze
//! lands by counting a host-fn's invocations (one per `cap.call` safepoint executed before the freeze).

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use svm_durable::{
    arm_freeze_after, init_durable_window, read_state, transform_module, STATE_UNWINDING,
};
use svm_interp::{run_capture_reserved_with_host, Host, Value};
use svm_ir::{Memory, Module};

const SIZE_LOG2: u8 = 18;
const WINDOW: usize = 1 << SIZE_LOG2;

// Three back-to-back cap.calls (three safepoints); the result is unobservable here — we only count
// how many executed before the freeze, via the host fn.
const SRC: &str = "memory 18\n\
    func (i32) -> (i64) {\n\
    block0(v0: i32):\n\
    \x20 v1 = cap.call 13 0 () -> (i64) v0 ()\n\
    \x20 v2 = cap.call 13 0 () -> (i64) v0 ()\n\
    \x20 v3 = cap.call 13 0 () -> (i64) v0 ()\n\
    \x20 v4 = i64.add v1 v2\n\
    \x20 v5 = i64.add v4 v3\n\
    \x20 return v5\n\
    }\n";

fn instrument() -> Module {
    let mut m = svm_text::parse_module(SRC).expect("parse");
    m.memory = Some(Memory {
        size_log2: SIZE_LOG2,
    });
    let inst = transform_module(&m).expect("transform");
    svm_verify::verify_module(&inst).expect("verify");
    inst
}

// Run armed to freeze after `n` safepoints; returns (host-fn calls executed, froze?).
fn run_armed(n: i64) -> (u64, bool) {
    let inst = instrument();
    let calls = Arc::new(AtomicU64::new(0));
    let sink = Arc::clone(&calls);
    let mut host = Host::new();
    host.set_durable(true);
    let hf = host.grant_host_fn(Box::new(move |_op, _args, _mem| {
        sink.fetch_add(1, Ordering::Relaxed);
        Ok(vec![0])
    }));

    let mut win = init_durable_window(WINDOW);
    arm_freeze_after(&mut win, n);
    let mut fuel = 100_000u64;
    let (res, snap) = run_capture_reserved_with_host(
        &inst,
        0,
        &[Value::I32(hf)],
        &mut fuel,
        &win,
        SIZE_LOG2,
        &mut host,
    );
    assert!(res.is_ok(), "armed run trapped: {res:?}");
    (
        calls.load(Ordering::Relaxed),
        read_state(&snap) == STATE_UNWINDING,
    )
}

#[test]
fn arming_freezes_after_exactly_n_safepoints() {
    // Arm-after-1 is the first-safepoint freeze (one cap.call executes, then its poll unwinds) —
    // the before-start harness reached *only* this; arming reaches it during the run.
    let (calls1, froze1) = run_armed(1);
    assert!(froze1, "armed run froze (state left UNWINDING)");
    assert_eq!(calls1, 1, "arm-after-1 freezes at the first safepoint");

    // Arm-after-2 makes real forward progress first: two cap.calls run before the freeze — the
    // capability the freeze-before-start harness could never reach.
    let (calls2, froze2) = run_armed(2);
    assert!(froze2, "armed run froze");
    assert_eq!(
        calls2, 2,
        "arm-after-2 freezes only at the second safepoint"
    );

    // Arm-after-3 freezes at the last safepoint.
    let (calls3, froze3) = run_armed(3);
    assert!(froze3, "armed run froze");
    assert_eq!(calls3, 3, "arm-after-3 freezes at the third safepoint");
}

#[test]
fn arming_past_the_last_safepoint_runs_to_completion() {
    // More safepoints requested than the program has: the countdown never reaches 0, the state word
    // stays ARMED (which every poll reads as NORMAL), and the run completes normally — no freeze.
    let (calls, froze) = run_armed(99);
    assert!(!froze, "the run completed (never promoted to UNWINDING)");
    assert_eq!(calls, 3, "all three safepoints executed");
}

#[test]
fn an_unarmed_durable_run_is_untouched() {
    // No arming: a plain durable run completes, touches none of the trigger machinery, and leaves
    // the (NORMAL) state word alone.
    let inst = instrument();
    let calls = Arc::new(AtomicU64::new(0));
    let sink = Arc::clone(&calls);
    let mut host = Host::new();
    host.set_durable(true);
    let hf = host.grant_host_fn(Box::new(move |_op, _args, _mem| {
        sink.fetch_add(1, Ordering::Relaxed);
        Ok(vec![0])
    }));
    let win = init_durable_window(WINDOW);
    let mut fuel = 100_000u64;
    let (res, snap) = run_capture_reserved_with_host(
        &inst,
        0,
        &[Value::I32(hf)],
        &mut fuel,
        &win,
        SIZE_LOG2,
        &mut host,
    );
    assert!(res.is_ok(), "unarmed durable run trapped: {res:?}");
    assert_eq!(calls.load(Ordering::Relaxed), 3, "ran to completion");
    assert_ne!(
        read_state(&snap),
        STATE_UNWINDING,
        "never entered UNWINDING"
    );
}
