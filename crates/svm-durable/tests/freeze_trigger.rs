//! The **mid-run freeze trigger** (DURABILITY.md §12, "freeze after N safepoints"). The freeze
//! mechanism unwinds at the first poll that sees `UNWINDING`; the *before-start* harness sets that
//! word before the run, so it freezes at the very first safepoint. `arm_freeze_after` instead lets
//! the run make forward progress and promotes the word to `UNWINDING` at the N-th **fiber safepoint**
//! (`cont.resume`/`suspend`), so the freeze lands *mid-run* — deterministically, in a single-threaded
//! test. Here the root makes an observable host-fn call before each fiber interaction, so the number
//! of calls that fired before the freeze pins where it landed. (cap.call is *not* a counted safepoint
//! — only the fiber ops are, so the trigger point is identical on the interpreter and the JIT.)

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use svm_durable::{
    arm_freeze_after, init_durable_window, read_state, transform_module, STATE_UNWINDING,
};
use svm_interp::{run_capture_reserved_with_host, Host, Value};
use svm_ir::{Memory, Module};

const SIZE_LOG2: u8 = 18;
const WINDOW: usize = 1 << SIZE_LOG2;

// Root (v0 = host-fn handle): a host-fn call (observable, *not* a safepoint) before each fiber
// interaction, around three resumes of a fiber that suspends twice then returns. The five fiber
// safepoints, in order, are: resume#1, suspend#1, resume#2, suspend#2, resume#3.
const SRC: &str = "memory 18\n\
    func (i32) -> (i64) {\n\
    block0(v0: i32):\n\
    \x20 v1 = ref.func 1\n\
    \x20 v2 = i64.const 4096\n\
    \x20 v3 = cont.new v1 v2\n\
    \x20 v4 = cap.call 13 0 () -> (i64) v0 ()\n\
    \x20 v5 = i64.const 0\n\
    \x20 v6, v7 = cont.resume v3 v5\n\
    \x20 v8 = cap.call 13 0 () -> (i64) v0 ()\n\
    \x20 v9, v10 = cont.resume v3 v5\n\
    \x20 v11 = cap.call 13 0 () -> (i64) v0 ()\n\
    \x20 v12, v13 = cont.resume v3 v5\n\
    \x20 v14 = cap.call 13 0 () -> (i64) v0 ()\n\
    \x20 return v14\n\
    }\n\
    func (i64, i64) -> (i64) {\n\
    block0(v0: i64, v1: i64):\n\
    \x20 v2 = i64.const 1\n\
    \x20 v3 = suspend v2\n\
    \x20 v4 = i64.const 2\n\
    \x20 v5 = suspend v4\n\
    \x20 v6 = i64.const 0\n\
    \x20 return v6\n\
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

// Run armed to freeze after `n` fiber safepoints; returns (observable host-fn calls, froze?).
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
fn arming_freezes_mid_run_after_n_fiber_safepoints() {
    // Each resume cycle is two fiber safepoints (resume + suspend), so progress steps every two
    // counts; the host-fn the *root* runs between resumes is the observable. Arm-after-1 freezes at
    // the very first fiber safepoint (only the first host-fn, before resume#1, has fired).
    assert_eq!(run_armed(1), (1, true), "freeze at the 1st fiber safepoint");
    // Arming deeper lets the run make real forward progress before freezing — exactly what the
    // freeze-before-start harness cannot reach.
    assert_eq!(
        run_armed(3),
        (2, true),
        "freeze at the 3rd (one more cycle)"
    );
    assert_eq!(run_armed(5), (3, true), "freeze at the 5th (last resume)");
}

#[test]
fn arming_past_the_last_safepoint_runs_to_completion() {
    // More fiber safepoints requested than the program has (5): the countdown never reaches 0, the
    // state word stays ARMED (which every poll reads as NORMAL), and the run completes — no freeze,
    // all four host-fn calls fire.
    assert_eq!(run_armed(99), (4, false), "no freeze; ran to completion");
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
    assert_eq!(calls.load(Ordering::Relaxed), 4, "ran to completion");
    assert_ne!(
        read_state(&snap),
        STATE_UNWINDING,
        "never entered UNWINDING"
    );
}
