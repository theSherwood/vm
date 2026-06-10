//! §5 the **kill-path across sibling vCPUs** — one host interrupt stops a whole *multithreaded*
//! runaway domain, not just its busy threads.
//!
//! Every vCPU runs the same finalized code, so a **spinning** sibling already polls the one baked
//! interrupt cell and trips `OutOfFuel` on its own. The subtle case is a **parked** sibling —
//! blocked in a futex `wait` or `thread.join` — which isn't executing guest code and so can't poll:
//! `os_thread_rt` re-checks the interrupt cell on a bounded interval while parked, so it wakes and
//! unwinds too. Without that, `join_all` at teardown would hang forever on a vCPU that never
//! finishes. These tests would *hang* (CI timeout) if the mechanism regressed; passing means the
//! whole domain terminates as `OutOfFuel`.
//!
//! Gated to the targets with the JIT thread runtime (`os_thread_rt`).
#![cfg(any(
    all(unix, target_arch = "x86_64"),
    all(unix, target_arch = "aarch64"),
    all(windows, target_arch = "x86_64")
))]

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use svm_jit::{compile_and_run_with_host_interruptible, JitOutcome, TrapKind};
use svm_text::parse_module;
use svm_verify::verify_module;

/// Compile `src`, arm the kill-path, and let a watchdog request the kill after `delay_ms`. Returns
/// the outcome — which must be `Trapped(OutOfFuel)` once *every* vCPU (the root + its spawned
/// siblings) has unwound (`run_inner` joins them all before returning).
fn run_killed(src: &str, delay_ms: u64) -> JitOutcome {
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");
    let interrupt = Arc::new(AtomicU64::new(0));
    let wd = interrupt.clone();
    let watchdog = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(delay_ms));
        wd.store(1, Ordering::SeqCst); // request the domain kill
    });
    // The guest makes no `cap.call`, so the thunk is never invoked (null ctx never read).
    let outcome = compile_and_run_with_host_interruptible(
        &m,
        0,
        &[0i64],
        svm_run::cap_thunk,
        core::ptr::null_mut(),
        Arc::as_ptr(&interrupt),
    )
    .expect("jit compiles");
    watchdog.join().unwrap();
    outcome
}

/// The root vCPU spawns a sibling that **spins**, then spins itself: both poll the shared baked
/// interrupt cell, so the single host kill stops both and the run reports `OutOfFuel`.
#[test]
fn killpath_stops_spinning_sibling() {
    let src = "memory 16
func (i64) -> (i64) {
block0(v0: i64):
  v1 = thread.spawn 1 v0 v0
  br block1(v0)
block1(v2: i64):
  v3 = i64.const 1
  v4 = i64.add v2 v3
  br block1(v4)
}
func (i64, i64) -> (i64) {
block0(v0: i64, v1: i64):
  br block1(v0)
block1(v2: i64):
  v3 = i64.const 1
  v4 = i64.add v2 v3
  br block1(v4)
}
";
    assert_eq!(
        run_killed(src, 100),
        JitOutcome::Trapped(TrapKind::OutOfFuel),
        "a multithreaded spinning domain must be killed whole"
    );
}

/// The root vCPU spawns a sibling that parks in an **infinite futex `wait`** (timeout `-1`, never
/// notified), then the root spins. The root trips on the interrupt; the *parked* sibling has no
/// guest code running to poll it, so the bounded re-check while parked is what wakes it — it then
/// unwinds (and trips `OutOfFuel` in the spin it falls into). The run terminates; without the
/// re-check, `join_all` would hang on the parked sibling forever.
#[test]
fn killpath_wakes_parked_futex_waiter() {
    let src = "memory 16
func (i64) -> (i64) {
block0(v0: i64):
  v1 = thread.spawn 1 v0 v0
  br block1(v0)
block1(v2: i64):
  v3 = i64.const 1
  v4 = i64.add v2 v3
  br block1(v4)
}
func (i64, i64) -> (i64) {
block0(v0: i64, v1: i64):
  v2 = i64.const 0
  v3 = i32.const 0
  v4 = i64.const -1
  v5 = i32.atomic.wait v2 v3 v4
  br block1(v0)
block1(v6: i64):
  v7 = i64.const 1
  v8 = i64.add v6 v7
  br block1(v8)
}
";
    assert_eq!(
        run_killed(src, 100),
        JitOutcome::Trapped(TrapKind::OutOfFuel),
        "a sibling parked in an infinite futex wait must be woken and killed"
    );
}
