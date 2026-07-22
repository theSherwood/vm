//! §5 the **fuel/epoch kill-path on the JIT** — a host can stop a *runaway* guest (an infinite
//! loop / unbounded tail recursion), matching the interpreter's `Trap::OutOfFuel`. The interpreter
//! has always bounded execution via its per-step fuel counter; this proves the production backend
//! now has the matching, **guest-undisableable** kill-path: the lowering polls a host-owned
//! interrupt cell at every loop back-edge and function entry and traps `OutOfFuel` the moment the
//! host sets it. Both backends agree on the outcome for a non-terminating program: it terminates,
//! reported as OutOfFuel, rather than hanging the host thread.
//!
//! The differential here is on the **outcome**, not the window/step-count: the interpreter trips on
//! a deterministic fuel budget, the JIT on a wall-clock watchdog (the realistic host mechanism), so
//! they stop at different points — but both must stop, and both must report OutOfFuel.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use svm_interp::{run, Host, Trap, Value};
use svm_jit::{compile_and_run, compile_and_run_with_host_interruptible, JitOutcome, TrapKind};
use svm_text::parse_module;
use svm_verify::verify_module;

/// Serialize this binary's tests (the ISSUES.md I4 pattern, applied per I33): every test here
/// races a wall-clock watchdog against deliberately-runaway guest code, and sibling tests
/// competing for the process's cores distort exactly that timing — I33 recorded the runaway-child
/// leg flaking under full-workspace parallel load (twice locally, once on macOS CI, all
/// 2026-07-20) while passing consistently in isolation. A poisoned lock (an earlier test failed)
/// is fine to reuse — take the inner guard.
fn serial() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// A non-terminating **intra-function loop** (block1 branches to itself forever) — caught by the
/// per-back-edge kill-path check.
const INFINITE_LOOP: &str = "\
func (i64) -> (i64) {
block 0 (v0: i64) {
  br 1(v0)
}
block 1 (v1: i64) {
  v2 = i64.const 1
  v3 = i64.add v1 v2
  br 1(v3)
  }
}
";

/// A non-terminating **tail-recursion** (function 0 tail-calls itself forever) — runs in O(1)
/// native stack, so it never faults; only the *function-entry* kill-path check can stop it.
const INFINITE_TAIL_RECURSION: &str = "\
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 1
  v2 = i64.add v0 v1
  return_call 0(v2)
  }
}
";

/// A **finite** countdown N → 0 — used to prove an *armed-but-never-tripped* run still completes
/// correctly (the poll sees the cell stay zero every iteration and never false-trips).
const FINITE_COUNTDOWN: &str = "\
func (i32) -> (i32) {
block 0 (v0: i32) {
  br 1(v0)
}
block 1 (v1: i32) {
  v2 = i32.const -1
  v3 = i32.add v1 v2
  br_if v3 1(v3) 2(v3)
}
block 2 (v4: i32) {
  return v4
  }
}
";

/// Run a non-terminating module on the JIT with the kill-path armed by a watchdog thread that sets
/// the interrupt cell after `delay`. Returns the JIT outcome — which must be `Trapped(OutOfFuel)`,
/// reached within a bounded time of the watchdog firing (else the mechanism is broken and the test
/// would hang — a CI timeout is then the failure signal).
fn jit_with_watchdog(src: &str, delay: Duration) -> JitOutcome {
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");

    let interrupt = Arc::new(AtomicU64::new(0));
    let wd = interrupt.clone();
    let watchdog = std::thread::spawn(move || {
        std::thread::sleep(delay);
        wd.store(1, Ordering::SeqCst); // request the kill
    });

    // The guest makes no `cap.call`, so the thunk is never invoked — a null ctx is never read.
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

#[test]
fn jit_killpath_stops_infinite_loop() {
    let _serial = serial();
    // Interp: a small fuel budget bounds the infinite loop → OutOfFuel.
    let m = parse_module(INFINITE_LOOP).expect("parse");
    verify_module(&m).expect("verify");
    let mut fuel = 100_000u64;
    let interp = run(&m, 0, &[Value::I64(0)], &mut fuel);
    assert!(
        matches!(interp, Err(Trap::OutOfFuel)),
        "interp must bound the infinite loop, got {interp:?}"
    );

    // JIT: a watchdog arms the kill-path; the per-back-edge poll trips it → OutOfFuel.
    let jit = jit_with_watchdog(INFINITE_LOOP, Duration::from_millis(100));
    assert_eq!(
        jit,
        JitOutcome::Trapped(TrapKind::OutOfFuel),
        "JIT must stop the runaway loop with OutOfFuel"
    );
}

#[test]
fn jit_killpath_stops_infinite_tail_recursion() {
    let _serial = serial();
    // Interp: fuel bounds the unbounded tail recursion → OutOfFuel.
    let m = parse_module(INFINITE_TAIL_RECURSION).expect("parse");
    verify_module(&m).expect("verify");
    let mut fuel = 100_000u64;
    let interp = run(&m, 0, &[Value::I64(0)], &mut fuel);
    assert!(
        matches!(interp, Err(Trap::OutOfFuel)),
        "interp must bound the tail recursion, got {interp:?}"
    );

    // JIT: the *function-entry* check (not the back-edge one) is what catches this — tail calls
    // never grow the stack, so without it the guest would spin forever.
    let jit = jit_with_watchdog(INFINITE_TAIL_RECURSION, Duration::from_millis(100));
    assert_eq!(
        jit,
        JitOutcome::Trapped(TrapKind::OutOfFuel),
        "JIT must stop the runaway tail recursion with OutOfFuel"
    );
}

#[test]
fn jit_armed_finite_run_completes_normally() {
    let _serial = serial();
    // Arm the kill-path on a *finite* program whose watchdog is set far enough out that it never
    // fires before the program finishes: the per-iteration poll sees the cell stay zero and never
    // false-trips, so the run returns its real result. (Also the interp/JIT agree on that result.)
    let m = parse_module(FINITE_COUNTDOWN).expect("parse");
    verify_module(&m).expect("verify");

    let mut fuel = 10_000_000u64;
    let interp = run(&m, 0, &[Value::I32(1000)], &mut fuel).expect("interp ok");
    assert_eq!(interp, vec![Value::I32(0)], "countdown returns 0");

    let interrupt = Arc::new(AtomicU64::new(0)); // armed, but we never set it
    let jit = compile_and_run_with_host_interruptible(
        &m,
        0,
        &[1000i64],
        svm_run::cap_thunk,
        core::ptr::null_mut(),
        Arc::as_ptr(&interrupt),
    )
    .expect("jit compiles");
    assert_eq!(
        jit,
        JitOutcome::Returned(vec![0]),
        "an armed-but-untripped finite run must complete normally"
    );
}

#[test]
fn jit_unarmed_path_is_unchanged() {
    let _serial = serial();
    // Sanity: the ordinary (kill-path-not-armed) entry still runs the same finite program to
    // completion — arming is strictly opt-in, so existing call sites are unaffected.
    let m = parse_module(FINITE_COUNTDOWN).expect("parse");
    verify_module(&m).expect("verify");
    let jit = compile_and_run(&m, 0, &[1000i64]).expect("jit");
    assert_eq!(jit, JitOutcome::Returned(vec![0]));
}

/// A §14 parent that instantiates func 1 as a nested child (64 KiB carve at offset 0) and `join`s it.
/// The child **spins forever**, so a host kill must reach *into the child* — the child polls the
/// **parent's** interrupt cell (it is compiled with the same baked address), trips `OutOfFuel`,
/// `join` propagates it, and the parent unwinds. Without the child polling that cell, the synchronous
/// `instantiate` never returns and the whole run hangs.
const PARENT_WITH_RUNAWAY_CHILD: &str = "\
memory 17
func (i32) -> (i64) {
block 0 (v0: i32) {
  v1 = i64.const 1
  v2 = i64.const 0
  v3 = i64.const 16
  v4 = i64.const 0
  v5 = cap.call 6 0 (i64, i64, i64, i64) -> (i32) v0 (v1, v2, v3, v4)
  v6 = cap.call 6 1 (i32) -> (i64) v0 (v5)
  return v6
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  br 1(v0)
}
block 1 (v1: i64) {
  v2 = i64.const 1
  v3 = i64.add v1 v2
  br 1(v3)
  }
}
";

#[test]
fn jit_killpath_stops_runaway_child() {
    let _serial = serial();
    if !svm_jit::fiber_supported() {
        return; // no JIT nesting runtime here — an instantiate is an inert CapFault, not a child run
    }
    let m = parse_module(PARENT_WITH_RUNAWAY_CHILD).expect("parse");
    verify_module(&m).expect("verify");

    let win = 1u64 << 17;
    let mut host = Host::new();
    let inst = host.grant_instantiator(0, win); // the parent's nesting authority over the window

    let interrupt = Arc::new(AtomicU64::new(0));
    let wd = interrupt.clone();
    let watchdog = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(100));
        wd.store(1, Ordering::SeqCst);
    });
    let outcome = compile_and_run_with_host_interruptible(
        &m,
        0,
        &[inst as i64],
        svm_run::cap_thunk,
        &mut host as *mut Host as *mut core::ffi::c_void,
        Arc::as_ptr(&interrupt),
    )
    .expect("jit compiles");
    watchdog.join().unwrap();

    assert_eq!(
        outcome,
        JitOutcome::Trapped(TrapKind::OutOfFuel),
        "a runaway nested JIT child must be killed (and not hang the parent's instantiate)"
    );
}
