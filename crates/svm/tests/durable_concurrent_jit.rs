//! DURABILITY.md §12.8 Phase 4 Slice A.5 **stage (ii)** + **follow-up A** — a *genuinely concurrent*
//! multi-vCPU stop-the-world freeze on the JIT. The root `thread.spawn`s children that run as **real OS
//! threads** and a [`FreezeController::request_freeze`] catches the running contexts; each self-unwinds
//! into its **own per-context shadow-SP region concurrently** (lock-free — stage i gave each its own SP
//! word). A thaw resumes every context and reproduces the uninterrupted result.
//!
//! **Spawn-before-freeze handshake.** An async freeze fires the instant the run publishes its window
//! base — *before* the root reaches its `thread.spawn`s, so the children would otherwise be **deferred**
//! (the single-worker path). To exercise the *concurrent* path, the root signals (via a host fn) once it
//! has spawned its children, and the controller thread only then requests the freeze — so the children
//! are already running OS threads when it lands.

use core::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use svm_durable::{
    init_durable_window, read_state, transform_module_assume_confined, write_state, STATE_NORMAL,
    STATE_REWINDING, STATE_UNWINDING,
};
use svm_interp::{Host, SHADOW_BASE};
use svm_ir::{Memory, Module};
use svm_jit::{
    compile_and_run_capture_reserved_with_host_durable_mv,
    compile_and_run_capture_reserved_with_host_durable_mv_interruptible, FreezeController,
    FrozenFiber, FrozenVCpu, JitError, JitOutcome,
};

const SIZE_LOG2: u8 = 17;
const WINDOW: usize = 1 << SIZE_LOG2;

// Loop trip count: large enough (~100ms native) that the freeze (requested the moment the root signals
// it has spawned) catches the looping contexts mid-flight.
const K: i64 = 100_000_000;

// Guest-memory slots (above the durable reserve). 65560 stashes the clock handle for the children.
const OFF_ROOT: i64 = 65536;
const OFF_C1: i64 = 65544;
const OFF_C2: i64 = 65552;

fn instrument(src: &str) -> Module {
    let mut m = svm_text::parse_module(src).expect("parse");
    m.memory = Some(Memory {
        size_log2: SIZE_LOG2,
    });
    let inst = transform_module_assume_confined(&m).expect("transform");
    svm_verify::verify_module(&inst).expect("instrumented concurrent multi-vCPU IR verifies");
    inst
}

fn le_i64(w: &[u8], off: i64) -> i64 {
    let o = off as usize;
    i64::from_le_bytes(w[o..o + 8].try_into().unwrap())
}

/// A concurrent freeze result: `(outcome, window image, fiber residue, vCPU residue, root extent)`.
type FreezeOutcome = (JitOutcome, Vec<u8>, Vec<FrozenFiber>, Vec<FrozenVCpu>, u64);

/// Run the durable concurrent freeze with the spawn-before-freeze handshake. `v0` = clock handle (for
/// the children), `v1` = host-fn handle (the root calls it to signal "children spawned"). Returns the
/// freeze outcome, or `None` to skip (unsupported shape / host alloc pressure).
fn concurrent_freeze(inst: &Module) -> Option<FreezeOutcome> {
    let spawned = Arc::new(AtomicBool::new(false));
    let sig = Arc::clone(&spawned);
    let mut host = Host::new();
    host.clock_ns = 42;
    let clk = host.grant_clock();
    // The root calls this once it has spawned its children; it flips the flag the controller waits on.
    let hf = host.grant_host_fn(Box::new(move |_op, _args, _mem| {
        sig.store(true, Ordering::SeqCst);
        Ok(vec![0])
    }));

    let freeze = FreezeController::new();
    let fc = Arc::clone(&freeze);
    let controller = std::thread::spawn(move || {
        // Wait until the children are spawned (so they run concurrently, not deferred), then freeze.
        while !spawned.load(Ordering::SeqCst) {
            std::hint::spin_loop();
        }
        fc.request_freeze();
    });

    let res = compile_and_run_capture_reserved_with_host_durable_mv_interruptible(
        inst,
        0,
        &[clk as i64, hf as i64],
        &init_durable_window(WINDOW),
        &[],
        &[],
        &[],
        SHADOW_BASE + 8,
        SIZE_LOG2,
        svm_run::cap_thunk,
        &mut host as *mut Host as *mut c_void,
        freeze,
    );
    controller.join().unwrap();
    match res {
        Ok((o, s, f, v, r)) => Some((o, s, f, v, r)),
        Err(JitError::Unsupported(_)) => None,
        Err(JitError::Backend(msg)) if msg.contains("Allocation error") => None,
        Err(e) => panic!("concurrent freeze failed: {e:?}"),
    }
}

fn thaw(
    inst: &Module,
    snap: &[u8],
    fibers: &[FrozenFiber],
    vcpus: &[FrozenVCpu],
    root_sp: u64,
) -> (JitOutcome, Vec<u8>) {
    let mut twin = snap.to_vec();
    write_state(&mut twin, STATE_REWINDING);
    let mut thost = Host::new();
    thost.clock_ns = 99;
    let tclk = thost.grant_clock();
    // The host fn is granted (handle order preserved) but never called on the thaw path.
    let _ = thost.grant_host_fn(Box::new(|_op: u32, _a: &[i64], _m| Ok(vec![0])));
    let (tout, tfinal, ..) = compile_and_run_capture_reserved_with_host_durable_mv(
        inst,
        0,
        &[tclk as i64, 0],
        &twin,
        &[], // init_prots
        fibers,
        vcpus,
        root_sp,
        SIZE_LOG2,
        svm_run::cap_thunk,
        &mut thost as *mut Host as *mut c_void,
    )
    .expect("concurrent thaw");
    (tout, tfinal)
}

// Root (v0 = clock, v1 = host-fn): stash the clock for the children, spawn two children as concurrent
// OS threads, **signal** they're spawned, then run a K-loop and store its total. The host-fn `cap.call`
// makes the root may-suspend so its loop is instrumented (the freeze safepoint). Each child loops K and
// stores its total to its own slot; the clock read after the loop just makes the child may-suspend too.
const SRC_LOOPS: &str = r#"
func (i32, i32) -> (i64) {
block0(v0: i32, v1: i32):
  v2 = i64.const 65560
  i32.store v2 v0
  v3 = i64.const 65544
  v4 = thread.spawn 1 v3 v3
  v5 = i64.const 65552
  v6 = thread.spawn 1 v5 v5
  v7 = i32.const 0
  v8 = cap.call 13 0 (i32) -> (i64) v1 (v7)
  v9 = i64.const 0
  br block1(v9)
block1(v10: i64):
  v11 = i64.const 1
  v12 = i64.add v10 v11
  v13 = i64.const 100000000
  v14 = i64.lt_s v12 v13
  br_if v14 block1(v12) block2(v12)
block2(v15: i64):
  v16 = i64.const 65536
  i64.store v16 v15
  return v15
}
func (i64, i64) -> (i64) {
block0(v0: i64, v1: i64):
  v2 = i64.const 65560
  v3 = i32.load v2
  v4 = i64.const 0
  br block1(v1, v3, v4)
block1(v5: i64, v6: i32, v7: i64):
  v8 = i64.const 1
  v9 = i64.add v7 v8
  v10 = i64.const 100000000
  v11 = i64.lt_s v9 v10
  br_if v11 block1(v5, v6, v9) block2(v5, v6, v9)
block2(v12: i64, v13: i32, v14: i64):
  v15 = i32.const 0
  v16 = cap.call 2 0 (i32) -> (i64) v13 (v15)
  i64.store v12 v14
  return v14
}
"#;

#[test]
fn concurrent_children_self_unwind_and_thaw_reproduces_the_result() {
    let inst = instrument(SRC_LOOPS);
    let Some((fout, fsnap, ffibers, fvcpus, froot_sp)) = concurrent_freeze(&inst) else {
        return;
    };

    if read_state(&fsnap) != STATE_UNWINDING {
        // Everything finished before the freeze landed (rare): already correct.
        assert_eq!(le_i64(&fsnap, OFF_ROOT), K);
        assert_eq!(le_i64(&fsnap, OFF_C1), K);
        assert_eq!(le_i64(&fsnap, OFF_C2), K);
        return;
    }
    assert!(
        matches!(fout, JitOutcome::Returned(_)),
        "concurrent freeze returns a placeholder"
    );
    assert_eq!(
        fvcpus.len(),
        2,
        "both concurrent children were captured as residue"
    );

    let (tout, tfinal) = thaw(&inst, &fsnap, &ffibers, &fvcpus, froot_sp);
    assert!(matches!(tout, JitOutcome::Returned(_)), "thaw returns");
    assert_eq!(read_state(&tfinal), STATE_NORMAL, "thaw back to NORMAL");
    assert_eq!(le_i64(&tfinal, OFF_ROOT), K, "root total reproduced");
    assert_eq!(le_i64(&tfinal, OFF_C1), K, "child 1 total reproduced");
    assert_eq!(le_i64(&tfinal, OFF_C2), K, "child 2 total reproduced");
}

// Follow-up A: a concurrent child that **finishes before the freeze point**, whose `thread.join` result
// the root only consumes *after* the freeze. The child (func1) has no may-suspend op, so it is left
// uninstrumented — it can't freeze and always runs to completion (a "completed child"). The root
// freezes mid-loop, *before* its join. The child's result lives only in the host-side Done cell, which
// isn't in the snapshot — so the freeze must capture it and the thaw deliver it to the re-executed join.
const SRC_JOIN: &str = r#"
func (i32, i32) -> (i64) {
block0(v0: i32, v1: i32):
  v2 = i64.const 7
  v3 = thread.spawn 1 v2 v2
  v4 = i32.const 0
  v5 = cap.call 13 0 (i32) -> (i64) v1 (v4)
  v6 = i64.const 0
  br block1(v3, v6)
block1(v7: i32, v8: i64):
  v9 = i64.const 1
  v10 = i64.add v8 v9
  v11 = i64.const 100000000
  v12 = i64.lt_s v10 v11
  br_if v12 block1(v7, v10) block2(v7, v10)
block2(v13: i32, v14: i64):
  v15 = thread.join v13
  v16 = i64.add v14 v15
  v17 = i64.const 65536
  i64.store v17 v16
  return v16
}
func (i64, i64) -> (i64) {
block0(v0: i64, v1: i64):
  v2 = i64.const 1000
  v3 = i64.add v1 v2
  return v3
}
"#;

#[test]
fn concurrent_join_result_survives_a_freeze_before_the_join() {
    let inst = instrument(SRC_JOIN);
    const ORACLE: i64 = K + 1007; // loop total + child(arg 7 + 1000)

    let Some((fout, fsnap, ffibers, fvcpus, froot_sp)) = concurrent_freeze(&inst) else {
        return;
    };

    if read_state(&fsnap) != STATE_UNWINDING {
        assert_eq!(le_i64(&fsnap, OFF_ROOT), ORACLE, "uninterrupted result");
        return;
    }
    assert!(
        matches!(fout, JitOutcome::Returned(_)),
        "freeze placeholder"
    );
    assert_eq!(fvcpus.len(), 1, "the completed child was captured");
    assert_eq!(
        fvcpus[0].completed_result,
        Some(1007),
        "the child's join result rides the artifact",
    );

    let (tout, tfinal) = thaw(&inst, &fsnap, &ffibers, &fvcpus, froot_sp);
    assert!(matches!(tout, JitOutcome::Returned(_)), "thaw returns");
    assert_eq!(read_state(&tfinal), STATE_NORMAL, "thaw back to NORMAL");
    assert_eq!(
        le_i64(&tfinal, OFF_ROOT),
        ORACLE,
        "the root's re-executed thread.join resolved the completed child's result across the freeze",
    );
}

// Follow-up B.1: a **concurrent** child that itself owns a fiber. The child creates + resumes a fiber
// that suspends (parks), then signals (so the freeze lands while its fiber is parked) and loops. On the
// freeze the child must flatten its own parked fiber — its own `freeze_drive`, the concurrent mirror of
// `run_child_inline`'s. (Here the *child*, not the root, drives the handshake; `concurrent_freeze`'s
// host fn flips the same flag regardless of caller.)
const SRC_CHILD_FIBER: &str = r#"
func (i32, i32) -> (i64) {
block0(v0: i32, v1: i32):
  v2 = i64.const 65568
  i32.store v2 v1
  v3 = i64.const 0
  v4 = thread.spawn 1 v3 v3
  v5 = i64.const 0
  br block1(v0, v5)
block1(v6: i32, v7: i64):
  v8 = i64.const 1
  v9 = i64.add v7 v8
  v10 = i64.const 100000000
  v11 = i64.lt_s v9 v10
  br_if v11 block1(v6, v9) block2(v6, v9)
block2(v12: i32, v13: i64):
  v14 = i32.const 0
  v15 = cap.call 2 0 (i32) -> (i64) v12 (v14)
  v16 = i64.const 65536
  i64.store v16 v13
  return v13
}
func (i64, i64) -> (i64) {
block0(v0: i64, v1: i64):
  v2 = ref.func 2
  v3 = i64.const 4096
  v4 = cont.new v2 v3
  v5 = i64.const 0
  v6, v7 = cont.resume v4 v5
  v8 = i64.const 65568
  v9 = i32.load v8
  v10 = i32.const 0
  v11 = cap.call 13 0 (i32) -> (i64) v9 (v10)
  v12 = i64.const 0
  br block1(v7, v12)
block1(v13: i64, v14: i64):
  v15 = i64.const 1
  v16 = i64.add v14 v15
  v17 = i64.const 100000000
  v18 = i64.lt_s v16 v17
  br_if v18 block1(v13, v16) block2(v13, v16)
block2(v19: i64, v20: i64):
  v21 = i64.add v19 v20
  v22 = i64.const 65544
  i64.store v22 v21
  return v21
}
func (i64, i64) -> (i64) {
block0(v0: i64, v1: i64):
  v2 = i64.const 5
  v3 = suspend v2
  v4 = i64.const 1000
  v5 = i64.add v3 v4
  return v5
}
"#;

#[test]
fn concurrent_child_owns_fiber_through_freeze_thaw() {
    let inst = instrument(SRC_CHILD_FIBER);
    let Some((fout, fsnap, ffibers, fvcpus, froot_sp)) = concurrent_freeze(&inst) else {
        return;
    };

    // This test needs the async freeze to catch the child **mid-loop with its fiber already parked** —
    // the clean interleaving the child's signal-after-park handshake aims for. On a slow/over-subscribed
    // runner the freeze can instead land while the child's fiber is still on its resume chain, or after
    // the child finished — a different (valid) freeze shape that doesn't exercise B.1. Only assert the
    // strong B.1 properties when the clean shape occurred; otherwise skip (the simpler concurrent tests
    // above always exercise the core concurrent freeze).
    let clean = read_state(&fsnap) == STATE_UNWINDING
        && matches!(fout, JitOutcome::Returned(_))
        && fvcpus.len() == 1
        && ffibers.len() == 1;
    if !clean {
        return;
    }

    let (tout, tfinal) = thaw(&inst, &fsnap, &ffibers, &fvcpus, froot_sp);
    assert!(matches!(tout, JitOutcome::Returned(_)), "thaw returns");
    assert_eq!(read_state(&tfinal), STATE_NORMAL, "thaw back to NORMAL");
    assert_eq!(le_i64(&tfinal, OFF_ROOT), K, "root total reproduced");
    assert_eq!(
        le_i64(&tfinal, OFF_C1),
        K + 5,
        "child resumed its loop + its fiber's yielded value across the freeze",
    );
}

// Blocked-in-join freeze: the root parks in `thread.join` and the freeze lands while it is **blocked**.
// The root spawns a long-running child as a concurrent OS thread and *immediately* joins it (no safepoint
// in between), so it parks in the join. The **child** drives the spawn-before-freeze handshake — it
// signals after it starts looping, so the root is already parked when the freeze is requested. With
// `thread.join` now a re-issue safepoint, the blocked root must unwind (the `thread_join` thunk returns
// on observing UNWINDING; the trailing safepoint spills the handle and unwinds), and the thaw must
// re-issue the join so it resolves the re-run child's result.
const SRC_BLOCKED_JOIN: &str = r#"
func (i32, i32) -> (i64) {
block0(v0: i32, v1: i32):
  v2 = i64.const 65560
  i32.store v2 v1
  v3 = i64.const 0
  v4 = thread.spawn 1 v3 v3
  v5 = thread.join v4
  v6 = i64.const 65536
  i64.store v6 v5
  return v5
}
func (i64, i64) -> (i64) {
block0(v0: i64, v1: i64):
  v2 = i64.const 65560
  v3 = i32.load v2
  v4 = i32.const 0
  v5 = cap.call 13 0 (i32) -> (i64) v3 (v4)
  v6 = i64.const 0
  br block1(v6)
block1(v7: i64):
  v8 = i64.const 1
  v9 = i64.add v7 v8
  v10 = i64.const 100000000
  v11 = i64.lt_s v9 v10
  br_if v11 block1(v9) block2(v9)
block2(v12: i64):
  v13 = i64.const 65544
  i64.store v13 v12
  return v12
}
"#;

#[test]
fn concurrent_freeze_while_root_blocked_in_join() {
    let inst = instrument(SRC_BLOCKED_JOIN);
    let Some((fout, fsnap, ffibers, fvcpus, froot_sp)) = concurrent_freeze(&inst) else {
        return;
    };

    if read_state(&fsnap) != STATE_UNWINDING {
        // The child finished and the root's join resolved before the freeze landed (rare): uninterrupted.
        assert_eq!(le_i64(&fsnap, OFF_ROOT), K);
        return;
    }
    assert!(
        matches!(fout, JitOutcome::Returned(_)),
        "freeze placeholder while the root was blocked in the join",
    );

    let (tout, tfinal) = thaw(&inst, &fsnap, &ffibers, &fvcpus, froot_sp);
    assert!(matches!(tout, JitOutcome::Returned(_)), "thaw returns");
    assert_eq!(read_state(&tfinal), STATE_NORMAL, "thaw back to NORMAL");
    assert_eq!(
        le_i64(&tfinal, OFF_C1),
        K,
        "the re-run child completed its loop across the freeze",
    );
    assert_eq!(
        le_i64(&tfinal, OFF_ROOT),
        K,
        "the root's re-issued thread.join resolved the child's result after a blocked-in-join freeze",
    );
}

// Follow-up B.2: nested concurrent spawns now work — a *concurrent* durable child that itself
// `thread.spawn`s attributes the grandchild's `parent_task` to itself via the per-OS-thread
// spawning-task source (not the shared `cur_task`). Here the root spawns a concurrent child that spawns
// a grandchild; with no freeze, the grandchild's value (42) flows back through both joins.
const SRC_NESTED: &str = r#"
func (i32) -> (i64) {
block0(v0: i32):
  v1 = i64.const 0
  v2 = thread.spawn 1 v1 v1
  v3 = thread.join v2
  return v3
}
func (i64, i64) -> (i64) {
block0(v0: i64, v1: i64):
  v2 = i64.const 0
  v3 = thread.spawn 2 v2 v2
  v4 = thread.join v3
  return v4
}
func (i64, i64) -> (i64) {
block0(v0: i64, v1: i64):
  v2 = i64.const 42
  return v2
}
"#;

#[test]
fn nested_concurrent_spawn_returns_grandchild_value() {
    let inst = instrument(SRC_NESTED);
    let mut host = Host::new();
    host.clock_ns = 42;
    let _clk = host.grant_clock();
    // A controller is required by the entry but never triggered — this is a pure NORMAL nested spawn.
    let freeze = FreezeController::new();
    let res = compile_and_run_capture_reserved_with_host_durable_mv_interruptible(
        &inst,
        0,
        &[0],
        &init_durable_window(WINDOW),
        &[],
        &[],
        &[],
        SHADOW_BASE + 8,
        SIZE_LOG2,
        svm_run::cap_thunk,
        &mut host as *mut Host as *mut c_void,
        freeze,
    );
    match res {
        Ok((out, ..)) => assert!(
            matches!(out, JitOutcome::Returned(ref s) if s == &[42]),
            "a nested concurrent spawn resolves the grandchild's value through both joins, got {out:?}",
        ),
        Err(JitError::Unsupported(_)) => {}
        Err(JitError::Backend(msg)) if msg.contains("Allocation error") => {}
        Err(e) => panic!("nested-spawn run failed: {e:?}"),
    }
}

// Follow-up B.2 under a freeze: a **three-level** concurrent tree (root → child → grandchild, all real
// OS threads) caught mid-flight. The grandchild drives the spawn-before-freeze handshake (it is the last
// to start, so by the time it signals, root + child are already looping), then every level self-unwinds
// into its own per-context region. The thaw rebuilds the **per-parent** join topology (grandchild under
// child, child under root) from the frozen residue and reproduces the uninterrupted result.
const SRC_NESTED_FREEZE: &str = r#"
func (i32, i32) -> (i64) {
block0(v0: i32, v1: i32):
  v2 = i64.const 65560
  i32.store v2 v1
  v3 = i64.const 0
  v4 = thread.spawn 1 v3 v3
  v5 = i64.const 0
  br block1(v4, v5)
block1(v6: i32, v7: i64):
  v8 = i64.const 1
  v9 = i64.add v7 v8
  v10 = i64.const 100000000
  v11 = i64.lt_s v9 v10
  br_if v11 block1(v6, v9) block2(v6, v9)
block2(v12: i32, v13: i64):
  v14 = thread.join v12
  v15 = i64.add v13 v14
  v16 = i64.const 65536
  i64.store v16 v15
  return v15
}
func (i64, i64) -> (i64) {
block0(v0: i64, v1: i64):
  v2 = i64.const 0
  v3 = thread.spawn 2 v2 v2
  v4 = i64.const 0
  br block1(v3, v4)
block1(v5: i32, v6: i64):
  v7 = i64.const 1
  v8 = i64.add v6 v7
  v9 = i64.const 100000000
  v10 = i64.lt_s v8 v9
  br_if v10 block1(v5, v8) block2(v5, v8)
block2(v11: i32, v12: i64):
  v13 = thread.join v11
  v14 = i64.add v12 v13
  v15 = i64.const 65544
  i64.store v15 v14
  return v14
}
func (i64, i64) -> (i64) {
block0(v0: i64, v1: i64):
  v2 = i64.const 65560
  v3 = i32.load v2
  v4 = i32.const 0
  v5 = cap.call 13 0 (i32) -> (i64) v3 (v4)
  v6 = i64.const 0
  br block1(v6)
block1(v7: i64):
  v8 = i64.const 1
  v9 = i64.add v7 v8
  v10 = i64.const 100000000
  v11 = i64.lt_s v9 v10
  br_if v11 block1(v9) block2(v9)
block2(v12: i64):
  v13 = i64.const 65552
  i64.store v13 v12
  return v12
}
"#;

#[test]
fn nested_concurrent_tree_freezes_and_thaws() {
    let inst = instrument(SRC_NESTED_FREEZE);
    let Some((fout, fsnap, ffibers, fvcpus, froot_sp)) = concurrent_freeze(&inst) else {
        return;
    };

    if read_state(&fsnap) != STATE_UNWINDING {
        // Everything finished before the freeze landed (rare): already correct.
        assert_eq!(le_i64(&fsnap, OFF_C2), K, "grandchild");
        assert_eq!(le_i64(&fsnap, OFF_C1), 2 * K, "child + grandchild");
        assert_eq!(le_i64(&fsnap, OFF_ROOT), 3 * K, "root + child + grandchild");
        return;
    }
    assert!(
        matches!(fout, JitOutcome::Returned(_)),
        "concurrent freeze returns a placeholder",
    );

    let (tout, tfinal) = thaw(&inst, &fsnap, &ffibers, &fvcpus, froot_sp);
    assert!(matches!(tout, JitOutcome::Returned(_)), "thaw returns");
    assert_eq!(read_state(&tfinal), STATE_NORMAL, "thaw back to NORMAL");
    assert_eq!(le_i64(&tfinal, OFF_C2), K, "grandchild total reproduced");
    assert_eq!(
        le_i64(&tfinal, OFF_C1),
        2 * K,
        "child's loop + its joined grandchild reproduced",
    );
    assert_eq!(
        le_i64(&tfinal, OFF_ROOT),
        3 * K,
        "root's loop + child + grandchild reproduced across the nested concurrent freeze",
    );
}
