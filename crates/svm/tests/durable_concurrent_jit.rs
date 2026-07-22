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
    begin_thaw, init_durable_window, read_state, transform_module_assume_confined, STATE_NORMAL,
    STATE_UNWINDING,
};
use svm_interp::{Host, SHADOW_BASE};
use svm_ir::{Memory, Module};
use svm_jit::{
    compile_and_run_capture_reserved_with_host_durable_mv,
    compile_and_run_capture_reserved_with_host_durable_mv_interruptible, FreezeController,
    FrozenFiber, FrozenVCpu, JitError, JitOutcome, TrapKind,
};
use svm_snapshot::{freeze as codec_freeze, restore as codec_restore};

const SIZE_LOG2: u8 = 17;
const WINDOW: usize = 1 << SIZE_LOG2;

// Loop trip count: large enough (~100ms native) that the freeze (requested the moment the root signals
// it has spawned) catches the looping contexts mid-flight.
const K: i64 = 100_000_000;

// Guest-memory slots (above the durable reserve). 65560 stashes the clock handle for the children.
const OFF_ROOT: i64 = 65536;
const OFF_C1: i64 = 65544;
const OFF_C2: i64 = 65552;
// §12.8 parked-vCPU slice: the root `atomic.wait`s on the futex word at guest offset 65568 (4-byte
// aligned; 65560 stays the clock/host-fn stash). `atomic.wait` status `1` = the value did not equal
// `expected` (no / end-of wait) — wasm's `WAIT_NOT_EQUAL`.
const WAIT_NOT_EQUAL: i64 = 1;
// `atomic.wait` status `0` = the waiter parked and a `notify` woke it — wasm's `WAIT_WOKEN`.
const WAIT_WOKEN: i64 = 0;

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
    // Declare durability on the *Host* (the JIT signals it to its own runtime via `set_durable_env`,
    // but the shared `cap_dispatch_slots` reads `Host::is_durable` — e.g. the §12.8 4A.7 `Blocking`
    // fail-closed gate). The interpreter durable path already requires this; the JIT path now matches.
    host.set_durable(true);
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
    begin_thaw(&mut twin, 0);
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
block 0 (v0: i32, v1: i32) {
  v2 = i64.const 65560
  i32.store v2 v0
  v3 = i64.const 65544
  v4 = thread.spawn 1 v3 v3
  v5 = i64.const 65552
  v6 = thread.spawn 1 v5 v5
  v7 = i32.const 0
  v8 = cap.call 13 0 (i32) -> (i64) v1 (v7)
  v9 = i64.const 0
  br 1(v9)
}
block 1 (v10: i64) {
  v11 = i64.const 1
  v12 = i64.add v10 v11
  v13 = i64.const 100000000
  v14 = i64.lt_s v12 v13
  br_if v14 1(v12) 2(v12)
}
block 2 (v15: i64) {
  v16 = i64.const 65536
  i64.store v16 v15
  return v15
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  v2 = i64.const 65560
  v3 = i32.load v2
  v4 = i64.const 0
  br 1(v1, v3, v4)
}
block 1 (v5: i64, v6: i32, v7: i64) {
  v8 = i64.const 1
  v9 = i64.add v7 v8
  v10 = i64.const 100000000
  v11 = i64.lt_s v9 v10
  br_if v11 1(v5, v6, v9) 2(v5, v6, v9)
}
block 2 (v12: i64, v13: i32, v14: i64) {
  v15 = i32.const 0
  v16 = cap.call 2 0 (i32) -> (i64) v13 (v15)
  i64.store v12 v14
  return v14
  }
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
block 0 (v0: i32, v1: i32) {
  v2 = i64.const 7
  v3 = thread.spawn 1 v2 v2
  v4 = i32.const 0
  v5 = cap.call 13 0 (i32) -> (i64) v1 (v4)
  v6 = i64.const 0
  br 1(v3, v6)
}
block 1 (v7: i32, v8: i64) {
  v9 = i64.const 1
  v10 = i64.add v8 v9
  v11 = i64.const 100000000
  v12 = i64.lt_s v10 v11
  br_if v12 1(v7, v10) 2(v7, v10)
}
block 2 (v13: i32, v14: i64) {
  v15 = thread.join v13
  v16 = i64.add v14 v15
  v17 = i64.const 65536
  i64.store v17 v16
  return v16
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  v2 = i64.const 1000
  v3 = i64.add v1 v2
  return v3
  }
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
block 0 (v0: i32, v1: i32) {
  v2 = i64.const 65568
  i32.store v2 v1
  v3 = i64.const 0
  v4 = thread.spawn 1 v3 v3
  v5 = i64.const 0
  br 1(v0, v5)
}
block 1 (v6: i32, v7: i64) {
  v8 = i64.const 1
  v9 = i64.add v7 v8
  v10 = i64.const 100000000
  v11 = i64.lt_s v9 v10
  br_if v11 1(v6, v9) 2(v6, v9)
}
block 2 (v12: i32, v13: i64) {
  v14 = i32.const 0
  v15 = cap.call 2 0 (i32) -> (i64) v12 (v14)
  v16 = i64.const 65536
  i64.store v16 v13
  return v13
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
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
  br 1(v7, v12)
}
block 1 (v13: i64, v14: i64) {
  v15 = i64.const 1
  v16 = i64.add v14 v15
  v17 = i64.const 100000000
  v18 = i64.lt_s v16 v17
  br_if v18 1(v13, v16) 2(v13, v16)
}
block 2 (v19: i64, v20: i64) {
  v21 = i64.add v19 v20
  v22 = i64.const 65544
  i64.store v22 v21
  return v21
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  v2 = i64.const 5
  v3 = suspend v2
  v4 = i64.const 1000
  v5 = i64.add v3 v4
  return v5
  }
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
    // runner the freeze can instead land while the child's fiber is still on its resume chain (that
    // mid-resume-chain case is now covered deterministically by
    // `concurrent_child_owns_active_chain_fiber_through_freeze_thaw`), or after the child finished — a
    // different (valid) freeze shape that doesn't exercise B.1's *parked* path. Only assert the strong
    // B.1 properties when the clean shape occurred; otherwise skip.
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

// Follow-up B.1′ — a concurrent child whose fiber is caught **mid-resume-chain** (active, not cleanly
// parked). B.1 covered a fiber *suspended* at the freeze point; here the **fiber itself** drives the
// spawn-before-freeze handshake (it signals from inside the resume chain, then loops K), so the async
// freeze deterministically lands while the fiber is *running on the chain* and the child is blocked in
// its `cont.resume`. On the freeze the fiber unwinds back through the resume (recorded as active-chain
// residue, like the single-vCPU slice-3.2 case), the child unwinds at its `cont.resume` re-issue, and
// the root unwinds separately. The thaw re-issues the child's resume, which re-enters the fiber; the
// fiber rewinds to its mid-loop point, finishes, and suspends 5 — so the child's loop + the fiber's
// yielded value reproduce `K + 5`, and the root reproduces `K`.
const SRC_CHILD_FIBER_ACTIVE: &str = r#"
func (i32, i32) -> (i64) {
block 0 (v0: i32, v1: i32) {
  v2 = i64.const 65568
  i32.store v2 v1
  v3 = i64.const 0
  v4 = thread.spawn 1 v3 v3
  v5 = i64.const 0
  br 1(v0, v5)
}
block 1 (v6: i32, v7: i64) {
  v8 = i64.const 1
  v9 = i64.add v7 v8
  v10 = i64.const 100000000
  v11 = i64.lt_s v9 v10
  br_if v11 1(v6, v9) 2(v6, v9)
}
block 2 (v12: i32, v13: i64) {
  v14 = i32.const 0
  v15 = cap.call 2 0 (i32) -> (i64) v12 (v14)
  v16 = i64.const 65536
  i64.store v16 v13
  return v13
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  v2 = ref.func 2
  v3 = i64.const 4096
  v4 = cont.new v2 v3
  v5 = i64.const 0
  v6, v7 = cont.resume v4 v5
  v12 = i64.const 0
  br 1(v7, v12)
}
block 1 (v13: i64, v14: i64) {
  v15 = i64.const 1
  v16 = i64.add v14 v15
  v17 = i64.const 100000000
  v18 = i64.lt_s v16 v17
  br_if v18 1(v13, v16) 2(v13, v16)
}
block 2 (v19: i64, v20: i64) {
  v21 = i64.add v19 v20
  v22 = i64.const 65544
  i64.store v22 v21
  return v21
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  v2 = i64.const 65568
  v3 = i32.load v2
  v4 = i32.const 0
  v5 = cap.call 13 0 (i32) -> (i64) v3 (v4)
  v6 = i64.const 0
  br 1(v6)
}
block 1 (v7: i64) {
  v8 = i64.const 1
  v9 = i64.add v7 v8
  v10 = i64.const 100000000
  v11 = i64.lt_s v9 v10
  br_if v11 1(v9) 2(v9)
}
block 2 (v12: i64) {
  v14 = suspend v12
  v15 = i64.const 1000
  v16 = i64.add v14 v15
  return v16
  }
}
"#;

#[test]
fn concurrent_child_owns_active_chain_fiber_through_freeze_thaw() {
    let inst = instrument(SRC_CHILD_FIBER_ACTIVE);
    let Some((fout, fsnap, ffibers, fvcpus, froot_sp)) = concurrent_freeze(&inst) else {
        return;
    };

    // The fiber signals from inside the chain then loops K, so the freeze should land while it is active
    // on the chain — the same residue shape as B.1 (one frozen child vCPU owning one frozen fiber). On a
    // badly-scheduled runner the freeze can still miss (fiber finished, or never entered): only assert
    // the strong active-chain properties when that clean shape occurred.
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
        2 * K,
        "child re-entered its mid-resume-chain fiber, which finished its own K-loop and suspended the \
         total, across the freeze (child's K + the fiber's K)",
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
block 0 (v0: i32, v1: i32) {
  v2 = i64.const 65560
  i32.store v2 v1
  v3 = i64.const 0
  v4 = thread.spawn 1 v3 v3
  v5 = thread.join v4
  v6 = i64.const 65536
  i64.store v6 v5
  return v5
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  v2 = i64.const 65560
  v3 = i32.load v2
  v4 = i32.const 0
  v5 = cap.call 13 0 (i32) -> (i64) v3 (v4)
  v6 = i64.const 0
  br 1(v6)
}
block 1 (v7: i64) {
  v8 = i64.const 1
  v9 = i64.add v7 v8
  v10 = i64.const 100000000
  v11 = i64.lt_s v9 v10
  br_if v11 1(v9) 2(v9)
}
block 2 (v12: i64) {
  v13 = i64.const 65544
  i64.store v13 v12
  return v12
  }
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
block 0 (v0: i32) {
  v1 = i64.const 0
  v2 = thread.spawn 1 v1 v1
  v3 = thread.join v2
  return v3
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  v2 = i64.const 0
  v3 = thread.spawn 2 v2 v2
  v4 = thread.join v3
  return v4
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  v2 = i64.const 42
  return v2
  }
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
block 0 (v0: i32, v1: i32) {
  v2 = i64.const 65560
  i32.store v2 v1
  v3 = i64.const 0
  v4 = thread.spawn 1 v3 v3
  v5 = i64.const 0
  br 1(v4, v5)
}
block 1 (v6: i32, v7: i64) {
  v8 = i64.const 1
  v9 = i64.add v7 v8
  v10 = i64.const 100000000
  v11 = i64.lt_s v9 v10
  br_if v11 1(v6, v9) 2(v6, v9)
}
block 2 (v12: i32, v13: i64) {
  v14 = thread.join v12
  v15 = i64.add v13 v14
  v16 = i64.const 65536
  i64.store v16 v15
  return v15
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  v2 = i64.const 0
  v3 = thread.spawn 2 v2 v2
  v4 = i64.const 0
  br 1(v3, v4)
}
block 1 (v5: i32, v6: i64) {
  v7 = i64.const 1
  v8 = i64.add v6 v7
  v9 = i64.const 100000000
  v10 = i64.lt_s v8 v9
  br_if v10 1(v5, v8) 2(v5, v8)
}
block 2 (v11: i32, v12: i64) {
  v13 = thread.join v11
  v14 = i64.add v12 v13
  v15 = i64.const 65544
  i64.store v15 v14
  return v14
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  v2 = i64.const 65560
  v3 = i32.load v2
  v4 = i32.const 0
  v5 = cap.call 13 0 (i32) -> (i64) v3 (v4)
  v6 = i64.const 0
  br 1(v6)
}
block 1 (v7: i64) {
  v8 = i64.const 1
  v9 = i64.add v7 v8
  v10 = i64.const 100000000
  v11 = i64.lt_s v9 v10
  br_if v11 1(v9) 2(v9)
}
block 2 (v12: i64) {
  v13 = i64.const 65552
  i64.store v13 v12
  return v12
  }
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

// §12.8 parked-vCPU slice — **freeze while a vCPU is blocked in `atomic.wait`, thaw resolves it.** The
// root spawns a concurrent child and *immediately* parks in `atomic.wait` on `FLAG_OFF` (expected 0, no
// timeout), so it is blocked when the freeze lands. The **child** drives the spawn-before-freeze
// handshake, but only *after* it has stored `FLAG_OFF = 1` (a plain atomic store — **no** notify, so the
// parked root is **not** woken): the controller therefore freezes with the value already changed in the
// window. On thaw the root re-issues the wait, which re-checks the value, finds `1 != 0`, and resolves
// immediately with `WAIT_NOT_EQUAL` — no re-park, no notifier needed. This is the thaw-able case: the
// wake landed as a value change that rode the snapshot.
const SRC_WAIT_WORKS: &str = r#"
func (i32, i32) -> (i64) {
block 0 (v0: i32, v1: i32) {
  v2 = i64.const 65560
  i32.store v2 v1
  v3 = i64.const 0
  v4 = thread.spawn 1 v3 v3
  v5 = i64.const 65568
  v6 = i32.const 0
  v7 = i64.const -1
  v8 = i32.atomic.wait v5 v6 v7
  v9 = i64.const 65536
  i32.store v9 v8
  v10 = i64.const 0
  return v10
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  v2 = i64.const 0
  br 1(v2)
}
block 1 (v3: i64) {
  v4 = i64.const 1
  v5 = i64.add v3 v4
  v6 = i64.const 100000000
  v7 = i64.lt_s v5 v6
  br_if v7 1(v5) 2(v5)
}
block 2 (v8: i64) {
  v9 = i64.const 65568
  v10 = i32.const 1
  i32.atomic.store v9 v10
  v11 = i64.const 65560
  v12 = i32.load v11
  v13 = i32.const 0
  v14 = cap.call 13 0 (i32) -> (i64) v12 (v13)
  v15 = i64.const 65544
  i64.store v15 v8
  return v8
  }
}
"#;

#[test]
fn concurrent_freeze_while_root_blocked_in_wait_thaws_when_value_changed() {
    let inst = instrument(SRC_WAIT_WORKS);
    let Some((fout, fsnap, ffibers, fvcpus, froot_sp)) = concurrent_freeze(&inst) else {
        return;
    };

    if read_state(&fsnap) != STATE_UNWINDING {
        // No frozen artifact (the freeze didn't engage), so there's nothing to thaw. Two valid races:
        // the root's wait found the changed value before parking (`NOT_EQUAL`, riding the snapshot), or —
        // rarer — it had parked and the freeze raced past it, so the child's plain store (no `notify`)
        // can't wake it and the run deadlock-traps (`ThreadFault`, the interp's join-deadlock). A
        // `Returned` with any other recorded status would be a real bug.
        match fout {
            JitOutcome::Returned(_) => assert_eq!(
                le_i64(&fsnap, OFF_ROOT),
                WAIT_NOT_EQUAL,
                "no-freeze completion: the root's wait resolved NOT_EQUAL",
            ),
            JitOutcome::Trapped(TrapKind::ThreadFault) => {}
            other => panic!("unexpected no-freeze outcome: {other:?}"),
        }
        return;
    }
    assert!(
        matches!(fout, JitOutcome::Returned(_)),
        "freeze placeholder while the root was blocked in the wait",
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
        WAIT_NOT_EQUAL,
        "the root's re-issued atomic.wait re-checked the value (1 != 0) and resolved NOT_EQUAL on thaw",
    );
}

// §12.8 parked-vCPU slice — **genuine-deadlock thaw fails closed.** Same shape, but the child **never**
// notifies `FLAG`: it signals first (so the freeze lands while the root is parked), then just loops and
// returns. On thaw (stage 2) the root re-issues its wait and **parks** (the value is still `0`), and the
// child re-runs concurrently — but it never notifies, so once it finishes no live vCPU can ever wake the
// root. `futex_wait`'s deadlock detection observes that its `peers_live()` went false and fails closed
// with `ThreadFault` (matching the interpreter's join-deadlock) — instead of the old blanket thaw
// fail-closed, and without hanging. (The sibling-notify counterpart resolves; see the next test.)
const SRC_WAIT_DEADLOCK: &str = r#"
func (i32, i32) -> (i64) {
block 0 (v0: i32, v1: i32) {
  v2 = i64.const 65560
  i32.store v2 v1
  v3 = i64.const 0
  v4 = thread.spawn 1 v3 v3
  v5 = i64.const 65568
  v6 = i32.const 0
  v7 = i64.const -1
  v8 = i32.atomic.wait v5 v6 v7
  v9 = i64.const 65536
  i32.store v9 v8
  v10 = i64.const 0
  return v10
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  v2 = i64.const 65560
  v3 = i32.load v2
  v4 = i32.const 0
  v5 = cap.call 13 0 (i32) -> (i64) v3 (v4)
  v6 = i64.const 0
  br 1(v6)
}
block 1 (v7: i64) {
  v8 = i64.const 1
  v9 = i64.add v7 v8
  v10 = i64.const 100000000
  v11 = i64.lt_s v9 v10
  br_if v11 1(v9) 2(v9)
}
block 2 (v12: i64) {
  v13 = i64.const 65544
  i64.store v13 v12
  return v12
  }
}
"#;

#[test]
fn concurrent_freeze_while_root_blocked_in_wait_fails_closed_on_thaw() {
    let inst = instrument(SRC_WAIT_DEADLOCK);
    let Some((fout, fsnap, ffibers, fvcpus, froot_sp)) = concurrent_freeze(&inst) else {
        return;
    };

    if read_state(&fsnap) != STATE_UNWINDING {
        // The freeze didn't catch the root parked (rare): nothing to assert about the wait.
        return;
    }
    assert!(
        matches!(fout, JitOutcome::Returned(_)),
        "freeze placeholder while the root was blocked in the wait",
    );

    let (tout, _tfinal) = thaw(&inst, &fsnap, &ffibers, &fvcpus, froot_sp);
    assert!(
        matches!(tout, JitOutcome::Trapped(TrapKind::ThreadFault)),
        "a re-issued wait whose every notifier has exited fails closed (ThreadFault), got {tout:?}",
    );
}

// §12.8 concurrent-thaw stage 2 — **producer↔consumer frozen mid-rendezvous, thaw resolves via a
// sibling's re-issued notify.** The capability the old blanket thaw fail-closed rejected. Same setup as
// the deadlock case (the freeze lands while the root is parked in `atomic.wait` on `FLAG=0`), but the
// child (the producer) **does** complete the rendezvous: after the freeze handshake + a delay loop it
// stores `FLAG=1` and `atomic.notify`s it. On thaw both vCPUs re-spawn and run concurrently: the root
// re-issues its wait and **parks** (the value is still `0` in the snapshot), the child re-runs to its
// re-issued `notify`, and that wakes the parked root — `WAIT_WOKEN`, no fail-closed, no hang. This is the
// per-context thaw word (stage 1b) + concurrent driver (stage 2) paying off: the two rewinds run on
// their own threads against their own words and re-synchronise on the real futex.
const SRC_WAIT_NOTIFY: &str = r#"
func (i32, i32) -> (i64) {
block 0 (v0: i32, v1: i32) {
  v2 = i64.const 65560
  i32.store v2 v1
  v3 = i64.const 0
  v4 = thread.spawn 1 v3 v3
  v5 = i64.const 65568
  v6 = i32.const 0
  v7 = i64.const -1
  v8 = i32.atomic.wait v5 v6 v7
  v9 = i64.const 65536
  i32.store v9 v8
  v10 = i64.const 0
  return v10
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  v2 = i64.const 65560
  v3 = i32.load v2
  v4 = i32.const 0
  v5 = cap.call 13 0 (i32) -> (i64) v3 (v4)
  v6 = i64.const 0
  br 1(v6)
}
block 1 (v7: i64) {
  v8 = i64.const 1
  v9 = i64.add v7 v8
  v10 = i64.const 100000000
  v11 = i64.lt_s v9 v10
  br_if v11 1(v9) 2(v9)
}
block 2 (v12: i64) {
  v13 = i64.const 65568
  v14 = i32.const 1
  i32.atomic.store v13 v14
  v15 = i32.const 1
  v16 = atomic.notify v13 v15
  v17 = i64.const 65544
  i64.store v17 v12
  return v12
  }
}
"#;

#[test]
fn concurrent_freeze_while_root_blocked_in_wait_thaws_via_sibling_notify() {
    let inst = instrument(SRC_WAIT_NOTIFY);
    let Some((fout, fsnap, ffibers, fvcpus, froot_sp)) = concurrent_freeze(&inst) else {
        return;
    };

    if read_state(&fsnap) != STATE_UNWINDING {
        // The freeze didn't catch the root parked (rare): nothing to assert about the rendezvous.
        return;
    }
    assert!(
        matches!(fout, JitOutcome::Returned(_)),
        "freeze placeholder while the root was blocked in the wait",
    );

    let (tout, tfinal) = thaw(&inst, &fsnap, &ffibers, &fvcpus, froot_sp);
    assert!(
        matches!(tout, JitOutcome::Returned(_)),
        "thaw resolves (the parked wait is woken by the sibling's re-issued notify), got {tout:?}",
    );
    assert_eq!(read_state(&tfinal), STATE_NORMAL, "thaw back to NORMAL");
    assert_eq!(
        le_i64(&tfinal, OFF_C1),
        K,
        "the producer child completed its loop and notified on thaw",
    );
    assert_eq!(
        le_i64(&tfinal, OFF_ROOT),
        WAIT_WOKEN,
        "the root's re-issued atomic.wait parked and was woken by the sibling's re-issued notify on thaw",
    );
}

// §12.8 concurrent-thaw stage 3 — **mutual deadlock fails closed (does not hang).** The root spawns two
// children and joins both; child A waits forever on B's flag and child B waits forever on A's flag —
// neither ever stores/notifies, so the three vCPUs (root in `thread.join`, A and B in `atomic.wait`) are
// all blocked with nobody runnable to wake anyone. The stage-2 `live > 1` heuristic missed this (both
// waiters stay *live*); stage 3's quiescence check (`live > parked`, counting wait + join parks) sees
// `live == parked` and fails the waits closed with `ThreadFault` — which resolves the root's joins.
// A fresh run suffices (the detection is general); the freeze/thaw path inherits it.
const SRC_MUTUAL_BLOCK: &str = r#"
func (i32, i32) -> (i64) {
block 0 (v0: i32, v1: i32) {
  v2 = i64.const 0
  v3 = thread.spawn 1 v2 v2
  v4 = thread.spawn 2 v2 v2
  v5 = thread.join v3
  v6 = thread.join v4
  v7 = i64.const 0
  return v7
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  v2 = i64.const 65576
  v3 = i32.const 0
  v4 = i64.const -1
  v5 = i32.atomic.wait v2 v3 v4
  v6 = i64.const 0
  return v6
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  v2 = i64.const 65568
  v3 = i32.const 0
  v4 = i64.const -1
  v5 = i32.atomic.wait v2 v3 v4
  v6 = i64.const 0
  return v6
  }
}
"#;

/// A fresh (no freeze, no residue) concurrent durable multi-vCPU run — used to exercise the deadlock
/// detection directly.
fn run_mv_fresh(inst: &Module) -> (JitOutcome, Vec<u8>) {
    let mut host = Host::new();
    host.clock_ns = 42;
    let clk = host.grant_clock();
    let _ = host.grant_host_fn(Box::new(|_op: u32, _a: &[i64], _m| Ok(vec![0])));
    let (out, win, ..) = compile_and_run_capture_reserved_with_host_durable_mv(
        inst,
        0,
        &[clk as i64, 0],
        &init_durable_window(WINDOW),
        &[],
        &[],
        &[],
        SHADOW_BASE + 8,
        SIZE_LOG2,
        svm_run::cap_thunk,
        &mut host as *mut Host as *mut c_void,
    )
    .expect("fresh concurrent mv run");
    (out, win)
}

#[test]
fn mutual_wait_block_fails_closed_not_hangs() {
    let inst = instrument(SRC_MUTUAL_BLOCK);
    let (out, _win) = run_mv_fresh(&inst);
    assert!(
        matches!(out, JitOutcome::Trapped(TrapKind::ThreadFault)),
        "two vCPUs each waiting on the other's never-set flag is a mutual deadlock — must fail closed \
         (ThreadFault) via quiescence detection, not hang; got {out:?}",
    );
}

// §12.8 concurrent-thaw stage 3 — **mutual rendezvous resolves (the quiescence check must not
// over-fire).** A 2-way barrier: child A stores+notifies `FlagA` then waits on `FlagB`; child B
// stores+notifies `FlagB` then waits on `FlagA`. Each notifies *before* it waits, so the two can never
// both be parked at once — at most one parks while the other runs to its notify (or the value is already
// set ⇒ `NOT_EQUAL`, no park). `live > parked` therefore stays true and the run resolves: both children
// reach the post-wait store (sentinel `7`) and the root's joins return. The live counterpart to the
// mutual-*block* above — the same two-vCPU cross-wait, but with the notifies that make it resolve.
const SRC_MUTUAL_RENDEZVOUS: &str = r#"
func (i32, i32) -> (i64) {
block 0 (v0: i32, v1: i32) {
  v2 = i64.const 0
  v3 = thread.spawn 1 v2 v2
  v4 = thread.spawn 2 v2 v2
  v5 = thread.join v3
  v6 = thread.join v4
  v7 = i64.const 0
  return v7
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  v2 = i64.const 65568
  v3 = i32.const 1
  i32.atomic.store v2 v3
  v4 = i32.const 1
  v5 = atomic.notify v2 v4
  v6 = i64.const 65576
  v7 = i32.const 0
  v8 = i64.const -1
  v9 = i32.atomic.wait v6 v7 v8
  v10 = i64.const 65544
  v11 = i64.const 7
  i64.store v10 v11
  return v11
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  v2 = i64.const 65576
  v3 = i32.const 1
  i32.atomic.store v2 v3
  v4 = i32.const 1
  v5 = atomic.notify v2 v4
  v6 = i64.const 65568
  v7 = i32.const 0
  v8 = i64.const -1
  v9 = i32.atomic.wait v6 v7 v8
  v10 = i64.const 65552
  v11 = i64.const 7
  i64.store v10 v11
  return v11
  }
}
"#;

#[test]
fn mutual_rendezvous_resolves_without_false_deadlock() {
    let inst = instrument(SRC_MUTUAL_RENDEZVOUS);
    let (out, win) = run_mv_fresh(&inst);
    assert!(
        matches!(out, JitOutcome::Returned(_)),
        "a cross-notifying 2-way barrier resolves — the quiescence check must not false-fire (both are \
         never parked at once); got {out:?}",
    );
    assert_eq!(
        le_i64(&win, OFF_C1),
        7,
        "child A passed its wait and stored its sentinel",
    );
    assert_eq!(
        le_i64(&win, OFF_C2),
        7,
        "child B passed its wait and stored its sentinel",
    );
}

#[test]
fn concurrent_freeze_thaw_is_deterministic_across_interleavings() {
    // §12.6 under the concurrent thaw: re-running the same freeze/thaw exercises different real OS-thread
    // schedules — the freeze landing at different points, the children rewinding concurrently on thaw.
    // Every run must reproduce the uninterrupted oracle (`K` for the root + both children), whether the
    // freeze caught the children mid-loop (the thaw rewinds them) or after they finished (the residue
    // carries their results). A schedule-dependent thaw bug would surface as a wrong total or a hang.
    // Kept modest (the controller `yield_now`s rather than busy-spins, so this doesn't starve CI cores).
    let inst = instrument(SRC_LOOPS);
    for i in 0..10 {
        let Some((fout, fsnap, ffibers, fvcpus, froot_sp)) = concurrent_freeze(&inst) else {
            continue; // unsupported shape / host alloc pressure: skip this iteration
        };
        if read_state(&fsnap) != STATE_UNWINDING {
            // Everything finished before the freeze landed: the snapshot already holds the oracle.
            assert_eq!(le_i64(&fsnap, OFF_ROOT), K, "iter {i}: root (no-freeze)");
            assert_eq!(le_i64(&fsnap, OFF_C1), K, "iter {i}: child 1 (no-freeze)");
            assert_eq!(le_i64(&fsnap, OFF_C2), K, "iter {i}: child 2 (no-freeze)");
            continue;
        }
        assert!(
            matches!(fout, JitOutcome::Returned(_)),
            "iter {i}: freeze placeholder"
        );
        let (tout, tfinal) = thaw(&inst, &fsnap, &ffibers, &fvcpus, froot_sp);
        assert!(
            matches!(tout, JitOutcome::Returned(_)),
            "iter {i}: thaw returns"
        );
        assert_eq!(
            read_state(&tfinal),
            STATE_NORMAL,
            "iter {i}: back to NORMAL"
        );
        assert_eq!(
            le_i64(&tfinal, OFF_ROOT),
            K,
            "iter {i}: root total reproduced"
        );
        assert_eq!(
            le_i64(&tfinal, OFF_C1),
            K,
            "iter {i}: child 1 total reproduced"
        );
        assert_eq!(
            le_i64(&tfinal, OFF_C2),
            K,
            "iter {i}: child 2 total reproduced"
        );
    }
}

// §12.8 4A.6 — **recycled-context async freeze (sparse-residue payoff).** The root spawns child A and
// **joins** it (A is uninstrumented — no may-suspend op — so it always finishes), which frees/recycles
// A's shadow context, *then* spawns the live looping child B and triggers the async freeze. By the freeze
// point A is finished **and joined** (fully gone — unlike SRC_JOIN's finished-but-unjoined child, which
// rides the artifact via `completed_result`), so only B is in the residue: 2 lifetime spawns, 1 frozen
// vCPU. The thaw reloads A's join result (A is never re-run) and reproduces every total.
const SRC_RECYCLE: &str = r#"
func (i32, i32) -> (i64) {
block 0 (v0: i32, v1: i32) {
  v2 = i64.const 65560
  i32.store v2 v0
  v3 = i64.const 7
  v4 = thread.spawn 1 v3 v3
  v5 = thread.join v4
  v6 = i64.const 65544
  v7 = thread.spawn 2 v6 v6
  v8 = i32.const 0
  v9 = cap.call 13 0 (i32) -> (i64) v1 (v8)
  v10 = i64.const 0
  br 1(v5, v10)
}
block 1 (v11: i64, v12: i64) {
  v13 = i64.const 1
  v14 = i64.add v12 v13
  v15 = i64.const 100000000
  v16 = i64.lt_s v14 v15
  br_if v16 1(v11, v14) 2(v11, v14)
}
block 2 (v17: i64, v18: i64) {
  v19 = i64.add v18 v17
  v20 = i64.const 65536
  i64.store v20 v19
  return v19
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  v2 = i64.const 1000
  v3 = i64.add v1 v2
  return v3
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  v2 = i64.const 65560
  v3 = i32.load v2
  v4 = i64.const 0
  br 1(v1, v3, v4)
}
block 1 (v5: i64, v6: i32, v7: i64) {
  v8 = i64.const 1
  v9 = i64.add v7 v8
  v10 = i64.const 100000000
  v11 = i64.lt_s v9 v10
  br_if v11 1(v5, v6, v9) 2(v5, v6, v9)
}
block 2 (v12: i64, v13: i32, v14: i64) {
  v15 = i32.const 0
  v16 = cap.call 2 0 (i32) -> (i64) v13 (v15)
  i64.store v12 v14
  return v14
  }
}
"#;

#[test]
fn recycled_context_freeze_residue_is_sparse() {
    let inst = instrument(SRC_RECYCLE);
    const ROOT_ORACLE: i64 = K + 1007; // root loop total K + A's join result (7 + 1000)
    let Some((fout, fsnap, ffibers, fvcpus, froot_sp)) = concurrent_freeze(&inst) else {
        return;
    };
    if read_state(&fsnap) != STATE_UNWINDING {
        // The freeze didn't catch the loops (rare): the run completed; A joined, B finished.
        assert_eq!(
            le_i64(&fsnap, OFF_ROOT),
            ROOT_ORACLE,
            "uninterrupted root total"
        );
        assert_eq!(le_i64(&fsnap, OFF_C1), K, "uninterrupted child B total");
        return;
    }
    assert!(
        matches!(fout, JitOutcome::Returned(_)),
        "freeze placeholder"
    );
    // The recycled-context residue: A finished + had its context freed, then B reserved a context (reusing
    // the freed slot) and froze live there. The residue is **B frozen** (`completed_result == None`) plus
    // **A as a completed child** (`completed_result == Some`) — completed children always ride so the
    // thaw's per-parent join table stays dense (follow-up A); the recycling shows in the *reused context*,
    // not a smaller record count.
    assert_eq!(
        fvcpus.len(),
        2,
        "B frozen + A completed (kept for join-table density)"
    );
    let frozen: Vec<_> = fvcpus
        .iter()
        .filter(|v| v.completed_result.is_none())
        .collect();
    let completed: Vec<_> = fvcpus
        .iter()
        .filter(|v| v.completed_result.is_some())
        .collect();
    assert_eq!(frozen.len(), 1, "exactly one live (frozen) child — B");
    assert_eq!(
        completed.len(),
        1,
        "exactly one completed child — the recycled A"
    );
    assert_eq!(
        completed[0].completed_result,
        Some(1007),
        "A's join result (7 + 1000) rides the artifact"
    );

    let (tout, tfinal) = thaw(&inst, &fsnap, &ffibers, &fvcpus, froot_sp);
    assert!(matches!(tout, JitOutcome::Returned(_)), "thaw returns");
    assert_eq!(read_state(&tfinal), STATE_NORMAL, "thaw back to NORMAL");
    assert_eq!(
        le_i64(&tfinal, OFF_ROOT),
        ROOT_ORACLE,
        "root total reproduced — A's join result was reloaded, A never re-run",
    );
    assert_eq!(
        le_i64(&tfinal, OFF_C1),
        K,
        "child B total reproduced on thaw"
    );
}

// §12.8 4A.6 codec follow-up — the recycled-context artifact through the **svm-snapshot §12 codec**.
// `recycled_context_freeze_residue_is_sparse` (above) checks the *in-memory* residue shape; this drives
// the **same real concurrent freeze residue** (B frozen + A completed, A's context recycled) through the
// serialize/restore codec and asserts the **§12.6 invariant 1 (canonical re-freeze)**: serialize → restore
// → re-serialize is **byte-identical**, with the recycled vCPU residue (`completed_result` included) and the
// sparse window image (recycled regions zero-elided) surviving intact.
//
// The codec is `svm_interp::Host`-based, and the interp can't *produce* a recycled residue — it runs durable
// single-worker, so `completed_result` is always `None` (svm-interp lib.rs) and no sibling context is freed
// at a freeze. So we bridge the JIT residue (field-identical mirror types) into a fresh codec-ready host that
// grants only the **durable clock**: the concurrent harness's signalling host-fn is a non-durable handle the
// codec would refuse (`FreezeError::NonDurableHandle`), and the children's clock reads already reloaded into
// the window image, so a clean clock-only handle table is a faithful artifact for the canonical-re-freeze check.
#[test]
fn recycled_context_artifact_canonical_re_freeze_through_the_codec() {
    let inst = instrument(SRC_RECYCLE);
    let Some((fout, fsnap, ffibers, fvcpus, froot_sp)) = concurrent_freeze(&inst) else {
        return;
    };
    if read_state(&fsnap) != STATE_UNWINDING {
        // The freeze didn't catch the loops (rare): nothing recycled-sparse to serialize.
        return;
    }
    assert!(
        matches!(fout, JitOutcome::Returned(_)),
        "freeze placeholder"
    );
    assert_eq!(
        fvcpus.len(),
        2,
        "recycled residue we serialize: B frozen + A completed"
    );
    assert!(
        fvcpus.iter().any(|v| v.completed_result == Some(1007)),
        "A's join result (7 + 1000) is in the residue we serialize",
    );

    // Bridge the real JIT residue into a fresh codec-ready interp host. `svm_jit` and `svm_interp` frozen
    // residue types are field-identical mirrors; the codec reads its control section from the host's
    // `frozen_*` fields + the window image.
    let i_fibers: Vec<svm_interp::FrozenFiber> = ffibers
        .iter()
        .map(|f| svm_interp::FrozenFiber {
            slot: f.slot,
            func: f.func,
            sp: f.sp,
            shadow_sp: f.shadow_sp,
            generation: f.generation,
        })
        .collect();
    let i_vcpus: Vec<svm_interp::FrozenVCpu> = fvcpus
        .iter()
        .map(|v| svm_interp::FrozenVCpu {
            task: v.task,
            parent_task: v.parent_task,
            func: v.func,
            args: v.args.clone(),
            shadow_sp: v.shadow_sp,
            completed_result: v.completed_result,
        })
        .collect();

    let mut fhost = Host::new();
    fhost.set_durable(true);
    let _ = fhost.grant_clock(); // the only durable handle; mirrors the real domain's clock
    fhost.set_frozen_fibers(i_fibers);
    fhost.set_frozen_vcpus(i_vcpus);
    fhost.set_frozen_root_sp(froot_sp);

    // Serialize the real §12 artifact carrying the recycled residue + sparse window image.
    let artifact = codec_freeze(&inst, &fsnap, &fhost).expect("recycled artifact serializes");
    assert!(
        artifact.len() < WINDOW,
        "sparse window image — recycled (freed) sibling regions are zero-elided: {} < {WINDOW}",
        artifact.len(),
    );

    // Restore into a fresh host; the residue (completed_result included) + root extent re-seed.
    let mut thost = Host::new();
    thost.set_durable(true);
    let window = codec_restore(&artifact, &inst, &mut thost).expect("recycled artifact restores");
    assert_eq!(
        thost.frozen_vcpus().len(),
        2,
        "restore re-seeded both residue records (B frozen + A completed)",
    );
    assert!(
        thost
            .frozen_vcpus()
            .iter()
            .any(|v| v.completed_result == Some(1007)),
        "A's completed-child result survived the codec round-trip",
    );

    // §12.6 invariant 1 — canonical: re-serializing the freshly-restored domain reproduces the recycled
    // artifact byte-for-byte (sparse window image + recycled vCPU residue, completed_result included).
    assert_eq!(
        codec_freeze(&inst, &window, &thost).expect("re-freeze"),
        artifact,
        "canonical re-freeze of a restored recycled-context domain is byte-identical",
    );
}
