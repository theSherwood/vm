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
    FrozenVCpu, JitError, JitOutcome,
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

/// Run the durable concurrent freeze with the spawn-before-freeze handshake. `v0` = clock handle (for
/// the children), `v1` = host-fn handle (the root calls it to signal "children spawned"). Returns the
/// freeze outcome, or `None` to skip (unsupported shape / host alloc pressure).
fn concurrent_freeze(inst: &Module) -> Option<(JitOutcome, Vec<u8>, Vec<FrozenVCpu>, u64)> {
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
        Ok((o, s, _f, v, r)) => Some((o, s, v, r)),
        Err(JitError::Unsupported(_)) => None,
        Err(JitError::Backend(msg)) if msg.contains("Allocation error") => None,
        Err(e) => panic!("concurrent freeze failed: {e:?}"),
    }
}

fn thaw(inst: &Module, snap: &[u8], vcpus: &[FrozenVCpu], root_sp: u64) -> (JitOutcome, Vec<u8>) {
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
        &[],
        &[],
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
    let Some((fout, fsnap, fvcpus, froot_sp)) = concurrent_freeze(&inst) else {
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

    let (tout, tfinal) = thaw(&inst, &fsnap, &fvcpus, froot_sp);
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

    let Some((fout, fsnap, fvcpus, froot_sp)) = concurrent_freeze(&inst) else {
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

    let (tout, tfinal) = thaw(&inst, &fsnap, &fvcpus, froot_sp);
    assert!(matches!(tout, JitOutcome::Returned(_)), "thaw returns");
    assert_eq!(read_state(&tfinal), STATE_NORMAL, "thaw back to NORMAL");
    assert_eq!(
        le_i64(&tfinal, OFF_ROOT),
        ORACLE,
        "the root's re-executed thread.join resolved the completed child's result across the freeze",
    );
}
