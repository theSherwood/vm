//! DURABILITY.md §12.8 Phase 4 Slice A.5 **stage (ii)** — a *genuinely concurrent* multi-vCPU
//! stop-the-world freeze on the JIT. The root `thread.spawn`s two children that run as **real OS
//! threads** (not the single-worker deferred model); an async [`FreezeController::request_freeze`]
//! catches all three contexts (root + 2 children) mid-compute-loop, and each **self-unwinds into its
//! own per-context shadow-SP region concurrently** (lock-free — the stage-i relocation gave each its
//! own SP word). The coordinator (root) joins the children via the run's `join_all`, then snapshots a
//! fully-quiesced window. A thaw resumes every context and reproduces the uninterrupted result.
//!
//! Each context writes its loop total to a distinct **guest-memory** slot rather than via
//! `thread.join`, so the result is captured in the window image and is **robust to freeze timing**:
//! whatever froze resumes on thaw and writes its slot; whatever (improbably) finished first already
//! wrote it into the snapshot. (Surviving a *join* result across a concurrent freeze — a child that
//! finishes before the freeze point — needs join-table capture and is a separate follow-up.)

use core::ffi::c_void;
use svm_durable::{
    init_durable_window, read_state, transform_module_assume_confined, write_state, STATE_NORMAL,
    STATE_REWINDING, STATE_UNWINDING,
};
use svm_interp::{Host, SHADOW_BASE};
use svm_ir::{Memory, Module};
use svm_jit::{
    compile_and_run_capture_reserved_with_host_durable_mv,
    compile_and_run_capture_reserved_with_host_durable_mv_interruptible, FreezeController,
    JitError, JitOutcome,
};

const SIZE_LOG2: u8 = 17;
const WINDOW: usize = 1 << SIZE_LOG2;

// Loop trip count: large enough (~100ms native) that an async freeze request landing within
// microseconds catches every context mid-flight — no context completes 100M iterations first.
const K: i64 = 100_000_000;

// Guest-memory result slots (above the durable reserve; the confined toolchain bases guest data here).
const OFF_ROOT: i64 = 65536;
const OFF_C1: i64 = 65544;
const OFF_C2: i64 = 65552;

// func0 (root, v0 = clock handle): stash the clock handle for the children, spawn child(OFF_C1) +
// child(OFF_C2) as concurrent OS threads, run its own K-loop, store its total at OFF_ROOT, return.
// func1 (child, v1 = its result slot): load the clock handle, run a K-loop, store K at its slot.
//
// Each function reads the clock once **after** its loop — purely to be "may-suspend" so the durable
// transform instruments the loop with a back-edge poll (the freeze safepoint); the clock value is
// discarded, so the stored result (the loop total) is freeze-timing-invariant. No `thread.join` —
// `join_all` at teardown waits for the children (or their freeze-unwind), and each total lands in the
// window, so the round-trip result is robust to *when* each context froze.
const SRC: &str = r#"
func (i32) -> (i64) {
block0(v0: i32):
  v1 = i64.const 65560
  i32.store v1 v0
  v2 = i64.const 65544
  v3 = thread.spawn 1 v2 v2
  v4 = i64.const 65552
  v5 = thread.spawn 1 v4 v4
  v6 = i64.const 0
  br block1(v0, v6)
block1(v7: i32, v8: i64):
  v9 = i64.const 1
  v10 = i64.add v8 v9
  v11 = i64.const 100000000
  v12 = i64.lt_s v10 v11
  br_if v12 block1(v7, v10) block2(v7, v10)
block2(v13: i32, v14: i64):
  v15 = i32.const 0
  v16 = cap.call 2 0 (i32) -> (i64) v13 (v15)
  v17 = i64.const 65536
  i64.store v17 v14
  return v14
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

fn instrument() -> Module {
    let mut m = svm_text::parse_module(SRC).expect("parse");
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

#[test]
fn concurrent_children_self_unwind_and_thaw_reproduces_the_result() {
    let inst = instrument();

    // A controller thread requests the freeze the instant the run publishes its window base — the real
    // async stop-the-world trigger. With 100M-iteration loops it catches the root and both children
    // mid-flight with overwhelming probability.
    let freeze = FreezeController::new();
    let fc = std::sync::Arc::clone(&freeze);
    let controller = std::thread::spawn(move || fc.request_freeze());

    let mut fhost = Host::new();
    fhost.clock_ns = 42;
    let fclk = fhost.grant_clock();
    let (fout, fsnap, _ffibers, fvcpus, froot_sp) =
        match compile_and_run_capture_reserved_with_host_durable_mv_interruptible(
            &inst,
            0,
            &[fclk as i64],
            &init_durable_window(WINDOW),
            &[],
            &[],
            &[],
            SHADOW_BASE + 8, // empty root extent (frame base) — unused once the root froze
            SIZE_LOG2,
            svm_run::cap_thunk,
            &mut fhost as *mut Host as *mut c_void,
            freeze,
        ) {
            Ok(t) => t,
            // Mirror the other JIT durable tests: skip on an unsupported shape / host alloc pressure.
            Err(JitError::Unsupported(_)) => return,
            Err(JitError::Backend(msg)) if msg.contains("Allocation error") => return,
            Err(e) => panic!("concurrent freeze failed: {e:?}"),
        };
    controller.join().unwrap();

    if read_state(&fsnap) != STATE_UNWINDING {
        // The whole domain finished before the request landed (astronomically unlikely at 100M iters):
        // the run is simply uninterrupted, and every slot is already its total. Correctness still holds.
        assert_eq!(le_i64(&fsnap, OFF_ROOT), K, "uninterrupted root total");
        assert_eq!(le_i64(&fsnap, OFF_C1), K, "uninterrupted child-1 total");
        assert_eq!(le_i64(&fsnap, OFF_C2), K, "uninterrupted child-2 total");
        return;
    }

    // The freeze caught the contexts mid-loop: the root unwound (placeholder return) and both children
    // self-unwound concurrently into their own regions, each recorded as residue.
    assert!(
        matches!(fout, JitOutcome::Returned(_)),
        "concurrent freeze returns a placeholder, not a trap"
    );
    assert_eq!(
        fvcpus.len(),
        2,
        "both concurrent children self-unwound and were captured as residue"
    );

    // Thaw: re-spawn the frozen children (REWINDING, resume their loops to completion) and rewind the
    // root, then run to completion. Every loop finishes and stores its total — reproducing the
    // uninterrupted result across a genuinely-concurrent freeze.
    let mut twin = fsnap.clone();
    write_state(&mut twin, STATE_REWINDING);
    let mut thost = Host::new();
    thost.clock_ns = 99;
    let tclk = thost.grant_clock();
    let (tout, tfinal, ..) = match compile_and_run_capture_reserved_with_host_durable_mv(
        &inst,
        0,
        &[tclk as i64],
        &twin,
        &[],
        &[],
        &fvcpus,
        froot_sp,
        SIZE_LOG2,
        svm_run::cap_thunk,
        &mut thost as *mut Host as *mut c_void,
    ) {
        Ok(t) => t,
        Err(e) => panic!("concurrent thaw failed: {e:?}"),
    };
    assert!(
        matches!(tout, JitOutcome::Returned(_)),
        "thaw returns a value, not a trap"
    );
    assert_eq!(
        le_i64(&tfinal, OFF_ROOT),
        K,
        "root resumed its loop and stored its total"
    );
    assert_eq!(
        le_i64(&tfinal, OFF_C1),
        K,
        "child 1 resumed its loop and stored its total"
    );
    assert_eq!(
        le_i64(&tfinal, OFF_C2),
        K,
        "child 2 resumed its loop and stored its total"
    );
    assert_eq!(
        read_state(&tfinal),
        STATE_NORMAL,
        "thaw flips the domain back to NORMAL"
    );
}
