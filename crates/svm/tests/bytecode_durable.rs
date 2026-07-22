//! Durability freeze/thaw harness for the bytecode engine (INTERP_PERF.md Slice 1c-6).
//!
//! Durable freeze/thaw is **IR-driven** (DURABILITY.md §2): the `svm-durable` transform rewrites a
//! module so that, with the window's state word set to `UNWINDING`, each function flattens its live
//! continuation into the in-window shadow stack and returns; with `REWINDING`, it rebuilds itself and
//! resumes. The native (here: bytecode) continuation is **never** serialized — so for a single-vCPU,
//! single-fiber program the bytecode engine supports freeze/thaw simply by *running the transformed
//! module* over a seeded window. (Multi-fiber freeze needs the per-fiber shadow-SP swap + the freeze
//! driver that flattens idle fibers; the bytecode durable entry refuses `cont.*`/`thread.*` modules,
//! falling back to the tree-walker — out of scope for this slice.)
//!
//! This checks the bytecode engine against the tree-walker oracle and the `svm-snapshot` §12 codec:
//!   1. a NORMAL durable run agrees with the tree-walker;
//!   2. an UNWINDING freeze produces a **byte-identical** snapshot *and* §12 artifact;
//!   3. restore+re-freeze is byte-identical (the §12.6 canonical invariant);
//!   4. thawing the bytecode artifact (REWINDING) reproduces the uninterrupted result and ends NORMAL.

use svm_durable::{
    begin_thaw, init_durable_window, read_state, transform_module, write_state, STATE_NORMAL,
    STATE_UNWINDING,
};
use svm_interp::{bytecode, run_capture_reserved_with_host, Host, Trap, Value};
use svm_snapshot::{freeze, restore};
use svm_text::parse_module;
use svm_verify::verify_module;

const SIZE_LOG2: u8 = 17; // 128 KiB ≥ the durable reserve
const WINDOW: usize = 1 << SIZE_LOG2;

fn window_with(state: i32) -> Vec<u8> {
    let mut w = init_durable_window(WINDOW);
    write_state(&mut w, state);
    w
}

/// Run the transformed entry on the **tree-walker** over `window`, clock seeded to `clock_v`.
fn tw_run(
    inst: &svm_ir::Module,
    clock_v: i64,
    window: &[u8],
) -> (Result<Vec<Value>, Trap>, Vec<u8>, i64, Host) {
    let mut h = Host::new();
    h.set_durable(true);
    let clk = h.grant_clock();
    h.clock_ns = clock_v;
    let mut fuel = 1_000_000u64;
    let (r, win) = run_capture_reserved_with_host(
        inst,
        0,
        &[Value::I32(clk)],
        &mut fuel,
        window,
        SIZE_LOG2,
        &mut h,
    );
    (r, win, h.clock_ns, h)
}

/// Run the transformed entry on the **bytecode engine** over `window`, clock seeded to `clock_v`.
fn bc_run(
    inst: &svm_ir::Module,
    clock_v: i64,
    window: &[u8],
) -> (Result<Vec<Value>, Trap>, Vec<u8>, i64, Host) {
    let mut h = Host::new();
    h.set_durable(true);
    let clk = h.grant_clock();
    h.clock_ns = clock_v;
    let mut fuel = 1_000_000u64;
    let (r, win) = bytecode::compile_and_run_capture_reserved_with_host(
        inst,
        0,
        &[Value::I32(clk)],
        &mut fuel,
        window,
        SIZE_LOG2,
        &mut h,
    )
    .expect("bytecode engine must drive a single-fiber durable module (Slice 1c-6)");
    (r, win, h.clock_ns, h)
}

/// Freeze on the tree-walker and the bytecode engine, then thaw the **bytecode** artifact, all checked
/// against the tree-walker oracle + the §12 codec.
fn check(src: &str) {
    let m = parse_module(src).expect("parse");
    let inst = transform_module(&m).expect("module must be in transform scope");
    verify_module(&inst).expect("instrumented IR verifies");

    let clock_v = 1000i64;

    // (1) NORMAL: the bytecode durable run agrees with the tree-walker (and is the baseline result).
    let (base_tw, _, _, _) = tw_run(&inst, clock_v, &window_with(STATE_NORMAL));
    let (base_bc, _, _, _) = bc_run(&inst, clock_v, &window_with(STATE_NORMAL));
    let base = base_tw.expect("baseline is trap-free");
    assert_eq!(
        base_bc,
        Ok(base.clone()),
        "NORMAL durable run: tree-walker != bytecode\n{src}"
    );

    // (2) freeze: UNWINDING leaves a byte-identical snapshot and §12 artifact across backends.
    let (fr_tw, snap_tw, _, host_tw) = tw_run(&inst, clock_v, &window_with(STATE_UNWINDING));
    let (fr_bc, snap_bc, clock_after, host_bc) =
        bc_run(&inst, clock_v, &window_with(STATE_UNWINDING));
    assert!(
        fr_tw.is_ok() && fr_bc.is_ok(),
        "freeze returns a placeholder\n{src}"
    );
    assert_eq!(
        read_state(&snap_bc),
        STATE_UNWINDING,
        "bytecode leaves the artifact UNWINDING\n{src}"
    );
    assert_eq!(
        snap_tw, snap_bc,
        "freeze snapshot: tree-walker != bytecode\n{src}"
    );

    let art_tw = freeze(&inst, &snap_tw, &host_tw).expect("tree-walker freeze serializes");
    let art_bc = freeze(&inst, &snap_bc, &host_bc).expect("bytecode freeze serializes");
    assert_eq!(
        art_tw, art_bc,
        "§12 artifact: tree-walker != bytecode\n{src}"
    );

    // (3) restore + re-freeze is byte-identical (the §12.6 canonical invariant).
    let mut rhost = Host::new();
    let rwin = restore(&art_bc, &inst, &mut rhost).expect("artifact restores");
    assert_eq!(
        freeze(&inst, &rwin, &rhost).expect("re-freeze"),
        art_bc,
        "re-serialize of a restored domain is byte-identical\n{src}"
    );

    // (4) thaw the bytecode artifact on the bytecode engine (clock continues from the freeze): the
    // frozen point's result is *reloaded* (not re-issued), so the run reproduces the baseline and ends
    // NORMAL.
    let mut thaw_win = rwin;
    begin_thaw(&mut thaw_win, 0);
    let (thaw_res, final_win, _, _) = bc_run(&inst, clock_after, &thaw_win);
    assert_eq!(
        thaw_res,
        Ok(base),
        "bytecode thaw must reproduce the uninterrupted result\n{src}"
    );
    assert_eq!(
        read_state(&final_win),
        STATE_NORMAL,
        "thaw must flip the state word back to NORMAL\n{src}"
    );
}

/// A single-fiber durable program with two may-suspend calls (`Clock.now`, iface 2 op 0 — an unwind
/// point): the first value is live across the second call, so a freeze after the first call spills it
/// into the shadow stack and a thaw reloads it. `base = clock_v + (clock_v + 1)`.
const TWO_CLOCK_READS: &str = r#"memory 17
func (i32) -> (i64) {
block 0 (v0: i32) {
  v1 = cap.call 2 0 () -> (i64) v0 ()
  v2 = cap.call 2 0 () -> (i64) v0 ()
  v3 = i64.add v1 v2
  return v3
  }
}
"#;

#[test]
fn single_fiber_clock_freeze_thaw_round_trip() {
    check(TWO_CLOCK_READS);
}

/// **Multiple** live values across the suspend point: the freeze must spill `v3` and `v8` (both
/// derived from the first call's result and used after the second call) into the continuation block's
/// params, and the thaw must restore them. `base = 5*clock_v + 8`.
const MULTI_LIVE: &str = r#"memory 17
func (i32) -> (i64) {
block 0 (v0: i32) {
  v1 = cap.call 2 0 () -> (i64) v0 ()
  v2 = i64.const 7
  v3 = i64.add v1 v2
  v7 = i64.const 3
  v8 = i64.mul v1 v7
  v5 = cap.call 2 0 () -> (i64) v0 ()
  v6 = i64.add v3 v5
  v9 = i64.add v6 v8
  return v9
  }
}
"#;

#[test]
fn single_fiber_multi_live_freeze_thaw() {
    check(MULTI_LIVE);
}
