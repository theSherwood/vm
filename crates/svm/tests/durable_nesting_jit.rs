//! JIT durable-nesting parity (DURABILITY.md §4, "JIT parity") — the interpreter is the oracle.
//!
//! Three slices, in order:
//! * **run** (`jit_durable_same_module_child_matches_interp`): a durable same-module child runs on the
//!   JIT (was fail-closed `-EINVAL`); a runnable one is pure-compute (an instrumented child hits a
//!   `cap.call` against its empty powerbox → `CapFault`);
//! * **powerbox** (`jit_durable_depth2_grandchild_matches_interp`): a durable child gets a
//!   one-capability `Instantiator` powerbox and nests a grandchild (depth-2, `NORMAL`);
//! * **freeze export** (`jit_freeze_captures_live_nested_child_matching_interp`): a freeze catches a
//!   live nested child and captures a `FrozenNested` re-attach record whose geometry matches the
//!   interpreter's.
//!
//! Thaw (re-attaching a frozen nested child on the JIT), separate-module durable children, and
//! `coro_spawn` are follow-ups.

use core::ffi::c_void;
use svm_durable::{
    init_durable_window, read_state, transform_module, write_state, STATE_UNWINDING,
};
use svm_interp::{run_capture_reserved_with_host, Host, Value};
use svm_jit::{
    compile_and_run_capture_reserved_with_host_durable,
    compile_and_run_capture_reserved_with_host_durable_nested, JitOutcome,
};
use svm_text::parse_module;
use svm_verify::verify_module;

const SIZE_LOG2: u8 = 18;
const WINDOW: usize = 1 << SIZE_LOG2;

/// A durable **same-module** parent: `instantiate`s its own func 1 (op 0) confined to a 128 KiB
/// sub-window, `join`s it, and returns the child's result. Func 1 sums 0..100 = 4950 — pure compute
/// (no `cap.call`), so it is not may-suspend and runs atomically in the carve. (Identical in shape to
/// `durable_nesting.rs::PARENT_SELF_LOOP`.)
const PARENT_SELF_LOOP: &str = "memory 18
func (i32) -> (i64) {
block0(v0: i32):
  v1 = i64.const 1
  v2 = i64.const 131072
  v3 = i64.const 17
  v4 = i64.const 0
  v5 = cap.call 6 0 (i64, i64, i64, i64) -> (i32) v0 (v1, v2, v3, v4)
  v6 = cap.call 6 1 (i32) -> (i64) v0 (v5)
  return v6
}
func (i64, i64) -> (i64) {
block0(v0: i64, v1: i64):
  v2 = i64.const 0
  v3 = i64.const 0
  br block1(v2, v3)
block1(v4: i64, v5: i64):
  v6 = i64.const 100
  v7 = i64.lt_s v4 v6
  br_if v7 block2(v4, v5) block3(v5)
block2(v8: i64, v9: i64):
  v10 = i64.add v9 v8
  v11 = i64.const 1
  v12 = i64.add v8 v11
  br block1(v12, v10)
block3(v13: i64):
  return v13
}
";

fn instrument(src: &str) -> svm_ir::Module {
    let m = parse_module(src).expect("parse");
    let inst = transform_module(&m).expect("transform");
    verify_module(&inst).expect("instrumented module verifies");
    inst
}

/// Slice 1: a durable same-module `instantiate` + `join` returns the nested child's total (4950)
/// on **both** backends — the JIT now runs the durable child instead of failing closed.
#[test]
fn jit_durable_same_module_child_matches_interp() {
    let inst = instrument(PARENT_SELF_LOOP);

    // Interp: the durable domain runs the nested child; its 4950 comes back through join.
    let mut hi = Host::new();
    hi.set_durable(true);
    let ih = hi.grant_instantiator(0, WINDOW as u64);
    let mut fuel = 50_000_000u64;
    let (ir, _imem) = run_capture_reserved_with_host(
        &inst,
        0,
        &[Value::I32(ih)],
        &mut fuel,
        &init_durable_window(WINDOW),
        SIZE_LOG2,
        &mut hi,
    );
    assert_eq!(
        ir.expect("interp durable run ok"),
        vec![Value::I64(4950)],
        "interp runs the durable same-module nested child"
    );

    // JIT: slice 1 admits the durable same-module child (it runs atomically in its carve), so it
    // returns 4950 too — the boundary flipped from fail-closed to a positive differential.
    // Gated on the nesting runtime being available on this target (as the other JIT nesting tests).
    if !svm_jit::fiber_supported() {
        return;
    }
    let mut hj = Host::new();
    hj.set_durable(true);
    let jh = hj.grant_instantiator(0, WINDOW as u64);
    let win = init_durable_window(WINDOW);
    let (jo, _jmem, _residue) = compile_and_run_capture_reserved_with_host_durable(
        &inst,
        0,
        &[jh as i64],
        &win,
        &[],
        &[],
        SIZE_LOG2,
        svm_run::cap_thunk,
        &mut hj as *mut Host as *mut c_void,
    )
    .expect("durable run compiles");
    assert!(
        matches!(jo, JitOutcome::Returned(ref s) if s == &[4950]),
        "JIT runs the durable same-module nested child, matching the interp's 4950: {jo:?}"
    );
}

const D2_SIZE_LOG2: u8 = 19;
const D2_WINDOW: usize = 1 << D2_SIZE_LOG2;

/// A durable **same-module depth-2** chain: root (func 0) instantiates a child (func 1) which itself
/// instantiates a grandchild (func 2, the 0..100 = 4950 loop) and joins it. The child is *instrumented*
/// (it does a `cap.call`), so this exercises the child's **Instantiator powerbox** (it resolves its own
/// window and carves the grandchild) and the carve control-word seeding. All `NORMAL` (no freeze).
/// (Identical in shape to `durable_nesting.rs::PARENT_DEPTH2`.)
const PARENT_DEPTH2: &str = "memory 19
func (i32) -> (i64) {
block0(v0: i32):
  v1 = i64.const 1
  v2 = i64.const 262144
  v3 = i64.const 18
  v4 = i64.const 0
  v5 = cap.call 6 0 (i64, i64, i64, i64) -> (i32) v0 (v1, v2, v3, v4)
  v6 = cap.call 6 1 (i32) -> (i64) v0 (v5)
  return v6
}
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i32.const 256
  v2 = i64.const 2
  v3 = i64.const 131072
  v4 = i64.const 17
  v5 = i64.const 0
  v6 = cap.call 6 0 (i64, i64, i64, i64) -> (i32) v1 (v2, v3, v4, v5)
  v7 = cap.call 6 1 (i32) -> (i64) v1 (v6)
  return v7
}
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const 0
  v2 = i64.const 0
  br block1(v1, v2)
block1(v3: i64, v4: i64):
  v5 = i64.const 100
  v6 = i64.lt_s v3 v5
  br_if v6 block2(v3, v4) block3(v4)
block2(v7: i64, v8: i64):
  v9 = i64.add v8 v7
  v10 = i64.const 1
  v11 = i64.add v7 v10
  br block1(v11, v9)
block3(v12: i64):
  return v12
}
";

/// The child powerbox slice: a durable depth-2 same-module chain (root → child → grandchild) returns
/// the grandchild's total (4950) on **both** backends. The JIT child now carries an Instantiator
/// powerbox, so its own `instantiate` of the grandchild resolves and runs instead of failing closed.
#[test]
fn jit_durable_depth2_grandchild_matches_interp() {
    let inst = instrument(PARENT_DEPTH2);

    // Interp: the durable chain runs; the grandchild's 4950 propagates up through both joins.
    let mut hi = Host::new();
    hi.set_durable(true);
    let ih = hi.grant_instantiator(0, D2_WINDOW as u64);
    let mut fuel = 50_000_000u64;
    let (ir, _imem) = run_capture_reserved_with_host(
        &inst,
        0,
        &[Value::I32(ih)],
        &mut fuel,
        &init_durable_window(D2_WINDOW),
        D2_SIZE_LOG2,
        &mut hi,
    );
    assert_eq!(
        ir.expect("interp durable depth-2 run ok"),
        vec![Value::I64(4950)],
        "interp runs the durable depth-2 chain"
    );

    // JIT: the child's Instantiator powerbox lets it carve + run the grandchild, so 4950 comes back.
    if !svm_jit::fiber_supported() {
        return;
    }
    let mut hj = Host::new();
    hj.set_durable(true);
    let jh = hj.grant_instantiator(0, D2_WINDOW as u64);
    let win = init_durable_window(D2_WINDOW);
    let (jo, _jmem, _residue) = compile_and_run_capture_reserved_with_host_durable(
        &inst,
        0,
        &[jh as i64],
        &win,
        &[],
        &[],
        D2_SIZE_LOG2,
        svm_run::cap_thunk,
        &mut hj as *mut Host as *mut c_void,
    )
    .expect("durable depth-2 run compiles");
    assert!(
        matches!(jo, JitOutcome::Returned(ref s) if s == &[4950]),
        "JIT runs the durable depth-2 chain (child powerbox nests the grandchild): {jo:?}"
    );
}

/// A durable parent that instantiates + joins an **instrumented** child (func 1). Func 1 is a
/// may-suspend loop — its `block3` (loop exit) holds a `cap.call` (never reached under a freeze), so
/// the transform makes it may-suspend and prepends a **loop-header poll** at `block1`. Born
/// `UNWINDING`, the child spills at that first poll (loop not yet entered) instead of completing — the
/// live child a freeze must capture. (A *pure-compute* loop like `PARENT_SELF_LOOP`'s child has no
/// poll site, so the synchronous JIT would run it to completion; only an instrumented child unwinds
/// mid-run — see DURABILITY.md §4 "Freeze model".)
const FREEZE_PARENT: &str = "memory 18
func (i32) -> (i64) {
block0(v0: i32):
  v1 = i64.const 1
  v2 = i64.const 131072
  v3 = i64.const 17
  v4 = i64.const 0
  v5 = cap.call 6 0 (i64, i64, i64, i64) -> (i32) v0 (v1, v2, v3, v4)
  v6 = cap.call 6 1 (i32) -> (i64) v0 (v5)
  return v6
}
func (i32) -> (i64) {
block0(v0: i32):
  v1 = i64.const 0
  br block1(v1, v1)
block1(v2: i64, v3: i64):
  v4 = i64.const 100
  v5 = i64.lt_s v2 v4
  br_if v5 block2(v2, v3) block3(v3)
block2(v6: i64, v7: i64):
  v8 = i64.add v7 v6
  v9 = i64.const 1
  v10 = i64.add v6 v9
  br block1(v10, v8)
block3(v11: i64):
  v12 = i32.const 256
  v13 = cap.call 6 1 (i32) -> (i64) v12 (v12)
  return v13
}
";

/// Freeze export — the first nested-freeze artifact on the JIT: a freeze-from-start (window
/// `UNWINDING`) that catches a **live** §14 child captures a `FrozenNested` re-attach record whose
/// geometry matches the interpreter's.
///
/// A subtlety of the synchronous model (DURABILITY.md §4 "Freeze model"): the two backends freeze a
/// **different child body** to produce a *live* residue, but the record is the same. The interpreter
/// freezes a *pure-compute* child (`PARENT_SELF_LOOP`'s func 1) by **never scheduling it** — the
/// parent, already `UNWINDING`, executes `instantiate` (enqueue only) then spills at its trailing poll.
/// The synchronous JIT runs the child *inline* in `instantiate`, so a pure child would run to
/// completion; it needs an **instrumented** child (`FREEZE_PARENT`'s func 1 — a may-suspend loop) that
/// unwinds at its first loop-header poll. **Both parents make the identical `instantiate` call**
/// (off = 131072, size_log2 = 17, entry = 1), so the `FrozenNested` — carve geometry + slot, the data
/// a thaw re-attaches — is identical regardless of the child body. We assert the JIT's captured record
/// equals the interpreter's. (Freeze→thaw→result round-trip parity is the thaw slice; the records can
/// coincide here even though the frozen continuations differ.)
#[test]
fn jit_freeze_captures_live_nested_child_matching_interp() {
    // Interp: freeze `PARENT_SELF_LOOP` (its pure-compute child frozen at entry → one residue record).
    let iinst = instrument(PARENT_SELF_LOOP);
    let mut hi = Host::new();
    hi.set_durable(true);
    let ih = hi.grant_instantiator(0, WINDOW as u64);
    let mut iwin = init_durable_window(WINDOW);
    write_state(&mut iwin, STATE_UNWINDING);
    let mut fuel = 50_000_000u64;
    let (ir, isnap) = run_capture_reserved_with_host(
        &iinst,
        0,
        &[Value::I32(ih)],
        &mut fuel,
        &iwin,
        SIZE_LOG2,
        &mut hi,
    );
    assert!(
        ir.is_ok(),
        "interp subtree freeze returns a placeholder: {ir:?}"
    );
    assert_eq!(
        read_state(&isnap),
        STATE_UNWINDING,
        "interp artifact frozen"
    );
    let ires = hi.frozen_nested().to_vec();
    assert_eq!(ires.len(), 1, "interp captured one live nested child");

    // JIT: freeze `FREEZE_PARENT` (its instrumented child unwinds inline → one residue record). Same
    // `instantiate` call, so the same re-attach geometry.
    if !svm_jit::fiber_supported() {
        return;
    }
    let jinst = instrument(FREEZE_PARENT);
    let mut hj = Host::new();
    hj.set_durable(true);
    let jh = hj.grant_instantiator(0, WINDOW as u64);
    let mut jwin = init_durable_window(WINDOW);
    write_state(&mut jwin, STATE_UNWINDING);
    let (jo, jsnap, _fibers, jnested) = compile_and_run_capture_reserved_with_host_durable_nested(
        &jinst,
        0,
        &[jh as i64],
        &jwin,
        &[],
        &[],
        SIZE_LOG2,
        svm_run::cap_thunk,
        &mut hj as *mut Host as *mut c_void,
    )
    .expect("durable freeze run compiles");
    assert_eq!(read_state(&jsnap), STATE_UNWINDING, "JIT artifact frozen");
    assert_eq!(
        jnested.len(),
        1,
        "JIT captured one live nested child: {jo:?}"
    );
    // Record match: the child's re-attach geometry (identical `instantiate` call on both parents).
    assert_eq!(jnested[0].parent_task, ires[0].parent_task, "parent_task");
    assert_eq!(jnested[0].slot, ires[0].slot, "slot");
    assert_eq!(jnested[0].carve_off, ires[0].carve_off, "carve_off");
    assert_eq!(jnested[0].size_log2, ires[0].size_log2, "size_log2");
    assert_eq!(jnested[0].entry, ires[0].entry, "entry");
}
