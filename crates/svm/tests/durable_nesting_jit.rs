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
//!   interpreter's;
//! * **thaw / round-trip** (`jit_nested_freeze_thaw_round_trips`): a thaw re-attaches + rewinds the
//!   frozen nested child so freeze→thaw ≡ uninterrupted (delivers 4950) — the correctness proof;
//! * **depth-2 freeze** (`jit_depth2_freeze_coalesces_grandchild_at_root`): a freeze catching a live
//!   child *and* grandchild coalesces both `FrozenNested` records at the root via a shared residue sink;
//! * **depth-2 thaw / round-trip** (`jit_depth2_freeze_thaw_round_trips`): a thaw recursively
//!   re-attaches the grandchild under its re-attached parent-child, so depth-2 freeze→thaw ≡
//!   uninterrupted (delivers 4950);
//! * **pure/completed child** (`jit_pure_child_freeze_thaw_round_trips`): a non-instrumented child
//!   *completes* under a freeze (rather than unwinding) yet still records + thaws by re-run — so the
//!   JIT needs no separate `completed_result` residue.
//!
//! Separate-module durable children and `coro_spawn` are follow-ups.

use core::ffi::c_void;
use svm_durable::{
    begin_thaw, init_durable_window, read_state, transform_module, write_state, STATE_NORMAL,
    STATE_UNWINDING,
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
block 0 (v0: i32) {
  v1 = i64.const 1
  v2 = i64.const 131072
  v3 = i64.const 17
  v4 = i64.const 0
  v5 = cap.call 6 0 (i64, i64, i64, i64) -> (i32) v0 (v1, v2, v3, v4)
  v6 = cap.call 6 1 (i32) -> (i64) v0 (v5)
  return v6
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  v2 = i64.const 0
  v3 = i64.const 0
  br 1(v2, v3)
}
block 1 (v4: i64, v5: i64) {
  v6 = i64.const 100
  v7 = i64.lt_s v4 v6
  br_if v7 2(v4, v5) 3(v5)
}
block 2 (v8: i64, v9: i64) {
  v10 = i64.add v9 v8
  v11 = i64.const 1
  v12 = i64.add v8 v11
  br 1(v12, v10)
}
block 3 (v13: i64) {
  return v13
  }
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
block 0 (v0: i32) {
  v1 = i64.const 1
  v2 = i64.const 262144
  v3 = i64.const 18
  v4 = i64.const 0
  v5 = cap.call 6 0 (i64, i64, i64, i64) -> (i32) v0 (v1, v2, v3, v4)
  v6 = cap.call 6 1 (i32) -> (i64) v0 (v5)
  return v6
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i32.const 256
  v2 = i64.const 2
  v3 = i64.const 131072
  v4 = i64.const 17
  v5 = i64.const 0
  v6 = cap.call 6 0 (i64, i64, i64, i64) -> (i32) v1 (v2, v3, v4, v5)
  v7 = cap.call 6 1 (i32) -> (i64) v1 (v6)
  return v7
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 0
  v2 = i64.const 0
  br 1(v1, v2)
}
block 1 (v3: i64, v4: i64) {
  v5 = i64.const 100
  v6 = i64.lt_s v3 v5
  br_if v6 2(v3, v4) 3(v4)
}
block 2 (v7: i64, v8: i64) {
  v9 = i64.add v8 v7
  v10 = i64.const 1
  v11 = i64.add v7 v10
  br 1(v11, v9)
}
block 3 (v12: i64) {
  return v12
  }
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
/// may-suspend loop that sums 0..100 = 4950: its `block4` holds a `cap.call` on a **statically-present
/// but never-taken** branch (`br_if 0`), so the transform makes it may-suspend and prepends a
/// **loop-header poll** at `block1`, yet the op never executes at runtime. So the child (a) can be
/// **frozen** — born `UNWINDING` it spills at that first poll (loop not yet entered), the live child a
/// freeze must capture — *and* (b) **thaws + runs uninterrupted cleanly** to 4950 (the dead branch is
/// never reached, so no `CapFault`). A *pure-compute* loop like `PARENT_SELF_LOOP`'s child has no poll
/// site, so the synchronous JIT would run it to completion (DURABILITY.md §4 "Freeze model").
const FREEZE_PARENT: &str = "memory 18
func (i32) -> (i64) {
block 0 (v0: i32) {
  v1 = i64.const 1
  v2 = i64.const 131072
  v3 = i64.const 17
  v4 = i64.const 0
  v5 = cap.call 6 0 (i64, i64, i64, i64) -> (i32) v0 (v1, v2, v3, v4)
  v6 = cap.call 6 1 (i32) -> (i64) v0 (v5)
  return v6
  }
}
func (i32) -> (i64) {
block 0 (v0: i32) {
  v1 = i64.const 0
  br 1(v1, v1, v0)
}
block 1 (v2: i64, v3: i64, v4: i32) {
  v5 = i64.const 100
  v6 = i64.lt_s v2 v5
  br_if v6 2(v2, v3, v4) 3(v3, v4)
}
block 2 (v7: i64, v8: i64, v9: i32) {
  v10 = i64.add v8 v7
  v11 = i64.const 1
  v12 = i64.add v7 v11
  br 1(v12, v10, v9)
}
block 3 (v13: i64, v14: i32) {
  v15 = i32.const 0
  br_if v15 4(v14) 5(v13)
}
block 4 (v16: i32) {
  v17 = cap.call 6 1 (i32) -> (i64) v16 (v16)
  return v17
}
block 5 (v18: i64) {
  return v18
  }
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
        &[], // no fiber seed
        &[], // no nested seed (this is a freeze, not a thaw)
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

/// The thaw side, closing the round-trip: **freeze → thaw → result** on the JIT delivers the
/// uninterrupted total (4950), the true correctness proof the freeze-export slice deferred. On thaw the
/// runtime re-attaches each frozen §14 child and **rewinds** it over its carve (thaw mode: `begin_thaw`
/// the carve — global state `NORMAL`, ctx-0 thaw word `REWINDING`, continuation preserved) to
/// completion *before* the parent re-enters, publishing its result at its join slot; the parent then
/// rewinds, reloads its `instantiate` handle, and its re-executed `join` resolves to the child's total.
#[test]
fn jit_nested_freeze_thaw_round_trips() {
    if !svm_jit::fiber_supported() {
        return;
    }
    let inst = instrument(FREEZE_PARENT);

    // (0) Uninterrupted baseline: the dead `cap.call` branch is never taken, so the chain returns 4950.
    let mut h0 = Host::new();
    h0.set_durable(true);
    let ih0 = h0.grant_instantiator(0, WINDOW as u64);
    let (o0, _w0, _f0, _n0) = compile_and_run_capture_reserved_with_host_durable_nested(
        &inst,
        0,
        &[ih0 as i64],
        &init_durable_window(WINDOW),
        &[], // init_prots
        &[], // fiber seed
        &[], // nested seed
        SIZE_LOG2,
        svm_run::cap_thunk,
        &mut h0 as *mut Host as *mut c_void,
    )
    .expect("uninterrupted compiles");
    assert!(
        matches!(o0, JitOutcome::Returned(ref s) if s == &[4950]),
        "uninterrupted total is 4950: {o0:?}"
    );

    // (1) Freeze-from-start: capture the artifact window + the nested-child residue.
    let mut hf = Host::new();
    hf.set_durable(true);
    let ihf = hf.grant_instantiator(0, WINDOW as u64);
    let mut fwin = init_durable_window(WINDOW);
    write_state(&mut fwin, STATE_UNWINDING);
    let (_of, artifact, _ff, nested) = compile_and_run_capture_reserved_with_host_durable_nested(
        &inst,
        0,
        &[ihf as i64],
        &fwin,
        &[], // init_prots
        &[], // fiber seed
        &[], // nested seed (freeze)
        SIZE_LOG2,
        svm_run::cap_thunk,
        &mut hf as *mut Host as *mut c_void,
    )
    .expect("freeze compiles");
    assert_eq!(read_state(&artifact), STATE_UNWINDING, "artifact frozen");
    assert_eq!(nested.len(), 1, "one nested child captured");

    // (2) Thaw the artifact with the nested seed: the child rewinds to completion, the parent's join
    // resolves, and the round-trip delivers the uninterrupted total.
    let mut twin = artifact.clone();
    begin_thaw(&mut twin, 0);
    let mut ht = Host::new();
    ht.set_durable(true);
    let iht = ht.grant_instantiator(0, WINDOW as u64);
    let (ot, tsnap, _ft, _nt) = compile_and_run_capture_reserved_with_host_durable_nested(
        &inst,
        0,
        &[iht as i64],
        &twin,
        &[],     // init_prots
        &[],     // fiber seed
        &nested, // nested seed — re-attach + rewind the frozen §14 child(ren)
        SIZE_LOG2,
        svm_run::cap_thunk,
        &mut ht as *mut Host as *mut c_void,
    )
    .expect("thaw compiles");
    assert!(
        matches!(ot, JitOutcome::Returned(ref s) if s == &[4950]),
        "thaw delivers the uninterrupted total (freeze→thaw ≡ uninterrupted): {ot:?}"
    );
    assert_eq!(read_state(&tsnap), STATE_NORMAL, "thaw back to NORMAL");
}

/// A durable **depth-2** chain where **both** nested levels are instrumented and unwind under a freeze:
/// root (func 0) → child (func 1, instrumented — it `instantiate`s the grandchild) → grandchild
/// (func 2, an instrumented dead-branch loop like `FREEZE_PARENT`'s child). Born `UNWINDING`, the child
/// executes `instantiate(grandchild)` then spills at its trailing poll, and the grandchild spills at its
/// first loop-header poll — so a freeze captures **two** live `FrozenNested` records.
const FREEZE_DEPTH2: &str = "memory 19
func (i32) -> (i64) {
block 0 (v0: i32) {
  v1 = i64.const 1
  v2 = i64.const 262144
  v3 = i64.const 18
  v4 = i64.const 0
  v5 = cap.call 6 0 (i64, i64, i64, i64) -> (i32) v0 (v1, v2, v3, v4)
  v6 = cap.call 6 1 (i32) -> (i64) v0 (v5)
  return v6
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i32.const 256
  v2 = i64.const 2
  v3 = i64.const 131072
  v4 = i64.const 17
  v5 = i64.const 0
  v6 = cap.call 6 0 (i64, i64, i64, i64) -> (i32) v1 (v2, v3, v4, v5)
  v7 = cap.call 6 1 (i32) -> (i64) v1 (v6)
  return v7
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 0
  br 1(v1, v1)
}
block 1 (v2: i64, v3: i64) {
  v4 = i64.const 100
  v5 = i64.lt_s v2 v4
  br_if v5 2(v2, v3) 3(v3)
}
block 2 (v6: i64, v7: i64) {
  v8 = i64.add v7 v6
  v9 = i64.const 1
  v10 = i64.add v6 v9
  br 1(v10, v8)
}
block 3 (v11: i64) {
  v12 = i32.const 0
  br_if v12 4(v11) 5(v11)
}
block 4 (v13: i64) {
  v14 = i32.const 256
  v15 = cap.call 6 1 (i32) -> (i64) v14 (v14)
  return v15
}
block 5 (v16: i64) {
  return v16
  }
}
";

/// Depth-2 freeze — the shared residue sink: a freeze catching a live **child and grandchild** captures
/// **both** `FrozenNested` records at the root, tagged `parent_task 0` (the root's direct child) and
/// `parent_task 1` (the grandchild, recorded by the child's nursery and coalesced at the root via the
/// shared sink). Without the sink the grandchild's record would be orphaned in the child's nursery and
/// the root would see only one. (Thaw of a depth-2 subtree is a follow-up.)
#[test]
fn jit_depth2_freeze_coalesces_grandchild_at_root() {
    if !svm_jit::fiber_supported() {
        return;
    }
    let inst = instrument(FREEZE_DEPTH2);
    let mut hj = Host::new();
    hj.set_durable(true);
    let jh = hj.grant_instantiator(0, D2_WINDOW as u64);
    let mut jwin = init_durable_window(D2_WINDOW);
    write_state(&mut jwin, STATE_UNWINDING);
    let (jo, jsnap, _fibers, jnested) = compile_and_run_capture_reserved_with_host_durable_nested(
        &inst,
        0,
        &[jh as i64],
        &jwin,
        &[], // init_prots
        &[], // fiber seed
        &[], // nested seed (freeze)
        D2_SIZE_LOG2,
        svm_run::cap_thunk,
        &mut hj as *mut Host as *mut c_void,
    )
    .expect("depth-2 durable freeze compiles");
    assert_eq!(read_state(&jsnap), STATE_UNWINDING, "artifact frozen");
    assert_eq!(
        jnested.len(),
        2,
        "both the child and grandchild coalesced at the root: {jo:?}"
    );
    // The child: the root's direct child (`parent_task 0`), carve at the root-relative 256 KiB.
    let child = jnested
        .iter()
        .find(|n| n.parent_task == 0)
        .expect("a root-child record (parent_task 0)");
    assert_eq!(
        child.carve_off, 262144,
        "child carve is root-relative 256 KiB"
    );
    assert_eq!(child.entry, 1, "child entry is func 1");
    // The grandchild: recorded by the child's nursery (`parent_task 1`) — the shared sink coalesced it
    // at the root rather than orphaning it in the child's nursery.
    let gchild = jnested
        .iter()
        .find(|n| n.parent_task == 1)
        .expect("a grandchild record (parent_task 1) — coalesced via the shared sink");
    assert_eq!(gchild.entry, 2, "grandchild entry is func 2");
}

/// Depth-2 thaw — the round-trip: **freeze → thaw → result** for a live child *and* grandchild delivers
/// the uninterrupted total (4950). On thaw the root re-attaches its direct child (rewind), and
/// `compile_child_and_run` recursively re-attaches the grandchild over the *child's* window before the
/// child runs — so the rewinding child's `join(grandchild)` resolves and the child completes, whose
/// result the root's `join` in turn resolves.
#[test]
fn jit_depth2_freeze_thaw_round_trips() {
    if !svm_jit::fiber_supported() {
        return;
    }
    let inst = instrument(FREEZE_DEPTH2);

    // (0) Uninterrupted baseline: the whole chain returns 4950 (the grandchild's dead branch is never
    // taken, so no `CapFault`).
    let mut h0 = Host::new();
    h0.set_durable(true);
    let ih0 = h0.grant_instantiator(0, D2_WINDOW as u64);
    let (o0, _w0, _f0, _n0) = compile_and_run_capture_reserved_with_host_durable_nested(
        &inst,
        0,
        &[ih0 as i64],
        &init_durable_window(D2_WINDOW),
        &[],
        &[],
        &[],
        D2_SIZE_LOG2,
        svm_run::cap_thunk,
        &mut h0 as *mut Host as *mut c_void,
    )
    .expect("uninterrupted compiles");
    assert!(
        matches!(o0, JitOutcome::Returned(ref s) if s == &[4950]),
        "depth-2 uninterrupted total is 4950: {o0:?}"
    );

    // (1) Freeze-from-start: capture the artifact + the two nested-child records.
    let mut hf = Host::new();
    hf.set_durable(true);
    let ihf = hf.grant_instantiator(0, D2_WINDOW as u64);
    let mut fwin = init_durable_window(D2_WINDOW);
    write_state(&mut fwin, STATE_UNWINDING);
    let (_of, artifact, _ff, nested) = compile_and_run_capture_reserved_with_host_durable_nested(
        &inst,
        0,
        &[ihf as i64],
        &fwin,
        &[],
        &[],
        &[],
        D2_SIZE_LOG2,
        svm_run::cap_thunk,
        &mut hf as *mut Host as *mut c_void,
    )
    .expect("freeze compiles");
    assert_eq!(read_state(&artifact), STATE_UNWINDING, "artifact frozen");
    assert_eq!(nested.len(), 2, "child + grandchild captured");

    // (2) Thaw with the nested seed: the child rewinds, the grandchild is recursively re-attached under
    // it, both joins resolve, and the round-trip delivers the uninterrupted total.
    let mut twin = artifact.clone();
    begin_thaw(&mut twin, 0);
    let mut ht = Host::new();
    ht.set_durable(true);
    let iht = ht.grant_instantiator(0, D2_WINDOW as u64);
    let (ot, tsnap, _ft, _nt) = compile_and_run_capture_reserved_with_host_durable_nested(
        &inst,
        0,
        &[iht as i64],
        &twin,
        &[],
        &[],
        &nested,
        D2_SIZE_LOG2,
        svm_run::cap_thunk,
        &mut ht as *mut Host as *mut c_void,
    )
    .expect("thaw compiles");
    assert!(
        matches!(ot, JitOutcome::Returned(ref s) if s == &[4950]),
        "depth-2 thaw delivers the uninterrupted total (freeze→thaw ≡ uninterrupted): {ot:?}"
    );
    assert_eq!(read_state(&tsnap), STATE_NORMAL, "thaw back to NORMAL");
}

/// A **pure-compute (non-instrumented) child** freeze→thaw round-trips too — a distinct path from the
/// instrumented-child tests. Born `UNWINDING` under a freeze, `PARENT_SELF_LOOP`'s func 1 has no poll
/// site, so it **runs to completion** in its carve (4950) rather than unwinding, yet its carve is left
/// `UNWINDING` (the seeded state word is never cleared), so the freeze still records it as a re-attach
/// record; thaw **re-runs** it (safe — a JIT nested child has no host caps, only idempotent window
/// writes) and delivers the same total. So the JIT needs no separate `completed_result` residue (the
/// interp's, for a *side-effecting* completed-but-unjoined child): record-and-re-run subsumes it here.
#[test]
fn jit_pure_child_freeze_thaw_round_trips() {
    if !svm_jit::fiber_supported() {
        return;
    }
    let inst = instrument(PARENT_SELF_LOOP);

    // Freeze-from-start: the pure child completes (4950) but its carve stays `UNWINDING`, so it rides
    // as one re-attach record.
    let mut hf = Host::new();
    hf.set_durable(true);
    let ihf = hf.grant_instantiator(0, WINDOW as u64);
    let mut fwin = init_durable_window(WINDOW);
    write_state(&mut fwin, STATE_UNWINDING);
    let (_of, artifact, _ff, nested) = compile_and_run_capture_reserved_with_host_durable_nested(
        &inst,
        0,
        &[ihf as i64],
        &fwin,
        &[],
        &[],
        &[],
        SIZE_LOG2,
        svm_run::cap_thunk,
        &mut hf as *mut Host as *mut c_void,
    )
    .expect("freeze compiles");
    assert_eq!(read_state(&artifact), STATE_UNWINDING, "artifact frozen");
    assert_eq!(
        nested.len(),
        1,
        "the pure child rides as one re-attach record"
    );

    // Thaw: re-run the child → 4950, the parent's join resolves, round-trip ≡ uninterrupted.
    let mut twin = artifact.clone();
    begin_thaw(&mut twin, 0);
    let mut ht = Host::new();
    ht.set_durable(true);
    let iht = ht.grant_instantiator(0, WINDOW as u64);
    let (ot, tsnap, _ft, _nt) = compile_and_run_capture_reserved_with_host_durable_nested(
        &inst,
        0,
        &[iht as i64],
        &twin,
        &[],
        &[],
        &nested,
        SIZE_LOG2,
        svm_run::cap_thunk,
        &mut ht as *mut Host as *mut c_void,
    )
    .expect("thaw compiles");
    assert!(
        matches!(ot, JitOutcome::Returned(ref s) if s == &[4950]),
        "pure-child freeze→thaw ≡ uninterrupted (4950): {ot:?}"
    );
    assert_eq!(read_state(&tsnap), STATE_NORMAL, "thaw back to NORMAL");
}
