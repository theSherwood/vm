//! Phase-3.2.1 — **multi-vCPU** durable freeze/thaw (interpreter, no live fibers).
//!
//! Phases 1–2 + 3.1/3.3 froze a single vCPU. This pins the first *multi-vCPU* freeze/thaw: a domain
//! whose root has spawned a `thread.spawn` child, frozen mid-run and thawed, equals the uninterrupted
//! run. The mechanism (DURABILITY.md §12.8 slice 3.2.1): the freeze/thaw run serializes onto a single
//! worker, the runtime swaps each vCPU's own state + shadow-SP words (`context = task id`) into the
//! shared window per dispatch, each unwinding child records a `FrozenVCpu` residue, and a thaw
//! re-spawns those children under `REWINDING` + rebuilds the root's join table — all transform-free
//! (`svm-durable` is unchanged; `thread.spawn` is not a checkpoint, so the reload lives in the runtime).
//!
//! Reload-not-reissue is the observable: both the root and the child read the (advancing) clock once;
//! a thaw on a host whose clock has moved on must reload each saved reading, not re-issue it.

use svm_durable::{
    init_durable_window, transform_module_assume_confined, write_state, STATE_REWINDING,
    STATE_UNWINDING,
};
use svm_interp::{run_capture_reserved_with_host, Host, Value};
use svm_ir::{Memory, Module};

const SIZE_LOG2: u8 = 17;
const WINDOW: usize = 1 << SIZE_LOG2;

// Root stashes the clock handle at a fixed guest byte (above the durable reserve), spawns a child over
// the shared window running it, calls the clock once itself, then joins the child and sums. The child
// loads the handle, calls the clock once, and returns clock + 10. `Clock.now` returns the counter then
// advances by one, so two calls yield the multiset {N, N+1} regardless of order — the baseline sum is
// order-invariant (it runs multi-worker), while the freeze/thaw run is single-worker.
const SRC: &str = r#"
func (i32) -> (i64) {
block0(v0: i32):
  v1 = i64.const 65536
  i32.store v1 v0
  v2 = i64.const 0
  v3 = i64.const 0
  v4 = thread.spawn 1 v2 v3
  v5 = i32.const 0
  v6 = cap.call 2 0 (i32) -> (i64) v0 (v5)
  v7 = thread.join v4
  v8 = i64.add v6 v7
  return v8
}
func (i64, i64) -> (i64) {
block0(v0: i64, v1: i64):
  v2 = i64.const 65536
  v3 = i32.load v2
  v4 = i32.const 0
  v5 = cap.call 2 0 (i32) -> (i64) v3 (v4)
  v6 = i64.const 10
  v7 = i64.add v5 v6
  return v7
}
"#;

fn instrument() -> Module {
    let mut m = svm_text::parse_module(SRC).expect("parse");
    m.memory = Some(Memory {
        size_log2: SIZE_LOG2,
    });
    // The guest uses linear memory (the handle stash above the reserve), so transform on the
    // cooperating-toolchain path.
    let inst = transform_module_assume_confined(&m).expect("transform");
    svm_verify::verify_module(&inst).expect("instrumented multi-vCPU IR verifies");
    inst
}

#[test]
fn two_vcpu_domain_freezes_and_thaws() {
    let inst = instrument();

    // Uninterrupted baseline: clock 42 → reads {42, 43}; result = root_read + (child_read + 10) =
    // 42 + 43 + 10 = 95 regardless of which vCPU got which reading (addition commutes).
    let want = {
        let mut h = Host::new();
        h.set_durable(true);
        h.clock_ns = 42;
        let clk = h.grant_clock();
        let mut fuel = 1_000_000u64;
        let (r, _) = run_capture_reserved_with_host(
            &inst,
            0,
            &[Value::I32(clk)],
            &mut fuel,
            &init_durable_window(WINDOW),
            SIZE_LOG2,
            &mut h,
        );
        r.expect("uninterrupted")
    };
    assert_eq!(want, vec![Value::I64(95)], "uninterrupted: 42 + (43 + 10)");

    // Freeze: UNWINDING from the start. Single-worker — the root runs (spawns the child, reads the
    // clock → 42), unwinds at its poll; then the child runs (reads the clock → 43), unwinds into its
    // own region. Capture the window + the child's residue.
    let (frozen, root_sp, snap, clock_after) = {
        let mut h = Host::new();
        h.set_durable(true);
        h.clock_ns = 42;
        let clk = h.grant_clock();
        let mut win = init_durable_window(WINDOW);
        write_state(&mut win, STATE_UNWINDING);
        let mut fuel = 1_000_000u64;
        let (r, snap) = run_capture_reserved_with_host(
            &inst,
            0,
            &[Value::I32(clk)],
            &mut fuel,
            &win,
            SIZE_LOG2,
            &mut h,
        );
        assert!(r.is_ok(), "freeze returns a placeholder: {r:?}");
        (
            h.frozen_vcpus().to_vec(),
            h.frozen_root_sp().expect("root extent recorded"),
            snap,
            h.clock_ns,
        )
    };
    assert_eq!(
        frozen.len(),
        1,
        "the spawned vCPU was captured as a FrozenVCpu, not lost"
    );
    assert_eq!(frozen[0].task, 1, "the child is task 1 (root is task 0)");
    assert_eq!(
        clock_after, 44,
        "the freeze ran both clock reads exactly once (42, 43 → counter 44)"
    );

    // Thaw on a host whose clock has *advanced* (44): both reads must reload their saved values (42,
    // 43), not re-issue (which would give 44, 45 → 99). Re-spawn the child + re-enter under REWINDING.
    let r_thaw = {
        let mut win = snap.clone();
        write_state(&mut win, STATE_REWINDING);
        let mut h = Host::new();
        h.set_durable(true);
        h.clock_ns = clock_after;
        let clk = h.grant_clock();
        h.set_frozen_vcpus(frozen);
        h.set_frozen_root_sp(root_sp);
        let mut fuel = 1_000_000u64;
        let (r, _) = run_capture_reserved_with_host(
            &inst,
            0,
            &[Value::I32(clk)],
            &mut fuel,
            &win,
            SIZE_LOG2,
            &mut h,
        );
        r
    };
    assert_eq!(
        r_thaw,
        Ok(want),
        "thawed two-vCPU domain reloads the saved clock reads (95), not re-issued ones (99)"
    );
}

// Slice 3.2.2 — the vCPU + fiber combination 3.2.1 could not do. The root spawns a child vCPU AND
// owns a fiber: with `context = task id` the child (task 1) and the fiber (slot 0 → context 1) would
// collide; the top-down vCPU layout puts the child at MAX_SHADOW_CTX while the fiber stays at context
// 1. The root spawns the child, creates a fiber that suspends (yielding 5, parked at the freeze), then
// unwinds at its `cont.resume` (its first may-suspend point). On thaw the parked fiber re-seeds and
// the child re-spawns into non-overlapping regions; the child's clock read reloads, not re-issues.
const SRC_FIBER_AND_VCPU: &str = r#"
func (i32) -> (i64) {
block0(v0: i32):
  v1 = i64.const 65536
  i32.store v1 v0
  v2 = i64.const 0
  v3 = i64.const 0
  v4 = thread.spawn 1 v2 v3
  v5 = ref.func 2
  v6 = i64.const 4096
  v7 = cont.new v5 v6
  v8 = i64.const 0
  v9, v10 = cont.resume v7 v8
  v13 = thread.join v4
  v14 = i64.add v10 v13
  return v14
}
func (i64, i64) -> (i64) {
block0(v0: i64, v1: i64):
  v2 = i64.const 65536
  v3 = i32.load v2
  v4 = i32.const 0
  v5 = cap.call 2 0 (i32) -> (i64) v3 (v4)
  v6 = i64.const 10
  v7 = i64.add v5 v6
  return v7
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

fn instrument_src(src: &str) -> Module {
    let mut m = svm_text::parse_module(src).expect("parse");
    m.memory = Some(Memory {
        size_log2: SIZE_LOG2,
    });
    let inst = transform_module_assume_confined(&m).expect("transform");
    svm_verify::verify_module(&inst).expect("verify");
    inst
}

#[test]
fn vcpu_and_fiber_coexist_through_freeze_thaw() {
    let inst = instrument_src(SRC_FIBER_AND_VCPU);

    // Baseline: fiber yields 5; child reads clock (42) + 10 = 52; result = 5 + 52 = 57.
    let want = {
        let mut h = Host::new();
        h.set_durable(true);
        h.clock_ns = 42;
        let clk = h.grant_clock();
        let mut fuel = 1_000_000u64;
        let (r, _) = run_capture_reserved_with_host(
            &inst,
            0,
            &[Value::I32(clk)],
            &mut fuel,
            &init_durable_window(WINDOW),
            SIZE_LOG2,
            &mut h,
        );
        r.expect("uninterrupted")
    };
    assert_eq!(want, vec![Value::I64(57)], "uninterrupted: 5 + (42 + 10)");

    // Freeze: UNWINDING. The root unwinds at cont.resume with the fiber parked (context 1); the child
    // unwinds into its top-down context. Capture window + both residues.
    let (frozen_fibers, frozen_vcpus, root_sp, snap, clock_after) = {
        let mut h = Host::new();
        h.set_durable(true);
        h.clock_ns = 42;
        let clk = h.grant_clock();
        let mut win = init_durable_window(WINDOW);
        write_state(&mut win, STATE_UNWINDING);
        let mut fuel = 1_000_000u64;
        let (r, snap) = run_capture_reserved_with_host(
            &inst,
            0,
            &[Value::I32(clk)],
            &mut fuel,
            &win,
            SIZE_LOG2,
            &mut h,
        );
        assert!(r.is_ok(), "freeze returns a placeholder: {r:?}");
        (
            h.frozen_fibers().to_vec(),
            h.frozen_vcpus().to_vec(),
            h.frozen_root_sp().expect("root extent recorded"),
            snap,
            h.clock_ns,
        )
    };
    assert_eq!(frozen_fibers.len(), 1, "the root's fiber was flattened");
    assert_eq!(frozen_vcpus.len(), 1, "the spawned vCPU was captured");
    assert_eq!(frozen_vcpus[0].task, 1, "child is task 1");
    assert!(
        clock_after > 42,
        "the freeze ran the child's clock read once"
    );

    // Thaw on a host whose clock has advanced: the child reloads 42, not re-issues (which would use
    // clock_after). The fiber re-seeds and the child re-spawns into non-overlapping regions.
    let r_thaw = {
        let mut win = snap.clone();
        write_state(&mut win, STATE_REWINDING);
        let mut h = Host::new();
        h.set_durable(true);
        h.clock_ns = clock_after;
        let clk = h.grant_clock();
        h.set_frozen_fibers(frozen_fibers);
        h.set_frozen_vcpus(frozen_vcpus);
        h.set_frozen_root_sp(root_sp);
        let mut fuel = 1_000_000u64;
        let (r, _) = run_capture_reserved_with_host(
            &inst,
            0,
            &[Value::I32(clk)],
            &mut fuel,
            &win,
            SIZE_LOG2,
            &mut h,
        );
        r
    };
    assert_eq!(
        r_thaw,
        Ok(want),
        "thawed vCPU+fiber domain reloads the saved clock read (57), not a re-issued one"
    );
}

// Slice 3.4 — a *spawned child* that **owns a fiber** (3.2.2 only covered the *root* owning one). The
// root just spawns + joins; the child reads the clock, creates a fiber that suspends (yielding 5,
// parked at the freeze), then unwinds. The child-owned parked fiber must be flattened into its own
// shadow region and re-seeded on thaw — which needs the *child* to run its own `freeze_drive` (the
// root's drive runs before the child exists, so it can't see the child's fiber), and the per-vCPU
// frozen-fiber residue to **accumulate** across vCPUs rather than clobber. The fiber sits at registry
// slot 0 → context 1 (grows up); the child sits at a top-down vCPU context — non-overlapping.
// The root spawns the child (`thread.spawn` is not a may-suspend op, so it executes before the freeze
// point), reads the clock (its first may-suspend op → it unwinds at the trailing poll, parking its
// continuation at the later `thread.join`), then joins + sums. The child's **first** may-suspend op is
// its own `cont.resume`, so the fiber runs + suspends (parks, yielding 5) *before* the child unwinds —
// leaving a child-owned parked fiber for the child's `freeze_drive` to flatten.
const SRC_CHILD_FIBER: &str = r#"
func (i32) -> (i64) {
block0(v0: i32):
  v1 = i64.const 0
  v2 = i64.const 0
  v3 = thread.spawn 1 v1 v2
  v4 = i32.const 0
  v5 = cap.call 2 0 (i32) -> (i64) v0 (v4)
  v6 = thread.join v3
  v7 = i64.add v5 v6
  return v7
}
func (i64, i64) -> (i64) {
block0(v0: i64, v1: i64):
  v2 = ref.func 2
  v3 = i64.const 4096
  v4 = cont.new v2 v3
  v5 = i64.const 0
  v6, v7 = cont.resume v4 v5
  v8 = i64.const 100
  v9 = i64.add v7 v8
  return v9
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
fn child_owns_fiber_through_freeze_thaw() {
    let inst = instrument_src(SRC_CHILD_FIBER);

    // Baseline: the root reads clock (42); the child's fiber yields 5 → child returns 5 + 100 = 105;
    // result = 42 + 105 = 147.
    let want = {
        let mut h = Host::new();
        h.set_durable(true);
        h.clock_ns = 42;
        let clk = h.grant_clock();
        let mut fuel = 1_000_000u64;
        let (r, _) = run_capture_reserved_with_host(
            &inst,
            0,
            &[Value::I32(clk)],
            &mut fuel,
            &init_durable_window(WINDOW),
            SIZE_LOG2,
            &mut h,
        );
        r.expect("uninterrupted")
    };
    assert_eq!(want, vec![Value::I64(147)], "uninterrupted: 42 + (5 + 100)");

    // Freeze: UNWINDING. The root unwinds (no fiber); the child reads the clock, parks its fiber at
    // cont.resume, and unwinds into its top-down context. Capture window + both residues.
    let (frozen_fibers, frozen_vcpus, root_sp, snap, clock_after) = {
        let mut h = Host::new();
        h.set_durable(true);
        h.clock_ns = 42;
        let clk = h.grant_clock();
        let mut win = init_durable_window(WINDOW);
        write_state(&mut win, STATE_UNWINDING);
        let mut fuel = 1_000_000u64;
        let (r, snap) = run_capture_reserved_with_host(
            &inst,
            0,
            &[Value::I32(clk)],
            &mut fuel,
            &win,
            SIZE_LOG2,
            &mut h,
        );
        assert!(r.is_ok(), "freeze returns a placeholder: {r:?}");
        (
            h.frozen_fibers().to_vec(),
            h.frozen_vcpus().to_vec(),
            h.frozen_root_sp().expect("root extent recorded"),
            snap,
            h.clock_ns,
        )
    };
    assert_eq!(
        frozen_fibers.len(),
        1,
        "the *child's* fiber was flattened (its own freeze_drive ran)"
    );
    assert_eq!(frozen_vcpus.len(), 1, "the spawned vCPU was captured");
    assert_eq!(frozen_vcpus[0].task, 1, "child is task 1");
    assert!(clock_after > 42, "the freeze ran the child's clock read once");

    // Thaw on a host whose clock has advanced: the root reloads 42, not re-issues. The child's fiber
    // re-seeds and the child re-spawns; forward execution reproduces the uninterrupted 147.
    let r_thaw = {
        let mut win = snap.clone();
        write_state(&mut win, STATE_REWINDING);
        let mut h = Host::new();
        h.set_durable(true);
        h.clock_ns = clock_after;
        let clk = h.grant_clock();
        h.set_frozen_fibers(frozen_fibers);
        h.set_frozen_vcpus(frozen_vcpus);
        h.set_frozen_root_sp(root_sp);
        let mut fuel = 1_000_000u64;
        let (r, _) = run_capture_reserved_with_host(
            &inst,
            0,
            &[Value::I32(clk)],
            &mut fuel,
            &win,
            SIZE_LOG2,
            &mut h,
        );
        r
    };
    assert_eq!(
        r_thaw,
        Ok(want),
        "thawed child-owned-fiber domain reloads the saved clock read (147), not a re-issued one"
    );
}
