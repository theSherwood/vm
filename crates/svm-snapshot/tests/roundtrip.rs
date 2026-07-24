//! The snapshot codec end-to-end: freeze → **serialize** → restore → thaw ≡ uninterrupted run,
//! through the real §12 artifact (not a kept-in-memory window), plus the two §12.6 invariants
//! (canonical re-serialize; identity-gated refusal) and the non-durable freeze refusal.

use svm_durable::{
    arm_freeze_after, begin_thaw, init_durable_window, transform_module,
    transform_module_assume_confined, write_state, STATE_UNWINDING,
};
use svm_interp::{run_capture_reserved_with_host, Host, Value};
use svm_ir::{Memory, Module};
use svm_snapshot::{freeze, restore, FreezeError, RestoreError};

const SIZE_LOG2: u8 = 18;
const WINDOW: usize = 1 << SIZE_LOG2;

// Calls Clock.now and *uses* the result after the call, so a thaw that re-issued the call
// (clock now 0) instead of reloading the saved 42 would be observable (100 vs 142).
const SRC: &str = r#"
func (i32) -> (i64) {
block 0 (v0: i32) {
  v1 = i32.const 0
  v2 = cap.call 2 0 (i32) -> (i64) v0 (v1)
  v3 = i64.const 100
  v4 = i64.add v2 v3
  return v4
  }
}
"#;

// A *different* instrumented module (adds 200), for the identity-gate test.
const SRC_OTHER: &str = r#"
func (i32) -> (i64) {
block 0 (v0: i32) {
  v1 = i32.const 0
  v2 = cap.call 2 0 (i32) -> (i64) v0 (v1)
  v3 = i64.const 200
  v4 = i64.add v2 v3
  return v4
  }
}
"#;

// A fiber'd module (slice 3.1.5): root resumes a fiber that suspends once, then resumes it again;
// the fiber returns 7 + 100. Freezing between the two resumes parks the fiber, so the artifact
// must carry the fiber's continuation (window shadow region) + its residue (Section 2).
const SRC_FIBER: &str = r#"
func () -> (i64) {
block 0 () {
  v0 = ref.func 1
  v1 = i64.const 4096
  v2 = cont.new v0 v1
  v3 = i64.const 0
  v4, v5 = cont.resume v2 v3
  v6 = i64.const 7
  v7, v8 = cont.resume v2 v6
  return v8
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  v2 = i64.const 42
  v3 = suspend v2
  v4 = i64.const 100
  v5 = i64.add v3 v4
  return v5
  }
}
"#;

fn instrument(src: &str) -> Module {
    let mut m = svm_text::parse_module(src).expect("parse");
    m.memory = Some(Memory {
        size_log2: SIZE_LOG2,
    });
    let inst = transform_module(&m).expect("transform");
    svm_verify::verify_module(&inst).expect("verify");
    inst
}

#[test]
fn freeze_serialize_restore_thaw_through_the_codec() {
    let inst = instrument(SRC);

    // Baseline: uninterrupted run, clock at 42.
    let mut host = Host::new();
    host.clock_ns = 42;
    let clk = host.grant_clock();
    let mut fuel = 100_000u64;
    let (baseline, _) = run_capture_reserved_with_host(
        &inst,
        0,
        &[Value::I32(clk)],
        &mut fuel,
        &init_durable_window(WINDOW),
        SIZE_LOG2,
        &mut host,
    );
    assert_eq!(
        baseline,
        Ok(vec![Value::I64(142)]),
        "uninterrupted: 42 + 100"
    );

    // Freeze run: UNWINDING → the poll unwinds out, leaving shadow state in the window.
    let mut fhost = Host::new();
    fhost.clock_ns = 42;
    let clk = fhost.grant_clock();
    let mut win = init_durable_window(WINDOW);
    write_state(&mut win, STATE_UNWINDING);
    let mut fuel = 100_000u64;
    let (frozen, snapshot) = run_capture_reserved_with_host(
        &inst,
        0,
        &[Value::I32(clk)],
        &mut fuel,
        &win,
        SIZE_LOG2,
        &mut fhost,
    );
    assert_eq!(
        frozen,
        Ok(vec![Value::I64(0)]),
        "freeze returns a placeholder"
    );

    // Serialize the real artifact.
    let artifact = freeze(&inst, &snapshot, &fhost).expect("freeze");

    // Restore into a FRESH host (clock now 0 — D-scope: resources aren't in the artifact).
    let mut thost = Host::new();
    let window = restore(&artifact, &inst, &mut thost).expect("restore");

    // §12.6 invariant 1 — canonical: re-serializing the freshly-restored domain at the same
    // safepoint reproduces the artifact byte-for-byte.
    assert_eq!(
        freeze(&inst, &window, &thost).expect("re-freeze"),
        artifact,
        "re-serialize of a restored domain is byte-identical"
    );

    // Thaw: flip to REWINDING and re-enter; the stack rebuilds from the restored window. The
    // guest receives the clock as a handle argument; restore reinstated it at its original
    // slot/generation, so the same handle value (`(generation << 8) | slot`) still resolves.
    let mut win = window;
    begin_thaw(&mut win, 0);
    let caps = thost.capture_durable_handles().expect("durable");
    let clk = ((caps[0].generation << 8) | caps[0].slot) as i32;
    let mut fuel = 100_000u64;
    let (thawed, _) = run_capture_reserved_with_host(
        &inst,
        0,
        &[Value::I32(clk)],
        &mut fuel,
        &win,
        SIZE_LOG2,
        &mut thost,
    );
    assert_eq!(thawed, baseline, "thawed run equals the uninterrupted run");
    assert_eq!(
        thawed,
        Ok(vec![Value::I64(142)]),
        "saved cap result (42) reloaded, not re-issued (which would give 100)"
    );
}

#[test]
fn fiber_freeze_serialize_restore_thaw_through_the_codec() {
    let inst = instrument(SRC_FIBER);

    // Baseline: uninterrupted run → 107.
    let mut host = Host::new();
    let mut fuel = 100_000u64;
    let (baseline, _) = run_capture_reserved_with_host(
        &inst,
        0,
        &[],
        &mut fuel,
        &init_durable_window(WINDOW),
        SIZE_LOG2,
        &mut host,
    );
    assert_eq!(
        baseline,
        Ok(vec![Value::I64(107)]),
        "uninterrupted: 7 + 100"
    );

    // Freeze run: UNWINDING from the start unwinds the root at resume #1 (fiber parked), then the
    // freeze driver flattens the parked fiber into its shadow region and exports its residue.
    let mut fhost = Host::new();
    fhost.set_durable(true);
    let mut win = init_durable_window(WINDOW);
    write_state(&mut win, STATE_UNWINDING);
    let mut fuel = 100_000u64;
    let (frozen, snapshot) =
        run_capture_reserved_with_host(&inst, 0, &[], &mut fuel, &win, SIZE_LOG2, &mut fhost);
    assert!(frozen.is_ok(), "freeze returns a placeholder: {frozen:?}");
    assert_eq!(fhost.frozen_fibers().len(), 1, "one fiber flattened");

    // Serialize the real artifact (now carrying Section 2 — the fiber residue).
    let artifact = freeze(&inst, &snapshot, &fhost).expect("freeze");

    // Restore into a FRESH host: re-grants handles (none here) and re-seeds the frozen fibers.
    let mut thost = Host::new();
    thost.set_durable(true);
    let window = restore(&artifact, &inst, &mut thost).expect("restore");
    assert_eq!(
        thost.frozen_fibers().len(),
        1,
        "restore re-seeded the frozen fiber"
    );

    // §12.6 invariant 1 — canonical: re-serializing the restored domain at the same safepoint
    // reproduces the artifact byte-for-byte (Section 2 included).
    assert_eq!(
        freeze(&inst, &window, &thost).expect("re-freeze"),
        artifact,
        "re-serialize of a restored fiber'd domain is byte-identical"
    );

    // Thaw: flip to REWINDING and re-enter. The root rewinds and re-issues cont.resume; the
    // re-seeded fiber re-enters its entry, rewinds, re-parks; forward execution then completes.
    let mut win = window;
    begin_thaw(&mut win, 0);
    let mut fuel = 100_000u64;
    let (thawed, _) =
        run_capture_reserved_with_host(&inst, 0, &[], &mut fuel, &win, SIZE_LOG2, &mut thost);
    assert_eq!(
        thawed, baseline,
        "thawed fiber'd run equals the uninterrupted run"
    );
}

#[test]
fn restore_refuses_a_mismatched_module() {
    // Freeze under SRC, then try to restore against a different instrumented module.
    let inst = instrument(SRC);
    let mut host = Host::new();
    host.grant_clock();
    let win = init_durable_window(WINDOW);
    let artifact = freeze(&inst, &win, &host).expect("freeze");

    let other = instrument(SRC_OTHER);
    let mut thost = Host::new();
    let err = restore(&artifact, &other, &mut thost).expect_err("digest mismatch must refuse");
    assert_eq!(err, RestoreError::ModuleMismatch, "R5 identity gate");
}

#[test]
fn freeze_refuses_a_non_durable_handle() {
    let inst = instrument(SRC);
    let mut host = Host::new();
    host.grant_clock();
    host.grant_io_ring(); // non-durable (carries out-of-line ring state)
    let win = init_durable_window(WINDOW);
    match freeze(&inst, &win, &host) {
        Err(FreezeError::NonDurableHandle(h)) => assert_eq!(h.slot, 1),
        other => panic!("expected NonDurableHandle refusal, got {other:?}"),
    }
}

/// §12.5 handle hardening: draining the non-durable handle turns the refusal above into a successful
/// serialize. `Host::drain_non_durable` closes the out-of-line `IoRing` (the guest relinquishes that
/// authority), leaving only the re-grantable `Clock`, so the domain becomes snapshottable.
#[test]
fn freeze_succeeds_after_draining_a_non_durable_handle() {
    let inst = instrument(SRC);
    let mut host = Host::new();
    host.grant_clock();
    host.grant_io_ring(); // non-durable — blocks the freeze until drained
    let win = init_durable_window(WINDOW);
    assert!(
        matches!(
            freeze(&inst, &win, &host),
            Err(FreezeError::NonDurableHandle(_))
        ),
        "the live io_ring refuses the freeze"
    );

    let drained = host.drain_non_durable();
    assert_eq!(drained.len(), 1, "the io_ring was drained");
    freeze(&inst, &win, &host).expect("a drained domain serializes");
}

// Multi-vCPU (slice 3.2.1): the root spawns a child over the shared window; both read the (advancing)
// clock once, then the root joins and sums. Frozen mid-run, the artifact must carry each vCPU's
// continuation (its own window shadow region) + the control-section residue (spawned vCPUs + the
// root's extent). The handle stash uses linear memory, so transform on the confined path.
const SRC_MULTIVCPU: &str = r#"
func (i32) -> (i64) {
block 0 (v0: i32) {
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
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  v2 = i64.const 65536
  v3 = i32.load v2
  v4 = i32.const 0
  v5 = cap.call 2 0 (i32) -> (i64) v3 (v4)
  v6 = i64.const 10
  v7 = i64.add v5 v6
  return v7
  }
}
"#;

#[test]
fn multivcpu_freeze_serialize_restore_thaw_through_the_codec() {
    let mut m = svm_text::parse_module(SRC_MULTIVCPU).expect("parse");
    m.memory = Some(Memory {
        size_log2: SIZE_LOG2,
    });
    let inst = svm_durable::transform_module_assume_confined(&m).expect("transform");
    svm_verify::verify_module(&inst).expect("verify");

    // Baseline: clock 42 → reads {42, 43}; result = 42 + (43 + 10) = 95 (order-invariant sum).
    let baseline = {
        let mut host = Host::new();
        host.set_durable(true);
        host.clock_ns = 42;
        let clk = host.grant_clock();
        let mut fuel = 100_000u64;
        let (r, _) = run_capture_reserved_with_host(
            &inst,
            0,
            &[Value::I32(clk)],
            &mut fuel,
            &init_durable_window(WINDOW),
            SIZE_LOG2,
            &mut host,
        );
        r.expect("uninterrupted")
    };
    assert_eq!(
        baseline,
        vec![Value::I64(95)],
        "uninterrupted: 42 + (43 + 10)"
    );

    // Freeze run: UNWINDING → both vCPUs unwind into their own regions; capture the window.
    let mut fhost = Host::new();
    fhost.set_durable(true);
    fhost.clock_ns = 42;
    let clk = fhost.grant_clock();
    let mut win = init_durable_window(WINDOW);
    write_state(&mut win, STATE_UNWINDING);
    let mut fuel = 100_000u64;
    let (frozen, snapshot) = run_capture_reserved_with_host(
        &inst,
        0,
        &[Value::I32(clk)],
        &mut fuel,
        &win,
        SIZE_LOG2,
        &mut fhost,
    );
    assert!(frozen.is_ok(), "freeze returns a placeholder: {frozen:?}");
    assert_eq!(fhost.frozen_vcpus().len(), 1, "one spawned vCPU flattened");

    // Serialize the real artifact (carrying the control section's vCPU residue + root extent).
    let artifact = freeze(&inst, &snapshot, &fhost).expect("freeze");

    // Restore into a FRESH host whose clock has advanced (D-scope: resources aren't in the artifact).
    let mut thost = Host::new();
    thost.set_durable(true);
    thost.clock_ns = 1000; // far past the saved reads; a re-issue would be observable
    let window = restore(&artifact, &inst, &mut thost).expect("restore");
    assert_eq!(
        thost.frozen_vcpus().len(),
        1,
        "restore re-seeded the spawned vCPU residue"
    );

    // §12.6 invariant 1 — canonical: re-serializing the freshly-restored domain reproduces the
    // artifact byte-for-byte (control section, vCPU residue included).
    assert_eq!(
        freeze(&inst, &window, &thost).expect("re-freeze"),
        artifact,
        "re-serialize of a restored multi-vCPU domain is byte-identical"
    );

    // Thaw: REWINDING re-enter. The child is re-spawned and rewinds; the root rewinds, joins, sums.
    // Both clock reads reload (42, 43 → 95), not re-issue (which would use clock 1000+ → ≠ 95).
    let mut win = window;
    begin_thaw(&mut win, 0);
    let clk = {
        let caps = thost.capture_durable_handles().expect("durable");
        ((caps[0].generation << 8) | caps[0].slot) as i32
    };
    let mut fuel = 100_000u64;
    let (thawed, _) = run_capture_reserved_with_host(
        &inst,
        0,
        &[Value::I32(clk)],
        &mut fuel,
        &win,
        SIZE_LOG2,
        &mut thost,
    );
    assert_eq!(
        thawed,
        Ok(baseline),
        "thawed multi-vCPU run equals the uninterrupted run"
    );
    assert_eq!(
        thawed,
        Ok(vec![Value::I64(95)]),
        "saved clock reads reloaded, not re-issued"
    );
}

// Slice 3.2.2 — vCPU + fiber coexistence through the codec. The root spawns a child vCPU and owns a
// fiber (which parks); the artifact's control section must carry BOTH the fiber residue and the
// spawned-vCPU residue (+ root extent), and the thaw must reconstruct both into non-colliding regions.
const SRC_FIBER_AND_VCPU: &str = r#"
func (i32) -> (i64) {
block 0 (v0: i32) {
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
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  v2 = i64.const 65536
  v3 = i32.load v2
  v4 = i32.const 0
  v5 = cap.call 2 0 (i32) -> (i64) v3 (v4)
  v6 = i64.const 10
  v7 = i64.add v5 v6
  return v7
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
fn vcpu_and_fiber_freeze_serialize_restore_thaw_through_the_codec() {
    let mut m = svm_text::parse_module(SRC_FIBER_AND_VCPU).expect("parse");
    m.memory = Some(Memory {
        size_log2: SIZE_LOG2,
    });
    let inst = svm_durable::transform_module_assume_confined(&m).expect("transform");
    svm_verify::verify_module(&inst).expect("verify");

    // Baseline: fiber yields 5; child reads clock (42) + 10; result = 5 + 52 = 57.
    let baseline = {
        let mut host = Host::new();
        host.set_durable(true);
        host.clock_ns = 42;
        let clk = host.grant_clock();
        let mut fuel = 100_000u64;
        let (r, _) = run_capture_reserved_with_host(
            &inst,
            0,
            &[Value::I32(clk)],
            &mut fuel,
            &init_durable_window(WINDOW),
            SIZE_LOG2,
            &mut host,
        );
        r.expect("uninterrupted")
    };
    assert_eq!(
        baseline,
        vec![Value::I64(57)],
        "uninterrupted: 5 + (42 + 10)"
    );

    // Freeze run.
    let mut fhost = Host::new();
    fhost.set_durable(true);
    fhost.clock_ns = 42;
    let clk = fhost.grant_clock();
    let mut win = init_durable_window(WINDOW);
    write_state(&mut win, STATE_UNWINDING);
    let mut fuel = 100_000u64;
    let (frozen, snapshot) = run_capture_reserved_with_host(
        &inst,
        0,
        &[Value::I32(clk)],
        &mut fuel,
        &win,
        SIZE_LOG2,
        &mut fhost,
    );
    assert!(frozen.is_ok(), "freeze returns a placeholder: {frozen:?}");
    assert_eq!(fhost.frozen_fibers().len(), 1, "fiber flattened");
    assert_eq!(fhost.frozen_vcpus().len(), 1, "spawned vCPU captured");

    // Serialize: the control section now carries BOTH a fiber record and a vCPU record.
    let artifact = freeze(&inst, &snapshot, &fhost).expect("freeze");

    // Restore on a fresh host whose clock advanced.
    let mut thost = Host::new();
    thost.set_durable(true);
    thost.clock_ns = 1000;
    let window = restore(&artifact, &inst, &mut thost).expect("restore");
    assert_eq!(thost.frozen_fibers().len(), 1, "fiber residue re-seeded");
    assert_eq!(thost.frozen_vcpus().len(), 1, "vCPU residue re-seeded");

    // §12.6 invariant 1 — canonical re-serialize is byte-identical (both residues included).
    assert_eq!(
        freeze(&inst, &window, &thost).expect("re-freeze"),
        artifact,
        "re-serialize of a restored vCPU+fiber domain is byte-identical"
    );

    // Thaw: the fiber re-seeds and re-parks, the child re-spawns and reloads its clock; result == 57.
    let mut win = window;
    begin_thaw(&mut win, 0);
    let clk = {
        let caps = thost.capture_durable_handles().expect("durable");
        ((caps[0].generation << 8) | caps[0].slot) as i32
    };
    let mut fuel = 100_000u64;
    let (thawed, _) = run_capture_reserved_with_host(
        &inst,
        0,
        &[Value::I32(clk)],
        &mut fuel,
        &win,
        SIZE_LOG2,
        &mut thost,
    );
    assert_eq!(
        thawed,
        Ok(baseline),
        "thawed vCPU+fiber run reloads its clock read (57), not re-issues it"
    );
}

// Recycling step 2 (DURABILITY.md §12.8): the control section carries each fiber's **generation**, so a
// thaw re-seeds a recycled (generation > 0) fiber at the generation its guest handle expects. The
// end-to-end durable round-trip of a recycled fiber isn't reachable with a freeze-at-first-safepoint
// harness (a recycled parked fiber requires a prior fiber-finish, i.e. a prior safepoint), so this
// pins the codec leg directly: a forced gen-N residue survives freeze → serialize → restore intact.
#[test]
fn fiber_residue_generation_round_trips_through_the_codec() {
    let inst = instrument(SRC_FIBER);

    // A real freeze produces a (gen-0) fiber residue + the matching window image.
    let mut fhost = Host::new();
    fhost.set_durable(true);
    let mut win = init_durable_window(WINDOW);
    write_state(&mut win, STATE_UNWINDING);
    let mut fuel = 100_000u64;
    let (frozen, snapshot) =
        run_capture_reserved_with_host(&inst, 0, &[], &mut fuel, &win, SIZE_LOG2, &mut fhost);
    assert!(frozen.is_ok(), "freeze placeholder: {frozen:?}");
    let mut residue = fhost.frozen_fibers().to_vec();
    assert_eq!(residue.len(), 1, "one fiber flattened");
    assert_eq!(
        residue[0].generation, 0,
        "a non-recycled fiber is generation 0"
    );

    // Stamp a non-zero generation (as slot recycling would) and serialize from a host carrying it.
    residue[0].generation = 7;
    let mut shost = Host::new();
    shost.set_durable(true);
    shost.set_frozen_fibers(residue);
    let artifact = freeze(&inst, &snapshot, &shost).expect("freeze");

    // Restore: the generation must come back intact (format v2 carries it).
    let mut thost = Host::new();
    thost.set_durable(true);
    let _ = restore(&artifact, &inst, &mut thost).expect("restore");
    let back = thost.frozen_fibers();
    assert_eq!(back.len(), 1, "residue restored");
    assert_eq!(
        back[0].generation, 7,
        "the fiber residue's generation round-trips through the codec"
    );
}

// A fiber'd module that **recycles a slot before freezing** (recycling step 2/3 + the mid-run freeze
// trigger, end to end). The root first runs fiber A (func 2) to completion — that *frees* registry
// slot 0 and bumps its generation to 1 — then creates the real fiber B (func 1) which reuses slot 0
// at **generation 1**, resumes B once (it suspends/parks), and resumes it again to completion (7 +
// 100). Freezing while B is parked must flatten B carrying generation 1, and the thaw must re-seed it
// there so B's guest handle ((1 << 24) | 0) still resolves. This is the case the freeze-before-start
// harness can't reach: a recycled parked fiber needs a prior fiber-finish (a prior safepoint), so the
// freeze has to land *mid-run* — which `arm_freeze_after` makes deterministic.
const SRC_RECYCLE: &str = r#"
func () -> (i64) {
block 0 () {
  v0 = ref.func 2
  v1 = i64.const 4096
  v2 = cont.new v0 v1
  v3 = i64.const 0
  v4, v5 = cont.resume v2 v3
  v6 = ref.func 1
  v7 = i64.const 4096
  v8 = cont.new v6 v7
  v9 = i64.const 0
  v10, v11 = cont.resume v8 v9
  v12 = i64.const 7
  v13, v14 = cont.resume v8 v12
  return v14
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  v2 = i64.const 42
  v3 = suspend v2
  v4 = i64.const 100
  v5 = i64.add v3 v4
  return v5
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  v2 = i64.const 0
  return v2
  }
}
"#;

#[test]
fn recycled_fiber_freeze_serialize_restore_thaw_through_the_codec() {
    let inst = instrument(SRC_RECYCLE);

    // Baseline: uninterrupted → 107 (A recycles slot 0, B reuses it and completes 7 + 100).
    let mut host = Host::new();
    let mut fuel = 100_000u64;
    let (baseline, _) = run_capture_reserved_with_host(
        &inst,
        0,
        &[],
        &mut fuel,
        &init_durable_window(WINDOW),
        SIZE_LOG2,
        &mut host,
    );
    assert_eq!(
        baseline,
        Ok(vec![Value::I64(107)]),
        "uninterrupted: 7 + 100"
    );

    // Freeze run: arm so the freeze lands mid-run, once slot 0 has been recycled and B is parked.
    // The trigger fires at the third safepoint (resume A; resume B #1; B's suspend), so B is parked
    // and slot 0 is at generation 1 when the root unwinds.
    let mut fhost = Host::new();
    fhost.set_durable(true);
    let mut win = init_durable_window(WINDOW);
    arm_freeze_after(&mut win, 3);
    let mut fuel = 100_000u64;
    let (frozen, snapshot) =
        run_capture_reserved_with_host(&inst, 0, &[], &mut fuel, &win, SIZE_LOG2, &mut fhost);
    assert!(frozen.is_ok(), "freeze returns a placeholder: {frozen:?}");
    let residue = fhost.frozen_fibers();
    assert_eq!(residue.len(), 1, "exactly the parked fiber B flattened");
    assert_eq!(residue[0].slot, 0, "B reused the recycled slot 0");
    assert_eq!(
        residue[0].generation, 1,
        "the recycled slot's generation (1) is recorded in the residue"
    );

    // Serialize → restore: the recycled generation survives the codec and re-seeds the fiber.
    let artifact = freeze(&inst, &snapshot, &fhost).expect("freeze");
    let mut thost = Host::new();
    thost.set_durable(true);
    let window = restore(&artifact, &inst, &mut thost).expect("restore");
    assert_eq!(thost.frozen_fibers().len(), 1, "restore re-seeded fiber B");
    assert_eq!(
        thost.frozen_fibers()[0].generation,
        1,
        "the re-seeded fiber is at generation 1 (its handle is (1 << 24) | 0)"
    );

    // §12.6 invariant 1 — canonical re-serialize is byte-identical.
    assert_eq!(
        freeze(&inst, &window, &thost).expect("re-freeze"),
        artifact,
        "re-serialize of a restored recycled-fiber domain is byte-identical"
    );

    // Thaw: the root rewinds and re-issues resume B; the gen-1 handle resolves to the re-seeded
    // fiber, which re-parks, and forward execution then resumes it to completion → 107.
    let mut win = window;
    begin_thaw(&mut win, 0);
    let mut fuel = 100_000u64;
    let (thawed, _) =
        run_capture_reserved_with_host(&inst, 0, &[], &mut fuel, &win, SIZE_LOG2, &mut thost);
    assert_eq!(
        thawed, baseline,
        "thawed recycled-fiber run equals the uninterrupted run"
    );
    assert_eq!(thawed, Ok(vec![Value::I64(107)]));
}

// Slice 3.4 (v4): a **nested** vCPU tree (root → child → grandchild) through the §12 codec — proves a
// non-zero `parent_task` round-trips (a grandchild's parent is its child, not the root) and the
// per-parent join table rebuilds on thaw.
const SRC_NESTED: &str = r#"
func (i32) -> (i64) {
block 0 (v0: i32) {
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
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  v2 = i64.const 65536
  v3 = i32.load v2
  v4 = i64.const 0
  v5 = i64.const 0
  v6 = thread.spawn 2 v4 v5
  v7 = i32.const 0
  v8 = cap.call 2 0 (i32) -> (i64) v3 (v7)
  v9 = thread.join v6
  v10 = i64.add v8 v9
  return v10
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  v2 = i64.const 65536
  v3 = i32.load v2
  v4 = i32.const 0
  v5 = cap.call 2 0 (i32) -> (i64) v3 (v4)
  return v5
  }
}
"#;

#[test]
fn nested_spawn_tree_freeze_serialize_restore_thaw_through_the_codec() {
    let mut m = svm_text::parse_module(SRC_NESTED).expect("parse");
    m.memory = Some(Memory {
        size_log2: SIZE_LOG2,
    });
    let inst = svm_durable::transform_module_assume_confined(&m).expect("transform");
    svm_verify::verify_module(&inst).expect("verify");

    // Baseline: clock 42 → reads {42,43,44} sum to 129 (order-invariant).
    let baseline = {
        let mut host = Host::new();
        host.set_durable(true);
        host.clock_ns = 42;
        let clk = host.grant_clock();
        let mut fuel = 100_000u64;
        let (r, _) = run_capture_reserved_with_host(
            &inst,
            0,
            &[Value::I32(clk)],
            &mut fuel,
            &init_durable_window(WINDOW),
            SIZE_LOG2,
            &mut host,
        );
        r.expect("uninterrupted")
    };
    assert_eq!(
        baseline,
        vec![Value::I64(129)],
        "uninterrupted: 42 + 43 + 44"
    );

    // Freeze run.
    let mut fhost = Host::new();
    fhost.set_durable(true);
    fhost.clock_ns = 42;
    let clk = fhost.grant_clock();
    let mut win = init_durable_window(WINDOW);
    write_state(&mut win, STATE_UNWINDING);
    let mut fuel = 100_000u64;
    let (frozen, snapshot) = run_capture_reserved_with_host(
        &inst,
        0,
        &[Value::I32(clk)],
        &mut fuel,
        &win,
        SIZE_LOG2,
        &mut fhost,
    );
    assert!(frozen.is_ok(), "freeze placeholder: {frozen:?}");
    assert_eq!(
        fhost.frozen_vcpus().len(),
        2,
        "child + grandchild flattened"
    );

    let artifact = freeze(&inst, &snapshot, &fhost).expect("freeze");

    // Restore into a fresh host with an advanced clock.
    let mut thost = Host::new();
    thost.set_durable(true);
    thost.clock_ns = 1000;
    let window = restore(&artifact, &inst, &mut thost).expect("restore");
    // The grandchild's non-zero `parent_task` survived the v4 codec (parent = the child, task 1).
    let mut seeded = thost.frozen_vcpus().to_vec();
    seeded.sort_by_key(|f| f.task);
    assert_eq!(seeded.len(), 2, "both vCPUs re-seeded");
    assert_eq!(
        (seeded[1].task, seeded[1].parent_task),
        (2, 1),
        "grandchild restores with parent_task = child (not root)"
    );

    // §12.6 canonical: re-serialize is byte-identical (parent_task included).
    assert_eq!(
        freeze(&inst, &window, &thost).expect("re-freeze"),
        artifact,
        "re-serialize of a restored nested tree is byte-identical"
    );

    // Thaw: the grandchild's handle resolves in the child's rebuilt table; all reads reload → 129.
    let mut win = window;
    begin_thaw(&mut win, 0);
    let clk = {
        let caps = thost.capture_durable_handles().expect("durable");
        ((caps[0].generation << 8) | caps[0].slot) as i32
    };
    let mut fuel = 100_000u64;
    let (thawed, _) = run_capture_reserved_with_host(
        &inst,
        0,
        &[Value::I32(clk)],
        &mut fuel,
        &win,
        SIZE_LOG2,
        &mut thost,
    );
    assert_eq!(
        thawed,
        Ok(baseline),
        "thawed nested tree equals the uninterrupted run (reload, not re-issue)"
    );
}

/// DURABILITY.md §13.4 step 3 (v13) — the **serve trio** round-trips: a domain frozen with a
/// non-empty inbound queue, completion cells, and a live ticket counter restores them exactly
/// (queue in FIFO order, cells intact, counter monotonic), and the §12.6 canonical re-freeze
/// of the restored domain is byte-identical.
#[test]
fn serve_state_round_trips_through_the_codec() {
    use svm_interp::SvcDispatch;

    let inst = instrument(SRC);
    let win = init_durable_window(WINDOW);

    let mut host = Host::new();
    host.set_svc_state(
        vec![
            SvcDispatch {
                export: 0,
                op: 1,
                args: vec![7, -3],
                ticket: 5,
            },
            SvcDispatch {
                export: 2,
                op: 0,
                args: vec![],
                ticket: 6,
            },
        ],
        vec![(3, 999), (4, -9)],
        7,
    );
    let artifact = freeze(&inst, &win, &host).expect("freeze with serve state");

    let mut rhost = Host::new();
    let rwin = restore(&artifact, &inst, &mut rhost).expect("restore");
    assert_eq!(
        rhost.svc_state(),
        host.svc_state(),
        "queue (FIFO), completion cells, and ticket counter restore exactly"
    );

    // §12.6: re-freezing the restored domain reproduces the artifact byte-for-byte.
    let refrozen = freeze(&inst, &rwin, &rhost).expect("re-freeze");
    assert_eq!(
        refrozen, artifact,
        "canonical re-serialize is byte-identical"
    );
}

/// The serve section is **elided** when the trio is empty (a never-served domain), so the
/// artifact's TLV walk carries no `TAG_SERVE` (4) — and restoring such an artifact leaves the
/// host's serve state at its empty default.
#[test]
fn empty_serve_state_elides_the_section() {
    let inst = instrument(SRC);
    let win = init_durable_window(WINDOW);
    let host = Host::new();
    let artifact = freeze(&inst, &win, &host).expect("freeze without serve state");

    // Walk the TLV container (magic, u16 version, then tag/len/body) and collect tags.
    let mut tags = Vec::new();
    let mut at = 6; // 4-byte magic + 2-byte version
    while at < artifact.len() {
        let (tag, n) = read_uleb_at(&artifact, at);
        at += n;
        let (len, n) = read_uleb_at(&artifact, at);
        at += n + len as usize;
        tags.push(tag);
    }
    assert!(
        !tags.contains(&4),
        "no TAG_SERVE section for an empty serve trio: {tags:?}"
    );

    let mut rhost = Host::new();
    restore(&artifact, &inst, &mut rhost).expect("restore");
    assert_eq!(
        rhost.svc_state(),
        (Vec::new(), Vec::new(), 0),
        "restored serve state stays at the empty default"
    );
}

/// Minimal LEB128 read for the TLV walk above: returns `(value, bytes_consumed)`.
fn read_uleb_at(b: &[u8], at: usize) -> (u64, usize) {
    let (mut v, mut shift, mut n) = (0u64, 0u32, 0usize);
    loop {
        let byte = b[at + n];
        v |= u64::from(byte & 0x7f) << shift;
        n += 1;
        if byte & 0x80 == 0 {
            return (v, n);
        }
        shift += 7;
    }
}

/// §13.4 slice 4b + v13 end-to-end: a **serving domain** frozen at its serve point round-trips
/// through the real artifact. The freeze run reaches `svc.poll` under `UNWINDING` (inert
/// sentinel — no drain), the artifact's serve section carries the untouched queue, and the
/// restored domain's re-issued serve op drains it: the handler finally runs against the
/// restored window and the completion cell fills — identical to the uninterrupted run.
#[test]
fn serving_freeze_serialize_restore_thaw_through_the_codec() {
    const SRC_SERVE: &str = r#"
memory 18
type 0 func (i64) -> (i64)
type 1 interface { bump: 0 }
export 0 interface "counter" 1 { bump: 1 }

func () -> (i64) {
block 0 () {
  vz = i32.const 0
  vn = cap.call 4294967295 9 () -> (i64) vz ()
  vc = i64.const 65600
  vafter = i64.load vc
  vk = i64.const 1000
  vm = i64.mul vn vk
  vr = i64.add vm vafter
  return vr
  }
}
func (i64) -> (i64) {
block 0 (vx: i64) {
  vc = i64.const 65600
  i64.store vc vx
  vone = i64.const 1
  vr = i64.add vx vone
  return vr
  }
}
"#;
    let mut m = svm_text::parse_module(SRC_SERVE).expect("parse");
    m.memory = Some(Memory {
        size_log2: SIZE_LOG2,
    });
    let inst = std::sync::Arc::new(transform_module_assume_confined(&m).expect("transform"));
    svm_verify::verify_module(&inst).expect("verify");

    // Freeze a domain with one queued dispatch, stopped at its serve point.
    let mut host = Host::new();
    host.set_durable(true);
    host.set_self_module(&inst);
    let ticket = host.svc_enqueue(0, 0, vec![41]).expect("enqueue");
    let mut win = init_durable_window(WINDOW);
    write_state(&mut win, STATE_UNWINDING);
    let mut fuel = 1_000_000u64;
    let (r, snap) =
        run_capture_reserved_with_host(&inst, 0, &[], &mut fuel, &win, SIZE_LOG2, &mut host);
    assert!(r.is_ok(), "freeze returns a placeholder: {r:?}");
    let artifact = freeze(&inst, &snap, &host).expect("freeze artifact");

    // Restore into a fresh host: the serve section re-seeds the trio; thaw drains it.
    let mut rhost = Host::new();
    let rwin = restore(&artifact, &inst, &mut rhost).expect("restore");
    assert_eq!(
        rhost.svc_state().0.len(),
        1,
        "the artifact carried the queued dispatch"
    );
    rhost.set_durable(true);
    rhost.set_self_module(&inst);
    let mut twin = rwin.clone();
    begin_thaw(&mut twin, 0);
    let mut fuel = 1_000_000u64;
    let (thawed, _) =
        run_capture_reserved_with_host(&inst, 0, &[], &mut fuel, &twin, SIZE_LOG2, &mut rhost);
    assert_eq!(
        thawed,
        Ok(vec![Value::I64(1041)]),
        "the restored domain served the restored dispatch (served*1000 + cell)"
    );
    assert_eq!(
        rhost.svc_result(ticket),
        Some(42),
        "the completion cell filled on thaw"
    );
}
