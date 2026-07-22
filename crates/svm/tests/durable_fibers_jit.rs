//! Phase-3 slice 3.3.1 (DURABILITY.md §12.8): the **JIT** maintains the per-fiber shadow-SP swap
//! (D-fiber-cont option A), mirroring the interpreter (`durable_fibers.rs`). On a durable run the
//! JIT keeps the active shadow-SP word (`SHADOW_SP_OFF`) pointing at the running context's per-fiber
//! shadow region, swapping it on every `cont.resume`/return — so a freeze that lands while a fiber
//! runs on the native stack spills into that fiber's own region, never a sibling's.
//!
//! Observed directly (no freeze needed): a host-fn capability reads the active shadow-SP each time
//! it is called, and we drive a root that probes, runs two fibers (each probes), and probes again —
//! the JIT must route each context to a distinct region just as the interpreter does.
//!
//! Native stack switching exists on x86-64 unix, aarch64 unix, and x86-64 Windows
//! (`svm_fiber::supported()`); elsewhere the JIT bails `Unsupported` on fiber ops, so this is gated.
#![cfg(any(
    all(unix, target_arch = "x86_64"),
    all(unix, target_arch = "aarch64"),
    all(windows, target_arch = "x86_64")
))]

use core::ffi::c_void;
use std::sync::{Arc, Mutex};
use svm_durable::{
    arm_freeze_after, begin_thaw, init_durable_window, transform_module,
    transform_module_assume_confined, write_state, STATE_UNWINDING,
};
use svm_interp::{
    run_capture_reserved_with_host, FrozenFiber as InterpFrozen, Host, Value, DURABLE_RESERVE,
    SHADOW_BASE, SHADOW_STRIDE,
};
use svm_jit::{
    compile_and_run_capture_reserved_with_host_durable, FrozenFiber as JitFrozen, JitOutcome,
};
use svm_snapshot::{freeze, restore};
use svm_text::parse_module;
use svm_verify::verify_module;

/// A pure-arithmetic fiber module the cross-backend freeze/thaw tests share: root resumes a fiber
/// twice (the fiber suspends once, yielding 42, then returns 7 + 100 = 107). No caps ⇒ deterministic.
const FIBER_SRC: &str = "memory 17\n\
    func () -> (i64) {\n\
    block 0 () {\n\
    \x20 v0 = ref.func 1\n\
    \x20 v1 = i64.const 4096\n\
    \x20 v2 = cont.new v0 v1\n\
    \x20 v3 = i64.const 0\n\
    \x20 v4, v5 = cont.resume v2 v3\n\
    \x20 v6 = i64.const 7\n\
    \x20 v7, v8 = cont.resume v2 v6\n\
    \x20 return v8\n\
      }\n\
    }\n\
    func (i64, i64) -> (i64) {\n\
    block 0 (v0: i64, v1: i64) {\n\
    \x20 v2 = i64.const 42\n\
    \x20 v3 = suspend v2\n\
    \x20 v4 = i64.const 100\n\
    \x20 v5 = i64.add v3 v4\n\
    \x20 return v5\n\
      }\n\
    }\n";

fn jit_seed(interp: &[InterpFrozen]) -> Vec<JitFrozen> {
    interp
        .iter()
        .map(|f| JitFrozen {
            slot: f.slot,
            func: f.func,
            sp: f.sp,
            shadow_sp: f.shadow_sp,
            generation: f.generation,
        })
        .collect()
}

const WINDOW_LOG2: u8 = 17; // 128 KiB ≥ DURABLE_RESERVE (64 KiB)
const WINDOW: usize = 1 << WINDOW_LOG2;

#[test]
fn jit_durable_fiber_switch_routes_shadow_sp_per_context() {
    // Same module as the interpreter's `durable_fiber_switch_routes_shadow_sp_per_context`: root
    // (v0 = host-fn handle) probes, creates+resumes fiber A, creates+resumes fiber B, probes again.
    // Each fiber probes via a cap.call whose handle arrives (as i64) in the resume arg.
    // §12.8 4A.5: each probe passes `durable.shadow_base` (the active context's own region base, from
    // the runtime-private register) to the host fn, which records it — directly exercising per-context
    // routing (vs. the legacy single swapped `SHADOW_SP_OFF` word, now retired).
    let src = "memory 17\n\
        func (i32) -> (i64) {\n\
        block 0 (v0: i32) {\n\
        \x20 v1 = durable.shadow_base\n\
        \x20 v2 = cap.call 13 0 (i64) -> (i64) v0 (v1)\n\
        \x20 v3 = ref.func 1\n\
        \x20 v4 = i64.const 4096\n\
        \x20 v5 = cont.new v3 v4\n\
        \x20 v6 = i64.extend_i32_u v0\n\
        \x20 v7, v8 = cont.resume v5 v6\n\
        \x20 v9 = i64.const 8192\n\
        \x20 v10 = cont.new v3 v9\n\
        \x20 v11, v12 = cont.resume v10 v6\n\
        \x20 v13 = durable.shadow_base\n\
        \x20 v14 = cap.call 13 0 (i64) -> (i64) v0 (v13)\n\
        \x20 return v2\n\
          }\n\
        }\n\
        func (i64, i64) -> (i64) {\n\
        block 0 (v0: i64, v1: i64) {\n\
        \x20 v2 = i32.wrap_i64 v1\n\
        \x20 v3 = durable.shadow_base\n\
        \x20 v4 = cap.call 13 0 (i64) -> (i64) v2 (v3)\n\
        \x20 v5 = suspend v4\n\
        \x20 return v5\n\
          }\n\
        }\n";
    // The fibers `suspend` (rather than return) so both stay concurrently live in their own slots —
    // otherwise §12.8 recycling step 3 would reclaim fiber A's finished slot for fiber B, routing B
    // into A's region (context 1) instead of a distinct one (context 2).
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");

    // Each host-fn call records the active shadow-SP the JIT has installed for the running context.
    let probes: Arc<Mutex<Vec<u64>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&probes);
    let mut host = Host::new();
    let hf = host.grant_host_fn(Box::new(move |_op, args, _mem| {
        sink.lock().unwrap().push(args[0] as u64);
        Ok(vec![0])
    }));

    // A zeroed window (state = NORMAL); the per-context shadow-base comes from the runtime register,
    // not the window, so no seed is needed.
    let init = vec![0u8; WINDOW];

    let (outcome, _win, _residue) = compile_and_run_capture_reserved_with_host_durable(
        &m,
        0,
        &[hf as i64],
        &init,
        &[], // freeze-style: no page protections to re-establish
        &[], // no fibers to re-seed (a probe run, not a thaw)
        WINDOW_LOG2,
        svm_run::cap_thunk,
        &mut host as *mut Host as *mut c_void,
    )
    .expect("JIT compiles + runs the durable fiber module");
    assert!(
        matches!(outcome, JitOutcome::Returned(_)),
        "durable fiber run returned, got {outcome:?}"
    );

    let seen = probes.lock().unwrap().clone();
    assert_eq!(seen.len(), 4, "four probes: root, fiber A, fiber B, root");
    let root = SHADOW_BASE; // context 0
    let a = SHADOW_BASE + SHADOW_STRIDE; // fiber slot 0 → context 1
    let b = SHADOW_BASE + 2 * SHADOW_STRIDE; // fiber slot 1 → context 2
    assert_eq!(seen[0], root, "root runs in context 0's region");
    assert_eq!(
        seen[1], a,
        "fiber A runs in its own region (the JIT swapped in)"
    );
    assert_eq!(seen[2], b, "fiber B runs in a distinct region");
    assert_eq!(
        seen[3], root,
        "the JIT restored the root's region on return"
    );
    assert!(
        a != root && b != root && a != b,
        "per-context regions are distinct (no collision)"
    );
}

/// Phase-3 slice 3.3.2: the **JIT freeze driver** flattens a parked fiber into its shadow region
/// exactly as the interpreter does. Freezing the *same* instrumented fiber module (UNWINDING from
/// the start) on both backends drives the root's unwind and then flattens the parked fiber; the
/// resulting durable reserve — both contexts' shadow regions + the state/SP words — must be
/// **byte-identical** across backends (the same emitted IR spills the same values to the same
/// offsets), the cross-backend §7 property extended to fibers.
#[test]
fn jit_freeze_driver_flattens_a_fiber_matching_interp() {
    // Pure-arithmetic fiber (no caps → deterministic): root resumes a fiber that suspends once then
    // would return 7 + 100. Freezing from the start parks the fiber after its suspend.
    let src = "memory 17\n\
        func () -> (i64) {\n\
        block 0 () {\n\
        \x20 v0 = ref.func 1\n\
        \x20 v1 = i64.const 4096\n\
        \x20 v2 = cont.new v0 v1\n\
        \x20 v3 = i64.const 0\n\
        \x20 v4, v5 = cont.resume v2 v3\n\
        \x20 v6 = i64.const 7\n\
        \x20 v7, v8 = cont.resume v2 v6\n\
        \x20 return v8\n\
          }\n\
        }\n\
        func (i64, i64) -> (i64) {\n\
        block 0 (v0: i64, v1: i64) {\n\
        \x20 v2 = i64.const 42\n\
        \x20 v3 = suspend v2\n\
        \x20 v4 = i64.const 100\n\
        \x20 v5 = i64.add v3 v4\n\
        \x20 return v5\n\
          }\n\
        }\n";
    let m = parse_module(src).expect("parse");
    let inst = transform_module(&m).expect("transform");
    verify_module(&inst).expect("verify");

    // Interp freeze: UNWINDING → the root unwinds at resume #1 and the freeze driver flattens the
    // parked fiber into its region.
    let mut ihost = Host::new();
    ihost.set_durable(true);
    let mut iwin = init_durable_window(WINDOW);
    write_state(&mut iwin, STATE_UNWINDING);
    let mut ifuel = 1_000_000u64;
    let (ires, isnap) =
        run_capture_reserved_with_host(&inst, 0, &[], &mut ifuel, &iwin, WINDOW_LOG2, &mut ihost);
    assert!(
        ires.is_ok(),
        "interp freeze returns a placeholder: {ires:?}"
    );

    // JIT freeze: the new JIT freeze driver must flatten the same fiber identically.
    let mut jhost = Host::new();
    let mut jwin = init_durable_window(WINDOW);
    write_state(&mut jwin, STATE_UNWINDING);
    let (jout, jsnap, _residue) = compile_and_run_capture_reserved_with_host_durable(
        &inst,
        0,
        &[],
        &jwin,
        &[],
        &[], // freeze: no seed
        WINDOW_LOG2,
        svm_run::cap_thunk,
        &mut jhost as *mut Host as *mut c_void,
    )
    .expect("JIT freeze compiles + runs");
    assert!(
        matches!(jout, JitOutcome::Returned(_)),
        "JIT freeze returns a placeholder, got {jout:?}"
    );

    // The whole durable reserve (control words + both contexts' flattened shadow regions) must
    // match byte-for-byte: the interp and JIT freeze the fiber into the identical artifact.
    let reserve = DURABLE_RESERVE as usize;
    assert_eq!(
        &isnap[..reserve],
        &jsnap[..reserve],
        "interp/JIT freeze the parked fiber into a byte-identical durable reserve"
    );
}

/// Phase-3 slice 3.3.3 (export): the JIT **freeze** exports the same `FrozenFiber` residue the
/// interpreter does, so a JIT-frozen fiber domain serializes to a **byte-identical §12 artifact**
/// (window image + Section-2 fiber residue) — the artifact is backend-independent for fibers too.
#[test]
fn jit_and_interp_freeze_a_fiber_to_an_identical_artifact() {
    let m = parse_module(FIBER_SRC).expect("parse");
    let inst = transform_module(&m).expect("transform");
    verify_module(&inst).expect("verify");

    // Interp freeze → artifact_i.
    let mut ihost = Host::new();
    ihost.set_durable(true);
    let mut iwin = init_durable_window(WINDOW);
    write_state(&mut iwin, STATE_UNWINDING);
    let mut ifuel = 1_000_000u64;
    let (ires, isnap) =
        run_capture_reserved_with_host(&inst, 0, &[], &mut ifuel, &iwin, WINDOW_LOG2, &mut ihost);
    assert!(ires.is_ok());
    let artifact_i = freeze(&inst, &isnap, &ihost).expect("interp artifact");

    // JIT freeze → residue; serialize through the same codec (set the JIT residue on a host).
    let mut jhost = Host::new();
    let mut jwin = init_durable_window(WINDOW);
    write_state(&mut jwin, STATE_UNWINDING);
    let (_jout, jsnap, residue) = compile_and_run_capture_reserved_with_host_durable(
        &inst,
        0,
        &[],
        &jwin,
        &[],
        &[],
        WINDOW_LOG2,
        svm_run::cap_thunk,
        &mut jhost as *mut Host as *mut c_void,
    )
    .expect("JIT freeze");
    assert_eq!(
        residue.len(),
        1,
        "the JIT freeze exported the flattened fiber"
    );
    // Re-state the JIT's residue (as interp `FrozenFiber`) on a host and serialize identically.
    let mut jhost2 = Host::new();
    jhost2.set_frozen_fibers(
        residue
            .iter()
            .map(|f| InterpFrozen {
                slot: f.slot,
                func: f.func,
                sp: f.sp,
                shadow_sp: f.shadow_sp,
                generation: f.generation,
            })
            .collect(),
    );
    let artifact_j = freeze(&inst, &jsnap, &jhost2).expect("JIT artifact");

    assert_eq!(
        artifact_i, artifact_j,
        "interp/JIT freeze a fiber'd domain to a byte-identical §12 artifact (incl. Section 2)"
    );
}

/// Phase-3 slice 3.3.3 (thaw): an **interpreter-frozen** fiber artifact, restored through the §12
/// codec, **resumes on the JIT** and reproduces the uninterrupted result — the JIT re-seeds the
/// fiber from Section 2 and re-enters it under REWINDING (rewind → re-park → run forward). This
/// crosses both the backend boundary and the serialize/restore one, for fibers.
#[test]
fn interp_frozen_fiber_artifact_thaws_on_the_jit() {
    let m = parse_module(FIBER_SRC).expect("parse");
    let inst = transform_module(&m).expect("transform");
    verify_module(&inst).expect("verify");

    // Uninterrupted baseline (interp NORMAL): 7 + 100 = 107.
    let mut bhost = Host::new();
    let mut bfuel = 1_000_000u64;
    let (base, _) = run_capture_reserved_with_host(
        &inst,
        0,
        &[],
        &mut bfuel,
        &init_durable_window(WINDOW),
        WINDOW_LOG2,
        &mut bhost,
    );
    assert_eq!(base, Ok(vec![Value::I64(107)]), "uninterrupted: 7 + 100");

    // Interp freeze → serialize.
    let mut ihost = Host::new();
    ihost.set_durable(true);
    let mut iwin = init_durable_window(WINDOW);
    write_state(&mut iwin, STATE_UNWINDING);
    let mut ifuel = 1_000_000u64;
    let (ires, isnap) =
        run_capture_reserved_with_host(&inst, 0, &[], &mut ifuel, &iwin, WINDOW_LOG2, &mut ihost);
    assert!(ires.is_ok());
    let artifact = freeze(&inst, &isnap, &ihost).expect("freeze");

    // Restore (re-seeds the frozen fibers into a fresh host) → bridge the residue to the JIT.
    let mut thost = Host::new();
    let mut thaw_win = restore(&artifact, &inst, &mut thost).expect("restore");
    begin_thaw(&mut thaw_win, 0);
    let seed = jit_seed(thost.frozen_fibers());
    assert_eq!(seed.len(), 1, "the artifact carried one frozen fiber");

    // Thaw on the JIT: re-seed the fiber, re-enter under REWINDING, run to completion.
    let mut jhost = Host::new();
    let (jout, _win, _res) = compile_and_run_capture_reserved_with_host_durable(
        &inst,
        0,
        &[],
        &thaw_win,
        &[],
        &seed,
        WINDOW_LOG2,
        svm_run::cap_thunk,
        &mut jhost as *mut Host as *mut c_void,
    )
    .expect("JIT thaw");
    match jout {
        JitOutcome::Returned(slots) => {
            assert_eq!(
                slots,
                vec![107],
                "JIT thaw of the fiber artifact reproduces 107"
            )
        }
        other => panic!("JIT thaw: expected Returned([107]), got {other:?}"),
    }
}

/// Phase-3 slice 3.2 (active-resume-chain, cross-backend): an **interpreter-frozen** fiber that was
/// *running* (mid-`cap.call`), not idle, at freeze — restored through the §12 codec and **thawed on
/// the JIT**. The JIT must re-seed the active-chain fiber and re-enter it so it rewinds at its leaf
/// point and runs *forward*, **reloading** the saved clock value (47) rather than re-issuing the
/// clock against the JIT thaw host's advanced clock. The clock handle reaches the fiber via guest
/// memory (`transform_module_assume_confined`).
#[test]
fn interp_frozen_active_chain_fiber_thaws_on_the_jit() {
    let src = "memory 17\n\
        func (i32) -> (i64) {\n\
        block 0 (v0: i32) {\n\
        \x20 v1 = i64.const 65536\n\
        \x20 i32.store v1 v0\n\
        \x20 v2 = ref.func 1\n\
        \x20 v3 = i64.const 4096\n\
        \x20 v4 = cont.new v2 v3\n\
        \x20 v5 = i64.const 0\n\
        \x20 v6, v7 = cont.resume v4 v5\n\
        \x20 return v7\n\
          }\n\
        }\n\
        func (i64, i64) -> (i64) {\n\
        block 0 (v0: i64, v1: i64) {\n\
        \x20 v2 = i64.const 65536\n\
        \x20 v3 = i32.load v2\n\
        \x20 v4 = i32.const 0\n\
        \x20 v5 = cap.call 2 0 (i32) -> (i64) v3 (v4)\n\
        \x20 v6 = i64.const 5\n\
        \x20 v7 = i64.add v5 v6\n\
        \x20 return v7\n\
          }\n\
        }\n";
    let mut m = parse_module(src).expect("parse");
    m.memory = Some(svm_ir::Memory {
        size_log2: WINDOW_LOG2,
    });
    let inst = transform_module_assume_confined(&m).expect("transform");
    verify_module(&inst).expect("verify");

    // Interp freeze: clock 42 → F calls the clock (42), then unwinds mid-call; capture the artifact.
    let mut ihost = Host::new();
    ihost.set_durable(true);
    ihost.clock_ns = 42;
    let iclk = ihost.grant_clock();
    let mut iwin = init_durable_window(WINDOW);
    write_state(&mut iwin, STATE_UNWINDING);
    let mut ifuel = 1_000_000u64;
    let (ires, isnap) = run_capture_reserved_with_host(
        &inst,
        0,
        &[Value::I32(iclk)],
        &mut ifuel,
        &iwin,
        WINDOW_LOG2,
        &mut ihost,
    );
    assert!(ires.is_ok(), "interp freeze placeholder: {ires:?}");
    let artifact = freeze(&inst, &isnap, &ihost).expect("freeze serializes the active-chain fiber");

    // Restore + bridge the residue to the JIT.
    let mut thost = Host::new();
    let mut thaw_win = restore(&artifact, &inst, &mut thost).expect("restore");
    begin_thaw(&mut thaw_win, 0);
    let seed = jit_seed(thost.frozen_fibers());
    assert_eq!(seed.len(), 1, "the artifact carried the active-chain fiber");

    // Thaw on the JIT, clock advanced to 99 — the fiber must reload 42 (→ 47), not re-issue (→ 104).
    let mut jhost = Host::new();
    jhost.clock_ns = 99;
    let jclk = jhost.grant_clock();
    let (jout, _win, _res) = compile_and_run_capture_reserved_with_host_durable(
        &inst,
        0,
        &[jclk as i64],
        &thaw_win,
        &[],
        &seed,
        WINDOW_LOG2,
        svm_run::cap_thunk,
        &mut jhost as *mut Host as *mut c_void,
    )
    .expect("JIT thaw");
    match jout {
        JitOutcome::Returned(slots) => assert_eq!(
            slots,
            vec![47],
            "JIT thaw reloads the saved clock (47), not a re-issued one (104)"
        ),
        other => panic!("JIT thaw: expected Returned([47]), got {other:?}"),
    }
}

/// A churn module that **recycles a fiber slot before freezing** (recycling step 2/3 + the mid-run
/// freeze trigger, cross-backend). Fiber A (func 2) runs to completion — freeing registry slot 0 and
/// bumping its generation to 1 — then fiber B (func 1) reuses slot 0 at **generation 1**, is parked
/// once, and would resume to completion (7 + 100). Arming the freeze at the 3rd fiber safepoint
/// (resume A; resume B; B's suspend) lands it with B parked at generation 1.
const RECYCLE_SRC: &str = "memory 17\n\
    func () -> (i64) {\n\
    block 0 () {\n\
    \x20 v0 = ref.func 2\n\
    \x20 v1 = i64.const 4096\n\
    \x20 v2 = cont.new v0 v1\n\
    \x20 v3 = i64.const 0\n\
    \x20 v4, v5 = cont.resume v2 v3\n\
    \x20 v6 = ref.func 1\n\
    \x20 v7 = i64.const 4096\n\
    \x20 v8 = cont.new v6 v7\n\
    \x20 v9 = i64.const 0\n\
    \x20 v10, v11 = cont.resume v8 v9\n\
    \x20 v12 = i64.const 7\n\
    \x20 v13, v14 = cont.resume v8 v12\n\
    \x20 return v14\n\
      }\n\
    }\n\
    func (i64, i64) -> (i64) {\n\
    block 0 (v0: i64, v1: i64) {\n\
    \x20 v2 = i64.const 42\n\
    \x20 v3 = suspend v2\n\
    \x20 v4 = i64.const 100\n\
    \x20 v5 = i64.add v3 v4\n\
    \x20 return v5\n\
      }\n\
    }\n\
    func (i64, i64) -> (i64) {\n\
    block 0 (v0: i64, v1: i64) {\n\
    \x20 v2 = i64.const 0\n\
    \x20 return v2\n\
      }\n\
    }\n";

/// Recycling step 4 (cross-backend): with the **mid-run freeze trigger** (`arm_freeze_after`) the JIT
/// and the interpreter freeze a *recycled* parked fiber (generation 1) at the same armed safepoint, to
/// a **byte-identical** durable reserve + residue; and the artifact **thaws on the JIT** to the
/// uninterrupted result. This is the recycled durable round-trip the freeze-before-start harness could
/// not reach — both backends count the same fiber safepoints, so the armed freeze lands identically.
#[test]
fn jit_and_interp_freeze_a_recycled_fiber_identically_and_thaw_on_the_jit() {
    let m = parse_module(RECYCLE_SRC).expect("parse");
    let inst = transform_module(&m).expect("transform");
    verify_module(&inst).expect("verify");

    // Uninterrupted baseline: 107 (A recycles slot 0, B reuses it and completes 7 + 100).
    let mut bhost = Host::new();
    let mut bfuel = 1_000_000u64;
    let (base, _) = run_capture_reserved_with_host(
        &inst,
        0,
        &[],
        &mut bfuel,
        &init_durable_window(WINDOW),
        WINDOW_LOG2,
        &mut bhost,
    );
    assert_eq!(base, Ok(vec![Value::I64(107)]), "uninterrupted: 7 + 100");

    // Interp freeze, armed to fire at the 3rd fiber safepoint (B parked, slot 0 recycled to gen 1).
    let mut ihost = Host::new();
    ihost.set_durable(true);
    let mut iwin = init_durable_window(WINDOW);
    arm_freeze_after(&mut iwin, 3);
    let mut ifuel = 1_000_000u64;
    let (ires, isnap) =
        run_capture_reserved_with_host(&inst, 0, &[], &mut ifuel, &iwin, WINDOW_LOG2, &mut ihost);
    assert!(ires.is_ok(), "interp armed freeze: {ires:?}");
    let ir = ihost.frozen_fibers();
    assert_eq!(ir.len(), 1, "interp flattened the parked fiber B");
    assert_eq!(
        (ir[0].slot, ir[0].generation),
        (0, 1),
        "B = recycled slot 0, gen 1 (interp)"
    );

    // JIT freeze, identically armed: the JIT's per-thunk trigger must promote at the same safepoint.
    let mut jhost = Host::new();
    let mut jwin = init_durable_window(WINDOW);
    arm_freeze_after(&mut jwin, 3);
    let (jout, jsnap, jr) = compile_and_run_capture_reserved_with_host_durable(
        &inst,
        0,
        &[],
        &jwin,
        &[],
        &[], // freeze: no seed
        WINDOW_LOG2,
        svm_run::cap_thunk,
        &mut jhost as *mut Host as *mut c_void,
    )
    .expect("JIT armed freeze compiles + runs");
    assert!(
        matches!(jout, JitOutcome::Returned(_)),
        "JIT freeze placeholder, got {jout:?}"
    );
    assert_eq!(jr.len(), 1, "JIT flattened the parked fiber B");
    assert_eq!(
        (jr[0].slot, jr[0].generation),
        (0, 1),
        "B = recycled slot 0, gen 1 (JIT)"
    );

    // Byte-identical durable reserve (control words + both contexts' flattened shadow regions): the
    // two backends armed-freeze the recycled fiber into the same image.
    let reserve = DURABLE_RESERVE as usize;
    assert_eq!(
        &isnap[..reserve],
        &jsnap[..reserve],
        "interp/JIT armed-freeze the recycled fiber into a byte-identical durable reserve"
    );

    // Serialize the interp artifact, restore, and thaw on the JIT: the gen-1 handle resolves to the
    // re-seeded fiber, which re-parks, and forward execution resumes it to completion → 107.
    let artifact = freeze(&inst, &isnap, &ihost).expect("freeze");
    let mut thost = Host::new();
    let mut thaw_win = restore(&artifact, &inst, &mut thost).expect("restore");
    begin_thaw(&mut thaw_win, 0);
    let seed = jit_seed(thost.frozen_fibers());
    assert_eq!(seed.len(), 1, "the artifact carried the recycled fiber");
    assert_eq!(seed[0].generation, 1, "re-seeded at generation 1");

    let mut jhost2 = Host::new();
    let (jout2, _win, _res) = compile_and_run_capture_reserved_with_host_durable(
        &inst,
        0,
        &[],
        &thaw_win,
        &[],
        &seed,
        WINDOW_LOG2,
        svm_run::cap_thunk,
        &mut jhost2 as *mut Host as *mut c_void,
    )
    .expect("JIT thaw");
    match jout2 {
        JitOutcome::Returned(slots) => {
            assert_eq!(slots, vec![107], "JIT thaw of the recycled fiber → 107")
        }
        other => panic!("JIT thaw: expected Returned([107]), got {other:?}"),
    }
}
