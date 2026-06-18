//! Phase-3 slice 3.3 (DURABILITY.md §12.8): the **JIT** freezes a *multi-vCPU* durable domain
//! exactly as the interpreter does. A durable freeze runs **single-worker** — the interp serializes
//! onto one cooperative worker, and the JIT (whose vCPUs are 1:1 OS threads) instead runs each
//! `thread.spawn`ed child **inline** on the spawning thread while the window state ≠ NORMAL. The one
//! shared set of durable control words (state + active shadow-SP) is never raced; each child unwinds
//! into its own top-down shadow context and records a `FrozenVCpu` residue.
//!
//! Pinned here (freeze side): freezing the *same* instrumented two-vCPU module (UNWINDING from the
//! start) on both backends must (1) flatten the child into a **byte-identical durable reserve**, and
//! (2) export the **same `FrozenVCpu` residue** (task id, entry func, spawn args, flattened
//! shadow-SP) — the cross-backend §7 property extended to spawned vCPUs. The thaw side (runtime
//! re-attach under REWINDING) is a follow-up; the interpreter already pins the full freeze→thaw
//! round-trip in `svm-durable/tests/multivcpu.rs`.
//!
//! Native stack switching (for the inline child's guarded run) exists on x86-64 unix, aarch64 unix,
//! and x86-64 Windows; elsewhere the JIT bails `Unsupported` on `thread.*`/`cont.*`, so this is gated.
#![cfg(any(
    all(unix, target_arch = "x86_64"),
    all(unix, target_arch = "aarch64"),
    all(windows, target_arch = "x86_64")
))]

use core::ffi::c_void;
use svm_durable::{
    init_durable_window, transform_module_assume_confined, write_state, STATE_REWINDING,
    STATE_UNWINDING,
};
use svm_interp::{run_capture_reserved_with_host, Host, Value, DURABLE_RESERVE, SHADOW_BASE};
use svm_ir::{Memory, Module};
use svm_jit::{
    compile_and_run_capture_reserved_with_host_durable_mv, FrozenVCpu as JitVCpu, JitError,
    JitOutcome,
};

const SIZE_LOG2: u8 = 17;
const WINDOW: usize = 1 << SIZE_LOG2;

// Same module as the interpreter's `two_vcpu_domain_freezes_and_thaws`: the root stashes the clock
// handle at a fixed guest byte (above the durable reserve), spawns a child over the shared window
// running it, reads the clock once, then joins the child and sums. The child loads the handle, reads
// the clock once, returns clock + 10.
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
    let inst = transform_module_assume_confined(&m).expect("transform");
    svm_verify::verify_module(&inst).expect("instrumented multi-vCPU IR verifies");
    inst
}

/// The JIT runs a spawned child **inline** during a freeze and flattens it into the same durable
/// reserve, exporting the same `FrozenVCpu` residue as the interpreter.
#[test]
fn jit_freezes_a_spawned_vcpu_matching_interp() {
    let inst = instrument();

    // Interp freeze: UNWINDING from the start (single-worker). The root runs (spawns the child, reads
    // the clock → 42), unwinds at its poll; then the child runs (reads the clock → 43), unwinds into
    // its own top-down region. Capture the window image + the child's residue.
    let (ifrozen, isnap) = {
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
        assert!(r.is_ok(), "interp freeze returns a placeholder: {r:?}");
        (h.frozen_vcpus().to_vec(), snap)
    };
    assert_eq!(ifrozen.len(), 1, "interp captured the spawned vCPU");
    assert_eq!(ifrozen[0].task, 1, "the child is task 1 (root is task 0)");

    // JIT freeze: the child runs inline (single-worker) and unwinds into its own region. Skip on
    // Unsupported / host allocation pressure (mirroring the other cross-backend JIT durable tests).
    let mut jhost = Host::new();
    jhost.set_durable(true);
    jhost.clock_ns = 42;
    let clk = jhost.grant_clock();
    let mut jwin = init_durable_window(WINDOW);
    write_state(&mut jwin, STATE_UNWINDING);
    let (jout, jsnap, jfibers, jvcpus, _jroot_sp) =
        match compile_and_run_capture_reserved_with_host_durable_mv(
            &inst,
            0,
            &[clk as i64],
            &jwin,
            &[],
            &[],         // freeze: no fiber seed
            &[],         // freeze: no vcpu seed
            SHADOW_BASE, // freeze: root_sp unused
            SIZE_LOG2,
            svm_run::cap_thunk,
            &mut jhost as *mut Host as *mut c_void,
        ) {
        Ok(t) => t,
        Err(JitError::Unsupported(_)) => return,
        Err(JitError::Backend(msg)) if msg.contains("Allocation error") => return,
        Err(e) => panic!("JIT failed to compile a verified multi-vCPU module: {e:?}\n{inst:#?}"),
    };
    assert!(
        matches!(jout, JitOutcome::Returned(_)),
        "JIT freeze returns a placeholder, got {jout:?}"
    );
    assert!(jfibers.is_empty(), "no fibers in this module");

    // (1) The two backends flatten the child into a byte-identical durable reserve (control words +
    // both contexts' shadow regions): the same emitted IR spills the same values to the same offsets.
    let reserve = DURABLE_RESERVE as usize;
    assert_eq!(
        &isnap[..reserve],
        &jsnap[..reserve],
        "interp/JIT freeze the spawned vCPU into a byte-identical durable reserve"
    );

    // (2) The exported `FrozenVCpu` residue matches field-for-field (task id, entry func, spawn args,
    // flattened shadow-SP) — so a JIT-frozen multi-vCPU domain re-attaches its children exactly as an
    // interp-frozen one does.
    assert_eq!(jvcpus.len(), 1, "the JIT exported the spawned vCPU");
    assert_eq!(jvcpus[0].task, ifrozen[0].task, "same task id");
    assert_eq!(jvcpus[0].func, ifrozen[0].func, "same entry func");
    assert_eq!(jvcpus[0].args, ifrozen[0].args, "same spawn args");
    assert_eq!(
        jvcpus[0].shadow_sp, ifrozen[0].shadow_sp,
        "same flattened shadow-SP extent"
    );
}

/// The JIT **thaws** a multi-vCPU domain it froze: the spawned child is re-attached + run (rewinds
/// from its restored extent, runs forward to completion), and the root re-enters under `REWINDING` and
/// resolves its `thread.join`. Thawing on a host whose clock has *advanced* must reproduce the
/// uninterrupted result — both vCPUs **reload** their saved clock reads (42, 43), they do not re-issue
/// them (which would read the advanced clock) — the §12.6 freeze/thaw equivalence, multi-vCPU, on the JIT.
#[test]
fn jit_thaws_its_own_multivcpu_freeze() {
    let inst = instrument();

    // Uninterrupted baseline: clock 42 → reads {42, 43}; result = 42 + (43 + 10) = 95 (order-invariant).
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

    // JIT freeze (UNWINDING): capture the window image, the child residue, and the root's extent.
    let mut fhost = Host::new();
    fhost.set_durable(true);
    fhost.clock_ns = 42;
    let fclk = fhost.grant_clock();
    let mut fwin = init_durable_window(WINDOW);
    write_state(&mut fwin, STATE_UNWINDING);
    let (fout, fsnap, _ff, fvcpus, froot_sp) =
        match compile_and_run_capture_reserved_with_host_durable_mv(
            &inst,
            0,
            &[fclk as i64],
            &fwin,
            &[],
            &[],
            &[],
            SHADOW_BASE,
            SIZE_LOG2,
            svm_run::cap_thunk,
            &mut fhost as *mut Host as *mut c_void,
        ) {
            Ok(t) => t,
            Err(JitError::Unsupported(_)) => return,
            Err(JitError::Backend(msg)) if msg.contains("Allocation error") => return,
            Err(e) => panic!("JIT freeze failed: {e:?}\n{inst:#?}"),
        };
    assert!(matches!(fout, JitOutcome::Returned(_)), "freeze placeholder");
    assert_eq!(fvcpus.len(), 1, "the freeze captured the spawned child");
    assert_eq!(
        fhost.clock_ns, 44,
        "the freeze ran both clock reads once (42, 43 → 44)"
    );

    // JIT thaw on a host whose clock has *advanced* to 44: re-attach the child + restore the root's
    // extent, re-enter under REWINDING. Reload (42, 43) → 95; a re-issue would read {44, 45} → 99.
    let mut twin = fsnap.clone();
    write_state(&mut twin, STATE_REWINDING);
    let mut thost = Host::new();
    thost.set_durable(true);
    thost.clock_ns = 44;
    let tclk = thost.grant_clock();
    assert_eq!(tclk, fclk, "fresh host re-grants the same clock handle");
    let (tout, _tsnap, _tf, tvcpus, _troot) =
        match compile_and_run_capture_reserved_with_host_durable_mv(
            &inst,
            0,
            &[tclk as i64],
            &twin,
            &[],
            &[],        // no fibers
            &fvcpus,    // re-attach the frozen child
            froot_sp,   // restore the root's extent
            SIZE_LOG2,
            svm_run::cap_thunk,
            &mut thost as *mut Host as *mut c_void,
        ) {
            Ok(t) => t,
            Err(JitError::Unsupported(_)) => return,
            Err(JitError::Backend(msg)) if msg.contains("Allocation error") => return,
            Err(e) => panic!("JIT thaw failed: {e:?}\n{inst:#?}"),
        };
    assert!(tvcpus.is_empty(), "a thaw re-freezes nothing");
    match tout {
        JitOutcome::Returned(rs) => assert_eq!(
            rs,
            vec![95],
            "thawed two-vCPU domain reloads the saved clock reads (95), not re-issued ones (99)"
        ),
        other => panic!("thaw did not return cleanly: {other:?}"),
    }
}

/// An **interpreter-frozen** multi-vCPU domain **thaws on the JIT** and reproduces the uninterrupted
/// result — crossing the backend boundary. The interp's `FrozenVCpu` residue + root extent drive the
/// JIT thaw directly (the residue is a portable host-side record), so the re-attached child reloads its
/// saved clock read on the JIT just as it would on the interp.
#[test]
fn interp_frozen_multivcpu_thaws_on_the_jit() {
    let inst = instrument();

    // Interp freeze (UNWINDING): capture the window image, the child residue, and the root's extent.
    let (ivcpus, iroot_sp, isnap) = {
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
        assert!(r.is_ok(), "interp freeze placeholder: {r:?}");
        (
            h.frozen_vcpus().to_vec(),
            h.frozen_root_sp().expect("root extent recorded"),
            snap,
        )
    };
    assert_eq!(ivcpus.len(), 1, "interp captured the spawned child");

    // Bridge the interp residue to the JIT (same fields), then thaw on the JIT with an advanced clock.
    let seed: Vec<JitVCpu> = ivcpus
        .iter()
        .map(|v| JitVCpu {
            task: v.task,
            func: v.func,
            args: v.args.clone(),
            shadow_sp: v.shadow_sp,
        })
        .collect();
    let mut twin = isnap.clone();
    write_state(&mut twin, STATE_REWINDING);
    let mut thost = Host::new();
    thost.set_durable(true);
    thost.clock_ns = 44;
    let tclk = thost.grant_clock();
    let (tout, ..) = match compile_and_run_capture_reserved_with_host_durable_mv(
        &inst,
        0,
        &[tclk as i64],
        &twin,
        &[],
        &[],
        &seed,
        iroot_sp,
        SIZE_LOG2,
        svm_run::cap_thunk,
        &mut thost as *mut Host as *mut c_void,
    ) {
        Ok(t) => t,
        Err(JitError::Unsupported(_)) => return,
        Err(JitError::Backend(msg)) if msg.contains("Allocation error") => return,
        Err(e) => panic!("JIT thaw of interp freeze failed: {e:?}\n{inst:#?}"),
    };
    match tout {
        JitOutcome::Returned(rs) => assert_eq!(
            rs,
            vec![95],
            "interp-frozen multi-vCPU domain thaws on the JIT to the uninterrupted result (95)"
        ),
        other => panic!("cross-backend thaw did not return cleanly: {other:?}"),
    }
}
