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
use svm_durable::{init_durable_window, transform_module, write_state, STATE_UNWINDING};
use svm_interp::{
    run_capture_reserved_with_host, Host, DURABLE_RESERVE, SHADOW_BASE, SHADOW_SP_OFF,
    SHADOW_STRIDE,
};
use svm_jit::{compile_and_run_capture_reserved_with_host_durable, JitOutcome};
use svm_text::parse_module;
use svm_verify::verify_module;

const WINDOW_LOG2: u8 = 17; // 128 KiB ≥ DURABLE_RESERVE (64 KiB)
const WINDOW: usize = 1 << WINDOW_LOG2;

#[test]
fn jit_durable_fiber_switch_routes_shadow_sp_per_context() {
    // Same module as the interpreter's `durable_fiber_switch_routes_shadow_sp_per_context`: root
    // (v0 = host-fn handle) probes, creates+resumes fiber A, creates+resumes fiber B, probes again.
    // Each fiber probes via a cap.call whose handle arrives (as i64) in the resume arg.
    let src = "memory 17\n\
        func (i32) -> (i64) {\n\
        block0(v0: i32):\n\
        \x20 v1 = cap.call 13 0 () -> (i64) v0 ()\n\
        \x20 v2 = ref.func 1\n\
        \x20 v3 = i64.const 4096\n\
        \x20 v4 = cont.new v2 v3\n\
        \x20 v5 = i64.extend_i32_u v0\n\
        \x20 v6, v7 = cont.resume v4 v5\n\
        \x20 v8 = i64.const 8192\n\
        \x20 v9 = cont.new v2 v8\n\
        \x20 v10, v11 = cont.resume v9 v5\n\
        \x20 v12 = cap.call 13 0 () -> (i64) v0 ()\n\
        \x20 return v1\n\
        }\n\
        func (i64, i64) -> (i64) {\n\
        block0(v0: i64, v1: i64):\n\
        \x20 v2 = i32.wrap_i64 v1\n\
        \x20 v3 = cap.call 13 0 () -> (i64) v2 ()\n\
        \x20 return v3\n\
        }\n";
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");

    // Each host-fn call records the active shadow-SP the JIT has installed for the running context.
    let probes: Arc<Mutex<Vec<u64>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&probes);
    let mut host = Host::new();
    let hf = host.grant_host_fn(Box::new(move |_op, _args, mem| {
        let m = mem.expect("durable module has a window");
        let bytes = m.read_bytes(SHADOW_SP_OFF, 8).expect("shadow-SP readable");
        let sp = u64::from_le_bytes(bytes.try_into().unwrap());
        sink.lock().unwrap().push(sp);
        Ok(vec![0])
    }));

    // Seed the window so the root's active shadow-SP starts at its (context-0) region base.
    let mut init = vec![0u8; WINDOW];
    init[SHADOW_SP_OFF as usize..SHADOW_SP_OFF as usize + 8]
        .copy_from_slice(&SHADOW_BASE.to_le_bytes());

    let (outcome, _win) = compile_and_run_capture_reserved_with_host_durable(
        &m,
        0,
        &[hf as i64],
        &init,
        &[], // freeze-style: no page protections to re-establish
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
        block0():\n\
        \x20 v0 = ref.func 1\n\
        \x20 v1 = i64.const 4096\n\
        \x20 v2 = cont.new v0 v1\n\
        \x20 v3 = i64.const 0\n\
        \x20 v4, v5 = cont.resume v2 v3\n\
        \x20 v6 = i64.const 7\n\
        \x20 v7, v8 = cont.resume v2 v6\n\
        \x20 return v8\n\
        }\n\
        func (i64, i64) -> (i64) {\n\
        block0(v0: i64, v1: i64):\n\
        \x20 v2 = i64.const 42\n\
        \x20 v3 = suspend v2\n\
        \x20 v4 = i64.const 100\n\
        \x20 v5 = i64.add v3 v4\n\
        \x20 return v5\n\
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
    let (jout, jsnap) = compile_and_run_capture_reserved_with_host_durable(
        &inst,
        0,
        &[],
        &jwin,
        &[],
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
