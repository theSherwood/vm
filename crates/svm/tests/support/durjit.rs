//! Cross-backend arm of the freeze/thaw equivalence property (DURABILITY.md §7/§12.6).
//!
//! The instrumented IR the durable transform emits is ordinary control flow + window
//! loads/stores — nothing VM-specific — so the *same* artifact must drive identically on
//! the reference interpreter and the Cranelift JIT. This module reuses the in-scope
//! generator (`durgen`, shared with the interp-only `durable_fuzz` smoke) and asserts the
//! two backends agree on three claims:
//!
//!   1. **NORMAL** — instrumented run returns the same value on both backends;
//!   2. **freeze** — both leave a byte-identical shadow region *and* serialize through the
//!      `svm-snapshot` §12 codec to a **byte-identical artifact** (header digest + handle
//!      table + window image), so the persisted artifact is backend-independent; plus the
//!      §12.6 **canonical** invariant (restore → re-serialize reproduces the artifact);
//!   3. **thaw portability** — the artifact frozen by the **interpreter** is *restored through
//!      the codec* and resumed on the **JIT** under a *different* host clock, still reproducing
//!      the uninterrupted result (so the JIT reloads the saved `cap.call` value, it does not
//!      re-issue the call) and flipping the state word back to `NORMAL`. This crosses both the
//!      backend boundary and the serialize/restore one.
//!
//! Shared by the stable `cargo test` driver (`crates/svm/tests/durable_jit.rs`) and the
//! libFuzzer target (`fuzz/fuzz_targets/durable_jit.rs`), mirroring `irgen`/`durgen`.

#![allow(dead_code)] // each includer uses a subset

#[path = "../../../svm-durable/tests/support/durgen.rs"]
pub mod durgen;

use core::ffi::c_void;
use durgen::{
    gen_loop_module, gen_module, gen_recycle_fiber_module, Gen, RecycleModule, SIZE_LOG2, WINDOW,
};
use svm_durable::{
    arm_freeze_after, begin_thaw, init_durable_window, read_state, transform_module, write_state,
    DURABLE_RESERVE, SHADOW_SP_OFF, STATE_NORMAL, STATE_UNWINDING,
};
use svm_interp::{run_capture_reserved_with_host, FrozenFiber as InterpFrozen, Host, Trap, Value};
use svm_ir::{Module, ValType};
use svm_jit::{
    compile_and_run_capture_reserved_with_host, compile_and_run_capture_reserved_with_host_durable,
    FrozenFiber as JitFrozen, JitError, JitOutcome,
};
use svm_snapshot::{freeze, restore};

fn from_slot(t: ValType, s: i64) -> Value {
    match t {
        ValType::I32 => Value::I32(s as i32),
        ValType::I64 => Value::I64(s),
        ValType::F32 => Value::F32(f32::from_bits(s as u32)),
        ValType::F64 => Value::F64(f64::from_bits(s as u64)),
        ValType::V128 => {
            let mut b = [0u8; 16];
            b[..8].copy_from_slice(&s.to_le_bytes());
            Value::V128(b)
        }
        ValType::Ref => Value::Ref(s as u64),
    }
}

fn read_sp(w: &[u8]) -> usize {
    let mut b = [0u8; 8];
    b.copy_from_slice(&w[SHADOW_SP_OFF as usize..SHADOW_SP_OFF as usize + 8]);
    u64::from_le_bytes(b) as usize
}

fn window_with(state: i32) -> Vec<u8> {
    let mut w = init_durable_window(WINDOW);
    write_state(&mut w, state);
    w
}

/// Run the instrumented entry on the interpreter against `window`, with the clock seeded to
/// `clock_v`. Returns the result and the post-run window snapshot.
/// Returns the result, the post-run window snapshot, and the host's final `clock_ns`
/// (how far the monotonic clock advanced — used to seed the continuation host on thaw).
fn interp_run(
    inst: &Module,
    clock_v: i64,
    window: &[u8],
) -> (Result<Vec<Value>, Trap>, Vec<u8>, i64, Host) {
    let mut h = Host::new();
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

/// Run the instrumented entry on the JIT against `window`, with the clock seeded to
/// `clock_v`. `None` if the JIT declines to compile (it only ever sees lowered ops, so this
/// is a safety valve, not an expected path). On `Some`, also returns the final `clock_ns`.
fn jit_run(inst: &Module, clock_v: i64, window: &[u8]) -> Option<(JitOutcome, Vec<u8>, i64, Host)> {
    let mut h = Host::new();
    let clk = h.grant_clock();
    h.clock_ns = clock_v;
    let slots = [clk as i64];
    match compile_and_run_capture_reserved_with_host(
        inst,
        0,
        &slots,
        window,
        SIZE_LOG2,
        svm_run::cap_thunk,
        &mut h as *mut Host as *mut c_void,
    ) {
        Ok((o, win)) => Some((o, win, h.clock_ns, h)),
        Err(JitError::Unsupported(_)) => None,
        // A transient host **allocation** failure — e.g. Windows `ERROR_COMMITMENT_LIMIT`
        // (os error 1455) under the cumulative compile/window-commit churn of `cargo test
        // --workspace` — is an environment condition, not a backend divergence. Skip the
        // case (like an unsupported op) rather than fail the equivalence property. The marker
        // is Cranelift's `ModuleError::Allocation` Display ("Allocation error: …"), specific
        // to memory exhaustion, so a genuine codegen/lowering bug still panics below.
        Err(JitError::Backend(msg)) if msg.contains("Allocation error") => None,
        Err(e) => panic!("JIT failed to compile a verified instrumented module: {e:?}\n{inst:#?}"),
    }
}

/// Assert the JIT `Returned` the same scalar values the interpreter produced.
fn assert_returned_eq(inst: &Module, want: &[Value], got: JitOutcome, phase: &str) {
    match got {
        JitOutcome::Returned(slots) => {
            let results = &inst.funcs[0].results;
            assert_eq!(
                slots.len(),
                results.len(),
                "{phase}: result arity\n{inst:#?}"
            );
            for (i, (t, s)) in results.iter().zip(slots).enumerate() {
                assert_eq!(
                    from_slot(*t, s),
                    want[i],
                    "interp/JIT disagree in {phase} phase\n{inst:#?}"
                );
            }
        }
        other => panic!("JIT {phase}: expected Returned, got {other:?}\n{inst:#?}"),
    }
}

/// Check the cross-backend property on one generated in-scope **call-chain** module.
pub fn fuzz_one_xbackend(g: &mut Gen) {
    let m = gen_module(g);
    let clock_v = g.u64v() as i64;
    check_xbackend(&m, clock_v);
}

/// Check the cross-backend property on one generated **poll-free-loop** module (Phase-4 Slice A):
/// the loop-header back-edge poll must freeze byte-identically on both backends and thaw across the
/// backend boundary, like any other resume point.
pub fn fuzz_loop_one_xbackend(g: &mut Gen) {
    let m = gen_loop_module(g);
    let clock_v = g.u64v() as i64;
    check_xbackend(&m, clock_v);
}

fn check_xbackend(m: &Module, clock_v: i64) {
    let inst = transform_module(m).expect("an in-scope module must transform");
    svm_verify::verify_module(&inst).expect("instrumented IR must verify");

    // Interpreter reference: the uninterrupted result and the frozen artifact. `clock_after`
    // is how far the clock advanced during freeze — the thaw host continues from there
    // (D-scope: the host clock is not in the artifact), so any suspend points re-performed
    // after the resume match the baseline while the frozen point's result is reloaded.
    let (base_i, _, _, _) = interp_run(&inst, clock_v, &window_with(STATE_NORMAL));
    let base = base_i.expect("generated programs are trap-free");
    let (fr_i, snap_i, clock_after, host_i) =
        interp_run(&inst, clock_v, &window_with(STATE_UNWINDING));
    assert!(fr_i.is_ok(), "interp freeze returns a placeholder");
    assert_eq!(read_state(&snap_i), STATE_UNWINDING);

    // (1) NORMAL: the JIT agrees with the interpreter's value.
    let Some((j_base, _, _, _)) = jit_run(&inst, clock_v, &window_with(STATE_NORMAL)) else {
        return;
    };
    assert_returned_eq(&inst, &base, j_base, "NORMAL");

    // (2) freeze: the persisted artifact is byte-identical across backends.
    let Some((j_fr, snap_j, _, host_j)) = jit_run(&inst, clock_v, &window_with(STATE_UNWINDING))
    else {
        return;
    };
    assert!(
        matches!(j_fr, JitOutcome::Returned(_)),
        "JIT freeze returns a placeholder, not a trap\n{inst:#?}"
    );
    assert_eq!(
        read_state(&snap_j),
        STATE_UNWINDING,
        "JIT leaves the artifact UNWINDING\n{inst:#?}"
    );
    let sp = read_sp(&snap_i).max(read_sp(&snap_j));
    assert_eq!(
        &snap_i[..sp],
        &snap_j[..sp],
        "interp/JIT freeze artifacts diverge over the live shadow region [0,{sp})\n{inst:#?}"
    );

    // (2b) The real §12 artifact is backend-independent: serializing each backend's freeze
    // through the snapshot codec yields *byte-identical* bytes (header digest + handle table +
    // sparse window image) — a stronger claim than the raw shadow-region compare above.
    let art_i = freeze(&inst, &snap_i, &host_i).expect("interp freeze serializes");
    let art_j = freeze(&inst, &snap_j, &host_j).expect("JIT freeze serializes");
    assert_eq!(
        art_i, art_j,
        "interp/JIT produce a byte-identical §12 artifact\n{inst:#?}"
    );

    // (2c) §12.6 canonical invariant: restore the artifact (re-granting its handle table into a
    // fresh host) and re-serialize at the same safepoint — byte-identical to the original, so
    // restore reconstructed the window + handle table exactly.
    let mut rhost = Host::new();
    let rwin = restore(&art_i, &inst, &mut rhost).expect("artifact restores");
    assert_eq!(
        freeze(&inst, &rwin, &rhost).expect("re-freeze"),
        art_i,
        "re-serialize of a restored domain is byte-identical\n{inst:#?}"
    );

    // (3) thaw portability through the codec: resume the **restored** interpreter artifact on
    // the JIT, on a host whose clock *continues* from the freeze. Must reproduce `base` (the
    // frozen point's result reloaded, not re-issued — a re-issue would consume the next tick)
    // and end NORMAL. This crosses both the backend boundary and the serialize/restore one.
    let mut thaw_win = rwin;
    begin_thaw(&mut thaw_win, 0);
    let Some((j_thaw, final_j, _, _)) = jit_run(&inst, clock_after, &thaw_win) else {
        return;
    };
    assert_returned_eq(&inst, &base, j_thaw, "thaw");
    assert_eq!(
        read_state(&final_j),
        STATE_NORMAL,
        "JIT thaw must flip the state word back to NORMAL\n{inst:#?}"
    );
}

/// Cross-backend recycling freeze/thaw property (recycling step 4): a churn module recycles a slot
/// to generation > 0, parks the real fiber there, and is frozen **mid-run** via `arm_freeze_after`.
/// The interpreter and the JIT must flatten the recycled fiber to a **byte-identical** durable
/// reserve + §12 artifact (residue generation included), and the interp-frozen artifact must **thaw
/// on the JIT** to the uninterrupted result — the recycled durable round-trip, both backends.
pub fn fuzz_recycle_fiber_one_xbackend(g: &mut Gen) {
    let RecycleModule {
        module: m,
        arm,
        generation,
    } = gen_recycle_fiber_module(g);
    let inst = transform_module(&m).expect("an in-scope recycling fiber module must transform");
    svm_verify::verify_module(&inst).expect("instrumented recycling IR must verify");

    // Interpreter reference: uninterrupted baseline, then the mid-run armed freeze (B parked at the
    // recycled generation).
    let base = {
        let mut h = Host::new();
        h.set_durable(true);
        let mut fuel = 1_000_000u64;
        let (r, _) = run_capture_reserved_with_host(
            &inst,
            0,
            &[],
            &mut fuel,
            &init_durable_window(WINDOW),
            SIZE_LOG2,
            &mut h,
        );
        r.expect("generated recycling programs are trap-free")
    };
    let (isnap, ihost, ifibers) = {
        let mut h = Host::new();
        h.set_durable(true);
        let mut win = init_durable_window(WINDOW);
        arm_freeze_after(&mut win, arm);
        let mut fuel = 1_000_000u64;
        let (r, snap) =
            run_capture_reserved_with_host(&inst, 0, &[], &mut fuel, &win, SIZE_LOG2, &mut h);
        assert!(
            r.is_ok(),
            "interp armed freeze returns a placeholder\n{inst:#?}"
        );
        let fibers = h.frozen_fibers().to_vec();
        (snap, h, fibers)
    };
    assert_eq!(
        (ifibers[0].slot, ifibers[0].generation),
        (0, generation),
        "interp froze the recycled slot 0 at the bumped generation\n{inst:#?}"
    );

    // JIT armed freeze (skip on Unsupported / host allocation pressure, like `jit_run`).
    let mut jhost = Host::new();
    let mut jwin = init_durable_window(WINDOW);
    arm_freeze_after(&mut jwin, arm);
    let (jout, jsnap, jfibers) = match compile_and_run_capture_reserved_with_host_durable(
        &inst,
        0,
        &[],
        &jwin,
        &[],
        &[], // freeze: no seed
        SIZE_LOG2,
        svm_run::cap_thunk,
        &mut jhost as *mut Host as *mut c_void,
    ) {
        Ok(t) => t,
        Err(JitError::Unsupported(_)) => return,
        Err(JitError::Backend(msg)) if msg.contains("Allocation error") => return,
        Err(e) => panic!("JIT failed to compile a verified recycling module: {e:?}\n{inst:#?}"),
    };
    assert!(
        matches!(jout, JitOutcome::Returned(_)),
        "JIT armed freeze returns a placeholder\n{inst:#?}"
    );
    assert_eq!(
        (jfibers[0].slot, jfibers[0].generation),
        (0, generation),
        "JIT froze the recycled slot 0 at the bumped generation\n{inst:#?}"
    );

    // (2) The two backends armed-freeze the recycled fiber into a byte-identical durable reserve...
    let reserve = DURABLE_RESERVE as usize;
    assert_eq!(
        &isnap[..reserve],
        &jsnap[..reserve],
        "interp/JIT armed-freeze the recycled fiber into a byte-identical durable reserve\n{inst:#?}"
    );
    // ...and through the §12 codec to a byte-identical artifact (residue generation included).
    let art_i = freeze(&inst, &isnap, &ihost).expect("interp recycle freeze serializes");
    let mut jhost2 = Host::new();
    jhost2.set_frozen_fibers(
        jfibers
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
    let art_j = freeze(&inst, &jsnap, &jhost2).expect("JIT recycle freeze serializes");
    assert_eq!(
        art_i, art_j,
        "interp/JIT produce a byte-identical §12 artifact for the recycled fiber\n{inst:#?}"
    );

    // (3) Thaw portability: restore the interp artifact and resume on the JIT — the gen-bearing
    // handle resolves to the re-seeded fiber, which re-parks; forward execution reproduces `base`.
    let mut thost = Host::new();
    let mut thaw_win = restore(&art_i, &inst, &mut thost).expect("recycled artifact restores");
    begin_thaw(&mut thaw_win, 0);
    let seed: Vec<JitFrozen> = thost
        .frozen_fibers()
        .iter()
        .map(|f| JitFrozen {
            slot: f.slot,
            func: f.func,
            sp: f.sp,
            shadow_sp: f.shadow_sp,
            generation: f.generation,
        })
        .collect();
    assert_eq!(seed.len(), 1, "the artifact carried the recycled fiber");
    let mut jhost3 = Host::new();
    let jthaw = match compile_and_run_capture_reserved_with_host_durable(
        &inst,
        0,
        &[],
        &thaw_win,
        &[],
        &seed,
        SIZE_LOG2,
        svm_run::cap_thunk,
        &mut jhost3 as *mut Host as *mut c_void,
    ) {
        Ok((o, _, _)) => o,
        Err(JitError::Unsupported(_)) => return,
        Err(JitError::Backend(msg)) if msg.contains("Allocation error") => return,
        Err(e) => panic!("JIT failed to thaw the recycled artifact: {e:?}\n{inst:#?}"),
    };
    assert_returned_eq(&inst, &base, jthaw, "recycle-thaw");
}
