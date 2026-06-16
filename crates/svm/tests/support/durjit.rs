//! Cross-backend arm of the freeze/thaw equivalence property (DURABILITY.md §7/§12.6).
//!
//! The instrumented IR the durable transform emits is ordinary control flow + window
//! loads/stores — nothing VM-specific — so the *same* artifact must drive identically on
//! the reference interpreter and the Cranelift JIT. This module reuses the in-scope
//! generator (`durgen`, shared with the interp-only `durable_fuzz` smoke) and asserts the
//! two backends agree on three claims:
//!
//!   1. **NORMAL** — instrumented run returns the same value on both backends;
//!   2. **freeze** — both leave a *byte-identical* shadow region (the persisted artifact
//!      is backend-independent), still flagged `UNWINDING`;
//!   3. **thaw portability** — the artifact frozen by the **interpreter**, resumed on the
//!      **JIT** under a *different* host clock, still reproduces the uninterrupted result
//!      (so the JIT reloads the saved `cap.call` value, it does not re-issue the call) and
//!      flips the state word back to `NORMAL`.
//!
//! Shared by the stable `cargo test` driver (`crates/svm/tests/durable_jit.rs`) and the
//! libFuzzer target (`fuzz/fuzz_targets/durable_jit.rs`), mirroring `irgen`/`durgen`.

#![allow(dead_code)] // each includer uses a subset

#[path = "../../../svm-durable/tests/support/durgen.rs"]
pub mod durgen;

use core::ffi::c_void;
use durgen::{gen_module, Gen, SIZE_LOG2, WINDOW};
use svm_durable::{
    init_durable_window, read_state, transform_module, write_state, SHADOW_SP_OFF, STATE_NORMAL,
    STATE_REWINDING, STATE_UNWINDING,
};
use svm_interp::{run_capture_reserved_with_host, Host, Trap, Value};
use svm_ir::{Module, ValType};
use svm_jit::{compile_and_run_capture_reserved_with_host, JitError, JitOutcome};

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
) -> (Result<Vec<Value>, Trap>, Vec<u8>, i64) {
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
    (r, win, h.clock_ns)
}

/// Run the instrumented entry on the JIT against `window`, with the clock seeded to
/// `clock_v`. `None` if the JIT declines to compile (it only ever sees lowered ops, so this
/// is a safety valve, not an expected path). On `Some`, also returns the final `clock_ns`.
fn jit_run(inst: &Module, clock_v: i64, window: &[u8]) -> Option<(JitOutcome, Vec<u8>, i64)> {
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
        Ok((o, win)) => Some((o, win, h.clock_ns)),
        Err(JitError::Unsupported(_)) => None,
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

/// Check the cross-backend property on one generated in-scope module.
pub fn fuzz_one_xbackend(g: &mut Gen) {
    let m = gen_module(g);
    let inst = transform_module(&m).expect("an in-scope module must transform");
    svm_verify::verify_module(&inst).expect("instrumented IR must verify");

    let clock_v = g.u64v() as i64;

    // Interpreter reference: the uninterrupted result and the frozen artifact. `clock_after`
    // is how far the clock advanced during freeze — the thaw host continues from there
    // (D-scope: the host clock is not in the artifact), so any suspend points re-performed
    // after the resume match the baseline while the frozen point's result is reloaded.
    let (base_i, _, _) = interp_run(&inst, clock_v, &window_with(STATE_NORMAL));
    let base = base_i.expect("generated programs are trap-free");
    let (fr_i, snap_i, clock_after) = interp_run(&inst, clock_v, &window_with(STATE_UNWINDING));
    assert!(fr_i.is_ok(), "interp freeze returns a placeholder");
    assert_eq!(read_state(&snap_i), STATE_UNWINDING);

    // (1) NORMAL: the JIT agrees with the interpreter's value.
    let Some((j_base, _, _)) = jit_run(&inst, clock_v, &window_with(STATE_NORMAL)) else {
        return;
    };
    assert_returned_eq(&inst, &base, j_base, "NORMAL");

    // (2) freeze: the persisted artifact is byte-identical across backends.
    let Some((j_fr, snap_j, _)) = jit_run(&inst, clock_v, &window_with(STATE_UNWINDING)) else {
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

    // (3) thaw portability: resume the interpreter-frozen artifact on the JIT, on a host
    // whose clock *continues* from the freeze. Must reproduce `base` (the frozen point's
    // result reloaded, not re-issued — a re-issue would consume the next tick) and end
    // NORMAL. This crosses backends: the artifact was frozen by the interpreter.
    let mut thaw_win = snap_i.clone();
    write_state(&mut thaw_win, STATE_REWINDING);
    let Some((j_thaw, final_j, _)) = jit_run(&inst, clock_after, &thaw_win) else {
        return;
    };
    assert_returned_eq(&inst, &base, j_thaw, "thaw");
    assert_eq!(
        read_state(&final_j),
        STATE_NORMAL,
        "JIT thaw must flip the state word back to NORMAL\n{inst:#?}"
    );
}
