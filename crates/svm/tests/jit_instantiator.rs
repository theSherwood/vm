//! §14 `Instantiator` (iface 6) on the **JIT**, differentially vs. the interpreter. A guest holding
//! an `Instantiator` `instantiate`s a child confined to a power-of-two sub-window and `join`s it. The
//! JIT re-compiles the child as a top-level guest over its own fresh guarded window (nesting cost paid
//! at setup), seeded from / copied back to the parent's sub-region; the interpreter spawns a child
//! vCPU sharing the parent backing. Both must agree on the child's result *and* the final parent
//! window — the child confined to its slice on either backend (the §18 oracle, now covering nesting).
//!
//! The JIT nesting runtime needs a stack-switch-capable target (`fiber_supported`); elsewhere an
//! `instantiate` is a `CapFault`, so the differential is gated on it.

use svm_interp::{run_capture_reserved_with_host, Host, Trap, Value};
use svm_jit::{compile_and_run_capture_reserved_with_host, JitOutcome};
use svm_text::parse_module;
use svm_verify::verify_module;

type Both = (Result<Vec<Value>, Trap>, Vec<u8>, JitOutcome, Vec<u8>);

/// Run `src`'s entry (the granted `Instantiator` handle is `v0`) on **both** backends over a
/// fully-mapped `1<<win_log2`-byte window seeded with a non-zero pattern, with an `Instantiator`
/// granted over the whole window. Returns the interp result, the JIT outcome, and both final windows.
fn both(src: &str, win_log2: u8) -> Both {
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");
    let win = 1u64 << win_log2;
    let init: Vec<u8> = (0..win)
        .map(|i| (i as u8).wrapping_mul(31) ^ 0xa5)
        .collect();

    let mut hi = Host::new();
    let ih = hi.grant_instantiator(0, win);
    let mut hj = Host::new();
    let jh = hj.grant_instantiator(0, win);
    assert_eq!(ih, jh, "the Instantiator grant must encode identically");

    let mut fuel = 5_000_000u64;
    let (ir, imem) =
        run_capture_reserved_with_host(&m, 0, &[Value::I32(ih)], &mut fuel, &init, 0, &mut hi);
    let (jo, jmem) = compile_and_run_capture_reserved_with_host(
        &m,
        0,
        &[jh as i64],
        &init,
        0,
        svm_run::cap_thunk,
        &mut hj as *mut Host as *mut core::ffi::c_void,
    )
    .expect("jit");
    (ir, imem, jo, jmem)
}

/// The shared nesting program: func 0 (parent) instantiates func 1 (child) in a 4 KiB window at
/// `off`, joins it, returns the child's result. The child (ignoring its starter-cap arg) stores a
/// marker at its offset 0 and at an in-window offset (1695) inside its 4 KiB window, then
/// returns 42 — so the result and the two in-slice writes are observable on both backends.
fn nest_src(off: u64) -> String {
    format!(
        "memory 17\n\
         func (i32) -> (i64) {{\n\
         block 0 (v0: i32) {{\n\
         \x20 v1 = i64.const 1\n\
         \x20 v2 = i64.const {off}\n\
         \x20 v3 = i64.const 12\n\
         \x20 v4 = i64.const 0\n\
         \x20 v5 = cap.call 6 0 (i64, i64, i64, i64) -> (i32) v0 (v1, v2, v3, v4)\n\
         \x20 v6 = cap.call 6 1 (i32) -> (i64) v0 (v5)\n\
         \x20 return v6\n\
           }}\n\
         }}\n\
         func (i64) -> (i64) {{\n\
         block 0 (v0: i64) {{\n\
         \x20 v1 = i64.const 0\n\
         \x20 v2 = i32.const 171\n\
         \x20 i32.store8 v1 v2\n\
         \x20 v3 = i64.const 1695\n\
         \x20 v4 = i32.const 200\n\
         \x20 i32.store8 v3 v4\n\
         \x20 v5 = i64.const 42\n\
         \x20 return v5\n\
           }}\n\
         }}\n"
    )
}

#[test]
fn jit_instantiator_matches_interp() {
    if !svm_jit::fiber_supported() {
        return; // no JIT nesting runtime on this target; an instantiate is a CapFault there
    }
    const OFF: u64 = 64 << 10; // a 64 KiB-aligned slot in the 128 KiB window
    const SIZE: u64 = 4096;
    let (ir, imem, jo, jmem) = both(&nest_src(OFF), 17);

    // Both backends ran the child and returned its result through join.
    let ival = ir.expect("interp ran ok").pop().expect("one result");
    assert_eq!(ival, Value::I64(42), "interp child result");
    assert!(
        matches!(jo, JitOutcome::Returned(ref s) if s == &[42]),
        "jit: {jo:?}"
    );

    // The escape-oracle: the two backends leave a byte-identical parent window — the child confined
    // to its slice on both, its writes materialized into the parent's sub-region.
    assert_eq!(
        imem, jmem,
        "interp/JIT parent windows diverge after nesting"
    );

    // Pin the effect: the child's stores landed in its slice (171 @ OFF, 200 @ OFF+1695), and every
    // byte outside the child's 4 KiB window is exactly as the parent seeded it.
    let init: Vec<u8> = (0..(128u64 << 10))
        .map(|i| (i as u8).wrapping_mul(31) ^ 0xa5)
        .collect();
    assert_eq!(jmem[OFF as usize], 171, "child store @0 missing on the JIT");
    assert_eq!(
        jmem[(OFF + 1695) as usize],
        200,
        "child in-window store missing on the JIT"
    );
    for i in 0..(128u64 << 10) {
        if !(OFF..OFF + SIZE).contains(&i) {
            assert_eq!(
                jmem[i as usize], init[i as usize],
                "JIT child escaped to parent byte {i}"
            );
        }
    }
}

/// An out-of-range carve (a child window that doesn't fit the holder) is `-EINVAL` on the JIT too —
/// no child runs, the handle is a negative value the guest can branch on.
#[test]
fn jit_instantiator_rejects_out_of_range_carve() {
    if !svm_jit::fiber_supported() {
        return;
    }
    let src = "memory 17\n\
         func (i32) -> (i64) {\n\
         block 0 (v0: i32) {\n\
         \x20 v1 = i64.const 1\n\
         \x20 v2 = i64.const 131072\n\
         \x20 v3 = i64.const 12\n\
         \x20 v4 = i64.const 0\n\
         \x20 v5 = cap.call 6 0 (i64, i64, i64, i64) -> (i32) v0 (v1, v2, v3, v4)\n\
         \x20 v6 = i64.extend_i32_s v5\n\
         \x20 return v6\n\
           }\n\
         }\n\
         func (i64) -> (i64) {\n\
         block 0 (v0: i64) {\n\
         \x20 v1 = i64.const 0\n\
         \x20 return v1\n\
           }\n\
         }\n";
    let (ir, _imem, jo, _jmem) = both(src, 17);
    let ival = ir.expect("interp ran ok").pop().expect("one result");
    assert_eq!(ival, Value::I64(-22), "interp out-of-range carve");
    assert!(
        matches!(jo, JitOutcome::Returned(ref s) if s == &[-22]),
        "jit: {jo:?}"
    );
}

/// A child trap propagates to the parent on `join` — identically on both backends. Two cases: an
/// explicit `unreachable`, and a **width-overrun** at the top of the child's window (an 8-byte store
/// straddling its 64 KiB top) — the decisive escape-TCB property: the child cannot reach past its
/// slice, the overrun is caught by *its own* guard page (JIT) / byte-exact bound (interp), and
/// surfaces as the parent's trap. (64 KiB is page-aligned on 4 KiB and 16 KiB hosts, so the JIT
/// guard sits exactly at the window top either way.)
#[test]
fn jit_instantiator_child_trap_propagates() {
    if !svm_jit::fiber_supported() {
        return;
    }
    // child body → the §14 child (entry func 1), confined to a 64 KiB window at 64 KiB of a 256 KiB
    // parent. `body` is the child's block-0 instructions; both children take/ignore one i64 arg.
    let make = |body: &str| {
        format!(
            "memory 18\n\
             func (i32) -> (i64) {{\n\
             block 0 (v0: i32) {{\n\
             \x20 v1 = i64.const 1\n\
             \x20 v2 = i64.const 65536\n\
             \x20 v3 = i64.const 16\n\
             \x20 v4 = i64.const 0\n\
             \x20 v5 = cap.call 6 0 (i64, i64, i64, i64) -> (i32) v0 (v1, v2, v3, v4)\n\
             \x20 v6 = cap.call 6 1 (i32) -> (i64) v0 (v5)\n\
             \x20 return v6\n\
               }}\n\
             }}\n\
             func (i64) -> (i64) {{\n\
             block 0 (v0: i64) {{\n\
             {body}\n\
               }}\n\
             }}\n"
        )
    };
    // (a) explicit unreachable.
    let (ir, _i, jo, _j) = both(&make("  unreachable"), 18);
    assert!(
        ir.is_err(),
        "interp: child unreachable should propagate, got {ir:?}"
    );
    assert!(matches!(jo, JitOutcome::Trapped(_)), "jit: {jo:?}");

    // (b) width-overrun at the child's window top: store i64 at child offset 65535 → [65535, 65543)
    // crosses 65536. The child is confined to [0, 64 KiB); the overrun cannot reach the parent.
    let overrun = "  v1 = i64.const 65535\n\
                   \x20 v2 = i64.const 7\n\
                   \x20 i64.store v1 v2\n\
                   \x20 v3 = i64.const 0\n\
                   \x20 return v3";
    let (ir, _i, jo, _j) = both(&make(overrun), 18);
    assert!(
        ir.is_err(),
        "interp: child width-overrun should trap, got {ir:?}"
    );
    assert!(matches!(jo, JitOutcome::Trapped(_)), "jit: {jo:?}");
}

/// §3.6 parity — `child_offer` (op 14) on the JIT is a **probeable refusal, never a trap**:
/// the op is eval-loop-serviced (it mints from the tree-walk scheduler's live-child registry,
/// which the JIT runtime does not have), so the JIT lowers it to a `-EINVAL` result — the
/// same refusal the interpreter gives a bad child handle. Differential: both backends agree.
#[test]
fn jit_child_offer_refuses_probeably_and_matches_interp() {
    if !svm_jit::fiber_supported() {
        return;
    }
    let src = "memory 16\n\
         func (i32) -> (i64) {\n\
         block 0 (v0: i32) {\n\
         \x20 vbad = i32.const 99\n\
         \x20 vz = i64.const 0\n\
         \x20 vc = cap.call 6 14 (i32, i64) -> (i32) v0 (vbad, vz)\n\
         \x20 vr = i64.extend_i32_s vc\n\
         \x20 return vr\n\
           }\n\
         }\n";
    let (ir, _imem, jo, _jmem) = both(src, 16);
    assert_eq!(
        ir,
        Ok(vec![Value::I64(-22)]),
        "interp: a bad child handle refuses -EINVAL"
    );
    assert_eq!(
        jo,
        JitOutcome::Returned(vec![-22]),
        "jit: op 14 refuses probeably with the same value — no trap, no wrong answer"
    );
}
