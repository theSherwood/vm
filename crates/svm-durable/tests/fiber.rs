//! Phase-3.1 (option A): the durable transform recognizes the §12 fiber control ops
//! (`cont.new`/`cont.resume`/`suspend`) as may-suspend points and instruments them. This pins
//! the **NORMAL-inertness** invariant for a fiber'd module — instrumented runs identically to
//! un-instrumented — and that the instrumented IR verifies.
//!
//! Slices 3.1.2–3 wire both **fiber thaw arms**: a `cont.resume` rewinds by re-issuing the resume
//! (reloading its spilled handle + arg), and a `suspend` rewinds by flipping to `NORMAL` and
//! re-executing `suspend` to re-park the fiber. No fiber arm fails closed any more. A full fiber
//! *thaw* still isn't exercisable here — a parked fiber's continuation isn't captured until the
//! freeze driver flattens it into its shadow stack and the snapshot records its metadata (slices
//! 3.1.4–5) — so these tests pin verification + NORMAL-inertness + the arm wiring, structurally.

use svm_durable::{
    init_durable_window, transform_module, write_state, STATE_REWINDING, STATE_UNWINDING,
};
use svm_interp::{
    run_capture_reserved_with_host, Host, Trap, Value, SHADOW_BASE, SHADOW_SP_OFF, SHADOW_STRIDE,
};
use svm_ir::{Inst, Memory, Module, Terminator};

const SIZE_LOG2: u8 = 17;
const WINDOW: usize = 1 << SIZE_LOG2;

// Root creates a fiber and resumes it twice; the fiber suspends once (yielding its arg) then
// returns arg+100. (The §12 raw-fiber shape from `jit_fibers.rs`.)
const SRC: &str = r#"
func () -> (i32, i64, i32, i64) {
block0():
  v0 = ref.func 1
  v1 = i64.const 4096
  v2 = cont.new v0 v1
  v3 = i64.const 10
  v4, v5 = cont.resume v2 v3
  v6 = i64.const 7
  v7, v8 = cont.resume v2 v6
  return v4 v5 v7 v8
}
func (i64, i64) -> (i64) {
block0(v0: i64, v1: i64):
  v2 = suspend v1
  v3 = i64.const 100
  v4 = i64.add v2 v3
  return v4
}
"#;

fn run_normal(m: &Module) -> Result<Vec<Value>, Trap> {
    let mut host = Host::new();
    let mut fuel = 1_000_000u64;
    let (r, _) = run_capture_reserved_with_host(
        m,
        0,
        &[],
        &mut fuel,
        &init_durable_window(WINDOW),
        SIZE_LOG2,
        &mut host,
    );
    r
}

#[test]
fn fiber_module_is_inert_under_instrumentation_in_normal() {
    let mut m = svm_text::parse_module(SRC).expect("parse");
    m.memory = Some(Memory {
        size_log2: SIZE_LOG2,
    });

    let base = run_normal(&m).expect("baseline fiber run");
    // Sanity: resume(10) → (SUSPENDED=0, yielded 10); resume(7) → (RETURNED=1, 7+100).
    assert_eq!(
        base,
        vec![
            Value::I32(0),
            Value::I64(10),
            Value::I32(1),
            Value::I64(107)
        ],
    );

    let inst = transform_module(&m).expect("a fiber'd module transforms");
    svm_verify::verify_module(&inst).expect("instrumented fiber'd IR verifies");
    let got = run_normal(&inst).expect("instrumented fiber'd module runs in NORMAL");

    assert_eq!(
        got, base,
        "instrumentation is inert in NORMAL for a fiber'd module"
    );
}

/// Count `cont.resume` / `suspend` ops and `Unreachable`-terminated blocks across every function.
fn op_and_trap_counts(m: &Module) -> (usize, usize, usize) {
    let mut resumes = 0;
    let mut suspends = 0;
    let mut unreachable = 0;
    for f in &m.funcs {
        for b in &f.blocks {
            for i in &b.insts {
                match i {
                    Inst::ContResume { .. } => resumes += 1,
                    Inst::Suspend { .. } => suspends += 1,
                    _ => {}
                }
            }
            if matches!(b.term, Terminator::Unreachable) {
                unreachable += 1;
            }
        }
    }
    (resumes, suspends, unreachable)
}

/// Slices 3.1.2–3: both fiber thaw arms are wired. A `cont.resume` rewind arm re-issues the
/// resume (3.1.2); a `suspend` rewind arm re-executes `suspend` to re-park the fiber (3.1.3).
/// Structurally each resume/suspend point keeps its forward op and gains a re-issue in its rewind
/// arm (so both counts double), and the **only** `Unreachable` block left is the shared forged-id
/// TRAP — no fiber arm fails closed any more.
#[test]
fn both_fiber_thaw_arms_reissue_their_op() {
    let mut m = svm_text::parse_module(SRC).expect("parse");
    m.memory = Some(Memory {
        size_log2: SIZE_LOG2,
    });
    let (res0, susp0, _) = op_and_trap_counts(&m);
    assert_eq!((res0, susp0), (2, 1), "source: 2 cont.resume, 1 suspend");

    let inst = transform_module(&m).expect("transform");
    svm_verify::verify_module(&inst).expect("instrumented IR verifies");
    let (res1, susp1, traps) = op_and_trap_counts(&inst);

    // Each resume/suspend point keeps its forward op and gains a re-issuing rewind arm → both
    // counts double.
    assert_eq!(
        res1,
        2 * res0,
        "each cont.resume gains a re-issuing rewind arm"
    );
    assert_eq!(
        susp1,
        2 * susp0,
        "each suspend gains a re-parking rewind arm"
    );
    // No fiber arm fails closed now: the only Unreachable blocks are the per-function forged-id
    // TRAPs — one in the root and one in the fiber (both functions are instrumented).
    assert_eq!(
        traps, 2,
        "only the per-function forged-id TRAP blocks are Unreachable"
    );
}

fn le_u64(b: &[u8]) -> u64 {
    u64::from_le_bytes(b.try_into().unwrap())
}
fn le_u32(b: &[u8]) -> u32 {
    u32::from_le_bytes(b.try_into().unwrap())
}

/// Slice 3.1.4 — the **freeze driver** flattens an idle parked fiber into its *own* shadow region.
/// Root resumes fiber F, which suspends and parks; freezing the durable domain (UNWINDING) drives
/// the root's unwind *and* then drives F so its post-suspend poll unwinds it with zero forward
/// progress. We check both contexts left a frame in their (distinct) regions, and that F unwound
/// **at its suspend point** (resume id 1) — i.e. it did not run any of its post-suspend code.
#[test]
fn freeze_driver_flattens_a_parked_fiber_into_its_region() {
    // Root resumes F once; F suspends 42 then (if ever run forward) would compute 42+7 and return.
    const SRC2: &str = r#"
func () -> (i64) {
block0():
  v0 = ref.func 1
  v1 = i64.const 4096
  v2 = cont.new v0 v1
  v3 = i64.const 99
  v4, v5 = cont.resume v2 v3
  return v5
}
func (i64, i64) -> (i64) {
block0(v0: i64, v1: i64):
  v2 = i64.const 42
  v3 = suspend v2
  v4 = i64.const 7
  v5 = i64.add v3 v4
  return v5
}
"#;
    let mut m = svm_text::parse_module(SRC2).expect("parse");
    m.memory = Some(Memory {
        size_log2: SIZE_LOG2,
    });
    let inst = transform_module(&m).expect("transform");
    svm_verify::verify_module(&inst).expect("instrumented IR verifies");

    // Freeze: seed the window in UNWINDING so the run unwinds instead of completing.
    let mut win = init_durable_window(WINDOW);
    write_state(&mut win, STATE_UNWINDING);
    let mut host = Host::new();
    host.set_durable(true);
    let mut fuel = 1_000_000u64;
    let (res, snap) =
        run_capture_reserved_with_host(&inst, 0, &[], &mut fuel, &win, SIZE_LOG2, &mut host);
    assert!(
        res.is_ok(),
        "freeze returns a placeholder, not a trap: {res:?}"
    );

    // Region bases: root is context 0, the single fiber (slot 0) is context 1.
    let root_base = SHADOW_BASE;
    let fiber_base = SHADOW_BASE + SHADOW_STRIDE;

    // The root unwound its `cont.resume` frame into context 0's region.
    let root_region = &snap[root_base as usize..(root_base + SHADOW_STRIDE) as usize];
    assert!(
        root_region.iter().any(|&b| b != 0),
        "the root unwound a frame into context 0's region"
    );

    // The freeze run hands back the flattened fiber's metadata; its continuation lives in its own
    // region, with the active shadow-SP left at the root's (thaw-ready — slice 3.1.5).
    let frozen = host.frozen_fibers();
    assert_eq!(frozen.len(), 1, "one fiber was flattened");
    assert_eq!(frozen[0].slot, 0, "the single fiber holds handle 0");
    let fiber_sp = frozen[0].shadow_sp;
    assert!(
        fiber_sp > fiber_base && fiber_sp <= fiber_base + SHADOW_STRIDE,
        "the fiber flattened a frame into its own region [{fiber_base}, +stride): sp={fiber_sp}"
    );
    let active_sp = le_u64(&snap[SHADOW_SP_OFF as usize..SHADOW_SP_OFF as usize + 8]);
    assert!(
        active_sp >= root_base && active_sp < fiber_base,
        "the active shadow-SP is left at the root's region for thaw: {active_sp}"
    );
    // The fiber unwound *at its suspend point* (resume id 1 = the first/only point), proving the
    // driver made zero forward progress past the `suspend` (it never reached `42 + 7`).
    let rid = le_u32(&snap[(fiber_sp - 4) as usize..fiber_sp as usize]);
    assert_eq!(
        rid, 1,
        "the fiber unwound at its suspend point (zero forward progress)"
    );
}

/// Slice 3.1.5 — the **end-to-end single-fiber round-trip**: `freeze → (window + fiber metadata) →
/// thaw ≡ uninterrupted`, interpreter-only. Root resumes a fiber, which suspends; freezing captures
/// the root's *and* the fiber's continuations (the latter via the driver + the exported
/// [`svm_interp::FrozenFiber`] residue). Thawing re-seeds the fiber, re-enters the root under
/// REWINDING — the resumer re-issues `cont.resume`, the fiber rewinds and re-parks — then runs
/// forward to completion, matching the uninterrupted result. (Metadata is passed in-memory; the
/// byte-level snapshot Section-2 codec is the follow-up `svm-snapshot` slice.)
#[test]
fn single_fiber_freeze_thaw_round_trips() {
    const SRC3: &str = r#"
func () -> (i64) {
block0():
  v0 = ref.func 1
  v1 = i64.const 4096
  v2 = cont.new v0 v1
  v3 = i64.const 0
  v4, v5 = cont.resume v2 v3
  v6 = i64.const 7
  v7, v8 = cont.resume v2 v6
  return v8
}
func (i64, i64) -> (i64) {
block0(v0: i64, v1: i64):
  v2 = i64.const 42
  v3 = suspend v2
  v4 = i64.const 100
  v5 = i64.add v3 v4
  return v5
}
"#;
    let mut m = svm_text::parse_module(SRC3).expect("parse");
    m.memory = Some(Memory {
        size_log2: SIZE_LOG2,
    });
    let inst = transform_module(&m).expect("transform");
    svm_verify::verify_module(&inst).expect("instrumented IR verifies");

    // Uninterrupted baseline: resume #1 suspends 42; resume #2 (arg 7) returns 7 + 100 = 107.
    let want = run_normal(&inst).expect("uninterrupted run");
    assert_eq!(want, vec![Value::I64(107)], "uninterrupted result");

    // Freeze from the start: the run unwinds at resume #1's poll (fiber parked after suspend).
    let mut win = init_durable_window(WINDOW);
    write_state(&mut win, STATE_UNWINDING);
    let mut fhost = Host::new();
    fhost.set_durable(true);
    let mut fuel = 1_000_000u64;
    let (rf, frozen_win) =
        run_capture_reserved_with_host(&inst, 0, &[], &mut fuel, &win, SIZE_LOG2, &mut fhost);
    assert!(rf.is_ok(), "freeze returns a placeholder: {rf:?}");
    let frozen = fhost.frozen_fibers().to_vec();
    assert_eq!(frozen.len(), 1, "the parked fiber was flattened");

    // Thaw: restore the captured window (REWINDING), re-seed the fiber, re-enter the root.
    let mut thaw_win = frozen_win;
    write_state(&mut thaw_win, STATE_REWINDING);
    let mut thost = Host::new();
    thost.set_durable(true);
    thost.set_frozen_fibers(frozen);
    let mut tfuel = 1_000_000u64;
    let (rt, _) =
        run_capture_reserved_with_host(&inst, 0, &[], &mut tfuel, &thaw_win, SIZE_LOG2, &mut thost);
    assert_eq!(
        rt.expect("thaw runs to completion"),
        want,
        "freeze → thaw reproduces the uninterrupted result"
    );
}
