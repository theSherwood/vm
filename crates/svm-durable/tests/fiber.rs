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
    begin_thaw, init_durable_window, transform_module, transform_module_assume_confined,
    write_state, STATE_UNWINDING,
};
use svm_interp::{run_capture_reserved_with_host, Host, Trap, Value, SHADOW_BASE, SHADOW_STRIDE};
use svm_ir::{Inst, Memory, Module, Terminator};

const SIZE_LOG2: u8 = 17;
const WINDOW: usize = 1 << SIZE_LOG2;

// Root creates a fiber and resumes it twice; the fiber suspends once (yielding its arg) then
// returns arg+100. (The §12 raw-fiber shape from `jit_fibers.rs`.)
const SRC: &str = r#"
func () -> (i32, i64, i32, i64) {
block 0 () {
  v0 = ref.func 1
  v1 = i64.const 4096
  v2 = cont.new v0 v1
  v3 = i64.const 10
  v4, v5 = cont.resume v2 v3
  v6 = i64.const 7
  v7, v8 = cont.resume v2 v6
  return v4 v5 v7 v8
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  v2 = suspend v1
  v3 = i64.const 100
  v4 = i64.add v2 v3
  return v4
  }
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
block 0 () {
  v0 = ref.func 1
  v1 = i64.const 4096
  v2 = cont.new v0 v1
  v3 = i64.const 99
  v4, v5 = cont.resume v2 v3
  return v5
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  v2 = i64.const 42
  v3 = suspend v2
  v4 = i64.const 7
  v5 = i64.add v3 v4
  return v5
  }
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
    // §12.8 4A.5: the root's shadow-SP word is the first 8 bytes of its own region (at `root_base`),
    // not the legacy global `SHADOW_SP_OFF`.
    let active_sp = le_u64(&snap[root_base as usize..root_base as usize + 8]);
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
    begin_thaw(&mut thaw_win, 0);
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

/// Phase-3 slice 3.2 (active-resume-chain): a fiber that's **running** (mid-`cap.call`), not idle,
/// when freeze lands. Unlike a parked fiber (flattened by the driver), this one unwinds *in place*
/// during the root's run — its base-frame return happens under UNWINDING — and must be captured as
/// `Frozen` + residue (not `Done`), so a thaw re-seeds it and it rewinds at its leaf point and runs
/// **forward** (the active analogue of an idle fiber's re-park). The fiber does a clock `cap.call`
/// then returns; freezing mid-call must **reload** the saved clock value on thaw, not re-issue it.
///
/// The clock handle reaches the fiber through guest memory (`transform_module_assume_confined`),
/// since the fiber entry's `i64` args can't be narrowed to the `i32` cap handle without a conversion
/// op the transform doesn't model.
#[test]
fn active_resume_chain_fiber_freezes_and_thaws() {
    // Root stashes the clock handle at the first guest byte, then resumes F. F loads the handle,
    // calls the clock, and returns clock + 5 — no `suspend`, so at freeze F is *running*.
    const SRC: &str = r#"
func (i32) -> (i64) {
block 0 (v0: i32) {
  v1 = i64.const 65536
  i32.store v1 v0
  v2 = ref.func 1
  v3 = i64.const 4096
  v4 = cont.new v2 v3
  v5 = i64.const 0
  v6, v7 = cont.resume v4 v5
  return v7
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  v2 = i64.const 65536
  v3 = i32.load v2
  v4 = i32.const 0
  v5 = cap.call 2 0 (i32) -> (i64) v3 (v4)
  v6 = i64.const 5
  v7 = i64.add v5 v6
  return v7
  }
}
"#;
    let mut m = svm_text::parse_module(SRC).expect("parse");
    m.memory = Some(Memory {
        size_log2: SIZE_LOG2,
    });
    // The guest uses linear memory (the handle stash), so transform on the cooperating-toolchain
    // path; the stash is above the durable reserve.
    let inst = transform_module_assume_confined(&m).expect("transform");
    svm_verify::verify_module(&inst).expect("verify");

    // Uninterrupted baseline: clock starts at 42 → F returns 42 + 5 = 47.
    let want = {
        let mut h = Host::new();
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
    assert_eq!(want, vec![Value::I64(47)], "uninterrupted: clock 42 + 5");

    // Freeze: UNWINDING from the start — F runs the cap.call, then its poll unwinds it *in place*
    // (it never reaches its real return). Capture the window + the active-chain fiber's residue.
    let (frozen, snap, clock_after) = {
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
        (h.frozen_fibers().to_vec(), snap, h.clock_ns)
    };
    assert_eq!(
        frozen.len(),
        1,
        "the active (mid-cap.call) fiber was captured as Frozen, not dropped as Done"
    );
    assert!(
        clock_after > 42,
        "the freeze actually called the clock once"
    );

    // Thaw on a host whose clock has *advanced* (clock_after): the fiber must reload the saved 42,
    // not re-issue the clock (which would yield clock_after + 5). Re-seed the fiber and re-enter.
    let r_thaw = {
        let mut win = snap.clone();
        begin_thaw(&mut win, 0);
        let mut h = Host::new();
        h.set_durable(true);
        h.clock_ns = clock_after;
        let clk = h.grant_clock();
        h.set_frozen_fibers(frozen);
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
        "thawed active-chain fiber reloads the saved clock (47), not a re-issued one"
    );
}

// Recycling step 1 (DURABILITY.md §12.8): a fiber **guest handle** carries a generation in its high
// bits (`FIBER_GEN_SHIFT`), and `cont.resume` rejects a handle whose generation doesn't match the
// slot's current one — the ABA guard a recycled slot will rely on. All live generations are 0 until
// recycling is wired, so the genuine handle (== slot) resumes, while a forged generation-1 handle for
// the same slot faults. (Non-durable run — the check lives in the registry, independent of freeze.)
#[test]
fn forged_fiber_generation_is_rejected() {
    // Genuine handle (slot 0, generation 0 == 0): the fiber runs and returns 99.
    const OK: &str = r#"
func () -> (i64) {
block 0 () {
  v0 = ref.func 1
  v1 = i64.const 4096
  v2 = cont.new v0 v1
  v4 = i64.const 0
  v5, v6 = cont.resume v2 v4
  return v6
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  v2 = i64.const 99
  return v2
  }
}
"#;
    // Forged handle `(1 << 24) | 0`: same slot 0 (the mask clamps it), generation 1 ≠ 0 ⇒ FiberFault.
    const FORGED: &str = r#"
func () -> (i64) {
block 0 () {
  v0 = ref.func 1
  v1 = i64.const 4096
  v2 = cont.new v0 v1
  v3 = i64.const 16777216
  v4 = i64.const 0
  v5, v6 = cont.resume v3 v4
  return v6
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  v2 = i64.const 99
  return v2
  }
}
"#;
    let mut ok = svm_text::parse_module(OK).expect("parse OK");
    ok.memory = Some(Memory {
        size_log2: SIZE_LOG2,
    });
    assert_eq!(
        run_normal(&ok),
        Ok(vec![Value::I64(99)]),
        "genuine handle (generation 0) resumes"
    );

    let mut forged = svm_text::parse_module(FORGED).expect("parse FORGED");
    forged.memory = Some(Memory {
        size_log2: SIZE_LOG2,
    });
    assert_eq!(
        run_normal(&forged),
        Err(Trap::FiberFault),
        "a forged generation is rejected (the recycled-slot ABA guard)"
    );
}

// Recycling step 3 (DURABILITY.md §12.8): a finished fiber's slot is reclaimed for the next `cont.new`,
// so the registry is bounded by the *peak concurrent* fiber count, not the lifetime total. The reused
// slot keeps its bumped generation, so a stale guest handle to the former occupant fails to resume —
// the ABA guard (step 1) that makes reuse safe. (Non-durable run; recycling lives in the registry.)
#[test]
fn recycling_reuses_a_freed_slot_with_a_bumped_generation() {
    // Fiber A (handle slot 0, gen 0) runs to completion; the next cont.new reuses slot 0 at gen 1, so
    // its handle is `(1 << 24) | 0 == 16777216`. Returning the i64 handle makes the reuse observable.
    const REUSE: &str = r#"
func () -> (i64) {
block 0 () {
  v0 = ref.func 1
  v1 = i64.const 4096
  v2 = cont.new v0 v1
  v3 = i64.const 0
  v4, v5 = cont.resume v2 v3
  v6 = cont.new v0 v1
  return v6
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  v2 = i64.const 7
  return v2
  }
}
"#;
    let mut reuse = svm_text::parse_module(REUSE).expect("parse REUSE");
    reuse.memory = Some(Memory {
        size_log2: SIZE_LOG2,
    });
    assert_eq!(
        run_normal(&reuse),
        Ok(vec![Value::I64(16777216)]),
        "the freed slot 0 is reused at generation 1 ⇒ handle (1<<24)|0"
    );

    // After slot 0 is recycled (now holds fiber B at gen 1), resuming A's stale gen-0 handle (i64 0)
    // must fault — even though slot 0 is live — because the generation no longer matches.
    const STALE: &str = r#"
func () -> (i64) {
block 0 () {
  v0 = ref.func 1
  v1 = i64.const 4096
  v2 = cont.new v0 v1
  v3 = i64.const 0
  v4, v5 = cont.resume v2 v3
  v6 = cont.new v0 v1
  v9 = i64.const 0
  v7, v8 = cont.resume v9 v3
  return v8
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  v2 = i64.const 7
  return v2
  }
}
"#;
    let mut stale = svm_text::parse_module(STALE).expect("parse STALE");
    stale.memory = Some(Memory {
        size_log2: SIZE_LOG2,
    });
    assert_eq!(
        run_normal(&stale),
        Err(Trap::FiberFault),
        "a stale (gen-0) handle to a recycled slot faults (the ABA guard)"
    );
}

// ---------------------------------------------------------------------------
// DURABILITY.md §13.4 step 2 — **event-park freeze**: a fiber parked in a FUTEX WAIT (a
// `RegFiber::ParkedOn` event-park, which used to fail every freeze closed) freezes and thaws.
// The root resumes the fiber (it waits on cell 64 expecting 0 → parks), polls it once more
// (resume#2 → `FIBER_PARKED`, the cooperative poll), then stores 7, notifies the cell, and
// resumes a third time to collect the result. The freeze is armed at the SECOND fiber
// safepoint (the countdown ticks only at `cont.resume`/`suspend`), so it lands at resume#2's
// trailing poll with the fiber **unwoken**-parked: the freeze driver consumes the fiber's
// futex waiter entry and flattens it with an inert status (the `MemoryWait` point spills
// `out − nres`, so the placeholder is never captured). On thaw the root re-issues resume#2;
// the re-seeded fiber rewinds to its `MemoryWait` re-issue arm and the re-executed wait
// re-checks the restored cell — still 0 (the store is after the freeze point) — so it
// re-parks, re-deriving its own waiter state from the restored world (the O10 re-issue rule
// turned inward). The root's replayed store + notify then wake it exactly as in the
// uninterrupted run, so both runs agree on `(FIBER_RETURNED, 107, 7)`.
// ---------------------------------------------------------------------------

const SRC_FUTEX_PARKED_FIBER: &str = r#"
func () -> (i32, i64, i64) {
block 0 () {
  v0 = ref.func 1
  v1 = i64.const 4096
  v2 = cont.new v0 v1
  v3 = i64.const 0
  v4, v5 = cont.resume v2 v3
  v6, v7 = cont.resume v2 v3
  v8 = i64.const 65600
  v9 = i64.const 7
  i64.store v8 v9
  v10 = i32.const 1
  v11 = atomic.notify v8 v10
  v12, v13 = cont.resume v2 v3
  v14 = i64.load v8
  return v12 v13 v14
  }
}
func (i64, i64) -> (i64) {
block 0 (va: i64, vb: i64) {
  v2 = i64.const 65600
  v3 = i32.const 0
  v4 = i64.const -1
  v5 = i32.atomic.wait v2 v3 v4
  v6 = i64.load v2
  v7 = i64.const 100
  v8 = i64.add v6 v7
  return v8
  }
}
"#;

#[test]
fn a_futex_event_parked_fiber_freezes_and_the_thawed_wait_reissues() {
    use svm_durable::arm_freeze_after;

    let mut m = svm_text::parse_module(SRC_FUTEX_PARKED_FIBER).expect("parse");
    m.memory = Some(Memory {
        size_log2: SIZE_LOG2,
    });
    let inst = transform_module_assume_confined(&m).expect("transform");
    svm_verify::verify_module(&inst).expect("verify");

    // Baseline (no freeze): park → poll (FIBER_PARKED) → store+notify → collect. (The futex
    // cell sits above `DURABLE_RESERVE` — the low 64 KiB belongs to the durable header/shadow.)
    assert_eq!(
        run_normal(&inst),
        Ok(vec![Value::I32(1), Value::I64(107), Value::I64(7)]),
        "uninterrupted run: the notified wait completes with 7 + 100"
    );

    // Freeze at the second fiber safepoint (the countdown ticks only at `cont.resume`/`suspend`:
    // resume#1 = 1, resume#2 = 2): the freeze lands at resume#2's trailing poll with the fiber
    // **unwoken**-parked in its wait — the exact state that used to fail closed.
    let (frozen_fibers, root_sp, snap) = {
        let mut h = Host::new();
        h.set_durable(true);
        let mut win = init_durable_window(WINDOW);
        arm_freeze_after(&mut win, 2);
        let mut fuel = 1_000_000u64;
        let (r, snap) =
            run_capture_reserved_with_host(&inst, 0, &[], &mut fuel, &win, SIZE_LOG2, &mut h);
        assert!(r.is_ok(), "freeze returns a placeholder: {r:?}");
        (
            h.frozen_fibers().to_vec(),
            h.frozen_root_sp().expect("root extent recorded"),
            snap,
        )
    };
    assert_eq!(
        frozen_fibers.len(),
        1,
        "the event-parked fiber was flattened (the old fail-closed would have trapped)"
    );

    // Thaw: the re-issued resume#2 re-enters the re-seeded fiber under REWINDING; its
    // `MemoryWait` arm re-executes the wait, which re-checks the restored cell (still 0) and
    // re-parks — re-deriving its waiter entry in the fresh scheduler — so the root's replayed
    // store + notify wake it just as in the uninterrupted run.
    let r_thaw = {
        let mut win = snap.clone();
        begin_thaw(&mut win, 0);
        let mut h = Host::new();
        h.set_durable(true);
        h.set_frozen_fibers(frozen_fibers);
        h.set_frozen_root_sp(root_sp);
        let mut fuel = 1_000_000u64;
        let (r, _) =
            run_capture_reserved_with_host(&inst, 0, &[], &mut fuel, &win, SIZE_LOG2, &mut h);
        r
    };
    assert_eq!(
        r_thaw,
        Ok(vec![Value::I32(1), Value::I64(107), Value::I64(7)]),
        "(FIBER_RETURNED, restored cell + 100, cell) — the thawed park re-derived itself"
    );
}

// ---------------------------------------------------------------------------
// §13.4 step 2, the **woken** branch: a fiber whose event already fired (`ParkedOn { woken }`)
// but that no resume has claimed yet also freezes — its frames carry the delivered status, so
// the flatten needs no placeholder and consumes no waiter (the wake already did). Fiber A parks
// on the cell (resume#1 = safepoint 1), the root's store + notify wake it, and the freeze lands
// at the resume of a second fiber B (safepoint 2), which suspend-parks — so the driver flattens
// one woken event-park (A) and one ordinary suspend-park (B). Freeze-side only: re-entering a
// flattened *woken* park from post-rewind NORMAL execution is the step-3 thaw wiring
// (DURABILITY.md §13.4).
// ---------------------------------------------------------------------------

const SRC_WOKEN_PARKED_FIBER: &str = r#"
func () -> (i32, i64) {
block 0 () {
  v0 = ref.func 1
  v1 = i64.const 4096
  v2 = cont.new v0 v1
  v3 = i64.const 0
  v4, v5 = cont.resume v2 v3
  v6 = i64.const 65600
  v7 = i64.const 7
  i64.store v6 v7
  v8 = i32.const 1
  v9 = atomic.notify v6 v8
  v10 = ref.func 2
  v11 = i64.const 8192
  v12 = cont.new v10 v11
  v13, v14 = cont.resume v12 v3
  return v13 v14
  }
}
func (i64, i64) -> (i64) {
block 0 (va: i64, vb: i64) {
  v2 = i64.const 65600
  v3 = i32.const 0
  v4 = i64.const -1
  v5 = i32.atomic.wait v2 v3 v4
  v6 = i64.load v2
  v7 = i64.const 100
  v8 = i64.add v6 v7
  return v8
  }
}
func (i64, i64) -> (i64) {
block 0 (va: i64, vb: i64) {
  v2 = suspend vb
  return v2
  }
}
"#;

#[test]
fn a_woken_event_parked_fiber_freezes_without_a_placeholder() {
    use svm_durable::arm_freeze_after;

    let mut m = svm_text::parse_module(SRC_WOKEN_PARKED_FIBER).expect("parse");
    m.memory = Some(Memory {
        size_log2: SIZE_LOG2,
    });
    let inst = transform_module_assume_confined(&m).expect("transform");
    svm_verify::verify_module(&inst).expect("verify");

    let mut h = Host::new();
    h.set_durable(true);
    let mut win = init_durable_window(WINDOW);
    arm_freeze_after(&mut win, 2);
    let mut fuel = 1_000_000u64;
    let (r, _) = run_capture_reserved_with_host(&inst, 0, &[], &mut fuel, &win, SIZE_LOG2, &mut h);
    assert!(r.is_ok(), "freeze returns a placeholder: {r:?}");
    let mut slots: Vec<usize> = h.frozen_fibers().iter().map(|f| f.slot).collect();
    slots.sort_unstable();
    assert_eq!(
        slots,
        vec![0, 1],
        "both the woken event-park (A) and the suspend-park (B) were flattened"
    );
}

// ---------------------------------------------------------------------------
// §13.4 step 2, the fail-closed gate: an **unwoken capability park** cannot freeze — a
// `Leaf` (cap.call) point spills its results *including* the call's, so a placeholder would
// be reloaded on thaw as if it were the call's real result (reload-not-reissue). The driver
// probes the park's kind by which scheduler map holds the waiter; a cap park is not in
// `wait_waiters`, so the freeze refuses with `FiberFault`. The fiber parks in a blocking
// stdin read (the racing-fibers shape from `fiber_parks.rs`, handle passed through memory —
// the transform has no conversions).
// ---------------------------------------------------------------------------

const SRC_CAP_PARKED_FIBER: &str = r#"
func (i32) -> (i64) {
block 0 (v0: i32) {
  v1 = i64.const 65608
  i32.store v1 v0
  v2 = ref.func 1
  v3 = i64.const 4096
  v4 = cont.new v2 v3
  v5 = i64.const 0
  v6, v7 = cont.resume v4 v5
  v8, v9 = cont.resume v4 v5
  return v9
  }
}
func (i64, i64) -> (i64) {
block 0 (va: i64, vb: i64) {
  v2 = i64.const 65608
  v3 = i32.load v2
  v4 = i64.const 65600
  v5 = i64.const 4
  v6 = cap.call 0 0 (i64, i64) -> (i64) v3 (v4, v5)
  return v6
  }
}
"#;

#[test]
fn an_unwoken_cap_parked_fiber_fails_the_freeze_closed() {
    use svm_durable::arm_freeze_after;
    use svm_interp::StreamRole;

    let mut m = svm_text::parse_module(SRC_CAP_PARKED_FIBER).expect("parse");
    m.memory = Some(Memory {
        size_log2: SIZE_LOG2,
    });
    let inst = transform_module_assume_confined(&m).expect("transform");
    svm_verify::verify_module(&inst).expect("verify");

    let mut h = Host::new();
    h.set_durable(true);
    let handle = h.grant_stream(StreamRole::In);
    h.set_stdin_blocking(true);
    let mut win = init_durable_window(WINDOW);
    arm_freeze_after(&mut win, 2);
    let mut fuel = 1_000_000u64;
    let (r, _) = run_capture_reserved_with_host(
        &inst,
        0,
        &[Value::I32(handle)],
        &mut fuel,
        &win,
        SIZE_LOG2,
        &mut h,
    );
    assert_eq!(
        r,
        Err(Trap::FiberFault),
        "a cap-parked fiber refuses the freeze (its placeholder would masquerade as a result)"
    );
}

// ---------------------------------------------------------------------------
// §13.4 step 4 (per-fiber thaw re-arm) — a **woken** event-park thaws through a
// **post-rewind** claim. The freeze lands at the resume of a second fiber B (safepoint 2)
// with fiber A already woken-but-unclaimed; on thaw the root's rewind re-issues only the
// B resume — by the time it claims A (the next op), the root runs NORMAL again. Without the
// re-arm the fresh `Pending` claim would start A from its entry and orphan its spilled
// frame (observable: a transient `FIBER_PARKED` poll instead of the result). With it, A's
// non-empty shadow region forces its context `REWINDING` at the switch, so A rewinds to its
// `MemoryWait` arm, re-checks the restored cell (7 ≠ 0), and completes — both timelines
// agree on `(FIBER_RETURNED, 107, 7)`.
// ---------------------------------------------------------------------------

// Root: seed witness cell 65608 = 5; resume A (reads witness, parks on 65600); store 7 +
// notify (A woken); overwrite witness = 9 (AFTER A's read); resume B (the freeze point);
// then LOOP resuming A until it completes (the cooperative-poll contract — a thawed park may
// re-park transiently). A returns `witness_read * 1000 + cell`: 5007 proves the rewind
// reloaded A's SPILLED pre-park read (5); a fresh start would re-read 9 → 9007.
const SRC_WOKEN_PARK_COLLECTED_LATE: &str = r#"
func () -> (i32, i64, i64) {
block 0 () {
  vwc = i64.const 65608
  vfive = i64.const 5
  i64.store vwc vfive
  v0 = ref.func 1
  v1 = i64.const 4096
  v2 = cont.new v0 v1
  v3 = i64.const 0
  v4, v5 = cont.resume v2 v3
  v6 = i64.const 65600
  v7 = i64.const 7
  i64.store v6 v7
  v8 = i32.const 1
  v9 = atomic.notify v6 v8
  vnine = i64.const 9
  i64.store vwc vnine
  v10 = ref.func 2
  v11 = i64.const 8192
  v12 = cont.new v10 v11
  v13, v14 = cont.resume v12 v3
  br 1(v2, v3)
}
block 1 (vk: i64, vz: i64) {
  vs, vv = cont.resume vk vz
  vpk = i32.const 3
  vp = i32.eq vs vpk
  br_if vp 1(vk, vz) 2(vs, vv)
}
block 2 (vrs: i32, vrv: i64) {
  vc = i64.const 65600
  vload = i64.load vc
  return vrs vrv vload
  }
}
func (i64, i64) -> (i64) {
block 0 (va: i64, vb: i64) {
  vwc = i64.const 65608
  vw = i64.load vwc
  v2 = i64.const 65600
  v3 = i32.const 0
  v4 = i64.const -1
  v5 = i32.atomic.wait v2 v3 v4
  v6 = i64.load v2
  vt = i64.const 1000
  vm = i64.mul vw vt
  v8 = i64.add vm v6
  return v8
  }
}
func (i64, i64) -> (i64) {
block 0 (va: i64, vb: i64) {
  v2 = suspend vb
  return v2
  }
}
"#;

#[test]
fn a_woken_event_park_thaws_through_a_post_rewind_claim() {
    use svm_durable::arm_freeze_after;

    let mut m = svm_text::parse_module(SRC_WOKEN_PARK_COLLECTED_LATE).expect("parse");
    m.memory = Some(Memory {
        size_log2: SIZE_LOG2,
    });
    let inst = transform_module_assume_confined(&m).expect("transform");
    svm_verify::verify_module(&inst).expect("verify");

    // Baseline: the woken park is claimed directly (LiveWoken) — (RETURNED, 107, 7).
    assert_eq!(
        run_normal(&inst),
        Ok(vec![Value::I32(1), Value::I64(5007), Value::I64(7)]),
        "uninterrupted run: witness_read(5)*1000 + cell(7)"
    );

    // Freeze at the second fiber safepoint — the resume of B — with A woken-but-unclaimed.
    let (frozen_fibers, root_sp, snap) = {
        let mut h = Host::new();
        h.set_durable(true);
        let mut win = init_durable_window(WINDOW);
        arm_freeze_after(&mut win, 2);
        let mut fuel = 1_000_000u64;
        let (r, snap) =
            run_capture_reserved_with_host(&inst, 0, &[], &mut fuel, &win, SIZE_LOG2, &mut h);
        assert!(r.is_ok(), "freeze returns a placeholder: {r:?}");
        (
            h.frozen_fibers().to_vec(),
            h.frozen_root_sp().expect("root extent recorded"),
            snap,
        )
    };
    let mut slots: Vec<usize> = frozen_fibers.iter().map(|f| f.slot).collect();
    slots.sort_unstable();
    assert_eq!(
        slots,
        vec![0, 1],
        "A (woken park) and B (suspend park) both flattened"
    );

    // Thaw: the rewind re-issues only the B resume; the A claim happens in post-rewind
    // NORMAL execution — the re-arm makes it rewind instead of starting fresh.
    let r_thaw = {
        let mut win = snap.clone();
        begin_thaw(&mut win, 0);
        let mut h = Host::new();
        h.set_durable(true);
        h.set_frozen_fibers(frozen_fibers);
        h.set_frozen_root_sp(root_sp);
        let mut fuel = 1_000_000u64;
        let (r, _) =
            run_capture_reserved_with_host(&inst, 0, &[], &mut fuel, &win, SIZE_LOG2, &mut h);
        r
    };
    assert_eq!(
        r_thaw,
        Ok(vec![Value::I32(1), Value::I64(5007), Value::I64(7)]),
        "the post-rewind claim rewinds the woken park (spilled witness 5 reloaded, not re-read as 9)"
    );
}
