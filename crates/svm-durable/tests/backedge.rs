//! Phase-4 Slice A — **back-edge polls**: a freeze lands inside a poll-free compute loop.
//!
//! Before this slice a freeze could only land at a *may-suspend* safepoint (`cap.call` /
//! `cont.resume` / `suspend`), so a vCPU in a tight compute loop with no such op never reached one
//! and the freeze hung until the loop exited (DURABILITY.md, the R6 latency caveat). The transform
//! now prepends a **state-word poll to every loop header** (a block reached by a back-edge), which
//! dominates the loop body, so the freeze is observed within one iteration.
//!
//! The guest below reads the clock **once, before the loop**, then runs a poll-free accumulator
//! loop. We arm the deterministic *back-edge* countdown (`arm_freeze_after_backedges`) so the run
//! makes progress past the `cap.call` and into the loop, then freezes mid-iteration at a header
//! poll. The property is the usual one (§12.6):
//!
//! > freeze → serialize → restore → thaw → run-to-end  ≡  uninterrupted run
//!
//! The clock seed (42) is baked into the loop-carried accumulator that the header poll spills, so a
//! thaw on a **fresh** host (clock 0) must still reproduce the oracle — proving the loop state was
//! reloaded, not recomputed, and that the pre-loop `cap.call` is not re-issued.

use svm_durable::{
    arm_freeze_after_backedges, init_durable_window, read_state, transform_module, begin_thaw, STATE_UNWINDING,
};
use svm_interp::{run_capture_reserved_with_host, Host, Trap, Value};
use svm_ir::Memory;

const SIZE_LOG2: u8 = 18;
const WINDOW: usize = 1 << SIZE_LOG2;

// v0 = clock handle. Read the clock once (v2), seed the accumulator with it, then loop five times
// adding 1 each iteration — a header (block1) reached by a back-edge, with **no** may-suspend op in
// the body. Oracle: clock(42) + 5 = 47.
const SRC: &str = r#"
func (i32) -> (i64) {
block0(v0: i32):
  v1 = i32.const 0
  v2 = cap.call 2 0 (i32) -> (i64) v0 (v1)
  v3 = i64.const 0
  br block1(v3, v2)
block1(v4: i64, v5: i64):
  v6 = i64.const 1
  v7 = i64.add v4 v6
  v8 = i64.add v5 v6
  v9 = i64.const 5
  v10 = i64.lt_s v7 v9
  br_if v10 block1(v7, v8) block2(v8)
block2(v11: i64):
  return v11
}
"#;

fn module() -> svm_ir::Module {
    let mut m = svm_text::parse_module(SRC).expect("parse");
    m.memory = Some(Memory {
        size_log2: SIZE_LOG2,
    });
    let inst = transform_module(&m).expect("transform");
    svm_verify::verify_module(&inst).expect("verify");
    inst
}

// Run with the given seeded window; returns (result, final window image).
fn run(inst: &svm_ir::Module, clock_ns: i64, win: &[u8]) -> (Result<Vec<Value>, Trap>, Vec<u8>) {
    let mut host = Host::new();
    host.set_durable(true);
    host.clock_ns = clock_ns;
    let clk = host.grant_clock();
    let mut fuel = 100_000u64;
    run_capture_reserved_with_host(
        inst,
        0,
        &[Value::I32(clk)],
        &mut fuel,
        win,
        SIZE_LOG2,
        &mut host,
    )
}

#[test]
fn freeze_inside_a_poll_free_loop_round_trips() {
    let inst = module();

    // ---- Baseline: the uninterrupted run (the oracle). Clock seeded at 42. ----
    let (baseline, _) = run(&inst, 42, &init_durable_window(WINDOW));
    assert_eq!(baseline, Ok(vec![Value::I64(47)]), "uninterrupted: 42 + 5");

    // ---- Freeze: arm the back-edge countdown so the freeze lands *mid-loop* (not at the ----
    // pre-loop cap.call). The 3rd branch terminator promotes the word to UNWINDING; the next
    // header poll then unwinds, spilling the loop-carried (counter, accumulator).
    let mut win = init_durable_window(WINDOW);
    arm_freeze_after_backedges(&mut win, 3);
    let (frozen, snapshot) = run(&inst, 42, &win);
    assert_eq!(
        frozen,
        Ok(vec![Value::I64(0)]),
        "freeze returns a placeholder (the loop unwound, did not finish)"
    );
    assert_eq!(
        read_state(&snapshot),
        STATE_UNWINDING,
        "froze inside the loop — the back-edge poll observed UNWINDING"
    );

    // ---- Restore + thaw on a FRESH host (clock now 0). If the loop state were recomputed ----
    // instead of reloaded — or the pre-loop cap.call re-issued — the seed would be 0 and the
    // result would differ. It must reproduce the oracle.
    let mut win = snapshot.clone();
    begin_thaw(&mut win, 0);
    let (thawed, _) = run(&inst, 0, &win);
    assert_eq!(
        thawed, baseline,
        "thawed run reproduces the uninterrupted run despite a fresh (clock 0) host"
    );
}

#[test]
fn unarmed_run_of_the_loop_is_inert() {
    // With no arming the header poll is a not-taken branch: the instrumented loop behaves exactly
    // like the original and runs to completion.
    let inst = module();
    let (r, snap) = run(&inst, 7, &init_durable_window(WINDOW));
    assert_eq!(r, Ok(vec![Value::I64(12)]), "7 + 5, instrumentation inert");
    assert_ne!(
        read_state(&snap),
        STATE_UNWINDING,
        "never entered UNWINDING"
    );
}

#[test]
fn freeze_at_a_later_back_edge_lands_deeper_in_the_loop() {
    // Arming deeper lets the loop make more progress before the freeze — the header poll fires on
    // whichever iteration the countdown lands on, and the thaw still reproduces the oracle.
    let inst = module();
    let (baseline, _) = run(&inst, 42, &init_durable_window(WINDOW));

    for backedges in 1..=5 {
        let mut win = init_durable_window(WINDOW);
        arm_freeze_after_backedges(&mut win, backedges);
        let (_, snapshot) = run(&inst, 42, &win);
        // Whether or not this count lands a freeze (a high count may exit the loop first), a thaw
        // of the resulting image must reproduce the oracle.
        let mut win = snapshot.clone();
        if read_state(&win) == STATE_UNWINDING {
            begin_thaw(&mut win, 0);
            let (thawed, _) = run(&inst, 0, &win);
            assert_eq!(
                thawed, baseline,
                "thaw at back-edge count {backedges} reproduces oracle"
            );
        }
    }
}
