//! Phase-4 Slice A (4A.2) — **back-edge polls on the JIT**: the loop-header poll the transform
//! emits is ordinary control flow + window loads/stores, so the JIT compiles it like any other
//! block and must agree with the interpreter on a compute-loop freeze. The interp-only generator
//! (`durgen`) emits only *forward* branches, so the `durable_jit` fuzz never exercises a real loop;
//! these two hand-written modules pin the cross-backend property for back-edges directly.
//!
//! Two complementary claims (DURABILITY.md §7/§12.6):
//!
//!   1. **freeze parity** — a freeze-from-start whose *first* poll is a loop header (`cap.call`
//!      after the loop) leaves a **byte-identical** durable reserve on both backends, and the
//!      interp artifact thaws on the JIT to the uninterrupted result.
//!   2. **mid-loop thaw portability** — a `cap.call`-before-loop module frozen **mid-iteration** on
//!      the interpreter (via the deterministic back-edge countdown) thaws on the **JIT** under a
//!      *fresh* host, reproducing the oracle — so the JIT correctly rewinds a `LoopHeader` resume
//!      point with real loop state and reloads the saved `cap.call` value rather than re-issuing it.

use core::ffi::c_void;
use std::sync::Arc;
use svm_durable::{
    arm_freeze_after_backedges, begin_thaw, init_durable_window, read_state, transform_module,
    write_state, DURABLE_RESERVE, STATE_NORMAL, STATE_UNWINDING,
};
use svm_interp::{run_capture_reserved_with_host, Host, Value};
use svm_ir::{Memory, Module};
use svm_jit::{
    compile_and_run_capture_reserved_with_host, compile_and_run_capture_reserved_with_host_durable,
    compile_and_run_capture_reserved_with_host_durable_interruptible, FreezeController, JitError,
    JitOutcome,
};

const SIZE_LOG2: u8 = 18;
const WINDOW: usize = 1 << SIZE_LOG2;

fn module(src: &str) -> Module {
    let mut m = svm_text::parse_module(src).expect("parse");
    m.memory = Some(Memory {
        size_log2: SIZE_LOG2,
    });
    let inst = transform_module(&m).expect("transform");
    svm_verify::verify_module(&inst).expect("verify");
    inst
}

fn window_with(state: i32) -> Vec<u8> {
    let mut w = init_durable_window(WINDOW);
    write_state(&mut w, state);
    w
}

// Interpreter run; returns (result, final window). `clock` seeds Clock.now.
fn interp(inst: &Module, clock: i64, win: &[u8], durable: bool) -> (Vec<Value>, Vec<u8>) {
    let mut h = Host::new();
    h.set_durable(durable);
    h.clock_ns = clock;
    let clk = h.grant_clock();
    let mut fuel = 1_000_000u64;
    let (r, out) = run_capture_reserved_with_host(
        inst,
        0,
        &[Value::I32(clk)],
        &mut fuel,
        win,
        SIZE_LOG2,
        &mut h,
    );
    (r.expect("interp run trapped"), out)
}

// JIT run; `None` if the JIT declines to compile (a safety valve — it only sees lowered ops).
fn jit(inst: &Module, clock: i64, win: &[u8]) -> Option<(JitOutcome, Vec<u8>)> {
    let mut h = Host::new();
    h.clock_ns = clock;
    let clk = h.grant_clock();
    let slots = [clk as i64];
    match compile_and_run_capture_reserved_with_host(
        inst,
        0,
        &slots,
        win,
        SIZE_LOG2,
        svm_run::cap_thunk,
        &mut h as *mut Host as *mut c_void,
    ) {
        Ok(t) => Some(t),
        Err(JitError::Unsupported(_)) => None,
        Err(JitError::Backend(msg)) if msg.contains("Allocation error") => None,
        Err(e) => panic!("JIT failed on a verified instrumented module: {e:?}\n{inst:#?}"),
    }
}

fn jit_i64(out: &JitOutcome) -> i64 {
    match out {
        JitOutcome::Returned(slots) => slots[0],
        other => panic!("expected Returned, got {other:?}"),
    }
}

// Loop FIRST (the header is the first poll site), `cap.call` after it. Loop adds 1 five times,
// then reads the clock and adds it: oracle = 5 + clock.
const LOOP_FIRST: &str = r#"
func (i32) -> (i64) {
block 0 (v0: i32) {
  v1 = i64.const 0
  br 1(v0, v1)
}
block 1 (v2: i32, v3: i64) {
  v4 = i64.const 1
  v5 = i64.add v3 v4
  v6 = i64.const 5
  v7 = i64.lt_s v5 v6
  br_if v7 1(v2, v5) 2(v2, v5)
}
block 2 (v8: i32, v9: i64) {
  v10 = i32.const 0
  v11 = cap.call 2 0 (i32) -> (i64) v8 (v10)
  v12 = i64.add v9 v11
  return v12
  }
}
"#;

// `cap.call` FIRST (clock seeds the accumulator), then a poll-free loop adds 1 five times:
// oracle = clock + 5. Freezing mid-loop bakes the clock into the spilled accumulator.
const CAP_FIRST: &str = r#"
func (i32) -> (i64) {
block 0 (v0: i32) {
  v1 = i32.const 0
  v2 = cap.call 2 0 (i32) -> (i64) v0 (v1)
  v3 = i64.const 0
  br 1(v3, v2)
}
block 1 (v4: i64, v5: i64) {
  v6 = i64.const 1
  v7 = i64.add v4 v6
  v8 = i64.add v5 v6
  v9 = i64.const 5
  v10 = i64.lt_s v7 v9
  br_if v10 1(v7, v8) 2(v8)
}
block 2 (v11: i64) {
  return v11
  }
}
"#;

#[test]
fn freeze_from_start_at_a_loop_header_is_byte_identical_across_backends() {
    let inst = module(LOOP_FIRST);
    let clock = 42;

    // Oracle.
    let (base, _) = interp(&inst, clock, &window_with(STATE_NORMAL), false);
    assert_eq!(base, vec![Value::I64(47)], "5 + clock(42)");

    // Both backends freeze-from-start at the header poll (acc = 0, the first entry), before the
    // post-loop cap.call. The durable reserves must be byte-identical.
    let (fi, snap_i) = interp(&inst, clock, &window_with(STATE_UNWINDING), false);
    assert_eq!(fi, vec![Value::I64(0)], "interp froze (placeholder)");
    assert_eq!(read_state(&snap_i), STATE_UNWINDING);

    let Some((fj, snap_j)) = jit(&inst, clock, &window_with(STATE_UNWINDING)) else {
        return;
    };
    assert!(
        matches!(fj, JitOutcome::Returned(_)),
        "JIT froze, not trapped"
    );
    assert_eq!(read_state(&snap_j), STATE_UNWINDING, "JIT left UNWINDING");
    assert_eq!(
        &snap_i[..DURABLE_RESERVE as usize],
        &snap_j[..DURABLE_RESERVE as usize],
        "interp/JIT freeze a loop header into a byte-identical durable reserve\n{inst:#?}"
    );

    // The interp-frozen artifact thaws on the JIT to the oracle (clock did not advance before the
    // header freeze, so the continuation clock is unchanged).
    let mut thaw = snap_i.clone();
    begin_thaw(&mut thaw, 0);
    let (tj, final_j) = jit(&inst, clock, &thaw).expect("JIT thaw compiles");
    assert_eq!(
        jit_i64(&tj),
        47,
        "JIT thaw of the interp artifact reproduces the oracle"
    );
    assert_eq!(
        read_state(&final_j),
        STATE_NORMAL,
        "JIT thaw flips back to NORMAL"
    );
}

// A long poll-free loop (100M iterations adding 1), the clock folded in after: oracle = 100M + clock.
// Big enough that an async controller firing at run start reliably catches it mid-loop, yet a thaw
// from any iteration completes the remainder natively in well under a second.
const LONG_LOOP: &str = r#"
func (i32) -> (i64) {
block 0 (v0: i32) {
  v1 = i64.const 0
  br 1(v0, v1)
}
block 1 (v2: i32, v3: i64) {
  v4 = i64.const 1
  v5 = i64.add v3 v4
  v6 = i64.const 100000000
  v7 = i64.lt_s v5 v6
  br_if v7 1(v2, v5) 2(v2, v5)
}
block 2 (v8: i32, v9: i64) {
  v10 = i32.const 0
  v11 = cap.call 2 0 (i32) -> (i64) v8 (v10)
  v12 = i64.add v9 v11
  return v12
  }
}
"#;

// Run the durable entry with an async freeze controller; returns (result, final window).
fn jit_async_freeze(
    inst: &Module,
    clock: i64,
    freeze: Arc<FreezeController>,
) -> (JitOutcome, Vec<u8>) {
    let mut h = Host::new();
    h.clock_ns = clock;
    let clk = h.grant_clock();
    let slots = [clk as i64];
    let win = window_with(STATE_NORMAL);
    let (out, snap, _) = compile_and_run_capture_reserved_with_host_durable_interruptible(
        inst,
        0,
        &slots,
        &win,
        &[],
        &[],
        SIZE_LOG2,
        svm_run::cap_thunk,
        &mut h as *mut Host as *mut c_void,
        freeze,
    )
    .expect("durable interruptible run compiles");
    (out, snap)
}

// Thaw a frozen window on the JIT via the durable entry (REWINDING); returns (result, final window).
fn jit_durable_thaw(inst: &Module, clock: i64, snap: &[u8]) -> (JitOutcome, Vec<u8>) {
    let mut win = snap.to_vec();
    begin_thaw(&mut win, 0);
    let mut h = Host::new();
    h.clock_ns = clock;
    let clk = h.grant_clock();
    let slots = [clk as i64];
    let (out, final_win, _) = compile_and_run_capture_reserved_with_host_durable(
        inst,
        0,
        &slots,
        &win,
        &[],
        &[],
        SIZE_LOG2,
        svm_run::cap_thunk,
        &mut h as *mut Host as *mut c_void,
    )
    .expect("durable thaw compiles");
    (out, final_win)
}

#[test]
fn async_controller_freezes_a_running_compute_loop_on_the_jit() {
    let inst = module(LONG_LOOP);

    // Oracle is the closed form: 100M loop iterations (acc += 1) + clock(42). (Running 100M
    // iterations on the tree-walk interpreter would blow the fuel budget and be slow; the JIT runs
    // them natively. The deterministic small-loop round-trip is covered by the tests above.)
    const ORACLE: i64 = 100_000_000 + 42;

    // A controller thread requests a freeze the instant the run publishes its window base — the real
    // async stop-the-world trigger, no deterministic countdown. It catches the long loop mid-flight
    // with overwhelming probability (the loop runs ~100ms; the request lands within microseconds).
    let freeze = FreezeController::new();
    let fc = Arc::clone(&freeze);
    let controller = std::thread::spawn(move || fc.request_freeze());
    let (out, snap) = jit_async_freeze(&inst, 42, freeze);
    controller.join().unwrap();

    if read_state(&snap) == STATE_UNWINDING {
        // Froze mid-loop (the expected path). The window holds the unwound loop-header continuation;
        // a thaw must reproduce the oracle — the loop state was reloaded and the post-loop cap.call
        // runs once on thaw. (The clock is folded in only after the loop, so use the baseline clock.)
        assert!(
            matches!(out, JitOutcome::Returned(_)),
            "async freeze returns a placeholder, not a trap"
        );
        let (thawed, final_win) = jit_durable_thaw(&inst, 42, &snap);
        assert_eq!(
            jit_i64(&thawed),
            ORACLE,
            "thaw of an async-frozen compute loop reproduces the oracle"
        );
        assert_eq!(
            read_state(&final_win),
            STATE_NORMAL,
            "thaw flips back to NORMAL"
        );
    } else {
        // The loop finished before the request landed (rare): the run is simply the uninterrupted
        // result. Correctness still holds — the async path never corrupts a completed run.
        assert_eq!(
            jit_i64(&out),
            ORACLE,
            "uninterrupted async run is still correct"
        );
    }
}

#[test]
fn interp_mid_loop_freeze_thaws_on_the_jit() {
    let inst = module(CAP_FIRST);

    // Oracle (clock 42): clock seeds the accumulator, +5 ⇒ 47.
    let (base, _) = interp(&inst, 42, &window_with(STATE_NORMAL), false);
    assert_eq!(base, vec![Value::I64(47)], "clock(42) + 5");

    // Freeze mid-loop on the interpreter via the back-edge countdown: the clock (42) is already
    // baked into the spilled accumulator.
    let mut win = init_durable_window(WINDOW);
    arm_freeze_after_backedges(&mut win, 3);
    let (fi, snap) = interp(&inst, 42, &win, true);
    assert_eq!(fi, vec![Value::I64(0)], "interp froze mid-loop");
    assert_eq!(read_state(&snap), STATE_UNWINDING);

    // Thaw on the JIT under a FRESH host (clock 0). The JIT must rewind the LoopHeader point with
    // the real mid-loop accumulator and reload the baked-in clock — not re-issue the cap.call
    // (which would now read 0 and give the wrong total).
    let mut thaw = snap.clone();
    begin_thaw(&mut thaw, 0);
    let (tj, final_j) = jit(&inst, 0, &thaw).expect("JIT thaw compiles");
    assert_eq!(
        jit_i64(&tj),
        47,
        "JIT thaw of an interp mid-loop freeze reproduces the oracle (reload, not re-issue)\n{inst:#?}"
    );
    assert_eq!(
        read_state(&final_j),
        STATE_NORMAL,
        "JIT thaw flips back to NORMAL"
    );
}
