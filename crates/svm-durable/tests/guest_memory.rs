//! A memory-using durable guest round-trips through the **confined** path (R9 / §12.7).
//!
//! The durable region is a reserved low slice `[0, DURABLE_RESERVE)` of the guest's own
//! window; guest memory lives in `[DURABLE_RESERVE, window)` (the wasm shadow-stack
//! convention — runtime state below `__heap_base`, the program's data above). A cooperating
//! toolchain bases the guest's data there, so the two never overlap. This test instruments
//! such a guest via [`transform_module_assume_confined`] and shows guest memory survives
//! freeze→serialize→restore→thaw (it rides the window image), while the strict
//! [`transform_module`] still rejects the same guest (it can't prove the confinement).

use svm_durable::{
    init_durable_window, read_state, transform_module, transform_module_assume_confined,
    write_state, TransformError, DURABLE_RESERVE, STATE_NORMAL, STATE_REWINDING, STATE_UNWINDING,
};
use svm_interp::{run_capture_reserved_with_host, Host, Value};
use svm_ir::{Memory, Module};

const SIZE_LOG2: u8 = 18; // 256 KiB window: 64 KiB reserve + ~192 KiB guest-usable
const WINDOW: usize = 1 << SIZE_LOG2;

fn module(src: &str) -> Module {
    let mut m = svm_text::parse_module(src).expect("parse");
    m.memory = Some(Memory {
        size_log2: SIZE_LOG2,
    });
    m
}

fn run(inst: &Module, clock_ns: i64, window: &[u8]) -> (Vec<Value>, Vec<u8>) {
    let mut host = Host::new();
    host.clock_ns = clock_ns;
    let clk = host.grant_clock();
    let mut fuel = 1_000_000u64;
    let (r, win) = run_capture_reserved_with_host(
        inst,
        0,
        &[Value::I32(clk)],
        &mut fuel,
        window,
        SIZE_LOG2,
        &mut host,
    );
    (r.expect("runs to completion"), win)
}

// Store 77 into guest memory at `DURABLE_RESERVE` (the first usable byte), call the clock,
// then load it back *after* the call. The stored value must survive the freeze (it lives in
// the window image), and the address `v1` is live across the call (used by the reload).
// Baseline (clock 42): 42 + 77 = 119.
fn guest_src() -> String {
    format!(
        "func (i32) -> (i64) {{\n\
block0(v0: i32):\n\
  v1 = i64.const {addr}\n\
  v2 = i64.const 77\n\
  i64.store v1 v2\n\
  v3 = i32.const 0\n\
  v4 = cap.call 2 0 (i32) -> (i64) v0 (v3)\n\
  v5 = i64.load v1\n\
  v6 = i64.add v4 v5\n\
  return v6\n\
}}\n",
        addr = DURABLE_RESERVE
    )
}

#[test]
fn guest_memory_survives_freeze_thaw_via_confined_path() {
    let m = module(&guest_src());
    let inst = transform_module_assume_confined(&m).expect("confined transform");
    svm_verify::verify_module(&inst).expect("instrumented IR must verify");

    // Baseline: the uninterrupted run.
    let (baseline, _) = run(&inst, 42, &init_durable_window(WINDOW));
    assert_eq!(baseline, vec![Value::I64(119)], "42 + stored 77");

    // Freeze: the store happens, then the poll after the call unwinds.
    let mut win = init_durable_window(WINDOW);
    write_state(&mut win, STATE_UNWINDING);
    let (_, snapshot) = run(&inst, 42, &win);
    assert_eq!(
        read_state(&snapshot),
        STATE_UNWINDING,
        "froze, did not complete"
    );

    // Thaw on a fresh host: the guest's stored 77 rides the restored window image, the cap
    // result (42) is reloaded, and the post-call reload reads 77 back.
    let mut win = snapshot.clone();
    write_state(&mut win, STATE_REWINDING);
    let (thawed, final_win) = run(&inst, 0, &win);
    assert_eq!(
        thawed, baseline,
        "guest memory + durable state round-tripped"
    );
    assert_eq!(read_state(&final_win), STATE_NORMAL, "thaw ends NORMAL");
}

#[test]
fn strict_path_rejects_the_same_memory_using_guest() {
    let m = module(&guest_src());
    assert_eq!(transform_module(&m), Err(TransformError::GuestUsesMemory));
}

#[test]
fn window_smaller_than_the_reserve_is_rejected() {
    // A window that cannot even hold the reserved region is too small.
    let mut m = svm_text::parse_module(
        "func (i32) -> (i64) {\nblock0(v0: i32):\n  v1 = i32.const 0\n  v2 = cap.call 2 0 (i32) -> (i64) v0 (v1)\n  return v2\n}\n",
    )
    .unwrap();
    m.memory = Some(Memory { size_log2: 12 }); // 4 KiB < 64 KiB reserve
    assert_eq!(transform_module(&m), Err(TransformError::MemoryTooSmall));
}
