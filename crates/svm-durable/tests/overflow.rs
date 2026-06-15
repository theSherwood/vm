//! The shadow stack traps on overflow instead of corrupting guest memory (R9 / §12.7).
//!
//! The shadow stack mirrors the call stack; the freeze-path `UNWIND` check refuses to push
//! a frame whose top would cross `DURABLE_RESERVE` into the guest's region. On the interp a
//! real overflow is unreachable (its own `MAX_CALL_DEPTH` caps recursion long before 64 KiB
//! of frames accrue), so we drive the guard directly: seed the shadow-SP near the top of the
//! reserve, so the very next push would cross it.

use svm_durable::{
    init_durable_window, transform_module, write_state, DURABLE_RESERVE, SHADOW_BASE,
    SHADOW_SP_OFF, STATE_UNWINDING,
};
use svm_interp::{run_capture_reserved_with_host, Host, Trap, Value};
use svm_ir::{Memory, Module};

const SIZE_LOG2: u8 = 18;
const WINDOW: usize = 1 << SIZE_LOG2;

const LEAF: &str = r#"
func (i32) -> (i64) {
block0(v0: i32):
  v1 = i32.const 0
  v2 = cap.call 2 0 (i32) -> (i64) v0 (v1)
  v3 = i64.const 100
  v4 = i64.add v2 v3
  return v4
}
"#;

fn instrument() -> Module {
    let mut m = svm_text::parse_module(LEAF).expect("parse");
    m.memory = Some(Memory {
        size_log2: SIZE_LOG2,
    });
    let inst = transform_module(&m).expect("transform");
    svm_verify::verify_module(&inst).expect("verify");
    inst
}

/// Freeze with the shadow-SP pre-seeded to `sp` (simulating an already-`sp`-deep stack).
fn freeze_with_sp(inst: &Module, sp: u64) -> Result<Vec<Value>, Trap> {
    let mut win = init_durable_window(WINDOW);
    write_state(&mut win, STATE_UNWINDING);
    win[SHADOW_SP_OFF as usize..SHADOW_SP_OFF as usize + 8].copy_from_slice(&sp.to_le_bytes());
    let mut host = Host::new();
    host.clock_ns = 42;
    let clk = host.grant_clock();
    let mut fuel = 1_000_000u64;
    let (r, _) = run_capture_reserved_with_host(
        inst,
        0,
        &[Value::I32(clk)],
        &mut fuel,
        &win,
        SIZE_LOG2,
        &mut host,
    );
    r
}

#[test]
fn shadow_overflow_traps_instead_of_corrupting() {
    let inst = instrument();

    // SP already at the very top of the reserve: the next frame push would cross
    // DURABLE_RESERVE into guest memory → the check traps instead.
    assert!(
        freeze_with_sp(&inst, DURABLE_RESERVE - 8).is_err(),
        "a push past the reserve traps, never writes guest memory"
    );

    // From the base, the same freeze fits and returns its placeholder.
    assert!(
        freeze_with_sp(&inst, SHADOW_BASE).is_ok(),
        "a freeze that fits within the reserve still works"
    );
}
