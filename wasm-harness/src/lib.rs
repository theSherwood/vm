//! Throwaway viability harness: run an SVM guest through the **bytecode engine** inside a
//! wasm sandbox. Exports a single `extern "C"` entry that Node calls with an i64 and gets back
//! the guest's i64 result — proving the engine executes (not just compiles) under wasm.

use svm_interp::{bytecode, Value};

/// The §ROI-spike "alu" hash recurrence: loops `n` times mixing an LCG, returns the accumulator.
/// Pure compute (no guest memory, no caps) — the cleanest end-to-end execution probe.
const ALU: &str = r#"
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const 0
  v2 = i64.const 0
  br block1(v0, v1, v2)
block1(v3: i64, v4: i64, v5: i64):
  v6 = i64.lt_s v5 v3
  br_if v6 block2(v3, v4, v5) block3(v4)
block2(v7: i64, v8: i64, v9: i64):
  v10 = i64.const 6364136223846793005
  v11 = i64.mul v8 v10
  v12 = i64.const 1442695040888963407
  v13 = i64.add v11 v12
  v14 = i64.add v13 v9
  v15 = i64.const 1
  v16 = i64.add v9 v15
  br block1(v7, v14, v16)
block3(v17: i64):
  return v17
}
"#;

/// Parse the guest, run it on the bytecode engine with arg `n`, return its i64 result.
/// Returns `i64::MIN` as an in-band sentinel for any parse/compile/trap failure.
#[no_mangle]
pub extern "C" fn run_guest(n: i64) -> i64 {
    let m = match svm_text::parse_module(ALU) {
        Ok(m) => m,
        Err(_) => return i64::MIN,
    };
    let mut fuel = u64::MAX;
    match bytecode::compile_and_run(&m, 0, &[Value::I64(n)], &mut fuel) {
        Some(Ok(vals)) => match vals.first() {
            Some(Value::I64(x)) => *x,
            _ => i64::MIN,
        },
        _ => i64::MIN,
    }
}
