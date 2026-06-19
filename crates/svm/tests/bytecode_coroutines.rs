//! Equality harness for the bytecode engine's **§14 coroutine seam** (INTERP_PERF.md Slice 1c-5d):
//! the cooperative `Instantiator.spawn_coroutine` / `resume` + `Yielder.yield` round-trip. The
//! coroutine child runs inline over a confined `nested_view` window with a Yielder-only powerbox.
//!
//! Adapted from `crates/svm/tests/coroutine.rs`: the parent (func 0) spawns a coroutine confined to
//! `[64 KiB, 128 KiB)` and resumes it three times; the child (func 1) yields 100, then 200+r1, then
//! returns 999+r2, where r1/r2 are the values the parent delivers (10, 20). Result:
//! `100 + 210 + 1019 + RETURNED*1_000_000 = 1_001_329`. The host grants the Instantiator capability;
//! the handle reaches the guest as func 0's argument. `.expect(Some)` gates that the bytecode engine
//! drove the module (didn't fall back).

use svm_interp::{bytecode, run_with_host, Host, Value};
use svm_text::parse_module;

const CORO: &str = r#"memory 17
func (i32) -> (i64) {
block0(v0: i32):
  v1 = i64.const 1
  v2 = i64.const 65536
  v3 = i64.const 16
  v4 = i64.const 0
  v5 = cap.call 6 2 (i64, i64, i64, i64) -> (i32) v0 (v1, v2, v3, v4)
  v6 = i64.const 0
  v7, v8 = cap.call 6 3 (i32, i64) -> (i32, i64) v0 (v5, v6)
  v9 = i64.const 10
  v10, v11 = cap.call 6 3 (i32, i64) -> (i32, i64) v0 (v5, v9)
  v12 = i64.const 20
  v13, v14 = cap.call 6 3 (i32, i64) -> (i32, i64) v0 (v5, v12)
  v15 = i64.add v8 v11
  v16 = i64.add v15 v14
  v17 = i64.extend_i32_s v13
  v18 = i64.const 1000000
  v19 = i64.mul v17 v18
  v20 = i64.add v16 v19
  return v20
}
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i32.wrap_i64 v0
  v2 = i64.const 0
  v3 = i32.const 7
  i32.store8 v2 v3
  v4 = i64.const 100
  v5 = cap.call 7 0 (i64) -> (i64) v1 (v4)
  v6 = i64.const 200
  v7 = i64.add v6 v5
  v8 = cap.call 7 0 (i64) -> (i64) v1 (v7)
  v9 = i64.const 999
  v10 = i64.add v9 v8
  return v10
}
"#;

fn check_coro(src: &str, want: i64) {
    let m = parse_module(src).expect("parse");

    let mut h_tw = Host::new();
    let inst_tw = h_tw.grant_instantiator(0, 128 << 10);
    let mut f_tw = 5_000_000u64;
    let tw = run_with_host(&m, 0, &[Value::I32(inst_tw)], &mut f_tw, &mut h_tw);

    let mut h_bc = Host::new();
    let inst_bc = h_bc.grant_instantiator(0, 128 << 10);
    let mut f_bc = 5_000_000u64;
    let bc =
        bytecode::compile_and_run_with_host(&m, 0, &[Value::I32(inst_bc)], &mut f_bc, &mut h_bc)
            .expect("bytecode engine must support coroutines (Slice 1c-5d)");

    assert_eq!(tw, bc, "coroutine: tree-walker != bytecode\n{src}");
    assert_eq!(bc, Ok(vec![Value::I64(want)]), "coroutine result\n{src}");
}

#[test]
fn coroutine_resume_suspend_round_trip() {
    check_coro(CORO, 100 + 210 + 1019 + 1_000_000);
}

/// Resuming a coroutine handle that was never spawned (or already finished) is an inert `CapFault`
/// on both engines.
const FORGED_RESUME: &str = r#"memory 17
func (i32) -> (i64) {
block0(v0: i32):
  v1 = i32.const 9
  v2 = i64.const 0
  v3, v4 = cap.call 6 3 (i32, i64) -> (i32, i64) v0 (v1, v2)
  return v4
}
"#;

#[test]
fn coroutine_forged_resume_faults_identically() {
    let m = parse_module(FORGED_RESUME).expect("parse");

    let mut h_tw = Host::new();
    let inst_tw = h_tw.grant_instantiator(0, 128 << 10);
    let mut f_tw = 5_000_000u64;
    let tw = run_with_host(&m, 0, &[Value::I32(inst_tw)], &mut f_tw, &mut h_tw);

    let mut h_bc = Host::new();
    let inst_bc = h_bc.grant_instantiator(0, 128 << 10);
    let mut f_bc = 5_000_000u64;
    let bc =
        bytecode::compile_and_run_with_host(&m, 0, &[Value::I32(inst_bc)], &mut f_bc, &mut h_bc)
            .expect("bytecode supports the module");

    assert_eq!(tw, bc, "forged coroutine resume must fault identically");
}
