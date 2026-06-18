//! Equality harness for the bytecode engine's **§12 fiber seam** (INTERP_PERF.md Slice 1c-5b):
//! cooperative continuation switching (`cont.new` / `cont.resume` / `suspend`), single-vCPU and
//! inline-driven (no M:N pool, no DPOR — that is the thread slice). The random generator never emits
//! `cont.*`, so — per the TDD plan — this authors fiber modules by hand and asserts the bytecode
//! engine agrees bit-for-bit with the reference tree-walker `run`.
//!
//! `cont.new`/`cont.resume`/`suspend` don't touch the powerbox, so an empty host is fine on both
//! engines. The `.expect(Some)` on `compile_and_run` gates that the bytecode engine actually drove
//! the module (didn't fall back to the tree-walker).

use svm_interp::{bytecode, run, Trap, Value};
use svm_text::parse_module;

fn check_fiber(src: &str, args: &[Value]) {
    let m = parse_module(src).expect("parse");
    let mut f_tw = 1_000_000u64;
    let tw = run(&m, 0, args, &mut f_tw);
    let mut f_bc = 1_000_000u64;
    let bc = bytecode::compile_and_run(&m, 0, args, &mut f_bc)
        .expect("bytecode engine must support fiber ops (Slice 1c-5b)");
    assert!(
        !matches!(tw, Err(Trap::OutOfFuel)) && !matches!(bc, Err(Trap::OutOfFuel)),
        "unexpected OutOfFuel\n tw={tw:?}\n bc={bc:?}\n{src}"
    );
    assert_eq!(tw, bc, "fiber: tree-walker != bytecode\n{src}");
}

/// A fiber that runs to completion (no suspend): `cont.resume` delivers `arg=7`, the fiber returns
/// `arg + 100 = 107`, so the resumer sees `(RETURNED, 107)`.
const RUN_TO_COMPLETION: &str = r#"
func () -> (i64) {
block0():
  v0 = ref.func 1
  v1 = i64.const 0
  v2 = cont.new v0 v1
  v3 = i64.const 7
  v4, v5 = cont.resume v2 v3
  return v5
}
func (i64, i64) -> (i64) {
block0(vsp: i64, varg: i64):
  v0 = i64.const 100
  v1 = i64.add varg v0
  return v1
}
"#;

/// The status of a run-to-completion fiber is RETURNED — return it (proves the i32 status result).
const RETURN_STATUS: &str = r#"
func () -> (i32) {
block0():
  v0 = ref.func 1
  v1 = i64.const 0
  v2 = cont.new v0 v1
  v3 = i64.const 7
  v4, v5 = cont.resume v2 v3
  return v4
}
func (i64, i64) -> (i64) {
block0(vsp: i64, varg: i64):
  return varg
}
"#;

/// Suspend round-trip: first resume (arg=10) suspends with value `10+1=11` (status SUSPENDED); the
/// second resume (arg=20) delivers 20 as the `suspend` result, the fiber returns `20+5=25` (status
/// RETURNED). Result: `v5 + v8 = 11 + 25 = 36`.
const SUSPEND_ROUNDTRIP: &str = r#"
func () -> (i64) {
block0():
  v0 = ref.func 1
  v1 = i64.const 0
  v2 = cont.new v0 v1
  v3 = i64.const 10
  v4, v5 = cont.resume v2 v3
  v6 = i64.const 20
  v7, v8 = cont.resume v2 v6
  v9 = i64.add v5 v8
  return v9
}
func (i64, i64) -> (i64) {
block0(vsp: i64, varg: i64):
  v0 = i64.const 1
  v1 = i64.add varg v0
  v2 = suspend v1
  v3 = i64.const 5
  v4 = i64.add v2 v3
  return v4
}
"#;

/// A fiber that suspends in a loop, accumulating the resume args, then returns the total. The
/// resumer feeds it `n, n-1, ..., 1` and a final `0` sentinel; exercises repeated park/resume of the
/// same fiber and the suspend-result delivery across many switches.
const SUSPEND_LOOP: &str = r#"
func () -> (i64) {
block0():
  v0 = ref.func 1
  v1 = i64.const 0
  v2 = cont.new v0 v1
  v3 = i64.const 3
  v4, v5 = cont.resume v2 v3
  v6 = i64.const 4
  v7, v8 = cont.resume v2 v6
  v9 = i64.const 5
  v10, v11 = cont.resume v2 v9
  v12 = i64.add v5 v8
  v13 = i64.add v12 v11
  return v13
}
func (i64, i64) -> (i64) {
block0(vsp: i64, varg: i64):
  v0 = suspend varg
  v1 = suspend v0
  v2 = i64.add varg v0
  v3 = i64.add v2 v1
  return v3
}
"#;

#[test]
fn fiber_run_to_completion() {
    check_fiber(RUN_TO_COMPLETION, &[]);
}

#[test]
fn fiber_return_status() {
    check_fiber(RETURN_STATUS, &[]);
}

#[test]
fn fiber_suspend_roundtrip() {
    check_fiber(SUSPEND_ROUNDTRIP, &[]);
}

#[test]
fn fiber_suspend_loop() {
    check_fiber(SUSPEND_LOOP, &[]);
}

/// Resuming a never-created handle is an inert `FiberFault` on both engines.
const FORGED_RESUME: &str = r#"
func () -> (i64) {
block0():
  v0 = i32.const 99
  v1 = i64.const 5
  v2, v3 = cont.resume v0 v1
  return v3
}
"#;

/// The root activation cannot `suspend` (no resumer) — `FiberFault` on both engines.
const ROOT_SUSPEND: &str = r#"
func () -> (i64) {
block0():
  v0 = i64.const 5
  v1 = suspend v0
  return v1
}
"#;

#[test]
fn fiber_forged_resume_faults_identically() {
    check_fiber(FORGED_RESUME, &[]);
}

#[test]
fn fiber_root_suspend_faults_identically() {
    check_fiber(ROOT_SUSPEND, &[]);
}
