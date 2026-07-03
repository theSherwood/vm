//! Functional test for the software stack-overflow guard (feature `stack-check`, STACK_GUARD.md).
//!
//! With the guard on, a fiber that recurses without bound must trap `StackOverflow` (the prologue
//! check fires ~`RED_ZONE` above the fiber's low bound) instead of running off its control stack —
//! and a normal, shallow fiber must still run to completion (the check doesn't false-trigger). The
//! recursion runs on the fiber's own 256 KiB control stack, not the host stack, so this is bounded.
#![cfg(all(
    feature = "stack-check",
    any(
        all(unix, target_arch = "x86_64"),
        all(unix, target_arch = "aarch64"),
        all(windows, target_arch = "x86_64")
    )
))]

use svm_jit::{compile_and_run, JitOutcome, TrapKind};
use svm_text::parse_module;

// Root creates a fiber and resumes it to completion. The fiber entry (func 1) calls func 2, which
// recurses into itself forever via a non-tail `call` (frames accumulate on the fiber's control stack).
const RECURSE: &str = "\
func () -> (i64) {
block0():
  v0 = ref.func 1
  v1 = i64.const 4096
  v2 = cont.new v0 v1
  v3 = i64.const 0
  v4, v5 = cont.resume v2 v3
  return v5
}
func (i64, i64) -> (i64) {
block0(v0: i64, v1: i64):
  v2 = call 2 (v0)
  return v2
}
func (i64) -> (i64) {
block0(v0: i64):
  v1 = call 2 (v0)
  return v1
}
";

// Root creates a fiber that immediately returns 7 — no deep stack use, must run fine under the guard.
const SHALLOW: &str = "\
func () -> (i64) {
block0():
  v0 = ref.func 1
  v1 = i64.const 4096
  v2 = cont.new v0 v1
  v3 = i64.const 0
  v4, v5 = cont.resume v2 v3
  return v5
}
func (i64, i64) -> (i64) {
block0(v0: i64, v1: i64):
  v2 = i64.const 7
  return v2
}
";

#[test]
fn unbounded_fiber_recursion_traps_stack_overflow() {
    let m = parse_module(RECURSE).expect("parse");
    match compile_and_run(&m, 0, &[]).expect("jit compile/run") {
        JitOutcome::Trapped(TrapKind::StackOverflow) => {}
        other => panic!("expected StackOverflow trap, got {other:?}"),
    }
}

// Multi-vCPU: the root spawns a vCPU (its own OS thread) whose thread-entry creates + resumes a
// recursing fiber. The limit is per-vCPU by construction (each fiber entry supplies its own via an
// ABI param, no shared cell), so the spawned vCPU's fiber must trap StackOverflow on *its* stack.
const SPAWNED_RECURSE: &str = "\
func () -> (i64) {
block0():
  v0 = i64.const 0
  v1 = thread.spawn 1 v0 v0
  v2 = thread.join v1
  return v2
}
func (i64, i64) -> (i64) {
block0(v0: i64, v1: i64):
  v2 = ref.func 2
  v3 = i64.const 4096
  v4 = cont.new v2 v3
  v5 = i64.const 0
  v6, v7 = cont.resume v4 v5
  return v7
}
func (i64, i64) -> (i64) {
block0(v0: i64, v1: i64):
  v2 = call 3 (v0)
  return v2
}
func (i64) -> (i64) {
block0(v0: i64):
  v1 = call 3 (v0)
  return v1
}
";

#[test]
fn spawned_vcpu_fiber_recursion_traps_stack_overflow() {
    let m = parse_module(SPAWNED_RECURSE).expect("parse");
    // A spawned vCPU's fiber overflow is detect-and-killed and surfaces as a trap on the run.
    match compile_and_run(&m, 0, &[]).expect("jit compile/run") {
        JitOutcome::Trapped(TrapKind::StackOverflow) => {}
        other => panic!("expected StackOverflow from the spawned vCPU's fiber, got {other:?}"),
    }
}

#[test]
fn shallow_fiber_runs_under_the_guard() {
    let m = parse_module(SHALLOW).expect("parse");
    match compile_and_run(&m, 0, &[]).expect("jit compile/run") {
        JitOutcome::Returned(slots) => assert_eq!(slots, vec![7]),
        other => panic!("expected Returned([7]), got {other:?}"),
    }
}
