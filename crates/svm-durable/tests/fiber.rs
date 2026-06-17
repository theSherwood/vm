//! Phase-3.1 (option A): the durable transform recognizes the §12 fiber control ops
//! (`cont.new`/`cont.resume`/`suspend`) as may-suspend points and instruments them. This pins
//! the **NORMAL-inertness** invariant for a fiber'd module — instrumented runs identically to
//! un-instrumented — and that the instrumented IR verifies.
//!
//! Slice 3.1.2 wires the **resumer-side thaw arm**: a `cont.resume` rewinds by re-issuing the
//! resume (reloading its spilled handle + arg), mirroring a propagated call. The **fiber side**
//! — re-parking a `suspend`ed fiber on thaw (slice 3.1.3) — is still fail-closed (its arm traps),
//! so a full fiber thaw isn't exercised here yet; the freeze driver + snapshot wiring (3.1.4–5)
//! complete the round-trip.

use svm_durable::{init_durable_window, transform_module};
use svm_interp::{run_capture_reserved_with_host, Host, Trap, Value};
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

/// Slice 3.1.2: the `cont.resume` resumer-side thaw arm is wired (re-issues the resume), while a
/// `suspend` point's arm still fails closed. Structurally: instrumenting adds one re-issued
/// `cont.resume` per resume point (its rewind arm) on top of the one kept in the forward segment,
/// but adds **no** new `suspend` (the `Yield` arm is a bare `Unreachable` trap, slice 3.1.3).
#[test]
fn resume_thaw_arm_reissues_while_suspend_stays_fail_closed() {
    let mut m = svm_text::parse_module(SRC).expect("parse");
    m.memory = Some(Memory {
        size_log2: SIZE_LOG2,
    });
    let (res0, susp0, _) = op_and_trap_counts(&m);
    assert_eq!((res0, susp0), (2, 1), "source: 2 cont.resume, 1 suspend");

    let inst = transform_module(&m).expect("transform");
    svm_verify::verify_module(&inst).expect("instrumented IR verifies");
    let (res1, susp1, traps) = op_and_trap_counts(&inst);

    // Each of the 2 resume points keeps its forward `cont.resume` and gains a re-issue in its
    // rewind arm → the count doubles. The single `suspend` gains no re-issue (its arm traps).
    assert_eq!(
        res1,
        2 * res0,
        "each cont.resume gains a re-issuing rewind arm"
    );
    assert_eq!(susp1, susp0, "a suspend point's thaw arm does not re-issue");
    // At least the one `suspend`'s fail-closed arm (plus the shared forged-id TRAP block).
    assert!(
        traps >= 2,
        "the suspend arm and the forged-id trap are Unreachable"
    );
}
