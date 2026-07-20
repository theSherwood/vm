//! **Phase 6 — the reactor model.** A live, stateful `Session`: instantiate once, then call an
//! exported function repeatedly with the guest window (here a BSS accumulator) persisting between
//! calls. The differential `DiffSession` steps the tree-walker, bytecode engine, and JIT in lockstep
//! across the call sequence and asserts they agree on results, output, and persistent state — the
//! first direct exercise of the bytecode engine under the powerbox (Followup F10).
//!
//! Gated `#![cfg(unix)]` like the other JIT differential suites.
#![cfg(unix)]

use svm_run::{instantiate, Backend, RunConfig, Value};

/// A reactor program with no capabilities: the paramless `_start` (func 0, run once by
/// [`Instance::start`]) zeroes an accumulator in a BSS window slot (offset 1024, page 0, no data
/// segment → persisted, not reset by `init_data`); `add(sp, x)` adds `x` to it and returns the
/// running total. Non-`_start` exports keep the `(i64 sp, …)` reactor calling convention
/// (`call_export` supplies `sp`); only func 0 / `_start` is paramless (IMPORTS.md phase 4).
const COUNTER: &str = "\
memory 15
export \"_start\" 0
export \"add\" 1
func () -> (i32) {
block0():
  v0 = i64.const 1024
  v1 = i64.const 0
  i64.store v0 v1
  v2 = i32.const 0
  return v2
}
func (i64, i64) -> (i64) {
block0(v0: i64, v1: i64):
  v2 = i64.const 1024
  v3 = i64.load v2
  v4 = i64.add v3 v1
  i64.store v2 v4
  return v4
}
";

fn counter_module() -> svm_ir::Module {
    svm_text::parse_module(COUNTER).expect("parse")
}

/// On a single backend, state persists across `call_export` — the accumulator grows.
#[test]
fn state_persists_across_calls_on_each_backend() {
    for backend in [Backend::TreeWalk, Backend::Bytecode, Backend::Jit] {
        let instance = instantiate(counter_module()).expect("instantiate");
        let mut session = instance
            .start(backend, &RunConfig::default())
            .unwrap_or_else(|e| panic!("{backend:?} start: {e}"));

        // Each call adds and returns the running total — proving the window persists between calls.
        let totals: Vec<i64> = [5i64, 3, 10, 100]
            .iter()
            .map(|&x| match session.call_export("add", &[Value::I64(x)]) {
                Ok(v) => match v.as_slice() {
                    [Value::I64(t)] => *t,
                    other => panic!("{backend:?}: unexpected results {other:?}"),
                },
                Err(e) => panic!("{backend:?} call: {e}"),
            })
            .collect();
        assert_eq!(totals, vec![5, 8, 18, 118], "{backend:?} running totals");
    }
}

/// The stateful differential: all three backends agree across the whole call sequence (results +
/// persistent window). A divergence at any step is an error.
#[test]
fn all_three_backends_agree_across_the_call_sequence() {
    let instance = instantiate(counter_module()).expect("instantiate");
    let mut diff = instance
        .start_diff(&RunConfig::default())
        .expect("start_diff");

    let mut running = 0i64;
    for x in [7i64, 11, 13, 1000, -4] {
        running += x;
        let got = diff
            .call_export("add", &[Value::I64(x)])
            .unwrap_or_else(|e| panic!("diff call: {e}"));
        assert_eq!(
            got,
            vec![Value::I64(running)],
            "all backends agree on the running total after adding {x}"
        );
    }
}

/// A fresh `start` resets state — sessions are independent (no leakage through the persisted window).
#[test]
fn sessions_are_independent() {
    let instance = instantiate(counter_module()).expect("instantiate");

    let mut a = instance
        .start(Backend::Jit, &RunConfig::default())
        .expect("a");
    assert_eq!(
        a.call_export("add", &[Value::I64(5)]).unwrap(),
        vec![Value::I64(5)]
    );
    assert_eq!(
        a.call_export("add", &[Value::I64(5)]).unwrap(),
        vec![Value::I64(10)]
    );

    // A second session starts from zero, unaffected by `a`'s accumulated state.
    let mut b = instance
        .start(Backend::Jit, &RunConfig::default())
        .expect("b");
    assert_eq!(
        b.call_export("add", &[Value::I64(1)]).unwrap(),
        vec![Value::I64(1)]
    );
}

/// A missing export is fail-closed.
#[test]
fn unknown_export_fails_closed() {
    let instance = instantiate(counter_module()).expect("instantiate");
    let mut s = instance
        .start(Backend::TreeWalk, &RunConfig::default())
        .expect("start");
    assert!(s.call_export("nope", &[]).is_err());
}
