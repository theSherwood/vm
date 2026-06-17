//! Phase-3.1 (option A), first slice: the durable transform now recognizes the §12 fiber control
//! ops (`cont.new`/`cont.resume`/`suspend`) as may-suspend points and instruments them. This pins
//! the **NORMAL-inertness** invariant for a fiber'd module — instrumented runs identically to
//! un-instrumented — and that the instrumented IR verifies.
//!
//! Fiber *freeze/thaw* (re-issuing `cont.resume`, re-parking a `suspend`ed fiber, per-fiber shadow
//! stacks) is **not** wired yet: those resume arms fail closed (trap) for now, so this slice does
//! not exercise a fiber thaw — only that instrumentation is transparent in NORMAL.

use svm_durable::{init_durable_window, transform_module};
use svm_interp::{run_capture_reserved_with_host, Host, Trap, Value};
use svm_ir::{Memory, Module};

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
