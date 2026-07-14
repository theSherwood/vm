//! JIT durable-nesting parity — slice 1: a durable **same-module** nested child runs on the JIT,
//! matching the interpreter (DURABILITY.md §4, "JIT parity").
//!
//! Until this slice the JIT's durable `instantiate` failed closed entirely (`instantiator_rt.rs`,
//! `-EINVAL`). Slice 1 admits a same-module durable child: its funcs are the parent's own funcs, run
//! in the carve as an ordinary top-level guest. A *runnable* same-module child on the JIT is a
//! pure-compute (non-may-suspend) func — it has no poll sites, so it runs atomically to completion
//! with no durable control-word setup (a would-be *instrumented* child hits a `cap.call` against its
//! empty powerbox → `CapFault`, so it never reaches an unwind). The child here sums 0..100 = 4950;
//! both backends must return it.
//!
//! Freezing a *live* nested child on the JIT — which needs the carve's ctx-0 control words + shadow
//! base seeded to match the interpreter, plus a child powerbox — is the next slice; separate-module
//! durable children and `coro_spawn` stay fail-closed.

use core::ffi::c_void;
use svm_durable::{init_durable_window, transform_module};
use svm_interp::{run_capture_reserved_with_host, Host, Value};
use svm_jit::{compile_and_run_capture_reserved_with_host_durable, JitOutcome};
use svm_text::parse_module;
use svm_verify::verify_module;

const SIZE_LOG2: u8 = 18;
const WINDOW: usize = 1 << SIZE_LOG2;

/// A durable **same-module** parent: `instantiate`s its own func 1 (op 0) confined to a 128 KiB
/// sub-window, `join`s it, and returns the child's result. Func 1 sums 0..100 = 4950 — pure compute
/// (no `cap.call`), so it is not may-suspend and runs atomically in the carve. (Identical in shape to
/// `durable_nesting.rs::PARENT_SELF_LOOP`.)
const PARENT_SELF_LOOP: &str = "memory 18
func (i32) -> (i64) {
block0(v0: i32):
  v1 = i64.const 1
  v2 = i64.const 131072
  v3 = i64.const 17
  v4 = i64.const 0
  v5 = cap.call 6 0 (i64, i64, i64, i64) -> (i32) v0 (v1, v2, v3, v4)
  v6 = cap.call 6 1 (i32) -> (i64) v0 (v5)
  return v6
}
func (i64, i64) -> (i64) {
block0(v0: i64, v1: i64):
  v2 = i64.const 0
  v3 = i64.const 0
  br block1(v2, v3)
block1(v4: i64, v5: i64):
  v6 = i64.const 100
  v7 = i64.lt_s v4 v6
  br_if v7 block2(v4, v5) block3(v5)
block2(v8: i64, v9: i64):
  v10 = i64.add v9 v8
  v11 = i64.const 1
  v12 = i64.add v8 v11
  br block1(v12, v10)
block3(v13: i64):
  return v13
}
";

fn instrument(src: &str) -> svm_ir::Module {
    let m = parse_module(src).expect("parse");
    let inst = transform_module(&m).expect("transform");
    verify_module(&inst).expect("instrumented module verifies");
    inst
}

/// Slice 1: a durable same-module `instantiate` + `join` returns the nested child's total (4950)
/// on **both** backends — the JIT now runs the durable child instead of failing closed.
#[test]
fn jit_durable_same_module_child_matches_interp() {
    let inst = instrument(PARENT_SELF_LOOP);

    // Interp: the durable domain runs the nested child; its 4950 comes back through join.
    let mut hi = Host::new();
    hi.set_durable(true);
    let ih = hi.grant_instantiator(0, WINDOW as u64);
    let mut fuel = 50_000_000u64;
    let (ir, _imem) = run_capture_reserved_with_host(
        &inst,
        0,
        &[Value::I32(ih)],
        &mut fuel,
        &init_durable_window(WINDOW),
        SIZE_LOG2,
        &mut hi,
    );
    assert_eq!(
        ir.expect("interp durable run ok"),
        vec![Value::I64(4950)],
        "interp runs the durable same-module nested child"
    );

    // JIT: slice 1 admits the durable same-module child (it runs atomically in its carve), so it
    // returns 4950 too — the boundary flipped from fail-closed to a positive differential.
    // Gated on the nesting runtime being available on this target (as the other JIT nesting tests).
    if !svm_jit::fiber_supported() {
        return;
    }
    let mut hj = Host::new();
    hj.set_durable(true);
    let jh = hj.grant_instantiator(0, WINDOW as u64);
    let win = init_durable_window(WINDOW);
    let (jo, _jmem, _residue) = compile_and_run_capture_reserved_with_host_durable(
        &inst,
        0,
        &[jh as i64],
        &win,
        &[],
        &[],
        SIZE_LOG2,
        svm_run::cap_thunk,
        &mut hj as *mut Host as *mut c_void,
    )
    .expect("durable run compiles");
    assert!(
        matches!(jo, JitOutcome::Returned(ref s) if s == &[4950]),
        "JIT runs the durable same-module nested child, matching the interp's 4950: {jo:?}"
    );
}
