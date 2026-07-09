//! JIT durable-nesting parity — the boundary pin (DURABILITY.md §4).
//!
//! §14 subtree freeze/thaw is **interpreter-only** today. The interpreter runs (and freezes/thaws) a
//! durable nested child; the JIT's durable `instantiate` **fails closed** — `svm-jit`'s
//! `instantiator_rt.rs` returns `-EINVAL` when the nursery's `durable` flag is set (it is, at the
//! common bottom of every run entry via `set_durable`) — and `FrozenNested` appears nowhere in
//! `svm-jit`. This test pins that boundary with the *same* durable-instrumented same-module nesting
//! program on both backends:
//!
//!   * the interpreter admits the durable `instantiate` + `join` and returns the child's result
//!     (777); the JIT refuses the `instantiate` (`-EINVAL`), so the subsequent `join` of that
//!     non-handle traps closed instead of returning 777.
//!
//! When a JIT slice closes the gap (durable child transform at `instantiate`, control words seeded in
//! the carve, `FrozenNested` export), this test flips to a *positive* differential — the JIT returns
//! 777 too, over a byte-identical durable reserve, the all-or-nothing oracle (DURABILITY.md §4,
//! "JIT parity"). Until then it locks the fail-closed path so a refactor can't silently re-open
//! durable JIT nesting unnoticed.

use core::ffi::c_void;
use svm_durable::{init_durable_window, transform_module};
use svm_interp::{run_capture_reserved_with_host, Host, Value};
use svm_jit::{compile_and_run_capture_reserved_with_host_durable, JitOutcome};
use svm_text::parse_module;
use svm_verify::verify_module;

const SIZE_LOG2: u8 = 18;
const WINDOW: usize = 1 << SIZE_LOG2;

/// A durable **same-module** parent: `instantiate`s its own func 1 (op 0) confined to a 128 KiB
/// sub-window, `join`s it, and returns the child's result. Func 1 is a trivial pure-compute child
/// (returns 777). On the interpreter the whole chain runs; on the JIT the `instantiate` is refused
/// (`-EINVAL`), so the `join` of that non-handle traps. (Identical in shape to
/// `durable_nesting.rs::PARENT_SELF`.)
const PARENT_SELF: &str = "memory 18
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
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const 777
  return v1
}
";

fn instrument(src: &str) -> svm_ir::Module {
    let m = parse_module(src).expect("parse");
    let inst = transform_module(&m).expect("transform");
    verify_module(&inst).expect("instrumented module verifies");
    inst
}

/// The parity boundary: the interpreter admits a durable same-module `instantiate`+`join` (child's
/// 777); the JIT fails the `instantiate` closed, so the run traps instead of returning 777. Flip to
/// a positive byte-identical differential when the first JIT durable-nesting slice lands.
#[test]
fn jit_durable_instantiate_fails_closed_while_interp_admits() {
    let inst = instrument(PARENT_SELF);

    // Interp: a durable domain admits the same-module durable child — instantiate + join run, so the
    // child's 777 comes back.
    let mut hi = Host::new();
    hi.set_durable(true);
    let ih = hi.grant_instantiator(0, WINDOW as u64);
    let mut fuel = 5_000_000u64;
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
        vec![Value::I64(777)],
        "interp admits the durable same-module instantiate + join"
    );

    // JIT: the durable run marks the §14 nursery durable, so `instantiate` fails closed and the
    // `join` of the refused handle traps — the run does not return 777. Gated on the nesting runtime
    // being available on this target (as the other JIT nesting tests are).
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
        matches!(jo, JitOutcome::Trapped(_)),
        "JIT durable nesting fails closed (instantiate -EINVAL → join traps); FrozenNested is \
         interp-only, so the run must not return 777: {jo:?}"
    );
}
