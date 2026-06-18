//! Equality harness for the bytecode engine's **§14 executor-child seam** (INTERP_PERF.md Slice
//! 1c-5g): `Instantiator.instantiate` / `Instantiator.join`. Unlike a coroutine (driven inline),
//! an instantiated child runs on the cooperative scheduler — confined to a power-of-two sub-window
//! of the holder's range (a `nested_view` over the shared backing), with an attenuated powerbox (an
//! `Instantiator` + an `AddressSpace`, each over its own window) and a `quota` fuel sub-budget — and
//! is joined through the shared §12 thread machinery.
//!
//! Adapted from `crates/svm/tests/instantiator.rs`. Each case is checked **bit-identical** to the
//! tree-walker `run_with_host`; `.expect(Some)` gates that the bytecode engine drove the module
//! (didn't fall back). The host grants the `Instantiator` capability (iface 6); the handle reaches
//! the guest as func 0's argument. `instantiate` is `cap.call 6 0`, `join` is `cap.call 6 1`.

use svm_interp::{bytecode, run_with_host, Host, Value};
use svm_text::parse_module;

/// Run `src`'s entry on both engines with an `Instantiator` granted over `[0, 1<<win_log2)`, and
/// assert the results are identical and equal to `want`.
fn check(src: &str, want: Result<Vec<Value>, ()>) {
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
            .expect("bytecode engine must support instantiate/join (Slice 1c-5g)");

    assert_eq!(tw, bc, "instantiate: tree-walker != bytecode\n{src}");
    match want {
        Ok(vals) => assert_eq!(bc, Ok(vals), "instantiate result\n{src}"),
        Err(()) => assert!(bc.is_err(), "expected a trap, got {bc:?}\n{src}"),
    }
}

/// Parent (func 0) instantiates the child (func 1) in a 4 KiB window at 64 KiB, joins it, then reads
/// back the marker the child wrote into the **shared** backing — proving the child ran confined on
/// the executor and its writes are visible to the parent (the §14 shared data plane). The child
/// writes 123 at its own offset 7 (→ backing 64 KiB + 7) and returns 42; the parent returns
/// `42 * 1000 + 123 = 42123`.
const SHARED_MEM: &str = r#"memory 17
func (i32) -> (i64) {
block0(v0: i32):
  v1 = i64.const 1
  v2 = i64.const 65536
  v3 = i64.const 12
  v4 = i64.const 0
  v5 = cap.call 6 0 (i64, i64, i64, i64) -> (i32) v0 (v1, v2, v3, v4)
  v6 = cap.call 6 1 (i32) -> (i64) v0 (v5)
  v7 = i64.const 65543
  v8 = i32.load8_u v7
  v9 = i64.extend_i32_u v8
  v10 = i64.const 1000
  v11 = i64.mul v6 v10
  v12 = i64.add v11 v9
  return v12
}
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const 7
  v2 = i32.const 123
  i32.store8 v1 v2
  v3 = i64.const 42
  return v3
}
"#;

#[test]
fn instantiate_join_shares_backing() {
    check(SHARED_MEM, Ok(vec![Value::I64(42123)]));
}

/// Depth-2 VM-in-VM (from `instantiator.rs`): the child, handed an `Instantiator` over *its* window,
/// itself instantiates a grandchild — confinement composes. The grandchild returns 77, propagated up
/// through two joins.
const DEPTH_TWO: &str = r#"memory 17
func (i32) -> (i64) {
block0(v0: i32):
  v1 = i64.const 1
  v2 = i64.const 65536
  v3 = i64.const 12
  v4 = i64.const 0
  v5 = cap.call 6 0 (i64, i64, i64, i64) -> (i32) v0 (v1, v2, v3, v4)
  v6 = cap.call 6 1 (i32) -> (i64) v0 (v5)
  return v6
}
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i32.wrap_i64 v0
  v2 = i64.const 0
  v3 = i32.const 171
  i32.store8 v2 v3
  v4 = i64.const 2
  v5 = i64.const 2048
  v6 = i64.const 10
  v7 = i64.const 0
  v8 = cap.call 6 0 (i64, i64, i64, i64) -> (i32) v1 (v4, v5, v6, v7)
  v9 = cap.call 6 1 (i32) -> (i64) v1 (v8)
  return v9
}
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const 0
  v2 = i32.const 200
  i32.store8 v1 v2
  v3 = i64.const 77
  return v3
}
"#;

#[test]
fn nesting_composes_to_depth_two() {
    check(DEPTH_TWO, Ok(vec![Value::I64(77)]));
}

/// A two-arg child receives its starter caps `(Instantiator, AddressSpace)`. It uses the
/// `AddressSpace` (iface 5, op 1 = `unmap`) to decommit the first 16 KiB of its **own** 64 KiB
/// window — a confined sub-window page op — and returns the unmap result (0). The parent returns it.
const ADDRESS_SPACE: &str = r#"memory 18
func (i32) -> (i64) {
block0(v0: i32):
  v1 = i64.const 1
  v2 = i64.const 65536
  v3 = i64.const 16
  v4 = i64.const 0
  v5 = cap.call 6 0 (i64, i64, i64, i64) -> (i32) v0 (v1, v2, v3, v4)
  v6 = cap.call 6 1 (i32) -> (i64) v0 (v5)
  return v6
}
func (i64, i64) -> (i64) {
block0(v0: i64, v1: i64):
  v2 = i32.wrap_i64 v1
  v3 = i64.const 0
  v4 = i64.const 16384
  v5 = cap.call 5 1 (i64, i64) -> (i64) v2 (v3, v4)
  return v5
}
"#;

#[test]
fn two_arg_child_manages_its_own_pages() {
    check(ADDRESS_SPACE, Ok(vec![Value::I64(0)]));
}

/// An out-of-range carve (a 4 KiB child at offset 128 KiB doesn't fit the 128 KiB holder) returns
/// `-EINVAL` (-22); the parent returns it without joining.
const BAD_CARVE: &str = r#"memory 17
func (i32) -> (i64) {
block0(v0: i32):
  v1 = i64.const 1
  v2 = i64.const 131072
  v3 = i64.const 12
  v4 = i64.const 0
  v5 = cap.call 6 0 (i64, i64, i64, i64) -> (i32) v0 (v1, v2, v3, v4)
  v6 = i64.extend_i32_s v5
  return v6
}
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const 0
  return v1
}
"#;

#[test]
fn out_of_range_carve_rejected() {
    check(BAD_CARVE, Ok(vec![Value::I64(-22)]));
}

/// A child trap (`unreachable`) must propagate through `join` as the parent's trap — identically on
/// both engines.
const CHILD_TRAP: &str = r#"memory 17
func (i32) -> (i64) {
block0(v0: i32):
  v1 = i64.const 1
  v2 = i64.const 0
  v3 = i64.const 12
  v4 = i64.const 0
  v5 = cap.call 6 0 (i64, i64, i64, i64) -> (i32) v0 (v1, v2, v3, v4)
  v6 = cap.call 6 1 (i32) -> (i64) v0 (v5)
  return v6
}
func (i64) -> (i64) {
block0(v0: i64):
  unreachable
}
"#;

#[test]
fn child_trap_propagates_through_join() {
    check(CHILD_TRAP, Err(()));
}
