//! Equality harness for the bytecode engine's **§14 separate-module executor child** (INTERP_PERF.md
//! Slice 1c-5h): `Instantiator.instantiate_module` (op 5). The host verifies a *different* module and
//! grants the parent a `Module` capability (iface 8); the parent passes it to op 5 to spawn a child
//! **domain** running that module, confined to a carve of the parent's window — the "plugin-in-plugin"
//! story. The child's data segments materialize into the carve at spawn, and the carve must equal the
//! module's declared memory. Joined via the shared thread machinery (op 1), exactly like op 0.
//!
//! Adapted from `crates/svm/tests/separate_module.rs`. Each case is checked **bit-identical** to the
//! tree-walker `run_with_host`; `.expect(Some)` gates that the bytecode engine drove the parent
//! module (didn't fall back). The parent entry takes `(instantiator, module handle)`.

use svm_interp::{bytecode, run_with_host, Host, Value};
use svm_text::parse_module;

/// The child ("plugin") module: a 64 KiB window with a data segment `"VM"` at offset 100. Its entry
/// (`(i64 instantiator) -> (i64)`) loads its own data byte at 100, stores a marker at offset 0, and
/// returns `byte + 1000` — exercising a foreign module's code, data, and window writes, confined.
const CHILD_SRC: &str = r#"memory 16
data 100 "VM"
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const 100
  v2 = i32.load8_u v1
  v3 = i64.const 0
  v4 = i32.const 7
  i32.store8 v3 v4
  v5 = i64.extend_i32_u v2
  v6 = i64.const 1000
  v7 = i64.add v5 v6
  return v7
}
"#;

/// Run `parent_src`'s entry on both engines with `(instantiator over the whole window, module handle
/// for `child_src`)` as its two args, and assert the results are identical and equal to `want`.
fn check(parent_src: &str, child_src: &str, want: Result<Vec<Value>, ()>) {
    let parent = parse_module(parent_src).expect("parse parent");
    let child = parse_module(child_src).expect("parse child");

    let mut h_tw = Host::new();
    let ih_tw = h_tw.grant_instantiator(0, 128 << 10);
    let mh_tw = h_tw.grant_module(&child);
    let mut f_tw = 5_000_000u64;
    let tw = run_with_host(
        &parent,
        0,
        &[Value::I32(ih_tw), Value::I32(mh_tw)],
        &mut f_tw,
        &mut h_tw,
    );

    let mut h_bc = Host::new();
    let ih_bc = h_bc.grant_instantiator(0, 128 << 10);
    let mh_bc = h_bc.grant_module(&child);
    let mut f_bc = 5_000_000u64;
    let bc = bytecode::compile_and_run_with_host(
        &parent,
        0,
        &[Value::I32(ih_bc), Value::I32(mh_bc)],
        &mut f_bc,
        &mut h_bc,
    )
    .expect("bytecode engine must support instantiate_module (Slice 1c-5h)");

    assert_eq!(
        tw, bc,
        "instantiate_module: tree-walker != bytecode\n{parent_src}"
    );
    match want {
        Ok(vals) => assert_eq!(bc, Ok(vals), "instantiate_module result\n{parent_src}"),
        Err(()) => assert!(bc.is_err(), "expected a trap, got {bc:?}\n{parent_src}"),
    }
}

/// `instantiate_module(module, entry 0, off 64 KiB, size_log2 16, fuel 0)` → `join` → the child's
/// result. The child read its own data segment ('V' = 86) and returned `86 + 1000 = 1086`.
const PARENT: &str = r#"memory 17
func (i32, i32) -> (i64) {
block0(v0: i32, v1: i32):
  v2 = i64.extend_i32_s v1
  v3 = i64.const 0
  v4 = i64.const 65536
  v5 = i64.const 16
  v6 = cap.call 6 5 (i64, i64, i64, i64, i64) -> (i32) v0 (v2, v3, v4, v5, v3)
  v7 = cap.call 6 1 (i32) -> (i64) v0 (v6)
  return v7
}
"#;

#[test]
fn module_child_runs_with_its_data_segments() {
    check(PARENT, CHILD_SRC, Ok(vec![Value::I64(1086)]));
}

/// The parent then reads the marker the module child wrote at the child's offset 0 (→ backing 64 KiB)
/// and the child's data byte — proving the foreign module ran confined over the shared backing.
const PARENT_READBACK: &str = r#"memory 17
func (i32, i32) -> (i64) {
block0(v0: i32, v1: i32):
  v2 = i64.extend_i32_s v1
  v3 = i64.const 0
  v4 = i64.const 65536
  v5 = i64.const 16
  v6 = cap.call 6 5 (i64, i64, i64, i64, i64) -> (i32) v0 (v2, v3, v4, v5, v3)
  v7 = cap.call 6 1 (i32) -> (i64) v0 (v6)
  v8 = i64.const 65536
  v9 = i32.load8_u v8
  v10 = i64.extend_i32_u v9
  v11 = i64.const 1000000
  v12 = i64.mul v7 v11
  v13 = i64.add v12 v10
  return v13
}
"#;

#[test]
fn module_child_writes_visible_to_parent() {
    // child returns 1086; marker at child offset 0 is 7 → 1086 * 1_000_000 + 7.
    check(
        PARENT_READBACK,
        CHILD_SRC,
        Ok(vec![Value::I64(1_086_000_007)]),
    );
}

/// A carve whose size doesn't equal the module's declared memory (size_log2 12 ≠ the child's 16) is
/// rejected with `-EINVAL` — §14 transparency (the plugin must run exactly as it would standalone).
const PARENT_BAD_SIZE: &str = r#"memory 17
func (i32, i32) -> (i64) {
block0(v0: i32, v1: i32):
  v2 = i64.extend_i32_s v1
  v3 = i64.const 0
  v4 = i64.const 65536
  v5 = i64.const 12
  v6 = cap.call 6 5 (i64, i64, i64, i64, i64) -> (i32) v0 (v2, v3, v4, v5, v3)
  v7 = i64.extend_i32_s v6
  return v7
}
"#;

#[test]
fn carve_must_equal_declared_memory() {
    check(PARENT_BAD_SIZE, CHILD_SRC, Ok(vec![Value::I64(-22)]));
}

/// A forged module handle is an inert `CapFault` on both engines (the parent passes 999 as the
/// module handle, which was never granted).
const PARENT_FORGED: &str = r#"memory 17
func (i32, i32) -> (i64) {
block0(v0: i32, v1: i32):
  v2 = i64.const 999
  v3 = i64.const 0
  v4 = i64.const 65536
  v5 = i64.const 16
  v6 = cap.call 6 5 (i64, i64, i64, i64, i64) -> (i32) v0 (v2, v3, v4, v5, v3)
  v7 = i64.extend_i32_s v6
  return v7
}
"#;

#[test]
fn forged_module_handle_faults_identically() {
    check(PARENT_FORGED, CHILD_SRC, Err(()));
}

/// A coroutine ("plugin") module: a 64 KiB window whose entry (`(i64 yielder) -> (i64)`) yields 100,
/// then `200 + r1`, then returns `999 + r2` (r1/r2 are the resume values). Stores a marker at offset
/// 0 to exercise its confined window too.
const CORO_CHILD: &str = r#"memory 16
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i32.wrap_i64 v0
  va = i64.const 0
  vb = i32.const 7
  i32.store8 va vb
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

/// `spawn_coroutine_module(module, entry 0, off 64 KiB, size_log2 16, fuel 0)` → resume three times
/// (delivering 0, 10, 20). Yields 100, 210, then returns 1019; `100 + 210 + 1019 + RETURNED*1e6`.
const PARENT_CORO: &str = r#"memory 17
func (i32, i32) -> (i64) {
block0(v0: i32, v1: i32):
  v2 = i64.extend_i32_s v1
  v3 = i64.const 0
  v4 = i64.const 65536
  v5 = i64.const 16
  v6 = i64.const 0
  v7 = cap.call 6 6 (i64, i64, i64, i64, i64) -> (i32) v0 (v2, v3, v4, v5, v6)
  v8 = i64.const 0
  v9, v10 = cap.call 6 3 (i32, i64) -> (i32, i64) v0 (v7, v8)
  v11 = i64.const 10
  v12, v13 = cap.call 6 3 (i32, i64) -> (i32, i64) v0 (v7, v11)
  v14 = i64.const 20
  v15, v16 = cap.call 6 3 (i32, i64) -> (i32, i64) v0 (v7, v14)
  v17 = i64.add v10 v13
  v18 = i64.add v17 v16
  v19 = i64.extend_i32_s v15
  v20 = i64.const 1000000
  v21 = i64.mul v19 v20
  v22 = i64.add v18 v21
  return v22
}
"#;

#[test]
fn coroutine_module_round_trip() {
    check(
        PARENT_CORO,
        CORO_CHILD,
        Ok(vec![Value::I64(100 + 210 + 1019 + 1_000_000)]),
    );
}
