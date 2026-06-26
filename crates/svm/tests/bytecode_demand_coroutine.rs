//! Equality harness for the bytecode engine's **§14 demand (fault-driven-yield) coroutines**
//! (INTERP_PERF.md Slice 1c-5j): `spawn_demand_coroutine` (op 4) and `spawn_demand_coroutine_module`
//! (op 7). A demand child starts with its whole window **unmapped**, so its first access to a page is
//! a *recoverable* fault that suspends to the parent (status `FAULTED` = 2, value = the fault address)
//! instead of trapping; the parent supplies the page (writing bytes into the shared window, or — for a
//! module child — relying on the data segments already materialized there) and resumes, and the
//! child's rewound access re-executes and reads it. The §14 userfaultfd-style lazy-paging model.
//!
//! The bytecode engine implements the "rewind the faulting op" with **no** hot-loop change: a demand
//! coroutine is stepped one op at a time (`budget = 1`), so a fault leaves the cursor on the faulting
//! op for free. Adapted from `crates/svm/tests/{coroutine,separate_module}.rs`; each case is checked
//! **bit-identical** to the tree-walker `run_with_host`.

use svm_interp::{bytecode, run_with_host, Host, Value};
use svm_text::parse_module;

const FAULTED: i64 = 2;
const RETURNED: i64 = 1;

/// op 4 (same-module demand): the parent spawns a demand coroutine, resumes it (first access FAULTs at
/// the child's page), supplies `123` at the fault address, and resumes again — the child's rewound
/// load reads it and RETURNs. Result `123 + FAULTED*1e6 + RETURNED*1e3`.
const DEMAND_SAME: &str = r#"memory 17
func (i32) -> (i64) {
block0(v0: i32):
  v1 = i64.const 1
  v2 = i64.const 65536
  v3 = i64.const 16
  v4 = i64.const 0
  v5 = cap.call 6 4 (i64, i64, i64, i64) -> (i32) v0 (v1, v2, v3, v4)
  v6 = i64.const 0
  v7, v8 = cap.call 6 3 (i32, i64) -> (i32, i64) v0 (v5, v6)
  v9 = i32.const 123
  i32.store8 v8 v9
  v10 = i64.const 0
  v11, v12 = cap.call 6 3 (i32, i64) -> (i32, i64) v0 (v5, v10)
  v13 = i64.extend_i32_s v7
  v14 = i64.const 1000000
  v15 = i64.mul v13 v14
  v16 = i64.extend_i32_s v11
  v17 = i64.const 1000
  v18 = i64.mul v16 v17
  v19 = i64.add v12 v15
  v20 = i64.add v19 v18
  return v20
}
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const 0
  v2 = i32.load8_u v1
  v3 = i64.extend_i32_u v2
  return v3
}
"#;

/// op 4: the fault address handed to the parent is the child's page in the *parent's* window
/// coordinates (window offset 64 KiB) — the parent returns it directly.
const DEMAND_FAULT_ADDR: &str = r#"memory 17
func (i32) -> (i64) {
block0(v0: i32):
  v1 = i64.const 1
  v2 = i64.const 65536
  v3 = i64.const 16
  v4 = i64.const 0
  v5 = cap.call 6 4 (i64, i64, i64, i64) -> (i32) v0 (v1, v2, v3, v4)
  v6 = i64.const 0
  v7, v8 = cap.call 6 3 (i32, i64) -> (i32, i64) v0 (v5, v6)
  return v8
}
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const 0
  v2 = i32.load8_u v1
  v3 = i64.extend_i32_u v2
  return v3
}
"#;

fn check_inst(src: &str, want: i64) {
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
            .expect("bytecode engine must support demand coroutines (Slice 1c-5j)");

    assert_eq!(tw, bc, "demand coroutine: tree-walker != bytecode\n{src}");
    assert_eq!(
        bc,
        Ok(vec![Value::I64(want)]),
        "demand coroutine result\n{src}"
    );
}

#[test]
fn demand_coroutine_faults_then_resumes() {
    check_inst(DEMAND_SAME, 123 + FAULTED * 1_000_000 + RETURNED * 1000);
}

#[test]
fn demand_coroutine_reports_fault_address() {
    check_inst(DEMAND_FAULT_ADDR, 65536);
}

/// op 7 (separate-module demand): a foreign module's data segments are materialized into the carve at
/// spawn, but its pages start unmapped — the child's first read of its data segment FAULTs to the
/// parent, which supplies the page by simply resuming (the bytes are already there). Lazy plugin load.
const MODULE_CHILD: &str = r#"memory 16
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

const DEMAND_MODULE_PARENT: &str = r#"memory 17
func (i32, i32) -> (i64) {
block0(v0: i32, v1: i32):
  v2 = i64.extend_i32_s v1
  v3 = i64.const 0
  v4 = i64.const 65536
  v5 = i64.const 16
  v6 = cap.call 6 7 (i64, i64, i64, i64, i64) -> (i32) v0 (v2, v3, v4, v5, v3)
  v7, v8 = cap.call 6 3 (i32, i64) -> (i32, i64) v0 (v6, v3)
  v9, v10 = cap.call 6 3 (i32, i64) -> (i32, i64) v0 (v6, v3)
  v11 = i64.extend_i32_s v7
  v12 = i64.const 1000000
  v13 = i64.mul v11 v12
  v14 = i64.extend_i32_s v9
  v15 = i64.const 100000
  v16 = i64.mul v14 v15
  v17 = i64.add v10 v13
  v18 = i64.add v17 v16
  return v18
}
"#;

#[test]
fn demand_module_coroutine_supplies_data_lazily() {
    let parent = parse_module(DEMAND_MODULE_PARENT).expect("parse parent");
    let child = parse_module(MODULE_CHILD).expect("parse child");

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
    .expect("bytecode engine must support demand module coroutines (Slice 1c-5j)");

    assert_eq!(tw, bc, "demand module coroutine: tree-walker != bytecode");
    // first resume FAULTED, second RETURNED 1086 ('V'=86 + 1000).
    assert_eq!(
        bc,
        Ok(vec![Value::I64(
            1086 + FAULTED * 1_000_000 + RETURNED * 100_000
        )])
    );
}
