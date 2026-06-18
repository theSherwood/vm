//! Equality harness for the bytecode engine's **§22 cross-module dispatch** (INTERP_PERF.md Slice
//! 1c-5e): `Jit.install` of a unit, then a module-0 `call_indirect` into it. The unit is
//! host-pre-compiled (so no in-guest blob seeding is needed — the bytecode entry builds memory from
//! the module, not an init image), and its code handle is passed to the guest as an argument.
//!
//! The guest installs the unit (→ a table slot) and `call_indirect`s that slot; on the bytecode
//! engine this exercises the runtime dispatch table + a cross-module activation. Compared against the
//! tree-walker `run_with_host`; `.expect(Some)` gates that bytecode drove it (didn't fall back).

use svm_interp::{bytecode, run_with_host, Host, Value};
use svm_run::grant_jit;
use svm_text::parse_module;
use svm_verify::verify_module;

/// The guest (func 0) takes `(jit_handle, code_handle, a, b)`: install the unit, then call its entry
/// indirectly through the freed slot. (1 guest func + a 16-slot table ⇒ install lands at slot 1.)
const GUEST: &str = r#"memory 16
func (i32, i32, i32, i32) -> (i32) {
block0(v0: i32, v1: i32, v2: i32, v3: i32):
  v4 = i64.extend_i32_u v1
  v5 = cap.call 11 3 (i64) -> (i64) v0 (v4)
  v6 = i32.wrap_i64 v5
  v7 = call_indirect (i32, i32) -> (i32) v6 (v2, v3)
  return v7
}
"#;

/// The unit: `service(a, b) = a*b + 100`.
const SERVICE: &str = r#"memory 16
func (i32, i32) -> (i32) {
block0(v0: i32, v1: i32):
  v2 = i32.mul v0 v1
  v3 = i32.const 100
  v4 = i32.add v2 v3
  return v4
}
"#;

/// A fresh host with the `Jit` cap (16-slot table) and the unit host-compiled; returns
/// `(host, jit_handle, code_handle)`. Granting/compiling into a fresh host is deterministic, so both
/// engines get identical handles.
fn host_with_unit(guest: &svm_ir::Module) -> (Host, i32, i32) {
    let mut host = Host::new();
    let jit = grant_jit(&mut host, guest, 4); // sets the blob validator; 2^4 = 16-slot table
    let svc = {
        let m = parse_module(SERVICE).expect("parse service");
        verify_module(&m).expect("verify service");
        svm_encode::encode_module(&m)
    };
    let code = host
        .jit_compile(jit, &svc)
        .expect("no trap")
        .expect("compile ok")
        .handle;
    (host, jit, code)
}

/// Install, then `uninstall` the slot, then `call_indirect` it — the freed slot traps
/// (`IndirectCallType`) identically on both engines.
const GUEST_UNINSTALL: &str = r#"memory 16
func (i32, i32, i32, i32) -> (i32) {
block0(v0: i32, v1: i32, v2: i32, v3: i32):
  v4 = i64.extend_i32_u v1
  v5 = cap.call 11 3 (i64) -> (i64) v0 (v4)
  v6 = cap.call 11 4 (i64) -> (i64) v0 (v5)
  v7 = i32.wrap_i64 v5
  v8 = call_indirect (i32, i32) -> (i32) v7 (v2, v3)
  return v8
}
"#;

#[test]
fn uninstall_then_call_indirect_traps_identically() {
    let m = parse_module(GUEST_UNINSTALL).expect("parse guest");
    verify_module(&m).expect("verify guest");

    let (mut h_tw, jit_tw, code_tw) = host_with_unit(&m);
    let mut f_tw = 50_000_000u64;
    let tw = run_with_host(
        &m,
        0,
        &[
            Value::I32(jit_tw),
            Value::I32(code_tw),
            Value::I32(6),
            Value::I32(7),
        ],
        &mut f_tw,
        &mut h_tw,
    );

    let (mut h_bc, jit_bc, code_bc) = host_with_unit(&m);
    let mut f_bc = 50_000_000u64;
    let bc = bytecode::compile_and_run_with_host(
        &m,
        0,
        &[
            Value::I32(jit_bc),
            Value::I32(code_bc),
            Value::I32(6),
            Value::I32(7),
        ],
        &mut f_bc,
        &mut h_bc,
    )
    .expect("bytecode supports install/uninstall (Slice 1c-5e)");

    assert_eq!(tw, bc, "uninstall+call_indirect: tree-walker != bytecode");
    assert!(
        matches!(bc, Err(svm_interp::Trap::IndirectCallType)),
        "{bc:?}"
    );
}

/// Host-compile `src` into `host`'s `Jit` domain `jit`, returning the code handle.
fn compile_unit(host: &mut Host, jit: i32, src: &str) -> i32 {
    let m = parse_module(src).expect("parse unit");
    verify_module(&m).expect("verify unit");
    host.jit_compile(jit, &svm_encode::encode_module(&m))
        .expect("no trap")
        .expect("compile ok")
        .handle
}

/// Unit A: `add(a, b) = a + b` — installed at slot 1.
const UNIT_A: &str = r#"memory 16
func (i32, i32) -> (i32) {
block0(v0: i32, v1: i32):
  v2 = i32.add v0 v1
  return v2
}
"#;

/// Unit B (invoked): `call_indirect[slot 1](a, b) + 1` — reaches the installed A.
const UNIT_B: &str = r#"memory 16
func (i32, i32) -> (i32) {
block0(v0: i32, v1: i32):
  v2 = i32.const 1
  v3 = call_indirect (i32, i32) -> (i32) v2 (v0, v1)
  v4 = i32.const 1
  v5 = i32.add v3 v4
  return v5
}
"#;

/// `Jit.invoke` of a unit that itself `call_indirect`s an *installed* unit (DESIGN.md §22 new→new):
/// install A at slot 1, then invoke B which calls A; `B(6,7) = A(6,7) + 1 = 14`.
const GUEST_INVOKE: &str = r#"memory 16
func (i32, i32, i32, i32, i32) -> (i32) {
block0(v0: i32, v1: i32, v2: i32, v3: i32, v4: i32):
  v5 = i64.extend_i32_u v1
  v6 = cap.call 11 3 (i64) -> (i64) v0 (v5)
  v7 = i64.extend_i32_u v2
  v8 = cap.call 11 1 (i64, i32, i32) -> (i32) v0 (v7, v3, v4)
  return v8
}
"#;

fn invoke_setup(guest: &svm_ir::Module) -> (Host, i32, i32, i32) {
    let mut host = Host::new();
    let jit = grant_jit(&mut host, guest, 4);
    let code_a = compile_unit(&mut host, jit, UNIT_A);
    let code_b = compile_unit(&mut host, jit, UNIT_B);
    (host, jit, code_a, code_b)
}

#[test]
fn invoke_unit_that_calls_installed_unit_agrees() {
    let m = parse_module(GUEST_INVOKE).expect("parse guest");
    verify_module(&m).expect("verify guest");

    let (mut h_tw, jit, ca, cb) = invoke_setup(&m);
    let args = [
        Value::I32(jit),
        Value::I32(ca),
        Value::I32(cb),
        Value::I32(6),
        Value::I32(7),
    ];
    let mut f_tw = 50_000_000u64;
    let tw = run_with_host(&m, 0, &args, &mut f_tw, &mut h_tw);

    let (mut h_bc, jit2, ca2, cb2) = invoke_setup(&m);
    let args2 = [
        Value::I32(jit2),
        Value::I32(ca2),
        Value::I32(cb2),
        Value::I32(6),
        Value::I32(7),
    ];
    let mut f_bc = 50_000_000u64;
    let bc = bytecode::compile_and_run_with_host(&m, 0, &args2, &mut f_bc, &mut h_bc)
        .expect("bytecode engine must support Jit.invoke (Slice 1c-5e)");

    assert_eq!(tw, bc, "invoke: tree-walker != bytecode");
    assert_eq!(bc, Ok(vec![Value::I32(14)]), "B(6,7) = A(6,7) + 1 = 14");
}

#[test]
fn install_then_cross_module_call_indirect_agrees() {
    let m = parse_module(GUEST).expect("parse guest");
    verify_module(&m).expect("verify guest");

    let (mut h_tw, jit_tw, code_tw) = host_with_unit(&m);
    let mut f_tw = 50_000_000u64;
    let tw = run_with_host(
        &m,
        0,
        &[
            Value::I32(jit_tw),
            Value::I32(code_tw),
            Value::I32(6),
            Value::I32(7),
        ],
        &mut f_tw,
        &mut h_tw,
    );

    let (mut h_bc, jit_bc, code_bc) = host_with_unit(&m);
    let mut f_bc = 50_000_000u64;
    let bc = bytecode::compile_and_run_with_host(
        &m,
        0,
        &[
            Value::I32(jit_bc),
            Value::I32(code_bc),
            Value::I32(6),
            Value::I32(7),
        ],
        &mut f_bc,
        &mut h_bc,
    )
    .expect("bytecode engine must support install + cross-module call_indirect (Slice 1c-5e)");

    assert_eq!(tw, bc, "install/call_indirect: tree-walker != bytecode");
    assert_eq!(
        bc,
        Ok(vec![Value::I32(142)]),
        "service(6,7) = 6*7 + 100 = 142"
    );
}
