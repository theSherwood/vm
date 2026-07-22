//! Equality harness for the bytecode engine's **synchronous capability seam** (INTERP_PERF.md
//! Slice 1c-5a). The random module generator never emits `cap.call`, so — per the TDD plan for the
//! seam rewrite — this authors capability-using modules by hand and asserts the bytecode engine
//! (with a live powerbox) agrees bit-for-bit with the reference tree-walker `run_with_host`.
//!
//! It uses a **deterministic** host function (`cap_id::HOST_FN` = 13) so results are reproducible:
//! `f(op, args) = op*100 + sum(args)`. Each engine gets its own freshly-granted host with the same
//! closure, so the granted handle index matches.
//!
//! The `.expect(...)` on `compile_and_run_with_host`'s `Option` is the gate: if the bytecode engine
//! does not *support* a synchronous `cap.call` (returns `None` → would fall back), the test fails.

use svm_interp::{bytecode, run_with_host, Host, StreamRole, Trap, Value};
use svm_text::parse_module;

/// A fresh host granting one deterministic host-fn capability; returns the host and its handle.
fn host_with_det_fn() -> (Host, i32) {
    let mut h = Host::new();
    let handle = h.grant_host_fn(Box::new(|op: u32, args: &[i64], _mem| {
        Ok(vec![op as i64 * 100 + args.iter().sum::<i64>()])
    }));
    (h, handle)
}

/// Run `src`'s entry on both engines with equal fuel and equal powerboxes (the handle is passed as
/// the first argument, ahead of `extra_args`), and assert the results are identical. Asserts the
/// bytecode engine actually *supported* the module (did not fall back).
fn check_cap(src: &str, extra_args: &[Value]) {
    let m = parse_module(src).expect("parse");

    let (mut h_tw, handle_tw) = host_with_det_fn();
    let mut args_tw = vec![Value::I32(handle_tw)];
    args_tw.extend_from_slice(extra_args);
    let mut f_tw = 1_000_000u64;
    let tw = run_with_host(&m, 0, &args_tw, &mut f_tw, &mut h_tw);

    let (mut h_bc, handle_bc) = host_with_det_fn();
    let mut args_bc = vec![Value::I32(handle_bc)];
    args_bc.extend_from_slice(extra_args);
    let mut f_bc = 1_000_000u64;
    let bc = bytecode::compile_and_run_with_host(&m, 0, &args_bc, &mut f_bc, &mut h_bc)
        .expect("bytecode engine must support a synchronous cap.call module (Slice 1c-5a)");

    assert!(
        !matches!(tw, Err(Trap::OutOfFuel)) && !matches!(bc, Err(Trap::OutOfFuel)),
        "unexpected OutOfFuel\n tw={tw:?}\n bc={bc:?}\n{src}"
    );
    assert_eq!(tw, bc, "cap.call: tree-walker != bytecode\n{src}");
}

/// One call: `f(0, [a, b]) = a + b`.
const SUM_ARGS: &str = r#"
func (i32, i64, i64) -> (i64) {
block 0 (v0: i32, v1: i64, v2: i64) {
  v3 = cap.call 13 0 (i64, i64) -> (i64) v0 (v1, v2)
  return v3
  }
}
"#;

/// Non-zero op selector: `f(5, [a]) = 500 + a`.
const OP_SELECTOR: &str = r#"
func (i32, i64) -> (i64) {
block 0 (v0: i32, v1: i64) {
  v2 = cap.call 13 5 (i64) -> (i64) v0 (v1)
  return v2
  }
}
"#;

/// Two calls summed, plus host result feeding back into a second call's args (data dependence
/// through the powerbox): `r1 = f(0,[a]) = a; r2 = f(0,[r1,b]) = a + b; return r1 + r2`.
const CHAINED: &str = r#"
func (i32, i64, i64) -> (i64) {
block 0 (v0: i32, v1: i64, v2: i64) {
  v3 = cap.call 13 0 (i64) -> (i64) v0 (v1)
  v4 = cap.call 13 0 (i64, i64) -> (i64) v0 (v3, v2)
  v5 = i64.add v3 v4
  return v5
  }
}
"#;

/// A cap.call inside a loop, accumulating — exercises the seam across back-edges and with the
/// suspend/resume cursor. The handle and `n` are threaded through block params (block-local SSA).
/// Computes sum_{i=0}^{n-1} f(0,[i]) = sum_{i=0}^{n-1} i.
const LOOP_CALL: &str = r#"
func (i32, i64) -> (i64) {
block 0 (v0: i32, v1: i64) {
  v2 = i64.const 0
  v3 = i64.const 0
  br 1(v0, v1, v2, v3)
}
block 1 (v4: i32, v5: i64, v6: i64, v7: i64) {
  v8 = i64.lt_s v6 v5
  br_if v8 2(v4, v5, v6, v7) 3(v7)
}
block 2 (v9: i32, v10: i64, v11: i64, v12: i64) {
  v13 = cap.call 13 0 (i64) -> (i64) v9 (v11)
  v14 = i64.add v12 v13
  v15 = i64.const 1
  v16 = i64.add v11 v15
  br 1(v9, v10, v16, v14)
}
block 3 (v17: i64) {
  return v17
  }
}
"#;

#[test]
fn cap_sum_args() {
    check_cap(SUM_ARGS, &[Value::I64(11), Value::I64(31)]);
    check_cap(SUM_ARGS, &[Value::I64(-5), Value::I64(5)]);
}

#[test]
fn cap_op_selector() {
    check_cap(OP_SELECTOR, &[Value::I64(7)]);
}

#[test]
fn cap_chained() {
    check_cap(CHAINED, &[Value::I64(100), Value::I64(23)]);
}

#[test]
fn cap_in_loop() {
    check_cap(LOOP_CALL, &[Value::I64(10)]);
    check_cap(LOOP_CALL, &[Value::I64(0)]);
}

// ---- §7 reflection (cap.self.count / cap.self.get) --------------------------------------------

/// A fresh host with a deterministic 3-cap powerbox (stream-out, exit, host-fn), granted in order so
/// handle `i` is the `i`-th grant on every fresh host.
fn host_with_powerbox() -> Host {
    let mut h = Host::new();
    let _ = h.grant_stream(StreamRole::Out); // handle 0, type_id 0
    let _ = h.grant_exit(); // handle 1, type_id 1
    let _ = h.grant_host_fn(Box::new(|_o, _a, _m| Ok(vec![0]))); // handle 2, type_id 13
    h
}

fn check_self(src: &str, args: &[Value]) {
    let m = parse_module(src).expect("parse");

    let mut h_tw = host_with_powerbox();
    let mut f_tw = 1_000_000u64;
    let tw = run_with_host(&m, 0, args, &mut f_tw, &mut h_tw);

    let mut h_bc = host_with_powerbox();
    let mut f_bc = 1_000_000u64;
    let bc = bytecode::compile_and_run_with_host(&m, 0, args, &mut f_bc, &mut h_bc)
        .expect("bytecode engine must support cap.self.* (Slice 1c-5a)");

    assert_eq!(tw, bc, "cap.self.*: tree-walker != bytecode\n{src}");
}

const SELF_COUNT: &str = r#"
func () -> (i32) {
block 0 () {
  v0 = cap.self.count
  return v0
  }
}
"#;

/// `cap.self.get(i)` returns `(handle, type_id)`; sum them so the result depends on both.
const SELF_GET: &str = r#"
func (i32) -> (i32) {
block 0 (v0: i32) {
  v1, v2 = cap.self.get v0
  v3 = i32.add v1 v2
  return v3
  }
}
"#;

#[test]
fn cap_self_count() {
    check_self(SELF_COUNT, &[]);
}

#[test]
fn cap_self_get() {
    check_self(SELF_GET, &[Value::I32(0)]);
    check_self(SELF_GET, &[Value::I32(1)]);
    check_self(SELF_GET, &[Value::I32(2)]);
}

/// A forged/out-of-range handle must trap identically on both engines (inert `CapFault`), not
/// diverge — the authority check is the host's, reused unchanged.
#[test]
fn cap_forged_handle_traps_identically() {
    // Pass a bogus handle (99) instead of the granted one.
    let m = parse_module(SUM_ARGS).expect("parse");

    let (mut h_tw, _) = host_with_det_fn();
    let mut f_tw = 1_000_000u64;
    let tw = run_with_host(
        &m,
        0,
        &[Value::I32(99), Value::I64(1), Value::I64(2)],
        &mut f_tw,
        &mut h_tw,
    );

    let (mut h_bc, _) = host_with_det_fn();
    let mut f_bc = 1_000_000u64;
    let bc = bytecode::compile_and_run_with_host(
        &m,
        0,
        &[Value::I32(99), Value::I64(1), Value::I64(2)],
        &mut f_bc,
        &mut h_bc,
    )
    .expect("bytecode supports the module");

    assert_eq!(tw, bc, "forged handle must trap identically");
}
