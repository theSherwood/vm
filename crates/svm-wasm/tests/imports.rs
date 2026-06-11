//! Differential tests for **function imports / the host ABI** (the wasm `call` → SVM `cap.call`
//! lowering). A wasm import binds to a capability by the convention `module` = decimal `type_id`,
//! `name` = decimal `op`; the transpiler threads one capability handle as the leading param of every
//! function, and the embedder grants the matching capability and passes its handle as the entry's
//! leading argument. These tests run a transpiled import-using module on **both** backends under one
//! reference `Host`, asserting they agree — and against a hand-computed oracle.
//!
//! Unlike `transpile.rs`'s capability-free `run`/`eval`, these need a powerbox: the interpreter via
//! `run_with_host`, the JIT via `compile_and_run_with_host` over the production `svm_run::cap_thunk`.

use std::ffi::c_void;
use svm_interp::{run_with_host, Host, Value};
use svm_jit::{compile_and_run_with_host, JitOutcome};

/// Transpile WAT importing one capability, verify, then run export `entry` on interp + JIT under a
/// `Host` the same `grant` populates on each (so the handle encoding matches). The granted handle is
/// passed as the leading argument (the threaded capability handle), followed by `extra_args`. Asserts
/// the single i64/i32 result agrees across backends and returns it.
fn run_import(
    wat: &str,
    entry: &str,
    grant: impl Fn(&mut Host) -> i32,
    extra_args: &[Value],
) -> i64 {
    let wasm = wat::parse_str(wat).expect("assemble wat");
    let t = svm_wasm::transpile(&wasm).expect("transpile");
    svm_verify::verify_module(&t.module).expect("verify transpiled IR");
    let idx = t
        .exports
        .iter()
        .find(|(n, _)| n == entry)
        .unwrap_or_else(|| panic!("no export {entry}"))
        .1;

    // Interpreter: grant the cap, pass its handle (i32) as the entry's leading arg.
    let mut hi = Host::new();
    let h = grant(&mut hi);
    let mut iargs: Vec<Value> = vec![Value::I32(h)];
    iargs.extend_from_slice(extra_args);
    let mut fuel = 100_000_000u64;
    let interp = run_with_host(&t.module, idx, &iargs, &mut fuel, &mut hi).expect("interp run");

    // JIT: the same grant (so the handle value matches), driven through the production cap thunk.
    let mut hj = Host::new();
    let hj_handle = grant(&mut hj);
    assert_eq!(h, hj_handle, "handle encoding must match across hosts");
    let mut slots: Vec<i64> = vec![h as i64];
    slots.extend(extra_args.iter().map(|v| match v {
        Value::I32(x) => *x as i64,
        Value::I64(x) => *x,
        other => panic!("unsupported arg {other:?}"),
    }));
    let jit = match compile_and_run_with_host(
        &t.module,
        idx,
        &slots,
        svm_run::cap_thunk,
        &mut hj as *mut Host as *mut c_void,
    )
    .expect("jit compile")
    {
        JitOutcome::Returned(v) => v,
        other => panic!("jit did not return: {other:?}"),
    };

    let iv = match interp[0] {
        Value::I64(x) => x,
        Value::I32(x) => x as i64,
        other => panic!("unexpected interp value {other:?}"),
    };
    let jv = match t.module.funcs[idx as usize].results[0] {
        svm_ir::ValType::I32 => jit[0] as u32 as i64 & 0xFFFF_FFFF,
        _ => jit[0],
    };
    let iv_cmp = match t.module.funcs[idx as usize].results[0] {
        svm_ir::ValType::I32 => iv as u32 as i64 & 0xFFFF_FFFF,
        _ => iv,
    };
    assert_eq!(iv_cmp, jv, "interp != jit (result {iv} vs {})", jit[0]);
    iv
}

/// The same deterministic transform the `Blocking` capability's `work(arg)` applies (op 0). Mirrors
/// `svm_interp::Host::mix` so the test can compute the oracle independently.
fn mix(arg: i64) -> i64 {
    arg.wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407)
}

/// A `call` to an imported **no-arg** capability op (Clock.now — type_id 2, op 0, `() -> i64`):
/// loop-call it `N` times and sum. The reference clock is deterministic (0, 1, 2, …), so the sum is
/// `0+1+…+(N-1)` on both backends.
#[test]
fn import_clock_now_loop_sum() {
    let wat = r#"
(module
  (import "2" "0" (func $now (result i64)))
  (func (export "run") (param $n i64) (result i64)
    (local $acc i64) (local $i i64)
    (block $done
      (loop $loop
        (br_if $done (i64.ge_s (local.get $i) (local.get $n)))
        (local.set $acc (i64.add (local.get $acc) (call $now)))
        (local.set $i (i64.add (local.get $i) (i64.const 1)))
        (br $loop)))
    (local.get $acc)))
"#;
    let n = 12i64;
    let got = run_import(wat, "run", |h| h.grant_clock(), &[Value::I64(n)]);
    assert_eq!(got, n * (n - 1) / 2);
}

/// A `call` to an imported op **with a scalar arg + result** (Blocking.work — type_id 10, op 0,
/// `(i64) -> i64`, a pure deterministic `mix`): loop-call `work(i)` and sum. Exercises argument
/// marshalling through the `cap.call` (the `hostcall` bench shape) on both backends.
#[test]
fn import_blocking_work_sum() {
    let wat = r#"
(module
  (import "10" "0" (func $work (param i64) (result i64)))
  (func (export "run") (param $n i64) (result i64)
    (local $acc i64) (local $i i64)
    (block $done
      (loop $loop
        (br_if $done (i64.ge_s (local.get $i) (local.get $n)))
        (local.set $acc (i64.add (local.get $acc) (call $work (local.get $i))))
        (local.set $i (i64.add (local.get $i) (i64.const 1)))
        (br $loop)))
    (local.get $acc)))
"#;
    let n = 10i64;
    let got = run_import(
        wat,
        "run",
        |h| h.grant_blocking(std::time::Duration::ZERO, None),
        &[Value::I64(n)],
    );
    let want: i64 = (0..n).map(mix).fold(0i64, |a, x| a.wrapping_add(x));
    assert_eq!(got, want);
}

/// The import handle is threaded across **calls between defined functions**: `run` calls a helper
/// that itself calls the imported op. Proves a non-entry function still reaches the capability (the
/// leading handle param is forwarded, not just held by the entry).
#[test]
fn import_handle_threads_through_defined_call() {
    let wat = r#"
(module
  (import "10" "0" (func $work (param i64) (result i64)))
  (func $helper (param $x i64) (result i64)
    (i64.add (call $work (local.get $x)) (call $work (i64.add (local.get $x) (i64.const 1)))))
  (func (export "run") (param $a i64) (result i64)
    (call $helper (local.get $a))))
"#;
    let a = 7i64;
    let got = run_import(
        wat,
        "run",
        |h| h.grant_blocking(std::time::Duration::ZERO, None),
        &[Value::I64(a)],
    );
    assert_eq!(got, mix(a).wrapping_add(mix(a + 1)));
}

/// `call_indirect` through the table still works when the module has imports: the threaded handle is
/// prepended to the indirect signature + args, so a table-dispatched defined function reaches the
/// capability. A 2-entry dispatch picks `work(arg)` vs `work(arg)+1` by index.
#[test]
fn import_handle_threads_through_call_indirect() {
    let wat = r#"
(module
  (import "10" "0" (func $work (param i64) (result i64)))
  (type $unary (func (param i64) (result i64)))
  (table 2 funcref)
  (elem (i32.const 0) $a $b)
  (func $a (param $x i64) (result i64) (call $work (local.get $x)))
  (func $b (param $x i64) (result i64) (i64.add (call $work (local.get $x)) (i64.const 1)))
  (func (export "run") (param $sel i32) (param $x i64) (result i64)
    (call_indirect (type $unary) (local.get $x) (local.get $sel))))
"#;
    let got_a = run_import(
        wat,
        "run",
        |h| h.grant_blocking(std::time::Duration::ZERO, None),
        &[Value::I32(0), Value::I64(5)],
    );
    assert_eq!(got_a, mix(5));
    let got_b = run_import(
        wat,
        "run",
        |h| h.grant_blocking(std::time::Duration::ZERO, None),
        &[Value::I32(1), Value::I64(5)],
    );
    assert_eq!(got_b, mix(5).wrapping_add(1));
}

// ---- the clean-error surface (the convention's guard rails) ----

fn err(wat: &str) -> svm_wasm::Error {
    let wasm = wat::parse_str(wat).expect("assemble wat");
    match svm_wasm::transpile(&wasm) {
        Ok(_) => panic!("expected an Unsupported error, got Ok"),
        Err(e) => e,
    }
}

/// A non-numeric import name is a clean error (caught, not silently mis-bound) — the convention needs
/// `module` = decimal type_id, `name` = decimal op.
#[test]
fn import_non_numeric_name_is_clean_error() {
    let e = err(
        r#"(module (import "env" "host_fn" (func (result i64))) (func (export "f") (result i64) (call 0)))"#,
    );
    assert!(matches!(e, svm_wasm::Error::Unsupported(_)), "{e:?}");
}

/// Imports spanning two distinct capability interfaces (type_ids) is unsupported in v1 (one handle is
/// threaded) — a clean error, not a wrong binding.
#[test]
fn import_multiple_interfaces_is_clean_error() {
    let e = err(r#"
(module
  (import "2" "0" (func $now (result i64)))
  (import "10" "0" (func $work (param i64) (result i64)))
  (func (export "f") (result i64) (call $now)))
"#);
    assert!(matches!(e, svm_wasm::Error::Unsupported(_)), "{e:?}");
}

/// A non-function import (here a memory) is unsupported — only function imports map to `cap.call`.
#[test]
fn import_memory_is_clean_error() {
    let e = err(
        r#"(module (import "env" "mem" (memory 1)) (func (export "f") (result i32) (i32.const 0)))"#,
    );
    assert!(matches!(e, svm_wasm::Error::Unsupported(_)), "{e:?}");
}
