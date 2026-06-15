//! The wasm **start section** (`(start $f)`): the start function runs once at instantiation, before
//! any export. SVM has no instantiation hook (a run calls one entry over a fresh window), so the
//! transpiler remaps each export to a wrapper that calls `start` then the real export — internal
//! `call`s reach the export directly, so `start` runs exactly once before the chosen entry.
//!
//! Previously the start section was *silently ignored* (the default section arm), so a start
//! function never ran — a silent miscompile. These pin that it now runs (and runs only once).

use svm_interp::Value;

/// Transpile WAT → IR, verify, run `entry(args)` on interp **and** JIT, assert they agree, return i64.
fn run(wat: &str, entry: &str, args: &[Value]) -> i64 {
    let wasm = wat::parse_str(wat).expect("assemble wat");
    let t = svm_wasm::transpile(&wasm).expect("transpile");
    svm_verify::verify_module(&t.module).expect("verify transpiled IR");
    let idx = t
        .exports
        .iter()
        .find(|(n, _)| n == entry)
        .unwrap_or_else(|| panic!("no export {entry}"))
        .1;
    let mut fuel = 10_000_000u64;
    let interp = svm_interp::run(&t.module, idx, args, &mut fuel).expect("interp run");
    let jit_args: Vec<i64> = args
        .iter()
        .map(|v| match v {
            Value::I32(x) => *x as i64,
            Value::I64(x) => *x,
            other => panic!("unsupported arg {other:?}"),
        })
        .collect();
    let jit = match svm_jit::compile_and_run(&t.module, idx, &jit_args).expect("jit") {
        svm_jit::JitOutcome::Returned(v) => v,
        other => panic!("jit did not return: {other:?}"),
    };
    let iv = match interp[0] {
        Value::I32(x) => x as i64,
        Value::I64(x) => x,
        other => panic!("unexpected interp value {other:?}"),
    };
    assert_eq!(iv, jit[0], "interp != jit");
    iv
}

/// The start function writes a sentinel to memory that the export reads back — so a non-zero result
/// proves `start` ran (it was silently skipped before).
#[test]
fn start_runs_before_export() {
    let wat = r#"
      (module
        (memory 1)
        (func $init (i32.store (i32.const 0) (i32.const 42)))
        (func (export "get") (result i32) (i32.load (i32.const 0)))
        (start $init))"#;
    assert_eq!(run(wat, "get", &[]), 42, "start must run before the export");
}

/// The wrapper threads the export's params and results correctly: `add(x) = x + mem[0]`, with
/// `mem[0] = 100` set by `start` → `add(5) = 105`.
#[test]
fn start_wrapper_threads_params_and_results() {
    let wat = r#"
      (module
        (memory 1)
        (func $init (i32.store (i32.const 0) (i32.const 100)))
        (func (export "add") (param $x i32) (result i32)
          (i32.add (local.get $x) (i32.load (i32.const 0))))
        (start $init))"#;
    assert_eq!(run(wat, "add", &[Value::I32(5)]), 105);
}

/// `start` runs **exactly once**, and internal `call`s bypass the wrapper. `start` increments a
/// counter at mem[0]; `run` calls an (also-exported) `helper` *internally* and returns the counter.
/// If internal calls hit the wrapper, `start` would run twice (→ 2); reaching the real `helper`
/// directly keeps it at 1.
#[test]
fn start_runs_once_internal_calls_bypass_wrapper() {
    let wat = r#"
      (module
        (memory 1)
        (func $init
          (i32.store (i32.const 0) (i32.add (i32.load (i32.const 0)) (i32.const 1))))
        (func $helper (result i32) (i32.load (i32.const 0)))
        (func (export "run") (result i32) (call $helper))
        (export "helper" (func $helper))
        (start $init))"#;
    assert_eq!(
        run(wat, "run", &[]),
        1,
        "start runs once; internal call bypasses the wrapper"
    );
}

/// A start function with a non-`() -> ()` signature is a clean error (not a panic / miscompile).
#[test]
fn start_with_bad_signature_rejected() {
    // `$s` takes a param, so it can't be a valid start function.
    let wat = r#"(module (func $s (param i32)) (start $s))"#;
    let wasm = match wat::parse_str(wat) {
        Ok(w) => w,
        Err(_) => return, // some assemblers reject this at parse time — that's also fine
    };
    match svm_wasm::transpile(&wasm) {
        Err(svm_wasm::Error::Unsupported(_)) => {}
        Err(e) => panic!("expected Unsupported, got {e:?}"),
        Ok(_) => panic!("expected a bad-start-signature error"),
    }
}
