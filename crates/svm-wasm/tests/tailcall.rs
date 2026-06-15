//! The wasm **tail-call proposal** (`return_call` / `return_call_indirect`): a block-terminating call
//! where the callee replaces the current frame. svm-wasm lowers them to the IR's `Terminator::
//! ReturnCall`/`ReturnCallIndirect`, which both backends execute as **true** tail calls (the interp
//! replaces the frame in place; the JIT emits Cranelift `return_call`/`return_call_indirect`). The
//! deep-recursion test would overflow a non-tail call on either backend, so it pins real tail-ness.

use svm_interp::Value;

/// Transpile WAT → IR, verify, run `entry(args)` on interp **and** JIT (generous fuel for the deep
/// recursion), assert they agree, return the i64 result.
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
    let mut fuel = 2_000_000_000u64;
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

/// `return_call` is a **true** tail call: `sum(n, acc)` tail-recurses `n` times. With `n = 200_000`
/// — far beyond any call-depth bound / native stack — it completes only because the callee replaces
/// the frame (a non-tail call would `StackOverflow` on the interp / smash the stack on the JIT).
/// Result = `acc0 + Σ_{1..=n} = n(n+1)/2`.
#[test]
fn return_call_is_a_true_tail_call() {
    let wat = r#"
      (module
        (func $sum (param $n i64) (param $acc i64) (result i64)
          (if (result i64) (i64.eqz (local.get $n))
            (then (local.get $acc))
            (else (return_call $sum
                     (i64.sub (local.get $n) (i64.const 1))
                     (i64.add (local.get $acc) (local.get $n))))))
        (func (export "run") (param $n i64) (result i64)
          (call $sum (local.get $n) (i64.const 0))))"#;
    let n: i64 = 200_000;
    assert_eq!(run(wat, "run", &[Value::I64(n)]), n * (n + 1) / 2);
}

/// `return_call_indirect`: a tail call through the function table (the §3c masked + type-checked
/// dispatch), selecting `$a` (x+10) or `$b` (x*3) by index.
#[test]
fn return_call_indirect_dispatches() {
    let wat = r#"
      (module
        (type $t (func (param i64) (result i64)))
        (table 2 funcref)
        (elem (i32.const 0) $a $b)
        (func $a (param i64) (result i64) (i64.add (local.get 0) (i64.const 10)))
        (func $b (param i64) (result i64) (i64.mul (local.get 0) (i64.const 3)))
        (func (export "run") (param $sel i32) (param $x i64) (result i64)
          (return_call_indirect (type $t) (local.get $x) (local.get $sel))))"#;
    assert_eq!(
        run(wat, "run", &[Value::I32(0), Value::I64(7)]),
        17,
        "$a: 7+10"
    );
    assert_eq!(
        run(wat, "run", &[Value::I32(1), Value::I64(7)]),
        21,
        "$b: 7*3"
    );
}

/// Mutually-recursive tail calls (even/odd) — a tail call into a *different* function each step, the
/// classic case stackless rewrites can't express but tail calls handle in O(1) frames.
#[test]
fn mutual_tail_recursion() {
    let wat = r#"
      (module
        (func $even (param $n i64) (result i64)
          (if (result i64) (i64.eqz (local.get $n))
            (then (i64.const 1))
            (else (return_call $odd (i64.sub (local.get $n) (i64.const 1))))))
        (func $odd (param $n i64) (result i64)
          (if (result i64) (i64.eqz (local.get $n))
            (then (i64.const 0))
            (else (return_call $even (i64.sub (local.get $n) (i64.const 1))))))
        (func (export "is_even") (param $n i64) (result i64)
          (call $even (local.get $n))))"#;
    assert_eq!(run(wat, "is_even", &[Value::I64(123_456)]), 1, "even");
    assert_eq!(run(wat, "is_even", &[Value::I64(123_457)]), 0, "odd");
}
