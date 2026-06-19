//! §RT **reference types** — `funcref`/`externref` values, `ref.null`/`is_null`/`func`, typed
//! `select`, and the core mutable-table ops (`table.get`/`set`/`size`/`fill`). Both ref types are an
//! i32 index in SVM (funcref → §3c function index, externref → §7 capability handle), so this all
//! lowers to existing IR (loads/stores/const/select + the synthesized fill loop) — no new ops.
//!
//! Table mutation is observed **within a single invoke** (SVM persists the window across the calls in
//! one `run`, not across separate invokes), so each test sets then reads in one function.

use svm_interp::Value;

/// Transpile WAT → IR, verify, run `entry` on interp **and** JIT, assert they agree, return the i32.
fn eval(wat: &str, entry: &str) -> i32 {
    let wasm = wat::parse_str(wat).expect("assemble wat");
    let t = svm_wasm::transpile(&wasm).expect("transpile");
    svm_verify::verify_module(&t.module).expect("verify transpiled IR");
    let idx = t
        .exports
        .iter()
        .find(|(n, _)| n == entry)
        .unwrap_or_else(|| panic!("no export {entry}"))
        .1;
    let mut fuel = 100_000_000u64;
    let interp = svm_interp::run(&t.module, idx, &[], &mut fuel).expect("interp run");
    let jit = match svm_jit::compile_and_run(&t.module, idx, &[]).expect("jit") {
        svm_jit::JitOutcome::Returned(v) => v,
        other => panic!("jit did not return: {other:?}"),
    };
    let iv = match interp[0] {
        Value::I32(x) => x,
        other => panic!("expected i32, got {other:?}"),
    };
    assert_eq!(iv, jit[0] as i32, "interp != jit");
    iv
}

/// `ref.func` + `table.set` + `call_indirect` — the vtable pattern: install two functions into table
/// slots, then dispatch through them. Set-then-dispatch in one invoke (window persists).
#[test]
fn ref_func_table_set_dispatch() {
    let wat = r#"
    (module
      (table 4 funcref)
      (type $r (func (result i32)))
      (elem declare func $a $b)
      (func $a (result i32) (i32.const 11))
      (func $b (result i32) (i32.const 22))
      (func (export "f") (result i32)
        (table.set (i32.const 1) (ref.func $a))
        (table.set (i32.const 2) (ref.func $b))
        (i32.add
          (call_indirect (type $r) (i32.const 1))
          (call_indirect (type $r) (i32.const 2)))))
    "#;
    assert_eq!(eval(wat, "f"), 33);
}

/// `table.get` round-trips a stored funcref index; `table.size` is the declared size.
#[test]
fn table_get_and_size() {
    let wat = r#"
    (module
      (table 7 funcref)
      (type $r (func (result i32)))
      (elem declare func $g)
      (func $g (result i32) (i32.const 99))
      (func (export "get") (result i32)
        (table.set (i32.const 5) (ref.func $g))
        ;; copy slot 5 → slot 0 *through* table.get/set, then dispatch slot 0.
        (table.set (i32.const 0) (table.get (i32.const 5)))
        (call_indirect (type $r) (i32.const 0)))
      (func (export "size") (result i32) (table.size)))
    "#;
    assert_eq!(eval(wat, "get"), 99);
    assert_eq!(eval(wat, "size"), 7);
}

/// `ref.null` + `ref.is_null`: a null funcref is null; a real one isn't.
#[test]
fn ref_null_and_is_null() {
    let wat = r#"
    (module
      (elem declare func $x)
      (func $x)
      (func (export "f") (result i32)
        ;; is_null(null) * 10 + is_null(ref.func) = 10 + 0 = 10
        (i32.add
          (i32.mul (ref.is_null (ref.null func)) (i32.const 10))
          (ref.is_null (ref.func $x)))))
    "#;
    assert_eq!(eval(wat, "f"), 10);
}

/// `table.fill`: fill a range of slots with one funcref, then dispatch through one of them.
#[test]
fn table_fill_range() {
    let wat = r#"
    (module
      (table 8 funcref)
      (type $r (func (result i32)))
      (elem declare func $c)
      (func $c (result i32) (i32.const 7))
      (func (export "f") (result i32)
        (table.fill (i32.const 2) (ref.func $c) (i32.const 4)) ;; slots 2..6 = $c
        (i32.add
          (call_indirect (type $r) (i32.const 2))
          (call_indirect (type $r) (i32.const 5)))))            ;; 7 + 7 = 14
    "#;
    assert_eq!(eval(wat, "f"), 14);
}

/// Typed `select (result funcref)` — picks between two funcrefs by a condition, then dispatches.
/// (The harness runs no-arg exports, so the condition is baked into two module variants.)
#[test]
fn typed_select_funcref() {
    let pick_a = r#"
    (module (table 1 funcref) (type $r (func (result i32)))
      (elem declare func $a $b)
      (func $a (result i32) (i32.const 1)) (func $b (result i32) (i32.const 2))
      (func (export "f") (result i32)
        (table.set (i32.const 0)
          (select (result funcref) (ref.func $a) (ref.func $b) (i32.const 1)))
        (call_indirect (type $r) (i32.const 0))))"#;
    let pick_b = r#"
    (module (table 1 funcref) (type $r (func (result i32)))
      (elem declare func $a $b)
      (func $a (result i32) (i32.const 1)) (func $b (result i32) (i32.const 2))
      (func (export "f") (result i32)
        (table.set (i32.const 0)
          (select (result funcref) (ref.func $a) (ref.func $b) (i32.const 0)))
        (call_indirect (type $r) (i32.const 0))))"#;
    assert_eq!(eval(pick_a, "f"), 1, "cond=1 selects a");
    assert_eq!(eval(pick_b, "f"), 2, "cond=0 selects b");
}

/// `externref` is an opaque i32 handle that flows through values unchanged — a function taking and
/// returning an externref is the identity on the bits (here via a local round-trip).
#[test]
fn externref_passthrough() {
    let wat = r#"
    (module
      (func (export "f") (result i32)
        (local $r externref)
        ;; stash a "handle" (null here) and read it back; is_null proves the round-trip.
        (local.set $r (ref.null extern))
        (ref.is_null (local.get $r))))
    "#;
    assert_eq!(eval(wat, "f"), 1);
}
