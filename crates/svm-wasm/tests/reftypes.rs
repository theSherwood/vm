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

/// `table.copy`: install two funcs, copy their slots elsewhere (overlap-safe memmove), dispatch the
/// copies.
#[test]
fn table_copy_slots() {
    let wat = r#"
    (module
      (table 8 funcref)
      (type $r (func (result i32)))
      (elem declare func $a $b)
      (func $a (result i32) (i32.const 3))
      (func $b (result i32) (i32.const 4))
      (func (export "f") (result i32)
        (table.set (i32.const 4) (ref.func $a))
        (table.set (i32.const 5) (ref.func $b))
        (table.copy (i32.const 0) (i32.const 4) (i32.const 2)) ;; slots 4..6 → 0..2
        (i32.add
          (call_indirect (type $r) (i32.const 0))
          (call_indirect (type $r) (i32.const 1)))))            ;; 3 + 4 = 7
    "#;
    assert_eq!(eval(wat, "f"), 7);
}

/// `table.copy` with **overlapping** ranges (`dest > src`) — the memmove path must read the whole
/// source before overwriting, so a forward byte-copy that clobbers slot `src+1` before reading it
/// would corrupt the result. Copy slots 0..3 → 1..4, then dispatch the (shifted-up) copies.
#[test]
fn table_copy_overlapping_slots() {
    let wat = r#"
    (module
      (table 8 funcref)
      (type $r (func (result i32)))
      (elem declare func $a $b $c)
      (func $a (result i32) (i32.const 3))
      (func $b (result i32) (i32.const 4))
      (func $c (result i32) (i32.const 5))
      (func (export "f") (result i32)
        (table.set (i32.const 0) (ref.func $a))
        (table.set (i32.const 1) (ref.func $b))
        (table.set (i32.const 2) (ref.func $c))
        (table.copy (i32.const 1) (i32.const 0) (i32.const 3)) ;; slots 0..3 → 1..4 (overlap, dest>src)
        (i32.add
          (call_indirect (type $r) (i32.const 1))              ;; $a → 3
          (i32.add
            (call_indirect (type $r) (i32.const 2))            ;; $b → 4
            (call_indirect (type $r) (i32.const 3))))))        ;; $c → 5, total 12
    "#;
    assert_eq!(eval(wat, "f"), 12);
}

/// `table.init` from a **passive** element segment: copy its funcref entries into the table, dispatch.
#[test]
fn table_init_from_passive_elem() {
    let wat = r#"
    (module
      (table 8 funcref)
      (type $r (func (result i32)))
      (elem $e func $a $b)            ;; passive segment (no table/offset)
      (func $a (result i32) (i32.const 5))
      (func $b (result i32) (i32.const 6))
      (func (export "f") (result i32)
        (table.init $e (i32.const 2) (i32.const 0) (i32.const 2)) ;; elem[0..2] → table[2..4]
        (elem.drop $e)                                            ;; no-op
        (i32.add
          (call_indirect (type $r) (i32.const 2))
          (call_indirect (type $r) (i32.const 3)))))              ;; 5 + 6 = 11
    "#;
    assert_eq!(eval(wat, "f"), 11);
}

/// `table.grow`: grow a table from 2→6 slots (filling the new slots with a funcref), confirm the new
/// `table.size`, and dispatch through a grown slot. The pre-grow size is the grow result.
#[test]
fn table_grow_and_use() {
    let wat = r#"
    (module
      (table 2 10 funcref)            ;; initial 2, max 10
      (type $r (func (result i32)))
      (elem declare func $g)
      (func $g (result i32) (i32.const 42))
      (func (export "grow") (result i32)
        (table.grow (ref.func $g) (i32.const 4)))   ;; returns old size = 2
      (func (export "size_after") (result i32)
        (drop (table.grow (ref.func $g) (i32.const 4)))
        (table.size))                                ;; 2 + 4 = 6
      (func (export "use_grown") (result i32)
        (drop (table.grow (ref.func $g) (i32.const 4))) ;; slots 2..6 = $g
        (call_indirect (type $r) (i32.const 5))))       ;; dispatch a grown slot → 42
    "#;
    assert_eq!(eval(wat, "grow"), 2);
    assert_eq!(eval(wat, "size_after"), 6);
    assert_eq!(eval(wat, "use_grown"), 42);
}

/// `table.grow` past the declared maximum fails (returns `-1`) and leaves the size unchanged.
#[test]
fn table_grow_over_max_fails() {
    let wat = r#"
    (module
      (table 2 3 funcref)             ;; max 3, so growing by 4 must fail
      (elem declare func $g)
      (func $g)
      (func (export "f") (result i32)
        ;; failed grow returns -1; size stays 2. encode as grow_result*100 + size.
        (i32.add
          (i32.mul (table.grow (ref.func $g) (i32.const 4)) (i32.const 100))
          (table.size))))
    "#;
    // -1 * 100 + 2 = -98
    assert_eq!(eval(wat, "f"), -98);
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
