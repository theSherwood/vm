//! §12 wasm threads — **atomics** (slice 1): the full-width `*.atomic.*` ops transpile to SVM's IR
//! atomics and execute **identically on interp and JIT**. These are single-threaded op-correctness
//! tests — they pin the lowering (operand-stack order, memarg-offset folding, i32/i64 widths, the
//! rmw "yields the *old* value" contract, the wait/notify status codes) without spawning, since
//! genuine multi-threading needs the spawn convention (slice 2). The memories are declared `shared`
//! (the threads-proposal flag the transpiler now accepts).
//!
//! Narrow atomics (`*.atomic.rmw8`/`load16_u`/…) have no IR form and are a clean `Unsupported` — the
//! `narrow_atomics_rejected` test pins that decision.

use svm_interp::Value;

/// Transpile WAT → IR, verify, run `entry(args)` on interp **and** JIT, assert they agree, return the
/// i64 result. (Mirrors `transpile.rs::run`.)
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
    let results = &t.module.funcs[idx as usize].results;
    let mut fuel = 100_000_000u64;
    let interp = svm_interp::run(&t.module, idx, args, &mut fuel).expect("interp run");
    let jit_args: Vec<i64> = args
        .iter()
        .map(|v| match v {
            Value::I32(x) => *x as i64,
            Value::I64(x) => *x,
            other => panic!("unsupported arg {other:?}"),
        })
        .collect();
    let jit = match svm_jit::compile_and_run(&t.module, idx, &jit_args).expect("jit compile") {
        svm_jit::JitOutcome::Returned(v) => v,
        other => panic!("jit did not return: {other:?}"),
    };
    assert_eq!(jit.len(), interp.len(), "result count");
    for (i, rt) in results.iter().enumerate() {
        let (a, b) = match (rt, interp[i]) {
            (svm_ir::ValType::I32, Value::I32(x)) => (x as u32 as u64, jit[i] as u32 as u64),
            (svm_ir::ValType::I64, Value::I64(x)) => (x as u64, jit[i] as u64),
            _ => panic!("result type / value mismatch at {i}"),
        };
        assert_eq!(a, b, "interp != jit at result {i}");
    }
    match interp[0] {
        Value::I64(x) => x,
        Value::I32(x) => x as i64,
        other => panic!("unexpected interp value {other:?}"),
    }
}

/// Every i32 RMW yields the **old** value and leaves the right value in memory. We thread the result
/// of each op into the next so a single returned i32 witnesses the whole chain (and a divergence in
/// any one op between backends fails the in-`run` differential).
#[test]
fn rmw_i32_yields_old_and_updates() {
    // mem[0]=100; add 5 (old 100, →105); sub 3 (old 105, →102); xchg 7 (old 102, →7);
    // and 6 (old 7, →6); or 1 (old 6, →7); xor 5 (old 7, →2). Sum of olds = 100+105+102+7+6+7 = 327,
    // final mem = 2. Return final*1000 + (sum_olds & 0x3FF) is overkill; instead return final value
    // and separately assert one old via a focused export.
    let wat = r#"
      (module
        (memory 1 1 shared)
        (func (export "final") (result i32)
          (i32.atomic.store (i32.const 0) (i32.const 100))
          (drop (i32.atomic.rmw.add  (i32.const 0) (i32.const 5)))
          (drop (i32.atomic.rmw.sub  (i32.const 0) (i32.const 3)))
          (drop (i32.atomic.rmw.xchg (i32.const 0) (i32.const 7)))
          (drop (i32.atomic.rmw.and  (i32.const 0) (i32.const 6)))
          (drop (i32.atomic.rmw.or   (i32.const 0) (i32.const 1)))
          (drop (i32.atomic.rmw.xor  (i32.const 0) (i32.const 5)))
          (i32.atomic.load (i32.const 0)))
        (func (export "old_add") (result i32)
          (i32.atomic.store (i32.const 0) (i32.const 100))
          (i32.atomic.rmw.add (i32.const 0) (i32.const 5)))    ;; old = 100
      )"#;
    assert_eq!(run(wat, "final", &[]), 2, "and/or/xor chain → 2");
    assert_eq!(run(wat, "old_add", &[]), 100, "rmw yields the old value");
}

/// i64 RMW + the i64 width (8-byte atomic, not aliasing an adjacent slot).
#[test]
fn rmw_i64_full_width() {
    let wat = r#"
      (module
        (memory 1 1 shared)
        (func (export "f") (result i64)
          (i64.atomic.store (i32.const 8) (i64.const 1000000000000))
          (drop (i64.atomic.rmw.add (i32.const 8) (i64.const 23)))
          (i64.atomic.load (i32.const 8))))"#;
    assert_eq!(run(wat, "f", &[]), 1_000_000_000_023);
}

/// `cmpxchg`: a matching expected swaps and yields the old; a non-matching expected leaves memory and
/// still yields the old. Returns `old_match*10 + (final != replacement_on_mismatch ? 1 : 0)`-ish — we
/// keep it simple: two exports.
#[test]
fn cmpxchg_match_and_mismatch() {
    let wat = r#"
      (module
        (memory 1 1 shared)
        ;; expected matches → swaps to 77, yields old 42
        (func (export "hit") (result i32)
          (i32.atomic.store (i32.const 0) (i32.const 42))
          (drop (i32.atomic.rmw.cmpxchg (i32.const 0) (i32.const 42) (i32.const 77)))
          (i32.atomic.load (i32.const 0)))                       ;; → 77
        ;; expected mismatches → no swap, mem stays 42
        (func (export "miss") (result i32)
          (i32.atomic.store (i32.const 0) (i32.const 42))
          (drop (i32.atomic.rmw.cmpxchg (i32.const 0) (i32.const 9) (i32.const 77)))
          (i32.atomic.load (i32.const 0)))                       ;; → 42
        ;; the yielded old is always the pre-op value
        (func (export "old") (result i32)
          (i32.atomic.store (i32.const 0) (i32.const 42))
          (i32.atomic.rmw.cmpxchg (i32.const 0) (i32.const 9) (i32.const 77)))  ;; → 42
      )"#;
    assert_eq!(run(wat, "hit", &[]), 77, "matching cmpxchg swaps");
    assert_eq!(
        run(wat, "miss", &[]),
        42,
        "mismatching cmpxchg leaves memory"
    );
    assert_eq!(run(wat, "old", &[]), 42, "cmpxchg yields the old value");
}

/// memarg `offset` is folded into the effective address (atomics carry it like a plain load/store).
#[test]
fn atomic_offset_folding() {
    let wat = r#"
      (module
        (memory 1 1 shared)
        (func (export "f") (result i32)
          (i32.atomic.store offset=16 (i32.const 0) (i32.const 555))  ;; writes mem[16]
          (i32.atomic.load (i32.const 16)))                          ;; reads mem[16] directly
      )"#;
    assert_eq!(run(wat, "f", &[]), 555);
}

/// `wait` on a **non-matching** value returns status 1 (not-equal, never blocks); `notify` with no
/// waiters returns 0. Single-threaded, so only the non-blocking paths — the blocking wake is slice 2.
#[test]
fn wait_not_equal_and_notify_zero() {
    let wat = r#"
      (module
        (memory 1 1 shared)
        ;; mem[0]=1; wait expecting 0 → value differs → status 1 (no block)
        (func (export "wait_neq") (result i32)
          (i32.atomic.store (i32.const 0) (i32.const 1))
          (memory.atomic.wait32 (i32.const 0) (i32.const 0) (i64.const -1)))   ;; → 1
        ;; notify 0 waiters → 0 woken
        (func (export "notify0") (result i32)
          (memory.atomic.notify (i32.const 0) (i32.const 10)))                ;; → 0
      )"#;
    assert_eq!(
        run(wat, "wait_neq", &[]),
        1,
        "wait on a differing value → 1"
    );
    assert_eq!(run(wat, "notify0", &[]), 0, "notify with no waiters → 0");
}

/// A standalone `atomic.fence` is a no-op functionally (lowered to the IR fence: honoured by interp,
/// a real seq-cst barrier on the JIT) — it must not change the surrounding computation.
#[test]
fn fence_is_transparent() {
    let wat = r#"
      (module
        (memory 1 1 shared)
        (func (export "f") (result i32)
          (i32.atomic.store (i32.const 0) (i32.const 7))
          (atomic.fence)
          (drop (i32.atomic.rmw.add (i32.const 0) (i32.const 35)))
          (atomic.fence)
          (i32.atomic.load (i32.const 0))))"#;
    assert_eq!(run(wat, "f", &[]), 42);
}

/// Narrow atomics have no IR form (SVM atomics are 32/64-bit only) — they're a clean `Unsupported`,
/// not a miscompile. Pins the "reject for now" decision (CAS-loop emulation is a possible follow-up).
#[test]
fn narrow_atomics_rejected() {
    let wat = r#"
      (module
        (memory 1 1 shared)
        (func (export "f") (result i32)
          (i32.atomic.rmw8.add_u (i32.const 0) (i32.const 1))))"#;
    let wasm = wat::parse_str(wat).expect("assemble wat");
    match svm_wasm::transpile(&wasm) {
        Err(svm_wasm::Error::Unsupported(_)) => {}
        Err(e) => panic!("expected narrow atomic Unsupported, got a different error: {e:?}"),
        Ok(_) => panic!("expected narrow atomic to be Unsupported, but it transpiled"),
    }
}
