//! §12 wasm threads — **spawn** (slice 2): the wasi-threads ABI maps onto SVM's native
//! `thread.spawn`. A module imports `wasi:thread/spawn`, exports `wasi_thread_start(tid, start_arg)`,
//! and declares shared memory; the transpiler lowers a `call $wasi_thread_spawn` to `thread.spawn` of
//! a synthesized shim (allocating a unique tid via a reserved counter slot and packing it with
//! `start_arg`). This is the same bytes `wasmtime-wasi-threads` runs — concurrency lives *in* the VM
//! (DESIGN §1a), not bolted onto the host.
//!
//! These are genuine multi-threaded runs: the interpreter drives its M:N executor (`svm_interp::run`),
//! the JIT spawns real 1:1 OS threads (`compile_and_run`). The totals are interleaving-invariant, so
//! the two backends must agree (the in-`run` differential), and the value pins correctness.

use svm_interp::Value;

/// Transpile WAT → IR, verify, run `entry(args)` on interp **and** JIT (both genuinely multi-threaded
/// for a spawning module), assert they agree, return the i64 result.
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
    let mut fuel = 1_000_000_000u64;
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
    let iv = match interp[0] {
        Value::I32(x) => x as i64,
        Value::I64(x) => x,
        other => panic!("unexpected interp value {other:?}"),
    };
    assert_eq!(iv, jit[0], "interp != jit");
    iv
}

/// A wasi-threads-ABI module: a parallel atomic counter. `run` initializes the worker count, spawns
/// `nworkers` threads via the `wasi:thread/spawn` import, then futex-waits on a "remaining" counter
/// each worker decrements (notifying as it finishes); returns the shared counter, which is exactly
/// `nworkers * steps` on every interleaving. mem[0] = i32 total, mem[4] = i32 remaining.
fn parallel_counter_wat(nworkers: i32, steps: i32) -> String {
    format!(
        r#"
      (module
        (import "wasi" "thread-spawn" (func $spawn (param i32) (result i32)))
        (memory 1 1 shared)
        ;; the spawned worker: `steps` atomic increments of mem[0], then signal completion.
        (func (export "wasi_thread_start") (param $tid i32) (param $start_arg i32)
          (local $j i32)
          (block $done
            (loop $lp
              (br_if $done (i32.ge_u (local.get $j) (i32.const {steps})))
              (drop (i32.atomic.rmw.add (i32.const 0) (i32.const 1)))
              (local.set $j (i32.add (local.get $j) (i32.const 1)))
              (br $lp)))
          (drop (i32.atomic.rmw.sub (i32.const 4) (i32.const 1)))   ;; remaining -= 1
          (drop (memory.atomic.notify (i32.const 4) (i32.const -1))))  ;; wake the main thread
        (func (export "run") (result i32)
          (local $i i32)
          (i32.atomic.store (i32.const 4) (i32.const {nworkers}))    ;; remaining = nworkers
          (block $spawned
            (loop $sp
              (br_if $spawned (i32.ge_u (local.get $i) (i32.const {nworkers})))
              (drop (call $spawn (local.get $i)))                     ;; start_arg = i
              (local.set $i (i32.add (local.get $i) (i32.const 1)))
              (br $sp)))
          (block $finished
            (loop $wait
              (br_if $finished (i32.eq (i32.atomic.load (i32.const 4)) (i32.const 0)))
              ;; futex-wait on mem[4] for the current value (re-checked atomically); 2s safety timeout.
              (drop (memory.atomic.wait32 (i32.const 4)
                       (i32.atomic.load (i32.const 4))
                       (i64.const 2000000000)))
              (br $wait)))
          (i32.atomic.load (i32.const 0))))
      "#
    )
}

#[test]
fn parallel_counter_4x1000() {
    let wat = parallel_counter_wat(4, 1000);
    assert_eq!(run(&wat, "run", &[]), 4000, "4 workers × 1000 increments");
}

#[test]
fn parallel_counter_8x500() {
    let wat = parallel_counter_wat(8, 500);
    assert_eq!(run(&wat, "run", &[]), 4000, "8 workers × 500 increments");
}

/// `wasi:thread/spawn` returns a **unique positive tid** (the first is 1). A module that spawns one
/// worker (which just signals done) and returns the spawn call's result witnesses the tid allocation.
#[test]
fn spawn_returns_positive_tid() {
    let wat = r#"
      (module
        (import "wasi" "thread-spawn" (func $spawn (param i32) (result i32)))
        (memory 1 1 shared)
        (func (export "wasi_thread_start") (param $tid i32) (param $start_arg i32)
          (drop (i32.atomic.rmw.sub (i32.const 4) (i32.const 1)))
          (drop (memory.atomic.notify (i32.const 4) (i32.const -1))))
        (func (export "run") (result i32)
          (local $tid i32)
          (i32.atomic.store (i32.const 4) (i32.const 1))
          (local.set $tid (call $spawn (i32.const 0)))               ;; → tid (1)
          (block $finished
            (loop $wait
              (br_if $finished (i32.eq (i32.atomic.load (i32.const 4)) (i32.const 0)))
              (drop (memory.atomic.wait32 (i32.const 4)
                       (i32.atomic.load (i32.const 4))
                       (i64.const 2000000000)))
              (br $wait)))
          (local.get $tid)))"#;
    assert_eq!(run(wat, "run", &[]), 1, "first spawned tid is 1");
}

/// `wasi:thread/spawn` without the `wasi_thread_start` export is a clean error (not a panic).
#[test]
fn spawn_without_start_export_rejected() {
    let wat = r#"
      (module
        (import "wasi" "thread-spawn" (func $spawn (param i32) (result i32)))
        (memory 1 1 shared)
        (func (export "run") (result i32)
          (call $spawn (i32.const 0))))"#;
    let wasm = wat::parse_str(wat).expect("assemble wat");
    match svm_wasm::transpile(&wasm) {
        Err(svm_wasm::Error::Unsupported(_)) => {}
        Err(e) => panic!("expected Unsupported, got {e:?}"),
        Ok(_) => panic!("expected a missing-wasi_thread_start error"),
    }
}
