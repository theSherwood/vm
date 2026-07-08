//! Differential tests for **function imports / the host ABI** (the wasm `call` → SVM `cap.call`
//! lowering). A wasm import binds to a capability by the convention `module` = decimal `type_id`,
//! `name` = decimal `op`; the transpiler threads one capability handle **per distinct import
//! interface (module)** as the leading params of every function, and the embedder grants one
//! capability per interface and passes their handles as the entry's leading arguments (in
//! first-appearance order). These tests run a transpiled import-using module on **both** backends
//! under one reference `Host`, asserting they agree — and against a hand-computed oracle.
//!
//! Unlike `transpile.rs`'s capability-free `run`/`eval`, these need a powerbox: the interpreter via
//! `run_with_host`, the JIT via `compile_and_run_with_host` over the production `svm_run::cap_thunk`.

use std::ffi::c_void;
use svm_interp::{run_with_host, Host, Value};
use svm_jit::{compile_and_run_with_host, JitOutcome};

/// Serialize this binary's tests (ISSUES.md I4). `spawn_alongside_capability_import` runs 6 real
/// OS-thread workers doing futex park/notify; on macOS CI the binary intermittently died `SIGABRT`
/// in that path while *sibling* tests ran concurrently in the same process — and because tests
/// interleave, the abort could never be attributed to one test. Every test takes this lock, so the
/// threaded run has the process to itself and any recurrence is localized to the single test that
/// held the lock. A poisoned lock (an earlier test failed) is fine to reuse — take the inner guard.
fn serial() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

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
    let _serial = serial();
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
    let _serial = serial();
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
    let _serial = serial();
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
    let _serial = serial();
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

// ---- the binding surface (the convention's guard rails) ----

/// A non-numeric import name is now a §7 **named import** (resolved to a capability at load by the
/// embedder, e.g. an `svm-wasi` shim), not an error: the module declares it in its import table —
/// `"<module>.<name>"` — and call sites lower to `CallImport`. The numeric `module`=type_id /
/// `name`=op convention still produces an inline `cap.call`.
#[test]
fn import_non_numeric_name_is_a_named_import() {
    let _serial = serial();
    let wasm = wat::parse_str(
        r#"(module (import "env" "host_fn" (func (result i64))) (func (export "f") (result i64) (call 0)))"#,
    )
    .expect("assemble wat");
    let t = svm_wasm::transpile(&wasm).expect("non-numeric import is a named import, not an error");
    assert_eq!(t.module.imports.len(), 1, "one §7 named import declared");
    assert_eq!(t.module.imports[0].name, "env.host_fn");
}

/// Run a module importing **N** capability interfaces (distinct modules): grant one capability per
/// interface (in slot/first-appearance order), then pass the N handles as the entry's N leading args,
/// followed by `extra_args`. Asserts interp == JIT and returns the single result.
fn run_import_multi(
    wat: &str,
    entry: &str,
    grants: &[&dyn Fn(&mut Host) -> i32],
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

    // Interpreter: grant each cap, pass the handles (i32) as the entry's leading args, in slot order.
    let mut hi = Host::new();
    let handles: Vec<i32> = grants.iter().map(|g| g(&mut hi)).collect();
    let mut iargs: Vec<Value> = handles.iter().map(|h| Value::I32(*h)).collect();
    iargs.extend_from_slice(extra_args);
    let mut fuel = 100_000_000u64;
    let interp = run_with_host(&t.module, idx, &iargs, &mut fuel, &mut hi).expect("interp run");

    // JIT: the same grants (so handle values match) through the production cap thunk.
    let mut hj = Host::new();
    let jhandles: Vec<i32> = grants.iter().map(|g| g(&mut hj)).collect();
    assert_eq!(handles, jhandles, "handle encoding must match across hosts");
    let mut slots: Vec<i64> = jhandles.iter().map(|h| *h as i64).collect();
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
    assert_eq!(iv, jit[0], "interp != jit");
    iv
}

/// Imports spanning two distinct capability interfaces now bind to **two** handles — one per interface
/// (module), threaded as the entry's two leading params in first-appearance order. Here Clock
/// (type_id 2) gets slot 0 and Blocking (type_id 10) slot 1; the guest calls both in one function.
#[test]
fn import_multiple_interfaces_bind_to_distinct_handles() {
    let _serial = serial();
    let wat = r#"
(module
  (import "2" "0" (func $now (result i64)))
  (import "10" "0" (func $work (param i64) (result i64)))
  (func (export "f") (result i64)
    (i64.add (call $now) (call $work (i64.const 5)))))
"#;
    // Slot 0 = Clock (first appearance), slot 1 = Blocking. The reference clock's first read is 0.
    let got = run_import_multi(
        wat,
        "f",
        &[&|h: &mut Host| h.grant_clock(), &|h: &mut Host| {
            h.grant_blocking(std::time::Duration::ZERO, None)
        }],
        &[],
    );
    assert_eq!(got, mix(5), "clock.now (=0) + work(5) (=mix(5))");
}

/// The two handles are distinct interfaces *and distinct call paths*: a defined helper threads both
/// handles through a normal `call`, proving the N-handle prefix rides cross-function calls (not just
/// the entry). The helper sums clock + work; the entry calls it twice.
#[test]
fn import_two_handles_thread_through_defined_call() {
    let _serial = serial();
    let wat = r#"
(module
  (import "2" "0" (func $now (result i64)))
  (import "10" "0" (func $work (param i64) (result i64)))
  (func $step (param $x i64) (result i64)
    (i64.add (call $now) (call $work (local.get $x))))
  (func (export "f") (result i64)
    (i64.add (call $step (i64.const 5)) (call $step (i64.const 7)))))
"#;
    let got = run_import_multi(
        wat,
        "f",
        &[&|h: &mut Host| h.grant_clock(), &|h: &mut Host| {
            h.grant_blocking(std::time::Duration::ZERO, None)
        }],
        &[],
    );
    // step(5): clock=0 + mix(5). step(7): clock=1 + mix(7). Sum = 1 + mix(5) + mix(7).
    assert_eq!(got, 1i64.wrapping_add(mix(5)).wrapping_add(mix(7)));
}

/// **`wasi:thread/spawn` *alongside* a capability import** (§12 — the per-thread handle stash). The
/// spawning thread writes its capability handle into a reserved window slot before each spawn; the
/// synthesized shim reads it back on the new thread and threads it into `wasi_thread_start`, so a
/// spawned worker can `cap.call`. Here each of `n` workers computes `work(its start_arg)` (the
/// `Blocking` capability, a deterministic `mix`) and atomically adds it to a shared sum — which is
/// `Σ mix(i)` on every interleaving (so interp's M:N executor and the JIT's real OS threads must
/// agree). This proves capabilities reach spawned threads, the gap this slice closes.
#[test]
fn spawn_alongside_capability_import() {
    let _serial = serial();
    let wat = r#"
(module
  (import "10" "0" (func $work (param i64) (result i64)))     ;; Blocking cap (handle slot 0)
  (import "wasi" "thread-spawn" (func $spawn (param i32) (result i32)))
  (memory 1 1 shared)
  (func (export "wasi_thread_start") (param $tid i32) (param $start_arg i32)
    ;; sum (i64 at mem[8]) += work(start_arg)
    (drop (i64.atomic.rmw.add (i32.const 8)
            (call $work (i64.extend_i32_u (local.get $start_arg)))))
    (drop (i32.atomic.rmw.sub (i32.const 4) (i32.const 1)))   ;; remaining -= 1
    (drop (memory.atomic.notify (i32.const 4) (i32.const -1))))
  (func (export "run") (param $n i32) (result i64)
    (local $i i32) (local $r i32)
    (i32.atomic.store (i32.const 4) (local.get $n))           ;; remaining = n
    (block $spawned (loop $sp
      (br_if $spawned (i32.ge_u (local.get $i) (local.get $n)))
      (drop (call $spawn (local.get $i)))                     ;; start_arg = i
      (local.set $i (i32.add (local.get $i) (i32.const 1)))
      (br $sp)))
    (block $finished (loop $wait
      (local.set $r (i32.atomic.load (i32.const 4)))
      (br_if $finished (i32.eqz (local.get $r)))
      (drop (memory.atomic.wait32 (i32.const 4) (local.get $r) (i64.const 2000000000)))
      (br $wait)))
    (i64.atomic.load (i32.const 8))))
"#;
    let n = 6i64;
    let got = run_import(
        wat,
        "run",
        |h| h.grant_blocking(std::time::Duration::ZERO, None),
        &[Value::I32(n as i32)],
    );
    let want: i64 = (0..n).map(mix).fold(0i64, |a, x| a.wrapping_add(x));
    assert_eq!(got, want, "Σ mix(i) over {n} spawned workers using the cap");
}

/// An **imported memory** is now supported (the wasi-threads shape — the host owns the one shared
/// linear memory). SVM treats it exactly like a defined memory: the window's linear region at offset
/// 0. (Imported table/global/tag stay unsupported.)
#[test]
fn import_memory_is_supported() {
    let _serial = serial();
    let wasm = wat::parse_str(
        r#"(module (import "env" "memory" (memory 1)) (func (export "f") (result i32)
             (i32.store (i32.const 0) (i32.const 42)) (i32.load (i32.const 0))))"#,
    )
    .expect("assemble wat");
    let t = svm_wasm::transpile(&wasm).expect("imported memory should transpile");
    svm_verify::verify_module(&t.module).expect("verify");
    let idx = t.exports.iter().find(|(n, _)| n == "f").unwrap().1;
    let mut fuel = 1_000_000u64;
    assert_eq!(
        svm_interp::run(&t.module, idx, &[], &mut fuel).expect("run"),
        vec![Value::I32(42)],
        "an imported memory behaves like a defined one"
    );
}
