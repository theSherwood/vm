//! **Concurrency benchmark: SVM vs Wasmtime + wasi-threads** (the `--threads` mode).
//!
//! The same wasm bytes — a wasi-threads-ABI parallel atomic counter (imports `wasi:thread/spawn` and
//! a shared memory, exports `wasi_thread_start`) — run on:
//!   * **SVM**: transpiled to IR and JIT-compiled; `thread.spawn` is a native 1:1 OS-thread vCPU over
//!     the shared window (concurrency *in* the VM — DESIGN §1a).
//!   * **Wasmtime + `wasmtime-wasi-threads`**: each spawned thread **re-instantiates** the module on a
//!     new OS thread sharing the imported memory (concurrency bolted onto the host).
//!
//! Both lower atomics via Cranelift to the same hardware instructions, so steady-state compute is
//! parity; the interesting axis is **spawn/teardown cost** — SVM's raw OS thread vs Wasmtime's
//! per-thread instantiation. We time only the parallel `run` (spawn + work + join), compiled once,
//! best-of-`reps`, across a spawn-heavy and a compute-heavy point.

use std::sync::Arc;
use std::time::{Duration, Instant};
use wasmtime::{Engine, Linker, Module, Store};

/// A wasi-threads parallel atomic counter: spawn `nworkers`, each does `steps` atomic increments of
/// the shared counter (mem[0]); `run` futex-waits on a remaining-count (mem[4]) the workers decrement,
/// and returns the total (`nworkers * steps`, interleaving-invariant). The memory is **imported**
/// (the wasi-threads shape — the host owns the one shared memory).
fn parallel_wat(nworkers: u32, steps: u32) -> String {
    format!(
        r#"
(module
  (import "wasi" "thread-spawn" (func $spawn (param i32) (result i32)))
  (import "env" "memory" (memory 1 64 shared))
  (func (export "wasi_thread_start") (param $tid i32) (param $start_arg i32)
    (local $j i32)
    (block $done (loop $lp
      (br_if $done (i32.ge_u (local.get $j) (i32.const {steps})))
      (drop (i32.atomic.rmw.add (i32.const 0) (i32.const 1)))
      (local.set $j (i32.add (local.get $j) (i32.const 1)))
      (br $lp)))
    (drop (i32.atomic.rmw.sub (i32.const 4) (i32.const 1)))
    (drop (memory.atomic.notify (i32.const 4) (i32.const -1))))
  (func (export "run") (result i32)
    (local $i i32) (local $r i32)
    (i32.atomic.store (i32.const 0) (i32.const 0))           ;; reset total (the window persists across reps)
    (i32.atomic.store (i32.const 4) (i32.const {nworkers}))  ;; remaining = nworkers
    (block $spawned (loop $sp
      (br_if $spawned (i32.ge_u (local.get $i) (i32.const {nworkers})))
      (drop (call $spawn (local.get $i)))
      (local.set $i (i32.add (local.get $i) (i32.const 1)))
      (br $sp)))
    ;; Standard futex wait: load remaining ONCE into $r, and wait expecting that same $r — so a
    ;; worker that drives remaining to 0 (and notifies) in the gap makes the wait return immediately
    ;; (value != expected) instead of blocking on a stale value with no further notify (a lost wakeup).
    (block $finished (loop $wait
      (local.set $r (i32.atomic.load (i32.const 4)))
      (br_if $finished (i32.eqz (local.get $r)))
      (drop (memory.atomic.wait32 (i32.const 4) (local.get $r) (i64.const 5000000000)))
      (br $wait)))
    (i32.atomic.load (i32.const 0))))
"#
    )
}

/// Time the parallel `run` on SVM (compiled once; `thread.spawn` = native OS threads). Returns
/// `(result, best_of_reps)`.
fn run_svm(wasm: &[u8], reps: usize) -> (i64, Duration) {
    use svm_jit::{CompiledModule, JitOutcome, Quota, INERT_CAP_THUNK};
    let t = svm_wasm::transpile(wasm).expect("svm transpile");
    let run_idx = t
        .exports
        .iter()
        .find(|(n, _)| n == "run")
        .expect("run export")
        .1;
    let mut cm = CompiledModule::compile(
        &t.module,
        run_idx,
        INERT_CAP_THUNK,
        std::ptr::null_mut(),
        svm_ir::DEFAULT_RESERVED_LOG2,
        None,
        None,
        None,
        None,
        Quota::default(),
        0,
    )
    .expect("svm jit compile");
    let mut best = Duration::MAX;
    let mut result = 0i64;
    for _ in 0..reps {
        let t0 = Instant::now();
        let (out, _) = cm.run(&[], None, None, None).expect("svm run");
        best = best.min(t0.elapsed());
        result = match out {
            JitOutcome::Returned(v) => v[0],
            other => panic!("svm run did not return: {other:?}"),
        };
    }
    (result, best)
}

/// Host state for `wasmtime-wasi-threads`: holds the spawn context (an `Arc` so it is cheap to clone
/// per thread, as the wasi-threads `T: Clone` bound requires). Recursive via `Arc` (which breaks the
/// type cycle).
#[derive(Clone)]
struct ThreadsHost {
    ctx: Option<Arc<wasmtime_wasi_threads::WasiThreadsCtx<ThreadsHost>>>,
}

/// Time the parallel `run` on Wasmtime + wasi-threads (compiled once; each spawn re-instantiates the
/// module on a new OS thread sharing the imported memory). Returns `(result, best_of_reps)`.
fn run_wasmtime(engine: &Engine, wasm: &[u8], reps: usize) -> (i32, Duration) {
    let module = Module::new(engine, wasm).expect("wasmtime compile");
    let mut linker: Linker<ThreadsHost> = Linker::new(engine);
    let mut store = Store::new(engine, ThreadsHost { ctx: None });
    // Satisfies the `wasi:thread/spawn` import and the shared-memory import.
    wasmtime_wasi_threads::add_to_linker(&mut linker, &store, &module, |h| {
        h.ctx.as_deref().expect("wasi-threads ctx set")
    })
    .expect("add wasi-threads");
    let ctx = Arc::new(
        wasmtime_wasi_threads::WasiThreadsCtx::new(module.clone(), Arc::new(linker.clone()), false)
            .expect("wasi-threads ctx"),
    );
    store.data_mut().ctx = Some(ctx);
    let instance = linker
        .instantiate(&mut store, &module)
        .expect("wasmtime instantiate");
    let run = instance
        .get_typed_func::<(), i32>(&mut store, "run")
        .expect("run export");
    let mut best = Duration::MAX;
    let mut result = 0i32;
    for _ in 0..reps {
        let t0 = Instant::now();
        result = run.call(&mut store, ()).expect("wasmtime run");
        best = best.min(t0.elapsed());
    }
    (result, best)
}

/// Run the concurrency comparison and print a small table.
pub fn run(engine: &Engine, reps: usize) {
    // (label, nworkers, steps): a spawn-heavy point (many short threads) and a compute-heavy point
    // (few long threads). Spawn-heavy exposes SVM's lighter thread.spawn vs Wasmtime's per-thread
    // re-instantiation; compute-heavy is steady-state (shared-Cranelift atomics ⇒ ~parity).
    let cases = [
        ("spawn-heavy   (64×1k)", 64u32, 1_000u32),
        ("balanced      (8×100k)", 8, 100_000),
        ("compute-heavy (4×2M)", 4, 2_000_000),
    ];
    println!("concurrency: SVM (native thread.spawn) vs Wasmtime+wasi-threads, same wasm bytes");
    println!("  best of {reps} runs; ratio = SVM ÷ Wasmtime (<1 ⇒ SVM faster)\n");
    println!(
        "  {:<24} {:>12} {:>12} {:>8}   {}",
        "case", "SVM", "Wasmtime", "ratio", "result"
    );
    for (label, nw, steps) in cases {
        let wasm = wat::parse_str(&parallel_wat(nw, steps)).expect("assemble wat");
        let (svm_res, svm_t) = run_svm(&wasm, reps);
        let (wt_res, wt_t) = run_wasmtime(engine, &wasm, reps);
        let want = (nw as i64) * (steps as i64);
        assert_eq!(svm_res, want, "SVM {label} wrong result");
        assert_eq!(wt_res as i64, want, "Wasmtime {label} wrong result");
        let ratio = svm_t.as_secs_f64() / wt_t.as_secs_f64();
        println!(
            "  {:<24} {:>10.3}ms {:>10.3}ms {:>8.2}   {}",
            label,
            svm_t.as_secs_f64() * 1e3,
            wt_t.as_secs_f64() * 1e3,
            ratio,
            want,
        );
    }
    println!(
        "\n  note: both lower atomics via Cranelift to the same hardware ops (compute ≈ parity);\n  \
         the spawn-heavy row is the differentiator — SVM's raw OS-thread `thread.spawn` over the\n  \
         shared window vs Wasmtime re-instantiating the module per spawned thread."
    );
}
