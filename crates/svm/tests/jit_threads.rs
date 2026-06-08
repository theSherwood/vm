//! Differential interp↔JIT tests for §12 **threads** (`thread.spawn`/`thread.join`).
//!
//! The JIT runs threaded modules on a cooperative green-thread scheduler (the entry becomes root vCPU
//! 0; spawned vCPUs are fibers; `join` blocks by suspending back to the scheduler). It is single OS
//! thread for now, so a run is deterministic — and for programs whose result is interleaving-invariant
//! (the ones below), the interp↔JIT differential oracle applies directly: the JIT must produce exactly
//! what the reference interpreter (true M:N executor) does.
//!
//! Stack switching exists only on x86-64 unix (`svm_fiber::supported()`); elsewhere the JIT bails
//! `Unsupported`, so these tests are gated to that target.
#![cfg(all(unix, target_arch = "x86_64"))]

use svm_interp::{run, Trap, Value};
use svm_jit::{
    compile_and_run, compile_and_run_parallel, compile_and_run_scheduled, JitOutcome, TrapKind,
};
use svm_text::parse_module;
use svm_verify::verify_module;

/// A 4-thread × 100 atomic-increment counter (→ 400), shared by the differential and seed-sweep tests.
const ATOMIC4: &str = "memory 16\n\
    func () -> (i64) {\n\
    block0():\n\
    \x20 v0 = i64.const 0\n\
    \x20 br block1(v0)\n\
    block1(v1: i64):\n\
    \x20 v2 = i64.const 4\n\
    \x20 v3 = i64.lt_u v1 v2\n\
    \x20 br_if v3 block2(v1) block3()\n\
    block2(v4: i64):\n\
    \x20 v5 = i64.const 100\n\
    \x20 v6 = thread.spawn 1 v5 v5\n\
    \x20 v7 = i64.const 4\n\
    \x20 v8 = i64.mul v4 v7\n\
    \x20 v9 = i64.const 16\n\
    \x20 v10 = i64.add v9 v8\n\
    \x20 i32.store v10 v6\n\
    \x20 v11 = i64.const 1\n\
    \x20 v12 = i64.add v4 v11\n\
    \x20 br block1(v12)\n\
    block3():\n\
    \x20 v13 = i64.const 0\n\
    \x20 br block4(v13)\n\
    block4(v14: i64):\n\
    \x20 v15 = i64.const 4\n\
    \x20 v16 = i64.lt_u v14 v15\n\
    \x20 br_if v16 block5(v14) block6()\n\
    block5(v17: i64):\n\
    \x20 v18 = i64.const 4\n\
    \x20 v19 = i64.mul v17 v18\n\
    \x20 v20 = i64.const 16\n\
    \x20 v21 = i64.add v20 v19\n\
    \x20 v22 = i32.load v21\n\
    \x20 v23 = thread.join v22\n\
    \x20 v24 = i64.const 1\n\
    \x20 v25 = i64.add v17 v24\n\
    \x20 br block4(v25)\n\
    block6():\n\
    \x20 v26 = i64.const 0\n\
    \x20 v27 = i64.atomic.load v26\n\
    \x20 return v27\n\
    }\n\
    func (i64, i64) -> (i64) {\n\
    block0(vsp: i64, v0: i64):\n\
    \x20 br block1(v0)\n\
    block1(v1: i64):\n\
    \x20 v2 = i64.const 0\n\
    \x20 v3 = i64.eq v1 v2\n\
    \x20 br_if v3 block2() block3(v1)\n\
    block3(v4: i64):\n\
    \x20 v5 = i64.const 0\n\
    \x20 v6 = i64.const 1\n\
    \x20 v7 = i64.atomic.rmw.add v5 v6\n\
    \x20 v8 = i64.const -1\n\
    \x20 v9 = i64.add v4 v8\n\
    \x20 br block1(v9)\n\
    block2():\n\
    \x20 v10 = i64.const 0\n\
    \x20 return v10\n\
    }\n";

fn to_slot(v: &Value) -> i64 {
    match v {
        Value::I32(x) => *x as i64,
        Value::I64(x) => *x,
        Value::F32(x) => x.to_bits() as i64,
        Value::F64(x) => x.to_bits() as i64,
    }
}

fn trap_matches(t: &Trap, k: &TrapKind) -> bool {
    matches!(
        (t, k),
        (Trap::ThreadFault, TrapKind::ThreadFault)
            | (Trap::FiberFault, TrapKind::FiberFault)
            | (Trap::MemoryFault, TrapKind::MemoryFault)
    )
}

fn assert_jit_matches_interp(src: &str) {
    let m = parse_module(src).unwrap_or_else(|e| panic!("parse: {e:?}\n{src}"));
    verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}\n{src}"));
    let mut fuel = 50_000_000u64;
    let interp = run(&m, 0, &[], &mut fuel);
    let jit = compile_and_run(&m, 0, &[]).expect("jit compile/run");
    match (&interp, &jit) {
        (Ok(vals), JitOutcome::Returned(slots)) => {
            let want: Vec<i64> = vals.iter().map(to_slot).collect();
            assert_eq!(&want, slots, "interp vs jit results differ\n{src}");
        }
        (Err(t), JitOutcome::Trapped(k)) if trap_matches(t, k) => {}
        _ => panic!("interp {interp:?} vs jit {jit:?} disagree\n{src}"),
    }
}

/// Spawn three worker threads that each just return their argument; the root joins all three and sums
/// the results (10 + 20 + 30 = 60). No shared memory — exercises spawn/join + result delivery.
#[test]
fn thread_spawn_join_sums_results() {
    let src = "func () -> (i64) {\n\
        block0():\n\
        \x20 v0 = i64.const 0\n\
        \x20 v1 = i64.const 10\n\
        \x20 v2 = thread.spawn 1 v0 v1\n\
        \x20 v3 = i64.const 20\n\
        \x20 v4 = thread.spawn 1 v0 v3\n\
        \x20 v5 = i64.const 30\n\
        \x20 v6 = thread.spawn 1 v0 v5\n\
        \x20 v7 = thread.join v2\n\
        \x20 v8 = thread.join v4\n\
        \x20 v9 = thread.join v6\n\
        \x20 v10 = i64.add v7 v8\n\
        \x20 v11 = i64.add v10 v9\n\
        \x20 return v11\n\
        }\n\
        func (i64, i64) -> (i64) {\n\
        block0(v0: i64, v1: i64):\n\
        \x20 return v1\n\
        }\n";
    assert_jit_matches_interp(src);
}

/// Four worker threads each `atomic.rmw.add` a shared counter 100×; the total is exactly 400 on every
/// interleaving (a non-atomic RMW would lose updates). Layout: `mem[0]` i64 counter, `mem[16+4i]` i32
/// handle of worker `i`.
#[test]
fn thread_atomic_counter() {
    assert_jit_matches_interp(ATOMIC4);
}

/// **Real multi-core execution.** The same counter on a 4-worker parallel pool: vCPUs run on real OS
/// threads, contending on the shared counter via hardware atomics. The total is exactly 400 every run
/// (a lost update — a scheduler race or a non-atomic RMW — would drop it). Many runs to shake out
/// races (the scheduler glue is also loom-verified in `svm-jit`'s `par` module).
#[test]
fn thread_parallel_atomic_counter() {
    let m = parse_module(ATOMIC4).expect("parse");
    verify_module(&m).expect("verify");
    for _ in 0..40 {
        match compile_and_run_parallel(&m, 0, &[], 4).expect("jit") {
            JitOutcome::Returned(x) => assert_eq!(x, [400], "parallel total wrong"),
            other => panic!("unexpected {other:?}"),
        }
    }
}

/// The deterministic seeded scheduler (the verification backbone): every seed yields the invariant
/// 400, and each seed is *reproducible* (same seed → same result) — the JIT analogue of the
/// interpreter's `run_scheduled` sweep, so a scheduler bug surfaces as a replayable failing seed.
#[test]
fn thread_seed_sweep_is_invariant_and_reproducible() {
    let m = parse_module(ATOMIC4).expect("parse");
    verify_module(&m).expect("verify");
    for seed in 0..24u64 {
        let a = compile_and_run_scheduled(&m, 0, &[], seed).expect("jit");
        let b = compile_and_run_scheduled(&m, 0, &[], seed).expect("jit");
        match (&a, &b) {
            (JitOutcome::Returned(x), JitOutcome::Returned(y)) => {
                assert_eq!(x, &[400], "seed {seed}: not the invariant total");
                assert_eq!(x, y, "seed {seed}: not reproducible");
            }
            _ => panic!("seed {seed}: unexpected {a:?} / {b:?}"),
        }
    }
}

/// Futex handoff: the producer writes a payload to `mem[8]`, spawns a consumer that `atomic.wait`s on
/// the flag at `mem[0]`, then sets the flag (release) and notifies; the consumer returns the payload
/// it reads. On every interleaving the result is the written payload — exercising `atomic.wait` /
/// `atomic.notify` end to end.
#[test]
fn thread_futex_handoff() {
    let src = "memory 16\n\
        func () -> (i64) {\n\
        block0():\n\
        \x20 v0 = i64.const 8\n\
        \x20 v1 = i64.const 987654\n\
        \x20 i64.atomic.store.release v0 v1\n\
        \x20 v2 = i64.const 0\n\
        \x20 v3 = thread.spawn 1 v2 v2\n\
        \x20 v4 = i64.const 0\n\
        \x20 v5 = i32.const 1\n\
        \x20 i32.atomic.store.release v4 v5\n\
        \x20 v6 = i64.const 0\n\
        \x20 v7 = i32.const 1\n\
        \x20 v8 = atomic.notify v6 v7\n\
        \x20 v9 = thread.join v3\n\
        \x20 return v9\n\
        }\n\
        func (i64, i64) -> (i64) {\n\
        block0(vsp: i64, v0: i64):\n\
        \x20 v1 = i64.const 0\n\
        \x20 v2 = i32.const 0\n\
        \x20 v3 = i64.const 1000000000\n\
        \x20 v4 = i32.atomic.wait v1 v2 v3\n\
        \x20 v5 = i64.const 8\n\
        \x20 v6 = i64.atomic.load.acquire v5\n\
        \x20 return v6\n\
        }\n";
    assert_jit_matches_interp(src);
}

/// Joining the same handle twice is inert the second time → `ThreadFault`, on both backends.
#[test]
fn thread_double_join_traps() {
    let src = "func () -> (i64) {\n\
        block0():\n\
        \x20 v0 = i64.const 0\n\
        \x20 v1 = i64.const 7\n\
        \x20 v2 = thread.spawn 1 v0 v1\n\
        \x20 v3 = thread.join v2\n\
        \x20 v4 = thread.join v2\n\
        \x20 return v4\n\
        }\n\
        func (i64, i64) -> (i64) {\n\
        block0(v0: i64, v1: i64):\n\
        \x20 return v1\n\
        }\n";
    assert_jit_matches_interp(src);
}
