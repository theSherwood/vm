//! Differential interp↔JIT tests for §12 **threads** (`thread.spawn`/`thread.join`).
//!
//! The JIT runs each spawned vCPU as a **real 1:1 OS thread** (`os_thread_rt`), all sharing the one
//! `Arc<Region>` window (hardware atomics + the condvar futex). For programs whose result is
//! interleaving-invariant (the ones below), the interp↔JIT differential oracle applies directly: the
//! JIT must produce exactly what the reference interpreter (M:N executor) does.
//!
//! The JIT thread/fiber runtime exists on x86-64 unix, aarch64 unix, and x86-64 Windows today
//! (`svm_fiber::supported()`); elsewhere the JIT bails `Unsupported`, so these tests are gated to it.
#![cfg(any(
    all(unix, target_arch = "x86_64"),
    all(unix, target_arch = "aarch64"),
    all(windows, target_arch = "x86_64")
))]

use svm_interp::{run, Trap, Value};
use svm_jit::{compile_and_run, JitOutcome, TrapKind};
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
        Value::V128(b) => i64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]),
        Value::Ref(x) => *x as i64,
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

/// **Real multi-core execution.** The same counter with four vCPUs on real OS threads (1:1, the VM's
/// only thread primitive — no scheduler), contending on the shared counter via hardware atomics. The
/// total is exactly 400 every run (a lost update — a non-atomic RMW — would drop it). Many runs to
/// shake out races; the deterministic exhaustive-interleaving check is the interpreter oracle
/// (`run_scheduled`/`explore_all`), against which this run is differential-tested.
#[test]
fn thread_parallel_atomic_counter() {
    let m = parse_module(ATOMIC4).expect("parse");
    verify_module(&m).expect("verify");
    for _ in 0..40 {
        match compile_and_run(&m, 0, &[]).expect("jit") {
            JitOutcome::Returned(x) => assert_eq!(x, [400], "parallel total wrong"),
            other => panic!("unexpected {other:?}"),
        }
    }
}

/// Futex handoff: the producer writes a payload to `mem[8]`, spawns a consumer that `atomic.wait`s on
/// the flag at `mem[0]`, then sets the flag (release) and notifies; the consumer returns the payload
/// it reads. On every interleaving the result is the written payload — exercising `atomic.wait` /
/// `atomic.notify` end to end.
const FUTEX: &str = "memory 16\n\
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

#[test]
fn thread_futex_handoff() {
    assert_jit_matches_interp(FUTEX);
}

/// The futex handoff with real parallel vCPUs: the consumer genuinely parks on `atomic.wait` (on its
/// own OS thread, possibly before the producer sets the flag) and is woken by the producer's
/// `atomic.notify` — the real block→notify path. The result is the payload (987654) every run.
#[test]
fn thread_parallel_futex_handoff() {
    let m = parse_module(FUTEX).expect("parse");
    verify_module(&m).expect("verify");
    for _ in 0..40 {
        match compile_and_run(&m, 0, &[]).expect("jit") {
            JitOutcome::Returned(x) => assert_eq!(x, [987654], "parallel futex payload wrong"),
            other => panic!("unexpected {other:?}"),
        }
    }
}

/// **Fibers + threads in one module** (the closed gap): a worker thread internally drives a generator
/// fiber. `main` spawns the worker with arg 5; the worker `cont.new`s a generator (func 2), resumes it
/// (it `suspend`s 42), and returns 42 + 5 = 47; `main` joins → 47. The JIT now gives each vCPU its own
/// fiber runtime, so this runs (cooperatively) and matches the interpreter rather than bailing.
#[test]
fn thread_with_fiber_inside() {
    let src = "func () -> (i64) {\n\
        block0():\n\
        \x20 v0 = i64.const 0\n\
        \x20 v1 = i64.const 5\n\
        \x20 v2 = thread.spawn 1 v0 v1\n\
        \x20 v3 = thread.join v2\n\
        \x20 return v3\n\
        }\n\
        func (i64, i64) -> (i64) {\n\
        block0(vsp: i64, varg: i64):\n\
        \x20 v0 = ref.func 2\n\
        \x20 v1 = i64.const 0\n\
        \x20 v2 = cont.new v0 v1\n\
        \x20 v3 = i64.const 0\n\
        \x20 v4, v5 = cont.resume v2 v3\n\
        \x20 v6 = i64.add v5 varg\n\
        \x20 return v6\n\
        }\n\
        func (i64, i64) -> (i64) {\n\
        block0(vsp: i64, varg: i64):\n\
        \x20 v0 = i64.const 42\n\
        \x20 v1 = suspend v0\n\
        \x20 v2 = i64.const 0\n\
        \x20 return v2\n\
        }\n";
    assert_jit_matches_interp(src);
}

/// Fibers + threads on the **parallel** pool: four worker threads each drive their own generator fiber
/// (each returns `iters + 42` where the fiber yields 42) and `main` sums the joins. Each vCPU has its
/// own fiber runtime, so the per-thread coroutines don't interfere — runs on real cores, exact sum
/// every time. (`mem[16+4i]` holds worker `i`'s handle; result = Σ (i*10 + 42) for i in 0..4.)
#[test]
fn thread_parallel_with_fibers() {
    let src = "memory 16\n\
        func () -> (i64) {\n\
        block0():\n\
        \x20 v0 = i64.const 0\n\
        \x20 br block1(v0)\n\
        block1(v1: i64):\n\
        \x20 v2 = i64.const 4\n\
        \x20 v3 = i64.lt_u v1 v2\n\
        \x20 br_if v3 block2(v1) block3()\n\
        block2(v4: i64):\n\
        \x20 v5 = i64.const 10\n\
        \x20 v6 = i64.mul v4 v5\n\
        \x20 v7 = thread.spawn 1 v4 v6\n\
        \x20 v8 = i64.const 4\n\
        \x20 v9 = i64.mul v4 v8\n\
        \x20 v10 = i64.const 16\n\
        \x20 v11 = i64.add v10 v9\n\
        \x20 i32.store v11 v7\n\
        \x20 v12 = i64.const 1\n\
        \x20 v13 = i64.add v4 v12\n\
        \x20 br block1(v13)\n\
        block3():\n\
        \x20 v14 = i64.const 0\n\
        \x20 br block4(v14, v14)\n\
        block4(v15: i64, v16: i64):\n\
        \x20 v17 = i64.const 4\n\
        \x20 v18 = i64.lt_u v15 v17\n\
        \x20 br_if v18 block5(v15, v16) block6(v16)\n\
        block5(v19: i64, v20: i64):\n\
        \x20 v21 = i64.const 4\n\
        \x20 v22 = i64.mul v19 v21\n\
        \x20 v23 = i64.const 16\n\
        \x20 v24 = i64.add v23 v22\n\
        \x20 v25 = i32.load v24\n\
        \x20 v26 = thread.join v25\n\
        \x20 v27 = i64.add v20 v26\n\
        \x20 v28 = i64.const 1\n\
        \x20 v29 = i64.add v19 v28\n\
        \x20 br block4(v29, v27)\n\
        block6(v30: i64):\n\
        \x20 return v30\n\
        }\n\
        func (i64, i64) -> (i64) {\n\
        block0(vsp: i64, varg: i64):\n\
        \x20 v0 = ref.func 2\n\
        \x20 v1 = i64.const 0\n\
        \x20 v2 = cont.new v0 v1\n\
        \x20 v3 = i64.const 0\n\
        \x20 v4, v5 = cont.resume v2 v3\n\
        \x20 v6 = i64.add v5 varg\n\
        \x20 return v6\n\
        }\n\
        func (i64, i64) -> (i64) {\n\
        block0(vsp: i64, varg: i64):\n\
        \x20 v0 = i64.const 42\n\
        \x20 v1 = suspend v0\n\
        \x20 v2 = i64.const 0\n\
        \x20 return v2\n\
        }\n";
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");
    // Σ over i in 0..4 of (i*10 + 42) = (0+10+20+30) + 4*42 = 60 + 168 = 228.
    for _ in 0..40 {
        match compile_and_run(&m, 0, &[]).expect("jit") {
            JitOutcome::Returned(x) => assert_eq!(x, [228], "parallel fiber+thread sum wrong"),
            other => panic!("unexpected {other:?}"),
        }
    }
    // And it matches the interpreter (cooperative).
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

/// **The fiber handle namespace is domain-wide on both backends (D57 3b-ii).** The JIT's fiber
/// table is now one `SharedFiberTable` per domain (like the interp's run-shared registry), so a
/// fiber created on a spawned vCPU gets the *next domain slot* — not slot 0 of a private
/// per-thread table. Root creates fiber 0 before spawning; the worker's `cont.new` must yield
/// handle **1** (the old per-thread tables gave it 0, diverging from the interpreter). Both
/// fibers are also driven to completion, proving the distinct slots both work.
#[test]
fn fiber_namespace_is_domain_wide() {
    let src = "func () -> (i64) {\n\
        block0():\n\
        \x20 v0 = ref.func 2\n\
        \x20 v1 = i64.const 0\n\
        \x20 v2 = cont.new v0 v1\n\
        \x20 v3 = thread.spawn 1 v1 v1\n\
        \x20 v4 = thread.join v3\n\
        \x20 v5 = i64.const 7\n\
        \x20 v6, v7 = cont.resume v2 v5\n\
        \x20 v8 = i64.const 100\n\
        \x20 v9 = i64.mul v4 v8\n\
        \x20 v10 = i64.add v9 v7\n\
        \x20 v11 = i64.add v10 v2\n\
        \x20 return v11\n\
        }\n\
        func (i64, i64) -> (i64) {\n\
        block0(vsp: i64, varg: i64):\n\
        \x20 v0 = ref.func 2\n\
        \x20 v1 = cont.new v0 vsp\n\
        \x20 v2 = i64.const 1\n\
        \x20 v3, v4 = cont.resume v1 v2\n\
        \x20 v5 = i64.const 42\n\
        \x20 v6 = i64.sub v4 v5\n\
        \x20 v7 = i64.add v1 v6\n\
        \x20 return v7\n\
        }\n\
        func (i64, i64) -> (i64) {\n\
        block0(vsp: i64, varg: i64):\n\
        \x20 v0 = i64.const 41\n\
        \x20 v1 = i64.add varg v0\n\
        \x20 return v1\n\
        }\n";
    assert_jit_matches_interp(src);
    // Pin the absolute result too (worker handle 1 ⇒ join 1 ⇒ 100, + root fiber's 7+41 = 148):
    // the old per-thread JIT tables produced 48 here (worker handle 0).
    let m = parse_module(src).expect("parse");
    let mut fuel = 1_000_000u64;
    assert_eq!(run(&m, 0, &[], &mut fuel), Ok(vec![Value::I64(148)]));
}

/// **Stackful fiber migration, differentially (D57 3c — the headline).** A fiber created and
/// part-run on the root (it suspends, capturing its first-resume argument 5 in its parked
/// *native stack*) is resumed from a **spawned vCPU on another OS thread**: the JIT claims the
/// slot through the loom-verified `Ownership` CAS and resumes the saved stack with the *same*
/// `svm-fiber` switch — and must produce exactly what the interpreter's migrating registry
/// (3b-i, the oracle) does: `10*7 + 5 = 75`, the `+ 5` proving the stack state captured on the
/// root survived the migration intact (a restart would lose it). Until 3c this test pinned the
/// staged divergence (the JIT's foreign claim faulted); it is now the migration differential.
#[test]
fn fiber_suspended_on_root_migrates_to_spawned_vcpu() {
    let src = "func () -> (i64) {\n\
        block0():\n\
        \x20 v0 = ref.func 2\n\
        \x20 v1 = i64.const 4096\n\
        \x20 v2 = cont.new v0 v1\n\
        \x20 v3 = i64.const 5\n\
        \x20 v4, v5 = cont.resume v2 v3\n\
        \x20 v6 = thread.spawn 1 v2 v2\n\
        \x20 v7 = thread.join v6\n\
        \x20 return v7\n\
        }\n\
        func (i64, i64) -> (i64) {\n\
        block0(vsp: i64, varg: i64):\n\
        \x20 v0 = i64.const 7\n\
        \x20 v1, v2 = cont.resume varg v0\n\
        \x20 return v2\n\
        }\n\
        func (i64, i64) -> (i64) {\n\
        block0(vsp: i64, varg: i64):\n\
        \x20 v0 = suspend varg\n\
        \x20 v1 = i64.const 10\n\
        \x20 v2 = i64.mul v0 v1\n\
        \x20 v3 = i64.add v2 varg\n\
        \x20 return v3\n\
        }\n";
    // Both backends agree (the differential)…
    assert_jit_matches_interp(src);
    // …and the absolute migration semantics: the fiber *continued* on the worker, stack intact.
    let m = parse_module(src).expect("parse");
    let mut fuel = 1_000_000u64;
    assert_eq!(run(&m, 0, &[], &mut fuel), Ok(vec![Value::I64(75)]));
}

/// **Concurrent steal stress (the D57 3c empirical net, layer 5).** K=16 fibers are created on
/// the root; **round 1** spawns 4 workers that race over an atomic work index, each claiming and
/// first-resuming fibers (every fiber yields `3*id + 1` and parks, published `RUNNABLE`); after a
/// join barrier, **round 2** spawns 4 *fresh* OS threads that race a second index and resume each
/// parked fiber to completion (`7*id + 2`) — so **all 16 second resumes are cross-thread
/// migrations of saved native stacks**, claimed concurrently through the `Ownership` CAS while
/// siblings race. The atomic sum is schedule-invariant: `Σ (10·id + 3) = 1248` — any torn switch,
/// lost claim, or double-resume changes it (or faults via the guard page / single-owner assert).
/// 30 reps on the JIT (real cores) + the interp differential.
#[test]
fn concurrent_fiber_steal_stress() {
    // mem[16]: round-1 work index; mem[20]: round-2 work index; mem[512+8i]: fiber i's i64 handle;
    // mem[128]: the i64 atomic sum. Fiber identity rides its `sp` (= its index i). (The handle array
    // is at 512/stride-8 — not 24/stride-4 — because an i64 handle is 8 bytes and a stride-8 array
    // from 24 would overlap the sum at 128.)
    // Worker body (funcs 1 and 2, differing only in the work-index address):
    //   loop: i = i32.atomic.rmw.add idx 1; if i >= 16 return 0;
    //         h = i64.load(512+8i); (st, v) = cont.resume h 0; i64.atomic.rmw.add mem[128] v
    let worker = |idx_addr: u64| -> String {
        format!(
            "func (i64, i64) -> (i64) {{\n\
             block0(v0: i64, v1: i64):\n\
             \x20 br block1()\n\
             block1():\n\
             \x20 v2 = i64.const {idx_addr}\n\
             \x20 v3 = i32.const 1\n\
             \x20 v4 = i32.atomic.rmw.add v2 v3\n\
             \x20 v5 = i32.const 16\n\
             \x20 v6 = i32.lt_u v4 v5\n\
             \x20 br_if v6 block2(v4) block3()\n\
             block2(v7: i32):\n\
             \x20 v8 = i64.extend_i32_u v7\n\
             \x20 v9 = i64.const 8\n\
             \x20 v10 = i64.mul v8 v9\n\
             \x20 v11 = i64.const 512\n\
             \x20 v12 = i64.add v11 v10\n\
             \x20 v13 = i64.load v12\n\
             \x20 v14 = i64.const 0\n\
             \x20 v15, v16 = cont.resume v13 v14\n\
             \x20 v17 = i64.const 128\n\
             \x20 v18 = i64.atomic.rmw.add v17 v16\n\
             \x20 br block1()\n\
             block3():\n\
             \x20 v19 = i64.const 0\n\
             \x20 return v19\n\
             }}\n"
        )
    };
    // Root: create the 16 fibers (handle at 24+4i, sp = i), then two rounds of spawn-4 / join-4.
    let mut root = String::from(
        "memory 16\n\
         func () -> (i64) {\n\
         block0():\n\
         \x20 v0 = i64.const 0\n\
         \x20 br block1(v0)\n\
         block1(v1: i64):\n\
         \x20 v2 = i64.const 16\n\
         \x20 v3 = i64.lt_u v1 v2\n\
         \x20 br_if v3 block2(v1) block3()\n\
         block2(v4: i64):\n\
         \x20 v5 = ref.func 3\n\
         \x20 v6 = cont.new v5 v4\n\
         \x20 v7 = i64.const 8\n\
         \x20 v8 = i64.mul v4 v7\n\
         \x20 v9 = i64.const 512\n\
         \x20 v10 = i64.add v9 v8\n\
         \x20 i64.store v10 v6\n\
         \x20 v11 = i64.const 1\n\
         \x20 v12 = i64.add v4 v11\n\
         \x20 br block1(v12)\n\
         block3():\n",
    );
    // Two rounds: spawn 4 workers of func `r` (1 then 2), storing thread handles at 256+…, then
    // join all 4 — the barrier between first-resumes and the migrating second-resumes.
    let mut v = 13;
    for r in 1..=2u32 {
        for w in 0..4 {
            root.push_str(&format!(
                "\x20 v{a} = i64.const 0\n\
                 \x20 v{b} = thread.spawn {r} v{a} v{a}\n\
                 \x20 v{c} = i64.const {addr}\n\
                 \x20 i32.store v{c} v{b}\n",
                a = v,
                b = v + 1,
                c = v + 2,
                addr = 256 + (r as u64 - 1) * 16 + w * 4,
            ));
            v += 3;
        }
        for w in 0..4 {
            root.push_str(&format!(
                "\x20 v{a} = i64.const {addr}\n\
                 \x20 v{b} = i32.load v{a}\n\
                 \x20 v{c} = thread.join v{b}\n",
                a = v,
                b = v + 1,
                c = v + 2,
                addr = 256 + (r as u64 - 1) * 16 + w * 4,
            ));
            v += 3;
        }
    }
    root.push_str(&format!(
        "\x20 v{a} = i64.const 128\n\
         \x20 v{b} = i64.atomic.load v{a}\n\
         \x20 return v{b}\n\
         }}\n",
        a = v,
        b = v + 1,
    ));
    // The fiber body: yield 3*sp+1 (parking RUNNABLE for round 2), then return 7*sp+2.
    root.push_str(&worker(16));
    root.push_str(&worker(20));
    root.push_str(
        "func (i64, i64) -> (i64) {\n\
         block0(v0: i64, v1: i64):\n\
         \x20 v2 = i64.const 3\n\
         \x20 v3 = i64.mul v0 v2\n\
         \x20 v4 = i64.const 1\n\
         \x20 v5 = i64.add v3 v4\n\
         \x20 v6 = suspend v5\n\
         \x20 v7 = i64.const 7\n\
         \x20 v8 = i64.mul v0 v7\n\
         \x20 v9 = i64.const 2\n\
         \x20 v10 = i64.add v8 v9\n\
         \x20 return v10\n\
         }\n",
    );
    let m = parse_module(&root).unwrap_or_else(|e| panic!("parse: {e:?}\n{root}"));
    verify_module(&m).expect("verify");
    // Σ over id in 0..16 of (3·id+1) + (7·id+2) = 10·(0+…+15) + 3·16 = 1200 + 48 = 1248.
    let reps = if cfg!(windows) { 10 } else { 30 };
    for _ in 0..reps {
        match compile_and_run(&m, 0, &[]).expect("jit") {
            JitOutcome::Returned(x) => assert_eq!(x, [1248], "steal-stress sum wrong"),
            other => panic!("unexpected {other:?}"),
        }
    }
    // And the interpreter (its real M:N pool) agrees on the invariant.
    let mut fuel = 100_000_000u64;
    assert_eq!(run(&m, 0, &[], &mut fuel), Ok(vec![Value::I64(1248)]));
}
