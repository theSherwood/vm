//! Equality harness for the bytecode engine's **§12 thread seam** (INTERP_PERF.md Slice 1c-5c):
//! `thread.spawn` / `thread.join` / `memory.wait` / `memory.notify`, run on the bytecode engine's
//! cooperative single-vCPU scheduler (one shared `Mem`).
//!
//! The reference oracle establishes concurrency correctness with **interleaving-invariant**
//! (determinate) programs — the same ones used here — so any *correct* schedule yields the same
//! result. We therefore compare the bytecode engine's result against the tree-walker `run` (the real
//! M:N executor): both must produce the determinate answer. The `.expect(Some)` on `compile_and_run`
//! gates that the bytecode engine actually drove the module (didn't fall back).

use svm_interp::{bytecode, run, Trap, Value};
use svm_text::parse_module;

fn check_threads(src: &str, want: Result<i64, Trap>) {
    let m = parse_module(src).expect("parse");
    let mut f_tw = 50_000_000u64;
    let tw = run(&m, 0, &[], &mut f_tw);
    let mut f_bc = 50_000_000u64;
    let bc = bytecode::compile_and_run(&m, 0, &[], &mut f_bc)
        .expect("bytecode engine must support thread ops (Slice 1c-5c)");
    // Both engines must agree (determinate program), and match the expected invariant.
    assert_eq!(tw, bc, "thread: tree-walker != bytecode\n{src}");
    let got = bc.map(|v| match v.first() {
        Some(Value::I64(x)) => *x,
        Some(Value::I32(x)) => *x as i64,
        other => panic!("unexpected result {other:?}"),
    });
    assert_eq!(got, want, "thread: result mismatch\n{src}");
}

/// Two threads each `atomic.rmw.add 1`; main joins both and reads the counter — always 2.
const TINY_ATOMIC: &str = r#"
memory 16
func () -> (i64) {
block0():
  vsp = i64.const 0
  va = i64.const 1
  vh0 = thread.spawn 1 vsp va
  vh1 = thread.spawn 1 vsp va
  vj0 = thread.join vh0
  vj1 = thread.join vh1
  vaddr = i64.const 0
  vr = i64.atomic.load vaddr
  return vr
}
func (i64, i64) -> (i64) {
block0(vsp: i64, varg: i64):
  vaddr = i64.const 0
  vrmw = i64.atomic.rmw.add vaddr varg
  vz = i64.const 0
  return vz
}
"#;

/// 8 vCPUs each `atomic.rmw.add` a shared counter 500× — total exactly 4000 on every interleaving.
const ATOMIC_COUNTER: &str = r#"
memory 16
func () -> (i64) {
block0():
  v0 = i64.const 0
  br block1(v0)
block1(v1: i64):
  v2 = i64.const 8
  v3 = i64.lt_u v1 v2
  br_if v3 block2(v1) block3()
block2(v4: i64):
  v5 = i64.const 500
  v6 = thread.spawn 1 v5 v5
  v7 = i64.const 4
  v8 = i64.mul v4 v7
  v9 = i64.const 16
  v10 = i64.add v9 v8
  i32.store v10 v6
  v11 = i64.const 1
  v12 = i64.add v4 v11
  br block1(v12)
block3():
  v13 = i64.const 0
  br block4(v13)
block4(v14: i64):
  v15 = i64.const 8
  v16 = i64.lt_u v14 v15
  br_if v16 block5(v14) block6()
block5(v17: i64):
  v18 = i64.const 4
  v19 = i64.mul v17 v18
  v20 = i64.const 16
  v21 = i64.add v20 v19
  v22 = i32.load v21
  v23 = thread.join v22
  v24 = i64.const 1
  v25 = i64.add v17 v24
  br block4(v25)
block6():
  v26 = i64.const 0
  v27 = i64.atomic.load v26
  return v27
}
func (i64, i64) -> (i64) {
block0(vsp: i64, v0: i64):
  br block1(v0)
block1(v1: i64):
  v2 = i64.const 0
  v3 = i64.eq v1 v2
  br_if v3 block2() block3(v1)
block3(v4: i64):
  v5 = i64.const 0
  v6 = i64.const 1
  v7 = i64.atomic.rmw.add v5 v6
  v8 = i64.const -1
  v9 = i64.add v4 v8
  br block1(v9)
block2():
  v10 = i64.const 0
  return v10
}
"#;

/// Futex handoff: producer writes payload at mem[8], spawns a consumer that `atomic.wait`s on the
/// flag at mem[0], then sets the flag and notifies. Consumer returns the payload (987654) on every
/// interleaving — exercises wait/notify + the not-equal fast path + the logical-clock timeout path.
const FUTEX_HANDOFF: &str = r#"
memory 16
func () -> (i64) {
block0():
  v0 = i64.const 8
  v1 = i64.const 987654
  i64.atomic.store.release v0 v1
  v2 = i64.const 0
  v3 = thread.spawn 1 v2 v2
  v4 = i64.const 0
  v5 = i32.const 1
  i32.atomic.store.release v4 v5
  v6 = i64.const 0
  v7 = i32.const 1
  v8 = atomic.notify v6 v7
  v9 = thread.join v3
  return v9
}
func (i64, i64) -> (i64) {
block0(vsp: i64, v0: i64):
  v1 = i64.const 0
  v2 = i32.const 0
  v3 = i64.const 1000000000
  v4 = i32.atomic.wait v1 v2 v3
  v5 = i64.const 8
  v6 = i64.atomic.load.acquire v5
  return v6
}
"#;

/// Joining a handle that was never spawned is an inert `ThreadFault` on both engines.
const FORGED_JOIN: &str = r#"
memory 16
func () -> (i64) {
block0():
  v0 = i32.const 0
  v1 = thread.join v0
  return v1
}
"#;

/// **Fiber migration** (D57): the root creates fiber 2 and resumes it once (it suspends, capturing
/// its first-resume arg 5); then it spawns a thread that resumes that *same* fiber on the other vCPU
/// with 7. The fiber computes `10*7 + 5 = 75` and returns it; the thread returns it; the root joins
/// and returns 75. Requires the **run-shared** fiber registry (Slice 1c-5f) — a per-vCPU registry
/// would `FiberFault` when the thread resumes a fiber it didn't create.
const MIGRATE: &str = r#"
func () -> (i64) {
block0():
  v0 = ref.func 2
  v1 = i64.const 4096
  v2 = cont.new v0 v1
  v3 = i64.const 5
  v4, v5 = cont.resume v2 v3
  v6 = i64.extend_i32_u v2
  v7 = thread.spawn 1 v6 v6
  v8 = thread.join v7
  return v8
}
func (i64, i64) -> (i64) {
block0(vsp: i64, varg: i64):
  v0 = i32.wrap_i64 varg
  v1 = i64.const 7
  v2, v3 = cont.resume v0 v1
  return v3
}
func (i64, i64) -> (i64) {
block0(vsp: i64, varg: i64):
  v0 = suspend varg
  v1 = i64.const 10
  v2 = i64.mul v0 v1
  v3 = i64.add v2 varg
  return v3
}
"#;

#[test]
fn threads_tiny_atomic() {
    check_threads(TINY_ATOMIC, Ok(2));
}

#[test]
fn threads_atomic_counter() {
    check_threads(ATOMIC_COUNTER, Ok(4000));
}

#[test]
fn threads_futex_handoff() {
    check_threads(FUTEX_HANDOFF, Ok(987654));
}

#[test]
fn threads_forged_join_faults_identically() {
    check_threads(FORGED_JOIN, Err(Trap::ThreadFault));
}

#[test]
fn threads_fiber_migration() {
    check_threads(MIGRATE, Ok(75));
}
