//! Differential interp↔JIT tests for §12 **fibers** (`cont.new`/`cont.resume`/`suspend`).
//!
//! Fibers are single-threaded and cooperative, so a run is deterministic — which means the same
//! interp-vs-JIT differential oracle that guards the scalar slice applies directly here. The JIT now
//! lowers the fiber ops to its host fiber runtime (native stack switching via `svm-fiber`); for every
//! program below the JIT must produce exactly what the reference interpreter does.
//!
//! Stack switching exists on x86-64 unix, aarch64 unix, and x86-64 Windows today
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
        (Trap::FiberFault, TrapKind::FiberFault)
            | (Trap::MemoryFault, TrapKind::MemoryFault)
            | (Trap::DivByZero, TrapKind::DivByZero)
    )
}

/// Run `src` on both backends and assert they agree (results bit-for-bit, or the same trap kind).
fn assert_jit_matches_interp(src: &str) {
    let m = parse_module(src).unwrap_or_else(|e| panic!("parse: {e:?}\n{src}"));
    verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}\n{src}"));
    let mut fuel = 10_000_000u64;
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

/// A fiber `(i64 sp, i64 arg)` that `suspend`s its arg, then on the next resume adds 100 and returns.
/// Root: resume(10) -> (SUSPENDED, 10); resume(7) -> (RETURNED, 107).
#[test]
fn fiber_suspend_then_resume() {
    let src = "func () -> (i32, i64, i32, i64) {\n\
        block0():\n\
        \x20 v0 = ref.func 1\n\
        \x20 v1 = i64.const 4096\n\
        \x20 v2 = cont.new v0 v1\n\
        \x20 v3 = i64.const 10\n\
        \x20 v4, v5 = cont.resume v2 v3\n\
        \x20 v6 = i64.const 7\n\
        \x20 v7, v8 = cont.resume v2 v6\n\
        \x20 return v4 v5 v7 v8\n\
        }\n\
        func (i64, i64) -> (i64) {\n\
        block0(v0: i64, v1: i64):\n\
        \x20 v2 = suspend v1\n\
        \x20 v3 = i64.const 100\n\
        \x20 v4 = i64.add v2 v3\n\
        \x20 return v4\n\
        }\n";
    assert_jit_matches_interp(src);
}

/// A generator fiber yields 1, 2, 3 then returns 4; the root loops resuming it and sums every
/// delivered value (10) — repeated resume/suspend with the handle threaded as a block param.
#[test]
fn fiber_generator_loop() {
    let src = "func () -> (i64) {\n\
        block0():\n\
        \x20 v0 = ref.func 1\n\
        \x20 v1 = i64.const 4096\n\
        \x20 v2 = cont.new v0 v1\n\
        \x20 v3 = i64.const 0\n\
        \x20 br block1(v2, v3)\n\
        block1(v4: i32, v5: i64):\n\
        \x20 v6 = i64.const 0\n\
        \x20 v7, v8 = cont.resume v4 v6\n\
        \x20 v9 = i64.add v5 v8\n\
        \x20 br_if v7 block2(v9) block1(v4, v9)\n\
        block2(v10: i64):\n\
        \x20 return v10\n\
        }\n\
        func (i64, i64) -> (i64) {\n\
        block0(v0: i64, v1: i64):\n\
        \x20 v2 = i64.const 1\n\
        \x20 v3 = suspend v2\n\
        \x20 v4 = i64.const 2\n\
        \x20 v5 = suspend v4\n\
        \x20 v6 = i64.const 3\n\
        \x20 v7 = suspend v6\n\
        \x20 v8 = i64.const 4\n\
        \x20 return v8\n\
        }\n";
    assert_jit_matches_interp(src);
}

/// A three-level resume chain (root → A → B): B suspends back to A, A suspends back to root, then the
/// chain unwinds to completion. The decisive test for the yielder-stack nesting discipline — a
/// `suspend` must return to the *correct* resumer.
#[test]
fn fiber_nested_resume_chain() {
    let src = "func () -> (i64, i64) {\n\
        block0():\n\
        \x20 v0 = ref.func 1\n\
        \x20 v1 = i64.const 4096\n\
        \x20 v2 = cont.new v0 v1\n\
        \x20 v3 = i64.const 0\n\
        \x20 v4, v5 = cont.resume v2 v3\n\
        \x20 v6, v7 = cont.resume v2 v3\n\
        \x20 return v5 v7\n\
        }\n\
        func (i64, i64) -> (i64) {\n\
        block0(v0: i64, v1: i64):\n\
        \x20 v2 = ref.func 2\n\
        \x20 v3 = i64.const 8192\n\
        \x20 v4 = cont.new v2 v3\n\
        \x20 v5 = i64.const 0\n\
        \x20 v6, v7 = cont.resume v4 v5\n\
        \x20 v8 = suspend v7\n\
        \x20 v9, v10 = cont.resume v4 v5\n\
        \x20 return v10\n\
        }\n\
        func (i64, i64) -> (i64) {\n\
        block0(v0: i64, v1: i64):\n\
        \x20 v2 = i64.const 11\n\
        \x20 v3 = suspend v2\n\
        \x20 v4 = i64.const 22\n\
        \x20 return v4\n\
        }\n";
    assert_jit_matches_interp(src);
}

/// Resuming a fiber that already returned is inert on both backends → `FiberFault`.
#[test]
fn fiber_resume_after_return_traps() {
    let src = "func () -> (i64) {\n\
        block0():\n\
        \x20 v0 = ref.func 1\n\
        \x20 v1 = i64.const 4096\n\
        \x20 v2 = cont.new v0 v1\n\
        \x20 v3 = i64.const 1\n\
        \x20 v4, v5 = cont.resume v2 v3\n\
        \x20 v6, v7 = cont.resume v2 v3\n\
        \x20 return v7\n\
        }\n\
        func (i64, i64) -> (i64) {\n\
        block0(v0: i64, v1: i64):\n\
        \x20 return v1\n\
        }\n";
    assert_jit_matches_interp(src);
}

/// A `suspend` from the root computation (no running fiber) traps on both backends → `FiberFault`.
#[test]
fn fiber_root_suspend_traps() {
    let src = "func () -> (i64) {\n\
        block0():\n\
        \x20 v0 = i64.const 5\n\
        \x20 v1 = suspend v0\n\
        \x20 return v1\n\
        }\n";
    assert_jit_matches_interp(src);
}

/// **Fiber handle *values* are identical across backends** (the D57 3b-i gate). Historically the
/// interp's table held the root as slot 0 (first `cont.new` → handle 1) while the JIT ran the root
/// off-table (first handle 0) — a documented safe divergence (DESIGN §3a) the fiber fuzzer had to
/// work around. The run-shared interp registry holds only `cont.new`-created fibers, unifying the
/// namespace: handles are 0, 1, … on **both** backends and may flow into observable output. Pins
/// both the cross-backend agreement and the absolute values (so the namespace can't drift again).
#[test]
fn fiber_handle_values_match_across_backends() {
    let src = "func () -> (i64, i64, i32, i64) {\n\
        block0():\n\
        \x20 v0 = ref.func 1\n\
        \x20 v1 = i64.const 4096\n\
        \x20 v2 = cont.new v0 v1\n\
        \x20 v3 = cont.new v0 v1\n\
        \x20 v4 = i64.const 3\n\
        \x20 v5, v6 = cont.resume v3 v4\n\
        \x20 v7 = i64.extend_i32_u v2\n\
        \x20 v8 = i64.extend_i32_u v3\n\
        \x20 return v7 v8 v5 v6\n\
        }\n\
        func (i64, i64) -> (i64) {\n\
        block0(v0: i64, v1: i64):\n\
        \x20 return v1\n\
        }\n";
    // Cross-backend agreement (the differential)…
    assert_jit_matches_interp(src);
    // …and the absolute namespace: first handle 0, second 1, resume ran fiber 1 to completion.
    let m = parse_module(src).expect("parse");
    let mut fuel = 1_000_000u64;
    let got = run(&m, 0, &[], &mut fuel).expect("interp");
    assert_eq!(
        got,
        vec![Value::I64(0), Value::I64(1), Value::I32(1), Value::I64(3)],
        "the unified handle namespace must start at 0 and number densely"
    );
}

/// `cont.new` over a **wrong-type funcref** (the funcref names a function whose signature is not the
/// fiber entry type `(i64, i64) -> i64`) faults on first resume on *both* backends → `FiberFault`. The
/// verifier only checks the funcref is an `i32` (§12 — the type is a runtime use-site check), so this
/// path is reachable; both backends type-check lazily at first resume, and the fault is a *fiber* fault
/// (the forged-handle / dead / bomb family), not a generic `IndirectCallType`.
#[test]
fn fiber_wrong_type_funcref_traps() {
    let src = "func () -> (i32, i64) {\n\
        block0():\n\
        \x20 v0 = ref.func 1\n\
        \x20 v1 = i64.const 4096\n\
        \x20 v2 = cont.new v0 v1\n\
        \x20 v3 = i64.const 1\n\
        \x20 v4, v5 = cont.resume v2 v3\n\
        \x20 return v4 v5\n\
        }\n\
        func (i32) -> (i32) {\n\
        block0(v0: i32):\n\
        \x20 return v0\n\
        }\n";
    assert_jit_matches_interp(src);
}

/// A fiber whose body actually uses the **data stack** (its `sp`) and shared memory: it stores `arg`
/// to `mem[sp]`, suspends, then reloads and returns it — exercising the two-stack split end to end and
/// the `mem_base`/`sp` threading into the fiber entry.
#[test]
fn fiber_uses_data_stack_and_memory() {
    let src = "memory 16\n\
        func () -> (i64) {\n\
        block0():\n\
        \x20 v0 = ref.func 1\n\
        \x20 v1 = i64.const 1024\n\
        \x20 v2 = cont.new v0 v1\n\
        \x20 v3 = i64.const 777\n\
        \x20 v4, v5 = cont.resume v2 v3\n\
        \x20 v6 = i64.const 0\n\
        \x20 v7, v8 = cont.resume v2 v6\n\
        \x20 return v8\n\
        }\n\
        func (i64, i64) -> (i64) {\n\
        block0(v0: i64, v1: i64):\n\
        \x20 i64.store v0 v1\n\
        \x20 v2 = i64.const 0\n\
        \x20 v3 = suspend v2\n\
        \x20 v4 = i64.load v0\n\
        \x20 return v4\n\
        }\n";
    assert_jit_matches_interp(src);
}

/// W5 JIT/DWARF Stage 4c — the fiber-rooted backtrace. A fiber entry `func 1` calls helper `func 2`,
/// which `suspend`s; the root resumes the fiber once and **returns without completing it**, leaving it
/// parked. The host then walks the suspended fiber's control stack and symbolizes its guest frames.
#[test]
fn fiber_backtrace_walks_a_suspended_fibers_guest_stack() {
    use svm_ir::DEFAULT_RESERVED_LOG2;
    use svm_jit::{CompiledModule, JitOutcome, Quota, INERT_CAP_THUNK};

    // `func 1` (entry) calls `func 2` (helper) at fib.c:5; the helper `suspend`s at fib.c:9.
    let src = r#"
func () -> (i64) {
block0():
  v0 = ref.func 1
  v1 = i64.const 4096
  v2 = cont.new v0 v1
  v3 = i64.const 5
  v4, v5 = cont.resume v2 v3
  return v5
}

func (i64, i64) -> (i64) {
block0(v0: i64, v1: i64):
  v2 = call 2(v0, v1)
  return v2
}

func (i64, i64) -> (i64) {
block0(v0: i64, v1: i64):
  v2 = suspend v1
  return v2
}

debug.file 0 "fib.c"
debug.loc 1 0 0 0 5 3
debug.loc 2 0 0 0 9 3
"#;
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");
    let mut cm = CompiledModule::compile(
        &m,
        0,
        INERT_CAP_THUNK,
        core::ptr::null_mut(),
        DEFAULT_RESERVED_LOG2,
        None,
        None,
        None,
        None,
        Quota::default(),
        0,
    )
    .expect("compile");

    // Before the run no fiber has been created, so there is nothing to walk.
    assert!(
        cm.fiber_backtrace(0).is_empty(),
        "no parked fiber before the run"
    );

    // The root resumes the fiber (it suspends) and returns its yielded value, leaving fiber 0 parked.
    let (outcome, _) = cm.run(&[], None, None, None).expect("run");
    let JitOutcome::Returned(ref v) = outcome else {
        panic!("expected the root to return, got {outcome:?}");
    };
    assert_eq!(
        v.as_slice(),
        &[5i64],
        "root returned the fiber's suspended value, leaving it parked"
    );

    // The fiber-rooted walk: the parked fiber's guest call stack, innermost frame first.
    let bt = cm.fiber_backtrace(0);
    let frames: Vec<(u32, &str, u32)> = bt
        .iter()
        .map(|f| (f.func, f.file.as_str(), f.line))
        .collect();
    assert_eq!(
        frames,
        vec![(2, "fib.c", 9), (1, "fib.c", 5)],
        "backtrace is [helper (suspended, innermost), entry (its caller)]"
    );
}

// ---- Recycling ABA guard, cross-backend (DURABILITY.md §12.8 steps 1/3) ----
//
// A fiber guest handle carries a generation in its high bits (`FIBER_GEN_SHIFT == 16`), and
// `cont.resume` rejects a handle whose generation doesn't match the slot's current one — the ABA
// guard that makes slot recycling safe. The interpreter pins this directly
// (`svm-durable/tests/fiber.rs::{forged_fiber_generation_is_rejected,
// recycling_reuses_a_freed_slot_with_a_bumped_generation}`) and the JIT registry has a unit test
// (`fiber_registry::claim_gen_rejects_a_stale_generation`); these pin the **wired** JIT path
// (`cont.resume` → `fiber_resume` → `resolve` + `claim_gen`) against the interpreter oracle, so the
// generation guard is enforced identically end-to-end on both backends.

/// A genuine handle (slot 0, generation 0) resumes; a forged generation-1 handle for the same slot
/// (`(1 << 16) | 0 == 65536`, which the slot mask clamps back to slot 0) faults — on both backends.
#[test]
fn fiber_forged_generation_faults_identically() {
    // Genuine handle: the fiber runs and returns 99.
    assert_jit_matches_interp(
        "func () -> (i64) {\n\
        block0():\n\
        \x20 v0 = ref.func 1\n\
        \x20 v1 = i64.const 4096\n\
        \x20 v2 = cont.new v0 v1\n\
        \x20 v4 = i64.const 0\n\
        \x20 v5, v6 = cont.resume v2 v4\n\
        \x20 return v6\n\
        }\n\
        func (i64, i64) -> (i64) {\n\
        block0(v0: i64, v1: i64):\n\
        \x20 v2 = i64.const 99\n\
        \x20 return v2\n\
        }\n",
    );
    // Forged handle `(1 << 16) | 0`: same slot 0, generation 1 ≠ 0 ⇒ FiberFault on both backends.
    assert_jit_matches_interp(
        "func () -> (i64) {\n\
        block0():\n\
        \x20 v0 = ref.func 1\n\
        \x20 v1 = i64.const 4096\n\
        \x20 v2 = cont.new v0 v1\n\
        \x20 v3 = i32.const 65536\n\
        \x20 v4 = i64.const 0\n\
        \x20 v5, v6 = cont.resume v3 v4\n\
        \x20 return v6\n\
        }\n\
        func (i64, i64) -> (i64) {\n\
        block0(v0: i64, v1: i64):\n\
        \x20 v2 = i64.const 99\n\
        \x20 return v2\n\
        }\n",
    );
}

/// A finished fiber's slot is recycled at a bumped generation: the next `cont.new` reuses slot 0 at
/// generation 1 (handle `(1 << 16) | 0 == 65536`), and the *stale* generation-0 handle to the former
/// occupant then faults even though slot 0 is live. Both backends must agree on each.
#[test]
fn recycled_slot_generation_guard_agrees() {
    // Fiber A (slot 0, gen 0) finishes; the next cont.new reuses slot 0 at gen 1 — returning the i32
    // handle makes the reuse observable (65536). Both backends must produce the same handle.
    assert_jit_matches_interp(
        "func () -> (i32) {\n\
        block0():\n\
        \x20 v0 = ref.func 1\n\
        \x20 v1 = i64.const 4096\n\
        \x20 v2 = cont.new v0 v1\n\
        \x20 v3 = i64.const 0\n\
        \x20 v4, v5 = cont.resume v2 v3\n\
        \x20 v6 = cont.new v0 v1\n\
        \x20 return v6\n\
        }\n\
        func (i64, i64) -> (i64) {\n\
        block0(v0: i64, v1: i64):\n\
        \x20 v2 = i64.const 7\n\
        \x20 return v2\n\
        }\n",
    );
    // After slot 0 is recycled (now gen 1), resuming A's stale gen-0 handle (i32 0) must fault on
    // both backends — even though slot 0 is live — because the generation no longer matches.
    assert_jit_matches_interp(
        "func () -> (i64) {\n\
        block0():\n\
        \x20 v0 = ref.func 1\n\
        \x20 v1 = i64.const 4096\n\
        \x20 v2 = cont.new v0 v1\n\
        \x20 v3 = i64.const 0\n\
        \x20 v4, v5 = cont.resume v2 v3\n\
        \x20 v6 = cont.new v0 v1\n\
        \x20 v9 = i32.const 0\n\
        \x20 v7, v8 = cont.resume v9 v3\n\
        \x20 return v8\n\
        }\n\
        func (i64, i64) -> (i64) {\n\
        block0(v0: i64, v1: i64):\n\
        \x20 v2 = i64.const 7\n\
        \x20 return v2\n\
        }\n",
    );
}
