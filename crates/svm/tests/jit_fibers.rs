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
