//! `gc.roots` — conservative root enumeration for a guest GC (GC.md §3).
//!
//! The op scans every fiber of the domain (the caller's own live frames, the parked root, and every
//! registry fiber's frames) for the deduplicated set of candidate words that fall in the half-open
//! guest range `[heap_lo, heap_hi)`, writes the first `cap` of them (ascending) into guest memory at
//! `buf`, and returns the total found. **Both backends** implement it: the interpreter scans its
//! reified `Value` frames; the JIT walks the live native control stacks of its fibers (parked fibers'
//! saved extents + the running resume chain + the root computation's frames). Correctness here is
//! *soundness* (every live in-range root is reported), not interp↔JIT equality — the two backends
//! legitimately over-approximate differently (GC.md §3.2). Where the stack-switch substrate is absent
//! the JIT still bails `Unsupported` and the interpreter covers it.

use svm_encode::{decode_module, encode_module};
use svm_interp::{run, Value};
use svm_text::{parse_module, print_module};
use svm_verify::verify_module;

/// `gc.roots` survives text and binary round-trips (new `GC_ROOTS` opcode + the 4-operand/1-result
/// shape) so it flows through the whole pipeline like any other instruction.
#[test]
fn gc_roots_round_trips() {
    let src = "memory 16\n\
        func () -> (i64) {\n\
        block0():\n\
        \x20 v0 = i64.const 0\n\
        \x20 v1 = i64.const 4096\n\
        \x20 v2 = i64.const 0\n\
        \x20 v3 = i64.const 8\n\
        \x20 v4 = gc.roots v0 v1 v2 v3\n\
        \x20 return v4\n\
        }\n";
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");
    assert_eq!(
        parse_module(&print_module(&m)),
        Ok(m.clone()),
        "text round-trip"
    );
    assert_eq!(
        decode_module(&encode_module(&m)),
        Ok(m.clone()),
        "binary round-trip"
    );
}

fn run_i64s(src: &str) -> Vec<i64> {
    let m = parse_module(src).unwrap_or_else(|e| panic!("parse: {e:?}\n{src}"));
    verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}\n{src}"));
    let mut fuel = 1_000_000u64;
    run(&m, 0, &[], &mut fuel)
        .unwrap_or_else(|t| panic!("interp trapped: {t:?}\n{src}"))
        .into_iter()
        .map(|v| match v {
            Value::I64(x) => x,
            other => panic!("expected i64 result, got {other:?}"),
        })
        .collect()
}

/// A root word held in the **caller's own frame** is enumerated (the op is call-clobbering, so the
/// caller's live values are scannable). `heap_lo` is itself an in-range guest value, so the candidate
/// set is `{heap_lo, sentinel}` written ascending.
#[test]
fn gc_roots_finds_caller_frame_root() {
    // range [0x5000, 0x6000); sentinel 0x5050; buf at offset 0, cap 8 slots.
    let src = "memory 16\n\
        func () -> (i64, i64, i64) {\n\
        block0():\n\
        \x20 v0 = i64.const 20560\n\
        \x20 v1 = i64.const 20480\n\
        \x20 v2 = i64.const 24576\n\
        \x20 v3 = i64.const 0\n\
        \x20 v4 = i64.const 8\n\
        \x20 v5 = gc.roots v1 v2 v3 v4\n\
        \x20 v6 = i64.load v3\n\
        \x20 v7 = i64.const 8\n\
        \x20 v8 = i64.load v7\n\
        \x20 return v5 v6 v8\n\
        }\n";
    // candidates in [0x5000,0x6000): heap_lo 0x5000 and the sentinel 0x5050 ⇒ count 2, ascending.
    assert_eq!(run_i64s(src), vec![2, 0x5000, 0x5050]);
}

/// A word outside `[heap_lo, heap_hi)` is **not** reported. The sentinel 0x5050 is out of the
/// [0x8000,0x9000) range; only `heap_lo` (0x8000, itself in range) is enumerated.
#[test]
fn gc_roots_excludes_out_of_range() {
    let src = "memory 16\n\
        func () -> (i64, i64) {\n\
        block0():\n\
        \x20 v0 = i64.const 20560\n\
        \x20 v1 = i64.const 32768\n\
        \x20 v2 = i64.const 36864\n\
        \x20 v3 = i64.const 0\n\
        \x20 v4 = i64.const 8\n\
        \x20 v5 = gc.roots v1 v2 v3 v4\n\
        \x20 v6 = i64.load v3\n\
        \x20 return v5 v6\n\
        }\n";
    // Only heap_lo (0x8000) is in range; the 0x5050 sentinel is excluded ⇒ count 1.
    assert_eq!(run_i64s(src), vec![1, 0x8000]);
}

/// A root word held on a **suspended fiber's** stack is enumerated from the run-shared registry —
/// the part the guest cannot reach itself. The fiber keeps 0x7050 live across its `suspend`; the
/// root then calls `gc.roots` over [0x7000,0x7100) and the fiber's word shows up alongside `heap_lo`.
#[test]
fn gc_roots_scans_suspended_fiber_stack() {
    let src = "memory 16\n\
        func () -> (i64, i64) {\n\
        block0():\n\
        \x20 v0 = ref.func 1\n\
        \x20 v1 = i64.const 4096\n\
        \x20 v2 = cont.new v0 v1\n\
        \x20 v3 = i64.const 0\n\
        \x20 v4, v5 = cont.resume v2 v3\n\
        \x20 v6 = i64.const 28672\n\
        \x20 v7 = i64.const 28928\n\
        \x20 v8 = i64.const 0\n\
        \x20 v9 = i64.const 8\n\
        \x20 v10 = gc.roots v6 v7 v8 v9\n\
        \x20 v11 = i64.const 8\n\
        \x20 v12 = i64.load v11\n\
        \x20 return v10 v12\n\
        }\n\
        func (i64, i64) -> (i64) {\n\
        block0(v0: i64, v1: i64):\n\
        \x20 v2 = i64.const 28752\n\
        \x20 v3 = i64.const 0\n\
        \x20 v4 = suspend v3\n\
        \x20 return v2\n\
        }\n";
    // candidates in [0x7000,0x7100): heap_lo 0x7000 (root frame) and 0x7050 (fiber's live stack) ⇒
    // count 2, with the fiber word as the second (ascending) slot.
    assert_eq!(run_i64s(src), vec![2, 0x7050]);
}

/// On targets **without** the stack-switch substrate, `gc.roots` (like the other fiber ops) bails
/// `Unsupported` on the JIT — the interpreter covers it there.
#[cfg(not(any(
    all(unix, target_arch = "x86_64"),
    all(unix, target_arch = "aarch64"),
    all(windows, target_arch = "x86_64")
)))]
#[test]
fn gc_roots_is_unsupported_on_the_jit_without_fibers() {
    use svm_jit::{compile_and_run, JitError};
    let src = "memory 16\n\
        func () -> (i64) {\n\
        block0():\n\
        \x20 v1 = i64.const 0\n\
        \x20 v2 = i64.const 4096\n\
        \x20 v3 = i64.const 0\n\
        \x20 v4 = i64.const 8\n\
        \x20 v5 = gc.roots v1 v2 v3 v4\n\
        \x20 return v5\n\
        }\n";
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");
    assert!(
        matches!(compile_and_run(&m, 0, &[]), Err(JitError::Unsupported(_))),
        "the JIT must bail Unsupported on gc.roots where stack switching is absent"
    );
}

/// **`gc.roots` on the JIT** — a conservative native-stack walk over the live fiber stacks. The fiber
/// receives the heap pointer `0x7050` as its `arg` and keeps it live across its `suspend` (returning
/// it afterward), so the switch spills it onto the fiber's own control stack; the root then calls
/// `gc.roots` over `[0x7000, 0x8000)` and the JIT's walker must recover that word from the suspended
/// fiber's stack (the part the guest cannot reach itself). The pointer is a fiber *parameter* — which
/// regalloc cannot rematerialize, so it is genuinely on the fiber's stack across the suspend — and the
/// root passes it only as the (immediately-dead) `cont.resume` arg, never retaining it; so `0x7050`'s
/// presence in the result proves the fiber-stack scan specifically, not the root's own frame.
///
/// Soundness framing (GC.md §3.2), not interp↔JIT equality: the JIT over-approximates from raw stack
/// words, the interpreter from reified `Value` frames, so the exact candidate *set* legitimately
/// differs. We assert the live fiber root is reported and that out-of-window words don't flood in.
#[cfg(any(
    all(unix, target_arch = "x86_64"),
    all(unix, target_arch = "aarch64"),
    all(windows, target_arch = "x86_64")
))]
#[test]
fn gc_roots_scans_suspended_fiber_stack_on_the_jit() {
    use svm_jit::{compile_and_run, JitOutcome};
    // range [0x7000, 0x8000); the root passes arg 0x7050 (28752) to the fiber, which keeps it live
    // across its suspend and returns it; buf at offset 0, cap 8 slots.
    let src = "memory 16\n\
        func () -> (i64, i64, i64) {\n\
        block0():\n\
        \x20 v0 = ref.func 1\n\
        \x20 v1 = i64.const 4096\n\
        \x20 v2 = cont.new v0 v1\n\
        \x20 v3 = i64.const 28752\n\
        \x20 v4, v5 = cont.resume v2 v3\n\
        \x20 v6 = i64.const 28672\n\
        \x20 v7 = i64.const 32768\n\
        \x20 v8 = i64.const 0\n\
        \x20 v9 = i64.const 8\n\
        \x20 v10 = gc.roots v6 v7 v8 v9\n\
        \x20 v11 = i64.const 0\n\
        \x20 v12 = i64.load v11\n\
        \x20 v13 = i64.const 8\n\
        \x20 v14 = i64.load v13\n\
        \x20 return v10 v12 v14\n\
        }\n\
        func (i64, i64) -> (i64) {\n\
        block0(v0: i64, v1: i64):\n\
        \x20 v2 = i64.const 0\n\
        \x20 v3 = suspend v2\n\
        \x20 return v1\n\
        }\n";
    let m = parse_module(src).unwrap_or_else(|e| panic!("parse: {e:?}"));
    verify_module(&m).expect("verify");
    let slots = match compile_and_run(&m, 0, &[]).expect("jit compile/run") {
        JitOutcome::Returned(s) => s,
        other => panic!("jit did not return: {other:?}"),
    };
    let (count, buf0, buf1) = (slots[0], slots[1], slots[2]);
    assert!(
        count >= 1,
        "the suspended fiber's root must be found (count {count})"
    );
    assert!(
        count <= 2,
        "only in-window words (0x7000/0x7050) should be reported, got count {count} \
         (out-of-window stack words must be filtered)"
    );
    assert!(
        buf0 == 0x7050 || buf1 == 0x7050,
        "the fiber's live heap pointer 0x7050 must be enumerated; got buf=[{buf0:#x}, {buf1:#x}]"
    );
}

/// **End-to-end cross-vCPU stop-the-world root scan** (GC.md §2 + §3) — the motivating case the op
/// exists for. A *collector* vCPU (the root) enumerates the roots of a *mutator* fiber that is parked
/// on a **different** vCPU, over the domain-shared fiber table, synchronized by a real futex
/// handshake (the §2 STW barrier realized at N=2 with `memory.wait`/`memory.notify`):
///
/// 1. The root spawns a mutator vCPU (`thread.spawn`) and waits on a `READY` window flag.
/// 2. The mutator creates a fiber, resumes it once so it `suspend`s holding the heap pointer `0x7050`
///    (its `arg`, kept live across the suspend → spilled onto the fiber's control stack), then sets
///    `READY` + `notify`s and parks itself on a `GO` flag — alive but quiescent, the real STW state.
/// 3. The root, now under STW (the only other vCPU is parked, its fiber `RUNNABLE`), calls `gc.roots`
///    over `[0x7000, 0x8000)`. The walker must reach across to the *other vCPU's* parked fiber's stack
///    and recover `0x7050` — the part neither the guest nor the collector's own vCPU can see.
/// 4. The root releases the mutator (`GO` + `notify`) and `thread.join`s it.
///
/// `0x7050` lives **only** on the mutator fiber's control stack (the root passes it to nothing; the
/// mutator passes it only as the immediately-dead `cont.resume` arg), so its presence proves the
/// cross-vCPU scan. The §3.3 sanity check (refuse if another vCPU holds a *running* fiber) must **not**
/// fire here — the mutator's fiber is `RUNNABLE`, not `RUNNING` — proving it does not false-refuse a
/// legitimately-parked sibling. Both backends share the fiber registry/table across vCPUs, so this is
/// asserted on each (soundness per §3.2, not interp↔JIT equality).
#[cfg(any(
    all(unix, target_arch = "x86_64"),
    all(unix, target_arch = "aarch64"),
    all(windows, target_arch = "x86_64")
))]
#[test]
fn gc_roots_cross_vcpu_stop_the_world_scan() {
    // Window flags: READY at 128, GO at 136 (i32). gc.roots buffer at offset 0, cap 8. Heap range
    // [0x7000, 0x8000); the mutator fiber holds 0x7050 (28752). `sp` args are unused parameters.
    let src = "memory 16\n\
        func () -> (i64, i64, i64) {\n\
        block0():\n\
        \x20 v0 = i64.const 0\n\
        \x20 v1 = i64.const 0\n\
        \x20 v2 = thread.spawn 1 v0 v1\n\
        \x20 br block1(v2)\n\
        block1(v3: i32):\n\
        \x20 v4 = i64.const 128\n\
        \x20 v5 = i32.atomic.load.acquire v4\n\
        \x20 v6 = i32.const 0\n\
        \x20 v7 = i32.ne v5 v6\n\
        \x20 br_if v7 block3(v3) block2(v3)\n\
        block2(v8: i32):\n\
        \x20 v9 = i64.const 128\n\
        \x20 v10 = i32.const 0\n\
        \x20 v11 = i64.const 1000000000\n\
        \x20 v12 = i32.atomic.wait v9 v10 v11\n\
        \x20 br block1(v8)\n\
        block3(v13: i32):\n\
        \x20 v14 = i64.const 28672\n\
        \x20 v15 = i64.const 32768\n\
        \x20 v16 = i64.const 0\n\
        \x20 v17 = i64.const 8\n\
        \x20 v18 = gc.roots v14 v15 v16 v17\n\
        \x20 v19 = i64.const 136\n\
        \x20 v20 = i32.const 1\n\
        \x20 i32.atomic.store.release v19 v20\n\
        \x20 v21 = i64.const 136\n\
        \x20 v22 = i32.const 1\n\
        \x20 v23 = atomic.notify v21 v22\n\
        \x20 v24 = thread.join v13\n\
        \x20 v25 = i64.const 0\n\
        \x20 v26 = i64.load v25\n\
        \x20 v27 = i64.const 8\n\
        \x20 v28 = i64.load v27\n\
        \x20 return v18 v26 v28\n\
        }\n\
        func (i64, i64) -> (i64) {\n\
        block0(v0: i64, v1: i64):\n\
        \x20 v2 = ref.func 2\n\
        \x20 v3 = i64.const 0\n\
        \x20 v4 = cont.new v2 v3\n\
        \x20 v5 = i64.const 28752\n\
        \x20 v6, v7 = cont.resume v4 v5\n\
        \x20 v8 = i64.const 128\n\
        \x20 v9 = i32.const 1\n\
        \x20 i32.atomic.store.release v8 v9\n\
        \x20 v10 = i64.const 128\n\
        \x20 v11 = i32.const 1\n\
        \x20 v12 = atomic.notify v10 v11\n\
        \x20 br block1()\n\
        block1():\n\
        \x20 v13 = i64.const 136\n\
        \x20 v14 = i32.atomic.load.acquire v13\n\
        \x20 v15 = i32.const 0\n\
        \x20 v16 = i32.ne v14 v15\n\
        \x20 br_if v16 block3() block2()\n\
        block2():\n\
        \x20 v17 = i64.const 136\n\
        \x20 v18 = i32.const 0\n\
        \x20 v19 = i64.const 1000000000\n\
        \x20 v20 = i32.atomic.wait v17 v18 v19\n\
        \x20 br block1()\n\
        block3():\n\
        \x20 v21 = i64.const 0\n\
        \x20 return v21\n\
        }\n\
        func (i64, i64) -> (i64) {\n\
        block0(v0: i64, v1: i64):\n\
        \x20 v2 = i64.const 0\n\
        \x20 v3 = suspend v2\n\
        \x20 return v1\n\
        }\n";
    let m = parse_module(src).unwrap_or_else(|e| panic!("parse: {e:?}"));
    verify_module(&m).expect("verify");

    // Assert the cross-vCPU root is found on a given backend's (count, buf0, buf1) result.
    let check = |slots: &[i64], backend: &str| {
        let (count, buf0, buf1) = (slots[0], slots[1], slots[2]);
        assert!(
            count >= 1 && count <= 2,
            "{backend}: expected 1-2 in-window roots (0x7000/0x7050), got count {count}"
        );
        assert!(
            buf0 == 0x7050 || buf1 == 0x7050,
            "{backend}: the mutator fiber's heap pointer 0x7050 must be enumerated across vCPUs; \
             got buf=[{buf0:#x}, {buf1:#x}]"
        );
    };

    // Interpreter (M:N executor; shares the run registry across spawned vCPUs).
    let mut fuel = 100_000_000u64;
    let interp: Vec<i64> = run(&m, 0, &[], &mut fuel)
        .unwrap_or_else(|t| panic!("interp trapped: {t:?}"))
        .iter()
        .map(|v| match v {
            Value::I64(x) => *x,
            other => panic!("expected i64, got {other:?}"),
        })
        .collect();
    check(&interp, "interp");

    // JIT (real 1:1 OS threads; domain-shared fiber table).
    use svm_jit::{compile_and_run, JitOutcome};
    let jit = match compile_and_run(&m, 0, &[]).expect("jit compile/run") {
        JitOutcome::Returned(s) => s,
        other => panic!("jit did not return: {other:?}"),
    };
    check(&jit, "jit");
}
