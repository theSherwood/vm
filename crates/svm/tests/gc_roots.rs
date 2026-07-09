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

/// `gc.roots` survives text and binary round-trips (new `GC_ROOTS` opcode + the 5-operand/1-result
/// shape: `heap_lo heap_hi mask buf cap`) so it flows through the whole pipeline like any other
/// instruction. `mask = -1` is the untagged (identity) payload mask.
#[test]
fn gc_roots_round_trips() {
    let src = "memory 16\n\
        func () -> (i64) {\n\
        block0():\n\
        \x20 v0 = i64.const 0\n\
        \x20 v1 = i64.const 4096\n\
        \x20 vmask = i64.const -1\n\
        \x20 v2 = i64.const 0\n\
        \x20 v3 = i64.const 8\n\
        \x20 v4 = gc.roots v0 v1 vmask v2 v3\n\
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
        \x20 vmask = i64.const -1\n\
        \x20 v5 = gc.roots v1 v2 vmask v3 v4\n\
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
        \x20 vmask = i64.const -1\n\
        \x20 v5 = gc.roots v1 v2 vmask v3 v4\n\
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
        \x20 vmask = i64.const -1\n\
        \x20 v10 = gc.roots v6 v7 vmask v8 v9\n\
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
        \x20 vmask = i64.const -1\n\
        \x20 v5 = gc.roots v1 v2 vmask v3 v4\n\
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
    // One-word heap window [0x7050, 0x7051): only the exact pointer 0x7050 can match, so the
    // conservative scan's result is deterministic across platforms (a stray stack word equal to
    // exactly 0x7050 is ~2^-64; dedup caps the count at 1). We assert *soundness* — the live fiber
    // root is reported — not an occupancy upper bound (GC.md §3.2: the JIT deliberately
    // over-approximates from raw stack words, so its candidate count is not something to bound).
    // The root passes arg 0x7050 (28752) to the fiber, which keeps it live across its suspend and
    // returns it; buf at offset 0, cap 8 slots.
    let src = "memory 16\n\
        func () -> (i64, i64, i64) {\n\
        block0():\n\
        \x20 v0 = ref.func 1\n\
        \x20 v1 = i64.const 4096\n\
        \x20 v2 = cont.new v0 v1\n\
        \x20 v3 = i64.const 28752\n\
        \x20 v4, v5 = cont.resume v2 v3\n\
        \x20 v6 = i64.const 28752\n\
        \x20 v7 = i64.const 28753\n\
        \x20 v8 = i64.const 0\n\
        \x20 v9 = i64.const 8\n\
        \x20 vmask = i64.const -1\n\
        \x20 v10 = gc.roots v6 v7 vmask v8 v9\n\
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
    let (count, buf0) = (slots[0], slots[1]);
    assert!(
        count >= 1,
        "the suspended fiber's root must be found (count {count})"
    );
    assert!(
        buf0 == 0x7050,
        "the fiber's live heap pointer 0x7050 must be enumerated; got buf0={buf0:#x}"
    );
}

/// **`gc.roots` scans a running *ancestor* fiber** — the path the tight running-scan
/// (`fiber_rt::gc_roots`; STACK_GUARD_FLIP.md #4) validates. Chain: root → A → B, all RUNNING; B calls
/// `gc.roots`. B is the innermost (scanned via `current_sp`); A is a running *ancestor* (scanned via
/// its child B's `resumer_sp` — the SP A saved when it resumed B). A holds the heap pointer `0x7ab8`
/// (its resume-arg *parameter*, so regalloc can't rematerialize it) live across resuming B by storing
/// it afterward, so `0x7ab8` sits on A's stack above A's saved SP. B never sees it (its arg is 0) and
/// the root passes it only as A's immediately-dead resume arg — so `0x7ab8`'s presence proves the
/// *ancestor* scan specifically. `heap_lo` (`0x7ab0`) lives only on B, proving the innermost scan;
/// finding both in one scan pins the whole running-chain walk (a too-tight ancestor bound would drop
/// `0x7ab8`).
#[cfg(any(
    all(unix, target_arch = "x86_64"),
    all(unix, target_arch = "aarch64"),
    all(windows, target_arch = "x86_64")
))]
#[test]
fn gc_roots_scans_running_ancestor_fiber_on_the_jit() {
    use svm_jit::{compile_and_run, JitOutcome};
    // 16-byte heap window [0x7ab0, 0x7ac0): only 0x7ab0 (heap_lo, on B) and 0x7ab8 (P_A, on A) fall in
    // it (a stray stack word in a 16-byte range is ~2^-60). buf at 0, cap 4. A stores P_A at offset
    // 0x1000 — outside both the buffer and the heap window — purely to keep it live past the resume.
    let src = "memory 16\n\
        func () -> (i64, i64, i64) {\n\
        block0():\n\
        \x20 v0 = ref.func 1\n\
        \x20 v1 = i64.const 4096\n\
        \x20 v2 = cont.new v0 v1\n\
        \x20 v3 = i64.const 31416\n\
        \x20 v4, v5 = cont.resume v2 v3\n\
        \x20 v6 = i64.const 0\n\
        \x20 v7 = i64.load v6\n\
        \x20 v8 = i64.const 8\n\
        \x20 v9 = i64.load v8\n\
        \x20 return v5 v7 v9\n\
        }\n\
        func (i64, i64) -> (i64) {\n\
        block0(v0: i64, v1: i64):\n\
        \x20 v2 = ref.func 2\n\
        \x20 v3 = i64.const 4096\n\
        \x20 v4 = cont.new v2 v3\n\
        \x20 v5 = i64.const 0\n\
        \x20 v6, v7 = cont.resume v4 v5\n\
        \x20 v8 = i64.const 4096\n\
        \x20 i64.store v8 v1\n\
        \x20 return v7\n\
        }\n\
        func (i64, i64) -> (i64) {\n\
        block0(v0: i64, v1: i64):\n\
        \x20 v2 = i64.const 31408\n\
        \x20 v3 = i64.const 31424\n\
        \x20 v4 = i64.const 0\n\
        \x20 v5 = i64.const 4\n\
        \x20 vmask = i64.const -1\n\
        \x20 v6 = gc.roots v2 v3 vmask v4 v5\n\
        \x20 return v6\n\
        }\n";
    let m = parse_module(src).unwrap_or_else(|e| panic!("parse: {e:?}"));
    verify_module(&m).expect("verify");
    let slots = match compile_and_run(&m, 0, &[]).expect("jit compile/run") {
        JitOutcome::Returned(s) => s,
        other => panic!("jit did not return: {other:?}"),
    };
    let (count, buf0, buf1) = (slots[0], slots[1], slots[2]);
    assert!(
        count >= 2,
        "both the caller (B) and ancestor (A) roots must be found (count {count})"
    );
    assert_eq!(
        buf0, 0x7ab0,
        "heap_lo 0x7ab0 (on innermost B) must be enumerated; got {buf0:#x}"
    );
    assert_eq!(
        buf1, 0x7ab8,
        "running ancestor A's live pointer 0x7ab8 must be enumerated; got {buf1:#x}"
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
    // Window flags: READY at 128, GO at 136 (i32). gc.roots buffer at offset 0, cap 8. One-word
    // heap window [0x7050, 0x7051) so the conservative scan's result is deterministic (only the
    // exact pointer matches; soundness per GC.md §3.2, not an occupancy bound). The mutator fiber
    // holds 0x7050 (28752). `sp` args are unused parameters.
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
        \x20 v14 = i64.const 28752\n\
        \x20 v15 = i64.const 28753\n\
        \x20 v16 = i64.const 0\n\
        \x20 v17 = i64.const 8\n\
        \x20 vmask = i64.const -1\n\
        \x20 v18 = gc.roots v14 v15 vmask v16 v17\n\
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

    // Assert the cross-vCPU root is found on a given backend's (count, buf0, _) result. Soundness
    // only (GC.md §3.2): the one-word window makes the result deterministic without bounding the
    // conservative candidate count.
    let check = |slots: &[i64], backend: &str| {
        let (count, buf0) = (slots[0], slots[1]);
        assert!(
            count >= 1,
            "{backend}: the mutator fiber's root must be found (count {count})"
        );
        assert!(
            buf0 == 0x7050,
            "{backend}: the mutator fiber's heap pointer 0x7050 must be enumerated across vCPUs; \
             got buf0={buf0:#x}"
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

/// How many `gc.roots` result words the runner reads back out of captured guest memory (= the `cap`
/// the root passes). With ≤6 in-window roots plus the `heap_lo` word, every in-window candidate is
/// well within this; the unused tail of the buffer stays zero.
#[cfg(any(
    all(unix, target_arch = "x86_64"),
    all(unix, target_arch = "aarch64"),
    all(windows, target_arch = "x86_64")
))]
const BUF_WORDS: usize = 16;

/// Build a module whose root creates one parked fiber per value in `roots` (each fiber holds its
/// value live across a `suspend`, so it is spilled onto that fiber's control stack / lives in its
/// reified frame), then calls `gc.roots` over `[lo, hi)` writing the deduped candidates to guest
/// memory at offset 0 (cap `BUF_WORDS`) and **returns just the count**. The result buffer is read
/// back from *captured memory* (see [`run_multi_fiber_words`]), not returned — the guest's return
/// arity stays at 1, since the Tail ABI's return-register budget differs across targets (8 values
/// fit on x86-64 but not aarch64). Shared fiber body (func 1) is `(sp, arg) -> arg` across a suspend.
#[cfg(any(
    all(unix, target_arch = "x86_64"),
    all(unix, target_arch = "aarch64"),
    all(windows, target_arch = "x86_64")
))]
fn build_multi_fiber_module(roots: &[i64], lo: i64, hi: i64) -> svm_ir::Module {
    use std::fmt::Write as _;
    let mut src = String::from("memory 16\nfunc () -> (i64) {\nblock0():\n");
    let mut v: u32 = 0;
    for &r in roots {
        // ref.func 1; sp (unused param); cont.new; arg = the root; resume → fiber parks holding arg.
        writeln!(src, "  v{v} = ref.func 1").unwrap();
        writeln!(src, "  v{} = i64.const 4096", v + 1).unwrap();
        writeln!(src, "  v{} = cont.new v{v} v{}", v + 2, v + 1).unwrap();
        writeln!(src, "  v{} = i64.const {r}", v + 3).unwrap();
        writeln!(
            src,
            "  v{}, v{} = cont.resume v{} v{}",
            v + 4,
            v + 5,
            v + 2,
            v + 3
        )
        .unwrap();
        v += 6;
    }
    writeln!(src, "  v{v} = i64.const {lo}").unwrap();
    writeln!(src, "  v{} = i64.const {hi}", v + 1).unwrap();
    writeln!(src, "  v{} = i64.const 0", v + 2).unwrap(); // buf offset
    writeln!(src, "  v{} = i64.const {BUF_WORDS}", v + 3).unwrap(); // cap
    writeln!(src, "  v{} = i64.const -1", v + 4).unwrap(); // payload mask (untagged identity)
    writeln!(
        src,
        "  v{} = gc.roots v{v} v{} v{} v{} v{}",
        v + 5,
        v + 1,
        v + 4,
        v + 2,
        v + 3
    )
    .unwrap();
    writeln!(src, "  return v{}", v + 5).unwrap();
    src.push_str("}\n");
    // Shared fiber body: keep `arg` (v1) live across the suspend, then return it.
    src.push_str("func (i64, i64) -> (i64) {\nblock0(v0: i64, v1: i64):\n");
    src.push_str("  v2 = i64.const 0\n  v3 = suspend v2\n  return v1\n}\n");

    let m = parse_module(&src).unwrap_or_else(|e| panic!("parse: {e:?}\n{src}"));
    verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}\n{src}"));
    m
}

/// Run `build_multi_fiber_module`'s output on both backends and return the `gc.roots` result buffer
/// (the first `BUF_WORDS` words written to guest memory at offset 0) per backend — read from
/// **captured memory**, so it doesn't depend on the guest's return arity. Soundness framing (GC.md
/// §3.2): each backend is checked independently against the planted roots, not against each other.
#[cfg(any(
    all(unix, target_arch = "x86_64"),
    all(unix, target_arch = "aarch64"),
    all(windows, target_arch = "x86_64")
))]
fn run_multi_fiber_words(m: &svm_ir::Module) -> [Vec<i64>; 2] {
    fn words(mem: &[u8]) -> Vec<i64> {
        (0..BUF_WORDS)
            .map(|i| i64::from_le_bytes(mem[i * 8..i * 8 + 8].try_into().unwrap()))
            .collect()
    }
    // Seed `BUF_WORDS` words of zeroed memory: the capture APIs snapshot exactly `init_mem.len()`
    // post-run bytes (escape-oracle convention), so this is what makes the gc.roots buffer at
    // offset 0 visible (its unwritten tail stays zero, which never matches a planted sentinel).
    let init_mem = [0u8; BUF_WORDS * 8];
    // Interp (shares the run registry across spawned vCPUs); capture the final guest memory.
    let mut fuel = 50_000_000u64;
    let (res, imem) = svm_interp::run_capture(m, 0, &[], &mut fuel, &init_mem);
    res.unwrap_or_else(|t| panic!("interp trapped: {t:?}"));
    // JIT (domain-shared fiber table); capture the final guest memory.
    let (outcome, jmem) =
        svm_jit::compile_and_run_capture(m, 0, &[], &init_mem).expect("jit compile/run");
    assert!(
        matches!(outcome, svm_jit::JitOutcome::Returned(_)),
        "jit did not return: {outcome:?}"
    );
    [words(&imem), words(&jmem)]
}

/// **`gc.roots` enumerates the roots of EVERY parked fiber, not just one.** Four fibers each park
/// holding a distinct in-window pointer (`0x7010`..`0x7040`); a fifth parks holding an out-of-window
/// value (`0x9000_0000`). Scanning `[0x7000, 0x7100)` must report every in-window pointer (soundness
/// — a superset is fine, GC.md §3.2) and must filter the out-of-window one. This exercises the
/// all-slots table walk + dedup + range filter across many fibers — the logic a single-fiber test
/// cannot reach. Asserted independently on each backend (JIT walks native stacks; interp walks
/// reified frames).
#[cfg(any(
    all(unix, target_arch = "x86_64"),
    all(unix, target_arch = "aarch64"),
    all(windows, target_arch = "x86_64")
))]
#[test]
fn gc_roots_enumerates_every_parked_fiber() {
    const IN: [i64; 4] = [0x7010, 0x7020, 0x7030, 0x7040];
    const OOR: i64 = 0x9000_0000;
    let m = build_multi_fiber_module(&[IN[0], IN[1], IN[2], IN[3], OOR], 0x7000, 0x7100);
    for (backend, words) in ["interp", "jit"].iter().zip(run_multi_fiber_words(&m)) {
        for &r in &IN {
            assert!(
                words.contains(&r),
                "{backend}: in-window root {r:#x} from a parked fiber was not enumerated; \
                 got {words:#x?}"
            );
        }
        assert!(
            !words.contains(&OOR),
            "{backend}: out-of-window value {OOR:#x} must be filtered out; got {words:#x?}"
        );
    }
}

/// **Randomized multi-fiber soundness sweep.** Over many seeds, park a random number (1..=6) of
/// fibers holding distinct in-window pointers plus a few out-of-window values, then assert `gc.roots`
/// reports *every* planted in-window root and *no* out-of-window value — on both backends. A
/// self-contained, hang-proof fuzz (pure fibers, no threads/recursion) that stresses the table walk,
/// dedup, and range filter across varied fiber counts without entangling the `fiber_fuzz` harness.
/// Soundness only (a superset is allowed, §3.2), so conservative false positives never flake it.
#[cfg(any(
    all(unix, target_arch = "x86_64"),
    all(unix, target_arch = "aarch64"),
    all(windows, target_arch = "x86_64")
))]
#[test]
fn gc_roots_multi_fiber_soundness_sweep() {
    // Tiny deterministic PRNG (xorshift64*), so any failure replays exactly.
    let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = || {
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        s.wrapping_mul(0x2545_F491_4F6C_DD1D)
    };

    // In-window pointers live in [0x7000, 0x8000); out-of-window decoys sit just below/above it.
    const LO: i64 = 0x7000;
    const HI: i64 = 0x8000;
    let mut checked = 0u64;
    for _ in 0..64 {
        let n_in = 1 + (next() % 6) as usize; // 1..=6 in-window roots (≤7 in-range words incl. heap_lo → all in the 8 read-back slots)
                                              // Distinct in-window pointers, 16-byte spaced so they never collide and stay < HI.
        let in_roots: Vec<i64> = (0..n_in).map(|i| LO + 0x10 * (i as i64 + 1)).collect();
        // 0..=2 out-of-window decoys (below LO and above HI).
        let n_oor = (next() % 3) as usize;
        let oor: Vec<i64> = (0..n_oor)
            .map(|i| (if i % 2 == 0 { LO - 0x100 } else { HI + 0x100 }) + i as i64)
            .collect();

        let mut roots = in_roots.clone();
        roots.extend(&oor);
        // Shuffle creation order so the planted roots aren't enumerated in table order by accident.
        for i in (1..roots.len()).rev() {
            roots.swap(i, (next() as usize) % (i + 1));
        }

        let m = build_multi_fiber_module(&roots, LO, HI);
        for (backend, words) in ["interp", "jit"].iter().zip(run_multi_fiber_words(&m)) {
            for &r in &in_roots {
                assert!(
                    words.contains(&r),
                    "{backend}: in-window root {r:#x} missing (in={in_roots:#x?}); got {words:#x?}"
                );
            }
            for &d in &oor {
                assert!(
                    !words.contains(&d),
                    "{backend}: out-of-window decoy {d:#x} leaked; got {words:#x?}"
                );
            }
        }
        checked += 1;
    }
    assert!(
        checked == 64,
        "expected to check 64 random multi-fiber configs"
    );
}

// ---- §GC payload mask: tagged-pointer roots + the host-leak safety constraint -------------------

/// The tagged-pointer constant a guest holds as a live root: tag `0x03` in the top byte, bare window
/// offset `0x7050` below. A naive scan (mask `-1`) would see the *huge* tagged word and reject it as
/// out-of-window; the §GC payload mask strips the tag so the bare offset is recovered.
const TAGGED_7050: i64 = (0x03 << 56) | 0x7050; // 0x0300_0000_0000_7050
/// Top-byte-strip mask (`0x00FF_FFFF_FFFF_FFFF`) — clears the tag byte, preserves the low 56 bits.
const STRIP_MASK: i64 = 0x00FF_FFFF_FFFF_FFFF;
/// A **fold-down** mask that keeps only the low 24 bits (`0x00FF_FFFF`): it would pull a host pointer
/// down into the guest window and leak host-address bits. The verifier + both runtimes reject it.
const FOLD_DOWN_MASK: i64 = 0x00FF_FFFF;

/// **The payload mask recovers a tagged root** (interpreter). A fiber parks holding the tagged word
/// `TAGGED_7050` live across its `suspend` (so it sits in the reified frame the run-shared registry
/// exposes). The root scans `[0x7000, 0x7100)` with `STRIP_MASK`: the tag byte is stripped, so the
/// bare offset `0x7050` is enumerated alongside `heap_lo` (0x7000, which masks to itself). The raw
/// tagged word never appears — only its masked offset crosses the boundary. (With mask `-1` the
/// tagged word is huge and excluded, so the count would be 1; the recovered `0x7050` is the proof.)
#[test]
fn gc_roots_strips_pointer_tag_via_mask() {
    let src = format!(
        "memory 16\n\
        func () -> (i64, i64, i64) {{\n\
        block0():\n\
        \x20 v0 = ref.func 1\n\
        \x20 v1 = i64.const 4096\n\
        \x20 v2 = cont.new v0 v1\n\
        \x20 v3 = i64.const {TAGGED_7050}\n\
        \x20 v4, v5 = cont.resume v2 v3\n\
        \x20 v6 = i64.const 28672\n\
        \x20 v7 = i64.const 28928\n\
        \x20 vmask = i64.const {STRIP_MASK}\n\
        \x20 v8 = i64.const 0\n\
        \x20 v9 = i64.const 8\n\
        \x20 v10 = gc.roots v6 v7 vmask v8 v9\n\
        \x20 v11 = i64.const 0\n\
        \x20 v12 = i64.load v11\n\
        \x20 v13 = i64.const 8\n\
        \x20 v14 = i64.load v13\n\
        \x20 return v10 v12 v14\n\
        }}\n\
        func (i64, i64) -> (i64) {{\n\
        block0(v0: i64, v1: i64):\n\
        \x20 v2 = i64.const 0\n\
        \x20 v3 = suspend v2\n\
        \x20 return v1\n\
        }}\n"
    );
    // {heap_lo 0x7000, masked tagged root 0x7050} ⇒ count 2, ascending; the raw tagged word (huge)
    // is filtered, so only the bare offset 0x7050 lands in the buffer.
    assert_eq!(run_i64s(&src), vec![2, 0x7000, 0x7050]);
}

/// **The payload mask recovers a tagged root on the JIT** (native-stack scan). The fiber receives the
/// tagged word as its `arg` and keeps it live across `suspend`, so the switch spills the *tagged*
/// word onto the fiber's control stack. Scanning the one-word window `[0x7050, 0x7051)` with
/// `STRIP_MASK`, the JIT's walker masks that spilled word down to `0x7050` and reports it — proving
/// the mask is applied to raw stack words too (with mask `-1` the huge tagged word can't match the
/// window, so its recovery here is the proof). Soundness framing, not interp↔JIT equality (§3.2).
#[cfg(any(
    all(unix, target_arch = "x86_64"),
    all(unix, target_arch = "aarch64"),
    all(windows, target_arch = "x86_64")
))]
#[test]
fn gc_roots_strips_pointer_tag_on_the_jit() {
    use svm_jit::{compile_and_run, JitOutcome};
    let src = format!(
        "memory 16\n\
        func () -> (i64, i64) {{\n\
        block0():\n\
        \x20 v0 = ref.func 1\n\
        \x20 v1 = i64.const 4096\n\
        \x20 v2 = cont.new v0 v1\n\
        \x20 v3 = i64.const {TAGGED_7050}\n\
        \x20 v4, v5 = cont.resume v2 v3\n\
        \x20 v6 = i64.const 28752\n\
        \x20 v7 = i64.const 28753\n\
        \x20 vmask = i64.const {STRIP_MASK}\n\
        \x20 v8 = i64.const 0\n\
        \x20 v9 = i64.const 8\n\
        \x20 v10 = gc.roots v6 v7 vmask v8 v9\n\
        \x20 v11 = i64.const 0\n\
        \x20 v12 = i64.load v11\n\
        \x20 return v10 v12\n\
        }}\n\
        func (i64, i64) -> (i64) {{\n\
        block0(v0: i64, v1: i64):\n\
        \x20 v2 = i64.const 0\n\
        \x20 v3 = suspend v2\n\
        \x20 return v1\n\
        }}\n"
    );
    let m = parse_module(&src).unwrap_or_else(|e| panic!("parse: {e:?}"));
    verify_module(&m).expect("verify");
    let slots = match compile_and_run(&m, 0, &[]).expect("jit compile/run") {
        JitOutcome::Returned(s) => s,
        other => panic!("jit did not return: {other:?}"),
    };
    let (count, buf0) = (slots[0], slots[1]);
    assert!(
        count >= 1,
        "the masked tagged root must be recovered (count {count})"
    );
    assert!(
        buf0 == 0x7050,
        "the tag byte must be stripped to the bare offset 0x7050; got buf0={buf0:#x}"
    );
}

/// **The verifier rejects a constant fold-down mask.** A `gc.roots` whose mask clears more than the
/// top byte (here the low-24-bits `FOLD_DOWN_MASK`) could fold a host pointer into the guest window
/// and leak host-address bits past the range filter (GC.md §3, §6). The verifier traces the constant
/// mask and fails closed.
#[test]
fn gc_roots_rejects_fold_down_mask_in_verifier() {
    let src = format!(
        "memory 16\n\
        func () -> (i64) {{\n\
        block0():\n\
        \x20 v0 = i64.const 0\n\
        \x20 v1 = i64.const 4096\n\
        \x20 vmask = i64.const {FOLD_DOWN_MASK}\n\
        \x20 v2 = i64.const 0\n\
        \x20 v3 = i64.const 8\n\
        \x20 v4 = gc.roots v0 v1 vmask v2 v3\n\
        \x20 return v4\n\
        }}\n"
    );
    let m = parse_module(&src).expect("parse");
    assert!(
        matches!(
            verify_module(&m),
            Err(svm_verify::VerifyError::GcRootsMaskUnsafe { .. })
        ),
        "verifier must reject a fold-down gc.roots mask"
    );
}

/// **Both runtimes defend against a fold-down mask** even when the module skips verification (an
/// unverified / adversarial module). The interpreter and the JIT each check the mask at the op and
/// trap rather than fold a host word into the reported set.
#[test]
fn gc_roots_runtime_rejects_fold_down_mask() {
    use svm_interp::Trap;
    let src = format!(
        "memory 16\n\
        func () -> (i64) {{\n\
        block0():\n\
        \x20 v0 = i64.const 0\n\
        \x20 v1 = i64.const 4096\n\
        \x20 vmask = i64.const {FOLD_DOWN_MASK}\n\
        \x20 v2 = i64.const 0\n\
        \x20 v3 = i64.const 8\n\
        \x20 v4 = gc.roots v0 v1 vmask v2 v3\n\
        \x20 return v4\n\
        }}\n"
    );
    let m = parse_module(&src).expect("parse");
    // Deliberately skip `verify_module` — this is the unverified-module defense path.
    let mut fuel = 1_000_000u64;
    assert!(
        matches!(run(&m, 0, &[], &mut fuel), Err(Trap::Malformed)),
        "interpreter must trap on a fold-down mask"
    );

    // The JIT applies the same check inside its `gc_roots` thunk (where the substrate supports the op).
    #[cfg(any(
        all(unix, target_arch = "x86_64"),
        all(unix, target_arch = "aarch64"),
        all(windows, target_arch = "x86_64")
    ))]
    {
        use svm_jit::{compile_and_run, JitOutcome};
        match compile_and_run(&m, 0, &[]).expect("jit compile/run") {
            JitOutcome::Trapped(_) => {}
            other => panic!("JIT must trap on a fold-down mask; got {other:?}"),
        }
    }
}
