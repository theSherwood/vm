//! DESIGN.md §22: the long-lived `CompiledModule` split (`compile` → `run`) and
//! **incremental definition** (`define_extra`) — the enabling primitives for the
//! guest-driven `Jit` capability (Model A).
//!
//! What is established here, differentially against the reference interpreter:
//! - `compile().run()` ≡ the old one-shot `compile_and_run` (the refactor is
//!   behavior-preserving), and a `CompiledModule` survives — and can re-`run` after —
//!   multiple runs.
//! - `define_extra` compiles a self-contained unit against the **parent's** baked
//!   environment: same confinement mask (escape-oracle checked), same table mask, the
//!   parent's `distinct` type-id space (unknown signatures trap fail-closed), and
//!   unit-local direct calls.
//! - The **W^X / incremental-finalize spike** (DESIGN.md §22 / Open questions):
//!   cranelift-jit 0.132's define→finalize→define→finalize cycle leaves already-finalized
//!   code intact and runnable — exercised by interleaving `define_extra` calls with runs
//!   of both the parent entry and earlier units.
//! - **Mask invariance** (DESIGN.md §22 "the baked function-table mask"): extra units are never
//!   installed in the function table, so `call_indirect` from an extra unit dispatches
//!   through the parent's table with the parent's mask — byte-identical dispatch, even
//!   when the cumulative function count crosses a power-of-two boundary.

use svm_interp::{run, Value};
use svm_ir::{FuncType, ValType, DEFAULT_RESERVED_LOG2};
use svm_jit::{CompiledModule, JitOutcome, Quota, TrapKind, INERT_CAP_THUNK};
use svm_text::parse_module;
use svm_verify::verify_module;

/// Compile `src`'s module long-lived with an empty powerbox (entry = func 0).
fn compile(src: &str) -> CompiledModule {
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");
    CompiledModule::compile(
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
    .expect("compile")
}

/// Interpreter oracle: run `src`'s func 0 on `args`, expecting success.
fn interp(src: &str, args: &[Value]) -> Vec<i64> {
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");
    let mut fuel = 1_000_000u64;
    run(&m, 0, args, &mut fuel)
        .expect("interp")
        .into_iter()
        .map(|v| match v {
            Value::I32(x) => x as i64,
            Value::I64(x) => x,
            other => panic!("scalar result expected, got {other:?}"),
        })
        .collect()
}

const ADD: &str =
    "func (i32, i32) -> (i32) {\nblock 0 (v0: i32, v1: i32) {\n  v2 = i32.add v0 v1\n  return v2\n  }\n}\n";

/// The split is behavior-preserving and the module survives a run: `compile().run()`
/// matches the interpreter, twice, on the same `CompiledModule`.
#[test]
fn compile_then_run_twice_matches_interp() {
    let mut cm = compile(ADD);
    for (a, b) in [(2, 40), (-7, 7)] {
        let (out, _) = cm
            .run(&[a as i64, b as i64], None, None, None)
            .expect("run");
        let want = interp(ADD, &[Value::I32(a), Value::I32(b)]);
        assert!(
            matches!(out, JitOutcome::Returned(ref s) if s == &want),
            "{out:?} != {want:?}"
        );
    }
}

/// `run` rejects an args buffer shorter than the entry's parameter count (the trampoline
/// would read out of bounds from safe code otherwise).
#[test]
fn run_rejects_short_args() {
    let mut cm = compile(ADD);
    assert!(cm.run(&[1], None, None, None).is_err());
}

/// `define_extra` basics: a pure extra function runs (via its trampoline) and matches the
/// interpreter running the same code as its own module.
#[test]
fn define_extra_pure_function_matches_interp() {
    let mut cm = compile(ADD);
    let extra_src = "func (i32, i32) -> (i32) {\nblock 0 (v0: i32, v1: i32) {\n  v2 = i32.mul v0 v1\n  return v2\n  }\n}\n";
    let extra = parse_module(extra_src).expect("parse");
    verify_module(&extra).expect("verify");
    let ptrs = cm.define_extra(&extra.funcs).expect("define_extra");
    assert_eq!(ptrs.len(), 1);
    let (out, _) = unsafe { cm.run_extra(ptrs[0].tramp, 2, 1, &[6, 7], None) }.expect("run_extra");
    let want = interp(extra_src, &[Value::I32(6), Value::I32(7)]);
    assert!(matches!(out, JitOutcome::Returned(ref s) if s == &want));
}

/// Unit-local direct calls: an extra unit's `FuncIdx` space is its own — func 0 of the
/// unit directly calls func 1 of the unit, not anything in the parent.
#[test]
fn define_extra_unit_local_direct_calls() {
    let mut cm = compile(ADD);
    let extra_src = "func (i32) -> (i32) {\nblock 0 (v0: i32) {\n  v1 = call 1 (v0)\n  v2 = i32.add v1 v1\n  return v2\n  }\n}\nfunc (i32) -> (i32) {\nblock 0 (v0: i32) {\n  v1 = i32.const 10\n  v2 = i32.add v0 v1\n  return v2\n  }\n}\n";
    let extra = parse_module(extra_src).expect("parse");
    verify_module(&extra).expect("verify");
    let ptrs = cm.define_extra(&extra.funcs).expect("define_extra");
    assert_eq!(ptrs.len(), 2);
    let (out, _) = unsafe { cm.run_extra(ptrs[0].tramp, 1, 1, &[5], None) }.expect("run_extra");
    let want = interp(extra_src, &[Value::I32(5)]); // (5 + 10) * 2 = 30
    assert!(matches!(out, JitOutcome::Returned(ref s) if s == &want));
}

/// The W^X / incremental-finalize spike (DESIGN.md §22): two `define_extra` calls — two
/// `finalize_definitions` cycles after the parent's — with runs of the parent entry and the
/// *first* unit interleaved **after the second finalize**. Already-finalized code must stay
/// intact and runnable across later finalizes.
#[test]
fn incremental_finalize_keeps_earlier_code_runnable() {
    let mut cm = compile(ADD);
    let unit1_src =
        "func (i32) -> (i32) {\nblock 0 (v0: i32) {\n  v1 = i32.const 1\n  v2 = i32.add v0 v1\n  return v2\n  }\n}\n";
    let unit2_src =
        "func (i32) -> (i32) {\nblock 0 (v0: i32) {\n  v1 = i32.const 2\n  v2 = i32.mul v0 v1\n  return v2\n  }\n}\n";
    let unit1 = parse_module(unit1_src).expect("parse");
    let unit2 = parse_module(unit2_src).expect("parse");
    verify_module(&unit1).expect("verify");
    verify_module(&unit2).expect("verify");

    let p1 = cm.define_extra(&unit1.funcs).expect("first define_extra");
    let p2 = cm.define_extra(&unit2.funcs).expect("second define_extra");

    // After the SECOND finalize: unit 1's code (finalized earlier) still runs…
    let (out, _) = unsafe { cm.run_extra(p1[0].tramp, 1, 1, &[41], None) }.expect("unit1");
    assert!(matches!(out, JitOutcome::Returned(ref s) if s == &[42]));
    // …unit 2 runs…
    let (out, _) = unsafe { cm.run_extra(p2[0].tramp, 1, 1, &[21], None) }.expect("unit2");
    assert!(matches!(out, JitOutcome::Returned(ref s) if s == &[42]));
    // …and the parent entry (finalized before both) still runs.
    let (out, _) = cm.run(&[40, 2], None, None, None).expect("parent");
    assert!(matches!(out, JitOutcome::Returned(ref s) if s == &[42]));
}

/// Mask invariance (DESIGN.md §22 "the baked function-table mask"): the parent declares ONE
/// function (table mask = 0); the extra unit holds FOUR functions, so the *cumulative*
/// count crosses a power-of-two boundary — but extra functions are thunk-reachable only
/// and never enter the table, so a `call_indirect` from extra code dispatches through the
/// parent's 1-entry table with the parent's mask. Index 0 hits the parent function; index 3
/// wraps (`3 & 0 = 0`) to the same slot — exactly what parent code itself would do.
#[test]
fn define_extra_call_indirect_uses_parent_table_and_mask() {
    // Parent: func 0 (i32, i32) -> (i32) is the add — also the entry.
    let mut cm = compile(ADD);
    let extra_src = concat!(
        // f0: call_indirect slot 0 with the parent's signature.
        "func (i32, i32) -> (i32) {\nblock 0 (v0: i32, v1: i32) {\n  v2 = i32.const 0\n  v3 = call_indirect (i32, i32) -> (i32) v2 (v0, v1)\n  return v3\n  }\n}\n",
        // f1: call_indirect slot 3 — masked by the parent's mask 0, wraps to slot 0.
        "func (i32, i32) -> (i32) {\nblock 0 (v0: i32, v1: i32) {\n  v2 = i32.const 3\n  v3 = call_indirect (i32, i32) -> (i32) v2 (v0, v1)\n  return v3\n  }\n}\n",
        // f2, f3: padding so the unit pushes the cumulative function count past a
        // power-of-two boundary (1 parent + 4 extra = 5 > 4).
        "func () -> (i32) {\nblock 0 () {\n  v0 = i32.const 0\n  return v0\n  }\n}\n",
        "func () -> (i32) {\nblock 0 () {\n  v0 = i32.const 0\n  return v0\n  }\n}\n",
    );
    let extra = parse_module(extra_src).expect("parse");
    verify_module(&extra).expect("verify");
    let ptrs = cm.define_extra(&extra.funcs).expect("define_extra");
    // Slot 0 → the parent's add.
    let (out, _) = unsafe { cm.run_extra(ptrs[0].tramp, 2, 1, &[30, 12], None) }.expect("idx 0");
    assert!(
        matches!(out, JitOutcome::Returned(ref s) if s == &[42]),
        "{out:?}"
    );
    // Slot 3 wraps under the parent's mask (0) to slot 0 → the same add. The dispatch
    // lowering and its mask constant are byte-identical to the parent's.
    let (out, _) = unsafe { cm.run_extra(ptrs[1].tramp, 2, 1, &[40, 2], None) }.expect("idx 3");
    assert!(
        matches!(out, JitOutcome::Returned(ref s) if s == &[42]),
        "{out:?}"
    );
}

/// The complement of mask invariance: **extra code is invisible to guest dispatch** (DESIGN.md §22
/// Model A — parent→extra calls do not exist; the only entry into extra code is the host
/// trampoline). The parent dispatches `call_indirect` over every index a guest could name;
/// then an extra function with the *same signature* as the table's functions is defined; the
/// sweep must be outcome-identical — no index reaches the new code, including the padding
/// slots and wrapped indices. (A guest array mixing old and new procedures therefore cannot
/// be uniform funcrefs under Model A: new procedures are reached via the Phase-2 `invoke`
/// op, i.e. tagged dispatch. Uniform funcref arrays are exactly what Model B2's table
/// installation would buy.)
#[test]
fn parent_call_indirect_cannot_reach_extra_code() {
    // Parent: f0 = the dispatching entry, f1 = +10, f2 = *2 (both (i32) -> (i32)).
    // Table is padded to 4 slots; slot 3 is padding (traps), idx ≥ 4 wraps (mask 3).
    let parent_src = concat!(
        "func (i32, i32) -> (i32) {\nblock 0 (v0: i32, v1: i32) {\n  v2 = call_indirect (i32) -> (i32) v0 (v1)\n  return v2\n  }\n}\n",
        "func (i32) -> (i32) {\nblock 0 (v0: i32) {\n  v1 = i32.const 10\n  v2 = i32.add v0 v1\n  return v2\n  }\n}\n",
        "func (i32) -> (i32) {\nblock 0 (v0: i32) {\n  v1 = i32.const 2\n  v2 = i32.mul v0 v1\n  return v2\n  }\n}\n",
    );
    let mut cm = compile(parent_src);
    let sweep = |cm: &mut CompiledModule| -> Vec<JitOutcome> {
        (0..8)
            .map(|idx| cm.run(&[idx, 100], None, None, None).expect("run").0)
            .collect()
    };
    let before = sweep(&mut cm);
    // Sanity: the sweep exercises real dispatch — hits (+10, *2), a self-type-mismatch,
    // padding traps, and wraparound.
    assert!(matches!(before[1], JitOutcome::Returned(ref s) if s == &[110]));
    assert!(matches!(before[2], JitOutcome::Returned(ref s) if s == &[200]));
    assert!(matches!(
        before[3],
        JitOutcome::Trapped(TrapKind::IndirectCallType)
    ));

    // An extra function with the SAME signature as f1/f2 — if it leaked into the table
    // anywhere, some index would now return x + 1000.
    let extra_src = "func (i32) -> (i32) {\nblock 0 (v0: i32) {\n  v1 = i32.const 1000\n  v2 = i32.add v0 v1\n  return v2\n  }\n}\n";
    let extra = parse_module(extra_src).expect("parse");
    verify_module(&extra).expect("verify");
    let ptrs = cm.define_extra(&extra.funcs).expect("define_extra");
    // The new code is alive and callable — through the host trampoline only.
    let (out, _) = unsafe { cm.run_extra(ptrs[0].tramp, 1, 1, &[100], None) }.expect("run_extra");
    assert!(matches!(out, JitOutcome::Returned(ref s) if s == &[1100]));

    // Every guest-nameable index dispatches byte-identically to before.
    let after = sweep(&mut cm);
    assert_eq!(
        before, after,
        "extra code must be unreachable from the table"
    );
}

/// Fail-closed type ids: a `call_indirect` in an extra unit whose signature no table entry
/// carries traps `IndirectCallType` (it must NOT silently call anything). The signature gets
/// a real interned id — stable for a future install — but until something with that
/// signature sits in the table, every dispatch with it is inert.
#[test]
fn define_extra_unknown_signature_traps_fail_closed() {
    let mut cm = compile(ADD); // parent declares only (i32, i32) -> (i32)
    let extra_src = "func (i64) -> (i64) {\nblock 0 (v0: i64) {\n  v1 = i32.const 0\n  v2 = call_indirect (i64) -> (i64) v1 (v0)\n  return v2\n  }\n}\n";
    let extra = parse_module(extra_src).expect("parse");
    verify_module(&extra).expect("verify");
    let ptrs = cm.define_extra(&extra.funcs).expect("define_extra");
    let (out, _) = unsafe { cm.run_extra(ptrs[0].tramp, 1, 1, &[7], None) }.expect("run_extra");
    assert!(
        matches!(out, JitOutcome::Trapped(TrapKind::IndirectCallType)),
        "unknown signature must trap fail-closed, got {out:?}"
    );
}

const MEM_PARENT: &str = "memory 16\nfunc () -> (i32) {\nblock 0 () {\n  v0 = i64.const 8\n  v1 = i32.load v0\n  return v1\n  }\n}\n";

/// Extra code shares the parent's window environment (DESIGN.md §22 "vmctx sharing"): it is
/// compiled against the same confinement mask + backed extent, so its memory effects match
/// the interpreter's — an in-window store lands at the same byte (final-memory equality),
/// and a store beyond the backed `mapped` extent (but inside the reserved mask domain)
/// detect-and-kills as a `MemoryFault` on both backends (§4 guard-when-bounded).
#[test]
fn define_extra_masking_matches_interp_memory_effects() {
    let mut cm = compile(MEM_PARENT);
    // In-window: store 0xAB at offset 8, read it back.
    let store_src = "memory 16\nfunc () -> (i32) {\nblock 0 () {\n  v0 = i64.const 8\n  v1 = i32.const 171\n  i32.store8 v0 v1\n  v2 = i64.const 8\n  v3 = i32.load8_u v2\n  return v3\n  }\n}\n";
    let extra = parse_module(store_src).expect("parse");
    verify_module(&extra).expect("verify");
    let mut fuel = 1_000_000u64;
    let want = run(&parse_module(store_src).expect("parse"), 0, &[], &mut fuel).expect("interp");
    assert_eq!(want, vec![Value::I32(171)]);

    let ptrs = cm.define_extra(&extra.funcs).expect("define_extra");
    let (out, final_mem) =
        unsafe { cm.run_extra(ptrs[0].tramp, 0, 1, &[], None) }.expect("run_extra");
    assert!(
        matches!(out, JitOutcome::Returned(ref s) if s == &[171]),
        "{out:?}"
    );
    // Escape-oracle: exactly the stored byte changed.
    assert_eq!(final_mem[8], 0xab);
    assert!(final_mem[..8].iter().all(|&b| b == 0));

    // Beyond `mapped` (64 KiB) but inside the reserved mask domain: a guard fault —
    // detect-and-kill — on the JIT, agreeing with the interpreter.
    let fault_src = "memory 16\nfunc () -> (i32) {\nblock 0 () {\n  v0 = i64.const 1048584\n  v1 = i32.const 171\n  i32.store8 v0 v1\n  v2 = i32.const 0\n  return v2\n  }\n}\n";
    let extra = parse_module(fault_src).expect("parse");
    verify_module(&extra).expect("verify");
    let mut fuel = 1_000_000u64;
    assert_eq!(
        run(&parse_module(fault_src).expect("parse"), 0, &[], &mut fuel),
        Err(svm_interp::Trap::MemoryFault),
        "interp: a store past the backed extent must fault"
    );
    let ptrs = cm.define_extra(&extra.funcs).expect("define_extra");
    let (out, _) = unsafe { cm.run_extra(ptrs[0].tramp, 0, 1, &[], None) }.expect("run_extra");
    assert!(
        matches!(out, JitOutcome::Trapped(TrapKind::MemoryFault)),
        "jit: extra code past the backed extent must detect-and-kill, got {out:?}"
    );
}

/// The append-only type-id registry (DESIGN.md §22 groundwork): a novel signature introduced by
/// one unit is interned under a stable id that a later unit — mentioning it only at a call
/// site — shares; parent ids never move; and until a table entry carries the id, dispatch
/// with it stays fail-closed. (End-to-end observability arrives with the table `install` op;
/// this pins the registry semantics it will rely on.)
#[test]
fn type_ids_are_interned_append_only_across_units() {
    let mut cm = compile(ADD);
    let add_sig = FuncType {
        params: vec![ValType::I32, ValType::I32],
        results: vec![ValType::I32],
    };
    let novel = FuncType {
        params: vec![ValType::I64],
        results: vec![ValType::I64],
    };
    // The parent's only function signature is id 0; the novel signature is unknown.
    assert_eq!(cm.interned_type_id(&add_sig), Some(0));
    assert_eq!(cm.interned_type_id(&novel), None);

    // Unit A introduces (i64) -> (i64) as a *function* signature.
    let unit_a_src =
        "func (i64) -> (i64) {\nblock 0 (v0: i64) {\n  v1 = i64.const 1\n  v2 = i64.add v0 v1\n  return v2\n  }\n}\n";
    let unit_a = parse_module(unit_a_src).expect("parse");
    verify_module(&unit_a).expect("verify");
    cm.define_extra(&unit_a.funcs).expect("unit A");
    let id = cm.interned_type_id(&novel).expect("interned by unit A");

    // Unit B mentions the same signature only at a call site — same id, nothing remapped.
    let unit_b_src = "func (i64) -> (i64) {\nblock 0 (v0: i64) {\n  v1 = i32.const 0\n  v2 = call_indirect (i64) -> (i64) v1 (v0)\n  return v2\n  }\n}\n";
    let unit_b = parse_module(unit_b_src).expect("parse");
    verify_module(&unit_b).expect("verify");
    let ptrs = cm.define_extra(&unit_b.funcs).expect("unit B");
    assert_eq!(cm.interned_type_id(&novel), Some(id), "stable across units");
    assert_eq!(
        cm.interned_type_id(&add_sig),
        Some(0),
        "parent ids never move"
    );

    // No table entry carries the novel id, so dispatch with it stays fail-closed.
    let (out, _) = unsafe { cm.run_extra(ptrs[0].tramp, 1, 1, &[7], None) }.expect("run_extra");
    assert!(matches!(
        out,
        JitOutcome::Trapped(TrapKind::IndirectCallType)
    ));
}

/// An empty unit is a no-op.
#[test]
fn define_extra_empty_unit() {
    let mut cm = compile(ADD);
    assert!(cm.define_extra(&[]).expect("empty").is_empty());
}

/// B2 `install` (JIT level, DESIGN.md §22 slice #4): a `define_extra` unit installed into a
/// **pre-reserved** table slot becomes `call_indirect`-able — old→new. The parent entry is
/// `(slot, a, b) -> call_indirect[slot](a, b)`; with a reserved table the unit lands in the
/// first padding slot, and dispatching that slot runs the unit over the live window. An
/// un-installed slot still traps `IndirectCallType` fail-closed.
#[test]
fn install_makes_unit_call_indirectable() {
    let parent_src = "func (i32, i32, i32) -> (i32) {\nblock 0 (v0: i32, v1: i32, v2: i32) {\n  v3 = call_indirect (i32, i32) -> (i32) v0 (v1, v2)\n  return v3\n  }\n}\n";
    let m = parse_module(parent_src).expect("parse");
    verify_module(&m).expect("verify");
    // Reserve a 16-slot table (log2 = 4) so there is padding for install (parent has 1 func).
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
        4,
    )
    .expect("compile");

    let unit_src = "func (i32, i32) -> (i32) {\nblock 0 (v0: i32, v1: i32) {\n  v2 = i32.mul v0 v1\n  v3 = i32.const 100\n  v4 = i32.add v2 v3\n  return v4\n  }\n}\n";
    let unit = parse_module(unit_src).expect("parse");
    verify_module(&unit).expect("verify");
    let defs = cm.define_extra(&unit.funcs).expect("define_extra");
    let slot = cm.install(defs[0].code, defs[0].type_id).expect("install");
    assert_eq!(
        slot, 1,
        "first padding slot is just past the parent's 1 function"
    );

    // old→new: parent `call_indirect` of the installed slot reaches the unit.
    let (out, _) = cm.run(&[slot as i64, 6, 7], None, None, None).expect("run");
    assert!(
        matches!(out, JitOutcome::Returned(ref s) if s == &[142]),
        "{out:?}"
    ); // 6 * 7 + 100

    // An un-installed (padding) slot still traps fail-closed.
    let (out, _) = cm.run(&[2, 6, 7], None, None, None).expect("run");
    assert!(matches!(
        out,
        JitOutcome::Trapped(TrapKind::IndirectCallType)
    ));
}

/// `install` returns `None` when every reserved slot is full (here: no reservation, so the
/// natural padding is the only room, and a 1-function module padded to 1 slot has none).
#[test]
fn install_full_table_returns_none() {
    let mut cm = compile(ADD); // natural table, 1 func → next_pow2(1) = 1 slot, zero padding
    let unit = parse_module(ADD).expect("parse");
    let defs = cm.define_extra(&unit.funcs).expect("define_extra");
    assert!(
        cm.install(defs[0].code, defs[0].type_id).is_none(),
        "no padding to install into"
    );
}

/// **Spike: `finalize_definitions` is safe while a sibling thread executes finalized code**
/// (DESIGN.md §22 threaded *compile*, the W^X question). A worker thread hammers an
/// already-finalized leaf function in a tight loop via its `extern "C"` trampoline, while the
/// main thread does hundreds of `define_extra`s — each running `finalize_definitions`, which
/// `mprotect`s the *new* code pages and issues a cross-core pipeline flush. If finalize ever
/// re-protected or disturbed the *running* leaf's page, the sibling would `SIGSEGV`/`SIGBUS` or
/// read garbage; instead it must keep returning `42`. This corroborates the source finding that
/// `ArenaMemoryProvider::finalize` skips already-finalized segments (`Segment::finalize` early-
/// returns on `finalized`, and the `set_rw` resize path skips finalized segments), so executing
/// code — always on a finalized segment — is never touched. The sibling holds only the raw code
/// pointer (a `usize`), never `cm`, so there is no Rust aliasing with the main thread's `&mut`.
#[test]
fn concurrent_finalize_does_not_disturb_running_code() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    // The buffer-ABI trampoline shape (`svm_jit::mem::Entry`): args, results, mem_base, fn_table,
    // trap_cell. A no-memory, no-call leaf never faults, so it is safe to call outside the
    // detect-and-kill guard.
    type Entry = extern "C" fn(*const i64, *mut i64, *mut u8, *const std::ffi::c_void, *mut i64);

    let mut cm = compile(ADD);
    // Leaf `() -> (i64)` returning 42 — no memory, no calls.
    let leaf_src = "func () -> (i64) {\nblock 0 () {\n  v0 = i64.const 42\n  return v0\n  }\n}\n";
    let leaf = parse_module(leaf_src).expect("parse");
    verify_module(&leaf).expect("verify");
    let defs = cm.define_extra(&leaf.funcs).expect("define leaf");
    let tramp = defs[0].tramp as usize; // Send across the thread boundary as a plain integer

    let stop = Arc::new(AtomicBool::new(false));
    let stop_w = Arc::clone(&stop);
    let worker = std::thread::spawn(move || {
        // SAFETY: `tramp` is a finalized buffer-ABI trampoline for the leaf; the arena never frees
        // finalized code, so it stays valid for the whole run. The leaf touches no memory, so
        // calling it outside the signal guard cannot fault.
        let f: Entry = unsafe { std::mem::transmute(tramp as *const u8) };
        let mut results = [0i64];
        let mut trap = 0i64;
        let mut calls = 0u64;
        while !stop_w.load(Ordering::Relaxed) {
            for _ in 0..1000 {
                f(
                    std::ptr::null(),
                    results.as_mut_ptr(),
                    std::ptr::null_mut(),
                    std::ptr::null(),
                    &mut trap,
                );
                assert_eq!(
                    results[0], 42,
                    "running code corrupted by a concurrent finalize"
                );
                assert_eq!(trap, 0, "running code trapped during a concurrent finalize");
            }
            calls += 1000;
        }
        calls
    });

    // Hammer `finalize_definitions` from the main thread while the worker executes the leaf.
    for k in 0..400i64 {
        let src = format!(
            "func () -> (i64) {{\nblock 0 () {{\n  v0 = i64.const {k}\n  return v0\n  }}\n}}\n"
        );
        let m = parse_module(&src).expect("parse");
        verify_module(&m).expect("verify");
        cm.define_extra(&m.funcs)
            .expect("define_extra under concurrent execution");
    }
    stop.store(true, Ordering::Relaxed);
    let calls = worker
        .join()
        .expect("worker thread panicked → finalize disturbed running code");
    assert!(
        calls > 0,
        "the worker must have executed during the finalizes"
    );
}
