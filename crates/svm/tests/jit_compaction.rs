//! JIT.md §6 #1 — **code-memory compaction reclaim** for the guest-driven `Jit`.
//!
//! `cranelift-jit`'s `JITModule` has **no per-function free**: every `define_extra` consumes the
//! code arena and nothing is ever returned, so a long REPL that redefines functions leaks code
//! memory until the 256 MiB arena is exhausted (`-ENOMEM`). Slot reclaim (`uninstall`) frees a
//! `call_indirect` *table slot* but not the *code* behind a stale definition. The reclaim strategy
//! (JIT.md §6) is therefore **whole-module recompaction**: at a quiescent point, rebuild the *live*
//! unit set into a **fresh** `CompiledModule` and drop the old one — RAII frees its entire arena.
//!
//! This is an **embedder-orchestrated** operation (the embedder owns liveness/handle policy and the
//! base compile inputs), so these tests drive the exact pattern an embedder uses, against the
//! primitives `CompiledModule` provides:
//! - [`CompiledModule::install_at`] — reinstall a unit at its *exact* old slot, so a funcref a guest
//!   already holds keeps resolving to the same unit across the swap;
//! - [`CompiledModule::extra_fn_count`] — the occupancy proxy an embedder watches to trigger
//!   compaction, and which restarts near zero in the fresh module (the visible reclaim).
//!
//! What is pinned: recompaction is **behaviorally transparent** (every live slot dispatches
//! identically before and after), **reproduces exact slot indices** (including around `uninstall`
//! gaps), and **bounds occupancy by the live set, not the cumulative history** (the reclaim).

use svm_ir::{Func, Module, DEFAULT_RESERVED_LOG2};
use svm_jit::{CompiledModule, JitOutcome, Quota, TrapKind, INERT_CAP_THUNK};
use svm_text::parse_module;
use svm_verify::verify_module;

/// Compile `m` long-lived with an empty powerbox, entry func 0, and a `table_log2`-slot reserved
/// `call_indirect` table (B2 install room). This is the single place the 11-arg `compile` is spelled
/// out — both the initial build and every recompaction go through it, so the fresh module shares the
/// original's baked environment exactly (same mask, same table size).
fn compile_reserved(m: &Module, table_log2: u8) -> CompiledModule {
    CompiledModule::compile(
        m,
        0,
        INERT_CAP_THUNK,
        core::ptr::null_mut(),
        DEFAULT_RESERVED_LOG2,
        None,
        None,
        None,
        None,
        Quota::default(),
        table_log2,
    )
    .expect("compile")
}

/// A unit `(i32, i32) -> (i32)` computing `a * b + k` — a distinct definition per `k`, so a slot's
/// behavior is a function of *which* definition currently occupies it.
fn unit_mul_add(k: i32) -> Vec<Func> {
    let src = format!(
        "func (i32, i32) -> (i32) {{\nblock0(v0: i32, v1: i32):\n  v2 = i32.mul v0 v1\n  v3 = i32.const {k}\n  v4 = i32.add v2 v3\n  return v4\n}}\n"
    );
    let m = parse_module(&src).expect("parse unit");
    verify_module(&m).expect("verify unit");
    m.funcs
}

/// The dispatching parent: `(slot, a, b) -> call_indirect[slot](a, b)`. Func 0, one real function,
/// so every reserved slot ≥ 1 is installable padding.
const DISPATCH_PARENT: &str = "func (i32, i32, i32) -> (i32) {\nblock0(v0: i32, v1: i32, v2: i32):\n  v3 = call_indirect (i32, i32) -> (i32) v0 (v1, v2)\n  return v3\n}\n";

/// Dispatch `slot` with `(a, b)` through the parent entry; expect a returned scalar.
fn dispatch(cm: &mut CompiledModule, slot: u32, a: i32, b: i32) -> i64 {
    let (out, _) = cm
        .run(&[slot as i64, a as i64, b as i64], None, None, None)
        .expect("run");
    match out {
        JitOutcome::Returned(s) if s.len() == 1 => s[0],
        other => panic!("expected one scalar, got {other:?}"),
    }
}

/// One live unit retained for compaction: its IR + the table slot it is installed at. (An embedder
/// holds the analogous record per live `CompiledCode` handle.)
struct Live {
    funcs: Vec<Func>,
    slot: u32,
}

/// Rebuild the live set into a fresh module: recompile the base, then `define_extra` + `install_at`
/// each live unit at its exact slot. This is the embedder's compaction step (minus handle remap,
/// which here is just "use the new module"). Returns the fresh module; the old one is dropped by the
/// caller, freeing its arena.
fn recompact(base: &Module, table_log2: u8, live: &[Live]) -> CompiledModule {
    let mut fresh = compile_reserved(base, table_log2);
    for u in live {
        let defs = fresh.define_extra(&u.funcs).expect("re-define live unit");
        assert!(
            fresh.install_at(u.slot, defs[0].code, defs[0].type_id),
            "install_at must reproduce slot {}",
            u.slot
        );
    }
    fresh
}

/// **The core reclaim property.** A REPL redefines one function many times: each redefinition
/// `uninstall`s the old slot and `install`s the new code, so the table never grows — but the *code*
/// of every superseded definition stays in the arena (`extra_fn_count` climbs monotonically). After
/// many redefinitions, compaction rebuilds only the *one* live definition into a fresh module: it
/// behaves identically, keeps the same slot, and its occupancy is the live set's, not the history's.
#[test]
fn recompaction_reclaims_superseded_definitions() {
    let base = parse_module(DISPATCH_PARENT).expect("parse parent");
    verify_module(&base).expect("verify parent");
    let table_log2 = 6; // 64 slots — ample install room
    let mut cm = compile_reserved(&base, table_log2);

    // Redefine "f" 40 times, reusing one slot (uninstall old → install new), as a REPL would on
    // `def f = ...`. The slot index stays put; the code arena accumulates 40 dead definitions.
    let mut slot = None;
    let last_k = 39;
    for k in 0..=last_k {
        let funcs = unit_mul_add(k);
        let defs = cm.define_extra(&funcs).expect("define_extra");
        if let Some(s) = slot {
            assert!(cm.uninstall(s), "uninstall previous definition");
            assert!(
                cm.install_at(s, defs[0].code, defs[0].type_id),
                "reinstall at the freed slot"
            );
        } else {
            slot = Some(cm.install(defs[0].code, defs[0].type_id).expect("install"));
        }
    }
    let slot = slot.unwrap();
    // The live definition (k = 39): 6 * 7 + 39 = 81.
    assert_eq!(dispatch(&mut cm, slot, 6, 7), 81);
    // Occupancy reflects all 40 definitions (≥ 40 functions lowered, plus trampolines).
    let dirty = cm.extra_fn_count();
    assert!(
        dirty >= 40,
        "expected the arena to carry every superseded definition, got {dirty}"
    );

    // Compact: the live set is the single current definition of "f" at its slot.
    let live = vec![Live {
        funcs: unit_mul_add(last_k),
        slot,
    }];
    let mut fresh = recompact(&base, table_log2, &live);

    // Transparent: the live slot dispatches identically in the fresh module.
    assert_eq!(dispatch(&mut fresh, slot, 6, 7), 81);
    assert_eq!(dispatch(&mut fresh, slot, 10, 10), 139); // 10*10 + 39 = 139
    let clean = fresh.extra_fn_count();
    // Reclaim: the fresh module carries only the live unit, not the 40-deep history.
    assert!(
        clean < dirty && clean <= 4,
        "compaction must bound occupancy by the live set: {clean} (was {dirty})"
    );

    // The old module is still valid until dropped (no use-after-free in the swap); dropping it frees
    // its arena. The fresh module is now the live one.
    assert_eq!(dispatch(&mut cm, slot, 1, 1), 40); // old still works: 1*1 + 39
    drop(cm);
    assert_eq!(dispatch(&mut fresh, slot, 1, 1), 40); // fresh unaffected by the drop
}

/// Recompaction reproduces **exact** slot indices, including around an `uninstall` gap — a guest
/// holding funcref `s` keeps reaching the same unit across the swap. Three units are installed at
/// slots 1/2/3; slot 2 is uninstalled (a gap); compaction must put the survivors back at 1 and 3,
/// leaving 2 trapping — `install_at`, not `install` (which would repack into 1,2), is what makes
/// this exact.
#[test]
fn recompaction_preserves_slots_across_a_gap() {
    let base = parse_module(DISPATCH_PARENT).expect("parse parent");
    verify_module(&base).expect("verify parent");
    let table_log2 = 4; // 16 slots
    let mut cm = compile_reserved(&base, table_log2);

    let mut slots = Vec::new();
    for k in [100, 200, 300] {
        let defs = cm.define_extra(&unit_mul_add(k)).expect("define_extra");
        slots.push(cm.install(defs[0].code, defs[0].type_id).expect("install"));
    }
    assert_eq!(slots, vec![1, 2, 3], "dense initial install");
    // Drop the middle one — slot 2 becomes a trapping gap.
    assert!(cm.uninstall(slots[1]));
    assert_eq!(dispatch(&mut cm, slots[0], 2, 3), 106); // 2*3 + 100
    assert_eq!(dispatch(&mut cm, slots[2], 2, 3), 306); // 2*3 + 300
    let (gap, _) = cm.run(&[slots[1] as i64, 2, 3], None, None, None).unwrap();
    assert!(matches!(
        gap,
        JitOutcome::Trapped(TrapKind::IndirectCallType)
    ));

    // Live set = the two survivors at their original slots (1 and 3), with a hole at 2.
    let live = vec![
        Live {
            funcs: unit_mul_add(100),
            slot: slots[0],
        },
        Live {
            funcs: unit_mul_add(300),
            slot: slots[2],
        },
    ];
    let mut fresh = recompact(&base, table_log2, &live);

    // Exact-slot transparency: 1 and 3 resolve as before, 2 is still a trapping hole.
    assert_eq!(dispatch(&mut fresh, slots[0], 2, 3), 106);
    assert_eq!(dispatch(&mut fresh, slots[2], 2, 3), 306);
    let (gap, _) = fresh
        .run(&[slots[1] as i64, 2, 3], None, None, None)
        .unwrap();
    assert!(
        matches!(gap, JitOutcome::Trapped(TrapKind::IndirectCallType)),
        "the uninstalled gap must remain a trapping padding slot: {gap:?}"
    );
}

/// `install_at` guards: it refuses an out-of-range slot, a real module-function slot (protecting the
/// original program's funcrefs), and an already-occupied slot (it never silently overwrites a live
/// installation — the target must be padding, which a freshly-compacted module's reserved slots are).
#[test]
fn install_at_rejects_invalid_targets() {
    let base = parse_module(DISPATCH_PARENT).expect("parse parent");
    verify_module(&base).expect("verify parent");
    let mut cm = compile_reserved(&base, 4); // 16 slots, 1 real func
    let defs = cm.define_extra(&unit_mul_add(7)).expect("define_extra");
    let (code, tid) = (defs[0].code, defs[0].type_id);

    assert!(!cm.install_at(0, code, tid), "slot 0 is the real function");
    assert!(!cm.install_at(16, code, tid), "slot 16 is out of range");
    assert!(!cm.install_at(99, code, tid), "way out of range");

    assert!(cm.install_at(5, code, tid), "slot 5 is installable padding");
    assert!(
        !cm.install_at(5, code, tid),
        "install_at must refuse an occupied slot"
    );
    // Quiescent between runs — an embedder checks this before swapping in a compacted module.
    assert!(!cm.is_running());
}
