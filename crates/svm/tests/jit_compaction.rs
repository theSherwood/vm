//! DESIGN.md §22 — **code-memory compaction reclaim** for the guest-driven `Jit`.
//!
//! `cranelift-jit`'s `JITModule` has **no per-function free**: every `define_extra` consumes the
//! code arena and nothing is ever returned, so a long REPL that redefines functions leaks code
//! memory until the 256 MiB arena is exhausted (`-ENOMEM`). Slot reclaim (`uninstall`) frees a
//! `call_indirect` *table slot* but not the *code* behind a stale definition. The reclaim strategy
//! (DESIGN.md §22) is therefore **whole-module recompaction**: at a quiescent point, rebuild the *live*
//! unit set into a **fresh** `CompiledModule` and drop the old one — RAII frees its entire arena.
//!
//! This is an **embedder-orchestrated** operation (the embedder owns liveness/handle policy and the
//! base compile inputs), so these tests drive the exact pattern an embedder uses, against the
//! primitives `CompiledModule` provides:
//! - [`CompiledModule::install_at`] — reinstall a unit at its *exact* old slot, so a funcref a guest
//!   already holds keeps resolving to the same unit across the swap;
//! - [`CompiledModule::extra_fn_count`] / [`CompiledModule::extra_byte_count`] — code-arena
//!   occupancy measures (function count, and the **byte-accurate** sum of emitted code) an embedder
//!   watches to trigger compaction; both restart near zero in the fresh module (the visible
//!   reclaim). `JitSession`'s auto-compaction watermarks on the byte count.
//!
//! What is pinned: recompaction is **behaviorally transparent** (every live slot dispatches
//! identically before and after), **reproduces exact slot indices** (including around `uninstall`
//! gaps), and **bounds occupancy by the live set, not the cumulative history** (the reclaim).

use svm_encode::encode_module;
use svm_interp::Host;
use svm_ir::{Func, Module, DEFAULT_RESERVED_LOG2};
use svm_jit::{CompiledModule, JitOutcome, Quota, TrapKind, INERT_CAP_THUNK};
use svm_run::{grant_jit, recompact_jit, JitSession};
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

// ---------------------------------------------------------------------------------------------
// Embedder-integrated compaction (svm-run `recompact_jit` + the `Host` unit/handle plumbing).
//
// These go through the *real* `Host` unit tracking — `jit_compile` stores a unit's IR and mints a
// `CompiledCode` handle; `set_jit_unit_native` records its native pointers; `jit_live_units` /
// `installed_slots` are what `recompact_jit` enumerates — driving the compile/install/release
// sequence from Rust exactly as `svm_run::jit_native_op` does from a guest `cap.call`. The dispatch
// itself runs real JIT `call_indirect` over the live window.
//
// Oracle: **compacting-JIT vs non-compacting-JIT** over a persistent-window REPL. (An interp↔JIT
// differential across *multiple* runs is blocked by the reference interp rebuilding its dispatch
// table per run — the separately-tracked shared-table refactor; DESIGN.md §22 "Remaining work" #2 — so
// single-run correctness stays differential in `jit_cap.rs` and *transparency across the swap* is
// pinned here against the non-compacting JIT, the production backend.)

/// A unit blob `(i32, i32) -> (i32)` = `a * b + k`, declaring the parent's `memory 16` so it passes
/// the `Jit` memory-match precondition.
fn unit_blob(k: i32) -> Vec<u8> {
    let src = format!(
        "memory 16\nfunc (i32, i32) -> (i32) {{\nblock0(v0: i32, v1: i32):\n  v2 = i32.mul v0 v1\n  v3 = i32.const {k}\n  v4 = i32.add v2 v3\n  return v4\n}}\n"
    );
    let m = parse_module(&src).expect("parse blob");
    verify_module(&m).expect("verify blob");
    encode_module(&m)
}

/// The REPL shell program: `(slot, x) -> (i32)` dispatches `unit[slot](x, x)` and **accumulates the
/// result into window[0]** — so the running total persists across prompts (each prompt is a fresh
/// `run` seeded with the prior prompt's final window), and compaction between prompts must leave
/// both the installed slot and that window state untouched.
const REPL_SHELL: &str = "memory 16\nfunc (i32, i32) -> (i32) {\nblock0(v0: i32, v1: i32):\n  v2 = call_indirect (i32, i32) -> (i32) v0 (v1, v1)\n  v3 = i64.const 0\n  v4 = i32.load v3\n  v5 = i32.add v4 v2\n  i32.store v3 v5\n  return v5\n}\n";

/// Compile a unit into the live module exactly as the guest-driven `compile` op does: validate +
/// store it in the `Host` (minting a `CompiledCode` handle), lower it (`define_extra`), and register
/// its native pointers. Returns `(unit, code_handle)`.
fn host_define(host: &mut Host, cm: &mut CompiledModule, jit_h: i32, blob: &[u8]) -> (u32, i32) {
    let c = host
        .jit_compile(jit_h, blob)
        .expect("jit domain")
        .expect("validate");
    let funcs = host.jit_unit_funcs(c.domain, c.unit).expect("unit funcs");
    let defs = cm.define_extra(&funcs).expect("define_extra");
    host.set_jit_unit_native(
        c.domain,
        c.unit,
        defs[0].tramp as usize,
        defs[0].code as usize,
        defs[0].type_id,
    );
    (c.unit, c.handle)
}

/// Run one REPL of `n` redefine+call prompts over a persistent window, optionally compacting every
/// `compact_every` prompts. Each prompt **redefines** the one live function (release + uninstall the
/// previous, compile + install the new at the reused slot) then **calls** it, accumulating. Returns
/// `(per-call results, final window low bytes, final extra-fn occupancy)`.
fn run_repl(table_log2: u8, n: usize, compact_every: Option<usize>) -> (Vec<i64>, Vec<u8>, usize) {
    let base = parse_module(REPL_SHELL).expect("parse shell");
    verify_module(&base).expect("verify shell");
    let mut host = Host::new();
    let jit_h = grant_jit(&mut host, &base, table_log2);
    let domain = host.resolve_jit_domain(jit_h).expect("domain");
    let mut cm = CompiledModule::compile(
        &base,
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
    .expect("compile shell");

    let mut mem = vec![0u8; 1 << 18];
    let mut results = Vec::new();
    let mut cur: Option<(i32 /*handle*/, u32 /*slot*/)> = None;
    for i in 0..n {
        // Redefine: drop the previous definition (the REPL releasing its handle + freeing the slot),
        // so it becomes dead code the next compaction reclaims.
        if let Some((h, slot)) = cur.take() {
            host.jit_release(h).expect("release");
            assert!(cm.uninstall(slot), "uninstall previous");
        }
        let (unit, handle) = host_define(&mut host, &mut cm, jit_h, &unit_blob(10));
        let (code, type_id) = host.jit_unit_install(domain, unit);
        let slot = cm.install(code as *const u8, type_id).expect("install");
        cur = Some((handle, slot));

        // Call: dispatch the live slot with x = i + 2; the accumulator lives in window[0].
        let x = (i as i64) + 2;
        let (out, m2) = cm
            .run(&[slot as i64, x], Some(&mem), Some(1 << 18), None)
            .expect("dispatch run");
        mem = m2;
        match out {
            JitOutcome::Returned(s) if s.len() == 1 => results.push(s[0]),
            other => panic!("dispatch returned {other:?}"),
        }

        if let Some(every) = compact_every {
            if (i + 1) % every == 0 {
                // Quiescent point (between runs): reclaim the dead definitions.
                cm = recompact_jit(
                    &base,
                    0,
                    DEFAULT_RESERVED_LOG2,
                    table_log2,
                    &mut host,
                    domain,
                    &cm,
                )
                .expect("recompact");
            }
        }
    }
    (results, mem[..16].to_vec(), cm.extra_fn_count())
}

/// **Compaction is transparent and reclaims** through the real `Host`/`recompact_jit` path: a
/// 20-prompt REPL that redefines one function each prompt produces byte-identical per-call results
/// and window state whether or not it compacts every 5 prompts — but the compacting run's code-arena
/// occupancy stays bounded by the one live definition while the non-compacting run accumulates all 20.
#[test]
fn repl_recompaction_is_transparent_and_reclaims() {
    let (r_plain, m_plain, occ_plain) = run_repl(6, 20, None);
    let (r_comp, m_comp, occ_comp) = run_repl(6, 20, Some(5));

    assert_eq!(r_plain, r_comp, "per-call results must be identical");
    assert_eq!(m_plain, m_comp, "persistent window must be identical");
    // Sanity: the accumulator actually advanced (x=2..21, each unit(x,x) = x*x + 10).
    let expect_last: i64 = (0..20).map(|i| ((i + 2) * (i + 2) + 10) as i64).sum();
    assert_eq!(*r_comp.last().unwrap(), expect_last);

    // Reclaim: the non-compacting run carries all 20 definitions; the compacting run only the live one.
    assert!(
        occ_plain >= 40,
        "non-compacting arena should hold every definition, got {occ_plain}"
    );
    assert!(
        occ_comp <= 4 && occ_comp < occ_plain,
        "compaction must bound occupancy by the live set: {occ_comp} (plain {occ_plain})"
    );
}

/// Compaction carries an **invoke-only, never-installed** unit when its `CompiledCode` handle is
/// still live (the `jit_live_units` branch of `recompact_jit`): its trampoline pointer moves across
/// the swap, so `recompact_jit` must remap the `Host` unit→native record for the existing handle to
/// keep invoking the right code — alongside a separately-installed unit reached by `call_indirect`.
#[test]
fn recompaction_carries_live_invoke_only_unit() {
    let base = parse_module(REPL_SHELL).expect("parse shell");
    verify_module(&base).expect("verify shell");
    let table_log2 = 4;
    let mut host = Host::new();
    let jit_h = grant_jit(&mut host, &base, table_log2);
    let domain = host.resolve_jit_domain(jit_h).expect("domain");
    let mut cm = CompiledModule::compile(
        &base,
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
    .expect("compile shell");

    // Unit U: invoke-only (handle kept, never installed). a*b + 7.
    let (u_unit, _u_handle) = host_define(&mut host, &mut cm, jit_h, &unit_blob(7));
    // Unit V: installed at a slot, reached by call_indirect. a*b + 99.
    let (v_unit, _v_handle) = host_define(&mut host, &mut cm, jit_h, &unit_blob(99));
    let (v_code, v_tid) = host.jit_unit_install(domain, v_unit);
    let v_slot = cm.install(v_code as *const u8, v_tid).expect("install V");

    // Before: invoke U directly through its trampoline (3*4 + 7 = 19).
    let u_tramp_before = host.jit_unit_native(domain, u_unit);
    let (out, _) = unsafe { cm.run_extra(u_tramp_before as *const u8, 2, 1, &[3, 4], None) }
        .expect("invoke U");
    assert!(matches!(out, JitOutcome::Returned(ref s) if s == &[19]));

    // Compact: both U (live handle) and V (installed) must be carried.
    let mut cm2 = recompact_jit(
        &base,
        0,
        DEFAULT_RESERVED_LOG2,
        table_log2,
        &mut host,
        domain,
        &cm,
    )
    .expect("recompact");
    drop(cm); // free the old arena; both units now live only in cm2

    // After: U's trampoline moved, but its handle still names (domain, u_unit) → the Host record was
    // remapped, so invoking through the *new* trampoline gives the same result.
    let u_tramp_after = host.jit_unit_native(domain, u_unit);
    assert_ne!(
        u_tramp_after, u_tramp_before,
        "trampoline must have moved into the fresh arena"
    );
    let (out, _) = unsafe { cm2.run_extra(u_tramp_after as *const u8, 2, 1, &[3, 4], None) }
        .expect("invoke U after compaction");
    assert!(
        matches!(out, JitOutcome::Returned(ref s) if s == &[19]),
        "live invoke-only unit must survive compaction: {out:?}"
    );

    // And V is still reachable at its exact slot via call_indirect (5*6 + 99 = 129).
    let (out, _) = cm2
        .run(&[v_slot as i64, 5, 6], None, Some(1 << 18), None)
        .expect("dispatch V");
    // (slot, x) shell: but V's slot dispatch uses the shell entry (slot, x) -> unit[slot](x,x).
    match out {
        JitOutcome::Returned(s) if s.len() == 1 => assert_eq!(s[0], 5 * 5 + 99),
        other => panic!("dispatch V returned {other:?}"),
    }
}

// ---------------------------------------------------------------------------------------------
// Auto-compacting guest-driven REPL (`svm_run::JitSession`) — the capstone: a *real* guest that
// drives compile/invoke/release through `cap.call` across many prompts over a persistent window,
// with the session auto-compacting at an occupancy watermark. This closes the §6 loop: a long REPL
// that JITs a fresh unit every prompt never exhausts the 256 MiB code arena, and the guest never
// observes the reclaim.

/// The REPL guest entry `(jit_handle, x) -> (i32)`: compile the blob into a fresh unit, invoke it
/// with `(x, x)`, **release** it (so it becomes dead code the next compaction reclaims), and
/// accumulate the result into window[0] (persisted across prompts). `BLOBLEN` is patched to the
/// blob's byte length before parsing.
const REPL_INVOKE_SHELL: &str = "memory 16\nfunc (i32, i32) -> (i32) {\nblock0(v0: i32, v1: i32):\n  v2 = i64.const 4096\n  v3 = i64.const BLOBLEN\n  v4 = cap.call 11 0 (i64, i64) -> (i64) v0 (v2, v3)\n  v5 = cap.call 11 1 (i64, i32, i32) -> (i32) v0 (v4, v1, v1)\n  v6 = cap.call 11 2 (i64) -> (i64) v0 (v4)\n  v7 = i64.const 0\n  v8 = i32.load v7\n  v9 = i32.add v8 v5\n  i32.store v7 v9\n  return v9\n}\n";

/// Drive a `watermark`-auto-compacting `JitSession` for `n` prompts; the blob is seeded at window
/// offset 4096 once and reused every prompt. Returns `(per-prompt results, final window[..16],
/// final occupancy, compactions run)`.
fn run_session(watermark: usize, n: usize) -> (Vec<i64>, Vec<u8>, usize, usize) {
    let blob = unit_blob(10); // x*x + 10
    let base_src = REPL_INVOKE_SHELL.replace("BLOBLEN", &blob.len().to_string());
    let base = parse_module(&base_src).expect("parse shell");
    verify_module(&base).expect("verify shell");

    let mut host = Host::new();
    let jit_h = grant_jit(&mut host, &base, 0); // invoke-only: no install table needed
    let domain = host.resolve_jit_domain(jit_h).expect("domain");
    // The session takes ownership of the host (boxed `Mutex<Host>`, stable address).
    let mut session = JitSession::new(&base, 0, DEFAULT_RESERVED_LOG2, 0, domain, watermark, host)
        .expect("session");

    // Seed the blob into the carried window at offset 4096 (where the guest `cap.call compile`s it);
    // it persists across prompts like any guest state.
    session.seed_window(4096, &blob);

    let mut results = Vec::new();
    for i in 0..n {
        let x = (i as i64) + 2;
        let out = session.run_prompt(&[jit_h as i64, x]).expect("prompt");
        match out {
            JitOutcome::Returned(s) if s.len() == 1 => results.push(s[0]),
            other => panic!("prompt returned {other:?}"),
        }
    }
    (
        results,
        session.window()[..16].to_vec(),
        session.occupancy(),
        session.compactions(),
    )
}

/// **Auto-compaction keeps a long REPL's arena bounded, transparently — on a byte-accurate
/// watermark.** A 30-prompt session that JITs + invokes + releases a fresh unit every prompt
/// produces identical per-prompt results and window state whether auto-compaction is off
/// (`watermark = 0`) or on (a byte watermark of ~4 units), and with it on the session compacts and
/// keeps its **byte** occupancy near the watermark while off it grows with every prompt. The
/// watermark is derived from a one-prompt probe so the test is robust to per-platform code sizes.
#[test]
fn jit_session_auto_compacts_transparently() {
    let n = 30;
    // Probe: one prompt's code-byte cost (a fresh unit + trampoline), to set a watermark that
    // triggers roughly every ~4 prompts regardless of the target's instruction encoding.
    let (_, _, unit_bytes, _) = run_session(0, 1);
    assert!(
        unit_bytes > 0,
        "a compiled unit must report nonzero code bytes"
    );
    let watermark = unit_bytes * 4;

    let (r_off, w_off, occ_off, c_off) = run_session(0, n);
    let (r_on, w_on, occ_on, c_on) = run_session(watermark, n);

    assert_eq!(r_off, r_on, "per-prompt results must be identical");
    assert_eq!(w_off, w_on, "persistent window must be identical");
    // The accumulator advanced: each prompt added x*x + 10 for x = 2..(n+1).
    let expected: i64 = (0..n as i64).map(|i| (i + 2) * (i + 2) + 10).sum();
    assert_eq!(*r_on.last().unwrap(), expected);

    assert_eq!(c_off, 0, "watermark 0 disables auto-compaction");
    assert!(c_on > 0, "the byte watermark must trip auto-compaction");
    assert!(
        occ_off > watermark,
        "without compaction the byte occupancy grows past the watermark: {occ_off}"
    );
    // After each prompt the session compacts once occupancy reaches the watermark; since every unit
    // is released, a compaction drops occupancy to ~0, so the final occupancy is bounded by the
    // watermark plus at most one prompt's worth.
    assert!(
        occ_on <= watermark + unit_bytes && occ_on < occ_off,
        "auto-compaction must bound byte occupancy near the watermark: {occ_on} (off {occ_off}, wm {watermark})"
    );
}

/// A **multi-threaded** REPL shell: `(jit, x) -> (i32)` spawns a worker that compiles+invokes+
/// releases `(7,7)=59` while main compiles+invokes+releases `(x,x)=x*x+10`, joins (so the prompt is
/// quiescent at its end), and accumulates `main+worker` into window[0]. `BLOBLEN` is patched in. Each
/// prompt redefines (compile) + releases two units, so the arena accumulates dead code that
/// compaction reclaims — exactly the single-threaded REPL pattern, but threaded.
const REPL_THREADED_SHELL: &str = "memory 16\nfunc (i32, i32) -> (i32) {\nblock0(v0: i32, v1: i32):\n  v2 = i64.extend_i32_u v0\n  v3 = i64.const 2048\n  v4 = thread.spawn 1 v3 v2\n  v5 = i64.const 4096\n  v6 = i64.const BLOBLEN\n  v7 = cap.call 11 0 (i64, i64) -> (i64) v0 (v5, v6)\n  v8 = cap.call 11 1 (i64, i32, i32) -> (i32) v0 (v7, v1, v1)\n  v9 = cap.call 11 2 (i64) -> (i64) v0 (v7)\n  v10 = thread.join v4\n  v11 = i32.wrap_i64 v10\n  v12 = i64.const 0\n  v13 = i32.load v12\n  v14 = i32.add v13 v8\n  v15 = i32.add v14 v11\n  i32.store v12 v15\n  return v15\n}\nfunc (i64, i64) -> (i64) {\nblock0(v0: i64, v1: i64):\n  v2 = i32.wrap_i64 v1\n  v3 = i64.const 4096\n  v4 = i64.const BLOBLEN\n  v5 = cap.call 11 0 (i64, i64) -> (i64) v2 (v3, v4)\n  v6 = i32.const 7\n  v7 = cap.call 11 1 (i64, i32, i32) -> (i32) v2 (v5, v6, v6)\n  v8 = cap.call 11 2 (i64) -> (i64) v2 (v5)\n  v9 = i64.extend_i32_u v7\n  return v9\n}\n";

/// Drive a `watermark`-auto-compacting `JitSession` for a **multi-threaded** guest: each prompt
/// spawns a worker that concurrently `Jit.compile`s (so the session's `Mutex<Host>` serialization is
/// exercised), the prompt joins it (quiescent end), and compaction runs between prompts. Returns
/// `(per-prompt results, final window[..8], final occupancy, compactions)`.
#[cfg(any(
    all(unix, target_arch = "x86_64"),
    all(unix, target_arch = "aarch64"),
    all(windows, target_arch = "x86_64")
))]
fn run_threaded_session(watermark: usize, n: usize) -> (Vec<i64>, Vec<u8>, usize, usize) {
    let blob = unit_blob(10);
    let base_src = REPL_THREADED_SHELL.replace("BLOBLEN", &blob.len().to_string());
    let base = parse_module(&base_src).expect("parse threaded shell");
    verify_module(&base).expect("verify threaded shell");

    let mut host = Host::new();
    let jit_h = grant_jit(&mut host, &base, 0);
    let domain = host.resolve_jit_domain(jit_h).expect("domain");
    let mut session = JitSession::new(&base, 0, DEFAULT_RESERVED_LOG2, 0, domain, watermark, host)
        .expect("session");
    session.seed_window(4096, &blob);

    let mut results = Vec::new();
    for i in 0..n {
        let x = (i as i64) + 2;
        let out = session.run_prompt(&[jit_h as i64, x]).expect("prompt");
        match out {
            JitOutcome::Returned(s) if s.len() == 1 => results.push(s[0]),
            other => panic!("threaded prompt returned {other:?}"),
        }
    }
    (
        results,
        session.window()[..8].to_vec(),
        session.occupancy(),
        session.compactions(),
    )
}

/// **Compaction works for a multi-threaded guest — byte-accurate watermark.** A 12-prompt session
/// whose every prompt spawns a worker that concurrently `Jit.compile`s produces identical per-prompt
/// results and window state whether auto-compaction is off (`watermark=0`) or on (a byte watermark
/// of ~2 prompts), and with it on the session compacts and bounds its **byte** occupancy. This pins
/// both the threaded-compile serialization through `JitSession`'s `Mutex<Host>` and that compaction
/// (a quiescent, between-prompts rebuild re-baking the locked thunk) is transparent across it.
#[test]
#[cfg(any(
    all(unix, target_arch = "x86_64"),
    all(unix, target_arch = "aarch64"),
    all(windows, target_arch = "x86_64")
))]
fn threaded_session_compacts_transparently() {
    let n = 12;
    // Probe one prompt's code-byte cost (main + worker each compile a unit), so the watermark is
    // robust to per-platform code sizes.
    let (_, _, prompt_bytes, _) = run_threaded_session(0, 1);
    assert!(
        prompt_bytes > 0,
        "a prompt's compiled units must report nonzero code bytes"
    );
    let watermark = prompt_bytes * 2;

    let (r_off, w_off, occ_off, c_off) = run_threaded_session(0, n);
    let (r_on, w_on, occ_on, c_on) = run_threaded_session(watermark, n);

    assert_eq!(
        r_off, r_on,
        "per-prompt results must match with/without compaction"
    );
    assert_eq!(w_off, w_on, "persistent window must match");
    // Each prompt adds main (x*x+10, x=2..n+1) + worker (7*7+10 = 59).
    let expected: i64 = (0..n as i64).map(|i| (i + 2) * (i + 2) + 10 + 59).sum();
    assert_eq!(*r_on.last().unwrap(), expected);

    assert_eq!(c_off, 0, "watermark 0 disables auto-compaction");
    assert!(
        c_on > 0,
        "the byte watermark must trip auto-compaction in the threaded session"
    );
    assert!(
        occ_off > watermark,
        "no-compaction byte occupancy grows past the watermark, got {occ_off}"
    );
    assert!(
        occ_on <= watermark + prompt_bytes && occ_on < occ_off,
        "auto-compaction bounds the threaded session's byte occupancy: {occ_on} (off {occ_off}, wm {watermark})"
    );
}
