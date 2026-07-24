//! `svm-durable` — the IR→IR freeze/thaw transform (DESIGN.md / DURABILITY.md D60).
//!
//! A **tooling-tier, non-TCB** crate (like `svm-text`): it depends only on `svm-ir`
//! and emits ordinary, verifier-passing IR — no new instructions, no escape-TCB
//! surface. An embedder running pre-instrumented modules links none of it.
//!
//! This is the **Phase 1** slice of the plan (DURABILITY.md §9): it instruments a
//! function so an in-flight may-suspend op (a `cap.call`, or a `Call` into a suspended
//! chain) can be *unwound* into guest-resident shadow state and later *rewound* back into
//! execution, byte-for-byte. The codec is exactly the §2 mechanism:
//!
//! * a **state word** (`NORMAL | UNWINDING | REWINDING`) in the window,
//! * a per-fiber **shadow stack** in the window (DURABILITY.md §12.7),
//! * **unwind** = after a may-suspend call, if `UNWINDING`, spill the live values +
//!   resume id and return out to the host;
//! * **rewind** = in the prologue, if `REWINDING`, `br_table` on the saved resume id,
//!   reload the live values, and continue from the resume point.
//!
//! # Scope (arbitrary single-vCPU CFGs)
//!
//! The transform handles an arbitrary-CFG function (branches, loops, joins) with any
//! number of may-suspend operations, each either:
//!
//! * a **leaf** `cap.call` — the host performs the operation; on thaw the deepest frame
//!   reloads the saved result and flips the state word back to `NORMAL`; or
//! * a **propagated** `Call` to a may-suspend callee (a function that transitively
//!   reaches a `cap.call`) — frames stack up across the call chain. On thaw a non-deepest
//!   frame reloads its pre-call live set and **re-issues the call** (leaving the state
//!   `REWINDING` so the callee rewinds in turn); only the innermost leaf flips to
//!   `NORMAL`. This is the DURABILITY.md §12.7 "re-issue vs. continue" branch (R8).
//!
//! Each original block is split at its suspend ops into forward segments; branch targets
//! are remapped to the target block's first segment; a `br_table` in the prologue dispatch
//! routes a thaw to the resume point that was in flight (one arm per point). A function is
//! **may-suspend** iff it contains a `cap.call` or (transitively) a `Call` to a may-suspend
//! function; only may-suspend functions are instrumented. Covered end-to-end on the real
//! interpreter (`tests/roundtrip.rs`, `chain.rs`, `multipoint.rs`, `multiblock.rs`) and
//! across the interp/JIT differential (`crates/svm/tests/durable_jit.rs`).
//!
//! Each resume point spills only its **minimal live set** — the values used after the op
//! (block-local SSA makes this a backward scan within the block), plus a propagated call's
//! own operands. An unmodelled instruction in the tail falls back to spilling the whole
//! range, so the analysis never under-spills.
//!
//! Out of scope (rejected / treated non-suspending): `call_indirect` and indirect tail
//! calls to a may-suspend target (unresolved); a direct tail call into a may-suspend
//! callee (the frame is replaced — no poll to unwind at).
//!
//! The remaining extensions (DURABILITY.md §9) are fibers / multi-vCPU / STW (Phase 3).

#![forbid(unsafe_code)]

use svm_ir::{
    BinOp, Block, BlockIdx, CmpOp, Func, FuncIdx, Inst, IntTy, LoadOp, Module, StoreOp, Terminator,
    ValIdx, ValType,
};

// `FuncIdx` is used by `SuspendKind::Propagated` below.

/// The reserved self-namespace op for `svc.poll` (IMPORTS.md §3.6). A local copy — this crate
/// depends only on `svm-ir` — pinned equal to `svm_interp::CAP_SELF_SVC_POLL` by a dev-test
/// (`tests/serve.rs`).
pub const SVC_POLL_OP: u32 = 9;
/// The reserved self-namespace op for `svc.wait`; pinned equal to `svm_interp::CAP_SELF_SVC_WAIT`.
pub const SVC_WAIT_OP: u32 = 10;

// ---- State word values (the §2 state machine) ----

/// Normal forward execution; polls and prologues fall straight through.
pub const STATE_NORMAL: i32 = 0;
/// Freeze in progress: every poll after a may-suspend call unwinds out to the host.
pub const STATE_UNWINDING: i32 = 1;
/// Thaw in progress: every prologue rebuilds its frame from the shadow stack.
pub const STATE_REWINDING: i32 = 2;
/// Freeze **armed** (the deterministic mid-run trigger): the run executes normally, but the runtime
/// counts down [`ARM_COUNTDOWN_OFF`] at each **fiber safepoint** (`cont.resume`/`suspend`) and, on
/// reaching 0, promotes the word to [`STATE_UNWINDING`] so that op's trailing poll begins the freeze.
/// Transparent to the instrumented IR — every emitted poll/prologue tests only `UNWINDING`/`REWINDING`,
/// so an `ARMED` run reads as `NORMAL` until the runtime promotes it. Lets a single-threaded test
/// freeze *after N fiber safepoints of forward progress* (e.g. after a fiber has been recycled), which
/// the freeze-before-start harness cannot; it also models an async controller flipping `UNWINDING` from
/// another thread, which the existing mechanism already picks up at the next poll. Both backends count
/// the same set (the fiber ops, routed through runtime thunks), so an armed freeze lands at the same
/// safepoint on each — `cap.call` is not counted (no cross-backend choke; its freeze is already
/// reachable at the first safepoint).
pub const STATE_ARMED: i32 = 3;

// ---- Durable runtime region layout ----
//
// The control state + shadow stack occupy a fixed **reserved low slice** `[0,
// DURABLE_RESERVE)` of the domain's *own* window; the guest's memory is `[DURABLE_RESERVE,
// window)`. This is the wasm shadow-stack convention (runtime metadata + call stack below
// `__heap_base`, the program's heap above it): the durable reserve is part of the guest's
// memory allotment, and a cooperating toolchain bases the guest's data/heap at
// `DURABLE_RESERVE` so the two never overlap (see `transform_module_assume_confined`).
//
// This is *placement*, not an isolation boundary: the window is per-domain and the runtime
// masks every access into it, so a guest that writes the reserve can only corrupt its own
// durability — never another domain or the host — and that fails safe (a forged resume id
// hits the `br_table` default → `Unreachable`; a wild shadow-SP stays masked in-window; the
// host validates the artifact on restore). Hardening the reserve against an *adversarial*
// guest (a guard-paged, per-fiber placement, DURABILITY.md §12.7) is optional
// defense-in-depth, not required for a cooperating toolchain.

/// Window byte offset of the `i32` state word.
pub const STATE_OFF: u64 = 0;
/// Window byte offset of the `i64` shadow-stack pointer (a window byte offset itself).
pub const SHADOW_SP_OFF: u64 = 8;
/// Window byte offset of the `i64` **arm countdown** — the number of fiber safepoints still to pass
/// before an [`STATE_ARMED`] run promotes itself to [`STATE_UNWINDING`]. Decremented by the runtime at
/// each fiber safepoint (`cont.resume`/`suspend`); inert unless the state word is `ARMED`. Lives in the
/// reserve's previously-unused `[16, 64)` gap, so it is byte-identical to before for any run that never
/// arms (countdown stays 0).
pub const ARM_COUNTDOWN_OFF: u64 = 16;
/// Window byte offset of the `i64` **back-edge arm countdown** — the number of loop back-edges
/// (branch terminators) still to pass before an [`STATE_ARMED`] run promotes itself to
/// [`STATE_UNWINDING`], so a loop-header poll begins the freeze. The deterministic trigger for the
/// Phase-4 Slice A back-edge polls, separate from [`ARM_COUNTDOWN_OFF`] (which counts only fiber
/// safepoints) so an ordinary or fiber-armed run is byte-identical (this slot stays 0). Lives in the
/// reserve's `[24, 64)` gap.
pub const ARM_BACKEDGE_OFF: u64 = 24;
/// §12.8 concurrent-thaw stage 1: byte offset of the per-context **thaw** state word
/// (`REWINDING`/`NORMAL`) **within a context's region** — just past the 8-byte in-region shadow-SP word
/// at the region base, addressed via the `durable.shadow_base` register (like the SP word). Each frozen
/// vCPU rewinds against its *own* thaw word, so the thaw can run them as concurrent OS threads with no
/// shared word: one vCPU finishing its rewind (flipping its word to `NORMAL`) can't disturb a sibling
/// still `REWINDING`, and a forward vCPU's callee prologue can't read another's `REWINDING`. The
/// **freeze** state (`UNWINDING`) stays at the global [`STATE_OFF`] — a freeze is stop-the-world, so the
/// single word is the natural broadcast every poll reads. Must equal `svm-interp`/`svm-jit`'s copy.
pub const STATE_IN_REGION_OFF: u64 = 8;
/// §12.8 concurrent-thaw stage 1: bytes reserved at a context region's base before its shadow frames —
/// the 8-byte shadow-SP word plus the 4-byte thaw state word at [`STATE_IN_REGION_OFF`], padded to 8 to
/// keep frames 8-aligned. [`shadow_frame_base`]-equivalents in every backend start here. Must equal
/// `svm-interp`/`svm-jit`'s copy.
pub const REGION_HEADER_LEN: u64 = 16;

/// Window byte offset where the shadow stack begins (grows upward, bounded by `DURABLE_RESERVE`).
pub const SHADOW_BASE: u64 = 64;
/// Per-context shadow-region stride: context `i` owns `[SHADOW_BASE + i*SHADOW_STRIDE, +SHADOW_STRIDE)`
/// (§12.8 4A.5). The transform itself never addresses a region (it emits `durable.shadow_base`-relative
/// loads the runtime resolves), but the [`write_thaw_state`] host helper indexes a context's region.
/// Must equal `svm-interp`/`svm-jit`'s `SHADOW_STRIDE`.
pub const SHADOW_STRIDE: u64 = 1 << 12;
/// Size of the reserved low region (one 64 KiB wasm page): `[0, DURABLE_RESERVE)` holds the
/// state word, shadow-SP, and shadow stack; the guest's memory is `[DURABLE_RESERVE, window)`.
/// A durable module's declared window must be at least this large (it is counted against the
/// guest's memory budget). A policy default — embedders may standardize a different value.
pub const DURABLE_RESERVE: u64 = 1 << 16;

// Block layout of an instrumented function with `S` forward segments (each original
// block is split at its suspend ops into `points+1` segments; non-suspend blocks are one
// segment) and `P` resume points total (an 8-block shape for one block / one point):
//   0                  PROLOGUE  — dispatch on the state word, then enter segment 0 of blk 0
//   1 ..= S            forward segments (each original block's segments, in block order)
//   1+S                DISPATCH  — read resume id, br_table to an arm
//   2+S ..= 1+S+2P     UNWIND_g  — a (check, spill) pair per point: trap if the push would
//                                  overflow the reserve, else spill + return placeholder
//   2+S+2P ..= 1+S+3P  ARM_g     — reload + resume, per resume point
//   2+S+3P             TRAP      — forged/reserved resume id, or shadow-stack overflow

/// Reasons the Phase-1 transform declines a module.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum TransformError {
    /// A function uses `cap.call` but the module declares no memory window — the
    /// shadow stack and state word have nowhere to live.
    NoMemory,
    /// The declared window is too small to hold the durable region + a shadow frame.
    MemoryTooSmall,
    /// A `cap.call`-bearing function is outside the Phase-1 shape (not a single block,
    /// not exactly one `cap.call`, or not a `return` terminator).
    UnsupportedShape,
    /// A prefix instruction's result type isn't modelled by the Phase-1 transform
    /// (e.g. SIMD, conversions, concurrency ops before the call).
    UnsupportedInst,
    /// The module is being instrumented via the strict [`transform_module`] path but a
    /// function uses a guest linear-memory instruction (load/store/atomic), which could
    /// alias the reserved durable region `[0, DURABLE_RESERVE)` (R9). The strict path fails
    /// closed for *untrusted* modules. A durable module from a cooperating toolchain that
    /// reserves `[0, DURABLE_RESERVE)` (basing the guest's data/heap at `DURABLE_RESERVE`)
    /// should instead use [`transform_module_assume_confined`]. (Cap-mediated window effects
    /// — e.g. a Memory capability's map/unmap — are a separate facet the embedder withholds.)
    GuestUsesMemory,
}

/// Instrument every may-suspend function in `m` for freeze/thaw. Functions that can
/// never suspend are returned unchanged. The result is ordinary IR; run it through
/// `svm_verify::verify_module` before executing.
pub fn transform_module(m: &Module) -> Result<Module, TransformError> {
    transform_module_inner(m, true)
}

/// Like [`transform_module`], but **allows the guest to use linear memory**, on the caller's
/// guarantee that the guest is confined to its usable region `[DURABLE_RESERVE, window)` and
/// never touches the reserved durable slice `[0, DURABLE_RESERVE)`.
///
/// This is the intended path for durable modules produced by a **cooperating toolchain**:
/// just as a wasm toolchain reserves low memory for the shadow stack and bases the heap at
/// `__heap_base`, the producer reserves `[0, DURABLE_RESERVE)` and bases the guest's
/// data/heap at `DURABLE_RESERVE`. The reserve is budget-accounted (the declared window must
/// be ≥ `DURABLE_RESERVE`). The contract is *not* statically enforced here — a guest that
/// violates it can corrupt only its own durability, and fails safe (see the region notes) —
/// so prefer [`transform_module`] (fails closed, no memory) for *untrusted* modules.
pub fn transform_module_assume_confined(m: &Module) -> Result<Module, TransformError> {
    transform_module_inner(m, false)
}

fn transform_module_inner(m: &Module, enforce_r9: bool) -> Result<Module, TransformError> {
    let func_results: Vec<Vec<ValType>> = m.funcs.iter().map(|f| f.results.clone()).collect();
    let may_suspend = compute_may_suspend(m);
    let any_instrumented = may_suspend.iter().any(|&s| s);

    // R9 enforcement: the durable region shares the window with guest memory at fixed low
    // addresses, with nothing confining the guest away from it. Rather than risk silent
    // mutual corruption, refuse to instrument a module any of whose functions touch linear
    // memory. (The generated/tested durable guests are pure SSA + `cap.call`, so they pass;
    // the relocation that would lift this restriction is §12.7 future work.)
    if enforce_r9
        && any_instrumented
        && m.funcs
            .iter()
            .any(|f| f.blocks.iter().any(|b| b.insts.iter().any(is_guest_mem_op)))
    {
        return Err(TransformError::GuestUsesMemory);
    }

    let mut out = m.clone();
    let mut max_frame = 0u64;

    for (i, f) in m.funcs.iter().enumerate() {
        if may_suspend[i] {
            let (nf, frame_size) = transform_func(f, &func_results, &may_suspend)?;
            out.funcs[i] = nf;
            max_frame = max_frame.max(frame_size);
        }
    }

    if any_instrumented {
        let mem = out.memory.ok_or(TransformError::NoMemory)?;
        // The reserved region `[0, DURABLE_RESERVE)` must fit in the declared window (it is
        // part of the guest's allotment; guest memory is the remainder `[DURABLE_RESERVE,
        // window)`), and a single shadow frame must fit in `[SHADOW_BASE, DURABLE_RESERVE)`.
        // A live call chain stacks one frame per suspended activation; the reserve bounds the
        // total depth (overflow-trapping the shadow stack is DURABILITY.md §12.7 future work).
        if mem.size() < DURABLE_RESERVE || SHADOW_BASE + max_frame > DURABLE_RESERVE {
            return Err(TransformError::MemoryTooSmall);
        }
    }
    Ok(out)
}

/// A guest linear-memory instruction (one that reads or writes a window address). These
/// can alias the durable region, so they are rejected in an instrumented module (R9). An
/// `AtomicFence` carries no address and so is not included.
fn is_guest_mem_op(inst: &Inst) -> bool {
    matches!(
        inst,
        Inst::Load { .. }
            | Inst::Store { .. }
            | Inst::AtomicLoad { .. }
            | Inst::AtomicStore { .. }
            | Inst::AtomicRmw { .. }
            | Inst::AtomicCmpxchg { .. }
            | Inst::V128Load { .. }
            | Inst::V128Store { .. }
            | Inst::MemoryWait { .. } // reads the window value at `addr`
    )
}

/// The value operands an instruction reads, or `None` for a variant this pass does not
/// model — in which case liveness conservatively assumes it uses everything (so we never
/// drop a still-live value). Covers the set `result_types` admits (anything else is already
/// rejected as `UnsupportedInst` before liveness runs).
fn inst_operands(i: &Inst) -> Option<Vec<ValIdx>> {
    use Inst::*;
    Some(match i {
        ConstI32(_) | ConstI64(_) | ConstF32(_) | ConstF64(_) | ConstV128(_) | RefFunc { .. } => {
            vec![]
        }
        IntBin { a, b, .. }
        | IntCmp { a, b, .. }
        | FBin { a, b, .. }
        | FCmp { a, b, .. }
        | PtrAdd { a, b } => vec![*a, *b],
        IntUn { a, .. } | FUn { a, .. } | Eqz { a, .. } | PtrCast { a, .. } => vec![*a],
        Select { cond, a, b } => vec![*cond, *a, *b],
        Load { addr, .. } | AtomicLoad { addr, .. } | V128Load { addr, .. } => vec![*addr],
        Store { addr, value, .. }
        | AtomicStore { addr, value, .. }
        | AtomicRmw { addr, value, .. }
        | V128Store { addr, value, .. } => vec![*addr, *value],
        AtomicCmpxchg {
            addr,
            expected,
            replacement,
            ..
        } => vec![*addr, *expected, *replacement],
        AtomicFence { .. } => vec![],
        MemoryWait {
            addr,
            expected,
            timeout,
            ..
        } => vec![*addr, *expected, *timeout],
        MemoryNotify { addr, count } => vec![*addr, *count],
        Call { args, .. } => args.clone(),
        CapCall { handle, args, .. } => {
            let mut v = Vec::with_capacity(args.len() + 1);
            v.push(*handle);
            v.extend_from_slice(args);
            v
        }
        CallIndirect { idx, args, .. } => {
            let mut v = Vec::with_capacity(args.len() + 1);
            v.push(*idx);
            v.extend_from_slice(args);
            v
        }
        ContNew { func, sp } => vec![*func, *sp],
        ContResume { k, arg } => vec![*k, *arg],
        Suspend { value } => vec![*value],
        _ => return None,
    })
}

/// The value operands a terminator reads (a closed set, all handled).
fn term_operands(t: &Terminator) -> Vec<ValIdx> {
    match t {
        Terminator::Br { args, .. } => args.clone(),
        Terminator::BrIf {
            cond,
            then_args,
            else_args,
            ..
        } => {
            let mut v = vec![*cond];
            v.extend_from_slice(then_args);
            v.extend_from_slice(else_args);
            v
        }
        Terminator::BrTable {
            idx,
            targets,
            default,
        } => {
            let mut v = vec![*idx];
            for (_, a) in targets {
                v.extend_from_slice(a);
            }
            v.extend_from_slice(&default.1);
            v
        }
        Terminator::Return(vals) => vals.clone(),
        Terminator::ReturnCall { args, .. } => args.clone(),
        Terminator::ReturnCallIndirect { idx, args, .. } => {
            let mut v = vec![*idx];
            v.extend_from_slice(args);
            v
        }
        Terminator::Unreachable => vec![],
    }
}

/// The block targets a terminator branches to (a closed set; tail calls / returns carry none).
fn term_targets(t: &Terminator) -> Vec<BlockIdx> {
    match t {
        Terminator::Br { target, .. } => vec![*target],
        Terminator::BrIf {
            then_blk, else_blk, ..
        } => vec![*then_blk, *else_blk],
        Terminator::BrTable {
            targets, default, ..
        } => {
            let mut v: Vec<BlockIdx> = targets.iter().map(|(t, _)| *t).collect();
            v.push(default.0);
            v
        }
        Terminator::Return(_)
        | Terminator::ReturnCall { .. }
        | Terminator::ReturnCallIndirect { .. }
        | Terminator::Unreachable => vec![],
    }
}

/// Mark each function that can suspend: it contains a `cap.call`, or (transitively) a
/// direct `Call` to a may-suspend function. A least-fixed-point over the direct-call
/// graph. `call_indirect` targets are unresolved and treated as non-suspending (see the
/// module-level scope note).
fn compute_may_suspend(m: &Module) -> Vec<bool> {
    let mut ms = vec![false; m.funcs.len()];
    for (i, f) in m.funcs.iter().enumerate() {
        if f.blocks.iter().any(|b| {
            b.insts.iter().any(|x| {
                // `cap.call` suspends to the host; a fiber `cont.resume`/`suspend` switches
                // stacks and is a freeze safepoint too (`cont.new` alone merely allocates).
                matches!(
                    x,
                    Inst::CapCall { .. }
                        | Inst::ContResume { .. }
                        | Inst::Suspend { .. }
                        | Inst::ThreadJoin { .. }
                        | Inst::MemoryWait { .. }
                )
            })
        }) {
            ms[i] = true;
        }
    }
    loop {
        let mut changed = false;
        for (i, f) in m.funcs.iter().enumerate() {
            if ms[i] {
                continue;
            }
            let calls_ms = f.blocks.iter().any(|b| {
                b.insts
                    .iter()
                    .any(|x| matches!(x, Inst::Call { func, .. } if ms[*func as usize]))
                    // a direct tail call into a may-suspend callee also suspends (rejected
                    // by `transform_func` as out of scope, but it must be marked so the
                    // module fails closed rather than leaving the caller uninstrumented)
                    || matches!(&b.term, Terminator::ReturnCall { func, .. } if ms[*func as usize])
            });
            if calls_ms {
                ms[i] = true;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    ms
}

/// The single may-suspend operation in an instrumented block.
enum SuspendKind {
    /// `cap.call`: the host performs the op; the deepest frame reloads its result.
    Leaf,
    /// `Call` to a may-suspend callee: re-issued on thaw so the callee rewinds in turn.
    Propagated { callee: FuncIdx, args: Vec<ValIdx> },
    /// `cont.resume` (resumer side): like a propagated call, **re-issued on thaw** so the fiber
    /// rewinds in turn and redelivers its `(status: i32, value: i64)` (slice 3.1.2). The
    /// re-issued resume reconstructs the fiber via its own rewind (the `Yield` re-park, slice
    /// 3.1.3) — until that lands, a thaw that actually re-enters a suspended fiber still relies
    /// on the fiber side being wired.
    Resume { k: ValIdx, arg: ValIdx },
    /// `thread.join` (§12.8 next slice): a vCPU blocked joining a child is a freeze safepoint too — the
    /// `thread_join` thunk returns a sentinel on observing `UNWINDING`, the trailing poll unwinds, and
    /// the join is **re-issued on thaw** (like a propagated call / `cont.resume`): by then the child has
    /// been re-spawned and run to completion, so the re-executed join resolves immediately to its
    /// recorded result. `handle` is the joined vCPU handle's block-local index (spilled + reloaded).
    ThreadJoin { handle: ValIdx },
    /// `<ty>.atomic.wait` (§12.8 parked-vCPU slice): a vCPU blocked in a futex wait is a freeze
    /// safepoint too — the `thread_wait` thunk returns on observing `UNWINDING`, the trailing poll
    /// unwinds, and the wait is **re-issued on thaw** (like `thread.join`): the re-executed wait
    /// re-checks the guest value, so a wake that landed as a value change (already in the snapshot, or
    /// replayed by another re-run vCPU) resolves it immediately with `WAIT_NOT_EQUAL`. A re-issue that
    /// would still *park* on the single-worker thaw can't be satisfied (no concurrent notifier) and
    /// fails closed (`ThreadFault`, matching the interp's join-deadlock). `ty` reconstructs the op;
    /// `addr`/`expected`/`timeout` are its block-local operands (spilled + reloaded).
    MemoryWait {
        ty: IntTy,
        addr: ValIdx,
        expected: ValIdx,
        timeout: ValIdx,
    },
    /// `suspend` (fiber side): unwinds the fiber's stack like a leaf, and thaw **re-parks** the
    /// fiber — flips the state word to `NORMAL` (a parked fiber's suspend is the globally-deepest
    /// frozen frame) and re-executes `suspend` so control returns to the resumer awaiting a future
    /// `cont.resume` (slice 3.1.3). `value` is the suspended value's block-local index.
    Yield { value: ValIdx },
    /// A **serve op** (`cap.call CAP_SELF svc.poll/svc.wait` — DURABILITY.md §13.4 slice 4b): a
    /// serving domain frozen at (or parked in) its serve point is a freeze safepoint — the serve
    /// arm delivers an inert sentinel on observing `UNWINDING` (no drain, no park; the queue
    /// stays untouched for the snapshot's serve section), the trailing poll unwinds, and the op
    /// is **re-issued on thaw** (like `atomic.wait`): the re-executed drain runs against the
    /// *restored* queue — re-execution is the recovery, so the sentinel is never captured. A
    /// mid-handler freeze is refused up front (the serve epilogue's fail-closed gate), so this
    /// point is always the globally-deepest frozen frame on its thread: flip the state word to
    /// `NORMAL` itself, then reload the handle + args and re-execute. The op immediates
    /// (`type_id`/`op`/`sig`) reconstruct the instruction; `handle`/`args` are its block-local
    /// operands (spilled + reloaded).
    SvcServe {
        type_id: u32,
        op: u32,
        sig: svm_ir::FuncType,
        handle: ValIdx,
        args: Vec<ValIdx>,
    },
    /// A **loop-header poll** (Phase-4 Slice A): a state-word check prepended to a loop header's
    /// entry (the header dominates its body, so a poll-free compute loop is caught every iteration
    /// at bounded latency, closing the R6 latency caveat). It has no in-flight op — it is always
    /// the globally-deepest frozen frame (any may-suspend op in the body would have unwound at its
    /// own poll first), so on thaw it behaves like a leaf: flip the state word to `NORMAL`, reload
    /// the header's block params, and re-enter the header body.
    LoopHeader,
}

/// One resume point's metadata across the whole function (global id by vector order).
struct PointPlan {
    kind: SuspendKind,        // leaf cap.call or propagated call
    nres: usize,              // result count of the suspend op
    out: usize,               // value count after the op (= continuation param count)
    save_end: usize,          // spillable range `[0, save_end)` (excludes a call's results)
    slot_types: Vec<ValType>, // types of values `0..out` (continuation params / dead-slot zeros)
    spilled: Vec<usize>,      // block-local indices actually spilled (sorted, live ∪ call-args)
    frame_offsets: Vec<u64>,  // window offset of each spilled value (parallel to `spilled`)
    frame_size: u64,
    rid_off: u64,
    cont_seg: u32, // new block index of the continuation segment (after the op)
}

/// Per-original-block analysis: value types, the value count after each instruction, and
/// the positions of its may-suspend ops.
struct BlockInfo {
    types: Vec<ValType>,
    vend: Vec<usize>,
    scs: Vec<usize>,
    plen: usize,
}

/// Instrument one may-suspend function. `Ok((func, max_frame_size))` on success, or an
/// error for an out-of-scope shape. (Non-may-suspend functions are not passed here.)
///
/// Each original block is split at its may-suspend ops into forward segments; a poll after
/// each op unwinds (per-point spill + resume id) or continues to the next segment, and the
/// prologue's `br_table` dispatch routes a thaw to the in-flight point's arm, which reloads
/// and resumes into the continuation segment. Branch targets are remapped to segment 0 of
/// the target block. See the block-layout map near the constants.
fn transform_func(
    f: &Func,
    func_results: &[Vec<ValType>],
    may_suspend: &[bool],
) -> Result<(Func, u64), TransformError> {
    // Out of scope: a direct tail call into a may-suspend callee (the frame is replaced, so
    // there is no poll to unwind at). An *indirect* tail call is treated as non-suspending
    // (its target is unresolved — same stance as `call_indirect`, see the module doc).
    for blk in &f.blocks {
        if matches!(&blk.term, Terminator::ReturnCall { func, .. } if may_suspend[*func as usize]) {
            return Err(TransformError::UnsupportedShape);
        }
    }

    let nb = f.blocks.len();
    // Per-block analysis (value types / counts / suspend positions).
    let mut binfo: Vec<BlockInfo> = Vec::with_capacity(nb);
    for blk in &f.blocks {
        let mut types = blk.params.clone();
        let mut vend = Vec::with_capacity(blk.insts.len());
        for inst in &blk.insts {
            types.extend(result_types(inst, &types, func_results)?);
            vend.push(types.len());
        }
        let scs: Vec<usize> = blk
            .insts
            .iter()
            .enumerate()
            .filter(|(_, inst)| match inst {
                Inst::CapCall { .. }
                | Inst::ContResume { .. }
                | Inst::Suspend { .. }
                | Inst::ThreadJoin { .. }
                | Inst::MemoryWait { .. } => true,
                Inst::Call { func, .. } => may_suspend[*func as usize],
                _ => false,
            })
            .map(|(pos, _)| pos)
            .collect();
        binfo.push(BlockInfo {
            types,
            vend,
            scs,
            plen: blk.params.len(),
        });
    }

    let inblock_points: usize = binfo.iter().map(|bi| bi.scs.len()).sum();
    if inblock_points == 0 {
        return Err(TransformError::UnsupportedShape); // may-suspend, but no in-block op
    }

    // A block is a *loop header* if a back-edge (a branch whose target index ≤ its source block)
    // targets it. A poll prepended to the header's entry — which dominates the loop body — is hit
    // every iteration, so a poll-free compute loop freezes at bounded latency (Phase-4 Slice A,
    // the R6 caveat). Each header adds one resume point (a `LoopHeader` poll) and one segment (the
    // poll itself, ahead of the header's body segments).
    let mut is_header = vec![false; nb];
    for (b, blk) in f.blocks.iter().enumerate() {
        for t in term_targets(&blk.term) {
            if (t as usize) <= b {
                is_header[t as usize] = true;
            }
        }
    }
    let header_count = is_header.iter().filter(|&&h| h).count();
    let total_points = inblock_points + header_count;

    // Block-index layout (see the map near the constants).
    let mut seg_base = Vec::with_capacity(nb);
    let mut acc = 1u32; // segment indices start right after the PROLOGUE
    for (b, bi) in binfo.iter().enumerate() {
        seg_base.push(acc);
        // points + 1 segments, + 1 poll segment ahead of the body when this block is a header.
        acc += bi.scs.len() as u32 + 1 + is_header[b] as u32;
    }
    let s_total = acc - 1;
    // Body segment `k` of block `b`. A header's poll segment sits at `seg_base[b]` (= `seg0(b)`,
    // the branch-entry target), so body segments start one past it.
    let seg = |b: usize, k: usize| seg_base[b] + is_header[b] as u32 + k as u32;
    let p_total = total_points as u32;
    let dispatch_blk = 1 + s_total;
    // Each resume point has a UNWIND *pair*: a check block (traps if the push would exceed
    // the reserve) and a spill block. `check_blk(g) = unwind_base + 2g`, spill is +1.
    let unwind_base = 2 + s_total;
    let arm_base = unwind_base + 2 * p_total;
    let trap_blk = arm_base + p_total;
    let p = f.params.len();

    // Remap a terminator's block targets to segment 0 of each target block.
    let seg0 = |t: BlockIdx| seg_base[t as usize];
    let remap = |term: &Terminator| -> Terminator {
        match term {
            Terminator::Br { target, args } => Terminator::Br {
                target: seg0(*target),
                args: args.clone(),
            },
            Terminator::BrIf {
                cond,
                then_blk,
                then_args,
                else_blk,
                else_args,
            } => Terminator::BrIf {
                cond: *cond,
                then_blk: seg0(*then_blk),
                then_args: then_args.clone(),
                else_blk: seg0(*else_blk),
                else_args: else_args.clone(),
            },
            Terminator::BrTable {
                idx,
                targets,
                default,
            } => Terminator::BrTable {
                idx: *idx,
                targets: targets.iter().map(|(t, a)| (seg0(*t), a.clone())).collect(),
                default: (seg0(default.0), default.1.clone()),
            },
            // Return / Unreachable / (in)direct tail calls carry no block targets.
            other => other.clone(),
        }
    };

    // ---- PROLOGUE — dispatch on the state word ----
    let mut pb = Bb::new(f.params.clone());
    let (st_a, st_off) = pb.thaw_word_addr();
    let st = pb.one(load(LoadOp::I32, st_a, st_off));
    let rw = pb.one(Inst::ConstI32(STATE_REWINDING));
    let is_rw = pb.one(icmp(IntTy::I32, CmpOp::Eq, st, rw));
    let prologue = pb.finish(Terminator::BrIf {
        cond: is_rw,
        then_blk: dispatch_blk,
        then_args: vec![], // the arm reloads everything from the frame
        else_blk: seg(0, 0),
        else_args: (0..p as u32).collect(),
    });

    // ---- forward segments + collect the per-point resume plans (global order) ----
    let mut seg_blocks: Vec<Block> = Vec::with_capacity(s_total as usize);
    let mut points: Vec<PointPlan> = Vec::with_capacity(total_points);
    for (b, blk) in f.blocks.iter().enumerate() {
        let bi = &binfo[b];
        let m = bi.scs.len();
        // segment k's incoming value count: block params for k==0, else the previous op's out
        let in_of = |k: usize| {
            if k == 0 {
                bi.plen
            } else {
                bi.vend[bi.scs[k - 1]]
            }
        };
        // Loop-header poll: a state-word check at the header's entry (`seg_base[b]` = `seg0(b)`,
        // the branch-entry target), ahead of the body segments. On UNWINDING it spills the
        // header's block params (the loop-carried live set) into a fresh `LoopHeader` resume
        // point and returns; otherwise it falls through into the body. Built before the in-block
        // points so its `gid` precedes them (the index layout only requires `points` order to
        // match the unwind/arm block order, which it does).
        if is_header[b] {
            let plen = bi.plen;
            let slot_types = bi.types[0..plen].to_vec();
            if slot_types.contains(&ValType::V128) {
                return Err(TransformError::UnsupportedInst); // v128 spill/reload: future work
            }
            let spilled: Vec<usize> = (0..plen).collect(); // all params (loop-carried, live)
            let mut frame_offsets = Vec::with_capacity(plen);
            let mut off = 0u64;
            for &i in &spilled {
                off = align_up(off, vsize(slot_types[i]));
                frame_offsets.push(off);
                off += vsize(slot_types[i]);
            }
            let frame_size = align_up(off + 4, 16);
            let gid = points.len() as u32;
            points.push(PointPlan {
                kind: SuspendKind::LoopHeader,
                nres: 0,
                out: plen,
                save_end: plen,
                slot_types: slot_types.clone(),
                spilled,
                frame_offsets,
                rid_off: frame_size - 4,
                frame_size,
                cont_seg: seg(b, 0), // re-enter the header body, past the poll
            });
            let mut psb = Bb::new(slot_types);
            let (st_a, st_off) = psb.freeze_word_addr();
            let st = psb.one(load(LoadOp::I32, st_a, st_off));
            let unw = psb.one(Inst::ConstI32(STATE_UNWINDING));
            let is_unw = psb.one(icmp(IntTy::I32, CmpOp::Eq, st, unw));
            let live: Vec<ValIdx> = (0..plen as u32).collect();
            seg_blocks.push(psb.finish(Terminator::BrIf {
                cond: is_unw,
                then_blk: unwind_base + 2 * gid, // the point's UNWIND check block
                then_args: live.clone(),
                else_blk: seg(b, 0), // the header body
                else_args: live,
            }));
        }
        for k in 0..=m {
            let mut sb = Bb::new(bi.types[0..in_of(k)].to_vec());
            if k < m {
                // segment body up to & including the suspend op, then the poll
                let pos = bi.scs[k];
                let seg_start = if k == 0 { 0 } else { bi.scs[k - 1] + 1 };
                sb.insts.extend_from_slice(&blk.insts[seg_start..=pos]);
                let out = bi.vend[pos];
                sb.next = out as u32;
                let (st_a, st_off) = sb.freeze_word_addr();
                let st = sb.one(load(LoadOp::I32, st_a, st_off));
                let unw = sb.one(Inst::ConstI32(STATE_UNWINDING));
                let is_unw = sb.one(icmp(IntTy::I32, CmpOp::Eq, st, unw));
                let gid = points.len() as u32;
                let live: Vec<ValIdx> = (0..out as u32).collect();
                seg_blocks.push(sb.finish(Terminator::BrIf {
                    cond: is_unw,
                    then_blk: unwind_base + 2 * gid, // the point's UNWIND check block
                    then_args: live.clone(),
                    else_blk: seg(b, k + 1),
                    else_args: live,
                }));

                // resume plan for this point
                let kind = match &blk.insts[pos] {
                    // §13.4 slice 4b: a serve op re-issues (the sentinel it returned under
                    // `UNWINDING` must never reload as the served count) — before the
                    // generic `Leaf` arm.
                    Inst::CapCall {
                        type_id: svm_ir::CAP_SELF_TYPE_ID,
                        op: sop @ (SVC_POLL_OP | SVC_WAIT_OP),
                        sig,
                        handle,
                        args,
                    } => SuspendKind::SvcServe {
                        type_id: svm_ir::CAP_SELF_TYPE_ID,
                        op: *sop,
                        sig: sig.clone(),
                        handle: *handle,
                        args: args.clone(),
                    },
                    Inst::CapCall { .. } => SuspendKind::Leaf,
                    Inst::Call { func, args } => SuspendKind::Propagated {
                        callee: *func,
                        args: args.clone(),
                    },
                    Inst::ContResume { k, arg } => SuspendKind::Resume { k: *k, arg: *arg },
                    Inst::Suspend { value } => SuspendKind::Yield { value: *value },
                    Inst::ThreadJoin { handle } => SuspendKind::ThreadJoin { handle: *handle },
                    Inst::MemoryWait {
                        ty,
                        addr,
                        expected,
                        timeout,
                    } => SuspendKind::MemoryWait {
                        ty: *ty,
                        addr: *addr,
                        expected: *expected,
                        timeout: *timeout,
                    },
                    _ => unreachable!(
                        "suspend position is a cap.call / call / fiber / thread.join / atomic.wait op"
                    ),
                };
                let nres = match (&kind, &blk.insts[pos]) {
                    (SuspendKind::Leaf, Inst::CapCall { sig, .. }) => sig.results.len(),
                    (SuspendKind::SvcServe { .. }, Inst::CapCall { sig, .. }) => sig.results.len(),
                    (SuspendKind::Propagated { callee, .. }, _) => {
                        func_results[*callee as usize].len()
                    }
                    (SuspendKind::Resume { .. }, _) => 2, // (status, value)
                    (SuspendKind::Yield { .. }, _) => 1,  // the resume arg
                    (SuspendKind::ThreadJoin { .. }, _) => 1, // the join result (i64)
                    (SuspendKind::MemoryWait { .. }, _) => 1, // the wait status (i32)
                    _ => unreachable!(),
                };
                // Spillable range: values `[0, save_end)`. A leaf reloads its own result too
                // (`save_end == out`); a propagated frame re-issues its call, so the call's
                // results `[save_end, out)` are recomputed, not spilled.
                let save_end = match kind {
                    SuspendKind::Leaf => out,
                    // The op's results are recomputed (re-issue) or redelivered (resume), so
                    // they aren't spilled — same as a propagated call.
                    SuspendKind::Propagated { .. }
                    | SuspendKind::Resume { .. }
                    | SuspendKind::Yield { .. }
                    | SuspendKind::ThreadJoin { .. }
                    | SuspendKind::MemoryWait { .. }
                    | SuspendKind::SvcServe { .. } => out - nres,
                    // Header polls are built separately (above), never from an in-block op.
                    SuspendKind::LoopHeader => unreachable!("loop-header point not from an op"),
                };
                let slot_types = bi.types[0..out].to_vec();

                // Minimal live-set: spill only values used *after* the op (block-local SSA ⇒
                // a value's whole live range is in this block, so "live across" = referenced
                // by a later instruction or the terminator), plus a propagated call's own
                // operands (needed to re-issue it). An unrecognized instruction in the tail ⇒
                // fall back to spilling the whole spillable range (never under-spill).
                let mut used = vec![false; out];
                let mut conservative = false;
                for inst in &blk.insts[pos + 1..] {
                    match inst_operands(inst) {
                        Some(ops) => ops.iter().for_each(|&o| {
                            if (o as usize) < out {
                                used[o as usize] = true;
                            }
                        }),
                        None => {
                            conservative = true;
                            break;
                        }
                    }
                }
                for o in term_operands(&blk.term) {
                    if (o as usize) < out {
                        used[o as usize] = true;
                    }
                }
                if let SuspendKind::Propagated { args, .. } = &kind {
                    for &a in args {
                        used[a as usize] = true; // operands of the re-issued call
                    }
                }
                if let SuspendKind::Resume { k, arg } = &kind {
                    used[*k as usize] = true; // operands of the re-issued cont.resume
                    used[*arg as usize] = true;
                }
                if let SuspendKind::Yield { value } = &kind {
                    used[*value as usize] = true; // operand of the re-executed suspend
                }
                if let SuspendKind::ThreadJoin { handle } = &kind {
                    used[*handle as usize] = true; // operand of the re-issued thread.join
                }
                if let SuspendKind::MemoryWait {
                    addr,
                    expected,
                    timeout,
                    ..
                } = &kind
                {
                    used[*addr as usize] = true; // operands of the re-issued atomic.wait
                    used[*expected as usize] = true;
                    used[*timeout as usize] = true;
                }
                if let SuspendKind::SvcServe { handle, args, .. } = &kind {
                    used[*handle as usize] = true; // operands of the re-issued serve op
                    for &a in args {
                        used[a as usize] = true;
                    }
                }
                let spilled: Vec<usize> = if conservative {
                    (0..save_end).collect()
                } else {
                    (0..save_end).filter(|&i| used[i]).collect()
                };
                if spilled.iter().any(|&i| slot_types[i] == ValType::V128) {
                    return Err(TransformError::UnsupportedInst); // v128 spill/reload: future work
                }
                // Frame layout (DURABILITY.md §12.7): packed spilled values, resume id on top.
                let mut frame_offsets = Vec::with_capacity(spilled.len());
                let mut off = 0u64;
                for &i in &spilled {
                    off = align_up(off, vsize(slot_types[i]));
                    frame_offsets.push(off);
                    off += vsize(slot_types[i]);
                }
                let frame_size = align_up(off + 4, 16);
                points.push(PointPlan {
                    kind,
                    nres,
                    out,
                    save_end,
                    slot_types,
                    spilled,
                    frame_offsets,
                    rid_off: frame_size - 4,
                    frame_size,
                    cont_seg: seg(b, k + 1),
                });
            } else {
                // last segment: the tail after the final suspend op + the remapped terminator
                let seg_start = if m == 0 { 0 } else { bi.scs[m - 1] + 1 };
                sb.insts.extend_from_slice(&blk.insts[seg_start..]);
                seg_blocks.push(sb.finish(remap(&blk.term)));
            }
        }
    }

    // ---- DISPATCH — read the resume id at SP-4 and br_table to the matching arm ----
    // `sp_a` is the **active context's shadow-SP word address** from the runtime-private register
    // (`durable.shadow_base`, §12.8 4A.5) — per-context, so concurrent vCPUs each address their own SP
    // word with no shared location (vs. the former fixed global `SHADOW_SP_OFF`). The runtime seeds the
    // register; a guest cannot redirect it. Used identically at every SP site (dispatch/unwind/arm).
    let mut db = Bb::new(vec![]);
    let sp_a = db.one(Inst::DurableShadowBase);
    let sp = db.one(load(LoadOp::I64, sp_a, 0));
    let four = db.one(Inst::ConstI64(4));
    let sp_m4 = db.one(ibin(IntTy::I64, BinOp::Sub, sp, four));
    let rid = db.one(load(LoadOp::I32, sp_m4, 0));
    // id 0 is reserved ("no resume" ⇒ trap); id g+1 selects ARM_g.
    let mut targets: Vec<(BlockIdx, Vec<ValIdx>)> = vec![(trap_blk, vec![])];
    for g in 0..p_total {
        targets.push((arm_base + g, vec![]));
    }
    let dispatch = db.finish(Terminator::BrTable {
        idx: rid,
        targets,
        default: (trap_blk, vec![]),
    });

    // ---- UNWIND (check + spill pair) / ARM_g, per resume point ----
    let mut unwind_blocks: Vec<Block> = Vec::with_capacity(2 * total_points);
    let mut arm_blocks: Vec<Block> = Vec::with_capacity(total_points);
    for (gid, pt) in points.iter().enumerate() {
        // index in `pt.spilled` (and thus the reloaded vec) of a block-local value, if spilled
        let spill_slot = |i: usize| pt.spilled.binary_search(&i).ok();

        // UNWIND check: a push of this frame must not run past the reserve into guest memory
        // (R9 / DURABILITY.md §12.7). The shadow stack mirrors the call stack, so this only
        // trips for a chain deeper than `DURABLE_RESERVE` holds — a clean trap, never silent
        // corruption. It lives on the (cold) freeze path, not the per-call path.
        let mut cb = Bb::new(pt.slot_types.clone());
        let sp_a = cb.one(Inst::DurableShadowBase);
        let sp = cb.one(load(LoadOp::I64, sp_a, 0));
        let fsz = cb.one(Inst::ConstI64(pt.frame_size as i64));
        let newsp = cb.one(ibin(IntTy::I64, BinOp::Add, sp, fsz));
        let reserve = cb.one(Inst::ConstI64(DURABLE_RESERVE as i64));
        let over = cb.one(icmp(IntTy::I64, CmpOp::GtU, newsp, reserve));
        let live: Vec<ValIdx> = (0..pt.out as u32).collect();
        unwind_blocks.push(cb.finish(Terminator::BrIf {
            cond: over,
            then_blk: trap_blk,
            then_args: vec![],
            else_blk: unwind_base + 2 * gid as u32 + 1, // the spill block
            else_args: live,
        }));

        // UNWIND spill: spill the live (∪ call-arg) values + the resume id, commit the new SP.
        let mut ub = Bb::new(pt.slot_types.clone());
        let sp_a = ub.one(Inst::DurableShadowBase);
        let sp = ub.one(load(LoadOp::I64, sp_a, 0)); // this activation's frame base
        for (j, &i) in pt.spilled.iter().enumerate() {
            ub.zero(store(
                store_op(pt.slot_types[i]),
                sp,
                i as u32,
                pt.frame_offsets[j],
            ));
        }
        let rid = ub.one(Inst::ConstI32(gid as i32 + 1));
        ub.zero(store(StoreOp::I32, sp, rid, pt.rid_off));
        let fsz = ub.one(Inst::ConstI64(pt.frame_size as i64));
        let newsp = ub.one(ibin(IntTy::I64, BinOp::Add, sp, fsz));
        ub.zero(store(StoreOp::I64, sp_a, newsp, 0));
        let ret: Vec<ValIdx> = f.results.iter().map(|&t| ub.one(zero_const(t))).collect();
        unwind_blocks.push(ub.finish(Terminator::Return(ret)));

        // ARM: reload the spilled set (self-contained — no incoming params), pop, resume.
        let mut ab = Bb::new(vec![]);
        let sp_a = ab.one(Inst::DurableShadowBase);
        let sp = ab.one(load(LoadOp::I64, sp_a, 0));
        let fsz = ab.one(Inst::ConstI64(pt.frame_size as i64));
        let base = ab.one(ibin(IntTy::I64, BinOp::Sub, sp, fsz));
        let reloaded: Vec<ValIdx> = pt
            .spilled
            .iter()
            .enumerate()
            .map(|(j, &i)| ab.one(load(load_op(pt.slot_types[i]), base, pt.frame_offsets[j])))
            .collect();
        ab.zero(store(StoreOp::I64, sp_a, base, 0)); // pop: SP = frame base

        // For a propagated call, re-issue it (its operands were all spilled). For a leaf,
        // flip the state word back to NORMAL.
        let op_results: Vec<ValIdx> = match &pt.kind {
            // A leaf cap.call and a loop-header poll are both the globally-deepest frozen frame
            // with no op to re-issue: flip the state word back to `NORMAL`. The leaf then reloads
            // its cap.call result; the header reloads its block params and re-enters the body
            // (`cont_seg`). Neither produces an `op_results` value.
            SuspendKind::Leaf | SuspendKind::LoopHeader => {
                let (st_a, st_off) = ab.thaw_word_addr();
                let normal_v = ab.one(Inst::ConstI32(STATE_NORMAL));
                ab.zero(store(StoreOp::I32, st_a, normal_v, st_off));
                vec![]
            }
            SuspendKind::Propagated { callee, args } => {
                let mapped: Vec<ValIdx> = args
                    .iter()
                    .map(|&a| reloaded[spill_slot(a as usize).expect("call arg spilled")])
                    .collect();
                ab.many(
                    Inst::Call {
                        func: *callee,
                        args: mapped,
                    },
                    pt.nres,
                )
            }
            // `cont.resume` re-issue (slice 3.1.2): reload the (spilled) handle + arg and resume
            // the fiber again. On thaw the fiber rewinds in turn (its `Yield` re-park) and
            // redelivers `(status, value)` — the resumer threads those two results into its
            // continuation just like a propagated call. The resumer does **not** flip the state
            // word: the resumee's `Yield` arm (the globally-deepest frame) does.
            SuspendKind::Resume { k, arg } => {
                let kk = reloaded[spill_slot(*k as usize).expect("resume handle spilled")];
                let aa = reloaded[spill_slot(*arg as usize).expect("resume arg spilled")];
                ab.many(Inst::ContResume { k: kk, arg: aa }, pt.nres)
            }
            // `suspend` re-park (slice 3.1.3): a parked fiber's suspend is the globally-deepest
            // frozen frame, so flip the state word to NORMAL, then re-execute `suspend` — which
            // parks this fiber and hands `value` back to the resumer (in NORMAL). Its result, the
            // value the *next* resume delivers, threads into the continuation exactly as a leaf's
            // reloaded cap.call result does.
            SuspendKind::Yield { value } => {
                let (st_a, st_off) = ab.thaw_word_addr();
                let normal_v = ab.one(Inst::ConstI32(STATE_NORMAL));
                ab.zero(store(StoreOp::I32, st_a, normal_v, st_off));
                let v = reloaded[spill_slot(*value as usize).expect("suspend value spilled")];
                ab.many(Inst::Suspend { value: v }, pt.nres)
            }
            // `thread.join` re-issue: reload the spilled vCPU handle and re-execute the join. By thaw the
            // child has been re-spawned and run to completion, so the join resolves to its recorded result
            // immediately (no block) — its result is *re-issued* (not reloaded), like `cont.resume`, so
            // §12.6 holds (the child's side effects are replayed on its own rewind). But unlike a
            // propagated call / resume, the join has **no in-thread callee** to flip the state word: the
            // joined child rewinds as a *separate* vCPU (and the thaw driver resets the word to
            // `REWINDING` afterward), so on this thread the join is the globally-deepest frozen frame —
            // it flips the state to `NORMAL` itself, like a leaf, *before* re-issuing.
            SuspendKind::ThreadJoin { handle } => {
                let (st_a, st_off) = ab.thaw_word_addr();
                let normal_v = ab.one(Inst::ConstI32(STATE_NORMAL));
                ab.zero(store(StoreOp::I32, st_a, normal_v, st_off));
                let hh =
                    reloaded[spill_slot(*handle as usize).expect("thread.join handle spilled")];
                ab.many(Inst::ThreadJoin { handle: hh }, pt.nres)
            }
            // `atomic.wait` re-issue: like `thread.join`, the wait is the globally-deepest frozen frame
            // on this thread (the notifier is a *separate* vCPU), so flip the state word to `NORMAL`
            // itself, then reload the spilled `addr`/`expected`/`timeout` and re-execute the wait. The
            // re-issued wait re-checks the value: a wake that landed as a value change resolves it with
            // `WAIT_NOT_EQUAL` (no block); a would-park fails closed in the thunk (`ThreadFault`).
            SuspendKind::MemoryWait {
                ty,
                addr,
                expected,
                timeout,
            } => {
                let (st_a, st_off) = ab.thaw_word_addr();
                let normal_v = ab.one(Inst::ConstI32(STATE_NORMAL));
                ab.zero(store(StoreOp::I32, st_a, normal_v, st_off));
                let aa = reloaded[spill_slot(*addr as usize).expect("atomic.wait addr spilled")];
                let ee =
                    reloaded[spill_slot(*expected as usize).expect("atomic.wait expected spilled")];
                let tt =
                    reloaded[spill_slot(*timeout as usize).expect("atomic.wait timeout spilled")];
                ab.many(
                    Inst::MemoryWait {
                        ty: *ty,
                        addr: aa,
                        expected: ee,
                        timeout: tt,
                    },
                    pt.nres,
                )
            }
            // Serve-op re-issue (§13.4 slice 4b): the mid-handler gate guarantees this point is
            // the globally-deepest frozen frame on its thread (no handler was in flight), so —
            // like `atomic.wait` — flip the state word to `NORMAL` itself, then reload the
            // handle + args and re-execute. The re-executed drain runs against the restored
            // queue (the snapshot's serve section); an empty queue re-parks `svc.wait` exactly
            // as an uninterrupted run would.
            SuspendKind::SvcServe {
                type_id,
                op,
                sig,
                handle,
                args,
            } => {
                let (st_a, st_off) = ab.thaw_word_addr();
                let normal_v = ab.one(Inst::ConstI32(STATE_NORMAL));
                ab.zero(store(StoreOp::I32, st_a, normal_v, st_off));
                let hh = reloaded[spill_slot(*handle as usize).expect("serve-op handle spilled")];
                let aa: Vec<ValIdx> = args
                    .iter()
                    .map(|&a| reloaded[spill_slot(a as usize).expect("serve-op arg spilled")])
                    .collect();
                ab.many(
                    Inst::CapCall {
                        type_id: *type_id,
                        op: *op,
                        sig: sig.clone(),
                        handle: hh,
                        args: aa,
                    },
                    pt.nres,
                )
            }
        };

        // Assemble the continuation's `out` args slot-by-slot: a reloaded value, a re-issued
        // call result (`[save_end, out)`), or a zero placeholder for a dead-but-present slot.
        let cont_args: Vec<ValIdx> = (0..pt.out)
            .map(|i| {
                if let Some(j) = spill_slot(i) {
                    reloaded[j]
                } else if i >= pt.save_end {
                    op_results[i - pt.save_end]
                } else {
                    ab.one(zero_const(pt.slot_types[i])) // dead across the op; value unused
                }
            })
            .collect();
        arm_blocks.push(ab.finish(Terminator::Br {
            target: pt.cont_seg,
            args: cont_args,
        }));
    }

    // ---- TRAP — br_table default / forged resume id ----
    let trap = Block {
        params: vec![],
        insts: vec![],
        term: Terminator::Unreachable,
    };

    // Assemble in the order the index layout assumes.
    let mut blocks = Vec::with_capacity((2 + s_total + 2 * p_total + 1) as usize);
    blocks.push(prologue);
    blocks.extend(seg_blocks);
    blocks.push(dispatch);
    blocks.extend(unwind_blocks);
    blocks.extend(arm_blocks);
    blocks.push(trap);

    let max_frame = points.iter().map(|pt| pt.frame_size).max().unwrap_or(0);
    let func = Func {
        params: f.params.clone(),
        results: f.results.clone(),
        blocks,
    };
    Ok((func, max_frame))
}

// ---- window helpers for freeze/thaw drivers and tests ----

/// A fresh durable window of `size` bytes: state = `NORMAL`, and the root context's per-context
/// shadow-SP word (§12.8 4A.5) — the first 8 bytes of its region at `SHADOW_BASE` — set to its empty
/// frame base (`SHADOW_BASE + REGION_HEADER_LEN`, just past the SP + thaw words). The legacy global
/// `SHADOW_SP_OFF` is unused; the per-context thaw words default to `NORMAL` (zero).
pub fn init_durable_window(size: usize) -> Vec<u8> {
    let mut w = vec![0u8; size];
    write_state(&mut w, STATE_NORMAL);
    w[SHADOW_BASE as usize..SHADOW_BASE as usize + 8]
        .copy_from_slice(&(SHADOW_BASE + REGION_HEADER_LEN).to_le_bytes());
    w
}

/// Window byte offset of context `ctx`'s **thaw** state word (§12.8 concurrent-thaw stage 1) — its
/// region base plus [`STATE_IN_REGION_OFF`]. Per-context, so a thaw can set each frozen vCPU rewinding
/// independently (vs. the global [`STATE_OFF`] freeze word).
pub fn thaw_state_off(ctx: usize) -> u64 {
    SHADOW_BASE + ctx as u64 * SHADOW_STRIDE + STATE_IN_REGION_OFF
}

/// Overwrite the global **freeze** state word (`UNWINDING`/`ARMED`/`NORMAL`) in a window image — the
/// stop-the-world trigger every poll reads. Thaw (`REWINDING`) goes through [`write_thaw_state`].
pub fn write_state(window: &mut [u8], state: i32) {
    window[STATE_OFF as usize..STATE_OFF as usize + 4].copy_from_slice(&state.to_le_bytes());
}

/// Overwrite context `ctx`'s per-context **thaw** state word (`REWINDING`/`NORMAL`) — used to drive a
/// thaw (the runtime sets each frozen context `REWINDING` before its rewinding re-entry).
pub fn write_thaw_state(window: &mut [u8], ctx: usize, state: i32) {
    let off = thaw_state_off(ctx) as usize;
    window[off..off + 4].copy_from_slice(&state.to_le_bytes());
}

/// Set up a window for a **thaw** of context `ctx` (§12.8 concurrent-thaw stage 1): clear the global
/// **freeze** word back to `NORMAL` (the frozen artifact left it `UNWINDING`, but a thaw is not a
/// freeze — leaving it would make the rewinding code's polls re-unwind) and set `ctx`'s per-context
/// **thaw** word to `REWINDING`. Mirrors what the runtime does on a real snapshot-restore thaw (the
/// interp's `drive` clear + per-context `REWINDING`; the JIT thaw driver).
pub fn begin_thaw(window: &mut [u8], ctx: usize) {
    write_state(window, STATE_NORMAL);
    write_thaw_state(window, ctx, STATE_REWINDING);
}

/// Read context `ctx`'s per-context **thaw** state word — after a thaw, a completed rewind reads
/// `NORMAL` (the deepest frame's re-issue flipped it).
pub fn read_thaw_state(window: &[u8], ctx: usize) -> i32 {
    let off = thaw_state_off(ctx) as usize;
    let mut b = [0u8; 4];
    b.copy_from_slice(&window[off..off + 4]);
    i32::from_le_bytes(b)
}

/// Arm a window to **freeze after `safepoints` further fiber safepoints** (the deterministic mid-run
/// trigger): the run proceeds normally, and the runtime promotes the state word to `UNWINDING` at the
/// `safepoints`-th fiber safepoint (`cont.resume`/`suspend`) so that op's trailing poll begins the
/// freeze. `safepoints == 1` freezes at the first fiber safepoint; larger values let the run make
/// forward progress first (e.g. past a fiber recycle). A non-positive count is clamped to 1.
pub fn arm_freeze_after(window: &mut [u8], safepoints: i64) {
    let n = safepoints.max(1);
    window[ARM_COUNTDOWN_OFF as usize..ARM_COUNTDOWN_OFF as usize + 8]
        .copy_from_slice(&n.to_le_bytes());
    write_state(window, STATE_ARMED);
}

/// Arm a window to **freeze after `backedges` further loop back-edges** (the deterministic Phase-4
/// Slice A trigger for back-edge polls): the run proceeds normally, and the runtime promotes the
/// state word to `UNWINDING` at the `backedges`-th branch terminator so the next loop-header poll
/// begins the freeze — reaching a poll-free compute loop that no fiber-safepoint countdown can. Sets
/// the back-edge countdown ([`ARM_BACKEDGE_OFF`]); the fiber-safepoint countdown stays 0 so the two
/// triggers never interfere. A non-positive count is clamped to 1.
pub fn arm_freeze_after_backedges(window: &mut [u8], backedges: i64) {
    let n = backedges.max(1);
    window[ARM_BACKEDGE_OFF as usize..ARM_BACKEDGE_OFF as usize + 8]
        .copy_from_slice(&n.to_le_bytes());
    write_state(window, STATE_ARMED);
}

/// Read the state word from a window image.
pub fn read_state(window: &[u8]) -> i32 {
    let mut b = [0u8; 4];
    b.copy_from_slice(&window[STATE_OFF as usize..STATE_OFF as usize + 4]);
    i32::from_le_bytes(b)
}

// ---- small IR construction helpers ----

/// A block under construction that tracks the next block-local value index.
struct Bb {
    params: Vec<ValType>,
    insts: Vec<Inst>,
    next: u32,
}

impl Bb {
    fn new(params: Vec<ValType>) -> Self {
        let next = params.len() as u32;
        Bb {
            params,
            insts: Vec::new(),
            next,
        }
    }
    /// Push a single-result instruction; returns its value index.
    fn one(&mut self, i: Inst) -> ValIdx {
        let idx = self.next;
        self.insts.push(i);
        self.next += 1;
        idx
    }
    /// Push an instruction that defines `nres` consecutive values; returns their indices.
    fn many(&mut self, i: Inst, nres: usize) -> Vec<ValIdx> {
        let start = self.next;
        self.insts.push(i);
        self.next += nres as u32;
        (start..self.next).collect()
    }
    /// Push a zero-result instruction (a store).
    fn zero(&mut self, i: Inst) {
        self.insts.push(i);
    }
    /// §12.8 concurrent-thaw stage 1: the **freeze** state word's address (`UNWINDING`), as
    /// `(base, offset)` for a `load`. **Always global** ([`STATE_OFF`]) — a freeze is genuinely
    /// stop-the-world, so the single word is the natural broadcast every context's poll reads (the arm
    /// trigger / `request_freeze` set it). Read by the loop-header and in-block `UNWINDING` polls.
    fn freeze_word_addr(&mut self) -> (ValIdx, u64) {
        (self.one(Inst::ConstI64(STATE_OFF as i64)), 0)
    }
    /// §12.8 concurrent-thaw stage 1: the **thaw** state word's address (`REWINDING`/`NORMAL`), as
    /// `(base, offset)` for a `load`/`store` — the running context's own region word
    /// (`durable.shadow_base` + [`STATE_IN_REGION_OFF`], like the per-context shadow-SP word), so
    /// concurrent vCPUs each rewind against their own (the relocation's whole point). Read by the
    /// prologue's `REWINDING` dispatch and written `NORMAL` by the deepest frame's re-issue (thaw end).
    fn thaw_word_addr(&mut self) -> (ValIdx, u64) {
        (self.one(Inst::DurableShadowBase), STATE_IN_REGION_OFF)
    }
    fn finish(self, term: Terminator) -> Block {
        Block {
            params: self.params,
            insts: self.insts,
            term,
        }
    }
}

fn load(op: LoadOp, addr: ValIdx, offset: u64) -> Inst {
    Inst::Load {
        op,
        addr,
        offset,
        align: 0,
    }
}

fn store(op: StoreOp, addr: ValIdx, value: ValIdx, offset: u64) -> Inst {
    Inst::Store {
        op,
        addr,
        value,
        offset,
        align: 0,
    }
}

fn ibin(ty: IntTy, op: BinOp, a: ValIdx, b: ValIdx) -> Inst {
    Inst::IntBin { ty, op, a, b }
}

fn icmp(ty: IntTy, op: CmpOp, a: ValIdx, b: ValIdx) -> Inst {
    Inst::IntCmp { ty, op, a, b }
}

fn zero_const(t: ValType) -> Inst {
    match t {
        ValType::I32 => Inst::ConstI32(0),
        ValType::I64 => Inst::ConstI64(0),
        ValType::F32 => Inst::ConstF32(0),
        ValType::F64 => Inst::ConstF64(0),
        ValType::V128 => Inst::ConstV128([0; 16]),
        // An opaque `ref` is i64-width (GC.md §6 reservation); its zero is the i64 zero word.
        ValType::Ref => Inst::ConstI64(0),
        // §3.5 `cap` is i32-width handle data.
        ValType::Cap => Inst::ConstI32(0),
    }
}

fn vsize(t: ValType) -> u64 {
    match t {
        ValType::I32 | ValType::F32 | ValType::Cap => 4,
        ValType::I64 | ValType::F64 | ValType::Ref => 8,
        ValType::V128 => 16,
    }
}

fn align_up(x: u64, a: u64) -> u64 {
    (x + a - 1) & !(a - 1)
}

fn store_op(t: ValType) -> StoreOp {
    match t {
        ValType::I32 | ValType::Cap => StoreOp::I32,
        ValType::I64 | ValType::Ref => StoreOp::I64, // `ref` spills as its opaque i64 word
        ValType::F32 => StoreOp::F32,
        ValType::F64 => StoreOp::F64,
        ValType::V128 => unreachable!("v128 spill rejected earlier"),
    }
}

fn load_op(t: ValType) -> LoadOp {
    match t {
        ValType::I32 | ValType::Cap => LoadOp::I32,
        ValType::I64 | ValType::Ref => LoadOp::I64, // `ref` reloads as its opaque i64 word
        ValType::F32 => LoadOp::F32,
        ValType::F64 => LoadOp::F64,
        ValType::V128 => unreachable!("v128 reload rejected earlier"),
    }
}

/// Result types of an instruction, given the types of all earlier values in the block
/// and each function's result types. Covers the scalar/memory/call subset a Phase-1
/// prefix can use; returns `UnsupportedInst` for anything else (SIMD, conversions,
/// concurrency ops), so the transform fails closed rather than mis-typing a frame.
fn result_types(
    inst: &Inst,
    types: &[ValType],
    func_results: &[Vec<ValType>],
) -> Result<Vec<ValType>, TransformError> {
    use Inst::*;
    Ok(match inst {
        ConstI32(_) => vec![ValType::I32],
        ConstI64(_) => vec![ValType::I64],
        ConstF32(_) => vec![ValType::F32],
        ConstF64(_) => vec![ValType::F64],
        ConstV128(_) => vec![ValType::V128],
        IntBin { ty, .. } | IntUn { ty, .. } => vec![ty.val()],
        FBin { ty, .. } | FUn { ty, .. } => vec![ty.val()],
        IntCmp { .. } | FCmp { .. } | Eqz { .. } => vec![ValType::I32],
        AtomicLoad { ty, .. } | AtomicRmw { ty, .. } | AtomicCmpxchg { ty, .. } => vec![ty.val()],
        Store { .. } | AtomicStore { .. } | AtomicFence { .. } => vec![],
        Select { a, .. } => vec![types[*a as usize]],
        Load { op, .. } => vec![load_result_ty(*op)],
        Call { func, .. } => func_results
            .get(*func as usize)
            .cloned()
            .ok_or(TransformError::UnsupportedShape)?,
        CapCall { sig, .. } => sig.results.clone(),
        CallIndirect { ty, .. } => ty.results.clone(),
        PtrAdd { .. } | PtrCast { .. } => vec![ValType::I64],
        RefFunc { .. } => vec![ValType::I32],
        // Fiber control ops (§12 / Phase 3): an i64 handle, a `(status, value)` pair, a resume arg.
        ContNew { .. } => vec![ValType::I64],
        ContResume { .. } => vec![ValType::I32, ValType::I64],
        Suspend { .. } => vec![ValType::I64],
        // §12 thread ops (Phase 3.2): `thread.spawn` yields an `i32` handle, `thread.join` an `i64`
        // result. Neither is a may-suspend checkpoint — they are copied verbatim into their segment and
        // their results spill/reload like any scalar; the multi-vCPU freeze/thaw choreography is the
        // runtime's (durable §12.8 slice 3.2.1), so the transform only needs to type them.
        ThreadSpawn { .. } => vec![ValType::I32],
        ThreadJoin { .. } => vec![ValType::I64],
        // §12 futex ops: `atomic.wait` yields an `i32` status (woken / not-equal / timed-out),
        // `atomic.notify` an `i32` woken count. `atomic.wait` is a may-suspend re-issue safepoint
        // (the parked-vCPU slice); `atomic.notify` is copied verbatim into its segment.
        MemoryWait { .. } | MemoryNotify { .. } => vec![ValType::I32],
        _ => return Err(TransformError::UnsupportedInst),
    })
}

fn load_result_ty(op: LoadOp) -> ValType {
    use LoadOp::*;
    match op {
        I32 | I32_8S | I32_8U | I32_16S | I32_16U => ValType::I32,
        I64 | I64_8S | I64_8U | I64_16S | I64_16U | I64_32S | I64_32U => ValType::I64,
        F32 => ValType::F32,
        F64 => ValType::F64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use svm_ir::Memory;

    fn parse_with_mem(src: &str, size_log2: u8) -> Module {
        let mut m = svm_text::parse_module(src).expect("parse");
        m.memory = Some(Memory { size_log2 });
        m
    }

    #[test]
    fn no_cap_call_is_left_unchanged() {
        let m = parse_with_mem(
            "func (i32) -> (i32) {\nblock 0 (v0: i32) {\n  return v0\n  }\n}\n",
            12,
        );
        let out = transform_module(&m).expect("transform");
        assert_eq!(out.funcs, m.funcs, "function without cap.call is untouched");
    }

    #[test]
    fn instrumented_function_verifies() {
        let m = parse_with_mem(
            "func (i32) -> (i64) {\nblock 0 (v0: i32) {\n  v1 = i32.const 0\n  v2 = cap.call 2 0 (i32) -> (i64) v0 (v1)\n  v3 = i64.const 100\n  v4 = i64.add v2 v3\n  return v4\n  }\n}\n",
            18,
        );
        let out = transform_module(&m).expect("transform");
        svm_verify::verify_module(&out).expect("instrumented IR must verify");
        assert_eq!(
            out.funcs[0].blocks.len(),
            8,
            "one point: 4n+4 blocks with n=1"
        );
    }

    #[test]
    fn two_cap_calls_become_two_resume_points() {
        // Two suspend points in one block ⇒ two br_table arms ⇒ 3·2 + 4 = 10 blocks.
        let m = parse_with_mem(
            "func (i32) -> (i64) {\nblock 0 (v0: i32) {\n  v1 = i32.const 0\n  v2 = cap.call 2 0 (i32) -> (i64) v0 (v1)\n  v3 = cap.call 2 0 (i32) -> (i64) v0 (v1)\n  v4 = i64.add v2 v3\n  return v4\n  }\n}\n",
            18,
        );
        let out = transform_module(&m).expect("two resume points are in scope");
        svm_verify::verify_module(&out).expect("instrumented IR must verify");
        assert_eq!(
            out.funcs[0].blocks.len(),
            12,
            "two-point layout: 4n+4 with n=2"
        );
    }

    #[test]
    fn propagated_chain_instruments_each_frame() {
        // A two-level chain: the caller suspends on its `call` to the leaf, the leaf on
        // its `cap.call`. Both are may-suspend, so both get the 7-block instrumentation.
        let m = parse_with_mem(
            "func (i32) -> (i64) {\nblock 0 (v0: i32) {\n  v1 = call 1 (v0)\n  return v1\n  }\n}\nfunc (i32) -> (i64) {\nblock 0 (v0: i32) {\n  v1 = i32.const 0\n  v2 = cap.call 2 0 (i32) -> (i64) v0 (v1)\n  return v2\n  }\n}\n",
            18,
        );
        let out = transform_module(&m).expect("transform");
        svm_verify::verify_module(&out).expect("instrumented chain must verify");
        assert_eq!(
            out.funcs[0].blocks.len(),
            8,
            "caller (propagated) instrumented"
        );
        assert_eq!(out.funcs[1].blocks.len(), 8, "callee (leaf) instrumented");
    }

    #[test]
    fn non_suspending_callee_is_left_unchanged() {
        // func 0 (leaf cap.call) calls func 1 (a pure helper) as a *prefix* op. The helper
        // never suspends, so it is not instrumented and func 0's only suspend point is its
        // own cap.call; the helper's result is spilled/reloaded, never re-issued.
        let m = parse_with_mem(
            "func (i32) -> (i64) {\nblock 0 (v0: i32) {\n  v1 = call 1 (v0)\n  v2 = i32.const 0\n  v3 = cap.call 2 0 (i32) -> (i64) v0 (v2)\n  v4 = i64.add v1 v3\n  return v4\n  }\n}\nfunc (i32) -> (i64) {\nblock 0 (v0: i32) {\n  v1 = i64.const 5\n  return v1\n  }\n}\n",
            18,
        );
        let helper_before = m.funcs[1].clone();
        let out = transform_module(&m).expect("transform");
        svm_verify::verify_module(&out).expect("verify");
        assert_eq!(out.funcs[0].blocks.len(), 8, "leaf instrumented");
        assert_eq!(
            out.funcs[1], helper_before,
            "non-suspending helper untouched"
        );
    }

    #[test]
    fn instrumented_module_with_guest_memory_op_is_rejected() {
        // A guest store could alias the durable region at `[0, SHADOW_BASE)` → R9 fails closed.
        let m = parse_with_mem(
            "func (i32) -> (i64) {\nblock 0 (v0: i32) {\n  v1 = i32.const 0\n  v2 = cap.call 2 0 (i32) -> (i64) v0 (v1)\n  v3 = i64.const 7\n  i64.store v1 v3\n  return v2\n  }\n}\n",
            18,
        );
        assert_eq!(transform_module(&m), Err(TransformError::GuestUsesMemory));
    }

    #[test]
    fn guest_memory_op_in_uninstrumented_module_is_fine() {
        // No `cap.call` anywhere ⇒ nothing is instrumented ⇒ no durable region ⇒ the
        // guest's own memory use is left untouched.
        let m = parse_with_mem(
            "func (i32) -> (i64) {\nblock 0 (v0: i32) {\n  v1 = i64.const 7\n  i64.store v0 v1\n  v2 = i64.load v0\n  return v2\n  }\n}\n",
            18,
        );
        let out = transform_module(&m).expect("no instrumentation, memory use is fine");
        assert_eq!(out.funcs, m.funcs, "left unchanged");
    }

    #[test]
    fn cap_call_without_memory_is_rejected() {
        let mut m = svm_text::parse_module(
            "func (i32) -> (i64) {\nblock 0 (v0: i32) {\n  v1 = i32.const 0\n  v2 = cap.call 2 0 (i32) -> (i64) v0 (v1)\n  return v2\n  }\n}\n",
        )
        .unwrap();
        m.memory = None;
        assert_eq!(transform_module(&m), Err(TransformError::NoMemory));
    }
}
