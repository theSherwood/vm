//! `svm-durable` — the IR→IR freeze/thaw transform (DESIGN.md / DURABILITY.md D60).
//!
//! A **tooling-tier, non-TCB** crate (like `svm-text`): it depends only on `svm-ir`
//! and emits ordinary, verifier-passing IR — no new instructions, no escape-TCB
//! surface. An embedder running pre-instrumented modules links none of it.
//!
//! This is the **Phase 1** slice of the plan (DURABILITY.md §9): it instruments a
//! function so a single in-flight `cap.call` can be *unwound* into guest-resident
//! shadow state and later *rewound* back into execution, byte-for-byte. The codec is
//! exactly the §2 mechanism:
//!
//! * a **state word** (`NORMAL | UNWINDING | REWINDING`) in the window,
//! * a per-fiber **shadow stack** in the window (DURABILITY.md §12.7),
//! * **unwind** = after a may-suspend call, if `UNWINDING`, spill the live values +
//!   resume id and return out to the host;
//! * **rewind** = in the prologue, if `REWINDING`, `br_table` on the saved resume id,
//!   reload the live values, and continue from the resume point.
//!
//! # Scope (single-block, multi-suspend — the go/no-go gate, now with chains)
//!
//! The transform handles a **single-block** function with a `return` terminator and any
//! number of may-suspend operations (≥ 1), each either:
//!
//! * a **leaf** `cap.call` — the host performs the operation; on thaw the deepest frame
//!   reloads the saved result and flips the state word back to `NORMAL`; or
//! * a **propagated** `Call` to a may-suspend callee (a function that transitively
//!   reaches a `cap.call`) — frames stack up across the call chain. On thaw a non-deepest
//!   frame reloads its pre-call live set and **re-issues the call** (leaving the state
//!   `REWINDING` so the callee rewinds in turn); only the innermost leaf flips to
//!   `NORMAL`. This is the DURABILITY.md §12.7 "re-issue vs. continue" branch (R8).
//!
//! The block is split at each suspend op into forward segments, and a `br_table` in the
//! prologue dispatch routes a thaw to the resume point that was in flight (one arm per
//! point). A function is **may-suspend** iff it contains a `cap.call` or (transitively) a
//! `Call` to a may-suspend function; only may-suspend functions are instrumented.
//!
//! This is enough to prove the core risk end-to-end on the real interpreter
//! (`tests/roundtrip.rs`, `tests/chain.rs`, `tests/multipoint.rs`): does an in-window
//! shadow stack + `br_table` rewind round-trip a frozen *call chain* with multiple resume
//! points? The remaining extensions (DURABILITY.md §9):
//!
//! * multi-block CFGs (split any block at its may-suspend calls — today a single block);
//! * fibers / multi-vCPU / STW (Phase 3).
//!
//! A `call_indirect` whose runtime target may suspend is **out of scope**: the closure
//! cannot resolve the target, so such a call is treated as non-suspending. Durable
//! modules must reach suspension through direct calls (the generator only emits those).
//!
//! Two Phase-1 simplifications are called out where they occur: the durable runtime
//! region is carved at **low** window addresses (real placement is per-fiber and
//! guard-paged, §12.7), and the captured **live set is over-approximated** to "all
//! values defined before the call" (correct, just spills more than the minimal live
//! set — the §12.7 optimization).

#![forbid(unsafe_code)]

use svm_ir::{
    BinOp, Block, BlockIdx, CmpOp, Func, FuncIdx, Inst, IntTy, LoadOp, Module, StoreOp,
    Terminator, ValIdx, ValType,
};

// `FuncIdx` is used by `SuspendKind::Propagated` below.

// ---- State word values (the §2 state machine) ----

/// Normal forward execution; polls and prologues fall straight through.
pub const STATE_NORMAL: i32 = 0;
/// Freeze in progress: every poll after a may-suspend call unwinds out to the host.
pub const STATE_UNWINDING: i32 = 1;
/// Thaw in progress: every prologue rebuilds its frame from the shadow stack.
pub const STATE_REWINDING: i32 = 2;

// ---- Durable runtime region layout (Phase 1: low window addresses) ----
//
// Real placement is per-fiber, guard-paged, quota-charged (DURABILITY.md §12.7). For
// Phase 1 (single vCPU, no fibers) we carve a fixed low region and require instrumented
// modules not to use `[0, SHADOW_BASE)` themselves.

/// Window byte offset of the `i32` state word.
pub const STATE_OFF: u64 = 0;
/// Window byte offset of the `i64` shadow-stack pointer (a window byte offset itself).
pub const SHADOW_SP_OFF: u64 = 8;
/// Window byte offset where the shadow stack begins (grows upward).
pub const SHADOW_BASE: u64 = 64;

// Block layout of an instrumented function with `S` forward segments (each original
// block is split at its suspend ops into `points+1` segments; non-suspend blocks are one
// segment) and `P` resume points total (reduces to the 7-block shape for one block / one
// point):
//   0                  PROLOGUE  — dispatch on the state word, then enter segment 0 of blk 0
//   1 ..= S            forward segments (each original block's segments, in block order)
//   1+S                DISPATCH  — read resume id, br_table to an arm
//   2+S ..= 1+S+P      UNWIND_g  — spill + return placeholder, per resume point
//   2+S+P ..= 1+S+2P   ARM_g     — reload + resume, per resume point
//   2+S+2P             TRAP      — forged / reserved resume id

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
}

/// Instrument every may-suspend function in `m` for freeze/thaw. Functions that can
/// never suspend are returned unchanged. The result is ordinary IR; run it through
/// `svm_verify::verify_module` before executing.
pub fn transform_module(m: &Module) -> Result<Module, TransformError> {
    let func_results: Vec<Vec<ValType>> = m.funcs.iter().map(|f| f.results.clone()).collect();
    let may_suspend = compute_may_suspend(m);
    let mut out = m.clone();
    let mut max_frame = 0u64;
    let mut any_instrumented = false;

    for (i, f) in m.funcs.iter().enumerate() {
        if may_suspend[i] {
            let (nf, frame_size) = transform_func(f, &func_results, &may_suspend)?;
            out.funcs[i] = nf;
            max_frame = max_frame.max(frame_size);
            any_instrumented = true;
        }
    }

    if any_instrumented {
        let mem = out.memory.ok_or(TransformError::NoMemory)?;
        // The check guarantees room for the durable region + the *largest single* frame.
        // A live call chain stacks one frame per suspended activation, so the window must
        // be sized for the deepest expected chain (a true guard-paged, quota-charged
        // shadow stack with overflow trapping is DURABILITY.md §12.7 / R9 future work).
        if mem.size() < SHADOW_BASE + max_frame {
            return Err(TransformError::MemoryTooSmall);
        }
    }
    Ok(out)
}

/// Mark each function that can suspend: it contains a `cap.call`, or (transitively) a
/// direct `Call` to a may-suspend function. A least-fixed-point over the direct-call
/// graph. `call_indirect` targets are unresolved and treated as non-suspending (see the
/// module-level scope note).
fn compute_may_suspend(m: &Module) -> Vec<bool> {
    let mut ms = vec![false; m.funcs.len()];
    for (i, f) in m.funcs.iter().enumerate() {
        if f.blocks.iter().any(|b| {
            b.insts.iter().any(|x| matches!(x, Inst::CapCall { .. }))
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
}

/// One resume point's metadata across the whole function (global id by vector order).
struct PointPlan {
    kind: SuspendKind,        // leaf cap.call or propagated call
    nres: usize,              // result count of the suspend op
    live_types: Vec<ValType>, // values visible at the poll (UNWIND params; arg count = len)
    saved_len: usize,         // first `saved_len` of `live_types` are spilled
    frame_offsets: Vec<u64>,  // window offset of each saved value
    frame_size: u64,
    rid_off: u64,
    cont_seg: u32,            // new block index of the continuation segment (after the op)
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
                Inst::CapCall { .. } => true,
                Inst::Call { func, .. } => may_suspend[*func as usize],
                _ => false,
            })
            .map(|(pos, _)| pos)
            .collect();
        binfo.push(BlockInfo { types, vend, scs, plen: blk.params.len() });
    }

    let total_points: usize = binfo.iter().map(|bi| bi.scs.len()).sum();
    if total_points == 0 {
        return Err(TransformError::UnsupportedShape); // may-suspend, but no in-block op
    }

    // Block-index layout (see the map near the constants).
    let mut seg_base = Vec::with_capacity(nb);
    let mut acc = 1u32; // segment indices start right after the PROLOGUE
    for bi in &binfo {
        seg_base.push(acc);
        acc += bi.scs.len() as u32 + 1; // points + 1 segments
    }
    let s_total = acc - 1;
    let seg = |b: usize, k: usize| seg_base[b] + k as u32;
    let p_total = total_points as u32;
    let dispatch_blk = 1 + s_total;
    let unwind_base = 2 + s_total;
    let arm_base = 2 + s_total + p_total;
    let trap_blk = 2 + s_total + 2 * p_total;
    let p = f.params.len();

    // Remap a terminator's block targets to segment 0 of each target block.
    let seg0 = |t: BlockIdx| seg_base[t as usize];
    let remap = |term: &Terminator| -> Terminator {
        match term {
            Terminator::Br { target, args } => Terminator::Br { target: seg0(*target), args: args.clone() },
            Terminator::BrIf { cond, then_blk, then_args, else_blk, else_args } => Terminator::BrIf {
                cond: *cond,
                then_blk: seg0(*then_blk),
                then_args: then_args.clone(),
                else_blk: seg0(*else_blk),
                else_args: else_args.clone(),
            },
            Terminator::BrTable { idx, targets, default } => Terminator::BrTable {
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
    let st_a = pb.one(Inst::ConstI64(STATE_OFF as i64));
    let st = pb.one(load(LoadOp::I32, st_a, 0));
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
        let in_of = |k: usize| if k == 0 { bi.plen } else { bi.vend[bi.scs[k - 1]] };
        for k in 0..=m {
            let mut sb = Bb::new(bi.types[0..in_of(k)].to_vec());
            if k < m {
                // segment body up to & including the suspend op, then the poll
                let pos = bi.scs[k];
                let seg_start = if k == 0 { 0 } else { bi.scs[k - 1] + 1 };
                sb.insts.extend_from_slice(&blk.insts[seg_start..=pos]);
                let out = bi.vend[pos];
                sb.next = out as u32;
                let st_a = sb.one(Inst::ConstI64(STATE_OFF as i64));
                let st = sb.one(load(LoadOp::I32, st_a, 0));
                let unw = sb.one(Inst::ConstI32(STATE_UNWINDING));
                let is_unw = sb.one(icmp(IntTy::I32, CmpOp::Eq, st, unw));
                let gid = points.len() as u32;
                let live: Vec<ValIdx> = (0..out as u32).collect();
                seg_blocks.push(sb.finish(Terminator::BrIf {
                    cond: is_unw,
                    then_blk: unwind_base + gid,
                    then_args: live.clone(),
                    else_blk: seg(b, k + 1),
                    else_args: live,
                }));

                // resume plan for this point
                let kind = match &blk.insts[pos] {
                    Inst::CapCall { .. } => SuspendKind::Leaf,
                    Inst::Call { func, args } => SuspendKind::Propagated { callee: *func, args: args.clone() },
                    _ => unreachable!("suspend position is a cap.call or call"),
                };
                let nres = match (&kind, &blk.insts[pos]) {
                    (SuspendKind::Leaf, Inst::CapCall { sig, .. }) => sig.results.len(),
                    (SuspendKind::Propagated { callee, .. }, _) => func_results[*callee as usize].len(),
                    _ => unreachable!(),
                };
                // Over-capture: spill *all* values visible at the op (block params + locals).
                // A leaf reloads its own result too; a propagated frame re-issues its call,
                // so the call's results are recomputed, not spilled.
                let live_types = bi.types[0..out].to_vec();
                let saved_len = match kind {
                    SuspendKind::Leaf => out,
                    SuspendKind::Propagated { .. } => out - nres,
                };
                if live_types[0..saved_len].contains(&ValType::V128) {
                    return Err(TransformError::UnsupportedInst); // v128 spill/reload: future work
                }
                let mut frame_offsets = Vec::with_capacity(saved_len);
                let mut off = 0u64;
                for &t in &live_types[0..saved_len] {
                    off = align_up(off, vsize(t));
                    frame_offsets.push(off);
                    off += vsize(t);
                }
                let frame_size = align_up(off + 4, 16);
                points.push(PointPlan {
                    kind,
                    nres,
                    live_types,
                    saved_len,
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
    let mut db = Bb::new(vec![]);
    let sp_a = db.one(Inst::ConstI64(SHADOW_SP_OFF as i64));
    let sp = db.one(load(LoadOp::I64, sp_a, 0));
    let four = db.one(Inst::ConstI64(4));
    let sp_m4 = db.one(ibin(IntTy::I64, BinOp::Sub, sp, four));
    let rid = db.one(load(LoadOp::I32, sp_m4, 0));
    // id 0 is reserved ("no resume" ⇒ trap); id g+1 selects ARM_g.
    let mut targets: Vec<(BlockIdx, Vec<ValIdx>)> = vec![(trap_blk, vec![])];
    for g in 0..p_total {
        targets.push((arm_base + g, vec![]));
    }
    let dispatch = db.finish(Terminator::BrTable { idx: rid, targets, default: (trap_blk, vec![]) });

    // ---- UNWIND_g / ARM_g, per resume point ----
    let mut unwind_blocks: Vec<Block> = Vec::with_capacity(total_points);
    let mut arm_blocks: Vec<Block> = Vec::with_capacity(total_points);
    for (gid, pt) in points.iter().enumerate() {
        // UNWIND: spill the saved prefix of the live set + the resume id, push the frame.
        let mut ub = Bb::new(pt.live_types.clone());
        let sp_a = ub.one(Inst::ConstI64(SHADOW_SP_OFF as i64));
        let sp = ub.one(load(LoadOp::I64, sp_a, 0)); // this activation's frame base
        for (j, &t) in pt.live_types[0..pt.saved_len].iter().enumerate() {
            ub.zero(store(store_op(t), sp, j as u32, pt.frame_offsets[j])); // value j is block-local idx j
        }
        let rid = ub.one(Inst::ConstI32(gid as i32 + 1));
        ub.zero(store(StoreOp::I32, sp, rid, pt.rid_off));
        let fsz = ub.one(Inst::ConstI64(pt.frame_size as i64));
        let newsp = ub.one(ibin(IntTy::I64, BinOp::Add, sp, fsz));
        ub.zero(store(StoreOp::I64, sp_a, newsp, 0));
        let ret: Vec<ValIdx> = f.results.iter().map(|&t| ub.one(zero_const(t))).collect();
        unwind_blocks.push(ub.finish(Terminator::Return(ret)));

        // ARM: reload the saved set (self-contained — no incoming params), pop, resume.
        let mut ab = Bb::new(vec![]);
        let sp_a = ab.one(Inst::ConstI64(SHADOW_SP_OFF as i64));
        let sp = ab.one(load(LoadOp::I64, sp_a, 0));
        let fsz = ab.one(Inst::ConstI64(pt.frame_size as i64));
        let base = ab.one(ibin(IntTy::I64, BinOp::Sub, sp, fsz));
        let mut reloaded = Vec::with_capacity(pt.saved_len);
        for (j, &t) in pt.live_types[0..pt.saved_len].iter().enumerate() {
            reloaded.push(ab.one(load(load_op(t), base, pt.frame_offsets[j])));
        }
        ab.zero(store(StoreOp::I64, sp_a, base, 0)); // pop: SP = frame base

        // The continuation segment expects the full live set (reloaded), then the op's
        // results (reloaded for a leaf, re-issued for a propagated call).
        let mut cont_args: Vec<ValIdx> = reloaded.clone();
        let arm = match &pt.kind {
            SuspendKind::Leaf => {
                let st_a = ab.one(Inst::ConstI64(STATE_OFF as i64));
                let normal_v = ab.one(Inst::ConstI32(STATE_NORMAL));
                ab.zero(store(StoreOp::I32, st_a, normal_v, 0));
                ab.finish(Terminator::Br { target: pt.cont_seg, args: cont_args })
            }
            SuspendKind::Propagated { callee, args } => {
                // Operands of the original call index into the saved set (all defined before
                // the call), so they map straight onto `reloaded`.
                let mapped: Vec<ValIdx> = args.iter().map(|&a| reloaded[a as usize]).collect();
                let results = ab.many(Inst::Call { func: *callee, args: mapped }, pt.nres);
                cont_args.extend(results);
                ab.finish(Terminator::Br { target: pt.cont_seg, args: cont_args })
            }
        };
        arm_blocks.push(arm);
    }

    // ---- TRAP — br_table default / forged resume id ----
    let trap = Block { params: vec![], insts: vec![], term: Terminator::Unreachable };

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

/// A fresh durable window of `size` bytes: state = `NORMAL`, shadow-SP = `SHADOW_BASE`.
pub fn init_durable_window(size: usize) -> Vec<u8> {
    let mut w = vec![0u8; size];
    write_state(&mut w, STATE_NORMAL);
    w[SHADOW_SP_OFF as usize..SHADOW_SP_OFF as usize + 8]
        .copy_from_slice(&SHADOW_BASE.to_le_bytes());
    w
}

/// Overwrite the state word in a window image (used to drive freeze/thaw).
pub fn write_state(window: &mut [u8], state: i32) {
    window[STATE_OFF as usize..STATE_OFF as usize + 4].copy_from_slice(&state.to_le_bytes());
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
    }
}

fn vsize(t: ValType) -> u64 {
    match t {
        ValType::I32 | ValType::F32 => 4,
        ValType::I64 | ValType::F64 => 8,
        ValType::V128 => 16,
    }
}

fn align_up(x: u64, a: u64) -> u64 {
    (x + a - 1) & !(a - 1)
}

fn store_op(t: ValType) -> StoreOp {
    match t {
        ValType::I32 => StoreOp::I32,
        ValType::I64 => StoreOp::I64,
        ValType::F32 => StoreOp::F32,
        ValType::F64 => StoreOp::F64,
        ValType::V128 => unreachable!("v128 spill rejected earlier"),
    }
}

fn load_op(t: ValType) -> LoadOp {
    match t {
        ValType::I32 => LoadOp::I32,
        ValType::I64 => LoadOp::I64,
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
            "func (i32) -> (i32) {\nblock0(v0: i32):\n  return v0\n}\n",
            12,
        );
        let out = transform_module(&m).expect("transform");
        assert_eq!(out.funcs, m.funcs, "function without cap.call is untouched");
    }

    #[test]
    fn instrumented_function_verifies() {
        let m = parse_with_mem(
            "func (i32) -> (i64) {\nblock0(v0: i32):\n  v1 = i32.const 0\n  v2 = cap.call 2 0 (i32) -> (i64) v0 (v1)\n  v3 = i64.const 100\n  v4 = i64.add v2 v3\n  return v4\n}\n",
            18,
        );
        let out = transform_module(&m).expect("transform");
        svm_verify::verify_module(&out).expect("instrumented IR must verify");
        assert_eq!(out.funcs[0].blocks.len(), 7, "7 instrumented blocks");
    }

    #[test]
    fn two_cap_calls_become_two_resume_points() {
        // Two suspend points in one block ⇒ two br_table arms ⇒ 3·2 + 4 = 10 blocks.
        let m = parse_with_mem(
            "func (i32) -> (i64) {\nblock0(v0: i32):\n  v1 = i32.const 0\n  v2 = cap.call 2 0 (i32) -> (i64) v0 (v1)\n  v3 = cap.call 2 0 (i32) -> (i64) v0 (v1)\n  v4 = i64.add v2 v3\n  return v4\n}\n",
            18,
        );
        let out = transform_module(&m).expect("two resume points are in scope");
        svm_verify::verify_module(&out).expect("instrumented IR must verify");
        assert_eq!(out.funcs[0].blocks.len(), 10, "two-point layout: 3n+4 with n=2");
    }

    #[test]
    fn propagated_chain_instruments_each_frame() {
        // A two-level chain: the caller suspends on its `call` to the leaf, the leaf on
        // its `cap.call`. Both are may-suspend, so both get the 7-block instrumentation.
        let m = parse_with_mem(
            "func (i32) -> (i64) {\nblock0(v0: i32):\n  v1 = call 1 (v0)\n  return v1\n}\nfunc (i32) -> (i64) {\nblock0(v0: i32):\n  v1 = i32.const 0\n  v2 = cap.call 2 0 (i32) -> (i64) v0 (v1)\n  return v2\n}\n",
            18,
        );
        let out = transform_module(&m).expect("transform");
        svm_verify::verify_module(&out).expect("instrumented chain must verify");
        assert_eq!(out.funcs[0].blocks.len(), 7, "caller (propagated) instrumented");
        assert_eq!(out.funcs[1].blocks.len(), 7, "callee (leaf) instrumented");
    }

    #[test]
    fn non_suspending_callee_is_left_unchanged() {
        // func 0 (leaf cap.call) calls func 1 (a pure helper) as a *prefix* op. The helper
        // never suspends, so it is not instrumented and func 0's only suspend point is its
        // own cap.call; the helper's result is spilled/reloaded, never re-issued.
        let m = parse_with_mem(
            "func (i32) -> (i64) {\nblock0(v0: i32):\n  v1 = call 1 (v0)\n  v2 = i32.const 0\n  v3 = cap.call 2 0 (i32) -> (i64) v0 (v2)\n  v4 = i64.add v1 v3\n  return v4\n}\nfunc (i32) -> (i64) {\nblock0(v0: i32):\n  v1 = i64.const 5\n  return v1\n}\n",
            18,
        );
        let helper_before = m.funcs[1].clone();
        let out = transform_module(&m).expect("transform");
        svm_verify::verify_module(&out).expect("verify");
        assert_eq!(out.funcs[0].blocks.len(), 7, "leaf instrumented");
        assert_eq!(out.funcs[1], helper_before, "non-suspending helper untouched");
    }

    #[test]
    fn cap_call_without_memory_is_rejected() {
        let mut m = svm_text::parse_module(
            "func (i32) -> (i64) {\nblock0(v0: i32):\n  v1 = i32.const 0\n  v2 = cap.call 2 0 (i32) -> (i64) v0 (v1)\n  return v2\n}\n",
        )
        .unwrap();
        m.memory = None;
        assert_eq!(transform_module(&m), Err(TransformError::NoMemory));
    }
}
