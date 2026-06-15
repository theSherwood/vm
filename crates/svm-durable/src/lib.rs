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

// Block layout of an instrumented function with `n` suspend points (computed by
// `block indices` below; reduces to the 7-block single-point shape for `n == 1`):
//   0                  PROLOGUE  — dispatch on the state word
//   1 ..= n+1          S_0 .. S_n — forward segments split at each suspend op (S_n = tail)
//   n+2 ..= 2n+1       UNWIND_0 .. UNWIND_{n-1} — spill + return placeholder, per point
//   2n+2               DISPATCH  — read resume id, br_table to an arm
//   2n+3 ..= 3n+2      ARM_0 .. ARM_{n-1} — reload + resume, per point
//   3n+3               TRAP      — forged / reserved resume id

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

/// One suspend point's resume metadata: where it is, what kind, and its frame layout.
struct PointPlan {
    pos: usize,            // instruction index of the suspend op
    kind: SuspendKind,     // leaf cap.call or propagated call
    out: usize,            // value count after the suspend op (= S_{k+1} param count)
    nres: usize,           // result count of the suspend op
    saved_types: Vec<ValType>,
    frame_offsets: Vec<u64>,
    frame_size: u64,
    rid_off: u64,
}

/// Instrument one may-suspend function. `Ok((func, max_frame_size))` on success, or an
/// error for an out-of-scope shape. (Non-may-suspend functions are not passed here.)
///
/// The single block is split at each may-suspend op into forward segments `S_0..S_n`; a
/// poll after each op unwinds (per-point spill + resume id) or continues, and a `br_table`
/// in the prologue dispatch routes a thaw to the matching arm. See the block-layout map.
fn transform_func(
    f: &Func,
    func_results: &[Vec<ValType>],
    may_suspend: &[bool],
) -> Result<(Func, u64), TransformError> {
    // Shape: single block, `return` terminator.
    if f.blocks.len() != 1 {
        return Err(TransformError::UnsupportedShape);
    }
    let b = &f.blocks[0];
    if !matches!(b.term, Terminator::Return(_)) {
        return Err(TransformError::UnsupportedShape);
    }

    // Value types over the whole block, plus the value count after each instruction
    // (`vend[pos]`), so we can slice the block into segments at the suspend ops.
    let p = f.params.len();
    let mut types = f.params.clone();
    let mut vend = Vec::with_capacity(b.insts.len());
    for inst in &b.insts {
        types.extend(result_types(inst, &types, func_results)?);
        vend.push(types.len());
    }

    // Find the may-suspend ops (any number ≥ 1): a `cap.call` (leaf) or a `Call` to a
    // may-suspend callee (propagated). Each becomes a resume point with id `k+1`.
    let mut points: Vec<PointPlan> = Vec::new();
    for (pos, inst) in b.insts.iter().enumerate() {
        let kind = match inst {
            Inst::CapCall { .. } => SuspendKind::Leaf,
            Inst::Call { func, args } if may_suspend[*func as usize] => {
                SuspendKind::Propagated { callee: *func, args: args.clone() }
            }
            _ => continue,
        };
        let out = vend[pos];
        let nres = match (&kind, &b.insts[pos]) {
            (SuspendKind::Leaf, Inst::CapCall { sig, .. }) => sig.results.len(),
            (SuspendKind::Propagated { callee, .. }, _) => func_results[*callee as usize].len(),
            _ => unreachable!("kind matches the instruction"),
        };
        // A leaf reloads its own result (the host produced it); a propagated frame re-issues
        // its call, so the call's results are recomputed, not spilled.
        let save_end = match kind {
            SuspendKind::Leaf => out,
            SuspendKind::Propagated { .. } => out - nres,
        };
        let saved_types: Vec<ValType> = types[p..save_end].to_vec();
        if saved_types.contains(&ValType::V128) {
            return Err(TransformError::UnsupportedInst); // v128 spill/reload: future work
        }
        // Frame layout (DURABILITY.md §12.7): packed values, then resume id in the top word.
        let mut frame_offsets = Vec::with_capacity(saved_types.len());
        let mut off = 0u64;
        for &t in &saved_types {
            off = align_up(off, vsize(t));
            frame_offsets.push(off);
            off += vsize(t);
        }
        let frame_size = align_up(off + 4, 16);
        points.push(PointPlan {
            pos,
            kind,
            out,
            nres,
            saved_types,
            frame_offsets,
            rid_off: frame_size - 4,
            frame_size,
        });
    }
    // A may-suspend function must contain at least one suspend op in its own block.
    if points.is_empty() {
        return Err(TransformError::UnsupportedShape);
    }

    // Block-index layout (see the map near the constants). `n` = number of resume points.
    let n = points.len() as u32;
    let s_blk = |k: u32| 1 + k; // S_0 .. S_n  (k in 0..=n)
    let unwind_blk = |k: u32| n + 2 + k; // UNWIND_0 .. UNWIND_{n-1}
    let dispatch_blk = 2 * n + 2;
    let arm_blk = |k: u32| 2 * n + 3 + k; // ARM_0 .. ARM_{n-1}
    let trap_blk = 3 * n + 3;
    // Param (incoming value) count of segment S_k: params for S_0, else the previous
    // point's `out`.
    let in_of = |k: usize| if k == 0 { p } else { points[k - 1].out };

    // ---- PROLOGUE — dispatch on the state word ----
    let mut pb = Bb::new(f.params.clone());
    let st_a = pb.one(Inst::ConstI64(STATE_OFF as i64));
    let st = pb.one(load(LoadOp::I32, st_a, 0));
    let rw = pb.one(Inst::ConstI32(STATE_REWINDING));
    let is_rw = pb.one(icmp(IntTy::I32, CmpOp::Eq, st, rw));
    let prologue = pb.finish(Terminator::BrIf {
        cond: is_rw,
        then_blk: dispatch_blk,
        then_args: (0..p as u32).collect(),
        else_blk: s_blk(0),
        else_args: (0..p as u32).collect(),
    });

    // ---- S_0 .. S_{n-1}: forward segment + suspend op + poll ----
    let mut seg_blocks: Vec<Block> = Vec::with_capacity(points.len() + 1);
    for (k, pt) in points.iter().enumerate() {
        let seg_start = if k == 0 { 0 } else { points[k - 1].pos + 1 };
        let mut sb = Bb::new(types[0..in_of(k)].to_vec());
        sb.insts.extend_from_slice(&b.insts[seg_start..=pt.pos]); // segment + the suspend op
        sb.next = pt.out as u32;
        let st_a = sb.one(Inst::ConstI64(STATE_OFF as i64));
        let st = sb.one(load(LoadOp::I32, st_a, 0));
        let unw = sb.one(Inst::ConstI32(STATE_UNWINDING));
        let is_unw = sb.one(icmp(IntTy::I32, CmpOp::Eq, st, unw));
        let live: Vec<ValIdx> = (0..pt.out as u32).collect();
        seg_blocks.push(sb.finish(Terminator::BrIf {
            cond: is_unw,
            then_blk: unwind_blk(k as u32),
            then_args: live.clone(),
            else_blk: s_blk(k as u32 + 1),
            else_args: live,
        }));
    }
    // ---- S_n: the tail (post-last-suspend insts + the real return) ----
    let last = points.len() - 1;
    seg_blocks.push(Block {
        params: types[0..points[last].out].to_vec(),
        insts: b.insts[points[last].pos + 1..].to_vec(),
        term: b.term.clone(),
    });

    // ---- UNWIND_k: spill point k's saved set + resume id k+1, return a placeholder ----
    let mut unwind_blocks: Vec<Block> = Vec::with_capacity(points.len());
    for (k, pt) in points.iter().enumerate() {
        let mut ub = Bb::new(types[0..pt.out].to_vec());
        let sp_a = ub.one(Inst::ConstI64(SHADOW_SP_OFF as i64));
        let sp = ub.one(load(LoadOp::I64, sp_a, 0)); // this activation's frame base
        for (j, &t) in pt.saved_types.iter().enumerate() {
            // saved value block-local index `p + j` (the saved set is contiguous [p, save_end))
            ub.zero(store(store_op(t), sp, (p + j) as u32, pt.frame_offsets[j]));
        }
        let rid = ub.one(Inst::ConstI32(k as i32 + 1));
        ub.zero(store(StoreOp::I32, sp, rid, pt.rid_off));
        let fsz = ub.one(Inst::ConstI64(pt.frame_size as i64));
        let newsp = ub.one(ibin(IntTy::I64, BinOp::Add, sp, fsz));
        ub.zero(store(StoreOp::I64, sp_a, newsp, 0));
        let ret: Vec<ValIdx> = f.results.iter().map(|&t| ub.one(zero_const(t))).collect();
        unwind_blocks.push(ub.finish(Terminator::Return(ret)));
    }

    // ---- DISPATCH — read the resume id at SP-4 and br_table to the matching arm ----
    let mut db = Bb::new(f.params.clone());
    let sp_a = db.one(Inst::ConstI64(SHADOW_SP_OFF as i64));
    let sp = db.one(load(LoadOp::I64, sp_a, 0));
    let four = db.one(Inst::ConstI64(4));
    let sp_m4 = db.one(ibin(IntTy::I64, BinOp::Sub, sp, four));
    let rid = db.one(load(LoadOp::I32, sp_m4, 0));
    // id 0 is reserved ("no resume" ⇒ trap); id k+1 selects ARM_k.
    let mut targets: Vec<(BlockIdx, Vec<ValIdx>)> = vec![(trap_blk, vec![])];
    for k in 0..points.len() as u32 {
        targets.push((arm_blk(k), (0..p as u32).collect()));
    }
    let dispatch = db.finish(Terminator::BrTable {
        idx: rid,
        targets,
        default: (trap_blk, vec![]),
    });

    // ---- ARM_k: reload point k's saved set, pop, resume the suspend op, continue ----
    let mut arm_blocks: Vec<Block> = Vec::with_capacity(points.len());
    for (k, pt) in points.iter().enumerate() {
        let mut ab = Bb::new(f.params.clone());
        let sp_a = ab.one(Inst::ConstI64(SHADOW_SP_OFF as i64));
        let sp = ab.one(load(LoadOp::I64, sp_a, 0));
        let fsz = ab.one(Inst::ConstI64(pt.frame_size as i64));
        let base = ab.one(ibin(IntTy::I64, BinOp::Sub, sp, fsz));
        let mut reloaded = Vec::with_capacity(pt.saved_types.len());
        for (j, &t) in pt.saved_types.iter().enumerate() {
            reloaded.push(ab.one(load(load_op(t), base, pt.frame_offsets[j])));
        }
        ab.zero(store(StoreOp::I64, sp_a, base, 0)); // pop: SP = frame base

        // S_{k+1} expects `out` args: params, the reloaded values, then the suspend op's
        // results (reloaded for a leaf, re-issued for a propagated call).
        let mut cont_args: Vec<ValIdx> = (0..p as u32).collect();
        cont_args.extend(reloaded.iter().copied());
        let next_fwd = s_blk(k as u32 + 1);
        let arm = match &pt.kind {
            SuspendKind::Leaf => {
                // Deepest frame: the host already performed the `cap.call`; its result is in
                // the reloaded set. Flip the state word back to NORMAL and continue forward.
                let st_a = ab.one(Inst::ConstI64(STATE_OFF as i64));
                let normal_v = ab.one(Inst::ConstI32(STATE_NORMAL));
                ab.zero(store(StoreOp::I32, st_a, normal_v, 0));
                ab.finish(Terminator::Br { target: next_fwd, args: cont_args })
            }
            SuspendKind::Propagated { callee, args } => {
                // Non-deepest frame (R8): leave the state REWINDING and re-issue the call so
                // the callee rewinds. Map the call's operands to params / reloaded values.
                let mapped: Vec<ValIdx> = args
                    .iter()
                    .map(|&a| if (a as usize) < p { a } else { reloaded[a as usize - p] })
                    .collect();
                let results = ab.many(Inst::Call { func: *callee, args: mapped }, pt.nres);
                cont_args.extend(results);
                ab.finish(Terminator::Br { target: next_fwd, args: cont_args })
            }
        };
        arm_blocks.push(arm);
    }

    // ---- TRAP — br_table default / forged resume id ----
    let trap = Block { params: vec![], insts: vec![], term: Terminator::Unreachable };

    // Assemble in the order the index layout assumes.
    let mut blocks = Vec::with_capacity((3 * n + 4) as usize);
    blocks.push(prologue);
    blocks.extend(seg_blocks);
    blocks.extend(unwind_blocks);
    blocks.push(dispatch);
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
