//! Stage 1–2 — the **first Futamura projection** over the IR (see `DESIGN.md` §20c).
//!
//! [`specialize`] / [`specialize_with`] / [`specialize_with_config`] take a function (an
//! *interpreter*) and a list of which parameters are **static** (a known constant at
//! specialization time) versus **dynamic** (a runtime value), and produce a residual function
//! specialized to the static inputs. Combined with a **program the caller declares constant** —
//! a readonly data segment, a caller-promised constant region, or explicit overlay bytes
//! ([`SpecConfig`]) — specializing the interpreter against that program folds the opcode loads to
//! constants, resolves the dispatch `br_table` to a single edge, and unrolls the interpreter loop
//! following the program. The dispatch loop disappears; what's left is the *compiled* program.
//! `spec(interp, program)(input) ≡ interp(program, input)`.
//!
//! The engine is **online polyvariant symbolic execution**, weval's shape:
//!
//! - Each SSA value is abstractly either a known [`Known`] constant or *dynamic* (a value in
//!   the residual block being built). Pure integer ops with all-constant operands fold (reusing
//!   the Stage-0 arithmetic, so it matches the interpreter exactly); a trapping fold (div/rem
//!   by zero) is emitted residually so it still traps. Anything with a dynamic operand is
//!   emitted into the residual.
//! - A **load from a constant address the caller declared constant** folds to those bytes
//!   ([`read_const_mem`]) — the "constant memory" read. By default that means a readonly data
//!   segment; a caller can also promise an arbitrary region or supply overlay bytes
//!   ([`SpecConfig`]). Constancy is a *caller contract*, not enforced here — a false promise is a
//!   miscompile, never an escape (the residual is re-verified). Any other load is emitted
//!   residually (faithful), so unpromised mutable memory is never folded.
//! - **Value-stack renaming (Stage 2).** A caller may designate a private byte range (the
//!   interpreter's operand stack / locals) as *renameable* ([`specialize_with`]). Stores into it
//!   at constant addresses update an **abstract memory** instead of emitting a store; loads read
//!   that abstract memory instead of emitting a load — so the in-memory stack is lifted into SSA
//!   and disappears from the residual. Narrow (`i8`/`i16`/`i32`-of-`i64`) cells are renamed too: a
//!   constant cell keeps its raw bytes and is re-extended (sign/zero) per the load op, so char/short
//!   locals fold exactly. Soundness is kept by construction: the region is assumed zero-initialized
//!   and private, every write to it is a tracked constant-address store, and any access that can't
//!   be resolved abstractly (a dynamic address that might alias the region, a *narrow store of a
//!   dynamic value* — which would need residual masking to read back — a partial-width overlap, a
//!   call) returns [`SpecError::Unsupported`] rather than guessing.
//! - The **context** threaded through the CFG is `(call stack, the constant valuation of the live
//!   abstract-memory cells)`, where each stack frame is `(source block, the constant valuation of
//!   its live SSA values)` — a one-frame stack for a single function, deeper when calls are
//!   CFG-inlined. One residual block is generated per context and memoized, so distinct constants
//!   (e.g. the program counter / stack pointer) drive loop unrolling, while repeated contexts
//!   reconnect — bounding termination. Dynamic SSA values (across every frame) *and* dynamic memory
//!   cells become the residual block's parameters; constant ones are baked in.
//!
//! **Untrusted for escape** like the rest of the crate: the residual is meant to be
//! re-verified before it runs. The differential harness (`tests/specialize.rs`) is the spec —
//! the residual must equal the interpreter on the reference interpreter for every input.
//!
//! **Cross-function `call`.** A direct [`Inst::Call`] (and a [`Terminator::ReturnCall`] tail call)
//! is **inlined at the call site** — the callee is symbolically executed in the *caller's* context,
//! sharing the same abstract memory, so a callee that reads constant memory or touches the renamed
//! operand stack folds exactly as inline code would. The call disappears; the callee's residual is
//! spliced into the caller. Two paths, picked automatically:
//!
//! - **Straight-line (the fast path).** A callee whose control flow resolves statically is traced
//!   into the caller's current residual block (static recursion unrolls, bounded by an inline-fuel
//!   budget). No new blocks; the result flows on inline.
//! - **CFG inlining (dynamic control flow).** When tracing hits a branch that stays *dynamic* (a
//!   data-dependent branch that must survive as a residual branch), the engine instead inlines the
//!   callee's CFG as residual blocks: the symbolic-execution **context becomes a call stack** of
//!   frames `(func, block, params)`, the caller's live values are threaded through the callee as
//!   block parameters (dead ones are cleaned up by the optimizer), and each callee `return` becomes
//!   a branch to the caller's continuation. Recursion + dynamic control flow, loops in the callee,
//!   and `unreachable` callee paths all work; one residual function still comes out.
//!
//! Indirect calls (`call_indirect`, `return_call_indirect`) and host/capability calls are not
//! inlined.
//!
//! **Scope.** Integer and **scalar float** ops — arithmetic, compares, fused multiply-add,
//! float↔int conversions, reinterpret/demote/promote casts — are specialized (folded where the
//! operands are constant, bit-for-bit the interpreter). Remaining **pure, single-result** value ops
//! — v128 (SIMD) lane ops, pointer ops — are emitted faithfully into the residual even though they
//! are not constant-folded yet, so dispatch is still eliminated around them. Direct calls are
//! inlined (above). Effectful, multi-result, or other cross-function ops (indirect/host calls,
//! atomics, fibers/threads), and memory accesses the engine can't resolve, return
//! [`SpecError::Unsupported`] rather than guessing.

use std::collections::{BTreeMap, HashMap, VecDeque};

use svm_ir::{ConvOp, Func, Inst, IntTy, LoadOp, Module, StoreOp, Terminator, ValType};
use svm_verify::func_value_types;

use crate::{fold_int_bin, fold_int_cmp, fold_int_un, Known};

/// How one parameter of the function being specialized is bound.
#[derive(Clone, Copy, Debug)]
pub enum SpecArg {
    /// Static: a known `i32` constant at specialization time (baked into the residual).
    ConstI32(i32),
    /// Static: a known `i64` constant at specialization time.
    ConstI64(i64),
    /// Dynamic: a runtime value (becomes a parameter of the residual function).
    Dynamic,
}

/// Why specialization could not produce a residual.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SpecError {
    /// The requested function index does not exist.
    BadFunc,
    /// `args.len()` did not match the function's parameter count.
    ArityMismatch,
    /// An instruction (or a memory access) outside the supported subset appeared.
    Unsupported,
    /// The residual exceeded the block budget — a likely-divergent specialization.
    Budget,
}

/// The abstract value of an SSA value during symbolic execution.
#[derive(Clone, Copy)]
enum Abs {
    /// A compile-time constant.
    Const(Known),
    /// A runtime value, identified by its index in the residual block currently being built.
    Dyn(u32),
}

/// The constant valuation of a frame's threaded SSA values (block params, then any further values
/// captured when the frame is suspended at a call): `Some` for a baked-in constant, `None` for a
/// dynamic value carried as a residual block parameter.
type ParamPattern = Vec<Option<Known>>;
/// Live abstract-memory cells at a program point, sorted by address: `(addr, width, value)`.
type MemPattern = Vec<(u64, u32, Option<Known>)>;

/// One activation in the symbolic call stack: a position in a source function plus the constant
/// valuation of its live SSA values. `ip` is the instruction index to resume at — `0` for a
/// freshly-entered block (where `env` is exactly the block's parameters), or, for a frame suspended
/// at a [`Inst::Call`] that needed CFG inlining, the index just after the call (where `env` has been
/// extended with the call's results before the frame resumes).
#[derive(Clone, PartialEq, Eq, Hash)]
struct Frame {
    func: u32,
    block: u32,
    ip: usize,
    env: ParamPattern,
}

/// One residual block still to be generated: the symbolic call stack (innermost/active frame last)
/// plus the abstract memory threaded into it. The full context is the memoization key.
struct Task {
    frames: Vec<Frame>,
    mem: MemPattern,
}

/// A frame with its SSA values resolved to concrete abstract values (constants or residual SSA
/// indices) — the working form while a residual block is built, and the input to [`Spec::branch_to`].
#[derive(Clone)]
struct FrameAbs {
    func: u32,
    block: u32,
    ip: usize,
    env: Vec<Abs>,
}

/// What executing the active frame's straight-line body produced.
enum Exec {
    /// Ran to the terminator with `env` fully populated.
    Done,
    /// Hit a call needing CFG inlining: suspend the active frame here and enter the callee.
    Suspend {
        callee: u32,
        args: Vec<Abs>,
        resume_ip: usize,
    },
}

/// The error channel for the straight-line inliner: distinguishes "this callee needs CFG inlining"
/// (a control-flow decision, recoverable by the caller) from a genuine [`SpecError`].
enum InlineErr {
    /// The callee's control flow stayed dynamic — fall back to CFG inlining.
    NeedsCfg,
    /// A real failure (unsupported op/access, or budget exhausted).
    Spec(SpecError),
}

/// The default ceiling on residual blocks before we declare likely divergence.
const DEFAULT_BUDGET: usize = 1 << 16;

/// The ceiling on block-steps a single inlined call site may take (across all nesting). A trace
/// that exceeds it — runaway / unbounded-recursion inlining — gives up with [`SpecError::Budget`].
/// Shared as fuel across nested inlines, so it also bounds inline recursion depth.
const INLINE_FUEL: usize = 1 << 16;

/// What the caller promises about memory, to steer specialization. All fields default to empty,
/// which reproduces the plain Stage-1 behavior (readonly data segments still fold).
///
/// **These are caller contracts, not enforced invariants.** Declaring a region constant — or an
/// overlay's bytes — is a promise that those bytes do not change between specialization time and
/// every execution of the residual. If the promise is false (self-modifying code, a racing
/// thread, …) the residual computes the wrong answer. It is still *safe*: the residual is meant to
/// be re-verified, so confinement and capability checks hold regardless — a broken promise is a
/// miscompile, never an escape. (This mirrors weval's `assume_const_memory`.)
#[derive(Clone, Debug, Default)]
pub struct SpecConfig {
    /// A private, zero-initialized scratch range `[lo, hi)` (the interpreter's operand stack /
    /// locals) whose stores/loads are renamed into SSA and elided from the residual (Stage 2).
    pub rename: Option<(u64, u64)>,
    /// Window ranges `[lo, hi)` the caller promises are constant at specialization time. Loads from
    /// them fold to the module's initial data image (readonly **or not**); bytes not covered by any
    /// data segment read as zero (the demand-zeroed window).
    pub const_regions: Vec<(u64, u64)>,
    /// Explicit constant bytes at a base window address, for a program not described by a data
    /// segment (e.g. one written into the window before the call). Loads fully inside an overlay
    /// fold to its bytes. Overlays take precedence over data segments.
    pub const_overlays: Vec<(u64, Vec<u8>)>,
    /// Caller promise: the [`rename`](Self::rename) region is **private** — touched only by the
    /// constant-address accesses the engine renames, never by a dynamic-address load/store. With it
    /// set, a dynamic-address access (whose target the engine can't pin) is emitted as a faithful
    /// residual access instead of conservatively refusing — letting an interpreter use a renamed
    /// operand stack *and* a pointer-addressed heap at once. Unsound if violated (a dynamic write
    /// into the region would desync the elided renamed cells), so it is opt-in and off by default.
    pub rename_is_private: bool,
}

/// Specialize with no caller memory hints (only readonly data segments fold).
pub fn specialize(module: &Module, func: u32, args: &[SpecArg]) -> Result<Module, SpecError> {
    specialize_with_config(module, func, args, &SpecConfig::default())
}

/// Specialize with a renameable memory region (Stage 2 value-stack renaming), no other hints.
pub fn specialize_with(
    module: &Module,
    func: u32,
    args: &[SpecArg],
    rename: Option<(u64, u64)>,
) -> Result<Module, SpecError> {
    specialize_with_config(
        module,
        func,
        args,
        &SpecConfig {
            rename,
            ..SpecConfig::default()
        },
    )
}

/// Specialize `module.funcs[func]` against the static/dynamic binding in `args`, steered by
/// `config`. Produces a module with a single residual function (index 0); the original memory and
/// data segments are carried through, so any residual loads still resolve.
pub fn specialize_with_config(
    module: &Module,
    func: u32,
    args: &[SpecArg],
    config: &SpecConfig,
) -> Result<Module, SpecError> {
    let f = module.funcs.get(func as usize).ok_or(SpecError::BadFunc)?;
    if args.len() != f.params.len() {
        return Err(SpecError::ArityMismatch);
    }

    // The entry context, and the residual function's parameters (the dynamic args, in order).
    let mut params = Vec::with_capacity(args.len());
    let mut residual_params = Vec::new();
    for (i, arg) in args.iter().enumerate() {
        match arg {
            SpecArg::ConstI32(v) => params.push(Some(Known::I32(*v))),
            SpecArg::ConstI64(v) => params.push(Some(Known::I64(*v))),
            SpecArg::Dynamic => {
                params.push(None);
                residual_params.push(f.params[i]);
            }
        }
    }

    let has_memory = module.memory.is_some();
    let value_types = module
        .funcs
        .iter()
        .map(|f| func_value_types(f, &module.funcs, has_memory))
        .collect();
    let mut spec = Spec {
        module,
        config,
        value_types,
        memo: HashMap::new(),
        queue: VecDeque::new(),
        next_id: 0,
    };
    // The entry context is a single frame: the function being specialized, at its entry block.
    spec.intern(
        vec![Frame {
            func,
            block: 0,
            ip: 0,
            env: params,
        }],
        Vec::new(),
    );

    let mut blocks = Vec::new();
    while let Some(task) = spec.queue.pop_front() {
        if blocks.len() >= DEFAULT_BUDGET {
            return Err(SpecError::Budget);
        }
        let block = spec.build_block(task)?;
        blocks.push(block);
    }

    Ok(Module {
        funcs: vec![Func {
            params: residual_params,
            results: f.results.clone(),
            blocks,
        }],
        memory: module.memory,
        data: module.data.clone(),
        imports: vec![],
        debug_info: None,
    })
}

struct Spec<'a> {
    module: &'a Module,
    config: &'a SpecConfig,
    /// Per-function, per-block, per-value source types (`value_types[func][block][value_idx]`) —
    /// used to type the SSA values threaded into a residual block as block parameters.
    value_types: Vec<Vec<Vec<ValType>>>,
    /// `(call stack, memory pattern) → residual block id`. The memo that makes the loop terminate
    /// and that closes residual loops.
    memo: HashMap<(Vec<Frame>, MemPattern), u32>,
    queue: VecDeque<Task>,
    next_id: u32,
}

impl Spec<'_> {
    /// Get (or create) the residual block id for a context, enqueuing it the first time it is
    /// seen. Ids are assigned in enqueue order and blocks are produced in that same (FIFO) order,
    /// so id == position in the output `blocks`.
    fn intern(&mut self, frames: Vec<Frame>, mem: MemPattern) -> u32 {
        let key = (frames, mem);
        if let Some(&id) = self.memo.get(&key) {
            return id;
        }
        let id = self.next_id;
        self.next_id += 1;
        self.queue.push_back(Task {
            frames: key.0.clone(),
            mem: key.1.clone(),
        });
        self.memo.insert(key, id);
        id
    }

    fn build_block(&mut self, task: Task) -> Result<svm_ir::Block, SpecError> {
        let module = self.module;

        // Reconstruct every frame's env and the memory cells from the context, assigning a fresh
        // residual block parameter to each dynamic lane. The canonical order — frames outermost→
        // innermost, each frame's dynamic env slots in order, then dynamic memory cells by address —
        // is shared with `branch_to`, so a successor passes its arguments in exactly the order this
        // block declares its parameters. Constant lanes are baked back in.
        let mut params: Vec<ValType> = Vec::new();
        let mut rnext: u32 = 0;
        let mut frames: Vec<FrameAbs> = Vec::with_capacity(task.frames.len());
        for fr in &task.frames {
            let types = &self.value_types[fr.func as usize][fr.block as usize];
            let mut env = Vec::with_capacity(fr.env.len());
            for (i, slot) in fr.env.iter().enumerate() {
                match slot {
                    Some(k) => env.push(Abs::Const(*k)),
                    None => {
                        env.push(Abs::Dyn(rnext));
                        rnext += 1;
                        params.push(types[i]);
                    }
                }
            }
            frames.push(FrameAbs {
                func: fr.func,
                block: fr.block,
                ip: fr.ip,
                env,
            });
        }
        let mut mem: BTreeMap<u64, (u32, Abs)> = BTreeMap::new();
        for &(addr, width, slot) in &task.mem {
            match slot {
                Some(k) => {
                    mem.insert(addr, (width, Abs::Const(k)));
                }
                None => {
                    mem.insert(addr, (width, Abs::Dyn(rnext)));
                    rnext += 1;
                    params.push(cell_type(width));
                }
            }
        }

        // Execute the active (innermost) frame's block from its resume point. `fuel` bounds any
        // straight-line call inlining within this block.
        let active = frames.pop().expect("a context has at least one frame");
        let src = &module.funcs[active.func as usize].blocks[active.block as usize];
        let mut env = active.env;
        let mut out: Vec<Inst> = Vec::new();
        let mut fuel = INLINE_FUEL;
        let exec = self.exec_insts(
            &src.insts, active.ip, &mut env, &mut mem, &mut out, &mut rnext, &mut fuel,
        )?;

        let term = match exec {
            // A call needs CFG inlining: suspend the active frame (env captured, resume just past
            // the call) and branch to the callee's entry. The caller's live values ride along as
            // this edge's arguments and reappear, threaded, until the callee returns.
            Exec::Suspend {
                callee,
                args,
                resume_ip,
            } => {
                frames.push(FrameAbs {
                    func: active.func,
                    block: active.block,
                    ip: resume_ip,
                    env,
                });
                frames.push(FrameAbs {
                    func: callee,
                    block: 0,
                    ip: 0,
                    env: args,
                });
                let (target, args) = self.branch_to(&frames, &mem);
                Terminator::Br { target, args }
            }
            Exec::Done => self.finish_term(
                &src.term,
                frames,
                active.func,
                &env,
                &mut mem,
                &mut out,
                &mut rnext,
                &mut fuel,
            )?,
        };

        Ok(svm_ir::Block {
            params,
            insts: out,
            term,
        })
    }

    /// Execute the active block's straight-line body from `start_ip`, pushing each instruction's
    /// abstract result(s) onto `env`. A direct [`Inst::Call`] is first attempted as a straight-line
    /// inline; if that callee needs CFG inlining, execution stops with [`Exec::Suspend`] so
    /// [`Self::build_block`] can split the block at the call. Other instructions go through
    /// [`Self::eval_inst`].
    #[allow(clippy::too_many_arguments)]
    fn exec_insts(
        &self,
        insts: &[Inst],
        start_ip: usize,
        env: &mut Vec<Abs>,
        mem: &mut BTreeMap<u64, (u32, Abs)>,
        out: &mut Vec<Inst>,
        rnext: &mut u32,
        fuel: &mut usize,
    ) -> Result<Exec, SpecError> {
        for (k, inst) in insts.iter().enumerate().skip(start_ip) {
            if let Inst::Call { func, args } = inst {
                let args_abs: Vec<Abs> = args.iter().map(|&a| env[a as usize]).collect();
                match self.try_straightline(*func, &args_abs, mem, out, rnext, fuel)? {
                    Some(results) => env.extend(results),
                    None => {
                        return Ok(Exec::Suspend {
                            callee: *func,
                            args: args_abs,
                            resume_ip: k + 1,
                        })
                    }
                }
            } else if let Some(res) = self.eval_inst(inst, env, mem, out, rnext)? {
                env.push(res);
            }
        }
        Ok(Exec::Done)
    }

    /// Attempt to inline a direct call as straight-line code into the current residual block. On
    /// success returns the callee's result values (the emissions are kept). If the callee's control
    /// flow stays dynamic, every emission/memory effect is rolled back and `None` is returned, so
    /// the caller falls back to CFG inlining. A real failure surfaces as [`SpecError`].
    fn try_straightline(
        &self,
        callee: u32,
        args_abs: &[Abs],
        mem: &mut BTreeMap<u64, (u32, Abs)>,
        out: &mut Vec<Inst>,
        rnext: &mut u32,
        fuel: &mut usize,
    ) -> Result<Option<Vec<Abs>>, SpecError> {
        let saved_len = out.len();
        let saved_rnext = *rnext;
        let saved_mem = mem.clone();
        match self.inline_call(callee, args_abs, mem, out, rnext, fuel) {
            Ok(results) => Ok(Some(results)),
            Err(InlineErr::NeedsCfg) => {
                out.truncate(saved_len);
                *rnext = saved_rnext;
                *mem = saved_mem;
                Ok(None)
            }
            Err(InlineErr::Spec(e)) => Err(e),
        }
    }

    /// Inline a direct call as a single straight-line trace into the *caller's* context, sharing the
    /// live abstract memory (`mem`) and residual stream (`out`/`rnext`) — so a callee that folds
    /// constant memory or touches the renamed operand stack behaves as if written inline. Static
    /// recursion unrolls (bounded by `fuel`, shared across nested inlines so it also caps recursion
    /// depth). Returns [`InlineErr::NeedsCfg`] the moment control flow stays dynamic (a dynamic
    /// branch, or an `unreachable` path that needs to become a real terminator), so the caller can
    /// fall back to CFG inlining; a callee tail call is itself inlined.
    fn inline_call(
        &self,
        func: u32,
        args_abs: &[Abs],
        mem: &mut BTreeMap<u64, (u32, Abs)>,
        out: &mut Vec<Inst>,
        rnext: &mut u32,
        fuel: &mut usize,
    ) -> Result<Vec<Abs>, InlineErr> {
        let g = self
            .module
            .funcs
            .get(func as usize)
            .ok_or(InlineErr::Spec(SpecError::BadFunc))?;
        let mut cur_args: Vec<Abs> = args_abs.to_vec();
        let mut block_idx = 0u32;
        loop {
            *fuel = fuel
                .checked_sub(1)
                .ok_or(InlineErr::Spec(SpecError::Budget))?;
            let blk = g
                .blocks
                .get(block_idx as usize)
                .ok_or(InlineErr::Spec(SpecError::BadFunc))?;
            // Seed this block's local env with its incoming parameter values, then run the body.
            let mut genv = cur_args;
            self.exec_insts_sl(&blk.insts, &mut genv, mem, out, rnext, fuel)?;
            match &blk.term {
                Terminator::Return(vals) => {
                    return Ok(vals.iter().map(|&v| genv[v as usize]).collect());
                }
                // A callee tail call: its results are the callee's results, so inline and forward.
                Terminator::ReturnCall { func, args } => {
                    let a: Vec<Abs> = args.iter().map(|&x| genv[x as usize]).collect();
                    return self.inline_call(*func, &a, mem, out, rnext, fuel);
                }
                // Intra-callee control flow must resolve to a single successor (straight-line trace);
                // a dynamic branch hands off to CFG inlining.
                Terminator::Br { target, args } => {
                    cur_args = args.iter().map(|&a| genv[a as usize]).collect();
                    block_idx = *target;
                }
                Terminator::BrIf {
                    cond,
                    then_blk,
                    then_args,
                    else_blk,
                    else_args,
                } => {
                    let c = match genv[*cond as usize] {
                        Abs::Const(c) => {
                            c.as_i32().ok_or(InlineErr::Spec(SpecError::Unsupported))?
                        }
                        Abs::Dyn(_) => return Err(InlineErr::NeedsCfg),
                    };
                    let (blk, args) = if c != 0 {
                        (*then_blk, then_args)
                    } else {
                        (*else_blk, else_args)
                    };
                    cur_args = args.iter().map(|&a| genv[a as usize]).collect();
                    block_idx = blk;
                }
                Terminator::BrTable {
                    idx,
                    targets,
                    default,
                } => {
                    let i = match genv[*idx as usize] {
                        Abs::Const(c) => {
                            c.as_i32().ok_or(InlineErr::Spec(SpecError::Unsupported))? as u32
                                as usize
                        }
                        Abs::Dyn(_) => return Err(InlineErr::NeedsCfg),
                    };
                    let (blk, args) = targets.get(i).unwrap_or(default);
                    cur_args = args.iter().map(|&a| genv[a as usize]).collect();
                    block_idx = *blk;
                }
                // An `unreachable` callee path must become a real residual terminator — only the CFG
                // path can emit one — so hand off. An indirect tail call is not inlined at all.
                Terminator::Unreachable => return Err(InlineErr::NeedsCfg),
                Terminator::ReturnCallIndirect { .. } => {
                    return Err(InlineErr::Spec(SpecError::Unsupported))
                }
            }
        }
    }

    /// Straight-line instruction executor used while tracing an inlined callee: like
    /// [`Self::exec_insts`] but a nested call must also stay straight-line (its
    /// [`InlineErr::NeedsCfg`] propagates so the whole attempt rolls back to the outermost call).
    fn exec_insts_sl(
        &self,
        insts: &[Inst],
        env: &mut Vec<Abs>,
        mem: &mut BTreeMap<u64, (u32, Abs)>,
        out: &mut Vec<Inst>,
        rnext: &mut u32,
        fuel: &mut usize,
    ) -> Result<(), InlineErr> {
        for inst in insts {
            if let Inst::Call { func, args } = inst {
                let a: Vec<Abs> = args.iter().map(|&x| env[x as usize]).collect();
                let results = self.inline_call(*func, &a, mem, out, rnext, fuel)?;
                env.extend(results);
            } else if let Some(res) = self
                .eval_inst(inst, env, mem, out, rnext)
                .map_err(InlineErr::Spec)?
            {
                env.push(res);
            }
        }
        Ok(())
    }

    /// Abstractly evaluate one instruction. Returns the abstract value of its result (`None` for a
    /// result-less instruction such as a store), emitting any residual instruction needed.
    fn eval_inst(
        &self,
        inst: &Inst,
        env: &[Abs],
        mem: &mut BTreeMap<u64, (u32, Abs)>,
        out: &mut Vec<Inst>,
        rnext: &mut u32,
    ) -> Result<Option<Abs>, SpecError> {
        let abs = match *inst {
            Inst::ConstI32(v) => Abs::Const(Known::I32(v)),
            Inst::ConstI64(v) => Abs::Const(Known::I64(v)),
            Inst::ConstF32(b) => Abs::Const(Known::F32(b)),
            Inst::ConstF64(b) => Abs::Const(Known::F64(b)),

            Inst::IntBin { ty, op, a, b } => {
                let (av, bv) = (env[a as usize], env[b as usize]);
                if let (Abs::Const(x), Abs::Const(y)) = (av, bv) {
                    if let Some(k) = fold_int_bin(ty, op, x, y) {
                        return Ok(Some(Abs::Const(k)));
                    }
                }
                let a = materialize(av, out, rnext);
                let b = materialize(bv, out, rnext);
                out.push(Inst::IntBin { ty, op, a, b });
                Abs::Dyn(bump(rnext))
            }
            Inst::IntCmp { ty, op, a, b } => {
                let (av, bv) = (env[a as usize], env[b as usize]);
                if let (Abs::Const(x), Abs::Const(y)) = (av, bv) {
                    if let Some(k) = fold_int_cmp(ty, op, x, y) {
                        return Ok(Some(Abs::Const(k)));
                    }
                }
                let a = materialize(av, out, rnext);
                let b = materialize(bv, out, rnext);
                out.push(Inst::IntCmp { ty, op, a, b });
                Abs::Dyn(bump(rnext))
            }
            Inst::IntUn { ty, op, a } => {
                let av = env[a as usize];
                if let Abs::Const(x) = av {
                    if let Some(k) = fold_int_un(ty, op, x) {
                        return Ok(Some(Abs::Const(k)));
                    }
                }
                let a = materialize(av, out, rnext);
                out.push(Inst::IntUn { ty, op, a });
                Abs::Dyn(bump(rnext))
            }
            Inst::Eqz { ty, a } => {
                let av = env[a as usize];
                if let Abs::Const(x) = av {
                    let z = match ty {
                        IntTy::I32 => x.as_i32().map(|v| v == 0),
                        IntTy::I64 => x.as_i64().map(|v| v == 0),
                    };
                    if let Some(b) = z {
                        return Ok(Some(Abs::Const(Known::I32(b as i32))));
                    }
                }
                let a = materialize(av, out, rnext);
                out.push(Inst::Eqz { ty, a });
                Abs::Dyn(bump(rnext))
            }
            Inst::Convert { op, a } => {
                let av = env[a as usize];
                if let Abs::Const(x) = av {
                    let folded = match op {
                        ConvOp::ExtendI32S => x.as_i32().map(|v| Known::I64(v as i64)),
                        ConvOp::ExtendI32U => x.as_i32().map(|v| Known::I64(v as u32 as i64)),
                        ConvOp::WrapI64 => x.as_i64().map(|v| Known::I32(v as i32)),
                    };
                    if let Some(k) = folded {
                        return Ok(Some(Abs::Const(k)));
                    }
                }
                let a = materialize(av, out, rnext);
                out.push(Inst::Convert { op, a });
                Abs::Dyn(bump(rnext))
            }
            // `select` with a constant condition forwards the chosen operand's abstract value.
            Inst::Select { cond, a, b } => {
                if let Abs::Const(c) = env[cond as usize] {
                    if let Some(c) = c.as_i32() {
                        return Ok(Some(if c != 0 {
                            env[a as usize]
                        } else {
                            env[b as usize]
                        }));
                    }
                }
                let cond = materialize(env[cond as usize], out, rnext);
                let a = materialize(env[a as usize], out, rnext);
                let b = materialize(env[b as usize], out, rnext);
                out.push(Inst::Select { cond, a, b });
                Abs::Dyn(bump(rnext))
            }

            Inst::Load {
                op,
                addr,
                offset,
                align,
            } => return self.eval_load(op, addr, offset, align, env, mem, out, rnext),
            Inst::Store {
                op,
                addr,
                value,
                offset,
                align,
            } => return self.eval_store(op, addr, value, offset, align, env, mem, out, rnext),

            // Any other pure, single-result value op. A scalar **float** op with all-constant
            // operands folds (bit-for-bit the interpreter; a `FToITrap` that would trap is left
            // unfolded so it still traps). Otherwise it is emitted faithfully into the residual —
            // folded constants flow in as operands, dynamics pass through; this also covers v128
            // (SIMD), casts, and pointer ops, which aren't folded yet. Effectful / multi-result /
            // memory / call ops are not handled here and fall through to Unsupported.
            _ => {
                if let Some(k) = fold_float(inst, env) {
                    return Ok(Some(Abs::Const(k)));
                }
                let abs =
                    emit_residual_pure(inst, env, out, rnext).ok_or(SpecError::Unsupported)?;
                return Ok(Some(abs));
            }
        };
        Ok(Some(abs))
    }

    /// A load: fold from a renameable cell, fold from readonly data, or emit a residual load.
    #[allow(clippy::too_many_arguments)]
    fn eval_load(
        &self,
        op: LoadOp,
        addr: u32,
        offset: u64,
        align: u8,
        env: &[Abs],
        mem: &BTreeMap<u64, (u32, Abs)>,
        out: &mut Vec<Inst>,
        rnext: &mut u32,
    ) -> Result<Option<Abs>, SpecError> {
        let width = op.info().2 as u64;
        if let Abs::Const(Known::I64(base)) = env[addr as usize] {
            let base = base as u64;
            let eff = base.wrapping_add(offset);
            if within_region(self.config.rename, eff, width) {
                // The renameable region must be resolved entirely abstractly. Only integer loads
                // can be (the abstract domain tracks integer cells); a float load into it can't.
                if !matches!(op.info().1, ValType::I32 | ValType::I64) {
                    return Err(SpecError::Unsupported);
                }
                // An exact cell — same address *and* width — resolves directly. A constant cell's
                // raw bytes are re-extended per this load op (so an `i8` cell loaded `*_u`/`*_s`
                // zero-/sign-extends correctly); a dynamic cell is only renamed at its full natural
                // width, where loading it back is the identity (no residual fixup needed).
                if let Some(&(wc, val)) = mem.get(&eff) {
                    if wc as u64 == width {
                        return Ok(Some(match val {
                            Abs::Const(k) => {
                                let raw = known_raw(k, width);
                                Abs::Const(extend_loaded(raw, op).ok_or(SpecError::Unsupported)?)
                            }
                            Abs::Dyn(i) if is_full_natural_load(op, width) => Abs::Dyn(i),
                            Abs::Dyn(_) => return Err(SpecError::Unsupported),
                        }));
                    }
                }
                // Anything else touching the cell (a different-width or straddling access) can't be
                // resolved abstractly without composing bytes — refuse rather than guess.
                if mem
                    .iter()
                    .any(|(&b, &(wc, _))| b < eff + width && eff < b + wc as u64)
                {
                    return Err(SpecError::Unsupported);
                }
                // Untouched region cell ⇒ the zero-initialized backing, extended per the load.
                return Ok(Some(Abs::Const(
                    extend_loaded(0, op).ok_or(SpecError::Unsupported)?,
                )));
            }
            // Outside the region: a readonly constant-memory read folds; otherwise residual.
            if let Some(k) = read_const_mem(self.config, self.module, base, offset, op) {
                return Ok(Some(Abs::Const(k)));
            }
            let addr = materialize(env[addr as usize], out, rnext);
            out.push(Inst::Load {
                op,
                addr,
                offset,
                align,
            });
            return Ok(Some(Abs::Dyn(bump(rnext))));
        }
        // Dynamic address: with a region active it might alias the renamed stack, so refuse —
        // unless the caller has promised the region is private to the renamed accesses.
        if self.config.rename.is_some() && !self.config.rename_is_private {
            return Err(SpecError::Unsupported);
        }
        let addr = materialize(env[addr as usize], out, rnext);
        out.push(Inst::Load {
            op,
            addr,
            offset,
            align,
        });
        Ok(Some(Abs::Dyn(bump(rnext))))
    }

    /// A store: rename into the abstract region, or emit a residual store outside it.
    #[allow(clippy::too_many_arguments)]
    fn eval_store(
        &self,
        op: StoreOp,
        addr: u32,
        value: u32,
        offset: u64,
        align: u8,
        env: &[Abs],
        mem: &mut BTreeMap<u64, (u32, Abs)>,
        out: &mut Vec<Inst>,
        rnext: &mut u32,
    ) -> Result<Option<Abs>, SpecError> {
        let width = store_width(op) as u64;
        if let Abs::Const(Known::I64(base)) = env[addr as usize] {
            let base = base as u64;
            let eff = base.wrapping_add(offset);
            if within_region(self.config.rename, eff, width) {
                // Only integer stores can be renamed (the abstract domain tracks integer cells).
                if !matches!(op.info().1, ValType::I32 | ValType::I64) {
                    return Err(SpecError::Unsupported);
                }
                let cell = match env[value as usize] {
                    // A constant is truncated to the store width and kept as the cell's raw bytes,
                    // so a later load re-extends it correctly (an `i8` store of `0x1FF` ⇒ `0xFF`).
                    Abs::Const(k) => Abs::Const(cell_const(known_raw(k, width), width)),
                    // A dynamic value is only renamed when stored at its full natural width — then
                    // loading it back at that width is the identity. A *narrow* dynamic store would
                    // need residual masking to read back soundly, so refuse it instead.
                    Abs::Dyn(i) if is_full_natural_store(op, width) => Abs::Dyn(i),
                    Abs::Dyn(_) => return Err(SpecError::Unsupported),
                };
                // Invalidate any overlapping cell, then record this one. No residual store.
                mem.retain(|&b, &mut (wc, _)| !(b < eff + width && eff < b + wc as u64));
                mem.insert(eff, (width as u32, cell));
                return Ok(None);
            }
            if disjoint_from_region(self.config.rename, eff, width) {
                let addr = materialize(env[addr as usize], out, rnext);
                let value = materialize(env[value as usize], out, rnext);
                out.push(Inst::Store {
                    op,
                    addr,
                    value,
                    offset,
                    align,
                });
                return Ok(None);
            }
            return Err(SpecError::Unsupported); // straddles the region boundary
        }
        // Dynamic address: with a region active it might alias the renamed stack, so refuse —
        // unless the caller has promised the region is private to the renamed accesses.
        if self.config.rename.is_some() && !self.config.rename_is_private {
            return Err(SpecError::Unsupported);
        }
        let addr = materialize(env[addr as usize], out, rnext);
        let value = materialize(env[value as usize], out, rnext);
        out.push(Inst::Store {
            op,
            addr,
            value,
            offset,
            align,
        });
        Ok(None)
    }

    /// Evaluate the active frame's terminator, given the suspended caller frames (`outer`) and the
    /// active function. A branch stays within the active frame (replacing it with its target); a
    /// `return` pops the active frame and either ends the residual function or resumes the caller; a
    /// `return_call` is straight-line-inlined or, failing that, replaces the active frame (a tail
    /// call keeps the same return continuation).
    #[allow(clippy::too_many_arguments)]
    fn finish_term(
        &mut self,
        term: &Terminator,
        outer: Vec<FrameAbs>,
        func: u32,
        env: &[Abs],
        mem: &mut BTreeMap<u64, (u32, Abs)>,
        out: &mut Vec<Inst>,
        rnext: &mut u32,
        fuel: &mut usize,
    ) -> Result<Terminator, SpecError> {
        Ok(match term {
            Terminator::Return(vals) => {
                let results: Vec<Abs> = vals.iter().map(|&v| env[v as usize]).collect();
                self.return_from(outer, &results, mem, out, rnext)
            }
            Terminator::Br { target, args } => {
                let stack = succ_stack(&outer, func, *target, args, env);
                let (target, args) = self.branch_to(&stack, mem);
                Terminator::Br { target, args }
            }
            Terminator::BrIf {
                cond,
                then_blk,
                then_args,
                else_blk,
                else_args,
            } => match env[*cond as usize] {
                // Static condition: dispatch resolves to the one taken edge.
                Abs::Const(c) => {
                    let taken = c.as_i32().map(|c| c != 0).ok_or(SpecError::Unsupported)?;
                    let (blk, args) = if taken {
                        (*then_blk, then_args)
                    } else {
                        (*else_blk, else_args)
                    };
                    let stack = succ_stack(&outer, func, blk, args, env);
                    let (target, args) = self.branch_to(&stack, mem);
                    Terminator::Br { target, args }
                }
                // Dynamic condition: specialize both successors and keep the branch.
                Abs::Dyn(cond) => {
                    let then_stack = succ_stack(&outer, func, *then_blk, then_args, env);
                    let (then_blk, then_args) = self.branch_to(&then_stack, mem);
                    let else_stack = succ_stack(&outer, func, *else_blk, else_args, env);
                    let (else_blk, else_args) = self.branch_to(&else_stack, mem);
                    Terminator::BrIf {
                        cond,
                        then_blk,
                        then_args,
                        else_blk,
                        else_args,
                    }
                }
            },
            Terminator::BrTable {
                idx,
                targets,
                default,
            } => match env[*idx as usize] {
                Abs::Const(c) => {
                    let i = c.as_i32().ok_or(SpecError::Unsupported)? as u32 as usize;
                    let (blk, args) = targets.get(i).unwrap_or(default);
                    let stack = succ_stack(&outer, func, *blk, args, env);
                    let (target, args) = self.branch_to(&stack, mem);
                    Terminator::Br { target, args }
                }
                Abs::Dyn(idx) => {
                    let targets = targets
                        .iter()
                        .map(|(blk, args)| {
                            let stack = succ_stack(&outer, func, *blk, args, env);
                            self.branch_to(&stack, mem)
                        })
                        .collect();
                    let default_stack = succ_stack(&outer, func, default.0, &default.1, env);
                    let default = self.branch_to(&default_stack, mem);
                    Terminator::BrTable {
                        idx,
                        targets,
                        default,
                    }
                }
            },
            Terminator::Unreachable => Terminator::Unreachable,
            // A direct tail call: straight-line-inline if we can, else replace the active frame with
            // the callee, keeping this frame's return continuation (the suspended callers).
            Terminator::ReturnCall { func: callee, args } => {
                let args_abs: Vec<Abs> = args.iter().map(|&a| env[a as usize]).collect();
                match self.try_straightline(*callee, &args_abs, mem, out, rnext, fuel)? {
                    Some(results) => self.return_from(outer, &results, mem, out, rnext),
                    None => {
                        let mut stack = outer;
                        stack.push(FrameAbs {
                            func: *callee,
                            block: 0,
                            ip: 0,
                            env: args_abs,
                        });
                        let (target, args) = self.branch_to(&stack, mem);
                        Terminator::Br { target, args }
                    }
                }
            }
            Terminator::ReturnCallIndirect { .. } => return Err(SpecError::Unsupported),
        })
    }

    /// Return `results` from the active frame: end the residual function if no caller is suspended,
    /// otherwise resume the innermost caller — its env gains the call's results and it continues from
    /// the instruction after the call (a branch to that continuation context).
    fn return_from(
        &mut self,
        mut outer: Vec<FrameAbs>,
        results: &[Abs],
        mem: &BTreeMap<u64, (u32, Abs)>,
        out: &mut Vec<Inst>,
        rnext: &mut u32,
    ) -> Terminator {
        match outer.pop() {
            None => {
                let vals = results
                    .iter()
                    .map(|&a| materialize(a, out, rnext))
                    .collect();
                Terminator::Return(vals)
            }
            Some(mut caller) => {
                caller.env.extend_from_slice(results);
                outer.push(caller);
                let (target, args) = self.branch_to(&outer, mem);
                Terminator::Br { target, args }
            }
        }
    }

    /// Resolve one outgoing edge into a residual block id + dynamic arguments. The successor inherits
    /// the full call stack and the current abstract memory; constant lanes join the context, dynamic
    /// lanes are passed as residual block arguments in the canonical order (frames outermost→
    /// innermost, each frame's dynamic env slots in order, then dynamic memory cells by address —
    /// matching [`Self::build_block`]'s parameter declaration).
    fn branch_to(
        &mut self,
        stack: &[FrameAbs],
        mem: &BTreeMap<u64, (u32, Abs)>,
    ) -> (u32, Vec<u32>) {
        let mut frames = Vec::with_capacity(stack.len());
        let mut dyn_args = Vec::new();
        for fr in stack {
            let mut env = Vec::with_capacity(fr.env.len());
            for &a in &fr.env {
                match a {
                    Abs::Const(k) => env.push(Some(k)),
                    Abs::Dyn(i) => {
                        env.push(None);
                        dyn_args.push(i);
                    }
                }
            }
            frames.push(Frame {
                func: fr.func,
                block: fr.block,
                ip: fr.ip,
                env,
            });
        }
        let mut mem_pat = Vec::with_capacity(mem.len());
        for (&addr, &(width, val)) in mem.iter() {
            match val {
                Abs::Const(k) => mem_pat.push((addr, width, Some(k))),
                Abs::Dyn(i) => {
                    mem_pat.push((addr, width, None));
                    dyn_args.push(i);
                }
            }
        }
        let id = self.intern(frames, mem_pat);
        (id, dyn_args)
    }
}

/// Build the successor call stack for an intra-function branch: the suspended `outer` frames
/// unchanged, with a fresh active frame entering `block` of `func` whose env is the edge's `args`
/// mapped through the current `env`.
fn succ_stack(
    outer: &[FrameAbs],
    func: u32,
    block: u32,
    args: &[u32],
    env: &[Abs],
) -> Vec<FrameAbs> {
    let mut stack = outer.to_vec();
    let tenv = args.iter().map(|&a| env[a as usize]).collect();
    stack.push(FrameAbs {
        func,
        block,
        ip: 0,
        env: tenv,
    });
    stack
}

/// Fold a scalar float op whose operands are all compile-time constants, reusing the shared,
/// interpreter-exact fold helpers. Returns `None` if any operand is dynamic, the op isn't a scalar
/// float op, or folding it would trap (a `FToITrap` out of range) — in which case the caller emits
/// it residually so it computes/traps at run time exactly as the source would.
fn fold_float(inst: &Inst, env: &[Abs]) -> Option<Known> {
    let cst = |i: u32| match env[i as usize] {
        Abs::Const(k) => Some(k),
        Abs::Dyn(_) => None,
    };
    match *inst {
        Inst::FBin { ty, op, a, b } => crate::fold_fbin(ty, op, cst(a)?, cst(b)?),
        Inst::FUn { ty, op, a } => crate::fold_fun(ty, op, cst(a)?),
        Inst::FCmp { ty, op, a, b } => crate::fold_fcmp(ty, op, cst(a)?, cst(b)?),
        Inst::Fma { ty, a, b, c } => crate::fold_fma(ty, cst(a)?, cst(b)?, cst(c)?),
        Inst::FToISat { op, a } => crate::fold_ftoi_sat(op, cst(a)?),
        Inst::FToITrap { op, a } => crate::fold_ftoi_trap(op, cst(a)?),
        Inst::IToFConv { op, a } => crate::fold_itof(op, cst(a)?),
        Inst::Cast { op, a } => crate::fold_cast(op, cst(a)?),
        _ => None,
    }
}

/// Emit a pure, single-result value op faithfully into the residual: materialize each operand
/// (a constant becomes a `const`; a dynamic reuses its residual value), then clone the op with its
/// operands rewritten. Returns `None` for anything not a pure value op (memory / call / effectful /
/// multi-result), which the caller turns into [`SpecError::Unsupported`].
///
/// "Pure value op" reuses the optimizer's [`crate::is_removable_if_dead`] whitelist (all such ops
/// are single-result and side-effect-free), plus the trapping-but-deterministic float→int
/// conversion, which is safe to emit residually (it traps at run time exactly as the source would).
fn emit_residual_pure(
    inst: &Inst,
    env: &[Abs],
    out: &mut Vec<Inst>,
    rnext: &mut u32,
) -> Option<Abs> {
    if !(crate::is_removable_if_dead(inst) || matches!(inst, Inst::FToITrap { .. })) {
        return None;
    }
    let mut clone = inst.clone();
    crate::map_operands(&mut clone, &mut |old| {
        materialize(env[old as usize], out, rnext)
    });
    out.push(clone);
    Some(Abs::Dyn(bump(rnext)))
}

/// Turn an abstract value into a concrete residual SSA index, emitting a `const` for a constant.
fn materialize(abs: Abs, out: &mut Vec<Inst>, rnext: &mut u32) -> u32 {
    match abs {
        Abs::Dyn(i) => i,
        Abs::Const(k) => {
            out.push(k.to_const_inst());
            bump(rnext)
        }
    }
}

/// Take the next residual value index.
fn bump(rnext: &mut u32) -> u32 {
    let i = *rnext;
    *rnext += 1;
    i
}

/// The block-parameter type of a renameable memory cell of the given byte width.
fn cell_type(width: u32) -> ValType {
    match width {
        4 => ValType::I32,
        _ => ValType::I64,
    }
}

/// Whether `[eff, eff+width)` lies fully inside the renameable region.
fn within_region(region: Option<(u64, u64)>, eff: u64, width: u64) -> bool {
    match region {
        Some((lo, hi)) => eff >= lo && eff.checked_add(width).is_some_and(|end| end <= hi),
        None => false,
    }
}

/// Whether `[eff, eff+width)` is entirely outside the renameable region (vacuously true if none).
fn disjoint_from_region(region: Option<(u64, u64)>, eff: u64, width: u64) -> bool {
    match region {
        Some((lo, hi)) => eff
            .checked_add(width)
            .is_some_and(|end| end <= lo || eff >= hi),
        None => true,
    }
}

/// The byte width of a store op.
fn store_width(op: StoreOp) -> u32 {
    match op {
        StoreOp::I32 | StoreOp::F32 | StoreOp::I64_32 => 4,
        StoreOp::I64 | StoreOp::F64 => 8,
        StoreOp::I32_8 | StoreOp::I64_8 => 1,
        StoreOp::I32_16 | StoreOp::I64_16 => 2,
    }
}

/// The low `width` bytes, as the unsigned in-memory content (`width >= 8` ⇒ all bytes).
fn width_mask(width: u64) -> u64 {
    if width >= 8 {
        u64::MAX
    } else {
        (1u64 << (8 * width)) - 1
    }
}

/// The raw little-endian content (zero-extended) a constant cell of `width` bytes holds. Cells only
/// ever hold integer constants (a float store into the rename region bails), but a float's raw bits
/// *are* its memory content, so handle it the same way for totality.
fn known_raw(k: Known, width: u64) -> u64 {
    let v = match k {
        Known::I32(x) => x as u32 as u64,
        Known::I64(x) => x as u64,
        Known::F32(b) => b as u64,
        Known::F64(b) => b,
    };
    v & width_mask(width)
}

/// The canonical constant a renamed cell stores for `width` raw bytes: `i64` for a full 8-byte
/// cell, `i32` otherwise — matching the pre-existing full-width representation so memo contexts
/// (which key on the constant) stay canonical across widths.
fn cell_const(raw: u64, width: u64) -> Known {
    if width == 8 {
        Known::I64(raw as i64)
    } else {
        Known::I32(raw as u32 as i32)
    }
}

/// Whether a dynamic value stored by `op` occupies its full natural width — `i32.store`/`i64.store`.
/// Only then is renaming a dynamic cell sound without a residual fixup (the stored value's high bits
/// survive, so a same-width load reads it back unchanged).
fn is_full_natural_store(op: StoreOp, width: u64) -> bool {
    matches!((op, width), (StoreOp::I32, 4) | (StoreOp::I64, 8))
}

/// The load counterpart of [`is_full_natural_store`]: `i32.load`/`i64.load` read a full natural cell
/// back as the identity, so a dynamic cell may be returned directly.
fn is_full_natural_load(op: LoadOp, width: u64) -> bool {
    matches!((op, width), (LoadOp::I32, 4) | (LoadOp::I64, 8))
}

/// Apply load `op`'s width + sign/zero extension to the assembled little-endian content `raw`,
/// producing the loaded integer constant exactly as the interpreter would. Returns `None` for a
/// float load (the abstract domain tracks integer constants only).
fn extend_loaded(raw: u64, op: LoadOp) -> Option<Known> {
    let (_, vt, width, signed) = op.info();
    Some(match (vt, width, signed) {
        (ValType::I32, 1, false) => Known::I32(raw as u8 as i32),
        (ValType::I32, 1, true) => Known::I32(raw as u8 as i8 as i32),
        (ValType::I32, 2, false) => Known::I32(raw as u16 as i32),
        (ValType::I32, 2, true) => Known::I32(raw as u16 as i16 as i32),
        (ValType::I32, 4, _) => Known::I32(raw as u32 as i32),
        (ValType::I64, 1, false) => Known::I64(raw as u8 as i64),
        (ValType::I64, 1, true) => Known::I64(raw as u8 as i8 as i64),
        (ValType::I64, 2, false) => Known::I64(raw as u16 as i64),
        (ValType::I64, 2, true) => Known::I64(raw as u16 as i16 as i64),
        (ValType::I64, 4, false) => Known::I64(raw as u32 as i64),
        (ValType::I64, 4, true) => Known::I64(raw as u32 as i32 as i64),
        (ValType::I64, 8, _) => Known::I64(raw as i64),
        _ => return None,
    })
}

/// Read an integer load from constant memory. The effective address `base + offset` must lie
/// fully in range (so the interpreter would not fault) and resolve to bytes the caller has
/// promised constant — a `const_overlay`, a `const_region`, or (the default) a **readonly** data
/// segment. Returns the loaded value, sign/zero-extended per `op`, matching the interpreter's
/// little-endian load exactly. Returns `None` (⇒ emit a residual load) otherwise.
fn read_const_mem(
    config: &SpecConfig,
    module: &Module,
    base: u64,
    offset: u64,
    op: LoadOp,
) -> Option<Known> {
    let (_, vt, width, _) = op.info();
    if !matches!(vt, ValType::I32 | ValType::I64) {
        return None;
    }
    let mem = module.memory?;
    let eff = base.checked_add(offset)?;
    let end = eff.checked_add(width as u64)?;
    if end > mem.size() {
        return None; // could fault at the window top — let the residual load reproduce it
    }
    let bytes = const_bytes(config, module, eff, width)?;
    let mut raw: u64 = 0;
    for (i, &byte) in bytes.iter().enumerate() {
        raw |= (byte as u64) << (8 * i);
    }
    extend_loaded(raw, op)
}

/// Resolve `width` constant bytes at window address `eff`, if the caller has promised that range
/// constant. Precedence: an explicit overlay; then a caller `const_region` or a readonly data
/// segment, both read from the module's initial data image (uncovered bytes are zero).
fn const_bytes(config: &SpecConfig, module: &Module, eff: u64, width: u32) -> Option<Vec<u8>> {
    let width = width as u64;
    for (obase, bytes) in &config.const_overlays {
        if eff >= *obase {
            let rel = eff - *obase;
            if rel + width <= bytes.len() as u64 {
                let s = rel as usize;
                return Some(bytes[s..s + width as usize].to_vec());
            }
        }
    }
    let promised = config
        .const_regions
        .iter()
        .any(|&(lo, hi)| eff >= lo && eff + width <= hi)
        || module.data.iter().any(|d| {
            d.readonly && eff >= d.offset && eff + width <= d.offset + d.bytes.len() as u64
        });
    if !promised {
        return None;
    }
    Some((0..width).map(|i| image_byte(module, eff + i)).collect())
}

/// The byte at window address `addr` in the module's initial data image: the last data segment
/// covering it wins (segments are applied in order at instantiation), else the window is zero.
fn image_byte(module: &Module, addr: u64) -> u8 {
    let mut byte = 0u8;
    for d in &module.data {
        if addr >= d.offset && addr < d.offset + d.bytes.len() as u64 {
            byte = d.bytes[(addr - d.offset) as usize];
        }
    }
    byte
}
