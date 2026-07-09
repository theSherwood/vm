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
//! An **indirect** call (`call_indirect` / `return_call_indirect`, and `ref.func`) is inlined too
//! when its table index resolves to a **constant, in-range, signature-matching** function — the
//! module-0 table is the identity map, so a folded funcref dispatches deterministically to that
//! callee, which is then inlined like a direct call. A dynamic / out-of-range / mismatched index
//! can't be specialized (the single-function residual carries no table) and returns
//! [`SpecError::Unsupported`]. Host/capability calls are never inlined.
//!
//! **Outlining (residual-call mode).** With [`SpecConfig::outline_calls`] (and no rename region),
//! calls are *not* inlined: each `(callee, arg pattern)` is specialized to its own residual function
//! — memoized so call sites with the same static binding share one — and emitted as a residual
//! `call`, giving a **multi-function** residual. This bounds code growth and specializes
//! **dynamic-depth recursion** (a recursive callee with a dynamic argument becomes a finite
//! self-recursive residual where inlining would diverge). Constant arguments are baked in; the
//! dynamic ones are passed.
//!
//! **Scope.** Integer, **scalar float**, and **v128 (SIMD)** ops — arithmetic, compares, fused
//! multiply-add, float↔int conversions, reinterpret/demote/promote casts; and the SIMD lane ops —
//! splat / extract / replace, lane int+float arithmetic / compares / shifts, bitwise, shuffle,
//! swizzle, **and the exotic ones** (saturating add/sub, widen/narrow, lane convert, dot, pairwise,
//! pmin/pmax, avgr, popcnt, any/all-true, bitmask, q15) — are specialized (folded where the operands
//! are constant, bit-for-bit the interpreter). Remaining **pure, single-result** value ops (e.g.
//! pointer ops, and any lane op with a dynamic operand) are emitted faithfully into the residual, so
//! dispatch is still eliminated around them. Direct calls are inlined (above). Effectful,
//! multi-result, or other cross-function ops (indirect/host calls, atomics, fibers/threads), and
//! memory accesses the engine can't resolve, return [`SpecError::Unsupported`] rather than guessing.

use alloc::collections::{BTreeMap, VecDeque};
use alloc::vec; // the `vec!` macro
use alloc::vec::Vec;
use core::cell::RefCell;

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
#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct Frame {
    func: u32,
    block: u32,
    ip: usize,
    env: ParamPattern,
    /// The argument pattern this activation was *entered* with — its recursion signature. Used by
    /// selective outlining to recognize an unbounded-recursion back-edge (a call whose `(func, entry)`
    /// equals an ancestor activation's): bounded recursion has a different `entry` each level (a
    /// decreasing constant) and keeps inlining, unbounded recursion repeats its `entry` and is cut by
    /// outlining. **Empty outside selective mode**, so it doesn't change the memo key for the inline /
    /// full-outline paths.
    entry: ParamPattern,
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
    /// This activation's recursion signature — see [`Frame::entry`]. Empty outside selective mode.
    entry: ParamPattern,
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
    /// **Outline calls** instead of inlining them: a direct (or constant-index indirect) call is
    /// specialized to a *separate* residual function — memoized per `(callee, arg pattern)` so call
    /// sites with the same static binding share one — and emitted as a residual `call`, producing a
    /// **multi-function** residual. This bounds code growth (a callee specialized once, called N
    /// times) and specializes **dynamic-depth recursion** (a recursive callee with a dynamic
    /// argument becomes a finite self-recursive residual, where inlining would diverge). Composes with
    /// [`rename`](Self::rename): the renamed region's live abstract cells are threaded across each
    /// residual call boundary (passed in as extra arguments, returned as extra results), so the region
    /// stays in SSA exactly as across an inlined call. Off by default (the residual is a single
    /// inlined function).
    pub outline_calls: bool,
    /// **Selective outlining**: outline *only* the calls that need it for termination — an unbounded-
    /// recursion back-edge (a call re-entering an activation already on the stack with the same
    /// argument pattern) — and **inline everything else** (straight-line and bounded recursion via the
    /// usual CFG inlining). The residual is then a *tight* recursive function with its leaves and
    /// structure folded in, instead of one tiny function per call site (full [`outline_calls`]). Like
    /// `outline_calls` it composes with [`rename`](Self::rename) (region cells are threaded across the
    /// outlined back-edge) and implies outlining is enabled (no need to also set `outline_calls`). Off
    /// by default.
    pub selective_outline: bool,
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
/// `config`. Produces a module whose residual entry is function 0; the original memory and data
/// segments are carried through, so any residual loads still resolve. With
/// [`SpecConfig::outline_calls`] the residual is **multi-function** (the entry plus one specialized
/// function per outlined `(callee, arg pattern)`); otherwise it is a single inlined function.
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

    // The entry context: the constant valuation of each parameter (a static const, or `None` for a
    // dynamic value carried as a residual parameter).
    let entry_pattern: ParamPattern = args
        .iter()
        .map(|arg| match arg {
            SpecArg::ConstI32(v) => Some(Known::I32(*v)),
            SpecArg::ConstI64(v) => Some(Known::I64(*v)),
            SpecArg::Dynamic => None,
        })
        .collect();

    let has_memory = module.memory.is_some();
    let value_types: Vec<Vec<Vec<ValType>>> = module
        .funcs
        .iter()
        .map(|f| func_value_types(f, &module.funcs, has_memory))
        .collect();

    // When outlining, the renamed region's live abstract cells are threaded across each residual call
    // boundary (passed in as extra arguments, returned as extra results); see `outline_call`. The
    // single-function inline path keeps the region entirely internal (no threading).
    let funcs = if config.outline_calls || config.selective_outline {
        outline_funcs(module, config, &value_types, func, entry_pattern)?
    } else {
        vec![
            build_func(
                module,
                config,
                &value_types,
                None,
                func,
                &entry_pattern,
                &Vec::new(),
                false,
            )?
            .0,
        ]
    };

    Ok(Module {
        funcs,
        memory: module.memory,
        data: module.data.clone(),
        imports: vec![],
        // The residual's functions are freshly built (specialized/renumbered), so the source
        // module's name→funcidx exports no longer apply; a residual is addressed by index.
        exports: vec![],
        debug_info: None,
    })
}

/// Build one residual function for `(callee, pattern)`: a fresh [`Spec`] symbolically executes the
/// callee from its entry, with `outline` either `None` (inline every call into this one function) or
/// `Some` (outline calls into shared residual functions via the shared state). The residual's
/// parameters are the dynamic entries of `pattern`, in order; its results match the callee's.
#[allow(clippy::too_many_arguments)]
fn build_func(
    module: &Module,
    config: &SpecConfig,
    value_types: &[Vec<Vec<ValType>>],
    outline: Option<&RefCell<OutlineState>>,
    callee: u32,
    pattern: &ParamPattern,
    mem_pat: &MemPattern,
    thread_cells: bool,
) -> Result<(Func, CellSig), SpecError> {
    let cf = module
        .funcs
        .get(callee as usize)
        .ok_or(SpecError::BadFunc)?;
    // Residual params: the dynamic call arguments, then the dynamic threaded region cells (by
    // address) — matching `build_block`'s entry-block parameter order (frame env, then memory cells).
    let mut residual_params: Vec<ValType> = pattern
        .iter()
        .zip(&cf.params)
        .filter_map(|(slot, ty)| slot.is_none().then_some(*ty))
        .collect();
    for &(_, width, slot) in mem_pat {
        if slot.is_none() {
            residual_params.push(cell_type(width));
        }
    }

    // Selective outlining only makes sense when outlining is enabled; when it is, populate the
    // per-frame recursion signatures (otherwise they stay empty, leaving the memo key unchanged).
    let selective = outline.is_some() && config.selective_outline;
    let mut spec = Spec {
        module,
        config,
        value_types,
        outline,
        selective,
        thread_cells,
        out_cells: None,
        memo: BTreeMap::new(),
        queue: VecDeque::new(),
        next_id: 0,
    };
    let entry = if selective {
        pattern.clone()
    } else {
        Vec::new()
    };
    spec.intern(
        vec![Frame {
            func: callee,
            block: 0,
            ip: 0,
            env: pattern.clone(),
            entry,
        }],
        mem_pat.clone(),
    );

    let mut blocks = Vec::new();
    while let Some(task) = spec.queue.pop_front() {
        if blocks.len() >= DEFAULT_BUDGET {
            return Err(SpecError::Budget);
        }
        blocks.push(spec.build_block(task)?);
    }
    // The threaded region cells flow back as extra results (after the callee's own), in the address
    // order fixed at the first `return` (see `return_from`); empty when nothing is threaded.
    let out_cells = spec.out_cells.unwrap_or_default();
    let mut results = cf.results.clone();
    for &(_, width) in &out_cells {
        results.push(cell_type(width));
    }
    Ok((
        Func {
            params: residual_params,
            results,
            blocks,
        },
        out_cells,
    ))
}

/// The outlining driver: polyvariant interprocedural specialization. A shared memo maps each
/// `(callee, arg pattern, incoming region cells)` to a residual function index. Functions are built
/// **eagerly, depth-first** ([`request_outline`]): a callee is built the first time it is referenced
/// — so its threaded region-cell *out* signature is known before the caller emits the `call` — and an
/// index is reserved up front so a recursion back-edge resolves to it mid-build.
fn outline_funcs(
    module: &Module,
    config: &SpecConfig,
    value_types: &[Vec<Vec<ValType>>],
    entry: u32,
    entry_pattern: ParamPattern,
) -> Result<Vec<Func>, SpecError> {
    let state = RefCell::new(OutlineState {
        memo: BTreeMap::new(),
        funcs: Vec::new(),
    });
    // The entry is residual function 0. It does **not** thread region cells in/out: the rename region
    // is private scratch, established fresh (zero) on entry and discarded on return, so the residual
    // entry's signature is just the source function's.
    {
        let mut s = state.borrow_mut();
        s.memo.insert(
            (entry, entry_pattern.clone(), Vec::new()),
            (0, Some(Vec::new())),
        );
        s.funcs.push(None);
    }
    let (entry_func, _) = build_func(
        module,
        config,
        value_types,
        Some(&state),
        entry,
        &entry_pattern,
        &Vec::new(),
        false,
    )?;
    state.borrow_mut().funcs[0] = Some(entry_func);

    // Everything reachable was built eagerly during the entry build.
    Ok(state
        .into_inner()
        .funcs
        .into_iter()
        .map(|f| f.expect("every reserved outline slot is filled"))
        .collect())
}

/// Shared state for outlining: `(callee, arg pattern, incoming region cells) → (residual index,
/// out-cell signature)`, plus the reserved function slots (filled as builds complete). The out-cell
/// signature is `None` while a function is *in progress* (so a recursion back-edge can detect the
/// cycle). Lives behind a `RefCell` so the (`&self`) block executor can mint a residual callee while
/// emitting the `call`.
type OutlineKey = (u32, ParamPattern, MemPattern);
/// A threaded region's live-cell signature: `(address, width)` in address order.
type CellSig = Vec<(u64, u32)>;
struct OutlineState {
    memo: BTreeMap<OutlineKey, (u32, Option<CellSig>)>,
    funcs: Vec<Option<Func>>,
}

/// Get the residual function index and threaded out-cell signature for `(callee, arg_pat, mem_pat)`,
/// building it eagerly the first time it is seen. A reference to an *in-progress* function (a
/// recursion back-edge) resolves to its reserved index; this is sound only when no region cells are
/// threaded (the out signature is then empty and known) — recursion through a rename region is
/// rejected, since the live-cell set grows per level and can't be cut into a fixed signature.
fn request_outline(
    module: &Module,
    config: &SpecConfig,
    value_types: &[Vec<Vec<ValType>>],
    state: &RefCell<OutlineState>,
    callee: u32,
    arg_pat: ParamPattern,
    mem_pat: MemPattern,
) -> Result<(u32, CellSig), SpecError> {
    let key = (callee, arg_pat, mem_pat);
    let idx = {
        let mut s = state.borrow_mut();
        if let Some((idx, sig)) = s.memo.get(&key) {
            return match sig {
                Some(sig) => Ok((*idx, sig.clone())),
                // In progress: a recursion back-edge. Only resolvable with no threaded cells.
                None if key.2.is_empty() => Ok((*idx, Vec::new())),
                None => Err(SpecError::Unsupported),
            };
        }
        if s.funcs.len() >= DEFAULT_BUDGET {
            return Err(SpecError::Budget);
        }
        let idx = s.funcs.len() as u32;
        s.funcs.push(None);
        s.memo.insert(key.clone(), (idx, None));
        idx
    };
    // Build outside the borrow (the build re-enters `request_outline` for nested calls).
    let (func, sig) = build_func(
        module,
        config,
        value_types,
        Some(state),
        callee,
        &key.1,
        &key.2,
        true,
    )?;
    let mut s = state.borrow_mut();
    s.funcs[idx as usize] = Some(func);
    s.memo.get_mut(&key).expect("reserved").1 = Some(sig.clone());
    Ok((idx, sig))
}

/// A call's specialization pattern: the constant arguments baked (`Some`), the dynamic ones marked
/// `None`. This is both the outlining key's pattern and a frame's recursion signature.
fn arg_pattern(args_abs: &[Abs]) -> ParamPattern {
    args_abs
        .iter()
        .map(|a| match a {
            Abs::Const(k) => Some(*k),
            Abs::Dyn(_) => None,
        })
        .collect()
}

/// The dynamic operands of an abstract argument list (the residual `call`'s value arguments).
fn dyn_args(args_abs: &[Abs]) -> Vec<u32> {
    args_abs
        .iter()
        .filter_map(|a| match a {
            Abs::Dyn(i) => Some(*i),
            Abs::Const(_) => None,
        })
        .collect()
}

struct Spec<'a> {
    module: &'a Module,
    config: &'a SpecConfig,
    /// Per-function, per-block, per-value source types (`value_types[func][block][value_idx]`) —
    /// used to type the SSA values threaded into a residual block as block parameters.
    value_types: &'a [Vec<Vec<ValType>>],
    /// `Some` ⇒ outline calls into shared residual functions via this state; `None` ⇒ inline them.
    outline: Option<&'a RefCell<OutlineState>>,
    /// Selective outlining: inline calls, outlining only unbounded-recursion back-edges. Implies
    /// `outline.is_some()`; when set, frames carry their recursion signature ([`Frame::entry`]).
    selective: bool,
    /// Whether this residual function threads the renamed region's live cells across its boundary:
    /// the incoming cells are extra parameters and the live cells flow back as extra results. `false`
    /// for the entry function and the single-function inline path (the region is internal there).
    thread_cells: bool,
    /// The threaded out-cell signature (`(addr, width)` by address), fixed at the first `return` and
    /// required to match at every other return. `None` until the first return; stays `None` when
    /// nothing is threaded.
    out_cells: Option<CellSig>,
    /// `(call stack, memory pattern) → residual block id`. The memo that makes the loop terminate
    /// and that closes residual loops.
    memo: BTreeMap<(Vec<Frame>, MemPattern), u32>,
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

    /// The recursion signature to stamp on a freshly-entered activation: the call's argument pattern
    /// in selective mode, empty otherwise (so non-selective memo keys are unchanged).
    fn entry_sig(&self, args: &[Abs]) -> ParamPattern {
        if self.selective {
            arg_pattern(args)
        } else {
            Vec::new()
        }
    }

    /// Whether a call to `(callee, pattern)` is an **unbounded-recursion back-edge**: in selective
    /// mode, the same `(func, entry)` is already live on the call stack (the active activation or a
    /// suspended ancestor). Such a call must outline to terminate; everything else inlines. Bounded
    /// recursion has a *different* `entry` each level (a decreasing constant), so it never matches and
    /// keeps unrolling.
    fn is_recursion(
        &self,
        callee: u32,
        pattern: &ParamPattern,
        ancestors: &[FrameAbs],
        active: (u32, &ParamPattern),
    ) -> bool {
        self.selective
            && ((active.0 == callee && active.1 == pattern)
                || ancestors
                    .iter()
                    .any(|f| f.func == callee && &f.entry == pattern))
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
                entry: fr.entry.clone(),
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
        let FrameAbs {
            func: active_func,
            block: active_block,
            ip: active_ip,
            mut env,
            entry: active_entry,
        } = frames.pop().expect("a context has at least one frame");
        let src = &module.funcs[active_func as usize].blocks[active_block as usize];
        let mut out: Vec<Inst> = Vec::new();
        let mut fuel = INLINE_FUEL;
        // `frames` is now exactly the suspended ancestors; together with `(active_func, active_entry)`
        // it is the call stack selective outlining checks for a recursion back-edge.
        let exec = self.exec_insts(
            &src.insts,
            active_ip,
            &mut env,
            &mut mem,
            &mut out,
            &mut rnext,
            &mut fuel,
            &frames,
            active_func,
            &active_entry,
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
                let callee_entry = self.entry_sig(&args);
                frames.push(FrameAbs {
                    func: active_func,
                    block: active_block,
                    ip: resume_ip,
                    env,
                    entry: active_entry,
                });
                frames.push(FrameAbs {
                    func: callee,
                    block: 0,
                    ip: 0,
                    env: args,
                    entry: callee_entry,
                });
                let (target, args) = self.branch_to(&frames, &mem);
                Terminator::Br { target, args }
            }
            Exec::Done => self.finish_term(
                &src.term,
                frames,
                active_func,
                &active_entry,
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
        // The current call stack, for selective outlining's recursion check: the suspended ancestors
        // plus the active activation's `(func, entry)`.
        ancestors: &[FrameAbs],
        active_func: u32,
        active_entry: &ParamPattern,
    ) -> Result<Exec, SpecError> {
        for (k, inst) in insts.iter().enumerate().skip(start_ip) {
            if let Some((callee, args_abs)) = self.callee_of(inst, env)? {
                match self.outline {
                    // Full outline: every call becomes a residual call to the shared specialized callee.
                    Some(state) if !self.selective => {
                        let results =
                            self.outline_call(state, callee, &args_abs, mem, out, rnext)?;
                        env.extend(results);
                    }
                    // Selective: inline if we can (straight-line / bounded recursion); on dynamic
                    // control flow, outline only a recursion back-edge, else fall back to CFG inlining.
                    Some(state) => {
                        match self.try_straightline(callee, &args_abs, mem, out, rnext, fuel)? {
                            Some(results) => env.extend(results),
                            None => {
                                let pat = arg_pattern(&args_abs);
                                if self.is_recursion(
                                    callee,
                                    &pat,
                                    ancestors,
                                    (active_func, active_entry),
                                ) {
                                    let results = self
                                        .outline_call(state, callee, &args_abs, mem, out, rnext)?;
                                    env.extend(results);
                                } else {
                                    return Ok(Exec::Suspend {
                                        callee,
                                        args: args_abs,
                                        resume_ip: k + 1,
                                    });
                                }
                            }
                        }
                    }
                    // Inline mode: straight-line if we can, else CFG inlining.
                    None => {
                        match self.try_straightline(callee, &args_abs, mem, out, rnext, fuel)? {
                            Some(results) => env.extend(results),
                            None => {
                                return Ok(Exec::Suspend {
                                    callee,
                                    args: args_abs,
                                    resume_ip: k + 1,
                                })
                            }
                        }
                    }
                }
            } else if let Some(res) = self.eval_inst(inst, env, mem, out, rnext)? {
                env.push(res);
            }
        }
        Ok(Exec::Done)
    }

    /// Emit a residual `call` to the specialized callee for `(callee, arg pattern, region cells)` and
    /// return its results as fresh residual values. Constant arguments are baked into the callee (so
    /// call sites with the same static binding share it); the dynamic arguments are passed, in order.
    ///
    /// **Renamed region threading.** When a rename region is active, the caller's live abstract cells
    /// (`mem`) cross the call boundary as data: the constant cells are baked into the callee's key, the
    /// dynamic ones are appended to the call arguments, and the callee's live-out cells come back as
    /// extra results — with which `mem` is rebuilt. So the operand stack stays in SSA across the call
    /// (never spilled to the window), exactly as it is across an inlined call.
    fn outline_call(
        &self,
        state: &RefCell<OutlineState>,
        callee: u32,
        args_abs: &[Abs],
        mem: &mut BTreeMap<u64, (u32, Abs)>,
        out: &mut Vec<Inst>,
        rnext: &mut u32,
    ) -> Result<Vec<Abs>, SpecError> {
        let arg_pat = arg_pattern(args_abs);
        let mut args = dyn_args(args_abs);
        // Thread the whole current abstract memory: constants into the key, dynamics by value.
        let mut mem_pat: MemPattern = Vec::with_capacity(mem.len());
        for (&addr, &(width, val)) in mem.iter() {
            match val {
                Abs::Const(k) => mem_pat.push((addr, width, Some(k))),
                Abs::Dyn(i) => {
                    mem_pat.push((addr, width, None));
                    args.push(i);
                }
            }
        }
        let (ridx, out_sig) = request_outline(
            self.module,
            self.config,
            self.value_types,
            state,
            callee,
            arg_pat,
            mem_pat,
        )?;
        out.push(Inst::Call { func: ridx, args });
        let nres = self.module.funcs[callee as usize].results.len();
        let results: Vec<Abs> = (0..nres).map(|_| Abs::Dyn(bump(rnext))).collect();
        // The live-out cells are the call's trailing results; rebuild the abstract memory from them.
        mem.clear();
        for (addr, width) in out_sig {
            mem.insert(addr, (width, Abs::Dyn(bump(rnext))));
        }
        Ok(results)
    }

    /// If `inst` is an inlinable call — a direct [`Inst::Call`], or an [`Inst::CallIndirect`] whose
    /// table index resolves to a constant in-range, signature-matching function — return the concrete
    /// callee index and its argument values. `Ok(None)` for a non-call. A `CallIndirect` whose index
    /// is dynamic / out of range / mismatched can't be specialized (the single-function residual has
    /// no table to dispatch through), so it surfaces as [`SpecError::Unsupported`].
    fn callee_of(&self, inst: &Inst, env: &[Abs]) -> Result<Option<(u32, Vec<Abs>)>, SpecError> {
        Ok(match inst {
            Inst::Call { func, args } => {
                Some((*func, args.iter().map(|&a| env[a as usize]).collect()))
            }
            Inst::CallIndirect { ty, idx, args } => {
                let callee = self
                    .resolve_indirect(ty, env[*idx as usize])
                    .ok_or(SpecError::Unsupported)?;
                Some((callee, args.iter().map(|&a| env[a as usize]).collect()))
            }
            _ => None,
        })
    }

    /// Resolve a `call_indirect` table index to a concrete function, or `None` if it can't be pinned
    /// at specialization time. The module-0 function table is the identity map (slot `i` → func `i`)
    /// padded with empty slots, and for any in-range index the table-size mask is a no-op — so a
    /// **constant, in-range, signature-matching** index dispatches deterministically to `funcs[idx]`
    /// on every backend. A dynamic index, an out-of-range index, or a signature mismatch returns
    /// `None` (the call can't be specialized — the runtime would dispatch or trap through a table the
    /// residual doesn't carry).
    fn resolve_indirect(&self, ty: &svm_ir::FuncType, idx: Abs) -> Option<u32> {
        let i = match idx {
            Abs::Const(k) => k.as_i32()?,
            Abs::Dyn(_) => return None,
        };
        let u = i as u32 as usize;
        let f = self.module.funcs.get(u)?;
        (f.params == ty.params && f.results == ty.results).then_some(u as u32)
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
                // path can emit one — so hand off.
                Terminator::Unreachable => return Err(InlineErr::NeedsCfg),
                // An indirect tail call whose index resolves to a constant callee is itself inlined.
                Terminator::ReturnCallIndirect { ty, idx, args } => {
                    let callee = self
                        .resolve_indirect(ty, genv[*idx as usize])
                        .ok_or(InlineErr::Spec(SpecError::Unsupported))?;
                    let a: Vec<Abs> = args.iter().map(|&x| genv[x as usize]).collect();
                    return self.inline_call(callee, &a, mem, out, rnext, fuel);
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
            if let Some((callee, a)) = self.callee_of(inst, env).map_err(InlineErr::Spec)? {
                let results = self.inline_call(callee, &a, mem, out, rnext, fuel)?;
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
            Inst::ConstV128(b) => Abs::Const(Known::V128(b)),
            // `ref.func` is the function index as a plain `i32` (a funcref is forgeable data, §3c).
            // Folding it to a constant lets a downstream `call_indirect` resolve its callee.
            Inst::RefFunc { func } => Abs::Const(Known::I32(func as i32)),

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

            // Any other pure, single-result value op. A scalar **float** or **v128 (SIMD)** op with
            // all-constant operands folds (bit-for-bit the interpreter; a `FToITrap` that would trap
            // is left unfolded so it still traps). Otherwise it is emitted faithfully into the
            // residual — folded constants flow in as operands, dynamics pass through; this also
            // covers the not-yet-folded SIMD ops, casts, and pointer ops. Effectful / multi-result /
            // memory / call ops are not handled here and fall through to Unsupported.
            _ => {
                let fold =
                    fold_float(inst, env).or_else(|| crate::fold_simd(inst, |i| cst(env, i)));
                if let Some(k) = fold {
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
        active_entry: &ParamPattern,
        env: &[Abs],
        mem: &mut BTreeMap<u64, (u32, Abs)>,
        out: &mut Vec<Inst>,
        rnext: &mut u32,
        fuel: &mut usize,
    ) -> Result<Terminator, SpecError> {
        Ok(match term {
            Terminator::Return(vals) => {
                let results: Vec<Abs> = vals.iter().map(|&v| env[v as usize]).collect();
                self.return_from(outer, &results, mem, out, rnext)?
            }
            Terminator::Br { target, args } => {
                let stack = succ_stack(&outer, func, *target, args, env, active_entry.clone());
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
                    let stack = succ_stack(&outer, func, blk, args, env, active_entry.clone());
                    let (target, args) = self.branch_to(&stack, mem);
                    Terminator::Br { target, args }
                }
                // Dynamic condition: specialize both successors and keep the branch.
                Abs::Dyn(cond) => {
                    let then_stack = succ_stack(
                        &outer,
                        func,
                        *then_blk,
                        then_args,
                        env,
                        active_entry.clone(),
                    );
                    let (then_blk, then_args) = self.branch_to(&then_stack, mem);
                    let else_stack = succ_stack(
                        &outer,
                        func,
                        *else_blk,
                        else_args,
                        env,
                        active_entry.clone(),
                    );
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
                    let stack = succ_stack(&outer, func, *blk, args, env, active_entry.clone());
                    let (target, args) = self.branch_to(&stack, mem);
                    Terminator::Br { target, args }
                }
                Abs::Dyn(idx) => {
                    let targets = targets
                        .iter()
                        .map(|(blk, args)| {
                            let stack =
                                succ_stack(&outer, func, *blk, args, env, active_entry.clone());
                            self.branch_to(&stack, mem)
                        })
                        .collect();
                    let default_stack = succ_stack(
                        &outer,
                        func,
                        default.0,
                        &default.1,
                        env,
                        active_entry.clone(),
                    );
                    let default = self.branch_to(&default_stack, mem);
                    Terminator::BrTable {
                        idx,
                        targets,
                        default,
                    }
                }
            },
            Terminator::Unreachable => Terminator::Unreachable,
            // A direct tail call.
            Terminator::ReturnCall { func: callee, args } => {
                let args_abs: Vec<Abs> = args.iter().map(|&a| env[a as usize]).collect();
                self.tail_call(
                    *callee,
                    args_abs,
                    outer,
                    func,
                    active_entry,
                    mem,
                    out,
                    rnext,
                    fuel,
                )?
            }
            // An indirect tail call whose index resolves to a constant callee.
            Terminator::ReturnCallIndirect { ty, idx, args } => {
                let callee = self
                    .resolve_indirect(ty, env[*idx as usize])
                    .ok_or(SpecError::Unsupported)?;
                let args_abs: Vec<Abs> = args.iter().map(|&a| env[a as usize]).collect();
                self.tail_call(
                    callee,
                    args_abs,
                    outer,
                    func,
                    active_entry,
                    mem,
                    out,
                    rnext,
                    fuel,
                )?
            }
        })
    }

    /// Specialize a tail call to `callee` (a `return_call`). In full-outline mode it becomes a
    /// residual `return_call` to the shared specialized callee. In selective mode it inlines unless it
    /// is a recursion back-edge (then the residual `return_call`). Otherwise (inline mode) it is
    /// straight-line-inlined or, failing that, replaces the active frame (a tail call keeps this
    /// frame's return continuation). `active_func`/`active_entry` identify the activation being
    /// replaced, for the recursion check.
    #[allow(clippy::too_many_arguments)]
    fn tail_call(
        &mut self,
        callee: u32,
        args_abs: Vec<Abs>,
        outer: Vec<FrameAbs>,
        active_func: u32,
        active_entry: &ParamPattern,
        mem: &mut BTreeMap<u64, (u32, Abs)>,
        out: &mut Vec<Inst>,
        rnext: &mut u32,
        fuel: &mut usize,
    ) -> Result<Terminator, SpecError> {
        if let Some(state) = self.outline {
            // Threading the renamed region across a *tail* call (where the callee's results become
            // this function's results) isn't supported — there's no return point to append this
            // function's own out-cells at. Fail closed; a rename region forces it.
            let outline_tail = |args_abs: &[Abs]| -> Result<Terminator, SpecError> {
                if self.config.rename.is_some() {
                    return Err(SpecError::Unsupported);
                }
                let (ridx, _) = request_outline(
                    self.module,
                    self.config,
                    self.value_types,
                    state,
                    callee,
                    arg_pattern(args_abs),
                    Vec::new(),
                )?;
                Ok(Terminator::ReturnCall {
                    func: ridx,
                    args: dyn_args(args_abs),
                })
            };
            if !self.selective {
                return outline_tail(&args_abs);
            }
            // Selective: try to inline; outline only a recursion back-edge.
            if let Some(results) =
                self.try_straightline(callee, &args_abs, mem, out, rnext, fuel)?
            {
                return self.return_from(outer, &results, mem, out, rnext);
            }
            let pat = arg_pattern(&args_abs);
            if self.is_recursion(callee, &pat, &outer, (active_func, active_entry)) {
                return outline_tail(&args_abs);
            }
            let mut stack = outer;
            stack.push(FrameAbs {
                func: callee,
                block: 0,
                ip: 0,
                env: args_abs,
                entry: pat,
            });
            let (target, args) = self.branch_to(&stack, mem);
            return Ok(Terminator::Br { target, args });
        }
        match self.try_straightline(callee, &args_abs, mem, out, rnext, fuel)? {
            Some(results) => self.return_from(outer, &results, mem, out, rnext),
            None => {
                let mut stack = outer;
                stack.push(FrameAbs {
                    func: callee,
                    block: 0,
                    ip: 0,
                    env: args_abs,
                    entry: Vec::new(),
                });
                let (target, args) = self.branch_to(&stack, mem);
                Ok(Terminator::Br { target, args })
            }
        }
    }

    /// Return `results` from the active frame: end the residual function if no caller is suspended,
    /// otherwise resume the innermost caller — its env gains the call's results and it continues from
    /// the instruction after the call (a branch to that continuation context).
    ///
    /// When this function threads region cells ([`Spec::thread_cells`]), the live cells flow out as
    /// extra return values, after the function's own results. The cell set (`(addr, width)` by
    /// address) is fixed at the first return and must match at every other — a function whose returns
    /// leave the renamed region in different shapes can't be given one residual signature, so it fails
    /// closed.
    fn return_from(
        &mut self,
        mut outer: Vec<FrameAbs>,
        results: &[Abs],
        mem: &BTreeMap<u64, (u32, Abs)>,
        out: &mut Vec<Inst>,
        rnext: &mut u32,
    ) -> Result<Terminator, SpecError> {
        Ok(match outer.pop() {
            None => {
                let mut vals: Vec<u32> = results
                    .iter()
                    .map(|&a| materialize(a, out, rnext))
                    .collect();
                if self.thread_cells {
                    let sig: Vec<(u64, u32)> = mem.iter().map(|(&a, &(w, _))| (a, w)).collect();
                    match &self.out_cells {
                        Some(prev) if *prev != sig => return Err(SpecError::Unsupported),
                        _ => self.out_cells = Some(sig),
                    }
                    for &(_, val) in mem.values() {
                        vals.push(materialize(val, out, rnext));
                    }
                }
                Terminator::Return(vals)
            }
            Some(mut caller) => {
                caller.env.extend_from_slice(results);
                outer.push(caller);
                let (target, args) = self.branch_to(&outer, mem);
                Terminator::Br { target, args }
            }
        })
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
                entry: fr.entry.clone(),
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
/// mapped through the current `env`. The branch stays inside the *same* activation, so it carries the
/// active frame's recursion signature (`entry`) forward unchanged.
fn succ_stack(
    outer: &[FrameAbs],
    func: u32,
    block: u32,
    args: &[u32],
    env: &[Abs],
    entry: ParamPattern,
) -> Vec<FrameAbs> {
    let mut stack = outer.to_vec();
    let tenv = args.iter().map(|&a| env[a as usize]).collect();
    stack.push(FrameAbs {
        func,
        block,
        ip: 0,
        env: tenv,
        entry,
    });
    stack
}

/// An operand's compile-time constant, if it has one (a dynamic value has none).
fn cst(env: &[Abs], i: u32) -> Option<Known> {
    match env[i as usize] {
        Abs::Const(k) => Some(k),
        Abs::Dyn(_) => None,
    }
}

/// Fold a scalar float op whose operands are all compile-time constants, reusing the shared,
/// interpreter-exact fold helpers. Returns `None` if any operand is dynamic, the op isn't a scalar
/// float op, or folding it would trap (a `FToITrap` out of range) — in which case the caller emits
/// it residually so it computes/traps at run time exactly as the source would.
fn fold_float(inst: &Inst, env: &[Abs]) -> Option<Known> {
    let cst = |i: u32| cst(env, i);
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
        // A v128 never reaches a renamed cell (a v128 store into the region bails); take its low 8
        // bytes for totality.
        Known::V128(b) => u64::from_le_bytes(b[..8].try_into().unwrap()),
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
