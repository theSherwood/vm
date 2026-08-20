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

use alloc::collections::{BTreeMap, BTreeSet, VecDeque};
use alloc::vec; // the `vec!` macro
use alloc::vec::Vec;
use core::cell::RefCell;

use svm_ir::{
    BinOp, Block, CastOp, ConvOp, Func, Inst, IntTy, LoadOp, Module, StoreOp, Terminator, ValType,
};
use svm_verify::func_value_types;

use crate::{fold_int_bin, fold_int_cmp, fold_int_un, Known};

/// Diagnostic: name the op/context that forced a [`SpecError::Unsupported`]. A no-op unless the
/// `trace` feature is on (which pulls in `std` for `eprintln`), so it never touches the `no_std`
/// build. Used to find the first blocker when specializing a large real function (e.g. `luaV_execute`).
#[cfg(feature = "trace")]
macro_rules! trace_unsup {
    ($($t:tt)*) => {{ extern crate std; std::eprintln!($($t)*); }};
}
#[cfg(not(feature = "trace"))]
macro_rules! trace_unsup {
    ($($t:tt)*) => {{}};
}

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
    /// The **entry seed** flag for deopt argument threading: `true` only for the initial entry-block
    /// context, where the threaded entry arguments are the entry frame's own parameters (no extra
    /// block params); `false` everywhere else, where they are reconstructed as extra block parameters.
    /// Always `false` unless the deopt handler takes the entry's arguments, so it leaves every other
    /// residual's memo partition unchanged.
    seed: bool,
    /// **Edge deopt** (see [`SpecConfig::deopt_edges`]): this context was reached by a branch edge the
    /// caller marked cold, so instead of *projecting* the target block (which would diverge — e.g. a
    /// dynamic-callee re-dispatch), [`Spec::build_block`] spills the live rename cells and tail-calls
    /// the deopt handler, exactly like a block-level [`SpecConfig::deopt_targets`] bail. Part of the
    /// memo key so a deopt-edge context is distinct from a normal one reaching the same block/state.
    deopt: bool,
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

/// How a cut call treats the live rename cells across the opaque boundary (see [`Spec::emit_cut_call`]).
#[derive(Clone, Copy, PartialEq, Eq)]
enum CutMode {
    /// [`SpecConfig::cut_calls`]: the callee touches no renamed state — cells preserved, no spill.
    Opaque,
    /// [`SpecConfig::cut_calls_read_state`]: the callee reads the state — spill (any width), no reload.
    ReadState,
    /// [`SpecConfig::cut_calls_touch_state`]: the callee reads/writes the state — spill then reload.
    TouchState,
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
    /// Treat the [`rename`](Self::rename) region as **live memory backed by the constant image**
    /// rather than zero-initialized scratch: an untouched renamed cell reads its **seed** from the
    /// constant-memory sources (an overlay, a [`const_regions`](Self::const_regions) promise, or a
    /// readonly data segment) instead of reading zero. This is what lets the specializer SSA-lift —
    /// and then mutate — real interpreter state captured from a live run (a `lua_State`'s stack
    /// pointers, a `CallInfo`, the register file), so the guards those fields drive fold while writes
    /// still update the abstract cell. Off by default (the region stays zero-init scratch), so
    /// existing renames — whose regions never overlap a constant source — are unaffected.
    ///
    /// **Correctness contract:** seeding makes the region *alias real memory*. If any renamed cell is
    /// **written**, the residual elides that store, so the window is left stale after the call — sound
    /// only when that memory is dead once the residual returns. When the memory persists (its final
    /// value is read by the caller), pair this with write-back so the live cells are spilled on exit.
    pub rename_seed_from_image: bool,
    /// **Additional** rename regions beyond [`rename`](Self::rename), with identical semantics (same
    /// `rename_is_private` / `rename_seed_from_image` treatment). A real interpreter's mutable state is
    /// several disjoint objects — a `lua_State`, its stack array, the current `CallInfo` — at unrelated
    /// window addresses, so one contiguous range can't cover them without also sweeping in the
    /// allocations between (a `global_State`, other frames). Each `[lo, hi)` here is SSA-lifted like
    /// `rename`; an access is "in the region set" if it lies fully within *any* one, and disjoint only
    /// if disjoint from *all*. Empty by default.
    pub rename_extra: Vec<(u64, u64)>,
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
    /// **Bounded-target dynamic `call_indirect` (approach A).** When `Some(cap)`, a pre-pass
    /// ([`crate::lower_indirect_dispatch`]) rewrites every dynamic-index `call_indirect` whose
    /// demanded signature is matched by at most `cap` module functions into an explicit masked
    /// dispatch over direct calls (one arm per target), so the specializer folds through it instead
    /// of returning [`SpecError::Unsupported`]. Sites with more than `cap` matching targets
    /// (megamorphic dispatch) are left untouched. `None` (default) keeps the constant-index-only
    /// behavior. See `lower_indirect` for the masked-compare soundness and the one known trap-kind
    /// gap on an out-of-signature index.
    pub indirect_targets_cap: Option<usize>,
    /// **Cut set — opaque runtime call-outs.** Source function indices the specializer must **not**
    /// inline, outline, or fold through. Each call to a cut callee is emitted verbatim as a residual
    /// `call`: its arguments are materialized (constants baked to `const` insts, dynamics passed
    /// through), its results become fresh unknowns, and the callee is **carried into the residual
    /// module** unspecialized (with its transitive direct-call closure) so the call resolves. This is
    /// what lets a **rolled loop keep a stateful runtime call-out** — a GC step, an allocator, an
    /// error raiser — as an opaque call while the dispatch and arithmetic around it still fold: the
    /// residual branch a rolled loop needs no longer has to *project* the callee's deeply-stateful
    /// body (the GC/`setjmp` wall), only *call* it.
    ///
    /// **Memory contract (caller promise, like [`rename_is_private`](Self::rename_is_private)).** The
    /// specializer's only abstract memory caches are the rename-region cells (held in SSA) and the
    /// constant-image folds ([`const_regions`](Self::const_regions) / readonly data / overlays); plain
    /// window memory is never cached, so a cut callee's reads/writes there flow through faithfully via
    /// the residual loads/stores already emitted around it. A cut callee must therefore touch only
    /// plain memory **disjoint from every rename and constant region**. Under that promise a cut call
    /// preserves the rename cells across the boundary (so the loop-carried register file stays in SSA
    /// and the loop rolls) when [`rename_is_private`](Self::rename_is_private) is set; without the
    /// private promise it rejects rather than miscompile (an opaque callee might alias the region, and
    /// neither preserving nor dropping the cell is sound). A cut callee that must *read or mutate*
    /// renamed state uses [`cut_calls_touch_state`](Self::cut_calls_touch_state) instead. Inline mode
    /// only; combining with [`outline_calls`](Self::outline_calls) /
    /// [`selective_outline`](Self::selective_outline) is rejected. Empty by default.
    pub cut_calls: Vec<u32>,
    /// **Cut calls that read/write the renamed state** (slice 1): a runtime call-out that touches the
    /// interpreter's register file / VM state — a GC that scans the stack, a `poscall` that rewrites
    /// registers. Same opacity as [`cut_calls`](Self::cut_calls) (emitted verbatim, results unknown,
    /// callee carried), but the live rename cells are **spilled to the window before the call and
    /// reloaded after** — so the callee observes the current state through memory and its writes are
    /// seen by later renamed accesses. The rename region must alias the state the callee touches (as
    /// with any live-backed rename). This is the general safepoint-spill discipline (a call is a
    /// safepoint; live registers spill to the frame and reload), and it is interpreter-agnostic — it
    /// operates on the generic rename region, nothing Lua-specific. Spilled/reloaded cells must be a
    /// natural 4- or 8-byte width; anything else fails closed. Empty by default.
    pub cut_calls_touch_state: Vec<u32>,
    /// **Cut calls that *read* the renamed state but do not rewrite its cells** — the safepoint spill
    /// for a **non-moving** collector or a read-only scan (a GC that marks the register file but never
    /// relocates it, a read barrier). Like [`cut_calls_touch_state`](Self::cut_calls_touch_state) the
    /// live rename cells are **spilled to the window before the call** (any natural width — the tag
    /// bytes spill too, so the scan sees correctly-typed registers), but they are **not reloaded**
    /// afterward: the abstract cells keep their pre-call SSA values. That is exactly right when the
    /// callee reads the state but leaves the registers unchanged — and, crucially, it keeps the folded
    /// tag constants *folded* across the call (a reload would turn every tag into a fresh unknown and
    /// re-open the per-opcode type checks the dispatch fold depends on). This is what lets an
    /// **allocating** loop keep a rolled, dispatch-free fast path while the allocator/GC stay opaque
    /// call-outs (a `luaC_checkGC` step, a `luaH_new`), where the collector scanning the stack would
    /// otherwise force projecting the whole GC.
    ///
    /// **Caller promise** (like [`rename_is_private`](Self::rename_is_private)): the callee must not
    /// write, nor **relocate**, any renamed region — a GC that reallocates the stack would invalidate
    /// the baked cell addresses. Spilled cells use the natural 1/2/4/8-byte widths; anything else fails
    /// closed. Empty by default.
    pub cut_calls_read_state: Vec<u32>,
    /// **Guard-and-deopt targets** (slice 2): `(func, block)` source blocks that are *cold paths* the
    /// specializer must **not** project — a type-check failure, an overflow slow path, an error raise,
    /// a GC-needed branch. When a dynamic branch would enter one, the residual instead **deopts**:
    /// it spills all live rename cells to the window (making it a valid interpreter resume image) and
    /// tail-calls [`deopt_handler`](Self::deopt_handler) to resume the original interpreter from that
    /// state. Fast paths fold; cold paths bail out — the standard guarded-fast-path JIT discipline,
    /// expressed generically over `(func, block)` so it fits any interpreter ported to svm, not just
    /// Lua. Requires [`deopt_handler`](Self::deopt_handler). Empty by default.
    pub deopt_targets: Vec<(u32, u32)>,
    /// The resume function tail-called on a deopt (see [`deopt_targets`](Self::deopt_targets)),
    /// carried into the residual like a cut callee. Two signatures are accepted, both returning the
    /// entry's results (the tail-call forwards them):
    ///
    /// - **`() -> <entry results>`** — resumes purely from the **written-back window state** (the
    ///   captured-VM-state model: the VM state, including a resumable `pc`, lives at known window
    ///   addresses the handler reads). Nothing is threaded.
    /// - **`<entry params> -> <entry results>`** — also receives the **entry's argument values**. The
    ///   specializer threads the entry's dynamic parameters through the CFG to every deopt edge and
    ///   passes them to the handler, so an interpreter whose input arrives as parameters (not only via
    ///   the window) can resume. Constant entry arguments are baked in; dynamic ones are threaded.
    ///
    /// `None` unless [`deopt_targets`] / [`deopt_edges`](Self::deopt_edges) is used.
    pub deopt_handler: Option<u32>,
    /// **Guard-and-deopt on a control-flow *edge*** — `(func, from_block, to_block)` triples. Like
    /// [`deopt_targets`](Self::deopt_targets), but the bail is keyed on a *branch edge* rather than a
    /// block, so a **shared** target block can be deopted on one cold in-edge while still projected on
    /// its hot ones. The motivating case: `OP_CALL` branches on whether the (dynamic, cut-looked-up)
    /// callee was a Lua function; the "Lua function" edge re-enters the dispatch loop with a dynamic
    /// new frame and **diverges** (an unknown function projected), while the shared dispatch block is
    /// the hot loop header — so the block can't be a `deopt_targets`. Marking just that edge cold bails
    /// it to the baseline (it is dead for a C-function call like `print`) and bounds the projection.
    /// Requires [`deopt_handler`](Self::deopt_handler). Empty by default.
    pub deopt_edges: Vec<(u32, u32, u32)>,
    /// **Carry the whole source module at its original function indices** (for cut call-outs whose
    /// closure contains *indirect* calls). The normal cut carry (`cut_calls*`) renumbers the carried
    /// closure to compact residual indices — which is unsound the moment a carried function dispatches
    /// through the function table, because the runtime funcref values it loads (a `global_State`'s
    /// allocator / panic / finalizer pointers) are **original** indices, and there is no table to
    /// remap. Real allocators/GCs do exactly this (`(*g->frealloc)(…)`, `g->panic(L)`), so their cut
    /// closure can't be compacted.
    ///
    /// With this set, the residual instead keeps **every** source function at its own index (plus the
    /// source data / data-funcref image), so the identity function table stays valid and any carried
    /// indirect call resolves; the cut map is the identity, and the specialized entry is **appended at
    /// the end** (index `module.funcs.len()`, *not* 0 — the source entry is left intact so a re-entrant
    /// carried callee, e.g. a GC finalizer, still finds the original). The residual is therefore large
    /// (it contains the whole runtime), but only the appended entry is the rolled fast path; the rest
    /// is the cold runtime, reached only through the opaque cut calls. Inline mode only. Off by default.
    pub carry_whole_module: bool,
    /// With [`carry_whole_module`](Self::carry_whole_module): keep the **import-bearing** function
    /// bodies (the guest I/O layer) verbatim instead of replacing them with trap stubs. Off by default
    /// (a standalone residual can't verify an unresolved import, so it stubs). Set it when the residual
    /// will be **embedded back into a runnable module** and its host binds the imports — then the caller
    /// must re-attach the source module's `imports` manifest (this pass leaves it empty, to keep
    /// `String`-cloning the manifest out of the on-ramped `svm-peval` guest build).
    pub carry_keep_imports: bool,
    /// **Runtime-input cells — the Futamura "dynamic input" for a rolled residual.** Rename-region
    /// cells `(address, width)` whose entry value is *not* known at specialization time: instead of
    /// reading its seed from the constant image (as [`rename_seed_from_image`](Self::rename_seed_from_image)
    /// would), each cell becomes a fresh **dynamic residual parameter**, appended after the dynamic
    /// call arguments in address order.
    ///
    /// This is what turns the interpreter's own state capture into an *input-taking* residual: seed
    /// the whole VM state from the image so the dispatch (opcode/`pc`/tag cells) folds, but mark the
    /// one register (or few) that carry the program's runtime input dynamic — so the loop condition
    /// that reads them stays dynamic and the loop **rolls** instead of unrolling to a constant. It is
    /// the direct generalization of the toy-VM demos where the residual took `(input)` and the entry
    /// threaded it into a register; here the register *is* the input, named by its window address.
    ///
    /// Each cell must lie fully within a rename region (it is renamed like any region cell — carried
    /// in SSA, spilled on deopt, written back on exit); a dynamic cell outside every region is
    /// rejected. Widths are the natural 4/8. Empty by default.
    pub dynamic_cells: Vec<(u64, u32)>,
    /// **Project *through* a Lua-to-Lua call — the precall post-state model.** In Lua 5.4 an `OP_CALL`
    /// does `newci = luaD_precall(L, ra, nres); ci = newci; goto startfunc`, re-entering the *same*
    /// dispatch loop with the callee's frame; `newci` is the **return value** of the (cut) `precall`,
    /// and the shared dispatch block folds `cl = *ci->func`, `proto`, `code` from it. Left to the plain
    /// [`cut_calls_touch_state`](Self::cut_calls_touch_state) treatment `precall`'s result is a fresh
    /// unknown, so the "was the callee a Lua function?" branch stays dynamic and the Lua edge must
    /// deopt ([`deopt_edges`](Self::deopt_edges)).
    ///
    /// With this set, at each `precall` cut the specializer reads the callee closure from `ra` (a
    /// constant stack slot in the fold) and **binds the result statically**: a C closure / light-C
    /// function ⇒ `NULL` (the C call is handled in place, the branch folds to the continue edge); a Lua
    /// closure whose `Proto` is one of [`PrecallModel::frames`] ⇒ the deterministic `CallInfo` address
    /// `precall` will (re)use for it, and `L->ci` is set to that address — so the branch folds to the
    /// Lua re-dispatch edge and the callee's own dispatch folds *inline* (the register cells are left
    /// folded: Lua's callee base reuses the in-place argument slots, which `precall` does not clobber).
    /// An unrecognized callee (dynamic `ra`, or a Lua proto not listed) falls back to the plain
    /// touch-state spill+reload. The callee frame's bytes (its `CallInfo`, `Proto`, code, closure) must
    /// be supplied via [`const_overlays`](Self::const_overlays), captured from a profiling run (the
    /// arena is deterministic and the `CallInfo` is loop-invariant once Lua caches `ci->next`).
    ///
    /// `precall` must also appear in a cut set (so it is carried and emitted opaquely). `None` by
    /// default. The paired return side (`poscall`) is not yet modeled — a callee return still bails via
    /// a [`deopt_edges`](Self::deopt_edges) / [`deopt_targets`](Self::deopt_targets) to the baseline.
    pub precall_model: Option<PrecallModel>,
    /// **Tail-call post-state model** (see [`PretailcallModel`]) — the `OP_TAILCALL` counterpart of
    /// [`precall_model`](Self::precall_model): `return f(x)` goes through `luaD_pretailcall`, which
    /// *reuses* the current frame instead of pushing one. `None` (the default) leaves
    /// `luaD_pretailcall` to the ordinary cut treatment (unknown result ⇒ both arms explored).
    pub pretailcall_model: Option<PretailcallModel>,
}

/// The static description a [`SpecConfig::precall_model`] needs to bind a `luaD_precall` cut's result
/// (and, for a Lua callee, install the callee frame). Call sites are keyed by `ra` — the callee's
/// **stack-slot address**, which is a constant in the fold (`base + A·sizeof(TValue)`, both constant)
/// and distinct per call site — *not* by the callee closure, which `OP_CLOSURE` allocates at runtime
/// (an opaque cut) so its pointer is dynamic. The addresses come from the caller's capture; nothing
/// here is Lua-specific to the engine beyond the `ra` argument position.
#[derive(Clone, Debug)]
pub struct PrecallModel {
    /// The cut callee that sets up a call frame (`luaD_precall`). Must also be in a cut set.
    pub precall: u32,
    /// Which argument of the `precall` call is the callee's stack slot `ra` (0-based index into the
    /// call's argument list).
    pub ra_arg: usize,
    /// The address of the `L->ci` field — the rename cell set to the callee `CallInfo` on a Lua call,
    /// so the callee frame is active for the folded dispatch (and a later spill/deopt is a valid
    /// resume image).
    pub l_ci_addr: u64,
    /// **Lua call sites** (see [`LuaSite`]) — at a `precall` whose `ra` matches, the callee frame is
    /// installed and the branch folds to the Lua re-dispatch edge.
    pub lua_sites: Vec<LuaSite>,
    /// **C call sites**: `ra` values at which the callee is a C function — the result is bound to
    /// `NULL` (the C call is handled in place inside the opaque `precall`, so its effects still happen)
    /// and the branch folds to the continue edge; the register cells are reloaded (the C function ran).
    pub c_sites: Vec<u64>,
    /// **Return side** (see [`PoscallModel`]). `None` ⇒ the callee return is not modeled and must bail
    /// via a deopt (slice 1); `Some` ⇒ `poscall` pops the frame in the abstract state so folding
    /// continues in the caller.
    pub poscall: Option<PoscallModel>,
}

/// The return side of the call model: how a `luaD_poscall` cut updates the abstract frame so folding
/// continues in the **caller** instead of bailing. `poscall` moves results down, pops `L->ci` to
/// `ci->previous`, and returns; the shared post-return block then re-reads `L->ci` to resume the
/// caller (or, for the entry frame, exits `luaV_execute` — which folds because the frame's
/// `callstatus` is a constant overlay). The model reads the returning frame's `ci` (a constant in the
/// fold — `L->ci` was pinned to it at the matching `precall`), computes the caller `ci` from its
/// `previous` field, and installs that at `L->ci`.
#[derive(Clone, Debug)]
pub struct PoscallModel {
    /// The cut callee that pops a call frame (`luaD_poscall`). Must also be in a cut set.
    pub poscall: u32,
    /// Byte offset from a `CallInfo` to its `previous` pointer (the caller frame).
    pub ci_previous_off: u64,
    /// Byte offset from a `CallInfo` to its `func` slot (the caller's call slot `ra`, where `poscall`
    /// writes the results). Only read when [`selective`](Self::selective) applies.
    pub ci_func_off: u64,
    /// Byte offset from a result slot's base to its `TValue` tag byte. Only read for `selective`.
    pub tag_off: u64,
    /// **Selective reload** — the lever that lets a *call-bearing loop roll*. Each `(ci, tag)` names a
    /// returning frame whose single result is a value with the (promised, captured) tag `tag`: instead
    /// of reloading **every** cell as an unknown (which would turn the caller's folded register tags
    /// dynamic and force the loop to unroll), only the result **value** cell (`ci->func`, 8 bytes) is
    /// reloaded dynamic and its **tag** cell (`ci->func + tag_off`) is pinned to `tag` — every other
    /// caller cell stays folded. A returning `ci` not listed here gets the plain full reload. Empty by
    /// default (the whole-frame reload of the non-rolling case).
    pub selective: Vec<(u64, u8)>,
}

/// The static description for a `luaD_pretailcall` cut (`return f(x)` — `OP_TAILCALL`). Unlike
/// `luaD_precall`, a tail call **reuses the current frame**: the callee closure and its arguments are
/// moved down to the frame's own `func` slot, `L->ci` is unchanged, and the eventual return pops
/// straight to the *caller's* caller. Sites are keyed by the (constant) `ra` of the callee's temp
/// slot, exactly like [`PrecallModel::lua_sites`]. For a matching Lua site the cut's `int` result is
/// bound **negative** (the "Lua callee, frame moved" arm), `L->ci` is pinned to the site's
/// `callee_ci` (the *reused* node — normally already there), and the site's pins install the moved
/// closure and the callee's `savedpc`. A non-matching `ra` falls back to the plain state-touching cut
/// (unknown result, full reload).
#[derive(Clone, Debug)]
pub struct PretailcallModel {
    /// The cut callee that performs the frame replacement (`luaD_pretailcall`). Must also be in a
    /// cut set (it is emitted opaquely either way; this model only chooses the abstract post-state).
    pub pretailcall: u32,
    /// Which argument of the call is the callee's stack slot `ra` (0-based).
    pub ra_arg: usize,
    /// The address of the `L->ci` field (same cell as [`PrecallModel::l_ci_addr`]).
    pub l_ci_addr: u64,
    /// Byte offset from a stack slot's base to its `TValue` tag byte (same as
    /// [`PoscallModel::tag_off`]) — used for the moved-argument reloads.
    pub tag_off: u64,
    /// Tail-call sites (see [`TailSite`]).
    pub sites: Vec<TailSite>,
}

/// One tail-call site. Unlike a [`LuaSite`], a tail call **moves** the callee closure and its
/// arguments down onto the reused frame *inside the opaque cut* — so the frame's argument cells hold
/// values the fold last saw at *other* addresses. `pins` carry the moved closure slot and the
/// per-callee `ci` fields (`savedpc`), exactly like a shared-`CallInfo` sequential site; `args`
/// apply the selective-reload discipline to the moved arguments so the callee reads the *runtime*
/// values, not the frame's stale abstract cells.
#[derive(Clone, Debug)]
pub struct TailSite {
    /// The (constant) stack-slot address of the callee's temp position — the site key.
    pub ra: u64,
    /// The reused frame node.
    pub callee_ci: u64,
    /// `(address, value)` cells pinned constant after the cut (closure slot, `ci` fields).
    pub pins: Vec<(u64, u64)>,
    /// The callee's argument slots *after* the move-down: `(value cell address, promised tag)`.
    /// Each value cell reloads **dynamic** (the cut moved a runtime value into it) and its tag
    /// pins to the captured constant, so the callee's fast-path tag checks fold while the values
    /// flow at runtime.
    pub args: Vec<(u64, u8)>,
}

/// One Lua call site for the [`PrecallModel`]. On a `precall` whose `ra` matches, the result (`newci`)
/// is bound to `callee_ci` (so the "was it a Lua function?" branch folds to the re-dispatch edge),
/// `L->ci` is set to it, and each `pin` cell is set to its constant value; the register cells are left
/// folded (Lua's callee base reuses the in-place argument slots, untouched by `precall`) so the
/// callee's own dispatch folds inline. All addresses/values come from a profiling capture; the callee
/// frame's bytes (its `CallInfo`, closure, `Proto`, code) must be supplied via
/// [`SpecConfig::const_overlays`].
#[derive(Clone, Debug)]
pub struct LuaSite {
    /// The callee stack slot `ra` that identifies this call site (constant in the fold).
    pub ra: u64,
    /// The deterministic `CallInfo` address `precall` (re)uses for this callee — bound as the result
    /// and installed at `L->ci`.
    pub callee_ci: u64,
    /// Extra `(addr, value)` cells to fold at the call — structural pointers the dispatch setup needs
    /// that are otherwise dynamic (e.g. the callee's function stack slot → its `LClosure` pointer,
    /// which `OP_CLOSURE` allocated opaquely). Each is written into the abstract memory as a constant.
    pub pins: Vec<(u64, u64)>,
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
    // Approach A: optionally rewrite dynamic-index `call_indirect`s into bounded direct-call
    // dispatch before specializing, so the engine folds through them (see `lower_indirect`). The
    // rewrite preserves function signatures/indices, so `func` and `args` still refer to the same
    // entry.
    let lowered;
    let module = match config.indirect_targets_cap {
        Some(cap) => {
            lowered = crate::lower_indirect_dispatch(module, cap);
            &lowered
        }
        None => module,
    };

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

    // Runtime-input cells become dynamic residual parameters: an entry `MemPattern` of dynamic
    // (`None`) cells in address order, shared by every entry build path (its cells are appended after
    // the dynamic call args, matching `build_func`'s residual-param order and `build_block`'s entry
    // parameters). Each must be a natural 4/8-byte cell fully inside a rename region — it is renamed
    // like any region cell (carried in SSA, spilled on deopt, written back on exit). Built with
    // explicit pushes + insertion sort (no `.collect()`/`slice::sort` — the LLVM→IR on-ramp can't
    // resolve driftsort's `sqrt_approx`, and svm-peval is itself on-ramped in the guest peval tests).
    let mut entry_mem: MemPattern = Vec::new();
    if !config.dynamic_cells.is_empty() {
        let regions = all_regions(config);
        for &(addr, width) in &config.dynamic_cells {
            if width != 4 && width != 8 {
                return Err(SpecError::Unsupported);
            }
            if !within_region(&regions, addr, width as u64) {
                return Err(SpecError::Unsupported);
            }
            // Insert keeping `entry_mem` sorted by address (the canonical memory-cell order).
            let mut i = entry_mem.len();
            while i > 0 && entry_mem[i - 1].0 > addr {
                i -= 1;
            }
            entry_mem.insert(i, (addr, width, None));
        }
    }

    let has_memory = module.memory.is_some();
    let value_types: Vec<Vec<Vec<ValType>>> = module
        .funcs
        .iter()
        .map(|f| func_value_types(f, &module.funcs, has_memory))
        .collect();

    // When outlining, the renamed region's live abstract cells are threaded across each residual call
    // boundary (passed in as extra arguments, returned as extra results); see `outline_call`. The
    // single-function inline path keeps the region entirely internal (no threading).
    // Deopt targets/edges require a handler; reject early so the config can't be silently ignored (with
    // no cut callees and no handler, `carry_roots` would be empty and the plain inline path taken).
    if (!config.deopt_targets.is_empty() || !config.deopt_edges.is_empty())
        && config.deopt_handler.is_none()
    {
        return Err(SpecError::Unsupported);
    }
    // Functions carried verbatim into the residual: every cut callee (opaque and state-touching) plus
    // the deopt handler, closed over their direct calls. Empty ⇒ the plain single-function inline path.
    let carry_roots = cut_roots(config);

    // Whole-module carry: keep every source function at its original index (identity cut), so a carried
    // cut callee's *indirect* calls (an allocator's `frealloc`, a GC's finalizer) resolve through the
    // unchanged identity table. The specialized entry is appended after them (see the field docs).
    if config.carry_whole_module {
        if config.outline_calls || config.selective_outline {
            return Err(SpecError::Unsupported);
        }
        // Identity cut over the roots (each cut callee / deopt handler is carried at its own index).
        let mut cut: BTreeMap<u32, u32> = BTreeMap::new();
        for &r in &carry_roots {
            if r as usize >= module.funcs.len() {
                return Err(SpecError::BadFunc);
            }
            cut.insert(r, r);
        }
        let mut deopt_targets: BTreeSet<(u32, u32)> = BTreeSet::new();
        for &t in &config.deopt_targets {
            deopt_targets.insert(t);
        }
        let (deopt_handler, deopt_pass_args) =
            match resolve_deopt_handler(module, config, func, &cut)? {
                Some((ridx, pass)) => (Some(ridx), pass),
                None => (None, false),
            };
        let entry_func = build_func(
            module,
            config,
            &value_types,
            None,
            func,
            &entry_pattern,
            &entry_mem,
            &cut,
            &deopt_targets,
            deopt_handler,
            deopt_pass_args,
            false,
        )?
        .0;
        // Carry every source function at its own index, but replace any function that reaches a host
        // boundary (a `CallImport` / capability / symbol call) with a same-signature trap stub: a
        // standalone residual can't verify an unresolved import, and such functions are never on the
        // cut closure's execution path (they are the guest's I/O layer, not its allocator/GC). Their
        // slots stay filled so every other function's indices — and the identity table — are unchanged.
        // With [`SpecConfig::carry_keep_imports`] the import-bearing bodies are kept verbatim instead
        // of stubbed — for a residual meant to be **embedded back into a runnable module** (whose host
        // binds the imports), where the I/O layer must still work. The caller then re-attaches the
        // source `imports` manifest (svm-peval itself leaves it empty — cloning the `String`-named
        // manifest here would pull `String::clone` into the on-ramped guest build).
        let keep = config.carry_keep_imports;
        let mut funcs: Vec<Func> = module
            .funcs
            .iter()
            .map(|f| {
                if !keep && imports_a_boundary(f) {
                    trap_stub(f)
                } else {
                    f.clone()
                }
            })
            .collect();
        funcs.push(entry_func); // the rolled fast path — the residual entry, at index funcs.len()-1
        return Ok(Module {
            funcs,
            memory: module.memory,
            // Keep the data image so any baked data pointers resolve; the `call_indirect` identity
            // table needs no side data (a funcref is its own function index, and its runtime value is
            // read from the live heap the caller re-seeds). `imports` / `types` are dropped — every
            // import-bearing function is stubbed, and `call_indirect` carries its signature inline —
            // which also keeps this branch free of the `String`-cloning the LLVM→IR on-ramp can't
            // resolve when svm-peval is itself on-ramped (the `peval_futamura` guest test).
            data: module.data.clone(),
            data_ptrs: Vec::new(),
            data_funcrefs: Vec::new(),
            imports: Vec::new(),
            exports: Vec::new(),
            data_exports: Vec::new(),
            impl_exports: Vec::new(),
            types: Vec::new(),
            debug_info: None,
        });
    }
    let funcs = if config.outline_calls || config.selective_outline {
        if !carry_roots.is_empty() {
            // Carry numbers residual functions inline-style (entry 0, carried after); the outline
            // driver numbers them differently, so the combination is rejected.
            return Err(SpecError::Unsupported);
        }
        outline_funcs(
            module,
            config,
            &value_types,
            func,
            entry_pattern,
            &entry_mem,
        )?
    } else if carry_roots.is_empty() {
        vec![
            build_func(
                module,
                config,
                &value_types,
                None,
                func,
                &entry_pattern,
                &entry_mem,
                &BTreeMap::new(),
                &BTreeSet::new(),
                None,
                false,
                false,
            )?
            .0,
        ]
    } else {
        // Reserve residual index 0 for the specialized entry, then carry each root and its transitive
        // direct-call closure verbatim at indices 1.. — so an opaque cut `call` / deopt tail-call
        // resolves. `cut` maps a source callee index to its residual index; `carried` is the closure
        // in the order the indices were assigned.
        let (cut, carried) = plan_cut_carry(module, func, &carry_roots)?;
        // Deopt targets (as a set) and the handler's residual index. Validated: deopt requires a
        // handler whose signature matches the entry (a deopt tail-calls it), and the handler must be
        // carried (it is, via `cut_roots`). Built with explicit inserts, not `.collect()` — a
        // `BTreeSet::from_iter` sorts via `slice::sort`, which the LLVM→IR on-ramp can't resolve.
        let mut deopt_targets: BTreeSet<(u32, u32)> = BTreeSet::new();
        for &t in &config.deopt_targets {
            deopt_targets.insert(t);
        }
        let (deopt_handler, deopt_pass_args) =
            match resolve_deopt_handler(module, config, func, &cut)? {
                Some((ridx, pass)) => (Some(ridx), pass),
                None => (None, false),
            };
        let mut funcs = vec![
            build_func(
                module,
                config,
                &value_types,
                None,
                func,
                &entry_pattern,
                &entry_mem,
                &cut,
                &deopt_targets,
                deopt_handler,
                deopt_pass_args,
                false,
            )?
            .0,
        ];
        for &orig in &carried {
            funcs.push(carry_func(&module.funcs[orig as usize], &cut)?);
        }
        funcs
    };

    Ok(Module {
        data_ptrs: Vec::new(),
        data_funcrefs: Vec::new(),
        funcs,
        memory: module.memory,
        data: module.data.clone(),
        imports: vec![],
        // The residual's functions are freshly built (specialized/renumbered), so the source
        // module's name→funcidx exports no longer apply; a residual is addressed by index.
        // Interface offers are dropped for the same reason (their op funcidxs are stale).
        exports: vec![],
        // A residual is a finished, index-addressed program — no cross-unit data symbols to export.
        data_exports: vec![],
        impl_exports: vec![],
        types: vec![],
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
    cut: &BTreeMap<u32, u32>,
    deopt_targets: &BTreeSet<(u32, u32)>,
    deopt_handler: Option<u32>,
    deopt_pass_args: bool,
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
    // The types of the entry's dynamic parameters — the block-parameter types of the deopt argument-
    // threading channel (see [`Spec::cur_thread`]). Empty unless the handler takes the entry's args.
    let thread_types: Vec<ValType> = if deopt_pass_args {
        pattern
            .iter()
            .zip(&cf.params)
            .filter_map(|(slot, ty)| slot.is_none().then_some(*ty))
            .collect()
    } else {
        Vec::new()
    };

    // Selective outlining only makes sense when outlining is enabled; when it is, populate the
    // per-frame recursion signatures (otherwise they stay empty, leaving the memo key unchanged).
    let selective = outline.is_some() && config.selective_outline;
    let mut spec = Spec {
        module,
        config,
        regions: all_regions(config),
        value_types,
        outline,
        selective,
        thread_cells,
        cut,
        deopt_targets,
        deopt_handler,
        deopt_pass_args,
        entry_pattern: pattern,
        cur_thread: Vec::new(),
        thread_types,
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
    // The initial context is the deopt entry seed when the handler takes the entry's arguments (see
    // [`Task::seed`]); otherwise `false`, leaving the memo partition unchanged.
    spec.intern(
        vec![Frame {
            func: callee,
            block: 0,
            ip: 0,
            env: pattern.clone(),
            entry,
        }],
        mem_pat.clone(),
        deopt_pass_args,
        false, // the entry is never itself a deopt edge
    );

    let mut blocks = Vec::new();
    while let Some(task) = spec.queue.pop_front() {
        if blocks.len() >= DEFAULT_BUDGET {
            return Err(SpecError::Budget);
        }
        #[cfg(feature = "trace")]
        let where_ = task.frames.last().map(|f| (f.func, f.block, f.ip));
        match spec.build_block(task) {
            Ok(b) => blocks.push(b),
            Err(e) => {
                trace_unsup!(
                    "build_block failed at (func,block,ip)={:?}: {:?}",
                    where_,
                    e
                );
                return Err(e);
            }
        }
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
    entry_mem: &MemPattern,
) -> Result<Vec<Func>, SpecError> {
    let state = RefCell::new(OutlineState {
        memo: BTreeMap::new(),
        funcs: Vec::new(),
    });
    // The entry is residual function 0. It does **not** thread region cells in/out: the rename region
    // is private scratch, established fresh (zero) on entry and discarded on return, so the residual
    // entry's signature is just the source function's — plus any runtime-input (`dynamic_cells`)
    // parameters, which are part of the entry key.
    {
        let mut s = state.borrow_mut();
        s.memo.insert(
            (entry, entry_pattern.clone(), entry_mem.clone()),
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
        entry_mem,
        &BTreeMap::new(),
        &BTreeSet::new(),
        None,
        false,
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
        &BTreeMap::new(),
        &BTreeSet::new(),
        None,
        false,
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
    /// The full rename region set: [`SpecConfig::rename`] (if any) followed by
    /// [`SpecConfig::rename_extra`], precomputed once. An access is renamed if it lies fully within
    /// any one; disjoint only if disjoint from all.
    regions: Vec<(u64, u64)>,
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
    /// The cut set as a **source callee index → residual function index** map (see
    /// [`SpecConfig::cut_calls`]). A call whose callee is a key is emitted as an opaque residual `call`
    /// to the mapped (carried) function instead of being inlined. Empty unless the cut set is active.
    cut: &'a BTreeMap<u32, u32>,
    /// Guard-and-deopt targets ([`SpecConfig::deopt_targets`]) as `(func, block)`. Reaching one of
    /// these blocks emits a deopt exit instead of projecting it. Empty unless deopt is active (and only
    /// on the inline entry build — carried/outlined functions never carry deopt targets).
    deopt_targets: &'a BTreeSet<(u32, u32)>,
    /// The residual function index of the deopt resume handler ([`SpecConfig::deopt_handler`]), tail-
    /// called on a deopt. `None` unless deopt is active.
    deopt_handler: Option<u32>,
    /// Whether the deopt handler takes the entry's arguments (`<entry params> -> R`) rather than
    /// resuming purely from the window (`() -> R`). When set, the entry's dynamic parameters are
    /// threaded through the CFG (see [`Spec::cur_thread`]) and passed to the handler at each deopt.
    deopt_pass_args: bool,
    /// The entry function's [`ParamPattern`]: the argument shape a deopt tail-call reconstructs for the
    /// handler (constant lanes baked, dynamic lanes taken from [`Spec::cur_thread`]). Only read when
    /// `deopt_pass_args` is set.
    entry_pattern: &'a ParamPattern,
    /// **Working state, set per built block.** The current block's threaded entry-argument values (the
    /// dynamic entry params, in entry order), read by [`Spec::branch_to`] to forward them along every
    /// edge and by the deopt emitter to build the handler call. Empty unless `deopt_pass_args`.
    cur_thread: Vec<Abs>,
    /// The block-parameter types of the [`Spec::cur_thread`] channel — the entry's dynamic-parameter
    /// types, reconstructed as extra block params on every non-seed block. Empty unless `deopt_pass_args`.
    thread_types: Vec<ValType>,
    /// The threaded out-cell signature (`(addr, width)` by address), fixed at the first `return` and
    /// required to match at every other return. `None` until the first return; stays `None` when
    /// nothing is threaded.
    out_cells: Option<CellSig>,
    /// `(call stack, memory pattern) → residual block id`. The memo that makes the loop terminate
    /// and that closes residual loops.
    memo: BTreeMap<(Vec<Frame>, MemPattern, bool, bool), u32>,
    queue: VecDeque<Task>,
    next_id: u32,
}

impl Spec<'_> {
    /// Get (or create) the residual block id for a context, enqueuing it the first time it is
    /// seen. Ids are assigned in enqueue order and blocks are produced in that same (FIFO) order,
    /// so id == position in the output `blocks`. `seed` is the deopt entry-seed flag (see
    /// [`Task::seed`]); it is `false` for every non-initial context and always `false` unless deopt
    /// argument threading is active, so it leaves the memo partition unchanged in the common case.
    fn intern(&mut self, frames: Vec<Frame>, mem: MemPattern, seed: bool, deopt: bool) -> u32 {
        let key = (frames, mem, seed, deopt);
        if let Some(&id) = self.memo.get(&key) {
            return id;
        }
        let id = self.next_id;
        self.next_id += 1;
        self.queue.push_back(Task {
            frames: key.0.clone(),
            mem: key.1.clone(),
            seed,
            deopt,
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
        let task_is_deopt_edge = task.deopt;

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
        // Deopt argument threading: on every non-seed block, the entry's dynamic arguments arrive as
        // extra block parameters, declared *after* the memory cells — the fixed position `branch_to`
        // forwards them at. The entry seed block instead sources them from its own parameters (below).
        let mut thread_abs: Vec<Abs> = Vec::new();
        if self.deopt_pass_args && !task.seed {
            for &ty in &self.thread_types {
                thread_abs.push(Abs::Dyn(rnext));
                rnext += 1;
                params.push(ty);
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

        // Seed the threading channel at the entry block: the threaded arguments *are* the entry's own
        // dynamic parameters (its env here), so no extra params are declared. Then publish the current
        // block's threaded values for `branch_to` and the deopt emitter to read.
        if self.deopt_pass_args && task.seed {
            thread_abs = self
                .entry_pattern
                .iter()
                .zip(&env)
                .filter_map(|(slot, &a)| slot.is_none().then_some(a))
                .collect();
        }
        self.cur_thread = thread_abs;

        // Diagnostic (under the `trace` feature): one line per residual block built, naming the active
        // source `(func, block)`, the symbolic call-stack depth, and the live memory-cell count. This is
        // how a runaway fold (a loop unrolling instead of rolling, or unbounded inline recursion) is
        // spotted — the divergent frame shows up as a repeating/growing pattern before the budget trips.
        trace_unsup!(
            "BLOCK id~{} active=({},{}) ip={} nframes={} memcells={}",
            self.next_id,
            active_func,
            active_block,
            active_ip,
            frames.len() + 1,
            mem.len()
        );

        // Guard-and-deopt: entering a cold target block bails to the resume handler instead of
        // projecting the block (which would drag in the interpreter's cold, stateful machinery). Spill
        // all live rename cells to the window — making it a valid interpreter resume image — then tail-
        // call the handler, which resumes from that state (and, for a `<entry params> -> R` handler,
        // from the threaded entry arguments). Deopt targets are only set on the inline entry.
        if let Some(handler) = self.deopt_handler {
            // Bail to the resume handler either because this *block* is a cold target
            // ([`SpecConfig::deopt_targets`]) or because this context was reached by a cold *edge*
            // ([`SpecConfig::deopt_edges`], carried on the task) — both spill the live rename cells to
            // the window (a valid resume image) and tail-call the baseline.
            if task_is_deopt_edge || self.deopt_targets.contains(&(active_func, active_block)) {
                let mut out: Vec<Inst> = Vec::new();
                self.write_back_cells(&mem, &mut out, &mut rnext)?;
                let args = self.deopt_handler_args(&mut out, &mut rnext);
                return Ok(svm_ir::Block {
                    params,
                    insts: out,
                    term: Terminator::ReturnCall {
                        func: handler,
                        args,
                    },
                });
            }
        }

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
                let (target, args) = self.branch_to(&frames, &mem, false);
                Terminator::Br { target, args }
            }
            Exec::Done => self.finish_term(
                &src.term,
                frames,
                active_func,
                active_block,
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
                // Precall post-state model (see [`SpecConfig::precall_model`]): the `luaD_precall` cut is
                // projected specially — its result is bound to the callee frame the branch that follows
                // will switch on, instead of a blind unknown. It is still a cut (carried + emitted
                // opaquely); this only chooses the abstract post-state.
                if let Some(pm) = &self.config.precall_model {
                    if callee == pm.precall {
                        let ridx = self.cut[&callee];
                        let results =
                            self.emit_precall(ridx, callee, &args_abs, pm, mem, out, rnext)?;
                        env.extend(results);
                        continue;
                    }
                    if let Some(po) = &pm.poscall {
                        if callee == po.poscall {
                            let ridx = self.cut[&callee];
                            let results = self
                                .emit_poscall(ridx, callee, &args_abs, pm, po, mem, out, rnext)?;
                            env.extend(results);
                            continue;
                        }
                    }
                }
                // Tail-call post-state model (see [`SpecConfig::pretailcall_model`]): like the precall
                // model, but the frame is REUSED — result bound negative for the Lua arm, no new ci.
                if let Some(tm) = &self.config.pretailcall_model {
                    if callee == tm.pretailcall {
                        let ridx = self.cut[&callee];
                        let results =
                            self.emit_pretailcall(ridx, callee, &args_abs, tm, mem, out, rnext)?;
                        env.extend(results);
                        continue;
                    }
                }
                // Cut set: a call-out we deliberately keep opaque (see [`SpecConfig::cut_calls`]).
                // Emit it as a residual `call` to the carried callee and treat its results as
                // unknowns — never inline or fold through it. Only *explicitly listed* callees are cut;
                // the transitive closure is carried so cut callees resolve, but a direct call to a
                // closure member is inlined as usual. State-touching callees spill+reload the region.
                let mode = if self.config.cut_calls.contains(&callee) {
                    Some(CutMode::Opaque)
                } else if self.config.cut_calls_touch_state.contains(&callee) {
                    Some(CutMode::TouchState)
                } else if self.config.cut_calls_read_state.contains(&callee) {
                    Some(CutMode::ReadState)
                } else {
                    None
                };
                if let Some(mode) = mode {
                    let ridx = self.cut[&callee];
                    let results =
                        self.emit_cut_call(ridx, callee, &args_abs, mode, mem, out, rnext)?;
                    env.extend(results);
                    continue;
                }
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

    /// Emit an **opaque cut call** (see [`SpecConfig::cut_calls`]): a residual `call` to the carried
    /// callee `ridx`, with each argument materialized (a constant to a `const` inst, a dynamic passed
    /// through) and each result a fresh unknown. The call is emitted verbatim — the callee's body is
    /// never entered, so its stateful effects (a GC step, an error raise) stay behind the call
    /// boundary instead of being projected.
    ///
    /// **Memory across the boundary.** `mem` holds only rename-region cells; plain window memory is
    /// never cached, so the callee's plain-memory reads/writes already flow faithfully through the
    /// residual loads/stores around this call.
    ///
    /// - `touches_state == false` (an opaque [`SpecConfig::cut_calls`] callee): the callee is promised
    ///   not to touch the rename region. Under [`rename_is_private`](SpecConfig::rename_is_private) the
    ///   cells are **preserved** across the call (the loop-carried register file stays in SSA and the
    ///   loop rolls). Without that promise the callee might alias the region, and neither preserving nor
    ///   dropping is sound — a dropped cell would make a later in-region load read the region's seed —
    ///   so it is rejected rather than miscompiled.
    /// - `touches_state == true` (a [`SpecConfig::cut_calls_touch_state`] callee): the live cells are
    ///   **spilled to the window before the call and reloaded after** (the safepoint-spill discipline),
    ///   so the callee reads the current state through memory and its writes are seen by later renamed
    ///   accesses. Reloaded cells become fresh unknowns.
    #[allow(clippy::too_many_arguments)]
    fn emit_cut_call(
        &self,
        ridx: u32,
        callee: u32,
        args_abs: &[Abs],
        mode: CutMode,
        mem: &mut BTreeMap<u64, (u32, Abs)>,
        out: &mut Vec<Inst>,
        rnext: &mut u32,
    ) -> Result<Vec<Abs>, SpecError> {
        match mode {
            CutMode::TouchState => {
                // Spill the live state to the window so the opaque callee sees it (any natural width —
                // the register tag bytes spill too, so a callee that rewrites a register writes a
                // well-typed `TValue`). Reload happens after the call, below — the cells must survive
                // `mem` until then.
                self.write_back_cells(mem, out, rnext)?;
            }
            CutMode::ReadState => {
                // The callee reads the state but does not rewrite its cells: spill every live cell
                // (any natural width — tags included, so a stack scan sees typed registers), then keep
                // the abstract cells as-is. No reload — the folded tag constants stay folded, so the
                // dispatch stays collapsed across the call.
                self.write_back_cells(mem, out, rnext)?;
            }
            CutMode::Opaque => {
                if !self.config.rename_is_private && !mem.is_empty() {
                    // `mem` holds only rename-region cells. Without the private promise an opaque
                    // non-touching callee could still alias them, and neither preserving nor dropping
                    // is sound — refuse.
                    trace_unsup!("cut call with live non-private region cells");
                    return Err(SpecError::Unsupported);
                }
            }
        }
        // Materialize the arguments in call order (constants become `const` insts), then emit the call.
        let args: Vec<u32> = args_abs
            .iter()
            .map(|&a| materialize(a, out, rnext))
            .collect();
        out.push(Inst::Call { func: ridx, args });
        let nres = self.module.funcs[callee as usize].results.len();
        let results: Vec<Abs> = (0..nres).map(|_| Abs::Dyn(bump(rnext))).collect();
        if mode == CutMode::TouchState {
            // The opaque callee may have rewritten the state: reload every cell from the window so
            // later renamed accesses see its writes (each becomes a fresh unknown).
            self.reload_cells_natural(mem, out, rnext)?;
        }
        Ok(results)
    }

    /// Emit a `luaD_precall` cut with the **precall post-state model** ([`SpecConfig::precall_model`]).
    /// The call is emitted opaquely (like a cut), but its result — the new `CallInfo*` the following
    /// branch switches on — is bound to a **static** value chosen by the call site `ra` (a constant in
    /// the fold), so the "was it a Lua function?" branch folds and the Lua edge projects, not deopts:
    ///
    /// - **Lua site** (`ra` in [`PrecallModel::lua_sites`]): the result is the deterministic callee
    ///   `CallInfo` address and `L->ci` is set to it. The register cells are left folded (no reload) —
    ///   Lua's callee base reuses the in-place argument slots, which `precall` does not clobber — so
    ///   the callee's own dispatch folds inline.
    /// - **C site** (`ra` in [`PrecallModel::c_sites`]): the result is `NULL` (the C call is handled in
    ///   place inside the opaque call); the register cells are reloaded (the C function ran).
    /// - **Unrecognized** (dynamic `ra`, or `ra` in neither list): fall back to plain touch-state
    ///   (unknown result, spill + reload) — the pre-existing behavior.
    #[allow(clippy::too_many_arguments)]
    fn emit_precall(
        &self,
        ridx: u32,
        callee: u32,
        args_abs: &[Abs],
        pm: &PrecallModel,
        mem: &mut BTreeMap<u64, (u32, Abs)>,
        out: &mut Vec<Inst>,
        rnext: &mut u32,
    ) -> Result<Vec<Abs>, SpecError> {
        // The call site is identified by its (constant) callee stack slot `ra`.
        let ra = match args_abs.get(pm.ra_arg) {
            Some(Abs::Const(k)) => k.as_i64().map(|v| v as u64),
            _ => None,
        };
        let lua_site = ra.and_then(|ra| pm.lua_sites.iter().find(|s| s.ra == ra));
        let is_c = ra.is_some_and(|ra| pm.c_sites.contains(&ra));

        // Spill live cells so the opaque callee observes the current state, then emit the call.
        self.write_back_cells(mem, out, rnext)?;
        let args: Vec<u32> = args_abs
            .iter()
            .map(|&a| materialize(a, out, rnext))
            .collect();
        out.push(Inst::Call { func: ridx, args });
        let nres = self.module.funcs[callee as usize].results.len();
        // Result 0 is the `CallInfo*`; bind it per the classification, leaving any further results
        // fresh. The emitted IR `Call` produces all `nres` result values regardless of the abstract
        // binding, so a value slot is consumed for *every* result (result 0's slot is discarded when it
        // is bound to a constant) — otherwise `rnext` desyncs from the residual's value numbering and
        // later operands in the block reference the wrong value.
        let bind = |first: Abs, rnext: &mut u32| -> Vec<Abs> {
            (0..nres)
                .map(|k| {
                    let slot = Abs::Dyn(bump(rnext));
                    if k == 0 {
                        first
                    } else {
                        slot
                    }
                })
                .collect()
        };
        if let Some(site) = lua_site {
            // Lua callee: bind newci and install L->ci, pin the callee-frame structural cells; keep the
            // register cells folded (no reload).
            let ci = site.callee_ci as i64;
            mem.insert(pm.l_ci_addr, (8, Abs::Const(Known::I64(ci))));
            for &(addr, val) in &site.pins {
                mem.insert(addr, (8, Abs::Const(Known::I64(val as i64))));
            }
            Ok(bind(Abs::Const(Known::I64(ci)), rnext))
        } else if is_c {
            // C callee: result is NULL; reload the register cells (the C function ran).
            let r = bind(Abs::Const(Known::I64(0)), rnext);
            self.reload_cells_natural(mem, out, rnext)?;
            Ok(r)
        } else {
            // Unrecognized: plain touch-state (unknown result, reload).
            let r = (0..nres).map(|_| Abs::Dyn(bump(rnext))).collect();
            self.reload_cells_natural(mem, out, rnext)?;
            Ok(r)
        }
    }

    /// Emit a `luaD_pretailcall` cut with the **tail-call post-state model** ([`PretailcallModel`]).
    /// The call is emitted opaquely (spill first, like every state-touching cut); the model only
    /// chooses the abstract post-state:
    /// - **Lua site** (`ra` in [`PretailcallModel::sites`]): the frame was *reused* — the callee
    ///   closure and its arguments were moved down onto it. Result 0 binds **negative** so the
    ///   `n < 0` branch folds to the Lua re-dispatch edge; `L->ci` is pinned to the (unchanged)
    ///   frame node and the site's pins install the moved closure slot and the callee `savedpc`.
    /// - **Unrecognized `ra`**: plain touch-state (unknown result, full reload).
    #[allow(clippy::too_many_arguments)]
    fn emit_pretailcall(
        &self,
        ridx: u32,
        callee: u32,
        args_abs: &[Abs],
        tm: &PretailcallModel,
        mem: &mut BTreeMap<u64, (u32, Abs)>,
        out: &mut Vec<Inst>,
        rnext: &mut u32,
    ) -> Result<Vec<Abs>, SpecError> {
        let ra = match args_abs.get(tm.ra_arg) {
            Some(Abs::Const(k)) => k.as_i64().map(|v| v as u64),
            _ => None,
        };
        let site = ra.and_then(|ra| tm.sites.iter().find(|s| s.ra == ra));
        // Spill so the opaque cut observes the current state, then emit the call.
        self.write_back_cells(mem, out, rnext)?;
        let args: Vec<u32> = args_abs
            .iter()
            .map(|&a| materialize(a, out, rnext))
            .collect();
        out.push(Inst::Call { func: ridx, args });
        let nres = self.module.funcs[callee as usize].results.len();
        if let Some(site) = site {
            // The reused frame: L->ci unchanged (re-pinned for robustness), moved slots pinned.
            mem.insert(
                tm.l_ci_addr,
                (8, Abs::Const(Known::I64(site.callee_ci as i64))),
            );
            for &(addr, val) in &site.pins {
                mem.insert(addr, (8, Abs::Const(Known::I64(val as i64))));
            }
            // A value slot is consumed for every result (result 0's is discarded for the constant)
            // so `rnext` stays in sync with the residual's value numbering — and the results must be
            // bound BEFORE any further emission (the call's values precede the reloads' values).
            let results: Vec<Abs> = (0..nres)
                .map(|k| {
                    let slot = Abs::Dyn(bump(rnext));
                    if k == 0 {
                        Abs::Const(Known::I32(-1))
                    } else {
                        slot
                    }
                })
                .collect();
            // The moved arguments: the cut relocated runtime values onto the frame's arg slots, so
            // the fold's stale cells there are WRONG — reload each value dynamic and pin its tag
            // (the selective-reload discipline applied at the call-in side).
            for &(addr, tag) in &site.args {
                self.reload_one(addr, 8, mem, out, rnext);
                mem.insert(addr + tm.tag_off, (1, Abs::Const(Known::I32(tag as i32))));
            }
            Ok(results)
        } else {
            let r = (0..nres).map(|_| Abs::Dyn(bump(rnext))).collect();
            self.reload_cells_natural(mem, out, rnext)?;
            Ok(r)
        }
    }

    /// Emit a `luaD_poscall` cut with the **return post-state model** ([`PoscallModel`]). The call is
    /// emitted opaquely and the register cells reload (the callee's results are written down into the
    /// caller's frame — fresh unknowns), but `L->ci` is then re-pinned to the **caller** frame: the
    /// returning frame's `ci` is a constant here (`L->ci` was pinned to it at the matching `precall`),
    /// so the caller `ci = *(ci->previous)` folds and the shared post-return block resumes the caller's
    /// dispatch (or exits `luaV_execute` at the entry frame — its `callstatus` is a constant overlay).
    /// If the returning `ci` isn't a constant, fall back to plain touch-state (the caller can't be
    /// pinned, so the return still bails at the post-return block if that is a deopt target).
    #[allow(clippy::too_many_arguments)]
    fn emit_poscall(
        &self,
        ridx: u32,
        callee: u32,
        args_abs: &[Abs],
        pm: &PrecallModel,
        po: &PoscallModel,
        mem: &mut BTreeMap<u64, (u32, Abs)>,
        out: &mut Vec<Inst>,
        rnext: &mut u32,
    ) -> Result<Vec<Abs>, SpecError> {
        // The returning frame's `ci` — pinned to a constant at the matching precall (else fall back).
        let cur_ci = mem
            .get(&pm.l_ci_addr)
            .and_then(|&(_, a)| match a {
                Abs::Const(k) => k.as_i64().map(|v| v as u64),
                Abs::Dyn(_) => None,
            })
            .or_else(|| {
                read_const_mem(self.config, self.module, pm.l_ci_addr, 0, LoadOp::I64)?
                    .as_i64()
                    .map(|v| v as u64)
            });
        let caller_ci = cur_ci.and_then(|ci| {
            read_const_mem(
                self.config,
                self.module,
                ci,
                po.ci_previous_off,
                LoadOp::I64,
            )?
            .as_i64()
            .map(|v| v as u64)
        });

        // Is this a selective-reload return (result value dynamic, tag pinned, other cells folded)?
        let selective = cur_ci.and_then(|ci| {
            po.selective
                .iter()
                .find(|(c, _)| *c == ci)
                .map(|&(_, tag)| tag)
        });

        // Spill so the opaque poscall observes the current state, then emit the call.
        self.write_back_cells(mem, out, rnext)?;
        let args: Vec<u32> = args_abs
            .iter()
            .map(|&a| materialize(a, out, rnext))
            .collect();
        out.push(Inst::Call { func: ridx, args });
        let nres = self.module.funcs[callee as usize].results.len();
        let results: Vec<Abs> = (0..nres).map(|_| Abs::Dyn(bump(rnext))).collect();

        match (selective, cur_ci) {
            (Some(tag), Some(ci)) => {
                // Selective: reload only the result value cell (at ci->func) as a fresh unknown and pin
                // its tag to the promised constant; leave every other caller cell folded so the loop
                // rolls. `reload_one` emits the load and updates the abstract cell.
                let func_slot =
                    read_const_mem(self.config, self.module, ci, po.ci_func_off, LoadOp::I64)
                        .and_then(|k| k.as_i64())
                        .map(|v| v as u64);
                if let Some(func_slot) = func_slot {
                    self.reload_one(func_slot, 8, mem, out, rnext);
                    mem.insert(
                        func_slot + po.tag_off,
                        (1, Abs::Const(Known::I32(tag as i32))),
                    );
                }
            }
            _ => {
                // Non-rolling: the callee wrote results down through memory — reload every cell.
                self.reload_cells_natural(mem, out, rnext)?;
            }
        }
        // Re-pin L->ci to the caller frame so folding continues there.
        if let Some(caller) = caller_ci {
            mem.insert(pm.l_ci_addr, (8, Abs::Const(Known::I64(caller as i64))));
        }
        Ok(results)
    }

    /// Reload a single rename cell from the window at `width`, replacing its abstract value with a fresh
    /// unknown backed by the emitted `load` (a one-cell [`Self::reload_cells_natural`]).
    fn reload_one(
        &self,
        eff: u64,
        width: u32,
        mem: &mut BTreeMap<u64, (u32, Abs)>,
        out: &mut Vec<Inst>,
        rnext: &mut u32,
    ) {
        let op = reload_load_op(width).expect("natural reload width");
        out.push(Inst::ConstI64(eff as i64));
        let addr = bump(rnext);
        out.push(Inst::Load {
            op,
            addr,
            offset: 0,
        });
        mem.insert(eff, (width, Abs::Dyn(bump(rnext))));
    }

    /// Reload every rename cell from the window at its canonical width (`i32.load8_u`/`load16_u`/
    /// `i32.load`/`i64.load` for 1/2/4/8), replacing its abstract value with a fresh unknown backed by
    /// the `load` — the inverse of the spill ([`Self::write_back_cells`]), run after an opaque
    /// state-touching call so its writes become visible. A narrow cell reloads as its `i32`-canonical
    /// zero-extended bytes (the same shape [`Self::eval_store`] records), so a later same-width load
    /// reads it back. Non-1/2/4/8 widths fail closed.
    fn reload_cells_natural(
        &self,
        mem: &mut BTreeMap<u64, (u32, Abs)>,
        out: &mut Vec<Inst>,
        rnext: &mut u32,
    ) -> Result<(), SpecError> {
        let addrs: Vec<(u64, u32)> = mem.iter().map(|(&a, &(w, _))| (a, w)).collect();
        for (eff, width) in addrs {
            let op = reload_load_op(width).ok_or(SpecError::Unsupported)?;
            out.push(Inst::ConstI64(eff as i64));
            let addr = bump(rnext);
            out.push(Inst::Load {
                op,
                addr,
                offset: 0,
            });
            mem.insert(eff, (width, Abs::Dyn(bump(rnext))));
        }
        Ok(())
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
                    .ok_or_else(|| {
                        trace_unsup!(
                            "call_indirect unresolved ty={:?} idx_const={:?}",
                            ty,
                            match env[*idx as usize] {
                                Abs::Const(k) => Some(k.as_i32()),
                                Abs::Dyn(_) => None,
                            }
                        );
                        SpecError::Unsupported
                    })?;
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

            Inst::Load { op, addr, offset } => {
                return self.eval_load(op, addr, offset, env, mem, out, rnext)
            }
            Inst::Store {
                op,
                addr,
                value,
                offset,
            } => return self.eval_store(op, addr, value, offset, env, mem, out, rnext),
            Inst::MemCopy { dst, src, len } => {
                return self.eval_mem_copy(dst, src, len, env, mem, out, rnext)
            }
            Inst::MemFill { dst, val, len } => {
                return self.eval_mem_fill(dst, val, len, env, mem, out, rnext)
            }

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
                let abs = emit_residual_pure(inst, env, out, rnext).ok_or_else(|| {
                    trace_unsup!("eval_inst fallthrough: {:?}", inst);
                    SpecError::Unsupported
                })?;
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
        env: &[Abs],
        mem: &BTreeMap<u64, (u32, Abs)>,
        out: &mut Vec<Inst>,
        rnext: &mut u32,
    ) -> Result<Option<Abs>, SpecError> {
        let width = op.info().2 as u64;
        if let Abs::Const(Known::I64(base)) = env[addr as usize] {
            let base = base as u64;
            let eff = base.wrapping_add(offset);
            if within_region(&self.regions, eff, width) {
                // The renameable region resolves entirely abstractly. Integer *and constant float*
                // cells work (a float's raw bits are its content — see [`load_bits`]); only a
                // *dynamic* float cell can't, which the store side never creates.
                // An exact cell — same address *and* width — resolves directly. A constant cell's
                // raw bytes are reconstructed per this load op (an `i8` cell loaded `*_u`/`*_s`
                // zero-/sign-extends; an `f64` cell loaded `f64` reinterprets its bits); a dynamic
                // cell is renamed only at its full natural width, where loading it back is identity.
                if let Some(&(wc, val)) = mem.get(&eff) {
                    if wc as u64 == width {
                        return Ok(Some(match val {
                            Abs::Const(k) => {
                                let raw = known_raw(k, width);
                                Abs::Const(load_bits(raw, op).ok_or(SpecError::Unsupported)?)
                            }
                            Abs::Dyn(i) if is_full_natural_load(op, width) => Abs::Dyn(i),
                            // A dynamic cell (held as integer bits) loaded as a **float**: reinterpret
                            // back to the requested float type — the inverse of the store-side cast.
                            Abs::Dyn(i) if is_full_natural_float(op.info().1, width) => {
                                let cast = if width == 8 {
                                    CastOp::ReinterpI64F64
                                } else {
                                    CastOp::ReinterpI32F32
                                };
                                out.push(Inst::Cast { op: cast, a: i });
                                Abs::Dyn(bump(rnext))
                            }
                            // A **narrow** dynamic cell (sub-natural width, `i32`-canonical from the
                            // store side) read back at its own width by an **unsigned** integer load:
                            // the cell already holds the zero-extended low `width` bytes, so the read
                            // is the identity (an `i32` result) or a zero-extend (an `i64` result).
                            // Signed sub-word reads and float reads stay refused — the interpreter's
                            // tag/field moves that reach here are unsigned.
                            Abs::Dyn(i) => {
                                let (_, vt, _w, signed) = op.info();
                                if signed || !matches!(vt, ValType::I32 | ValType::I64) {
                                    trace_unsup!(
                                        "load: unrenamable dynamic cell eff={:#x} op={:?}",
                                        eff,
                                        op
                                    );
                                    return Err(SpecError::Unsupported);
                                }
                                match vt {
                                    ValType::I32 => Abs::Dyn(i),
                                    _ => {
                                        out.push(Inst::Convert {
                                            op: ConvOp::ExtendI32U,
                                            a: i,
                                        });
                                        Abs::Dyn(bump(rnext))
                                    }
                                }
                            }
                        }));
                    }
                }
                // Anything else touching the cell (a different-width or straddling access) can't be
                // resolved abstractly without composing bytes — refuse rather than guess.
                if mem
                    .iter()
                    .any(|(&b, &(wc, _))| b < eff + width && eff < b + wc as u64)
                {
                    trace_unsup!(
                        "load: straddling/overlap in rename region eff={:#x} w={}",
                        eff,
                        width
                    );
                    return Err(SpecError::Unsupported);
                }
                // Untouched region cell. If the caller declared the region live-backed by the
                // constant image, read its seed from the constant-memory sources (overlay / promised
                // const region / readonly data) so captured VM state folds; otherwise it is the
                // zero-initialized scratch backing. A *written* cell was resolved above, so the seed
                // is only ever the region's pre-call value — writes correctly shadow it.
                if self.config.rename_seed_from_image {
                    if let Some(k) = read_const_mem(self.config, self.module, base, offset, op) {
                        return Ok(Some(Abs::Const(k)));
                    }
                }
                return Ok(Some(Abs::Const(
                    load_bits(0, op).ok_or(SpecError::Unsupported)?,
                )));
            }
            // Outside the region: a readonly constant-memory read folds; otherwise residual.
            if let Some(k) = read_const_mem(self.config, self.module, base, offset, op) {
                return Ok(Some(Abs::Const(k)));
            }
            let addr = materialize(env[addr as usize], out, rnext);
            out.push(Inst::Load { op, addr, offset });
            return Ok(Some(Abs::Dyn(bump(rnext))));
        }
        // Dynamic address: with a region active it might alias the renamed stack, so refuse —
        // unless the caller has promised the region is private to the renamed accesses.
        if !self.regions.is_empty() && !self.config.rename_is_private {
            return Err(SpecError::Unsupported);
        }
        let addr = materialize(env[addr as usize], out, rnext);
        out.push(Inst::Load { op, addr, offset });
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
        env: &[Abs],
        mem: &mut BTreeMap<u64, (u32, Abs)>,
        out: &mut Vec<Inst>,
        rnext: &mut u32,
    ) -> Result<Option<Abs>, SpecError> {
        let width = store_width(op) as u64;
        if let Abs::Const(Known::I64(base)) = env[addr as usize] {
            let base = base as u64;
            let eff = base.wrapping_add(offset);
            if within_region(&self.regions, eff, width) {
                let is_int = matches!(op.info().1, ValType::I32 | ValType::I64);
                let cell = match env[value as usize] {
                    // A constant is kept as the cell's raw bytes (truncated to the store width), so a
                    // later load re-extends it correctly (an `i8` store of `0x1FF` ⇒ `0xFF`). This
                    // covers a **constant float** store too — its bits *are* its memory content — so a
                    // `TValue` value the interpreter moves via an `f64` load/store renames like any
                    // 8-byte cell.
                    Abs::Const(k) => Abs::Const(cell_const(known_raw(k, width), width)),
                    // A dynamic integer value renames directly at its full natural width (loading it
                    // back at that width is the identity).
                    Abs::Dyn(i) if is_int && is_full_natural_store(op, width) => Abs::Dyn(i),
                    // A dynamic full-natural **float** value (a `TValue` value moved via `f64`/`f32`):
                    // reinterpret its bits to a same-width integer so the cell is uniformly
                    // integer-typed; a later float load reinterprets back. This lets a rolled loop
                    // carry a dynamic register through the `TValue` value union.
                    Abs::Dyn(i) if is_full_natural_float(op.info().1, width) => {
                        let cast = if width == 8 {
                            CastOp::ReinterpF64I64
                        } else {
                            CastOp::ReinterpF32I32
                        };
                        out.push(Inst::Cast { op: cast, a: i });
                        Abs::Dyn(bump(rnext))
                    }
                    // A **narrow** dynamic integer store (a sub-natural width — the `TValue` tag byte
                    // the interpreter moves via `i32.store8`, or a 2-byte field): canonicalize the
                    // value to an `i32` holding the **zero-extended low `width` bytes** — the cell's
                    // physical content (matching `cell_const`/`writeback_store_op`, where a sub-8 cell
                    // is `i32`). A same-width load reads it back exactly (`eval_load`); a wider or
                    // overlapping access is refused by the straddle guard below (`mem.retain` +
                    // `within_region`/straddle checks) — which is precisely the masking-soundness
                    // condition (no other access can observe the untruncated high bits). This is what
                    // lets the renamer carry a **dynamic tagged value** (a looked-up function/table/
                    // string), whose tag is dynamic, instead of refusing every non-integer move.
                    Abs::Dyn(i) if is_int && width < 8 => {
                        let src32 = if matches!(op.info().1, ValType::I32) {
                            i
                        } else {
                            out.push(Inst::Convert {
                                op: ConvOp::WrapI64,
                                a: i,
                            });
                            bump(rnext)
                        };
                        // Widths 1/2 mask to the low bytes; width 4 already *is* the full i32.
                        let masked = if width >= 4 {
                            src32
                        } else {
                            out.push(Inst::ConstI32(((1i64 << (8 * width)) - 1) as i32));
                            let m = bump(rnext);
                            out.push(Inst::IntBin {
                                ty: IntTy::I32,
                                op: BinOp::And,
                                a: src32,
                                b: m,
                            });
                            bump(rnext)
                        };
                        Abs::Dyn(masked)
                    }
                    // A narrow dynamic **float** store (no natural interpreter shape) — refuse.
                    Abs::Dyn(_) => {
                        trace_unsup!("store: unrenamable dynamic cell eff={:#x} op={:?}", eff, op);
                        return Err(SpecError::Unsupported);
                    }
                };
                // Invalidate any overlapping cell, then record this one. No residual store.
                mem.retain(|&b, &mut (wc, _)| !(b < eff + width && eff < b + wc as u64));
                mem.insert(eff, (width as u32, cell));
                return Ok(None);
            }
            if disjoint_from_region(&self.regions, eff, width) {
                let addr = materialize(env[addr as usize], out, rnext);
                let value = materialize(env[value as usize], out, rnext);
                out.push(Inst::Store {
                    op,
                    addr,
                    value,
                    offset,
                });
                return Ok(None);
            }
            return Err(SpecError::Unsupported); // straddles the region boundary
        }
        // Dynamic address: with a region active it might alias the renamed stack, so refuse —
        // unless the caller has promised the region is private to the renamed accesses.
        if !self.regions.is_empty() && !self.config.rename_is_private {
            return Err(SpecError::Unsupported);
        }
        let addr = materialize(env[addr as usize], out, rnext);
        let value = materialize(env[value as usize], out, rnext);
        out.push(Inst::Store {
            op,
            addr,
            value,
            offset,
        });
        Ok(None)
    }

    /// A `memory.copy` of a **constant** length between **constant** addresses. When source and
    /// destination both lie fully inside the renameable region, model it as an abstract **cell copy**
    /// — Lua's 16-byte `TValue` struct moves (`MOVE`, loop-control) lower to `llvm.memcpy`, so this is
    /// the hinge that lets `luaV_execute` fold. Every source cell is shifted to the destination and
    /// the rest of the destination span is invalidated, so untouched bytes read back as the region's
    /// zero-init — matching a real copy of the source's uninitialized bytes. Both-disjoint copies are
    /// emitted residually; a dynamic length/address is residual only when the region is private (else
    /// it could alias renamed cells and is refused).
    #[allow(clippy::too_many_arguments)]
    fn eval_mem_copy(
        &self,
        dst: u32,
        src: u32,
        len: u32,
        env: &[Abs],
        mem: &mut BTreeMap<u64, (u32, Abs)>,
        out: &mut Vec<Inst>,
        rnext: &mut u32,
    ) -> Result<Option<Abs>, SpecError> {
        let consts = match (env[dst as usize], env[src as usize], env[len as usize]) {
            (Abs::Const(Known::I64(d)), Abs::Const(Known::I64(s)), Abs::Const(Known::I64(l))) => {
                Some((d as u64, s as u64, l as u64))
            }
            _ => None,
        };
        let emit_residual = |out: &mut Vec<Inst>, rnext: &mut u32| {
            let dst = materialize(env[dst as usize], out, rnext);
            let src = materialize(env[src as usize], out, rnext);
            let len = materialize(env[len as usize], out, rnext);
            out.push(Inst::MemCopy { dst, src, len });
        };
        let (da, sa, ln) = match consts {
            Some(t) => t,
            None => {
                // A dynamic span might alias the renamed region unless the caller promised it private.
                if !self.regions.is_empty() && !self.config.rename_is_private {
                    trace_unsup!("mem_copy: dynamic span with a non-private rename region");
                    return Err(SpecError::Unsupported);
                }
                emit_residual(out, rnext);
                return Ok(None);
            }
        };
        if ln == 0 {
            return Ok(None);
        }
        let region = self.regions.as_slice();
        if within_region(region, da, ln) && within_region(region, sa, ln) {
            // A source cell straddling the span boundary can't be split abstractly.
            if mem.iter().any(|(&a, &(w, _))| {
                a < sa + ln && sa < a + w as u64 && !(a >= sa && a + w as u64 <= sa + ln)
            }) {
                trace_unsup!("mem_copy: a source cell straddles the span");
                return Err(SpecError::Unsupported);
            }
            let moved: Vec<(u64, u32, Abs)> = mem
                .iter()
                .filter(|(&a, &(w, _))| a >= sa && a + w as u64 <= sa + ln)
                .map(|(&a, &(w, v))| (da + (a - sa), w, v))
                .collect();
            mem.retain(|&a, &mut (w, _)| !(a < da + ln && da < a + w as u64));
            for (a, w, v) in moved {
                mem.insert(a, (w, v));
            }
            return Ok(None);
        }
        if disjoint_from_region(region, da, ln) && disjoint_from_region(region, sa, ln) {
            emit_residual(out, rnext);
            return Ok(None);
        }
        trace_unsup!(
            "mem_copy: mixed/straddle da={:#x} sa={:#x} len={}",
            da,
            sa,
            ln
        );
        Err(SpecError::Unsupported)
    }

    /// A `memory.fill` of a **constant** length. A **zero** fill inside the renameable region simply
    /// invalidates the span (untouched cells read back as the zero-init); a fill disjoint from the
    /// region is residual. Other cases (non-zero fill into the region, straddle) are not modeled yet.
    #[allow(clippy::too_many_arguments)]
    fn eval_mem_fill(
        &self,
        dst: u32,
        val: u32,
        len: u32,
        env: &[Abs],
        mem: &mut BTreeMap<u64, (u32, Abs)>,
        out: &mut Vec<Inst>,
        rnext: &mut u32,
    ) -> Result<Option<Abs>, SpecError> {
        let consts = match (env[dst as usize], env[val as usize], env[len as usize]) {
            (Abs::Const(Known::I64(d)), Abs::Const(v), Abs::Const(Known::I64(l))) => {
                Some((d as u64, v.as_i32().unwrap_or(1), l as u64))
            }
            _ => None,
        };
        let emit_residual = |out: &mut Vec<Inst>, rnext: &mut u32| {
            let dst = materialize(env[dst as usize], out, rnext);
            let val = materialize(env[val as usize], out, rnext);
            let len = materialize(env[len as usize], out, rnext);
            out.push(Inst::MemFill { dst, val, len });
        };
        let (da, byte, ln) = match consts {
            Some(t) => t,
            None => {
                if !self.regions.is_empty() && !self.config.rename_is_private {
                    trace_unsup!("mem_fill: dynamic span with a non-private rename region");
                    return Err(SpecError::Unsupported);
                }
                emit_residual(out, rnext);
                return Ok(None);
            }
        };
        if ln == 0 {
            return Ok(None);
        }
        let region = self.regions.as_slice();
        if within_region(region, da, ln) {
            if byte != 0 {
                trace_unsup!("mem_fill: non-zero fill into the rename region byte={byte}");
                return Err(SpecError::Unsupported);
            }
            mem.retain(|&a, &mut (w, _)| !(a < da + ln && da < a + w as u64));
            return Ok(None);
        }
        if disjoint_from_region(region, da, ln) {
            emit_residual(out, rnext);
            return Ok(None);
        }
        trace_unsup!("mem_fill: straddle da={:#x} len={}", da, ln);
        Err(SpecError::Unsupported)
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
        active_block: u32,
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
                let d = self.is_deopt_edge(func, active_block, *target);
                let stack = succ_stack(&outer, func, *target, args, env, active_entry.clone());
                let (target, args) = self.branch_to(&stack, mem, d);
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
                    let d = self.is_deopt_edge(func, active_block, blk);
                    let stack = succ_stack(&outer, func, blk, args, env, active_entry.clone());
                    let (target, args) = self.branch_to(&stack, mem, d);
                    Terminator::Br { target, args }
                }
                // Dynamic condition: specialize both successors and keep the branch.
                Abs::Dyn(cond) => {
                    let then_d = self.is_deopt_edge(func, active_block, *then_blk);
                    let else_d = self.is_deopt_edge(func, active_block, *else_blk);
                    let then_stack = succ_stack(
                        &outer,
                        func,
                        *then_blk,
                        then_args,
                        env,
                        active_entry.clone(),
                    );
                    let (then_blk, then_args) = self.branch_to(&then_stack, mem, then_d);
                    let else_stack = succ_stack(
                        &outer,
                        func,
                        *else_blk,
                        else_args,
                        env,
                        active_entry.clone(),
                    );
                    let (else_blk, else_args) = self.branch_to(&else_stack, mem, else_d);
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
                    let d = self.is_deopt_edge(func, active_block, *blk);
                    let stack = succ_stack(&outer, func, *blk, args, env, active_entry.clone());
                    let (target, args) = self.branch_to(&stack, mem, d);
                    Terminator::Br { target, args }
                }
                Abs::Dyn(idx) => {
                    let targets = targets
                        .iter()
                        .map(|(blk, args)| {
                            let d = self.is_deopt_edge(func, active_block, *blk);
                            let stack =
                                succ_stack(&outer, func, *blk, args, env, active_entry.clone());
                            self.branch_to(&stack, mem, d)
                        })
                        .collect();
                    let default_d = self.is_deopt_edge(func, active_block, default.0);
                    let default_stack = succ_stack(
                        &outer,
                        func,
                        default.0,
                        &default.1,
                        env,
                        active_entry.clone(),
                    );
                    let default = self.branch_to(&default_stack, mem, default_d);
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
        // Cut set: a `return runtime_callout(...)` tail-bail (a common shape — e.g. an interpreter
        // checkpointing its state and tail-calling a resume/slow-path routine). Emit it verbatim as a
        // residual `return_call` to the carried callee, exactly like an opaque `call` — never inlined
        // or folded. State-touching cut callees would need the region spilled here, which the tail
        // position (no return point to reload at) can't do, so only the plain opaque cut set applies.
        if self.config.cut_calls.contains(&callee) {
            let ridx = self.cut[&callee];
            let args: Vec<u32> = args_abs
                .iter()
                .map(|&a| materialize(a, out, rnext))
                .collect();
            return Ok(Terminator::ReturnCall { func: ridx, args });
        }
        if let Some(state) = self.outline {
            // Threading the renamed region across a *tail* call (where the callee's results become
            // this function's results) isn't supported — there's no return point to append this
            // function's own out-cells at. Fail closed; a rename region forces it.
            let outline_tail = |args_abs: &[Abs]| -> Result<Terminator, SpecError> {
                if !self.regions.is_empty() {
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
            let (target, args) = self.branch_to(&stack, mem, false);
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
                let (target, args) = self.branch_to(&stack, mem, false);
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
                // Write-back: a live-backed rename region (`rename_seed_from_image`) aliases real,
                // persistent memory, so before the residual's true exit spill every written cell to
                // the window — otherwise the caller (e.g. Lua's `poscall` reading `L->top` and the
                // return registers) would see stale pre-call bytes. Only at the entry's exit
                // (`!thread_cells`); outlined callees thread their cells out as results instead, and
                // those flow back into this `mem` before the entry returns. Untouched (seed-only)
                // cells never entered `mem`, so their memory already holds the correct seed.
                if self.config.rename_seed_from_image && !self.thread_cells {
                    self.write_back_cells(mem, out, rnext)?;
                }
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
                let (target, args) = self.branch_to(&outer, mem, false);
                Terminator::Br { target, args }
            }
        })
    }

    /// Spill the live abstract cells to the window as residual stores (write-back for a live-backed
    /// rename region). Each written cell `(eff, width, val)` becomes `store.<width> eff, val`; a width
    /// with no natural store op fails closed. Emitted in address order for a deterministic residual.
    fn write_back_cells(
        &self,
        mem: &BTreeMap<u64, (u32, Abs)>,
        out: &mut Vec<Inst>,
        rnext: &mut u32,
    ) -> Result<(), SpecError> {
        for (&eff, &(width, val)) in mem.iter() {
            let op = writeback_store_op(width).ok_or_else(|| {
                trace_unsup!(
                    "write_back_cells: no store op for width {} @ {:#x}",
                    width,
                    eff
                );
                SpecError::Unsupported
            })?;
            out.push(Inst::ConstI64(eff as i64));
            let addr = bump(rnext);
            let value = materialize(val, out, rnext);
            out.push(Inst::Store {
                op,
                addr,
                value,
                offset: 0,
            });
        }
        Ok(())
    }

    /// The argument list for a deopt tail-call to the resume handler. Empty for a `() -> R` handler;
    /// for a `<entry params> -> R` handler, the entry's arguments in order — constant lanes
    /// materialized as `const` insts, dynamic lanes taken from the threaded channel
    /// ([`Spec::cur_thread`], set for the current block in `build_block`).
    fn deopt_handler_args(&self, out: &mut Vec<Inst>, rnext: &mut u32) -> Vec<u32> {
        if !self.deopt_pass_args {
            return Vec::new();
        }
        let mut thread = self.cur_thread.iter();
        let mut args = Vec::with_capacity(self.entry_pattern.len());
        for slot in self.entry_pattern {
            let abs = match slot {
                Some(k) => Abs::Const(*k),
                None => *thread
                    .next()
                    .expect("one threaded value per dynamic entry param"),
            };
            args.push(materialize(abs, out, rnext));
        }
        args
    }

    /// Resolve one outgoing edge into a residual block id + dynamic arguments. The successor inherits
    /// the full call stack and the current abstract memory; constant lanes join the context, dynamic
    /// lanes are passed as residual block arguments in the canonical order (frames outermost→
    /// innermost, each frame's dynamic env slots in order, then dynamic memory cells by address —
    /// matching [`Self::build_block`]'s parameter declaration).
    /// Whether a branch from `(func, block)` to `target` is a cold **deopt edge**
    /// ([`SpecConfig::deopt_edges`]): the successor is interned as a deopt task (spill + resume the
    /// baseline) rather than projected — bounding a divergent target (e.g. a dynamic-callee
    /// re-dispatch) without deopting the shared block on its other, hot, in-edges. Only with a handler.
    fn is_deopt_edge(&self, func: u32, block: u32, target: u32) -> bool {
        self.deopt_handler.is_some()
            && self
                .config
                .deopt_edges
                .iter()
                .any(|&(f, b, t)| f == func && b == block && t == target)
    }

    fn branch_to(
        &mut self,
        stack: &[FrameAbs],
        mem: &BTreeMap<u64, (u32, Abs)>,
        deopt: bool,
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
        // Forward the threaded entry arguments (deopt arg passing) as the edge's trailing arguments,
        // after the memory cells — the fixed position `build_block` reconstructs them at. They are
        // always dynamic, so they add no memo diversity; the successor is never the entry seed.
        for &a in &self.cur_thread {
            dyn_args.push(match a {
                Abs::Dyn(i) => i,
                // The threaded lanes are the entry's dynamic params (always `Dyn`); a `Const` here
                // would mean a constant entry arg leaked into the channel, which never happens.
                Abs::Const(_) => unreachable!("threaded entry args are always dynamic"),
            });
        }
        let id = self.intern(frames, mem_pat, false, deopt);
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
    // The canonical SSA type of a rename cell, matching `writeback_store_op` and the narrow-cell
    // handling in `eval_store`/`eval_load`: a full-width 8-byte cell is `i64`; every sub-8 cell
    // (1/2/4 bytes — including a dynamic `TValue` tag byte) is `i32` holding its zero-extended content.
    match width {
        8 => ValType::I64,
        _ => ValType::I32,
    }
}

/// Whether `[eff, eff+width)` lies fully inside the renameable region.
fn within_region(regions: &[(u64, u64)], eff: u64, width: u64) -> bool {
    let Some(end) = eff.checked_add(width) else {
        return false;
    };
    regions.iter().any(|&(lo, hi)| eff >= lo && end <= hi)
}

/// Whether `[eff, eff+width)` is entirely outside **every** renameable region (vacuously true if the
/// set is empty). An access that is neither within one region nor disjoint from all straddles a
/// region boundary — the caller treats that as unsupported.
fn disjoint_from_region(regions: &[(u64, u64)], eff: u64, width: u64) -> bool {
    match eff.checked_add(width) {
        Some(end) => regions.iter().all(|&(lo, hi)| end <= lo || eff >= hi),
        None => false,
    }
}

/// The source functions carried verbatim into the residual: every cut callee (opaque
/// [`SpecConfig::cut_calls`] and state-touching [`SpecConfig::cut_calls_touch_state`]) plus the deopt
/// handler ([`SpecConfig::deopt_handler`], carried only when [`SpecConfig::deopt_targets`] is used).
/// Deduplicated, in a deterministic order. Empty ⇒ the plain single-function inline path.
fn cut_roots(config: &SpecConfig) -> Vec<u32> {
    let mut roots: Vec<u32> = Vec::new();
    let mut push = |f: u32| {
        if !roots.contains(&f) {
            roots.push(f);
        }
    };
    for &f in &config.cut_calls {
        push(f);
    }
    for &f in &config.cut_calls_touch_state {
        push(f);
    }
    for &f in &config.cut_calls_read_state {
        push(f);
    }
    if !config.deopt_targets.is_empty() || !config.deopt_edges.is_empty() {
        if let Some(h) = config.deopt_handler {
            push(h);
        }
    }
    roots
}

/// Validate and resolve the deopt handler (see [`SpecConfig::deopt_handler`]) to its residual index.
/// `Ok(None)` when no deopt targets are configured. Otherwise a handler must be named, its signature
/// must be `() -> <entry results>` (a deopt tail-calls it with no args, forwarding its results), and it
/// must have been carried (it is — [`cut_roots`] includes it). Fails [`SpecError::Unsupported`] on a
/// missing/mis-typed handler, [`SpecError::BadFunc`] on an out-of-range index.
fn resolve_deopt_handler(
    module: &Module,
    config: &SpecConfig,
    entry: u32,
    cut: &BTreeMap<u32, u32>,
) -> Result<Option<(u32, bool)>, SpecError> {
    if config.deopt_targets.is_empty() && config.deopt_edges.is_empty() {
        return Ok(None);
    }
    let handler = config.deopt_handler.ok_or(SpecError::Unsupported)?;
    let hf = module
        .funcs
        .get(handler as usize)
        .ok_or(SpecError::BadFunc)?;
    let ef = module.funcs.get(entry as usize).ok_or(SpecError::BadFunc)?;
    if hf.results != ef.results {
        trace_unsup!("deopt handler results must match the entry's");
        return Err(SpecError::Unsupported);
    }
    // Two supported shapes: `() -> R` resumes purely from the written-back window; `P -> R` (same
    // params as the entry) also receives the entry's argument values, threaded to the deopt edge.
    let pass_args = if hf.params.is_empty() {
        false
    } else if hf.params == ef.params {
        true
    } else {
        trace_unsup!("deopt handler must be `() -> R` or `<entry params> -> R`");
        return Err(SpecError::Unsupported);
    };
    Ok(Some((cut[&handler], pass_args)))
}

/// Plan the cut-set carry (see [`SpecConfig::cut_calls`]). From the `roots` (the cut callees) walk the
/// transitive **direct**-call closure and give each carried source function a residual index: the
/// specialized entry is residual 0, and carried functions follow at 1.. in the deterministic
/// discovery order. Returns the `source index → residual index` map and the carried source indices in
/// that order (used to append them to the residual `funcs`).
///
/// Fails with [`SpecError::Unsupported`] if the closure reaches the entry (which is specialized, not
/// carried, so it can't also be a verbatim callee) or a function that dispatches through an indirect
/// call (the residual carries no table); [`SpecError::BadFunc`] on an out-of-range root.
fn plan_cut_carry(
    module: &Module,
    entry: u32,
    roots: &[u32],
) -> Result<(BTreeMap<u32, u32>, Vec<u32>), SpecError> {
    let mut carried: Vec<u32> = Vec::new();
    let mut seen: BTreeSet<u32> = BTreeSet::new();
    let mut work: VecDeque<u32> = VecDeque::new();
    for &r in roots {
        if r as usize >= module.funcs.len() {
            return Err(SpecError::BadFunc);
        }
        if seen.insert(r) {
            work.push_back(r);
        }
    }
    while let Some(fidx) = work.pop_front() {
        if fidx == entry {
            // The entry is residual function 0 (specialized); it cannot also be carried verbatim.
            return Err(SpecError::Unsupported);
        }
        carried.push(fidx);
        let callees = direct_callees(&module.funcs[fidx as usize]).inspect_err(|_e| {
            trace_unsup!(
                "plan_cut_carry: func {} has an indirect call (can't carry)",
                fidx
            );
        })?;
        for callee in callees {
            if callee as usize >= module.funcs.len() {
                return Err(SpecError::BadFunc);
            }
            if seen.insert(callee) {
                work.push_back(callee);
            }
        }
    }
    // Residual indices: entry is 0, carried functions follow in discovery order. Built with explicit
    // inserts rather than `.collect()` — `BTreeMap::from_iter` sorts its input via `slice::sort`, whose
    // driftsort helper the LLVM→IR on-ramp can't resolve (it would break the `peval_futamura` guest).
    let mut map = BTreeMap::new();
    for (i, &orig) in carried.iter().enumerate() {
        map.insert(orig, i as u32 + 1);
    }
    Ok((map, carried))
}

/// Whether `f` reaches a host boundary the standalone residual can't resolve — a `CallImport`,
/// `CallImportDyn`, `CapCall`, or `CallSym`. Used by [`SpecConfig::carry_whole_module`] to decide
/// which carried functions to replace with a trap stub (an unresolved import fails verification, and
/// these are the guest's I/O layer — off the allocator/GC path the cut closure actually runs).
fn imports_a_boundary(f: &Func) -> bool {
    f.blocks.iter().flat_map(|b| &b.insts).any(|i| {
        matches!(
            i,
            Inst::CallImport { .. }
                | Inst::CallImportDyn { .. }
                | Inst::CapCall { .. }
                | Inst::CallSym { .. }
        )
    })
}

/// A same-signature stub whose single block just traps — the placeholder for a carried function that
/// [`carry_whole_module`](SpecConfig::carry_whole_module) can't keep verbatim (see
/// [`imports_a_boundary`]). Filling the slot keeps every other function's index — and the identity
/// `call_indirect` table — unchanged; the body is unreachable on the cut closure's execution path.
fn trap_stub(f: &Func) -> Func {
    Func {
        params: f.params.clone(),
        results: f.results.clone(),
        blocks: vec![Block {
            params: f.params.clone(),
            insts: Vec::new(),
            term: Terminator::Unreachable,
        }],
    }
}

/// The direct-call callee indices referenced by `f` (`call` / `return_call`), in program order.
/// Fails with [`SpecError::Unsupported`] if `f` uses an indirect call — a carried function is emitted
/// verbatim, and the residual has no table to resolve one through.
fn direct_callees(f: &Func) -> Result<Vec<u32>, SpecError> {
    let mut out = Vec::new();
    for b in &f.blocks {
        for inst in &b.insts {
            match inst {
                Inst::Call { func, .. } => out.push(*func),
                Inst::CallIndirect { .. } => return Err(SpecError::Unsupported),
                _ => {}
            }
        }
        match &b.term {
            Terminator::ReturnCall { func, .. } => out.push(*func),
            Terminator::ReturnCallIndirect { .. } => return Err(SpecError::Unsupported),
            _ => {}
        }
    }
    Ok(out)
}

/// Carry a cut callee into the residual verbatim, rewriting its direct-call targets from source
/// indices to residual indices via `cut`. Every callee it references was included in the carry
/// closure, so each target is present in the map.
fn carry_func(f: &Func, cut: &BTreeMap<u32, u32>) -> Result<Func, SpecError> {
    let remap = |func: u32| -> Result<u32, SpecError> {
        cut.get(&func).copied().ok_or(SpecError::Unsupported)
    };
    let mut blocks = Vec::with_capacity(f.blocks.len());
    for b in &f.blocks {
        let mut insts = Vec::with_capacity(b.insts.len());
        for inst in &b.insts {
            match inst {
                Inst::Call { func, args } => insts.push(Inst::Call {
                    func: remap(*func)?,
                    args: args.clone(),
                }),
                other => insts.push(other.clone()),
            }
        }
        let term = match &b.term {
            Terminator::ReturnCall { func, args } => Terminator::ReturnCall {
                func: remap(*func)?,
                args: args.clone(),
            },
            other => other.clone(),
        };
        blocks.push(svm_ir::Block {
            params: b.params.clone(),
            insts,
            term,
        });
    }
    Ok(Func {
        params: f.params.clone(),
        results: f.results.clone(),
        blocks,
    })
}

/// The full rename region set: [`SpecConfig::rename`] then [`SpecConfig::rename_extra`].
fn all_regions(config: &SpecConfig) -> Vec<(u64, u64)> {
    config
        .rename
        .into_iter()
        .chain(config.rename_extra.iter().copied())
        .collect()
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

/// Whether a load/store's value type `vt` is a **full-natural float** at `width` — `f32`@4 / `f64`@8.
/// A dynamic value of this shape is renamed by reinterpreting its bits to the same-width integer (and
/// back on load), so the cell stays uniformly integer-typed.
fn is_full_natural_float(vt: ValType, width: u64) -> bool {
    matches!((vt, width), (ValType::F32, 4) | (ValType::F64, 8))
}

/// The load counterpart of [`is_full_natural_store`]: `i32.load`/`i64.load` read a full natural cell
/// back as the identity, so a dynamic cell may be returned directly.
fn is_full_natural_load(op: LoadOp, width: u64) -> bool {
    matches!((op, width), (LoadOp::I32, 4) | (LoadOp::I64, 8))
}

/// The store op that spills a renamed cell of `width` bytes back to the window (write-back). The op's
/// value type matches the cell's canonical type (`i32` for width < 8, `i64` for 8 — see
/// [`cell_const`]/[`cell_type`]): a width-8 cell holds an `i64`, everything narrower an `i32`, so the
/// low `width` bytes of that value are the cell's content. Returns `None` for an unsupported width.
fn writeback_store_op(width: u32) -> Option<StoreOp> {
    Some(match width {
        1 => StoreOp::I32_8,
        2 => StoreOp::I32_16,
        4 => StoreOp::I32,
        8 => StoreOp::I64,
        _ => return None,
    })
}

/// The full-width store op for a natural 4/8-byte cell — the spill side of a state-touching cut call
/// The reload op that reads a spilled cell back after a state-touching cut call
/// ([`Spec::reload_cells_natural`]), matching [`writeback_store_op`]'s canonical widths: a narrow cell
/// reloads **zero-extended** (`load8_u`/`load16_u`) as its `i32`-canonical bytes — the same shape
/// [`Spec::eval_store`] records — so a later same-width load reads it back. Non-1/2/4/8 fails closed.
fn reload_load_op(width: u32) -> Option<LoadOp> {
    Some(match width {
        1 => LoadOp::I32_8U,
        2 => LoadOp::I32_16U,
        4 => LoadOp::I32,
        8 => LoadOp::I64,
        _ => return None,
    })
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

/// Reconstruct the loaded constant from `width` raw little-endian bytes per load `op`. Integer ops
/// sign/zero-extend ([`extend_loaded`]); **float** ops reinterpret the bits (`f64`/`f32`), since a
/// float's memory content *is* its bit pattern. Lets a rename cell written by a float store — a
/// `TValue` value the interpreter moves via an `f64` load/store — read back as the right constant.
fn load_bits(raw: u64, op: LoadOp) -> Option<Known> {
    match op.info().1 {
        ValType::F64 => Some(Known::F64(raw)),
        ValType::F32 => Some(Known::F32(raw as u32)),
        _ => extend_loaded(raw, op),
    }
}

/// Read an integer or float load from constant memory. The effective address `base + offset` must lie
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
    let width = op.info().2;
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
    load_bits(raw, op)
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
