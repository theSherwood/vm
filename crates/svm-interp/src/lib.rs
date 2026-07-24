//! Reference interpreter — the **oracle** the JIT is differential-tested against
//! (`DESIGN.md` §18). It implements the IR's total semantics directly (§3b: every
//! op is a defined value or a defined trap — no UB).
//!
//! Robustness: the interpreter assumes a *verified* module, but must never panic
//! even on an unverified one (so it is safe to drive from a fuzzer). Any structural
//! surprise yields `Trap::Malformed` rather than an index panic. Runaway control
//! flow is bounded by `fuel` (a stand-in for §5 metering), so it always terminates.
#![forbid(unsafe_code)]

/// Phase-1b bytecode-dispatch engine (see `INTERP_PERF.md`) — a flat, operand-resolved execution
/// path, not yet the default; gated by the equality harness against this interpreter.
pub mod bytecode;

use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier, Condvar, Mutex, RwLock};
use std::time::{Duration, Instant};

use svm_ir::{
    AtomicRmwOp, BinOp, CastOp, CmpOp, ConvOp, Data, DebugInfo, FBinOp, FCmpOp, FToI, FUnOp,
    FloatTy, Func, FuncIdx, FuncType, IToF, Inst, IntTy, IntUnOp, LoadOp, Memory, Module, SsaLoc,
    StoreOp, Terminator, VBitBinOp, VCvtOp, VFCmpOp, VFloatBinOp, VFloatUnOp, VICmpOp, VIntBinOp,
    VIntUnOp, VNarrowOp, VPMinMaxOp, VSatBinOp, VShape, VShiftOp, VWidenOp, ValIdx, ValType,
    VarInfo, VarLoc, DEFAULT_RESERVED_LOG2,
};
use svm_mask::Window;
use svm_mem::RmwOp;
// Re-exported so an embedder can build a `Region::shared` over host memory (e.g. the wasm shared
// linear memory) and hand it to `compile_and_run_capture_over` — the parallel-wasm window backing.
// The `unsafe` of borrowing host memory lives in `svm_mem::Region::shared`, keeping this crate
// `#![forbid(unsafe_code)]`.
pub use svm_mem::Region;

/// A runtime value. Mirrors `ValType`.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Value {
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
    /// A `v128` SIMD vector (§17/D58): 16 raw little-endian bytes. Lane interpretation is
    /// per-op, never per-value — so the value carries only the bytes.
    V128([u8; 16]),
    /// An opaque 64-bit `ref` (GC.md §6 forward-compat reservation). Operationally identical to
    /// `I64` — a distinct variant only so type confusion is compile-caught; it carries raw bits.
    Ref(u64),
}

/// A raw value **slot** — the interpreter's in-frame storage for one live SSA value, replacing
/// the 24-byte tagged [`Value`] on the hot path. A scalar lives in `lo` as its bit pattern in the
/// low bits (mirroring the JIT / cap `val_to_slot` ABI, so `Reg::i64` equals `val_to_slot`); a
/// `v128` uses both words (little-endian: `lo` = bytes 0..8, `hi` = bytes 8..16). Reads are
/// **op-directed** — the executing instruction's static type says how to interpret the slot, so
/// there is no per-value tag to match and no `Result` to thread. `Value` stays the public type;
/// conversions happen only at the API / capability / debugger boundaries.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
struct Reg {
    lo: u64,
    hi: u64,
}

impl Reg {
    #[inline]
    fn from_i32(x: i32) -> Reg {
        Reg {
            lo: x as i64 as u64,
            hi: 0,
        } // sign-extend, matching `val_to_slot`
    }
    #[inline]
    fn from_i64(x: i64) -> Reg {
        Reg {
            lo: x as u64,
            hi: 0,
        }
    }
    #[inline]
    fn from_f32(x: f32) -> Reg {
        Reg {
            lo: x.to_bits() as u64,
            hi: 0,
        }
    }
    #[inline]
    fn from_f64(x: f64) -> Reg {
        Reg {
            lo: x.to_bits(),
            hi: 0,
        }
    }
    #[inline]
    fn from_v128(b: [u8; 16]) -> Reg {
        Reg {
            lo: u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]),
            hi: u64::from_le_bytes([b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]]),
        }
    }
    #[inline]
    fn i32(self) -> i32 {
        self.lo as i32
    }
    #[inline]
    fn i64(self) -> i64 {
        self.lo as i64
    }
    #[inline]
    fn f32(self) -> f32 {
        f32::from_bits(self.lo as u32)
    }
    #[inline]
    fn f64(self) -> f64 {
        f64::from_bits(self.lo)
    }
    #[inline]
    fn v128(self) -> [u8; 16] {
        let lo = self.lo.to_le_bytes();
        let hi = self.hi.to_le_bytes();
        [
            lo[0], lo[1], lo[2], lo[3], lo[4], lo[5], lo[6], lo[7], hi[0], hi[1], hi[2], hi[3],
            hi[4], hi[5], hi[6], hi[7],
        ]
    }
    /// Boundary in: pack a typed [`Value`] (entry args, host results) into a slot.
    #[inline]
    fn from_value(v: Value) -> Reg {
        match v {
            Value::I32(x) => Reg::from_i32(x),
            Value::I64(x) => Reg::from_i64(x),
            Value::F32(x) => Reg::from_f32(x),
            Value::F64(x) => Reg::from_f64(x),
            Value::V128(b) => Reg::from_v128(b),
            Value::Ref(x) => Reg { lo: x, hi: 0 },
        }
    }
    /// Boundary out: reconstruct a typed [`Value`] (API results, debugger reads). The type comes
    /// from the function/cap signature or the debugger's value-type lookup — the slot itself is
    /// untyped.
    #[inline]
    fn to_value(self, ty: ValType) -> Value {
        match ty {
            ValType::I32 => Value::I32(self.i32()),
            ValType::I64 => Value::I64(self.i64()),
            ValType::F32 => Value::F32(self.f32()),
            ValType::F64 => Value::F64(self.f64()),
            ValType::V128 => Value::V128(self.v128()),
            ValType::Ref => Value::Ref(self.lo),
            // `cap` is i32-width handle data everywhere in guest code (§3.5 reservation).
            ValType::Cap => Value::I32(self.i32()),
        }
    }
}

/// Reasons execution stopped without producing results.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Trap {
    /// Ran out of fuel (potential infinite loop) — see `run`.
    OutOfFuel,
    /// Integer division or remainder by zero (§3b).
    DivByZero,
    /// Signed `div_s` of `INT_MIN / -1`: the quotient `+2^31` is not representable, so
    /// it traps (§3b: trap only when there is no representable result). `rem_s` does
    /// **not** trap here — the remainder `0` *is* representable.
    IntOverflow,
    /// A memory access crossed the top of the window (guard-region fault, §4/§5).
    MemoryFault,
    /// Call recursion exceeded the interpreter's depth bound (host-stack guard).
    StackOverflow,
    /// `call_indirect` selected an empty table slot or a function whose signature
    /// did not match the call's type (the §3c table type-id check).
    IndirectCallType,
    /// Reached an `unreachable`/`trap` terminator (§3b).
    Unreachable,
    /// A trapping float→int conversion saw NaN or an out-of-range value (§3b).
    BadConversion,
    /// A `cap.call` named a handle that is forged, closed/revoked (dead generation),
    /// or the wrong interface type — the index was **inert** (§3c). Not an escape.
    CapFault,
    /// The guest invoked the `Exit` capability; carries the requested exit code. Not
    /// an error — the domain asked to terminate (§3e). Propagates like a trap.
    Exit(i32),
    /// A §12 fiber operation failed: `cont.resume` named a forged/dead/already-running
    /// fiber handle (inert, like [`Trap::CapFault`]), `suspend` ran at the root (no fiber
    /// to suspend to), or the fiber count exceeded the interpreter's bound. Not an escape.
    FiberFault,
    /// A §12 thread operation failed: `thread.join` named a forged / out-of-range / already-joined
    /// thread handle (inert, like [`Trap::CapFault`]), or `thread.spawn` exceeded the run's thread
    /// budget. Not an escape.
    ThreadFault,
    /// Structurally invalid in a way a verified module never is (defensive only).
    Malformed,
}

/// Maximum nested `call` depth before the interpreter traps, bounding the size of the
/// **explicit** guest call stack (a `Vec<Frame>`, §12) so adversarial (or merely deep)
/// guest recursion yields a clean `Trap::StackOverflow` rather than unbounded growth.
///
/// The interpreter no longer recurses on the host stack — the guest call stack is
/// reified (so a fiber's continuation is just its `Vec<Frame>`, suspendable; §12), and
/// the host stack stays O(1) regardless of guest depth. This is a reference-oracle limit,
/// not the production recursion ceiling (the JIT uses the guest's guard-paged data stack,
/// §5).
///
/// The bound must sit **above** a guest runtime's own C-stack-overflow detection so the
/// oracle observes the same *catchable* error the production engines do, rather than killing
/// the guest first. Concretely: Lua's `LUAI_MAXCCALLS` (200 nested C calls) expands to well
/// under 2048 reified frames, so `coroutine.lua`'s "infinite recursion of coroutines" test
/// raises a `pcall`-catchable "C stack overflow" on all three engines instead of the
/// tree-walker uniquely tripping this cap (an uncatchable §5 kill) at 256. Kept comfortably
/// below the durable shadow-reserve's frame budget (`DURABLE_RESERVE`, §12.7) so a deep
/// durable freeze still traps at *this* cap, never by corrupting guest memory.
const MAX_CALL_DEPTH: u32 = 2048;

/// Run `func` with `args`, consuming up to `*fuel` execution steps.
///
/// Returns the function's result values, or a `Trap`. Decrements `*fuel` per
/// instruction and per branch so that even an infinite loop terminates — important
/// for fuzzing and for never hanging a test.
pub fn run(m: &Module, func: FuncIdx, args: &[Value], fuel: &mut u64) -> Result<Vec<Value>, Trap> {
    // No capabilities granted: an empty powerbox (any `cap.call` is inert → `CapFault`).
    let mut host = Host::new();
    run_with_host(m, func, args, fuel, &mut host)
}

/// A traced run's outcome: the result, the **trap-time backtrace** (innermost frame first, as `IrPc`s;
/// empty on a clean finish), and the **trapping fiber** (§5 W3 / §23-D57 — `Some(handle)` for a fiber,
/// `Some(-1)` for the root, `None` on a clean finish). Returned by [`run_traced`] /
/// [`run_with_host_traced`] and the `*_fast_traced` fast-path counterparts.
pub type TracedRun = (Result<Vec<Value>, Trap>, Vec<IrPc>, Option<i64>);

/// Like [`run`], but also return the guest's **trap-time backtrace** (innermost frame first, as
/// `IrPc`s; empty on a clean finish) and the trapping fiber — see [`run_with_host_traced`].
pub fn run_traced(m: &Module, func: FuncIdx, args: &[Value], fuel: &mut u64) -> TracedRun {
    let mut host = Host::new();
    run_with_host_traced(m, func, args, fuel, &mut host)
}

/// The **fast** interpreter entry (INTERP_PERF.md Slice 1c): run on the [`bytecode`] engine when the
/// module is eligible, else fall back to the tree-walker [`run`]. The two are bit-for-bit equivalent
/// on the eligible set (the `bytecode_diff` harness gates this), so this is a transparent speedup.
///
/// `run` itself stays the tree-walker — it is the reference **oracle** the JIT (and the bytecode
/// engine) are differentially checked against, so it must not change. Speed-sensitive callers that
/// are *not* themselves the oracle use `run_fast`.
pub fn run_fast(
    m: &Module,
    func: FuncIdx,
    args: &[Value],
    fuel: &mut u64,
) -> Result<Vec<Value>, Trap> {
    let mut host = Host::new();
    run_with_host_fast(m, func, args, fuel, &mut host)
}

// ===========================================================================================
// Debugging — interpreter-rooted stepping/breakpoints (DEBUGGING.md W2/W8, Milestone A slice 1).
// Designs: S1 (location model = `IrPc`), S3 (logical-time = the probe's op count), S4 (the per-op
// seam in `run_inner`), S5 (driver-style `Inspector`). Single-threaded guests only for now; debug
// of multithreaded guests is Milestone B (it rides the `Policy` scheduler seam — S4).
// ===========================================================================================

/// A program location (DEBUGGING.md S1): which op of which block of which function, in which
/// module space (`0` = the guest's own program; `≥1` = an installed `Jit` unit, §22). This is the
/// granularity breakpoints and backtraces use; mapping it to source is W4 (the §3a debug-info
/// side-table), not yet built.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct IrPc {
    pub module: u32,
    pub func: FuncIdx,
    pub block: usize,
    pub inst: usize,
}

/// Why an [`Inspector`] paused the guest.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StopReason {
    /// Reached an op carrying a breakpoint.
    Breakpoint,
    /// Completed a single-step (`Inspector::step`).
    Step,
    /// About to execute an op that accesses a watched window range. Reported *before* the access
    /// takes effect (`addr`/`write` describe the access); `step` once to apply it and observe the
    /// new bytes. `addr` is the confined window offset the op touches.
    Watchpoint { addr: u64, write: bool },
    /// About to execute a capability call (§3c) — the host-boundary stop, enabled by
    /// [`Inspector::set_cap_call_stops`]. The handle/args are live in the frame (read them via
    /// [`Inspector::read_ir_value`]); `step` once to perform the call and see its results. This is
    /// the boundary W1 record/replay will hook (DEBUGGING.md S5).
    CapCall { type_id: u32, op: u32 },
}

/// Which accesses a watchpoint fires on (`Inspector::set_watchpoint`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum WatchKind {
    /// Reads only.
    Read,
    /// Writes / RMW / `notify` only (the common case — "what changes this?").
    Write,
    /// Either.
    ReadWrite,
}

impl WatchKind {
    fn fires_on(self, write: bool) -> bool {
        match self {
            WatchKind::Read => !write,
            WatchKind::Write => write,
            WatchKind::ReadWrite => true,
        }
    }
}

/// Handle for a set watchpoint, used to clear it (`Inspector::clear_watchpoint`).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct WatchId(u32);

impl WatchId {
    /// Mint a `WatchId` from a raw handle — for a backend that owns its own stable ids (the
    /// DAP-over-bytecode backend, whose watch set is re-applied across `seek` rebuilds, so the ids
    /// can't come from a per-run counter).
    pub fn from_raw(n: u32) -> WatchId {
        WatchId(n)
    }
}

/// The outcome of resuming an [`Inspector`] ([`Inspector::run_until_stop`] / [`Inspector::step`]).
#[derive(Clone, Debug)]
pub enum Stop {
    /// Paused *before* the op at `pc`; the guest state is live and inspectable.
    Break { reason: StopReason, pc: IrPc },
    /// The guest ran to completion (or trapped); no more stepping is possible.
    Finished(Result<Vec<Value>, Trap>),
    /// The (single-threaded) guest parked on `thread.join`/`atomic.wait` with nothing to wake it —
    /// out of scope for slice 1 (multithreaded debugging is Milestone B).
    Blocked,
}

/// One activation on the paused guest's call stack (innermost first), as reported by
/// [`Inspector::backtrace`]. `vals` are the frame's live SSA values by index — the interpreter
/// holds them directly (DEBUGGING.md S2), so a promoted local resolves to `vals[value_idx]` with no
/// Cranelift machinery.
#[derive(Clone, Debug)]
pub struct FrameInfo {
    pub pc: IrPc,
    pub vals: Vec<Value>,
    /// The source location of `pc`, if the module carries debug info (DEBUGGING.md §6/W4).
    pub source: Option<SourceLoc>,
}

/// The source-level types of `frame`'s block-local SSA values, so the debugger can reconstruct a
/// typed [`Value`] from an (untyped) storage [`Reg`]. Reuses the verifier's assignment (single
/// source of truth). Only the guest's own program (module 0) is typed here — installed §22 units
/// resolve against a different function space, so their values fall back to a raw `i64` view.
fn frame_value_types(v: &VCpu, frame: &Frame) -> Vec<ValType> {
    if frame.module != 0 {
        return Vec::new();
    }
    match v.funcs.get(frame.func as usize) {
        Some(f) => svm_verify::func_value_types(f, &v.funcs, v.mem.is_some())
            .into_iter()
            .nth(frame.block)
            .unwrap_or_default(),
        None => Vec::new(),
    }
}

/// Reconstruct a typed [`Value`] for block-local value `idx` of `frame` (debugger boundary). A
/// value whose type can't be derived (a §22 unit frame, or an out-of-range index on an unverified
/// module) reads back as a raw `i64` — total, never panics.
fn frame_value(frame: &Frame, types: &[ValType], idx: usize) -> Option<Value> {
    let slot = frame.vals.get(idx)?;
    Some(slot.to_value(types.get(idx).copied().unwrap_or(ValType::I64)))
}

/// A watched window range (DEBUGGING.md W2). Stop when an op accesses `[addr, addr+len)` with a
/// matching read/write kind.
struct Watch {
    id: WatchId,
    addr: u64,
    len: u64,
    kind: WatchKind,
}

/// Debug state **shared by every vCPU** of a debugged run (DEBUGGING.md W2/Milestone B):
/// breakpoints and watchpoints are global — a breakpoint fires in whichever thread reaches it, a
/// watchpoint on whichever thread touches the range. (Logical time and the pending single-step are
/// *per-vCPU*; they live in [`DebugCtx`].) Behind a `Mutex` only because a vCPU must stay `Send`
/// for the real worker pool; a debugged run is driven cooperatively on one thread, so the lock is
/// always uncontended.
struct DebugShared {
    breakpoints: BTreeSet<IrPc>,
    /// Window-range watchpoints. Empty in the common case, so the hot loop skips the (confining)
    /// `access_of` computation entirely when none are armed (S4 cost gating).
    watchpoints: Vec<Watch>,
    next_watch: u32,
    /// Pause before every `cap.call` (the host boundary, DEBUGGING.md S5).
    cap_stops: bool,
    /// While `true`, the per-op seam ignores breakpoints/watchpoints/steps (clock still ticks) — set
    /// during a scheduled-mode time-travel `seek` so it fast-forwards to a target turn without
    /// stopping (DEBUGGING.md W1). Breakpoints stay armed for the run that follows.
    suppress_stops: bool,
}

impl DebugShared {
    fn new() -> DebugShared {
        DebugShared {
            breakpoints: BTreeSet::new(),
            watchpoints: Vec::new(),
            next_watch: 0,
            cap_stops: false,
            suppress_stops: false,
        }
    }

    /// A `cap.call`-boundary stop, if armed and `inst` is one.
    fn cap_stop(&self, inst: &Inst) -> Option<StopReason> {
        match inst {
            Inst::CapCall { type_id, op, .. } if self.cap_stops => Some(StopReason::CapCall {
                type_id: *type_id,
                op: *op,
            }),
            // An executable `call.import` (IMPORTS.md phase 1) is a capability boundary too;
            // reported under the reserved import-dispatch type_id with the import index as the op
            // (the concrete binding is instantiation state, not visible at this layer).
            Inst::CallImport { import, .. } | Inst::CallSym { import, .. } if self.cap_stops => {
                Some(StopReason::CapCall {
                    type_id: svm_ir::CAP_IMPORT_TYPE_ID,
                    op: *import,
                })
            }
            _ => None,
        }
    }

    /// First watchpoint the `access` hits (overlapping bytes + matching read/write kind), if any.
    fn watch_hit(&self, access: MemAccess) -> Option<(u64, bool)> {
        let MemAccess::Range { base, width, write } = access else {
            return None;
        };
        let end = base.saturating_add(width as u64);
        self.watchpoints.iter().find_map(|w| {
            let w_end = w.addr.saturating_add(w.len);
            let overlaps = base < w_end && w.addr < end;
            (overlaps && w.kind.fires_on(write)).then_some((base, write))
        })
    }
}

/// Per-vCPU debug state, consulted by the per-op hook in [`run_inner`] (DEBUGGING.md S4). Present
/// only while an [`Inspector`] drives the vCPU. The breakpoint/watchpoint sets it consults are
/// **shared** across the run's vCPUs (see [`DebugShared`]); `clock`/`step_target` are this vCPU's.
struct DebugCtx {
    /// The run-wide breakpoint/watchpoint state (shared with the [`Inspector`] and sibling vCPUs).
    shared: Arc<Mutex<DebugShared>>,
    /// S3 logical time: the number of ops this vCPU has executed. Monotonic; the coordinate a
    /// future `seek` (W1) and step-back will target.
    clock: u64,
    /// `clock` value at the most recent stop. Suppresses an immediate re-trigger at the op we just
    /// paused on, so a `continue`/`step` first *steps off* the current breakpoint.
    resume_clock: u64,
    /// When `Some(t)`, stop once `clock` reaches `t` (a pending single-step).
    step_target: Option<u64>,
    /// Depth-aware stepping (DEBUGGING.md W2): when `Some(d)`, stop at the next op (after stepping
    /// off the current one) whose call depth is `<= d`. `d` = current depth runs over (skips into and
    /// out of) any call the current op makes (step-**over**); `d` = current depth − 1 runs until this
    /// function returns (step-**out**).
    step_max_depth: Option<usize>,
    /// Time-travel seek (DEBUGGING.md W1): when `Some(t)`, fast-forward this fresh re-execution to
    /// logical time `t` — pausing exactly at `clock == t` and **ignoring** breakpoints/watchpoints
    /// along the way (we are replaying to a known coordinate, not hunting for stops).
    seek_target: Option<u64>,
}

impl DebugCtx {
    fn new(shared: Arc<Mutex<DebugShared>>) -> DebugCtx {
        DebugCtx {
            shared,
            clock: 0,
            resume_clock: u64::MAX, // nothing stopped yet, so the first op may itself break
            step_target: None,
            step_max_depth: None,
            seek_target: None,
        }
    }

    fn shared(&self) -> std::sync::MutexGuard<'_, DebugShared> {
        self.shared.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn watches_armed(&self) -> bool {
        !self.shared().watchpoints.is_empty()
    }

    /// Decide whether to pause *before* the op at `pc`. `access` is the op's memory effect (only
    /// computed by the caller when watchpoints are armed; [`MemAccess::None`] otherwise); `inst` is
    /// the op itself (for the `cap.call` boundary stop). `Some(reason)` pauses (the op has not run,
    /// `clock` unchanged, so the continuation re-enters here); `None` charges one tick of logical
    /// time and lets the op run. The `resume_clock` guard makes resume step off the current op.
    fn before_op(
        &mut self,
        pc: IrPc,
        inst: &Inst,
        access: MemAccess,
        depth: usize,
    ) -> Option<StopReason> {
        // Time-travel seek (W1): replay straight to logical time `t`, past any breakpoints.
        if let Some(t) = self.seek_target {
            if self.clock >= t {
                self.seek_target = None;
                self.resume_clock = self.clock; // a subsequent step/continue first steps off here
                return Some(StopReason::Step);
            }
            self.clock += 1;
            return None;
        }
        let just_resumed = self.clock == self.resume_clock;
        let reason = if just_resumed {
            None
        } else {
            let sh = self.shared();
            if sh.suppress_stops {
                None // scheduled-seek fast-forward: run past stops (clock still ticks below)
            } else if sh.breakpoints.contains(&pc) {
                Some(StopReason::Breakpoint)
            } else if let Some((addr, write)) = sh.watch_hit(access) {
                Some(StopReason::Watchpoint { addr, write })
            } else if let Some(r) = sh.cap_stop(inst) {
                Some(r)
            } else if self.step_target == Some(self.clock) {
                Some(StopReason::Step)
            } else if matches!(self.step_max_depth, Some(d) if depth <= d) {
                // Step-over/out: we have stepped off the original op and the call depth is back at
                // (or below) the target, so a call we ran over has returned.
                Some(StopReason::Step)
            } else {
                None
            }
        };
        match reason {
            Some(r) => {
                self.resume_clock = self.clock;
                self.step_target = None;
                self.step_max_depth = None;
                Some(r)
            }
            None => {
                self.clock += 1;
                None
            }
        }
    }
}

/// A host-side, **observe-only** debugger for a single-threaded guest on the reference interpreter
/// (DEBUGGING.md W8/S5). It *owns and pumps* the run — `run_until_stop`/`step` drive the guest to
/// the next breakpoint/step, then `backtrace`/`read_ir_value`/`read_window` inspect the paused
/// state. It is a *host* capability shaped like §15 `Monitor`: it never widens the guest's
/// authority, and attaching with no breakpoints is behavior-identical to [`run`] (S7).
///
/// Single vCPU (multithreaded guests are Milestone B). A manifest module's `call.import`s execute
/// through the instantiation-time binding table ([`Host::set_import_bindings`]) exactly as under
/// [`run`] — an unbound slot is a fail-closed `CapFault`. The §3a source mapping (W4) and
/// time-travel (W1) are later slices.
pub struct Inspector {
    /// The vCPU under inspection in **single-threaded** mode (`attach`/`attach_with_host`). `None`
    /// in scheduled mode, where the threads live in the scheduler and the stopped one is held by the
    /// [`SchedDriver`] — see [`Inspector::cur`].
    v: Option<Box<VCpu>>,
    /// Present in **scheduled** (multithreaded) mode (`attach_scheduled`): the cooperative driver +
    /// its deterministic schedule, owned across `run_until_stop`/`step` calls (DEBUGGING.md
    /// Milestone B).
    sched: Option<SchedState>,
    /// The shared powerbox: capabilities the guest may call. The driver owns it for the run; while
    /// the guest is paused it is uncontended, so [`host`](Inspector::host) can lock it to read
    /// effects (captured stdout, clock, grants).
    host: Arc<Mutex<Host>>,
    /// The frontend-neutral debug-info waist (DEBUGGING.md §6), cloned from the module. `None` ⇒
    /// the debugger reports IR locations only ([`IrPc`]/SSA indices); present ⇒ it can resolve
    /// source locations and named variables.
    debug_info: Option<DebugInfo>,
    /// The run-wide breakpoint/watchpoint state, shared with every driven vCPU (so a breakpoint
    /// fires in whichever thread reaches it — DEBUGGING.md Milestone B). The [`Inspector`] mutates
    /// it through `set_breakpoint`/`set_watchpoint`/`set_cap_call_stops`.
    shared: Arc<Mutex<DebugShared>>,
    /// Which thread (vCPU) read-inspection (`backtrace`/`read_*`/`clock`) targets in scheduled mode
    /// (DEBUGGING.md Milestone B `select_fiber`). `None` ⇒ the thread that stopped; `Some(id)` ⇒ a
    /// thread the user `select_task`ed. Reset to `None` on each resume. Stepping always drives the
    /// stopped thread, regardless of focus.
    focus: Option<TaskId>,
    /// The recorded inputs needed to **re-execute** the run from scratch for time-travel `seek` (W1).
    /// Present in single-threaded mode; `None` in scheduled mode (seek pending there).
    seek_init: Option<SeekInit>,
    finished: Option<Result<Vec<Value>, Trap>>,
    /// Time-travel **checkpoint ladder** (W1): snapshots of the sole vCPU at ascending `clock`s,
    /// captured during single-threaded `seek` replays so a later `seek`/`step_back` restarts from the
    /// nearest one (`clock ≤ t`) instead of clock 0 — turning a backward sweep from O(t²) into
    /// ~O(t·stride). Kept sorted by `clock`. Empty in scheduled mode or once `checkpointing` is off.
    checkpoints: Vec<SeekCheckpoint>,
    /// Whether this run is eligible for checkpointing — the single-threaded, **root-only, non-fiber,
    /// non-durable, simple-memory** subset where `frames` + window bytes fully capture the
    /// continuation. Starts `true`; the first replay that observes state outside the subset clears it
    /// (and the ladder), after which `seek` is exactly the original replay-from-clock-0 path.
    checkpointing: bool,
}

/// Capture a checkpoint at most every this many ops. Small enough that `step_back` replays a bounded
/// tail, large enough that the per-checkpoint snapshot (frames + window bytes) amortizes well.
const SEEK_CHECKPOINT_STRIDE: u64 = 1024;

/// The run-mutable host substate a time-travel checkpoint restores — see [`Host::replay_substate`].
#[derive(Clone)]
struct HostReplaySubstate {
    stdin_pos: usize,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
    clock_ns: i64,
    cap_cursor: usize,
    cap_record: Vec<CapRecord>,
}

/// A single-threaded time-travel **checkpoint** (W1): the full re-executable state of the sole vCPU at
/// logical time `clock`, so [`Inspector::seek`] can restart a replay here rather than from clock 0.
/// Captured only for the root-only / non-fiber / non-durable / simple-memory subset (see
/// [`Inspector::checkpoint_of`]), where `frames` plus the window bytes fully determine the
/// continuation.
struct SeekCheckpoint {
    clock: u64,
    frames: Vec<Frame>,
    fuel: u64,
    /// Mapped window bytes (`Mem::snapshot`), reseeded via `Mem::seed` on restore; `None` for a
    /// memoryless run.
    mem: Option<Vec<u8>>,
    host: HostReplaySubstate,
}

/// The inputs a single-threaded run was started with, kept so [`Inspector::seek`] can re-execute it
/// deterministically from `clock 0` (DEBUGGING.md W1 stateless re-execution). All fields are cheap
/// to clone (shared `Arc`s + `Copy`), so a `seek`/`step_back` per keystroke is fine.
#[derive(Clone)]
struct SeekInit {
    funcs: Arc<[Func]>,
    func: FuncIdx,
    args: Arc<[Value]>,
    fuel: u64,
    memory: Option<Memory>,
    data: Arc<[Data]>,
    /// `Some(plan)` ⇒ this was a **scheduled** (multithreaded) run; `seek(t)` re-executes the plan
    /// for `t` scheduler turns (the global logical-time coordinate). `None` ⇒ single-threaded, where
    /// `seek(t)` targets the sole vCPU's op clock.
    schedule: Option<Arc<[u64]>>,
    /// The fuzzing seed, if the scheduled run was [`attach_scheduled_seeded`]. `seek` re-applies it
    /// so the same random interleaving is reproduced.
    ///
    /// [`attach_scheduled_seeded`]: Inspector::attach_scheduled_seeded
    seed: Option<u64>,
}

/// Scheduled (multithreaded) execution state owned by an [`Inspector`] across pumps (Milestone B).
struct SchedState {
    /// The deterministic cooperative scheduler holding every vCPU not currently selected.
    det: Arc<DetSched>,
    /// The schedule source, reused as the picker: an empty plan ⇒ the deterministic default order
    /// (smallest runnable `TaskId` each turn); a `Witness::plan` ⇒ replay that exact interleaving.
    dpor: Dpor,
    /// The re-entrant driver; while stopped, the offending vCPU is parked in `driver.held` and is
    /// the inspection target ([`Inspector::cur`]).
    driver: SchedDriver,
    /// The root vCPU's id, whose outcome is the guest's result.
    root_id: TaskId,
}

/// A resolved source location ([`Inspector::source_loc`]).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct SourceLoc {
    pub file: String,
    pub line: u32,
    /// 0 means "no column".
    pub col: u32,
}

/// Resolve an [`IrPc`] to its source position using `m`'s `-g` debug info (nearest-preceding `loc`
/// within the block), or `None` without debug info / for an installed §22 unit (`pc.module != 0`).
/// The free counterpart of [`Inspector::source_loc`]; pairs with [`run_with_host_traced`] to render
/// an interpreter trap-time backtrace to source (the symmetric analog of the JIT's `JitFrameLoc`).
pub fn source_loc(m: &Module, pc: IrPc) -> Option<SourceLoc> {
    source_loc_in(m.debug_info.as_ref()?, pc)
}

/// The `-g` source name of function `func` in module 0 (`debug_info.func_names`), or `None` when the
/// module carried no name for it — renderers fall back to `fn{func}`. Pairs with [`source_loc`] to
/// render an interpreter trap-time backtrace with function names (the analog of the JIT's
/// `JitFrameLoc::func_name`).
pub fn func_name(m: &Module, func: FuncIdx) -> Option<&str> {
    let di = m.debug_info.as_ref()?;
    di.func_names
        .iter()
        .find(|f| f.func == func)
        .map(|f| f.name.as_str())
}

/// The nearest-preceding-`loc` resolution shared by [`source_loc`] and [`Inspector::source_loc`]
/// (DEBUGGING.md §6/W4 S2): the latest `loc` in `pc`'s block at or before `pc.inst`.
fn source_loc_in(di: &DebugInfo, pc: IrPc) -> Option<SourceLoc> {
    if pc.module != 0 {
        return None;
    }
    let l = di
        .locs
        .iter()
        .filter(|l| l.func == pc.func && l.block as usize == pc.block && l.inst as usize <= pc.inst)
        .max_by_key(|l| l.inst)?;
    let file = di.files.get(l.file as usize)?.clone();
    Some(SourceLoc {
        file,
        line: l.line,
        col: l.col,
    })
}

/// The value of a source variable ([`Inspector::read_var`]). A promoted scalar resolves to a live
/// SSA [`Value`]; a window-resident variable resolves to its raw little-endian bytes (the S2
/// `Window` location — the caller interprets them per the variable's type).
#[derive(Clone, PartialEq, Debug)]
pub enum VarValue {
    Value(Value),
    Bytes(Vec<u8>),
}

/// One recorded crossing of the capability boundary (DEBUGGING.md W1 `CapTape`): the inputs the
/// guest passed and the result slots the host returned, for a **nondeterministic input** capability
/// (e.g. `Clock`). Replayed verbatim so a re-execution (time-travel `seek`) sees identical host
/// inputs without a live host.
#[derive(Clone, PartialEq, Debug)]
pub struct CapRecord {
    pub type_id: u32,
    pub op: u32,
    pub handle: i32,
    pub args: Vec<i64>,
    /// The result slots the host returned (`Err` for a cap that trapped, e.g. `exit`).
    pub result: Result<Vec<i64>, Trap>,
    /// Bytes the cap wrote **into the guest window** as `(ptr, bytes)` — e.g. a stdin `read` filling
    /// its buffer. Re-applied on replay so a buffer-filling input cap reproduces faithfully. Empty
    /// for slot-only caps (`Clock`).
    pub mem_writes: Vec<(u64, Vec<u8>)>,
}

/// An append-only log of the capability **inputs** crossing into a guest, captured during a run so a
/// later re-execution reproduces them without a live host (DEBUGGING.md W1). Records the slot-only
/// input caps (`Clock`) and buffer-filling ones (stdin `read`); RNG / host-fns and the `SchedTape`
/// are follow-ups.
#[derive(Clone, Default, PartialEq, Debug)]
pub struct CapTape {
    pub records: Vec<CapRecord>,
}

/// Whether a capability is a **nondeterministic input** whose result a re-execution must replay
/// (rather than re-derive). Deterministic / structural caps (window `Memory` ops, `SharedRegion`,
/// `Stream` *write*) re-run faithfully on a fresh powerbox, so they are left live. Inputs: `Clock`
/// (op 0 `now`), `Stream` op 0 (stdin `read`), and **any host-fn** (`cap_id::HOST_FN`) — the
/// embedder's escape hatch (RNG, a real clock, external I/O), whose closure is *gone* on the fresh
/// replay powerbox, so only the tape can reproduce it.
fn is_recorded_input(type_id: u32, op: u32) -> bool {
    type_id == cap_id::CLOCK || (type_id == cap_id::STREAM && op == 0) || type_id == cap_id::HOST_FN
}

/// A [`GuestMem`] wrapper that records every `write_bytes` a capability makes into the guest window
/// (DEBUGGING.md W1), so a buffer-filling input cap (stdin `read`) can be replayed by re-applying the
/// captured bytes. Every other operation delegates unchanged to the real window.
struct RecordingMem<'a> {
    inner: &'a mut dyn GuestMem,
    writes: Vec<(u64, Vec<u8>)>,
}

impl GuestMem for RecordingMem<'_> {
    fn read_bytes(&self, ptr: u64, len: u64) -> Option<Vec<u8>> {
        self.inner.read_bytes(ptr, len)
    }
    fn write_bytes(&mut self, ptr: u64, data: &[u8]) -> Option<()> {
        let r = self.inner.write_bytes(ptr, data);
        if r.is_some() {
            self.writes.push((ptr, data.to_vec()));
        }
        r
    }
    fn map(&mut self, offset: u64, len: u64, prot: i32) -> i64 {
        self.inner.map(offset, len, prot)
    }
    fn unmap(&mut self, offset: u64, len: u64) -> i64 {
        self.inner.unmap(offset, len)
    }
    fn protect(&mut self, offset: u64, len: u64, prot: i32) -> i64 {
        self.inner.protect(offset, len, prot)
    }
    fn map_region(
        &mut self,
        win_off: u64,
        region_off: u64,
        len: u64,
        prot: i32,
        region: u32,
        backing: RegionBacking,
    ) -> i64 {
        self.inner
            .map_region(win_off, region_off, len, prot, region, backing)
    }
    fn page_size(&self) -> i64 {
        self.inner.page_size()
    }
    fn region_page_size(&self) -> i64 {
        self.inner.region_page_size()
    }
    fn async_counter(&self, counter_addr: u64) -> Option<Arc<dyn AsyncCounter>> {
        self.inner.async_counter(counter_addr)
    }
}

impl Inspector {
    /// Attach to `m`'s `func(args)` with `fuel` and an **empty powerbox** (any `cap.call` faults),
    /// paused before the first op. Set breakpoints, then
    /// [`run_until_stop`](Inspector::run_until_stop) or [`step`](Inspector::step).
    pub fn attach(m: &Module, func: FuncIdx, args: &[Value], fuel: u64) -> Inspector {
        Inspector::attach_with_host(m, func, args, fuel, Host::new())
    }

    /// Like [`attach`](Inspector::attach), but with a caller-prepared [`Host`] (the powerbox):
    /// `grant_*` the capabilities the guest needs, pass their handle indices in `args`, then debug a
    /// capability-using guest (§3c/§3e). Read effects back through [`host`](Inspector::host) while
    /// paused or after [`Stop::Finished`]. `m` must be import-resolved (see the type docs).
    pub fn attach_with_host(
        m: &Module,
        func: FuncIdx,
        args: &[Value],
        fuel: u64,
        mut host: Host,
    ) -> Inspector {
        let funcs: Arc<[Func]> = m.funcs.to_vec().into();
        let args: Arc<[Value]> = args.into();
        let data: Arc<[Data]> = m.data.clone().into();
        host.record_caps(); // W1: tape the cap inputs so `seek` can re-execute faithfully
        let host = Arc::new(Mutex::new(host));
        let shared = Arc::new(Mutex::new(DebugShared::new()));
        let root = Self::fresh_single_root(
            Arc::clone(&funcs),
            func,
            &args,
            fuel,
            m.memory,
            &data,
            Arc::clone(&host),
            Arc::clone(&shared),
            None,
        );
        Inspector {
            v: Some(root),
            sched: None,
            host,
            debug_info: m.debug_info.clone(),
            shared,
            focus: None,
            seek_init: Some(SeekInit {
                funcs,
                func,
                args,
                fuel,
                memory: m.memory,
                data,
                schedule: None,
                seed: None,
            }),
            finished: None,
            checkpoints: Vec::new(),
            checkpointing: true,
        }
    }

    /// Build a fresh single-threaded root vCPU over `funcs`/`func`/`args` with its own scheduler (no
    /// workers run — the driver pumps it directly). `seek_target` fast-forwards a time-travel replay
    /// to a logical time and pauses there (W1); `None` for a normal attach. Shared by `attach` and
    /// `seek` so both build identical initial state.
    #[allow(clippy::too_many_arguments)]
    fn fresh_single_root(
        funcs: Arc<[Func]>,
        func: FuncIdx,
        args: &[Value],
        fuel: u64,
        memory: Option<Memory>,
        data: &[Data],
        host: Arc<Mutex<Host>>,
        shared: Arc<Mutex<DebugShared>>,
        seek_target: Option<u64>,
    ) -> Box<VCpu> {
        let mem = memory.map(|mc| {
            let mut mm = Mem::with_reservation(DEFAULT_RESERVED_LOG2, mc.size_log2);
            mm.init_data(data);
            mm
        });
        let quota = Quota::default();
        let sched = Arc::new(Scheduler::new(quota.max_vcpus, MAX_WORKERS));
        let dt = Arc::new(DomainTable::new(&funcs, 0));
        let mut root = VCpu::new(
            funcs,
            func,
            args,
            mem,
            host,
            fuel,
            0,
            0,
            SchedRef::Real(sched),
            quota,
            dt,
        );
        let mut d = DebugCtx::new(shared);
        d.seek_target = seek_target;
        root.debug = Some(Box::new(d));
        Box::new(root)
    }

    /// Attach to a **multithreaded** guest and debug a chosen interleaving (DEBUGGING.md Milestone
    /// B). The run is driven cooperatively on this thread under a fixed `schedule` — a
    /// [`Witness::plan`] from [`find_schedule`] to step a specific (e.g. failing) interleaving, or an
    /// empty `Vec` for the deterministic default order (smallest runnable thread each turn). Unlike
    /// the single-threaded [`attach`](Inspector::attach), this drives every `thread.spawn`ed vCPU on
    /// one OS thread, so the schedule is reproducible and breakpoints/steps are exact and
    /// race-free. Breakpoints fire in whichever thread reaches them; [`stopped_task`] reports which.
    /// Inspection becomes available after the first [`run_until_stop`]/[`step`]. `m` must be
    /// import-resolved.
    ///
    /// [`stopped_task`]: Inspector::stopped_task
    pub fn attach_scheduled(
        m: &Module,
        func: FuncIdx,
        args: &[Value],
        fuel: u64,
        schedule: Vec<u64>,
    ) -> Inspector {
        Self::scheduled_impl(m, func, args, fuel, schedule, None)
    }

    /// Like [`attach_scheduled`](Inspector::attach_scheduled), but drive a **random** interleaving
    /// from `seed` instead of a fixed plan — schedule fuzzing (DEBUGGING.md W1). Each turn picks a
    /// random runnable thread, so different seeds explore different interleavings (and may surface
    /// different race outcomes). The interleaving that ran is captured: [`sched_tape`] returns it as
    /// a portable plan you can `attach_scheduled` to replay deterministically (or share as a repro).
    /// `seek`/`step_back` reproduce this exact random run (same seed).
    ///
    /// [`sched_tape`]: Inspector::sched_tape
    pub fn attach_scheduled_seeded(
        m: &Module,
        func: FuncIdx,
        args: &[Value],
        fuel: u64,
        seed: u64,
    ) -> Inspector {
        Self::scheduled_impl(m, func, args, fuel, Vec::new(), Some(seed))
    }

    fn scheduled_impl(
        m: &Module,
        func: FuncIdx,
        args: &[Value],
        fuel: u64,
        schedule: Vec<u64>,
        seed: Option<u64>,
    ) -> Inspector {
        let funcs: Arc<[Func]> = m.funcs.to_vec().into();
        let args_arc: Arc<[Value]> = args.into();
        let data: Arc<[Data]> = m.data.clone().into();
        let schedule: Arc<[u64]> = schedule.into();
        let mut host0 = Host::new();
        host0.record_caps(); // W1: tape cap inputs so scheduled `seek` re-executes faithfully
        let host = Arc::new(Mutex::new(host0));
        let shared = Arc::new(Mutex::new(DebugShared::new()));
        let sched = Self::fresh_scheduled(
            Arc::clone(&funcs),
            func,
            args,
            fuel,
            m.memory,
            &data,
            Arc::clone(&shared),
            Arc::clone(&host),
            &schedule,
            seed,
            None,
        );
        Inspector {
            v: None,
            sched: Some(sched),
            host,
            debug_info: m.debug_info.clone(),
            shared,
            focus: None,
            seek_init: Some(SeekInit {
                funcs,
                func,
                args: args_arc,
                fuel,
                memory: m.memory,
                data,
                schedule: Some(schedule),
                seed,
            }),
            finished: None,
            // Scheduled (multithreaded) seek targets the global turn coordinate and is not
            // checkpointed in this slice — checkpointing is the single-threaded path only.
            checkpoints: Vec::new(),
            checkpointing: false,
        }
    }

    /// Build a fresh scheduled (multithreaded) run: a `DetSched` holding the root vCPU (debug +
    /// `memop`), to be driven under `schedule` (a plan, or empty for the deterministic default
    /// order). `turn_limit` stops the driver at that turn boundary for a time-travel `seek`. Shared
    /// by `attach_scheduled` and the scheduled branch of `seek` so both build identical state.
    #[allow(clippy::too_many_arguments)]
    fn fresh_scheduled(
        funcs: Arc<[Func]>,
        func: FuncIdx,
        args: &[Value],
        fuel: u64,
        memory: Option<Memory>,
        data: &[Data],
        shared: Arc<Mutex<DebugShared>>,
        host: Arc<Mutex<Host>>,
        schedule: &[u64],
        seed: Option<u64>,
        turn_limit: Option<u64>,
    ) -> SchedState {
        let mem = memory.map(|mc| {
            let mut mm = Mem::with_reservation(DEFAULT_RESERVED_LOG2, mc.size_log2);
            mm.init_data(data);
            mm
        });
        let det = Arc::new(DetSched::new(0, MAX_VCPUS));
        let root_id = {
            let mut s = det.lock();
            let id = s.next_task;
            s.next_task += 1;
            s.live += 1;
            let dt = Arc::new(DomainTable::new(&funcs, 0));
            let mut root = VCpu::new(
                Arc::clone(&funcs),
                func,
                args,
                mem,
                Arc::clone(&host),
                fuel,
                0,
                id,
                SchedRef::Det(Arc::clone(&det)),
                Quota::default(),
                dt,
            );
            root.memop = true; // one visible op per turn — the granularity steps/breakpoints align to
            root.debug = Some(Box::new(DebugCtx::new(Arc::clone(&shared))));
            s.runnable.push(Box::new(root));
            id
        };
        let mut dpor = Dpor::new(schedule.to_vec(), Vec::new());
        // Seeded fuzzing extends the schedule randomly past the (usually empty) plan; mixing the raw
        // seed avoids the xorshift zero fixpoint, and is applied identically here for `seek` to
        // reproduce the same random schedule.
        dpor.rng = seed.map(|s| s ^ 0x9E37_79B9_7F4A_7C15);
        SchedState {
            det,
            dpor,
            driver: SchedDriver {
                turn_limit,
                ..Default::default()
            },
            root_id,
        }
    }

    /// Time-travel to logical time `t` (DEBUGGING.md W1): re-execute the run from the start and pause
    /// at `t`, so `backtrace`/`read_*` show the guest exactly **as it was**. The coordinate `t` is
    /// the sole vCPU's op [`clock`](Inspector::clock) in single-threaded mode, and the global
    /// scheduler-[`turn`](Inspector::turn) count in scheduled (multithreaded) mode — where the result
    /// is a *global snapshot* and every thread is inspectable via [`threads`]/[`select_task`].
    /// Stateless re-execution (the §18 explorer's trick): exact for a deterministic guest, and
    /// faithful for nondeterministic *inputs* via the replayed [`CapTape`]. Breakpoints are preserved
    /// but ignored during the fast-forward. `Stop::Blocked` if the run can't be re-executed.
    ///
    /// [`threads`]: Inspector::threads
    /// [`select_task`]: Inspector::select_task
    pub fn seek(&mut self, t: u64) -> Stop {
        let Some(init) = self.seek_init.clone() else {
            return Stop::Blocked;
        };
        // A fresh powerbox seeded to *replay* the recorded cap inputs (W1) and keep recording past
        // `t`, so re-execution reproduces nondeterministic inputs (e.g. `Clock`, stdin `read`).
        let tape: Arc<[CapRecord]> = self.cap_tape().records.into();
        let mut fresh = Host::new();
        fresh.replay_caps(tape);
        fresh.record_caps();
        let host = Arc::new(Mutex::new(fresh));

        match &init.schedule {
            // Scheduled (multithreaded): replay the plan for `t` scheduler turns, landing at a global
            // turn-`t` snapshot. No vCPU is "stopped"; focus the thread that ran the last turn.
            Some(schedule) => {
                let mut sched = Self::fresh_scheduled(
                    init.funcs,
                    init.func,
                    &init.args,
                    init.fuel,
                    init.memory,
                    &init.data,
                    Arc::clone(&self.shared),
                    Arc::clone(&host),
                    schedule,
                    init.seed,
                    Some(t),
                );
                self.shared().suppress_stops = true; // fast-forward past breakpoints
                let (focus, finished, pc) = {
                    let SchedState {
                        det,
                        dpor,
                        driver,
                        root_id,
                    } = &mut sched;
                    let mut policy = Policy::Dpor(dpor);
                    let _ = driver.run(det, &mut policy);
                    driver.turn_limit = None; // continuing from here runs forward normally
                    let focus = dpor.trace.last().map(|e| e.tid).unwrap_or(*root_id);
                    let finished = det.lock().results.get(root_id).map(|o| o.result.clone());
                    let pc = det.lock().find_vcpu(focus).and_then(Self::pc_of);
                    (focus, finished, pc)
                };
                self.shared().suppress_stops = false;
                self.v = None;
                self.host = host;
                self.focus = Some(focus);
                self.finished = finished.clone();
                self.sched = Some(sched);
                match finished {
                    Some(r) => Stop::Finished(r),
                    None => Stop::Break {
                        reason: StopReason::Step,
                        pc: pc.unwrap_or(IrPc {
                            module: 0,
                            func: 0,
                            block: 0,
                            inst: 0,
                        }),
                    },
                }
            }
            // Single-threaded: re-execute the sole vCPU to op `clock == t` — restarting from the
            // nearest checkpoint ≤ t (W1) instead of clock 0 when this run is checkpointable.
            None => self.seek_single(&init, host, t),
        }
    }

    /// Single-threaded `seek` (DEBUGGING.md W1): drive the sole vCPU to logical time `t`, restarting
    /// from the nearest **checkpoint** (`clock ≤ t`) rather than clock 0 when the run is in the
    /// checkpointable subset, and laying down fresh checkpoints (every [`SEEK_CHECKPOINT_STRIDE`] ops)
    /// along the way. This bounds `seek`/`step_back` to the checkpoint stride instead of O(t). For a
    /// run outside the subset it is exactly the original replay-from-clock-0 (`checkpointing` is off,
    /// no checkpoint is found, none is captured).
    fn seek_single(&mut self, init: &SeekInit, host: Arc<Mutex<Host>>, t: u64) -> Stop {
        // Nearest checkpoint at or before `t` (the ladder is kept sorted by `clock`).
        let start = if self.checkpointing {
            self.checkpoints.iter().rev().find(|c| c.clock <= t)
        } else {
            None
        };
        let mut root = Self::fresh_single_root(
            init.funcs.clone(),
            init.func,
            &init.args,
            init.fuel,
            init.memory,
            &init.data,
            Arc::clone(&host),
            Arc::clone(&self.shared),
            None, // the seek target is set per chunk by the drive loop below
        );
        if let Some(cp) = start {
            root.restore_continuation(cp.frames.clone(), cp.fuel, cp.mem.as_deref(), cp.clock);
            host.lock()
                .unwrap_or_else(|e| e.into_inner())
                .restore_replay_substate(&cp.host);
        }
        self.host = host;
        self.finished = None;
        self.focus = None;

        let stop = self.drive_to(&mut root, t);
        self.v = Some(root);
        stop
    }

    /// Drive `root` forward to logical time `t`, pausing at each [`SEEK_CHECKPOINT_STRIDE`] boundary to
    /// snapshot a checkpoint (so a later `seek`/`step_back` restarts nearby), then on to `t`. With
    /// checkpointing off it is a single straight run to `t`. The chunking is transparent: each chunk
    /// runs purely the `seek_target` replay branch of [`DebugCtx::before_op`], so the cumulative effect
    /// equals one run from the start point to `t`.
    fn drive_to(&mut self, root: &mut VCpu, t: u64) -> Stop {
        loop {
            let clock = root.debug_clock();
            // Next pause: the upcoming stride boundary (strictly after `clock`), capped at `t`.
            let next = if self.checkpointing && clock < t {
                (clock / SEEK_CHECKPOINT_STRIDE + 1) * SEEK_CHECKPOINT_STRIDE
            } else {
                t
            }
            .min(t);
            root.dbg_seek_to(next);
            match root.run(u64::MAX) {
                Step::Done(r) => {
                    self.finished = Some(r.clone());
                    return Stop::Finished(r);
                }
                Step::Park(_) | Step::Yield => return Stop::Blocked,
                Step::Pause(_, pc) => {
                    if root.debug_clock() >= t {
                        return Stop::Break {
                            reason: StopReason::Step,
                            pc,
                        };
                    }
                    // Reached a stride boundary short of `t`: record a checkpoint and continue. If the
                    // continuation has left the checkpointable subset (a fiber/thread/coroutine/durable
                    // op, or non-pristine memory), abandon checkpointing for this run.
                    self.maybe_checkpoint(root);
                }
            }
        }
    }

    /// Snapshot `root` into the checkpoint ladder at its current `clock`, if checkpointing is still on
    /// and the continuation is [`VCpu::checkpointable`]; otherwise disable checkpointing and drop the
    /// ladder (the run is outside the snapshottable subset, so `seek` reverts to replay-from-0). A
    /// `clock` already present in the ladder is not duplicated.
    fn maybe_checkpoint(&mut self, root: &VCpu) {
        if !self.checkpointing {
            return;
        }
        let clock = root.debug_clock();
        let host_sub = {
            let h = self.host.lock().unwrap_or_else(|e| e.into_inner());
            // Leave the subset (and drop the ladder) if the continuation or the host has grown state a
            // checkpoint can't faithfully restore.
            if !root.checkpointable() || !h.checkpoint_safe() {
                drop(h);
                self.checkpointing = false;
                self.checkpoints.clear();
                return;
            }
            h.replay_substate()
        };
        if self.checkpoints.iter().any(|c| c.clock == clock) {
            return;
        }
        let host = host_sub;
        let cp = SeekCheckpoint {
            clock,
            frames: root.frames.clone(),
            fuel: root.fuel,
            mem: root.mem.as_ref().map(|m| m.window_snapshot()),
            host,
        };
        // Keep the ladder sorted by `clock` (boundaries are usually appended in order, but a fresh
        // replay-from-0 can fill gaps below an existing entry).
        let at = self.checkpoints.partition_point(|c| c.clock < clock);
        self.checkpoints.insert(at, cp);
    }

    /// The current call frame's pc, if any (innermost frame).
    fn pc_of(v: &VCpu) -> Option<IrPc> {
        v.frames.last().map(|f| IrPc {
            module: f.module,
            func: f.func,
            block: f.block,
            inst: f.inst,
        })
    }

    /// The global scheduler-turn count — the coordinate scheduled-mode [`seek`](Inspector::seek)
    /// targets (DEBUGGING.md W1). One turn per visible-op decision across all threads. `0` in
    /// single-threaded mode (which uses the op [`clock`](Inspector::clock) instead).
    pub fn turn(&self) -> u64 {
        self.sched.as_ref().map(|s| s.driver.turns).unwrap_or(0)
    }

    /// The **`SchedTape`** (DEBUGGING.md W1): the interleaving this scheduled run actually executed,
    /// as the ordered `TaskId` choice at each visible-op decision. Empty in single-threaded mode or
    /// before any turn runs. It is a portable, replayable artifact — `attach_scheduled` it to
    /// reproduce the exact interleaving deterministically (no seed needed), e.g. to share a race
    /// repro found by [`attach_scheduled_seeded`](Inspector::attach_scheduled_seeded). Under
    /// sequential consistency the schedule *is* the memory order, so this fully pins the run.
    pub fn sched_tape(&self) -> Vec<u64> {
        self.sched
            .as_ref()
            .map(|s| s.dpor.trace.iter().map(|e| e.tid).collect())
            .unwrap_or_default()
    }

    /// Step **backward** one unit of logical time (DEBUGGING.md W1): the global `turn` in scheduled
    /// mode, the op `clock` single-threaded. At time 0 this re-seeks to the initial state.
    pub fn step_back(&mut self) -> Stop {
        let now = if self.sched.is_some() {
            self.turn()
        } else {
            self.clock()
        };
        self.seek(now.saturating_sub(1))
    }

    /// The vCPU currently under inspection (the paused thread in scheduled mode, the sole vCPU in
    /// single-threaded mode). `None` in scheduled mode before the first stop, or once finished.
    fn cur(&self) -> Option<&VCpu> {
        match &self.sched {
            Some(s) => s.driver.held.as_ref().map(|h| h.v.as_ref()),
            None => self.v.as_deref(),
        }
    }

    fn cur_mut(&mut self) -> Option<&mut VCpu> {
        match &mut self.sched {
            Some(s) => s.driver.held.as_mut().map(|h| h.v.as_mut()),
            None => self.v.as_deref_mut(),
        }
    }

    /// The id of the thread (vCPU) currently stopped, in scheduled mode (DEBUGGING.md Milestone B).
    /// `None` in single-threaded mode, before the first stop, or once finished.
    pub fn stopped_task(&self) -> Option<u64> {
        self.sched
            .as_ref()
            .and_then(|s| s.driver.held.as_ref().map(|h| h.v.id))
    }

    /// Every live thread (vCPU) id, sorted — the stopped thread plus every other thread the
    /// scheduler is holding (runnable or parked on join/wait/spin). Single-threaded mode reports the
    /// sole vCPU; once a thread finishes it drops out (only its outcome remains).
    pub fn threads(&self) -> Vec<u64> {
        match &self.sched {
            None => self.cur().map(|v| vec![v.id]).unwrap_or_default(),
            Some(s) => {
                let mut ids = s.det.lock().live_ids();
                // The stopped thread is held *out* of the scheduler, so add it back.
                if let Some(h) = s.driver.held.as_ref() {
                    if !ids.contains(&h.v.id) {
                        ids.push(h.v.id);
                        ids.sort_unstable();
                    }
                }
                ids
            }
        }
    }

    /// Focus read-inspection (`backtrace`/`read_*`/`clock`) on thread `id` (DEBUGGING.md Milestone B
    /// `select_fiber`): while stopped, look at *any* live thread, not just the one that hit the
    /// breakpoint. Returns `false` (leaving the focus unchanged) if `id` is not a live thread.
    /// Stepping still drives the stopped thread; the focus resets on the next resume.
    pub fn select_task(&mut self, id: u64) -> bool {
        if self.threads().contains(&id) {
            self.focus = Some(id);
            true
        } else {
            false
        }
    }

    /// The thread read-inspection currently targets: the `select_task` focus, else the stopped
    /// (scheduled) or sole (single-threaded) thread.
    pub fn focused_task(&self) -> Option<u64> {
        self.focus.or_else(|| self.cur().map(|v| v.id))
    }

    /// Run `f` against the focused thread's vCPU (the `select_task` target, else the stopped/sole
    /// thread), resolving it wherever the scheduler parks it. `None` if there is no such live thread.
    fn with_focused<R>(&self, f: impl FnOnce(&VCpu) -> R) -> Option<R> {
        match (&self.sched, self.focus) {
            // A selected thread other than the stopped one lives inside the scheduler.
            (Some(s), Some(id)) if Some(id) != s.driver.held.as_ref().map(|h| h.v.id) => {
                s.det.lock().find_vcpu(id).map(f)
            }
            // Otherwise it's the stopped (held) / sole vCPU.
            _ => self.cur().map(f),
        }
    }

    /// Lock the powerbox to inspect host-side effects (e.g. `host().stdout`). Safe while the guest
    /// is paused or finished — it is not executing, so the lock is uncontended.
    pub fn host(&self) -> std::sync::MutexGuard<'_, Host> {
        self.host.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The recorded capability-input trace so far (DEBUGGING.md W1) — the `Clock` (and, later, other
    /// input) crossings this run has made. This is what [`seek`](Inspector::seek) replays to
    /// re-execute a nondeterministic guest faithfully, and the artifact a host can capture from a
    /// live run.
    pub fn cap_tape(&self) -> CapTape {
        self.host().cap_tape()
    }

    fn dbg(&mut self) -> &mut DebugCtx {
        self.cur_mut()
            .and_then(|v| v.debug.as_mut())
            .expect("an Inspector-driven vCPU always carries debug state")
    }

    /// The run-wide (cross-vCPU) breakpoint/watchpoint state. Uncontended while the guest is paused.
    fn shared(&self) -> std::sync::MutexGuard<'_, DebugShared> {
        self.shared.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Add a breakpoint at `pc` (idempotent). Applies to every thread (DEBUGGING.md Milestone B).
    pub fn set_breakpoint(&mut self, pc: IrPc) {
        self.shared().breakpoints.insert(pc);
    }

    /// Remove a breakpoint; returns whether one was present.
    pub fn clear_breakpoint(&mut self, pc: IrPc) -> bool {
        self.shared().breakpoints.remove(&pc)
    }

    /// Watch window range `[addr, addr+len)` for accesses of `kind`; the run pauses *before* the
    /// op that accesses it ([`StopReason::Watchpoint`]). Because the window is one contiguous buffer
    /// (DEBUGGING.md W2), this catches every code path — no per-op instrumentation needed. Returns a
    /// handle for [`clear_watchpoint`](Inspector::clear_watchpoint). A zero-length watch never fires.
    pub fn set_watchpoint(&mut self, addr: u64, len: u64, kind: WatchKind) -> WatchId {
        let mut d = self.shared();
        let id = WatchId(d.next_watch);
        d.next_watch += 1;
        d.watchpoints.push(Watch {
            id,
            addr,
            len,
            kind,
        });
        id
    }

    /// Remove a watchpoint; returns whether one was present.
    pub fn clear_watchpoint(&mut self, id: WatchId) -> bool {
        let mut d = self.shared();
        let before = d.watchpoints.len();
        d.watchpoints.retain(|x| x.id != id);
        d.watchpoints.len() != before
    }

    /// Enable/disable pausing before every `cap.call` ([`StopReason::CapCall`]) — the host-boundary
    /// stop. Useful for tracing a guest's capability use and the boundary W1 record/replay hooks.
    pub fn set_cap_call_stops(&mut self, on: bool) {
        self.shared().cap_stops = on;
    }

    /// Resume until the next breakpoint, completion, or block.
    pub fn run_until_stop(&mut self) -> Stop {
        if let Some(r) = &self.finished {
            return Stop::Finished(r.clone());
        }
        self.focus = None; // a `select_task` focus lasts only until the guest moves again
        if self.sched.is_some() {
            return self.pump_sched();
        }
        match self.v.as_mut().unwrap().run(u64::MAX) {
            Step::Pause(reason, pc) => Stop::Break { reason, pc },
            Step::Done(r) => {
                self.finished = Some(r.clone());
                Stop::Finished(r)
            }
            Step::Park(_) | Step::Yield => Stop::Blocked,
        }
    }

    /// Drive the cooperative multithreaded scheduler to the next stop (DEBUGGING.md Milestone B).
    fn pump_sched(&mut self) -> Stop {
        // `Done` means scheduler quiescence: read the root's outcome (a result ⇒ finished; absent ⇒
        // a join-deadlock / all-asleep, i.e. blocked). A `Paused` keeps the offending vCPU held.
        let outcome = {
            let s = self.sched.as_mut().unwrap();
            let SchedState {
                det,
                dpor,
                driver,
                root_id,
            } = s;
            let mut policy = Policy::Dpor(dpor);
            match driver.run(det, &mut policy) {
                DriverStop::Paused { reason, pc, .. } => return Stop::Break { reason, pc },
                DriverStop::Done => det.lock().results.remove(root_id),
                // Normal pumping never sets a turn limit (only `seek` does, on its own driver).
                DriverStop::TurnLimit => unreachable!("turn limit only set during seek"),
            }
        };
        match outcome {
            Some(o) => {
                self.finished = Some(o.result.clone());
                Stop::Finished(o.result)
            }
            None => Stop::Blocked,
        }
    }

    /// Execute exactly one op of the currently-stopped thread, then pause before the next (or
    /// finish/block). In scheduled mode before the first stop, resumes to the first stop instead.
    pub fn step(&mut self) -> Stop {
        if self.finished.is_some() || self.cur().is_none() {
            return self.run_until_stop();
        }
        let d = self.dbg();
        d.step_target = Some(d.clock + 1);
        self.run_until_stop()
    }

    /// **Step over** (DEBUGGING.md W2): execute the current op and stop at the next one in this
    /// frame — running any call it makes to completion rather than descending into it. (Op-level: a
    /// non-call advances one op, like [`step`](Inspector::step); a `call` runs the whole callee.)
    pub fn step_over(&mut self) -> Stop {
        self.step_to_depth(|depth| depth)
    }

    /// **Step out** (DEBUGGING.md W2): run until the current function returns, stopping at the op in
    /// the caller that the call returned to. From the outermost frame this runs to completion.
    pub fn step_out(&mut self) -> Stop {
        self.step_to_depth(|depth| depth.saturating_sub(1))
    }

    /// Shared driver for step-over/out: step off the current op, then stop at the next op whose call
    /// depth is `<= target(current_depth)`.
    fn step_to_depth(&mut self, target: impl Fn(usize) -> usize) -> Stop {
        if self.finished.is_some() || self.cur().is_none() {
            return self.run_until_stop();
        }
        let depth = self.cur().map(|v| v.frames.len()).unwrap_or(0);
        let d = self.dbg();
        d.resume_clock = d.clock; // step off the current op even on the very first step
        d.step_max_depth = Some(target(depth));
        d.step_target = None;
        self.run_until_stop()
    }

    /// The guest's result once it has [`Stop::Finished`]; `None` while still running.
    pub fn result(&self) -> Option<&Result<Vec<Value>, Trap>> {
        self.finished.as_ref()
    }

    /// Logical time (DEBUGGING.md S3): ops executed so far on the focused thread ([`select_task`],
    /// else the stopped/sole thread).
    ///
    /// [`select_task`]: Inspector::select_task
    pub fn clock(&self) -> u64 {
        self.with_focused(|v| v.debug.as_ref().map(|d| d.clock).unwrap_or(0))
            .unwrap_or(0)
    }

    /// The number of time-travel **checkpoints** currently cached (DEBUGGING.md W1) — snapshots of the
    /// sole vCPU at ascending logical times that let `seek`/`step_back` restart from the nearest one
    /// instead of clock 0. `0` for a run that hasn't been seeked far enough to lay one down, or one
    /// outside the checkpointable subset (multithreaded, or using fibers / a stateful host capability),
    /// which falls back to replay-from-0. Introspection / test hook; does not affect results.
    pub fn checkpoint_count(&self) -> usize {
        self.checkpoints.len()
    }

    /// The focused thread's call stack, innermost frame first ([`select_task`] chooses the thread;
    /// by default the stopped/sole one). Empty once the guest has finished (or, in scheduled mode,
    /// before the first stop). Each frame carries its source location when the module has debug info
    /// (DEBUGGING.md §6/W4).
    ///
    /// [`select_task`]: Inspector::select_task
    pub fn backtrace(&self) -> Vec<FrameInfo> {
        self.with_focused(|v| {
            v.frames
                .iter()
                .rev()
                .map(|f| {
                    let pc = IrPc {
                        module: f.module,
                        func: f.func,
                        block: f.block,
                        inst: f.inst,
                    };
                    let types = frame_value_types(v, f);
                    let vals = f
                        .vals
                        .iter()
                        .enumerate()
                        .map(|(i, s)| s.to_value(types.get(i).copied().unwrap_or(ValType::I64)))
                        .collect();
                    FrameInfo {
                        pc,
                        vals,
                        source: self.source_loc(pc),
                    }
                })
                .collect()
        })
        .unwrap_or_default()
    }

    /// The source location of an [`IrPc`], if the module carries debug info (DEBUGGING.md §6/W4).
    /// Locations are recorded per statement, so this uses **nearest-preceding** within the same
    /// `(func, block)` — the op's row is the last `debug.loc` at or before it (DWARF line-table
    /// semantics). Only the guest's own program (module 0) has source; installed §22 units return
    /// `None`.
    pub fn source_loc(&self, pc: IrPc) -> Option<SourceLoc> {
        source_loc_in(self.debug_info.as_ref()?, pc)
    }

    /// The `-g` source name of function `func` (`debug_info.func_names`), or `None` when the module
    /// carried no name for it — renderers (the DAP stack trace) fall back to `func{N}`. The method
    /// counterpart of the free [`func_name`], for callers that hold an `Inspector` (DEBUGGING.md §6).
    pub fn func_name(&self, func: FuncIdx) -> Option<&str> {
        self.debug_info
            .as_ref()?
            .func_names
            .iter()
            .find(|f| f.func == func)
            .map(|f| f.name.as_str())
    }

    /// Resolve a source variable by `name` in the frame `frame_from_top` levels up (0 = innermost)
    /// and read its current value — the W4→S2 bridge: `Ssa` reads the frame's value table, `Window`
    /// reads `width` bytes from `data-SP + off`. Returns `None` if there is no debug info, no such
    /// variable in that frame's function, or the read is out of range. `width` is the byte width to
    /// read for a window variable (its type size); ignored for SSA variables.
    pub fn read_var(&self, frame_from_top: usize, name: &str, width: usize) -> Option<VarValue> {
        let di = self.debug_info.as_ref()?;
        self.with_focused(|v| {
            let n = v.frames.len();
            let idx = n.checked_sub(1 + frame_from_top)?;
            let frame = v.frames.get(idx)?;
            // A var belongs to a function; match on the frame's function (module-0 program only).
            if frame.module != 0 {
                return None;
            }
            let var = pick_var(di, frame.func, name, frame.block, frame.inst)?;
            // Resolve a location list (S2) at the frame's current pc (nearest-preceding within the
            // stopped block). `None` ⇒ no covering entry, the var isn't live here.
            let resolve = |locs: &[SsaLoc]| loclist_value(locs, frame.block, frame.inst);
            // Read `width` window bytes at `base + off`, directly (not via `read_window`, to avoid
            // re-locking).
            let window_read = |base: u64, off: i64| -> Option<VarValue> {
                let bytes = v
                    .mem
                    .as_ref()?
                    .read_window(base.wrapping_add(off as u64), width)
                    .ok()?;
                Some(VarValue::Bytes(bytes))
            };
            let types = frame_value_types(v, frame);
            match &var.loc {
                VarLoc::Ssa { value } => {
                    frame_value(frame, &types, *value as usize).map(VarValue::Value)
                }
                VarLoc::SsaList(locs) => {
                    frame_value(frame, &types, resolve(locs)? as usize).map(VarValue::Value)
                }
                VarLoc::Window { off } => {
                    // Address = data-SP (block param v0) + off.
                    window_read(frame.vals.first()?.i64() as u64, *off)
                }
                VarLoc::WindowVia { base, off } => {
                    // Address = (the per-pc base SSA value, e.g. a wasm frame pointer) + off.
                    let base_val = frame.vals.get(resolve(base)? as usize)?;
                    window_read(base_val.i64() as u64, *off)
                }
                // A module-scoped global at a fixed absolute window address (frame-independent).
                VarLoc::Fixed { addr } => window_read(*addr, 0),
            }
        })
        .flatten()
    }

    /// The window address of a memory-located source variable (`Window` / `WindowVia`), resolved at
    /// the focused thread's frame `frame_from_top` levels up — the base a debugger uses to expand an
    /// aggregate or take a member's address. `None` for an SSA-valued var (no memory address) or an
    /// unmapped/not-yet-live one.
    pub fn var_addr(&self, frame_from_top: usize, name: &str) -> Option<u64> {
        let di = self.debug_info.as_ref()?;
        self.with_focused(|v| {
            let n = v.frames.len();
            let frame = v.frames.get(n.checked_sub(1 + frame_from_top)?)?;
            if frame.module != 0 {
                return None;
            }
            let var = pick_var(di, frame.func, name, frame.block, frame.inst)?;
            let base = match &var.loc {
                VarLoc::Window { off } => frame.vals.first()?.i64() as u64 + *off as u64,
                VarLoc::WindowVia { base, off } => {
                    let idx = loclist_value(base, frame.block, frame.inst)?;
                    frame.vals.get(idx as usize)?.i64() as u64 + *off as u64
                }
                VarLoc::Fixed { addr } => *addr,
                VarLoc::Ssa { .. } | VarLoc::SsaList(_) => return None,
            };
            Some(base)
        })
        .flatten()
    }
    /// by index, so a promoted local is a direct lookup — no debug-build mode needed (DEBUGGING.md
    /// W6).
    pub fn read_ir_value(&self, frame_from_top: usize, value_idx: usize) -> Option<Value> {
        self.with_focused(|v| {
            let n = v.frames.len();
            let idx = n.checked_sub(1 + frame_from_top)?;
            let frame = v.frames.get(idx)?;
            let types = frame_value_types(v, frame);
            frame_value(frame, &types, value_idx)
        })
        .flatten()
    }

    /// Read `len` bytes from confined window address `addr` of the focused thread ([`select_task`])
    /// — the S2 `Window` resolution + raw inspection. Uses the same confinement as a guest load, so
    /// an out-of-window read faults rather than escaping. Empty if the guest declared no memory.
    ///
    /// [`select_task`]: Inspector::select_task
    pub fn read_window(&self, addr: u64, len: usize) -> Result<Vec<u8>, Trap> {
        self.with_focused(|v| match v.mem.as_ref() {
            Some(m) => m.read_window(addr, len),
            None => Ok(Vec::new()),
        })
        .unwrap_or(Ok(Vec::new()))
    }
}

/// Like [`run`], but with a caller-provided [`Host`] (the powerbox): grant the entry
/// function's capabilities into `host`, pass their handle indices in `args`, then read
/// effects (`host.stdout`, etc.) back afterwards. This is how a capability-using guest
/// is driven (§3c/§3e).
pub fn run_with_host(
    m: &Module,
    func: FuncIdx,
    args: &[Value],
    fuel: &mut u64,
    host: &mut Host,
) -> Result<Vec<Value>, Trap> {
    run_with_host_traced(m, func, args, fuel, host).0
}

/// Like [`run_with_host`], but also return the guest's **trap-time backtrace** — the call stack
/// (innermost frame first, as `IrPc`s) at the point a trap was raised, or empty if the run finished
/// cleanly (DEBUGGING.md §5 / W3; the interpreter counterpart to the JIT's `last_trap_backtrace`).
/// The interpreter reifies its call stack and doesn't unwind it on a trap, so this is the exact frame
/// chain. Resolve each `IrPc` to source with [`source_loc`]. Useful for kill diagnostics and for the
/// interp↔JIT differential fuzzer to report *where* a divergence/trap occurred.
pub fn run_with_host_traced(
    m: &Module,
    func: FuncIdx,
    args: &[Value],
    fuel: &mut u64,
    host: &mut Host,
) -> TracedRun {
    if m.funcs.get(func as usize).is_none() {
        return (Err(Trap::Malformed), Vec::new(), None);
    }
    // One linear-memory window per run, zero-initialized and lazily paged. The whole module
    // shares it. The window is a large reserved range (§4 default policy) with only `mapped`
    // backed, so an out-of-`mapped` access faults (detect-and-kill) instead of wrapping.
    let mut mem = m.memory.map(|mc| {
        let mut mm = Mem::with_reservation(DEFAULT_RESERVED_LOG2, mc.size_log2);
        mm.init_data(&m.data); // §3a/D40 data segments (copy + RO-protect)
        mm
    });
    drive(&m.funcs, func, args, fuel, &mut mem, host)
}

/// Host-carrying counterpart of [`run_fast`]: try the [`bytecode`] engine first, fall back to the
/// tree-walker [`run_with_host`]. The bytecode engine drives no runtime seams yet (no scheduler,
/// powerbox, fibers, threads, durability), so it is used only when the run needs none of them:
///
/// * the host is **not durable** (durability needs the shadow-stack swap in [`drive`]), and
/// * [`bytecode::compile_and_run`] accepts the module — it returns `None` for any op that needs a
///   seam (capability / thread / fiber / continuation / `memory.wait` ops), so an accepted module is
///   pure compute + memory + (direct/indirect) calls, which a granted-capability host never affects.
///
/// On the `None` fallback path `compile_and_run` rejects the module *before* executing, so `fuel` is
/// untouched and the tree-walker runs with the full budget.
pub fn run_with_host_fast(
    m: &Module,
    func: FuncIdx,
    args: &[Value],
    fuel: &mut u64,
    host: &mut Host,
) -> Result<Vec<Value>, Trap> {
    if !host.is_durable() {
        if let Some(result) = bytecode::compile_and_run_with_host(m, func, args, fuel, host) {
            return result;
        }
    }
    run_with_host(m, func, args, fuel, host)
}

/// Trap-time-backtrace counterpart of [`run_with_host_fast`]: the [`run_with_host_traced`] semantics
/// (return the guest's trap-time call stack — innermost frame first, as [`IrPc`]s; empty on a clean
/// finish, resolvable with [`source_loc`]) on whichever backend runs. The [`bytecode`] engine carries
/// its own backtrace ([`bytecode::compile_and_run_with_host_traced`]); when it declines the module —
/// a durable host, an out-of-subset op, or a step that reaches a concurrency seam (backtraces are
/// single-vCPU scope, DEBUGGING.md S4) — this falls back to the tree-walker [`run_with_host_traced`],
/// so the backtrace is always *some* faithful engine's, never dropped. Both backends resolve to the
/// same source program points (the bytecode engine reports tree-walker-identical [`IrPc`]s, gated by
/// `bytecode_traced.rs`), so a kill diagnostic reads the same regardless of which one ran.
pub fn run_with_host_fast_traced(
    m: &Module,
    func: FuncIdx,
    args: &[Value],
    fuel: &mut u64,
    host: &mut Host,
) -> TracedRun {
    if !host.is_durable() {
        if let Some(result) = bytecode::compile_and_run_with_host_traced(m, func, args, fuel, host)
        {
            return result;
        }
    }
    run_with_host_traced(m, func, args, fuel, host)
}

/// Capability-free [`run_with_host_fast_traced`] (an empty powerbox), the traced counterpart of
/// [`run_fast`] — mirrors [`run_traced`] on the fast path.
pub fn run_fast_traced(m: &Module, func: FuncIdx, args: &[Value], fuel: &mut u64) -> TracedRun {
    let mut host = Host::new();
    run_with_host_fast_traced(m, func, args, fuel, &mut host)
}

/// Run the entry vCPU on the M:N executor: submit the root, become a worker on the calling thread,
/// and once every vCPU has finished, join any worker threads the executor spawned and read the root's
/// outcome back. `funcs` is cloned into an `Arc<[Func]>` the vCPUs own, so a spawned vCPU borrows
/// nothing and can run on a pooled thread. A single-threaded guest never spawns a worker — the calling
/// thread runs it to completion — so non-threaded runs pay no pool overhead.
fn drive(
    funcs: &[Func],
    entry: FuncIdx,
    args: &[Value],
    fuel: &mut u64,
    mem: &mut Option<Mem>,
    host: &mut Host,
) -> TracedRun {
    drive_arc(funcs.to_vec().into(), entry, args, fuel, mem, host)
}

/// [`drive`] over an already-shared function table — the §3.2 wired-offer dispatch reuses the
/// offer's `Arc<[Func]>` verbatim instead of re-copying the table per call.
fn drive_arc(
    funcs: Arc<[Func]>,
    entry: FuncIdx,
    args: &[Value],
    fuel: &mut u64,
    mem: &mut Option<Mem>,
    host: &mut Host,
) -> TracedRun {
    let workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .clamp(1, MAX_WORKERS);
    // §15: the domain's spawn quota (already clamped to the hard ceilings by `set_quota`) sizes the
    // executor's live-vCPU cap and each vCPU's fiber cap. Default = the ceilings (unchanged behavior).
    let quota = host.quota();
    // B2 `install`: the table reservation the root vCPU builds its dispatch table with (must
    // equal the JIT's `table_reserve_log2`). Read before the host is moved into the Arc below.
    let jit_table_log2 = host.jit_table_log2();
    // Durability is a domain property (DURABILITY.md §12.8): every vCPU of a durable run maintains
    // the per-context shadow-SP swap. Read before the host moves into the shared Arc.
    let durable = host.is_durable();
    // §12.8 concurrent-thaw stage 1: a thaw restores the frozen window with the global **freeze** word
    // still `UNWINDING` (the artifact froze there), while the per-context **thaw** word now carries the
    // `REWINDING` phase. Clear the leftover freeze word to `NORMAL` up front, so the loop polls don't
    // re-unwind mid-thaw and `durable_load_dstate` reads the thaw — the real snapshot-restore path
    // leaks the same `UNWINDING`. A freeze leaves the thaw word `NORMAL`, so this never fires on it.
    if durable {
        if let Some(m) = mem.as_mut() {
            if m.durable_thaw_state(0) == STATE_REWINDING {
                m.durable_set_state(STATE_NORMAL);
            }
        }
    }
    // Durable **freeze/thaw** runs (a freeze's global word is `UNWINDING`/`ARMED`, or a thaw's
    // per-context word is `REWINDING`, not `NORMAL`)
    // serialize onto a single worker (DURABILITY.md §12.8 slice 3.2.1): the shared active shadow-SP
    // word is used by one vCPU at a time, so concurrent unwind/rewind can't race it, and the runtime
    // re-points it per vCPU on each dispatch. Ordinary runs — incl. a `NORMAL` durable run, which never
    // touches the reserve — keep full multi-worker parallelism.
    let workers = if durable
        && mem
            .as_ref()
            // §12.8 concurrent-thaw stage 1: serialize for a freeze (global word) *or* a thaw (the
            // root re-enters under `REWINDING` in its own per-context thaw word — no longer the global
            // word). The root's context is 0; `durable_load_dstate(0)` combines both words:
            // non-`NORMAL` ⇒ freeze/thaw in progress.
            .map(|m| m.durable_load_dstate(0))
            .unwrap_or(STATE_NORMAL)
            != STATE_NORMAL
    {
        1
    } else {
        workers
    };
    // Thaw seeding (slice 3.1.5): fibers a freeze flattened, to re-create in the registry before the
    // root re-enters under REWINDING. Taken (cleared) here; empty for a freeze or ordinary run.
    let thaw_fibers = std::mem::take(&mut host.frozen_fibers);
    // Thaw seeding (slice 3.2.1): spawned vCPUs a freeze flattened, re-attached by the root's rewound
    // `thread.spawn` (in ascending task order). Taken here; empty for a freeze or ordinary run.
    let thaw_vcpus = std::mem::take(&mut host.frozen_vcpus);
    let thaw_nested = std::mem::take(&mut host.frozen_nested);
    // Thaw seeding (slice 3.2.1): the root's flattened shadow-SP extent (a multi-vCPU thaw only). `None`
    // ⇒ read the extent from the restored window's active-SP word (the single-vCPU path).
    let thaw_root_sp = host.frozen_root_sp.take();
    let sched = Arc::new(Scheduler::new(quota.max_vcpus, workers));
    // The powerbox is **shared** by every vCPU of the run (so spawned threads inherit it): move the
    // caller's host into an `Arc<Mutex<Host>>`, hand a clone to the root (and, on `thread.spawn`, to
    // each child), then unwrap it back into the caller after every vCPU is gone. The root still owns
    // the run's `mem`/`fuel`, read back from its outcome.
    let host_shared = Arc::new(Mutex::new(std::mem::take(host)));
    // §9/§12 async ring: wire the completion `notify` hook to this run's M:N scheduler, so an offload
    // worker waking a vCPU parked in `wait` is a `Scheduler::notify` on the confined counter key.
    {
        let sched_for_notify = Arc::clone(&sched);
        host_shared
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .set_async_notify(Arc::new(move |key, count| {
                // The async-ring completion counter is always a normal anonymous window page
                // (`async_counter_impl` rejects backed pages), so its rendezvous key is `Anon` (S1b).
                sched_for_notify.notify(FutexKey::Anon(key), count);
            }));
    }
    let root_id = {
        let mut s = sched.lock();
        let id = s.next_task;
        s.next_task += 1;
        s.live += 1;
        s.workers = 1; // the calling thread acts as worker 0
                       // The domain's shared dispatch table (B2 `install` reserves `jit_table_log2` slots; no
                       // effect when `0`). Every vCPU of the run shares this one `Arc`, so an install is visible
                       // across `thread.spawn`/`Jit.invoke` children (DESIGN.md §22).
        let dt = Arc::new(DomainTable::new(&funcs, jit_table_log2));
        let mut root = Box::new(VCpu::new(
            Arc::clone(&funcs),
            entry,
            args,
            mem.take(),
            Arc::clone(&host_shared),
            *fuel,
            0,
            id,
            SchedRef::Real(Arc::clone(&sched)),
            quota,
            Arc::clone(&dt),
        ));
        root.durable = durable;
        // The root's own durable state word is the window's initial phase (NORMAL / UNWINDING freeze /
        // REWINDING thaw); the runtime swaps it per vCPU from here (slice 3.2.1). §12.8 concurrent-thaw
        // stage 1: combine the global freeze word with the root's per-context thaw word (a thaw seeds
        // `REWINDING` into the latter).
        root.dstate = root
            .mem
            .as_ref()
            .map(|m| m.durable_load_dstate(root.vcpu_ctx))
            .unwrap_or(STATE_NORMAL);
        // The root's active shadow-SP: its flattened extent on a multi-vCPU thaw (recorded residue), or
        // the window's active-SP word otherwise (a fresh/freeze run leaves it at `SHADOW_BASE`; a
        // single-vCPU thaw's window already holds the root's extent). The runtime swaps it in per
        // dispatch (slice 3.2.1).
        let root_word = shadow_region_base(root.vcpu_ctx);
        root.root_shadow_sp = thaw_root_sp.unwrap_or_else(|| {
            root.mem
                .as_ref()
                .map(|m| m.durable_get_sp(root_word))
                .unwrap_or_else(|| shadow_frame_base(root.vcpu_ctx))
        });
        // Thaw seeding (slice 3.1.5): re-create each frozen fiber in the run-shared registry, in
        // ascending slot order, so the dense handle namespace matches the freeze (the root's
        // re-issued `cont.resume` names handle 0, …). Each fiber's flattened shadow-SP goes back in
        // the `shadow` table so the swap re-points to it when the root re-enters it under REWINDING.
        {
            let mut seed: Vec<FrozenFiber> = thaw_fibers;
            seed.sort_by_key(|f| f.slot);
            for (expected, ff) in seed.into_iter().enumerate() {
                let got = root
                    .registry
                    .seed_frozen(ff.func, ff.sp, ff.shadow_sp, ff.generation);
                debug_assert_eq!(got, expected, "frozen fibers re-seed densely from slot 0");
                debug_assert_eq!(got, ff.slot, "re-seeded slot matches the recorded handle");
            }
        }
        // Thaw re-spawn (slice 3.2.1): reconstruct the spawned vCPUs a freeze flattened. The root's
        // rewind *skips* its prologue `thread.spawn` (the REWINDING prologue jumps straight to the
        // resume ARM), so a child that existed before the freeze point is **not** re-created by the
        // root — the runtime re-creates it here, under `REWINDING`, with its flattened shadow-SP
        // restored, so it rewinds from its frozen point and runs forward. Children re-spawn in
        // ascending task (= spawn) order; their regions return via the restored shadow-SP, and the
        // root's `threads` (join) table is rebuilt to map each handle slot to its child — so the root's
        // re-executed `thread.join` (after its checkpoint) resolves. As of slice 3.2.2 the root may
        // also own fibers (top-down vCPU contexts vs. up-growing fiber contexts no longer collide).
        // Only the root's *direct* children are handled (flat spawns); nested spawns and a *spawned*
        // child owning fibers (per-child freeze_drive) are follow-ups.
        {
            let mut vseed: Vec<FrozenVCpu> = thaw_vcpus;
            // Ascending task = ascending spawn order across the whole tree, and a parent's id is always
            // < its children's (it was spawned first), so this order re-attaches **parents before
            // children** — essential for nested spawns (slice 3.4): a grandchild's handle is rebuilt
            // into its *parent child's* table, which must already exist.
            vseed.sort_by_key(|f| f.task);
            // Per-piece rebuild (none of "top `n`, densely" holds with recycling / nesting):
            //   • context — *derived* from the restored shadow-SP (`(sp − SHADOW_BASE) / STRIDE`), since
            //     the region rides in the absolute shadow-SP; collected into the occupancy mask so a
            //     post-thaw spawn lands in a genuinely-free context.
            //   • task id — *preserved* (`cid = ff.task`), so the §12.6 canonical re-freeze is byte-identical.
            //   • join handle — appended into the **parent's** `threads` in ascending-task (= spawn)
            //     order, so the guest's reloaded handle resolves in the table of whoever spawned it
            //     (the root for a direct child, a re-spawned child for a grandchild).
            let mut vcpu_mask: u16 = 0;
            // Re-spawned children held by task id so a grandchild can attach to its (already re-spawned)
            // parent; the root is mutated directly. `BTreeMap` keeps the enqueue order ascending-task.
            let mut children: std::collections::BTreeMap<TaskId, Box<VCpu>> =
                std::collections::BTreeMap::new();
            for ff in vseed {
                let cid = ff.task as TaskId;
                s.next_task = s.next_task.max(cid + 1);
                s.live += 1;
                let ctx = ((ff.shadow_sp - SHADOW_BASE) / SHADOW_STRIDE) as usize;
                vcpu_mask |= 1 << ctx;
                let child_mem = root.mem.as_ref().map(|m| m.fork_for_thread());
                let mut child = Box::new(VCpu::new(
                    Arc::clone(&funcs),
                    ff.func as FuncIdx,
                    &[
                        Value::I64(ff.args.first().copied().unwrap_or(0)),
                        Value::I64(ff.args.get(1).copied().unwrap_or(0)),
                    ],
                    child_mem,
                    Arc::clone(&host_shared),
                    *fuel,
                    0,
                    cid,
                    SchedRef::Real(Arc::clone(&sched)),
                    quota,
                    Arc::clone(&dt),
                ));
                child.registry = Arc::clone(&root.registry);
                child.durable = true;
                child.dstate = STATE_REWINDING; // re-enter under rewind, from its restored extent
                child.root_shadow_sp = ff.shadow_sp;
                child.vcpu_ctx = ctx; // freed on a post-thaw finish, like a freshly-spawned child
                child.parent_task = ff.parent_task as TaskId;
                child.spawn_residue = Some((ff.func as FuncIdx, ff.args.clone()));
                // Append the handle into the spawning vCPU's join table (root, or a re-spawned child).
                let parent = ff.parent_task as TaskId;
                if parent == id {
                    root.threads.push(Some(cid));
                } else if let Some(p) = children.get_mut(&parent) {
                    p.threads.push(Some(cid));
                }
                // (A parent not in the set can't happen on a dense freeze — every live ancestor unwinds
                // too; a missing handle would surface as a clean `ThreadFault`, not a mis-attach.)
                children.insert(cid, child);
            }
            // Seed the registry's vCPU-context occupancy from the re-spawned children (recycling).
            root.registry.seed_vcpu_mask(vcpu_mask);
            // Enqueue every re-spawned child, parents first (ascending task via the `BTreeMap`).
            for (_, child) in children {
                s.runnable.push_back(child);
            }
        }
        // §4 subtree thaw (DURABILITY.md): re-attach the §14 nested children a subtree freeze
        // recorded — now to **arbitrary depth** (parent→child→grandchild, …). Each child's whole
        // state — window, durable reserve, unwound continuation — is already in the restored image
        // (its carve is a sub-range of the *root's* window, at any nesting depth); this re-creates
        // each child *domain* around it: a nested view of the carve, a fresh attenuated powerbox
        // (the same grants, in the same order, as `instantiate` minted — so the child's reloaded
        // handle values still resolve), and a `REWINDING` re-entry from the extent its carve's own
        // shadow-SP word holds. Mirrors the `thread.spawn` [`FrozenVCpu`] two-phase re-attach
        // (above): the residue is grouped by `parent_task` so each child's handle is rebuilt into
        // **its own parent's** join table (the root for a direct child, a re-created child for a
        // grandchild) — so every re-executed `join` (root's *and* a child's) resolves and parks
        // until its rewound child completes, exactly as pre-freeze.
        {
            let mut nseed: Vec<FrozenNested> = thaw_nested;
            // Sort by `parent_task` then `slot`: a parent's task id is always < its children's (it
            // was instantiated first), so this re-attaches **parents before their grandchildren** —
            // a parent-child VCpu exists before any of its children attach to it. `slot` is the
            // deterministic tiebreak (the freeze's canonical order). The subtree freeze reproduces
            // the same dense task ids as the freeze (root = 0, then the sorted-order children get
            // 1, 2, …), so a grandchild's recorded `parent_task` equals its parent-child's fresh cid
            // here — the key `children` is stored under.
            nseed.sort_by(|a, b| a.parent_task.cmp(&b.parent_task).then(a.slot.cmp(&b.slot)));
            // Re-created child domains, held by their fresh cid so a grandchild can attach to its
            // (already re-created) parent-child; the root is mutated directly. `BTreeMap` keeps the
            // enqueue order ascending-cid (parents first).
            let mut children: std::collections::BTreeMap<TaskId, Box<VCpu>> =
                std::collections::BTreeMap::new();
            // Each re-created child's **absolute** (root-window-relative) carve offset. A record's
            // `carve_off` is relative to *its parent's* window (the parent's Instantiator base is 0
            // in its own view), so a grandchild's offset into the root image is its parent-child's
            // absolute base + its own recorded offset — accumulated down the chain here.
            let mut abs_off: std::collections::BTreeMap<TaskId, u64> =
                std::collections::BTreeMap::new();
            for fnr in nseed {
                let parent = fnr.parent_task as TaskId;
                // The parent-child's absolute carve base (0 for a direct child of the root).
                let parent_base = if parent == id {
                    0
                } else {
                    abs_off.get(&parent).copied().unwrap_or(0)
                };
                let abs_carve = parent_base + fnr.carve_off;
                // A **completed** child is not re-created (reload-not-reissue): its result is posted
                // straight into the scheduler and mapped to the recording parent's join slot, so the
                // parent's re-executed `thread.join` delivers it without re-running the child.
                if let Some(r) = fnr.completed_result {
                    let cid = s.next_task;
                    s.next_task += 1;
                    s.results.insert(
                        cid,
                        Outcome {
                            result: Ok(vec![Value::I64(r)]),
                            mem: None,
                            fuel: *fuel,
                            trap_bt: Vec::new(),
                            trap_fiber: None,
                        },
                    );
                    if parent == id {
                        while root.threads.len() <= fnr.slot {
                            root.threads.push(None);
                        }
                        root.threads[fnr.slot] = Some(cid);
                    } else if let Some(p) = children.get_mut(&parent) {
                        while p.threads.len() <= fnr.slot {
                            p.threads.push(None);
                        }
                        p.threads[fnr.slot] = Some(cid);
                    }
                    abs_off.insert(cid, abs_carve);
                    continue;
                }
                let csize = 1u64 << fnr.size_log2;
                // Resolve the child's function table: the parent's own for a same-module child, or a
                // re-granted **separate module** matched by content digest (host-supplied at restore).
                // A missing / mismatched re-grant leaves the join slot empty, so the recording
                // parent's re-executed `thread.join` fails closed — the per-child R5 identity gate.
                let cfuncs = match fnr.module_digest {
                    None => Some(Arc::clone(&funcs)),
                    Some(d) => host_shared
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .module_by_digest(&d),
                };
                let Some(cfuncs) = cfuncs else {
                    if parent == id {
                        while root.threads.len() <= fnr.slot {
                            root.threads.push(None);
                        }
                    } else if let Some(p) = children.get_mut(&parent) {
                        while p.threads.len() <= fnr.slot {
                            p.threads.push(None);
                        }
                    }
                    continue;
                };
                // Flip the child's carve from its frozen phase to a thaw: clear its own global
                // freeze word and set its context-0 thaw word — `begin_thaw`, at the **absolute**
                // carve offset (the carve rides the root image at any depth).
                if let Some(m) = root.mem.as_mut() {
                    let _ = m.write_bytes(abs_carve + STATE_OFF, &STATE_NORMAL.to_le_bytes());
                    let _ = m.write_bytes(
                        abs_carve + thaw_state_off(0),
                        &STATE_REWINDING.to_le_bytes(),
                    );
                }
                // The child's flattened extent: a nested child is a single-vCPU durable domain in
                // its carve (`vcpu_ctx == 0`), so its shadow-SP word sits at its context-0 **region
                // base** (`DurableShadowBase`, where the transform reads/writes it) — *not* the old
                // fixed global `SHADOW_SP_OFF`. A leaf child re-runs idempotently from base, so the
                // read location was immaterial before; a **depth-2 middle child** reloads its
                // `instantiate` checkpoint on rewind, so it must resume from the true drained extent.
                let child_extent = root
                    .mem
                    .as_ref()
                    .map(|m| m.durable_get_sp(abs_carve + shadow_region_base(0)))
                    .unwrap_or_else(|| shadow_frame_base(0));
                let child_mem = root
                    .mem
                    .as_ref()
                    .map(|m| m.nested_view(m.window.base() + abs_carve, fnr.size_log2));
                let mut ch = Host::new();
                ch.set_durable(true);
                let cinst = ch.grant_instantiator(0, csize);
                let want_as = cfuncs
                    .get(fnr.entry as usize)
                    .is_some_and(|f| f.params == [ValType::I64, ValType::I64]);
                let child_args = if want_as {
                    let cas = ch.grant_address_space(0, csize);
                    vec![Value::I64(cinst as i64), Value::I64(cas as i64)]
                } else {
                    vec![Value::I64(cinst as i64)]
                };
                let cid = s.next_task;
                s.next_task += 1;
                s.live += 1;
                // True nesting depth: a direct child of the root is depth 1; a grandchild is its
                // parent-child's depth + 1 (the parent is already in `children`, built first).
                let cdepth = if parent == id {
                    1
                } else {
                    children.get(&parent).map(|p| p.depth + 1).unwrap_or(1)
                };
                let cdt = Arc::new(DomainTable::new(&cfuncs, 0));
                let mut child = Box::new(VCpu::new(
                    Arc::clone(&cfuncs),
                    fnr.entry,
                    &child_args,
                    child_mem,
                    Arc::new(Mutex::new(ch)),
                    *fuel,
                    cdepth,
                    cid,
                    SchedRef::Real(Arc::clone(&sched)),
                    quota,
                    cdt,
                ));
                child.durable = true;
                child.nested_child = true;
                child.dstate = STATE_REWINDING;
                child.root_shadow_sp = child_extent;
                child.parent_task = parent;
                // §4 depth-2+: a great-grandchild's residue coalesces in the root host too (the same
                // shared sink the freeze used), so re-freezing a thawed deep subtree stays canonical.
                child.freeze_sink = Some(Arc::clone(&host_shared));
                let info = NestedChildInfo {
                    slot: fnr.slot,
                    carve_off: fnr.carve_off,
                    size_log2: fnr.size_log2,
                    entry: fnr.entry,
                    module_digest: fnr.module_digest,
                };
                // Rebuild the recording parent's join table + nested-child record at the recorded
                // slot (the root for a direct child, a re-created child for a grandchild).
                if parent == id {
                    while root.threads.len() <= fnr.slot {
                        root.threads.push(None);
                    }
                    root.threads[fnr.slot] = Some(cid);
                    root.nested_children.push(info);
                } else if let Some(p) = children.get_mut(&parent) {
                    while p.threads.len() <= fnr.slot {
                        p.threads.push(None);
                    }
                    p.threads[fnr.slot] = Some(cid);
                    p.nested_children.push(info);
                }
                abs_off.insert(cid, abs_carve);
                children.insert(cid, child);
            }
            // Enqueue every re-created child, parents first (ascending cid via the `BTreeMap`).
            for (_, child) in children {
                s.runnable.push_back(child);
            }
        }
        s.runnable.push_back(root);
        id
    };
    // Run as worker 0 until the run shuts down (every vCPU finished), then join spawned workers.
    worker_loop(&sched);
    let handles = std::mem::take(&mut sched.lock().handles);
    for h in handles {
        let _ = h.join();
    }
    // Drain any in-flight async-ring offload jobs (they hold the window's `Arc<Region>` and may still
    // be writing the futex counter) and drop the `notify` hook's `Arc<Scheduler>` before reading the
    // final window back. Safe to lock: all vCPUs are gone, so the shared host is otherwise idle.
    {
        let mut h = host_shared.lock().unwrap_or_else(|e| e.into_inner());
        h.quiesce_pool();
        h.clear_async_notify();
    }
    let (out, trap_origin) = {
        let mut s = sched.lock();
        let out = s.results.remove(&root_id).expect("root vCPU finished");
        (out, s.trap_origin.take())
    };
    *fuel = out.fuel;
    *mem = out.mem;
    // Every vCPU (which held an Arc clone) is finished and dropped now, so the shared host is uniquely
    // owned — unwrap it back into the caller so it observes the run's effects (stdout, grants, clock…).
    *host = Arc::try_unwrap(host_shared)
        .unwrap_or_else(|_| unreachable!("all vCPUs dropped before host readback"))
        .into_inner()
        .unwrap_or_else(|e| e.into_inner());
    // Prefer the trap-origin capture (the first vCPU to actually trap) over the root's own outcome,
    // which for a join-propagated child trap names the join site, not the origin. `None` ⇒ clean run
    // (use the root's empty backtrace). This mirrors the JIT's `root_trap_cap.or(worker_trap_cap)`.
    let (trap_bt, trap_fiber) = trap_origin.unwrap_or((out.trap_bt, out.trap_fiber));
    (out.result, trap_bt, trap_fiber)
}

/// Like [`run`], but seed the window with `init_mem` (its low bytes) and return the final
/// window contents (the same number of bytes) alongside the result. This is the
/// **escape-oracle** path (§18): a *verified* module must keep every access in-window, so a
/// run that completes without trapping must leave a window byte-identical to the JIT's. The
/// non-zero seed makes a divergent (e.g. under-masked) *read* observable, not just a write.
/// With no declared memory the snapshot is empty.
pub fn run_capture(
    m: &Module,
    func: FuncIdx,
    args: &[Value],
    fuel: &mut u64,
    init_mem: &[u8],
) -> (Result<Vec<Value>, Trap>, Vec<u8>) {
    // Default reservation policy (§4): a large reserved range, only `mapped` backed.
    run_capture_reserved(m, func, args, fuel, init_mem, DEFAULT_RESERVED_LOG2)
}

/// Like [`run_capture`], but with a host **reservation policy**: only the declared `1 << size_log2`
/// bytes are backed within a `2^reserved_log2` reservation, so an access outside `[0, mapped)` —
/// whether past the backed prefix, into the reserved-but-unmapped tail, or wildly out of range —
/// faults (`Trap::MemoryFault`) under trap-confinement (the §4 "guard-when-bounded" model; no
/// wrapping/aliasing). `reserved_log2` is raised
/// to at least `size_log2` (so `0` ⇒ fully mapped). This is the interpreter side of the
/// escape-oracle under the decoupled model and must be driven with the **same** `reserved_log2`
/// as the JIT's [`svm_jit::compile_and_run_capture_reserved`] to stay in differential lockstep.
pub fn run_capture_reserved(
    m: &Module,
    func: FuncIdx,
    args: &[Value],
    fuel: &mut u64,
    init_mem: &[u8],
    reserved_log2: u8,
) -> (Result<Vec<Value>, Trap>, Vec<u8>) {
    let mut host = Host::new();
    if m.funcs.get(func as usize).is_none() {
        return (Err(Trap::Malformed), Vec::new());
    }
    let mut mem = m.memory.map(|mc| {
        let mut mm = Mem::with_reservation(reserved_log2, mc.size_log2);
        mm.seed(init_mem);
        mm.init_data(&m.data); // §3a/D40 data segments (after the escape-oracle seed)
        mm
    });
    let (r, ..) = drive(&m.funcs, func, args, fuel, &mut mem, &mut host);
    let snap = mem
        .as_ref()
        .map(|mm| mm.snapshot(init_mem.len() as u64))
        .unwrap_or_default();
    (r, snap)
}

/// Like [`run_capture_reserved`], but with a caller-provided [`Host`] (the powerbox), so a
/// `cap.call` to a *granted* handle takes its **success** path while the final-window snapshot
/// still feeds the escape-oracle (§18). Pairs with the JIT's
/// [`svm_jit::compile_and_run_capture_reserved_with_host`]: running both lets the §3e Memory
/// capability's `map`/`unmap`/`protect` effects be byte-compared across backends, not just their
/// return values — a real generative escape-oracle for the capability path.
/// Escape-oracle snapshot span (the `_with_host` capture): byte-compare the low `SNAP_CAP` bytes of
/// the window — *including* reserved-tail pages the guest grew via the Memory cap, not just the
/// backed prefix. **Must match `svm_jit`'s `SNAP_CAP`** so both backends snapshot the same span.
const SNAP_CAP: usize = 1 << 18; // 256 KiB

pub fn run_capture_reserved_with_host(
    m: &Module,
    func: FuncIdx,
    args: &[Value],
    fuel: &mut u64,
    init_mem: &[u8],
    reserved_log2: u8,
    host: &mut Host,
) -> (Result<Vec<Value>, Trap>, Vec<u8>) {
    if m.funcs.get(func as usize).is_none() {
        return (Err(Trap::Malformed), Vec::new());
    }
    let mut mem = m.memory.map(|mc| {
        let mut mm = Mem::with_reservation(reserved_log2, mc.size_log2);
        mm.seed(init_mem);
        mm.init_data(&m.data);
        mm
    });
    let (r, ..) = drive(&m.funcs, func, args, fuel, &mut mem, host);
    // Snapshot past the backed prefix to also cover reserved-tail pages the guest grew (the §1a
    // growth path), matching the JIT's `_with_host` capture span so the escape-oracle byte-compares
    // them too.
    let snap = mem
        .as_ref()
        .map(|mm| mm.snapshot_window(SNAP_CAP))
        .unwrap_or_default();
    (r, snap)
}

/// The durable snapshot's window-image page granularity (DURABILITY.md §12.3 / `svm-snapshot`'s
/// `PAGE`). Protections are captured at this fixed size — independent of the host page size — so
/// an artifact is portable across hosts. A 4 KiB codec page sits within one host page (host
/// pages are `≥ 4 KiB`), so each codec page's protection is that of its containing host page.
pub const DURABLE_SNAPSHOT_PAGE: u64 = 4096;

/// Per-page protection of a captured window region, for the durable snapshot (DURABILITY.md
/// §12.3). A faithful view of the interpreter's page model; `Backed` is the §13
/// `SharedRegion`-aliased case a durable snapshot must reject (D-region — freeze refuses).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CapturedProt {
    /// Read/write (the default for a committed prefix page).
    Rw,
    /// Read-only — e.g. a D40 `readonly` data segment, or a guest `protect`.
    Ro,
    /// Unmapped / uncommitted — any access faults.
    Unmapped,
    /// A §13 `SharedRegion`-aliased page — not snapshottable in v1 (D-region).
    Backed,
}

/// [`run_capture_reserved_with_host`] that **seeds** an initial per-page protection map (the
/// durable-restore step — re-establishing `Ro`/`Unmapped` pages so a thawed guest faults exactly
/// as the frozen one would) and **returns** the post-run protection map (the durable-freeze step),
/// both one [`CapturedProt`] per [`DURABLE_SNAPSHOT_PAGE`]-byte page (DURABILITY.md §12.3). Pass
/// `init_prots = None` to capture only (every page starts at its default). A `Backed` entry in
/// `init_prots` is ignored — a §13 shared-region alias is not restorable (D-region; freeze refuses
/// it), so the embedder re-grants the region instead.
#[allow(clippy::too_many_arguments)]
pub fn run_capture_reserved_with_host_prots(
    m: &Module,
    func: FuncIdx,
    args: &[Value],
    fuel: &mut u64,
    init_mem: &[u8],
    init_prots: Option<&[CapturedProt]>,
    reserved_log2: u8,
    host: &mut Host,
) -> (Result<Vec<Value>, Trap>, Vec<u8>, Vec<CapturedProt>) {
    if m.funcs.get(func as usize).is_none() {
        return (Err(Trap::Malformed), Vec::new(), Vec::new());
    }
    let mut mem = m.memory.map(|mc| {
        let mut mm = Mem::with_reservation(reserved_log2, mc.size_log2);
        mm.seed(init_mem);
        mm.init_data(&m.data);
        if let Some(prots) = init_prots {
            mm.apply_prots(prots);
        }
        mm
    });
    let (r, ..) = drive(&m.funcs, func, args, fuel, &mut mem, host);
    let (snap, prots) = mem
        .as_ref()
        .map(|mm| (mm.snapshot_window(SNAP_CAP), mm.snapshot_prots(SNAP_CAP)))
        .unwrap_or_default();
    (r, snap, prots)
}

/// Run the guest confined to a §14 **nested sub-window** `[base, base+size)` of a fully-backed
/// parent of `parent_bytes` (the child runs over the parent's `Region`; `size = 1 << size_log2` is
/// the module's declared memory). The confinement unit ([`svm_mask::Window::sub`]) bounds every child
/// access to its slice (faulting otherwise), so a *verified* guest reaches only `[base, base+size)`. This is the
/// interpreter side of the **sub-window escape-oracle**: pair it with the JIT's
/// [`svm_jit::compile_and_run_capture_sub`] and byte-compare the whole parent — every byte outside
/// the slice must stay as seeded (confinement) and the slice must match the JIT (codegen). `init_mem`
/// seeds the whole parent; the returned `Vec` is the whole parent window (`parent_bytes` bytes).
pub fn run_capture_sub(
    m: &Module,
    func: FuncIdx,
    args: &[Value],
    fuel: &mut u64,
    init_mem: &[u8],
    base: u64,
    parent_bytes: u64,
) -> (Result<Vec<Value>, Trap>, Vec<u8>) {
    let mut host = Host::new();
    if m.funcs.get(func as usize).is_none() {
        return (Err(Trap::Malformed), Vec::new());
    }
    let mut mem = m.memory.map(|mc| {
        let mut mm = Mem::sub_window(base, mc.size_log2, parent_bytes);
        mm.seed_parent(init_mem); // seed the whole parent, not just the child slice
        mm.init_data_at(&m.data, base); // child-relative segments shifted into the slice
        mm
    });
    let (r, ..) = drive(&m.funcs, func, args, fuel, &mut mem, &mut host);
    let snap = mem
        .as_ref()
        .map(|mm| mm.snapshot_parent(parent_bytes))
        .unwrap_or_default();
    (r, snap)
}

/// Run a module under the **deterministic explorer** (§18) with scheduling decisions driven by
/// `seed`: a single OS thread interleaves the guest's vCPUs (green threads) cooperatively, so the run
/// is fully reproducible and sweeping seeds enumerates distinct interleavings. This is the
/// verification driver for concurrent guest code — no wall-clock, no OS-scheduler nondeterminism, so
/// a failing interleaving is replayable from its seed. Returns the entry vCPU's result (or the trap /
/// `ThreadFault` on a guest deadlock). Memory is default-reserved + data-initialized; no powerbox.
pub fn run_scheduled(
    m: &Module,
    func: FuncIdx,
    args: &[Value],
    fuel: u64,
    seed: u64,
) -> Result<Vec<Value>, Trap> {
    if m.funcs.get(func as usize).is_none() {
        return Err(Trap::Malformed);
    }
    let funcs: Arc<[Func]> = m.funcs.clone().into();
    let mem = m.memory.map(|mc| {
        let mut mm = Mem::with_reservation(DEFAULT_RESERVED_LOG2, mc.size_log2);
        mm.init_data(&m.data);
        mm
    });
    let det = Arc::new(DetSched::new(seed, MAX_VCPUS));
    let root_id = {
        let mut s = det.lock();
        let id = s.next_task;
        s.next_task += 1;
        s.live += 1;
        let dt = Arc::new(DomainTable::new(&funcs, 0)); // DPOR: no Jit install, natural table
        let root = Box::new(VCpu::new(
            funcs,
            func,
            args,
            mem,
            Arc::new(Mutex::new(Host::new())),
            fuel,
            0,
            id,
            SchedRef::Det(Arc::clone(&det)),
            Quota::default(), // deterministic oracle path: the fixed anti-bomb ceilings
            dt,
        ));
        s.runnable.push(root);
        id
    };
    run_det(&det);
    let out = det.lock().results.remove(&root_id);
    match out {
        Some(out) => out.result,
        None => Err(Trap::ThreadFault), // could not complete (a guest join-deadlock)
    }
}

/// Drives one run's scheduling choices for the exhaustive model checker, and records enough to walk
/// the next branch. `plan` is the choice to make at each decision point (a runnable set with >1
/// member); points past its end default to choice 0. `branches`/`chosen` log this run's actual
/// fan-out and choices so the caller can backtrack.
struct Choices {
    plan: Vec<usize>,
    branches: Vec<usize>,
    chosen: Vec<usize>,
    depth: usize,
}

impl Choices {
    fn new(plan: Vec<usize>) -> Choices {
        Choices {
            plan,
            branches: Vec::new(),
            chosen: Vec::new(),
            depth: 0,
        }
    }

    /// Pick a runnable index given `n` choices. A singleton runnable set is not a real decision (and
    /// isn't recorded), so the plan stays compact and stable across replays.
    fn pick(&mut self, n: usize) -> usize {
        if n == 1 {
            return 0;
        }
        let c = self.plan.get(self.depth).copied().unwrap_or(0).min(n - 1);
        self.branches.push(n);
        self.chosen.push(c);
        self.depth += 1;
        c
    }
}

/// The result of exhaustively exploring a concurrent program's interleavings ([`explore_all`]).
#[derive(Debug)]
pub struct Exhaustive {
    /// Every **distinct** terminal outcome observed across all explored schedules. For a correct
    /// program with an interleaving-invariant result this is a single element.
    pub outcomes: Vec<Result<Vec<Value>, Trap>>,
    /// How many complete schedules were run.
    pub schedules: u64,
    /// `true` if the whole interleaving tree was enumerated; `false` if `max_schedules` cut it short
    /// (so `outcomes` is a sound under-approximation, not a proof over *all* interleavings).
    pub complete: bool,
}

/// The shared object a visible op touches, used by [`explore_all`]'s DPOR to decide which
/// transitions **commute** (independent ⇒ their order is irrelevant) vs. **conflict** (dependent ⇒
/// both orders must be explored). Memory/atomic accesses are a confined byte range + read/write;
/// futex `wait`/`notify` are modelled as a read/write of their (confined, in-window) key, so they
/// also conflict with atomic accesses to the same word. `thread.spawn`/`join` carry no racy object —
/// their ordering is already enforced by the scheduler's *enabled* set (a child isn't runnable before
/// its spawn; a joiner/waiter is parked, not a scheduling choice).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum MemAccess {
    /// No shared object (pure control/sync whose order the enabled set already fixes).
    None,
    /// A `[base, base+width)` byte range; `write` is true for any store/RMW/notify.
    Range { base: u64, width: u32, write: bool },
    /// A §12 fiber op (`cont.new`/`cont.resume`/`suspend`) on the **run-shared registry** (D57).
    /// Every fiber op mutates that one shared object (a slot claim/publish, or the allocation
    /// cursor that determines the next handle value), so any two fiber ops by different vCPUs are
    /// conservatively dependent — racing `cont.resume`s decide a winner, and `cont.new` order
    /// decides handle values, so both orders must be explored. (Per-slot precision would only be a
    /// DPOR-efficiency upgrade, not a soundness need.)
    Fiber,
}

impl MemAccess {
    /// Two transitions are **dependent** (don't commute) iff they touch overlapping bytes and at
    /// least one writes — the standard read/write conflict relation DPOR reduces over.
    fn conflicts(self, other: MemAccess) -> bool {
        match (self, other) {
            (
                MemAccess::Range {
                    base: a,
                    width: wa,
                    write: wwa,
                },
                MemAccess::Range {
                    base: b,
                    width: wb,
                    write: wwb,
                },
            ) => (wwa || wwb) && a < b.saturating_add(wb as u64) && b < a.saturating_add(wa as u64),
            (MemAccess::Fiber, MemAccess::Fiber) => true,
            _ => false,
        }
    }
}

/// The confined object a visible instruction will access, computed from the live SSA values at the
/// decision point (mirrors what `load`/`store`/`atomic_*`/`prepare_wait` confine to). A confinement
/// failure (out-of-reserved) ⇒ [`MemAccess::None`]: the op will trap and the thread ends, contributing
/// no ordering constraint.
fn access_of(inst: &Inst, vals: &[Reg], mem: &Option<Mem>) -> MemAccess {
    // Fiber ops touch the run-shared registry, not the window — classified before the memory
    // check (a module with no linear memory can still race on fibers via handles in args).
    // `gc.roots` reads every fiber's frames, so it is conservatively dependent on all of them too
    // (it is meant to run under stop-the-world quiescence — GC.md §2 — so this only matters to the
    // adversarial concurrency explorer, which should still order it against every fiber op).
    if matches!(
        inst,
        Inst::ContNew { .. }
            | Inst::ContResume { .. }
            | Inst::Suspend { .. }
            | Inst::GcRoots { .. }
    ) {
        return MemAccess::Fiber;
    }
    let Some(m) = mem.as_ref() else {
        return MemAccess::None;
    };
    let range = |addr: ValIdx, offset: u64, width: u32, write: bool| -> MemAccess {
        match get(vals, addr).map(|s| s.i64()) {
            Ok(a) => match m.confine_checked(a as u64, offset, width) {
                Ok(base) => MemAccess::Range { base, width, write },
                Err(_) => MemAccess::None,
            },
            Err(_) => MemAccess::None,
        }
    };
    match inst {
        Inst::Load {
            op, addr, offset, ..
        } => range(*addr, *offset, op.info().2, false),
        Inst::Store {
            op, addr, offset, ..
        } => range(*addr, *offset, op.info().2, true),
        Inst::AtomicLoad {
            ty, addr, offset, ..
        } => range(*addr, *offset, atomic_width(*ty), false),
        Inst::AtomicStore {
            ty, addr, offset, ..
        } => range(*addr, *offset, atomic_width(*ty), true),
        Inst::AtomicRmw {
            ty, addr, offset, ..
        } => range(*addr, *offset, atomic_width(*ty), true),
        Inst::AtomicCmpxchg {
            ty, addr, offset, ..
        } => range(*addr, *offset, atomic_width(*ty), true),
        // Futex key: `wait` reads it (compared under the lock), `notify` writes it (the wake). Width 4
        // is the common i32 futex; an overlapping i64 atomic at the same word still conflicts by range.
        Inst::MemoryWait { ty, addr, .. } => range(*addr, 0, atomic_width(*ty), false),
        Inst::MemoryNotify { addr, .. } => range(*addr, 0, 4, true),
        _ => MemAccess::None, // ThreadSpawn / ThreadJoin: ordering via the enabled set, not a race
    }
}

/// One executed transition in a schedule: which vCPU ran (`tid`), the runnable vCPUs at that decision
/// (`enabled`, sorted — the choices that were available), and the object it touched (`access`). The
/// trace of these is what DPOR analyses for races.
struct SchedEvent {
    tid: TaskId,
    enabled: Vec<TaskId>,
    access: MemAccess,
}

/// Drives one schedule under DPOR with **sleep sets** (Flanagan–Godefroid). Follow `plan` (one
/// `TaskId` per decision); past its end pick the smallest enabled `TaskId` that is **not asleep**. As
/// it descends it carries the current `sleep` set — threads whose exploration from here would be
/// redundant: a thread sleeps in a subtree once a sibling *independent* with it has been explored, and
/// wakes the moment a *conflicting* transition runs. `prior[d]` supplies the siblings already explored
/// at depth `d` (with their accessed objects), so the inherited sleep is reconstructed during replay.
/// Records the executed `trace` for race analysis; sets `blocked` when every enabled vCPU is asleep
/// (a redundant prefix whose completions were covered by other schedules — the run stops, contributing
/// no outcome).
struct Dpor {
    plan: Vec<TaskId>,
    prior: Vec<BTreeMap<TaskId, MemAccess>>,
    depth: usize,
    sleep: BTreeMap<TaskId, MemAccess>,
    trace: Vec<SchedEvent>,
    pending: Option<(TaskId, Vec<TaskId>)>,
    blocked: bool,
    /// `Some(state)` ⇒ extend the schedule *randomly* (seeded fuzzing for the debugger's
    /// `SchedTape`, DEBUGGING.md W1) instead of the deterministic smallest-id greedy. Only set by the
    /// `Inspector` (which runs one schedule); the explorer leaves it `None` so its DPOR backtracking
    /// stays deterministic. The executed choices still land in `trace`, so the run replays exactly.
    rng: Option<u64>,
}

impl Dpor {
    fn new(plan: Vec<TaskId>, prior: Vec<BTreeMap<TaskId, MemAccess>>) -> Dpor {
        Dpor {
            plan,
            prior,
            depth: 0,
            sleep: BTreeMap::new(),
            trace: Vec::new(),
            pending: None,
            blocked: false,
            rng: None,
        }
    }

    /// Choose the next vCPU by `TaskId` (not runnable index — the runnable order is reshuffled by
    /// `swap_remove`, so addressing by id keeps the plan stable across replays). Returns `None` when
    /// every enabled vCPU is asleep, so the caller stops this (redundant) run. `enabled` is sorted.
    fn pick(&mut self, enabled: &[TaskId]) -> Option<TaskId> {
        let tid = if self.depth < self.plan.len() {
            // Forced replay (incl. a race-woken thread): the planned choice overrides the sleep set.
            self.plan[self.depth]
        } else {
            // Extension past the plan: the smallest enabled, non-asleep thread (deterministic), or —
            // when seeded for fuzzing — a random one. Either way the choice is recorded in `trace`.
            let chosen = if let Some(state) = &mut self.rng {
                let avail: Vec<TaskId> = enabled
                    .iter()
                    .copied()
                    .filter(|t| !self.sleep.contains_key(t))
                    .collect();
                if avail.is_empty() {
                    None
                } else {
                    // xorshift* (matches `DetState::rng`).
                    let mut x = *state;
                    x ^= x >> 12;
                    x ^= x << 25;
                    x ^= x >> 27;
                    *state = x;
                    let r = x.wrapping_mul(0x2545F4914F6CDD1D);
                    Some(avail[(r as usize) % avail.len()])
                }
            } else {
                enabled
                    .iter()
                    .copied()
                    .find(|t| !self.sleep.contains_key(t))
            };
            match chosen {
                Some(t) => t,
                None => {
                    self.blocked = true;
                    return None;
                }
            }
        };
        debug_assert!(enabled.contains(&tid), "planned tid must be runnable");
        self.pending = Some((tid, enabled.to_vec()));
        Some(tid)
    }

    /// Finalize the current decision into the trace once its `access` is known, and advance the sleep
    /// set to the child state: the thread that just ran leaves the set (its old next-transition entry is
    /// stale); the siblings explored before it (`prior[depth]`) join it; then everything that
    /// **conflicts** with the transition just taken wakes (is dropped), leaving only the independent
    /// threads asleep deeper — the FG sleep-set rule `sleep(s.p) = {q ∈ sleep(s) : indep(p, q)}`.
    fn finish(&mut self, access: MemAccess) {
        if let Some((tid, enabled)) = self.pending.take() {
            self.trace.push(SchedEvent {
                tid,
                enabled,
                access,
            });
            self.sleep.remove(&tid);
            if let Some(prior) = self.prior.get(self.depth) {
                for (&q, &qacc) in prior {
                    self.sleep.entry(q).or_insert(qacc);
                }
            }
            self.sleep.retain(|_, &mut qacc| !access.conflicts(qacc));
            self.depth += 1;
        }
    }
}

/// One node of the DPOR exploration along the current depth-first path: the vCPU `chosen` here (and the
/// object `chosen_acc` it touched), the `enabled` set, the `backtrack`/`done` sets (threads still to
/// explore vs. already explored from this state), each explored thread's access (`done_acc`), and
/// `prior_acc` — the siblings explored *before* the current `chosen`, which seed the child sleep set
/// during replay. The Flanagan–Godefroid bookkeeping, plus the access maps sleep sets need.
struct DporSlot {
    chosen: TaskId,
    chosen_acc: MemAccess,
    enabled: Vec<TaskId>,
    backtrack: BTreeSet<TaskId>,
    done: BTreeSet<TaskId>,
    done_acc: BTreeMap<TaskId, MemAccess>,
    prior_acc: BTreeMap<TaskId, MemAccess>,
}

/// **Exhaustive interleaving model checker** (§18) with **dynamic partial-order reduction** (DPOR):
/// enumerate every distinct schedule of a concurrent guest *modulo independent-operation reordering*
/// and report the set of terminal outcomes — turning "sweep random seeds and hope" into a proof, for
/// programs small enough to explore fully.
///
/// It's a *stateless* checker (CHESS / `shuttle`-style): each schedule is one fresh execution replayed
/// from a planned sequence of scheduling choices, with no VM-state snapshotting. vCPUs run at
/// **memory-op granularity** (`memop` + `quantum = 1`), so the decision points are exactly the
/// shared-state / sync operations ([`is_visible`]). DPOR (Flanagan–Godefroid stateless form, **with
/// sleep sets**) then only explores *both* orders of two transitions when they actually **conflict**
/// ([`MemAccess::conflicts`]: same bytes, one a write); independent operations keep one order. After
/// each run it detects races (for each transition, the latest earlier conflicting transition by a
/// *different* vCPU) and adds the conflicting vCPU to that earlier decision's `backtrack` set; **sleep
/// sets** then prune the residual redundancy (a thread that became redundant after an independent
/// sibling ran is held asleep down that subtree until a conflict wakes it), so the search visits
/// essentially one schedule per Mazurkiewicz trace. It DFS-backtracks to the deepest decision with an
/// unexplored, non-sleeping alternative — stopping when the tree is exhausted or `max_schedules` is hit.
/// The reduction is sound: reordering independent ops cannot change the terminal state, so the set of
/// reachable outcomes is identical to the unreduced enumeration ([`explore_all_bruteforce`], the
/// differential oracle) at a fraction of the schedules.
///
/// Asserting `outcomes == [expected]` (with `complete`) proves the invariant holds under every
/// interleaving.
pub fn explore_all(
    m: &Module,
    func: FuncIdx,
    args: &[Value],
    fuel: u64,
    max_schedules: u64,
) -> Exhaustive {
    let mut outcomes: Vec<Result<Vec<Value>, Trap>> = Vec::new();
    let (schedules, complete) =
        explore_core(m, func, args, fuel, max_schedules, |_idx, result, _plan| {
            if !outcomes.contains(result) {
                outcomes.push(result.clone());
            }
            false // never stop early: enumerate the whole (reduced) interleaving tree
        });
    Exhaustive {
        outcomes,
        schedules,
        complete,
    }
}

/// A replayable witness schedule (DEBUGGING.md W7): the exact sequence of scheduling choices
/// (`TaskId` per visible-op decision point) that produced `outcome`, recovered from the DPOR
/// explorer. Feed `plan` to [`replay_schedule`] to reproduce `outcome` deterministically (or, later,
/// to a multithreaded `Inspector` to *step* that interleaving — Milestone B).
#[derive(Clone, PartialEq, Debug)]
pub struct Witness {
    /// The scheduling choice (`TaskId`) at each decision point, in order.
    pub plan: Vec<u64>,
    /// The terminal outcome this schedule produced.
    pub outcome: Result<Vec<Value>, Trap>,
    /// Which explored schedule (1-based) yielded it.
    pub schedule_index: u64,
}

/// Model-check `func` across interleavings and return the **first** schedule whose outcome
/// satisfies `pred`, as a replayable [`Witness`] (DEBUGGING.md W7) — e.g. find a deadlock
/// (`|o| matches!(o, Err(Trap::ThreadFault))`), any trap, or a specific bad result. Explores with
/// DPOR (sound reduction) up to `max_schedules`. `None` means no explored schedule matched — and if
/// the search ran to completion (didn't hit `max_schedules`), no such schedule exists. Unlike
/// [`explore_all`] (which reports the *set* of outcomes), this hands back a concrete, reproducible
/// interleaving you can replay and inspect.
pub fn find_schedule(
    m: &Module,
    func: FuncIdx,
    args: &[Value],
    fuel: u64,
    max_schedules: u64,
    pred: impl Fn(&Result<Vec<Value>, Trap>) -> bool,
) -> Option<Witness> {
    let mut found: Option<Witness> = None;
    explore_core(m, func, args, fuel, max_schedules, |idx, result, plan| {
        if pred(result) {
            found = Some(Witness {
                plan: plan.to_vec(),
                outcome: result.clone(),
                schedule_index: idx,
            });
            true // stop: we have our witness
        } else {
            false
        }
    });
    found
}

/// Re-run a witness schedule deterministically (DEBUGGING.md W7 → W1): drive the interpreter with
/// `plan` (a [`Witness::plan`]) so the same interleaving — hence the same outcome — is reproduced.
/// A failing interleaving found by [`find_schedule`] is thus replayable on demand.
pub fn replay_schedule(
    m: &Module,
    func: FuncIdx,
    args: &[Value],
    fuel: u64,
    plan: &[u64],
) -> Result<Vec<Value>, Trap> {
    if m.funcs.get(func as usize).is_none() {
        return Err(Trap::Malformed);
    }
    let funcs: Arc<[Func]> = m.funcs.clone().into();
    // The plan forces each scheduling choice (`Dpor::pick`); an empty `prior` is fine — sleep sets
    // only affect *exploration*, never a forced replay.
    let mut dpor = Dpor::new(plan.to_vec(), Vec::new());
    run_one_schedule(
        &funcs,
        &m.memory,
        &m.data,
        func,
        args,
        fuel,
        Policy::Dpor(&mut dpor),
    )
}

/// The DPOR exploration engine shared by [`explore_all`] and [`find_schedule`] (DEBUGGING.md W7).
/// Runs schedules under dynamic partial-order reduction; for each non-redundant (non-sleep-blocked)
/// schedule it calls `on_outcome(schedule_index, &outcome, &executed_plan)`, where `executed_plan`
/// is the sequence of `TaskId` choices that produced `outcome` — a replayable witness (feed it to
/// [`replay_schedule`]). Returning `true` stops the search early (so `complete` is then `false`).
/// Returns `(schedules_run, complete)`.
fn explore_core(
    m: &Module,
    func: FuncIdx,
    args: &[Value],
    fuel: u64,
    max_schedules: u64,
    mut on_outcome: impl FnMut(u64, &Result<Vec<Value>, Trap>, &[TaskId]) -> bool,
) -> (u64, bool) {
    if m.funcs.get(func as usize).is_none() {
        on_outcome(1, &Err(Trap::Malformed), &[]);
        return (1, true);
    }
    let funcs: Arc<[Func]> = m.funcs.clone().into();

    let mut stack: Vec<DporSlot> = Vec::new();
    let mut schedules = 0u64;

    loop {
        // Replay the current path (`chosen` per depth), extending with the default choice past its end.
        // `prior` carries each slot's pre-`chosen` siblings so the controller reconstructs sleep sets.
        let plan: Vec<TaskId> = stack.iter().map(|s| s.chosen).collect();
        let prior: Vec<BTreeMap<TaskId, MemAccess>> =
            stack.iter().map(|s| s.prior_acc.clone()).collect();
        let mut dpor = Dpor::new(plan, prior);
        let result = run_one_schedule(
            &funcs,
            &m.memory,
            &m.data,
            func,
            args,
            fuel,
            Policy::Dpor(&mut dpor),
        );
        schedules += 1;
        // A sleep-blocked run is a redundant prefix (its completions were reached elsewhere): keep its
        // trace for race/backtrack bookkeeping, but don't surface an outcome from its truncated tail.
        let blocked = dpor.blocked;
        let trace = dpor.trace;
        if !blocked {
            let executed: Vec<TaskId> = trace.iter().map(|e| e.tid).collect();
            if on_outcome(schedules, &result, &executed) {
                return (schedules, false); // caller asked to stop (e.g. found a witness)
            }
        }

        // Sync the path stack with the trace just produced. Slots for the replayed prefix already
        // exist (their `chosen` matches by construction); push fresh slots for newly reached depths;
        // a forced/blocked choice that shortened the run drops the now-stale deeper slots.
        if trace.len() < stack.len() {
            stack.truncate(trace.len());
        }
        for (d, ev) in trace.iter().enumerate() {
            if d < stack.len() {
                debug_assert_eq!(stack[d].chosen, ev.tid);
                stack[d].enabled.clone_from(&ev.enabled);
                stack[d].chosen_acc = ev.access;
                stack[d].done.insert(ev.tid);
                stack[d].done_acc.insert(ev.tid, ev.access);
            } else {
                stack.push(DporSlot {
                    chosen: ev.tid,
                    chosen_acc: ev.access,
                    enabled: ev.enabled.clone(),
                    backtrack: BTreeSet::from([ev.tid]),
                    done: BTreeSet::from([ev.tid]),
                    done_acc: BTreeMap::from([(ev.tid, ev.access)]),
                    prior_acc: BTreeMap::new(),
                });
            }
        }

        // Race detection (Flanagan–Godefroid): for each transition `j`, find the latest earlier
        // transition `i` by a *different* vCPU that conflicts with it, and ensure the decision at `i`
        // will also try `j`'s vCPU (or, if it wasn't co-enabled there, every enabled vCPU — the
        // conservative "may-be-co-enabled" fallback). The recursion across runs then covers earlier
        // conflicts. A race-added thread overrides the sleep set (backtrack ∖ done isn't pruned by it).
        for j in 0..trace.len() {
            for i in (0..j).rev() {
                if trace[i].tid != trace[j].tid && trace[i].access.conflicts(trace[j].access) {
                    let q = trace[j].tid;
                    if stack[i].enabled.contains(&q) {
                        stack[i].backtrack.insert(q);
                    } else {
                        let enabled = stack[i].enabled.clone();
                        stack[i].backtrack.extend(enabled);
                    }
                    break;
                }
            }
        }

        // Backtrack to the deepest decision with an unexplored alternative; force it next run, recording
        // the now-explored siblings as its child's sleep seed (`prior_acc`).
        let mut next = None;
        for d in (0..stack.len()).rev() {
            if let Some(&p) = stack[d].backtrack.difference(&stack[d].done).next() {
                next = Some((d, p));
                break;
            }
        }
        match next {
            Some((d, p)) if schedules < max_schedules => {
                stack[d].prior_acc = stack[d].done_acc.clone();
                stack[d].done.insert(p);
                stack[d].chosen = p;
                stack.truncate(d + 1);
            }
            Some(_) => return (schedules, false), // hit `max_schedules` with work left
            None => return (schedules, true),     // interleaving tree exhausted
        }
    }
}

/// The **unreduced** exhaustive enumerator — explores *every* ordering of visible ops, including
/// reorderings of independent operations. Superseded by [`explore_all`] (DPOR) for real use; kept as
/// the differential oracle that proves DPOR's reduction is sound (same `outcomes`, fewer `schedules`).
#[doc(hidden)]
pub fn explore_all_bruteforce(
    m: &Module,
    func: FuncIdx,
    args: &[Value],
    fuel: u64,
    max_schedules: u64,
) -> Exhaustive {
    if m.funcs.get(func as usize).is_none() {
        return Exhaustive {
            outcomes: vec![Err(Trap::Malformed)],
            schedules: 1,
            complete: true,
        };
    }
    let funcs: Arc<[Func]> = m.funcs.clone().into();

    let mut plan: Vec<usize> = Vec::new();
    let mut outcomes: Vec<Result<Vec<Value>, Trap>> = Vec::new();
    let mut schedules = 0u64;
    let complete;

    loop {
        let mut choices = Choices::new(plan);
        let result = run_one_schedule(
            &funcs,
            &m.memory,
            &m.data,
            func,
            args,
            fuel,
            Policy::Brute(&mut choices),
        );
        schedules += 1;
        if !outcomes.contains(&result) {
            outcomes.push(result);
        }

        // Backtrack: bump the deepest decision that has an unexplored sibling, dropping everything
        // after it (those subtrees are re-explored fresh under the new prefix).
        let mut next = None;
        for i in (0..choices.branches.len()).rev() {
            if choices.chosen[i] + 1 < choices.branches[i] {
                let mut p = choices.chosen[..i].to_vec();
                p.push(choices.chosen[i] + 1);
                next = Some(p);
                break;
            }
        }
        match next {
            Some(p) if schedules < max_schedules => plan = p,
            Some(_) => {
                complete = false;
                break;
            }
            None => {
                complete = true;
                break;
            }
        }
    }

    Exhaustive {
        outcomes,
        schedules,
        complete,
    }
}

/// Run a single schedule under the exhaustive checker: a fresh memory image and root vCPU (at
/// memory-op granularity), driven by `policy` ([`Policy::Brute`] or [`Policy::Dpor`]). Returns the root
/// task's outcome.
fn run_one_schedule(
    funcs: &Arc<[Func]>,
    memory: &Option<Memory>,
    data: &[Data],
    func: FuncIdx,
    args: &[Value],
    fuel: u64,
    policy: Policy,
) -> Result<Vec<Value>, Trap> {
    let mem = memory.map(|mc| {
        let mut mm = Mem::with_reservation(DEFAULT_RESERVED_LOG2, mc.size_log2);
        mm.init_data(data);
        mm
    });
    let det = Arc::new(DetSched::new(0, MAX_VCPUS)); // seed unused under the exhaustive policy
    let root_id = {
        let mut s = det.lock();
        let id = s.next_task;
        s.next_task += 1;
        s.live += 1;
        let dt = Arc::new(DomainTable::new(funcs, 0)); // DPOR: no Jit install, natural table
        let mut root = VCpu::new(
            Arc::clone(funcs),
            func,
            args,
            mem,
            Arc::new(Mutex::new(Host::new())),
            fuel,
            0,
            id,
            SchedRef::Det(Arc::clone(&det)),
            Quota::default(), // exhaustive model-checker path: the fixed anti-bomb ceilings
            dt,
        );
        root.memop = true;
        s.runnable.push(Box::new(root));
        id
    };
    run_with_policy(&det, policy);
    let out = det.lock().results.remove(&root_id);
    out.map_or(Err(Trap::ThreadFault), |o| o.result)
}

/// One activation record on the **explicit** guest call stack (§12). Reifying the call
/// stack — rather than recursing on the host stack — is what makes fibers possible: a
/// fiber's continuation is exactly its `Vec<Frame>`, which `suspend` pauses and
/// `cont.resume` restarts.
#[derive(Clone)]
struct Frame {
    /// The function this activation is executing — stored as an **index** (not a borrow) so a
    /// `Frame` (hence a whole vCPU continuation) is self-contained and movable between worker
    /// threads. Resolved against [`Frame::module`]'s `Arc<[Func]>` at each use.
    func: FuncIdx,
    /// Which **function space** [`Frame::func`] indexes — `0` is the vCPU's primary module (its
    /// own program), `k ≥ 1` is `units[k-1]`, a guest-compiled `Jit` unit (DESIGN.md §22). A
    /// `call_indirect` always dispatches through the **module-0** table (matching the JIT, where
    /// every function — parent or unit — is lowered against the parent `fn_table`), so a unit's
    /// indirect call lands in module 0 (new→old); a **direct** call stays in the caller's module
    /// (unit-local). Ordinary vCPUs only ever use module 0.
    module: u32,
    /// Index of the block currently executing.
    block: usize,
    /// Index of the **next** instruction to execute within that block. Saved across a
    /// nested call so the caller resumes just past the `call` when the callee returns.
    inst: usize,
    /// Block-local SSA values produced so far (entry = the call arguments), as raw [`Reg`]s.
    vals: Vec<Reg>,
}

/// A `<setjmp.h>` checkpoint: the resume point of a `setjmp` call (see [`VCpu::setjmp_points`]). Since
/// `Frame::vals` is **replaced per block** (block-param SSA), the value state at the `setjmp` point can
/// not be reconstructed from the live (later-block) frame, so it is snapshotted here. `longjmp` truncates
/// the call stack to `depth` (the intervening frames discarded — C has no cleanups), overwrites the
/// surviving `setjmp` frame with `(block, inst, vals)`, sets `vals[result_idx]` to the long-jump value,
/// and resumes. The data-stack pointer rides in `vals[0]` (the §3d SP block-param), so restoring `vals`
/// restores it.
#[derive(Clone)]
struct SetJmpPoint {
    /// Call-stack length at `setjmp` (the `setjmp` frame is at index `depth - 1`).
    depth: usize,
    /// The `setjmp` frame's block, and the instruction index just *after* the `setjmp`.
    block: usize,
    inst: usize,
    /// The `setjmp` frame's value array at the `setjmp` point (its result slot included).
    vals: Vec<Reg>,
    /// Index in `vals` of the `setjmp` result — overwritten with the long-jump value on re-entry.
    result_idx: usize,
}

/// One slot of a vCPU's **`call_indirect` dispatch table** — the explicit, module-aware
/// generalization of "mask the index into `funcs`". Each slot names which module's function it
/// holds, so the table can mix the parent's functions (Model A: populated from module 0) with
/// later-installed unit functions (Model B2: `install` appends). `module == TABLE_EMPTY` is a
/// power-of-two padding slot — a forged index landing there traps, like the JIT's
/// `PADDING_TYPE_ID`. The table length is a power of two; the mask is `len - 1`.
#[derive(Clone, Copy)]
struct TableSlot {
    module: u32,
    func: FuncIdx,
}
const TABLE_EMPTY: u32 = u32::MAX;
/// The reserved module id of the unit a `Jit.invoke` is currently running ([`VCpu::new_invoke`]).
/// It is a **per-vCPU transient** module, kept *out* of the shared [`DomainTable`]'s `units`, so the
/// invoked unit is never an installed (`call_indirect`-reachable) module and never collides with a
/// concurrent install — even one the invoked unit performs on itself (install-during-own-invocation,
/// DESIGN.md §22). Far above any reachable install count, just below `TABLE_EMPTY`.
const INVOKE_MODULE: u32 = u32::MAX - 1;

#[inline]
fn pack_slot(module: u32, func: u32) -> u64 {
    ((module as u64) << 32) | func as u64
}
#[inline]
fn unpack_slot(w: u64) -> TableSlot {
    TableSlot {
        module: (w >> 32) as u32,
        func: w as u32,
    }
}

/// The domain's **shared, live** `call_indirect` dispatch table — the reference mirror of the JIT's
/// one `fn_table`, shared by every vCPU of a domain (root, `thread.spawn` children, `Jit.invoke`
/// children) via `Arc`. Sharing it is what makes guest-driven `install` faithful across threads and
/// across a nested invocation (DESIGN.md §22): a slot write is visible to every vCPU, exactly as the
/// JIT's one shared table is.
///
/// **No lock on the dispatch hot path.** Each slot is a single `u64` word (`module<<32 | func`), so
/// dispatch is one `Acquire` atomic load (free on x86) and `install`/`uninstall` one `Release`
/// store — no `Mutex`, no torn read (the table is pre-reserved and never resized). `units` (the
/// funcs backing installed modules) is append-only; *writers* serialize under its `Mutex`, while
/// *readers* keep a lock-free local clone (a prefix of `units`) and re-sync only on a miss — so a
/// steady-state `call_indirect` to module 0 (the common case) touches neither a lock nor `units`.
struct DomainTable {
    slots: Box<[AtomicU64]>,
    units: Mutex<Vec<Arc<[Func]>>>,
}

impl DomainTable {
    /// Build a domain's table from its module-0 functions: slot `i` = `(module 0, i)`, then trapping
    /// padding to a power-of-two length. `reserve_log2` reserves a *larger* table than the module
    /// needs (matching the JIT's `table_reserve_log2`) so `install` (Model B2) fills the padding;
    /// `0` ⇒ the natural `next_pow2(funcs.len())`. Real functions stay in `[0, funcs.len())` and
    /// padding starts at `funcs.len()` on both backends, so the slot index `install` returns agrees.
    fn new(funcs: &[Func], reserve_log2: u8) -> DomainTable {
        let len = (1usize << reserve_log2)
            .max(funcs.len().next_power_of_two())
            .max(1);
        let slots = (0..len)
            .map(|i| {
                AtomicU64::new(if i < funcs.len() {
                    pack_slot(0, i as u32)
                } else {
                    pack_slot(TABLE_EMPTY, 0)
                })
            })
            .collect();
        DomainTable {
            slots,
            units: Mutex::new(Vec::new()),
        }
    }

    fn len(&self) -> usize {
        self.slots.len()
    }

    /// Dispatch-path read: one `Acquire` load. Ordered after the matching `install` `Release` store,
    /// so a reader that observes a filled slot also observes the pushed unit (the `units` `Mutex` in
    /// `install` releases before the slot store, and `units_snapshot` re-acquires it).
    #[inline]
    fn slot(&self, i: usize) -> TableSlot {
        unpack_slot(self.slots[i].load(Ordering::Acquire))
    }

    /// `Jit.install` (Model B2): append `unit` (module id = its 1-based index) and fill the first
    /// padding slot, returning the slot. `None` if every reserved slot is full. Writers serialize
    /// under the `units` lock (rare — only inside a synchronous `cap.call`, the guest suspended); the
    /// slot store is `Release` so the pushed unit is visible to any reader that observes the slot.
    fn install(&self, unit: Arc<[Func]>) -> Option<u32> {
        let mut units = self.units.lock().unwrap_or_else(|e| e.into_inner());
        let slot = self
            .slots
            .iter()
            .position(|s| (s.load(Ordering::Relaxed) >> 32) as u32 == TABLE_EMPTY)?;
        units.push(unit);
        let module = units.len() as u32; // module k ≡ units[k-1]
        self.slots[slot].store(pack_slot(module, 0), Ordering::Release);
        Some(slot as u32)
    }

    /// `Jit.uninstall` (Model B2 reclaim): clear an installed slot (`≥ n_real`, currently filled)
    /// back to trapping padding so the index is reusable and a stale `call_indirect` of it traps.
    /// `units` stays (append-only; the unit is just no longer reachable). Serialized like `install`.
    fn uninstall(&self, slot: usize, n_real: usize) -> bool {
        let _g = self.units.lock().unwrap_or_else(|e| e.into_inner());
        if slot >= n_real
            && slot < self.slots.len()
            && (self.slots[slot].load(Ordering::Relaxed) >> 32) as u32 != TABLE_EMPTY
        {
            self.slots[slot].store(pack_slot(TABLE_EMPTY, 0), Ordering::Release);
            true
        } else {
            false
        }
    }

    /// A clone of the installed-units prefix (Arc clones — cheap). A reader calls this only on a
    /// local-cache miss (a unit installed since its last sync); the `Mutex` acquire pairs with the
    /// `install` slot `Release` to make the pushed unit visible.
    fn units_snapshot(&self) -> Vec<Arc<[Func]>> {
        self.units.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }
}

/// Resolve module `m`'s functions for a running vCPU: module 0 is its primary program; `INVOKE_MODULE`
/// is the transient unit a `Jit.invoke` is running (`invoked`); `k ≥ 1` is an installed unit in the
/// shared [`DomainTable`]. `local_units` is the vCPU's lock-free clone of the shared installed units
/// (a prefix); a miss (a unit installed since the last sync) refreshes it. Returns `None` for an
/// out-of-range module (a forged/stale slot) → the caller traps.
fn resolve_module(
    funcs: &Arc<[Func]>,
    local_units: &mut Vec<Arc<[Func]>>,
    invoked: &Option<Arc<[Func]>>,
    dt: &DomainTable,
    m: u32,
) -> Option<Arc<[Func]>> {
    if m == 0 {
        return Some(Arc::clone(funcs));
    }
    if m == INVOKE_MODULE {
        return invoked.clone();
    }
    let i = (m - 1) as usize;
    if i >= local_units.len() {
        *local_units = dt.units_snapshot(); // a unit installed since our last sync
    }
    local_units.get(i).cloned()
}

/// Resolve a `call_indirect` through the shared [`DomainTable`]: mask the guest index, one `Acquire`
/// load of the slot, reject a padding slot, then type-check the target's signature structurally
/// (matching the JIT's masked load + `type_id` check, where the interned id ≡ structural equality).
/// Returns the target `(module, func)`. `local_units`/`invoked` resolve the slot's module (see
/// [`resolve_module`]).
fn dispatch_indirect(
    dt: &DomainTable,
    funcs: &Arc<[Func]>,
    local_units: &mut Vec<Arc<[Func]>>,
    invoked: &Option<Arc<[Func]>>,
    idx: i32,
    ty: &FuncType,
) -> Result<(u32, FuncIdx), Trap> {
    let mask = dt.len() - 1; // len is a power of two
    let slot = (idx as u32 as usize) & mask;
    let e = dt.slot(slot);
    if e.module == TABLE_EMPTY {
        return Err(Trap::IndirectCallType);
    }
    // Resolve the target module's funcs **by borrow** (the type-check needs no owned `Arc`, so the
    // hot path adds no refcount op over the old `&[Func]` lookup). A miss on an installed unit
    // re-syncs the local prefix first.
    let funcs_m: &[Func] = match e.module {
        0 => funcs,
        INVOKE_MODULE => invoked.as_deref().ok_or(Trap::IndirectCallType)?,
        m => {
            let i = (m - 1) as usize;
            if i >= local_units.len() {
                *local_units = dt.units_snapshot();
            }
            local_units
                .get(i)
                .map(|a| &**a)
                .ok_or(Trap::IndirectCallType)?
        }
    };
    let f = funcs_m.get(e.func as usize).ok_or(Trap::IndirectCallType)?;
    if f.params == ty.params && f.results == ty.results {
        Ok((e.module, e.func))
    } else {
        Err(Trap::IndirectCallType)
    }
}

/// Maximum number of fibers a single run may create (§12). Bounds the fiber table so a
/// fiber-bomb yields a clean [`Trap::FiberFault`] instead of unbounded host allocation —
/// the reference-oracle analogue of the quota that charges out-of-band stacks to the
/// guest, so a fiber-bomb OOMs *itself*, never the host. `1 << 24` (~16.7M): the hard ceiling equals
/// the fiber-handle index width ([`FIBER_GEN_SHIFT`]); the per-run spawn quota (`SVM_MAX_FIBERS`,
/// clamped to this) is the tunable anti-bomb policy.
const MAX_FIBERS: usize = 1 << 24;

/// Maximum number of **concurrently live** vCPUs (`thread.spawn`) across a run (§12). With the M:N
/// executor a vCPU is a cheap green thread (a parked one costs only its continuation, not an OS
/// thread), so this can be large; it's just an anti-bomb ceiling — exceeding it is a clean
/// [`Trap::ThreadFault`]. A spawned-and-joined loop creates unboundedly many vCPUs over its lifetime;
/// only simultaneous liveness is bounded.
const MAX_VCPUS: usize = 1 << 16;

/// Hard ceiling on the number of SQEs one async-ring `submit`/`submit_async` (§9/§12) will process in
/// a single call. The entry count comes straight from a guest register, so an unclamped value would
/// let a guest drive an unbounded host-side allocation (an uncatchable allocator `abort()`) and loop;
/// clamping here bounds both. Any real ring batch is far below this — a guest needing more issues
/// multiple submits. Mirrors the [`MAX_FIBERS`]/[`MAX_VCPUS`] anti-bomb ceilings.
const MAX_RING_BATCH: u64 = 1 << 16;

// §15 **spawn quota** — the single shared type lives in `svm-ir` (re-exported here and as
// `svm_jit::Quota`), so a powerbox embedder sets it once and it binds all three backends identically,
// with no facade conversion (Followup F6). The local `MAX_FIBERS`/`MAX_VCPUS` consts above mirror its
// hard ceilings (also used here for fiber/vCPU table sizing); `Quota::clamped` enforces them.
pub use svm_ir::Quota;

/// `cont.resume` status results (§12): the fiber `suspend`ed (resumable) vs. returned (done).
const FIBER_SUSPENDED: i32 = 0;
const FIBER_RETURNED: i32 = 1;
/// Extra §14 coroutine-`resume` status: the child suspended on a **page fault** (its `(status, value)`
/// is `(2, fault_addr)`) — the parent supplies the page and resumes (fault-driven yield / lazy paging).
const CORO_FAULTED: i32 = 2;
/// §3.6 slice 5a — the third `cont.resume` status (beside suspended/returned, extending the
/// family exactly as [`CORO_FAULTED`] did): the fiber hit an event park (`memory.wait`, a
/// blocking read, a live-callee call) and was set aside — **the fiber parked, not the vCPU**
/// (DESIGN.md "blocks the fiber, never the domain"). The resumer proceeds; re-resuming while
/// still blocked reports this again (a poll); after the event fires, a resume continues the
/// fiber past its park with the event's result delivered.
const FIBER_PARKED: i32 = 3;

/// `<ty>.atomic.wait` status results (§12), matching wasm: woken by a notify / value mismatch / timed
/// out.
const WAIT_WOKEN: i32 = 0;
const WAIT_NOT_EQUAL: i32 = 1;
const WAIT_TIMED_OUT: i32 = 2;

/// Upper bound on how long a `<ty>.atomic.wait` will actually block, regardless of the guest's
/// requested timeout (and what a negative — "infinite" — timeout is clamped to). A vCPU blocking
/// forever would never let the run's thread `scope` join; capping keeps the host live (a guest can
/// stall *itself* but not wedge the process). Legitimate waits return immediately on the notify, so
/// the cap only bounds the missed-notify fallback.
const MAX_WAIT: std::time::Duration = std::time::Duration::from_secs(10);

/// Maximum worker OS threads the executor will spawn for one run (the "N" of M:N). Capped at the
/// host parallelism. Workers are spawned **lazily** — a single-threaded guest never creates any.
const MAX_WORKERS: usize = 32;

/// A task identifier (a spawned vCPU). Distinct from the per-vCPU join *handle* (an index into the
/// spawner's child table); the executor keys results/waiters by `TaskId`.
type TaskId = u64;

/// A finished vCPU's outcome, parked in the scheduler until a `thread.join` claims it (or, for the
/// root, until [`drive`] reads it). Carries `mem`/`fuel` so the root's window can be snapshot and its
/// fuel read back after the worker that ran it is gone. (The powerbox is **shared** across all vCPUs of
/// the run — `Arc<Mutex<Host>>` — so it isn't carried here; `drive` reads it back by unwrapping the Arc.)
struct Outcome {
    result: Result<Vec<Value>, Trap>,
    mem: Option<Mem>,
    fuel: u64,
    /// The vCPU's **trap-time call stack** (innermost frame first), as `IrPc`s — captured from the
    /// live frames when (and only when) `result` is a trap, before the vCPU is dropped (DEBUGGING.md
    /// §5 / W3, the interpreter counterpart to the JIT's `last_trap_backtrace`). Empty on a clean
    /// finish. Resolved to source by the run wrapper via [`source_loc`]; host-side, off the hot path.
    trap_bt: Vec<IrPc>,
    /// The guest **fiber handle** running on this vCPU when it trapped (§5 W3 / §23-D57 per-fiber
    /// attribution, the interpreter counterpart to the JIT's `last_trap_fiber`): `Some(handle)` for a
    /// trap inside a fiber, `Some(-1)` for the root computation (no fiber), `None` on a clean finish.
    /// The handle uses the cross-backend `(generation << FIBER_GEN_SHIFT) | slot` encoding, so it
    /// compares equal to the JIT's for the same fiber.
    trap_fiber: Option<i64>,
}

/// The trapping vCPU's running-fiber handle for [`Outcome::trap_fiber`]: `-1` when the root is running
/// (no fiber), else the running fiber's guest handle (slot + live generation). Only meaningful at a
/// trap — call when `result.is_err()`.
fn trap_fiber_of(v: &VCpu) -> i64 {
    if v.cur == ROOT_FIBER {
        -1
    } else {
        fiber_handle(v.cur, v.registry.generation(v.cur))
    }
}

/// The `IrPc`s of `frames`, innermost frame first — the shape [`Outcome::trap_bt`] captures at a trap.
fn frames_to_pcs(frames: &[Frame]) -> Vec<IrPc> {
    frames
        .iter()
        .rev()
        .map(|f| IrPc {
            module: f.module,
            func: f.func,
            block: f.block,
            inst: f.inst,
        })
        .collect()
}

/// A futex wait/notify rendezvous coordinate (PROCESS.md S1b). The wait-queue is keyed by this, **not**
/// by the raw window address, so two aliases of one §13 `SharedRegion` (mapped at different window
/// offsets — in the same or different domains) rendezvous. The Linux distinction, exactly: a private
/// page keys on its virtual address (`FUTEX_PRIVATE`), a shared page on its backing identity + offset.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
enum FutexKey {
    /// A normal (anonymous) window page — keyed by confined absolute address. Anonymous pages are never
    /// aliased across domains, so the address is a sound identity.
    Anon(u64),
    /// A §13 `SharedRegion`-aliased page — keyed by `(backing identity, byte offset within the
    /// region)`, so every alias of the same region byte maps to the same key regardless of which
    /// window, window offset, **or domain** names it (the S1c residue: a per-window region id was
    /// canonical within one domain only — two domains granted the same backing got different
    /// keys, and a pipe ring between concurrently-running children never woke). The identity is
    /// the backing allocation's address (stable while any grant holds it — a mapped waiter's
    /// window keeps it alive; a wake against a freed-and-reused address is a futex-legal spurious
    /// wake). This is what lets a sibling/parent↔child pipe ring on shared memory wake its peer.
    Region(u64, u64),
}

/// Why a vCPU yielded its worker (returned by [`VCpu::run`]).
enum Blocked {
    /// Blocked in `thread.join` on child task `child` (the join handle's table slot is recorded in the
    /// vCPU's `pending`, set before it parks).
    Join { child: TaskId },
    /// Blocked in `atomic.wait`. `key` is the canonical rendezvous coordinate the wait-queue and
    /// `notify` match on (region-canonical for aliased pages — S1b); `addr` is the confined absolute
    /// address the driver re-reads to compare against `expected` under its lock (the futex
    /// compare-and-park atomicity — the value lives at a real address, the queue at a canonical key).
    /// Woken by a matching `notify` or after `timeout_ns` (already `MAX_WAIT`-clamped). The driver
    /// turns `timeout_ns` into a deadline on *its* clock — wall-clock for the real pool, a logical
    /// clock for the explorer.
    Wait {
        key: FutexKey,
        addr: u64,
        expected: u64,
        width: u32,
        timeout_ns: u64,
    },
    /// Blocked inside a capability call **through `handle`** (§3.6 slice 1: a blocking stream
    /// read with no data). Parked in the scheduler's handle-keyed waiter index, so a
    /// **revocation** of that handle (another fiber's `Stream.close`, D37 turned inward) can
    /// find and wake it with a probeable negative errno — the racing-fibers escape hatch
    /// (IMPORTS.md §3.6 consumer pinning / PROCESS.md §4 revocation-unparks). Nothing else
    /// wakes it today: a scheduler-run blocking read with no closer blocks forever, which is
    /// what blocking means.
    CapRead { handle: i32 },
    /// §3.6 slice 3 — parked inside a call to a **live callee** awaiting its reply, keyed by
    /// the dispatch ticket. `callee` is carried for the park-vs-reply race check: the park
    /// handler probes the callee's completion cell under the scheduler lock, so a reply that
    /// landed before the park wakes the caller immediately instead of stranding it.
    CapReply {
        ticket: u64,
        callee: Arc<Mutex<Host>>,
    },
    /// §3.6 slice 3 — a serving fiber parked in `svc.wait` on an empty queue, keyed by its
    /// domain identity. The frame was rewound before parking, so the wake re-executes the
    /// `svc.wait` (which then finds work). The park handler re-checks the queue under the
    /// scheduler lock (the park-vs-enqueue race). `deadline_ns` is the I38 **timed** form's
    /// optional timeout (the op's single optional arg; `< 0` = wait forever): a consumer whose
    /// deadline fires with no progress is re-admitted with [`Pending::SvcTimeout`] and its
    /// re-executed `svc.wait` returns `0` instead of re-parking — the multi-consumer
    /// wind-down primitive (a spare consumer can otherwise never exit: any sibling may
    /// work-steal every dispatch).
    SvcWait { key: usize, deadline_ns: i64 },
}

/// Set on a parked vCPU before it is re-enqueued, telling its driver how to finish the op on resume.
enum Pending {
    /// Finish a `thread.join`: take the child's result from `threads[slot]`.
    Join { slot: usize },
    /// Finish an `atomic.wait`, pushing this status (woken / not-equal / timed-out).
    Wait(i32),
    /// Finish a §14 co-fiber `yield`: push the value the parent's `resume` delivered (the result of
    /// the child's `Yielder` cap.call). Only ever set on a *coroutine* child the parent drives inline.
    CoResume(i64),
    /// Finish a revoked capability call ([`Blocked::CapRead`]): push this as the call's i64
    /// result — a negative errno the parked fiber probes on its own error path. The fiber is
    /// never killed and never traps; cancellation is a returned value (§3.6 revocation-unparks).
    CapResult(i64),
    /// A timed `svc.wait`'s deadline fired with nothing served ([`Blocked::SvcWait`]): the
    /// frame was rewound at the park, so nothing is pushed here — the flag makes the
    /// re-executed serve arm return `0` instead of re-parking (concurrent work that raced the
    /// timer is still admitted first; a non-zero count wins over the timeout).
    SvcTimeout,
}

/// One run of a vCPU until it finishes or yields.
enum Step {
    Done(Result<Vec<Value>, Trap>),
    Park(Blocked),
    /// Ran out its scheduling quantum mid-execution (deterministic-explorer preemption); re-enqueue
    /// and continue later. The real executor uses an unbounded quantum and never yields.
    Yield,
    /// **Debug pause** (DEBUGGING.md W2/S4): a breakpoint/step hit, before the op at this [`IrPc`].
    /// Only produced when an [`Inspector`] drives the vCPU; the [`VCpu`] continuation is intact, so
    /// the next `run` resumes exactly here. Carries the reason + location for the driver to report.
    Pause(StopReason, IrPc),
}

/// Internal `?`-friendly driver result; [`VCpu::run`] folds an `Err` into `Step::Done(Err)`.
enum Inner {
    Done(Vec<Value>),
    Park(Blocked),
    Yield,
    /// A §14 **co-fiber** child yielded a value to its instantiator-parent (`Yielder` cap.call). The
    /// child's continuation (frames/mem/host) is preserved in the `VCpu` so the parent's next `resume`
    /// continues it. Only produced while a coroutine child is driven inline by `resume`; a normal vCPU
    /// that reaches it (a `Yielder` with no resumer) is a `FiberFault`.
    CoYield(i64),
    /// A §14 **fault-driven yield**: a coroutine child (`fault_yields`) hit a recoverable page fault
    /// (an access to an unmapped page in its window) at this confined address. The faulting access has
    /// been rewound; the parent's `resume` supplies the page and re-runs it (userfaultfd-style lazy
    /// paging). Like `CoYield`, only produced for an inline-driven coroutine.
    CoFault(u64),
    /// A **debug pause** (DEBUGGING.md W2/S4): the per-op hook stopped before the op at this
    /// [`IrPc`]. Folded to [`Step::Pause`] by [`VCpu::run`]; never produced unless `debug` is set.
    Pause(StopReason, IrPc),
}

/// The **M:N executor** (§12): a bounded pool of worker OS threads runs many vCPUs (green threads)
/// from a shared run-queue. A vCPU that blocks on `thread.join`/`atomic.wait` **parks** — its owned
/// continuation ([`VCpu`]) is set aside, freeing the worker — and is re-enqueued when the awaited
/// event fires (child completion / `notify` / timeout). Thus thousands of vCPUs run on a handful of
/// threads. One mutex guards all scheduler state: coarse, but obviously race-free (the interpreter is
/// the reference oracle; the JIT is the performance path). Workers are spawned lazily, so a
/// single-threaded guest runs entirely on the calling thread with no pool at all.
struct Scheduler {
    mx: Mutex<Sched>,
    /// Workers wait here for runnable vCPUs (woken on new work, shutdown, or a timer deadline).
    /// `drive` runs as worker 0 and returns when shutdown fires, so no separate idle signal is needed.
    work: Condvar,
    /// Max concurrently-live vCPUs (anti-bomb) and max worker threads.
    cap: usize,
    max_workers: usize,
}

/// §3.6 slice 5a — one parked entity in a scheduler waiter map: a whole **vCPU** (the fiberless
/// / root-level park, as always) or a single **fiber** (a fiber-level park — the vCPU moved on;
/// only the fiber's continuation waits, in the registry's `ParkedOn` slot). A fiber wake flips
/// the slot claimable with the event's result delivered ([`FiberRegistry::wake_blocked`]); a
/// vCPU wake re-enqueues the box with a `Pending`, exactly as before.
enum Waiter {
    VCpu(Box<VCpu>),
    Fiber {
        reg: Arc<FiberRegistry>,
        slot: usize,
        /// §3.6 slice 5b — the parked fiber's **domain key** ([`Sched::svc_waiters`]). A wake
        /// also re-admits the domain's serve loop if it is parked in `svc.wait`, so a woken
        /// **handler** fiber gets resumed rather than waiting for the next unrelated enqueue
        /// (a non-handler fiber's spurious serve wake finds nothing runnable and re-parks).
        svc: usize,
    },
}

/// §3.6 slice 5b — wake a domain's `svc.wait`-parked serve loop from inside a wake path that
/// already holds the scheduler lock (the locked half of [`Scheduler::svc_wake`]). Idempotent:
/// a domain not parked in `svc.wait` is a no-op. Returns whether a vCPU was re-admitted (the
/// caller then signals the condvar; [`process_timers`]'s caller is a worker already awake).
fn svc_wake_locked(s: &mut Sched, key: usize) -> bool {
    match s.svc_waiters.remove(&key) {
        Some(vs) => {
            // Wake every parked consumer of the domain (see [`Sched::svc_waiters`]): only the
            // owner of a woken handler can resume it, and admission is race-free under the
            // powerbox lock — the others re-park on their re-executed `svc.wait`.
            let woke = !vs.is_empty();
            for v in vs {
                s.runnable.push_back(v);
            }
            woke
        }
        None => false,
    }
}

#[derive(Default)]
struct Sched {
    /// vCPUs ready to run.
    runnable: VecDeque<Box<VCpu>>,
    /// Finished tasks' outcomes, awaiting `join` (or the root, awaiting `drive`).
    results: BTreeMap<TaskId, Outcome>,
    /// A vCPU parked in `join`, keyed by the child it awaits.
    join_waiters: BTreeMap<TaskId, Box<VCpu>>,
    /// vCPUs parked in `wait`, keyed by canonical futex key (S1b); each tagged with a waiter id.
    wait_waiters: BTreeMap<FutexKey, Vec<(u64, Waiter)>>,
    /// vCPUs parked inside a capability call, **keyed by the handle they are parked through**
    /// (§3.6 slice 1 — the handle → parked-fibers index revocation-unparks needs). Woken only by
    /// [`Scheduler::cap_revoke`] with a negative errno; the wait_waiters/notify pair is the template.
    /// (`Box<VCpu>` deliberately, like every other parked-vCPU store — a `VCpu` is large and moves
    /// between this map and `runnable` as a pointer, never by value.)
    cap_waiters: BTreeMap<i32, Vec<Waiter>>,
    /// §3.6 slice 3 — callers parked awaiting a live-callee **reply**, keyed by the dispatch
    /// ticket (exactly one caller per ticket; woken by [`Scheduler::cap_reply_or_stash`] with
    /// the result).
    ticket_waiters: BTreeMap<u64, Waiter>,
    /// §3.6 slice 3 — serving vCPUs parked in `svc.wait` on an empty queue, keyed by their
    /// domain identity (the powerbox `Arc` pointer — all vCPUs of a domain share it). Woken by
    /// a caller's enqueue ([`Scheduler::svc_wake`]); resume re-executes the `svc.wait`.
    /// **Multi-consumer** (I39's top rung): a domain may park N vCPUs here (N spawned server
    /// threads draining one queue), so the value is a `Vec` and a wake re-admits **all** of
    /// them — the wake path knows only the domain key, never which vCPU owns a parked handler
    /// (per-vCPU `handler_parks`), so wake-all is the obviously-correct form: the queue pop is
    /// under the powerbox lock (each dispatch admitted exactly once), and a woken vCPU that
    /// finds nothing runnable simply re-parks.
    /// (`Box<VCpu>` deliberately, like every other parked-vCPU store — the box moves between
    /// this map and `runnable` as a pointer, never by value.)
    #[allow(clippy::vec_box)]
    svc_waiters: BTreeMap<usize, Vec<Box<VCpu>>>,
    /// Min-heap of `(deadline, waiter id, futex key)` for timing out `wait`s.
    timers: BinaryHeap<Reverse<(Instant, u64, FutexKey)>>,
    /// Min-heap of `(deadline, domain key, task)` for the I38 **timed `svc.wait`** — fired by
    /// [`process_timers`] like the futex timers (a waiter already woken by an enqueue is
    /// simply absent from `svc_waiters`; its stale timer is skipped).
    svc_timers: BinaryHeap<Reverse<(Instant, usize, TaskId)>>,
    /// OS-thread handles of spawned workers (joined by `drive` at the end).
    handles: Vec<std::thread::JoinHandle<()>>,
    /// vCPUs not yet finished (running + queued + parked). The run ends when this hits 0.
    live: usize,
    /// Worker threads in existence (incl. the calling thread, counted as 1).
    workers: usize,
    next_task: TaskId,
    next_wid: u64,
    shutdown: bool,
    /// §5 W3 / §23-D57 — the **trap-origin capture**: the `(backtrace, fiber)` of the *first* vCPU to
    /// trap on its own op, run-shared and **first-wins**. A child trap propagates to its `thread.join`er
    /// as a bare `Err(Trap)` (the parent re-traps with *its* frames at the join), so the root's own
    /// outcome would name the join site, not the origin. `drive` reads this instead, so the trap
    /// diagnostic names *where the guest actually trapped* — the interpreter counterpart to the JIT's
    /// `Domain` trap-capture handoff. `None` on a clean run.
    trap_origin: Option<(Vec<IrPc>, Option<i64>)>,
}

impl Scheduler {
    fn new(cap: usize, max_workers: usize) -> Scheduler {
        Scheduler {
            mx: Mutex::new(Sched::default()),
            work: Condvar::new(),
            cap,
            max_workers,
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Sched> {
        self.mx.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Allocate a task id + live slot and enqueue the vCPU built by `make`; spawn another worker if
    /// demand warrants. `None` if the live cap is hit (a thread-bomb).
    fn spawn(self: &Arc<Self>, make: impl FnOnce(TaskId) -> Box<VCpu>) -> Option<TaskId> {
        let mut s = self.lock();
        if s.live >= self.cap {
            return None;
        }
        let id = s.next_task;
        s.next_task += 1;
        s.live += 1;
        s.runnable.push_back(make(id));
        self.maybe_spawn_worker(&mut s);
        self.work.notify_one();
        Some(id)
    }

    /// Grow the pool toward `min(live, max_workers)` so parked/queued vCPUs have a thread to run them.
    fn maybe_spawn_worker(self: &Arc<Self>, s: &mut Sched) {
        if s.workers < self.max_workers && s.workers < s.live && !s.shutdown {
            s.workers += 1;
            let me = Arc::clone(self);
            s.handles.push(std::thread::spawn(move || worker_loop(&me)));
        }
    }

    /// Wake up to `count` vCPUs parked on `key`; return how many were woken.
    fn notify(&self, key: FutexKey, count: u32) -> u32 {
        let mut s = self.lock();
        let mut woken: Vec<Waiter> = Vec::new();
        if let Some(q) = s.wait_waiters.get_mut(&key) {
            while (woken.len() as u32) < count {
                match q.pop() {
                    Some((_, v)) => woken.push(v),
                    None => break,
                }
            }
            if q.is_empty() {
                s.wait_waiters.remove(&key);
            }
        }
        let n = woken.len() as u32;
        for w in woken {
            match w {
                Waiter::VCpu(mut v) => {
                    v.pending = Some(Pending::Wait(WAIT_WOKEN));
                    s.runnable.push_back(v);
                }
                // §3.6 5a: a fiber-level waiter — deliver the status into its set-aside
                // frames and make it claimable; its resumer re-admits it cooperatively
                // (for a handler fiber, that resumer is the domain's serve loop — 5b).
                Waiter::Fiber { reg, slot, svc } => {
                    reg.wake_blocked(slot, Reg::from_i32(WAIT_WOKEN));
                    svc_wake_locked(&mut s, svc);
                }
            }
        }
        if n > 0 {
            self.work.notify_all();
        }
        n
    }

    /// §3.6 revocation-unparks: wake **every** fiber parked in a capability call through
    /// `handle`, completing each one's call with the negative errno `status` (probeable on the
    /// fiber's own error path — never a trap, never a kill). Called at the revocation act
    /// (`Stream.close` from a sibling fiber); returns how many were woken. Granularity is
    /// per-connection by design: all fibers parked through the handle wake together.
    fn cap_revoke(&self, handle: i32, status: i64) -> u32 {
        let mut s = self.lock();
        let woken = s.cap_waiters.remove(&handle).unwrap_or_default();
        let n = woken.len() as u32;
        for w in woken {
            match w {
                Waiter::VCpu(mut v) => {
                    v.pending = Some(Pending::CapResult(status));
                    s.runnable.push_back(v);
                }
                Waiter::Fiber { reg, slot, svc } => {
                    reg.wake_blocked(slot, Reg::from_i64(status));
                    svc_wake_locked(&mut s, svc);
                }
            }
        }
        if n > 0 {
            self.work.notify_all();
        }
        n
    }

    /// §3.6 — deliver a served dispatch's result **atomically** against a racing caller park:
    /// wake the ticket's parked caller, or — under the SAME scheduler lock — stash the value in
    /// the callee's completion cell. The two-step form (a reply-wake miss, then a
    /// separate cell insert) had a TOCTOU window: the caller could park between the miss and
    /// the insert — its park-time cell probe empty, its `ticket_waiters` entry never woken (no
    /// second reply ever comes) — stranding it forever with the value sitting in the cell.
    /// Found hammering the multi-consumer suite; reachable (rarer) with a single consumer too.
    /// Lock order (scheduler, then callee powerbox) matches the caller's park handler and the
    /// fiber early-probe, so the pair can't deadlock.
    fn cap_reply_or_stash(&self, ticket: u64, result: i64, callee: &Arc<Mutex<Host>>) {
        let mut s = self.lock();
        match s.ticket_waiters.remove(&ticket) {
            Some(Waiter::VCpu(mut v)) => {
                v.pending = Some(Pending::CapResult(result));
                s.runnable.push_back(v);
                self.work.notify_all();
            }
            Some(Waiter::Fiber { reg, slot, svc }) => {
                reg.wake_blocked(slot, Reg::from_i64(result));
                if svc_wake_locked(&mut s, svc) {
                    self.work.notify_all();
                }
            }
            None => {
                callee
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .svc_results
                    .insert(ticket, result);
            }
        }
    }

    /// DURABILITY.md §13.4 step 2 — whether fiber `slot` of `reg` is parked in a **futex
    /// wait** (its waiter entry sits in `wait_waiters`). The freeze driver's park-kind probe:
    /// a futex park freezes (its `MemoryWait` thaw arm re-issues the wait), a cap park
    /// (`cap_waiters`/`ticket_waiters`) does not (its `Leaf` spill would reload the freeze
    /// placeholder as the call's result) — probe only, nothing is consumed.
    fn fiber_wait_parked(&self, reg: &Arc<FiberRegistry>, slot: usize) -> bool {
        let s = self.lock();
        s.wait_waiters.values().any(|q| {
            q.iter().any(|(_, w)| {
                matches!(w, Waiter::Fiber { reg: r, slot: sl, .. }
                    if Arc::ptr_eq(r, reg) && *sl == slot)
            })
        })
    }

    /// DURABILITY.md §13.4 step 2 — consume fiber `slot`'s futex waiter entry (the freeze
    /// takes ownership of the park; the thaw's re-issued wait re-derives it). A stale timer
    /// for the purged waiter finds it absent and is skipped, as for a notified waiter.
    fn purge_fiber_wait_park(&self, reg: &Arc<FiberRegistry>, slot: usize) {
        let mut s = self.lock();
        for q in s.wait_waiters.values_mut() {
            q.retain(|(_, w)| {
                !matches!(w, Waiter::Fiber { reg: r, slot: sl, .. }
                    if Arc::ptr_eq(r, reg) && *sl == slot)
            });
        }
        s.wait_waiters.retain(|_, q| !q.is_empty());
    }

    /// §3.6 slice 3 — a caller's enqueue landed on domain `key`'s queue: wake its vCPUs parked
    /// in `svc.wait`, if any (all of them — see [`Sched::svc_waiters`]). Resume re-executes the
    /// `svc.wait` (the frame was rewound at the park), which then finds the queue non-empty and
    /// serves — or, for a consumer that lost the admission race, re-parks.
    fn svc_wake(&self, key: usize) -> bool {
        let mut s = self.lock();
        if svc_wake_locked(&mut s, key) {
            self.work.notify_all();
            true
        } else {
            false
        }
    }
}

/// Move any expired `wait` timers' vCPUs back to the run-queue with a timed-out status. (A waiter
/// already woken by `notify` is simply absent — its stale timer is skipped.)
fn process_timers(s: &mut Sched) {
    let now = Instant::now();
    // Timed `svc.wait` deadlines (I38): re-admit the still-parked consumer with the timeout
    // pending — its rewound `svc.wait` re-executes, admits anything that raced the timer, and
    // returns its count (0 on a pure timeout) instead of re-parking.
    while let Some(&Reverse((dl, key, tid))) = s.svc_timers.peek() {
        if dl > now {
            break;
        }
        s.svc_timers.pop();
        if let Some(q) = s.svc_waiters.get_mut(&key) {
            if let Some(pos) = q.iter().position(|v| v.id == tid) {
                let mut v = q.remove(pos);
                v.pending = Some(Pending::SvcTimeout);
                s.runnable.push_back(v);
            }
            if q.is_empty() {
                s.svc_waiters.remove(&key);
            }
        }
    }
    while let Some(&Reverse((dl, wid, key))) = s.timers.peek() {
        if dl > now {
            break;
        }
        s.timers.pop();
        let mut woken = None;
        if let Some(q) = s.wait_waiters.get_mut(&key) {
            if let Some(pos) = q.iter().position(|(id, _)| *id == wid) {
                woken = Some(q.remove(pos).1);
            }
        }
        if let Some(w) = woken {
            if s.wait_waiters.get(&key).is_some_and(|q| q.is_empty()) {
                s.wait_waiters.remove(&key);
            }
            match w {
                Waiter::VCpu(mut v) => {
                    v.pending = Some(Pending::Wait(WAIT_TIMED_OUT));
                    s.runnable.push_back(v);
                }
                Waiter::Fiber { reg, slot, svc } => {
                    reg.wake_blocked(slot, Reg::from_i32(WAIT_TIMED_OUT));
                    svc_wake_locked(s, svc);
                }
            }
        }
    }
}

/// A worker: pull a runnable vCPU and dispatch it, sleeping (until work, a timer, or shutdown) when
/// idle. Returns when the run is shutting down and nothing is left to do.
fn worker_loop(sched: &Arc<Scheduler>) {
    loop {
        let next = {
            let mut s = sched.lock();
            loop {
                process_timers(&mut s);
                if let Some(v) = s.runnable.pop_front() {
                    break Some(v);
                }
                if s.shutdown {
                    break None;
                }
                let dl_futex = s.timers.peek().map(|Reverse((dl, _, _))| *dl);
                let dl_svc = s.svc_timers.peek().map(|Reverse((dl, _, _))| *dl);
                let dl_next = match (dl_futex, dl_svc) {
                    (Some(a), Some(b)) => Some(a.min(b)),
                    (a, None) => a,
                    (None, b) => b,
                };
                match dl_next {
                    Some(dl) => {
                        let now = Instant::now();
                        if dl > now {
                            let (g, _) = sched
                                .work
                                .wait_timeout(s, dl - now)
                                .unwrap_or_else(|e| e.into_inner());
                            s = g;
                        }
                    }
                    None => s = sched.work.wait(s).unwrap_or_else(|e| e.into_inner()),
                }
            }
        };
        match next {
            Some(v) => dispatch(sched, v),
            None => return,
        }
    }
}

/// Run one vCPU until it yields, then route the outcome: publish a result (waking a joiner) and
/// retire the slot, or park it on a join target / wait address.
fn dispatch(sched: &Arc<Scheduler>, mut v: Box<VCpu>) {
    // Durable multi-vCPU (DURABILITY.md §12.8 slice 3.2.1): swap THIS vCPU's per-context durable words
    // into the **shared** window before it runs — the state word ([`STATE_OFF`]) and the active
    // shadow-SP ([`SHADOW_SP_OFF`]). The freeze/thaw runs single-worker, so the one shared pair is each
    // vCPU's own context, swapped per dispatch: the shadow-SP points unwind/rewind at this vCPU's region
    // (`context = task id`), and the state word is this vCPU's own freeze phase — vital because a
    // rewinding vCPU flips the word to `NORMAL` after reloading, which must not disturb a sibling still
    // rewinding. Only at root context — a vCPU parked mid-fiber-resume keeps the fiber swap's own
    // bookkeeping (a no-op for the no-fiber slice, where `cur` is always `ROOT_FIBER`).
    if v.durable && v.cur == ROOT_FIBER {
        // §12.8 4A.5: at root, the active spill context is this vCPU's own; its SP word lives in *its*
        // region (`shadow_region_base(vcpu_ctx)`), not a shared offset.
        v.durable_sp_ctx = v.vcpu_ctx;
        let root_word = shadow_region_base(v.vcpu_ctx);
        if let Some(m) = v.mem.as_mut() {
            // §12.8 concurrent-thaw stage 1: route this vCPU's phase across the global freeze word and
            // its own per-context thaw word, so its rewind can't disturb a sibling's.
            m.durable_store_dstate(v.vcpu_ctx, v.dstate);
            m.durable_set_sp(root_word, v.root_shadow_sp);
        }
    }
    let step = v.run(u64::MAX);
    // Save this vCPU's durable words whenever it parks (it resumes later on this single worker and must
    // restore the same context). Skipped on `Done` (it won't run again; its residue, if any, is read
    // from the live words below).
    if v.durable && v.cur == ROOT_FIBER && !matches!(step, Step::Done(_)) {
        let root_word = shadow_region_base(v.vcpu_ctx);
        if let Some(m) = v.mem.as_ref() {
            // §12.8 concurrent-thaw stage 1: recombine the phase from the freeze (global) + thaw
            // (per-context) words — the rewind's re-issue flipped *its own* thaw word to `NORMAL`.
            v.dstate = m.durable_load_dstate(v.vcpu_ctx);
            v.root_shadow_sp = m.durable_get_sp(root_word);
        }
    }
    match step {
        Step::Done(result) => {
            // `froze` distinguishes a **freeze-unwind** (the run is `UNWINDING`; a spawned child
            // records `FrozenVCpu` residue and its region is kept for thaw) from a **genuine finish**
            // (NORMAL completion). Context recycling frees a finished child's shadow context back to
            // the registry, but a frozen child must keep it (it is re-spawned there on thaw).
            let froze = v.durable
                && result.is_ok()
                && v.mem.as_ref().map(|m| m.durable_state()) == Some(STATE_UNWINDING);
            // Freeze driver (DURABILITY.md §12.8 slice 3.1.4 / 3.4): a durable run left in `UNWINDING`
            // has drained THIS vCPU's native stack into its shadow region; now flatten the fibers it
            // parked into theirs, while the registry is alive, before the window is snapshotted. **Every**
            // vCPU drives its own (slice 3.4: a spawned child that owns fibers must flatten them too — the
            // root's drive runs before the children exist, so it can't see a child's fiber). `freeze_drive`
            // walks the *shared* registry's parked set and removes what it takes, so each vCPU's drive
            // catches exactly the fibers still parked when it runs (its own), with no double-flatten. A
            // drive trap (out-of-scope module) surfaces as the run's result.
            // §4 subtree freeze (DURABILITY.md): handle this vCPU's live §14 children before its
            // own residue is recorded. The **covered** shape — same-module, still-running children,
            // no unjoined `thread.spawn` siblings — is frozen for real: broadcast `UNWINDING` into
            // each live child's carve state word (the subtree STW; the child self-unwinds into its
            // carve's own durable reserve, which is inside this window's image, at its next poll) and
            // record it as [`FrozenNested`] re-attach residue, tagged with **this** vCPU's task id as
            // its `parent_task`. Depth is now covered to **arbitrary nesting** (parent→child→
            // grandchild, …): a §14 child records its *own* live children (its grandchildren of the
            // root) into the subtree's shared freeze-residue **sink** — the root host, reached via
            // [`VCpu::freeze_sink`] since the child's own powerbox is private — so the whole subtree's
            // residue coalesces where a thaw reads it, disambiguated by `parent_task` (the exact
            // shape by which `thread.spawn`'s shared host coalesces [`FrozenVCpu`]). A
            // **completed-but-unjoined** child rides via `completed_result` (its result taken from
            // the scheduler; no `UNWINDING` broadcast — nothing to unwind). Everything else stays
            // **fail-closed** (`ThreadFault`, like the join-deadlock): a suspended coroutine
            // (host-side native continuation), a separate-module child (its module identity can't ride
            // the artifact yet), a completed child that **trapped** (its trap can't ride yet), and
            // mixing with unjoined `thread.spawn` children (the two thaw seedings would contend for
            // the join table).
            let nested_refused = froze && {
                let live: Vec<NestedChildInfo> = v
                    .nested_children
                    .iter()
                    .filter(|c| v.threads.get(c.slot).is_some_and(Option::is_some))
                    .copied()
                    .collect();
                if v.coroutines.iter().any(Option::is_some) {
                    true
                } else if live.is_empty() {
                    false
                } else if v.threads.iter().enumerate().any(|(slot, t)| {
                    // A live `thread.spawn` sibling (a `threads` slot not backed by a
                    // `nested_children` entry): its thaw seeding and the §14 seeding would contend
                    // for the join table — fail closed.
                    t.is_some() && !v.nested_children.iter().any(|c| c.slot == slot)
                }) || v.mem.is_none()
                {
                    true // uncovered shape / malformed durable freeze
                } else {
                    // NB: a nested child (`v.nested_child`) is **no longer** refused here — a §14
                    // child may record its own live children (grandchildren), tagged with this
                    // vCPU's task id and pushed to the subtree's shared sink (depth-2+, §4).
                    let mut refuse = false;
                    for c in &live {
                        let cid = v.threads[c.slot].expect("filtered to Some");
                        let completed_result = if v.sched.has_result(cid) {
                            // Take the finished child's result; a clean `Ok(i64)` rides the artifact,
                            // a completed-with-trap child is not yet representable — fail closed.
                            match v.sched.take_result(cid).map(|o| o.result) {
                                Some(Ok(vals)) => Some(match vals.first() {
                                    Some(Value::I64(x)) => *x,
                                    Some(Value::I32(x)) => *x as i64,
                                    _ => 0,
                                }),
                                _ => {
                                    refuse = true;
                                    break;
                                }
                            }
                        } else {
                            // Still running: broadcast `UNWINDING` into its carve; it self-unwinds.
                            if let Some(m) = v.mem.as_mut() {
                                m.write_bytes(
                                    c.carve_off + STATE_OFF,
                                    &STATE_UNWINDING.to_le_bytes(),
                                );
                            }
                            None
                        };
                        // §4 depth-2: push to the subtree's **effective sink** (the root host for a
                        // nested child, our own host for the root), tagged with our own task id as
                        // the child's `parent_task`, so the whole nesting subtree's residue coalesces
                        // where a thaw reads it — the root records its child with `parent_task = 0`,
                        // a child records its grandchild with `parent_task = <child's task>`.
                        let sink = v.freeze_sink.clone().unwrap_or_else(|| Arc::clone(&v.host));
                        sink.lock()
                            .unwrap_or_else(|e| e.into_inner())
                            .frozen_nested
                            .push(FrozenNested {
                                parent_task: v.id as usize,
                                slot: c.slot,
                                carve_off: c.carve_off,
                                size_log2: c.size_log2,
                                entry: c.entry,
                                module_digest: c.module_digest,
                                completed_result,
                            });
                    }
                    refuse
                }
            };
            let result = if nested_refused {
                Err(Trap::ThreadFault)
            } else if froze {
                // Record this vCPU's own flattened extent (the live shadow-SP) *before* `freeze_drive`
                // repoints the active-SP word to flatten idle fibers and restores it to this extent.
                let self_sp = v
                    .mem
                    .as_ref()
                    .map(|m| m.durable_get_sp(shadow_region_base(v.vcpu_ctx)))
                    .unwrap_or_else(|| shadow_frame_base(v.vcpu_ctx));
                if let Some((func, args)) = v.spawn_residue.clone() {
                    // A **spawned** vCPU (slice 3.2.1) records *itself* as residue: its continuation now
                    // lives in its own region (extent = `self_sp`); a thaw re-spawns it there.
                    v.host
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .frozen_vcpus
                        .push(FrozenVCpu {
                            task: v.id as usize,
                            parent_task: v.parent_task as usize,
                            func: func as i32,
                            args,
                            shadow_sp: self_sp,
                            completed_result: None, // interp runs durable single-worker
                        });
                } else if !v.nested_child {
                    // The root: record its extent (the shared active-SP word will be overwritten by a
                    // later child, so the root's residue can't ride the window).
                    v.host
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .frozen_root_sp = Some(self_sp);
                }
                // (A §14 nested child records nothing: its extent is its carve's own shadow-SP
                // word, inside the parent's window image — carve-self-describing.)
                let mut r = v.freeze_drive().and(result);
                // Hand this vCPU's flattened fibers back to the embedder via the shared host. **Extend**
                // (not assign): every vCPU contributes its own, and the snapshot's canonical sort-by-slot
                // makes the accumulation order irrelevant to the artifact.
                if !v.frozen.is_empty() {
                    let frozen = std::mem::take(&mut v.frozen);
                    if v.nested_child {
                        // A nested child's fiber residue is child-local (its slots/extents name
                        // its own registry + carve) — it cannot ride the parent's tables. Fail
                        // closed until per-child fiber residue lands (DURABILITY.md §4).
                        drop(frozen);
                        r = Err(Trap::ThreadFault);
                    } else {
                        v.host
                            .lock()
                            .unwrap_or_else(|e| e.into_inner())
                            .frozen_fibers
                            .extend(frozen);
                    }
                }
                r
            } else {
                result
            };
            // Context recycling: a spawned vCPU that genuinely finished frees its shadow context for a
            // later spawn to reuse (a freeze-unwound child keeps it — it's re-spawned there on thaw).
            if v.vcpu_ctx > 0 && !froze {
                v.registry.free_vcpu_context(v.vcpu_ctx);
            }
            let id = v.id;
            // §5 W3: snapshot the trap-time call stack before the vCPU is dropped (only on a trap).
            let trap_bt = if result.is_err() {
                frames_to_pcs(&v.frames)
            } else {
                Vec::new()
            };
            let trap_fiber = result.is_err().then(|| trap_fiber_of(&v));
            let outcome = Outcome {
                result,
                mem: v.mem.take(),
                fuel: v.fuel,
                trap_bt,
                trap_fiber,
            };
            drop(v);
            let mut s = sched.lock();
            // First-wins trap-origin capture (§5 W3 / §23-D57): the first vCPU to trap records its own
            // backtrace + fiber, so a later join-propagated re-trap on the root can't overwrite the
            // true origin. A clean finish leaves it untouched.
            if outcome.result.is_err() {
                s.trap_origin
                    .get_or_insert_with(|| (outcome.trap_bt.clone(), outcome.trap_fiber));
            }
            if let Some(parent) = s.join_waiters.remove(&id) {
                s.runnable.push_back(parent);
                sched.work.notify_one();
            }
            s.results.insert(id, outcome);
            s.live -= 1;
            if s.live == 0 {
                s.shutdown = true;
                sched.work.notify_all();
            }
        }
        Step::Park(Blocked::Join { child }) => {
            let mut s = sched.lock();
            if s.results.contains_key(&child) {
                // Already finished between the join check and here — resume immediately.
                s.runnable.push_back(v);
                sched.work.notify_one();
            } else {
                s.join_waiters.insert(child, v);
            }
        }
        Step::Park(Blocked::Wait {
            key,
            addr,
            expected,
            width,
            timeout_ns,
        }) => {
            let deadline = Instant::now() + Duration::from_nanos(timeout_ns);
            let mut s = sched.lock();
            // Re-read the value **under the lock** so the compare-and-park is atomic vs. `notify`.
            // The value lives at the absolute `addr`; the queue/timer key is the canonical `key` (S1b).
            if v.atomic_value(addr, width) != expected {
                v.pending = Some(Pending::Wait(WAIT_NOT_EQUAL));
                s.runnable.push_back(v);
                sched.work.notify_one();
            } else {
                let wid = s.next_wid;
                s.next_wid += 1;
                s.timers.push(Reverse((deadline, wid, key)));
                s.wait_waiters
                    .entry(key)
                    .or_default()
                    .push((wid, Waiter::VCpu(v)));
                sched.work.notify_all(); // let idle workers recompute their timer deadline
            }
        }
        Step::Park(Blocked::CapRead { handle }) => {
            // §3.6 slice 1: park inside a capability call, keyed by the handle. The
            // park-vs-revoke race mirrors the futex compare-and-park: enqueue under the
            // scheduler lock, then re-check the handle's liveness — a `cap_revoke` that ran
            // between the empty-read decision and this insertion found no waiter, so if the
            // handle is now dead we wake ourselves with the same errno instead of parking
            // forever. (Lock order sched → host; the revoke path holds neither.)
            let mut s = sched.lock();
            let live = v
                .host
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .handle_live(handle);
            if live {
                s.cap_waiters
                    .entry(handle)
                    .or_default()
                    .push(Waiter::VCpu(v));
            } else {
                v.pending = Some(Pending::CapResult(CAP_REVOKED));
                s.runnable.push_back(v);
                sched.work.notify_one();
            }
        }
        Step::Park(Blocked::CapReply { ticket, callee }) => {
            // §3.6 slice 3: park awaiting a live-callee reply. Park-vs-reply race check under
            // the scheduler lock: a reply that landed before this park sits in the callee's
            // completion cell — take it and wake ourselves instead of stranding the caller.
            let mut s = sched.lock();
            let early = callee
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .svc_results
                .remove(&ticket);
            match early {
                Some(r) => {
                    v.pending = Some(Pending::CapResult(r));
                    s.runnable.push_back(v);
                    sched.work.notify_one();
                }
                None => {
                    s.ticket_waiters.insert(ticket, Waiter::VCpu(v));
                }
            }
        }
        Step::Park(Blocked::SvcWait { key, deadline_ns }) => {
            // §3.6 slice 3: park the serving vCPU on its empty queue. Park-vs-enqueue race
            // check under the scheduler lock: an enqueue that landed since the empty check
            // found no waiter, so re-run ourselves (the frame was rewound — the re-executed
            // `svc.wait` finds the work). Multi-consumer: push alongside any sibling
            // consumers already parked on this domain (never displace them).
            let mut s = sched.lock();
            let empty = v
                .host
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .svc_queue
                .is_empty();
            if empty {
                if deadline_ns >= 0 {
                    s.svc_timers.push(Reverse((
                        Instant::now() + std::time::Duration::from_nanos(deadline_ns as u64),
                        key,
                        v.id,
                    )));
                }
                s.svc_waiters.entry(key).or_default().push(v);
            } else {
                s.runnable.push_back(v);
                sched.work.notify_one();
            }
        }
        Step::Yield => {
            // Unreachable for the real pool (quantum is `u64::MAX`), but re-enqueue for safety.
            let mut s = sched.lock();
            s.runnable.push_back(v);
            sched.work.notify_one();
        }
        // Only an `Inspector`-driven vCPU pauses, and those are never on the executor (DEBUGGING.md S4).
        Step::Pause(..) => unreachable!("debug pause on a pooled vCPU"),
    }
}

/// A vCPU's executor handle: spawn/notify route to either the real OS-thread pool or the
/// single-threaded deterministic explorer. `Clone` is a cheap `Arc` bump (the child inherits it).
#[derive(Clone)]
enum SchedRef {
    Real(Arc<Scheduler>),
    Det(Arc<DetSched>),
}

impl SchedRef {
    fn spawn(&self, make: impl FnOnce(TaskId) -> Box<VCpu>) -> Option<TaskId> {
        match self {
            SchedRef::Real(s) => s.spawn(make),
            SchedRef::Det(d) => d.spawn(make),
        }
    }
    fn notify(&self, key: FutexKey, count: u32) -> u32 {
        match self {
            SchedRef::Real(s) => s.notify(key, count),
            SchedRef::Det(d) => d.notify(key, count),
        }
    }
    /// §3.6 revocation-unparks: wake every fiber parked in a capability call through `handle`
    /// with the negative errno `status` ([`Scheduler::cap_revoke`]). The explorer has no
    /// cap-call parks (a `Blocked::CapRead` there fails closed at the park), so it is a no-op.
    fn cap_revoke(&self, handle: i32, status: i64) -> u32 {
        match self {
            SchedRef::Real(s) => s.cap_revoke(handle, status),
            SchedRef::Det(_) => 0,
        }
    }
    /// §13.4 step 2 park-kind probe ([`Scheduler::fiber_wait_parked`]); the explorer hosts no
    /// durable freezes, so its arm answers `false` (→ the freeze fails closed).
    fn fiber_wait_parked(&self, reg: &Arc<FiberRegistry>, slot: usize) -> bool {
        match self {
            SchedRef::Real(s) => s.fiber_wait_parked(reg, slot),
            SchedRef::Det(_) => false,
        }
    }
    /// §13.4 step 2 waiter consume ([`Scheduler::purge_fiber_wait_park`]); explorer: no-op.
    fn purge_fiber_wait_park(&self, reg: &Arc<FiberRegistry>, slot: usize) {
        if let SchedRef::Real(s) = self {
            s.purge_fiber_wait_park(reg, slot);
        }
    }
    /// Atomic reply-or-stash ([`Scheduler::cap_reply_or_stash`]); the explorer has no caller
    /// parks, so its arm stashes straight into the cell.
    fn cap_reply_or_stash(&self, ticket: u64, result: i64, callee: &Arc<Mutex<Host>>) {
        match self {
            SchedRef::Real(s) => s.cap_reply_or_stash(ticket, result, callee),
            SchedRef::Det(_) => {
                callee
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .svc_results
                    .insert(ticket, result);
            }
        }
    }
    /// §3.6 slice 3 serve-wake ([`Scheduler::svc_wake`]); the explorer has no svc parks.
    fn svc_wake(&self, key: usize) -> bool {
        match self {
            SchedRef::Real(s) => s.svc_wake(key),
            SchedRef::Det(_) => false,
        }
    }
    /// Take a finished child's outcome (for a resuming `thread.join`).
    fn take_result(&self, id: TaskId) -> Option<Outcome> {
        match self {
            SchedRef::Real(s) => s.lock().results.remove(&id),
            SchedRef::Det(d) => d.lock().results.remove(&id),
        }
    }
    /// PROCESS.md S3 — non-destructive lifecycle probe for `poll`: `None` if `id` is still running,
    /// `Some(true)` if it returned cleanly, `Some(false)` if it trapped. Leaves the result in place so
    /// a later `join`/`take_result` still gets it.
    fn poll_status(&self, id: TaskId) -> Option<bool> {
        match self {
            SchedRef::Real(s) => s.lock().results.get(&id).map(|o| o.result.is_ok()),
            SchedRef::Det(d) => d.lock().results.get(&id).map(|o| o.result.is_ok()),
        }
    }
    /// Whether `id` has already completed (result posted, unjoined) — a non-destructive probe the
    /// §14 subtree freeze uses to tell a **live** child (broadcast + residue) from a
    /// **completed-but-unjoined** one (refused fail-closed until completed-result residue lands).
    fn has_result(&self, id: TaskId) -> bool {
        match self {
            SchedRef::Real(s) => s.lock().results.contains_key(&id),
            SchedRef::Det(d) => d.lock().results.contains_key(&id),
        }
    }
}

/// Upper bound on the deterministic explorer's per-step quantum (instructions before a forced
/// yield). A small bound interleaves vCPUs finely; the actual quantum each turn is seeded in
/// `1..=MAX_QUANTUM`, so varying the seed varies the interleaving.
const MAX_QUANTUM: u64 = 8;

/// The **deterministic explorer** (§18): a single-threaded, seed-driven executor for *verifying*
/// concurrent guest code. It runs the same vCPUs as the real pool but on one OS thread, choosing
/// which runnable vCPU to step (and for how long) from a seeded PRNG, and timing out `atomic.wait`s
/// on a **logical** clock. So a run is fully reproducible from its seed, and sweeping seeds explores
/// distinct interleavings — turning "run many times and hope" into systematic coverage, with any
/// failure replayable. No data races exist (one thread), so each seed realizes one valid sequential
/// interleaving of the shared-memory ops.
struct DetSched {
    st: Mutex<DetState>,
}

struct DetWaiter {
    key: FutexKey,
    deadline: u64, // logical ns
    vcpu: Box<VCpu>,
}

/// A vCPU the explorer parked because it was **spinning**: it ran a visible op that changed no memory
/// and returned it to the same local configuration (a busy-wait retry). It stays parked — not a
/// scheduling choice, so the spin doesn't multiply the interleaving tree — until another vCPU writes to
/// the `[base, base+width)` range it was reading, which may have changed the value it spins on.
struct SpinWaiter {
    vcpu: Box<VCpu>,
    base: u64,
    width: u32,
}

struct DetState {
    // `Box` (matching the join/wait waiter maps) keeps moving a large vCPU between the runnable set
    // and the waiter collections a pointer copy.
    #[allow(clippy::vec_box)]
    runnable: Vec<Box<VCpu>>,
    results: BTreeMap<TaskId, Outcome>,
    join_waiters: BTreeMap<TaskId, Box<VCpu>>,
    wait_waiters: Vec<DetWaiter>,
    /// vCPUs parked by spin-loop detection (memop explorer only), woken by a write to their read range.
    spin_waiters: Vec<SpinWaiter>,
    live: usize,
    next_task: TaskId,
    clock: u64, // logical nanoseconds, advanced only to fire a timeout
    rng: u64,
    cap: usize,
}

impl DetState {
    /// Find a **live** vCPU by id wherever the scheduler parks it — runnable, join/wait/spin-parked
    /// (DEBUGGING.md Milestone B: the debugger inspects any thread while stopped). Finished vCPUs are
    /// gone (only their [`Outcome`] remains in `results`), so they aren't found here.
    fn find_vcpu(&self, id: TaskId) -> Option<&VCpu> {
        if let Some(v) = self.runnable.iter().find(|v| v.id == id) {
            return Some(v);
        }
        if let Some(v) = self.join_waiters.get(&id) {
            return Some(v);
        }
        if let Some(w) = self.wait_waiters.iter().find(|w| w.vcpu.id == id) {
            return Some(&w.vcpu);
        }
        self.spin_waiters
            .iter()
            .find(|w| w.vcpu.id == id)
            .map(|w| &*w.vcpu)
    }

    /// The ids of every live (not-yet-finished) vCPU the scheduler holds, sorted.
    fn live_ids(&self) -> Vec<TaskId> {
        let mut ids: Vec<TaskId> = self
            .runnable
            .iter()
            .map(|v| v.id)
            .chain(self.join_waiters.keys().copied())
            .chain(self.wait_waiters.iter().map(|w| w.vcpu.id))
            .chain(self.spin_waiters.iter().map(|w| w.vcpu.id))
            .collect();
        ids.sort_unstable();
        ids
    }

    /// Move every spin-parked vCPU whose read range `[base, base+width)` overlaps the just-written
    /// `[w_base, w_base+w_width)` back to the runnable set — a write there may have changed the value it
    /// spins on, so it must re-check (it re-parks if still stuck). The interpreter is sequentially
    /// consistent, so a write is the *only* way a spinner's read can change.
    fn wake_spins(&mut self, w_base: u64, w_width: u32) {
        let mut i = 0;
        while i < self.spin_waiters.len() {
            let s = &self.spin_waiters[i];
            let overlap = w_base < s.base.saturating_add(s.width as u64)
                && s.base < w_base.saturating_add(w_width as u64);
            if overlap {
                let w = self.spin_waiters.swap_remove(i);
                self.runnable.push(w.vcpu);
            } else {
                i += 1;
            }
        }
    }
}

impl DetState {
    /// xorshift64* — the seeded source of all scheduling choices (so the whole run is a function of
    /// the seed).
    fn rng(&mut self) -> u64 {
        let mut x = self.rng;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.rng = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
}

impl DetSched {
    fn new(seed: u64, cap: usize) -> DetSched {
        DetSched {
            st: Mutex::new(DetState {
                runnable: Vec::new(),
                results: BTreeMap::new(),
                join_waiters: BTreeMap::new(),
                wait_waiters: Vec::new(),
                spin_waiters: Vec::new(),
                live: 0,
                next_task: 0,
                clock: 0,
                rng: seed | 1,
                cap,
            }),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, DetState> {
        self.st.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn spawn(&self, make: impl FnOnce(TaskId) -> Box<VCpu>) -> Option<TaskId> {
        let mut s = self.lock();
        if s.live >= s.cap {
            return None;
        }
        let id = s.next_task;
        s.next_task += 1;
        s.live += 1;
        s.runnable.push(make(id));
        Some(id)
    }

    /// Wake up to `count` vCPUs waiting on `key`, in deterministic (insertion) order.
    fn notify(&self, key: FutexKey, count: u32) -> u32 {
        let mut s = self.lock();
        let mut woken = 0u32;
        let mut i = 0;
        while woken < count && i < s.wait_waiters.len() {
            if s.wait_waiters[i].key == key {
                let mut w = s.wait_waiters.remove(i);
                w.vcpu.pending = Some(Pending::Wait(WAIT_WOKEN));
                s.runnable.push(w.vcpu);
                woken += 1;
            } else {
                i += 1;
            }
        }
        woken
    }
}

/// How a deterministic-explorer run resolves its two scheduling choices — which runnable vCPU to step
/// next, and for how long. `Seeded` is the random seed sweep ([`run_scheduled`]); `Exhaustive` is the
/// stateless model checker ([`explore_all`]), which follows a planned choice sequence and records the
/// branch factor at each point so the caller can DFS the whole interleaving tree.
enum Policy<'a> {
    Seeded,
    /// Unreduced enumeration ([`explore_all_bruteforce`]): pick by runnable *index* per `Choices`.
    Brute(&'a mut Choices),
    /// DPOR ([`explore_all`]): pick by `TaskId` per the plan, recording each step's [`MemAccess`].
    Dpor(&'a mut Dpor),
}

/// Run a deterministic-explorer instance to completion (every vCPU finished, or a genuine deadlock —
/// nothing runnable and nothing waiting on a timeout). Picks a runnable vCPU and a quantum per the
/// [`Policy`]; when none is runnable, advances the logical clock to the earliest `wait` deadline and
/// times that waiter out.
fn run_det(det: &Arc<DetSched>) {
    run_with_policy(det, Policy::Seeded);
}

fn run_with_policy(det: &Arc<DetSched>, mut policy: Policy) {
    match SchedDriver::default().run(det, &mut policy) {
        DriverStop::Done => {}
        // The explorer sets neither `debug` nor a turn limit, so only `Done` can occur (S4).
        DriverStop::Paused { .. } | DriverStop::TurnLimit => {
            unreachable!("debug pause / turn limit in the deterministic explorer")
        }
    }
}

/// Outcome of one [`SchedDriver::run`] pump.
enum DriverStop {
    /// A debugged vCPU paused *before* an op (breakpoint/step/watchpoint/cap.call). The vCPU is held
    /// inside the driver (`driver.held`) so the next pump resumes its turn without re-deciding the
    /// schedule; the held vCPU's id is the stopped thread ([`Inspector::stopped_task`]).
    Paused { reason: StopReason, pc: IrPc },
    /// The run reached a scheduler quiescence: every vCPU finished, or a join-deadlock, or (under a
    /// plan) every runnable vCPU is asleep. The caller reads outcomes from `det`.
    Done,
    /// Ran the requested number of scheduler turns (`turn_limit`) and stopped at that turn boundary —
    /// a global snapshot for scheduled-mode time-travel `seek` (DEBUGGING.md W1). No vCPU is held;
    /// every thread is back in the scheduler, inspectable via `threads`/`select_task`.
    TurnLimit,
}

/// A vCPU whose turn a debug stop interrupted (the op has not run). Resumed verbatim on the next
/// pump — same quantum, same spin-detection baseline — so the schedule decision for the turn stands.
struct HeldTurn {
    v: Box<VCpu>,
    quantum: u64,
    pre_fp: u64,
    writes_before: u64,
    spin_capable: bool,
}

/// The cooperative single-thread multi-vCPU scheduler loop, made **re-entrant** so a debugger can
/// pause it at a breakpoint and resume (DEBUGGING.md Milestone B). [`run_with_policy`] is the
/// non-pausing wrapper used by the model-checker/explorer (whose vCPUs carry no `debug`, so
/// [`DriverStop::Paused`] never occurs); the [`Inspector`] drives the same loop and stops on it.
#[derive(Default)]
struct SchedDriver {
    held: Option<HeldTurn>,
    /// Completed scheduler turns (one per visible-op decision) — the global logical-time coordinate
    /// scheduled-mode `seek` targets (DEBUGGING.md W1).
    turns: u64,
    /// When `Some(t)`, stop with [`DriverStop::TurnLimit`] once `turns` reaches `t` (a seek).
    turn_limit: Option<u64>,
}

impl SchedDriver {
    /// Pump turns until a debug stop or scheduler quiescence. On `Paused`, the interrupted vCPU is
    /// retained in `self.held` and the next `run` resumes exactly there.
    fn run(&mut self, det: &Arc<DetSched>, policy: &mut Policy) -> DriverStop {
        loop {
            // Scheduled-seek stop: at the requested turn boundary (no vCPU held — a global snapshot).
            if self.turn_limit == Some(self.turns) {
                return DriverStop::TurnLimit;
            }
            // Resume an interrupted turn verbatim, else pick the next vCPU + quantum under the lock
            // (release it before running — run_inner may re-enter via `spawn`/`notify`).
            let (mut v, quantum, pre_fp, writes_before, spin_capable) = match self.held.take() {
                Some(h) => (h.v, h.quantum, h.pre_fp, h.writes_before, h.spin_capable),
                None => {
                    let (v, quantum) = {
                        let mut s = det.lock();
                        if s.runnable.is_empty() {
                            if s.live == 0 {
                                return DriverStop::Done; // all done
                            }
                            // No one runnable: fire the earliest timeout (or deadlock if none).
                            let Some(idx) = (0..s.wait_waiters.len())
                                .min_by_key(|&i| s.wait_waiters[i].deadline)
                            else {
                                return DriverStop::Done; // live > 0 but quiescent: a join-deadlock
                            };
                            let mut w = s.wait_waiters.remove(idx);
                            s.clock = s.clock.max(w.deadline);
                            w.vcpu.pending = Some(Pending::Wait(WAIT_TIMED_OUT));
                            s.runnable.push(w.vcpu);
                            continue;
                        }
                        let n = s.runnable.len();
                        // One visible op per turn (`memop` vCPUs) so every shared access is a decision.
                        let (pick, quantum) = match &mut *policy {
                            Policy::Seeded => ((s.rng() as usize) % n, 1 + s.rng() % MAX_QUANTUM),
                            Policy::Brute(c) => (c.pick(n), 1),
                            Policy::Dpor(d) => {
                                // Address by `TaskId` (runnable order is reshuffled by `swap_remove`),
                                // so the plan replays identically. `enabled` sorted ⇒ stable default
                                // (smallest id). `None` ⇒ every runnable vCPU is asleep: stop here.
                                let mut enabled: Vec<TaskId> =
                                    s.runnable.iter().map(|v| v.id).collect();
                                enabled.sort_unstable();
                                match d.pick(&enabled) {
                                    Some(tid) => {
                                        let idx = s
                                            .runnable
                                            .iter()
                                            .position(|v| v.id == tid)
                                            .expect("planned tid is runnable");
                                        (idx, 1)
                                    }
                                    None => return DriverStop::Done,
                                }
                            }
                        };
                        let v = s.runnable.swap_remove(pick);
                        (v, quantum)
                    };
                    // Spin-loop detection (memop explorer only): snapshot the vCPU's local
                    // configuration and write count so a post-turn busy-wait retry (same config, no
                    // memory changed) is distinguishable from real progress.
                    let spin_capable = v.memop;
                    let pre_fp = if spin_capable {
                        v.local_fingerprint()
                    } else {
                        0
                    };
                    let writes_before = v.mem.as_ref().map_or(0, |m| m.writes);
                    (v, quantum, pre_fp, writes_before, spin_capable)
                }
            };

            let step = v.run(quantum);

            // A debug stop interrupts the turn before the op runs: hold the vCPU (its continuation is
            // intact) and hand control back. The DPOR decision isn't finalized — `d.finish` waits for
            // the turn to actually complete on a later pump.
            if let Step::Pause(reason, pc) = step {
                self.held = Some(HeldTurn {
                    v,
                    quantum,
                    pre_fp,
                    writes_before,
                    spin_capable,
                });
                return DriverStop::Paused { reason, pc };
            }
            self.turns += 1; // a visible-op decision completed — advance global logical time

            let acc = v.acc.take();
            // DPOR: finalize this decision's trace entry now that the step's accessed object is known.
            if let Policy::Dpor(d) = &mut *policy {
                d.finish(acc.unwrap_or(MemAccess::None));
            }
            // A turn that actually changed a byte may unblock spinners parked on that address — wake
            // them to re-check (they re-park if still stuck). Memory change is the only thing that
            // can, under sequential consistency, alter a parked spinner's read.
            let mem_changed = v.mem.as_ref().map_or(0, |m| m.writes) != writes_before;
            if mem_changed {
                if let Some(MemAccess::Range { base, width, .. }) = acc {
                    det.lock().wake_spins(base, width);
                }
            }
            match step {
                Step::Done(result) => {
                    let id = v.id;
                    // §5 W3: snapshot the trap-time call stack before the vCPU is dropped (trap only).
                    let trap_bt = if result.is_err() {
                        frames_to_pcs(&v.frames)
                    } else {
                        Vec::new()
                    };
                    let trap_fiber = result.is_err().then(|| trap_fiber_of(&v));
                    let outcome = Outcome {
                        result,
                        mem: v.mem.take(),
                        fuel: v.fuel,
                        trap_bt,
                        trap_fiber,
                    };
                    drop(v);
                    let mut s = det.lock();
                    if let Some(parent) = s.join_waiters.remove(&id) {
                        s.runnable.push(parent);
                    }
                    s.results.insert(id, outcome);
                    s.live -= 1;
                }
                Step::Park(Blocked::Join { child }) => {
                    let mut s = det.lock();
                    if s.results.contains_key(&child) {
                        s.runnable.push(v); // already done (pending already set)
                    } else {
                        s.join_waiters.insert(child, v);
                    }
                }
                Step::Park(Blocked::Wait {
                    key,
                    addr,
                    expected,
                    width,
                    timeout_ns,
                }) => {
                    let mut s = det.lock();
                    if v.atomic_value(addr, width) != expected {
                        v.pending = Some(Pending::Wait(WAIT_NOT_EQUAL));
                        s.runnable.push(v);
                    } else {
                        let deadline = s.clock.saturating_add(timeout_ns);
                        s.wait_waiters.push(DetWaiter {
                            key,
                            deadline,
                            vcpu: v,
                        });
                    }
                }
                // Blocking stream reads, live-callee calls, and svc.wait are not part of the
                // explored model (the same restriction as the other non-resumable drivers).
                // Fail closed rather than wedge.
                Step::Park(
                    Blocked::CapRead { .. } | Blocked::CapReply { .. } | Blocked::SvcWait { .. },
                ) => {
                    let id = v.id;
                    drop(v);
                    let mut s = det.lock();
                    if let Some(parent) = s.join_waiters.remove(&id) {
                        s.runnable.push(parent);
                    }
                    s.results.insert(
                        id,
                        Outcome {
                            result: Err(Trap::CapFault),
                            mem: None,
                            fuel: 0,
                            trap_bt: Vec::new(),
                            trap_fiber: None,
                        },
                    );
                    s.live -= 1;
                }
                Step::Yield => {
                    // Spin-park: the turn ran one visible op, changed no memory, and returned the vCPU
                    // to the same local configuration — a busy-wait whose only way forward is another
                    // vCPU writing what it just read. Park it off the runnable set until such a write
                    // wakes it. Anything else re-enqueues normally.
                    if spin_capable && !mem_changed && v.local_fingerprint() == pre_fp {
                        if let Some(MemAccess::Range { base, width, .. }) = acc {
                            let mut s = det.lock();
                            s.spin_waiters.push(SpinWaiter {
                                vcpu: v,
                                base,
                                width,
                            });
                            continue;
                        }
                    }
                    det.lock().runnable.push(v);
                }
                Step::Pause(..) => unreachable!("handled above"),
            }
        }
    }
}

// ---- Durable runtime ABI (DURABILITY.md §12.7/§12.8) ----
//
// These describe where a **durable** (freeze/thaw-instrumented) module's per-context shadow
// state lives in the window. They are the runtime half of a contract whose tooling half is
// `svm-durable`: the transform emits IR that reads/writes the *active* shadow-SP word at
// [`SHADOW_SP_OFF`]; the runtime (here) keeps that word pointing at the **currently-running
// context's** shadow region, swapping it on every fiber switch (D-fiber-cont option A — the
// switch knowledge lives in the runtime's resume chain, not in emitted IR). `svm-interp` is
// TCB and must not depend on the tooling-tier `svm-durable`, so these constants are duplicated
// and cross-checked against `svm_durable`'s in that crate's tests.
//
// Per-context layout: context `i` owns the shadow region `[SHADOW_BASE + i*SHADOW_STRIDE, +
// SHADOW_STRIDE)`, all within the reserved low slice `[0, DURABLE_RESERVE)`. The root
// computation is context 0 (so a single-context run is byte-identical to the pre-fiber layout,
// whose lone shadow stack started at `SHADOW_BASE`); a `cont.new`-created fiber in registry
// slot `s` is context `s + 1`.

/// Window byte offset of the `i32` durable **state word** (`NORMAL | UNWINDING | REWINDING`).
/// The freeze driver reads it to tell a freeze (UNWINDING) run from an ordinary one. Must equal
/// `svm_durable::STATE_OFF`.
pub const STATE_OFF: u64 = 0;
/// State-word values (must equal `svm_durable::STATE_*`). Only `UNWINDING` is read by the runtime
/// today (the freeze-driver trigger); the others are maintained entirely by the instrumented IR.
pub const STATE_NORMAL: i32 = 0;
pub const STATE_UNWINDING: i32 = 1;
pub const STATE_REWINDING: i32 = 2;
/// Freeze **armed**: the deterministic mid-run freeze trigger. The runtime counts down
/// [`ARM_COUNTDOWN_OFF`] at each safepoint and promotes the word to `UNWINDING` at 0; transparent to
/// the instrumented IR (which tests only `UNWINDING`/`REWINDING`). Must equal `svm_durable::STATE_ARMED`.
pub const STATE_ARMED: i32 = 3;

/// §12.8 4A.7 (parked-vCPU / `Blocking.work` latency). Reads the global durable **freeze** word at
/// [`STATE_OFF`] from the live window image: `true` iff an async stop-the-world freeze has already
/// landed ([`STATE_UNWINDING`]). A durable vCPU about to enter a host `Blocking` call consults this and
/// fails **closed** rather than starting an un-checkpointable, latency-unbounded offload (R6); cancelling
/// an *already in-flight* call is deferred (R2). `false` for a non-durable / malformed window (no word),
/// so the caller also gates on [`Host::is_durable`].
fn freeze_has_landed(mem: Option<&dyn GuestMem>) -> bool {
    mem.and_then(|m| m.read_bytes(STATE_OFF, 4))
        .and_then(|b| <[u8; 4]>::try_from(b.as_slice()).ok())
        .map(|b| i32::from_le_bytes(b) == STATE_UNWINDING)
        .unwrap_or(false)
}
/// Window byte offset of the `i64` **arm countdown** (safepoints left before an `ARMED` run promotes
/// to `UNWINDING`). Decremented by the runtime at each safepoint; inert unless `ARMED`. Must equal
/// `svm_durable::ARM_COUNTDOWN_OFF`.
pub const ARM_COUNTDOWN_OFF: u64 = 16;
/// Window byte offset of the `i64` **back-edge arm countdown** (loop back-edges left before an
/// `ARMED` run promotes to `UNWINDING`, so a loop-header poll begins the freeze). Decremented at each
/// branch terminator; inert unless `ARMED` and the slot is positive. Must equal
/// `svm_durable::ARM_BACKEDGE_OFF`.
pub const ARM_BACKEDGE_OFF: u64 = 24;
/// Window byte offset of the `i64` *active* shadow-stack pointer (the running context's, a
/// window byte offset itself). The instrumented IR reads/writes this; the runtime re-points it
/// on each fiber switch. Must equal `svm_durable::SHADOW_SP_OFF`.
pub const SHADOW_SP_OFF: u64 = 8;
/// Window byte offset where **context 0's** (the root's) shadow stack begins. Must equal
/// `svm_durable::SHADOW_BASE`.
pub const SHADOW_BASE: u64 = 64;
/// Per-context shadow-stack stride: context `i` occupies `[SHADOW_BASE + i*SHADOW_STRIDE, +
/// SHADOW_STRIDE)`. 4 KiB per context fits ~15 contexts in the 64 KiB reserve — a provisional
/// slice-1 value; precise per-fiber sizing + quota accounting is the open §12.8 sub-question.
///
/// NOTE (slice-1 limitation): the transform's shadow-overflow guard still trips at the global
/// `DURABLE_RESERVE` ceiling, not at a per-region bound, so a fiber recursed deeper than
/// `SHADOW_STRIDE` would grow into the next context's region before tripping. Shallow fibers
/// (every test today) stay confined; making the overflow bound per-region travels with the
/// sizing decision.
pub const SHADOW_STRIDE: u64 = 1 << 12;
/// Ceiling of the reserved durable region `[0, DURABLE_RESERVE)`. Must equal
/// `svm_durable::DURABLE_RESERVE`.
pub const DURABLE_RESERVE: u64 = 1 << 16;

/// The shadow-region base (window offset) of context `ctx_idx` (root = 0, fiber slot `s` =
/// `s + 1`). The per-context partition that keeps two fibers' frozen frames from colliding.
fn shadow_region_base(ctx_idx: usize) -> u64 {
    SHADOW_BASE + ctx_idx as u64 * SHADOW_STRIDE
}

/// Bytes reserved at each region's base for its **per-context shadow-SP word** (§12.8 4A.5): the SP
/// word lives at `shadow_region_base(ctx)`; frames grow upward from [`shadow_frame_base`]. So a vCPU
/// addresses *its own* SP word (via `durable.shadow_base`) with no shared location.
const SHADOW_SP_WORD_LEN: u64 = 8;
/// §12.8 concurrent-thaw stage 1: byte offset of a context's **thaw** state word (`REWINDING`/`NORMAL`)
/// within its region — just past the [`SHADOW_SP_WORD_LEN`]-byte SP word, addressed via
/// `durable.shadow_base` (like the SP word). The **freeze** word (`UNWINDING`) stays at the global
/// [`STATE_OFF`]. Must equal `svm_durable::STATE_IN_REGION_OFF`.
const STATE_IN_REGION_OFF: u64 = SHADOW_SP_WORD_LEN;
/// §12.8 concurrent-thaw stage 1: bytes reserved at a region's base before its frames — the SP word
/// plus the 4-byte thaw word, padded to 8 to keep frames 8-aligned. Must equal
/// `svm_durable::REGION_HEADER_LEN`.
const REGION_HEADER_LEN: u64 = 16;

/// The empty shadow-SP / frame base of context `ctx_idx`: just past its in-region SP + thaw words. The
/// empty (no-frames) extent of a context's shadow stack.
fn shadow_frame_base(ctx_idx: usize) -> u64 {
    shadow_region_base(ctx_idx) + REGION_HEADER_LEN
}

/// Byte offset of context `ctx_idx`'s per-context **thaw** state word (§12.8 concurrent-thaw stage 1).
fn thaw_state_off(ctx_idx: usize) -> u64 {
    shadow_region_base(ctx_idx) + STATE_IN_REGION_OFF
}

/// Whether context `ctx_idx`'s shadow region fits within the reserve — the capacity bound
/// `cont.new` checks before handing out a new fiber's region.
fn shadow_region_fits(ctx_idx: usize) -> bool {
    shadow_region_base(ctx_idx) + SHADOW_STRIDE <= DURABLE_RESERVE
}

/// The highest usable shadow-context index: the reserve holds `DURABLE_RESERVE / SHADOW_STRIDE`
/// contexts and index 0 is the root, so `1..=MAX_SHADOW_CTX` are the non-root regions.
const MAX_SHADOW_CTX: usize = (DURABLE_RESERVE / SHADOW_STRIDE) as usize - 1;

/// Bits a fiber **guest handle** reserves for the registry slot; the rest carry a **generation**
/// (DURABILITY.md §12.8 recycling step 1). [`MAX_FIBERS`] is `1 << 24`, so a slot always fits in the
/// low 24 bits and the generation occupies bits 24.. of the **`i64`** handle. A handle is
/// `(generation << FIBER_GEN_SHIFT) | slot` — and since a fresh slot's generation is 0, a non-recycled
/// run's handle is exactly its slot (byte-identical to before, and to the JIT, which likewise hands out
/// `slot`). The generation lets a later **recycled** slot reject a stale handle to its former occupant
/// (the ABA guard the JIT's `Ownership` word already carries internally). Widened 16→24 (from a 65 536
/// concurrent-fiber ceiling): the arena stack backend removed the `vm.max_map_count` VMA wall, so the
/// handle index became the binding limit — 24 bits allows ~16.7M concurrent fibers.
const FIBER_GEN_SHIFT: u32 = 24;

/// The generation bits a fiber guest handle carries (the field above the slot): an `i64` handle leaves
/// **40 bits** for the generation, so a stale handle is rejected modulo `2^40` — an ABA window so vast
/// (≈ a trillion recycles of one slot) that wraparound is unreachable in practice. Matches `svm_jit`'s
/// `FIBER_HANDLE_GEN_MASK`.
const FIBER_GEN_MASK: u64 = (1 << 40) - 1;

/// Encode a fiber guest handle from its registry `slot` and `generation` (low 40 bits).
fn fiber_handle(slot: usize, generation: u64) -> i64 {
    (((generation & FIBER_GEN_MASK) << FIBER_GEN_SHIFT) | slot as u64) as i64
}

/// The generation field a guest fiber handle carries (the high bits above the slot).
fn fiber_handle_generation(handle: i64) -> u64 {
    (handle as u64) >> FIBER_GEN_SHIFT
}

#[cfg(test)]
mod fiber_handle_layout_tests {
    //! The fiber-handle index was widened 16→24 bits (`MAX_FIBERS` 1<<16 → 1<<24) once the arena stack
    //! backend removed the `vm.max_map_count` VMA wall. These pin that the wider index round-trips and
    //! that the slot decode (a `next_power_of_two(len)-1` mask, as `claim`/`resolve` use) stays clear of
    //! the generation bits at any slot up to the new ceiling — i.e. > 65 535 fibers are addressable.
    use super::{fiber_handle, fiber_handle_generation, FIBER_GEN_SHIFT, MAX_FIBERS};

    #[test]
    fn index_width_is_24_bits() {
        assert_eq!(FIBER_GEN_SHIFT, 24);
        assert_eq!(MAX_FIBERS, 1 << 24);
    }

    #[test]
    fn handle_round_trips_beyond_the_old_16_bit_ceiling() {
        // A slot the old 16-bit index could not represent (> 65 535), with a non-zero generation.
        let slot = 1_000_000usize; // < MAX_FIBERS (1<<24 = 16 777 216)
        let generation = 0x3_ABCD_1234u64;
        let handle = fiber_handle(slot, generation);
        // Generation decodes back exactly.
        assert_eq!(fiber_handle_generation(handle), generation);
        // Slot decodes back via the same dynamic mask `claim`/`resolve` use (padded to the table len);
        // the generation bits sit strictly above it, so the slot is recovered cleanly.
        let mask = (slot + 1).next_power_of_two() - 1;
        assert_eq!((handle as u64 as usize) & mask, slot);
    }

    #[test]
    fn top_slot_and_generation_do_not_overlap() {
        // The largest addressable slot and a full-width generation must not collide in the i64 handle.
        let slot = (1usize << 24) - 1;
        let generation = (1u64 << 40) - 1;
        let handle = fiber_handle(slot, generation);
        assert_eq!(fiber_handle_generation(handle), generation);
        let mask = (1u64 << FIBER_GEN_SHIFT) - 1;
        assert_eq!((handle as u64) & mask, slot as u64);
    }
}

/// Re-point the active shadow-SP word from the outgoing context's region to the incoming one's,
/// on a fiber switch (D-fiber-cont option A). A no-op unless the run is `durable` and has a
/// window. The saved-SP of each *non-running* context lives host-side (the root's in
/// [`VCpu::root_shadow_sp`], a fiber's in the registry's `shadow` table); the running context's
/// live SP is the in-window word the instrumented IR maintains. In `NORMAL` execution the
/// shadow stacks are empty (frames are pushed only under `UNWINDING`), so a saved SP equals its
/// region base — but saving/restoring the real word keeps this correct for the freeze/thaw
/// choreography (slices 3.1.3–4), where a drained fiber carries a non-empty shadow stack.
#[allow(clippy::too_many_arguments)] // an internal swap helper threading the per-context durable state
fn shadow_switch(
    mem: &mut Option<Mem>,
    registry: &FiberRegistry,
    root_shadow_sp: &mut u64,
    root_ctx: usize,
    sp_ctx: &mut usize,
    durable: bool,
    out_ctx: usize,
    in_ctx: usize,
) {
    if !durable {
        return;
    }
    // The incoming context becomes the active spill context for `durable.shadow_base`.
    *sp_ctx = if in_ctx == ROOT_FIBER {
        root_ctx
    } else {
        shadow_context_index(in_ctx)
    };
    let Some(m) = mem.as_mut() else { return };
    // §12.8 4A.5: each context's SP word lives in its **own** region (`shadow_region_base`); the
    // off-table root uses this vCPU's `root_ctx`, a fiber slot `s` its context `s + 1`. The save/load
    // mirror the host-side caches — with per-context words the load is a redundant equal-write (the
    // incoming region already holds its SP), retained for choreography parity with freeze/thaw.
    let region_of = |ctx: usize| {
        shadow_region_base(if ctx == ROOT_FIBER {
            root_ctx
        } else {
            shadow_context_index(ctx)
        })
    };
    // Save the outgoing context's live SP to its host-side slot.
    let sp = m.durable_get_sp(region_of(out_ctx));
    if out_ctx == ROOT_FIBER {
        *root_shadow_sp = sp;
    } else {
        registry.set_saved_sp(out_ctx, sp);
    }
    // Load the incoming context's SP into its (own) region word.
    let in_sp = if in_ctx == ROOT_FIBER {
        *root_shadow_sp
    } else {
        registry.saved_sp(in_ctx)
    };
    m.durable_set_sp(region_of(in_ctx), in_sp);
    // §12.8 concurrent-thaw stage 1: carry the active **thaw** phase (`REWINDING`/`NORMAL`) from the
    // outgoing context to the incoming one. Within a vCPU the rewind is sequential (a `cont.resume`
    // resumer waits on its resumee), so the globally-deepest frame's flip to `NORMAL` must propagate
    // back up through the switches — exactly as the former single global state word did (a resumer
    // doesn't flip its own word; this carry does, on the return switch). The **freeze** word is global,
    // so a non-thaw switch carries `NORMAL` (a no-op). Cross-*vCPU* concurrency uses distinct words and
    // never routes through `shadow_switch`, so each vCPU's thaw stays independent.
    let ctx_idx_of = |ctx: usize| {
        if ctx == ROOT_FIBER {
            root_ctx
        } else {
            shadow_context_index(ctx)
        }
    };
    let phase = m.durable_thaw_state(ctx_idx_of(out_ctx));
    m.durable_set_thaw_state(ctx_idx_of(in_ctx), phase);
}

/// Sentinel "fiber slot" for a vCPU's **root computation**, which lives *off-table*: the shared
/// [`FiberRegistry`] holds only `cont.new`-created fibers, so the first handle of a run is `0` on
/// both backends (the JIT has always run the root off-table — this is the handle-namespace
/// unification of D57 step 3b-i, closing the documented interp↔JIT divergence). The root's parked
/// frames (while it is resuming a fiber) live in `VCpu::root_parked`.
const ROOT_FIBER: usize = usize::MAX;

/// A fiber **flattened for freeze** (DURABILITY.md §12.8 slice 3.1.4–5), as the snapshot carries it
/// (the eventual §12.4 Section-2 per-fiber record). Its continuation is bytes in its in-window
/// shadow region `[shadow_region_base(slot+1), shadow_sp)`; this is the small host-side residue:
/// where it sits and how to re-enter it on thaw. Re-entry recreates it as a `Pending` fiber so a
/// thaw `cont.resume` runs its entry under `REWINDING`, rebuilding then re-parking it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrozenFiber {
    /// Registry slot = the guest fiber handle (so re-seeding preserves handle values).
    pub slot: usize,
    /// The fiber's entry funcref (== func index; `ref.func`/the identity table), re-entered on thaw.
    pub func: i32,
    /// The fiber's data-stack base (the entry's first param). Inert for rewind (the prologue
    /// dispatches to the resume arm, ignoring params) but recorded for fidelity / forward use.
    pub sp: i64,
    /// Window offset of the flattened shadow-SP — the extent of its frozen continuation, restored
    /// into the registry's `shadow` table so the swap re-points to it when the fiber is resumed.
    pub shadow_sp: u64,
    /// The slot's **generation** at freeze (recycling step 2): re-seeded on thaw so a guest handle to a
    /// *recycled* (generation > 0) fiber still resolves (`(generation << 24) | slot`). 0 for a
    /// non-recycled fiber — then the handle is exactly its slot. 48-bit field (the `i64` handle's
    /// generation bits); serialized as `uleb(u64)` (snapshot format v3 — see `FORMAT_VERSION`).
    pub generation: u64,
}

/// A **spawned vCPU** (a `thread.spawn` child) flattened for freeze (DURABILITY.md §12.8 slice 3.2.1).
/// Like [`FrozenFiber`] but for a whole green thread rather than a fiber: under a multi-vCPU freeze the
/// child unwinds its own native stack into *its* per-context shadow region (`context = task id`), and
/// this is the host-side residue needed to reconstruct it on thaw. The **root** vCPU needs no residue —
/// its entry/args are supplied by the thaw caller (it is re-entered directly, like the single-vCPU
/// case). On thaw the child is re-attached by the runtime: the root's rewind re-executes `thread.spawn`
/// (not a transform checkpoint), and under `REWINDING` that op re-spawns the next frozen child from this
/// residue — in deterministic spawn order — instead of creating a fresh one (a reload-not-reissue done
/// in the runtime, so `svm-durable` is unchanged).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrozenVCpu {
    /// The child's task id at freeze — globally unique + monotonic in spawn order across the whole vCPU
    /// tree. Preserved on thaw so the §12.6 canonical re-freeze records the same task.
    pub task: usize,
    /// The task id of the vCPU that **spawned** this child (slice 3.4: nested spawns). The root's direct
    /// children carry the root's id (`0`); a grandchild carries its parent child's task. Thaw groups the
    /// residue by this, re-attaches parents before children, and rebuilds each parent's join table so a
    /// grandchild's reloaded handle resolves in its *parent's* table, not the root's.
    pub parent_task: usize,
    /// The child's entry function (the `thread.spawn` target), re-entered on thaw.
    pub func: i32,
    /// The child's spawn args (`[sp, arg]`, the fiber-style thread entry), replayed on re-spawn.
    pub args: Vec<i64>,
    /// Window offset of the child's flattened shadow-SP — the extent of its frozen continuation in its
    /// region; restored as the child's shadow-SP so its thaw re-entry rewinds from the right point.
    pub shadow_sp: u64,
    /// §12.8 4A.5 follow-up A: `Some(result)` for a **completed-but-unjoined** concurrent child (JIT
    /// only — the interp runs durable single-worker, so it always records `None`). The thaw delivers the
    /// result into the spawner's join table without re-running the child. `None` for a normal frozen
    /// child (re-spawned + rewound).
    pub completed_result: Option<i64>,
}

/// A vCPU's record of one live §14 child it instantiated: the child's join-table slot, its carve
/// geometry (window-relative), and its entry — everything the subtree freeze needs to broadcast
/// into the child's carve and record it as [`FrozenNested`] residue (DURABILITY.md §4).
#[derive(Clone, Copy, Debug)]
struct NestedChildInfo {
    /// The [`VCpu::threads`] slot the child's join handle names.
    slot: usize,
    /// The carve's **window-relative** base (holder base + requested offset) — the child's window
    /// is `[carve_off, carve_off + (1 << size_log2))` of the parent's.
    carve_off: u64,
    size_log2: u8,
    /// The child's entry function index — into the parent's own table for a same-module child, or
    /// into the child's granted module for a separate-module child (resolved by `module_digest`).
    entry: u32,
    /// The child's module content digest, or `None` for a **same-module** child (`instantiate`,
    /// op 0) that runs the parent's own funcs. `Some` for a **separate-module** child (op 5): its
    /// module is host-supplied at restore, and this digest resolves the re-granted module on thaw.
    module_digest: Option<[u8; 32]>,
}

/// A **§14 instantiated child** flattened for freeze (DURABILITY.md §4 "STW quiesces the subtree
/// as a unit" — first slice: depth-1, same-module, interp). The child's entire state — its window,
/// its durable reserve (shadow regions, state/SP words), its unwound continuation — lives in its
/// carve, a sub-range of the parent's window, so it is **already in the artifact's window image**;
/// this is the small host-side residue a thaw needs to re-attach it: where the carve is, how to
/// re-enter it, and which join slot the parent's reloaded handle names. (Carried through the
/// snapshot codec's Section 2 from `FORMAT_VERSION` v8.)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrozenNested {
    /// The task id of the vCPU that **instantiated** this child (DURABILITY.md §4, depth-2 nesting).
    /// A root's direct child carries the root's id (`0`); a grandchild carries its parent-child's
    /// task. Mirrors [`FrozenVCpu::parent_task`] for `thread.spawn`: the whole §14 nesting subtree
    /// coalesces its residue in the *root* host (the §14 child's own capability host is private, so
    /// a shared **freeze-residue sink** is threaded down the subtree — see [`VCpu::freeze_sink`]),
    /// and thaw groups by this to rebuild the per-parent join table, re-attaching parents before
    /// children so a grandchild's reloaded handle resolves in its *parent-child's* table, not the
    /// root's. Depth-1 residue carries `0` and is byte-identical over the codec (not yet on the
    /// wire — a depth-2 artifact is refused at freeze; see `svm-snapshot`).
    pub parent_task: usize,
    /// The parent's join-table slot for this child (the guest-held handle value).
    pub slot: usize,
    /// The carve's window-relative base.
    pub carve_off: u64,
    /// The carve's (= the child window's) size, `log2`.
    pub size_log2: u8,
    /// The child's entry function index — into the parent's own table (`module_digest == None`) or
    /// the child's granted module (`module_digest == Some`).
    pub entry: u32,
    /// `None` for a **same-module** child (runs the parent's funcs); `Some(digest)` for a
    /// **separate-module** child, whose module a thaw resolves against the restore host's re-granted
    /// modules (host-supplied at restore, D-scope — the module bytes never ride the artifact). A
    /// missing / mismatched re-grant makes the thaw fail closed (per-child R5 identity gate).
    pub module_digest: Option<[u8; 32]>,
    /// `Some(result)` for a **completed-but-unjoined** child — one that finished before the freeze
    /// point, so its `thread.join` result must survive in the artifact (its continuation is gone; the
    /// scheduler result cell isn't captured). The thaw delivers this straight into the parent's join
    /// table **without re-running** the child (reload-not-reissue — no double side effects); its carve
    /// gets no `UNWINDING` broadcast (nothing to unwind). `None` for a still-running child (re-attached
    /// + rewound on thaw). Mirrors [`FrozenVCpu::completed_result`] for `thread.spawn` children.
    pub completed_result: Option<i64>,
}

/// A §12 fiber as the run-shared registry holds it: a first-class suspendable computation whose
/// continuation is exactly its reified call stack. `cont.new` makes one (`Pending`);
/// `cont.resume` claims and switches into it; `suspend` parks it back, claimable again
/// (`Parked`). The states mirror the loom-verified single-owner `Ownership` protocol
/// (`svm-jit/src/fiber_registry.rs`, D57 step 3a): `Pending` ≈ `OWNED` (fresh), `Parked` ≈
/// `RUNNABLE` (the only claimable-by-anyone state besides a fresh `Pending`), `Running` ≈
/// `RUNNING` (never claimable), `Done` ≈ `FREE` (unrecycled).
enum RegFiber {
    /// Created by `cont.new`, not yet started: holds the `i32` funcref to launch on the
    /// first resume (resolved then through the function table as `(i64 sp, i64 arg) ->
    /// i64`) and the `i64` data-stack base `sp` to run it on. Claimable by any vCPU.
    Pending { func: i32, sp: i64 },
    /// **Voluntarily suspended** at a `suspend`, holding its reified call stack — in the pool,
    /// claimable by any vCPU (this is what makes a fiber *migratable*: the stack is pure data,
    /// so a foreign vCPU's claim is a safe hand-off).
    Parked(Vec<Frame>),
    /// Claimed by a vCPU — **never** claimable (a second `cont.resume` loses and faults).
    /// `None`: its frames are in flight as the claimant's live `frames`. `Some`: it is itself
    /// parked mid-`cont.resume` as an ancestor in its claimant's resume chain (the frames are
    /// stored here, but only that claimant pops back into them — a foreign claim would alias a
    /// running computation).
    Running(Option<Vec<Frame>>),
    /// §3.6 slice 5a — **event-parked** (a fiber-level park): blocked inside a capability or
    /// futex park while running as a fiber. Unlike `Parked`, NOT claimable until its event
    /// fires — a `cont.resume` of it reports `FIBER_PARKED` to the resumer without switching
    /// (a poll). The wake pushes the event's result onto the set-aside top frame and flips
    /// `woken`; a claim then delivers the frames **verbatim** (the result is already in place,
    /// so the resumer's `arg` is deliberately not pushed).
    ParkedOn { frames: Vec<Frame>, woken: bool },
    /// Returned: resuming it again traps. Slots are **not recycled** (matching both backends'
    /// historical tables, so handles stay dense and deterministic); recycling + the generation
    /// tag land with the JIT shared registry (3b-ii) so both backends adopt one policy together.
    Done,
    /// **Flattened for freeze** (DURABILITY.md §12.8 slice 3.1.4): the freeze driver drove this
    /// (formerly `Parked`) fiber under `UNWINDING`, so its continuation now lives in its in-window
    /// shadow region (extent recorded in the registry's `shadow` table) instead of as host frames.
    /// A `claim` of it currently loses (thaw re-entry is slice 3.1.5); never claimable meanwhile.
    Frozen,
}

/// What a successful [`FiberRegistry::claim`] hands the winning vCPU.
enum Claimed {
    /// A `Pending` fiber: launch `func(sp, arg)` on its data stack.
    Start { func: i32, sp: i64 },
    /// A `Parked` fiber: its reified call stack, ready to continue past its `suspend`.
    Live(Vec<Frame>),
    /// A **woken** event-parked fiber ([`RegFiber::ParkedOn`]): frames verbatim — the wake
    /// already delivered the park's result onto the top frame; do NOT push the resume arg.
    LiveWoken(Vec<Frame>),
    /// A **still-blocked** event-parked fiber: not a fault and not a claim — the resumer gets
    /// `(FIBER_PARKED, 0)` without a switch (the cooperative poll).
    StillParked,
}

/// The **run-shared fiber registry** (D57 step 3b-i, DESIGN.md §23): one
/// slot table shared by every vCPU of a domain (the root + its `thread.spawn` children),
/// replacing the old per-vCPU fiber tables. This is exactly the VM-side surface migratable
/// fibers need — (1) a **shared handle namespace**, so any vCPU can name any fiber, and (2) the
/// **single-owner arbiter**: a `cont.resume` *claims* the fiber, exactly one claimant wins, and
/// a loser gets a clean [`Trap::FiberFault`]. The interpreter's fiber is `Vec<Frame>` — pure
/// data — so cross-vCPU migration is a safe hand-off; this table's mutex is the claim arbiter
/// (the safe-Rust stand-in DESIGN.md §23 sanctions for the oracle; the JIT's lock-free
/// `Ownership`-word table is slice 3b-ii/3c). The lock is a leaf (nothing else is locked while
/// it is held) and is touched only by fiber ops, never the execution hot path.
struct FiberRegistry {
    mx: Mutex<RegState>,
}

/// The registry's locked state: the fiber slots, plus a parallel **durable shadow** table
/// (`shadow[s]` = the saved shadow-SP window offset of the fiber in slot `s`, for the
/// freeze/thaw codec — D-fiber-cont option A). The two vecs grow together in [`create`], so a
/// slot's index is the same in both. `shadow` is meaningful only for durable runs; a
/// non-durable run never reads it.
struct RegState {
    fibers: Vec<RegFiber>,
    shadow: Vec<u64>,
    /// Per-slot **generation** (recycling step 1): bumped when a slot is freed for reuse, and carried
    /// in the guest handle's high bits ([`FIBER_GEN_SHIFT`]) so a stale handle to a slot's former
    /// occupant is rejected on `claim`. Grows with `fibers`/`shadow` (same index).
    gens: Vec<u64>,
    /// Freed slots reclaimable for a new fiber (recycling step 3), a **min-heap** so `create` reuses the
    /// *lowest* free slot — keeping contexts dense and low (within `MAX_SHADOW_CTX`, and clear of the
    /// top-down vCPU pool) and bounding the table to the *peak concurrent* fiber count rather than the
    /// lifetime total. A finished slot's generation is already bumped, so reuse is ABA-safe.
    free: BinaryHeap<Reverse<usize>>,
    /// **Occupied** durable vCPU shadow contexts (slice 3.2.2 + context recycling): a bitmask over
    /// contexts `1..=MAX_SHADOW_CTX` (bit `c` set ⇒ context `c` is live). The root spawns children that
    /// grow **down** from `MAX_SHADOW_CTX` while fibers grow **up** from context 1; a child's context is
    /// *freed* (its bit cleared) when it genuinely finishes, so the bound is now *peak concurrent* vCPUs
    /// rather than the lifetime total. `MAX_SHADOW_CTX` is 15, so a `u16` holds every context bit.
    vcpu_mask: u16,
}

impl FiberRegistry {
    fn new() -> FiberRegistry {
        FiberRegistry {
            mx: Mutex::new(RegState {
                fibers: Vec::new(),
                shadow: Vec::new(),
                gens: Vec::new(),
                free: BinaryHeap::new(),
                vcpu_mask: 0,
            }),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, RegState> {
        self.mx.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The saved shadow-SP of the fiber in `slot` (its region base if it has never been frozen).
    /// Out-of-range slots return the root base — they can only arise from a corrupt chain, which
    /// the surrounding fiber logic already treats as `Malformed`.
    fn saved_sp(&self, slot: usize) -> u64 {
        self.lock().shadow.get(slot).copied().unwrap_or(SHADOW_BASE)
    }

    /// The `slot`'s current generation (recycling step 2) — recorded in its [`FrozenFiber`] residue at
    /// freeze so a thaw re-seeds it at the same generation. 0 for an out-of-range slot.
    fn generation(&self, slot: usize) -> u64 {
        self.lock().gens.get(slot).copied().unwrap_or(0)
    }

    /// Record the fiber in `slot`'s shadow-SP (called when it stops being the running context).
    fn set_saved_sp(&self, slot: usize, sp: u64) {
        let mut t = self.lock();
        if let Some(s) = t.shadow.get_mut(slot) {
            *s = sp;
        }
    }

    /// Reserve the next durable vCPU shadow-context (slice 3.2.2): the `thread.spawn` path calls this
    /// to claim a top-down region (`MAX_SHADOW_CTX`, `−1`, …) for a freshly spawned child. `None` if
    /// the reserve is full (the vCPU pool growing down would meet the fiber pool growing up) — a clean
    /// `ThreadFault`, never an overlap. Atomic with the fiber count under the registry lock.
    fn reserve_vcpu_context(&self) -> Option<usize> {
        let mut t = self.lock();
        // Hand out the **highest free** context above the fiber pool (`fibers.len()` occupies contexts
        // `1..=fibers.len()`). Top-down keeps the vCPU pool clear of the upward-growing fibers; reusing
        // a freed (cleared) bit is the recycling that lifts the lifetime cap to peak-concurrent.
        let floor = t.fibers.len();
        let mut c = MAX_SHADOW_CTX;
        while c > floor {
            if t.vcpu_mask & (1 << c) == 0 {
                t.vcpu_mask |= 1 << c;
                return Some(c);
            }
            c -= 1;
        }
        None // the reserve is full (the vCPU pool growing down would meet the fibers growing up)
    }

    /// Free a spawned vCPU's shadow context for reuse (context recycling): called when the child
    /// **genuinely finishes** (not a freeze-unwind, which keeps the region for thaw). A no-op for the
    /// root / a non-durable child (context 0).
    fn free_vcpu_context(&self, ctx: usize) {
        if (1..=MAX_SHADOW_CTX).contains(&ctx) {
            self.lock().vcpu_mask &= !(1 << ctx);
        }
    }

    /// Seed the durable vCPU-context occupancy a **thaw** re-establishes (context recycling): the
    /// re-spawned children reclaim *exactly* the contexts they held at freeze (derived from their
    /// restored shadow-SPs — recycling means these need not be the top `n`), so a post-thaw spawn
    /// allocates into a genuinely-free context. Set once after re-seeding, before forward execution.
    fn seed_vcpu_mask(&self, mask: u16) {
        self.lock().vcpu_mask = mask;
    }

    /// `cont.new`: allocate a slot — the guest handle — under the §15 quota, which is **per run** now
    /// that the table is run-shared (DESIGN.md §23 (per-run quota)). The `+ 1` counts the off-table root
    /// computation. **Recycling (step 3):** the lowest freed slot is reused (its already-bumped
    /// generation kept, so a stale handle to its former occupant fails `claim`); only when none is free
    /// does the table grow. So the table is bounded by the *peak concurrent* fiber count, not the
    /// lifetime total — and the quota / durable-reserve checks (on the grow path / the allocated
    /// context) likewise bound concurrency rather than lifetime.
    fn create(&self, func: i32, sp: i64, max_fibers: usize, durable: bool) -> Result<i64, Trap> {
        let mut t = self.lock();
        let reuse = t.free.peek().map(|&Reverse(s)| s);
        // Growing (no free slot ⇒ every existing slot is live) must honor the concurrency quota.
        if reuse.is_none() && t.fibers.len() + 1 >= max_fibers {
            return Err(Trap::FiberFault);
        }
        let slot = reuse.unwrap_or(t.fibers.len());
        // A durable fiber needs a distinct shadow region; refuse if the reserve has no room (a
        // clean `FiberFault`, like exhausting the quota — never an overflow into another
        // context's region). The fiber's context index is `slot + 1` (the root is context 0). The
        // fiber pool grows up from 1 and the spawned-vCPU pool grows down from `MAX_SHADOW_CTX`
        // (slice 3.2.2), so this fiber must stay strictly below the lowest live vCPU context (which,
        // with recycling, need not be a simple count from the top).
        let lowest_vcpu = {
            if t.vcpu_mask == 0 {
                MAX_SHADOW_CTX + 1
            } else {
                t.vcpu_mask.trailing_zeros() as usize
            }
        };
        if durable && (!shadow_region_fits(slot + 1) || slot + 1 >= lowest_vcpu) {
            return Err(Trap::FiberFault);
        }
        let generation = if reuse.is_some() {
            t.free.pop();
            t.fibers[slot] = RegFiber::Pending { func, sp };
            t.shadow[slot] = shadow_frame_base(slot + 1); // reused region: empty stack at its frame base
            t.gens[slot] // kept from the freed occupant's bump (the ABA guard)
        } else {
            t.fibers.push(RegFiber::Pending { func, sp });
            t.shadow.push(shadow_frame_base(slot + 1));
            t.gens.push(0); // a fresh slot is generation 0 ⇒ handle == slot
            0
        };
        Ok(fiber_handle(slot, generation))
    }

    /// `cont.resume`: resolve the (forgeable) handle — **masked** into the power-of-two-padded
    /// table (Spectre-safe, like `call_indirect`) — and **claim** the fiber for the calling vCPU.
    /// Exactly one of any racing claimants wins (the mutex arbitrates); out-of-range, `Running`
    /// (anyone's — incl. an ancestor in the caller's own resume chain), or `Done` is a lost claim
    /// ⇒ inert [`Trap::FiberFault`]. On a win the slot is `Running(None)` — if the *caller* then
    /// traps before running it (a bad `Pending` funcref), the slot stays claimed forever, which a
    /// later resume sees as an ordinary lost claim.
    fn claim(&self, handle: i64) -> Result<(usize, Claimed), Trap> {
        let mut t = self.lock();
        let mask = t.fibers.len().next_power_of_two() - 1; // len 0 ⇒ mask 0 ⇒ slot 0, caught below
        let slot = (handle as u64 as usize) & mask; // the generation bits are above the slot mask
        if slot >= t.fibers.len() {
            return Err(Trap::FiberFault);
        }
        // Generation check (recycling step 1/3): reject a handle whose generation doesn't match the
        // slot's current one — a stale handle to a slot's former occupant after the slot was recycled
        // (`finish` bumped the generation). Compared modulo `2^40` (the handle's field width); a forged
        // non-zero generation is rejected, exactly as a forged slot is masked-and-lost.
        if fiber_handle_generation(handle) != (t.gens[slot] & FIBER_GEN_MASK) {
            return Err(Trap::FiberFault);
        }
        match std::mem::replace(&mut t.fibers[slot], RegFiber::Running(None)) {
            RegFiber::Pending { func, sp } => Ok((slot, Claimed::Start { func, sp })),
            RegFiber::Parked(f) => Ok((slot, Claimed::Live(f))),
            // §3.6 slice 5a: a woken event-park continues verbatim (result already delivered);
            // a still-blocked one is a poll — the resumer learns FIBER_PARKED, no switch.
            RegFiber::ParkedOn {
                frames,
                woken: true,
            } => Ok((slot, Claimed::LiveWoken(frames))),
            old @ RegFiber::ParkedOn { woken: false, .. } => {
                t.fibers[slot] = old;
                Ok((slot, Claimed::StillParked))
            }
            old => {
                t.fibers[slot] = old; // lost: already running (or done) — put it back untouched
                Err(Trap::FiberFault)
            }
        }
    }

    /// §3.6 slice 5a — park the running fiber on an **event** (fiber-level park): its frames
    /// are set aside, not claimable until [`FiberRegistry::wake_blocked`] flips it. The
    /// suspend-shaped counterpart of [`FiberRegistry::park_suspended`] for parks the guest
    /// did not choose.
    fn park_blocked(&self, slot: usize, frames: Vec<Frame>) {
        let mut t = self.lock();
        debug_assert!(matches!(t.fibers[slot], RegFiber::Running(None)));
        t.fibers[slot] = RegFiber::ParkedOn {
            frames,
            woken: false,
        };
    }

    /// §3.6 slice 5a — the event fired: deliver `result` onto the parked fiber's top frame
    /// (the park op's return value, exactly what the `Pending` resume would have pushed) and
    /// make it claimable. `false` if the slot is not a blocked park (already woken, freed —
    /// the wake is then a no-op, matching every other idempotent wake path).
    fn wake_blocked(&self, slot: usize, result: Reg) -> bool {
        let mut t = self.lock();
        match &mut t.fibers[slot] {
            RegFiber::ParkedOn {
                frames,
                woken: woken @ false,
            } => {
                if let Some(f) = frames.last_mut() {
                    f.vals.push(result);
                }
                *woken = true;
                true
            }
            _ => false,
        }
    }

    /// Park the claimant's current fiber as an **active resumer** (it just executed
    /// `cont.resume`): its frames are stored but the slot stays `Running` — an ancestor in a
    /// resume chain is never claimable.
    fn park_resumer(&self, slot: usize, frames: Vec<Frame>) {
        let mut t = self.lock();
        debug_assert!(matches!(t.fibers[slot], RegFiber::Running(None)));
        t.fibers[slot] = RegFiber::Running(Some(frames));
    }

    /// Pop back into a parked resumer (its resumee suspended or returned): take its frames; the
    /// slot stays `Running(None)` (its frames are in flight again).
    fn unpark_resumer(&self, slot: usize) -> Result<Vec<Frame>, Trap> {
        let mut t = self.lock();
        match std::mem::replace(
            t.fibers.get_mut(slot).ok_or(Trap::Malformed)?,
            RegFiber::Running(None),
        ) {
            RegFiber::Running(Some(f)) => Ok(f),
            _ => Err(Trap::Malformed),
        }
    }

    /// `suspend`: publish the claimant's current fiber back to the pool — claimable by **any**
    /// vCPU again (the migration point).
    fn park_suspended(&self, slot: usize, frames: Vec<Frame>) {
        let mut t = self.lock();
        debug_assert!(matches!(t.fibers[slot], RegFiber::Running(None)));
        t.fibers[slot] = RegFiber::Parked(frames);
    }

    /// The claimant's current fiber returned: the slot is `Done`. **Recycling (step 3):** bump the
    /// slot's generation (so any stale guest handle to it now fails `claim` — the ABA guard) and add it
    /// to the free list, reclaimable for a new `cont.new`. Resuming the *old* handle faults either way
    /// (a `Done` slot, or — once reused — a generation mismatch).
    fn finish(&self, slot: usize) {
        let mut t = self.lock();
        debug_assert!(matches!(t.fibers[slot], RegFiber::Running(None)));
        t.fibers[slot] = RegFiber::Done;
        t.gens[slot] = t.gens[slot].wrapping_add(1);
        t.free.push(Reverse(slot));
    }

    /// Freeze (slice 3.2 active-chain): the running fiber base-returned **while unwinding for a
    /// freeze** (state == UNWINDING), so it didn't really finish — its continuation now lives in its
    /// shadow region. Mark the slot `Frozen` (re-enterable on thaw, *not* `Done`) and record its
    /// flattened shadow-SP, so the residue + a thaw re-seed reconstruct it like an idle-parked fiber.
    fn freeze_active(&self, slot: usize, shadow_sp: u64) {
        let mut t = self.lock();
        debug_assert!(matches!(t.fibers[slot], RegFiber::Running(None)));
        t.fibers[slot] = RegFiber::Frozen;
        t.shadow[slot] = shadow_sp;
    }

    /// Freeze driver (slice 3.1.4): take the lowest still-`Parked` fiber's frames and mark its slot
    /// `Frozen`, so the driver can flatten it into its shadow region and not revisit it. Returns
    /// `(slot, frames)`, or `None` once every fiber is flattened (no `Parked` slot remains).
    /// §3.6 slice 5a — whether any fiber is **event-parked** (`ParkedOn`). A durable freeze
    /// fails closed on one: its wake is host-side scheduler state (a waiter entry) that no
    /// snapshot can carry — durable event-parks are a recorded follow-up.
    fn has_blocked_parks(&self) -> bool {
        self.lock()
            .fibers
            .iter()
            .any(|f| matches!(f, RegFiber::ParkedOn { .. }))
    }

    fn take_parked_for_freeze(&self) -> Option<(usize, Vec<Frame>)> {
        let mut t = self.lock();
        let slot = t
            .fibers
            .iter()
            .position(|f| matches!(f, RegFiber::Parked(_)))?;
        match std::mem::replace(&mut t.fibers[slot], RegFiber::Frozen) {
            RegFiber::Parked(frames) => Some((slot, frames)),
            _ => unreachable!("position found a Parked slot"),
        }
    }

    /// DURABILITY.md §13.4 step 2 — the event-parked (`ParkedOn`) fibers as `(slot, woken)`,
    /// for the freeze driver's classification: a **woken** park carries its delivered result
    /// in its frames (the point's spill reloads it at thaw), an unwoken park freezes only if
    /// its thaw arm **re-issues** (a futex wait) — anything else fails the freeze closed.
    fn blocked_parks(&self) -> Vec<(usize, bool)> {
        self.lock()
            .fibers
            .iter()
            .enumerate()
            .filter_map(|(i, f)| match f {
                RegFiber::ParkedOn { woken, .. } => Some((i, *woken)),
                _ => None,
            })
            .collect()
    }

    /// Take the lowest still-event-parked fiber for the freeze flatten (the `ParkedOn`
    /// counterpart of [`Self::take_parked_for_freeze`]): `(slot, frames, woken)`, the slot
    /// marked `Frozen`.
    fn take_blocked_for_freeze(&self) -> Option<(usize, Vec<Frame>, bool)> {
        let mut t = self.lock();
        let slot = t
            .fibers
            .iter()
            .position(|f| matches!(f, RegFiber::ParkedOn { .. }))?;
        match std::mem::replace(&mut t.fibers[slot], RegFiber::Frozen) {
            RegFiber::ParkedOn { frames, woken } => Some((slot, frames, woken)),
            _ => unreachable!("position found a ParkedOn slot"),
        }
    }

    /// Thaw seeding (slice 3.1.5): re-create a frozen fiber at the next slot as `Pending` (so a
    /// thaw `cont.resume` re-enters its entry under `REWINDING`) with its flattened shadow-SP in the
    /// `shadow` table (so the swap re-points there). Seed in ascending slot order to rebuild the
    /// dense handle namespace; returns the slot, which must equal the recorded one.
    fn seed_frozen(&self, func: i32, sp: i64, shadow_sp: u64, generation: u64) -> usize {
        let mut t = self.lock();
        let slot = t.fibers.len();
        t.fibers.push(RegFiber::Pending { func, sp });
        t.shadow.push(shadow_sp);
        t.gens.push(generation); // restore the freeze-time generation so a recycled handle resolves
        slot
    }
}

/// The fixed fiber entry signature (§12): a fiber runs a function of type `(i64 sp, i64
/// arg) -> i64`. `sp` is the fiber's data-stack base (the §3d two-stack split — every
/// frontend-emitted function already takes the data-SP as its first param); `arg` carries
/// the first-resume value in and the final value out (a window pointer can carry richer
/// payloads).
fn fiber_sig() -> FuncType {
    FuncType {
        params: vec![ValType::I64, ValType::I64],
        results: vec![ValType::I64],
    }
}

/// Resolve a `thread.join` handle to a table slot (§12). Like [`resolve_fiber`], the handle is
/// forgeable, so it is **masked** into the power-of-two-padded table, then bounds- and
/// liveness-checked: out of range or an already-joined (`None`) slot is inert ([`Trap::ThreadFault`]).
fn resolve_thread<T>(threads: &[Option<T>], handle: i32) -> Result<usize, Trap> {
    if threads.is_empty() {
        return Err(Trap::ThreadFault);
    }
    let mask = threads.len().next_power_of_two() - 1;
    let slot = (handle as u32 as usize) & mask;
    if slot >= threads.len() || threads[slot].is_none() {
        return Err(Trap::ThreadFault);
    }
    Ok(slot)
}

/// Run one vCPU. `funcs` is an `Arc<[Func]>` the vCPU **owns** (a child gets its own cheap clone), so
/// a spawned vCPU borrows nothing from its parent and can run on a detached OS thread (the seam for a
/// `'static` worker pool). The shared runtime state — thread `budget`, the `parking` lot, the
/// `registry` of spawned threads — is `Arc`-shared across all vCPUs.
///
/// All the run state is **owned**, so a vCPU is `Send` and self-contained — the basis for moving it
/// A resolved §14 `Module` grant's pieces, as the eval loop carries them from the op decode
/// into the shared spawn logic (`Arc`s — spawning shares, never copies). `None` = a
/// same-module child (runs the parent's own program).
struct ChildMod {
    funcs: Arc<[Func]>,
    memory_log2: Option<u8>,
    data: Arc<[Data]>,
    durable: bool,
    digest: [u8; 32],
    imports: Arc<[svm_ir::Import]>,
    types: Arc<[svm_ir::TypeEntry]>,
    /// §3.6 — the whole granted module, registered as the child's **self module** at spawn so
    /// a separate-module child serves its *own* offers (enqueue admission and handler
    /// resolution go through it), exactly as a same-module child serves the parent's.
    module: Arc<Module>,
}

/// A §14 co-fiber child the parent drives with `resume`: its suspended continuation plus whether it is
/// **awaiting a resume value** — i.e. parked at a `yield` (so the next `resume` delivers its argument
/// as the yield's result) vs. freshly spawned at its entry (the first `resume` just starts it, its
/// argument unused). The child runs *inline* on the parent's thread, never on the executor.
struct Coro {
    vcpu: Box<VCpu>,
    awaiting_resume: bool,
    /// When set, the child is suspended at a **fault-driven yield** awaiting this (confined) page: the
    /// next `resume` supplies it (maps it read-write) before re-running the rewound faulting access.
    faulted_page: Option<u64>,
}

/// between worker threads and (next) parking its continuation on a blocking op.
struct VCpu {
    /// The owned function table; `Frame::func` resolves against it.
    funcs: Arc<[Func]>,
    /// The §12 fiber table — **shared by every vCPU of the domain** (root + `thread.spawn`
    /// children, D57 3b-i), so any vCPU can resume any fiber; the registry's claim is the
    /// single-owner arbiter. Nested §14 children / separate domains get a fresh registry.
    registry: Arc<FiberRegistry>,
    /// The resume chain: `chain[0]` is [`ROOT_FIBER`] (this vCPU's off-table root computation),
    /// `chain.last()` the running fiber's registry slot.
    chain: Vec<usize>,
    /// Registry slot of the running fiber ([`ROOT_FIBER`] when the root is running).
    cur: usize,
    /// The running fiber's reified call stack.
    frames: Vec<Frame>,
    /// The root computation's parked frames while it is resuming a fiber (the root lives
    /// off-table — see [`ROOT_FIBER`] — so its parked state can't go in the registry).
    root_parked: Option<Vec<Frame>>,
    /// Total frames across this vCPU's parked resume-chain ancestors (incl. a parked root) —
    /// maintained at fiber switches so the recursion depth bound (`MAX_CALL_DEPTH`) spans all
    /// active fibers without walking the shared registry on every call.
    parked_frames: usize,
    /// This run executes a **durable** (freeze/thaw-instrumented) module, so the runtime keeps the
    /// active shadow-SP word ([`SHADOW_SP_OFF`]) pointing at the running context's per-fiber shadow
    /// region, swapping it on every fiber switch (D-fiber-cont option A, DURABILITY.md §12.8). A
    /// non-durable run leaves this `false` and never touches the reserve. Set from
    /// [`Host::is_durable`] by `drive`, inherited by `thread.spawn` children.
    durable: bool,
    /// The root computation's saved shadow-SP (window offset) while it is parked resuming a fiber
    /// — the off-table root's slot in the per-context saved-SP table (a fiber's lives in the
    /// registry's `shadow`). The root is context 0, so this starts at [`shadow_frame_base`]`(0)` (its
    /// in-region SP word at `SHADOW_BASE`, frames just past it).
    root_shadow_sp: u64,
    /// §12.8 4A.5: the **active spill context** — whose region the running instrumented code addresses
    /// via `durable.shadow_base`. The root's `vcpu_ctx` while at root, a fiber's `slot + 1` while a
    /// fiber runs (maintained at the fiber switch), and the driven fiber's during `freeze_drive` (where
    /// `cur` is the `ROOT_FIBER` sentinel but the spill must land in the fiber's region).
    durable_sp_ctx: usize,
    /// Fibers the freeze driver flattened this run (slice 3.1.5), handed back to the embedder via
    /// the shared [`Host`] so a snapshot can record them and a thaw re-seed them. Empty otherwise.
    frozen: Vec<FrozenFiber>,
    /// `Some` on a **spawned** (`thread.spawn`) vCPU: its `(entry, [sp, arg])`, retained so that when
    /// it unwinds under a freeze it can emit its [`FrozenVCpu`] residue (its frames are gone by then).
    /// `None` on the root (whose entry/args the thaw caller supplies) and on every non-durable vCPU.
    /// (slice 3.2.1)
    spawn_residue: Option<(FuncIdx, Vec<i64>)>,
    /// This spawned vCPU's durable **shadow context** (`1..=MAX_SHADOW_CTX`), reserved at
    /// `thread.spawn` and freed back to the registry when the vCPU genuinely finishes (context
    /// recycling). 0 for the root and every non-durable vCPU (nothing to free).
    vcpu_ctx: usize,
    /// This vCPU's **saved durable state word** (`NORMAL | UNWINDING | REWINDING`), swapped into the
    /// shared window word ([`STATE_OFF`]) by the runtime when this vCPU runs and saved back when it
    /// parks (slice 3.2.1). Multi-vCPU freeze/thaw run single-worker, so the one shared state word is
    /// each vCPU's *own* context, swapped per dispatch — essential because a rewinding vCPU flips the
    /// word to `NORMAL` after reloading, which must not disturb siblings still rewinding. A non-durable
    /// run leaves it `NORMAL` and never touches the word.
    dstate: i32,
    /// This vCPU's linear-memory view (shared `Region` + address space; see [`Mem`]).
    mem: Option<Mem>,
    /// The domain's powerbox, **shared** by every vCPU of the run (`Arc<Mutex<Host>>`): a spawned
    /// thread inherits the same capability table + I/O sinks, so a handle granted to the domain works
    /// in any thread and I/O from any thread reaches the same sink (matching the JIT, whose `cap.call`s
    /// all hit the one host ctx). Locked briefly per `cap.call`.
    host: Arc<Mutex<Host>>,
    /// The residue sink for a §14 subtree freeze (DURABILITY.md §4) — where this vCPU's
    /// [`FrozenNested`] records are pushed. `None` = use `self.host` (the root, or a `thread.spawn`
    /// domain that shares its host). A §14 nested child, whose capability host is **private** (its
    /// own attenuated powerbox), inherits its parent's *effective* sink so a whole nesting subtree's
    /// residue coalesces in the root host — mirroring how `thread.spawn`'s shared host coalesces
    /// [`FrozenVCpu`] residue. Without this a grandchild's [`FrozenNested`] would be orphaned in the
    /// child's private host and lost to the snapshot.
    freeze_sink: Option<Arc<Mutex<Host>>>,
    /// Remaining fuel (metering, §5).
    fuel: u64,
    /// This vCPU's spawned children, by `thread.join` handle (slot) ⇒ child [`TaskId`]; `None` once
    /// joined (a re-join is inert).
    threads: Vec<Option<TaskId>>,
    /// Which [`VCpu::threads`] slots were created by a §14 **`Instantiator.instantiate`** (vs. a §12
    /// `thread.spawn`), with each child's carve geometry + entry. A durable freeze must distinguish
    /// them: a spawned vCPU rides the artifact as `FrozenVCpu` residue, while a §14 child is its
    /// **own domain** whose window (a sub-range of this vCPU's — shadow regions and state words
    /// included) already rides the parent's window image. On a freeze, a **live** same-module child
    /// is driven to self-unwind into its carve (the subtree STW broadcast) and recorded as
    /// [`FrozenNested`] residue; the un-covered shapes (separate-module child, completed-but-unjoined
    /// child, nested-in-nested) still fail closed (DURABILITY.md §4).
    nested_children: Vec<NestedChildInfo>,
    /// §3.6 slice 3 — live powerboxes of this vCPU's §14 children, keyed by their `threads`
    /// slot. Retained so the parent can mint a [`Binding::LiveImpl`] over a child's offer
    /// (`Instantiator.child_offer`, op 14) — the caller-parking linkage. Live-only (never
    /// frozen: a LiveImpl is non-durable), kept out of `NestedChildInfo` so the freeze
    /// records stay `Copy`.
    child_hosts: BTreeMap<usize, Arc<Mutex<Host>>>,
    /// This vCPU **is** a §14 instantiated child (depth-1). Its freeze-unwind is self-describing —
    /// its extent lives in its carve's shadow-SP word, inside the parent's window image — so its
    /// freeze completion records no host-side residue (and must not clobber the root's).
    nested_child: bool,
    /// This vCPU's §14 **co-fiber** children (`Instantiator.spawn_coroutine`): suspended continuations
    /// (their own frames/mem/host) driven *inline* by `resume`, by handle (slot). `None` once the
    /// coroutine has run to completion (a later `resume` is inert). Distinct from `threads` — a
    /// coroutine is cooperative (parent and child never run concurrently), not an executor vCPU.
    coroutines: Vec<Option<Coro>>,
    /// §14: this vCPU is a coroutine whose recoverable page faults **suspend to its parent** (fault-
    /// driven yield / lazy paging) instead of trapping. Set for `Instantiator.spawn_coroutine`
    /// children; `false` for every ordinary vCPU (a page fault is detect-and-kill).
    fault_yields: bool,
    /// Call-depth base for the stack-overflow bound.
    depth: u32,
    /// This task's own id (where its outcome is published on completion).
    id: TaskId,
    /// The id of the vCPU that spawned this one (slice 3.4: nested spawns) — `0` for the root and every
    /// non-durable vCPU. Stamped at `thread.spawn` and on a thaw re-attach so a freeze records each
    /// child's parent in its [`FrozenVCpu`], letting thaw rebuild the per-parent join-table topology.
    parent_task: TaskId,
    /// §12 per-vCPU **thread-local register** (`vcpu.tls.get`/`set`). One i64 of per-vCPU state,
    /// seeded to this vCPU's dense id at construction (root = 0), guest-overwritable. Read at the
    /// op's execution point — so a fiber that migrated here reads *this* vCPU's word.
    tls: i64,
    /// `<setjmp.h>` checkpoints — `setjmp` records this vCPU's resume point here keyed by the guest
    /// `jmp_buf` window address; `longjmp` looks it up. Per-vCPU (a checkpoint references *this* frame
    /// stack; cross-thread `longjmp` is UB in C and simply misses). Keyed by buffer address (not a
    /// growing token table) so a re-`setjmp` to the same buffer overwrites and a `pcall`-in-a-loop
    /// stays bounded; the trade-off is that a *copied* `jmp_buf` (rare/UB-adjacent) misses → traps.
    setjmp_points: BTreeMap<u64, SetJmpPoint>,
    /// Set when resuming from a park: how to finish the blocked op (see [`Pending`]).
    pending: Option<Pending>,
    /// The executor this vCPU runs under — the real OS-thread [`Scheduler`] or the deterministic
    /// [`DetSched`] (spawn enqueues here; notify wakes here).
    sched: SchedRef,
    /// When set, the `quantum` budget counts **visible (shared-memory / sync) operations** rather than
    /// raw instructions, so the vCPU yields at memory-op boundaries. The exhaustive model checker
    /// ([`explore_all`]) uses `memop = true` + `quantum = 1` to make every shared-state access a
    /// scheduling point; the real pool and the seeded explorer leave it `false`.
    memop: bool,
    /// The object touched by the **visible op this turn ran** (set in `memop` mode at the op's commit
    /// point; `None` if the turn ran no visible op). Read back by the DPOR driver ([`explore_all`]) to
    /// build the schedule trace; unused by the real pool / seeded explorer.
    acc: Option<MemAccess>,
    /// §15 spawn quota (fiber/vCPU ceilings) — inherited by every vCPU of the run from the root.
    quota: Quota,
    /// The domain's **shared, live** dispatch table (slots + installed units), shared by every vCPU
    /// of the domain via `Arc` so a guest-driven `install` is visible across `thread.spawn` children
    /// and `Jit.invoke` children (DESIGN.md §22). Reads are lock-free atomic loads (see
    /// [`DomainTable`]).
    dt: Arc<DomainTable>,
    /// A lock-free **local clone** (a prefix) of `dt`'s installed units (module `k` ≡ `units[k-1]`),
    /// re-synced only on a miss (see [`resolve_module`]) — so resolving a running unit frame or an
    /// installed `call_indirect` target never locks the shared `units`.
    units: Vec<Arc<[Func]>>,
    /// The transient unit a `Jit.invoke` is running on this vCPU (`Some` only for an invoke child),
    /// resolved as module [`INVOKE_MODULE`] — kept out of the shared `dt.units` so it is never
    /// installed/`call_indirect`-reachable and never collides with a concurrent install.
    invoked: Option<Arc<[Func]>>,
    /// **Debug seam** (DEBUGGING.md W2/S4): `Some` only when an [`Inspector`] drives this vCPU.
    /// `None` is the production hot path — the per-op hook in [`run_inner`] is gated on it, so an
    /// undebugged run pays a single null check per op and is otherwise byte-identical (S7). Not
    /// inherited across `thread.spawn` (slice 1 debugs single-threaded guests).
    debug: Option<Box<DebugCtx>>,
    /// PROCESS.md S3 `kill` — `Some` on a §14 child (and, inherited, its `thread.spawn` descendants):
    /// a shared flag the parent sets via `Instantiator.kill`. Polled at the per-op fuel `step`; when
    /// set the vCPU traps (`ThreadFault`, which `poll` reports as `2`), so the child's whole subtree
    /// self-terminates. `None` on the root and top-level threads (nothing above them to kill them).
    kill: Option<Arc<AtomicBool>>,
    /// PROCESS.md S3 `kill` — a parent's map from a §14 child's join-table **slot** to that child's
    /// kill flag ([`VCpu::kill`]), so `Instantiator.kill(child)` sets it. Sparse (only §14 children,
    /// not `thread.spawn` threads, which share their §14 ancestor's flag); empty on a leaf vCPU.
    child_kill: BTreeMap<usize, Arc<AtomicBool>>,
    /// §3.6 slice 5b — the serve loop's **running handler fiber**, set when the
    /// `svc.poll`/`svc.wait` arm switches into one and consumed when the serve frame re-executes
    /// (the handler returned, fiber-parked, or suspended). See [`ServeRun`].
    serve_run: Option<ServeRun>,
    /// §3.6 slice 5b — **event-parked handler fibers** of this vCPU's serve loop, registry slot
    /// → (fiber handle, dispatch ticket). A handler that fiber-parked is a
    /// completed-but-not-replied dispatch: its caller stays parked in `ticket_waiters`, the
    /// serve loop moves on. Each serve re-execution re-claims these — still-blocked ones are
    /// put back; a woken one is resumed, and its eventual return finally replies.
    handler_parks: BTreeMap<usize, (i64, u64)>,
    /// §3.6 slice 5b — dispatches completed by the current `svc.poll`/`svc.wait` activation
    /// (the op's result). Lives on the vCPU because the activation spans rewind-driven
    /// re-executions (and possibly a `svc.wait` park); reset when the count is delivered.
    serve_count: i64,
}

/// §3.6 slice 5b — the serve loop's in-flight handler: the registry slot/handle the handler
/// fiber occupies, the dispatch ticket its return answers, and the fiber the serve frame
/// itself runs as (`serve_cur`) — which distinguishes the serve frame's own rewound
/// re-execution from a nested `svc.*` executed *under* the handler (refused with a probeable
/// `-EINVAL`: the serve loop is the domain's outermost dispatcher).
struct ServeRun {
    slot: usize,
    handle: i64,
    ticket: u64,
    serve_cur: usize,
}

impl VCpu {
    /// A fresh vCPU whose root frame is `funcs[entry](args)`. A bad `entry` is caught by the driver's
    /// first block lookup ([`Trap::Malformed`]), so construction is infallible.
    #[allow(clippy::too_many_arguments)]
    fn new(
        funcs: Arc<[Func]>,
        entry: FuncIdx,
        args: &[Value],
        mem: Option<Mem>,
        host: Arc<Mutex<Host>>,
        fuel: u64,
        depth: u32,
        id: TaskId,
        sched: SchedRef,
        quota: Quota,
        dt: Arc<DomainTable>,
    ) -> VCpu {
        VCpu {
            funcs,
            registry: Arc::new(FiberRegistry::new()),
            chain: vec![ROOT_FIBER],
            cur: ROOT_FIBER,
            frames: vec![Frame {
                func: entry,
                module: 0,
                block: 0,
                inst: 0,
                vals: args.iter().map(|&x| Reg::from_value(x)).collect(),
            }],
            root_parked: None,
            parked_frames: 0,
            durable: false,
            root_shadow_sp: shadow_frame_base(0),
            durable_sp_ctx: 0,
            frozen: Vec::new(),
            spawn_residue: None,
            vcpu_ctx: 0,
            dstate: STATE_NORMAL,
            mem,
            host,
            freeze_sink: None,
            fuel,
            threads: Vec::new(),
            nested_children: Vec::new(),
            child_hosts: BTreeMap::new(),
            nested_child: false,
            coroutines: Vec::new(),
            fault_yields: false,
            depth,
            id,
            parent_task: 0,
            tls: id as i64, // §12 seed the per-vCPU TLS register to the dense vCPU id (root = 0)
            setjmp_points: BTreeMap::new(),
            pending: None,
            sched,
            memop: false,
            acc: None,
            quota,
            dt,
            units: Vec::new(),
            invoked: None,
            debug: None,
            kill: None,
            child_kill: BTreeMap::new(),
            serve_run: None,
            handler_parks: BTreeMap::new(),
            serve_count: 0,
        }
    }

    /// A vCPU that runs a guest-compiled **`Jit` unit**'s entry (`unit[0]`) over the parent's
    /// world — same window/host/fuel — for the `Jit.invoke` op (DESIGN.md §22/B2). Module 0 is
    /// the `parent` program; `dt` is the **shared, live** domain table, so the unit's `call_indirect`
    /// reaches the original program (new→old) *and* any already- **or later-** `install`ed units
    /// (new→new, incl. install-during-own-invocation), exactly as the JIT's invoked code dispatches
    /// through the live `fn_table`. The invoked unit runs as the transient [`INVOKE_MODULE`] (kept
    /// out of `dt.units`, so it is never itself `call_indirect`-reachable).
    #[allow(clippy::too_many_arguments)]
    fn new_invoke(
        parent: Arc<[Func]>,
        dt: Arc<DomainTable>,
        unit: Arc<[Func]>,
        args: &[Value],
        mem: Option<Mem>,
        host: Arc<Mutex<Host>>,
        fuel: u64,
        depth: u32,
        sched: SchedRef,
        quota: Quota,
    ) -> VCpu {
        VCpu {
            funcs: parent,
            // A unit cannot use `cont.*` (gated at compile), so its registry is never touched.
            registry: Arc::new(FiberRegistry::new()),
            chain: vec![ROOT_FIBER],
            cur: ROOT_FIBER,
            frames: vec![Frame {
                func: 0, // the unit's entry
                module: INVOKE_MODULE,
                block: 0,
                inst: 0,
                vals: args.iter().map(|&x| Reg::from_value(x)).collect(),
            }],
            root_parked: None,
            parked_frames: 0,
            durable: false,
            root_shadow_sp: shadow_frame_base(0),
            durable_sp_ctx: 0,
            frozen: Vec::new(),
            spawn_residue: None,
            vcpu_ctx: 0,
            dstate: STATE_NORMAL,
            mem,
            host,
            freeze_sink: None,
            fuel,
            threads: Vec::new(),
            nested_children: Vec::new(),
            child_hosts: BTreeMap::new(),
            nested_child: false,
            coroutines: Vec::new(),
            fault_yields: false,
            depth,
            id: 0, // unused: driven inline, never via the executor
            parent_task: 0,
            tls: 0, // §12 per-vCPU TLS seed (id 0)
            setjmp_points: BTreeMap::new(),
            pending: None,
            sched,
            memop: false,
            acc: None,
            quota,
            dt,
            units: Vec::new(),
            invoked: Some(unit),
            debug: None,
            kill: None,
            child_kill: BTreeMap::new(),
            serve_run: None,
            handler_parks: BTreeMap::new(),
            serve_count: 0,
        }
    }

    /// A 64-bit fingerprint of this vCPU's **local** execution configuration (its resume chain +
    /// reified call stacks: function / block / instruction / SSA values — everything *except* shared
    /// memory and the shared fiber registry, see below). The
    /// explorer compares it across one turn: a visible op that returns the vCPU to the same fingerprint
    /// has gone once around a loop with no local progress — a spin (livelock unless shared memory it
    /// reads changes). Collisions would risk a false spin-park, but two configs of the *same* vCPU one
    /// op apart colliding is ~2^-64; the values fully determine the hash (floats by bit pattern).
    fn local_fingerprint(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        fn hash_vals(h: &mut std::collections::hash_map::DefaultHasher, vals: &[Reg]) {
            vals.len().hash(h);
            for v in vals {
                // Untyped raw slots: hash the two words directly (the fingerprint is only compared
                // pre/post one turn of this vCPU, so any stable encoding is fine).
                (v.lo, v.hi).hash(h);
            }
        }
        fn hash_frames(h: &mut std::collections::hash_map::DefaultHasher, frames: &[Frame]) {
            frames.len().hash(h);
            for f in frames {
                (f.func, f.module, f.block, f.inst).hash(h);
                hash_vals(h, &f.vals);
            }
        }
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.cur.hash(&mut h);
        self.chain.hash(&mut h);
        hash_frames(&mut h, &self.frames);
        // The root's parked frames are local configuration (a suspend could pop back into them).
        // The **shared** fiber registry (D57) is deliberately *not* hashed: the compare is
        // pre/post one turn of *this* vCPU, and within a turn its chain-held slots can't change
        // (claimed exclusively by it) while pool fibers can't affect a spin — every fiber op is a
        // visible op that visibly changes this fingerprint (chain/vals), so a fingerprint-stable
        // turn touched no fiber state, and a failed claim is a trap (vCPU ends), not a retry.
        match &self.root_parked {
            Some(f) => {
                1u8.hash(&mut h);
                hash_frames(&mut h, f);
            }
            None => 0u8.hash(&mut h),
        }
        h.finish()
    }

    /// The current `width`-byte value at confined `key` (no checks; used by the executor for the
    /// futex compare under the scheduler lock). Zero if this vCPU has no memory.
    fn atomic_value(&self, key: u64, width: u32) -> u64 {
        self.mem.as_ref().map_or(0, |m| m.atomic_value(key, width))
    }

    /// This vCPU's logical-time `clock` (ops executed), or 0 if undebugged. The coordinate a
    /// single-threaded time-travel `seek`/checkpoint is keyed by.
    fn debug_clock(&self) -> u64 {
        self.debug.as_ref().map(|d| d.clock).unwrap_or(0)
    }

    /// Arm the time-travel replay to fast-forward to logical time `t` (W1): the next `run` advances
    /// `clock` to `t`, ignoring breakpoints, then pauses. Used by the chunked checkpoint drive.
    fn dbg_seek_to(&mut self, t: u64) {
        if let Some(d) = self.debug.as_mut() {
            d.seek_target = Some(t);
        }
    }

    /// Whether this vCPU's state is fully captured by `frames` + the window bytes — the subset a
    /// single-threaded time-travel **checkpoint** (W1) snapshots: the **root** computation is running
    /// (no fiber resume-chain or parked root), nothing durable/frozen, no `thread.spawn`/coroutine
    /// children, and memory has a pristine layout (no `map`/`unmap`/`protect`/grow or §13 region
    /// aliasing, so `snapshot`/`seed` of the mapped prefix round-trips). Outside this subset the
    /// `Inspector` stops checkpointing and falls back to replay-from-clock-0.
    fn checkpointable(&self) -> bool {
        self.cur == ROOT_FIBER
            && self.chain.as_slice() == [ROOT_FIBER]
            && self.root_parked.is_none()
            // §3.6 5a/5b: an event-parked fiber's frames (incl. a parked serve handler's) live
            // in the registry, outside the frames+window capture — no checkpoint.
            && !self.registry.has_blocked_parks()
            && self.handler_parks.is_empty()
            && self.frozen.is_empty()
            && !self.durable
            && self.threads.is_empty()
            && self.coroutines.is_empty()
            && self.invoked.is_none() // no guest-installed §22 units (would need the domain table rebuilt)
            && self.mem.as_ref().is_none_or(|m| m.snapshot_safe())
    }

    /// Restore a checkpoint's continuation into this freshly-built root vCPU (from
    /// [`Inspector::fresh_single_root`]): replace the call stack, fuel, window bytes, and logical clock
    /// so a subsequent `run` (with a `seek_target`) resumes the replay exactly at the checkpoint's
    /// logical time. The shared structure (funcs, fresh registry, host, scheduler) already matches a
    /// root-only run, which is the only kind that is checkpointed.
    fn restore_continuation(
        &mut self,
        frames: Vec<Frame>,
        fuel: u64,
        mem_bytes: Option<&[u8]>,
        clock: u64,
    ) {
        self.frames = frames;
        self.fuel = fuel;
        if let (Some(m), Some(bytes)) = (self.mem.as_mut(), mem_bytes) {
            m.seed(bytes);
        }
        if let Some(d) = self.debug.as_mut() {
            d.clock = clock;
        }
    }

    /// Run for up to `quantum` instructions, then finish / park / yield. The real executor passes
    /// `u64::MAX` (run to completion or park); the deterministic explorer passes a small seeded
    /// quantum to interleave vCPUs finely. Folds a trap into `Step::Done(Err)`.
    fn run(&mut self, quantum: u64) -> Step {
        match run_inner(self, quantum) {
            Ok(Inner::Done(v)) => Step::Done(Ok(v)),
            Ok(Inner::Park(b)) => Step::Park(b),
            Ok(Inner::Yield) => Step::Yield,
            // A `Yielder` cap.call / fault-driven yield on a vCPU the *executor* runs has no resumer to
            // yield to (a coroutine child is driven inline by `resume`, never enqueued here) — inert.
            Ok(Inner::CoYield(_)) | Ok(Inner::CoFault(_)) => Step::Done(Err(Trap::FiberFault)),
            Ok(Inner::Pause(r, pc)) => Step::Pause(r, pc),
            Err(t) => Step::Done(Err(t)),
        }
    }

    /// **Freeze driver** (DURABILITY.md §12.8 slice 3.1.4). Called once this vCPU's root has run to
    /// completion under `UNWINDING` (its native stack drained into the root's shadow region): flatten
    /// every still-**parked** fiber into *its own* shadow region so the window snapshot captures it.
    ///
    /// Each parked fiber is resumed under `UNWINDING` like a standalone root run — its frames become
    /// the active stack with `cur = ROOT_FIBER`, the active shadow-SP is pointed at the fiber's region
    /// base, and a placeholder resume value is delivered (mimicking `cont.resume`, so the post-suspend
    /// continuation is well-formed; the suspend's result slot is inert — the `Yield` thaw arm
    /// redelivers it). Because the transform places the poll **immediately** after the `suspend`, that
    /// poll fires before any of the fiber's guest code runs, so it unwinds with **zero forward
    /// progress** and its base-frame return (under `cur == ROOT_FIBER`) ends the sub-run. The fiber's
    /// flattened shadow-SP extent is recorded in the registry's `shadow` table for the snapshot.
    ///
    /// Each flattened fiber is recorded as a [`FrozenFiber`] in `self.frozen` (handed to the embedder
    /// via the [`Host`] for the snapshot / a thaw re-seed). The active shadow-SP is left pointing at
    /// the **root's** region on return, so the captured window is thaw-ready (the root rewinds first;
    /// each fiber's own SP travels in its `FrozenFiber`, re-seeded into the registry on thaw).
    ///
    /// Single-vCPU only (slice 3.1); the multi-vCPU stop-the-world choreography is slice 3.2. Handles
    /// idle parked fibers; a fiber still on an active resume chain at freeze unwinds with the root and
    /// is a 3.1.5/3.2 follow-up.
    fn freeze_drive(&mut self) -> Result<(), Trap> {
        // DURABILITY.md §13.4 step 2: serve-handler parks stay fail-closed until serve-state
        // capture (step 3) — their reply linkage (`handler_parks`) is per-vCPU serve state no
        // snapshot carries yet.
        if !self.handler_parks.is_empty() {
            return Err(Trap::FiberFault);
        }
        // §13.4 step 2 classification (before anything is consumed): an event-parked fiber
        // freezes when its thaw can re-derive the park — a WOKEN park's delivered result is
        // already in its frames (the point's spill reloads it), and an unwoken FUTEX park
        // re-issues at its `MemoryWait` thaw arm. An unwoken CAP park would spill the freeze
        // placeholder into a `Leaf` frame (reloaded as the call's result — unsound), so it
        // fails the whole freeze closed.
        for (slot, woken) in self.registry.blocked_parks() {
            if !woken && !self.sched.fiber_wait_parked(&self.registry, slot) {
                return Err(Trap::FiberFault);
            }
        }
        // §12.8 4A.5: this vCPU's root region word (where the root's SP lives); restored at the end so
        // the window is thaw-ready (the root rewinds first).
        let root_word = shadow_region_base(self.vcpu_ctx);
        let root_sp = self
            .mem
            .as_ref()
            .map(|m| m.durable_get_sp(root_word))
            .unwrap_or_else(|| shadow_frame_base(self.vcpu_ctx));
        while let Some((slot, frames)) = self.registry.take_parked_for_freeze() {
            // Placeholder resume value (inert; not spilled by `Yield`).
            self.flatten_fiber_for_freeze(slot, frames, Some(Reg::from_i64(0)))?;
        }
        // §13.4 step 2 — flatten the event-parked fibers (classified above): a woken park's
        // frames already carry its delivered result (no placeholder — the point's spill reloads
        // the real value at thaw); an unwoken futex park's waiter entry is consumed here and an
        // inert status is delivered — the `MemoryWait` point spills `out − nres` (the status is
        // never captured) and its thaw arm re-issues the wait, which re-checks the restored
        // guest value (the O10 re-issue rule turned inward).
        while let Some((slot, frames, woken)) = self.registry.take_blocked_for_freeze() {
            let placeholder = if woken {
                None
            } else {
                self.sched.purge_fiber_wait_park(&self.registry, slot);
                Some(Reg::from_i32(0))
            };
            self.flatten_fiber_for_freeze(slot, frames, placeholder)?;
        }
        // Leave the active shadow-SP at the root's region: the root rewinds first on thaw.
        self.durable_sp_ctx = self.vcpu_ctx;
        if let Some(m) = self.mem.as_mut() {
            m.durable_set_sp(root_word, root_sp);
        }
        Ok(())
    }

    /// One fiber's freeze flatten (the [`VCpu::freeze_drive`] loop body, shared by the
    /// suspend-parked and event-parked loops): resume the fiber's frames as a standalone
    /// sub-run under `UNWINDING` — the first poll fires with zero forward progress — so its
    /// continuation spills into *its own* shadow region, and record the [`FrozenFiber`]
    /// residue. `placeholder` is the inert resume/status value delivered when the park's
    /// point excludes it from the spill (`None` when the frames already carry a real
    /// delivered result — a woken event-park).
    fn flatten_fiber_for_freeze(
        &mut self,
        slot: usize,
        mut frames: Vec<Frame>,
        placeholder: Option<Reg>,
    ) -> Result<(), Trap> {
        // The entry funcref (== func index) + data-stack base, to re-enter the fiber on thaw.
        let func = frames.first().map(|f| f.func as i32).unwrap_or(0);
        let sp = match frames.first().and_then(|f| f.vals.first()) {
            Some(r) => r.i64(),
            _ => 0,
        };
        if let Some(p) = placeholder {
            if let Some(f) = frames.last_mut() {
                f.vals.push(p);
            }
        }
        // The fiber spills into *its* region: its SP word starts empty (frame base) and
        // `durable.shadow_base` must resolve to its region during the unwind sub-run (where `cur`
        // is the `ROOT_FIBER` sentinel), so set the active spill context explicitly.
        let fctx = shadow_context_index(slot);
        self.durable_sp_ctx = fctx;
        if let Some(m) = self.mem.as_mut() {
            m.durable_set_sp(shadow_region_base(fctx), shadow_frame_base(fctx));
        }
        self.frames = frames;
        self.cur = ROOT_FIBER;
        self.chain = vec![ROOT_FIBER];
        self.root_parked = None;
        self.parked_frames = 0;
        run_inner(self, u64::MAX)?; // the fiber unwinds; base return (cur == ROOT) ends the sub-run
        let shadow_sp = self
            .mem
            .as_ref()
            .map(|m| m.durable_get_sp(shadow_region_base(fctx)))
            .unwrap_or_else(|| shadow_frame_base(fctx));
        self.registry.set_saved_sp(slot, shadow_sp);
        self.frozen.push(FrozenFiber {
            slot,
            func,
            sp,
            shadow_sp,
            generation: self.registry.generation(slot),
        });
        Ok(())
    }
}

/// The shadow-context index of a fiber registry `slot` (the root is context 0, fiber slot `s` is
/// context `s + 1`). Mirrors [`shadow_switch`]'s mapping for the off-table root vs. a fiber.
fn shadow_context_index(slot: usize) -> usize {
    slot + 1
}

/// A **visible** instruction — one whose effect another vCPU can observe or that synchronizes with
/// one: a linear-memory access (atomic or plain) or a thread/futex op. These are the only points at
/// which interleaving order can change a program's outcome, so they are the scheduling decision
/// points the exhaustive model checker preempts on (`memop` granularity). Pure thread-local
/// computation (arithmetic, control flow, calls) is invisible and runs without a yield. `atomic.fence`
/// is omitted: both backends execute seq-cst, so a fence moves no data and adds no observable order.
fn is_visible(inst: &Inst) -> bool {
    matches!(
        inst,
        Inst::Load { .. }
            | Inst::Store { .. }
            | Inst::AtomicLoad { .. }
            | Inst::AtomicStore { .. }
            | Inst::AtomicRmw { .. }
            | Inst::AtomicCmpxchg { .. }
            | Inst::ThreadSpawn { .. }
            | Inst::ThreadJoin { .. }
            | Inst::MemoryWait { .. }
            | Inst::MemoryNotify { .. }
            // §12 fiber ops operate on the **run-shared** fiber registry (D57 3b-i), so another
            // vCPU can observe them (a racing `cont.resume` decides a winner; `cont.new` order
            // decides handle values) — they are scheduling decision points like memory ops.
            | Inst::ContNew { .. }
            | Inst::ContResume { .. }
            | Inst::Suspend { .. }
            // §GC `gc.roots` reads every fiber's registry frames and writes guest memory, so it is
            // observable to another vCPU — a scheduling decision point like the fiber ops.
            | Inst::GcRoots { .. }
    )
}

/// Drive a vCPU until it finishes (`Inner::Done`) or parks on a blocking op (`Inner::Park`). On
/// re-entry it first completes the parked op recorded in `pending`. The owned `funcs` is borrowed
/// locally as `fs` so the loop can mutate the other fields.
/// Finish a memory-op result in the eval loop. Pushes the loaded value (if any). For a
/// coroutine child's *recoverable* in-window page fault (`fault_yields`), it rewinds the op and
/// returns `Some(Inner::CoFault)` so the loop suspends to the parent (which supplies the page)
/// instead of trapping; any other fault propagates. `Ok(None)` means "handled, keep going". This
/// is the one home for the §14 fault-driven-yield decision, shared by the fast-pathed memory ops
/// and the `eval_inst` fallback so the logic isn't repeated per arm.
#[inline]
fn handle_mem(
    r: Result<Option<Reg>, Trap>,
    frame: &mut Frame,
    fault_yields: bool,
    mem: &Option<Mem>,
) -> Result<Option<Inner>, Trap> {
    match r {
        Ok(Some(v)) => {
            frame.vals.push(v);
            Ok(None)
        }
        Ok(None) => Ok(None),
        Err(Trap::MemoryFault) if fault_yields => match mem.as_ref().and_then(|m| m.take_fault()) {
            Some(addr) => {
                frame.inst -= 1; // re-execute the access on resume
                Ok(Some(Inner::CoFault(addr)))
            }
            None => Err(Trap::MemoryFault),
        },
        Err(t) => Err(t),
    }
}

fn run_inner(v: &mut VCpu, quantum: u64) -> Result<Inner, Trap> {
    let mut budget = quantum; // instructions left before a forced `Yield` (deterministic explorer)
                              // A timed `svc.wait`'s deadline fired (I38): consumed by the rewound serve arm below, which
                              // returns its count instead of re-parking. Only ever set in the same `run_inner` call that
                              // re-executes the arm (the pending is taken at resume, the rewound op runs first).
    let mut svc_timed_out = false;
    // Resuming from a park: finish the op the scheduler woke us for.
    match v.pending.take() {
        Some(Pending::Join { slot }) => {
            let child = v
                .threads
                .get(slot)
                .and_then(|x| *x)
                .ok_or(Trap::ThreadFault)?;
            v.threads[slot] = None; // a handle is joined once
            let out = v.sched.take_result(child).ok_or(Trap::Malformed)?;
            let vals = out.result?; // a child trap propagates as this vCPU's trap
            let top = v.frames.len() - 1;
            v.frames[top].vals.push(Reg::from_value(
                vals.first().copied().unwrap_or(Value::I64(0)),
            ));
        }
        Some(Pending::Wait(status)) => {
            let top = v.frames.len() - 1;
            v.frames[top].vals.push(Reg::from_i32(status));
        }
        Some(Pending::CoResume(value)) => {
            // The parent's `resume` delivered `value` — push it as the child `Yielder` cap.call's
            // result so the coroutine continues past its `yield`.
            let top = v.frames.len() - 1;
            v.frames[top].vals.push(Reg::from_i64(value));
        }
        Some(Pending::CapResult(status)) => {
            // §3.6 revocation-unparks: the handle this fiber's capability call was parked
            // through was revoked. The call completes with the negative errno — the fiber
            // resumes on its own error path (no trap, no kill; cancellation is a value).
            let top = v.frames.len() - 1;
            v.frames[top].vals.push(Reg::from_i64(status));
        }
        Some(Pending::SvcTimeout) => svc_timed_out = true,
        None => {}
    }

    let funcs = Arc::clone(&v.funcs); // module 0 (this vCPU's primary program), immutable for the run
    let spawn_quota = v.quota; // §15 fiber/vCPU ceilings (distinct from the Instantiator's i64 fuel quota)
                               // `dt` is the **shared** domain table (atomic slots + the writer-locked installed
                               // units); reads off it are lock-free. `units` here is this vCPU's **local clone** of
                               // the installed-units prefix, refreshed lazily on a miss (`resolve_module`) so neither
                               // a running unit frame nor the `Jit.install` arm needs the shared lock on the hot loop.
    let VCpu {
        registry,
        chain,
        cur,
        frames,
        root_parked,
        parked_frames,
        durable,
        root_shadow_sp,
        durable_sp_ctx, // §12.8 4A.5: active spill context, maintained at fiber switches
        frozen, // freeze-unwind of an active-chain fiber pushes here (slice 3.2); also `freeze_drive`
        spawn_residue: _,
        vcpu_ctx,  // §12.8 4A.5: this vCPU's root shadow context, for `durable.shadow_base`
        dstate: _, // swapped at the dispatch boundary, not inside `run_inner`
        mem,
        host,
        freeze_sink, // §4: the effective sink a §14 child inherits so its residue reaches the root host
        fuel,
        threads,
        nested_children,
        child_hosts,
        nested_child: _,
        coroutines,
        fault_yields,
        depth,
        pending,
        sched,
        funcs: _,
        id,
        parent_task: _, // a spawned child stamps *its own* id as the grandchild's parent below
        tls,
        memop,
        acc,
        quota: _,
        setjmp_points,
        dt,
        units,
        invoked,
        debug,
        kill,
        child_kill,
        serve_run,
        handler_parks,
        serve_count,
    } = v;
    let depth = *depth;
    let durable = *durable;
    let memop = *memop;
    let fault_yields = *fault_yields;

    // Reusable scratch for branch edge-args (block parameters). Each taken edge gathers its
    // args here and swaps the buffer into the frame's value slot, so steady-state branching —
    // notably a loop back-edge, run every iteration — allocates nothing (the displaced buffer
    // becomes the next edge's scratch). Lives across the whole `run_inner` call, so the two
    // buffers ping-pong for free once warmed.
    let mut edge_buf: Vec<Reg> = Vec::new();
    // Reusable scratch for `return` results. The common case (a callee returning to a caller in
    // the same fiber) gathers results here and copies them straight into the caller's value
    // file, so a call/return pair no longer allocates a results `Vec` per return. The rarer
    // root/fiber exits read out of the same buffer.
    let mut ret_buf: Vec<Reg> = Vec::new();

    // Resolve the running frame against *its* module (module 0 = this vCPU's program;
    // `INVOKE_MODULE` = the invoked unit; ≥ 1 = an installed Jit unit), cloning the module's `Arc`
    // into a local so the shared `dt` is never borrowed across the loop body. This is **cached
    // across loop iterations**: re-resolving (an atomic `Arc` refcount bump) on every block entry
    // showed up on the hot path — a branch / loop back-edge re-enters this loop, and almost always
    // stays in the same module — so we only re-resolve when `module` actually changes. A module's
    // code never mutates in place (units are append-only; module 0 is fixed), so a same-id reuse is
    // sound.
    let mut cur_module = frames.last().map(|f| f.module).unwrap_or(0);
    let mut cur_funcs: Arc<[Func]> =
        resolve_module(&funcs, units, invoked, dt, cur_module).ok_or(Trap::Malformed)?;

    // Drive the running fiber's top frame. A `call` pushes a new top and restarts here; a
    // `return` pops and appends results to the caller (which resumes past the call); a tail
    // call replaces the top in place (O(1) frames). `cont.resume`/`suspend` switch which
    // fiber's stack is in `frames` — see the comments on those arms.
    'frames: loop {
        let top = frames.len() - 1;
        if frames[top].module != cur_module {
            cur_module = frames[top].module;
            cur_funcs =
                resolve_module(&funcs, units, invoked, dt, cur_module).ok_or(Trap::Malformed)?;
        }
        let fs: &[Func] = &cur_funcs;
        let block = fs
            .get(frames[top].func as usize)
            .ok_or(Trap::Malformed)?
            .blocks
            .get(frames[top].block)
            .ok_or(Trap::Malformed)?;

        // Execute the remaining instructions of this block.
        while frames[top].inst < block.insts.len() {
            // An op counts against the scheduling quantum when it is "visible": every op in the real
            // executor / single-threaded debugger (`!memop`), or only a **shared-state / sync** op in
            // `memop` mode (so thread-local computation runs to the next memory op before a yield is
            // possible — the partial-order reduction that keeps exhaustive exploration tractable).
            let visible = !memop || is_visible(&block.insts[frames[top].inst]);
            // Deterministic-explorer preemption: yield at an instruction boundary (state consistent;
            // `inst` not yet advanced) when the quantum is spent. The real pool passes `u64::MAX`, so
            // it never yields. **This precedes the debug seam below**: a debug stop (breakpoint /
            // watch / step) at a budget-exhausted visible op must fire at the *start of its own turn*
            // (budget fresh, on the next pick), not inside the previous turn — otherwise a stop would
            // run the op in the prior turn, collapsing two one-visible-op turns into one and desyncing
            // a fixed replay plan (DEBUGGING.md Milestone B). Undebugged runs are unaffected (the
            // ordering only matters when something stops).
            if visible && budget == 0 {
                return Ok(Inner::Yield);
            }
            // Debug seam (DEBUGGING.md W2/S4): before each op, consult the inspector's probe. A
            // breakpoint/step hit returns `Inner::Pause` with the op not yet advanced, so the
            // continuation is intact and the next `run` resumes exactly here. Gated on `debug`
            // being `Some`, so an undebugged run is unaffected (S7). The `clock` it maintains is
            // the S3 logical-time coordinate (ops executed on this vCPU).
            if let Some(dbg) = debug.as_mut() {
                let pc = IrPc {
                    module: frames[top].module,
                    func: frames[top].func,
                    block: frames[top].block,
                    inst: frames[top].inst,
                };
                // Watchpoints reuse `access_of` — the same confined-range analysis the DPOR
                // explorer uses — but only when armed (it confines, so it isn't free). Breakpoints,
                // stepping, and the cap.call stop need no memory analysis.
                let inst = &block.insts[frames[top].inst];
                let access = if dbg.watches_armed() {
                    access_of(inst, &frames[top].vals, &*mem)
                } else {
                    MemAccess::None
                };
                if let Some(reason) = dbg.before_op(pc, inst, access, frames.len()) {
                    return Ok(Inner::Pause(reason, pc));
                }
            }
            // Charge the quantum for the visible op about to run, recording the object it touches (for
            // DPOR's race analysis) first — the confined address is a pure function of the live SSA
            // values here.
            if visible {
                if memop {
                    *acc = Some(access_of(
                        &block.insts[frames[top].inst],
                        &frames[top].vals,
                        mem,
                    ));
                }
                budget -= 1;
            }
            let inst = &block.insts[frames[top].inst];
            step(fuel, kill.as_deref())?;
            frames[top].inst += 1; // advance first, so a call-return resumes past this inst

            // Mid-run freeze trigger (DURABILITY.md §12, "freeze after N safepoints"): on a durable run
            // armed via `STATE_ARMED`, count down at each **fiber safepoint** (`cont.resume`/`suspend`)
            // and promote to `UNWINDING` at 0, so *this* op's trailing poll begins the freeze. Inert
            // unless armed; gated on `durable` so an ordinary run is untouched. Lets a run freeze after
            // forward progress (e.g. once a fiber is recycled), which the freeze-before-start harness
            // cannot reach. Counting only the fiber ops (routed through runtime thunks on both backends)
            // keeps the trigger point identical interp↔JIT — cap.call is *not* counted (the JIT's
            // cap.call thunk is host-supplied, so there is no cross-backend choke for it); a cap.call
            // freeze is already reachable at the first safepoint, and a production async trigger handles
            // general mid-run freeze.
            if durable && matches!(inst, Inst::ContResume { .. } | Inst::Suspend { .. }) {
                if let Some(m) = mem.as_mut() {
                    m.durable_tick_arm();
                }
            }

            // The three guest-driven `Jit` ops serviced in the eval loop (not the generic
            // host dispatch) are shared between their two dispatch forms — the resolved
            // `cap.call` and the phase-3 executable `call.import` routed through the
            // instance binding — as local macros, so the special semantics exist once.
            // §3.6 slice 5a — fiber-level park routing (DESIGN.md: "blocks the fiber, never
            // the domain"). When a park happens while a FIBER runs (and the real M:N
            // scheduler is driving — the deterministic explorer keeps whole-vCPU parks so
            // interleavings stay explorable), the fiber's frames are set aside as an event
            // park, a fiber-keyed waiter is registered and the event re-checked (a race that
            // already fired wakes the fiber immediately; a stale waiter entry is an
            // idempotent no-op later), and control unwinds one chain link to the resumer
            // with `(FIBER_PARKED, 0)` — exactly a `suspend` the guest didn't write. The
            // vCPU keeps running; it idles only when nothing in its chain is runnable.
            macro_rules! fiber_park {
                ($register_and_recheck:expr) => {{
                    let leaving = *cur;
                    registry.park_blocked(leaving, std::mem::take(frames));
                    ($register_and_recheck)(leaving);
                    chain.pop();
                    *cur = *chain.last().expect("chain keeps the root");
                    shadow_switch(
                        mem,
                        registry,
                        root_shadow_sp,
                        *vcpu_ctx,
                        durable_sp_ctx,
                        durable,
                        leaving,
                        *cur,
                    );
                    *frames = if *cur == ROOT_FIBER {
                        root_parked.take().ok_or(Trap::Malformed)?
                    } else {
                        registry.unpark_resumer(*cur)?
                    };
                    *parked_frames -= frames.len();
                    let rtop = frames.len() - 1;
                    frames[rtop].vals.push(Reg::from_i32(FIBER_PARKED));
                    frames[rtop].vals.push(Reg::from_i64(0));
                    continue 'frames;
                }};
            }
            macro_rules! jit_install_body {
                ($h:expr, $args:expr) => {{
                    let ch =
                        get(&frames[top].vals, *$args.first().ok_or(Trap::Malformed)?)?.i64() as i32;
                    let unit_funcs = {
                        let hg = host.lock().unwrap_or_else(|e| e.into_inner());
                        let domain = hg.resolve_jit_domain($h)?;
                        let (cd, cu) = hg.resolve_jit_code(ch)?;
                        if cd != domain {
                            return Err(Trap::CapFault);
                        }
                        hg.jit_unit_funcs(cd, cu).ok_or(Trap::CapFault)?
                    };
                    // Append the unit to the **shared** domain table (module id = its 1-based
                    // index) and fill the next empty slot — visible at once to every vCPU of the
                    // domain (DESIGN.md §22). The padding starts at `funcs.len()` on both backends,
                    // so the first install lands at the same index the JIT's `install` returns.
                    let res = match dt.install(unit_funcs) {
                        Some(slot) => slot as i64,
                        None => ENOSPC,
                    };
                    frames[top].vals.push(Reg::from_i64(res));
                }};
            }
            macro_rules! jit_uninstall_body {
                ($h:expr, $args:expr) => {{
                    {
                        let hg = host.lock().unwrap_or_else(|e| e.into_inner());
                        hg.resolve_jit_domain($h)?; // authority: a forged handle is inert
                    }
                    let slot = get(&frames[top].vals, *$args.first().ok_or(Trap::Malformed)?)?.i64()
                        as usize;
                    // A guest may only clear slots it installed (`≥ funcs.len()`, the module-0
                    // function count) — `dt.uninstall` enforces the range + filled checks.
                    let res = if dt.uninstall(slot, funcs.len()) {
                        0
                    } else {
                        EINVAL
                    };
                    frames[top].vals.push(Reg::from_i64(res));
                }};
            }
            macro_rules! jit_invoke_body {
                ($h:expr, $args:expr, $sig:expr) => {{
                    // arg0 = the CompiledCode handle; the rest are the invoke args. Args cross in
                    // the i64-slot ABI (the handle rides the low 32 bits of its slot, like every
                    // handle-as-arg — e.g. the Instantiator's module ops), so read it as a slot.
                    let ch =
                        get(&frames[top].vals, *$args.first().ok_or(Trap::Malformed)?)?.i64() as i32;
                    let unit_funcs = {
                        let hg = host.lock().unwrap_or_else(|e| e.into_inner());
                        let domain = hg.resolve_jit_domain($h)?;
                        let (cd, cu) = hg.resolve_jit_code(ch)?;
                        // A code handle is only valid on the domain that compiled it.
                        if cd != domain {
                            return Err(Trap::CapFault);
                        }
                        hg.jit_unit_funcs(cd, cu).ok_or(Trap::CapFault)?
                    };
                    let entry = unit_funcs.first().ok_or(Trap::CapFault)?;
                    // Strict arity: the cap.call's declared signature must match the unit entry's
                    // (minus the code-handle arg) — fail-closed, identically on the JIT path.
                    if $sig.params.len() != entry.params.len() + 1
                        || $sig.results.len() != entry.results.len()
                    {
                        return Err(Trap::CapFault);
                    }
                    // Marshal the invoke args through the i64-slot ABI (value → slot → entry-typed),
                    // exactly the JIT trampoline's decode.
                    let mut child_args = Vec::with_capacity(entry.params.len());
                    for (a, ty) in $args[1..].iter().zip(entry.params.clone()) {
                        let slot = get(&frames[top].vals, *a)?.i64();
                        child_args.push(slot_to_val(ty, slot));
                    }
                    // Nested run over the SAME window/fuel/powerbox: move them into an
                    // inline-driven child vCPU (like a §14 coroutine, but sharing the window) and
                    // move them back whatever the outcome — the parent's snapshot/teardown still
                    // needs them after a trap.
                    let child_mem = mem.take();
                    // The unit runs over the parent's world: module 0 = this vCPU's program and the
                    // **shared, live** domain table, so the unit's `call_indirect` reaches the
                    // original program (new→old) and any installed units (new→new, incl. one it
                    // installs during its own invocation), matching the JIT's invoked code over the
                    // live `fn_table`. A nested invoke costs call depth like a call, so invoke
                    // recursion is bounded by the same stack-overflow bound as ordinary recursion.
                    let mut child = VCpu::new_invoke(
                        Arc::clone(&funcs),
                        Arc::clone(dt),
                        unit_funcs,
                        &child_args,
                        child_mem,
                        Arc::clone(host),
                        *fuel,
                        depth + frames.len() as u32 + 1,
                        sched.clone(),
                        spawn_quota,
                    );
                    child.memop = memop;
                    let out = run_inner(&mut child, u64::MAX);
                    *mem = child.mem.take();
                    *fuel = child.fuel;
                    match out {
                        Ok(Inner::Done(results)) => {
                            // Results cross back as slots (arity already checked equal).
                            for (v, ty) in results.iter().zip(&$sig.results) {
                                frames[top]
                                    .vals
                                    .push(Reg::from_value(slot_to_val(*ty, val_to_slot(*v))));
                            }
                        }
                        // The unit cannot park/yield (concurrency is rejected at compile);
                        // defensive fail-closed.
                        Ok(_) => return Err(Trap::CapFault),
                        // A trap in invoked code is terminal for the domain (DESIGN.md §22),
                        // matching the JIT's trap-cell propagation.
                        Err(t) => return Err(t),
                    }
                }};
            }
            match inst {
                // Non-tail calls push a new frame and switch to it; the callee's results
                // are appended to this frame's `vals` when it returns (see `Return`).
                Inst::Call { func, args } => {
                    let argv = collect(&frames[top].vals, args)?;
                    if fs.get(*func as usize).is_none() {
                        return Err(Trap::Malformed);
                    }
                    if depth as usize + *parked_frames + frames.len() > MAX_CALL_DEPTH as usize {
                        return Err(Trap::StackOverflow);
                    }
                    frames.push(Frame {
                        func: *func,
                        module: frames[top].module, // a direct call stays in the caller's module
                        block: 0,
                        inst: 0,
                        vals: argv,
                    });
                    continue 'frames;
                }
                Inst::CallIndirect { ty, idx, args } => {
                    // Dispatch through the module-0 table (new→old; matches the JIT, where every
                    // function — parent or unit — is lowered against the parent `fn_table`).
                    let (cmod, cfunc) = dispatch_indirect(
                        dt,
                        &funcs,
                        units,
                        invoked,
                        get_i32(&frames[top].vals, *idx)?,
                        ty,
                    )?;
                    let argv = collect(&frames[top].vals, args)?;
                    if depth as usize + *parked_frames + frames.len() > MAX_CALL_DEPTH as usize {
                        return Err(Trap::StackOverflow);
                    }
                    frames.push(Frame {
                        func: cfunc,
                        module: cmod,
                        block: 0,
                        inst: 0,
                        vals: argv,
                    });
                    continue 'frames;
                }
                // §14 `Instantiator` (iface 6): serviced here, not in the generic host dispatch, because
                // `instantiate` spawns a child vCPU and `join` parks — both need the executor. The
                // handle still gates authority (resolve as Instantiator → its carve range `[base,
                // base+size)`); a forged/wrong-type handle is an inert `CapFault`.
                // §14 co-fiber `yield` (iface 7): suspend this (coroutine) child, handing `value` to
                // the instantiator-parent's `resume`. Serviced here — it must yield the running
                // continuation, which the generic dispatch can't. The cap.call's result (the resumed
                // value) is delivered on the next `resume` via `Pending::CoResume`; the inst pointer is
                // already advanced, so we return `CoYield` without pushing a result.
                Inst::CapCall {
                    type_id: cap_id::YIELDER,
                    op,
                    handle,
                    args,
                    ..
                } => {
                    if *op != 0 {
                        return Err(Trap::CapFault);
                    }
                    let h = get_i32(&frames[top].vals, *handle)?;
                    {
                        let hg = host.lock().unwrap_or_else(|e| e.into_inner());
                        hg.resolve_yielder(h)?; // authority: a forged/wrong handle is inert
                    }
                    let value = get_i64(&frames[top].vals, *args.first().ok_or(Trap::Malformed)?)?;
                    return Ok(Inner::CoYield(value));
                }
                // §13/§14 cross-domain **`grant`** (SharedRegion op 4): install this region — the
                // *same* shared backing — into a suspended coroutine child's powerbox, returning the
                // handle the **child** will use. Serviced here (the generic dispatch can't reach the
                // coroutine table). The parent delivers the returned handle to the child by existing
                // means (typically the next `resume`'s value); the child `map`s the region into its
                // own window — the zero-copy cross-domain data plane (§13). Executor (`instantiate`)
                // children and the JIT parent are follow-ups; a forged region handle or an
                // unknown/finished child is an inert `CapFault`.
                Inst::CapCall {
                    type_id: cap_id::SHARED_REGION,
                    op: 4,
                    handle,
                    args,
                    ..
                } => {
                    let h = get_i32(&frames[top].vals, *handle)?;
                    let backing = {
                        let hg = host.lock().unwrap_or_else(|e| e.into_inner());
                        hg.resolve_region(h)?
                    };
                    let ch = get_i32(&frames[top].vals, *args.first().ok_or(Trap::Malformed)?)?;
                    let coro = coroutines
                        .get_mut(ch as usize)
                        .and_then(|c| c.as_mut())
                        .ok_or(Trap::CapFault)?;
                    // Install the region into the child's powerbox. Guest-minting into the *child*
                    // table, so a full table yields -EMFILE rather than panicking (§3c / audit #1).
                    let child_handle = {
                        let mut chh = coro.vcpu.host.lock().unwrap_or_else(|e| e.into_inner());
                        chh.try_grant_shared_region_backed(backing)
                    };
                    frames[top]
                        .vals
                        .push(Reg::from_i64(child_handle.map_or(EMFILE, |h| h as i64)));
                }
                Inst::CapCall {
                    type_id: cap_id::INSTANTIATOR,
                    op,
                    handle,
                    args,
                    ..
                } => {
                    let h = get_i32(&frames[top].vals, *handle)?;
                    let (ibase, isize) = {
                        let hg = host.lock().unwrap_or_else(|e| e.into_inner());
                        hg.resolve_instantiator(h)?
                    };
                    // §14 **separate-module children** (ops 5/6/7 = `instantiate_module` /
                    // `spawn_coroutine_module` / `spawn_demand_coroutine_module`): exactly ops 0/2/4,
                    // except the first arg is a host-granted `Module` handle (iface 8) and the child
                    // domain runs *that* verified module — the "plugin-in-plugin" story (a guest can
                    // only instantiate modules it was given). Resolve the grant here, shift the
                    // remaining args by one, and fold into the shared op logic below; `join`/`resume`
                    // (ops 1/3) serve both kinds unchanged. A forged module handle is a `CapFault`.
                    #[allow(clippy::type_complexity)]
                    // `grant`/`named` carry the re-grant **handles** (not pre-resolved bindings): a pipe
                    // end must alias its shared backing into the child, not copy a parent-local index, so
                    // resolution happens at child construction via `regrant_into_child`. Validated here.
                    let (op, child_mod, askip, grant, named): (
                        u32,
                        Option<ChildMod>,
                        usize,
                        Option<i32>,
                        Vec<(String, i32)>,
                    ) = match *op {
                        mop @ 5..=7 => {
                            // The module handle crosses as an i64 arg (the slot ABI); low 32 bits.
                            let mh =
                                get_i64(&frames[top].vals, *args.first().ok_or(Trap::Malformed)?)?
                                    as i32;
                            let g = {
                                let hg = host.lock().unwrap_or_else(|e| e.into_inner());
                                let g = hg.resolve_module(mh)?;
                                ChildMod {
                                    funcs: g.funcs.clone(),
                                    memory_log2: g.memory_log2,
                                    data: g.data.clone(),
                                    durable: g.durable,
                                    digest: g.digest,
                                    imports: g.imports.clone(),
                                    types: g.types.clone(),
                                    module: Arc::clone(&g.module),
                                }
                            };
                            (
                                match mop {
                                    5 => 0, // instantiate_module → instantiate
                                    6 => 2, // spawn_coroutine_module → spawn_coroutine
                                    _ => 4, // spawn_demand_coroutine_module → spawn_demand_coroutine
                                },
                                Some(g),
                                1,
                                None,
                                Vec::new(),
                            )
                        }
                        // §14 `instantiate_granted(grant_handle, entry, off, size_log2, quota)`
                        // (PROCESS.md S2): exactly `instantiate` (op 0), except the first arg is a
                        // handle to one of the parent's own coordinate-free capabilities (`Stream` /
                        // `Exit` / `Clock`) that is **re-granted into the child's powerbox**, so a
                        // child can do I/O instead of being born destitute. The child receives its
                        // handle as a **third** entry arg (after `Instantiator`, `AddressSpace`); a
                        // forged / non-copyable handle is a `CapFault`.
                        8 => {
                            let gh =
                                get_i64(&frames[top].vals, *args.first().ok_or(Trap::Malformed)?)?
                                    as i32;
                            // Validate the grant is re-grantable now (coordinate-free cap or pipe end);
                            // a forged / non-grantable handle fails the spawn closed.
                            {
                                let hg = host.lock().unwrap_or_else(|e| e.into_inner());
                                hg.can_regrant(gh).then_some(()).ok_or(Trap::CapFault)?;
                            }
                            (0, None, 1, Some(gh), Vec::new())
                        }
                        // §14 `instantiate_named(grants_ptr, grants_n, entry, off, size_log2, quota)`
                        // (PROCESS.md S2): `instantiate` (op 0) plus a **grant list** — `grants_n`
                        // 16-byte records `{name_off: u32, name_len: u32, handle: i32, flags: u32}` at
                        // window-relative `grants_ptr`. Each record's `handle` (one of the parent's own
                        // coordinate-free caps) is re-granted into the child's powerbox **under its
                        // name**, so the child discovers it by `cap.self.resolve(name)` — the general
                        // multi-cap, name-based form of op 8 (no fixed arg-slot coupling). A forged /
                        // non-copyable handle, an out-of-window record/name, or non-UTF-8 name fails the
                        // whole spawn closed (`CapFault` / `MemoryFault`); `flags` is reserved-zero.
                        11 => {
                            let grants_ptr =
                                get_i64(&frames[top].vals, *args.first().ok_or(Trap::Malformed)?)?
                                    as u64;
                            let grants_n =
                                get_i64(&frames[top].vals, *args.get(1).ok_or(Trap::Malformed)?)?
                                    as u64;
                            let m = mem.as_ref().ok_or(Trap::Malformed)?;
                            let mut list: Vec<(String, i32)> = Vec::new();
                            for i in 0..grants_n {
                                let rec = m.read_window(grants_ptr + i * 16, 16)?;
                                let name_off =
                                    u32::from_le_bytes([rec[0], rec[1], rec[2], rec[3]]) as u64;
                                let name_len =
                                    u32::from_le_bytes([rec[4], rec[5], rec[6], rec[7]]) as usize;
                                let handle = i32::from_le_bytes([rec[8], rec[9], rec[10], rec[11]]);
                                let name_bytes = m.read_window(name_off, name_len)?;
                                let name =
                                    String::from_utf8(name_bytes).map_err(|_| Trap::CapFault)?;
                                {
                                    let hg = host.lock().unwrap_or_else(|e| e.into_inner());
                                    hg.can_regrant(handle).then_some(()).ok_or(Trap::CapFault)?;
                                }
                                list.push((name, handle));
                            }
                            (0, None, 2, None, list)
                        }
                        // §14 `instantiate_module_named(module, grants_ptr, grants_n, entry, off,
                        // size_log2, quota)` (STAGE1.md — the shell "exec" primitive): the union of op 5
                        // (run a host-granted **separate `Module`**) and op 11 (a **named grant list**
                        // re-granted into the child's powerbox). It is the only op that runs a foreign
                        // program *and* hands it capabilities — so a compiled command (its own module)
                        // can resolve an inherited `stdout` by name and do real I/O, not just return a
                        // status. `module` is arg 0, the grant list at args 1/2, the carve args follow
                        // (`askip = 3`). Forged module / non-copyable grant / bad record fail closed,
                        // exactly as ops 5 and 11 do individually.
                        13 => {
                            let mh =
                                get_i64(&frames[top].vals, *args.first().ok_or(Trap::Malformed)?)?
                                    as i32;
                            let g = {
                                let hg = host.lock().unwrap_or_else(|e| e.into_inner());
                                let g = hg.resolve_module(mh)?;
                                ChildMod {
                                    funcs: g.funcs.clone(),
                                    memory_log2: g.memory_log2,
                                    data: g.data.clone(),
                                    durable: g.durable,
                                    digest: g.digest,
                                    imports: g.imports.clone(),
                                    types: g.types.clone(),
                                    module: Arc::clone(&g.module),
                                }
                            };
                            let grants_ptr =
                                get_i64(&frames[top].vals, *args.get(1).ok_or(Trap::Malformed)?)?
                                    as u64;
                            let grants_n =
                                get_i64(&frames[top].vals, *args.get(2).ok_or(Trap::Malformed)?)?
                                    as u64;
                            let m = mem.as_ref().ok_or(Trap::Malformed)?;
                            let mut list: Vec<(String, i32)> = Vec::new();
                            for i in 0..grants_n {
                                let rec = m.read_window(grants_ptr + i * 16, 16)?;
                                let name_off =
                                    u32::from_le_bytes([rec[0], rec[1], rec[2], rec[3]]) as u64;
                                let name_len =
                                    u32::from_le_bytes([rec[4], rec[5], rec[6], rec[7]]) as usize;
                                let handle = i32::from_le_bytes([rec[8], rec[9], rec[10], rec[11]]);
                                let name_bytes = m.read_window(name_off, name_len)?;
                                let name =
                                    String::from_utf8(name_bytes).map_err(|_| Trap::CapFault)?;
                                {
                                    let hg = host.lock().unwrap_or_else(|e| e.into_inner());
                                    hg.can_regrant(handle).then_some(()).ok_or(Trap::CapFault)?;
                                }
                                list.push((name, handle));
                            }
                            (0, Some(g), 3, None, list)
                        }
                        o => (o, None, 0, None, Vec::new()),
                    };
                    // The function table the child's `entry` indexes — its own module's, or ours.
                    let cfs: &[Func] = child_mod.as_ref().map_or(fs, |cm| &cm.funcs);
                    match op {
                        // instantiate(entry, off, size_log2, fuel) -> child handle (or -EINVAL). The
                        // §14 data plane is shared memory: the parent seeds the child's sub-window
                        // directly (it sees the superset) before/after, so there is no scalar arg.
                        0 => {
                            let argn = |i: usize| -> Result<i64, Trap> {
                                Ok(get(
                                    &frames[top].vals,
                                    *args.get(i + askip).ok_or(Trap::Malformed)?,
                                )?
                                .i64())
                            };
                            let entry = argn(0)? as u64;
                            let off = argn(1)? as u64;
                            let size_log2 = argn(2)?;
                            let quota = argn(3)?;
                            // The child entry returns one `i64` and takes its starter capabilities as
                            // `i64` args, in order: `Instantiator`, then (if 2+ params) `AddressSpace`,
                            // then (S2 `instantiate_granted`, 3 params) the re-granted `Stream`/`Exit`/
                            // `Clock`. A missing/mistyped entry is rejected, not run. When a grant is
                            // supplied the entry **must** be the 3-arg form (so the child actually
                            // receives the handle); without one, 1- or 2-arg as before.
                            let want_as =
                                cfs.get(entry as usize).is_some_and(|f| f.params.len() >= 2);
                            let ok_entry = cfs.get(entry as usize).is_some_and(|f| {
                                f.results == [ValType::I64]
                                    && f.params.iter().all(|p| *p == ValType::I64)
                                    && if grant.is_some() {
                                        f.params.len() == 3
                                    } else {
                                        f.params.len() == 1 || f.params.len() == 2
                                    }
                            });
                            // The carve must be a power-of-two-aligned sub-window within `[0, isize)`
                            // — a child can only get what the holder sub-allocates (§14/D19). A
                            // separate-module child's carve must **equal its declared memory** (§14
                            // transparency: the plugin runs exactly as it would standalone — same
                            // window size, same wrap behaviour; a module with no memory can't nest).
                            let child_size = if (0..64).contains(&size_log2) {
                                1u64 << size_log2
                            } else {
                                0
                            };
                            let mod_ok = child_mod
                                .as_ref()
                                .is_none_or(|cm| cm.memory_log2 == Some(size_log2 as u8));
                            let fits = child_size != 0
                                && child_size <= isize
                                && off & (child_size - 1) == 0
                                && off.checked_add(child_size).is_some_and(|e| e <= isize);
                            // §4 enforcement: *a durable domain admits only freezable modules* — a
                            // separate-module child must carry the grant's durable attestation
                            // (`grant_durable_module`); an un-instrumented child could never
                            // drain-then-unwind, so admitting it would stop the subtree being
                            // snapshottable as a unit. A same-module child (`None`) runs the
                            // parent's own (already instrumented) funcs — always admissible.
                            let mod_durable_ok =
                                !durable || child_mod.as_ref().is_none_or(|cm| cm.durable);
                            if !ok_entry || !fits || !mod_ok || !mod_durable_ok {
                                frames[top].vals.push(Reg::from_i32(EINVAL as i32));
                            } else {
                                // `ibase`/`off` are holder-relative; the backing-absolute base
                                // adds the holder's own window base (0 for a top-level holder), so
                                // nesting composes at any depth.
                                let abs_base =
                                    mem.as_ref().map_or(0, |m| m.window.base()) + ibase + off;
                                let child_mem = mem
                                    .as_ref()
                                    .map(|m| m.nested_view(abs_base, size_log2 as u8));
                                // A separate-module child's **data segments** materialize into the
                                // carve at spawn (exactly as if the child wrote them; the verifier
                                // bounded them to its declared window == the carve). RO protection of
                                // `readonly` segments is skipped for nested children (documented —
                                // intra-domain self-corruption is a §1 non-goal).
                                if let (Some(cm), Some(m)) = (&child_mod, mem.as_ref()) {
                                    for d in cm.data.iter() {
                                        if d.offset.saturating_add(d.bytes.len() as u64)
                                            <= child_size
                                        {
                                            for (k, &b) in d.bytes.iter().enumerate() {
                                                m.set_byte(abs_base + d.offset + k as u64, b);
                                            }
                                        }
                                    }
                                }
                                // Attenuated powerbox: the child gets, over its *own* window (a strict
                                // subset of the parent's authority), an `Instantiator` (so it can
                                // itself nest — confinement composes to any depth) and an
                                // `AddressSpace` (so it can manage its own pages). These are its entry
                                // arguments. (Pass-through of the parent's *other* handles is a
                                // follow-up.)
                                let mut ch = Host::new();
                                // §4: *a durable domain may only spawn durable children* — the
                                // subtree freezes as a unit, so the child's own spawns/fibers must
                                // reserve shadow state like the parent's (and its own nested
                                // instantiates re-apply this same admission rule).
                                ch.set_durable(durable);
                                // §6: stamp the child's attestation — nested (its carve is a superset
                                // the parent reads), freezable iff durable, tier inherited from us.
                                let catt = {
                                    let hg = host.lock().unwrap_or_else(|e| e.into_inner());
                                    hg.child_attestation(durable)
                                };
                                ch.set_attestation(catt);
                                let cinst = ch.grant_instantiator(0, child_size);
                                let cas = ch.grant_address_space(0, child_size);
                                // S2: re-grant the parent capability into the child's fresh powerbox —
                                // a coordinate-free cap (`Stream`/`Exit`/`Clock`, a stdout/stderr stream
                                // sharing the parent's sink) or a **pipe end** (aliasing its shared FIFO,
                                // the cross-domain `cmd1 | cmd2` grant). Its handle is guest-visible data
                                // the child receives as its third entry arg. Pre-validated above, so the
                                // regrant cannot fail. `regrant_into_child` is the same helper the JIT's
                                // `spawn_granted_child` uses, so both backends stay in lockstep.
                                let cgrant = grant.and_then(|gh| {
                                    let mut hg = host.lock().unwrap_or_else(|e| e.into_inner());
                                    hg.regrant_into_child(gh, &mut ch)
                                });
                                // S2 named grant list (op 11): install each re-granted cap into the child
                                // **under its name** (so the child resolves it by `cap.self.resolve`).
                                // Empty for every other op.
                                let mut named_child: Vec<i32> = Vec::new();
                                for (name, gh) in &named {
                                    let cg = {
                                        let mut hg = host.lock().unwrap_or_else(|e| e.into_inner());
                                        hg.regrant_into_child(*gh, &mut ch)
                                    };
                                    if let Some(cg) = cg {
                                        ch.register_cap_name(name, cg);
                                        named_child.push(cg);
                                    }
                                }
                                // IMPORTS.md phase 3 / S2.1 + §3.3: bind the child module's
                                // import manifest against its granted powerbox (named offers
                                // first, then the shared reference policy — same binder the
                                // JIT's child builders call). §3.3 **withhold**: a `required`
                                // slot with nothing to bind fails the spawn closed — probeable
                                // `-EINVAL`, before any child code runs.
                                let manifest_ok = match &child_mod {
                                    Some(cm) => {
                                        ch.bind_child_manifest(&cm.imports, &cm.types).is_ok()
                                    }
                                    None => true,
                                };
                                if !manifest_ok {
                                    frames[top].vals.push(Reg::from_i32(EINVAL as i32));
                                } else {
                                    // §3.6: the child's **self module** — what its serve loop
                                    // resolves enqueue admission and handlers against. A
                                    // separate-module child serves its *own* offers; a
                                    // same-module child serves over the shared program (the
                                    // parent's registered module).
                                    ch.self_module = match &child_mod {
                                        Some(cm) => Some(Arc::clone(&cm.module)),
                                        None => host
                                            .lock()
                                            .unwrap_or_else(|e| e.into_inner())
                                            .self_module
                                            .clone(),
                                    };
                                    let child_host = Arc::new(Mutex::new(ch));
                                    // §3.6 slice 3: keep a live reference past the move into
                                    // the child vCPU, for `child_offer` (op 14).
                                    let child_host_keep = Arc::clone(&child_host);
                                    let mut child_args = vec![Value::I64(cinst as i64)];
                                    if want_as {
                                        child_args.push(Value::I64(cas as i64));
                                    }
                                    if let Some(cg) = cgrant {
                                        child_args.push(Value::I64(cg as i64));
                                    }
                                    // Quota: the child's fuel, sub-allocated from (and capped by) ours.
                                    let child_fuel = if quota <= 0 {
                                        *fuel
                                    } else {
                                        (quota as u64).min(*fuel)
                                    };
                                    let cfuncs = child_mod.as_ref().map_or_else(
                                        || Arc::clone(&funcs),
                                        |cm| Arc::clone(&cm.funcs),
                                    );
                                    let csched = sched.clone();
                                    // §4 subtree freeze (DURABILITY.md): the child's [`FrozenNested`]
                                    // residue must reach the **root** host, not the child's private
                                    // powerbox. Compute our *effective* sink — our inherited one, or our
                                    // own host if we are the root/a shared-host domain — and hand it down;
                                    // the child (and transitively any grandchild) pushes there, so a whole
                                    // nesting subtree's residue coalesces where a thaw reads it (mirrors how
                                    // `thread.spawn`'s shared host coalesces [`FrozenVCpu`]).
                                    let residue_sink =
                                        freeze_sink.clone().unwrap_or_else(|| Arc::clone(host));
                                    // §4 subtree STW (DURABILITY.md): the child inherits our **current
                                    // durable phase** (its own carve's, which is `UNWINDING` when we are
                                    // instantiating it mid-freeze), exactly as `thread.spawn` seeds
                                    // `child.dstate = child_state`. Without this the child's first
                                    // dispatch would store its default `NORMAL` over the freeze driver's
                                    // `UNWINDING` broadcast into the carve — clobbering the STW — so a
                                    // grandchild-of-the-root would never be driven to self-unwind (and
                                    // its residue would be lost). With it, a child instantiated while its
                                    // parent is unwinding is born unwinding: it runs its own
                                    // `instantiate`/`join` under `UNWINDING`, recording *its* live
                                    // children before it drains — the recursion that makes depth-2+ work.
                                    let child_dstate = mem
                                        .as_ref()
                                        .map(|m| m.durable_state())
                                        .unwrap_or(STATE_NORMAL);
                                    // S3 `kill`: a fresh flag for this child's subtree; the child (and its
                                    // inherited-flag `thread.spawn` descendants) polls it per op, the
                                    // parent sets it via `Instantiator.kill`.
                                    let kflag = Arc::new(AtomicBool::new(false));
                                    let kflag_child = Arc::clone(&kflag);
                                    let made = sched.spawn(move |id| {
                                        // A nested child is its **own** domain (own host/window/program),
                                        // so it gets its own dispatch table, not the parent's.
                                        let cdt = Arc::new(DomainTable::new(&cfuncs, 0));
                                        let mut child = VCpu::new(
                                            cfuncs,
                                            entry as u32,
                                            &child_args, // [Instantiator] or [Instantiator, AddressSpace]
                                            child_mem,
                                            child_host,
                                            child_fuel,
                                            depth + 1,
                                            id,
                                            csched,
                                            spawn_quota, // a nested child inherits the domain's spawn quota
                                            cdt,
                                        );
                                        child.memop = memop;
                                        // §4: durability is a subtree property — a durable parent's
                                        // instantiated child runs durable (freezable module enforced
                                        // at the admission check above).
                                        child.durable = durable;
                                        // Its freeze-unwind is carve-self-describing: record no
                                        // host-side extent (see `VCpu::nested_child`).
                                        child.nested_child = true;
                                        // §4 depth-2: the child pushes its own [`FrozenNested`] residue
                                        // (for any grandchild) into the subtree's shared sink, so it
                                        // coalesces in the root host rather than the child's private one.
                                        child.freeze_sink = Some(residue_sink);
                                        // §4 subtree STW: inherit the parent's freeze phase (see above),
                                        // so a mid-freeze instantiate composes recursively.
                                        child.dstate = child_dstate;
                                        child.kill = Some(kflag_child); // S3: parent-settable kill flag
                                        Box::new(child)
                                    });
                                    match made {
                                        Some(child_id) => {
                                            threads.push(Some(child_id));
                                            child_kill.insert(threads.len() - 1, kflag); // S3 kill map
                                                                                         // §14 children are tracked apart from `thread.spawn`
                                                                                         // vCPUs, with their carve geometry: a durable freeze
                                                                                         // broadcasts into a live child's carve and records it
                                                                                         // as residue — or fails closed on the un-covered
                                                                                         // shapes (see `VCpu::nested_children`).
                                                                                         // §3.6 slice 3: retain the child's live powerbox
                                                                                         // so `child_offer` (op 14) can mint a LiveImpl.
                                            child_hosts.insert(threads.len() - 1, child_host_keep);
                                            nested_children.push(NestedChildInfo {
                                                slot: threads.len() - 1,
                                                carve_off: ibase + off,
                                                size_log2: size_log2 as u8,
                                                entry: entry as u32,
                                                module_digest: child_mod
                                                    .as_ref()
                                                    .map(|cm| cm.digest),
                                            });
                                            frames[top]
                                                .vals
                                                .push(Reg::from_i32((threads.len() - 1) as i32));
                                        }
                                        None => return Err(Trap::ThreadFault),
                                    }
                                }
                            }
                        }
                        // join(child) -> result: park only this fiber until the child finishes (its
                        // result/trap is delivered on resume via `Pending::Join`); siblings run on.
                        1 => {
                            let ch =
                                get_i32(&frames[top].vals, *args.first().ok_or(Trap::Malformed)?)?;
                            let slot = resolve_thread(threads, ch)?;
                            let child = threads[slot].ok_or(Trap::ThreadFault)?;
                            *pending = Some(Pending::Join { slot });
                            return Ok(Inner::Park(Blocked::Join { child }));
                        }
                        // spawn_coroutine (op 2) / spawn_demand_coroutine (op 4) (entry, off,
                        // size_log2, fuel) -> child handle (or -EINVAL). Like instantiate, but the child
                        // is a **suspended coroutine** (its own confined window + a `Yielder` handle
                        // back to us, its single entry arg), driven cooperatively by `resume` — not run
                        // on the executor. op 4 additionally **demand-pages** the child's window (every
                        // page starts unmapped), so the child faults on first access and we supply the
                        // page — the §14 parent-virtualized-fault / userfaultfd-style lazy-paging model.
                        2 | 4 => {
                            let demand = op == 4;
                            let argn = |i: usize| -> Result<i64, Trap> {
                                Ok(get(
                                    &frames[top].vals,
                                    *args.get(i + askip).ok_or(Trap::Malformed)?,
                                )?
                                .i64())
                            };
                            let entry = argn(0)? as u64;
                            let off = argn(1)? as u64;
                            let size_log2 = argn(2)?;
                            let _quota = argn(3)?; // (per-coroutine fuel metering is a follow-up)
                                                   // A coroutine child entry is a fixed `(i64 yielder) -> (i64)`.
                            let ok_entry = cfs.get(entry as usize).is_some_and(|f| {
                                f.params == [ValType::I64] && f.results == [ValType::I64]
                            });
                            let child_size = if (0..64).contains(&size_log2) {
                                1u64 << size_log2
                            } else {
                                0
                            };
                            // A separate-module child's carve must equal its declared memory (§14
                            // transparency), exactly as for `instantiate_module`.
                            let mod_ok = child_mod
                                .as_ref()
                                .is_none_or(|cm| cm.memory_log2 == Some(size_log2 as u8));
                            let fits = child_size != 0
                                && child_size <= isize
                                && off & (child_size - 1) == 0
                                && off.checked_add(child_size).is_some_and(|e| e <= isize);
                            // §4 enforcement, exactly as for `instantiate`: a durable domain
                            // admits only freezable (durable-attested) separate-module children.
                            let mod_durable_ok =
                                !durable || child_mod.as_ref().is_none_or(|cm| cm.durable);
                            if !ok_entry || !fits || !mod_ok || !mod_durable_ok {
                                frames[top].vals.push(Reg::from_i32(EINVAL as i32));
                            } else {
                                // `ibase`/`off` are holder-relative; the backing-absolute base
                                // adds the holder's own window base (0 for a top-level holder), so
                                // nesting composes at any depth.
                                let abs_base =
                                    mem.as_ref().map_or(0, |m| m.window.base()) + ibase + off;
                                let child_mem = mem.as_ref().map(|m| {
                                    let cm = m.nested_view(abs_base, size_log2 as u8);
                                    if demand {
                                        cm.demand_page(); // every page starts unmapped (lazy paging)
                                    }
                                    cm
                                });
                                // A separate-module child's data segments materialize into the carve
                                // at spawn (see `instantiate`). For a **demand** coroutine they land
                                // in the parent's backing while the child's pages start unmapped — so
                                // a plugin's data segments are *supplied lazily*, page by page, as it
                                // first touches them (the §14 parent-as-pager model, for free).
                                if let (Some(cm), Some(m)) = (&child_mod, mem.as_ref()) {
                                    for d in cm.data.iter() {
                                        if d.offset.saturating_add(d.bytes.len() as u64)
                                            <= child_size
                                        {
                                            for (k, &b) in d.bytes.iter().enumerate() {
                                                m.set_byte(abs_base + d.offset + k as u64, b);
                                            }
                                        }
                                    }
                                }
                                let mut ch = Host::new();
                                // §4: a durable parent's co-fiber child is durable too (see
                                // `instantiate`); its module admission was enforced above.
                                ch.set_durable(durable);
                                // §6: a co-fiber child is nested (window-exposed), freezable iff durable.
                                let catt = {
                                    let hg = host.lock().unwrap_or_else(|e| e.into_inner());
                                    hg.child_attestation(durable)
                                };
                                ch.set_attestation(catt);
                                let cy = ch.grant_yielder(); // the child's handle to suspend back to us
                                let child_host = Arc::new(Mutex::new(ch));
                                let cfuncs = child_mod
                                    .as_ref()
                                    .map_or_else(|| Arc::clone(&funcs), |cm| Arc::clone(&cm.funcs));
                                // A co-fiber child is its own domain → its own dispatch table.
                                let cdt = Arc::new(DomainTable::new(&cfuncs, 0));
                                let mut child = VCpu::new(
                                    cfuncs,
                                    entry as u32,
                                    &[Value::I64(cy as i64)],
                                    child_mem,
                                    child_host,
                                    *fuel,
                                    depth + 1,
                                    0, // unused: a coroutine is driven inline, never via the executor
                                    sched.clone(),
                                    spawn_quota, // co-fiber child inherits the domain's spawn quota
                                    cdt,
                                );
                                child.fault_yields = true; // its page faults suspend to us, not trap
                                child.durable = durable; // §4: durability is a subtree property
                                coroutines.push(Some(Coro {
                                    vcpu: Box::new(child),
                                    awaiting_resume: false,
                                    faulted_page: None,
                                }));
                                frames[top]
                                    .vals
                                    .push(Reg::from_i32((coroutines.len() - 1) as i32));
                            }
                        }
                        // resume(child, value) -> (status: i32, value: i64): drive the coroutine
                        // **inline** until it `yield`s (SUSPENDED), faults on an unmapped page (FAULTED,
                        // value = fault address), or returns (RETURNED). The first resume starts it (its
                        // `value` arg unused); a resume after an explicit yield delivers `value` as the
                        // yield's result; a resume after a fault first **supplies** the faulted page
                        // (the parent has placed its bytes in the shared window) and re-runs the access.
                        // A child trap propagates to us.
                        3 => {
                            let ch =
                                get_i32(&frames[top].vals, *args.first().ok_or(Trap::Malformed)?)?;
                            let value =
                                get_i64(&frames[top].vals, *args.get(1).ok_or(Trap::Malformed)?)?;
                            let slot = ch as usize;
                            let mut coro = match coroutines.get_mut(slot).and_then(|c| c.take()) {
                                Some(c) => c,
                                None => return Err(Trap::CapFault), // forged or already-finished
                            };
                            if let Some(addr) = coro.faulted_page.take() {
                                // Supply the faulted page (map it RW, keeping the parent's bytes), then
                                // re-run the rewound access.
                                if let Some(m) = &coro.vcpu.mem {
                                    m.supply_page(addr);
                                }
                            } else if coro.awaiting_resume {
                                coro.vcpu.pending = Some(Pending::CoResume(value));
                            }
                            match run_inner(&mut coro.vcpu, u64::MAX) {
                                Ok(Inner::CoYield(yv)) => {
                                    coro.awaiting_resume = true;
                                    coroutines[slot] = Some(coro);
                                    frames[top].vals.push(Reg::from_i32(FIBER_SUSPENDED));
                                    frames[top].vals.push(Reg::from_i64(yv));
                                }
                                Ok(Inner::CoFault(addr)) => {
                                    coro.faulted_page = Some(addr);
                                    coroutines[slot] = Some(coro);
                                    frames[top].vals.push(Reg::from_i32(CORO_FAULTED));
                                    frames[top].vals.push(Reg::from_i64(addr as i64));
                                }
                                Ok(Inner::Done(result)) => {
                                    // Finished — `coroutines[slot]` stays `None` (a later resume inert).
                                    frames[top].vals.push(Reg::from_i32(FIBER_RETURNED));
                                    frames[top].vals.push(Reg::from_value(
                                        result.first().copied().unwrap_or(Value::I64(0)),
                                    ));
                                }
                                // A coroutine that parks used a blocking concurrency op (it has no
                                // executor driving it) — unsupported; surface as a fault.
                                Ok(Inner::Park(_)) | Ok(Inner::Yield) => {
                                    return Err(Trap::FiberFault)
                                }
                                // A co-fiber child never carries `debug`, so it cannot pause (S4).
                                Ok(Inner::Pause(..)) => {
                                    unreachable!("debug pause in a coroutine child")
                                }
                                Err(t) => return Err(t), // a child trap propagates to the parent
                            }
                        }
                        // poll(child) -> 0 running | 1 returned | 2 trapped (PROCESS.md S3). Never
                        // parks — the reap probe a shell loops for `WNOHANG` / `SIGCHLD`.
                        // **Non-destructive**: the join-table slot and the child's stashed result are
                        // left in place, so a later `join` still delivers it. A forged / already-joined
                        // / detached handle is a `ThreadFault`, exactly like `join`.
                        9 => {
                            let ch =
                                get_i32(&frames[top].vals, *args.first().ok_or(Trap::Malformed)?)?;
                            let slot = resolve_thread(threads, ch)?;
                            let child = threads[slot].ok_or(Trap::ThreadFault)?;
                            let status = match sched.poll_status(child) {
                                None => 0,        // still running
                                Some(true) => 1,  // returned cleanly
                                Some(false) => 2, // trapped
                            };
                            frames[top].vals.push(Reg::from_i32(status));
                        }
                        // detach(child) -> 0 (PROCESS.md S3): drop the parent's join claim. The child
                        // keeps running to completion (detach is not kill); its result is reaped now if
                        // it already finished, else discarded when the run tears down — never joinable
                        // again. A forged / already-joined handle is a `ThreadFault`. (Auto-reap of a
                        // still-running detached child's eventual result — vs. run-end cleanup — is a
                        // follow-up, as is `kill`, which needs a per-child §5 interrupt.)
                        10 => {
                            let ch =
                                get_i32(&frames[top].vals, *args.first().ok_or(Trap::Malformed)?)?;
                            let slot = resolve_thread(threads, ch)?;
                            let child = threads[slot].ok_or(Trap::ThreadFault)?;
                            threads[slot] = None; // no longer joinable
                            let _ = sched.take_result(child); // reap if already finished
                            frames[top].vals.push(Reg::from_i32(0));
                        }
                        // kill(child) -> 0 | -ESRCH (PROCESS.md S3): set the child's subtree kill flag;
                        // the child (and its `thread.spawn` descendants, which share the flag) traps at
                        // its next per-op poll (`ThreadFault` → `poll` reports 2). Idempotent; killing an
                        // already-finished child (no live flag) is a harmless success. The parent must
                        // then `poll`/`detach` rather than `join` (a `join` would propagate the child's
                        // trap to the parent). Reliably stops a **running** child; a child parked on a
                        // futex/join observes the kill when it next wakes (prompt parked-wake is a
                        // follow-up). A forged/joined handle is a `ThreadFault`.
                        12 => {
                            let ch =
                                get_i32(&frames[top].vals, *args.first().ok_or(Trap::Malformed)?)?;
                            let slot = resolve_thread(threads, ch)?;
                            if let Some(flag) = child_kill.get(&slot) {
                                flag.store(true, Ordering::Relaxed);
                            }
                            frames[top].vals.push(Reg::from_i32(0));
                        }
                        // §3.6 slice 3 — `child_offer(child_handle, export) -> cap | -EINVAL`:
                        // mint a **live-callee offer** over a running child's impl-export. A
                        // call through the returned cap enqueues on the child's inbound queue
                        // and parks this (the caller's) fiber until the child's serve loop
                        // replies — the caller-parking half of the unified model. Probeable
                        // `-EINVAL` for a bad child handle, a finished/joined child, or a
                        // malformed export. The offer's shape is the CHILD's export (its own
                        // registered module — a separate-module child's differs from ours),
                        // fetched first so the two powerbox locks are never held together,
                        // then interned structurally into our table (D59: the id ≡ the shape).
                        14 => {
                            let ch =
                                get_i32(&frames[top].vals, *args.first().ok_or(Trap::Malformed)?)?;
                            let export =
                                get_i32(&frames[top].vals, *args.get(1).ok_or(Trap::Malformed)?)?
                                    as u32;
                            let cap = resolve_thread(threads, ch)
                                .ok()
                                .and_then(|slot| child_hosts.get(&slot).cloned())
                                .and_then(|callee| {
                                    let sigs = callee
                                        .lock()
                                        .unwrap_or_else(|e| e.into_inner())
                                        .offer_shape(export)?;
                                    let mut hg = host.lock().unwrap_or_else(|e| e.into_inner());
                                    hg.wire_live_impl(&callee, export, &sigs).ok()
                                });
                            frames[top]
                                .vals
                                .push(Reg::from_i32(cap.unwrap_or(EINVAL as i32)));
                        }
                        // PROCESS.md §5 — `instantiate_detached(minter, module, grants_ptr,
                        // grants_n, entry, size_log2, quota) -> child | -EINVAL`: spawn a
                        // child from a granted module into a **fresh platform window**, minted
                        // through a `WindowMinter` capability — outside this domain's window,
                        // so no ancestor below the platform holds read authority and the child
                        // attests `window_exposed = false` (the distrust-spawner trust model).
                        // The §14 free data plane is gone BY DESIGN: data flows through the
                        // module's own segments and the named grants (streams / pipe ends /
                        // regions — the op-11 record format), the separate-process discipline.
                        // The spawner keeps kill/join/fuel authority — detachment severs
                        // *read*, not lifecycle. A quota miss / forged minter / bad entry or
                        // size refuses probeably (and a refused spawn charges nothing); a
                        // **durable** domain refuses outright (a detached window is outside
                        // the subtree snapshot — fail closed; multi-window freeze is the
                        // recorded §5/O6 follow-up). No D38 contact: the child's window is an
                        // ordinary reservation with its own guard, exactly a root run's.
                        15 => {
                            let argn = |i: usize| -> Result<i64, Trap> {
                                Ok(
                                    get(&frames[top].vals, *args.get(i).ok_or(Trap::Malformed)?)?
                                        .i64(),
                                )
                            };
                            let minter = argn(0)? as i32;
                            let mh = argn(1)? as i32;
                            let grants_ptr = argn(2)? as u64;
                            let grants_n = argn(3)? as u64;
                            let entry = argn(4)? as u64;
                            let size_log2 = argn(5)?;
                            let quota = argn(6)?;
                            // The module grant (a forged module handle is a CapFault, as ops
                            // 5/13); the child runs it as its own program + self module.
                            let cm = {
                                let hg = host.lock().unwrap_or_else(|e| e.into_inner());
                                let g = hg.resolve_module(mh)?;
                                ChildMod {
                                    funcs: g.funcs.clone(),
                                    memory_log2: g.memory_log2,
                                    data: g.data.clone(),
                                    durable: g.durable,
                                    digest: g.digest,
                                    imports: g.imports.clone(),
                                    types: g.types.clone(),
                                    module: Arc::clone(&g.module),
                                }
                            };
                            // Named grants (the op-11 record format), pre-validated fail-closed.
                            let mut glist: Vec<(String, i32)> = Vec::new();
                            for i in 0..grants_n {
                                let m = mem.as_ref().ok_or(Trap::Malformed)?;
                                let rec = m.read_window(grants_ptr + i * 16, 16)?;
                                let name_off =
                                    u32::from_le_bytes([rec[0], rec[1], rec[2], rec[3]]) as u64;
                                let name_len =
                                    u32::from_le_bytes([rec[4], rec[5], rec[6], rec[7]]) as usize;
                                let gh = i32::from_le_bytes([rec[8], rec[9], rec[10], rec[11]]);
                                let name_bytes = m.read_window(name_off, name_len)?;
                                let name =
                                    String::from_utf8(name_bytes).map_err(|_| Trap::CapFault)?;
                                {
                                    let hg = host.lock().unwrap_or_else(|e| e.into_inner());
                                    hg.can_regrant(gh).then_some(()).ok_or(Trap::CapFault)?;
                                }
                                glist.push((name, gh));
                            }
                            let cfs: &[Func] = &cm.funcs;
                            let want_as =
                                cfs.get(entry as usize).is_some_and(|f| f.params.len() >= 2);
                            let ok_entry = cfs.get(entry as usize).is_some_and(|f| {
                                f.results == [ValType::I64]
                                    && f.params.iter().all(|p| *p == ValType::I64)
                                    && (f.params.len() == 1 || f.params.len() == 2)
                            });
                            let child_size = if (0..64).contains(&size_log2) {
                                1u64 << size_log2
                            } else {
                                0
                            };
                            // §14 transparency: the detached window equals the module's
                            // declared memory (a module with no memory can't spawn).
                            let mod_ok = cm.memory_log2 == Some(size_log2 as u8);
                            let admitted = !durable
                                && ok_entry
                                && child_size != 0
                                && mod_ok
                                && host
                                    .lock()
                                    .unwrap_or_else(|e| e.into_inner())
                                    .window_minter_take(minter, child_size);
                            if !admitted {
                                frames[top].vals.push(Reg::from_i32(EINVAL as i32));
                            } else {
                                // The fresh platform window: its own reservation + guard,
                                // exactly a root run's — nothing of it in this domain's VA.
                                let mut fm =
                                    Mem::with_reservation(DEFAULT_RESERVED_LOG2, size_log2 as u8);
                                fm.init_data(&cm.data);
                                let mut ch = Host::new();
                                ch.set_attestation({
                                    let hg = host.lock().unwrap_or_else(|e| e.into_inner());
                                    hg.detached_child_attestation()
                                });
                                let cinst = ch.grant_instantiator(0, child_size);
                                let cas = ch.grant_address_space(0, child_size);
                                for (name, gh) in &glist {
                                    let cg = {
                                        let mut hg = host.lock().unwrap_or_else(|e| e.into_inner());
                                        hg.regrant_into_child(*gh, &mut ch)
                                    };
                                    if let Some(cg) = cg {
                                        ch.register_cap_name(name, cg);
                                    }
                                }
                                if ch.bind_child_manifest(&cm.imports, &cm.types).is_err() {
                                    frames[top].vals.push(Reg::from_i32(EINVAL as i32));
                                } else {
                                    ch.self_module = Some(Arc::clone(&cm.module));
                                    let child_host = Arc::new(Mutex::new(ch));
                                    let child_host_keep = Arc::clone(&child_host);
                                    let mut child_args = vec![Value::I64(cinst as i64)];
                                    if want_as {
                                        child_args.push(Value::I64(cas as i64));
                                    }
                                    let child_fuel = if quota <= 0 {
                                        *fuel
                                    } else {
                                        (quota as u64).min(*fuel)
                                    };
                                    let cfuncs = Arc::clone(&cm.funcs);
                                    let csched = sched.clone();
                                    let kflag = Arc::new(AtomicBool::new(false));
                                    let kflag_child = Arc::clone(&kflag);
                                    let made = sched.spawn(move |id| {
                                        // A detached child is its own domain: own dispatch
                                        // table, own window; NOT `nested_child` (no carve to
                                        // self-describe) and never in `nested_children` (no
                                        // carve geometry exists — and the durable refusal
                                        // above keeps it out of every freeze path).
                                        let cdt = Arc::new(DomainTable::new(&cfuncs, 0));
                                        let mut child = VCpu::new(
                                            cfuncs,
                                            entry as u32,
                                            &child_args,
                                            Some(fm),
                                            child_host,
                                            child_fuel,
                                            depth + 1,
                                            id,
                                            csched,
                                            spawn_quota,
                                            cdt,
                                        );
                                        child.memop = memop;
                                        child.kill = Some(kflag_child); // lifecycle stays the spawner's
                                        Box::new(child)
                                    });
                                    match made {
                                        Some(child_id) => {
                                            threads.push(Some(child_id));
                                            child_kill.insert(threads.len() - 1, kflag);
                                            // Live offers over a detached child work exactly as
                                            // nested (`child_offer` + caller parking): the
                                            // linkage is the powerbox Arc, not the window.
                                            child_hosts.insert(threads.len() - 1, child_host_keep);
                                            frames[top]
                                                .vals
                                                .push(Reg::from_i32((threads.len() - 1) as i32));
                                        }
                                        None => return Err(Trap::ThreadFault),
                                    }
                                }
                            }
                        }
                        _ => return Err(Trap::CapFault),
                    }
                }
                // Guest-driven `Jit.invoke` (iface 11 op 1, DESIGN.md §22): serviced here, not in
                // the generic dispatch, because it must **run guest code** — a nested evaluation of
                // the compiled unit's entry over THIS vCPU's window, fuel, and powerbox. That is
                // exactly the same-domain/same-window semantics the JIT backend gets by calling the
                // unit's native trampoline over the live window (`invoke_extra`), so the two
                // backends stay in differential lockstep. compile/release (ops 0/2) take the
                // generic dispatch below like any host-state op.
                // Guest-driven `Jit.install` (iface 11 op 3, DESIGN.md §22): install a compiled
                // unit into the `call_indirect` dispatch table, returning its slot index — a
                // funcref old code can `call_indirect` (old→new). Serviced here (it mutates this
                // vCPU's `units`/`table`, which the generic host dispatch can't reach). The JIT
                // mirrors it by writing the unit's natural entry + `type_id` into the same padding
                // slot (`CompiledModule::install`), so the returned index agrees.
                Inst::CapCall {
                    type_id: cap_id::JIT,
                    op: 3,
                    handle,
                    args,
                    ..
                } => {
                    let h = get_i32(&frames[top].vals, *handle)?;
                    jit_install_body!(h, args)
                }
                // `Jit.uninstall(slot)` (iface 11 op 4, DESIGN.md §22 reclaim): clear a
                // previously-installed table slot so the index is reusable and a stale
                // `call_indirect` of it traps. Serviced here (it mutates the table). A guest may
                // only clear slots it installed (`≥ funcs.len()`, the module-0 function count);
                // `0` on success, `-EINVAL` for a real-function/out-of-range/already-empty slot.
                Inst::CapCall {
                    type_id: cap_id::JIT,
                    op: 4,
                    handle,
                    args,
                    ..
                } => {
                    let h = get_i32(&frames[top].vals, *handle)?;
                    jit_uninstall_body!(h, args)
                }
                Inst::CapCall {
                    type_id: cap_id::JIT,
                    op: 1,
                    handle,
                    args,
                    sig,
                } => {
                    let h = get_i32(&frames[top].vals, *handle)?;
                    jit_invoke_body!(h, args, sig)
                }
                // §3.6 slices 2+3+5b — the service points. `svc.poll` (op 9) serves everything
                // currently runnable and returns the count of *completed* dispatches;
                // `svc.wait` (op 10) parks when nothing is runnable and no progress was made,
                // until a caller's enqueue — or an in-flight handler's wake — re-admits it.
                // Each dispatch runs as a handler over the domain's **one world** (same
                // functions, live window, powerbox, fuel), admitted as a **fiber of this
                // vCPU** (slice 5b): the serve frame rewinds and parks as its resumer, so a
                // handler that fiber-parks (futex / blocking read / live call) suspends back
                // here with `FIBER_PARKED` — a completed-but-not-replied dispatch whose caller
                // stays parked in `ticket_waiters` — and the serve loop moves on (a park
                // blocks the fiber, never the domain). Parked handlers are re-claimed on
                // every re-execution; their wakes also `svc_wake` this domain (the waiter's
                // domain key), so a `svc.wait`-parked serve loop resumes them. The whole arm
                // is a rewind-driven state machine — one fiber switch per execution, state in
                // `serve_run`/`handler_parks`/`serve_count`. A handler trap is terminal (one
                // world, no second state to shield); a handler `suspend` has no resumer to
                // receive it (`FiberFault`); a completed dispatch's result wakes its parked
                // caller (`cap_reply`) or rides the completion cell. Serviced here because
                // only the eval loop can run guest code; other tiers answer a probeable
                // `-EINVAL` from host-side dispatch.
                Inst::CapCall {
                    type_id: svm_ir::CAP_SELF_TYPE_ID,
                    op: op @ (CAP_SELF_SVC_POLL | CAP_SELF_SVC_WAIT),
                    sig,
                    args,
                    ..
                } => {
                    if let Some(sr_) = serve_run.as_ref() {
                        if *cur != sr_.serve_cur {
                            // A nested `svc.*` from *under* the running handler (the serve
                            // frame is a parked ancestor): probeable refusal — the serve loop
                            // is the domain's outermost dispatcher (re-entry into a domain is
                            // a fresh dispatch, never a nested drain).
                            if !sig.results.is_empty() {
                                frames[top].vals.push(Reg::from_i64(EINVAL));
                            }
                            continue;
                        }
                    }
                    // The serve frame back in control after a handler switch: the fiber exit
                    // paths pushed `(status, value)` onto this frame — pop them and settle
                    // that dispatch before the rewound op runs the machine again.
                    if let Some(run) = serve_run.take() {
                        // DURABILITY.md §13.4 step 3: a freeze that lands **mid-handler** fails
                        // closed. Under `UNWINDING` the handler's exit is an unwind return —
                        // its "(FIBER_RETURNED, 0)" would settle a bogus zero into the caller's
                        // completion cell, and even a genuine return's reply linkage is not yet
                        // in the snapshot (the step-4 serve_run record). Refuse the freeze
                        // (same shape as the `handler_parks` gate); the previous snapshot
                        // remains the recovery point.
                        if durable
                            && mem.as_ref().map(|m| m.durable_state()) == Some(STATE_UNWINDING)
                        {
                            return Err(Trap::FiberFault);
                        }
                        let value = frames[top].vals.pop().ok_or(Trap::Malformed)?.i64();
                        let status = frames[top].vals.pop().ok_or(Trap::Malformed)?.i32();
                        match status {
                            FIBER_RETURNED => {
                                // Reply-wake the parked caller, or stash in the completion
                                // cell — atomically, so a caller parking mid-settle can't
                                // strand ([`Scheduler::cap_reply_or_stash`]).
                                sched.cap_reply_or_stash(run.ticket, value, host);
                                *serve_count += 1;
                            }
                            FIBER_PARKED => {
                                handler_parks.insert(run.slot, (run.handle, run.ticket));
                            }
                            // A handler `suspend` has no resumer to receive its yield — the
                            // serve loop is not a `cont.resume` site. Same family as the root
                            // suspending: a fiber fault, terminal for the one world.
                            _ => return Err(Trap::FiberFault),
                        }
                    }
                    // Switch into handler-fiber frames: the `cont.resume` tail with the serve
                    // frame rewound, so the handler's every exit re-executes this op.
                    macro_rules! serve_switch {
                        ($slot:expr, $handle:expr, $ticket:expr, $new_frames:expr) => {{
                            *serve_run = Some(ServeRun {
                                slot: $slot,
                                handle: $handle,
                                ticket: $ticket,
                                serve_cur: *cur,
                            });
                            frames[top].inst -= 1;
                            let parked = std::mem::take(frames);
                            *parked_frames += parked.len();
                            if *cur == ROOT_FIBER {
                                *root_parked = Some(parked);
                            } else {
                                registry.park_resumer(*cur, parked);
                            }
                            shadow_switch(
                                mem,
                                registry,
                                root_shadow_sp,
                                *vcpu_ctx,
                                durable_sp_ctx,
                                durable,
                                *cur,
                                $slot,
                            );
                            chain.push($slot);
                            *cur = $slot;
                            *frames = $new_frames;
                            continue 'frames;
                        }};
                    }
                    // A woken parked handler resumes before new admissions (its dispatch is
                    // older than anything still queued); still-blocked ones are put back by
                    // the claim. A handler slot claimable any other way means the guest
                    // resumed a forged handle into it — the racing-claim fault family.
                    let parked_now: Vec<(usize, i64, u64)> = handler_parks
                        .iter()
                        .map(|(s_, (h_, t_))| (*s_, *h_, *t_))
                        .collect();
                    for (pslot, phandle, pticket) in parked_now {
                        match registry.claim(phandle)? {
                            (_, Claimed::StillParked) => {}
                            (_, Claimed::LiveWoken(f)) => {
                                handler_parks.remove(&pslot);
                                serve_switch!(pslot, phandle, pticket, f);
                            }
                            _ => return Err(Trap::FiberFault),
                        }
                    }
                    // Admit queued dispatches: un-servable ones settle inline with a probeable
                    // errno (the dispatch's fault, never the domain's — it keeps serving);
                    // the first servable one switches.
                    loop {
                        let d = {
                            let mut hg = host.lock().unwrap_or_else(|e| e.into_inner());
                            hg.svc_queue.pop_front()
                        };
                        let Some(d) = d else { break };
                        // The queue only holds servable dispatches (checked at enqueue), so a
                        // missing handler here is host-state corruption: fail closed.
                        let fidx = {
                            let hg = host.lock().unwrap_or_else(|e| e.into_inner());
                            hg.svc_handler_func(d.export, d.op).ok_or(Trap::CapFault)?
                        };
                        let params = &funcs.get(fidx as usize).ok_or(Trap::CapFault)?.params;
                        if d.args.len() != params.len() {
                            sched.cap_reply_or_stash(d.ticket, EINVAL, host);
                            continue;
                        }
                        let child_vals: Vec<Reg> = d
                            .args
                            .iter()
                            .zip(params.iter())
                            .map(|(s_, ty)| Reg::from_value(slot_to_val(*ty, *s_)))
                            .collect();
                        // The handler's fiber slot — an ordinary registry fiber (recycled on
                        // finish), so the §15 quota bounds concurrent parked handlers too.
                        // Exhaustion is backpressure to the dispatch, not a trap.
                        let handle = match registry.create(0, 0, spawn_quota.max_fibers, durable) {
                            Ok(h_) => h_,
                            Err(_) => {
                                sched.cap_reply_or_stash(d.ticket, EAGAIN, host);
                                continue;
                            }
                        };
                        // Claim it straight into `Running` (discarding the placeholder
                        // `Start`): handler first-frames are built here — their signatures
                        // are the impl_export's own, not the `(sp, arg)` fiber launch shape.
                        let (hslot, _) = registry.claim(handle)?;
                        serve_switch!(
                            hslot,
                            handle,
                            d.ticket,
                            vec![Frame {
                                func: fidx,
                                module: 0,
                                block: 0,
                                inst: 0,
                                vals: child_vals,
                            }]
                        );
                    }
                    // Nothing runnable. `svc.wait` with no progress parks, keyed by this
                    // domain's powerbox identity (a caller's enqueue — or a parked handler's
                    // wake — computes the same key and re-admits us); otherwise deliver the
                    // completed count and close the activation.
                    if *op == CAP_SELF_SVC_WAIT
                        && *serve_count == 0
                        && !std::mem::take(&mut svc_timed_out)
                    {
                        frames[top].inst -= 1;
                        let key =
                            host.lock().unwrap_or_else(|e| e.into_inner()).domain_id() as usize;
                        // I38 timed form: the op's single optional arg is a timeout in ns
                        // (`< 0` / absent = wait forever). Re-read on every (re-)park, so a
                        // spurious wake restarts the clock — "at least this long" semantics.
                        let deadline_ns = match args.first() {
                            Some(a) => get(&frames[top].vals, *a)?.i64(),
                            None => -1,
                        };
                        return Ok(Inner::Park(Blocked::SvcWait { key, deadline_ns }));
                    }
                    if !sig.results.is_empty() {
                        frames[top].vals.push(Reg::from_i64(*serve_count));
                    }
                    *serve_count = 0;
                }
                Inst::CapCall {
                    type_id,
                    op,
                    sig,
                    handle,
                    args,
                } => {
                    // Capability call (§3c): resolve the handle in the host-owned table
                    // (mask + type_id/generation check) and dispatch to the mock host.
                    // Args/results cross as i64 slots (the shared host-dispatch ABI).
                    // Synchronous in the reference (the async/submit-complete ABI is §12).
                    let h = get_i32(&frames[top].vals, *handle)?;
                    let mut argv = Vec::with_capacity(args.len());
                    for a in args {
                        argv.push(get(&frames[top].vals, *a)?.i64());
                    }
                    // §3.6 slice 3 — caller-side parking: a call through a live-callee offer
                    // does not dispatch here. It enqueues on the callee's inbound queue, wakes
                    // the callee's `svc.wait` (if parked), and parks THIS fiber until the
                    // handler's reply (`Blocked::CapReply`). A full callee queue is probeable
                    // backpressure (`-EAGAIN` as the call's result), never a trap.
                    let live = {
                        let hg = host.lock().unwrap_or_else(|e| e.into_inner());
                        hg.live_impl_of(h, *type_id)
                    };
                    if let Some((callee, export)) = live {
                        let (ticket, callee_id) = {
                            let mut cg = callee.lock().unwrap_or_else(|e| e.into_inner());
                            (cg.svc_enqueue(export, *op, argv), cg.domain_id())
                        };
                        match ticket {
                            Some(t) => {
                                sched.svc_wake(callee_id as usize);
                                if *cur != ROOT_FIBER {
                                    if let SchedRef::Real(sr) = sched {
                                        let regc = Arc::clone(registry);
                                        let calleec = Arc::clone(&callee);
                                        let svck = host
                                            .lock()
                                            .unwrap_or_else(|e| e.into_inner())
                                            .domain_id()
                                            as usize;
                                        fiber_park!(|slot: usize| {
                                            let mut sg = sr.lock();
                                            sg.ticket_waiters.insert(
                                                t,
                                                Waiter::Fiber {
                                                    reg: Arc::clone(&regc),
                                                    slot,
                                                    svc: svck,
                                                },
                                            );
                                            drop(sg);
                                            let early = calleec
                                                .lock()
                                                .unwrap_or_else(|e| e.into_inner())
                                                .svc_results
                                                .remove(&t);
                                            if let Some(r) = early {
                                                regc.wake_blocked(slot, Reg::from_i64(r));
                                            }
                                        });
                                    }
                                }
                                return Ok(Inner::Park(Blocked::CapReply { ticket: t, callee }));
                            }
                            None => {
                                if !sig.results.is_empty() {
                                    frames[top].vals.push(Reg::from_i64(EAGAIN));
                                }
                                continue;
                            }
                        }
                    }
                    let gm = mem.as_mut().map(|m| m as &mut dyn GuestMem);
                    // Lock the shared powerbox for the duration of this one cap.call (brief; no nested
                    // host locking). Threads of a domain serialize their capability calls here.
                    let mut hg = host.lock().unwrap_or_else(|e| e.into_inner());
                    let results = hg.cap_dispatch_slots(*type_id, *op, h, &argv, gm)?;
                    // §3.6 slice 1 — the two revocation-unparks hooks, direct-call route:
                    // (a) a blocking stream read with no data parks THIS fiber, keyed by the
                    //     handle it is parked through (the dispatch's placeholder result is
                    //     discarded; the wake delivers the real one via `Pending::CapResult`);
                    if hg.take_stdin_parked() {
                        drop(hg);
                        if *cur != ROOT_FIBER {
                            if let SchedRef::Real(sr) = sched {
                                let regc = Arc::clone(registry);
                                let hostc = Arc::clone(host);
                                let svck =
                                    hostc.lock().unwrap_or_else(|e| e.into_inner()).domain_id()
                                        as usize;
                                fiber_park!(|slot: usize| {
                                    let mut sg = sr.lock();
                                    sg.cap_waiters.entry(h).or_default().push(Waiter::Fiber {
                                        reg: Arc::clone(&regc),
                                        slot,
                                        svc: svck,
                                    });
                                    drop(sg);
                                    let live = hostc
                                        .lock()
                                        .unwrap_or_else(|e| e.into_inner())
                                        .handle_live(h);
                                    if !live {
                                        regc.wake_blocked(slot, Reg::from_i64(CAP_REVOKED));
                                    }
                                });
                            }
                        }
                        return Ok(Inner::Park(Blocked::CapRead { handle: h }));
                    }
                    // (b) a `Stream.close` that just revoked `h` wakes every sibling fiber
                    //     parked in a call through it, each completing with a probeable errno.
                    let closed = *type_id == cap_id::STREAM && *op == 2;
                    drop(hg);
                    if closed {
                        sched.cap_revoke(h, CAP_REVOKED);
                    }
                    for (s, ty) in results.iter().zip(&sig.results) {
                        frames[top].vals.push(Reg::from_value(slot_to_val(*ty, *s)));
                    }
                }
                // §7 executable named import (IMPORTS.md phase 1): dispatch through the reserved
                // [`svm_ir::CAP_IMPORT_TYPE_ID`] with the import index as the op — the host
                // translates it via the domain's instantiation-time binding table (import `i` →
                // bound `(type_id, op)` + granted handle) and re-dispatches. The module is never
                // rewritten; the handle operand is vestigial (the binding carries the handle), so
                // it is not read. Same shared host entry as the JIT thunk and the bytecode engine,
                // so all three backends agree over one implementation.
                // IMPORTS.md phase 3: an executable import bound to the guest-driven `Jit`
                // interface must reach the special servicing above (invoke runs guest code;
                // install/uninstall mutate the shared dispatch table — none reachable from the
                // generic host dispatch). Route by the instance binding; the vestigial handle
                // operand is ignored (the binding carries the real handle).
                Inst::CallImport {
                    import, sig, args, ..
                }
                | Inst::CallSym {
                    import, sig, args, ..
                } if matches!(
                    host.lock().unwrap_or_else(|e| e.into_inner()).import_binding(*import),
                    Some(b) if b.type_id == cap_id::JIT && matches!(b.op, 1 | 3 | 4)
                ) =>
                {
                    let b = host
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .import_binding(*import)
                        .ok_or(Trap::CapFault)?;
                    let h = b.handle;
                    match b.op {
                        3 => jit_install_body!(h, args),
                        4 => jit_uninstall_body!(h, args),
                        _ => jit_invoke_body!(h, args, sig),
                    }
                }
                Inst::CallImport {
                    import,
                    op,
                    sig,
                    args,
                    ..
                } => {
                    let mut argv = Vec::with_capacity(args.len());
                    for a in args {
                        argv.push(get(&frames[top].vals, *a)?.i64());
                    }
                    // §3.6 slice 4 — a slot bound (e.g. by `import.attach`) to a live-callee
                    // offer routes like the direct form: enqueue on the callee, park this
                    // fiber until the reply. Same backpressure and race story as slice 3.
                    let live = {
                        let hg = host.lock().unwrap_or_else(|e| e.into_inner());
                        hg.import_live_target(*import)
                    };
                    if let Some((callee, export, base_op)) = live {
                        let (ticket, callee_id) = {
                            let mut cg = callee.lock().unwrap_or_else(|e| e.into_inner());
                            (cg.svc_enqueue(export, base_op + *op, argv), cg.domain_id())
                        };
                        match ticket {
                            Some(t) => {
                                sched.svc_wake(callee_id as usize);
                                if *cur != ROOT_FIBER {
                                    if let SchedRef::Real(sr) = sched {
                                        let regc = Arc::clone(registry);
                                        let calleec = Arc::clone(&callee);
                                        let svck = host
                                            .lock()
                                            .unwrap_or_else(|e| e.into_inner())
                                            .domain_id()
                                            as usize;
                                        fiber_park!(|slot: usize| {
                                            let mut sg = sr.lock();
                                            sg.ticket_waiters.insert(
                                                t,
                                                Waiter::Fiber {
                                                    reg: Arc::clone(&regc),
                                                    slot,
                                                    svc: svck,
                                                },
                                            );
                                            drop(sg);
                                            let early = calleec
                                                .lock()
                                                .unwrap_or_else(|e| e.into_inner())
                                                .svc_results
                                                .remove(&t);
                                            if let Some(r) = early {
                                                regc.wake_blocked(slot, Reg::from_i64(r));
                                            }
                                        });
                                    }
                                }
                                return Ok(Inner::Park(Blocked::CapReply { ticket: t, callee }));
                            }
                            None => {
                                if !sig.results.is_empty() {
                                    frames[top].vals.push(Reg::from_i64(EAGAIN));
                                }
                                continue;
                            }
                        }
                    }
                    let gm = mem.as_mut().map(|m| m as &mut dyn GuestMem);
                    let mut hg = host.lock().unwrap_or_else(|e| e.into_inner());
                    // §3.5: the reserved import dispatch packs `(slot | consumer_op << 16)`.
                    let packed = *import | (*op << 16);
                    let results =
                        hg.cap_dispatch_slots(svm_ir::CAP_IMPORT_TYPE_ID, packed, 0, &argv, gm)?;
                    // Import-routed blocking reads don't park this slice (slot-parked calls are
                    // the §3.6 caller-parking slice); discard the flag so it can't leak into a
                    // later direct call's park decision — the read keeps its historical 0-EOF.
                    let _ = hg.take_stdin_parked();
                    for (s, ty) in results.iter().zip(&sig.results) {
                        frames[top].vals.push(Reg::from_value(slot_to_val(*ty, *s)));
                    }
                }
                // §7/§22 symbolic call: when the instance bound the name, dispatch exactly like
                // a flat `call.import` (op 0); the legacy handle operand is ignored.
                Inst::CallSym {
                    import, sig, args, ..
                } => {
                    let mut argv = Vec::with_capacity(args.len());
                    for a in args {
                        argv.push(get(&frames[top].vals, *a)?.i64());
                    }
                    let gm = mem.as_mut().map(|m| m as &mut dyn GuestMem);
                    let mut hg = host.lock().unwrap_or_else(|e| e.into_inner());
                    let results =
                        hg.cap_dispatch_slots(svm_ir::CAP_IMPORT_TYPE_ID, *import, 0, &argv, gm)?;
                    let _ = hg.take_stdin_parked(); // no slot-parking this slice (see call.import)
                    for (s, ty) in results.iter().zip(&sig.results) {
                        frames[top].vals.push(Reg::from_value(slot_to_val(*ty, *s)));
                    }
                }
                // §3.5 dynamic-mode dispatch by type-section reference: the reserved dyn entry
                // packs `(type_idx | op << 16)`; the handle operand is the live handle value.
                Inst::CallImportDyn {
                    ty,
                    op,
                    sig,
                    handle,
                    args,
                } => {
                    let h = get_i32(&frames[top].vals, *handle)?;
                    let mut argv = Vec::with_capacity(args.len());
                    for a in args {
                        argv.push(get(&frames[top].vals, *a)?.i64());
                    }
                    let gm = mem.as_mut().map(|m| m as &mut dyn GuestMem);
                    let mut hg = host.lock().unwrap_or_else(|e| e.into_inner());
                    let packed = *ty | (*op << 16);
                    let results =
                        hg.cap_dispatch_slots(svm_ir::CAP_DYN_TYPE_ID, packed, h, &argv, gm)?;
                    let _ = hg.take_stdin_parked(); // no dyn-parking this slice (see call.import)
                    for (s, tyv) in results.iter().zip(&sig.results) {
                        frames[top]
                            .vals
                            .push(Reg::from_value(slot_to_val(*tyv, *s)));
                    }
                }
                // §3.5 self-namespace extensions: reify own offer / intern own shape / probe
                // coverage — all through the shared dispatch entry (op packs `selfop | idx << 8`).
                Inst::ExportHandle { export } => {
                    let mut hg = host.lock().unwrap_or_else(|e| e.into_inner());
                    let results = hg.cap_dispatch_slots(
                        svm_ir::CAP_SELF_TYPE_ID,
                        8 | (*export << 8),
                        0,
                        &[],
                        None,
                    )?;
                    let r = *results.first().ok_or(Trap::CapFault)?;
                    frames[top].vals.push(Reg::from_i32(r as i32));
                }
                Inst::CapSelfTypeId { ty } => {
                    let mut hg = host.lock().unwrap_or_else(|e| e.into_inner());
                    let results = hg.cap_dispatch_slots(
                        svm_ir::CAP_SELF_TYPE_ID,
                        6 | (*ty << 8),
                        0,
                        &[],
                        None,
                    )?;
                    let r = *results.first().ok_or(Trap::CapFault)?;
                    frames[top].vals.push(Reg::from_i32(r as i32));
                }
                Inst::CapSelfCovers { handle, ty } => {
                    let h = get_i32(&frames[top].vals, *handle)?;
                    let mut hg = host.lock().unwrap_or_else(|e| e.into_inner());
                    let results = hg.cap_dispatch_slots(
                        svm_ir::CAP_SELF_TYPE_ID,
                        7 | (*ty << 8),
                        0,
                        &[h as i64],
                        None,
                    )?;
                    let r = *results.first().ok_or(Trap::CapFault)?;
                    frames[top].vals.push(Reg::from_i32(r as i32));
                }
                // Phase-2 `import.attach` (IMPORTS.md): rebind rebindable slot `import` to the
                // handle value — routed through the shared attach dispatch entry, so all three
                // backends agree over one implementation. Result is the `i32` status.
                Inst::ImportAttach { import, handle } => {
                    let h = get_i32(&frames[top].vals, *handle)?;
                    let mut hg = host.lock().unwrap_or_else(|e| e.into_inner());
                    // §3.6 slice 4 — rebind revokes the outgoing *connection*: fibers parked in
                    // a call through the slot's old binding handle wake with the revocation
                    // errno (the pinned racing-fibers trigger: "closing/REBINDING the client
                    // handle"). In-flight live-callee dispatches (ticket-parked) deliberately
                    // still complete — program order is call → results; rebind governs *future*
                    // calls through the slot.
                    let old = hg
                        .import_binding(*import)
                        .filter(|b| b.bound)
                        .map(|b| b.handle);
                    let results = hg.cap_dispatch_slots(
                        svm_ir::CAP_IMPORT_ATTACH_TYPE_ID,
                        *import,
                        0,
                        &[h as i64],
                        None,
                    )?;
                    let status = *results.first().ok_or(Trap::Malformed)?;
                    drop(hg);
                    if status == 0 {
                        if let Some(old) = old {
                            if old != h {
                                sched.cap_revoke(old, CAP_REVOKED);
                            }
                        }
                    }
                    frames[top].vals.push(Reg::from_i32(status as i32));
                }
                // §7 reflection (`cap.self.count`): how many capabilities this domain holds. Routed
                // through `self_dispatch` (op 0) — the same path the JIT's thunk takes, so they agree.
                Inst::CapSelfCount => {
                    let hg = host.lock().unwrap_or_else(|e| e.into_inner());
                    let r = hg.self_dispatch(0, &[])?;
                    frames[top].vals.push(Reg::from_i32(r[0] as i32));
                }
                // §6 attestation (`cap.self.attest`): the domain's platform-vouched provenance, packed
                // into an `i32` (op 4). Same `self_dispatch` path the JIT thunk takes, so they agree.
                Inst::CapSelfAttest => {
                    let hg = host.lock().unwrap_or_else(|e| e.into_inner());
                    let r = hg.self_dispatch(4, &[])?;
                    frames[top].vals.push(Reg::from_i32(r[0] as i32));
                }
                // §7 reflection (`cap.self.get`): the `idx`-th held capability as `(handle, type_id)`
                // (op 1). An out-of-range index is fail-closed (the guest bounds it by the count).
                Inst::CapSelfGet { idx } => {
                    let i = get_i32(&frames[top].vals, *idx)? as i64;
                    let hg = host.lock().unwrap_or_else(|e| e.into_inner());
                    let r = hg.self_dispatch(1, &[i])?;
                    drop(hg);
                    frames[top].vals.push(Reg::from_i32(r[0] as i32));
                    frames[top].vals.push(Reg::from_i32(r[1] as i32));
                }
                // §7 reflection (`cap.self.resolve`): resolve a name buffer to its handle (op 2).
                // Routed through the generic capability dispatch (which has the window to read the
                // name) — the same code the JIT thunk / bytecode engine reach, so all three agree.
                Inst::CapSelfResolve { name_ptr, name_len } => {
                    let ptr = get_i64(&frames[top].vals, *name_ptr)?;
                    let len = get_i64(&frames[top].vals, *name_len)?;
                    let gm = mem.as_mut().map(|m| m as &mut dyn GuestMem);
                    let mut hg = host.lock().unwrap_or_else(|e| e.into_inner());
                    let r =
                        hg.cap_dispatch_slots(svm_ir::CAP_SELF_TYPE_ID, 2, 0, &[ptr, len], gm)?;
                    drop(hg);
                    frames[top].vals.push(Reg::from_i32(r[0] as i32));
                }
                // §7 reflection (`cap.self.label`): write the handle's label into the window (op 3),
                // the reverse of resolve. Routed through the generic dispatch (which has the window).
                Inst::CapSelfLabel {
                    handle,
                    buf_ptr,
                    buf_cap,
                } => {
                    let h = get_i32(&frames[top].vals, *handle)? as i64;
                    let ptr = get_i64(&frames[top].vals, *buf_ptr)?;
                    let cap = get_i64(&frames[top].vals, *buf_cap)?;
                    let gm = mem.as_mut().map(|m| m as &mut dyn GuestMem);
                    let mut hg = host.lock().unwrap_or_else(|e| e.into_inner());
                    let r =
                        hg.cap_dispatch_slots(svm_ir::CAP_SELF_TYPE_ID, 3, 0, &[h, ptr, cap], gm)?;
                    drop(hg);
                    frames[top].vals.push(Reg::from_i32(r[0] as i32));
                }
                // §12 per-vCPU TLS register: read/write **this** vCPU's word (`tls`, destructured from
                // `v` above), so a fiber that migrated here sees the current vCPU's value.
                Inst::VcpuTlsGet => {
                    frames[top].vals.push(Reg::from_i64(*tls));
                }
                Inst::VcpuTlsSet { val } => {
                    *tls = get_i64(&frames[top].vals, *val)?;
                }
                // §12.8 4A.5 durable-runtime-internal: the active context's shadow-SP **word address** —
                // the base of *its own* region (the SP word is the region's first 8 bytes), so concurrent
                // vCPUs never share an SP word. The active context is the running fiber's (`cur + 1`) or,
                // off-table, this vCPU's root context (`vcpu_ctx`). The transform reads no guest-mutable
                // state, so a guest cannot redirect its own shadow stack.
                Inst::DurableShadowBase => {
                    frames[top]
                        .vals
                        .push(Reg::from_i64(shadow_region_base(*durable_sp_ctx) as i64));
                }
                // §12 fiber create: record a `Pending` fiber in the **run-shared** registry
                // (D57), yield its handle (the registry slot — the first handle of a run is 0 on
                // both backends, the unified namespace). No switch.
                Inst::ContNew { func, sp } => {
                    let funcref = get_i32(&frames[top].vals, *func)?;
                    let stack_base = get_i64(&frames[top].vals, *sp)?;
                    // `durable` runs assign the new fiber a distinct shadow region (and refuse if
                    // the reserve is full); a non-durable run ignores the region bookkeeping.
                    let handle =
                        registry.create(funcref, stack_base, spawn_quota.max_fibers, durable)?;
                    frames[top].vals.push(Reg::from_i64(handle));
                }
                // §12 fiber resume: **claim** fiber `k` — any vCPU may, so a fiber suspended on
                // one vCPU migrates to whichever claims it next; exactly one racing claimant wins
                // and a loser faults (D57) — and switch into it, delivering `arg`. The two results
                // `(status, value)` are appended to *this* frame later, when `k` suspends or
                // returns control here (see `Suspend` and `Return`).
                Inst::ContResume { k, arg } => {
                    let kh = get_i64(&frames[top].vals, *k)?;
                    let av = get_i64(&frames[top].vals, *arg)?;
                    // Materialize the target's frames: start a `Pending` fiber, or continue a
                    // parked one (delivering `arg` as the result of its `suspend`).
                    let (target, claimed) = registry.claim(kh)?;
                    let new_frames = match claimed {
                        Claimed::Start { func: funcref, sp } => {
                            // A forged / wrong-type fiber funcref is a **fiber** fault, not a
                            // generic `IndirectCallType`: the fault arises from a `cont.*` op, so it
                            // joins the forged-handle / dead / bomb family and matches the JIT, which
                            // raises `FiberFault` from its first-resume type-check (`fiber_rt`). The
                            // claim already took the slot to `Running`, so this fiber stays inert.
                            let callee = table_lookup(fs, funcref, &fiber_sig())
                                .map_err(|_| Trap::FiberFault)?;
                            // First entry: call `func(sp, arg)` on the fiber's data stack. Fibers
                            // are module-0 only (a unit cannot use `cont.*`, gated at compile).
                            vec![Frame {
                                func: callee,
                                module: 0,
                                block: 0,
                                inst: 0,
                                vals: vec![Reg::from_i64(sp), Reg::from_i64(av)],
                            }]
                        }
                        Claimed::Live(mut f) => {
                            f.last_mut()
                                .ok_or(Trap::Malformed)?
                                .vals
                                .push(Reg::from_i64(av));
                            f
                        }
                        // §3.6 slice 5a: a woken event-park continues verbatim — its park op's
                        // result was already delivered by the wake; the resume arg is not pushed.
                        Claimed::LiveWoken(f) => f,
                        // §3.6 slice 5a: still blocked — the cooperative poll. Report
                        // `(FIBER_PARKED, 0)` to the resumer without switching.
                        Claimed::StillParked => {
                            frames[top].vals.push(Reg::from_i32(FIBER_PARKED));
                            frames[top].vals.push(Reg::from_i64(0));
                            continue;
                        }
                    };
                    // Park the resumer — it stays claimed (`Running`), since an ancestor in a
                    // resume chain is never stealable — and switch to the target.
                    let parked = std::mem::take(frames);
                    *parked_frames += parked.len();
                    if *cur == ROOT_FIBER {
                        *root_parked = Some(parked);
                    } else {
                        registry.park_resumer(*cur, parked);
                    }
                    // Re-point the active shadow-SP from the resumer's region to the target's
                    // (durable runs only) so a freeze that lands while the target runs spills into
                    // the target's own region — never the resumer's.
                    shadow_switch(
                        mem,
                        registry,
                        root_shadow_sp,
                        *vcpu_ctx,
                        durable_sp_ctx,
                        durable,
                        *cur,
                        target,
                    );
                    chain.push(target);
                    *cur = target;
                    *frames = new_frames;
                    continue 'frames;
                }
                // §12 fiber suspend: hand `value` back to the resumer with status SUSPENDED;
                // publish this fiber back to the pool — claimable by **any** vCPU now, the D57
                // migration point — and pop back into the resumer.
                Inst::Suspend { value } => {
                    if *cur == ROOT_FIBER {
                        return Err(Trap::FiberFault); // no resumer (the root cannot suspend)
                    }
                    let v = get_i64(&frames[top].vals, *value)?;
                    let leaving = *cur;
                    registry.park_suspended(*cur, std::mem::take(frames));
                    chain.pop();
                    *cur = *chain.last().expect("chain keeps the root");
                    // Hand the active shadow-SP back to the resumer's region (durable runs only):
                    // the suspended fiber's SP is saved to its slot so a later resume restores it.
                    shadow_switch(
                        mem,
                        registry,
                        root_shadow_sp,
                        *vcpu_ctx,
                        durable_sp_ctx,
                        durable,
                        leaving,
                        *cur,
                    );
                    *frames = if *cur == ROOT_FIBER {
                        root_parked.take().ok_or(Trap::Malformed)?
                    } else {
                        registry.unpark_resumer(*cur)?
                    };
                    *parked_frames -= frames.len();
                    let rtop = frames.len() - 1;
                    frames[rtop].vals.push(Reg::from_i32(FIBER_SUSPENDED));
                    frames[rtop].vals.push(Reg::from_i64(v));
                    continue 'frames;
                }
                // `setjmp`: snapshot this frame's resume point (the value state is captured because
                // `vals` is replaced per block) keyed by the guest `jmp_buf` address, and fall through
                // returning 0. `frames[top].inst` is already advanced past the `setjmp` (line above), so
                // it is exactly the re-entry point. A re-`setjmp` to the same buffer overwrites.
                Inst::SetJmp { buf } => {
                    let key = get_i64(&frames[top].vals, *buf)? as u64;
                    let result_idx = frames[top].vals.len();
                    let mut snap = frames[top].vals.clone();
                    snap.push(Reg::from_i32(0)); // the result slot (overwritten by longjmp)
                    setjmp_points.insert(
                        key,
                        SetJmpPoint {
                            depth: frames.len(),
                            block: frames[top].block,
                            inst: frames[top].inst,
                            vals: snap,
                            result_idx,
                        },
                    );
                    frames[top].vals.push(Reg::from_i32(0)); // the direct call returns 0
                }
                // `longjmp`: look up the checkpoint by `jmp_buf` address, unwind the call stack to it
                // (the intervening frames discarded — C has no cleanups), restore the `setjmp` frame's
                // (block, inst, vals) with the result slot set to `val` (a `0` `val` becomes `1`, per C),
                // and resume there. A missing checkpoint, or one whose frame already returned (its
                // `depth` now exceeds the live stack), traps in-sandbox (§3b totality).
                Inst::LongJmp { buf, val } => {
                    let key = get_i64(&frames[top].vals, *buf)? as u64;
                    let v = get_i32(&frames[top].vals, *val)?;
                    let resume = if v == 0 { 1 } else { v };
                    let point = setjmp_points.get(&key).cloned().ok_or(Trap::Malformed)?;
                    if point.depth == 0 || point.depth > frames.len() {
                        return Err(Trap::Malformed); // the setjmp frame has already returned
                    }
                    frames.truncate(point.depth);
                    let f = &mut frames[point.depth - 1];
                    f.block = point.block;
                    f.inst = point.inst;
                    f.vals = point.vals;
                    f.vals[point.result_idx] = Reg::from_i32(resume);
                    continue 'frames;
                }
                // §GC conservative root enumeration (`gc.roots`): collect the deduplicated set of
                // candidate words in `[heap_lo, heap_hi)` across **every** fiber of the domain —
                // this computation's own live `frames` (the caller; the op is call-clobbering, so
                // its roots are already in `frames`), the parked root computation (`root_parked`),
                // and every registry fiber that holds frames: `Parked` (suspended) and
                // `Running(Some)` (a resume-chain ancestor). The currently-running fiber's slot is
                // `Running(None)` (its frames are `frames`, scanned above) so nothing double-counts.
                // Write the first `cap` candidates (ascending) into guest memory at `buf`, yield the
                // total found. Ambient + authority-neutral (GC.md §3): every candidate is in-window
                // guest data the heap already encodes; nothing host-side can appear in a `Value`.
                Inst::GcRoots {
                    heap_lo,
                    heap_hi,
                    mask,
                    buf,
                    cap,
                } => {
                    let lo = get_i64(&frames[top].vals, *heap_lo)? as u64;
                    let hi = get_i64(&frames[top].vals, *heap_hi)? as u64;
                    let mask = get_i64(&frames[top].vals, *mask)? as u64;
                    // Security: the payload mask may only clear the top byte (low 56 bits all-ones),
                    // else a host pointer could be folded into the guest window and leak host-address
                    // bits past the range filter (GC.md §3, §6). The verifier rejects a constant
                    // fold-down mask statically; this defends an unverified module / non-constant mask.
                    if mask | 0xFF00_0000_0000_0000 != u64::MAX {
                        return Err(Trap::Malformed);
                    }
                    let dst = get_i64(&frames[top].vals, *buf)? as u64;
                    let cap = get_i64(&frames[top].vals, *cap)?.max(0) as usize;
                    let mut roots = std::collections::BTreeSet::new();
                    gc_scan_frames(frames, lo, hi, mask, &mut roots);
                    if let Some(rp) = root_parked.as_ref() {
                        gc_scan_frames(rp, lo, hi, mask, &mut roots);
                    }
                    for fib in registry.lock().fibers.iter() {
                        if let RegFiber::Parked(f)
                        | RegFiber::Running(Some(f))
                        | RegFiber::ParkedOn { frames: f, .. } = fib
                        {
                            gc_scan_frames(f, lo, hi, mask, &mut roots);
                        }
                    }
                    let total = roots.len();
                    let mut bytes = Vec::with_capacity(total.min(cap) * 8);
                    for w in roots.into_iter().take(cap) {
                        bytes.extend_from_slice(&w.to_le_bytes());
                    }
                    // Reuse the §7 cap-buffer write path: confines `buf` to committed, writable
                    // window pages (a forged/unmapped buffer ⇒ `MemoryFault`), exactly like a host
                    // capability writing its result buffer.
                    let m = mem.as_mut().ok_or(Trap::Malformed)?;
                    m.write_bytes_impl(dst, &bytes).ok_or(Trap::MemoryFault)?;
                    frames[top].vals.push(Reg::from_i64(total as i64));
                }
                // §12 thread spawn: enqueue a new vCPU (green thread) running `funcs[func](arg)` over
                // the *shared* memory (the `Arc<Region>` bytes + §13 `Arc` regions; the child snapshots
                // the page-protection map). The executor runs it on a pooled worker. The child **shares
                // the domain's powerbox** (the same `Arc<Mutex<Host>>`), so a handle granted to the
                // domain works in the child and its I/O reaches the same sink; it gets its own fuel.
                // Yields an i32 thread handle (the table slot).
                Inst::ThreadSpawn { func, sp, arg } => {
                    if fs.get(*func as usize).is_none() {
                        return Err(Trap::Malformed);
                    }
                    let entry = *func;
                    let spv = get_i64(&frames[top].vals, *sp)?; // the thread's data-stack base
                    let av = get_i64(&frames[top].vals, *arg)?;
                    // Durable multi-vCPU (DURABILITY.md §12.8 slice 3.2.1): a child inherits the *current*
                    // state word — under a freeze it spawns into `UNWINDING` so it too unwinds at its next
                    // safepoint. (A child existing *before* the freeze point is reconstructed by the
                    // runtime at thaw, not re-spawned here — the root's rewind skips the prologue
                    // `thread.spawn`, so thaw re-attach is a `drive`-setup concern, not this op's.)
                    let child_state = mem
                        .as_ref()
                        .map(|m| m.durable_state())
                        .unwrap_or(STATE_NORMAL);
                    // Durable multi-vCPU (slice 3.2.2): reserve this child's shadow context top-down
                    // (`MAX_SHADOW_CTX`, −1, …) so it can't collide with a fiber's `slot+1` region.
                    // Fail closed (`ThreadFault`) if the reserve is full — the vCPU pool growing down
                    // would meet the fiber pool growing up. (Non-durable runs never touch the reserve.)
                    let child_ctx = if durable {
                        match registry.reserve_vcpu_context() {
                            Some(c) => c,
                            None => return Err(Trap::ThreadFault),
                        }
                    } else {
                        0
                    };
                    let child_mem = mem.as_ref().map(|m| m.fork_for_thread());
                    let child_host = Arc::clone(host); // inherit the domain powerbox
                    let child_fuel = *fuel; // the child's own metering budget (a copy)
                    let cfuncs = Arc::clone(&funcs);
                    let cdt = Arc::clone(dt); // **share** the domain table: a post-spawn install is visible here (§6 #2)
                                              // **Share** the fiber registry (D57 3b-i): one handle namespace per domain, so
                                              // the child can resume fibers the parent (or a sibling) created and suspended.
                    let creg = Arc::clone(registry);
                    let parent_id = *id; // the spawning vCPU's task — the child's `parent_task`
                    let csched = sched.clone();
                    // A debugged run shares one breakpoint/watchpoint set across all threads
                    // (DEBUGGING.md Milestone B): the child gets its own per-vCPU `DebugCtx` (fresh
                    // logical clock) over the same shared state, so a breakpoint fires in it too.
                    let cdebug = debug.as_ref().map(|d| Arc::clone(&d.shared));
                    // S3 `kill`: a `thread.spawn` descendant of a §14 child **inherits** its kill
                    // flag, so killing the §14 child terminates its whole thread subtree; `None` (root
                    // / top-level thread) stays unkillable.
                    let kill_inherit = kill.clone();
                    let made = sched.spawn(move |id| {
                        let mut child = VCpu::new(
                            cfuncs,
                            entry,
                            &[Value::I64(spv), Value::I64(av)], // (sp, arg) — the fiber-style entry
                            child_mem,
                            child_host,
                            child_fuel,
                            0,
                            id,
                            csched,
                            spawn_quota, // spawned vCPU inherits the domain's spawn quota
                            cdt,
                        );
                        child.registry = creg;
                        child.memop = memop; // inherit the explorer's memory-op granularity
                        child.durable = durable; // durability is a domain property (shared window/registry)
                                                 // Durable multi-vCPU (slice 3.2.2): this child owns the top-down context reserved
                                                 // above, so its shadow stack lives in its own region; it carries its own state word
                                                 // (swapped in by the runtime). Retain `(entry, [sp, arg])` so a freeze emits residue.
                        child.root_shadow_sp = shadow_frame_base(child_ctx);
                        child.vcpu_ctx = child_ctx; // freed back to the registry when it finishes
                        child.dstate = child_state;
                        child.parent_task = parent_id; // slice 3.4: who spawned it (nested-spawn thaw)
                        child.spawn_residue = Some((entry, vec![spv, av]));
                        child.debug = cdebug.map(|sh| Box::new(DebugCtx::new(sh)));
                        child.kill = kill_inherit; // S3: inherit the §14 subtree kill flag (or None)
                        Box::new(child)
                    });
                    match made {
                        Some(child_id) => {
                            threads.push(Some(child_id));
                            frames[top]
                                .vals
                                .push(Reg::from_i32((threads.len() - 1) as i32));
                        }
                        None => return Err(Trap::ThreadFault), // live cap (a thread-bomb)
                    }
                }
                // §12 thread join: park until vCPU `handle` finishes, then (on resume) take its i64
                // result. A forged / out-of-range / already-joined handle is inert (masked + checked
                // like a fiber handle); a trap in the joined vCPU propagates here (on resume).
                Inst::ThreadJoin { handle } => {
                    let h = get_i32(&frames[top].vals, *handle)?;
                    let slot = resolve_thread(threads, h)?;
                    let child = threads[slot].ok_or(Trap::ThreadFault)?;
                    *pending = Some(Pending::Join { slot });
                    return Ok(Inner::Park(Blocked::Join { child }));
                }
                // §12 futex wait: validate the address (confine/align/prot — traps surface here), then
                // park; the executor re-checks the value under its lock (atomic vs. `notify`) and either
                // resumes immediately (value changed → status 1) or blocks until notified / timed out.
                Inst::MemoryWait {
                    ty,
                    addr,
                    expected,
                    timeout,
                } => {
                    let width = atomic_width(*ty);
                    let a = get_i64(&frames[top].vals, *addr)? as u64;
                    let exp = get(&frames[top].vals, *expected)?.lo & width_mask(width);
                    let to_ns = get_i64(&frames[top].vals, *timeout)?;
                    let m = mem.as_ref().ok_or(Trap::Malformed)?;
                    let base = m.prepare_wait(a, *ty)?;
                    // S1b: the wait-queue key is region-canonical for an aliased page (so peers on the
                    // same `SharedRegion` byte rendezvous), while `addr` stays the absolute address the
                    // driver re-reads for the compare-and-park.
                    let key = m.futex_key(base);
                    let wait = if to_ns < 0 {
                        MAX_WAIT
                    } else {
                        Duration::from_nanos(to_ns as u64).min(MAX_WAIT)
                    };
                    if *cur != ROOT_FIBER {
                        if let SchedRef::Real(sr) = sched {
                            let regc = Arc::clone(registry);
                            let svck =
                                host.lock().unwrap_or_else(|e| e.into_inner()).domain_id() as usize;
                            fiber_park!(|slot: usize| {
                                let mut sg = sr.lock();
                                let wid = sg.next_wid;
                                sg.next_wid += 1;
                                sg.timers.push(Reverse((Instant::now() + wait, wid, key)));
                                sg.wait_waiters.entry(key).or_default().push((
                                    wid,
                                    Waiter::Fiber {
                                        reg: Arc::clone(&regc),
                                        slot,
                                        svc: svck,
                                    },
                                ));
                                // Compare-under-lock: a value that already changed wakes the
                                // fiber immediately with the not-equal status.
                                let curv = mem.as_ref().map_or(0, |mm| mm.atomic_value(a, width));
                                drop(sg);
                                if curv != exp {
                                    regc.wake_blocked(slot, Reg::from_i32(WAIT_NOT_EQUAL));
                                }
                                sr.work.notify_all(); // idle workers recompute timer deadlines
                            });
                        }
                    }
                    return Ok(Inner::Park(Blocked::Wait {
                        key,
                        addr: base,
                        expected: exp,
                        width,
                        timeout_ns: wait.as_nanos() as u64,
                    }));
                }
                // §12 futex notify: wake up to `count` vCPUs parked on the confined address.
                Inst::MemoryNotify { addr, count } => {
                    let a = get_i64(&frames[top].vals, *addr)? as u64;
                    // The count is **unsigned** "wake up to N" (wasm's `memory.atomic.notify` count is
                    // u32; the wake-all idiom is `-1` = u32::MAX). `notify` caps at the real waiter
                    // count, so reinterpreting the i32 bits as u32 is safe and faithful.
                    let n = get_i32(&frames[top].vals, *count)? as u32;
                    let m = mem.as_ref().ok_or(Trap::Malformed)?;
                    let base = m.confine_for_notify(a)?;
                    // S1b: notify on the same canonical key the waiter enqueued under.
                    let key = m.futex_key(base);
                    frames[top]
                        .vals
                        .push(Reg::from_i32(sched.notify(key, n) as i32));
                }
                // Fast paths for the **pure** value ops (no memory, no SIMD): dispatch here and
                // push the result directly, instead of through the `eval_inst` call (and its
                // `Option<Reg>` return). `eval_inst` is a very large function that never inlines,
                // so the call dominated the per-op cost on scalar/float code — eliminating it for
                // these ops is the bulk of the eval-loop speedup. The operation *semantics* live in
                // shared helpers (`bin64`/`fbin64`/`cast`/…) called from both here and `eval_inst`,
                // so only thin dispatch glue is repeated; `eval_inst` keeps a (now-unreachable, but
                // exhaustive and identical) copy of these arms and remains the path for memory,
                // SIMD, and the no-result/control ops.
                Inst::ConstI32(c) => frames[top].vals.push(Reg::from_i32(*c)),
                Inst::ConstI64(c) => frames[top].vals.push(Reg::from_i64(*c)),
                Inst::ConstF32(bits) => frames[top].vals.push(Reg::from_f32(f32::from_bits(*bits))),
                Inst::ConstF64(bits) => frames[top].vals.push(Reg::from_f64(f64::from_bits(*bits))),
                Inst::IntBin { ty, op, a, b } => {
                    let vals = &frames[top].vals;
                    let r = match ty {
                        IntTy::I32 => {
                            Reg::from_i32(bin32(*op, get_i32(vals, *a)?, get_i32(vals, *b)?)?)
                        }
                        IntTy::I64 => {
                            Reg::from_i64(bin64(*op, get_i64(vals, *a)?, get_i64(vals, *b)?)?)
                        }
                    };
                    frames[top].vals.push(r);
                }
                Inst::IntCmp { ty, op, a, b } => {
                    let vals = &frames[top].vals;
                    let r = match ty {
                        IntTy::I32 => cmp32(*op, get_i32(vals, *a)?, get_i32(vals, *b)?),
                        IntTy::I64 => cmp64(*op, get_i64(vals, *a)?, get_i64(vals, *b)?),
                    };
                    frames[top].vals.push(Reg::from_i32(r as i32));
                }
                Inst::IntUn { ty, op, a } => {
                    let vals = &frames[top].vals;
                    let r = match ty {
                        IntTy::I32 => Reg::from_i32(intun32(*op, get_i32(vals, *a)?)),
                        IntTy::I64 => Reg::from_i64(intun64(*op, get_i64(vals, *a)?)),
                    };
                    frames[top].vals.push(r);
                }
                Inst::Eqz { ty, a } => {
                    let vals = &frames[top].vals;
                    let r = match ty {
                        IntTy::I32 => get_i32(vals, *a)? == 0,
                        IntTy::I64 => get_i64(vals, *a)? == 0,
                    };
                    frames[top].vals.push(Reg::from_i32(r as i32));
                }
                Inst::Convert { op, a } => {
                    let vals = &frames[top].vals;
                    let r = match op {
                        ConvOp::ExtendI32S => Reg::from_i64(get_i32(vals, *a)? as i64),
                        ConvOp::ExtendI32U => Reg::from_i64(get_i32(vals, *a)? as u32 as i64),
                        ConvOp::WrapI64 => Reg::from_i32(get_i64(vals, *a)? as i32),
                    };
                    frames[top].vals.push(r);
                }
                Inst::Select { cond, a, b } => {
                    let vals = &frames[top].vals;
                    let r = if get_i32(vals, *cond)? != 0 {
                        get(vals, *a)?
                    } else {
                        get(vals, *b)?
                    };
                    frames[top].vals.push(r);
                }
                Inst::FBin { ty, op, a, b } => {
                    let vals = &frames[top].vals;
                    let r = match ty {
                        FloatTy::F32 => {
                            Reg::from_f32(fbin32(*op, get_f32(vals, *a)?, get_f32(vals, *b)?))
                        }
                        FloatTy::F64 => {
                            Reg::from_f64(fbin64(*op, get_f64(vals, *a)?, get_f64(vals, *b)?))
                        }
                    };
                    frames[top].vals.push(r);
                }
                Inst::FUn { ty, op, a } => {
                    let vals = &frames[top].vals;
                    let r = match ty {
                        FloatTy::F32 => Reg::from_f32(fun32(*op, get_f32(vals, *a)?)),
                        FloatTy::F64 => Reg::from_f64(fun64(*op, get_f64(vals, *a)?)),
                    };
                    frames[top].vals.push(r);
                }
                Inst::FCmp { ty, op, a, b } => {
                    let vals = &frames[top].vals;
                    let r = match ty {
                        FloatTy::F32 => fcmp32(*op, get_f32(vals, *a)?, get_f32(vals, *b)?),
                        FloatTy::F64 => fcmp64(*op, get_f64(vals, *a)?, get_f64(vals, *b)?),
                    };
                    frames[top].vals.push(Reg::from_i32(r as i32));
                }
                Inst::FToISat { op, a } => {
                    let r = fto_i(*op, get(&frames[top].vals, *a)?);
                    frames[top].vals.push(r);
                }
                Inst::FToITrap { op, a } => {
                    let r = trunc_trap(*op, get(&frames[top].vals, *a)?)?;
                    frames[top].vals.push(r);
                }
                Inst::IToFConv { op, a } => {
                    let r = i_to_f(*op, get(&frames[top].vals, *a)?);
                    frames[top].vals.push(r);
                }
                Inst::PtrAdd { a, b } => {
                    let vals = &frames[top].vals;
                    let r = Reg::from_i64(get_i64(vals, *a)?.wrapping_add(get_i64(vals, *b)?));
                    frames[top].vals.push(r);
                }
                Inst::PtrCast { a, .. } => {
                    let r = Reg::from_i64(get_i64(&frames[top].vals, *a)?);
                    frames[top].vals.push(r);
                }
                Inst::Cast { op, a } => {
                    let r = cast(*op, get(&frames[top].vals, *a)?);
                    frames[top].vals.push(r);
                }
                Inst::RefFunc { func } => frames[top].vals.push(Reg::from_i32(*func as i32)),
                // Everything else: one value, or none for `Store`/`AtomicStore`.
                // Fast-path the two common scalar memory ops out of the `eval_inst` call; the
                // §14 fault-driven-yield handling is shared via `handle_mem`. The expensive part
                // (confinement + page-protection in `Mem`) is unchanged — only the dispatch call
                // is removed.
                Inst::Load {
                    op, addr, offset, ..
                } => {
                    let r = (|| -> Result<Option<Reg>, Trap> {
                        let a = get_i64(&frames[top].vals, *addr)? as u64;
                        let m = mem.as_ref().ok_or(Trap::Malformed)?;
                        Ok(Some(Reg::from_value(m.load(a, *offset, *op)?)))
                    })();
                    if let Some(inner) = handle_mem(r, &mut frames[top], fault_yields, mem)? {
                        return Ok(inner);
                    }
                }
                Inst::Store {
                    op,
                    addr,
                    value,
                    offset,
                    ..
                } => {
                    let r = (|| -> Result<Option<Reg>, Trap> {
                        let a = get_i64(&frames[top].vals, *addr)? as u64;
                        let v = Value::I64(get(&frames[top].vals, *value)?.i64());
                        mem.as_mut()
                            .ok_or(Trap::Malformed)?
                            .store(a, *offset, *op, v)?;
                        Ok(None)
                    })();
                    if let Some(inner) = handle_mem(r, &mut frames[top], fault_yields, mem)? {
                        return Ok(inner);
                    }
                }
                // Everything else (other memory ops, SIMD, no-result/control): one `eval_inst`
                // call, through the same fault-yield handling.
                other => {
                    let r = eval_inst(other, &frames[top].vals, mem);
                    if let Some(inner) = handle_mem(r, &mut frames[top], fault_yields, mem)? {
                        return Ok(inner);
                    }
                }
            }
        }

        step(fuel, kill.as_deref())?;
        // Back-edge freeze trigger (DURABILITY.md Phase-4 Slice A): on a durable run armed for
        // back-edges (`ARM_BACKEDGE_OFF`), count down at each branch terminator and promote to
        // `UNWINDING` at 0 so the next loop-header poll begins the freeze — reaching a poll-free
        // compute loop that the fiber-safepoint countdown cannot. Inert unless armed; gated on
        // `durable` so an ordinary run is untouched (byte-identical).
        if durable
            && matches!(
                &block.term,
                Terminator::Br { .. } | Terminator::BrIf { .. } | Terminator::BrTable { .. }
            )
        {
            if let Some(m) = mem.as_mut() {
                m.durable_tick_arm_backedge();
            }
        }
        match &block.term {
            Terminator::Br { target, args } => {
                collect_into(&mut edge_buf, &frames[top].vals, args)?;
                std::mem::swap(&mut frames[top].vals, &mut edge_buf);
                frames[top].block = *target as usize;
                frames[top].inst = 0;
            }
            Terminator::BrIf {
                cond,
                then_blk,
                then_args,
                else_blk,
                else_args,
            } => {
                let (target, edge_args) = if get_i32(&frames[top].vals, *cond)? != 0 {
                    (*then_blk, then_args)
                } else {
                    (*else_blk, else_args)
                };
                collect_into(&mut edge_buf, &frames[top].vals, edge_args)?;
                std::mem::swap(&mut frames[top].vals, &mut edge_buf);
                frames[top].block = target as usize;
                frames[top].inst = 0;
            }
            Terminator::BrTable {
                idx,
                targets,
                default,
            } => {
                let i = get_i32(&frames[top].vals, *idx)? as u32 as usize;
                let (target, edge_args) = targets.get(i).unwrap_or(default);
                collect_into(&mut edge_buf, &frames[top].vals, edge_args)?;
                std::mem::swap(&mut frames[top].vals, &mut edge_buf);
                frames[top].block = *target as usize;
                frames[top].inst = 0;
            }
            Terminator::Return(out) => {
                collect_into(&mut ret_buf, &frames[top].vals, out)?;
                let popped = frames.pop();
                if let Some(caller) = frames.last_mut() {
                    // Caller in the same fiber resumes past its `call` (`inst` already advanced).
                    // Copy results straight in — no per-return results `Vec`.
                    caller.vals.extend_from_slice(&ret_buf);
                } else if *cur == ROOT_FIBER {
                    // The root returned: this vCPU is done. Reconstruct typed result `Value`s from
                    // the returning function's result signature (the public boundary is `Value`).
                    let results = match popped.as_ref().and_then(|p| fs.get(p.func as usize)) {
                        Some(f) => ret_buf
                            .iter()
                            .zip(&f.results)
                            .map(|(s, ty)| s.to_value(*ty))
                            .collect(),
                        None => ret_buf.iter().map(|s| s.to_value(ValType::I64)).collect(),
                    };
                    return Ok(Inner::Done(results));
                } else {
                    // A fiber's function returned: hand its single `i64` back to the resumer
                    // with status RETURNED; the fiber is now `Done` (resuming again traps) —
                    // **unless** this is a freeze-unwind (durable + UNWINDING): then the fiber is
                    // mid-resume-chain and unwound *for freeze*, not a real return, so capture its
                    // residue and mark it `Frozen` (re-enterable on thaw) instead of `Done`. Its
                    // continuation is already flattened into its shadow region; on thaw the resumer
                    // re-issues `cont.resume`, the fiber rewinds at its in-flight (leaf/propagated)
                    // point and runs *forward* — the active-chain analogue of an idle fiber's re-park.
                    let leaving = *cur;
                    // Distinguish a freeze-unwind from a genuine return under UNWINDING: an unwound
                    // fiber spilled frames into its shadow region (`shadow_sp` past the region base),
                    // whereas a non-instrumented fiber that truly returned left it empty. Only the
                    // former is `Frozen` (an instrumented fiber always unwinds at a poll before its
                    // real return, so this never mis-classifies one that should be `Done`).
                    let lctx = shadow_context_index(leaving);
                    let region_base = shadow_region_base(lctx);
                    let shadow_sp = mem
                        .as_ref()
                        .map(|m| m.durable_get_sp(region_base))
                        .unwrap_or_else(|| shadow_frame_base(lctx));
                    // §12.8 4A.5: an unwound fiber spilled *past* its frame base (the SP word occupies
                    // the region's first 8 bytes); an empty stack sits exactly at the frame base.
                    let freezing = durable
                        && shadow_sp > shadow_frame_base(lctx)
                        && mem.as_ref().map(|m| m.durable_state()) == Some(STATE_UNWINDING);
                    if freezing {
                        let (func, sp) = match popped.as_ref() {
                            Some(f) => (
                                f.func as i32,
                                match f.vals.first() {
                                    Some(r) => r.i64(),
                                    _ => 0,
                                },
                            ),
                            None => (0, 0),
                        };
                        registry.freeze_active(*cur, shadow_sp);
                        frozen.push(FrozenFiber {
                            slot: leaving,
                            func,
                            sp,
                            shadow_sp,
                            generation: registry.generation(leaving),
                        });
                    } else {
                        registry.finish(*cur);
                    }
                    chain.pop();
                    *cur = *chain.last().expect("chain keeps the root");
                    // Restore the resumer's active shadow-SP (durable runs only). The fiber is
                    // `Done`, so saving its SP is moot, but `shadow_switch` reads the live word
                    // before overwriting it — correct whether or not it had unwound frames.
                    shadow_switch(
                        mem,
                        registry,
                        root_shadow_sp,
                        *vcpu_ctx,
                        durable_sp_ctx,
                        durable,
                        leaving,
                        *cur,
                    );
                    *frames = if *cur == ROOT_FIBER {
                        root_parked.take().ok_or(Trap::Malformed)?
                    } else {
                        registry.unpark_resumer(*cur)?
                    };
                    *parked_frames -= frames.len();
                    let rtop = frames.len() - 1;
                    frames[rtop].vals.push(Reg::from_i32(FIBER_RETURNED));
                    frames[rtop]
                        .vals
                        .push(ret_buf.first().copied().unwrap_or(Reg::from_i64(0)));
                }
            }
            Terminator::Unreachable => return Err(Trap::Unreachable),
            // Tail calls replace the top frame in place — no depth growth.
            Terminator::ReturnCall { func, args } => {
                let argv = collect(&frames[top].vals, args)?;
                if fs.get(*func as usize).is_none() {
                    return Err(Trap::Malformed);
                }
                frames[top] = Frame {
                    func: *func,
                    module: frames[top].module, // a direct tail call stays in the caller's module
                    block: 0,
                    inst: 0,
                    vals: argv,
                };
            }
            Terminator::ReturnCallIndirect { ty, idx, args } => {
                let (cmod, cfunc) = dispatch_indirect(
                    dt,
                    &funcs,
                    units,
                    invoked,
                    get_i32(&frames[top].vals, *idx)?,
                    ty,
                )?;
                let argv = collect(&frames[top].vals, args)?;
                frames[top] = Frame {
                    func: cfunc,
                    module: cmod,
                    block: 0,
                    inst: 0,
                    vals: argv,
                };
            }
        }
    }
}

fn eval_inst(inst: &Inst, vals: &[Reg], mem: &mut Option<Mem>) -> Result<Option<Reg>, Trap> {
    // Single dispatch over `inst`: value-producing ops yield the produced [`Reg`]; the
    // no-result ops (`Store`/`AtomicStore`/`V128Store`, fences, and the control/fiber ops
    // serviced in the eval loop) take a `return Ok(None)` arm. Operand reads are op-directed
    // (`get_i32`/`get_f64`/`Reg::v128`, …) — the instruction's static type says how to read the
    // untyped slot, so there is no per-value tag to match. The `Mem` wrappers stay `Value`-based;
    // we convert at that boundary (`Value::I64(slot.i64())` covers any scalar store, since the
    // store keeps only the low `width` bytes; loads come back typed → `Reg::from_value`).
    let v = match inst {
        // ----- §3a/§12/§17 no-result memory writes (the only value-less data ops) -----
        Inst::Store {
            op,
            addr,
            value,
            offset,
            ..
        } => {
            let m = mem.as_mut().ok_or(Trap::Malformed)?;
            let a = get_i64(vals, *addr)? as u64;
            m.store(a, *offset, *op, Value::I64(get(vals, *value)?.i64()))?;
            return Ok(None);
        }
        // Bulk-memory ops (D62): both `MemCopy` and `MemMove` route to the overlap-safe `mem_copy`.
        Inst::MemCopy { dst, src, len } | Inst::MemMove { dst, src, len } => {
            let m = mem.as_mut().ok_or(Trap::Malformed)?;
            let d = get_i64(vals, *dst)? as u64;
            let s = get_i64(vals, *src)? as u64;
            let n = get_i64(vals, *len)? as u64;
            m.mem_copy(d, s, n)?;
            return Ok(None);
        }
        Inst::MemFill { dst, val, len } => {
            let m = mem.as_mut().ok_or(Trap::Malformed)?;
            let d = get_i64(vals, *dst)? as u64;
            let v = get(vals, *val)?.i32() as u8;
            let n = get_i64(vals, *len)? as u64;
            m.mem_fill(d, v, n)?;
            return Ok(None);
        }
        Inst::AtomicStore {
            ty,
            addr,
            value,
            offset,
            ..
        } => {
            let m = mem.as_mut().ok_or(Trap::Malformed)?;
            let a = get_i64(vals, *addr)? as u64;
            m.atomic_store(a, *offset, *ty, Value::I64(get(vals, *value)?.i64()))?;
            return Ok(None);
        }
        Inst::V128Store {
            addr,
            value,
            offset,
            ..
        } => {
            let m = mem.as_mut().ok_or(Trap::Malformed)?;
            let a = get_i64(vals, *addr)? as u64;
            m.store_v128(a, *offset, get(vals, *value)?.v128())?;
            return Ok(None);
        }
        Inst::ConstI32(c) => Reg::from_i32(*c),
        Inst::ConstI64(c) => Reg::from_i64(*c),
        // §7 executable named imports (+ phase-2 attach) need the host's import-binding table,
        // so they're serviced in the eval loop (like `cap.call`), never in this pure-op helper.
        // `CallSym` (the v8 link-form placeholder) never verifies, so it can never execute.
        Inst::CallImport { .. }
        | Inst::CallImportDyn { .. }
        | Inst::CallSym { .. }
        | Inst::ExportHandle { .. }
        | Inst::ImportAttach { .. } => return Err(Trap::Malformed),
        // §7 reflection intrinsics need the host table, so they're serviced in the eval loop
        // (like `cap.call`), never here.
        Inst::CapSelfCount
        | Inst::CapSelfAttest
        | Inst::CapSelfGet { .. }
        | Inst::CapSelfResolve { .. }
        | Inst::CapSelfLabel { .. }
        | Inst::CapSelfTypeId { .. }
        | Inst::CapSelfCovers { .. } => return Err(Trap::Malformed),
        // §12 per-vCPU TLS register needs the running vCPU's state, so it's serviced in the eval
        // loop (`run_inner`), never here.
        Inst::VcpuTlsGet | Inst::VcpuTlsSet { .. } => return Err(Trap::Malformed),
        // §12.8 4A.5 durable shadow base needs the running vCPU's context, so it's serviced in the
        // eval loop (`run_inner`), never here.
        Inst::DurableShadowBase => return Err(Trap::Malformed),
        Inst::IntBin { ty, op, a, b } => match ty {
            IntTy::I32 => Reg::from_i32(bin32(*op, get_i32(vals, *a)?, get_i32(vals, *b)?)?),
            IntTy::I64 => Reg::from_i64(bin64(*op, get_i64(vals, *a)?, get_i64(vals, *b)?)?),
        },
        Inst::IntCmp { ty, op, a, b } => {
            let r = match ty {
                IntTy::I32 => cmp32(*op, get_i32(vals, *a)?, get_i32(vals, *b)?),
                IntTy::I64 => cmp64(*op, get_i64(vals, *a)?, get_i64(vals, *b)?),
            };
            Reg::from_i32(r as i32)
        }
        Inst::IntUn { ty, op, a } => match ty {
            IntTy::I32 => Reg::from_i32(intun32(*op, get_i32(vals, *a)?)),
            IntTy::I64 => Reg::from_i64(intun64(*op, get_i64(vals, *a)?)),
        },
        Inst::Eqz { ty, a } => {
            let r = match ty {
                IntTy::I32 => get_i32(vals, *a)? == 0,
                IntTy::I64 => get_i64(vals, *a)? == 0,
            };
            Reg::from_i32(r as i32)
        }
        Inst::Convert { op, a } => match op {
            ConvOp::ExtendI32S => Reg::from_i64(get_i32(vals, *a)? as i64),
            ConvOp::ExtendI32U => Reg::from_i64(get_i32(vals, *a)? as u32 as i64),
            ConvOp::WrapI64 => Reg::from_i32(get_i64(vals, *a)? as i32),
        },
        Inst::Select { cond, a, b } => {
            if get_i32(vals, *cond)? != 0 {
                get(vals, *a)?
            } else {
                get(vals, *b)?
            }
        }
        Inst::ConstF32(bits) => Reg::from_f32(f32::from_bits(*bits)),
        Inst::ConstF64(bits) => Reg::from_f64(f64::from_bits(*bits)),
        Inst::FBin { ty, op, a, b } => match ty {
            FloatTy::F32 => Reg::from_f32(fbin32(*op, get_f32(vals, *a)?, get_f32(vals, *b)?)),
            FloatTy::F64 => Reg::from_f64(fbin64(*op, get_f64(vals, *a)?, get_f64(vals, *b)?)),
        },
        Inst::FUn { ty, op, a } => match ty {
            FloatTy::F32 => Reg::from_f32(fun32(*op, get_f32(vals, *a)?)),
            FloatTy::F64 => Reg::from_f64(fun64(*op, get_f64(vals, *a)?)),
        },
        Inst::Fma { ty, a, b, c } => match ty {
            // `mul_add` is the correctly-rounded fused FMA — bit-identical to Cranelift's `fma`.
            FloatTy::F32 => {
                Reg::from_f32(get_f32(vals, *a)?.mul_add(get_f32(vals, *b)?, get_f32(vals, *c)?))
            }
            FloatTy::F64 => {
                Reg::from_f64(get_f64(vals, *a)?.mul_add(get_f64(vals, *b)?, get_f64(vals, *c)?))
            }
        },
        Inst::FCmp { ty, op, a, b } => {
            let r = match ty {
                FloatTy::F32 => fcmp32(*op, get_f32(vals, *a)?, get_f32(vals, *b)?),
                FloatTy::F64 => fcmp64(*op, get_f64(vals, *a)?, get_f64(vals, *b)?),
            };
            Reg::from_i32(r as i32)
        }
        Inst::FToISat { op, a } => fto_i(*op, get(vals, *a)?),
        Inst::FToITrap { op, a } => trunc_trap(*op, get(vals, *a)?)?,
        Inst::IToFConv { op, a } => i_to_f(*op, get(vals, *a)?),
        Inst::PtrAdd { a, b } => Reg::from_i64(get_i64(vals, *a)?.wrapping_add(get_i64(vals, *b)?)),
        // `ptr.from_int`/`ptr.to_int` are a no-op off-CHERI: pass the i64 through.
        Inst::PtrCast { a, .. } => Reg::from_i64(get_i64(vals, *a)?),
        Inst::Cast { op, a } => cast(*op, get(vals, *a)?),
        // A funcref is just the function index as plain i32 data (§3c).
        Inst::RefFunc { func } => Reg::from_i32(*func as i32),
        Inst::Load {
            op, addr, offset, ..
        } => {
            let m = mem.as_ref().ok_or(Trap::Malformed)?;
            let a = get_i64(vals, *addr)? as u64;
            Reg::from_value(m.load(a, *offset, *op)?)
        }
        // The `order` is carried but execution is seq-cst (a sound strengthening; see `svm_ir::Ordering`).
        Inst::AtomicLoad {
            ty, addr, offset, ..
        } => {
            let m = mem.as_ref().ok_or(Trap::Malformed)?;
            let a = get_i64(vals, *addr)? as u64;
            Reg::from_value(m.atomic_load(a, *offset, *ty)?)
        }
        Inst::AtomicRmw {
            ty,
            op,
            addr,
            value,
            offset,
            ..
        } => {
            let m = mem.as_mut().ok_or(Trap::Malformed)?;
            let a = get_i64(vals, *addr)? as u64;
            let v = Value::I64(get(vals, *value)?.i64());
            Reg::from_value(m.atomic_rmw(a, *offset, *ty, *op, v)?)
        }
        Inst::AtomicCmpxchg {
            ty,
            addr,
            expected,
            replacement,
            offset,
            ..
        } => {
            let m = mem.as_mut().ok_or(Trap::Malformed)?;
            let a = get_i64(vals, *addr)? as u64;
            let exp = Value::I64(get(vals, *expected)?.i64());
            let rep = Value::I64(get(vals, *replacement)?.i64());
            Reg::from_value(m.atomic_cmpxchg(a, *offset, *ty, exp, rep)?)
        }
        // §12 standalone fence — issue the real hardware fence (a `Relaxed` fence is a no-op; `std`
        // would panic on it). Both backends are otherwise seq-cst, so this is the one place ordering
        // is observable.
        Inst::AtomicFence { order } => {
            use std::sync::atomic::{fence, Ordering as O};
            match order {
                svm_ir::Ordering::Relaxed => {}
                svm_ir::Ordering::Acquire => fence(O::Acquire),
                svm_ir::Ordering::Release => fence(O::Release),
                svm_ir::Ordering::AcqRel => fence(O::AcqRel),
                svm_ir::Ordering::SeqCst => fence(O::SeqCst),
            }
            return Ok(None);
        }

        // ----- §17 SIMD reference lane semantics (the differential oracle, D58) -----
        Inst::ConstV128(b) => Reg::from_v128(*b),
        Inst::V128Load { addr, offset, .. } => {
            let m = mem.as_ref().ok_or(Trap::Malformed)?;
            let a = get_i64(vals, *addr)? as u64;
            Reg::from_value(m.load_v128(a, *offset)?)
        }
        Inst::Splat { shape, a } => Reg::from_v128(simd_splat(*shape, get(vals, *a)?.lo)),
        Inst::ExtractLane {
            shape,
            lane,
            signed,
            a,
        } => simd_extract(*shape, *lane, *signed, get(vals, *a)?.v128()),
        Inst::ReplaceLane { shape, lane, a, b } => Reg::from_v128(simd_replace(
            *shape,
            *lane,
            get(vals, *a)?.v128(),
            get(vals, *b)?.lo,
        )),
        Inst::VIntBin { shape, op, a, b } => Reg::from_v128(simd_vint_bin(
            *shape,
            *op,
            get(vals, *a)?.v128(),
            get(vals, *b)?.v128(),
        )),
        Inst::VIntCmp { shape, op, a, b } => Reg::from_v128(simd_vint_cmp(
            *shape,
            *op,
            get(vals, *a)?.v128(),
            get(vals, *b)?.v128(),
        )),
        Inst::VFloatCmp { shape, op, a, b } => Reg::from_v128(simd_vfloat_cmp(
            *shape,
            *op,
            get(vals, *a)?.v128(),
            get(vals, *b)?.v128(),
        )),
        Inst::VShift { shape, op, a, amt } => Reg::from_v128(simd_vshift(
            *shape,
            *op,
            get(vals, *a)?.v128(),
            get_i32(vals, *amt)? as u32,
        )),
        Inst::VIntUn { shape, op, a } => {
            Reg::from_v128(simd_vint_un(*shape, *op, get(vals, *a)?.v128()))
        }
        Inst::VSatBin { shape, op, a, b } => Reg::from_v128(simd_vsat_bin(
            *shape,
            *op,
            get(vals, *a)?.v128(),
            get(vals, *b)?.v128(),
        )),
        Inst::VWiden { shape, op, a } => {
            Reg::from_v128(simd_widen(*shape, *op, get(vals, *a)?.v128()))
        }
        Inst::VNarrow { shape, op, a, b } => Reg::from_v128(simd_narrow(
            *shape,
            *op,
            get(vals, *a)?.v128(),
            get(vals, *b)?.v128(),
        )),
        Inst::VConvert { op, a } => Reg::from_v128(simd_convert(*op, get(vals, *a)?.v128())),
        Inst::VPMinMax { shape, op, a, b } => Reg::from_v128(simd_pminmax(
            *shape,
            *op,
            get(vals, *a)?.v128(),
            get(vals, *b)?.v128(),
        )),
        Inst::VPopcnt { a } => {
            let v = get(vals, *a)?.v128();
            let mut o = [0u8; 16];
            for i in 0..16 {
                o[i] = v[i].count_ones() as u8;
            }
            Reg::from_v128(o)
        }
        Inst::VAvgr { shape, a, b } => Reg::from_v128(simd_avgr(
            *shape,
            get(vals, *a)?.v128(),
            get(vals, *b)?.v128(),
        )),
        Inst::VDot { a, b } => {
            Reg::from_v128(simd_dot(get(vals, *a)?.v128(), get(vals, *b)?.v128()))
        }
        Inst::VDotI8 { a, b } => {
            Reg::from_v128(simd_dot_i8(get(vals, *a)?.v128(), get(vals, *b)?.v128()))
        }
        Inst::VExtMul { shape, op, a, b } => Reg::from_v128(simd_extmul(
            *shape,
            *op,
            get(vals, *a)?.v128(),
            get(vals, *b)?.v128(),
        )),
        Inst::VExtAddPairwise { shape, signed, a } => {
            Reg::from_v128(simd_extadd_pairwise(*shape, *signed, get(vals, *a)?.v128()))
        }
        Inst::VQ15MulrSat { a, b } => {
            Reg::from_v128(simd_q15mulr(get(vals, *a)?.v128(), get(vals, *b)?.v128()))
        }
        Inst::VFma {
            shape,
            neg,
            a,
            b,
            c,
        } => Reg::from_v128(simd_fma(
            *shape,
            *neg,
            get(vals, *a)?.v128(),
            get(vals, *b)?.v128(),
            get(vals, *c)?.v128(),
        )),
        Inst::VAnyTrue { a } => {
            Reg::from_i32((get(vals, *a)?.v128().iter().any(|&b| b != 0)) as i32)
        }
        Inst::VAllTrue { shape, a } => Reg::from_i32(simd_all_true(*shape, get(vals, *a)?.v128())),
        Inst::VBitmask { shape, a } => Reg::from_i32(simd_bitmask(*shape, get(vals, *a)?.v128())),
        Inst::VFloatBin { shape, op, a, b } => Reg::from_v128(simd_vfloat_bin(
            *shape,
            *op,
            get(vals, *a)?.v128(),
            get(vals, *b)?.v128(),
        )),
        Inst::VFloatUn { shape, op, a } => {
            Reg::from_v128(simd_vfloat_un(*shape, *op, get(vals, *a)?.v128()))
        }
        Inst::VBitBin { op, a, b } => Reg::from_v128(simd_vbit_bin(
            *op,
            get(vals, *a)?.v128(),
            get(vals, *b)?.v128(),
        )),
        Inst::VNot { a } => {
            let x = get(vals, *a)?.v128();
            let mut o = [0u8; 16];
            for i in 0..16 {
                o[i] = !x[i];
            }
            Reg::from_v128(o)
        }
        Inst::Bitselect { a, b, mask } => Reg::from_v128(simd_bitselect(
            get(vals, *a)?.v128(),
            get(vals, *b)?.v128(),
            get(vals, *mask)?.v128(),
        )),
        Inst::Shuffle { lanes, a, b } => Reg::from_v128(simd_shuffle(
            lanes,
            get(vals, *a)?.v128(),
            get(vals, *b)?.v128(),
        )),
        Inst::Swizzle { a, b } => {
            Reg::from_v128(simd_swizzle(get(vals, *a)?.v128(), get(vals, *b)?.v128()))
        }
        // The §17/D58 feature-detect hook: a deterministic constant in the fixed-128 MVP, so it
        // stays identical across the interp↔JIT oracle.
        Inst::SimdWidthBytes => Reg::from_i32(16),

        // Handled in `run_func` for the §12 fiber ops (which switch stacks) and in the eval
        // loop for calls/cap-calls; listed for exhaustiveness (no panic).
        Inst::Call { .. }
        | Inst::CallIndirect { .. }
        | Inst::CapCall { .. }
        | Inst::ContNew { .. }
        | Inst::ContResume { .. }
        | Inst::Suspend { .. }
        | Inst::SetJmp { .. }
        | Inst::LongJmp { .. }
        | Inst::GcRoots { .. }
        | Inst::ThreadSpawn { .. }
        | Inst::ThreadJoin { .. }
        | Inst::MemoryWait { .. }
        | Inst::MemoryNotify { .. } => return Ok(None),
    };
    Ok(Some(v))
}

// ----- §17 SIMD lane semantics (reference oracle, D58) -----
// Lanes are little-endian within the 16 bytes. Float lanes reuse the scalar `fbin*`/`fun*`
// helpers so a vector lane and its scalar op are bit-identical (NaN bits included).

/// Read lane `lane` (of `bytes` width) as a zero-extended `u64`.
fn lane_read(v: &[u8; 16], lane: usize, bytes: usize) -> u64 {
    let mut x = 0u64;
    for k in 0..bytes {
        x |= (v[lane * bytes + k] as u64) << (8 * k);
    }
    x
}

/// Write the low `bytes` of `x` into lane `lane`.
fn lane_write(v: &mut [u8; 16], lane: usize, bytes: usize, x: u64) {
    for k in 0..bytes {
        v[lane * bytes + k] = (x >> (8 * k)) as u8;
    }
}

/// `<shape>.splat`: broadcast the low `lane_bytes` of `bits` into every lane.
fn simd_splat(shape: VShape, bits: u64) -> [u8; 16] {
    let bytes = shape.lane_bytes() as usize;
    let mut o = [0u8; 16];
    for i in 0..shape.lanes() as usize {
        lane_write(&mut o, i, bytes, bits);
    }
    o
}

/// `<shape>.extract_lane`: read lane `lane` as the shape's scalar [`Value`]. Narrow integer
/// lanes sign- or zero-extend into the `i32` result per `signed`.
fn simd_extract(shape: VShape, lane: u8, signed: bool, v: [u8; 16]) -> Reg {
    let bytes = shape.lane_bytes() as usize;
    // Lane index is verifier-bounded; clamp defensively so this stays total on raw input.
    let lane = (lane as usize).min(shape.lanes() as usize - 1);
    let raw = lane_read(&v, lane, bytes);
    match shape {
        VShape::I8x16 | VShape::I16x8 => {
            let bits = (bytes * 8) as u32;
            let ext = if signed {
                let shift = 32 - bits;
                (((raw as u32) << shift) as i32) >> shift
            } else {
                raw as i32
            };
            Reg::from_i32(ext)
        }
        VShape::I32x4 => Reg::from_i32(raw as i32),
        VShape::I64x2 => Reg::from_i64(raw as i64),
        VShape::F32x4 => Reg::from_f32(f32::from_bits(raw as u32)),
        VShape::F64x2 => Reg::from_f64(f64::from_bits(raw)),
    }
}

/// `<shape>.replace_lane`: `v` with lane `lane` set to the low `lane_bytes` of `bits`.
fn simd_replace(shape: VShape, lane: u8, mut v: [u8; 16], bits: u64) -> [u8; 16] {
    let bytes = shape.lane_bytes() as usize;
    let lane = (lane as usize).min(shape.lanes() as usize - 1);
    lane_write(&mut v, lane, bytes, bits);
    v
}

/// Lane-wise integer add/sub/mul (wrapping at the lane width — only the low `lane_bytes`
/// are kept, which is exactly modular arithmetic).
fn simd_vint_bin(shape: VShape, op: VIntBinOp, a: [u8; 16], b: [u8; 16]) -> [u8; 16] {
    let bytes = shape.lane_bytes() as usize;
    let mut o = [0u8; 16];
    for i in 0..shape.lanes() as usize {
        let x = lane_read(&a, i, bytes);
        let y = lane_read(&b, i, bytes);
        let r = match op {
            VIntBinOp::Add => x.wrapping_add(y),
            VIntBinOp::Sub => x.wrapping_sub(y),
            VIntBinOp::Mul => x.wrapping_mul(y),
            // Unsigned compares use the zero-extended lane values directly; signed sign-extend first.
            VIntBinOp::MinU => x.min(y),
            VIntBinOp::MaxU => x.max(y),
            VIntBinOp::MinS => lane_sext(x, bytes).min(lane_sext(y, bytes)) as u64,
            VIntBinOp::MaxS => lane_sext(x, bytes).max(lane_sext(y, bytes)) as u64,
        };
        lane_write(&mut o, i, bytes, r);
    }
    o
}

/// Sign-extend the low `bytes` of a zero-extended lane value to a full `i64`.
fn lane_sext(x: u64, bytes: usize) -> i64 {
    let bits = bytes * 8;
    if bits >= 64 {
        x as i64
    } else {
        let shift = 64 - bits;
        ((x << shift) as i64) >> shift
    }
}

/// `<shape>.<cmp>`: per-lane integer comparison → an all-ones (true) / all-zeros (false) mask of the
/// lane width. `lane_read` zero-extends, so unsigned compares are direct; signed compares
/// sign-extend each lane first.
fn simd_vint_cmp(shape: VShape, op: VICmpOp, a: [u8; 16], b: [u8; 16]) -> [u8; 16] {
    let bytes = shape.lane_bytes() as usize;
    let mut o = [0u8; 16];
    for i in 0..shape.lanes() as usize {
        let xu = lane_read(&a, i, bytes);
        let yu = lane_read(&b, i, bytes);
        let (xs, ys) = (lane_sext(xu, bytes), lane_sext(yu, bytes));
        let t = match op {
            VICmpOp::Eq => xu == yu,
            VICmpOp::Ne => xu != yu,
            VICmpOp::LtS => xs < ys,
            VICmpOp::LtU => xu < yu,
            VICmpOp::GtS => xs > ys,
            VICmpOp::GtU => xu > yu,
            VICmpOp::LeS => xs <= ys,
            VICmpOp::LeU => xu <= yu,
            VICmpOp::GeS => xs >= ys,
            VICmpOp::GeU => xu >= yu,
        };
        lane_write(&mut o, i, bytes, if t { u64::MAX } else { 0 });
    }
    o
}

/// Map a vector float op onto the scalar [`FBinOp`]/[`FUnOp`] so lanes match scalars exactly.
fn vf_bin(op: VFloatBinOp) -> FBinOp {
    match op {
        VFloatBinOp::Add => FBinOp::Add,
        VFloatBinOp::Sub => FBinOp::Sub,
        VFloatBinOp::Mul => FBinOp::Mul,
        VFloatBinOp::Div => FBinOp::Div,
        VFloatBinOp::Min => FBinOp::Min,
        VFloatBinOp::Max => FBinOp::Max,
    }
}
fn vf_un(op: VFloatUnOp) -> FUnOp {
    match op {
        VFloatUnOp::Abs => FUnOp::Abs,
        VFloatUnOp::Neg => FUnOp::Neg,
        VFloatUnOp::Sqrt => FUnOp::Sqrt,
        VFloatUnOp::Ceil => FUnOp::Ceil,
        VFloatUnOp::Floor => FUnOp::Floor,
        VFloatUnOp::Trunc => FUnOp::Trunc,
        VFloatUnOp::Nearest => FUnOp::Nearest,
    }
}

/// `<f-shape>.<cmp>`: per-lane float comparison → an all-ones (true) / all-zeros (false) mask of the
/// lane width. Rust's `==`/`!=`/`<`/… already match wasm's ordered (`ne` unordered) NaN behaviour.
fn simd_vfloat_cmp(shape: VShape, op: VFCmpOp, a: [u8; 16], b: [u8; 16]) -> [u8; 16] {
    let mut o = [0u8; 16];
    match shape {
        VShape::F32x4 => {
            for i in 0..4 {
                let x = f32::from_bits(lane_read(&a, i, 4) as u32);
                let y = f32::from_bits(lane_read(&b, i, 4) as u32);
                let t = match op {
                    VFCmpOp::Eq => x == y,
                    VFCmpOp::Ne => x != y,
                    VFCmpOp::Lt => x < y,
                    VFCmpOp::Gt => x > y,
                    VFCmpOp::Le => x <= y,
                    VFCmpOp::Ge => x >= y,
                };
                lane_write(&mut o, i, 4, if t { u64::MAX } else { 0 });
            }
        }
        VShape::F64x2 => {
            for i in 0..2 {
                let x = f64::from_bits(lane_read(&a, i, 8));
                let y = f64::from_bits(lane_read(&b, i, 8));
                let t = match op {
                    VFCmpOp::Eq => x == y,
                    VFCmpOp::Ne => x != y,
                    VFCmpOp::Lt => x < y,
                    VFCmpOp::Gt => x > y,
                    VFCmpOp::Le => x <= y,
                    VFCmpOp::Ge => x >= y,
                };
                lane_write(&mut o, i, 8, if t { u64::MAX } else { 0 });
            }
        }
        // Verifier rejects an integer shape here; total fall-through returns zero.
        _ => {}
    }
    o
}

/// `<i-shape>.{shl,shr_s,shr_u}`: shift every lane by the same scalar amount, taken modulo the lane
/// bit-width (the wasm rule). `shl`/`shr_u` are logical on the zero-extended lane; `shr_s` is
/// arithmetic on the sign-extended lane.
fn simd_vshift(shape: VShape, op: VShiftOp, a: [u8; 16], amt: u32) -> [u8; 16] {
    let bytes = shape.lane_bytes() as usize;
    let sh = amt & (bytes as u32 * 8 - 1);
    let mut o = [0u8; 16];
    for i in 0..shape.lanes() as usize {
        let x = lane_read(&a, i, bytes);
        let r = match op {
            VShiftOp::Shl => x << sh,
            VShiftOp::ShrU => x >> sh,
            VShiftOp::ShrS => (lane_sext(x, bytes) >> sh) as u64,
        };
        lane_write(&mut o, i, bytes, r);
    }
    o
}

/// Lane int↔float / float↔float conversions. Rust's `as` casts already match wasm: int→float is
/// round-to-nearest, float→int is `trunc_sat` (NaN→0, clamp to the int range), and `f64 as f32` is
/// the IEEE round demote. `demote`/`promote_low` touch the low 2 lanes; demote zeroes lanes 2/3.
fn simd_convert(op: VCvtOp, a: [u8; 16]) -> [u8; 16] {
    let mut o = [0u8; 16];
    match op {
        VCvtOp::F32x4ConvertI32x4S => {
            for i in 0..4 {
                let x = lane_read(&a, i, 4) as u32 as i32;
                lane_write(&mut o, i, 4, (x as f32).to_bits() as u64);
            }
        }
        VCvtOp::F32x4ConvertI32x4U => {
            for i in 0..4 {
                let x = lane_read(&a, i, 4) as u32;
                lane_write(&mut o, i, 4, (x as f32).to_bits() as u64);
            }
        }
        VCvtOp::I32x4TruncSatF32x4S => {
            for i in 0..4 {
                let x = f32::from_bits(lane_read(&a, i, 4) as u32);
                lane_write(&mut o, i, 4, (x as i32) as u32 as u64);
            }
        }
        VCvtOp::I32x4TruncSatF32x4U => {
            for i in 0..4 {
                let x = f32::from_bits(lane_read(&a, i, 4) as u32);
                lane_write(&mut o, i, 4, (x as u32) as u64);
            }
        }
        VCvtOp::F32x4DemoteF64x2Zero => {
            for i in 0..2 {
                let x = f64::from_bits(lane_read(&a, i, 8));
                lane_write(&mut o, i, 4, (x as f32).to_bits() as u64);
            }
            // lanes 2/3 stay zero.
        }
        VCvtOp::F64x2PromoteLowF32x4 => {
            for i in 0..2 {
                let x = f32::from_bits(lane_read(&a, i, 4) as u32);
                lane_write(&mut o, i, 8, (x as f64).to_bits());
            }
        }
        VCvtOp::F64x2ConvertLowI32x4S => {
            for i in 0..2 {
                let x = lane_read(&a, i, 4) as u32 as i32;
                lane_write(&mut o, i, 8, (x as f64).to_bits());
            }
        }
        VCvtOp::F64x2ConvertLowI32x4U => {
            for i in 0..2 {
                let x = lane_read(&a, i, 4) as u32;
                lane_write(&mut o, i, 8, (x as f64).to_bits());
            }
        }
        VCvtOp::I32x4TruncSatF64x2SZero => {
            for i in 0..2 {
                let x = f64::from_bits(lane_read(&a, i, 8));
                lane_write(&mut o, i, 4, (x as i32) as u32 as u64);
            }
            // lanes 2/3 stay zero.
        }
        VCvtOp::I32x4TruncSatF64x2UZero => {
            for i in 0..2 {
                let x = f64::from_bits(lane_read(&a, i, 8));
                lane_write(&mut o, i, 4, (x as u32) as u64);
            }
            // lanes 2/3 stay zero.
        }
    }
    o
}

/// `<narrow>.narrow_{s,u}`: saturate every lane of two wide sources to the narrow width and
/// concatenate (`a` then `b`). The source is read as **signed**; `S`/`U` pick the saturation range.
fn simd_narrow(out: VShape, op: VNarrowOp, a: [u8; 16], b: [u8; 16]) -> [u8; 16] {
    let out_bytes = out.lane_bytes() as usize;
    let src = out.wider().expect("verifier ensures a wider source");
    let src_bytes = src.lane_bytes() as usize;
    let src_lanes = src.lanes() as usize; // = out.lanes() / 2
    let bits = out_bytes as u32 * 8;
    let (min, max) = match op {
        VNarrowOp::S => (-(1i128 << (bits - 1)), (1i128 << (bits - 1)) - 1),
        VNarrowOp::U => (0i128, (1i128 << bits) - 1),
    };
    let mut o = [0u8; 16];
    for i in 0..src_lanes {
        let s = lane_sext(lane_read(&a, i, src_bytes), src_bytes) as i128;
        lane_write(&mut o, i, out_bytes, s.clamp(min, max) as u64);
    }
    for i in 0..src_lanes {
        let s = lane_sext(lane_read(&b, i, src_bytes), src_bytes) as i128;
        lane_write(&mut o, src_lanes + i, out_bytes, s.clamp(min, max) as u64);
    }
    o
}

/// `<wide>.extend_{low,high}_{s,u}`: take the low or high half of the (half-width) source lanes and
/// sign/zero-extend each to the wide lane width.
fn simd_widen(out: VShape, op: VWidenOp, a: [u8; 16]) -> [u8; 16] {
    let (low, signed) = op.parts();
    let out_bytes = out.lane_bytes() as usize;
    let src_bytes = out_bytes / 2;
    let n = out.lanes() as usize; // result lanes = source lanes we consume
    let base = if low { 0 } else { n }; // the low or high half of the source lanes
    let mut o = [0u8; 16];
    for i in 0..n {
        let s = lane_read(&a, base + i, src_bytes);
        let v = if signed {
            lane_sext(s, src_bytes) as u64
        } else {
            s
        };
        lane_write(&mut o, i, out_bytes, v);
    }
    o
}

/// `<wide>.extmul_{low,high}_<src>_{s,u}`: widen the low/high half of both operands (sign/zero per
/// the op) and multiply lane-wise into the wide result. Products are computed in `i128` so they
/// can't overflow before the wrapping write at the wide lane width.
fn simd_extmul(out: VShape, op: VWidenOp, a: [u8; 16], b: [u8; 16]) -> [u8; 16] {
    let (low, signed) = op.parts();
    let out_bytes = out.lane_bytes() as usize;
    let src_bytes = out_bytes / 2;
    let n = out.lanes() as usize;
    let base = if low { 0 } else { n };
    let widen = |raw: u64| -> i128 {
        if signed {
            lane_sext(raw, src_bytes) as i128
        } else {
            raw as i128
        }
    };
    let mut o = [0u8; 16];
    for i in 0..n {
        let x = widen(lane_read(&a, base + i, src_bytes));
        let y = widen(lane_read(&b, base + i, src_bytes));
        lane_write(&mut o, i, out_bytes, (x * y) as u64);
    }
    o
}

/// `<wide>.extadd_pairwise_<src>_{s,u}`: widen every source lane (sign/zero) and sum adjacent pairs
/// into the wide result — `out[i] = w(a[2i]) + w(a[2i+1])`.
fn simd_extadd_pairwise(out: VShape, signed: bool, a: [u8; 16]) -> [u8; 16] {
    let out_bytes = out.lane_bytes() as usize;
    let src_bytes = out_bytes / 2;
    let n = out.lanes() as usize;
    let widen = |raw: u64| -> i128 {
        if signed {
            lane_sext(raw, src_bytes) as i128
        } else {
            raw as i128
        }
    };
    let mut o = [0u8; 16];
    for i in 0..n {
        let lo = widen(lane_read(&a, 2 * i, src_bytes));
        let hi = widen(lane_read(&a, 2 * i + 1, src_bytes));
        lane_write(&mut o, i, out_bytes, (lo + hi) as u64);
    }
    o
}

/// `i16x8.q15mulr_sat_s`: signed Q15 fixed-point multiply with rounding and saturation —
/// `out[i] = sat_i16((a[i]·b[i] + 0x4000) >> 15)`.
fn simd_q15mulr(a: [u8; 16], b: [u8; 16]) -> [u8; 16] {
    let mut o = [0u8; 16];
    for i in 0..8 {
        let x = lane_sext(lane_read(&a, i, 2), 2);
        let y = lane_sext(lane_read(&b, i, 2), 2);
        let r = (x * y + 0x4000) >> 15;
        let sat = r.clamp(i16::MIN as i64, i16::MAX as i64);
        lane_write(&mut o, i, 2, sat as u16 as u64);
    }
    o
}

/// `<i-shape>.{add,sub}_sat_{s,u}`: per-lane add/sub that **clamps** to the lane's signed/unsigned
/// range instead of wrapping. Computed in `i128` so the intermediate can't overflow. `i8x16`/`i16x8`.
fn simd_vsat_bin(shape: VShape, op: VSatBinOp, a: [u8; 16], b: [u8; 16]) -> [u8; 16] {
    let bytes = shape.lane_bytes() as usize;
    let bits = bytes as u32 * 8;
    let max_u = (1i128 << bits) - 1;
    let max_s = (1i128 << (bits - 1)) - 1;
    let min_s = -(1i128 << (bits - 1));
    let mut o = [0u8; 16];
    for i in 0..shape.lanes() as usize {
        let (xu, yu) = (
            lane_read(&a, i, bytes) as i128,
            lane_read(&b, i, bytes) as i128,
        );
        let (xs, ys) = (
            lane_sext(lane_read(&a, i, bytes), bytes) as i128,
            lane_sext(lane_read(&b, i, bytes), bytes) as i128,
        );
        let r = match op {
            VSatBinOp::AddU => (xu + yu).min(max_u),
            VSatBinOp::SubU => (xu - yu).max(0),
            VSatBinOp::AddS => (xs + ys).clamp(min_s, max_s),
            VSatBinOp::SubS => (xs - ys).clamp(min_s, max_s),
        };
        lane_write(&mut o, i, bytes, r as u64);
    }
    o
}

/// `<i-shape>.avgr_u`: per-lane unsigned rounding average `(a + b + 1) >> 1`, computed wide so the
/// `+1` can't overflow. `i8x16`/`i16x8`.
fn simd_avgr(shape: VShape, a: [u8; 16], b: [u8; 16]) -> [u8; 16] {
    let bytes = shape.lane_bytes() as usize;
    let mut o = [0u8; 16];
    for i in 0..shape.lanes() as usize {
        let x = lane_read(&a, i, bytes);
        let y = lane_read(&b, i, bytes);
        lane_write(&mut o, i, bytes, (x + y + 1) >> 1);
    }
    o
}

/// `i32x4.dot_i16x8_s`: signed dot product of adjacent `i16` pairs into `i32` lanes —
/// `out[i] = a[2i]·b[2i] + a[2i+1]·b[2i+1]`. Products are computed in `i32`; the pair sum can
/// overflow `i32` only for the `(-32768)·(-32768)` corner doubled, which wraps (matches wasm).
fn simd_dot(a: [u8; 16], b: [u8; 16]) -> [u8; 16] {
    let mut o = [0u8; 16];
    for i in 0..4 {
        let a0 = lane_sext(lane_read(&a, 2 * i, 2), 2) as i32;
        let a1 = lane_sext(lane_read(&a, 2 * i + 1, 2), 2) as i32;
        let b0 = lane_sext(lane_read(&b, 2 * i, 2), 2) as i32;
        let b1 = lane_sext(lane_read(&b, 2 * i + 1, 2), 2) as i32;
        let r = a0.wrapping_mul(b0).wrapping_add(a1.wrapping_mul(b1));
        lane_write(&mut o, i, 4, r as u32 as u64);
    }
    o
}

/// `i16x8.dot_i8x16_s`: signed `i8` dot of adjacent pairs into `i16` lanes (wrapping) — the
/// deterministic `relaxed_dot_i8x16_i7x16_s`. Products of `i8`s fit in `i16`; the pair sum wraps.
fn simd_dot_i8(a: [u8; 16], b: [u8; 16]) -> [u8; 16] {
    let mut o = [0u8; 16];
    for j in 0..8 {
        let a0 = lane_sext(lane_read(&a, 2 * j, 1), 1) as i32;
        let a1 = lane_sext(lane_read(&a, 2 * j + 1, 1), 1) as i32;
        let b0 = lane_sext(lane_read(&b, 2 * j, 1), 1) as i32;
        let b1 = lane_sext(lane_read(&b, 2 * j + 1, 1), 1) as i32;
        let r = a0 * b0 + a1 * b1; // exact in i32; wraps when written at i16 width
        lane_write(&mut o, j, 2, r as u16 as u64);
    }
    o
}

/// `<shape>.all_true`: `1` iff every lane is non-zero.
fn simd_all_true(shape: VShape, a: [u8; 16]) -> i32 {
    let bytes = shape.lane_bytes() as usize;
    (0..shape.lanes() as usize).all(|i| lane_read(&a, i, bytes) != 0) as i32
}

/// `<shape>.bitmask`: lane `i`'s high (sign) bit → bit `i` of the result.
fn simd_bitmask(shape: VShape, a: [u8; 16]) -> i32 {
    let bytes = shape.lane_bytes() as usize;
    let top = bytes as u32 * 8 - 1;
    let mut m = 0i32;
    for i in 0..shape.lanes() as usize {
        m |= (((lane_read(&a, i, bytes) >> top) & 1) as i32) << i;
    }
    m
}

/// `<i-shape>.{abs,neg}`: per-lane two's-complement `|x|` / `0 - x` (both wrapping at the lane
/// width — `abs(INT_MIN) == INT_MIN`, matching wasm/hardware).
fn simd_vint_un(shape: VShape, op: VIntUnOp, a: [u8; 16]) -> [u8; 16] {
    let bytes = shape.lane_bytes() as usize;
    let mut o = [0u8; 16];
    for i in 0..shape.lanes() as usize {
        let x = lane_sext(lane_read(&a, i, bytes), bytes);
        let r = match op {
            VIntUnOp::Abs => x.wrapping_abs(),
            VIntUnOp::Neg => x.wrapping_neg(),
        };
        lane_write(&mut o, i, bytes, r as u64);
    }
    o
}

fn simd_vfloat_bin(shape: VShape, op: VFloatBinOp, a: [u8; 16], b: [u8; 16]) -> [u8; 16] {
    let mut o = [0u8; 16];
    match shape {
        VShape::F32x4 => {
            for i in 0..4 {
                let x = f32::from_bits(lane_read(&a, i, 4) as u32);
                let y = f32::from_bits(lane_read(&b, i, 4) as u32);
                lane_write(&mut o, i, 4, fbin32(vf_bin(op), x, y).to_bits() as u64);
            }
        }
        VShape::F64x2 => {
            for i in 0..2 {
                let x = f64::from_bits(lane_read(&a, i, 8));
                let y = f64::from_bits(lane_read(&b, i, 8));
                lane_write(&mut o, i, 8, fbin64(vf_bin(op), x, y).to_bits());
            }
        }
        // Verifier rejects an integer shape here; total fall-through returns zero.
        _ => {}
    }
    o
}

/// Lane-wise fused multiply-add (`relaxed_madd`/`nmadd`): `±a·b + c` with a single rounding.
/// `f*::mul_add` is the correctly-rounded IEEE-754 FMA — bit-identical to Cranelift's `fma`, so the
/// interp↔JIT differential holds. `neg` negates the product (the `nmadd` form, `−a·b + c`).
fn simd_fma(shape: VShape, neg: bool, a: [u8; 16], b: [u8; 16], c: [u8; 16]) -> [u8; 16] {
    let mut o = [0u8; 16];
    match shape {
        VShape::F32x4 => {
            for i in 0..4 {
                let x = f32::from_bits(lane_read(&a, i, 4) as u32);
                let y = f32::from_bits(lane_read(&b, i, 4) as u32);
                let z = f32::from_bits(lane_read(&c, i, 4) as u32);
                let x = if neg { -x } else { x };
                lane_write(&mut o, i, 4, x.mul_add(y, z).to_bits() as u64);
            }
        }
        VShape::F64x2 => {
            for i in 0..2 {
                let x = f64::from_bits(lane_read(&a, i, 8));
                let y = f64::from_bits(lane_read(&b, i, 8));
                let z = f64::from_bits(lane_read(&c, i, 8));
                let x = if neg { -x } else { x };
                lane_write(&mut o, i, 8, x.mul_add(y, z).to_bits());
            }
        }
        _ => {}
    }
    o
}

fn simd_pminmax(shape: VShape, op: VPMinMaxOp, a: [u8; 16], b: [u8; 16]) -> [u8; 16] {
    // wasm pmin/pmax: pseudo-min/max defined as a one-sided compare-and-select.
    //   pmin(a, b) = b < a ? b : a
    //   pmax(a, b) = a < b ? b : a
    // This propagates NaN from the second operand and returns -0/+0 per the
    // chosen operand (no IEEE min/max canonicalization).
    let mut o = [0u8; 16];
    match shape {
        VShape::F32x4 => {
            for i in 0..4 {
                let x = f32::from_bits(lane_read(&a, i, 4) as u32);
                let y = f32::from_bits(lane_read(&b, i, 4) as u32);
                let r = match op {
                    VPMinMaxOp::Pmin => {
                        if y < x {
                            y
                        } else {
                            x
                        }
                    }
                    VPMinMaxOp::Pmax => {
                        if x < y {
                            y
                        } else {
                            x
                        }
                    }
                };
                lane_write(&mut o, i, 4, r.to_bits() as u64);
            }
        }
        VShape::F64x2 => {
            for i in 0..2 {
                let x = f64::from_bits(lane_read(&a, i, 8));
                let y = f64::from_bits(lane_read(&b, i, 8));
                let r = match op {
                    VPMinMaxOp::Pmin => {
                        if y < x {
                            y
                        } else {
                            x
                        }
                    }
                    VPMinMaxOp::Pmax => {
                        if x < y {
                            y
                        } else {
                            x
                        }
                    }
                };
                lane_write(&mut o, i, 8, r.to_bits());
            }
        }
        // Verifier rejects an integer shape here; total fall-through returns zero.
        _ => {}
    }
    o
}

fn simd_vfloat_un(shape: VShape, op: VFloatUnOp, a: [u8; 16]) -> [u8; 16] {
    let mut o = [0u8; 16];
    match shape {
        VShape::F32x4 => {
            for i in 0..4 {
                let x = f32::from_bits(lane_read(&a, i, 4) as u32);
                lane_write(&mut o, i, 4, fun32(vf_un(op), x).to_bits() as u64);
            }
        }
        VShape::F64x2 => {
            for i in 0..2 {
                let x = f64::from_bits(lane_read(&a, i, 8));
                lane_write(&mut o, i, 8, fun64(vf_un(op), x).to_bits());
            }
        }
        _ => {}
    }
    o
}

fn simd_vbit_bin(op: VBitBinOp, a: [u8; 16], b: [u8; 16]) -> [u8; 16] {
    let mut o = [0u8; 16];
    for i in 0..16 {
        o[i] = match op {
            VBitBinOp::And => a[i] & b[i],
            VBitBinOp::Or => a[i] | b[i],
            VBitBinOp::Xor => a[i] ^ b[i],
            VBitBinOp::AndNot => a[i] & !b[i],
        };
    }
    o
}

/// `v128.bitselect`: per-bit `(a & mask) | (b & !mask)`.
fn simd_bitselect(a: [u8; 16], b: [u8; 16], mask: [u8; 16]) -> [u8; 16] {
    let mut o = [0u8; 16];
    for i in 0..16 {
        o[i] = (a[i] & mask[i]) | (b[i] & !mask[i]);
    }
    o
}

/// `i8x16.shuffle`: result byte `i` is byte `lanes[i]` of the concatenation `a ++ b`.
fn simd_shuffle(lanes: &[u8; 16], a: [u8; 16], b: [u8; 16]) -> [u8; 16] {
    let mut o = [0u8; 16];
    for i in 0..16 {
        let sel = lanes[i] as usize;
        o[i] = if sel < 16 {
            a[sel]
        } else if sel < 32 {
            b[sel - 16]
        } else {
            0 // verifier rejects ≥32; total fall-through
        };
    }
    o
}

/// `i8x16.swizzle`: result byte `i` is `a[b[i]]` when `b[i] < 16`, else `0`.
fn simd_swizzle(a: [u8; 16], b: [u8; 16]) -> [u8; 16] {
    let mut o = [0u8; 16];
    for i in 0..16 {
        let sel = b[i] as usize;
        o[i] = if sel < 16 { a[sel] } else { 0 };
    }
    o
}

/// Resolve a `call_indirect`: mask the index into the power-of-two-padded function
/// table, then check the selected entry's signature against `ty` (the §3c table
/// type-id check). Masking — not branching — keeps the table load Spectre-v1 safe.
fn table_lookup(funcs: &[Func], idx: i32, ty: &FuncType) -> Result<FuncIdx, Trap> {
    let mask = funcs.len().next_power_of_two() - 1;
    let slot = (idx as u32 as usize) & mask;
    match funcs.get(slot) {
        Some(c) if c.params == ty.params && c.results == ty.results => Ok(slot as FuncIdx),
        _ => Err(Trap::IndirectCallType),
    }
}

fn fbin32(op: FBinOp, a: f32, b: f32) -> f32 {
    match op {
        FBinOp::Add => a + b,
        FBinOp::Sub => a - b,
        FBinOp::Mul => a * b,
        FBinOp::Div => a / b,
        FBinOp::Min => fmin32(a, b),
        FBinOp::Max => fmax32(a, b),
        FBinOp::Copysign => a.copysign(b),
    }
}

fn fbin64(op: FBinOp, a: f64, b: f64) -> f64 {
    match op {
        FBinOp::Add => a + b,
        FBinOp::Sub => a - b,
        FBinOp::Mul => a * b,
        FBinOp::Div => a / b,
        FBinOp::Min => fmin64(a, b),
        FBinOp::Max => fmax64(a, b),
        FBinOp::Copysign => a.copysign(b),
    }
}

fn fun32(op: FUnOp, a: f32) -> f32 {
    match op {
        FUnOp::Abs => a.abs(),
        FUnOp::Neg => -a,
        FUnOp::Sqrt => a.sqrt(),
        FUnOp::Ceil => a.ceil(),
        FUnOp::Floor => a.floor(),
        FUnOp::Trunc => a.trunc(),
        FUnOp::Nearest => a.round_ties_even(),
    }
}

fn fun64(op: FUnOp, a: f64) -> f64 {
    match op {
        FUnOp::Abs => a.abs(),
        FUnOp::Neg => -a,
        FUnOp::Sqrt => a.sqrt(),
        FUnOp::Ceil => a.ceil(),
        FUnOp::Floor => a.floor(),
        FUnOp::Trunc => a.trunc(),
        FUnOp::Nearest => a.round_ties_even(),
    }
}

fn fcmp32(op: FCmpOp, a: f32, b: f32) -> bool {
    match op {
        FCmpOp::Eq => a == b,
        FCmpOp::Ne => a != b,
        FCmpOp::Lt => a < b,
        FCmpOp::Le => a <= b,
        FCmpOp::Gt => a > b,
        FCmpOp::Ge => a >= b,
    }
}

fn fcmp64(op: FCmpOp, a: f64, b: f64) -> bool {
    match op {
        FCmpOp::Eq => a == b,
        FCmpOp::Ne => a != b,
        FCmpOp::Lt => a < b,
        FCmpOp::Le => a <= b,
        FCmpOp::Gt => a > b,
        FCmpOp::Ge => a >= b,
    }
}

// wasm min/max: NaN propagates; for ±0, min prefers -0 and max prefers +0.
fn fmin32(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        f32::NAN
    } else if a == b {
        if a.is_sign_negative() {
            a
        } else {
            b
        }
    } else if a < b {
        a
    } else {
        b
    }
}
fn fmax32(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        f32::NAN
    } else if a == b {
        if a.is_sign_negative() {
            b
        } else {
            a
        }
    } else if a > b {
        a
    } else {
        b
    }
}
fn fmin64(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if a == b {
        if a.is_sign_negative() {
            a
        } else {
            b
        }
    } else if a < b {
        a
    } else {
        b
    }
}
fn fmax64(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if a == b {
        if a.is_sign_negative() {
            b
        } else {
            a
        }
    } else if a > b {
        a
    } else {
        b
    }
}

// Float→int casts are saturating with NaN→0 (Rust `as` matches wasm `trunc_sat`).
fn fto_i(op: FToI, a: Reg) -> Reg {
    match op {
        FToI::F32I32S => Reg::from_i32(a.f32() as i32),
        FToI::F32I32U => Reg::from_i32(a.f32() as u32 as i32),
        FToI::F32I64S => Reg::from_i64(a.f32() as i64),
        FToI::F32I64U => Reg::from_i64(a.f32() as u64 as i64),
        FToI::F64I32S => Reg::from_i32(a.f64() as i32),
        FToI::F64I32U => Reg::from_i32(a.f64() as u32 as i32),
        FToI::F64I64S => Reg::from_i64(a.f64() as i64),
        FToI::F64I64U => Reg::from_i64(a.f64() as u64 as i64),
    }
}

/// Trapping float→int conversion (`trunc`, vs the saturating `trunc_sat`): NaN and
/// out-of-range inputs trap. Work in `f64` (promoting `f32` is exact), and trap
/// unless the truncation toward zero fits the target — `f > MIN-1 && f < MAX+1`
/// (using the exact float boundary constants; the `i64` signed lower bound is
/// closed because `-2^63 - 1` is not representable and rounds to `-2^63`).
fn trunc_trap(op: FToI, a: Reg) -> Result<Reg, Trap> {
    let (from, to, signed) = op.parts();
    let f: f64 = match from {
        FloatTy::F32 => a.f32() as f64,
        FloatTy::F64 => a.f64(),
    };
    if f.is_nan() {
        return Err(Trap::BadConversion);
    }
    // Bounds are written as explicit comparisons so the open-vs-closed distinction is
    // visible: the i64-signed *lower* bound is closed (`>=`) because `-2^63 - 1` is
    // not representable and rounds to `-2^63`; the rest are open.
    #[allow(clippy::manual_range_contains)]
    let in_range = match (to, signed) {
        (IntTy::I32, true) => f > -2_147_483_649.0 && f < 2_147_483_648.0,
        (IntTy::I32, false) => f > -1.0 && f < 4_294_967_296.0,
        (IntTy::I64, true) => f >= -9_223_372_036_854_775_808.0 && f < 9_223_372_036_854_775_808.0,
        (IntTy::I64, false) => f > -1.0 && f < 18_446_744_073_709_551_616.0,
    };
    if !in_range {
        return Err(Trap::BadConversion);
    }
    // In range, so the cast is exact (truncating toward zero, no saturation).
    Ok(match (to, signed) {
        (IntTy::I32, true) => Reg::from_i32(f as i32),
        (IntTy::I32, false) => Reg::from_i32(f as u32 as i32),
        (IntTy::I64, true) => Reg::from_i64(f as i64),
        (IntTy::I64, false) => Reg::from_i64(f as u64 as i64),
    })
}

fn i_to_f(op: IToF, a: Reg) -> Reg {
    match op {
        IToF::I32F32S => Reg::from_f32(a.i32() as f32),
        IToF::I32F32U => Reg::from_f32(a.i32() as u32 as f32),
        IToF::I64F32S => Reg::from_f32(a.i64() as f32),
        IToF::I64F32U => Reg::from_f32(a.i64() as u64 as f32),
        IToF::I32F64S => Reg::from_f64(a.i32() as f64),
        IToF::I32F64U => Reg::from_f64(a.i32() as u32 as f64),
        IToF::I64F64S => Reg::from_f64(a.i64() as f64),
        IToF::I64F64U => Reg::from_f64(a.i64() as u64 as f64),
    }
}

fn cast(op: CastOp, a: Reg) -> Reg {
    match op {
        CastOp::Demote => Reg::from_f32(a.f64() as f32),
        CastOp::Promote => Reg::from_f64(a.f32() as f64),
        CastOp::ReinterpI32F32 => Reg::from_f32(f32::from_bits(a.i32() as u32)),
        CastOp::ReinterpF32I32 => Reg::from_i32(a.f32().to_bits() as i32),
        CastOp::ReinterpI64F64 => Reg::from_f64(f64::from_bits(a.i64() as u64)),
        CastOp::ReinterpF64I64 => Reg::from_i64(a.f64().to_bits() as i64),
    }
}

// ----------------------------------------------------------------------------
// Capabilities — the host-owned handle table + a deterministic mock powerbox
// (§3c index model, §3e MVP interface set). This is the reference oracle's
// stand-in for real host capabilities: deterministic, in-process, so it can be a
// differential oracle. The *security* of the model lives in `Host::resolve`
// (use-site mask + type_id + generation check → forged indices are inert).
// ----------------------------------------------------------------------------

/// MVP interface type-ids (§3e). Phase-1: a `type_id` is just a small constant a
/// handle-table entry carries and `cap.call` re-checks. (A module-level interface
/// section that globalizes ids across linked modules is deferred to §13.)
pub mod cap_id {
    /// `Stream` — byte stream: op 0 `read`, op 1 `write`, op 2 `close` (§3e D43).
    pub const STREAM: u32 = 0;
    /// `Exit` — lifecycle: op 0 `exit(code)` (noreturn).
    pub const EXIT: u32 = 1;
    /// `Clock` — op 0 `now(clock_id) -> i64` nanoseconds.
    pub const CLOCK: u32 = 2;
    /// `Memory` — op 0 `map`, 1 `unmap`, 2 `protect`, 3 `page_size` (§3e; real page protection —
    /// see `Mem`).
    pub const MEMORY: u32 = 3;
    /// `SharedRegion` — a host-backed memory object aliased into the window (§13). op 0
    /// `map(window_offset, region_offset, len, prot)` aliases the region's pages into the window
    /// (the same backing may be mapped at *multiple* window offsets → zero-overhead aliasing, the
    /// magic-ring-buffer primitive); op 1 `unmap(window_offset, len)` drops the alias; op 2
    /// `len() -> i64` reports the region size; op 3 `page_size() -> i64`. Granting the handle is how
    /// two domains come to share memory; `create`/`grant` (guest-minted regions, cross-domain) are a
    /// §14 follow-up — today regions are host-granted, like `Memory`. A backing may be a fresh OS
    /// shared object (`memfd`) **or a real host file** (`svm-run`'s `FileBacking`, minted by an
    /// mmap-capable fs cap): mapping the latter aliases the file into the window zero-copy — the
    /// file-backed-mmap bridge (MMAP_CAPABILITY.md §4b).
    pub const SHARED_REGION: u32 = 4;
    /// `AddressSpace` — the §14 memory-management capability, **attenuable to a power-of-two
    /// window sub-range** `[base, base+size)`. Like `Memory` but every op is confined to the
    /// holder's sub-range (offsets are sub-range-relative, shifted by `base`): op 0 `map(off,len,prot)`,
    /// 1 `unmap(off,len)`, 2 `protect(off,len,prot)`, 3 `page_size() -> i64`, and 4
    /// **`sub(off, size_log2) -> handle`** — the **attenuation** primitive: mint a child `AddressSpace`
    /// over the power-of-two-aligned sub-range `[base+off, base+off + 2^size_log2)`, which must lie
    /// within the holder's range (a parent can only sub-allocate what it holds, §14). This is the
    /// memory half of the `Instantiator`: a guest carves a child's window from its own.
    pub const ADDRESS_SPACE: u32 = 5;
    /// `Instantiator` — the §14 nesting primitive: spawn a **child domain** confined to a
    /// power-of-two sub-window `[base, base+size)` of the holder's window (VM-in-VM). op 0
    /// `instantiate(entry, off, size_log2, fuel) -> child_handle` enqueues a child vCPU running the
    /// same module's `entry` (which returns one `i64` and takes one or two — its starter caps)
    /// confined to `[base+off, base+off+2^size_log2)` with an **attenuated** powerbox over the child's
    /// own window: an `Instantiator` (so it can recurse — confinement composes to any depth) and an
    /// `AddressSpace` (so it can manage its own pages), passed as the entry's arguments. A fuel quota
    /// caps it; returns immediately (non-blocking). op 1 `join(child_handle) -> result` parks **only
    /// the calling fiber** until that child finishes, then yields its result (siblings keep running —
    /// the child rides the same §12 executor). Holding the handle is the authority to nest (D19: a
    /// child can only get what the parent sub-allocates).
    pub const INSTANTIATOR: u32 = 6;
    /// `Yielder` — a §14 **co-fiber** child's handle back to its instantiator-parent. op 0
    /// `yield(value: i64) -> resumed: i64` suspends the child, handing `value` to the parent's
    /// `resume` (which returns it as the yield's status/value), and on the next `resume` returns the
    /// value the parent passed. The cooperative-coroutine primitive the §14 parent-virtualized-fault /
    /// lazy-paging model builds on (a child parks on a fault it cannot service; the parent supplies the
    /// page and resumes it). Granted to a coroutine child (`Instantiator.spawn_coroutine`) only.
    pub const YIELDER: u32 = 7;
    /// `Module` — a host-granted, host-**verified** module a guest may instantiate (§14). The handle
    /// confers only the authority to pass it to the `Instantiator`'s module ops (5/6/7 —
    /// `instantiate_module` / `spawn_coroutine_module` / `spawn_demand_coroutine_module`), which
    /// spawn a child domain running *that* module's code confined to a carve of the holder's window
    /// — the "plugin-in-plugin" story: a guest can only instantiate modules it was given (no ambient
    /// authority). It has no directly callable ops (`cap.call` on it is an inert `CapFault`).
    pub const MODULE: u32 = 8;
    /// §9/§12 `IoRing` — the submit/complete ring. `op 0 submit(sq_ptr, n, cq_ptr)` runs `n`
    /// deferred `cap.call`s (each a 64-byte SQE in the window) and writes their results as 32-byte
    /// CQEs, amortizing the boundary crossing — and, for *blocking* SQEs, **overlapping** them on a
    /// bounded host offload pool ([`OFFLOAD_POOL_THREADS`] threads; the §12 increment-2 win).
    pub const IO_RING: u32 = 9;
    /// §12 `Blocking` — a *mock* synchronous-only / blocking host capability (DNS-/FS-blocking-shaped)
    /// whose op 0 `work(arg) -> mix(arg)` is **window-independent and `&mut Host`-free**, so a
    /// `submit` batch can hand it to the offload pool instead of the guest's vCPU thread. Op 0 is also
    /// a perfectly ordinary synchronous `cap.call` (it then blocks the caller — the degenerate path).
    pub const BLOCKING: u32 = 10;
    /// `Jit` — the guest-driven JIT capability (DESIGN.md §22): submit serialized IR at runtime to
    /// be validated (decode + verify + the memory-match precondition, via the host-injected
    /// [`crate::JitValidator`]) and compiled into the **same domain** (same window, same powerbox —
    /// a module is not an isolation unit, DESIGN §8). op 0 `compile(ptr, len) -> code_handle | -errno`
    /// (fail-closed: nothing is installed on any validation failure); op 1
    /// `invoke(code_handle, args…) -> results` runs the compiled unit's entry (`funcs[0]`) over the
    /// caller's **live window** — serviced by the eval loop on the interpreter (it must run guest
    /// code, which the generic dispatch can't) and by the embedder's cap thunk on the JIT (it calls
    /// the unit's native trampoline); traps in invoked code are **terminal for the domain**; op 2
    /// `release(code_handle) -> 0 | -errno` revokes the handle (no code reclaim yet — DESIGN.md §22
    /// "Code reclaim"); op 3 `install(code_handle) -> slot_index | -errno` (Model B2) installs the
    /// unit into the `call_indirect` table's next reserved slot so old code (or another unit) can
    /// dispatch it at native speed (old→new), `-ENOSPC` if the table is full; op 4
    /// `uninstall(slot) -> 0 | -errno` clears an installed slot so the index is reusable and a
    /// stale `call_indirect` of it traps (slot reclaim — the code memory itself is not freed).
    pub const JIT: u32 = 11;
    /// `CompiledCode` — a unit minted by `Jit.compile`. Like `Module`, it has no directly callable
    /// ops (`cap.call` on it is an inert `CapFault`); it confers only the authority to be named in
    /// `Jit.invoke`/`release` on the domain handle that compiled it.
    pub const JIT_CODE: u32 = 12;
    /// `HostFn` — an **embedder-registered** capability (§7 "host-defined capabilities"): the host
    /// installs a handler closure with [`crate::Host::grant_host_fn`] and the guest reaches it like
    /// any capability (`cap.call HOST_FN op …`). The interface's *semantics* live entirely in the
    /// embedder's closure (e.g. an `svm-wasi` shim), **outside** this crate's TCB match — so a host
    /// can add capabilities without touching the VM. The handler reads/writes the guest window
    /// through the same masked `GuestMem` the built-in ops use (authority-TCB, not escape-TCB).
    pub const HOST_FN: u32 = 13;
    /// §15 / PROCESS.md §5 `Budget` — a passable, **splittable** resource-quota vector (fuel / mem /
    /// spawn), §15's "every meterable resource is a capability with a quota" promoted to an object.
    /// op 0 `split(fuel, mem, spawn) -> sub_handle | -errno`: mint a child `Budget` holding those
    /// amounts, **deducted** from the holder's remaining — attenuation (a child can never exceed the
    /// parent, D19); a field of `-1` means "all remaining"; asking for more than remains is `-EINVAL`.
    /// op 1 `read(field) -> remaining | -EINVAL`: report one field's remaining quota (`0` fuel, `1`
    /// mem, `2` spawn) — the §15 monitoring readout. Charging a domain's consumption against its budget
    /// (the `create(module, window, budget)` accounting) is the follow-up; this is the passable object
    /// + attenuation the rest builds on.
    pub const BUDGET: u32 = 14;
    /// PROCESS.md §5 **window minter** — the authority to mint **detached** windows: a child
    /// spawned through it (`Instantiator.instantiate_detached`, op 15) gets a fresh platform
    /// window *outside* the parent's — no ancestor below the minter holds read authority, and
    /// the child attests `window_exposed = false` (the jacl distrust-spawner trust anchor).
    /// The capability carries a **byte quota**, deducted at each mint (host-enforced); an
    /// ordinary granted authority (D46 `Resolver`-shaped: you can mint detached windows only
    /// if someone granted you that), embedder-granted at the root.
    pub const WINDOW_MINTER: u32 = 15;
    /// Base of the **guest-interface id space** (IMPORTS.md §3.2): ids for wired interface offers
    /// are interned per-`Host` from this base upward ([`super::Host::intern_interface`] — the id ≡
    /// the structural op-signature list, the D59 rule applied to capability interfaces). Far above
    /// the fixed built-ins and far below the reserved `u32::MAX`-family dispatch sentinels.
    pub const GUEST_IMPL_BASE: u32 = 0x1000_0000;
}

/// Canonical op-signature shapes for the built-in interfaces that are **pre-seeded** into the
/// per-`Host` intern (IMPORTS.md §3.5 "intern pre-seeding"). A guest interface declaration whose
/// op-signature list is structurally equal to a pre-seeded shape interns to the built-in id
/// ([`Host::intern_interface`]) — D59 extended across the host-native/guest-impl divide — so an
/// import slot requiring that shape accepts a real host handle or a guest impl of it
/// interchangeably (the virtualized-interface unlock: a guest can interpose a whole built-in
/// interface). Interning to a built-in id confers **no authority**: a call still needs a real
/// granted handle of the matching [`Binding`], generation- and type-checked at the use site
/// ([`Host::resolve_op`]) — pre-seeding only lets a structurally-equal declaration name the same
/// type, never mint the capability.
///
/// Only **specific** shapes are pre-seeded. `Stream`'s read/write/close triple is a genuine
/// interface identity. A generic single-op shape — `Clock`'s `(i64) -> (i64)`, say — is *not*: an
/// unrelated capability could share it by accident, so canonicalizing it would over-claim (and
/// `(i64) -> (i64)` is exactly the shape an ordinary guest offer uses). Handle-typed built-ins,
/// whose ops pass or return capabilities where the `cap`-vs-`i32` signature convention for
/// built-ins is unsettled, and `HOST_FN`, whose semantics are per-registration with no canonical
/// shape, are the deliberate exceptions — see IMPORTS.md §3.5.
fn preseeded_iface_shapes() -> [(u32, Vec<(&'static str, FuncType)>); 1] {
    let rw = FuncType {
        params: vec![ValType::I64, ValType::I64],
        results: vec![ValType::I64],
    };
    let unit = FuncType {
        params: vec![],
        results: vec![],
    };
    [(
        cap_id::STREAM,
        vec![("read", rw.clone()), ("write", rw), ("close", unit)],
    )]
}

/// The pre-seeded built-in id whose canonical shape structurally equals `sigs`, if any
/// ([`preseeded_iface_shapes`]). Consulted before allocating a guest id in
/// [`Host::intern_interface`], so a matching declaration resolves to the built-in.
fn preseeded_iface_id(sigs: &[FuncType]) -> Option<u32> {
    preseeded_iface_shapes().into_iter().find_map(|(id, ops)| {
        (ops.len() == sigs.len() && ops.iter().zip(sigs).all(|((_, s), t)| s == t)).then_some(id)
    })
}

/// The canonical op names + signatures of a pre-seeded built-in interface
/// ([`preseeded_iface_shapes`]) — for an embedder offering a host-native handle as a **whole
/// interface** (e.g. `svm-run`'s `IfaceShape::builtin`) without re-declaring its shape by hand.
/// Returns `None` for a built-in that is not pre-seeded (handle-typed built-ins, `HOST_FN`) or an
/// unknown id.
pub fn builtin_iface_shape(id: u32) -> Option<Vec<(&'static str, FuncType)>> {
    preseeded_iface_shapes()
        .into_iter()
        .find_map(|(bid, ops)| (bid == id).then_some(ops))
}

/// Negative-errno values returned by capability ops (§3e D42): `< 0` is `-errno`,
/// `>= 0` is success. Errors do **not** trap — traps stay reserved for escape/fatal.
const ENOMEM: i64 = -12; // resource quota exhausted (e.g. the Jit compile budget)
/// §3.6 revocation-unparks completion status (`-EBADF`): the errno a fiber's parked capability
/// call returns when the handle it was parked through is revoked out from under it. Probeable
/// on the fiber's own error path — never a trap (D42: errors return, traps stay for escape).
const CAP_REVOKED: i64 = -9;
/// A live-callee's dispatch queue was full at the enqueue (`-EAGAIN`): backpressure surfaces
/// as a probeable errno at the caller, per the §3.6 bounded fail-closed queue design.
const EAGAIN: i64 = -11;
const EFAULT: i64 = -14; // buffer not fully within the window
const EINVAL: i64 = -22; // bad op / argument
const EMFILE: i64 = -24; // handle table full — a guest-minted handle has nowhere to go (§3c)
const ENOSPC: i64 = -28; // no free table slot — the Jit install table is full

/// A `Trap` → small status code for an `IoRing` CQE, numbered to match the JIT's `TrapKind` codes
/// (so the whole system speaks one trap-code vocabulary). `0` is reserved for success in the CQE.
fn trap_status(t: &Trap) -> i64 {
    match t {
        Trap::DivByZero => 1,
        Trap::IntOverflow => 2,
        Trap::BadConversion => 3,
        Trap::Unreachable => 4,
        Trap::IndirectCallType => 5,
        Trap::CapFault | Trap::Malformed | Trap::Exit(_) => 6, // bad/unsupported async request
        Trap::MemoryFault => 8,
        Trap::FiberFault => 9,
        Trap::ThreadFault => 10,
        Trap::OutOfFuel => 11,
        // Matches the JIT's `TrapKind::StackOverflow` (13). The JIT produces it only under the
        // `stack-check` feature (a fiber's software stack-limit check); the default guard-page path
        // reports a stack overflow as `MemoryFault` (8) — the hardware can't distinguish it — so the
        // two configs report the same event under different codes, both a "stack blew up" outcome.
        Trap::StackOverflow => 13,
    }
}

/// Per-region cap on a **guest-minted** region (`AddressSpace.create_region`, §13/§14): an anti-bomb
/// ceiling so a single mint can't exhaust the host. Aggregate quota metering is §15 (D48: DoS is
/// contained by caps + the kill path, not prevented).
const MAX_MINTED_REGION: i64 = 256 << 20; // 256 MiB

/// Cap ABI `prot` bits for the `Memory` capability (§3e): the low two bits of the `i32`
/// argument. There is no `EXEC` bit — guest data is never executed as code (§3c).
const PROT_READ: i32 = 1;
const PROT_WRITE: i32 = 2;

/// A §13 `SharedRegion`'s backing — a host-owned shared object aliased into a window at one or more
/// offsets. The reference (interpreter) backing is a plain Rust buffer ([`VecBacking`]); a flat-window
/// backend (the JIT) supplies one wrapping a real OS shared-memory object (memfd / file mapping) whose
/// [`SharedBacking::os_fd`] it `mmap`s for true hardware aliasing. Cloning the `Arc` shares the *same*
/// object, so two mappings of it alias.
///
/// `Send + Sync`: a region is shared across vCPU threads (§12) — a `Backed` page aliased into more
/// than one thread's window names the same bytes. Concurrent access is the guest's race (§12);
/// implementors serialize or use atomics as they see fit (the reference [`VecBacking`] uses a
/// `Mutex`).
pub trait SharedBacking: Send + Sync {
    /// Region size in bytes.
    fn size(&self) -> u64;
    /// Read one region-relative byte (out of range ⇒ 0).
    fn read_byte(&self, off: u64) -> u8;
    /// Write one region-relative byte (out of range ⇒ ignored). Interior-mutable: a region is shared
    /// (`Arc`), so writes go through `&self`.
    fn write_byte(&self, off: u64, b: u8);
    /// An OS shared-memory handle a flat-window backend can `mmap` for real aliasing; `None` for the
    /// pure-Rust reference backing (the interpreter models aliasing in software instead). Unix
    /// (`memfd`/`shm`); the Windows analogue is [`os_section`](SharedBacking::os_section).
    fn os_fd(&self) -> Option<i32> {
        None
    }

    /// A Windows section handle (from `CreateFileMapping`) a flat-window backend maps into the window
    /// via `MapViewOfFile3` for real aliasing — the Windows analogue of [`os_fd`](SharedBacking::os_fd).
    /// Carried as an `isize` (a `HANDLE` is pointer-sized) to keep this trait platform-clean; `None`
    /// for the pure-Rust reference backing. Only the Windows JIT path consumes it.
    fn os_section(&self) -> Option<isize> {
        None
    }
}

/// A reference to a shared region backing (see [`SharedBacking`]); cloning shares the same object.
pub type RegionBacking = Arc<dyn SharedBacking>;

/// §4 / S4 — a host-served **pipe's** shared FIFO backing. `Arc<Mutex<…>>` so a pipe end can be
/// **re-granted into a §14 child** (the child's `Host` clones the `Arc`, aliasing the same queue) and
/// so concurrent parent/child access — the interpreter runs children on its M:N executor — is
/// serialized. The `write` end appends, the `read` end drains.
type PipeBacking = Arc<Mutex<VecDeque<u8>>>;

/// The reference [`SharedBacking`]: a plain in-process buffer behind a `Mutex` (so it is `Send +
/// Sync` and safe to alias across vCPU threads). The interpreter models aliasing by reading/writing
/// this shared buffer through several `Backed` pages.
struct VecBacking(Mutex<Vec<u8>>);

impl VecBacking {
    /// Lock, recovering from poisoning rather than panicking (the interpreter never panics, §robust).
    fn buf(&self) -> std::sync::MutexGuard<'_, Vec<u8>> {
        self.0.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl SharedBacking for VecBacking {
    fn size(&self) -> u64 {
        self.buf().len() as u64
    }
    fn read_byte(&self, off: u64) -> u8 {
        self.buf().get(off as usize).copied().unwrap_or(0)
    }
    fn write_byte(&self, off: u64, b: u8) {
        if let Some(s) = self.buf().get_mut(off as usize) {
            *s = b;
        }
    }
}

/// The guest window a capability handler borrows `(ptr, len)` buffers from (§7). Both
/// the interpreter's lazily-paged [`Mem`] and a JIT's flat window implement this, so a
/// single host dispatch ([`Host::cap_dispatch`]) serves both backends. **All offsets/pointers are
/// guest-relative** — the zero-based window the guest sees (a §14 child names its own `[0, size)`,
/// never its position in an ancestor's window); implementations translate to their backing. The
/// methods bounds-check `[ptr, ptr+len) ⊆ [0, size)` and return `None` (→ `-EFAULT`) otherwise.
pub trait GuestMem {
    fn read_bytes(&self, ptr: u64, len: u64) -> Option<Vec<u8>>;
    fn write_bytes(&mut self, ptr: u64, data: &[u8]) -> Option<()>;

    /// `Memory` capability ops (§3e): (re)commit / decommit / re-protect window pages. `offset`
    /// is page-aligned and `[offset, offset+len)` window-relative; `prot` is `READ|WRITE`. Each
    /// returns `0` or a negative errno (`-EINVAL`). The default is a success no-op — overridden
    /// by the interpreter's paged [`Mem`] (the reference semantics); a flat-window backend
    /// (e.g. a JIT) wires its own `mprotect`-backed implementation.
    fn map(&mut self, _offset: u64, _len: u64, _prot: i32) -> i64 {
        0
    }
    fn unmap(&mut self, _offset: u64, _len: u64) -> i64 {
        0
    }
    fn protect(&mut self, _offset: u64, _len: u64, _prot: i32) -> i64 {
        0
    }

    /// `SharedRegion` op 0 `map` (§13): alias `backing`'s `[region_off, region_off+len)` pages into
    /// the window at `[win_off, win_off+len)` with `prot`. The same `region`/`backing` mapped at two
    /// window offsets makes both ranges name the *same* bytes (zero-overhead aliasing). `0` or a
    /// negative errno. The default rejects it (`-EINVAL`): only the reference paged [`Mem`] models
    /// aliasing today; a flat-window backend wires its own shared mapping (§13 slice 2).
    fn map_region(
        &mut self,
        _win_off: u64,
        _region_off: u64,
        _len: u64,
        _prot: i32,
        _region: u32,
        _backing: RegionBacking,
    ) -> i64 {
        EINVAL
    }

    /// `Memory` op 3 `page_size() -> i64`: the host MMU page granularity this window is managed in —
    /// the unit `map`/`unmap`/`protect` round to. A guest queries it to align its own allocator to
    /// the real page (4 KiB / 16 KiB / …) and adapt, instead of assuming a fixed size. The default
    /// reports the host page; the paged [`Mem`] and the JIT's `MprotectWindow` override it with the
    /// exact value they round to, so the two backends stay in differential lockstep.
    fn page_size(&self) -> i64 {
        host_page_size() as i64
    }

    /// `SharedRegion` op 3 `page_size() -> i64`: the granularity a `SharedRegion` map aligns to —
    /// the host page on unix, the **allocation granularity** (64 KiB) on Windows, which
    /// `MapViewOfFile3` requires. Distinct from [`page_size`](GuestMem::page_size) (the protection
    /// granularity) so a guest aligns its region maps to a value that works on every backend. The
    /// default ([`host_region_granularity`]) is correct for both the paged [`Mem`] and the JIT's
    /// flat window, so the two stay in §13 lockstep without an override.
    fn region_page_size(&self) -> i64 {
        host_region_granularity() as i64
    }

    /// §9/§12 **async ring** support. Return a backend-neutral [`AsyncCounter`] for the 4-byte futex
    /// **completion counter** at `counter_addr`: an offload-pool worker atomic-increments it (the same
    /// path the backend's `wait`/`notify` value-check reads) and `notify`s its [`AsyncCounter::key`], so
    /// a vCPU parked in `wait` on the counter wakes race-free (the compare-under-lock guard). `Some`
    /// only for a normal in-window, naturally-aligned, writable page. `None` — the default — means the
    /// backend can't post async completions, so `submit_async` reports `-EINVAL` and the guest falls
    /// back to the synchronous `submit`. The reference paged [`Mem`] and the JIT's flat window both
    /// override it (each keyed to its own `wait`/`notify`: a window offset vs. an absolute address).
    fn async_counter(&self, _counter_addr: u64) -> Option<Arc<dyn AsyncCounter>> {
        None
    }
}

/// A `Send + Sync` handle an offload-pool worker uses to post an async-ring completion to the futex
/// **completion counter** (§9/§12). `increment` atomic-adds to the in-window counter through the same
/// path the backend's atomics take (a [`Region`] on the interpreter, a raw window write on the JIT);
/// `key` is the parking-lot key to hand the [`Host`]'s wake hook — a window offset on the interpreter
/// (the `Scheduler` key), an absolute window address on the JIT (the futex key) — each consistent with
/// that backend's `wait`/`notify`, so the worker's increment targets exactly what the parked vCPU's
/// value-check reads.
pub trait AsyncCounter: Send + Sync {
    fn increment(&self, delta: u64);
    fn key(&self) -> u64;
}

/// §4/§7 a JIT cap-path window **page map**: page index → state code (the flat-window backend, e.g.
/// `svm_run`, owns the encoding; absent ⇒ region default). Shared + persistent across a run's
/// `cap.call`s (see [`Host::cap_window_pages`]) so a guest-grown page stays borrowable.
pub type CapPageMap = Arc<Mutex<BTreeMap<u64, u8>>>;

/// A [`GuestMem`] over a flat, contiguous window slice — the JIT's representation. The
/// slice may include trailing guard bytes; `size` is the *logical* window so the §7
/// bounds check matches the interpreter exactly.
pub struct WindowMem<'a> {
    window: &'a mut [u8],
    size: u64,
}

impl<'a> WindowMem<'a> {
    pub fn new(window: &'a mut [u8], size: u64) -> WindowMem<'a> {
        WindowMem { window, size }
    }
}

impl GuestMem for WindowMem<'_> {
    fn read_bytes(&self, ptr: u64, len: u64) -> Option<Vec<u8>> {
        let end = ptr.checked_add(len)?;
        if end > self.size {
            return None;
        }
        Some(self.window[ptr as usize..end as usize].to_vec())
    }
    fn write_bytes(&mut self, ptr: u64, data: &[u8]) -> Option<()> {
        let end = ptr.checked_add(data.len() as u64)?;
        if end > self.size {
            return None;
        }
        self.window[ptr as usize..end as usize].copy_from_slice(data);
        Some(())
    }
}

/// Which standard stream a `Stream` handle is bound to.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StreamRole {
    In,
    Out,
    Err,
}

/// The host-side object a handle-table entry dispatches to — the mock equivalent of
/// §3c's `(methods, object)`. The guest never names or writes this (it lives in host
/// memory); it is selected only by a *granted* handle index.
#[derive(Clone, Copy, Debug)]
enum Binding {
    Stream(StreamRole),
    /// §3.6 slice 3 — a **live-callee offer**: a capability whose provider is another *running*
    /// domain (a §14 child), carried as an index into [`Host::live_impls`] (the entry holds the
    /// callee's live powerbox Arc + target impl-export — index-carried to keep `Binding: Copy`,
    /// like [`Binding::GuestImpl`]/[`Binding::PipeEnd`]). A call through it does not run a
    /// passive `drive_arc` sub-run — it **enqueues** onto the callee's inbound dispatch queue
    /// and **parks the calling fiber** until the callee's serve loop completes the dispatch
    /// (the caller-parking half of the unified model; serviced in the eval loop, which alone
    /// can park). Acyclic by construction: the parent's table points at the child's Host, never
    /// the reverse. Non-durable (a live run is not a snapshot artifact).
    LiveImpl(u32),
    /// A **host-served pipe** end (§4/S4, the personality's byte IPC): resolves under iface `STREAM`
    /// (a pipe end *is* a stream — the personality treats it as an fd), carrying the index of its FIFO
    /// in [`Host::pipes`] and which half it is. `write = true` appends to the FIFO (op 1), `false`
    /// drains it (op 0) — non-blocking: a read of an empty pipe returns `0`. Index-carrying, so
    /// non-durable and non-copyable, like [`Binding::SharedRegion`].
    PipeEnd {
        pipe: u32,
        write: bool,
    },
    /// PROCESS.md §5 — a **window minter** (detached-window authority), carrying the index of
    /// its remaining byte quota in [`Host::window_minters`]. Index-carrying (mutable quota
    /// state), so non-copyable and non-durable like [`Binding::SharedRegion`]; serviced by the
    /// eval loop's `instantiate_detached` arm, inert under the generic dispatch.
    WindowMinter(u32),
    Exit,
    Clock,
    Memory,
    /// A §13 `SharedRegion` handle, carrying the index of its backing in [`Host::regions`]. The
    /// backing (not the index) is the shared object; mapping it at several window offsets aliases.
    SharedRegion(u32),
    /// A §14 `AddressSpace` handle attenuated to the power-of-two window sub-range `[base, base+size)`
    /// in the **holder's own (guest-relative) coordinates** — a child's full-window grant is
    /// `[0, its size)` regardless of where its window sits in an ancestor's. Every op is confined to
    /// it; `sub` mints a further-attenuated child. The bounds live in the host-owned slot — the guest
    /// names only the forgeable handle.
    AddressSpace {
        base: u64,
        size: u64,
    },
    /// A §14 `Instantiator` handle conferring authority to spawn children confined to the window
    /// sub-range `[base, base+size)` in the **holder's own (guest-relative) coordinates**. The eval
    /// loop (not the generic dispatch) services it — spawning needs executor access — translating to
    /// backing-absolute via the holder's window base, so nesting composes at any depth.
    Instantiator {
        base: u64,
        size: u64,
    },
    /// A §14 `Yielder` handle a co-fiber child holds to suspend back to its instantiator-parent. The
    /// eval loop services it (it must yield the running coroutine's continuation, which the generic
    /// dispatch can't); a forged/wrong handle resolves nowhere and is an inert `CapFault`.
    Yielder,
    /// A §14 `Module` handle, carrying the index of its grant in [`Host::modules`]. Confers only the
    /// authority to instantiate (the Instantiator's module ops, serviced by the eval loop / nesting
    /// runtime); the generic dispatch treats any `cap.call` on it as an inert `CapFault`.
    Module(u32),
    /// A §9/§12 `IoRing` handle: authority to `submit` a batch of deferred `cap.call`s
    /// (io_uring-shaped), carrying the index of its [`RingState`] in [`Host::rings`] (the async-path
    /// completion buffer; the synchronous `submit` doesn't use it). The SQ/CQ ring buffers live in the
    /// guest window; the ops get their pointers as args.
    IoRing(u32),
    /// A §12 `Blocking` handle, carrying the index of its [`AsyncState`] in [`Host::blockings`] — a
    /// mock synchronous-only/blocking op the offload pool can overlap. Out-of-line (an index, not the
    /// `Arc`) so `Binding` stays `Copy`, like [`Binding::SharedRegion`]/[`Binding::Module`].
    Blocking(u32),
    /// A guest-driven `Jit` domain handle (iface 11, DESIGN.md §22), carrying the index of its
    /// [`JitDomainState`] in [`Host::jit_domains`]. Out-of-line so `Binding` stays `Copy`.
    JitDomain(u32),
    /// A `CompiledCode` handle minted by `Jit.compile` (iface 12): `(domain, unit)` indices into
    /// [`Host::jit_domains`]. No directly callable ops (like [`Binding::Module`]) — it is only
    /// *named* in `Jit.invoke`/`release`.
    JitCode {
        domain: u32,
        unit: u32,
    },
    /// An **embedder-registered** host-function capability (iface 13): carries the index of its
    /// handler closure in [`Host::host_fns`] (out-of-line so `Binding` stays `Copy`, like
    /// [`Binding::Blocking`]). All ops dispatch to that one closure, which interprets `op`.
    HostFn(u32),
    /// An mmap-capable host-function capability (§4b): like [`Binding::HostFn`] but its handler in
    /// [`Host::host_fns_region`] is also handed a [`RegionMinter`]. Resolves under the same iface 13.
    HostFnRegion(u32),
    /// A **wired interface offer** (IMPORTS.md §3.2): a guest-implemented capability, carrying the
    /// index of its [`GuestImplEntry`] in [`Host::guest_impls`] (out-of-line so `Binding` stays
    /// `Copy`, like [`Binding::HostFn`]). Op `i` dispatches to the offer's `ops[i]` function via
    /// the **generic dispatch** (one implementation, all three backends): a v1 **pure dispatch** —
    /// a fresh reference run over the offer's functions with no window and an empty powerbox, so
    /// the impl computes over its arguments alone. Exporter-domain state is the designed
    /// follow-up.
    GuestImpl(u32),
    /// A §15 / PROCESS.md §5 `Budget` handle, carrying the index of its [`BudgetState`] in
    /// [`Host::budgets`]. Authority over a passable, **splittable** resource-quota vector (fuel / mem /
    /// spawn): `split` attenuates a sub-budget out of the remaining, `read` reports it. Out-of-line (an
    /// index, not the state) so `Binding` stays `Copy`, like [`Binding::SharedRegion`].
    Budget(u32),
}

/// §15 / PROCESS.md §5 — a `Budget`'s **remaining** resource-quota vector. Three meterable resources
/// today (fuel / mem / spawn, the anti-bomb dials §15 already tracks); the vector is kept deliberately
/// short (O8). A field is a non-negative remaining amount. `split` moves quota from a parent entry into
/// a fresh child entry (never raising a total — attenuation, D19); `read` reports a field. Charging a
/// domain's live consumption against its budget is the follow-up (the `create(module, window, budget)`
/// accounting) — this type is the passable, splittable object the accounting will draw down.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BudgetState {
    fuel: i64,
    mem: i64,
    spawn: i64,
}

/// §6 (PROCESS.md) — a domain's platform-vouched **attestation**, reported by `cap.self.attest`. The
/// non-interposable trust anchor: a nested host can virtualize every handle-gated capability, but not
/// this (it is a D46 `cap.self` intrinsic, never a handle). Kept minimal (O5: a tier + two bits, not an
/// authority set) so the non-interposable surface stays tiny.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Attestation {
    /// §2 isolation tier — `0`/`1` in-process (defense-in-depth, **not** a Spectre boundary), `3`
    /// separate-process. The strongest isolation the domain may *require*; a domain needing protection
    /// from a distrusted host must see `3` or refuse.
    pub tier: u8,
    /// `true` iff an **ancestor** (beyond the platform) holds map/read/pager rights over the domain's
    /// backing — a §14 nested carve is exposed (the parent sees the superset); a platform-minted or
    /// root window is not.
    pub window_exposed: bool,
    /// `true` iff an ancestor may **snapshot** the domain (a snapshot is a complete read). A domain is
    /// *confidential* (freezable by nobody below the platform) or *ancestor-durable*, never both.
    pub freeze_exposed: bool,
}

impl Default for Attestation {
    /// A **root** domain: in-process (tier 1), platform-only window (no ancestor read), not
    /// ancestor-freezable. The embedder overrides via [`Host::set_attestation`]; the §14 spawn path
    /// stamps a nested child's (exposed) report.
    fn default() -> Attestation {
        Attestation {
            tier: 1,
            window_exposed: false,
            freeze_exposed: false,
        }
    }
}

impl Attestation {
    /// Pack into the `i32` `cap.self.attest` returns: `tier | (window_exposed << 8) |
    /// (freeze_exposed << 9)`.
    fn packed(self) -> i32 {
        (self.tier as i32)
            | ((self.window_exposed as i32) << 8)
            | ((self.freeze_exposed as i32) << 9)
    }
}

/// The value-typed subset of [`Binding`] a v1 snapshot can **re-grant** on restore
/// (DURABILITY.md §12.5). Every variant's entire state is value-typed — no out-of-line host
/// objects (`Host::regions`/`modules`/`rings`/…) and no native pointers — so re-granting it
/// into a fresh `Host` reconstructs the exact authority. The non-value bindings
/// (`SharedRegion`, `Module`, `IoRing`, `Blocking`, `JitDomain`, `JitCode`, `HostFn`) are
/// **not** durable: a live one makes the domain non-snapshottable.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DurableBinding {
    Stream(StreamRole),
    Exit,
    Clock,
    Memory,
    Yielder,
    AddressSpace { base: u64, size: u64 },
    Instantiator { base: u64, size: u64 },
}

/// One live, re-grantable handle-table entry captured for snapshot/restore (DURABILITY.md
/// §12.5 Section 3). The `(slot, generation)` pin is what keeps a **guest-held handle value**
/// valid across restore: the guest names `(generation << CAP_LOG2) | slot`, so restore must
/// reinstate the same pair. `type_id` is the interface the slot was granted under (the
/// resolve check is `type_id` + `generation`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct DurableHandle {
    pub slot: u32,
    pub generation: u32,
    pub type_id: u32,
    pub binding: DurableBinding,
}

/// Why a handle table can't be snapshotted in v1: a live slot holds a binding that carries
/// out-of-line host state or native pointers, so it isn't re-grantable (DURABILITY.md §12.5).
/// Freeze refuses with this rather than silently dropping authority, so restore is
/// all-or-nothing. Such handles must be closed/drained (§5) before a freeze can proceed.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct NonDurableHandle {
    pub slot: u32,
    pub type_id: u32,
    pub kind: NonDurableKind,
}

/// Which non-re-grantable binding kind a live slot held (the [`NonDurableHandle`] reason).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum NonDurableKind {
    SharedRegion,
    Module,
    IoRing,
    Blocking,
    JitDomain,
    JitCode,
    HostFn,
    Budget,
    Pipe,
    /// A wired interface offer (IMPORTS.md §3.2) — carries an out-of-line reference to the
    /// offering domain's functions, so it must be re-wired after restore, not snapshotted.
    GuestImpl,
    /// §3.6 a live-callee offer — points at a *running* domain's powerbox, which no snapshot
    /// can carry; re-wired after restore like a GuestImpl.
    LiveImpl,
    /// PROCESS.md §5 a window minter — mutable quota state (and the detached children it
    /// minted are outside the snapshot anyway); re-granted by the embedder after restore.
    WindowMinter,
}

/// One handle-table slot (§3c): host-owned, guest-unwritable. `generation` is
/// per-slot and only advances on (re)grant, so a closed handle's value can never
/// alias a later grant of the same slot (ABA-safe use-after-close detection, D37).
#[derive(Clone, Copy, Debug, Default)]
struct Slot {
    generation: u32,
    entry: Option<Binding>,
    type_id: u32,
}

/// `log2` of the handle-table capacity. A handle value packs `(generation, slot)`:
/// `slot = h & (cap-1)`, `generation = h >> CAP_LOG2`.
const CAP_LOG2: u32 = 8;
const CAP: usize = 1 << CAP_LOG2;

/// Bits the packed handle carries for the generation counter: the 32-bit `i32`/`u32` handle minus
/// the `CAP_LOG2` slot bits. `Slot::generation` is a full `u32`, so it is compared **masked to
/// these bits** (`GEN_MASK`) — otherwise, once the counter passed `2^GEN_BITS` regrants of one slot
/// the full-width `generation` could no longer equal the truncated value carried in any handle, and
/// the slot died permanently (a single-slot, fail-closed DoS — reachable via a `Jit.compile`→`release`
/// loop). Masking lets the slot recycle cleanly on wrap. ABA resistance is `2^GEN_BITS` (~16M)
/// regrants of one slot; a stale handle that aliased after a full wrap is still inert — it re-selects
/// one of *this domain's own* current grants, re-checked by `type_id` at the call (D37) — so this is
/// not an authority escape, only the documented "a forged index is inert" property.
const GEN_BITS: u32 = 32 - CAP_LOG2;
/// DURABILITY.md §13.3 — the process-wide [`Host::domain_id`] mint (uniqueness is all the
/// runtime needs; a snapshot records the ids it saw and a thaw re-links by record, so the
/// counter's absolute values never matter).
static NEXT_DOMAIN_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
const GEN_MASK: u32 = (1u32 << GEN_BITS) - 1;

/// Worker-thread count of the host **bounded blocking-offload pool** (§12 "Keeping cores busy under
/// blocking", path 2 — the io_uring increment-2 path). At most this many *blocking* SQEs run
/// concurrently; the `(K+1)`th queues — so the OS-thread cost of a `submit` batch is bounded by `K`,
/// never by the number of deferred ops. The guest's own vCPU thread is **not** multiplied (it parks
/// on the one `submit` while the pool absorbs the blocking) — the "0 blocked vCPU threads" win.
pub const OFFLOAD_POOL_THREADS: usize = 4;

/// Shared, thread-safe state behind a [`cap_id::BLOCKING`] capability — a *mock* synchronous-only host
/// op used to exercise the offload pool. Its `run` is **window-independent and `&mut Host`-free** (a
/// pure function of its argument plus this `Send + Sync` state), which is exactly the property that
/// lets a `submit` batch run it on the pool instead of the guest's vCPU thread. The result is
/// deterministic ⇒ both backends agree (the §18 oracle); the `active`/`max_active` counters let a
/// test *prove* a batch genuinely overlapped on the pool.
pub struct AsyncState {
    /// How long each op blocks before returning — the "synchronous blocking" the pool absorbs.
    /// `Duration::ZERO` in production (a pure compute); a test sets it to make the blocking real.
    block_for: Duration,
    /// Optional rendezvous: when set, every concurrent op waits here before completing, so a batch of
    /// exactly `width` ops on a `≥ width`-thread pool **deterministically** co-resides
    /// (`max_active == width`) without depending on sleep timing. `None` in production. A *direct*
    /// (non-batched) `cap.call` on a rendezvous-configured handle would block forever — it is a
    /// batch-overlap test fixture only.
    rendezvous: Option<Arc<Barrier>>,
    /// Ops currently in-flight (bumped on entry, dropped on completion).
    active: AtomicUsize,
    /// High-water mark of `active` across this `AsyncState`'s lifetime — the realized concurrency,
    /// read back via [`AsyncState::max_active`] to confirm a batch overlapped on `K` threads.
    max_active: AtomicUsize,
}

impl AsyncState {
    /// Run one blocking op: account the in-flight concurrency, (optionally) rendezvous + block, then
    /// return the deterministic transform of `arg`. Called either inline (a direct `cap.call`, on the
    /// caller's thread) or on an offload-pool worker (a batched `submit`) — same result either way.
    fn run(&self, arg: i64) -> i64 {
        let now = self.active.fetch_add(1, Ordering::SeqCst) + 1;
        self.max_active.fetch_max(now, Ordering::SeqCst);
        if let Some(b) = &self.rendezvous {
            b.wait();
        }
        if !self.block_for.is_zero() {
            std::thread::sleep(self.block_for);
        }
        self.active.fetch_sub(1, Ordering::SeqCst);
        Self::mix(arg)
    }

    /// A deterministic, non-trivial pure transform (one Knuth LCG step) — identical on every backend
    /// and thread, so a batch's CQE results are reproducible (and a divergence would show).
    fn mix(arg: i64) -> i64 {
        arg.wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407)
    }

    /// The peak realized concurrency — a test reads this after a batched `submit` to confirm the pool
    /// overlapped the blocking ops (`== min(batch, OFFLOAD_POOL_THREADS)`).
    pub fn max_active(&self) -> usize {
        self.max_active.load(Ordering::SeqCst)
    }
}

/// A job handed to the offload pool: a self-contained closure that writes its own result (it captures
/// the destination), so completion *order* is irrelevant — the data it leaves is deterministic.
type OffloadJob = Box<dyn FnOnce() + Send + 'static>;

/// The **bounded blocking-offload pool** (§12 path 2): [`OFFLOAD_POOL_THREADS`] long-lived workers
/// that run window-independent blocking SQEs *off* the guest's vCPU thread. A `submit` of `n` blocking
/// ops costs `K` OS threads regardless of `n` (waves of `K`). Each worker owns its **own** channel —
/// a single shared `Mutex<Receiver>` would serialize the blocking `recv`s and defeat the overlap.
struct OffloadPool {
    /// Per-worker job channels; a batch is round-robined across them.
    txs: Vec<std::sync::mpsc::Sender<OffloadJob>>,
    workers: Vec<std::thread::JoinHandle<()>>,
    /// Jobs dispatched but not yet finished — `(count, condvar)`. [`OffloadPool::dispatch`] (the async
    /// path) returns without waiting, so [`OffloadPool::quiesce`] uses this to drain in-flight work at
    /// run end before the window's `Arc<Region>` (which a late job may still write) is read back.
    inflight: Arc<(Mutex<usize>, Condvar)>,
    next: usize,
}

impl OffloadPool {
    fn new(k: usize) -> OffloadPool {
        let mut txs = Vec::with_capacity(k);
        let mut workers = Vec::with_capacity(k);
        for _ in 0..k {
            let (tx, rx) = std::sync::mpsc::channel::<OffloadJob>();
            txs.push(tx);
            workers.push(std::thread::spawn(move || {
                while let Ok(job) = rx.recv() {
                    job();
                }
            }));
        }
        OffloadPool {
            txs,
            workers,
            inflight: Arc::new((Mutex::new(0), Condvar::new())),
            next: 0,
        }
    }

    /// **Async dispatch** (the increment-3 path): round-robin `jobs` to the workers and return
    /// **immediately**. Each job is wrapped to decrement the in-flight count + notify on completion;
    /// the job itself posts its own completion (host-side result + futex counter + `notify`). The
    /// guest's vCPU parks via the futex `wait` rather than blocking here — the whole point of async.
    fn dispatch(&mut self, jobs: Vec<OffloadJob>) {
        if jobs.is_empty() {
            return;
        }
        {
            let (m, _) = &*self.inflight;
            *m.lock().unwrap() += jobs.len();
        }
        for job in jobs {
            let inflight = Arc::clone(&self.inflight);
            let wrapped: OffloadJob = Box::new(move || {
                job();
                let (m, c) = &*inflight;
                let mut g = m.lock().unwrap();
                *g -= 1;
                if *g == 0 {
                    c.notify_all();
                }
            });
            let w = self.next % self.txs.len();
            self.next = self.next.wrapping_add(1);
            self.txs[w].send(wrapped).expect("offload worker vanished");
        }
    }

    /// Block until every dispatched async job has finished. Called at run end so no worker still holds
    /// (and might still write) the window's `Arc<Region>` after the caller reads the final memory back.
    fn quiesce(&self) {
        let (m, c) = &*self.inflight;
        let mut g = m.lock().unwrap();
        while *g > 0 {
            g = c.wait(g).unwrap();
        }
    }

    /// Round-robin `jobs` across the workers and **block until all complete**. Each job writes its
    /// result through its own captured destination, so the caller reads results back by index after
    /// this returns. This is the synchronous-submit MVP: one boundary crossing, `K`-way overlap,
    /// then a single reap (fiber-parking / async resume is increment 3).
    fn run_batch(&self, jobs: Vec<OffloadJob>) {
        let n = jobs.len();
        if n == 0 {
            return;
        }
        let done = Arc::new((Mutex::new(0usize), Condvar::new()));
        for (i, job) in jobs.into_iter().enumerate() {
            let done = Arc::clone(&done);
            let wrapped: OffloadJob = Box::new(move || {
                job();
                let (m, c) = &*done;
                *m.lock().unwrap() += 1;
                c.notify_all();
            });
            // `send` only fails if a worker thread is gone — a host bug, not a guest-reachable path,
            // and the wait below would then hang, so surface it loudly.
            self.txs[i % self.txs.len()]
                .send(wrapped)
                .expect("offload worker vanished");
        }
        let (m, c) = &*done;
        let mut g = m.lock().unwrap();
        while *g < n {
            g = c.wait(g).unwrap();
        }
    }
}

impl Drop for OffloadPool {
    fn drop(&mut self) {
        // Dropping the senders closes each worker's channel → its `recv` returns `Err` → it exits.
        self.txs.clear();
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

/// §9/§12 async-ring per-handle state: completions posted by offload workers (or by inline ops) during
/// a `submit_async`, awaiting the guest's `reap` to flush them into the window. `Send + Sync` so a pool
/// worker pushes from its own thread; the guest reaps on its vCPU thread. The futex completion counter
/// lives in the *window* (so the guest can `wait` on it); this holds only the CQE payloads.
#[derive(Default)]
struct RingState {
    /// Ready completions `(user_data, result, status)`, FIFO — pushed by workers/inline, popped by reap.
    completed: Mutex<VecDeque<(i64, i64, i64)>>,
}

/// The interpreter's [`AsyncCounter`]: the futex completion counter is a normal anonymous window page,
/// so a worker increments it via the shared [`Region`] (the same real-atomic path cross-vCPU atomics
/// take) and the parking key is the window-relative offset (the `Scheduler`'s parking-lot key).
struct RegionCounter {
    region: Arc<Region>,
    off: u64,
}

impl AsyncCounter for RegionCounter {
    fn increment(&self, delta: u64) {
        self.region.atomic_rmw(self.off, 4, RmwOp::Add, delta);
    }
    fn key(&self) -> u64 {
        self.off
    }
}

/// An **embedder-registered host-capability handler** (iface [`cap_id::HOST_FN`]): given the `op`,
/// the slot-encoded `i64` args, and the guest window (`None` if the module has no memory), it runs
/// the operation and returns its result slots — or a [`Trap`] (e.g. `Trap::Exit`). This is how a
/// host adds a capability (e.g. an `svm-wasi` shim) **without** touching this crate: the semantics
/// live in the closure, reached only through a granted handle (the §3c masked/type-checked table).
pub type HostFn =
    Box<dyn FnMut(u32, &[i64], Option<&mut dyn GuestMem>) -> Result<Vec<i64>, Trap> + Send>;

/// The **one** extra authority an mmap-capable [`HostFnRegion`] handler gets over a plain [`HostFn`]:
/// mint a §13 `SharedRegion` and receive its handle. Deliberately narrow — the handler still cannot
/// reach the rest of the `Host` (no slot table, no other backings), so the escape hatch widens by
/// exactly this one capability (MMAP_CAPABILITY.md §4b, "a small, existing-shaped new power for the
/// closure"). Implemented by [`Host`] as a thin forward to [`Host::grant_shared_region_backed`].
pub trait RegionMinter {
    /// Grant a `SharedRegion` over `backing` and return the guest handle value (`< 0` on failure, e.g.
    /// the handle table is full). The guest maps it with the built-in `SharedRegion.map`.
    fn grant_region(&mut self, backing: RegionBacking) -> i32;
}

/// Like [`HostFn`] but the handler is also handed a [`RegionMinter`] — the escape hatch for the
/// zero-copy file-mmap bridge (§4b): an mmap-capable fs handler opens a file, mints a file-backed
/// `SharedRegion` over it, and returns the handle so the guest aliases the real file into its window.
/// Registered with [`Host::grant_host_fn_region`]; resolves under the same [`cap_id::HOST_FN`] as a
/// plain `HostFn`, so a guest reaches it identically.
pub type HostFnRegion = Box<
    dyn FnMut(
            u32,
            &[i64],
            Option<&mut dyn GuestMem>,
            &mut dyn RegionMinter,
        ) -> Result<Vec<i64>, Trap>
        + Send,
>;

/// A **wired interface offer**'s host-side state (IMPORTS.md §3.2), indexed by the id a
/// [`Binding::GuestImpl`] carries: the offering module's functions, the offer's per-op funcidx
/// list, the op signatures **derived** from those functions' declared types (never self-asserted),
/// and the interned interface id ([`Host::intern_interface`]). Minted only by the wiring party
/// ([`Host::wire_impl`]) — declaring an offer confers nothing; this entry existing in a domain's
/// table is what moves authority. Op `i` runs `funcs[ops[i]]` as a **v1 pure dispatch** (see
/// [`Binding::GuestImpl`]): windowless, empty powerbox, fixed fuel — arguments in, results out.
/// A slot's retained §3.5 requirement set: the manifest's `(names, sigs)`, shared with the
/// attach-time coverage walk.
type ImportReq = Arc<(Vec<String>, Vec<FuncType>)>;

/// §3.5 coverage walk: for every required `(name, sig)`, find the same-named,
/// signature-equal provider op; extra provider ops are ignored. Name-less providers (legacy
/// wires) fall back to exact positional matching. Returns the consumer→provider op remap, or
/// `None` (does not cover).
pub fn coverage_remap(
    req_names: &[String],
    req_sigs: &[FuncType],
    prov_names: &[String],
    prov_sigs: &[FuncType],
) -> Option<Arc<[u32]>> {
    if prov_names.is_empty() {
        // Legacy name-less provider: match each required op to the *first* provider op with an
        // equal signature (the pre-§3.5 "first sig-matching op" rule, generalized per-op).
        let mut remap = Vec::with_capacity(req_sigs.len());
        for sig in req_sigs {
            remap.push(prov_sigs.iter().position(|s| s == sig)? as u32);
        }
        return Some(remap.into());
    }
    let mut remap = Vec::with_capacity(req_sigs.len());
    for (n, sig) in req_names.iter().zip(req_sigs) {
        let p = prov_names.iter().position(|pn| pn == n)?;
        if prov_sigs.get(p)? != sig {
            return None;
        }
        remap.push(p as u32);
    }
    Some(remap.into())
}

#[derive(Clone)]
pub struct GuestImplEntry {
    pub funcs: Arc<[Func]>,
    pub ops: Arc<[u32]>,
    pub sigs: Arc<[FuncType]>,
    /// §3.5 declared op **names** (the coverage-binding contract), from the offer's interface
    /// declaration when wired through a module-aware path; empty for name-less legacy wires
    /// (coverage then falls back to exact positional matching). Names are never identity —
    /// `type_id` interns the shape alone.
    pub names: Arc<[String]>,
    pub type_id: u32,
    /// §3.1 **provenance depth**: how many domain boundaries stand between the holder and the
    /// implementation — `1` where the offer was wired, `+1` per re-grant into a child. Every
    /// wired offer is **ancestor-terminated** (guest code, never the platform); this is the
    /// honest bit `cap.self` reports (op 5), and it lives host-side, so a parent can interpose
    /// but cannot hide that it did.
    pub depth: u32,
    /// §3.2 v2 **provider instance** — the exporter's domain. `None` = a v1 *pure* offer
    /// (windowless, empty powerbox: arguments in, results out). `Some` = an **instanced** offer:
    /// ops run over this persistent window + powerbox ([`Host::wire_impl_instance`]), so state
    /// survives across calls — the stateful wrap ("an `Fs` backed by the provider's own
    /// window"). Shared (`Arc`) across re-grants: a parent and its children handed the same
    /// offer drive one service instance, like a pipe's shared backing. The blocking lock is
    /// deadlock-free by construction: a provider can never hold an offer
    /// ([`Host::grant_impl_cap`] refuses one), so provider chains are acyclic and the lock
    /// order is always domain-host → provider, never the reverse.
    pub state: Option<Arc<Mutex<ProviderState>>>,
}

/// §3.6 slice 2 — one queued inbound dispatch (the serve-loop core): a call targeting this
/// domain's impl-export `export`, op `op`, args in the i64-slot ABI, completion posted under
/// `ticket`. Admitted as a handler over the domain's **one world** at a `svc.poll` service
/// point — run-to-completion this slice (handler parking rides the fiber-admission slice),
/// which is exactly the passive instance's serialized observable behavior (the oracle).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SvcDispatch {
    pub export: u32,
    pub op: u32,
    pub args: Vec<i64>,
    pub ticket: u64,
}

/// §3.6 bounded dispatch-queue depth. Small and fixed: a full queue refuses the enqueue
/// fail-closed (probeable at the enqueuer) — backpressure, not buffering (the pinned design).
pub const SVC_QUEUE_CAP: usize = 64;

/// The reserved self-namespace op number for `svc.poll` (§3.6 slice 2): drain-and-serve all
/// queued dispatches, return the count. Rides `cap.call CAP_SELF_TYPE_ID` like
/// `cap.self.provenance` — no wire change, no new opcode; serviced by the eval loop (it runs
/// guest code, which host-side dispatch cannot), so a backend tier without the eval-loop arm
/// answers `-EINVAL`, probeable. `svc.wait` (park-until-dispatch) takes the next number when
/// the caller-parking slice gives it a real waker.
pub const CAP_SELF_SVC_POLL: u32 = 9;

/// The reserved self-namespace op for `svc.wait` (§3.6 slice 3): like `svc.poll`, but an empty
/// queue **parks the serving fiber** until a caller's enqueue wakes it — the classic serve loop.
/// Same encoding path and backend story as `svc.poll`.
pub const CAP_SELF_SVC_WAIT: u32 = 10;

/// §3.6 slice 3 — the side table a [`Binding::LiveImpl`] indexes: the callee's live powerbox
/// and the target impl-export. Index-carried so `Binding` stays `Copy`. Carries the export's
/// **shape** (fetched from the callee's module at wire time) so a re-grant into another
/// powerbox can intern it there without touching the callee's lock.
struct LiveImplEntry {
    callee: Arc<Mutex<Host>>,
    export: u32,
    sigs: Arc<[FuncType]>,
}

/// An instanced offer's **provider domain** (IMPORTS.md §3.2 v2): the persistent window (built
/// once from the provider module's memory declaration + data segments at wiring) and powerbox
/// its ops execute over. The powerbox starts empty; the wirer may re-grant its own capabilities
/// in ([`Host::grant_impl_cap`]) — how a wrap holds the real cap it forwards to.
pub struct ProviderState {
    mem: Option<Mem>,
    host: Host,
    /// §5.3 **provider-pays metering** (resolved 2026-07-20): the provider funds its own
    /// dispatch compute out of this drainable reserve — its code, its choice to offer.
    /// Each op call is capped by `min(GUEST_IMPL_FUEL, remaining)` and drains what it
    /// used; a dry reserve makes further calls an inert `CapFault` (probeable by the
    /// caller, visible to the wirer via [`Host::impl_fuel_remaining`] — the §15 "read the
    /// meters on what you granted" story). A provider worried about a hammering child
    /// rate-limits or kills the child itself; the platform just meters honestly.
    fuel: u64,
}

/// The default provider fuel reserve (§5.3 provider-pays): generous — a service is expected
/// to live for many calls — and wirer-adjustable via [`Host::set_impl_fuel_reserve`].
const PROVIDER_FUEL_RESERVE: u64 = 1 << 32;

/// The fixed, deterministic fuel budget for one wired-offer op dispatch (v1 pure dispatch —
/// see [`Binding::GuestImpl`]). A looping impl hits `OutOfFuel` and the caller's call traps,
/// fail-closed and identically on every backend. Caller-fuel threading is the designed
/// follow-up alongside exporter-domain state.
const GUEST_IMPL_FUEL: u64 = 1 << 26;

/// §3.5 `cap` **boundary translation**: for each slot the signature types `ValType::Cap`,
/// re-grant the capability the slot names from `src`'s handle table into `dst`'s, replacing the
/// value with the receiver-local packed handle. Non-`cap` slots pass through untouched — a
/// guest-visible `cap` is `i32`-width data, treated specially *only* at a boundary. A forged /
/// dead / non-re-grantable handle is a fail-closed [`Trap::CapFault`].
///
/// This is the guest↔guest half of §2.3's "objects are arguments": authority crosses an
/// offer-call boundary exactly where a signature says `cap`, and unmarked integers keep the
/// forgeability guarantee (a raw handle crossing domains would index the *receiver's* table and
/// is inert). The re-grant policy is [`Host::regrant_into_child`]'s — an offer is adopted one
/// domain-hop deeper (§3.1 provenance), a pipe end aliases its shared backing, a coordinate-free
/// cap copies.
fn translate_cap_slots(
    src: &mut Host,
    dst: &mut Host,
    types: &[ValType],
    slots: &mut [i64],
) -> Result<(), Trap> {
    for (ty, slot) in types.iter().zip(slots.iter_mut()) {
        if *ty == ValType::Cap {
            let translated = src
                .regrant_into_child(*slot as i32, dst)
                .ok_or(Trap::CapFault)?;
            *slot = translated as i64;
        }
    }
    Ok(())
}

// The `Host` *is* the region minter — the narrow authority a `HostFnRegion` handler is handed. It
// forwards to the ordinary grant path; nothing else of the `Host` is exposed through this trait.
impl RegionMinter for Host {
    fn grant_region(&mut self, backing: RegionBacking) -> i32 {
        self.grant_shared_region_backed(backing)
    }
}

/// The host: the **host-owned handle table** (the powerbox) plus deterministic mock
/// capability state (captured stdio, a monotonic clock). Construct with [`Host::new`],
/// `grant_*` the initial capabilities, then pass to [`run_with_host`]; afterwards read
/// back `stdout`/`stderr`. Deterministic by design so it serves as a §18 oracle.
pub struct Host {
    table: Vec<Slot>, // CAP slots, host-owned
    /// Bytes a `Stream{In}` handle's `read` draws from.
    pub stdin: Vec<u8>,
    stdin_pos: usize,
    /// **Opt-in blocking stdin** (a persistent interactive session, e.g. the browser Postgres console).
    /// When `true`, a `Stream{In}` `read` that finds the buffer exhausted does **not** return `0`
    /// (EOF — which makes a REPL-style guest shut down); instead it sets [`Self::stdin_parked`] so the
    /// driver can suspend the vCPU at that read and resume it once more bytes are pushed. Default
    /// `false`: the one-shot runs (`svm_run_pg`, the corpus/oracle) keep plain EOF-at-end semantics.
    pub stdin_block: bool,
    /// Transient: the last `Stream{In}` `read` parked (buffer empty under [`Self::stdin_block`]). The
    /// bytecode `CapCall` arm takes this to yield [`Outcome::StdinPark`] instead of completing the read.
    stdin_parked: bool,
    /// Bytes written by `Stream{Out}` / `Stream{Err}` `write`s.
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    /// PROCESS.md S2: optional **shared** stdout/stderr sinks. `None` (default) ⇒ writes go to the
    /// local `stdout`/`stderr` Vec (the existing public API — untouched for hosts that never share).
    /// `Some` ⇒ writes go to the shared buffer instead — set on a **child** host whose stdout/stderr
    /// was re-granted from a parent (`instantiate_granted`), so the child's output reaches the same
    /// buffer the granting embedder observes (POSIX stdio inheritance). Promoted lazily by
    /// [`Host::shared_stdout`]/[`Host::shared_stderr`]; read the effective bytes via [`Host::stdout_bytes`].
    out_sink: Option<Arc<Mutex<Vec<u8>>>>,
    err_sink: Option<Arc<Mutex<Vec<u8>>>>,
    /// Monotonic nanosecond counter; each `Clock.now` returns it then advances by one,
    /// so reads are deterministic and strictly increasing.
    pub clock_ns: i64,
    /// §13 `SharedRegion` backings, indexed by the id a [`Binding::SharedRegion`] carries. Each is a
    /// shared host buffer; aliasing a region into several window offsets clones this `Rc`.
    regions: Vec<RegionBacking>,
    /// PROCESS.md S1b/S1c — the **canonical-key futex** region hook. On a §13 `map`/`unmap` the JIT needs
    /// to canonicalize a `Backed` address to `(backing, region offset)` for its futex, but its futex
    /// thunk has no region map. `svm-run` installs a recorder here (over the JIT window's `mem_base`);
    /// the dispatch calls it after each successful `map` (`Some(win_off, region_off, len, backing_fd)`)
    /// and each `unmap` (`None`), so the JIT registry stays in step. `None` on the interpreter (whose own
    /// `PageProt::Backed` already canonicalizes) and on any host with no recorder installed.
    #[allow(clippy::type_complexity)]
    region_hook: Option<Arc<dyn Fn(u64, u64, Option<(u64, u64)>) + Send + Sync>>,
    /// §15 / PROCESS.md §5 `Budget` states, indexed by the id a [`Binding::Budget`] carries. Each is a
    /// remaining resource-quota vector; `split` moves quota from a parent's entry into a fresh child
    /// entry (append-only, like `regions` — a split budget's index stays valid for the run).
    budgets: Vec<BudgetState>,
    /// §4 / S4 **host-served pipe** FIFO backings, indexed by the id a [`Binding::PipeEnd`] carries.
    /// Each is a shared byte queue a `write` end appends to and a `read` end drains. The backing is
    /// `Arc`-shared ([`PipeBacking`]) so an end can be **re-granted into a §14 child** (the child's
    /// `Host` clones the `Arc`, so both domains see the same queue — the cross-domain `cmd1 | cmd2`
    /// wiring). Append-only vector (a pipe's index stays valid for the run), like `regions`/`budgets`.
    pipes: Vec<PipeBacking>,
    /// §6 (PROCESS.md) — this domain's platform-vouched provenance, reported verbatim by
    /// `cap.self.attest`. Defaults to a **root** report ([`Attestation::default`]); the embedder sets it
    /// for the top-level domain and the §14 spawn path stamps a nested child's (exposed) one.
    attestation: Attestation,
    /// §14 instantiable **modules**, indexed by the id a [`Binding::Module`] carries — host-verified
    /// code a guest holding the handle may spawn as a child domain (`Arc`s so a spawned child shares,
    /// not copies). Append-only for the life of the `Host`, so raw views handed to the JIT's nesting
    /// runtime ([`Host::resolve_module_parts`]) stay valid for the whole run.
    modules: Vec<ModuleGrant>,
    /// The backing factory for **guest-minted** §13/§14 regions (`AddressSpace.create_region`).
    /// `None` (the default) mints the pure-Rust reference [`VecBacking`]; a flat-window embedder
    /// installs an OS-shared-memory factory ([`Host::set_region_factory`], e.g.
    /// `svm_run::new_shared_region`) so a JIT guest can `map` what it mints for real aliasing.
    region_factory: Option<fn(usize) -> RegionBacking>,
    /// §12 `Blocking` capability backings, indexed by the id a [`Binding::Blocking`] carries. Each is
    /// a `Send + Sync` [`AsyncState`] a `submit` batch can run on the offload pool.
    blockings: Vec<Arc<AsyncState>>,
    /// §7 embedder-registered host-capability handlers, indexed by the id a [`Binding::HostFn`]
    /// carries ([`Host::grant_host_fn`]). A dispatch takes the closure out, runs it, and restores it.
    host_fns: Vec<HostFn>,
    /// §4b mmap-capable host-capability handlers (indexed by the id a [`Binding::HostFnRegion`]
    /// carries, [`Host::grant_host_fn_region`]) — a `HostFn` plus a [`RegionMinter`]. Same
    /// take-out/run/restore dispatch as `host_fns`.
    host_fns_region: Vec<HostFnRegion>,
    /// Wired interface offers (IMPORTS.md §3.2), indexed by the id a [`Binding::GuestImpl`]
    /// carries ([`Host::wire_impl`]).
    guest_impls: Vec<GuestImplEntry>,
    /// The per-`Host` **structural interface intern** (D59 applied to capability interfaces):
    /// index `i` holds the op-signature list whose interface id is `GUEST_IMPL_BASE + i`, so
    /// id-equality ≡ structural equality within this table. Flat and scanned linearly — offers
    /// are few; boring beats a map.
    iface_intern: Vec<Arc<[FuncType]>>,
    /// §3.5 grouped-import **op remaps**, parallel to `import_bindings`: slot `i`'s entry, when
    /// present, maps consumer-local op indices to provider op indices (frozen at the binding
    /// act by the coverage walk). `None` = flat binding (consumer op must be 0).
    import_remaps: Vec<Option<Arc<[u32]>>>,
    /// §3.5 per-slot **requirement sets** `(names, sigs)`, parallel to `import_bindings`,
    /// retained from the manifest at the binding act so a later `import.attach` can run the
    /// coverage walk against the attached capability (and refresh the slot's remap). `None` =
    /// no requirement recorded — attach falls back to the legacy exact-`type_id` check.
    import_reqs: Vec<Option<ImportReq>>,
    /// §3.5 **self-module registration**: the running module's type-section interfaces, offers,
    /// and function table, registered at run setup so `call.import.dyn`, `cap.self.type_id`,
    /// `cap.self.covers`, and `export.handle` resolve through one host-side entry on all three
    /// backends. `None` until registered (the ops then fail closed, probeable).
    self_module: Option<Arc<Module>>,
    /// The domain's one shared service state for offers it reifies (`export.handle` — all of a
    /// domain's reified offers share it), created lazily on first reification.
    self_instance: Option<Arc<Mutex<ProviderState>>>,
    /// Memoized reified-offer handles by impl-export index (re-reifying returns the same
    /// backing).
    self_reified: BTreeMap<u32, i32>,
    /// §3.6 slice 2 — the domain's **bounded inbound dispatch queue** (the serve-loop core):
    /// dispatches targeting this domain's offers queue here and are admitted as handlers over
    /// the domain's **one world** (live window + powerbox) at the guest's `svc.poll` service
    /// points. Bounded, fail-closed: a full queue refuses the enqueue (probeable at the
    /// enqueuer), per the pinned §3.6 design. Enqueued embedder-side this slice
    /// ([`Host::svc_enqueue`]); the cross-domain caller side is the §3.6 caller-parking slice.
    svc_queue: VecDeque<SvcDispatch>,
    /// Completion cells for served dispatches, keyed by ticket ([`Host::svc_result`] drains).
    svc_results: BTreeMap<u64, i64>,
    svc_next_ticket: u64,
    /// DURABILITY.md §13.3 — this domain's **stable identity** for every serve-path key
    /// (`svc_waiters`/`svc_timers` keys, fiber waiters' domain field, `svc_wake` targets):
    /// process-unique, minted at construction. The `Arc` pointer it replaces was
    /// snapshot-hostile (a thaw re-allocates every powerbox); the id is recordable in a
    /// snapshot and re-linkable on thaw.
    domain_id: u64,
    /// I36 slice 3 (JIT serve loop) — the JIT embedder's native context for **this domain's
    /// serve loop** (its `*mut CompiledModule` as a `usize`): registered around a JIT run so
    /// the embedder's cap thunk can invoke handler trampolines over the live window at a
    /// `svc.poll`/`svc.wait` service point. Opaque here (only the embedder dereferences it);
    /// `0` on interpreter runs. Distinct from the per-`Jit`-domain [`Host::jit_native_ctx`]
    /// (which requires a granted `Jit` capability a serving module need not hold).
    serve_native_ctx: usize,
    /// §3.6 slice 3 — live-callee offer entries ([`Binding::LiveImpl`] indexes here).
    live_impls: Vec<LiveImplEntry>,
    /// PROCESS.md §5 — the side table a [`Binding::WindowMinter`] indexes: each entry the
    /// minter's **remaining byte quota**, deducted at every detached mint (numeric,
    /// host-enforced; no refund on child completion — the quota bounds lifetime total, v1).
    window_minters: Vec<u64>,
    /// The §12 bounded blocking-offload pool, created lazily on the first batched `submit` that has a
    /// blocking SQE (so a `Host` that never offloads spawns no threads). Dropping it joins the
    /// workers ([`OffloadPool`]'s `Drop`).
    pool: Option<OffloadPool>,
    /// §9/§12 async-ring per-handle state, indexed by the id a [`Binding::IoRing`] carries — where a
    /// `submit_async` posts completions for the guest's `reap`.
    rings: Vec<Arc<RingState>>,
    /// §9/§12 the **async-completion `notify`** hook: an offload worker calls this (with the confined
    /// futex counter key) to wake the vCPU parked in `wait` on that key — i.e. an I/O completion *is* a
    /// futex notify (DESIGN §12). Installed per run by the executor that owns the wake mechanism
    /// (`drive` wires it to the M:N `Scheduler::notify`); `None` ⇒ no async support, so `submit_async`
    /// `-EINVAL`s and the guest falls back to the synchronous `submit`.
    async_notify: Option<Arc<dyn Fn(u64, u32) + Send + Sync>>,
    /// §4/§7 the **JIT cap-path window page map**, keyed by window base. The JIT's `cap_thunk` rebuilds
    /// its window view per `cap.call`, so without a persistent home a guest-*grown* heap page (committed
    /// via the Memory cap in an earlier call) would read back as unmapped and a cap-buffer borrow of it
    /// would fail-closed. Persisting it here (the per-run `Host` is the only state `cap_thunk` reaches)
    /// mirrors how the interpreter's `Mem` keeps its page map across calls. Page index → state code
    /// (`svm_run` owns the encoding); absent ⇒ region default. Reset when a new window base appears.
    cap_pages: Option<(usize, CapPageMap)>,
    /// §15 spawn quota (fiber/vCPU ceilings) the embedder sets for this domain ([`Host::set_quota`]);
    /// default = the hard anti-bomb ceilings, so an unconfigured run is unchanged. `drive` reads it to
    /// size the executor's live-vCPU cap and each vCPU's fiber cap.
    quota: Quota,
    /// Guest-driven `Jit` domains (iface 11), indexed by the id a [`Binding::JitDomain`] carries.
    /// Append-only for the life of the `Host` (units are never removed — `release` only revokes
    /// the *handle*; code reclaim is a DESIGN.md §22 follow-up), so unit `Arc`s and native pointers stay
    /// valid for the whole run.
    jit_domains: Vec<JitDomainState>,
    /// The host-injected validation gate for guest-submitted `Jit` blobs ([`JitValidator`]) —
    /// injected (like [`Host::region_factory`]) rather than implemented here so this crate keeps
    /// its tiny dependency set *and* both backends run the **identical** decode+verify gate.
    /// `None` (the default) fail-closes every `compile` (`-EINVAL`).
    jit_validator: Option<JitValidator>,
    /// The `call_indirect` table reservation (`log2` of the slot count) for B2 `install` — the
    /// run's root vCPU builds its table this large so installs have padding slots. Must equal the
    /// JIT's `table_reserve_log2` for the backends to agree on slot indices. `0` ⇒ natural size
    /// (no install room).
    jit_table_log2: u8,
    /// W1 record/replay (DEBUGGING.md): when `Some`, every nondeterministic-input `cap.call`
    /// ([`is_recorded_input`]) is appended here as it crosses, so a later re-execution can replay it.
    cap_record: Option<Vec<CapRecord>>,
    /// W1 replay: when `Some`, serve nondeterministic-input `cap.call`s from this tape (cursor) in
    /// order instead of the live host, so a fresh-powerbox re-execution reproduces the guest's inputs.
    cap_replay: Option<(Arc<[CapRecord]>, usize)>,
    /// This domain runs a **durable** (freeze/thaw-instrumented) module: `drive` propagates it to
    /// every vCPU so the runtime maintains the per-context shadow-SP swap (D-fiber-cont option A,
    /// DURABILITY.md §12.8). `false` (the default) ⇒ an ordinary run that never touches the
    /// durable reserve. Set with [`Host::set_durable`] before [`run_with_host`] / friends.
    durable: bool,
    /// The freeze/thaw fiber residue (slice 3.1.5): **out** of a freeze run (the driver flattens
    /// each parked fiber and records it here for the snapshot) and **in** to a thaw run (`drive`
    /// re-seeds the registry from it before re-entering under `REWINDING`). Empty for an ordinary run.
    frozen_fibers: Vec<FrozenFiber>,
    /// The freeze/thaw **spawned-vCPU** residue (slice 3.2.1): **out** of a multi-vCPU freeze (each
    /// `thread.spawn` child that unwinds records itself here for the snapshot) and **in** to a thaw
    /// (the root's re-executed `thread.spawn` re-attaches these, in spawn order, under `REWINDING`).
    /// Empty for a single-vCPU or ordinary run.
    frozen_vcpus: Vec<FrozenVCpu>,
    /// §14 nested-child residue of a subtree freeze (DURABILITY.md §4) — see [`FrozenNested`].
    frozen_nested: Vec<FrozenNested>,
    /// The freeze/thaw **root** vCPU's flattened shadow-SP extent (slice 3.2.1). The single shared
    /// active-SP word holds only the *last* context to run at freeze end (a spawned child), so the
    /// root's own extent — its implicit residue (the thaw caller re-enters the root directly) — is
    /// recorded here instead. `None` ⇒ a single-vCPU thaw, which reads the extent from the window's
    /// active-SP word as before. Set by the freeze driver; consumed by `drive` on thaw.
    frozen_root_sp: Option<u64>,
    /// §7 the **capability-name directory** (Followup F7): name → handle for the powerbox grants, so a
    /// guest can resolve a capability by name at runtime (`cap.self` op 2) — dlopen-style discovery, the
    /// dynamic counterpart to load-time name binding. Populated by the powerbox layer at grant time
    /// (`svm_run`); empty for a bare `Host` (resolution then finds nothing — fail-closed). First match
    /// wins on a duplicate name. A side table only — it never affects handle values or grant order.
    cap_names: Vec<(String, i32)>,
    /// §7 / IMPORTS.md phase 1: the **import-binding table** — entry `i` is the instantiation-time
    /// resolution of the module's import `i` ([`Host::set_import_bindings`]). Read by the
    /// [`svm_ir::CAP_IMPORT_TYPE_ID`] translation in [`Host::cap_dispatch_slots`], the one shared
    /// entry all three backends dispatch executable `call.import`s through. Empty for a module with
    /// no imports (or a legacy resolved one) — an executable `call.import` then `CapFault`s.
    import_bindings: Vec<BoundImport>,
}

/// One instantiation-time import binding (§7 / IMPORTS.md phase 1): the `(type_id, op)` the host's
/// policy resolved an import name to, plus the **granted handle** for its powerbox-prefix slot.
/// Installed by [`Host::set_import_bindings`]; consumed by the [`svm_ir::CAP_IMPORT_TYPE_ID`]
/// dispatch translation.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct BoundImport {
    pub type_id: u32,
    pub op: u32,
    pub handle: i32,
    /// Whether the slot currently holds a live binding (IMPORTS.md phase 2). A `required` import
    /// is always bound; a `rebindable` one may start unbound (`call.import` then `CapFault`s,
    /// fail-closed) until an `import.attach` fills it.
    pub bound: bool,
    /// Whether `import.attach` may retarget this slot (the manifest's `rebindable` mode). The
    /// verifier enforces this statically for guest code; the host re-checks fail-closed (the
    /// dispatch entry is also reachable by host-side callers).
    pub rebindable: bool,
}

impl BoundImport {
    /// A bound `required`-mode entry (the phase-1 shape): immutable for the instance's lifetime.
    pub fn required(type_id: u32, op: u32, handle: i32) -> BoundImport {
        BoundImport {
            type_id,
            op,
            handle,
            bound: true,
            rebindable: false,
        }
    }

    /// A `rebindable` entry: `handle` is the initial binding (`Some`) or the slot starts empty
    /// (`None` — `call.import` traps until an `import.attach` fills it). The `(type_id, op)`
    /// template is fixed either way: attach swaps *which object*, never *which interface*.
    pub fn rebindable(type_id: u32, op: u32, handle: Option<i32>) -> BoundImport {
        BoundImport {
            type_id,
            op,
            handle: handle.unwrap_or(0),
            bound: handle.is_some(),
            rebindable: true,
        }
    }
}

/// The host-injected validation gate for guest-submitted `Jit` blobs (DESIGN.md §22 "Security
/// argument"): `(blob bytes, expected declared memory)` → the verified functions, or a
/// negative errno. The embedder's implementation must run the full hinge —
/// `decode_module` + `verify_module` + the **memory-match precondition** (declared memory ==
/// the parent window) + reject data segments and §12 concurrency ops
/// ([`Func::uses_concurrency`]) — and the *same* function must be installed for the
/// interpreter and JIT runs of a differential pair, so both backends accept/reject
/// identically (`svm-run` provides the canonical one).
///
/// The third argument is the **symbol-table bytes** for host-assisted dynamic linking
/// (DESIGN.md §22): a guest-provided `name → slot | capability` table the validator resolves the
/// unit's §7 imports against *before* verify (rewrite-then-verify). It is empty (`&[]`) for the
/// ordinary closed-blob `compile` op — an empty table resolves nothing, so a unit with imports
/// fails closed — and carries the guest's table only for the `compile_linked` op.
pub type JitValidator = fn(&[u8], Option<u8>, &[u8]) -> Result<Arc<[Func]>, i64>;

/// A successful [`Host::jit_compile`]: the minted `CompiledCode` handle and the `(domain,
/// unit)` indices the JIT embedder needs to compile the unit natively and register its
/// trampoline ([`Host::set_jit_unit_native`]).
pub struct JitCompiled {
    pub handle: i32,
    pub domain: u32,
    pub unit: u32,
}

/// One guest-compiled `Jit` unit: the validated functions (the unit's entry is `funcs[0]`;
/// the rest are unit-local helpers reached by direct calls), plus the JIT embedder's
/// registrations for the entry (`0` in a reference/interpreter run): the buffer-ABI
/// trampoline (`native_code`, for `invoke`) and the natural-ABI entry + interned `type_id`
/// (`install_code`/`install_type_id`, for B2 table `install`).
struct JitUnit {
    funcs: Arc<[Func]>,
    native_code: usize,
    install_code: usize,
    install_type_id: u32,
}

/// Per-`Jit`-handle domain state.
struct JitDomainState {
    /// The memory-match precondition (DESIGN.md §22 "Security argument"): a submitted blob's declared
    /// memory must equal the parent module's, fixed when the capability is granted.
    mem_log2: Option<u8>,
    units: Vec<JitUnit>,
    /// Opaque native context the JIT embedder registered (its `*mut CompiledModule` as a
    /// `usize`); `0` in a reference run. Never dereferenced here — only stored and handed back
    /// ([`Host::jit_native_ctx`]).
    native_ctx: usize,
    /// Compile quota (DESIGN.md §22 "Code reclaim", the MVP byte-cap): remaining units / cumulative
    /// submitted-blob bytes this domain may still `compile`. A guest looping `compile` is
    /// bounded with `-ENOMEM` here — in the **shared** gate, so both backends reject
    /// identically — instead of pressuring the JIT's finite code arena (whose exhaustion path
    /// inside Cranelift is not a guest-reachable-safe failure mode). Blob bytes are the budget
    /// *proxy* for compiled bytes.
    units_left: u32,
    bytes_left: u64,
}

/// Default per-domain compile quota: generous for a long REPL session, far below what could
/// pressure the 256 MiB code arena. Tighten per domain with [`Host::set_jit_quota`].
const JIT_DEFAULT_MAX_UNITS: u32 = 4096;
const JIT_DEFAULT_MAX_BLOB_BYTES: u64 = 1 << 26; // 64 MiB of cumulative submitted IR

/// One §14 module grant: the verified module's functions, declared window size, data segments — what
/// spawning a child domain of it needs — and its first-class export table, so a parent can resolve a
/// child entry **by name** (`Module` op 0, F2) instead of hardcoding its funcidx.
struct ModuleGrant {
    funcs: Arc<[Func]>,
    memory_log2: Option<u8>,
    data: Arc<[Data]>,
    exports: Arc<[svm_ir::Export]>,
    /// The module's import manifest (IMPORTS.md phase 3 / S2.1): retained so a child spawn can
    /// bind each slot against the child's granted powerbox - the child executes `call.import`
    /// through these bindings; nothing is rewritten and no handle is stashed in its window.
    imports: Arc<[svm_ir::Import]>,
    /// The module's type section (§3.5): retained beside `imports` so the child-spawn binder
    /// can resolve each import's requirement set (names + signatures) for the coverage walk.
    types: Arc<[svm_ir::TypeEntry]>,
    /// DURABILITY.md §4: the granting host attests this module is **freezable** (it ran
    /// `svm_durable::transform_module` on it — a compile-mode fact only the host knows, like
    /// verification). A durable domain refuses to instantiate a grant without this bit, so its
    /// nesting subtree stays snapshottable as a unit.
    durable: bool,
    /// Content digest of the granted module's semantic image (§4 separate-module nesting): a §14
    /// **separate-module** child records this in its `FrozenNested` residue so a thaw can resolve it
    /// against the restore host's re-granted modules (host-supplied at restore, D-scope — the module
    /// bytes never ride the artifact). Computed by [`module_digest`], shared with the codec's R5 gate.
    digest: [u8; 32],
    /// §3.6 — the whole granted module, registered as a spawned child's **self module** so a
    /// separate-module child can serve its own offers (impl-export admission, handler
    /// resolution, reflection) exactly as the top-level program serves via `set_self_module`.
    module: Arc<Module>,
}

/// The §4 nested-child module-identity digest: a content hash of a grant's **semantic image**
/// (functions, memory, data, exports), computed the same way at freeze-grant and thaw-grant so a
/// separate-module child re-attaches against the matching re-granted module. Debug info and
/// (already-resolved) imports are excluded — they don't affect execution, and stripping them keeps
/// the identity tolerant of a debug-stripped re-grant.
fn module_digest(m: &Module) -> [u8; 32] {
    let canon = Module {
        funcs: m.funcs.clone(),
        memory: m.memory,
        data: m.data.clone(),
        exports: m.exports.clone(),
        // Interface offers and the interface section are semantic (they are what wiring
        // resolves against), so they ride the identity digest like function exports do.
        impl_exports: m.impl_exports.clone(),
        types: m.types.clone(),
        imports: Vec::new(),
        debug_info: None,
    };
    svm_encode::digest256(&svm_encode::encode_module(&canon))
}

impl Default for Host {
    fn default() -> Host {
        Host::new()
    }
}

impl Host {
    pub fn new() -> Host {
        Host {
            table: vec![Slot::default(); CAP],
            stdin: Vec::new(),
            stdin_pos: 0,
            stdin_block: false,
            stdin_parked: false,
            stdout: Vec::new(),
            stderr: Vec::new(),
            out_sink: None,
            err_sink: None,
            clock_ns: 0,
            regions: Vec::new(),
            region_hook: None,
            budgets: Vec::new(),
            pipes: Vec::new(),
            attestation: Attestation::default(),
            modules: Vec::new(),
            region_factory: None,
            blockings: Vec::new(),
            host_fns: Vec::new(),
            host_fns_region: Vec::new(),
            guest_impls: Vec::new(),
            iface_intern: Vec::new(),
            import_remaps: Vec::new(),
            import_reqs: Vec::new(),
            self_module: None,
            self_instance: None,
            self_reified: BTreeMap::new(),
            svc_queue: VecDeque::new(),
            svc_results: BTreeMap::new(),
            svc_next_ticket: 0,
            domain_id: NEXT_DOMAIN_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            serve_native_ctx: 0,
            live_impls: Vec::new(),
            window_minters: Vec::new(),
            pool: None,
            rings: Vec::new(),
            async_notify: None,
            cap_pages: None,
            quota: Quota::default(),
            jit_domains: Vec::new(),
            jit_validator: None,
            jit_table_log2: 0,
            cap_record: None,
            cap_replay: None,
            durable: false,
            frozen_fibers: Vec::new(),
            frozen_vcpus: Vec::new(),
            frozen_nested: Vec::new(),
            frozen_root_sp: None,
            cap_names: Vec::new(),
            import_bindings: Vec::new(),
        }
    }

    /// Install this domain's **import-binding table** (§7 / IMPORTS.md phase 1): entry `i` is the
    /// instantiation-time resolution of the module's import `i` — the bound `(type_id, op)` plus the
    /// granted handle. An executable [`svm_ir::CAP_IMPORT_TYPE_ID`] dispatch translates through it,
    /// so a `call.import` runs without the module ever being rewritten. Host-controlled only (the
    /// guest cannot reach this); an unbound index is a `CapFault` at use, fail-closed.
    /// The import manifest of a granted §14 `Module`, or `None` for a forged/closed handle. The
    /// JIT's op-13 child builder reads it (via `svm_run::child_bind_imports`) to bind the child's
    /// slots — the same manifest the interpreter's inline spawn reads from its `ModuleGrant`.
    pub fn module_imports(&self, handle: i32) -> Option<Arc<[svm_ir::Import]>> {
        self.resolve_module(handle).ok().map(|g| g.imports.clone())
    }

    /// The type section of a granted §14 `Module` (§3.5): read beside [`Host::module_imports`]
    /// by the child-manifest binder to resolve each import's requirement set.
    pub fn module_types(&self, handle: i32) -> Option<Arc<[svm_ir::TypeEntry]>> {
        self.resolve_module(handle).ok().map(|g| g.types.clone())
    }

    /// S2.1 / IMPORTS.md phase 3 + §3.3: bind a **child** module's import manifest against this
    /// (child) host's granted powerbox, so the child's `call.import`s dispatch through instance
    /// bindings — no rewrite, no window stash. Binding, per slot, in order:
    ///
    /// 1. **A named grant that is a wired offer** (§3.3 wrap/override): a cap registered under
    ///    exactly the import's name that resolves to a [`Binding::GuestImpl`] binds the slot to
    ///    its first op whose derived signature equals the declaration (structural, fail-closed —
    ///    a name match with no signature match never silently binds).
    /// 2. **The reference policy** (`write`/`read`/`exit` → first granted cap of the matching
    ///    interface, in grant order — the auto-granted Instantiator/AddressSpace occupy slots
    ///    0/1, so an S2 grant or named grant is found first for Stream/Exit).
    /// 3. **Withhold** (§3.3): nothing bindable — a `rebindable` slot starts empty (attachable
    ///    later); a `required` slot **fails the whole spawn closed** (`Err` with the offending
    ///    import index; callers surface `-EINVAL` before any child code runs).
    ///
    /// Shared by the interpreter's inline spawn and the JIT's child builders (differential
    /// lockstep).
    pub fn bind_child_manifest(
        &mut self,
        imports: &[svm_ir::Import],
        tsec: &[svm_ir::TypeEntry],
    ) -> Result<(), u32> {
        if imports.is_empty() {
            return Ok(());
        }
        let policy = |name: &str| match name {
            "write" => Some((cap_id::STREAM, 1u32)),
            "read" => Some((cap_id::STREAM, 0u32)),
            "exit" => Some((cap_id::EXIT, 0u32)),
            _ => None,
        };
        let first_of = |h: &Host, tid: u32| -> Option<i32> {
            (0..CAP).find_map(|slot| {
                let st = &h.table[slot];
                (st.entry.is_some() && st.type_id == tid)
                    .then_some(((st.generation & GEN_MASK) << CAP_LOG2 | slot as u32) as i32)
            })
        };
        // Resolve an import's §3.5 requirement set — `(names, sigs)` — through the child's
        // type section: a flat import is the singleton `[(name, sig)]`, a grouped import its
        // interface's named op list. `None` = a malformed reference (unverified module).
        let requirement = |im: &svm_ir::Import| -> Option<(Vec<String>, Vec<FuncType>)> {
            let named: Vec<(&str, &FuncType)> = match im.shape {
                svm_ir::ImportShape::Func(t) => match tsec.get(t as usize)? {
                    svm_ir::TypeEntry::Func(ft) => vec![(im.name.as_str(), ft)],
                    _ => return None,
                },
                svm_ir::ImportShape::Interface(t) => match tsec.get(t as usize)? {
                    svm_ir::TypeEntry::Interface(elems) => elems
                        .iter()
                        .map(|e| match tsec.get(e.ty as usize)? {
                            svm_ir::TypeEntry::Func(ft) => Some((e.name.as_str(), ft)),
                            _ => None,
                        })
                        .collect::<Option<Vec<_>>>()?,
                    _ => return None,
                },
            };
            Some((
                named.iter().map(|(n, _)| n.to_string()).collect(),
                named.iter().map(|&(_, ft)| ft.clone()).collect(),
            ))
        };
        let mut bindings = Vec::with_capacity(imports.len());
        let mut remaps: Vec<Option<Arc<[u32]>>> = Vec::with_capacity(imports.len());
        let mut reqs: Vec<(Vec<String>, Vec<FuncType>)> = Vec::with_capacity(imports.len());
        for (i, im) in imports.iter().enumerate() {
            let rebindable = im.mode == svm_ir::ImportMode::Rebindable;
            let Some((req_names, req_sigs)) = requirement(im) else {
                return Err(i as u32);
            };
            reqs.push((req_names.clone(), req_sigs.clone()));
            // §3.3: a named offer grant binds directly — the parent's wrap/override. §3.5: the
            // match is the coverage walk (name-keyed against a named provider; exact positional
            // against a name-less legacy wire), producing the slot's frozen op remap.
            if let Some(h) = self.resolve_cap_name(&im.name) {
                if let Ok(entry) = self.resolve_guest_impl(h) {
                    let (en, es) = (Arc::clone(&entry.names), Arc::clone(&entry.sigs));
                    let cov = coverage_remap(&req_names, &req_sigs, &en, &es);
                    match cov.and_then(|remap| {
                        let b =
                            self.bound_import_for_impl(h, remap[0], &req_sigs[0], rebindable)?;
                        Some((b, remap))
                    }) {
                        Some((b, remap)) => {
                            bindings.push(b);
                            remaps.push(Some(remap));
                            continue;
                        }
                        None if rebindable => {
                            bindings.push(BoundImport::rebindable(0, 0, None));
                            remaps.push(None);
                            continue;
                        }
                        None => return Err(i as u32),
                    }
                }
            }
            match policy(&im.name).and_then(|(tid, iop)| first_of(self, tid).map(|c| (tid, iop, c)))
            {
                Some((tid, iop, c)) => {
                    bindings.push(BoundImport::required(tid, iop, c));
                    remaps.push(None);
                }
                None if rebindable => {
                    bindings.push(BoundImport::rebindable(0, 0, None));
                    remaps.push(None);
                }
                None => return Err(i as u32),
            }
        }
        self.set_import_bindings(bindings);
        for (i, r) in remaps.into_iter().enumerate() {
            if let Some(r) = r {
                self.set_import_remap(i, r);
            }
        }
        // Retain every slot's requirement so a later `import.attach` coverage-checks against
        // the manifest's declared shape (§3.5), not just the current binding's type_id.
        for (i, (names, sigs)) in reqs.into_iter().enumerate() {
            self.set_import_req(i, names, sigs);
        }
        Ok(())
    }

    /// The interface `type_id` behind a live handle, or `None` for a forged/closed one. Used by
    /// the child-spawn manifest binding (S2.1) to select which granted child cap satisfies an
    /// import's interface.
    pub fn type_id_of(&self, handle: i32) -> Option<u32> {
        let h = handle as u32;
        let slot = (h as usize) & (CAP - 1);
        let gen = h >> CAP_LOG2;
        let st = &self.table[slot];
        (st.entry.is_some() && (st.generation & GEN_MASK) == gen).then_some(st.type_id)
    }

    /// Read import slot `i`'s live binding (IMPORTS.md phase 1): `Some` only when the slot is
    /// bound. Used by embedder thunks and the eval loop's special-op routing to translate a
    /// `CAP_IMPORT_TYPE_ID` dispatch *before* an interface-specific interception — the same
    /// translation [`Host::cap_dispatch_slots`] applies internally.
    pub fn import_binding(&self, i: u32) -> Option<BoundImport> {
        self.import_bindings
            .get(i as usize)
            .copied()
            .filter(|b| b.bound)
    }

    pub fn set_import_bindings(&mut self, bindings: Vec<BoundImport>) {
        debug_assert!(
            bindings
                .iter()
                .all(|b| b.type_id != svm_ir::CAP_IMPORT_TYPE_ID),
            "an import binding can never target the import-dispatch pseudo-type_id"
        );
        self.import_remaps = vec![None; bindings.len()];
        self.import_reqs = vec![None; bindings.len()];
        self.import_bindings = bindings;
    }

    /// Mark this domain **durable**: its module has been freeze/thaw-instrumented, so the runtime
    /// keeps the per-context shadow-SP word pointing at the running fiber's region (D-fiber-cont
    /// option A, DURABILITY.md §12.8). Set before handing the host to [`run_with_host`] /
    /// [`run_capture_reserved_with_host`]. A non-durable run leaves the reserve untouched.
    pub fn set_durable(&mut self, durable: bool) {
        self.durable = durable;
    }

    /// Enable **blocking stdin** (a persistent interactive session — see [`Self::stdin_block`]). With it
    /// on, a `read` on an exhausted stdin buffer parks the vCPU (surfacing [`VcpuEvent::StdinPark`]) so
    /// the driver can push more input and resume, rather than returning EOF and letting the guest exit.
    pub fn set_stdin_blocking(&mut self, on: bool) {
        self.stdin_block = on;
    }

    /// Append bytes to the stdin buffer (a resumed [`VcpuEvent::StdinPark`] then reads them). The
    /// consumed prefix is not reclaimed — fine for the interactive session's modest per-query input.
    pub fn push_stdin(&mut self, bytes: &[u8]) {
        self.stdin.extend_from_slice(bytes);
    }

    /// Take the transient "the last stdin read parked" flag (the `CapCall` arm uses it to yield
    /// [`Outcome::StdinPark`]). Crate-internal: the bytecode engine lives in a sibling module.
    pub(crate) fn take_stdin_parked(&mut self) -> bool {
        core::mem::take(&mut self.stdin_parked)
    }

    /// Whether this domain runs a durable module (see [`Host::set_durable`]). Read by `drive`.
    pub fn is_durable(&self) -> bool {
        self.durable
    }

    /// The fibers the freeze driver flattened on the last freeze run (slice 3.1.5) — the host-side
    /// residue a snapshot records (their continuations are in the captured window). Empty otherwise.
    pub fn frozen_fibers(&self) -> &[FrozenFiber] {
        &self.frozen_fibers
    }

    /// Seed the frozen fibers a **thaw** must reconstruct (from a snapshot's Section 2). `drive`
    /// re-creates each in the registry before re-entering the root under `REWINDING`. Set alongside
    /// the restored (REWINDING) window and [`Host::set_durable`].
    pub fn set_frozen_fibers(&mut self, frozen: Vec<FrozenFiber>) {
        self.frozen_fibers = frozen;
    }

    /// The spawned vCPUs a multi-vCPU freeze flattened on the last freeze run (slice 3.2.1) — the
    /// host-side residue a snapshot records (their continuations are in the captured window, each in
    /// its own per-context region). Empty for a single-vCPU or ordinary run.
    pub fn frozen_vcpus(&self) -> &[FrozenVCpu] {
        &self.frozen_vcpus
    }

    /// Seed the spawned vCPUs a **thaw** must reconstruct (from a snapshot's control section). The
    /// root's re-executed `thread.spawn` re-attaches these (in spawn order) under `REWINDING`. Set
    /// alongside the restored (REWINDING) window and [`Host::set_durable`].
    pub fn set_frozen_vcpus(&mut self, frozen: Vec<FrozenVCpu>) {
        self.frozen_vcpus = frozen;
    }

    /// The §14 nested children a subtree freeze flattened on the last freeze run (DURABILITY.md §4
    /// — each child's continuation is in its own carve inside the captured window; this is the
    /// re-attach residue). Empty unless a live nested child was frozen. **Not yet carried by the
    /// snapshot codec** — `svm-snapshot::freeze` refuses while this is non-empty.
    pub fn frozen_nested(&self) -> &[FrozenNested] {
        &self.frozen_nested
    }

    /// Seed the §14 nested children a **thaw** must re-attach, alongside the restored window and
    /// [`Host::set_durable`] (the in-memory counterpart of [`Host::set_frozen_vcpus`]).
    pub fn set_frozen_nested(&mut self, frozen: Vec<FrozenNested>) {
        self.frozen_nested = frozen;
    }

    /// The root vCPU's flattened shadow-SP extent from the last multi-vCPU freeze (slice 3.2.1), or
    /// `None` for a single-vCPU freeze (its extent is the window's active-SP word). A snapshot records
    /// it alongside [`Host::frozen_vcpus`].
    pub fn frozen_root_sp(&self) -> Option<u64> {
        self.frozen_root_sp
    }

    /// Seed the root vCPU's flattened shadow-SP extent a **thaw** must restore (slice 3.2.1). Set only
    /// for a multi-vCPU thaw, alongside [`Host::set_frozen_vcpus`].
    pub fn set_frozen_root_sp(&mut self, sp: u64) {
        self.frozen_root_sp = Some(sp);
    }

    /// Begin recording the nondeterministic capability **inputs** crossing into the guest, so a
    /// re-execution can replay them (DEBUGGING.md W1). Idempotent; keeps any records already logged.
    pub fn record_caps(&mut self) {
        if self.cap_record.is_none() {
            self.cap_record = Some(Vec::new());
        }
    }

    /// The capability inputs recorded so far (empty if [`record_caps`](Host::record_caps) was never
    /// called).
    pub fn cap_tape(&self) -> CapTape {
        CapTape {
            records: self.cap_record.clone().unwrap_or_default(),
        }
    }

    /// Serve nondeterministic-input `cap.call`s from `tape` (from the start) instead of the live host
    /// — used by re-execution / time-travel so it sees identical inputs without a live powerbox.
    fn replay_caps(&mut self, tape: Arc<[CapRecord]>) {
        self.cap_replay = Some((tape, 0));
    }

    /// Whether the only run-mutable state this host has accumulated is the **restorable** replay
    /// substate (I/O streams, clock, cap cursor) — i.e. no stateful host capability has left residue a
    /// checkpoint restore would silently drop. A fresh seek-host starts with all of these empty; the
    /// guest minting a §13 region / §14 module / §12 blocking / async ring / §22 JIT domain, or the
    /// embedder granting a host-fn, populates one. While they stay empty a checkpoint restored to an
    /// earlier logical time reproduces the host faithfully (W1); otherwise the `Inspector` stops
    /// checkpointing and falls back to replay-from-clock-0.
    fn checkpoint_safe(&self) -> bool {
        self.regions.is_empty()
            && self.modules.is_empty()
            && self.blockings.is_empty()
            && self.rings.is_empty()
            && self.host_fns.is_empty()
            && self.host_fns_region.is_empty()
            && self.jit_domains.is_empty()
    }

    /// Snapshot the run-mutable substate a time-travel **checkpoint** (W1) must restore so resuming a
    /// replay from logical time `c` sees the host exactly as it was then: the I/O streams the run has
    /// produced/consumed, the deterministic clock, and the cap-replay cursor (which record the next
    /// `Clock`/stdin input is served from). Everything else on a fresh seek-host is immutable for the
    /// run (the replay tape, the empty cap table) or absent (no granted powerbox), so this small set is
    /// the whole mutable frontier. Cheap clones (`stdout`/`stderr` are typically tiny in a debug run).
    fn replay_substate(&self) -> HostReplaySubstate {
        HostReplaySubstate {
            stdin_pos: self.stdin_pos,
            stdout: self.stdout.clone(),
            stderr: self.stderr.clone(),
            clock_ns: self.clock_ns,
            cap_cursor: self.cap_replay.as_ref().map(|(_, c)| *c).unwrap_or(0),
            cap_record: self.cap_record.clone().unwrap_or_default(),
        }
    }

    /// Restore a [`replay_substate`](Host::replay_substate) snapshot onto a fresh seek-host (one already
    /// seeded with the replay tape via [`replay_caps`](Host::replay_caps) + recording). Re-points the
    /// cap-replay cursor and reinstates the I/O streams + clock, so a replay resumed from the
    /// checkpoint's logical time draws the same subsequent inputs and accumulates onto the same output.
    fn restore_replay_substate(&mut self, s: &HostReplaySubstate) {
        self.stdin_pos = s.stdin_pos;
        self.stdout = s.stdout.clone();
        self.stderr = s.stderr.clone();
        self.clock_ns = s.clock_ns;
        if let Some(slot) = self.cap_replay.as_mut() {
            slot.1 = s.cap_cursor;
        }
        self.cap_record = Some(s.cap_record.clone());
    }

    /// §15: set this domain's spawn quota (fiber/vCPU ceilings). Each limit is clamped to its hard
    /// anti-bomb ceiling ([`MAX_FIBERS`]/[`MAX_VCPUS`]) — a quota can only *tighten* — and to ≥ 1. The
    /// quota is read at run start ([`run_with_host`]→`drive`); a guest exceeding it traps cleanly
    /// (`FiberFault`/`ThreadFault`). The JIT enforces the same quota via `svm_jit` (see `svm-run`).
    pub fn set_quota(&mut self, quota: Quota) {
        self.quota = quota.clamped();
    }
    /// This domain's spawn quota (the clamped value in effect).
    pub fn quota(&self) -> Quota {
        self.quota
    }

    /// §7 register `name -> handle` in the capability-name directory (Followup F7), so a guest can
    /// `cap.self`-resolve `name` to this handle at runtime. The powerbox layer (`svm_run`) calls this
    /// for each granted handle; an embedder may add its own names. First registration of a name wins.
    pub fn register_cap_name(&mut self, name: &str, handle: i32) {
        self.cap_names.push((name.to_string(), handle));
    }

    /// Resolve a capability `name` to the handle it was registered under ([`Host::register_cap_name`]),
    /// or `None` if no grant carries that name. The backing for the guest's `cap.self` op-2 resolve.
    /// `pub` as the read half of [`Host::register_cap_name`]: an embedder that adds its own names can
    /// also look them up — e.g. `svm_run`'s module-grouped imports grant one provider instance per
    /// module by registering the module name at first grant and resolving it for the siblings.
    pub fn resolve_cap_name(&self, name: &str) -> Option<i32> {
        self.cap_names
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, h)| *h)
    }

    /// The human-readable **label** registered for `handle` (the reverse of [`Host::resolve_cap_name`]),
    /// or `None` if the handle carries no label — the cosmetic name an embedder can use in diagnostics,
    /// and the backing for the guest's `cap.self.label` reflection (Followup F9). Authority-neutral: a
    /// label is not a nominal type_id and the verifier ignores it. First registration of a handle wins.
    pub fn cap_label(&self, handle: i32) -> Option<&str> {
        self.cap_names
            .iter()
            .find(|(_, h)| *h == handle)
            .map(|(n, _)| n.as_str())
    }

    /// §4/§7 the JIT cap-path window page map (see [`Host::cap_pages`]) for window `base`, persistent
    /// across this run's `cap.call`s so a guest-grown heap page stays borrowable. Returns a fresh empty
    /// map when the base changes (a new window / run reusing this `Host`), else the existing one.
    pub fn cap_window_pages(&mut self, base: usize) -> CapPageMap {
        match &self.cap_pages {
            Some((b, m)) if *b == base => Arc::clone(m),
            _ => {
                let m = Arc::new(Mutex::new(BTreeMap::new()));
                self.cap_pages = Some((base, Arc::clone(&m)));
                m
            }
        }
    }

    /// Capture the window's per-page protections for a durable freeze of a **JIT** run
    /// (DURABILITY.md §12.3). The JIT keeps protections in the OS page tables, so — unlike the
    /// interpreter, where [`run_capture_reserved_with_host_prots`] reads its software map — a
    /// freeze reconstructs them here from the two host-side sources, mirroring the interpreter's
    /// `snapshot_prots` so the backends agree page-for-page:
    ///
    /// * the module's `readonly` data segments → `Ro` (applied to the window at instantiation but
    ///   not recorded in `cap_pages`), and
    /// * the runtime page-state map the Memory cap maintained (`cap_pages`: `map`/`unmap`/`protect`)
    ///   — which **overrides** the segment/default for any page the guest changed.
    ///
    /// One [`CapturedProt`] per [`DURABLE_SNAPSHOT_PAGE`]-byte page over `npages`; `mapped` is the
    /// backed-prefix byte length (pages beyond it default `Unmapped`). A `Backed` (§13) page can't
    /// arise here — a domain holding a shared region isn't freezable (its handle is non-durable).
    pub fn capture_window_prots(
        &self,
        data: &[Data],
        mapped: u64,
        npages: usize,
    ) -> Vec<CapturedProt> {
        // Default: Rw in the backed prefix, Unmapped in the reserved tail.
        let mut out: Vec<CapturedProt> = (0..npages as u64)
            .map(|p| {
                if p * DURABLE_SNAPSHOT_PAGE < mapped {
                    CapturedProt::Rw
                } else {
                    CapturedProt::Unmapped
                }
            })
            .collect();
        // `readonly` data segments → Ro. Protection is **host-page** granular (the runtime's
        // `protect_ro` and the interpreter's `init_data` protect the whole host page), so a
        // segment marks every codec page of the host page(s) it touches — this is what makes the
        // result match `snapshot_prots` on a host whose page is larger than a codec page.
        let host = host_page_size();
        for d in data {
            if !d.readonly || d.bytes.is_empty() {
                continue;
            }
            let lo = (d.offset / host) * host / DURABLE_SNAPSHOT_PAGE;
            let hi =
                (d.offset + d.bytes.len() as u64).div_ceil(host) * host / DURABLE_SNAPSHOT_PAGE;
            for p in lo..hi.min(npages as u64) {
                out[p as usize] = CapturedProt::Ro;
            }
        }
        // Runtime map/unmap/protect (host-page-indexed) override the defaults.
        if let Some((_, map)) = &self.cap_pages {
            let m = map.lock().unwrap();
            for (p, slot) in out.iter_mut().enumerate() {
                let hp = (p as u64 * DURABLE_SNAPSHOT_PAGE) / host;
                match m.get(&hp) {
                    Some(1) => *slot = CapturedProt::Rw,
                    Some(2) => *slot = CapturedProt::Ro,
                    Some(3) => *slot = CapturedProt::Unmapped,
                    _ => {}
                }
            }
        }
        out
    }

    /// Install the §9/§12 async-completion `notify` hook (the executor that owns the wake mechanism
    /// wires it at run start; see [`Host::async_notify`]). The interp's `drive` calls this with the M:N
    /// `Scheduler::notify`; the JIT wires its futex via the same seam (`svm_jit::AsyncHostHooks`).
    /// Cleared at run end to drop the closure's executor reference.
    pub fn set_async_notify(&mut self, f: Arc<dyn Fn(u64, u32) + Send + Sync>) {
        self.async_notify = Some(f);
    }
    pub fn clear_async_notify(&mut self) {
        self.async_notify = None;
    }
    /// Drain any in-flight offload-pool jobs (run end), so no worker still holds the window backing (or
    /// the JIT's window/`Domain` pointers) when the caller frees them.
    pub fn quiesce_pool(&self) {
        if let Some(p) = &self.pool {
            p.quiesce();
        }
    }

    /// Install a host binding in a free slot and return the guest handle — a forgeable
    /// `i32` index encoding `(generation, slot)`. This is how the powerbox (and, later,
    /// attenuation) hands authority to the guest (§3c). Panics only if the table is
    /// full (a host bug, not reachable from guest code).
    /// Fallible grant: claim a free handle-table slot for `binding`, or `None` if the table is full
    /// (all `CAP` slots live). **Guest-minting** ops (`AddressSpace.sub`, `create_region`, the
    /// cross-domain `SharedRegion.grant`) must use this and surface `None` as `-EMFILE` — a guest can
    /// call them in a loop, and a panic here would unwind across the JIT's `extern "C"` cap thunk and
    /// abort the host (a guest must never crash the host; §5). Host-side powerbox setup uses the
    /// infallible [`Host::grant`] (it grants a bounded few into a fresh table at instantiation).
    fn try_grant(&mut self, type_id: u32, binding: Binding) -> Option<i32> {
        let slot = self.table.iter().position(|s| s.entry.is_none())?;
        let s = &mut self.table[slot];
        s.generation = s.generation.wrapping_add(1); // advance per (re)grant (ABA-safe)
        s.entry = Some(binding);
        s.type_id = type_id;
        Some((((s.generation & GEN_MASK) << CAP_LOG2) | slot as u32) as i32)
    }

    /// §7 capability **reflection** (backs `cap.self.count`/`cap.self.get`): the live handle-table
    /// entries this **domain** holds, as `(handle, type_id)` in slot order. The handle value is the
    /// same `(generation << CAP_LOG2) | slot` a grant returns, so the guest can *use* what it
    /// discovers. Read-only and authority-neutral — every handle is one the domain already holds,
    /// so this confers nothing and reveals nothing the host did not grant. The `Host` *is* the
    /// domain's table (a nested §14 child gets a fresh one), so this auto-scopes to the caller.
    /// §7 reflection dispatch (`cap.self.*`), shared by both backends: op 0 `count() -> [n]`; op 1
    /// `get(idx) -> [handle, type_id]` (out-of-range index is fail-closed). The interpreter calls this
    /// for its `CapSelf*` ops, and the JIT routes them through the `cap.call` thunk with the reserved
    /// [`CAP_SELF_TYPE_ID`] — so the two stay in lockstep over one implementation.
    fn self_dispatch(&self, op: u32, args: &[i64]) -> Result<Vec<i64>, Trap> {
        let caps = self.self_caps();
        match op {
            0 => Ok(vec![caps.len() as i64]),
            1 => {
                let idx = *args.first().ok_or(Trap::Malformed)?;
                let (handle, type_id) = usize::try_from(idx)
                    .ok()
                    .and_then(|i| caps.get(i).copied())
                    .ok_or(Trap::Malformed)?;
                Ok(vec![handle as i64, type_id as i64])
            }
            // §6 `cap.self.attest`: the domain's platform-vouched provenance, packed as
            // `tier | (window_exposed << 8) | (freeze_exposed << 9)` — the non-interposable trust anchor.
            4 => Ok(vec![self.attestation.packed() as i64]),
            // §3.1 `cap.self.provenance(handle) -> i32` (IMPORTS.md): the binding's provenance
            // class — `0` = **platform-terminated** (host-native vtable), `d ≥ 1` =
            // **ancestor-terminated** (a wired guest impl terminating `d` domain boundaries up:
            // 1 where it was wired, +1 per re-grant hop). Interface identity is structural (D59),
            // so this is the one honest bit typing deliberately cannot answer; it lives in the
            // non-interposable namespace — a parent can interpose every capability but cannot
            // hide that it did. A forged/closed handle is an inert `CapFault` (§3c).
            5 => {
                let h = *args.first().ok_or(Trap::Malformed)? as i32;
                if let Ok(e) = self.resolve_guest_impl(h) {
                    return Ok(vec![e.depth as i64]);
                }
                // Not a wired offer: any live binding is platform-terminated (the vtable is
                // host-native); dead/forged stays an inert CapFault via the canonical resolve.
                let expect = self.table[(h as u32 as usize) & (CAP - 1)].type_id;
                self.resolve(h, expect).map(|_| vec![0])
            }
            _ => Err(Trap::Malformed),
        }
    }

    fn self_caps(&self) -> Vec<(i32, u32)> {
        self.table
            .iter()
            .enumerate()
            .filter(|(_, s)| s.entry.is_some())
            .map(|(slot, s)| {
                let handle = (((s.generation & GEN_MASK) << CAP_LOG2) | slot as u32) as i32;
                (handle, s.type_id)
            })
            .collect()
    }

    /// Infallible grant for **host-controlled** powerbox setup (`grant_stream`/`grant_memory`/… and
    /// the `grant_*` embedder APIs): the host grants a bounded handful into a fresh `CAP`-slot table,
    /// so the table cannot be full. Never call this on a **guest-reachable** path — use
    /// [`Host::try_grant`] there (a guest can exhaust the table; see its docs).
    fn grant(&mut self, type_id: u32, binding: Binding) -> i32 {
        self.try_grant(type_id, binding)
            .expect("handle table full during host powerbox setup (bounded by construction)")
    }

    /// Classify the **live** handle table for a snapshot (DURABILITY.md §12.5 Section 3). On
    /// success, the re-grantable handles in ascending slot order. Refuses — `Err` naming the
    /// **first** offending slot — if any live slot holds a non-durable binding, so freeze is
    /// all-or-nothing rather than dropping authority on restore.
    pub fn capture_durable_handles(&self) -> Result<Vec<DurableHandle>, NonDurableHandle> {
        let mut out = Vec::new();
        for (slot, s) in self.table.iter().enumerate() {
            let Some(binding) = s.entry else { continue };
            let binding = match binding {
                Binding::Stream(role) => DurableBinding::Stream(role),
                Binding::Exit => DurableBinding::Exit,
                Binding::Clock => DurableBinding::Clock,
                Binding::Memory => DurableBinding::Memory,
                Binding::Yielder => DurableBinding::Yielder,
                Binding::AddressSpace { base, size } => DurableBinding::AddressSpace { base, size },
                Binding::Instantiator { base, size } => DurableBinding::Instantiator { base, size },
                Binding::SharedRegion(_) => {
                    return Err(self.non_durable(slot, NonDurableKind::SharedRegion))
                }
                Binding::Module(_) => return Err(self.non_durable(slot, NonDurableKind::Module)),
                Binding::IoRing(_) => return Err(self.non_durable(slot, NonDurableKind::IoRing)),
                Binding::Blocking(_) => {
                    return Err(self.non_durable(slot, NonDurableKind::Blocking))
                }
                Binding::JitDomain(_) => {
                    return Err(self.non_durable(slot, NonDurableKind::JitDomain))
                }
                Binding::JitCode { .. } => {
                    return Err(self.non_durable(slot, NonDurableKind::JitCode))
                }
                Binding::HostFn(_) | Binding::HostFnRegion(_) => {
                    return Err(self.non_durable(slot, NonDurableKind::HostFn))
                }
                Binding::GuestImpl(_) => {
                    return Err(self.non_durable(slot, NonDurableKind::GuestImpl))
                }
                Binding::LiveImpl(_) => {
                    return Err(self.non_durable(slot, NonDurableKind::LiveImpl))
                }
                Binding::Budget(_) => return Err(self.non_durable(slot, NonDurableKind::Budget)),
                Binding::PipeEnd { .. } => return Err(self.non_durable(slot, NonDurableKind::Pipe)),
                Binding::WindowMinter(_) => {
                    return Err(self.non_durable(slot, NonDurableKind::WindowMinter))
                }
            };
            out.push(DurableHandle {
                slot: slot as u32,
                generation: s.generation,
                type_id: s.type_id,
                binding,
            });
        }
        Ok(out)
    }

    /// **Drain the live non-durable handles** so the domain becomes snapshottable (DURABILITY.md §12.5
    /// "drainable non-durable bindings" / Phase-4 handle hardening). Closes every live slot holding a
    /// binding [`Self::capture_durable_handles`] would refuse on — the ones carrying out-of-line host
    /// state or native pointers (`SharedRegion`/`Module`/`IoRing`/`Blocking`/`JitDomain`/`JitCode`/
    /// `HostFn`) — and leaves the durable handles untouched. Each close frees the slot but **keeps its
    /// generation** (D37), so a guest's stale handle value becomes a dead generation and any later
    /// `cap.call` on it is an inert `CapFault`, never authority into a recycled slot. Returns the drained
    /// handles in ascending slot order (for the embedder to audit the relinquished authority). The exact
    /// complement of [`Self::capture_durable_handles`]'s refusal set: after a drain, `capture` succeeds,
    /// so a subtree that held non-durable authority can now be frozen.
    ///
    /// This is authority **relinquishment**, the embedder's counterpart to "freeze refuses unless the
    /// non-durable handles are closed/drained first": the guest could never reach these capabilities
    /// across a restore (they aren't re-grantable), so dropping them is the only way to make the freeze
    /// proceed. Call at a freeze safepoint — the STW quiesce + §12.8 4A.7 guarantee no vCPU is mid-host
    /// call, and the embedder drains any async offload residue ([`Self::quiesce_pool`]) first, so closing
    /// the slot orphans no in-flight work. The out-of-line backings (`rings`/`blockings`/`host_fns`/…)
    /// are released when this per-run `Host` is dropped after the snapshot.
    pub fn drain_non_durable(&mut self) -> Vec<NonDurableHandle> {
        let mut drained = Vec::new();
        for slot in 0..self.table.len() {
            let Some(binding) = self.table[slot].entry else {
                continue;
            };
            let kind = match binding {
                // Durable (value-typed) — re-grantable on restore, so keep them.
                Binding::Stream(_)
                | Binding::Exit
                | Binding::Clock
                | Binding::Memory
                | Binding::Yielder
                | Binding::AddressSpace { .. }
                | Binding::Instantiator { .. } => continue,
                Binding::SharedRegion(_) => NonDurableKind::SharedRegion,
                Binding::Module(_) => NonDurableKind::Module,
                Binding::IoRing(_) => NonDurableKind::IoRing,
                Binding::Blocking(_) => NonDurableKind::Blocking,
                Binding::JitDomain(_) => NonDurableKind::JitDomain,
                Binding::JitCode { .. } => NonDurableKind::JitCode,
                Binding::HostFn(_) | Binding::HostFnRegion(_) => NonDurableKind::HostFn,
                Binding::GuestImpl(_) => NonDurableKind::GuestImpl,
                Binding::LiveImpl(_) => NonDurableKind::LiveImpl,
                Binding::Budget(_) => NonDurableKind::Budget,
                Binding::PipeEnd { .. } => NonDurableKind::Pipe,
                Binding::WindowMinter(_) => NonDurableKind::WindowMinter,
            };
            drained.push(NonDurableHandle {
                slot: slot as u32,
                type_id: self.table[slot].type_id,
                kind,
            });
            self.table[slot].entry = None; // close: free the slot, retain the generation (D37)
        }
        drained
    }

    fn non_durable(&self, slot: usize, kind: NonDurableKind) -> NonDurableHandle {
        NonDurableHandle {
            slot: slot as u32,
            type_id: self.table[slot].type_id,
            kind,
        }
    }

    /// Reinstate a captured durable set into this table (DURABILITY.md §12.5), pinning each
    /// `(slot, generation)` so guest-held handle values stay valid across restore. Intended
    /// for the restore path on a **fresh** table; entries already at those slots are
    /// overwritten. Slots must be in range (the snapshot codec validates that against
    /// [`Host::handle_capacity`] before calling).
    pub fn restore_durable_handles(&mut self, handles: &[DurableHandle]) {
        for h in handles {
            let binding = match h.binding {
                DurableBinding::Stream(role) => Binding::Stream(role),
                DurableBinding::Exit => Binding::Exit,
                DurableBinding::Clock => Binding::Clock,
                DurableBinding::Memory => Binding::Memory,
                DurableBinding::Yielder => Binding::Yielder,
                DurableBinding::AddressSpace { base, size } => Binding::AddressSpace { base, size },
                DurableBinding::Instantiator { base, size } => Binding::Instantiator { base, size },
            };
            self.grant_at(h.slot, h.generation, h.type_id, binding);
        }
    }

    /// The handle-table capacity (`CAP`): valid slot indices are `0..handle_capacity()`. The
    /// snapshot codec validates a captured slot against this before [`Host::restore_durable_handles`].
    pub fn handle_capacity() -> u32 {
        CAP as u32
    }

    /// Pin `binding` at an exact `(slot, generation)` — the restore primitive behind
    /// [`Host::restore_durable_handles`] (DURABILITY.md §12.5). Unlike [`Host::grant`], it
    /// neither picks a free slot nor advances the generation: the snapshot already fixed both,
    /// and the guest holds handle values that encode them. Panics on an out-of-range slot (a
    /// codec bug, not reachable from guest code — like `grant`'s table-full panic).
    fn grant_at(&mut self, slot: u32, generation: u32, type_id: u32, binding: Binding) {
        let s = &mut self.table[slot as usize];
        s.generation = generation;
        s.entry = Some(binding);
        s.type_id = type_id;
    }

    /// Grant a `Stream` capability bound to `role` (a powerbox stdio grant, §3e).
    pub fn grant_stream(&mut self, role: StreamRole) -> i32 {
        self.grant(cap_id::STREAM, Binding::Stream(role))
    }

    /// §4 / S4 — mint a **host-served pipe** and grant both ends, returning `(write_handle,
    /// read_handle)`. Both are `Stream`-typed (a pipe end is a stream: read/write/close), backed by one
    /// FIFO in [`Host::pipes`]: bytes written to the write end are drained by the read end (non-blocking,
    /// FIFO order). The personality's byte-IPC primitive — a shell wiring `cmd1 | cmd2` grants each side
    /// one end. (Cross-domain granting of an end into a child is a follow-up — the FIFO would move to a
    /// shared backing, like `SharedRegion`; today both ends live in the minting domain, e.g. between its
    /// own fibers.)
    pub fn grant_pipe(&mut self) -> (i32, i32) {
        let pipe = self.pipes.len() as u32;
        self.pipes.push(Arc::new(Mutex::new(VecDeque::new())));
        let w = self.grant(cap_id::STREAM, Binding::PipeEnd { pipe, write: true });
        let r = self.grant(cap_id::STREAM, Binding::PipeEnd { pipe, write: false });
        (w, r)
    }

    /// Resolve a handle to a **pipe end** — `(is_write, shared FIFO backing)` — or `None` if it is not
    /// a live `PipeEnd`. The re-grant counterpart of [`Self::resolve_copyable`] for the one
    /// index-carrying cap a §14 child can be handed: it returns the shared backing (not the parent-local
    /// index) so [`Self::install_pipe_end`] can alias it into a child `Host`.
    fn resolve_pipe_end(&self, handle: i32) -> Option<(bool, PipeBacking)> {
        match self.resolve(handle, cap_id::STREAM) {
            Ok(Binding::PipeEnd { pipe, write }) => {
                Some((write, Arc::clone(self.pipes.get(pipe as usize)?)))
            }
            _ => None,
        }
    }

    /// Install a re-granted pipe end (its shared `backing` aliased from the granting domain) into this
    /// `Host` and return its handle. The child sees the **same** FIFO as the parent's other end — how a
    /// pipe crosses a §14 domain boundary.
    fn install_pipe_end(&mut self, write: bool, backing: PipeBacking) -> i32 {
        let pipe = self.pipes.len() as u32;
        self.pipes.push(backing);
        self.grant(cap_id::STREAM, Binding::PipeEnd { pipe, write })
    }

    /// PROCESS.md S2 — promote `stdout` to a **shared** sink and return a handle to it, so a child
    /// host can be pointed at the same buffer (stdio inheritance via `instantiate_granted`).
    /// Idempotent (a second call returns the same sink). After promotion, new writes go to the sink;
    /// read the effective bytes via [`Host::stdout_bytes`] (`stdout` itself no longer receives them).
    pub fn shared_stdout(&mut self) -> Arc<Mutex<Vec<u8>>> {
        if self.out_sink.is_none() {
            let taken = std::mem::take(&mut self.stdout);
            self.out_sink = Some(Arc::new(Mutex::new(taken)));
        }
        Arc::clone(self.out_sink.as_ref().unwrap())
    }

    /// S2 — the stderr analogue of [`Host::shared_stdout`].
    pub fn shared_stderr(&mut self) -> Arc<Mutex<Vec<u8>>> {
        if self.err_sink.is_none() {
            let taken = std::mem::take(&mut self.stderr);
            self.err_sink = Some(Arc::new(Mutex::new(taken)));
        }
        Arc::clone(self.err_sink.as_ref().unwrap())
    }

    /// S2 — the effective stdout: the shared sink's bytes if it was promoted (child stdio shared in),
    /// else the local `stdout` Vec. Use this instead of reading `stdout` directly when a child may
    /// have inherited this host's stdout.
    pub fn stdout_bytes(&self) -> Vec<u8> {
        match &self.out_sink {
            Some(s) => s.lock().unwrap_or_else(|e| e.into_inner()).clone(),
            None => self.stdout.clone(),
        }
    }

    /// S2 — the stderr analogue of [`Host::stdout_bytes`].
    pub fn stderr_bytes(&self) -> Vec<u8> {
        match &self.err_sink {
            Some(s) => s.lock().unwrap_or_else(|e| e.into_inner()).clone(),
            None => self.stderr.clone(),
        }
    }
    pub fn grant_exit(&mut self) -> i32 {
        self.grant(cap_id::EXIT, Binding::Exit)
    }
    pub fn grant_clock(&mut self) -> i32 {
        self.grant(cap_id::CLOCK, Binding::Clock)
    }
    pub fn grant_memory(&mut self) -> i32 {
        self.grant(cap_id::MEMORY, Binding::Memory)
    }
    /// Grant a §9/§12 `IoRing` capability — authority to `submit` batched/deferred `cap.call`s
    /// (synchronously via op 0, or asynchronously via op 1 `submit_async` + op 2 `reap`).
    pub fn grant_io_ring(&mut self) -> i32 {
        let idx = self.rings.len() as u32;
        self.rings.push(Arc::new(RingState::default()));
        self.grant(cap_id::IO_RING, Binding::IoRing(idx))
    }
    /// Grant a §12 `Blocking` capability — a mock synchronous/blocking host op the offload pool can
    /// overlap. `block_for` is how long each op blocks (`Duration::ZERO` for a pure compute);
    /// `rendezvous` (test-only) installs a width-`w` [`Barrier`] so a batch of exactly `w` ops on a
    /// `≥ w`-thread pool deterministically co-resides (proving overlap without timing).
    pub fn grant_blocking(&mut self, block_for: Duration, rendezvous: Option<usize>) -> i32 {
        let state = Arc::new(AsyncState {
            block_for,
            rendezvous: rendezvous.map(|w| Arc::new(Barrier::new(w))),
            active: AtomicUsize::new(0),
            max_active: AtomicUsize::new(0),
        });
        let idx = self.blockings.len() as u32;
        self.blockings.push(state);
        self.grant(cap_id::BLOCKING, Binding::Blocking(idx))
    }
    /// Read back the [`AsyncState`] behind a granted `Blocking` handle (a test inspects `max_active`
    /// to confirm a batched `submit` overlapped on the pool). `None` if the handle isn't a `Blocking`.
    pub fn blocking_state(&self, handle: i32) -> Option<Arc<AsyncState>> {
        match self.resolve(handle, cap_id::BLOCKING) {
            Ok(Binding::Blocking(idx)) => self.blockings.get(idx as usize).cloned(),
            _ => None,
        }
    }

    /// §7 Register an **embedder host-capability** handler and grant a handle to it (iface
    /// [`cap_id::HOST_FN`]). The guest reaches it with `cap.call HOST_FN <op> <handle> (args)`; the
    /// closure supplies the semantics, so a host adds a capability (e.g. an `svm-wasi` shim) without
    /// changing the VM. The handler is host code in the **authority** TCB — it sees the guest window
    /// (masked `GuestMem`) but is reached only through this masked, type-checked handle.
    pub fn grant_host_fn(&mut self, f: HostFn) -> i32 {
        let idx = self.host_fns.len() as u32;
        self.host_fns.push(f);
        self.grant(cap_id::HOST_FN, Binding::HostFn(idx))
    }

    /// §4b Register an **mmap-capable** embedder host-capability handler and grant a handle to it
    /// (also iface [`cap_id::HOST_FN`], so a guest resolves it exactly like a plain [`grant_host_fn`]).
    /// Identical to `grant_host_fn` except the handler is additionally handed a [`RegionMinter`] on
    /// each call, so it can mint a file-backed `SharedRegion` and return the handle — the delivery
    /// mechanism for the zero-copy file-mmap bridge. The extra authority is exactly region-minting
    /// (nothing else of the `Host` is reachable).
    pub fn grant_host_fn_region(&mut self, f: HostFnRegion) -> i32 {
        let idx = self.host_fns_region.len() as u32;
        self.host_fns_region.push(f);
        self.grant(cap_id::HOST_FN, Binding::HostFnRegion(idx))
    }

    /// Intern an interface's op-signature list and return its id (IMPORTS.md §3.2): structurally
    /// identical lists collide to the same id, so **id-equality ≡ structural equality** within
    /// this `Host` (D59 applied to capability interfaces — a parent-implemented interface is
    /// typewise indistinguishable from any other structurally-equal one; provenance, not typing,
    /// is the honest bit, §3.1). Guest ids allocate from [`cap_id::GUEST_IMPL_BASE`] upward — but a
    /// declaration structurally equal to a **pre-seeded built-in shape** ([`preseeded_iface_shapes`],
    /// IMPORTS.md §3.5) interns to that built-in's fixed id instead, so a guest that declares (e.g.)
    /// the `Stream` interface is typewise the same as a real host stream and an import slot requiring
    /// that shape accepts either. This grants no authority — only a real handle of the matching
    /// [`Binding`] can be called.
    pub fn intern_interface(&mut self, sigs: &[FuncType]) -> u32 {
        if let Some(id) = preseeded_iface_id(sigs) {
            return id;
        }
        if let Some(i) = self.iface_intern.iter().position(|s| **s == *sigs) {
            return cap_id::GUEST_IMPL_BASE + i as u32;
        }
        self.iface_intern.push(sigs.into());
        cap_id::GUEST_IMPL_BASE + (self.iface_intern.len() - 1) as u32
    }

    /// **Wire an interface offer into this domain's table** (IMPORTS.md §3.2) and return the
    /// handle — the authority-moving act. `funcs` is the offering module's function table and
    /// `ops` its offer's per-op funcidx list ([`svm_ir::ImplExport::ops`], verifier-checked
    /// in-range for a verified module; re-checked here fail-closed since the wirer is host-side
    /// code). Op signatures are **derived** from the named functions' declared types and interned
    /// ([`Host::intern_interface`]), so the entry's `type_id` is its structural identity. Only a
    /// wiring party holding both ends calls this — declaring the offer conferred nothing.
    ///
    /// Returns `None` (nothing minted) for an empty op list or an out-of-range funcidx.
    pub fn wire_impl(&mut self, funcs: &Arc<[Func]>, ops: &[u32]) -> Option<i32> {
        if ops.is_empty() || ops.iter().any(|&f| f as usize >= funcs.len()) {
            return None;
        }
        let sigs: Arc<[FuncType]> = ops
            .iter()
            .map(|&f| FuncType {
                params: funcs[f as usize].params.clone(),
                results: funcs[f as usize].results.clone(),
            })
            .collect();
        let type_id = self.intern_interface(&sigs);
        let idx = self.guest_impls.len() as u32;
        self.guest_impls.push(GuestImplEntry {
            funcs: Arc::clone(funcs),
            ops: ops.into(),
            sigs,
            names: Arc::from(Vec::new()),
            type_id,
            depth: 1,
            state: None,
        });
        Some(self.grant(type_id, Binding::GuestImpl(idx)))
    }

    /// Wire an **instanced** offer (IMPORTS.md §3.2 v2 — exporter-domain state): like
    /// [`Host::wire_impl`], but the offer gets a persistent **provider domain** — a window built
    /// once from `m`'s memory declaration + data segments, and its own (initially empty)
    /// powerbox — that every op dispatch runs over, so state survives across calls. The wirer
    /// may re-grant capabilities into the provider with [`Host::grant_impl_cap`] (the
    /// wrap-holding-its-real-cap story). Fail-closed like `wire_impl`; `m.funcs` must be
    /// verifier-passing (the host is trusted to wire only verified modules, as with every grant).
    pub fn wire_impl_instance(&mut self, m: &Module, ops: &[u32]) -> Option<i32> {
        let funcs: Arc<[Func]> = m.funcs.clone().into();
        if ops.is_empty() || ops.iter().any(|&f| f as usize >= funcs.len()) {
            return None;
        }
        let sigs: Arc<[FuncType]> = ops
            .iter()
            .map(|&f| FuncType {
                params: funcs[f as usize].params.clone(),
                results: funcs[f as usize].results.clone(),
            })
            .collect();
        // The provider's window, exactly as a run of `m` would build it (§3a data segments
        // included) — the exporter's own memory, not the wirer's and not any caller's.
        let mem = m.memory.map(|mc| {
            let mut mm = Mem::with_reservation(DEFAULT_RESERVED_LOG2, mc.size_log2);
            mm.init_data(&m.data);
            mm
        });
        let type_id = self.intern_interface(&sigs);
        let idx = self.guest_impls.len() as u32;
        self.guest_impls.push(GuestImplEntry {
            funcs,
            ops: ops.into(),
            sigs,
            names: Arc::from(Vec::new()),
            type_id,
            depth: 1,
            state: Some(Arc::new(Mutex::new(ProviderState {
                mem,
                host: Host::new(),
                fuel: PROVIDER_FUEL_RESERVE,
            }))),
        });
        Some(self.grant(type_id, Binding::GuestImpl(idx)))
    }

    /// §3.5: register the running module's self-referential surface (type-section interfaces,
    /// impl exports, function table, memory template) so `call.import.dyn`,
    /// `cap.self.type_id`, `cap.self.covers`, and `export.handle` resolve through one host-side
    /// entry on all three backends. Unregistered, those ops fail closed (probeable `CapFault`).
    pub fn set_self_module(&mut self, m: &Arc<Module>) {
        self.self_module = Some(Arc::clone(m));
    }

    /// §3.6 slice 2 — enqueue a dispatch onto this domain's bounded inbound queue, to be served
    /// at the guest's next `svc.poll` service point. Returns the completion ticket, or `None`
    /// **fail-closed** when the queue is full (backpressure is the enqueuer's problem, probeable)
    /// or the target isn't a well-formed offer op of the registered self module. Embedder-side
    /// this slice; a cross-domain caller's enqueue is the §3.6 caller-parking slice.
    pub fn svc_enqueue(&mut self, export: u32, op: u32, args: Vec<i64>) -> Option<u64> {
        if self.svc_queue.len() >= SVC_QUEUE_CAP || self.svc_handler_func(export, op).is_none() {
            return None;
        }
        let ticket = self.svc_next_ticket;
        self.svc_next_ticket += 1;
        self.svc_queue.push_back(SvcDispatch {
            export,
            op,
            args,
            ticket,
        });
        Some(ticket)
    }

    /// Take a served dispatch's result (first result slot; `0` for a no-result handler).
    /// `None` while unserved — the completion cell is filled at the `svc.poll` that ran it.
    pub fn svc_result(&mut self, ticket: u64) -> Option<i64> {
        self.svc_results.remove(&ticket)
    }

    /// DURABILITY.md §13.4 step 3 — this domain's **serve state** for the snapshot codec:
    /// `(queue, completion cells, next ticket)`. Plain data (the §13.2 inventory row
    /// "serialize as-is"); the completion cells come out in ascending-ticket order (the
    /// `BTreeMap` iteration), which is the artifact's canonical order.
    pub fn svc_state(&self) -> (Vec<SvcDispatch>, Vec<(u64, i64)>, u64) {
        (
            self.svc_queue.iter().cloned().collect(),
            self.svc_results.iter().map(|(&t, &v)| (t, v)).collect(),
            self.svc_next_ticket,
        )
    }

    /// DURABILITY.md §13.4 step 3 — restore this domain's serve state from a snapshot's
    /// serve section (the inverse of [`Host::svc_state`]). Replaces whatever is present.
    pub fn set_svc_state(&mut self, queue: Vec<SvcDispatch>, results: Vec<(u64, i64)>, next: u64) {
        self.svc_queue = queue.into();
        self.svc_results = results.into_iter().collect();
        self.svc_next_ticket = next;
    }

    /// DURABILITY.md §13.3 — this domain's stable identity (see the field doc).
    pub fn domain_id(&self) -> u64 {
        self.domain_id
    }

    /// I36 slice 3 — pop the next queued dispatch as `(export, op, args, ticket)`, for an
    /// **embedder-side** serve loop (the JIT cap thunk's native `svc.poll`/`svc.wait` arm,
    /// which invokes the handler's compiled trampoline itself). The interpreter backends
    /// drain the queue in their own serve arms and never call this.
    pub fn svc_pop(&mut self) -> Option<(u32, u32, Vec<i64>, u64)> {
        self.svc_queue
            .pop_front()
            .map(|d| (d.export, d.op, d.args, d.ticket))
    }

    /// I36 slice 3 — settle ticket `ticket`'s completion cell with `v`: the embedder serve
    /// loop's counterpart of the eval-loop settle (the enqueuer claims it via
    /// [`Host::svc_result`]).
    pub fn svc_settle(&mut self, ticket: u64, v: i64) {
        self.svc_results.insert(ticket, v);
    }

    /// I36 slice 3 — the handler [`FuncIdx`] a queued `(export, op)` dispatch runs, for the
    /// embedder serve loop (public face of [`Host::svc_handler_func`]).
    pub fn svc_handler(&self, export: u32, op: u32) -> Option<FuncIdx> {
        self.svc_handler_func(export, op)
    }

    /// I36 slice 3 — register (or clear, `0`) the JIT embedder's serve-loop native context
    /// (its `*mut CompiledModule` as a `usize`); see [`Host::serve_native_ctx`].
    pub fn set_serve_native_ctx(&mut self, ctx: usize) {
        self.serve_native_ctx = ctx;
    }

    /// The registered serve-loop native context (`0` ⇒ interpreter run / none registered).
    pub fn serve_native_ctx(&self) -> usize {
        self.serve_native_ctx
    }

    /// §3.6 — the shape of THIS domain's impl-export `export` (its op signatures, in op
    /// order), for a wirer minting a live offer over it: the offer *is* the callee's export,
    /// so its shape resolves against the callee's own registered module — which, for a
    /// separate-module child, differs from the wirer's. `None` fails the wire closed (no
    /// registered module / malformed export).
    fn offer_shape(&self, export: u32) -> Option<Vec<FuncType>> {
        let m = self.self_module.as_ref()?;
        let e = m.impl_exports.get(export as usize)?;
        let named = m.interface_named_ops(e.interface)?;
        Some(named.iter().map(|&(_, ft)| ft.clone()).collect())
    }

    /// §3.6 slice 3 — mint a **live-callee offer** into this (the wirer's) table: a capability
    /// whose provider is the *running* domain behind `callee` (a §14 child's live powerbox),
    /// targeting its impl-export `export`. `sigs` is the export's shape — the **callee's**
    /// ([`Host::offer_shape`] on it; the caller fetches it first so the two powerbox locks are
    /// never held together) — interned structurally into the wirer's table (D59: the id ≡ the
    /// shape, so a same-module child lands on the identical id the wirer's own module would).
    /// A call through the handle enqueues on the callee and parks the caller (the eval loop's
    /// caller-parking arm).
    pub fn wire_live_impl(
        &mut self,
        callee: &Arc<Mutex<Host>>,
        export: u32,
        sigs: &[FuncType],
    ) -> Result<i32, Trap> {
        Ok(self.install_live_impl(Arc::clone(callee), export, sigs.into()))
    }

    /// Install a live-callee offer entry + grant its handle — shared by the wire
    /// ([`Host::wire_live_impl`]) and the child re-grant ([`Host::regrant_into_child`]), so
    /// the two mint identical bindings.
    fn install_live_impl(
        &mut self,
        callee: Arc<Mutex<Host>>,
        export: u32,
        sigs: Arc<[FuncType]>,
    ) -> i32 {
        let type_id = self.intern_interface(&sigs);
        let idx = self.live_impls.len() as u32;
        self.live_impls.push(LiveImplEntry {
            callee,
            export,
            sigs,
        });
        self.grant(type_id, Binding::LiveImpl(idx))
    }

    /// Resolve `handle` as a live-callee offer, whatever its interface type — the re-grant
    /// path's lookup (which has only the handle, not the interned id). `None` for anything
    /// else, including forged handles.
    fn resolve_live_impl(&self, handle: i32) -> Option<&LiveImplEntry> {
        let tid = self.type_id_of(handle)?;
        match self.resolve(handle, tid).ok()? {
            Binding::LiveImpl(i) => self.live_impls.get(i as usize),
            _ => None,
        }
    }

    /// §3.6 slice 4 — the live-callee target behind import slot `i`, iff the slot is bound to a
    /// [`Binding::LiveImpl`] handle: `(callee, export, base_op)`. The `call.import` route's
    /// pre-dispatch probe (an ordinarily-bound slot answers `None` and flows to the ordinary
    /// dispatch); the consumer's op offsets from `base_op` (flat/identity mapping this slice —
    /// coverage-remapped grouped bindings ride a later slice).
    fn import_live_target(&self, i: u32) -> Option<(Arc<Mutex<Host>>, u32, u32)> {
        let b = self.import_binding(i)?;
        if !b.bound {
            return None;
        }
        match self.resolve(b.handle, b.type_id) {
            Ok(Binding::LiveImpl(idx)) => self
                .live_impls
                .get(idx as usize)
                .map(|e| (Arc::clone(&e.callee), e.export, b.op)),
            _ => None,
        }
    }

    /// The live-callee target behind `handle` iff it resolves to a [`Binding::LiveImpl`] of the
    /// right `type_id` — the eval loop's pre-dispatch probe (a non-LiveImpl handle answers
    /// `None` and flows to the ordinary dispatch).
    fn live_impl_of(&self, handle: i32, type_id: u32) -> Option<(Arc<Mutex<Host>>, u32)> {
        match self.resolve(handle, type_id) {
            Ok(Binding::LiveImpl(i)) => self
                .live_impls
                .get(i as usize)
                .map(|e| (Arc::clone(&e.callee), e.export)),
            _ => None,
        }
    }

    /// The function a queued `(export, op)` dispatch runs: the registered self module's
    /// impl-export op table entry, checked in-range against its function table. `None` fails
    /// the enqueue closed — the queue only ever holds servable dispatches.
    fn svc_handler_func(&self, export: u32, op: u32) -> Option<FuncIdx> {
        let m = self.self_module.as_ref()?;
        let e = m.impl_exports.get(export as usize)?;
        let f = *e.ops.get(op as usize)?;
        ((f as usize) < m.funcs.len()).then_some(f)
    }

    /// The named-op view of self interface `ty`: `(names, sigs)`, or `None` when no self module
    /// is registered or `ty` is not a well-formed interface entry.
    fn self_iface(&self, ty: u32) -> Option<(Vec<String>, Vec<FuncType>)> {
        let m = self.self_module.as_ref()?;
        let ops = m.interface_named_ops(ty)?;
        Some((
            ops.iter().map(|(n, _)| n.to_string()).collect(),
            ops.iter().map(|&(_, ft)| ft.clone()).collect(),
        ))
    }

    /// §3.5 `cap.self.type_id`: intern this domain's declared interface `ty` and return the
    /// runtime id — authority-neutral pure reflection (the shape is the module's own).
    pub fn self_type_id(&mut self, ty: u32) -> Result<u32, Trap> {
        let (_, sigs) = self.self_iface(ty).ok_or(Trap::CapFault)?;
        Ok(self.intern_interface(&sigs))
    }

    /// §3.5 `cap.self.covers`: does the live capability behind `handle` **cover** self
    /// interface `ty`? `1` covers, `0` live-but-does-not, `-EBADF` dead/forged.
    pub fn self_covers(&mut self, handle: i32, ty: u32) -> Result<i64, Trap> {
        let (names, sigs) = self.self_iface(ty).ok_or(Trap::CapFault)?;
        let Some(tid) = self.type_id_of(handle) else {
            return Ok(-9); // EBADF: dead or forged — probeable, never a trap
        };
        // Exact shape ⇒ covers by construction (same interned id).
        if tid == self.intern_interface(&sigs) {
            return Ok(1);
        }
        // A wired guest impl may cover a subset requirement — the name-keyed walk.
        if let Ok(e) = self.resolve_guest_impl(handle) {
            let (en, es) = (Arc::clone(&e.names), Arc::clone(&e.sigs));
            return Ok(coverage_remap(&names, &sigs, &en, &es).is_some() as i64);
        }
        Ok(0)
    }

    /// §3.5 `export.handle`: reify this domain's own impl export `k` as a capability — the only
    /// guest-reachable source of offer wiring rights (offer exposure is consent-based). All of
    /// a domain's reified offers share **one** service state (created lazily from the module's
    /// memory declaration + data segments); re-reifying returns the same handle.
    pub fn reify_export(&mut self, k: u32) -> Result<i32, Trap> {
        if let Some(&h) = self.self_reified.get(&k) {
            return Ok(h);
        }
        let m = self.self_module.clone().ok_or(Trap::CapFault)?;
        let e = m.impl_exports.get(k as usize).ok_or(Trap::CapFault)?;
        let funcs: Arc<[Func]> = m.funcs.clone().into();
        if e.ops.is_empty() || e.ops.iter().any(|&f| f as usize >= funcs.len()) {
            return Err(Trap::CapFault);
        }
        let named = m.interface_named_ops(e.interface).ok_or(Trap::CapFault)?;
        let sigs: Arc<[FuncType]> = named.iter().map(|&(_, ft)| ft.clone()).collect();
        let names: Arc<[String]> = named.iter().map(|(n, _)| n.to_string()).collect();
        let state = self
            .self_instance
            .get_or_insert_with(|| {
                let mem = m.memory.map(|mc| {
                    let mut mm = Mem::with_reservation(DEFAULT_RESERVED_LOG2, mc.size_log2);
                    mm.init_data(&m.data);
                    mm
                });
                Arc::new(Mutex::new(ProviderState {
                    mem,
                    host: Host::new(),
                    fuel: PROVIDER_FUEL_RESERVE,
                }))
            })
            .clone();
        let type_id = self.intern_interface(&sigs);
        let idx = self.guest_impls.len() as u32;
        self.guest_impls.push(GuestImplEntry {
            funcs,
            ops: Arc::from(e.ops.clone()),
            sigs,
            names,
            type_id,
            depth: 1,
            state: Some(state),
        });
        let h = self.grant(type_id, Binding::GuestImpl(idx));
        self.self_reified.insert(k, h);
        Ok(h)
    }

    /// §3.5: freeze a grouped slot's bind-time op remap (consumer-local op → provider op),
    /// computed by the coverage walk at the binding act. No-op for an out-of-range slot.
    pub fn set_import_remap(&mut self, slot: usize, remap: Arc<[u32]>) {
        if let Some(r) = self.import_remaps.get_mut(slot) {
            *r = Some(remap);
        }
    }

    /// §3.5: retain slot `slot`'s manifest requirement set so `import.attach` can coverage-check
    /// an attached capability against it. No-op for an out-of-range slot.
    pub fn set_import_req(&mut self, slot: usize, names: Vec<String>, sigs: Vec<FuncType>) {
        if let Some(r) = self.import_reqs.get_mut(slot) {
            *r = Some(Arc::new((names, sigs)));
        }
    }

    /// Re-grant one of **this** domain's capabilities into the provider instance behind `offer`,
    /// registered under `name` in the provider's §7 name directory (IMPORTS.md §3.2 v2): how a
    /// wrap comes to hold the real capability it forwards to. Same re-grant policy as a §14
    /// child (coordinate-free caps and pipe ends; stdio shares this domain's sinks) with one
    /// deliberate exception: **never another offer** — providers stay offer-free so provider
    /// chains are acyclic and the blocking provider lock can never deadlock. `None` (nothing
    /// granted) for a non-instanced offer, a non-grantable cap, or an offer-shaped `cap`.
    pub fn grant_impl_cap(&mut self, offer: i32, cap: i32, name: &str) -> Option<i32> {
        let state = self.resolve_guest_impl(offer).ok()?.state.clone()?;
        if self.resolve_guest_impl(cap).is_ok() {
            return None; // offers never nest in providers (acyclicity = deadlock-freedom)
        }
        let mut st = state.lock().unwrap_or_else(|e| e.into_inner());
        let h = self.regrant_into_child(cap, &mut st.host)?;
        st.host.register_cap_name(name, h);
        Some(h)
    }

    /// §5.3 provider-pays: the provider's remaining fuel reserve behind `offer` — the wirer's
    /// meter ("read the meters on what you granted", §15). `None` for a forged handle or a
    /// pure (non-instanced) offer.
    pub fn impl_fuel_remaining(&self, offer: i32) -> Option<u64> {
        let state = self.resolve_guest_impl(offer).ok()?.state.clone()?;
        let st = state.lock().unwrap_or_else(|e| e.into_inner());
        Some(st.fuel)
    }

    /// §5.3 provider-pays: set the provider's fuel reserve behind `offer` (the wirer pricing
    /// its own service — top-up or clamp). `None` (no change) for a forged handle or a pure
    /// offer.
    pub fn set_impl_fuel_reserve(&mut self, offer: i32, fuel: u64) -> Option<()> {
        let state = self.resolve_guest_impl(offer).ok()?.state.clone()?;
        state.lock().unwrap_or_else(|e| e.into_inner()).fuel = fuel;
        Some(())
    }

    /// Adopt a wired offer re-granted from a parent domain (IMPORTS.md §3.3 — the wrap/override
    /// leg of [`Host::regrant_into_child`]): install the entry under **this** host's interned id
    /// for its (unchanged) signature list, one provenance hop deeper, and grant the handle.
    fn adopt_guest_impl(&mut self, entry: GuestImplEntry) -> i32 {
        let type_id = self.intern_interface(&entry.sigs);
        let idx = self.guest_impls.len() as u32;
        self.guest_impls.push(GuestImplEntry {
            type_id,
            depth: entry.depth + 1,
            ..entry
        });
        self.grant(type_id, Binding::GuestImpl(idx))
    }

    /// Resolve `handle` to its wired-offer state (§3c: mask + generation, then the binding must
    /// actually be a [`Binding::GuestImpl`]) — the eval loop's lookup when servicing a dispatch
    /// (slice 3), and the wiring-time lookup for [`Host::bound_import_for_impl`]. A forged /
    /// closed / non-offer handle is an inert `CapFault`.
    pub fn resolve_guest_impl(&self, handle: i32) -> Result<&GuestImplEntry, Trap> {
        // The slot's own type_id feeds the canonical resolve (§3c mask + generation + type run
        // through the one hinge, never re-implemented); the binding-kind match below is the check
        // that the id actually names a wired offer.
        let expect = self.table[(handle as u32 as usize) & (CAP - 1)].type_id;
        match self.resolve(handle, expect)? {
            Binding::GuestImpl(idx) => self.guest_impls.get(idx as usize).ok_or(Trap::CapFault),
            _ => Err(Trap::CapFault),
        }
    }

    /// Build the [`BoundImport`] that binds an import slot to op `op` of the wired offer behind
    /// `handle` — **the §3.2 wiring-time signature check**, structural and fail-closed: `None`
    /// unless `handle` resolves to a wired offer, `op` is within its op list, and the slot's
    /// `declared` signature equals the op's derived signature exactly. On success the slot binds
    /// `(offer's type_id, op, handle)` like any other capability binding
    /// ([`Host::set_import_bindings`]).
    pub fn bound_import_for_impl(
        &self,
        handle: i32,
        op: u32,
        declared: &FuncType,
        rebindable: bool,
    ) -> Option<BoundImport> {
        let entry = self.resolve_guest_impl(handle).ok()?;
        let sig = entry.sigs.get(op as usize)?;
        if sig != declared {
            return None;
        }
        Some(if rebindable {
            BoundImport::rebindable(entry.type_id, op, Some(handle))
        } else {
            BoundImport::required(entry.type_id, op, handle)
        })
    }

    /// Grant a §14 `AddressSpace` capability over the window sub-range `[base, base+size)` (§14). The
    /// root grant is normally the whole window (`base = 0`, `size` the window size); the guest then
    /// `sub`-attenuates it to carve children. `size` must be a power of two and `base` a multiple of
    /// it (so the range and every sub-range are power-of-two aligned, §4/D19) — the caller's
    /// contract, mirroring how the host lays out windows.
    pub fn grant_address_space(&mut self, base: u64, size: u64) -> i32 {
        self.grant(cap_id::ADDRESS_SPACE, Binding::AddressSpace { base, size })
    }

    /// Grant a §14 `Instantiator` capability over the window sub-range `[base, base+size)` — the
    /// authority to spawn children (`instantiate`/`join`) confined to power-of-two sub-windows of it
    /// (§14). Like `grant_address_space`, `size` must be a power of two and `base` a multiple of it.
    pub fn grant_instantiator(&mut self, base: u64, size: u64) -> i32 {
        self.grant(cap_id::INSTANTIATOR, Binding::Instantiator { base, size })
    }

    /// Resolve a handle as an `Instantiator` (§14) and return its `(base, size)` sub-range, or a
    /// `CapFault` for a forged / closed / wrong-type handle. Used by the eval loop, which services
    /// `instantiate`/`join` itself (the generic dispatch can't reach the executor).
    fn resolve_instantiator(&self, handle: i32) -> Result<(u64, u64), Trap> {
        match self.resolve(handle, cap_id::INSTANTIATOR)? {
            Binding::Instantiator { base, size } => Ok((base, size)),
            _ => Err(Trap::CapFault),
        }
    }

    /// Grant a §14 `Yielder` capability (the co-fiber child's handle back to its parent). Used by the
    /// eval loop when standing up a coroutine child; not a powerbox-level grant.
    fn grant_yielder(&mut self) -> i32 {
        self.grant(cap_id::YIELDER, Binding::Yielder)
    }

    /// Confirm a handle resolves to *this* domain's `Yielder` (§14 co-fiber); a forged/wrong handle is
    /// a `CapFault`. The eval loop calls this before yielding the running coroutine's continuation.
    fn resolve_yielder(&self, handle: i32) -> Result<(), Trap> {
        match self.resolve(handle, cap_id::YIELDER)? {
            Binding::Yielder => Ok(()),
            _ => Err(Trap::CapFault),
        }
    }

    /// Grant a §14 **`Module` capability** over `m` — the authority to instantiate it as a child
    /// domain via the `Instantiator`'s module ops (the "plugin" grant). **`m` must already be
    /// verified** (`svm_verify::verify_module`): like every run entry, the host is trusted to grant
    /// only verifier-passing modules — a guest can never inject code, only spawn what it was given.
    pub fn grant_module(&mut self, m: &Module) -> i32 {
        self.grant_module_inner(m, false)
    }

    /// [`Host::grant_module`], additionally attesting the module is **freezable** (DURABILITY.md
    /// §4): the host ran `svm_durable::transform_module` on `m` before granting, so a *durable*
    /// domain may instantiate it as a child (an unmarked grant is refused there — the child could
    /// never drain-then-unwind, making the subtree non-snapshottable). Like verification, the
    /// attestation is the trusted host's: instrumentation is a compile-mode fact the runtime
    /// cannot re-derive from the IR.
    pub fn grant_durable_module(&mut self, m: &Module) -> i32 {
        self.grant_module_inner(m, true)
    }

    /// Grant a PROCESS.md §5 **window-minter** capability with a byte `quota`: the authority to
    /// spawn **detached** children (`Instantiator.instantiate_detached`, op 15) whose windows no
    /// ancestor below the platform can read. Embedder-granted (like `exec`/`fs` — nothing
    /// ambient); each mint deducts the child's window size from the remaining quota.
    pub fn grant_window_minter(&mut self, quota: u64) -> i32 {
        let idx = self.window_minters.len() as u32;
        self.window_minters.push(quota);
        self.grant(cap_id::WINDOW_MINTER, Binding::WindowMinter(idx))
    }

    /// Deduct `bytes` from the minter behind `handle` — the detached-spawn admission check.
    /// `false` (nothing deducted) for a forged/wrong-type handle or an exhausted quota: the
    /// spawn refuses probeably, never a trap.
    fn window_minter_take(&mut self, handle: i32, bytes: u64) -> bool {
        let idx = match self.resolve(handle, cap_id::WINDOW_MINTER) {
            Ok(Binding::WindowMinter(i)) => i as usize,
            _ => return false,
        };
        match self.window_minters.get_mut(idx) {
            Some(rem) if *rem >= bytes => {
                *rem -= bytes;
                true
            }
            _ => false,
        }
    }

    fn grant_module_inner(&mut self, m: &Module, durable: bool) -> i32 {
        let id = self.modules.len() as u32;
        self.modules.push(ModuleGrant {
            funcs: m.funcs.clone().into(),
            memory_log2: m.memory.map(|mc| mc.size_log2),
            data: m.data.clone().into(),
            exports: m.exports.clone().into(),
            imports: m.imports.clone().into(),
            types: m.types.clone().into(),
            durable,
            digest: module_digest(m),
            module: Arc::new(m.clone()),
        });
        self.grant(cap_id::MODULE, Binding::Module(id))
    }

    /// Find a granted **durable** module by its content digest (§4 separate-module thaw): the restore
    /// host re-grants the child's module, and its re-attach residue names it by [`module_digest`].
    /// Returns the grant's function table, or `None` (a missing / mismatched re-grant ⇒
    /// the thaw fails closed, the per-child R5 identity gate).
    fn module_by_digest(&self, digest: &[u8; 32]) -> Option<Arc<[Func]>> {
        self.modules
            .iter()
            .find(|g| g.durable && &g.digest == digest)
            .map(|g| Arc::clone(&g.funcs))
    }

    /// Resolve a handle as a §14 `Module` grant — the eval loop's lookup for the Instantiator's
    /// module ops. A forged / closed / wrong-type handle is a `CapFault`.
    fn resolve_module(&self, handle: i32) -> Result<&ModuleGrant, Trap> {
        match self.resolve(handle, cap_id::MODULE)? {
            Binding::Module(id) => self.modules.get(id as usize).ok_or(Trap::CapFault),
            _ => Err(Trap::CapFault),
        }
    }

    /// Resolve a §14 `Module` handle to **raw views** of its grant — the bridge the JIT's nesting
    /// runtime uses (via `svm-run`'s `module_resolver` callback; `svm-jit` cannot name `Host`).
    /// `None` for a forged/closed/wrong-type handle. The returned pointers borrow [`Host::modules`]
    /// (append-only), so they stay valid for as long as this `Host` lives — which outlives the run,
    /// the same lifetime contract as the `cap.call` ctx itself. `memory_log2` is `-1` when the
    /// module declares no memory. Host-side callers only; never reachable from a guest `cap.call`
    /// (the generic dispatch on a `Module` handle is an inert `CapFault`), so no host address ever
    /// leaks into a guest-readable value.
    #[allow(clippy::type_complexity)]
    pub fn resolve_module_parts(
        &self,
        handle: i32,
    ) -> Option<(*const Func, usize, i32, *const Data, usize)> {
        let g = self.resolve_module(handle).ok()?;
        Some((
            g.funcs.as_ptr(),
            g.funcs.len(),
            g.memory_log2.map_or(-1, |l| l as i32),
            g.data.as_ptr(),
            g.data.len(),
        ))
    }

    /// Grant a §13 `SharedRegion` capability backed by a fresh `len`-byte zero-filled host buffer,
    /// returning its handle. The guest `map`s it into its window (op 0) — at one or more offsets — to
    /// access the shared bytes as ordinary masked loads/stores. (Guest-minted regions and
    /// cross-domain `grant` are a §14 follow-up; this models the host↔guest data plane.)
    pub fn grant_shared_region(&mut self, len: usize) -> i32 {
        self.grant_shared_region_backed(Arc::new(VecBacking(Mutex::new(vec![0u8; len]))))
    }

    /// Grant a §13 `SharedRegion` over a caller-supplied [`SharedBacking`] — how a flat-window
    /// backend installs a region whose `os_fd` it can `mmap` for real hardware aliasing (the JIT
    /// side of the §13 differential). The pure-Rust [`grant_shared_region`] is the common case.
    pub fn grant_shared_region_backed(&mut self, backing: RegionBacking) -> i32 {
        let id = self.regions.len() as u32;
        self.regions.push(backing);
        self.grant(cap_id::SHARED_REGION, Binding::SharedRegion(id))
    }

    /// PROCESS.md S1b/S1c — install (or clear, with `None`) the **canonical-key futex** region hook.
    /// `svm-run` installs it for a JIT run so a §13 `map` records the aliased pages into the JIT's futex
    /// registry (and `unmap` forgets them); the interpreter needs none (its `PageProt::Backed` already
    /// canonicalizes). Called with `(win_off, len, Some((region_off, backing)))` on `map`,
    /// `(win_off, len, None)` on `unmap`.
    #[allow(clippy::type_complexity)]
    pub fn set_region_hook(
        &mut self,
        hook: Option<Arc<dyn Fn(u64, u64, Option<(u64, u64)>) + Send + Sync>>,
    ) {
        self.region_hook = hook;
    }

    /// Whether a canonical-key region hook is installed — so `svm-run`'s `cap.call` trampoline installs
    /// it lazily (once, over the run's `mem_base`) only when absent.
    pub fn has_region_hook(&self) -> bool {
        self.region_hook.is_some()
    }

    /// Grant a §15 / PROCESS.md §5 `Budget` — a splittable resource-quota vector `(fuel, mem, spawn)` —
    /// returning its handle. The embedder mints the **root** budget with the total resources it lends a
    /// domain; the guest `split`s sub-budgets out of it (attenuation) and `read`s remaining. A field of
    /// `-1` means "unbounded" (the anti-bomb ceilings still cap actual consumption; `read` reports it as
    /// `-1`). Charging live consumption against a budget is the follow-up — this is the passable object.
    pub fn grant_budget(&mut self, fuel: i64, mem: i64, spawn: i64) -> i32 {
        let id = self.budgets.len() as u32;
        self.budgets.push(BudgetState { fuel, mem, spawn });
        self.grant(cap_id::BUDGET, Binding::Budget(id))
    }

    /// Set this domain's §6 [`Attestation`] — what `cap.self.attest` reports. The embedder calls it on
    /// the **top-level** `Host` to declare the platform-vouched provenance (isolation tier, whether an
    /// ancestor can read/snapshot it); the §14 spawn path calls it internally to stamp a nested child's
    /// (exposed) report. The default (a fresh `Host`) is a root domain (tier 1, unexposed).
    pub fn set_attestation(&mut self, attestation: Attestation) {
        self.attestation = attestation;
    }

    /// This domain's current §6 [`Attestation`] (what `cap.self.attest` reports) — the read half of
    /// [`Self::set_attestation`], for an embedder that inspects a child host it built.
    pub fn attestation(&self) -> Attestation {
        self.attestation
    }

    /// The §6 [`Attestation`] to stamp on a §14 **nested child** this host spawns: the child inherits
    /// the parent's isolation tier, is always `window_exposed` (the parent sees its carve — the §14
    /// superset, so an ancestor holds read authority), and is `freeze_exposed` iff spawned into a
    /// **durable** subtree (an ancestor may snapshot it — a snapshot is a read).
    /// PROCESS.md §5 — a **detached** child's attestation: same in-process tier (never a
    /// Spectre boundary — real distrust is a separate process), but `window_exposed = false`
    /// (no ancestor below the platform holds read authority over its platform-minted window)
    /// and never ancestor-freezable. The jacl distrust-spawner report.
    fn detached_child_attestation(&self) -> Attestation {
        Attestation {
            tier: self.attestation.tier,
            window_exposed: false,
            freeze_exposed: false,
        }
    }

    fn child_attestation(&self, durable: bool) -> Attestation {
        Attestation {
            tier: self.attestation.tier,
            window_exposed: true,
            freeze_exposed: durable,
        }
    }

    /// Install the [`JitValidator`] — the decode+verify gate every `Jit.compile` runs. The
    /// embedder must install the **same** function for the interpreter and JIT runs of a
    /// differential pair (see [`JitValidator`]); without one, every `compile` is `-EINVAL`.
    pub fn set_jit_validator(&mut self, v: JitValidator) {
        self.jit_validator = Some(v);
    }

    /// Grant a guest-driven `Jit` capability (iface 11, opt-in like `Memory`). `mem_log2` is the
    /// parent module's declared memory — the memory-match precondition submitted blobs are
    /// checked against (DESIGN.md §22 "Security argument").
    pub fn grant_jit(&mut self, mem_log2: Option<u8>) -> i32 {
        self.grant_jit_with_table(mem_log2, 0)
    }

    /// Like [`Host::grant_jit`], but also reserve a `call_indirect` table of `2^table_log2`
    /// slots for B2 `install` (the run's root vCPU honours it; pass the **same** value as the
    /// JIT's `table_reserve_log2`). `0` ⇒ natural size (no install room).
    pub fn grant_jit_with_table(&mut self, mem_log2: Option<u8>, table_log2: u8) -> i32 {
        let id = self.jit_domains.len() as u32;
        self.jit_domains.push(JitDomainState {
            mem_log2,
            units: Vec::new(),
            native_ctx: 0,
            units_left: JIT_DEFAULT_MAX_UNITS,
            bytes_left: JIT_DEFAULT_MAX_BLOB_BYTES,
        });
        self.jit_table_log2 = self.jit_table_log2.max(table_log2);
        self.grant(cap_id::JIT, Binding::JitDomain(id))
    }

    /// The `call_indirect` table reservation (`log2`) the run's root vCPU should build for B2
    /// `install`; `0` ⇒ natural size.
    pub fn jit_table_log2(&self) -> u8 {
        self.jit_table_log2
    }

    /// Tighten (or widen) every granted `Jit` domain's compile quota — the §15-style resource
    /// bound on guest-driven compilation (units and cumulative submitted-blob bytes; enforced
    /// in the shared [`Host::jit_compile`] gate, so a quota'd `compile` fails `-ENOMEM`
    /// identically on both backends). Set before the run, like [`Host::set_quota`].
    pub fn set_jit_quota(&mut self, max_units: u32, max_blob_bytes: u64) {
        for d in &mut self.jit_domains {
            d.units_left = max_units;
            d.bytes_left = max_blob_bytes;
        }
    }

    /// Register the JIT embedder's native context (its `*mut CompiledModule` as a `usize`) on
    /// every granted `Jit` domain — called after the parent module is compiled, before the run.
    /// Stored opaquely; only the embedder's cap thunk dereferences it. A reference
    /// (interpreter) run never calls this, leaving `0`.
    pub fn set_jit_native_ctx(&mut self, ctx: usize) {
        for d in &mut self.jit_domains {
            d.native_ctx = ctx;
        }
    }

    /// The native context registered for `domain` (`0` ⇒ reference run / none registered).
    pub fn jit_native_ctx(&self, domain: u32) -> usize {
        self.jit_domains
            .get(domain as usize)
            .map_or(0, |d| d.native_ctx)
    }

    /// `Jit.compile` minus the backend-specific half (shared by the reference dispatch arm and
    /// the JIT embedder's thunk): resolve `handle` as a `Jit` domain, run the injected
    /// [`JitValidator`] (fail-closed `-EINVAL` when none is installed), store the validated unit,
    /// and mint its `CompiledCode` handle. `Ok(Err(errno))` is a guest-visible failure (nothing
    /// installed); `Err(Trap)` is a forged/wrong-type domain handle. The JIT embedder then
    /// compiles the unit natively and registers the trampoline via [`Host::set_jit_unit_native`].
    pub fn jit_compile(
        &mut self,
        handle: i32,
        bytes: &[u8],
    ) -> Result<Result<JitCompiled, i64>, Trap> {
        // The closed-blob path: no symbol table, so a unit with §7 imports fails closed.
        self.jit_compile_linked(handle, bytes, &[])
    }

    /// Like [`Self::jit_compile`], but the unit's §7 imports are resolved against the
    /// guest-provided **symbol-table bytes** before verify — host-assisted dynamic linking
    /// (DESIGN.md §22). The `compile_linked` op routes here; `compile` routes here with an empty
    /// table. The validator does the resolve-then-verify, so the symbol table stays
    /// guest-controlled yet a mis-link can never escape (it fails re-verification, `-EINVAL`).
    pub fn jit_compile_linked(
        &mut self,
        handle: i32,
        bytes: &[u8],
        symtab: &[u8],
    ) -> Result<Result<JitCompiled, i64>, Trap> {
        let domain = self.resolve_jit_domain(handle)?;
        // §4 (DURABILITY.md): *a durable domain admits only freezable modules* — and a §22
        // guest-submitted unit is a module installation. The durable transform is a host-side
        // compile pass this crate cannot run (no `svm-durable` dependency), so until an embedder
        // instrumentation hook exists, a durable domain's `compile` fails closed (`-EINVAL`,
        // guest-reachable errno like the other refusals here): an un-instrumented unit could
        // never drain-then-unwind, silently making the domain non-snapshottable.
        if self.durable {
            return Ok(Err(EINVAL));
        }
        let Some(validate) = self.jit_validator else {
            return Ok(Err(EINVAL));
        };
        let d = &mut self.jit_domains[domain as usize];
        // Compile quota first: charge the *attempt's* bytes (validation is the cost a looping
        // guest imposes), the unit slot only on success; out of either budget is `-ENOMEM`.
        if d.units_left == 0 || (bytes.len() as u64) > d.bytes_left {
            return Ok(Err(ENOMEM));
        }
        d.bytes_left -= bytes.len() as u64;
        let funcs = match validate(bytes, d.mem_log2, symtab) {
            Ok(f) if !f.is_empty() => f,
            Ok(_) => return Ok(Err(EINVAL)), // an empty unit has no entry to invoke
            Err(e) => return Ok(Err(e)),
        };
        d.units_left -= 1;
        let unit = d.units.len() as u32;
        d.units.push(JitUnit {
            funcs,
            native_code: 0,
            install_code: 0,
            install_type_id: 0,
        });
        // Guest-minting: a full handle table is -EMFILE, never a panic (§3c / audit #1). The
        // stored unit stays (append-only storage; harmless without a handle).
        match self.try_grant(cap_id::JIT_CODE, Binding::JitCode { domain, unit }) {
            Some(h) => Ok(Ok(JitCompiled {
                handle: h,
                domain,
                unit,
            })),
            None => Ok(Err(EMFILE)),
        }
    }

    /// Register the JIT embedder's code for a unit it compiled via `define_extra`: the
    /// buffer-ABI trampoline (`tramp`, for `invoke`) and the natural-ABI entry + interned
    /// `type_id` (`install_code`/`install_type_id`, for B2 table `install`). `0` (the default)
    /// means "no native code" — a native `invoke`/`install` of such a unit is rejected
    /// fail-closed.
    pub fn set_jit_unit_native(
        &mut self,
        domain: u32,
        unit: u32,
        tramp: usize,
        install_code: usize,
        install_type_id: u32,
    ) {
        if let Some(u) = self
            .jit_domains
            .get_mut(domain as usize)
            .and_then(|d| d.units.get_mut(unit as usize))
        {
            u.native_code = tramp;
            u.install_code = install_code;
            u.install_type_id = install_type_id;
        }
    }

    /// The natural-ABI entry pointer + interned `type_id` the JIT embedder registered for a
    /// unit (for B2 `install`); `(0, 0)` if none / a reference run.
    pub fn jit_unit_install(&self, domain: u32, unit: u32) -> (usize, u32) {
        self.jit_domains
            .get(domain as usize)
            .and_then(|d| d.units.get(unit as usize))
            .map_or((0, 0), |u| (u.install_code, u.install_type_id))
    }

    /// Resolve a handle as a `Jit` domain (a forged/closed/wrong-type handle is a `CapFault`).
    pub fn resolve_jit_domain(&self, handle: i32) -> Result<u32, Trap> {
        match self.resolve(handle, cap_id::JIT)? {
            Binding::JitDomain(d) => Ok(d),
            _ => Err(Trap::CapFault),
        }
    }

    /// Resolve a handle as a `CompiledCode` unit → `(domain, unit)`.
    pub fn resolve_jit_code(&self, handle: i32) -> Result<(u32, u32), Trap> {
        match self.resolve(handle, cap_id::JIT_CODE)? {
            Binding::JitCode { domain, unit } => Ok((domain, unit)),
            _ => Err(Trap::CapFault),
        }
    }

    /// The validated functions of a compiled unit (its entry is `funcs[0]`).
    pub fn jit_unit_funcs(&self, domain: u32, unit: u32) -> Option<Arc<[Func]>> {
        self.jit_domains
            .get(domain as usize)
            .and_then(|d| d.units.get(unit as usize))
            .map(|u| Arc::clone(&u.funcs))
    }

    /// The number of units a `Jit` domain has compiled (append-only; released units stay, their
    /// handle merely revoked). The code-memory compaction driver (DESIGN.md §22) walks `0..count`
    /// deciding which to carry into the fresh module.
    pub fn jit_unit_count(&self, domain: u32) -> u32 {
        self.jit_domains
            .get(domain as usize)
            .map_or(0, |d| d.units.len() as u32)
    }

    /// The units of `domain` still reachable through a **live `CompiledCode` handle** (a
    /// `Binding::JitCode` entry the guest can still `invoke`/`install`/`release`). Compaction must
    /// carry these (their trampoline pointers move, so the handle would dangle otherwise); a unit
    /// that is neither here nor occupying a table slot is dead and is reclaimed. Scans the
    /// host-owned handle table — small (`CAP` = 256) and done only at a quiescent compaction point.
    pub fn jit_live_units(&self, domain: u32) -> Vec<u32> {
        let mut units: Vec<u32> = self
            .table
            .iter()
            .filter_map(|s| match s.entry {
                Some(Binding::JitCode { domain: d, unit }) if d == domain => Some(unit),
                _ => None,
            })
            .collect();
        units.sort_unstable();
        units.dedup();
        units
    }

    /// The native trampoline registered for a unit (`0` ⇒ none).
    pub fn jit_unit_native(&self, domain: u32, unit: u32) -> usize {
        self.jit_domains
            .get(domain as usize)
            .and_then(|d| d.units.get(unit as usize))
            .map_or(0, |u| u.native_code)
    }

    /// `Jit.release`: revoke a `CompiledCode` handle (the slot is cleared; the per-slot
    /// generation makes the old handle value inert forever, D37). The unit's code/funcs stay —
    /// reclaim is a DESIGN.md §22 follow-up. A forged/closed handle is `Err` (the caller maps it to a
    /// non-fatal `-EINVAL`: release is guest-reachable in a loop, so it must not trap).
    pub fn jit_release(&mut self, code_handle: i32) -> Result<(), Trap> {
        self.resolve_jit_code(code_handle)?;
        let slot = (code_handle as u32 as usize) & (CAP - 1);
        self.table[slot].entry = None;
        Ok(())
    }

    /// Fallible [`grant_shared_region_backed`] for **guest-minting** paths (`create_region`, the
    /// cross-domain `grant`): `None` when the handle table is full (so the caller can return
    /// `-EMFILE` instead of panicking). Checks for a free slot **before** registering the backing, so
    /// a full table leaves `regions` untouched (no leaked backing).
    pub fn try_grant_shared_region_backed(&mut self, backing: RegionBacking) -> Option<i32> {
        if self.table.iter().all(|s| s.entry.is_some()) {
            return None; // table full — don't register a backing we can't hand out
        }
        let id = self.regions.len() as u32;
        self.regions.push(backing);
        self.try_grant(cap_id::SHARED_REGION, Binding::SharedRegion(id))
    }

    /// Install the backing factory for **guest-minted** regions (`AddressSpace.create_region`,
    /// §13/§14). A flat-window embedder passes an OS-shared-memory factory (e.g.
    /// `svm_run::new_shared_region`) so a JIT guest can `map` what it mints; without one, mints use
    /// the pure-Rust reference [`VecBacking`] (fine for the interpreter, unmappable by the JIT).
    pub fn set_region_factory(&mut self, f: fn(usize) -> RegionBacking) {
        self.region_factory = Some(f);
    }

    /// Resolve a handle as a §13 `SharedRegion` and return its backing (an `Arc` clone — the same
    /// shared object). Used by the eval loop's cross-domain `grant` (SharedRegion op 4); a forged /
    /// closed / wrong-type handle is a `CapFault`.
    fn resolve_region(&self, handle: i32) -> Result<RegionBacking, Trap> {
        match self.resolve(handle, cap_id::SHARED_REGION)? {
            Binding::SharedRegion(id) => {
                self.regions.get(id as usize).cloned().ok_or(Trap::CapFault)
            }
            _ => Err(Trap::CapFault),
        }
    }

    /// Close a handle (§3c): free the slot but keep its generation, so the old handle value is
    /// now a dead generation. A later `cap.call` on it completes with the probeable `-EBADF`
    /// errno (I41 graceful revocation — the once-issued generation is its own tombstone,
    /// [`Host::handle_revoked`]); only a **forged** generation still traps (D37).
    pub fn close(&mut self, handle: i32) {
        let slot = (handle as u32 as usize) & (CAP - 1);
        self.table[slot].entry = None;
    }

    /// Whether `handle` still names a live table entry (its slot has an entry and the packed
    /// generation matches). The §3.6 revocation-unparks park-vs-revoke race check: a fiber about
    /// to park through a handle confirms it wasn't revoked in the window since its empty read —
    /// authority-neutral (a `bool`, no binding escapes), and deliberately type-blind (liveness,
    /// not typing, is the question).
    pub fn handle_live(&self, handle: i32) -> bool {
        let slot = (handle as u32 as usize) & (CAP - 1);
        let gen = (handle as u32) >> CAP_LOG2;
        let s = &self.table[slot];
        s.entry.is_some() && (s.generation & GEN_MASK) == gen
    }

    /// ISSUES.md I41 (graceful revocation) — whether `handle` names a **revoked-once-valid**
    /// capability: its slot no longer resolves it (entry gone, or re-granted at a newer
    /// generation) but its generation is one the slot has actually issued. No tombstone storage
    /// is needed: a slot's counter advances only at (re)grant ([`Host::try_grant`]), so every
    /// generation `1..=generation` was once a live handle — a dead generation at or below the
    /// counter IS the tombstone. A **live** handle is not revoked (a wrong-type use of it stays
    /// the D37 trap), and a generation the slot never issued (0, or above the counter) is a
    /// forgery (trap). Once the full-width counter wraps past `GEN_MASK`, every masked
    /// generation has genuinely been issued, so this degrades exactly as [`Host::resolve`]'s
    /// own masked ABA acceptance does.
    fn handle_revoked(&self, handle: i32) -> bool {
        let h = handle as u32;
        let slot = (h as usize) & (CAP - 1);
        let gen = h >> CAP_LOG2;
        let s = &self.table[slot];
        let live = s.entry.is_some() && (s.generation & GEN_MASK) == gen;
        !live && gen >= 1 && (s.generation > GEN_MASK || gen <= s.generation)
    }

    /// Resolve a handle at a `cap.call` use site (§3c) — **the security hinge**: mask
    /// the index into the host-owned table (never branch), then re-check the entry's
    /// interface `type_id` and `generation`. A forged / closed / wrong-type index is
    /// inert: it faults, or at worst selects one of *this domain's own* granted
    /// `type_id` capabilities. The guest never supplies the binding.
    fn resolve(&self, handle: i32, type_id: u32) -> Result<Binding, Trap> {
        let h = handle as u32;
        let slot = (h as usize) & (CAP - 1); // mask, not branch (Spectre-v1 safe)
        let gen = h >> CAP_LOG2;
        let s = &self.table[slot];
        match s.entry {
            // Compare the generation masked to the bits a handle actually carries (`GEN_BITS`), so a
            // slot recycles cleanly when the full-width counter wraps instead of dying permanently.
            Some(b) if s.type_id == type_id && (s.generation & GEN_MASK) == gen => Ok(b),
            _ => Err(Trap::CapFault),
        }
    }

    /// PROCESS.md S2 — resolve `handle` to its `(type_id, binding)` for **re-granting into a child**
    /// (`Instantiator.instantiate_granted`), so a §14 child is not born destitute. Only a
    /// **coordinate-free, self-contained** capability qualifies — one a fresh child `Host` can hold
    /// as-is: `Stream` (stdio), `Exit`, `Clock`. Refused (`CapFault`), deliberately:
    /// - **index-carrying** caps (`SharedRegion`/`Module`/`IoRing`/`Blocking`/`HostFn*`) whose index
    ///   names a slot in *this* Host's side tables — a child needs those installed by their own
    ///   deep-copy path (e.g. the SharedRegion grant), not a raw binding copy; and
    /// - **window-coordinate** caps (`AddressSpace`/`Instantiator`, `Memory`) whose `{base,size}` are
    ///   in the holder's coordinates — the child already gets fresh ones over *its own* window, and
    ///   copying the parent's would confer authority in the wrong coordinate space.
    fn resolve_copyable(&self, handle: i32) -> Result<(u32, Binding), Trap> {
        let h = handle as u32;
        let slot = (h as usize) & (CAP - 1);
        let gen = h >> CAP_LOG2;
        let s = &self.table[slot];
        match s.entry {
            Some(b) if (s.generation & GEN_MASK) == gen => match b {
                Binding::Stream(_) | Binding::Exit | Binding::Clock => Ok((s.type_id, b)),
                _ => Err(Trap::CapFault),
            },
            _ => Err(Trap::CapFault),
        }
    }

    /// PROCESS.md S2 (JIT parity) — build a §14 **granted child** powerbox: a fresh `Host` holding an
    /// `Instantiator` + `AddressSpace` over its own window `[0, child_size)` and the parent's
    /// re-granted coordinate-free capability `grant_handle` (`Stream`/`Exit`/`Clock`). Returns the
    /// child `Host` and its three entry-arg handles `(instantiator, address_space, grant)`, or `None`
    /// for a forged / non-copyable handle ([`Self::resolve_copyable`]). A stdout/stderr `Stream` grant
    /// points the child's sink at the parent's shared buffer (stdio inheritance), so the child's output
    /// reaches the granting embedder rather than the child's discarded host buffer.
    ///
    /// This is the child-host construction the interpreter's own `instantiate_granted` (op 8) inlines,
    /// factored out so the **JIT** backend can build the *same* child powerbox host-side (via
    /// `svm_run::grant_child_build`) and keep both backends in differential lockstep — the child sees an
    /// identical set of handles and the same shared sink.
    ///
    /// The grant may be a coordinate-free cap (`Stream`/`Exit`/`Clock`) **or a pipe end** — the latter
    /// aliases its shared FIFO into the child (the cross-domain `cmd1 | cmd2` grant), see
    /// [`Self::regrant_into_child`].
    pub fn spawn_granted_child(
        &mut self,
        grant_handle: i32,
        child_size: u64,
    ) -> Option<(Host, i32, i32, i32)> {
        // Reject a non-grantable handle before building anything (fail closed, no state mutated).
        if !self.can_regrant(grant_handle) {
            return None;
        }
        let mut ch = Host::new();
        // §6: a granted child is nested (window-exposed) and non-durable (not ancestor-freezable).
        ch.set_attestation(self.child_attestation(false));
        let cinst = ch.grant_instantiator(0, child_size);
        let cas = ch.grant_address_space(0, child_size);
        let cg = self.regrant_into_child(grant_handle, &mut ch)?;
        Some((ch, cinst, cas, cg))
    }

    /// Whether `handle` names a capability this host may **re-grant into a §14 child** — a coordinate-free
    /// cap ([`Self::resolve_copyable`]) or a pipe end ([`Self::resolve_pipe_end`]). Used to fail a grant
    /// closed *before* any child state is built.
    fn can_regrant(&self, handle: i32) -> bool {
        self.resolve_pipe_end(handle).is_some()
            || self.resolve_live_impl(handle).is_some()
            || self.resolve_region(handle).is_ok()
            || self.resolve_guest_impl(handle).is_ok()
            || self.resolve_copyable(handle).is_ok()
    }

    /// Re-grant `handle` from this (parent) host into `child` — the §14 child-powerbox re-grant policy:
    /// a **pipe end** aliases its shared FIFO backing into the child (so parent and child share the same
    /// queue — the cross-domain pipe); a stdout/stderr `Stream` shares the parent's sink (stdio
    /// inheritance); every other coordinate-free cap copies its binding as-is. Returns the child handle,
    /// or `None` for a forged / non-grantable cap. (A pipe end is checked first: it is index-carrying,
    /// so `resolve_copyable` would refuse it.)
    fn regrant_into_child(&mut self, handle: i32, child: &mut Host) -> Option<i32> {
        if let Some((write, backing)) = self.resolve_pipe_end(handle) {
            return Some(child.install_pipe_end(write, backing));
        }
        // §3.6 sibling-as-service: re-granting a **live-callee offer** wires the SAME running
        // domain into the child — the child's calls enqueue on the original callee (park,
        // serve, reply, all as-built), so two siblings coordinate through a live peer their
        // parent introduced. The shape rides the entry (captured at wire), so the child-side
        // intern never touches the callee's lock.
        if let Some(e) = self.resolve_live_impl(handle) {
            let (callee, export, sigs) = (Arc::clone(&e.callee), e.export, Arc::clone(&e.sigs));
            return Some(child.install_live_impl(callee, export, sigs));
        }
        // Concurrent stages (STAGE1.md item 6): re-granting a §13 `SharedRegion` aliases the
        // SAME backing into the child's powerbox — the explicit parent↔child / sibling↔sibling
        // data plane (PROCESS.md §4: never implicit carve addresses). Each grantee maps it into
        // its own window; the canonical futex key (backing identity) makes wait/notify
        // rendezvous across the domains — the concurrent-pipeline substrate.
        if let Ok(backing) = self.resolve_region(handle) {
            return Some(child.grant_shared_region_backed(backing));
        }
        // A **wired interface offer** (IMPORTS.md §3.2/§3.3): re-granting it is how a parent
        // forwards, wraps, or overrides a capability for a child — the entry (shared function
        // table + op list) is adopted into the child's own table under the child's interned id,
        // one domain boundary deeper (§3.1 provenance: the impl terminates in an ancestor, and
        // the depth records how far up).
        if let Ok(entry) = self.resolve_guest_impl(handle) {
            let entry = entry.clone();
            return Some(child.adopt_guest_impl(entry));
        }
        let (tid, binding) = self.resolve_copyable(handle).ok()?;
        if let Binding::Stream(r @ (StreamRole::Out | StreamRole::Err)) = binding {
            let sink = if r == StreamRole::Out {
                self.shared_stdout()
            } else {
                self.shared_stderr()
            };
            if r == StreamRole::Out {
                child.out_sink = Some(sink);
            } else {
                child.err_sink = Some(sink);
            }
        }
        Some(child.grant(tid, binding))
    }

    /// PROCESS.md S2 (JIT parity) — build a §14 **named-grant child** powerbox: a fresh `Host` holding
    /// an `Instantiator` + `AddressSpace` over `[0, child_size)` and each of `grants` (a list of
    /// `(name, handle)`) re-granted under its name, so the child discovers them by `cap.self.resolve`.
    /// Returns the child `Host` and its `(instantiator, address_space)` handles, or `None` if **any**
    /// grant's handle is forged / non-copyable (the whole spawn fails closed, `CapFault`, matching the
    /// interpreter's op-11 path). stdout/stderr grants share the parent's sinks (stdio inheritance).
    ///
    /// The multi-cap, by-name analog of [`Self::spawn_granted_child`]; the JIT reads the guest's grant
    /// records from the window (host-side, `svm_run::grant_named_child_build`) and calls this, so both
    /// backends register the same names against the same shared sinks.
    pub fn spawn_named_child(
        &mut self,
        grants: &[(String, i32)],
        child_size: u64,
    ) -> Option<(Host, i32, i32)> {
        // Check every handle first — if any is non-grantable the spawn fails closed, before we mutate
        // anything (a partially-built child would leak a promoted sink / installed pipe).
        if !grants.iter().all(|(_, h)| self.can_regrant(*h)) {
            return None;
        }
        let mut ch = Host::new();
        // §6: a named-grant child is nested (window-exposed) and non-durable (not ancestor-freezable).
        ch.set_attestation(self.child_attestation(false));
        let cinst = ch.grant_instantiator(0, child_size);
        let cas = ch.grant_address_space(0, child_size);
        for (name, handle) in grants {
            // Pre-checked above, so this cannot fail; each cap (coordinate-free or pipe end) is
            // re-granted into the child under its name.
            let cg = self.regrant_into_child(*handle, &mut ch)?;
            ch.register_cap_name(name, cg);
        }
        Some((ch, cinst, cas))
    }

    /// **D45 allocation-free fast path for `Clock.now()`** (ISSUES.md I12). The generic
    /// [`Self::cap_dispatch_slots`] path is dominated, for a cheap cap, by the per-call `Vec` result
    /// allocation and the W1 record/replay gate — not by the work (a field read). This does the
    /// authority check ([`Self::resolve`], identical to the generic path: a forged/closed/wrong-type
    /// handle is an inert `CapFault`) and the read+advance **inline**, returning the `i64` directly.
    ///
    /// Returns `None` when a W1 record or replay tape is active — the caller must then use the full
    /// [`Self::cap_dispatch_slots`] so the crossing is taped/served faithfully (the clock is a recorded
    /// nondeterministic input, [`is_recorded_input`]). Semantics are otherwise byte-identical to the
    /// `Binding::Clock` arm, so interp == JIT still holds.
    #[inline]
    pub fn fast_clock_now(&mut self, handle: i32) -> Option<Result<i64, Trap>> {
        if self.cap_record.is_some() || self.cap_replay.is_some() {
            return None; // a tape is active — fall back so the input is recorded/replayed
        }
        Some(match self.resolve(handle, cap_id::CLOCK) {
            Ok(Binding::Clock) => {
                let now = self.clock_ns;
                self.clock_ns = self.clock_ns.wrapping_add(1);
                Ok(now)
            }
            // `resolve` already enforced `type_id == CLOCK`, so a success is always `Binding::Clock`;
            // any other binding at a CLOCK-typed slot would be a host bug — fail closed like a fault.
            Ok(_) => Err(Trap::CapFault),
            // I41: a revoked-once-valid handle is the probeable errno on the fast path too —
            // byte-identical to the generic dispatch's `handle_revoked` arm, so interp == JIT
            // holds whichever path a backend takes.
            Err(_) if self.handle_revoked(handle) => Ok(CAP_REVOKED),
            Err(t) => Err(t),
        })
    }

    /// Dispatch a `cap.call` (§3c): resolve the handle, then run the mock operation.
    /// Returns the op's result values (negative-errno encoded in an `i64` for the
    /// fallible ops, §3e D42), or a `Trap` for escape/exit. `mem` backs buffer args.
    /// Dispatch a `cap.call` (§3c): resolve the handle in the host-owned table, then run
    /// the bound capability op. Public and **slot-based** (`i64` per scalar; `i32` in
    /// the low bits) so both backends drive the same handlers without per-arg type tags
    /// — the interpreter converts its `Value`s, a JIT passes its slots directly. `mem`
    /// is `None` when the module declares no memory (buffer ops then return `-EFAULT`).
    pub fn cap_dispatch_slots(
        &mut self,
        type_id: u32,
        op: u32,
        handle: i32,
        args: &[i64],
        mem: Option<&mut dyn GuestMem>,
    ) -> Result<Vec<i64>, Trap> {
        // §7 executable named import (IMPORTS.md phase 1): the reserved pseudo-`type_id` carries the
        // **import index** in `op`; translate it through the instantiation-time binding table to the
        // bound `(type_id, op, granted handle)` and fall through to the ordinary flow. Translating
        // *before* the record/replay gate below means a taped `call.import` records exactly what the
        // equivalent resolved `cap.call` would — replay parity across the two forms. An unbound
        // index (no manifest binding installed) is a `CapFault`, fail-closed. The guest-supplied
        // handle argument is ignored: the binding carries the granted handle (the operand is
        // vestigial in static dispatch — IMPORTS.md §2.5).
        let (type_id, op, handle) = if type_id == svm_ir::CAP_IMPORT_TYPE_ID {
            // §3.5: `op` packs `(slot | consumer_op << 16)`. A grouped binding translates the
            // consumer-local op through its bind-time remap (frozen by the coverage walk); a
            // flat binding requires consumer_op 0 and uses the bound op. Fail-closed on an
            // out-of-range consumer op — probeable, like every capability fault.
            let slot = op & 0xFFFF;
            let cop = op >> 16;
            let b = self
                .import_bindings
                .get(slot as usize)
                .copied()
                .ok_or(Trap::CapFault)?;
            // An unbound rebindable slot (declared, never attached — phase 2) is fail-closed.
            if !b.bound {
                return Err(Trap::CapFault);
            }
            let eff_op = match self
                .import_remaps
                .get(slot as usize)
                .and_then(|r| r.as_ref())
            {
                Some(remap) => *remap.get(cop as usize).ok_or(Trap::CapFault)?,
                None if cop == 0 => b.op,
                None => return Err(Trap::CapFault),
            };
            (b.type_id, eff_op, b.handle)
        } else if type_id == svm_ir::CAP_DYN_TYPE_ID {
            // §3.5 dynamic mode by type-section reference: `op` packs `(type_idx | op << 16)`;
            // intern the registered self-module shape and re-enter with the effective id — the
            // ordinary §3c use-site check below does the rest (exact-id fast path).
            let ty = op & 0xFFFF;
            let dop = op >> 16;
            let id = self.self_type_id(ty)?;
            (id, dop, handle)
        } else {
            (type_id, op, handle)
        };
        // §3.5 self-namespace extensions (dispatch form, exempt from manifest-completeness like
        // the rest of `cap.self.*`): op packs `(selfop | idx << 8)` for selfop ≥ 6 —
        // 6 = `type_id` (intern self interface `idx`), 7 = `covers` (probe the handle argument
        // against self interface `idx`), 8 = `export.handle` (reify own offer `idx`).
        if type_id == svm_ir::CAP_SELF_TYPE_ID && (op & 0xFF) >= 6 {
            let idx = op >> 8;
            return match op & 0xFF {
                6 => Ok(vec![self.self_type_id(idx)? as i32 as i64]),
                7 => {
                    let h = *args.first().ok_or(Trap::CapFault)? as i32;
                    Ok(vec![self.self_covers(h, idx)?])
                }
                8 => Ok(vec![self.reify_export(idx)? as i64]),
                // §3.6 svc.poll/svc.wait reaching host-side dispatch = a backend tier without
                // the eval-loop servicing arm (only the eval loop can run guest handler code):
                // a probeable `-EINVAL`, never a trap — the guest's serve loop can fall back.
                // (The tree-walk eval loop intercepts these before dispatch; the bytecode
                // engine declines them at compile and falls back to the tree-walker.)
                CAP_SELF_SVC_POLL | CAP_SELF_SVC_WAIT if op >> 8 == 0 => Ok(vec![EINVAL]),
                _ => Err(Trap::CapFault),
            };
        }
        // W1 record/replay (DEBUGGING.md): only the nondeterministic *input* caps are taped —
        // deterministic / structural caps re-run faithfully on a fresh powerbox and are left live.
        if is_recorded_input(type_id, op) {
            // Replay: serve the next recorded crossing if it matches this call (a mismatch means the
            // re-execution diverged before this point — fall through to live as a best effort).
            let served = if let Some((tape, pos)) = &mut self.cap_replay {
                match tape.get(*pos) {
                    Some(rec)
                        if rec.type_id == type_id
                            && rec.op == op
                            && rec.handle == handle
                            && rec.args == args =>
                    {
                        *pos += 1;
                        Some(rec.clone())
                    }
                    _ => None,
                }
            } else {
                None
            };
            if let Some(rec) = served {
                // Re-apply any guest-window writes the cap made (a buffer-filling `read`), then
                // return the recorded result slots — no live host needed.
                if let Some(m) = mem {
                    for (ptr, bytes) in &rec.mem_writes {
                        m.write_bytes(*ptr, bytes);
                    }
                }
                return rec.result;
            }
            // Record: run live through a `RecordingMem` so any guest-window writes are captured,
            // then log the crossing for a future replay.
            let (result, mem_writes) = match mem {
                Some(m) => {
                    let mut rec_mem = RecordingMem {
                        inner: m,
                        writes: Vec::new(),
                    };
                    let r = self.cap_dispatch_slots_inner(
                        type_id,
                        op,
                        handle,
                        args,
                        Some(&mut rec_mem),
                    );
                    (r, rec_mem.writes)
                }
                None => (
                    self.cap_dispatch_slots_inner(type_id, op, handle, args, None),
                    Vec::new(),
                ),
            };
            if let Some(rec) = &mut self.cap_record {
                rec.push(CapRecord {
                    type_id,
                    op,
                    handle,
                    args: args.to_vec(),
                    result: result.clone(),
                    mem_writes,
                });
            }
            return result;
        }
        self.cap_dispatch_slots_inner(type_id, op, handle, args, mem)
    }

    /// The live capability dispatch (§3c) — resolve the handle in the host table and run the op. The
    /// public [`cap_dispatch_slots`](Host::cap_dispatch_slots) wraps this with W1 record/replay.
    fn cap_dispatch_slots_inner(
        &mut self,
        type_id: u32,
        op: u32,
        handle: i32,
        args: &[i64],
        mem: Option<&mut dyn GuestMem>,
    ) -> Result<Vec<i64>, Trap> {
        // Phase-2 `import.attach` (IMPORTS.md): (re)bind rebindable import slot `op` to the handle
        // in `args[0]`. The new handle must resolve **live under the slot's declared interface
        // type_id** (the §3c mask + type + generation check) — attach swaps which *object* the slot
        // names, never which *interface*. Authority-neutral (aliases a held capability into a named
        // slot). Structural misuse (unknown slot / non-rebindable) is a `CapFault` — the verifier
        // rules that out for guest code, so reaching it means a hostile/buggy direct caller; a
        // wrong-type or dead handle returns `-EINVAL` so a guest can probe a discovered handle and
        // fall back. Serialized by the Host lock like every dispatch (the §12 ordering guarantee:
        // concurrent `call.import`s on the slot see the old or the new binding atomically).
        if type_id == svm_ir::CAP_IMPORT_ATTACH_TYPE_ID {
            let slot = op as usize;
            let b = self.import_bindings.get(slot).ok_or(Trap::CapFault)?;
            if !b.rebindable {
                return Err(Trap::CapFault);
            }
            let new_handle = *args.first().ok_or(Trap::Malformed)? as i32;
            // §3.5: when the slot's manifest requirement was retained, attach is the coverage
            // walk — the attached capability must cover the declared `(name, sig)` set (exact
            // shape, or a covering wired guest impl), and the slot's remap refreshes with the
            // binding. Without a recorded requirement, the legacy exact-`type_id` check stands.
            if let Some(req) = self.import_reqs.get(slot).cloned().flatten() {
                let (req_names, req_sigs) = &*req;
                let Some(tid) = self.type_id_of(new_handle) else {
                    return Ok(vec![EINVAL]); // dead/forged — probeable, never a trap
                };
                let exact = tid == self.intern_interface(req_sigs);
                let cover = if exact {
                    Some((0..req_sigs.len() as u32).collect::<Arc<[u32]>>())
                } else {
                    self.resolve_guest_impl(new_handle).ok().and_then(|e| {
                        let (en, es) = (Arc::clone(&e.names), Arc::clone(&e.sigs));
                        coverage_remap(req_names, req_sigs, &en, &es)
                    })
                };
                let Some(remap) = cover else {
                    return Ok(vec![EINVAL]); // does not cover — fail closed, probeable
                };
                let b = &mut self.import_bindings[slot];
                b.type_id = tid;
                b.op = remap[0];
                b.handle = new_handle;
                b.bound = true;
                self.import_remaps[slot] = Some(remap);
                return Ok(vec![0]);
            }
            if self.resolve(new_handle, b.type_id).is_err() {
                return Ok(vec![EINVAL]);
            }
            let b = &mut self.import_bindings[slot];
            b.handle = new_handle;
            b.bound = true;
            return Ok(vec![0]);
        }
        // §7 reflection: the reserved pseudo-`type_id` has no handle to resolve — service it directly
        // (read-only over this domain's own powerbox). This is the JIT's entry point for `cap.self.*`.
        if type_id == svm_ir::CAP_SELF_TYPE_ID {
            // op 2 = `resolve(name_ptr, name_len) -> handle | -errno` (F7): look a capability **name**
            // up in this domain's name directory (populated at powerbox grant) and return the handle
            // it's bound to — runtime, in-guest, dlopen-style discovery, the dynamic counterpart to
            // load-time name binding. Confers no authority: it only re-finds a handle the guest was
            // already granted (a name with no grant is `-EINVAL`). Serviced here, not in the mem-less
            // `self_dispatch`, because it must read the name from the window (fail-closed: `-EFAULT`
            // out of bounds, `-EINVAL` on bad UTF-8 / unknown name).
            if op == 2 {
                let Some(mem) = mem else {
                    return Ok(vec![EFAULT]);
                };
                let ptr = *args.first().unwrap_or(&0) as u64;
                let len = *args.get(1).unwrap_or(&0) as u64;
                let Some(bytes) = mem.read_bytes(ptr, len) else {
                    return Ok(vec![EFAULT]);
                };
                let Ok(name) = std::str::from_utf8(&bytes) else {
                    return Ok(vec![EINVAL]);
                };
                return Ok(vec![self
                    .resolve_cap_name(name)
                    .map_or(EINVAL, |h| h as i64)]);
            }
            // op 3 = `label(handle, buf_ptr, buf_cap) -> len | 0 | -EFAULT` (F9): write the handle's
            // human-readable label into the window (the reverse of op 2). Returns the label's full
            // length (`0` if unlabeled); writes nothing if it doesn't fit (`buf_cap < len`) so the
            // guest can retry with a buffer of the returned size. Authority-neutral reflection.
            if op == 3 {
                let h = *args.first().unwrap_or(&0) as i32;
                let ptr = *args.get(1).unwrap_or(&0) as u64;
                let cap = *args.get(2).unwrap_or(&0) as u64;
                let Some(label) = self.cap_label(h).map(|s| s.to_string()) else {
                    return Ok(vec![0]); // no label for this handle
                };
                let len = label.len() as u64;
                if len <= cap {
                    let Some(mem) = mem else {
                        return Ok(vec![EFAULT]);
                    };
                    if mem.write_bytes(ptr, label.as_bytes()).is_none() {
                        return Ok(vec![EFAULT]);
                    }
                }
                return Ok(vec![len as i64]);
            }
            return self.self_dispatch(op, args);
        }
        // ISSUES.md I41 (graceful revocation): a call through a handle that was once granted and
        // has since been revoked completes with the **same probeable errno** the §3.6
        // revocation-unpark delivers ([`CAP_REVOKED`], `-EBADF`) — "cancellation is a value" now
        // holds whether the caller was parked mid-call or calls a moment later, removing the
        // dominant benign trap in a long-running server. The trap stays reserved for what D37
        // always meant it for: a **forgery** (a generation the slot never issued), and for
        // type-confusion on a live handle (`handle_revoked` is false for live handles, so the
        // wrong-type resolve failure below still traps).
        let resolved = match self.resolve(handle, type_id) {
            Ok(b) => b,
            Err(t) => {
                return if self.handle_revoked(handle) {
                    Ok(vec![CAP_REVOKED])
                } else {
                    Err(t)
                }
            }
        };
        match resolved {
            // §3.6 slice 3: a live-callee offer is serviced by the eval loop (enqueue + park —
            // host-side dispatch cannot park). Reaching it here means a backend tier without
            // the servicing arm: answer probeable, never trap.
            Binding::LiveImpl(_) => Ok(vec![EINVAL]),
            // PROCESS.md §5: a window minter is spawn *evidence* (an `instantiate_detached`
            // argument), not a dispatch target — inert probeable refusal.
            Binding::WindowMinter(_) => Ok(vec![EINVAL]),
            // §3.6 slice 1: `Stream.close` is **real** — the guest-side revocation act (D37
            // turned inward: the holder hangs up). Null the slot entry so every later use of the
            // handle is the clean use-after-close answer (I41: the probeable `-EBADF`, matching
            // the revocation-unpark — a forgery still traps), and so a sibling fiber parked in a
            // read through this handle can be woken by the caller (the eval loop calls
            // `Scheduler::cap_revoke` after this dispatch returns).
            // Handled here rather than in `stream_op` because closing needs the *handle*,
            // which the per-role op body deliberately never sees. Uniform across backends —
            // every backend routes through this one dispatch.
            Binding::Stream(_) if op == 2 => {
                self.close(handle);
                Ok(vec![0])
            }
            Binding::Stream(role) => self.stream_op(role, op, args, mem),
            Binding::PipeEnd { pipe, write } => self.pipe_op(pipe, write, op, args, mem),
            // A wired interface offer (IMPORTS.md §3.2): run op `op`'s function — **v1 pure
            // dispatch**. The op executes as a fresh reference run over the offer's own function
            // table with **no window and an empty powerbox**: it computes over its arguments alone,
            // so it gains exactly nothing from the wiring context (authority-neutral by
            // construction — "implementing an interface requires zero authority", and this v1
            // implements one *with* zero authority). A load/store or capability call inside the
            // impl faults/`CapFault`s fail-closed. Living in the generic dispatch keeps all three
            // backends on one implementation. Exporter-domain state (the stateful "parent `Fs`
            // backed by its own window") is the designed follow-up; fuel is a fixed deterministic
            // budget until caller-fuel threading lands with it.
            Binding::GuestImpl(idx) => {
                let entry = self
                    .guest_impls
                    .get(idx as usize)
                    .ok_or(Trap::CapFault)?
                    .clone();
                let f = *entry.ops.get(op as usize).ok_or(Trap::CapFault)?;
                let sig = entry.sigs.get(op as usize).ok_or(Trap::CapFault)?;
                if args.len() != sig.params.len() {
                    return Err(Trap::CapFault);
                }
                // §3.5 `cap` boundary translation happens per branch, because the receiver host
                // differs: `cap`-typed args are re-granted caller→provider *before* the run,
                // `cap`-typed results provider→caller *after* it. Non-`cap` slots are plain i32
                // data. A trap inside the impl (fault, fuel, CapFault) propagates verbatim — the
                // caller's call traps, fail-closed, identically on every backend.
                match &entry.state {
                    // §3.2 v2 **instanced** offer: run over the provider's persistent window +
                    // powerbox (exporter-domain state). The blocking lock is deadlock-free by
                    // construction — providers never hold offers (`grant_impl_cap` refuses
                    // them), so provider chains are acyclic and the lock order is always
                    // domain-host → provider; cross-domain contention on a shared provider
                    // serializes here, bounded by the impl fuel budget.
                    //
                    // §5.3 **provider pays**: each call is funded from the provider's drainable
                    // reserve (capped per-call by GUEST_IMPL_FUEL); a dry reserve is an inert
                    // CapFault the caller can probe and the wirer can meter
                    // ([`Host::impl_fuel_remaining`]) — its code, its choice to offer, its
                    // budget. A provider worried about a hammering caller rate-limits or kills
                    // that caller itself.
                    Some(state) => {
                        let mut st = state.lock().unwrap_or_else(|e| e.into_inner());
                        let st = &mut *st;
                        if st.fuel == 0 {
                            return Err(Trap::CapFault);
                        }
                        let mut arg_slots = args.to_vec();
                        translate_cap_slots(self, &mut st.host, &sig.params, &mut arg_slots)?;
                        let vals: Vec<Value> = sig
                            .params
                            .iter()
                            .zip(&arg_slots)
                            .map(|(ty, &s)| slot_to_val(*ty, s))
                            .collect();
                        let budget = st.fuel.min(GUEST_IMPL_FUEL);
                        let mut impl_fuel = budget;
                        let (res, _, _) = drive_arc(
                            entry.funcs.clone(),
                            f,
                            &vals,
                            &mut impl_fuel,
                            &mut st.mem,
                            &mut st.host,
                        );
                        st.fuel -= budget - impl_fuel; // drain what the call actually used
                        let mut result_slots: Vec<i64> =
                            res?.iter().map(|v| val_to_slot(*v)).collect();
                        translate_cap_slots(&mut st.host, self, &sig.results, &mut result_slots)?;
                        Ok(result_slots)
                    }
                    // v1 **pure** offer: windowless, empty powerbox — arguments in, results out,
                    // capped at the flat per-call budget (there is no provider domain to drain;
                    // the wirer accepted the bounded per-call price at wiring). The ephemeral
                    // host lives just long enough to hold any translated `cap` arg the impl
                    // forwards or returns (e.g. an identity offer over a `cap`).
                    None => {
                        let mut ephemeral = Host::new();
                        let mut arg_slots = args.to_vec();
                        translate_cap_slots(self, &mut ephemeral, &sig.params, &mut arg_slots)?;
                        let vals: Vec<Value> = sig
                            .params
                            .iter()
                            .zip(&arg_slots)
                            .map(|(ty, &s)| slot_to_val(*ty, s))
                            .collect();
                        let mut impl_fuel = GUEST_IMPL_FUEL;
                        let (res, _, _) = drive_arc(
                            entry.funcs.clone(),
                            f,
                            &vals,
                            &mut impl_fuel,
                            &mut None,
                            &mut ephemeral,
                        );
                        let mut result_slots: Vec<i64> =
                            res?.iter().map(|v| val_to_slot(*v)).collect();
                        translate_cap_slots(&mut ephemeral, self, &sig.results, &mut result_slots)?;
                        Ok(result_slots)
                    }
                }
            }
            // §7 embedder host-capability: hand `op`/args/window to the registered closure. Take it
            // out for the call so the closure can't alias `self.host_fns` (it doesn't need `Host`),
            // then restore it — a panic would only poison this one slot, never the host.
            Binding::HostFn(idx) => {
                let mut f = match self.host_fns.get_mut(idx as usize) {
                    Some(slot) => std::mem::replace(slot, Box::new(|_, _, _| Err(Trap::CapFault))),
                    None => return Err(Trap::CapFault),
                };
                let r = f(op, args, mem);
                self.host_fns[idx as usize] = f;
                r
            }
            // §4b mmap-capable host-cap: same take-out/run/restore as `HostFn`, but also hand the
            // handler `self` as the `RegionMinter`. Taking the closure out first means `self` is no
            // longer aliased by `host_fns_region[idx]`, so the `&mut dyn RegionMinter` borrow is sound.
            Binding::HostFnRegion(idx) => {
                let mut f = match self.host_fns_region.get_mut(idx as usize) {
                    Some(slot) => {
                        std::mem::replace(slot, Box::new(|_, _, _, _| Err(Trap::CapFault)))
                    }
                    None => return Err(Trap::CapFault),
                };
                let r = f(op, args, mem, self);
                self.host_fns_region[idx as usize] = f;
                r
            }
            Binding::Exit => {
                // op 0: exit(code: i32) — noreturn. Propagate as a (non-error) trap.
                let code = *args.first().ok_or(Trap::Malformed)? as i32;
                Err(Trap::Exit(code))
            }
            Binding::Clock => {
                // op 0: now(clock_id) -> i64 nanoseconds (deterministic, increasing).
                let now = self.clock_ns;
                self.clock_ns = self.clock_ns.wrapping_add(1);
                Ok(vec![now])
            }
            Binding::Memory => {
                // map(off,len,prot) / unmap(off,len) / protect(off,len,prot) on the window's
                // pages (§3e). With no window there is nothing to address (-EINVAL); the effect
                // is applied to whichever backend's memory `mem` wraps (interp `Mem` here, a
                // JIT's flat window via its own impl), keeping the two in differential lockstep.
                let Some(mem) = mem else {
                    return Ok(vec![EINVAL]);
                };
                let off = *args.first().unwrap_or(&0) as u64;
                let len = *args.get(1).unwrap_or(&0) as u64;
                let prot = *args.get(2).unwrap_or(&0) as i32;
                Ok(vec![match op {
                    0 => mem.map(off, len, prot),
                    1 => mem.unmap(off, len),
                    2 => mem.protect(off, len, prot),
                    3 => mem.page_size(),
                    _ => EINVAL,
                }])
            }
            Binding::SharedRegion(region) => {
                // §13: alias the host-backed region into the window. `map` (op 0) at several offsets
                // aliases the same bytes; loads/stores then go through the ordinary masked path.
                let Some(backing) = self.regions.get(region as usize).cloned() else {
                    return Ok(vec![EINVAL]);
                };
                let Some(mem) = mem else {
                    return Ok(vec![EINVAL]);
                };
                Ok(vec![match op {
                    0 => {
                        let win_off = *args.first().unwrap_or(&0) as u64;
                        let region_off = *args.get(1).unwrap_or(&0) as u64;
                        let len = *args.get(2).unwrap_or(&0) as u64;
                        let prot = *args.get(3).unwrap_or(&0) as i32;
                        // The backing's OS identity, stable across every alias of one region: the
                        // memfd on unix, the section HANDLE on Windows (`MapViewOfFile3`). Either makes
                        // two aliases key on the same `(backing, offset)` — a software-only region (no
                        // OS handle) can't be canonicalized on the JIT, so it stays `Anon`.
                        let backing_id = backing
                            .os_fd()
                            .map(|fd| fd as u64)
                            .or_else(|| backing.os_section().map(|s| s as u64));
                        let r = mem.map_region(win_off, region_off, len, prot, region, backing);
                        // S1b/S1c: on a successful map of an OS-fd-backed region, tell the JIT futex
                        // registry which pages now alias which region bytes (a no-op on the interp).
                        if r >= 0 {
                            if let (Some(hook), Some(id)) = (&self.region_hook, backing_id) {
                                hook(win_off, len, Some((region_off, id)));
                            }
                        }
                        r
                    }
                    1 => {
                        let win_off = *args.first().unwrap_or(&0) as u64;
                        let len = *args.get(1).unwrap_or(&0) as u64;
                        if let Some(hook) = &self.region_hook {
                            hook(win_off, len, None);
                        }
                        mem.unmap(win_off, len)
                    }
                    2 => backing.size() as i64,
                    3 => mem.region_page_size(),
                    _ => EINVAL,
                }])
            }
            Binding::Budget(idx) => {
                // §15 / PROCESS.md §5: a passable, splittable resource-quota vector.
                let idx = idx as usize;
                let Some(&BudgetState { fuel, mem, spawn }) = self.budgets.get(idx) else {
                    return Ok(vec![EINVAL]);
                };
                match op {
                    0 => {
                        // split(fuel, mem, spawn) -> sub_handle | -errno. For each field, `-1` = "all
                        // remaining"; a non-negative amount must be `<=` the holder's remaining
                        // (attenuation — a child can never exceed the parent, D19). An **unbounded**
                        // (`-1`) parent field grants any child amount and stays unbounded. Over-asking
                        // any bounded field is `-EINVAL` — the whole split fails closed, nothing deducted.
                        // Returns `(child_amount, parent_remaining_after)`.
                        let split_field = |arg: i64, rem: i64| -> Option<(i64, i64)> {
                            if arg < 0 {
                                // "all remaining": bounded parent → child takes it all (parent 0);
                                // unbounded parent → child unbounded, parent stays unbounded.
                                Some(if rem < 0 { (-1, -1) } else { (rem, 0) })
                            } else if rem < 0 {
                                (arg, -1).into() // unbounded parent, bounded request
                            } else if arg <= rem {
                                (arg, rem - arg).into()
                            } else {
                                None // over-attenuation of a bounded field
                            }
                        };
                        let (Some((cf, pf)), Some((cm, pm)), Some((cs, ps))) = (
                            split_field(*args.first().unwrap_or(&0), fuel),
                            split_field(*args.get(1).unwrap_or(&0), mem),
                            split_field(*args.get(2).unwrap_or(&0), spawn),
                        ) else {
                            return Ok(vec![EINVAL]);
                        };
                        // Mint the child budget first; draw down the parent only once the grant
                        // succeeds, so a full handle table (-EMFILE) leaves the parent's quota intact.
                        let child = self.budgets.len() as u32;
                        self.budgets.push(BudgetState {
                            fuel: cf,
                            mem: cm,
                            spawn: cs,
                        });
                        Ok(vec![
                            match self.try_grant(cap_id::BUDGET, Binding::Budget(child)) {
                                Some(h) => {
                                    let b = &mut self.budgets[idx];
                                    b.fuel = pf;
                                    b.mem = pm;
                                    b.spawn = ps;
                                    h as i64
                                }
                                None => {
                                    self.budgets.pop();
                                    EMFILE
                                }
                            },
                        ])
                    }
                    // read(field) -> remaining | -EINVAL: `0` fuel, `1` mem, `2` spawn (the §15
                    // monitoring readout). Window-independent — one field per call.
                    1 => Ok(vec![match *args.first().unwrap_or(&-1) {
                        0 => fuel,
                        1 => mem,
                        2 => spawn,
                        _ => EINVAL,
                    }]),
                    _ => Ok(vec![EINVAL]),
                }
            }
            Binding::AddressSpace { base, size } => {
                // §14: every op is confined to this capability's sub-range `[base, base+size)`. Offsets
                // are sub-range-relative; the handler bounds them and shifts by `base` into the window,
                // so a holder can never reach a byte outside its grant — the memory authority a child
                // gets from the `Instantiator`. `sub` (op 4) is **attenuation**: mint a child range.
                if op == 4 {
                    // sub(off, size_log2) -> child handle (attenuation). Mint an AddressSpace over the
                    // power-of-two-aligned `[base+off, base+off+child)` ⊆ `[base, base+size)`.
                    let off = *args.first().unwrap_or(&0) as u64;
                    let size_log2 = *args.get(1).unwrap_or(&-1);
                    if !(0..64).contains(&size_log2) {
                        return Ok(vec![EINVAL]);
                    }
                    let child = 1u64 << size_log2;
                    // child fits, `off` is child-aligned (power-of-two sub-window, D19), and the whole
                    // child range lies within this holder's range — "sub-allocate only what you hold".
                    let fits = child <= size
                        && off & (child - 1) == 0
                        && off.checked_add(child).is_some_and(|end| end <= size);
                    if !fits {
                        return Ok(vec![EINVAL]);
                    }
                    // Guest-minting: a full handle table yields -EMFILE, never a panic (§3c / audit #1).
                    return Ok(vec![match self.try_grant(
                        cap_id::ADDRESS_SPACE,
                        Binding::AddressSpace {
                            base: base + off,
                            size: child,
                        },
                    ) {
                        Some(h) => h as i64,
                        None => EMFILE,
                    }]);
                }
                if op == 5 {
                    // create_region(len) -> region handle — a **guest-minted** §13/§14 `SharedRegion`
                    // (the cross-domain data plane's `create`): the memory-management authority mints
                    // a fresh zero-filled shareable region and gets its handle, to `map` into its own
                    // window and/or `grant` into a child domain (SharedRegion op 4). Backing comes
                    // from the embedder's factory (OS shared memory under the JIT) or the reference
                    // `VecBacking`. Capped per-region (anti-bomb); real quota metering is §15 — DoS
                    // is contained, not prevented (D48).
                    let len = *args.first().unwrap_or(&0);
                    if len <= 0 || len > MAX_MINTED_REGION {
                        return Ok(vec![EINVAL]);
                    }
                    let backing = match self.region_factory {
                        Some(f) => f(len as usize),
                        None => Arc::new(VecBacking(Mutex::new(vec![0u8; len as usize]))),
                    };
                    // Guest-minting: a full handle table yields -EMFILE, never a panic (§3c / audit #1).
                    return Ok(vec![self
                        .try_grant_shared_region_backed(backing)
                        .map_or(EMFILE, |h| h as i64)]);
                }
                // map/unmap/protect/page_size — same shapes as `Memory`, but bounded to `[0, size)`
                // and shifted by `base` (a buffer/range straddling the sub-range boundary is -EINVAL).
                let Some(mem) = mem else {
                    return Ok(vec![EINVAL]);
                };
                if op == 3 {
                    return Ok(vec![mem.page_size()]);
                }
                let off = *args.first().unwrap_or(&0) as u64;
                let len = *args.get(1).unwrap_or(&0) as u64;
                let prot = *args.get(2).unwrap_or(&0) as i32;
                // The decisive confinement check: the range must be wholly within this sub-window.
                if off.checked_add(len).is_none_or(|end| end > size) {
                    return Ok(vec![EINVAL]);
                }
                Ok(vec![match op {
                    0 => mem.map(base + off, len, prot),
                    1 => mem.unmap(base + off, len),
                    2 => mem.protect(base + off, len, prot),
                    _ => EINVAL,
                }])
            }
            // The interpreter services `instantiate`/`join` in its eval loop (spawning a child vCPU
            // needs the executor the generic dispatch can't reach), so it never routes an Instantiator
            // here. A flat-window backend (the JIT) *does* — but only to **resolve this handle's
            // authority**: op 0 returns the carve range `[base, base+size)` so the JIT can compile and
            // run the child confined to a sub-window of it (the JIT owns the actual spawn). Other ops
            // are inert here (the JIT routes `join` to its own child table, never to the Host).
            Binding::Instantiator { base, size } => match op {
                0 => Ok(vec![base as i64, size as i64]),
                _ => Err(Trap::CapFault),
            },
            // The §14 `Yielder` (co-fiber `yield`) is serviced by the eval loop (it suspends the
            // running coroutine's continuation, which the generic dispatch can't); reaching here means
            // a `Yielder` cap.call slipped through (e.g. the JIT, which has no coroutine runtime) —
            // inert `CapFault`.
            Binding::Yielder => Err(Trap::CapFault),
            // A §14 `Module` handle confers instantiation authority (through the Instantiator's module
            // ops 5/6/7) plus one callable op:
            //   op 0 `resolve_export(name_ptr, name_len) -> funcidx | -errno` (F2): look a name up in
            //   the child module's first-class export table so a parent can address a child entry **by
            //   name** rather than hardcoding its funcidx; the result is passed as the `entry` to the
            //   Instantiator's module ops. The name is borrowed from the calling domain's window
            //   (fail-closed: `-EFAULT` out of bounds, `-EINVAL` on bad UTF-8 / unknown name). Only the
            //   funcidx (a small integer, already implicit in the granted module) crosses back — no
            //   host pointer or grant-internal data is ever exposed.
            Binding::Module(id) => match op {
                0 => {
                    let Some(mem) = mem else {
                        return Ok(vec![EFAULT]);
                    };
                    let ptr = *args.first().unwrap_or(&0) as u64;
                    let len = *args.get(1).unwrap_or(&0) as u64;
                    let Some(bytes) = mem.read_bytes(ptr, len) else {
                        return Ok(vec![EFAULT]);
                    };
                    let Some(g) = self.modules.get(id as usize) else {
                        return Ok(vec![EINVAL]);
                    };
                    let Ok(name) = std::str::from_utf8(&bytes) else {
                        return Ok(vec![EINVAL]);
                    };
                    Ok(vec![match g.exports.iter().find(|e| e.name == name) {
                        Some(e) => e.func as i64,
                        None => EINVAL,
                    }])
                }
                _ => Err(Trap::CapFault),
            },
            // Guest-driven `Jit` (iface 11, DESIGN.md §22). This generic arm is the **reference**
            // path (an interpreter run, or a wiring-bug fallback): op 0 `compile` validates +
            // stores the unit and mints its handle; op 2 `release` revokes one. op 1 `invoke` can
            // never be serviced here — it must *run guest code* (the interp eval loop intercepts
            // it before dispatch; the JIT embedder's thunk intercepts the whole iface) — so
            // reaching it is fail-closed.
            Binding::JitDomain(_) => match op {
                0 => {
                    // compile(ptr, len) -> code_handle | -errno. The blob is borrowed from guest
                    // memory; with no window there is nothing to read (-EFAULT, like Stream).
                    let Some(mem) = mem else {
                        return Ok(vec![EFAULT]);
                    };
                    let ptr = *args.first().unwrap_or(&0) as u64;
                    let len = *args.get(1).unwrap_or(&0) as u64;
                    let Some(bytes) = mem.read_bytes(ptr, len) else {
                        return Ok(vec![EFAULT]);
                    };
                    Ok(vec![match self.jit_compile(handle, &bytes)? {
                        Ok(c) => c.handle as i64,
                        Err(e) => e,
                    }])
                }
                1 => Err(Trap::CapFault),
                2 => {
                    // release(code_handle) -> 0 | -EINVAL. A forged/already-released handle is a
                    // non-fatal errno (guest-reachable in a loop; must not trap).
                    let ch = *args.first().unwrap_or(&0) as i32;
                    Ok(vec![match self.jit_release(ch) {
                        Ok(()) => 0,
                        Err(_) => EINVAL,
                    }])
                }
                5 => {
                    // compile_linked(ir_ptr, ir_len, symtab_ptr, symtab_len) -> code_handle |
                    // -errno (DESIGN.md §22 host-assisted dynamic linking). Like op 0 `compile`, but
                    // the unit may carry unresolved §7 imports, bound by name against the guest's
                    // symbol-table buffer before verify. Both buffers are borrowed from the
                    // window; with no window there is nothing to read (-EFAULT, like op 0).
                    let Some(mem) = mem else {
                        return Ok(vec![EFAULT]);
                    };
                    let ir_ptr = *args.first().unwrap_or(&0) as u64;
                    let ir_len = *args.get(1).unwrap_or(&0) as u64;
                    let st_ptr = *args.get(2).unwrap_or(&0) as u64;
                    let st_len = *args.get(3).unwrap_or(&0) as u64;
                    let Some(ir) = mem.read_bytes(ir_ptr, ir_len) else {
                        return Ok(vec![EFAULT]);
                    };
                    let Some(symtab) = mem.read_bytes(st_ptr, st_len) else {
                        return Ok(vec![EFAULT]);
                    };
                    Ok(vec![match self.jit_compile_linked(handle, &ir, &symtab)? {
                        Ok(c) => c.handle as i64,
                        Err(e) => e,
                    }])
                }
                _ => Ok(vec![EINVAL]),
            },
            // A `CompiledCode` handle has no directly callable ops (like `Module`): it is only
            // *named* in `Jit.invoke`/`release`.
            Binding::JitCode { .. } => Err(Trap::CapFault),
            // §9/§12 IoRing. op 0 `submit(sq_ptr, n, cq_ptr)` — synchronous batch (increment 1/2). op 1
            // `submit_async(sq_ptr, n, counter_addr)` — kick the batch onto the pool and return; each
            // completion posts to the ring's [`RingState`] + bumps the in-window futex counter + wakes
            // a parked vCPU (increment 3). op 2 `reap(cq_ptr, max)` — flush ready completions to the
            // window on the vCPU thread.
            Binding::IoRing(idx) => match op {
                0 => self.io_ring_submit(args, mem),
                1 => self.io_ring_submit_async(idx, args, mem),
                2 => self.io_ring_reap(idx, args, mem),
                _ => Ok(vec![EINVAL]),
            },
            // §12 Blocking: `op 0 work(arg) -> mix(arg)`. As a *direct* cap.call it runs inline and
            // blocks the caller (the degenerate single path); a batched `submit` instead overlaps it
            // on the offload pool. Either way the result is the same deterministic transform.
            Binding::Blocking(idx) => match op {
                0 => {
                    // §12.8 4A.7 (parked-vCPU / `Blocking.work` latency). If an async STW freeze has
                    // already landed (the global freeze word reads `UNWINDING`), a durable vCPU must
                    // not *enter* a new blocking host call: it can't be checkpointed and would extend
                    // snapshot latency by the whole call (R6). Fail **closed** — the freeze refuses —
                    // mirroring the `thread.wait` deadlock fail-closed (`Trap::ThreadFault`). Gated on
                    // `is_durable` so a non-durable guest's byte at window offset 0 can't spuriously
                    // refuse; cancelling an *already in-flight* call is deferred (R2). Both backends
                    // funnel through here (the JIT via `svm-run`'s `cap_thunk`), so the refusal is
                    // backend-agnostic by construction.
                    if self.is_durable() && freeze_has_landed(mem.as_deref()) {
                        return Err(Trap::ThreadFault);
                    }
                    let arg = *args.first().unwrap_or(&0);
                    Ok(vec![self.blockings[idx as usize].run(arg)])
                }
                _ => Ok(vec![EINVAL]),
            },
        }
    }

    /// §9/§12 the **submit/complete ring** (io_uring-shaped). `submit(sq_ptr, n, cq_ptr)` reads `n`
    /// 64-byte SQEs from `[sq_ptr, …)` (each a *deferred `cap.call`*) and writes a 32-byte CQE to
    /// `[cq_ptr, …)` per entry; returns the count completed. One boundary crossing for `n` ops (the
    /// §1a interface-amortization win).
    ///
    /// **Two execution classes (increment 2 — the bounded offload pool):**
    /// - **Inline** — ops that touch the window or `&mut Host` (Clock, Memory, Stream, …) run on the
    ///   submitting thread through the normal dispatch, in SQE order, exactly as increment 1.
    /// - **Offloaded** — `Blocking` ops (window-independent, `&mut Host`-free) are handed to the
    ///   bounded [`OffloadPool`] and run **concurrently** on `K` threads (waves of `K`), so the
    ///   guest's vCPU thread isn't multiplied by the blocking count (§12 "0 blocked vCPU threads").
    ///
    /// Window reads (SQE parse) and writes (CQE) stay on the submit thread; only the offloaded *op
    /// bodies* overlap, and each `Blocking` result is a deterministic pure transform — so the final
    /// window is **identical to running every op inline in order**, and both backends still agree (the
    /// §18 oracle). The submit blocks until the whole batch completes (fiber-parking is increment 3).
    ///
    /// SQE (64 B, little-endian): `u32 type_id | u32 op | i32 handle | u32 n_args | i64 args[4] |
    /// i64 user_data | i64 pad`. CQE (32 B): `i64 user_data | i64 result | i64 status (0=ok, else a
    /// TrapKind code) | i64 pad`. A nested `IoRing` op, or an op the dispatch can't service
    /// (Instantiator/Yielder/Module → `CapFault`), simply lands as a CQE with a non-zero `status` —
    /// never a host panic and never unbounded recursion.
    fn io_ring_submit(
        &mut self,
        args: &[i64],
        mem: Option<&mut dyn GuestMem>,
    ) -> Result<Vec<i64>, Trap> {
        const SQE: u64 = 64;
        const CQE: u64 = 32;
        const MAX_SQ_ARGS: usize = 4;
        // A ring with no window is inert (`-EFAULT`); otherwise borrow the window once and reborrow it
        // (`&mut *m`) for each SQE's read, inline dispatch, and CQE write.
        let m = match mem {
            Some(m) => m,
            None => return Ok(vec![EFAULT]),
        };
        let sq_ptr = *args.first().unwrap_or(&0) as u64;
        // Clamp the guest-supplied entry count to a bounded batch ceiling. Without this a forged `n`
        // (e.g. `i64::MAX`) would both `with_capacity`-allocate ~`n * size_of::<Pending>()` bytes (an
        // uncatchable allocator `abort()` — a host crash from a guest, defeating §5) and spin the
        // submit loop `n` times. Any real ring batch fits well under this; a guest needing more issues
        // multiple submits. The clamped value is what we report submitted.
        let n = ((*args.get(1).unwrap_or(&0)).max(0) as u64).min(MAX_RING_BATCH);
        let cq_ptr = *args.get(2).unwrap_or(&0) as u64;

        // One pending completion per SQE we managed to read; filled inline now, or by the pool below.
        // (An unreadable SQE writes its `-EFAULT` CQE immediately and is not tracked here.)
        struct Pending {
            at: u64,
            user_data: i64,
            result: i64,
            status: i64,
        }
        let mut pending: Vec<Pending> = Vec::with_capacity(n as usize);
        // Offloadable `Blocking` SQEs: `(index into `pending`, its state, its argument)`.
        let mut offload: Vec<(usize, Arc<AsyncState>, i64)> = Vec::new();

        for i in 0..n {
            let at = cq_ptr + i * CQE;
            // Read SQE i (a borrow-checked window read; out-of-window ⇒ -EFAULT completion).
            let raw = match m.read_bytes(sq_ptr + i * SQE, SQE) {
                Some(r) => r,
                None => {
                    Self::write_cqe(&mut *m, at, 0, 0, -EFAULT);
                    continue;
                }
            };
            let type_id = u32::from_le_bytes(raw[0..4].try_into().unwrap());
            let op = u32::from_le_bytes(raw[4..8].try_into().unwrap());
            let handle = i32::from_le_bytes(raw[8..12].try_into().unwrap());
            let n_args =
                (u32::from_le_bytes(raw[12..16].try_into().unwrap()) as usize).min(MAX_SQ_ARGS);
            let mut opargs = [0i64; MAX_SQ_ARGS];
            for (a, slot) in opargs.iter_mut().enumerate().take(n_args) {
                *slot = i64::from_le_bytes(raw[16 + a * 8..24 + a * 8].try_into().unwrap());
            }
            let user_data = i64::from_le_bytes(raw[48..56].try_into().unwrap());

            if type_id == cap_id::IO_RING {
                // A ring submitting to a ring would recurse without bound — inert CapFault.
                pending.push(Pending {
                    at,
                    user_data,
                    result: 0,
                    status: trap_status(&Trap::CapFault),
                });
            } else if type_id == cap_id::BLOCKING && op == 0 {
                // Offloadable iff the handle actually resolves to a `Blocking` binding; a forged /
                // wrong-type handle is an inert CapFault (the I2 check), never queued.
                match self.resolve(handle, cap_id::BLOCKING) {
                    Ok(Binding::Blocking(idx)) => {
                        let slot = pending.len();
                        pending.push(Pending {
                            at,
                            user_data,
                            result: 0, // filled from the pool below
                            status: 0,
                        });
                        offload.push((slot, Arc::clone(&self.blockings[idx as usize]), opargs[0]));
                    }
                    _ => pending.push(Pending {
                        at,
                        user_data,
                        result: 0,
                        status: trap_status(&Trap::CapFault),
                    }),
                }
            } else {
                // Inline: window-/host-touching ops run on the submit thread, in order.
                let (result, status) = match self.cap_dispatch_slots(
                    type_id,
                    op,
                    handle,
                    &opargs[..n_args],
                    Some(&mut *m),
                ) {
                    Ok(res) => (res.first().copied().unwrap_or(0), 0),
                    Err(t) => (0, trap_status(&t)),
                };
                pending.push(Pending {
                    at,
                    user_data,
                    result,
                    status,
                });
            }
        }

        // Run the offloadable blocking ops concurrently on the bounded pool (created lazily so a Host
        // that never offloads spawns no threads). Each job writes its result by index; the submit
        // thread parks until the whole batch posts completion, then we copy results back in order.
        if !offload.is_empty() {
            // §12.8 4A.7 (parked-vCPU / `Blocking.work` latency). A batched submit *parks* the vCPU
            // thread on the pool until the whole offload completes — no poll site. If an async STW
            // freeze has already landed, fail **closed** rather than start the batch (the same
            // fail-closed as a direct `Blocking.work` cap.call); the submit thread would otherwise
            // stall the freeze for the whole batch (R6). Cancelling in-flight pool work is deferred
            // (R2). Only the *offloadable* (blocking) batch is gated — an all-inline submit already
            // ran above and parks nothing.
            if self.is_durable() && freeze_has_landed(Some(&*m)) {
                return Err(Trap::ThreadFault);
            }
            let results: Arc<Vec<AtomicI64>> =
                Arc::new(offload.iter().map(|_| AtomicI64::new(0)).collect());
            let mut jobs: Vec<OffloadJob> = Vec::with_capacity(offload.len());
            for (k, (_slot, state, arg)) in offload.iter().enumerate() {
                let state = Arc::clone(state);
                let arg = *arg;
                let results = Arc::clone(&results);
                jobs.push(Box::new(move || {
                    results[k].store(state.run(arg), Ordering::SeqCst);
                }));
            }
            let pool = self
                .pool
                .get_or_insert_with(|| OffloadPool::new(OFFLOAD_POOL_THREADS));
            pool.run_batch(jobs);
            for (k, (slot, _, _)) in offload.iter().enumerate() {
                pending[*slot].result = results[k].load(Ordering::SeqCst);
            }
        }

        for p in &pending {
            Self::write_cqe(&mut *m, p.at, p.user_data, p.result, p.status);
        }
        Ok(vec![n as i64])
    }

    /// §9/§12 **async submit** (op 1, increment 3). `submit_async(sq_ptr, n, counter_addr)` reads `n`
    /// SQEs, kicks the **offloadable** (`Blocking`) ones onto the bounded pool, runs the inline ones
    /// immediately, and returns the count submitted **without waiting**. Each completion posts its CQE
    /// to the ring's host-side [`RingState`] and atomic-increments the 4-byte futex **completion
    /// counter** at `counter_addr`; an offloaded completion additionally `notify`s the counter key to
    /// wake a vCPU parked in `wait` on it — an I/O completion *is* a futex notify (DESIGN §12). The
    /// guest then parks on the counter, runs other fibers, and `reap`s once it advances.
    ///
    /// Requires the backend to expose the futex counter (`async_counter`) **and** the wake hook
    /// (`async_notify`); without them — the JIT pre-§3b, or the deterministic explorer — it returns
    /// `-EINVAL`, and the guest is expected to fall back to the synchronous `submit`. CQEs are written
    /// only by `reap` on the vCPU thread, so the single counter atomic is the *only* cross-thread
    /// window write an async ring performs.
    fn io_ring_submit_async(
        &mut self,
        ring_idx: u32,
        args: &[i64],
        mem: Option<&mut dyn GuestMem>,
    ) -> Result<Vec<i64>, Trap> {
        const SQE: u64 = 64;
        const MAX_SQ_ARGS: usize = 4;
        let m = match mem {
            Some(m) => m,
            None => return Ok(vec![EFAULT]),
        };
        let sq_ptr = *args.first().unwrap_or(&0) as u64;
        // Bounded batch ceiling, as in `io_ring_submit` — a forged `n` must not drive an unbounded
        // allocation or loop on the host.
        let n = ((*args.get(1).unwrap_or(&0)).max(0) as u64).min(MAX_RING_BATCH);
        let counter_addr = *args.get(2).unwrap_or(&0) as u64;

        // The backend must expose the futex counter handle + the wake hook, else there is no async
        // path here (the guest falls back to the synchronous `submit`).
        let counter = match m.async_counter(counter_addr) {
            Some(c) => c,
            None => return Ok(vec![EINVAL]),
        };
        let notify = match &self.async_notify {
            Some(f) => Arc::clone(f),
            None => return Ok(vec![EINVAL]),
        };
        let ring = Arc::clone(&self.rings[ring_idx as usize]);

        let mut jobs: Vec<OffloadJob> = Vec::new();
        let mut inline_done: u32 = 0; // completions ready before we return (counter bumped once below)
        for i in 0..n {
            let raw = match m.read_bytes(sq_ptr + i * SQE, SQE) {
                Some(r) => r,
                None => {
                    ring.completed.lock().unwrap().push_back((0, 0, -EFAULT));
                    inline_done += 1;
                    continue;
                }
            };
            let type_id = u32::from_le_bytes(raw[0..4].try_into().unwrap());
            let op = u32::from_le_bytes(raw[4..8].try_into().unwrap());
            let handle = i32::from_le_bytes(raw[8..12].try_into().unwrap());
            let n_args =
                (u32::from_le_bytes(raw[12..16].try_into().unwrap()) as usize).min(MAX_SQ_ARGS);
            let mut opargs = [0i64; MAX_SQ_ARGS];
            for (a, slot) in opargs.iter_mut().enumerate().take(n_args) {
                *slot = i64::from_le_bytes(raw[16 + a * 8..24 + a * 8].try_into().unwrap());
            }
            let user_data = i64::from_le_bytes(raw[48..56].try_into().unwrap());

            if type_id == cap_id::BLOCKING && op == 0 {
                if let Ok(Binding::Blocking(bidx)) = self.resolve(handle, cap_id::BLOCKING) {
                    // Offload: compute on a pool thread, post the completion, then bump+notify so a
                    // parked vCPU wakes (the counter write happens-before the notify, so the futex
                    // compare-under-lock can't lose the wakeup).
                    let state = Arc::clone(&self.blockings[bidx as usize]);
                    let arg = opargs[0];
                    let ring = Arc::clone(&ring);
                    let counter = Arc::clone(&counter);
                    let notify = Arc::clone(&notify);
                    jobs.push(Box::new(move || {
                        let r = state.run(arg);
                        ring.completed.lock().unwrap().push_back((user_data, r, 0));
                        counter.increment(1);
                        notify(counter.key(), u32::MAX);
                    }));
                    continue;
                }
                // forged / wrong-type Blocking handle → inert CapFault completion (the I2 check).
                ring.completed.lock().unwrap().push_back((
                    user_data,
                    0,
                    trap_status(&Trap::CapFault),
                ));
                inline_done += 1;
            } else if type_id == cap_id::IO_RING {
                // A ring submitting to a ring would recurse without bound — inert CapFault.
                ring.completed.lock().unwrap().push_back((
                    user_data,
                    0,
                    trap_status(&Trap::CapFault),
                ));
                inline_done += 1;
            } else {
                // Inline: window-/host-touching ops run now on the submit thread.
                let (result, status) = match self.cap_dispatch_slots(
                    type_id,
                    op,
                    handle,
                    &opargs[..n_args],
                    Some(&mut *m),
                ) {
                    Ok(res) => (res.first().copied().unwrap_or(0), 0),
                    Err(t) => (0, trap_status(&t)),
                };
                ring.completed
                    .lock()
                    .unwrap()
                    .push_back((user_data, result, status));
                inline_done += 1;
            }
        }

        // Account the inline completions on the counter once (no wake — the guest can't be parked
        // during its own submit). Offloaded ones bump the counter as they finish.
        if inline_done > 0 {
            counter.increment(inline_done as u64);
        }
        if !jobs.is_empty() {
            let pool = self
                .pool
                .get_or_insert_with(|| OffloadPool::new(OFFLOAD_POOL_THREADS));
            pool.dispatch(jobs);
        }
        Ok(vec![n as i64])
    }

    /// §9/§12 **reap** (op 2). `reap(cq_ptr, max) -> n_reaped` pops up to `max` ready completions from
    /// the ring's [`RingState`] and writes them as 32-byte CQEs to `[cq_ptr, …)`, on the vCPU thread.
    fn io_ring_reap(
        &mut self,
        ring_idx: u32,
        args: &[i64],
        mem: Option<&mut dyn GuestMem>,
    ) -> Result<Vec<i64>, Trap> {
        const CQE: u64 = 32;
        let m = match mem {
            Some(m) => m,
            None => return Ok(vec![EFAULT]),
        };
        let cq_ptr = *args.first().unwrap_or(&0) as u64;
        let max = (*args.get(1).unwrap_or(&0)).max(0) as u64;
        let ring = Arc::clone(&self.rings[ring_idx as usize]);
        let mut q = ring.completed.lock().unwrap();
        let mut i = 0u64;
        while i < max {
            let Some((ud, result, status)) = q.pop_front() else {
                break;
            };
            Self::write_cqe(&mut *m, cq_ptr + i * CQE, ud, result, status);
            i += 1;
        }
        Ok(vec![i as i64])
    }

    /// Write one 32-byte CQE (little-endian) at `at`. A bad address is dropped (the guest's bug; the
    /// `completed` count still reflects the SQEs the host ran).
    fn write_cqe(m: &mut dyn GuestMem, at: u64, user_data: i64, result: i64, status: i64) {
        let mut b = [0u8; 32];
        b[0..8].copy_from_slice(&user_data.to_le_bytes());
        b[8..16].copy_from_slice(&result.to_le_bytes());
        b[16..24].copy_from_slice(&status.to_le_bytes());
        let _ = m.write_bytes(at, &b);
    }

    /// `Stream` ops (§3e D43): 0 `read`, 1 `write`, 2 `close`. Buffers are `(ptr,len)`,
    /// borrow-only — the host reads/writes the guest window in place after the §7
    /// trampoline bounds-checks `[ptr,ptr+len) ⊆ [0,size)` (violation → `-EFAULT`).
    fn stream_op(
        &mut self,
        role: StreamRole,
        op: u32,
        args: &[i64],
        mem: Option<&mut dyn GuestMem>,
    ) -> Result<Vec<i64>, Trap> {
        let ret = |v: i64| Ok(vec![v]);
        match op {
            0 => {
                // read(buf, len) -> bytes read (>=0) or -errno; only stdin is readable.
                if role != StreamRole::In {
                    return ret(EINVAL);
                }
                let ptr = *args.first().ok_or(Trap::Malformed)? as u64;
                let len = *args.get(1).ok_or(Trap::Malformed)? as u64;
                let avail = &self.stdin[self.stdin_pos.min(self.stdin.len())..];
                // Blocking stdin (opt-in, e.g. a persistent REPL session): an exhausted buffer parks
                // the vCPU at this read rather than returning EOF. Signal the driver (which re-issues
                // the read after pushing more input) and return a placeholder the driver discards.
                if avail.is_empty() && self.stdin_block {
                    self.stdin_parked = true;
                    return ret(0);
                }
                let n = (len as usize).min(avail.len());
                let chunk = avail[..n].to_vec();
                let Some(m) = mem else {
                    return ret(EFAULT);
                };
                if m.write_bytes(ptr, &chunk).is_none() {
                    return ret(EFAULT);
                }
                self.stdin_pos += n;
                ret(n as i64)
            }
            1 => {
                // write(buf, len) -> bytes written (>=0) or -errno; stdin is not writable.
                if role == StreamRole::In {
                    return ret(EINVAL);
                }
                let ptr = *args.first().ok_or(Trap::Malformed)? as u64;
                let len = *args.get(1).ok_or(Trap::Malformed)? as u64;
                let Some(m) = mem else {
                    return ret(EFAULT);
                };
                let Some(bytes) = m.read_bytes(ptr, len) else {
                    return ret(EFAULT);
                };
                // S2: a re-granted stdout/stderr routes to a shared sink (so a child's output reaches
                // the embedder that granted it); otherwise to this host's local buffer.
                let sink = if role == StreamRole::Out {
                    &self.out_sink
                } else {
                    &self.err_sink
                };
                match sink {
                    Some(s) => s
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .extend_from_slice(&bytes),
                    None if role == StreamRole::Out => self.stdout.extend_from_slice(&bytes),
                    None => self.stderr.extend_from_slice(&bytes),
                }
                ret(len as i64)
            }
            2 => ret(0), // close: no-op in the MVP (exit reclaims all)
            _ => ret(EINVAL),
        }
    }

    /// §4 / S4 host-served **pipe** end, dispatched under iface `STREAM` (a pipe end *is* a stream).
    /// Non-blocking: the `read` half drains bytes the `write` half has queued (op 0), a `read` of an
    /// empty pipe returns `0`; the `write` half appends (op 1). Wrong-direction ops are `-EINVAL`. Same
    /// `(buf, len) -> n | -errno` shapes as `stream_op`, so a personality's fd layer treats a pipe end
    /// and a stdio stream identically.
    fn pipe_op(
        &mut self,
        pipe: u32,
        write: bool,
        op: u32,
        args: &[i64],
        mem: Option<&mut dyn GuestMem>,
    ) -> Result<Vec<i64>, Trap> {
        let ret = |v: i64| Ok(vec![v]);
        let Some(backing) = self.pipes.get(pipe as usize) else {
            return ret(EINVAL);
        };
        let mut fifo = backing.lock().unwrap_or_else(|e| e.into_inner());
        match op {
            0 => {
                // read(buf, len) -> n; only the read end is readable.
                if write {
                    return ret(EINVAL);
                }
                let ptr = *args.first().ok_or(Trap::Malformed)? as u64;
                let len = *args.get(1).ok_or(Trap::Malformed)? as u64;
                let n = (len as usize).min(fifo.len());
                let chunk: Vec<u8> = fifo.drain(..n).collect();
                let Some(m) = mem else {
                    return ret(EFAULT);
                };
                if m.write_bytes(ptr, &chunk).is_none() {
                    // The bytes are already drained; a fail-closed buffer is the guest's bug, but keep
                    // them out of the FIFO rather than re-queue (matches `stream_op`'s stdin semantics).
                    return ret(EFAULT);
                }
                ret(n as i64)
            }
            1 => {
                // write(buf, len) -> n; only the write end is writable.
                if !write {
                    return ret(EINVAL);
                }
                let ptr = *args.first().ok_or(Trap::Malformed)? as u64;
                let len = *args.get(1).ok_or(Trap::Malformed)? as u64;
                let Some(m) = mem else {
                    return ret(EFAULT);
                };
                let Some(bytes) = m.read_bytes(ptr, len) else {
                    return ret(EFAULT);
                };
                fifo.extend(bytes);
                ret(len as i64)
            }
            2 => ret(0), // close: no-op in the MVP (exit reclaims all)
            _ => ret(EINVAL),
        }
    }
}

// ----------------------------------------------------------------------------
// Linear memory — the trap-confinement *reference* (§4, invariant I1)
// ----------------------------------------------------------------------------

/// The **host** page size — the granularity of the protection model (RO/unmap) *and* the lazy
/// backing-store chunk. Queried so the interpreter's protection granularity matches the JIT's real
/// `mprotect` on the same host (§4 "pin page size", host-page default); both backends query the
/// same value, so they agree page-for-page on any platform (4 KiB / 16 KiB / …). Lazy paging keeps
/// interpreter memory bounded by what a (fuel-limited) run touches, so a huge declared window never
/// eagerly allocates — safe to fuzz.
fn host_page_size() -> u64 {
    // wasm has no host MMU and no `mprotect`: linear-memory pages are a fixed 64 KiB, so report that
    // (and avoid pulling the `page_size` crate into the wasm dependency graph). On native, query the
    // real host page so interpreter and JIT agree page-for-page.
    #[cfg(target_family = "wasm")]
    {
        65536
    }
    #[cfg(not(target_family = "wasm"))]
    match page_size::get() as u64 {
        0 => 4096,
        p => p,
    }
}

/// The granularity a `SharedRegion` map (§13) aligns to — distinct from [`host_page_size`] (the
/// protection granularity) because a *shared mapping* is coarser on Windows. On unix this is the
/// host page (`mmap(MAP_FIXED)` aliases at page granularity); on Windows it is the **allocation
/// granularity** (64 KiB), which `MapViewOfFile3` *requires* for both the placement address and the
/// section offset. Both the interpreter reference and the JIT's flat window report this for
/// `SharedRegion` op 3 (`region_page_size`), so a guest aligns its region maps to a single value that
/// works on every backend and the §13 differential stays in lockstep. `page_size::get_granularity`
/// returns `dwAllocationGranularity` on Windows and the page size on unix.
pub fn host_region_granularity() -> u64 {
    // wasm: no separate allocation granularity (no `MapViewOfFile3`), so a region aligns to the same
    // 64 KiB linear-memory page as the protection model.
    #[cfg(target_family = "wasm")]
    {
        host_page_size()
    }
    #[cfg(not(target_family = "wasm"))]
    match page_size::get_granularity() as u64 {
        0 => host_page_size(),
        g => g,
    }
}

/// Explicit per-page state in the guest-visible address space (§3e Memory cap / §4).
///
/// A page absent from the map takes the **default for its region**: read+write inside the
/// initial backed prefix `[0, mapped)`, and *unmapped* in the reserved tail `[mapped, reserved)`
/// — so growth into the tail must be made explicit by a `map` (a [`PageProt::Rw`] entry). This is
/// what lets the guest `map`/`unmap`/`protect` sparsely across the whole reserved window (the §1a
/// "sparse address space / lazy page supply" capability), in lockstep with the JIT's real page
/// tables (an uncommitted page is `PROT_NONE` there and faults identically).
///
/// A committed *anonymous* page is zero-filled and lives in [`Mem::pages`]; a [`PageProt::Backed`]
/// page's bytes instead live in a §13 `SharedRegion` buffer (keyed in [`Mem::regions`]) — the
/// primitive behind aliasing / the magic-ring-buffer trick. Crucially the access path
/// ([`Mem::byte`]/[`Mem::set_byte`]) just redirects where a page's bytes live; loads/stores stay
/// ordinary masked accesses (zero overhead), exactly as §13 specifies.
#[derive(Clone, Copy, PartialEq, Eq)]
enum PageProt {
    /// Explicitly `map`ped read-write — committed even in the reserved tail (where *absent* would
    /// mean unmapped). Within the initial prefix, plain read-write is left *absent* (the default),
    /// so this entry only appears for grown/re-committed pages.
    Rw,
    /// `protect`ed read-only: reads succeed, a store faults (the D40 const-segment mechanism).
    Ro,
    /// `unmap`ped: any access faults.
    Unmapped,
    /// §13 aliased page: its bytes live at `region_off` in the `SharedRegion` `region`
    /// ([`Mem::regions`]), not in an anonymous [`Mem::pages`] entry. `writable` mirrors the map
    /// `prot` (a store to a read-only alias faults). Two pages with the same `region` (mapped at
    /// different window offsets) name the same backing → aliasing.
    Backed {
        region: u32,
        region_off: u64,
        writable: bool,
    },
}

/// A guest linear-memory window. Confinement itself lives in [`svm_mask::Window`]
/// (the isolated, separately-fuzzed security unit, §4); `Mem` owns the lazily paged backing
/// store, threads accesses through that confinement, and carries the guest-visible page
/// protection map (`map`/`unmap`/`protect`, §3e). This is the semantics the JIT is
/// differential-tested against (§18).
struct Mem {
    window: Window,
    /// Host page size (`host_page_size()`): protection + storage-chunk granularity. Cached per
    /// `Mem` so every method shares the one host-queried value (matches the JIT's `mprotect`).
    page: u64,
    /// The anonymous-page backing: a [`svm_mem::Region`] (`#![forbid(unsafe_code)]`-friendly) sized
    /// to the window's reserved extent. On unix this is one demand-zeroed `mmap` — the shareable
    /// substrate parallel vCPUs run over with real hardware atomics (§12); elsewhere a paged
    /// fallback. §13 aliased pages live in `regions`, not here. Held in an `Arc` so a spawned vCPU
    /// (`thread.spawn`) shares the *same* bytes — `Region`'s accessors are all `&self`, so the `Arc`
    /// derefs transparently.
    back: Arc<Region>,
    /// The guest-visible **address space** — page-protection map + §13 region backings — behind a
    /// shared `RwLock` so all vCPUs of a run see one another's `map`/`unmap`/`protect` live (§12). A
    /// spawned vCPU ([`Mem::fork_for_thread`]) clones this `Arc`, sharing the same address space; the
    /// `RwLock` lets the many readers (every protection check) run concurrently while `map`/`unmap`
    /// take the brief write lock.
    space: Arc<RwLock<AddrSpace>>,
    /// Fast-path flag: set once any §13 region is aliased in (monotonic). While clear — the
    /// overwhelmingly common case — the per-byte path skips the address-space lock entirely and goes
    /// straight to `back`, since no page can be `Backed`. Shared with forked vCPUs.
    has_regions: Arc<AtomicBool>,
    /// Fast-path flag: set once the address space is **ever** mutated (`map`/`unmap`/`protect`, §13
    /// region alias, demand/supply paging) — monotonic, dirtied at the [`Mem::space_write`] choke
    /// point. While clear — the overwhelmingly common case (no syscalls, no coroutines, no regions)
    /// — [`Mem::check_prot`] knows every in-prefix page is plain RW and skips the address-space
    /// `RwLock` read entirely. Shared with the same topology as `space` (cloned for a forked thread,
    /// fresh for a §14 child, which has its own address space).
    prot_dirty: Arc<AtomicBool>,
    /// §14 fault-driven-yield side-channel: the confined address of the most recent **recoverable**
    /// page fault (an in-window access to an unmapped/read-only page — `check_prot` sets it,
    /// `confine_checked` clears it to [`NO_FAULT`]). A coroutine child with `fault_yields` reads it
    /// after a `MemoryFault` to distinguish a recoverable page fault (suspend to the parent, which
    /// supplies the page) from an out-of-window fault (a real trap). Per-`Mem` (each vCPU owns its
    /// own), written/read only by the owning thread; `AtomicU64` keeps `Mem: Sync` for the futex path.
    last_fault: AtomicU64,
    /// Monotonic count of operations that **actually changed** a byte (a `store`/`atomic.store`/
    /// `atomic.rmw`, or an `atomic.cmpxchg` that *swapped*). The deterministic explorer reads the
    /// per-turn delta to drive spin-loop detection (a turn that changed no memory and returned the vCPU
    /// to the same local configuration is a pure spin → park it) and spin wakeups (a change wakes
    /// spinners parked on the written address). Per-`Mem` (only the running vCPU writes through its own).
    writes: u64,
}

/// Sentinel for [`Mem::last_fault`] meaning "no recoverable fault pending" — never a valid confined
/// address (every access is bounded to `< reserved ≤ 2^MAX_JIT_WINDOW_LOG2`).
const NO_FAULT: u64 = u64::MAX;

/// The shared, synchronized guest address space (§12): the page-protection map plus the §13 region
/// backings, mutated by `map`/`unmap`/`protect` and read by every access check. Lives behind
/// `Mem::space`'s `RwLock`; all vCPUs of a run share one.
#[derive(Default)]
struct AddrSpace {
    /// Page index (`offset / page`) ⇒ explicit page state. A page absent from the map takes its
    /// region default: read+write inside the initial prefix `[0, mapped)`, unmapped in the
    /// reserved tail `[mapped, reserved)`. Entries appear for `protect`ed (`Ro`), `unmap`ped
    /// (`Unmapped`), and grown/re-committed tail (`Rw`) pages — anywhere in `[0, reserved)`.
    prot: BTreeMap<u64, PageProt>,
    /// §13 `SharedRegion` backings this window has aliased in, keyed by region id (the bytes a
    /// [`PageProt::Backed`] page redirects to). A clone of the `Host`'s `Arc`, so two windows — or
    /// two offsets in one window — that map the same region share the *same* bytes.
    regions: BTreeMap<u32, RegionBacking>,
}

impl Mem {
    /// A window whose mask domain is `1 << reserved_log2` bytes but whose backed region is the
    /// declared `1 << mapped_log2` prefix; an access into the reserved-but-unmapped tail faults
    /// (the §4 "guard-when-bounded" model). `reserved_log2` is raised to at least `mapped_log2`,
    /// so passing `0` yields a fully-mapped window. Lazy paging means a huge mask domain (or
    /// reservation) never eagerly allocates.
    fn with_reservation(reserved_log2: u8, mapped_log2: u8) -> Mem {
        let reserved_log2 = reserved_log2.max(mapped_log2);
        let window = Window::with_mapped(reserved_log2, 1u64 << mapped_log2.min(63));
        let page = host_page_size();
        Mem {
            back: Arc::new(Region::new(window.reserved(), page)),
            window,
            page,
            space: Arc::new(RwLock::new(AddrSpace::default())),
            has_regions: Arc::new(AtomicBool::new(false)),
            prot_dirty: Arc::new(AtomicBool::new(false)),
            last_fault: AtomicU64::new(NO_FAULT),
            writes: 0,
        }
    }

    /// Like [`Mem::with_reservation`], but the backing is a **caller-provided** [`Region`] (e.g. a
    /// [`Region::shared`] over the host's shared linear memory) rather than an engine-`mmap`ped one —
    /// the substrate the parallel-wasm backend runs over (every per-vCPU Worker executes over the same
    /// shared window). `back` must address ≥ the mapped prefix `1 << mapped_log2`; reserved-tail
    /// accesses beyond it read as zero (a confined guest stays in its prefix). The crate stays
    /// `#![forbid(unsafe_code)]` — the `unsafe` of borrowing host memory is in the embedder's
    /// `Region::shared` call, not here. Today still driven cooperatively; only the backing changes.
    fn with_reservation_over(reserved_log2: u8, mapped_log2: u8, back: Arc<Region>) -> Mem {
        let reserved_log2 = reserved_log2.max(mapped_log2);
        let window = Window::with_mapped(reserved_log2, 1u64 << mapped_log2.min(63));
        let page = host_page_size();
        Mem {
            back,
            window,
            page,
            space: Arc::new(RwLock::new(AddrSpace::default())),
            has_regions: Arc::new(AtomicBool::new(false)),
            prot_dirty: Arc::new(AtomicBool::new(false)),
            last_fault: AtomicU64::new(NO_FAULT),
            writes: 0,
        }
    }

    /// A fully-mapped **§14 sub-window**: a `1 << size_log2`-byte child window at absolute offset
    /// `base` inside a parent backing of `parent_bytes` bytes (the child runs over the parent's
    /// `Region`). The confinement unit ([`svm_mask::Window::sub`], fuzzed as the escape hinge) bounds
    /// every child access to `[base, base + size)` (faulting otherwise), so the child can reach **only
    /// its slice** — never
    /// the parent's other memory or outside the parent window. `base` is size-aligned by `Window::sub`;
    /// the whole slice is backed (no `map`-growth inside a child yet). The backing is sized to hold
    /// `[0, base + size)`.
    fn sub_window(base: u64, size_log2: u8, parent_bytes: u64) -> Mem {
        let window = Window::sub(base, size_log2, 1u64 << size_log2.min(63));
        let page = host_page_size();
        let need = window.base().saturating_add(window.reserved());
        Mem {
            back: Arc::new(Region::new(parent_bytes.max(need), page)),
            window,
            page,
            space: Arc::new(RwLock::new(AddrSpace::default())),
            has_regions: Arc::new(AtomicBool::new(false)),
            prot_dirty: Arc::new(AtomicBool::new(false)),
            last_fault: AtomicU64::new(NO_FAULT),
            writes: 0,
        }
    }

    /// Read/write the shared address space, recovering from a poisoned lock (the interpreter never
    /// panics while holding it) rather than propagating the panic.
    fn space_read(&self) -> std::sync::RwLockReadGuard<'_, AddrSpace> {
        self.space.read().unwrap_or_else(|e| e.into_inner())
    }
    fn space_write(&self) -> std::sync::RwLockWriteGuard<'_, AddrSpace> {
        // Any address-space mutation (the only reason to take the write lock) disables the lock-free
        // `check_prot` fast path, monotonically. Set *before* acquiring the lock so a concurrent
        // reader never observes a clear flag after a mutation has begun (a false positive just makes
        // it take the read lock — always safe; a false negative would not be).
        self.prot_dirty.store(true, Ordering::Release);
        self.space.write().unwrap_or_else(|e| e.into_inner())
    }

    /// Build the memory view a spawned vCPU (`thread.spawn`) starts with (§12): it shares the **same**
    /// everything — the `Arc<Region>` bytes *and* the `Arc<RwLock<AddrSpace>>` address space — so a
    /// `map`/`unmap`/`protect` (or §13 alias) by any vCPU is immediately visible to the others.
    /// Confinement (`window`/`page`) is copied (identical for every vCPU of the run).
    fn fork_for_thread(&self) -> Mem {
        Mem {
            window: self.window,
            page: self.page,
            back: Arc::clone(&self.back),
            space: Arc::clone(&self.space),
            has_regions: Arc::clone(&self.has_regions),
            prot_dirty: Arc::clone(&self.prot_dirty),
            last_fault: AtomicU64::new(NO_FAULT),
            writes: 0,
        }
    }

    /// Build the memory view a **§14 nested child** runs over: it shares this (parent's) `Arc<Region>`
    /// bytes — so the parent intrinsically sees all of the child's bytes (the superset, §14) — but
    /// **confines** the child to the fully-mapped sub-window `[abs_base, abs_base + 2^size_log2)`
    /// (window-absolute, in the shared backing's coordinates). The child sees a zero-based `[0, size)`
    /// and cannot learn it is nested; trap-confinement ([`Window::sub`]) bounds the child to its slice.
    ///
    /// Unlike [`fork_for_thread`](Mem::fork_for_thread), the child gets its **own** address space (a
    /// fresh, empty page-protection map + §13 region set), *not* the parent's: page protections are a
    /// per-domain view, and the prot map is keyed window-relative, so a shared map would alias the
    /// child's pages onto the parent's (a child `unmap` of *its* page 0 would hit the parent's). The
    /// domains share **bytes**, not page-protection state — cross-domain memory sharing is §13, and
    /// lazy paging is the parent fielding the child's faults (co-fiber), not a shared map.
    fn nested_view(&self, abs_base: u64, size_log2: u8) -> Mem {
        Mem {
            window: Window::sub(abs_base, size_log2, 1u64 << size_log2.min(63)),
            page: self.page,
            back: Arc::clone(&self.back),
            space: Arc::new(RwLock::new(AddrSpace::default())),
            has_regions: Arc::new(AtomicBool::new(false)),
            prot_dirty: Arc::new(AtomicBool::new(false)),
            last_fault: AtomicU64::new(NO_FAULT),
            writes: 0,
        }
    }

    /// One page's access state: `None` ⇒ faults (unmapped), `Some(writable)` ⇒ committed. A page
    /// absent from the map takes its region default — read+write in the initial prefix
    /// `[0, mapped)`, unmapped in the reserved tail (growth must be an explicit `map`).
    fn page_access(&self, prot: &BTreeMap<u64, PageProt>, page: u64) -> Option<bool> {
        match prot.get(&page) {
            Some(PageProt::Rw) => Some(true),
            Some(PageProt::Ro) => Some(false),
            Some(PageProt::Backed { writable, .. }) => Some(*writable),
            Some(PageProt::Unmapped) => None,
            None => (page * self.page < self.window.mapped()).then_some(true),
        }
    }

    /// Enforce the page state for a `width`-byte access at confined offset `base`: any access to an
    /// unmapped page, or a store to a read-only page, faults (§4/§5). Fast-pathed when the access
    /// lies wholly in the committed prefix and no page has been re-protected (the common case), so
    /// unprotected windows pay nothing.
    fn check_prot(&self, base: u64, width: u32, write: bool) -> Result<(), Trap> {
        // `base` is the *absolute* confined address; the page-map and `mapped` bound are
        // window-relative (`rel == base` for a top-level window; offset by the sub-window base for a
        // §14 child).
        let rel = base.wrapping_sub(self.window.base());
        let last = rel + width as u64 - 1;
        // Lock-free fast path: the address space has never been mutated, so every page is in its
        // region default — RW inside `[0, mapped)`. An in-prefix access is unconditionally fine and
        // skips the `RwLock` read entirely (the hot, overwhelmingly common case).
        if !self.prot_dirty.load(Ordering::Acquire) && last < self.window.mapped() {
            return Ok(());
        }
        let space = self.space_read();
        if space.prot.is_empty() && last < self.window.mapped() {
            return Ok(());
        }
        for page in (rel / self.page)..=(last / self.page) {
            match self.page_access(&space.prot, page) {
                // A **recoverable** in-window page fault: record the confined address so a §14
                // coroutine child can suspend to its parent (fault-driven yield) instead of trapping.
                None => return Err(self.page_fault(base)), // unmapped
                Some(false) if write => return Err(self.page_fault(base)), // read-only store
                _ => {}
            }
        }
        Ok(())
    }

    /// Record `base` as the pending recoverable page fault (for §14 fault-driven yield) and return the
    /// `MemoryFault` to propagate. A normal guest treats it as a trap (detect-and-kill); a coroutine
    /// child reads the recorded address and suspends to its parent instead.
    fn page_fault(&self, base: u64) -> Trap {
        self.last_fault.store(base, Ordering::Relaxed);
        Trap::MemoryFault
    }

    /// Take the pending recoverable page-fault address (set by [`check_prot`], cleared by
    /// [`confine_checked`]), clearing it. `None` if the last `MemoryFault` was an out-of-window fault
    /// (a real trap), not a recoverable page fault.
    fn take_fault(&self) -> Option<u64> {
        match self.last_fault.swap(NO_FAULT, Ordering::Relaxed) {
            NO_FAULT => None,
            addr => Some(addr),
        }
    }

    /// Mark **every** page of this window unmapped — demand-paging it (§14 lazy paging): a coroutine
    /// child started this way faults on first access of each page, suspending to the parent, which
    /// supplies the page and resumes. The parent virtualizes the whole sub-window.
    fn demand_page(&self) {
        // `div_ceil` so a child smaller than one host page (e.g. a 4 KiB sub-window on a 16 KiB-page
        // host) still gets its single covering page marked — confinement keeps its accesses in-window.
        let pages = self.window.reserved().div_ceil(self.page).max(1);
        let mut space = self.space_write();
        for p in 0..pages {
            space.prot.insert(p, PageProt::Unmapped);
        }
    }

    /// Supply the page containing the confined `abs_addr` (§14 lazy paging): mark it read-write
    /// **without zeroing**, so the bytes the parent placed in the shared backing survive — the
    /// faulting access then re-executes and reads them. Used by `resume` after a fault-driven yield.
    fn supply_page(&self, abs_addr: u64) {
        let page = abs_addr.wrapping_sub(self.window.base()) / self.page;
        self.space_write().prot.insert(page, PageProt::Rw);
    }

    /// **Trap-confinement** (§4): bounds-check the `width`-byte access and reject any that leaves the
    /// window — there is no masking. An access is admitted iff `[addr+offset, addr+offset+width)` lies
    /// within `[0, reserved)` (the reserved domain, which the tail-committed-by-`grow` case can reach);
    /// per-page committed-ness — including the reserved tail's unmapped default and the mapped prefix —
    /// is then enforced by [`Mem::check_prot`], so an in-reserved-but-uncommitted access still faults,
    /// matching the JIT's bounds-trap + `PROT_NONE` page tables. The arithmetic is `checked` (matching
    /// [`Window::checked`]) so a wild `addr+offset` that would overflow `u64` faults rather than
    /// wrapping into a valid offset.
    fn confine_checked(&self, addr: u64, offset: u64, width: u32) -> Result<u64, Trap> {
        // `confine` returns the **absolute** address `base + (addr+offset)` (`base == 0` for a
        // top-level window, the sub-window base for a §14 child); the guest `addr`/`offset` are
        // window-relative, so the bound is on `addr+offset` directly. The returned absolute address
        // indexes the (possibly parent-sized) backing.
        self.last_fault.store(NO_FAULT, Ordering::Relaxed); // fresh: clear any prior page fault
        match addr
            .checked_add(offset)
            .and_then(|e| e.checked_add(width as u64))
        {
            // An out-of-window fault is a real trap (not a recoverable page fault) — leave `last_fault`
            // cleared so `take_fault` returns `None` and the coroutine path propagates the trap.
            Some(end) if end <= self.window.reserved() => Ok(self.window.confine(addr, offset)),
            _ => Err(Trap::MemoryFault),
        }
    }

    /// Read `len` raw bytes from confined window address `addr` (DEBUGGING.md W2 inspection). Bounds
    /// it through the same `confine_checked` a guest load uses, so a read past the reserved window
    /// faults instead of escaping; uncommitted in-window pages read as zero (demand-zeroed backing).
    fn read_window(&self, addr: u64, len: usize) -> Result<Vec<u8>, Trap> {
        if len == 0 {
            return Ok(Vec::new());
        }
        let abs = self.confine_checked(addr, 0, len as u32)?;
        let mut out = vec![0u8; len];
        self.back.read_into(abs, &mut out);
        Ok(out)
    }

    fn load(&self, addr: u64, offset: u64, op: LoadOp) -> Result<Value, Trap> {
        let (_, rty, width, signed) = op.info();
        let base = self.confine_checked(addr, offset, width)?;
        self.check_prot(base, width, false)?;
        let raw = self.read_le(base, width);
        Ok(decode_loaded(rty, width, signed, raw))
    }

    fn store(&mut self, addr: u64, offset: u64, op: StoreOp, v: Value) -> Result<(), Trap> {
        let (_, _, width) = op.info();
        let base = self.confine_checked(addr, offset, width)?;
        self.check_prot(base, width, true)?;
        // `write_le` keeps only the low `width` bytes, so narrow stores truncate.
        self.write_le(base, width, store_bits(v));
        self.writes += 1;
        Ok(())
    }

    /// Bounds-check a whole `[addr, addr+len)` span against the reserved domain `[0, reserved)` — the
    /// `len: u64` bulk analogue of [`Self::confine_checked`], a *single* range check for the entire
    /// span (bulk-memory ops, D62). Returns the absolute base. Callers early-out on `len == 0` before
    /// calling, so this always sees `len >= 1` (matching the JIT's `len != 0`-guarded check).
    fn confine_span(&self, addr: u64, len: u64) -> Result<u64, Trap> {
        self.last_fault.store(NO_FAULT, Ordering::Relaxed);
        match addr.checked_add(len) {
            Some(end) if end <= self.window.reserved() => Ok(self.window.confine(addr, 0)),
            _ => Err(Trap::MemoryFault),
        }
    }

    /// Per-page committed-ness / RO enforcement over a whole span — the `len: u64` analogue of
    /// [`Self::check_prot`]. `len >= 1` (see [`Self::confine_span`]).
    fn check_prot_span(&self, base: u64, len: u64, write: bool) -> Result<(), Trap> {
        let rel = base.wrapping_sub(self.window.base());
        let last = rel + (len - 1);
        if !self.prot_dirty.load(Ordering::Acquire) && last < self.window.mapped() {
            return Ok(());
        }
        let space = self.space_read();
        if space.prot.is_empty() && last < self.window.mapped() {
            return Ok(());
        }
        for page in (rel / self.page)..=(last / self.page) {
            match self.page_access(&space.prot, page) {
                None => return Err(self.page_fault(base)),
                Some(false) if write => return Err(self.page_fault(base)),
                _ => {}
            }
        }
        Ok(())
    }

    /// Overlap-safe byte copy of a confined span (source snapshotted before any write). The interpreter
    /// is the differential oracle, not the fast path, so a snapshot `Vec` keeps `MemCopy`/`MemMove`
    /// byte-identical (both correct even under overlap) without direction-analysis subtlety.
    fn copy_span(&self, dbase: u64, sbase: u64, len: u64) {
        let n = len as usize;
        if !self.has_regions.load(Ordering::Relaxed) {
            let mut buf = vec![0u8; n];
            self.back.read_into(sbase, &mut buf);
            for (k, b) in buf.iter().enumerate() {
                self.back.set_byte(dbase + k as u64, *b);
            }
        } else {
            let buf: Vec<u8> = (0..len).map(|k| self.byte(sbase + k)).collect();
            for (k, b) in buf.iter().enumerate() {
                self.set_byte(dbase + k as u64, *b);
            }
        }
    }

    /// `Inst::MemCopy`/`MemMove` — copy `len` bytes `src`→`dst`, each span confined as a whole to
    /// `[0, reserved)` (fault before any write if either span escapes). Overlap-safe (so it serves both
    /// the non-overlapping `memcpy` and the overlapping `memmove`).
    fn mem_copy(&mut self, dst: u64, src: u64, len: u64) -> Result<(), Trap> {
        if len == 0 {
            return Ok(());
        }
        let sbase = self.confine_span(src, len)?;
        let dbase = self.confine_span(dst, len)?;
        self.check_prot_span(sbase, len, false)?;
        self.check_prot_span(dbase, len, true)?;
        self.copy_span(dbase, sbase, len);
        self.writes += 1;
        Ok(())
    }

    /// `Inst::MemFill` — set `len` bytes at `dst` to `val`, the span confined as a whole to
    /// `[0, reserved)` (fault before any write if it escapes).
    fn mem_fill(&mut self, dst: u64, val: u8, len: u64) -> Result<(), Trap> {
        if len == 0 {
            return Ok(());
        }
        let base = self.confine_span(dst, len)?;
        self.check_prot_span(base, len, true)?;
        if !self.has_regions.load(Ordering::Relaxed) {
            for k in 0..len {
                self.back.set_byte(base + k, val);
            }
        } else {
            for k in 0..len {
                self.set_byte(base + k, val);
            }
        }
        self.writes += 1;
        Ok(())
    }

    /// Bytecode-engine fast path for `MemCopy`/`MemMove` (Phase 2). Same whole-span confinement +
    /// per-page protection scan as the oracle [`Mem::mem_copy`], but the copy body is one bulk
    /// [`Region::copy_within`] (a `memmove`) instead of the oracle's scalar snapshot loop — the same
    /// single-threaded cooperative contract the bytecode engine already relies on for
    /// [`Mem::load_scalar`]/`read_word`. The `§13`-regions path keeps the byte-at-a-time
    /// [`Mem::copy_span`] (region redirection is not a `back`-contiguous span). The tree-walk oracle
    /// stays on scalar `mem_copy`, so it remains the independent reference the JIT + this path are
    /// differentially checked against.
    fn mem_copy_fast(&mut self, dst: u64, src: u64, len: u64) -> Result<(), Trap> {
        if len == 0 {
            return Ok(());
        }
        let sbase = self.confine_span(src, len)?;
        let dbase = self.confine_span(dst, len)?;
        self.check_prot_span(sbase, len, false)?;
        self.check_prot_span(dbase, len, true)?;
        if !self.has_regions.load(Ordering::Relaxed) {
            self.back.copy_within(dbase, sbase, len);
        } else {
            self.copy_span(dbase, sbase, len);
        }
        self.writes += 1;
        Ok(())
    }

    /// Bytecode-engine fast path for `MemFill` — the [`Mem::mem_copy_fast`] counterpart: one bulk
    /// [`Region::fill`] (a `memset`) instead of the oracle's scalar byte loop.
    fn mem_fill_fast(&mut self, dst: u64, val: u8, len: u64) -> Result<(), Trap> {
        if len == 0 {
            return Ok(());
        }
        let base = self.confine_span(dst, len)?;
        self.check_prot_span(base, len, true)?;
        if !self.has_regions.load(Ordering::Relaxed) {
            self.back.fill(base, len, val);
        } else {
            for k in 0..len {
                self.set_byte(base + k, val);
            }
        }
        self.writes += 1;
        Ok(())
    }

    /// Bytecode-engine scalar load (Phase 2): the slot-returning fast path. When no protection has
    /// ever been mutated (`!prot_dirty` — the common case: no syscalls / coroutines / §13 regions, so
    /// every prefix page is plain committed RW and `!prot_dirty ⟹ !has_regions`) and the access lies
    /// wholly in the backed prefix ([`Window::checked`]), read straight through — skipping the per-op
    /// `last_fault` atomic store in `confine_checked` and the `check_prot` page scan. **Semantically
    /// identical** to the cold [`Mem::load`] for this case (the escape-oracle byte-compares it); any
    /// other case (RO/unmapped/reserved-tail/regions, or a recoverable demand fault) falls to the cold
    /// path, which keeps the exact trap + `last_fault` semantics. Returns the [`Reg`] slot directly,
    /// dropping the `Value` round-trip the bytecode engine paid via `Reg::from_value(load(..))`.
    #[inline]
    fn load_scalar(&self, addr: u64, offset: u64, op: LoadOp) -> Result<Reg, Trap> {
        let (_, rty, width, signed) = op.info();
        if !self.prot_dirty.load(Ordering::Acquire) {
            if let Some(abs) = self.window.checked(addr, offset, width) {
                // `!prot_dirty ⟹ !has_regions`, so the bytes live in `back` (no §13 redirect); the
                // cooperative bytecode engine is the sole accessor, so a single non-atomic word read
                // is sound (and one instruction, not `width` atomic byte loads).
                let raw = self.back.read_word(abs, width);
                return Ok(Reg::from_value(decode_loaded(rty, width, signed, raw)));
            }
        }
        Ok(Reg::from_value(self.load(addr, offset, op)?))
    }

    /// Bytecode-engine scalar store (Phase 2): the fast path of [`Mem::store`], taking the value as
    /// raw slot `lo` bits (no `Value` wrap). Same fast-path gate + cold fallback as
    /// [`Mem::load_scalar`]; `write_le` truncates to the op width.
    #[inline]
    fn store_scalar(&mut self, addr: u64, offset: u64, op: StoreOp, lo: u64) -> Result<(), Trap> {
        let (_, _, width) = op.info();
        if !self.prot_dirty.load(Ordering::Acquire) {
            if let Some(abs) = self.window.checked(addr, offset, width) {
                self.back.write_word(abs, width, lo);
                self.writes += 1;
                return Ok(());
            }
        }
        self.store(addr, offset, op, Value::I64(lo as i64))
    }

    /// §17 `v128.load`: the **16-byte** bounds-checked access — the sole escape-TCB delta SIMD adds
    /// (D58). Shares the exact confinement + page-protection path as the scalar `load`, just
    /// with `width = 16`, so `svm-mask`'s width-parametric guard covers it unchanged.
    fn load_v128(&self, addr: u64, offset: u64) -> Result<Value, Trap> {
        let base = self.confine_checked(addr, offset, 16)?;
        self.check_prot(base, 16, false)?;
        let mut b = [0u8; 16];
        for (k, slot) in b.iter_mut().enumerate() {
            *slot = self.byte(base + k as u64);
        }
        Ok(Value::V128(b))
    }

    /// §17 `v128.store`: the 16-byte bounds-checked write (see [`Mem::load_v128`]).
    fn store_v128(&mut self, addr: u64, offset: u64, b: [u8; 16]) -> Result<(), Trap> {
        let base = self.confine_checked(addr, offset, 16)?;
        self.check_prot(base, 16, true)?;
        for (k, byte) in b.iter().enumerate() {
            self.set_byte(base + k as u64, *byte);
        }
        self.writes += 1;
        Ok(())
    }

    /// §12 atomics share the confinement + page-protection path with `load`/`store`, and add a
    /// **natural-alignment** requirement: a misaligned effective address traps (`MemoryFault`). The
    /// window base and reserved domain are width-aligned, so checking the confined address suffices.
    /// Single-threaded, an atomic's *value* semantics equal the non-atomic op; the JIT lowers these
    /// to hardware atomics so they stay correct once threads exist (§12). All operate on the full
    /// `ty` width (`i32`/`i64`).
    fn check_align(&self, base: u64, width: u32) -> Result<(), Trap> {
        if base.is_multiple_of(width as u64) {
            Ok(())
        } else {
            Err(Trap::MemoryFault)
        }
    }

    /// Whether `base`'s page is a §13 aliased (`Backed`) page. A naturally-aligned ≤8-byte atomic
    /// lies wholly within one host page, so the single page of `base` decides. Aliased pages keep the
    /// value-correct `read_le`/`write_le` path (their bytes live in an `Rc` region, not `back`);
    /// anonymous pages get `back`'s real hardware atomics (§12).
    fn is_backed(&self, base: u64) -> bool {
        self.has_regions.load(Ordering::Relaxed)
            && matches!(
                self.space_read()
                    .prot
                    .get(&(base.wrapping_sub(self.window.base()) / self.page)),
                Some(PageProt::Backed { .. })
            )
    }

    /// §9/§12 async-ring completion counter: confine + validate a 4-byte futex counter address (same
    /// gate as an `i32` atomic), require a normal anonymous page (a §13 alias's atomics route through
    /// `read_le`, not `back`, so an offload worker couldn't reach it consistently), and hand back the
    /// `Arc<Region>` + confined key for a worker to atomic-increment (matching `atomic_value`'s
    /// non-backed path) before it `notify`s.
    fn async_counter_impl(&self, counter_addr: u64) -> Option<Arc<dyn AsyncCounter>> {
        let base = self.confine_checked(counter_addr, 0, 4).ok()?;
        if !base.is_multiple_of(4) || self.is_backed(base) {
            return None;
        }
        self.check_prot(base, 4, true).ok()?;
        Some(Arc::new(RegionCounter {
            region: Arc::clone(&self.back),
            off: base,
        }))
    }

    /// Validate a `<ty>.atomic.wait` address: confine it, require natural alignment, and require the
    /// page be readable (`map`/`unmap`/`protect` state) — the same gate as a same-width atomic load.
    /// Returns the confined base (the parking-lot key). (§12 futex)
    fn prepare_wait(&self, addr: u64, ty: IntTy) -> Result<u64, Trap> {
        let width = atomic_width(ty);
        let base = self.confine_checked(addr, 0, width)?;
        self.check_align(base, width)?;
        self.check_prot(base, width, false)?;
        Ok(base)
    }

    /// The current `width`-byte value at confined `base` (no checks; `prepare_wait` ran them). Used
    /// for the futex compare under the parking lock — real atomic for anonymous pages, value-correct
    /// for §13 aliases.
    fn atomic_value(&self, base: u64, width: u32) -> u64 {
        if self.is_backed(base) {
            self.read_le(base, width)
        } else {
            self.back.atomic_load(base, width)
        }
    }

    /// Confine an `atomic.notify` address to its parking-lot key. `notify` reads no memory, so only
    /// bounds confinement applies (no alignment or protection check). (§12 futex)
    fn confine_for_notify(&self, addr: u64) -> Result<u64, Trap> {
        self.confine_checked(addr, 0, 1)
    }

    /// The canonical futex rendezvous key for a confined absolute address (PROCESS.md S1b). A §13
    /// `SharedRegion`-aliased (`Backed`) page keys on `(region, byte offset within the region)`, so two
    /// aliases of the same region byte — mapped at different window offsets, in the same or different
    /// domains — produce the **same** key and rendezvous. A normal anonymous page keys on its absolute
    /// address (never aliased across domains). This is the wait-queue/`notify` key; the value compare
    /// still uses the absolute address.
    fn futex_key(&self, base: u64) -> FutexKey {
        if self.has_regions.load(Ordering::Relaxed) {
            let rel = base.wrapping_sub(self.window.base());
            let space = self.space_read();
            if let Some(PageProt::Backed {
                region, region_off, ..
            }) = space.prot.get(&(rel / self.page))
            {
                // Cross-domain canonical identity (S1c residue): key on the backing
                // *allocation*, not the per-window region id — two domains that map the same
                // granted backing must produce the same key, or a concurrent pipe ring's
                // notify misses its peer. The fat `dyn` pointer's data address is the identity.
                let ident = space
                    .regions
                    .get(region)
                    .map(|b| Arc::as_ptr(b) as *const u8 as u64)
                    .unwrap_or(*region as u64);
                return FutexKey::Region(ident, region_off + rel % self.page);
            }
        }
        FutexKey::Anon(base)
    }

    fn atomic_load(&self, addr: u64, offset: u64, ty: IntTy) -> Result<Value, Trap> {
        let width = atomic_width(ty);
        let base = self.confine_checked(addr, offset, width)?;
        self.check_align(base, width)?;
        self.check_prot(base, width, false)?;
        let raw = if self.is_backed(base) {
            self.read_le(base, width)
        } else {
            self.back.atomic_load(base, width)
        };
        Ok(atomic_decode(ty, raw))
    }

    fn atomic_store(&mut self, addr: u64, offset: u64, ty: IntTy, v: Value) -> Result<(), Trap> {
        let width = atomic_width(ty);
        let base = self.confine_checked(addr, offset, width)?;
        self.check_align(base, width)?;
        self.check_prot(base, width, true)?;
        if self.is_backed(base) {
            self.write_le(base, width, store_bits(v));
        } else {
            self.back.atomic_store(base, width, store_bits(v));
        }
        self.writes += 1;
        Ok(())
    }

    /// Read the old value, apply `op` with `v`, write the result back, return the **old** value.
    fn atomic_rmw(
        &mut self,
        addr: u64,
        offset: u64,
        ty: IntTy,
        op: AtomicRmwOp,
        v: Value,
    ) -> Result<Value, Trap> {
        let width = atomic_width(ty);
        let base = self.confine_checked(addr, offset, width)?;
        self.check_align(base, width)?;
        self.check_prot(base, width, true)?;
        let old = if self.is_backed(base) {
            let old = self.read_le(base, width);
            self.write_le(base, width, atomic_rmw_apply(ty, op, old, store_bits(v)));
            old
        } else {
            self.back.atomic_rmw(base, width, rmw_op(op), store_bits(v))
        };
        self.writes += 1;
        Ok(atomic_decode(ty, old))
    }

    /// If `*addr == expected`, write `replacement`; always return the **old** value.
    fn atomic_cmpxchg(
        &mut self,
        addr: u64,
        offset: u64,
        ty: IntTy,
        expected: Value,
        replacement: Value,
    ) -> Result<Value, Trap> {
        let width = atomic_width(ty);
        let base = self.confine_checked(addr, offset, width)?;
        self.check_align(base, width)?;
        self.check_prot(base, width, true)?;
        let want = store_bits(expected) & width_mask(width);
        let old = if self.is_backed(base) {
            let old = self.read_le(base, width); // already the low `width` bytes, zero-extended
            if old == want {
                self.write_le(base, width, store_bits(replacement));
            }
            old
        } else {
            self.back
                .atomic_cmpxchg(base, width, store_bits(expected), store_bits(replacement))
        };
        // Count a write only when the compare succeeded (a failed cmpxchg leaves memory unchanged —
        // the distinction the spin detector needs to tell a spinning retry from a real acquire).
        if old == want {
            self.writes += 1;
        }
        Ok(atomic_decode(ty, old))
    }

    /// Validate a `map`/`unmap`/`protect` range (§3e): the offset must be page-aligned and the
    /// whole `[offset, offset+len)` must lie within the **reserved** window `[0, reserved)` — the
    /// guest may now grow into the reserved tail `[mapped, reserved)`, not just the initial backed
    /// prefix. Returns the inclusive page-index range it covers, or `Err(EINVAL)`.
    fn prot_pages(&self, offset: u64, len: u64) -> Result<core::ops::RangeInclusive<u64>, i64> {
        // `offset` is **guest-relative** — the zero-based window the guest sees (the whole `GuestMem`
        // surface speaks guest coordinates; a §14 child names its own `[0, size)`, never its position
        // in the parent). The prot map is keyed by the same relative pages
        // (`check_prot`/`page_access`); only backing-store accesses add `window.base()`.
        if len == 0 || !offset.is_multiple_of(self.page) {
            return Err(EINVAL);
        }
        let end = offset.checked_add(len).ok_or(EINVAL)?;
        if end > self.window.reserved() {
            return Err(EINVAL);
        }
        Ok((offset / self.page)..=((end - 1) / self.page)) // len need not be a page multiple; round up
    }

    /// Set one page's protection from cap `prot` bits: `WRITE` ⇒ read+write, `READ` only ⇒
    /// read-only, neither ⇒ unmapped. A read-write page in the initial prefix is left *absent*
    /// (its default); in the reserved tail it needs an explicit [`PageProt::Rw`] entry, since
    /// *absent* there means unmapped.
    /// Apply a `map`/`protect` protection to one page in the given prot map (the caller holds the
    /// address-space write lock). Uses `self`'s immutable `window`/`page` only.
    fn set_prot(&self, prot: &mut BTreeMap<u64, PageProt>, page: u64, flags: i32) {
        if flags & PROT_WRITE != 0 {
            if page * self.page < self.window.mapped() {
                prot.remove(&page); // read+write is the prefix default (no entry)
            } else {
                prot.insert(page, PageProt::Rw); // explicit commit in the reserved tail
            }
        } else if flags & PROT_READ != 0 {
            prot.insert(page, PageProt::Ro);
        } else {
            prot.insert(page, PageProt::Unmapped);
        }
    }

    /// Place initialized data segments at instantiation (§3a / D40): write every segment's bytes,
    /// then mark the pages of each `readonly` segment read-only (so the init writes themselves
    /// don't fault). RO protection is page-granular, so a producer keeps RO data on its own pages
    /// (the verifier already bounds each segment to `[0, size)`).
    fn init_data(&mut self, data: &[Data]) {
        self.init_data_at(data, 0);
    }

    /// Like [`init_data`], but place each (child-relative) segment at `win_base + offset` — the §14
    /// sub-window's slice base, so segment bytes and their read-only protections land in the child's
    /// region of the parent backing (matching trap-confinement, which confines child accesses to
    /// `[win_base, win_base + size)`). `win_base == 0` is the ordinary top-level window.
    fn init_data_at(&mut self, data: &[Data], win_base: u64) {
        // Byte writes first (no §13 regions exist at init ⇒ `set_byte` is lock-free)...
        for d in data {
            for (i, &b) in d.bytes.iter().enumerate() {
                self.set_byte(win_base + d.offset + i as u64, b);
            }
        }
        // ...then the read-only protections, under one address-space write lock. The prot map is
        // keyed by window-relative page (trap-confinement confines accesses to this window, and
        // `check_prot` looks up relative pages), so fold the window base out of the absolute address.
        let mut space = self.space_write();
        for d in data {
            if d.readonly && !d.bytes.is_empty() {
                let first = (win_base + d.offset).wrapping_sub(self.window.base());
                let last = first + d.bytes.len() as u64 - 1;
                for page in (first / self.page)..=(last / self.page) {
                    space.prot.insert(page, PageProt::Ro);
                }
            }
        }
    }

    /// Every page touched by `[ptr, ptr+len)` is committed (and writable, when `write`), and the
    /// range stays within `[0, reserved)`. The §7 borrow check: a buffer straddling an unmapped or
    /// (for writes) read-only page is rejected (`-EFAULT`), and grown tail pages are accepted.
    fn range_committed(&self, ptr: u64, len: u64, write: bool) -> bool {
        let Some(end) = ptr.checked_add(len) else {
            return false;
        };
        if end > self.window.reserved() {
            return false;
        }
        if len == 0 {
            return true;
        }
        let space = self.space_read();
        (ptr / self.page..=(end - 1) / self.page)
            .all(|page| matches!(self.page_access(&space.prot, page), Some(w) if w || !write))
    }

    /// Borrow-validate and read a `(ptr, len)` capability buffer (§7): every page of
    /// `[ptr, ptr+len)` must be committed. Returns the bytes, or `None` (→ `-EFAULT`).
    /// Confinement holds regardless; this explicit check is the recoverable guest-bug
    /// path, not a safety boundary.
    fn read_bytes_impl(&self, ptr: u64, len: u64) -> Option<Vec<u8>> {
        if !self.range_committed(ptr, len, false) {
            return None;
        }
        // `ptr` is guest-relative; `byte` indexes the (possibly parent-shared) backing absolutely.
        let base = self.window.base();
        Some((0..len).map(|k| self.byte(base + ptr + k)).collect())
    }

    /// Borrow-validate and write a `(ptr, len)` capability buffer (§7): every page must be
    /// committed and writable. `None` → `-EFAULT`.
    fn write_bytes_impl(&mut self, ptr: u64, data: &[u8]) -> Option<()> {
        if !self.range_committed(ptr, data.len() as u64, true) {
            return None;
        }
        let base = self.window.base();
        for (k, b) in data.iter().enumerate() {
            self.set_byte(base + ptr + k as u64, *b);
        }
        Some(())
    }

    /// The durable **active shadow-SP** word at [`SHADOW_SP_OFF`] (the running context's
    /// shadow-stack pointer). Read/written by the runtime on a fiber switch to keep it pointing at
    /// the current context's region (D-fiber-cont option A). Falls back to [`SHADOW_BASE`] if the
    /// word's page is somehow uncommitted (a malformed durable window) — `set` then no-ops.
    fn durable_get_sp(&self, sp_word: u64) -> u64 {
        self.read_bytes_impl(sp_word, 8)
            .and_then(|b| b.try_into().ok())
            .map(u64::from_le_bytes)
            .unwrap_or(SHADOW_BASE)
    }

    fn durable_set_sp(&mut self, sp_word: u64, sp: u64) {
        let _ = self.write_bytes_impl(sp_word, &sp.to_le_bytes());
    }

    /// The durable state word at [`STATE_OFF`] (the freeze driver's trigger). `STATE_NORMAL` if the
    /// word's page is somehow uncommitted (a non-durable / malformed window).
    fn durable_state(&self) -> i32 {
        self.read_bytes_impl(STATE_OFF, 4)
            .and_then(|b| b.try_into().ok())
            .map(i32::from_le_bytes)
            .unwrap_or(STATE_NORMAL)
    }

    /// Set the global durable **freeze** word at [`STATE_OFF`] (`UNWINDING`/`ARMED`/`NORMAL`) — the
    /// stop-the-world trigger every poll reads.
    fn durable_set_state(&mut self, state: i32) {
        let _ = self.write_bytes_impl(STATE_OFF, &state.to_le_bytes());
    }

    /// Read context `ctx`'s per-context **thaw** state word (`REWINDING`/`NORMAL`) at
    /// [`thaw_state_off`] (§12.8 concurrent-thaw stage 1). `STATE_NORMAL` if its page is uncommitted.
    fn durable_thaw_state(&self, ctx: usize) -> i32 {
        self.read_bytes_impl(thaw_state_off(ctx), 4)
            .and_then(|b| b.try_into().ok())
            .map(i32::from_le_bytes)
            .unwrap_or(STATE_NORMAL)
    }

    /// Set context `ctx`'s per-context **thaw** state word.
    fn durable_set_thaw_state(&mut self, ctx: usize, state: i32) {
        let _ = self.write_bytes_impl(thaw_state_off(ctx), &state.to_le_bytes());
    }

    /// Load a vCPU's unified durable phase from the two words it is split across (§12.8 concurrent-thaw
    /// stage 1): the global **freeze** word ([`STATE_OFF`]: `UNWINDING`/`ARMED`) takes precedence; else
    /// context `ctx`'s per-context **thaw** word (`REWINDING`/`NORMAL`). Mirror of [`Self::durable_store_dstate`].
    fn durable_load_dstate(&self, ctx: usize) -> i32 {
        let g = self.durable_state();
        if g != STATE_NORMAL {
            g
        } else {
            self.durable_thaw_state(ctx)
        }
    }

    /// Store a vCPU's unified durable phase, routing `REWINDING` to context `ctx`'s own **thaw** word and
    /// the freeze phases (`UNWINDING`/`ARMED`/`NORMAL`) to the global **freeze** word — so a rewinding
    /// vCPU flipping its own word to `NORMAL` can't disturb a sibling still `REWINDING` (the relocation
    /// the JIT needs for concurrent rewinds; the interp swaps it per dispatch, slice 3.2.1).
    fn durable_store_dstate(&mut self, ctx: usize, dstate: i32) {
        if dstate == STATE_REWINDING {
            self.durable_set_state(STATE_NORMAL);
            self.durable_set_thaw_state(ctx, STATE_REWINDING);
        } else {
            self.durable_set_state(dstate);
            self.durable_set_thaw_state(ctx, STATE_NORMAL);
        }
    }

    /// Tick the **mid-run freeze trigger** at a fiber safepoint (`cont.resume`/`suspend`): if the run
    /// is `STATE_ARMED`, decrement the arm countdown at [`ARM_COUNTDOWN_OFF`] and, when it reaches 0,
    /// promote the state word to `UNWINDING` — so the safepoint's trailing poll begins the freeze. A
    /// no-op unless armed (the common case: one `i32` read per safepoint, no write), so an unarmed run
    /// is byte-identical. Call once per fiber safepoint — see the `run_inner` dispatch.
    fn durable_tick_arm(&mut self) {
        self.durable_tick_countdown(ARM_COUNTDOWN_OFF);
    }

    /// Back-edge variant (Phase-4 Slice A): on an `ARMED` run, count down [`ARM_BACKEDGE_OFF`] at
    /// each branch terminator and promote to `UNWINDING` at 0, so the next loop-header poll begins
    /// the freeze — reaching a poll-free compute loop. Inert unless armed for back-edges (the slot
    /// is positive), so a fiber-armed or ordinary run is byte-identical. Call at branch terminators.
    fn durable_tick_arm_backedge(&mut self) {
        self.durable_tick_countdown(ARM_BACKEDGE_OFF);
    }

    /// Decrement the countdown at `off` on an `ARMED` run and promote to `UNWINDING` at 0. Guarded
    /// on the slot being **positive**, so the two countdowns (fiber-safepoint / back-edge) never
    /// interfere: arming one leaves the other at 0, where this is a no-op. An unarmed run is one
    /// `i32` state read, no write.
    fn durable_tick_countdown(&mut self, off: u64) {
        if self.durable_state() != STATE_ARMED {
            return;
        }
        let cur = self
            .read_bytes_impl(off, 8)
            .and_then(|b| b.try_into().ok())
            .map(i64::from_le_bytes)
            .unwrap_or(0);
        if cur <= 0 {
            return; // not armed for this countdown
        }
        let n = cur - 1;
        let _ = self.write_bytes_impl(off, &n.to_le_bytes());
        if n <= 0 {
            self.durable_set_state(STATE_UNWINDING);
        }
    }

    fn read_le(&self, base: u64, width: u32) -> u64 {
        // Hoist the `has_regions` check out of the per-byte loop: when no §13 region is aliased in
        // (the common case) read straight from `back`, one `has_regions` load instead of `width`.
        // Still per-byte (each `Region::byte` is a relaxed atomic — defined under §12 races).
        let mut raw = 0u64;
        if !self.has_regions.load(Ordering::Relaxed) {
            for k in 0..width as u64 {
                raw |= (self.back.byte(base + k) as u64) << (8 * k);
            }
        } else {
            for k in 0..width as u64 {
                raw |= (self.byte(base + k) as u64) << (8 * k);
            }
        }
        raw
    }

    fn write_le(&mut self, base: u64, width: u32, raw: u64) {
        if !self.has_regions.load(Ordering::Relaxed) {
            for k in 0..width as u64 {
                self.back.set_byte(base + k, (raw >> (8 * k)) as u8);
            }
        } else {
            for k in 0..width as u64 {
                self.set_byte(base + k, (raw >> (8 * k)) as u8);
            }
        }
    }

    /// Read one byte; unwritten anonymous pages read as zero. A [`PageProt::Backed`] page redirects
    /// to its §13 region buffer (so an aliased page reads whatever the shared backing holds).
    fn byte(&self, off: u64) -> u8 {
        // Fast path: no §13 region is mapped, so no page can be `Backed` — go straight to `back`
        // without touching the address-space lock (the hot, overwhelmingly common case).
        if !self.has_regions.load(Ordering::Relaxed) {
            return self.back.byte(off);
        }
        let idx = (off % self.page) as usize;
        let space = self.space_read();
        // The prot map is keyed by window-relative page (base folds out; the within-page `idx` is
        // unchanged since the base is page-aligned).
        if let Some(PageProt::Backed {
            region, region_off, ..
        }) = space
            .prot
            .get(&(off.wrapping_sub(self.window.base()) / self.page))
        {
            return space
                .regions
                .get(region)
                .map_or(0, |r| r.read_byte(*region_off + idx as u64));
        }
        self.back.byte(off)
    }

    fn set_byte(&self, off: u64, b: u8) {
        if !self.has_regions.load(Ordering::Relaxed) {
            self.back.set_byte(off, b);
            return;
        }
        let idx = (off % self.page) as usize;
        let space = self.space_read();
        if let Some(PageProt::Backed {
            region, region_off, ..
        }) = space
            .prot
            .get(&(off.wrapping_sub(self.window.base()) / self.page))
        {
            // §13 aliased page: write through to the shared region backing.
            if let Some(r) = space.regions.get(region) {
                r.write_byte(*region_off + idx as u64, b);
            }
            return;
        }
        self.back.set_byte(off, b);
    }

    /// Seed the low bytes of the window from `init` (escape-oracle, §18). Bytes past the
    /// window size are ignored — confinement only concerns `[0, size)`.
    fn seed(&mut self, init: &[u8]) {
        let n = (init.len() as u64).min(self.window.mapped());
        for i in 0..n {
            self.set_byte(i, init[i as usize]);
        }
    }

    /// Snapshot the low `n` bytes of the window (clamped to the backed `mapped` extent).
    fn snapshot(&self, n: u64) -> Vec<u8> {
        let n = n.min(self.window.mapped());
        (0..n).map(|i| self.byte(i)).collect()
    }

    /// Whether the live memory state is **fully captured** by a [`window_snapshot`](Mem::window_snapshot)
    /// then [`seed`](Mem::seed) round-trip of the mapped prefix — the precondition for time-travel
    /// checkpointing the window (W1). True when nothing has changed *how* `[0, mapped)` is read back or
    /// extended it: no §13 region aliasing, and every explicit page-protection entry is a benign
    /// in-prefix `Rw` commit (the only kind demand-paging inserts). A `protect`ed (`Ro`), `unmap`ped,
    /// region-`Backed`, or **grown** (tail `Rw`) page is *not* reproduced by reseeding a fresh window,
    /// so it makes the run un-checkpointable (it falls back to replay-from-clock-0). Plain in-prefix
    /// writes are fine — they leave the prefix `Rw` (absent from the map) and snapshot/seed round-trips
    /// their bytes.
    fn snapshot_safe(&self) -> bool {
        if self.has_regions.load(Ordering::Relaxed) {
            return false;
        }
        if !self.prot_dirty.load(Ordering::Acquire) {
            return true; // never mutated ⇒ the prefix is plain Rw throughout
        }
        let mapped_pages = self.window.mapped() / self.page;
        let space = self.space.read().unwrap_or_else(|e| e.into_inner());
        space.regions.is_empty()
            && space
                .prot
                .iter()
                .all(|(&pg, p)| matches!(p, PageProt::Rw) && pg < mapped_pages)
    }

    /// The full mapped window, for a time-travel checkpoint (restored with [`seed`](Mem::seed)).
    fn window_snapshot(&self) -> Vec<u8> {
        self.snapshot(self.window.mapped())
    }

    /// Seed the **whole parent backing** of a §14 sub-window (parent-absolute bytes), so the
    /// escape-oracle starts with non-zero bytes *outside* the child's slice — a child write that
    /// escaped its `[base, base+size)` slice would then perturb a byte the snapshot catches.
    fn seed_parent(&self, init: &[u8]) {
        for (i, &b) in init.iter().enumerate() {
            self.set_byte(i as u64, b);
        }
    }

    /// Snapshot the **whole parent backing** `[0, parent_bytes)` of a §14 sub-window (paired with
    /// the JIT's `compile_and_run_capture_sub`, which returns the full parent window).
    fn snapshot_parent(&self, parent_bytes: u64) -> Vec<u8> {
        (0..parent_bytes).map(|i| self.byte(i)).collect()
    }

    /// Snapshot the low `min(reserved, max(mapped, snap_cap))` bytes for the escape-oracle —
    /// **including grown reserved-tail pages** (a page absent from the map reads zero, matching the
    /// JIT's freshly-committed tail). Page-wise (one map lookup per committed page, not per byte) so
    /// widening past the backed prefix stays cheap.
    fn snapshot_window(&self, snap_cap: usize) -> Vec<u8> {
        let snap = self
            .window
            .reserved()
            .min(self.window.mapped().max(snap_cap as u64)) as usize;
        let mut out = vec![0u8; snap];
        self.back.read_into(0, &mut out); // anonymous bytes (untouched / grown-tail read as zero)
                                          // §13 aliased pages live in their region backing, not in `back` — fill them from there.
        let space = self.space_read();
        for (&idx, p) in &space.prot {
            let PageProt::Backed {
                region, region_off, ..
            } = p
            else {
                continue;
            };
            let start = (idx * self.page) as usize;
            if start >= snap {
                continue;
            }
            let n = (self.page as usize).min(snap - start);
            if let Some(r) = space.regions.get(region) {
                for k in 0..n {
                    out[start + k] = r.read_byte(*region_off + k as u64);
                }
            }
        }
        out
    }

    /// Per-page protections over the same span as [`snapshot_window`], one [`CapturedProt`] per
    /// [`DURABLE_SNAPSHOT_PAGE`]-byte page (DURABILITY.md §12.3). A page absent from the map is
    /// `Rw` in the committed prefix and `Unmapped` in the reserved tail — the same default the
    /// access path and the JIT's page tables use.
    fn snapshot_prots(&self, snap_cap: usize) -> Vec<CapturedProt> {
        let snap = self
            .window
            .reserved()
            .min(self.window.mapped().max(snap_cap as u64));
        let mapped = self.window.mapped();
        let space = self.space_read();
        (0..snap / DURABLE_SNAPSHOT_PAGE)
            .map(|i| {
                let byte_off = i * DURABLE_SNAPSHOT_PAGE;
                match space.prot.get(&(byte_off / self.page)) {
                    Some(PageProt::Rw) => CapturedProt::Rw,
                    Some(PageProt::Ro) => CapturedProt::Ro,
                    Some(PageProt::Unmapped) => CapturedProt::Unmapped,
                    Some(PageProt::Backed { .. }) => CapturedProt::Backed,
                    None if byte_off < mapped => CapturedProt::Rw,
                    None => CapturedProt::Unmapped,
                }
            })
            .collect()
    }

    /// Re-establish a captured protection map on this window (the durable-restore step, the
    /// inverse of [`snapshot_prots`]): mark each `Ro`/`Unmapped` page so a thawed guest faults
    /// exactly as the frozen one would. `Rw` is the prefix default (left absent) but is set
    /// explicitly for a reserved-tail page (a grown commit); `Backed` is skipped — a §13
    /// shared-region alias isn't restorable here (D-region), the embedder re-grants the region.
    fn apply_prots(&mut self, prots: &[CapturedProt]) {
        let mapped = self.window.mapped();
        let mut space = self.space_write();
        for (i, &p) in prots.iter().enumerate() {
            let byte_off = i as u64 * DURABLE_SNAPSHOT_PAGE;
            let host_page = byte_off / self.page;
            match p {
                CapturedProt::Ro => {
                    space.prot.insert(host_page, PageProt::Ro);
                }
                CapturedProt::Unmapped => {
                    space.prot.insert(host_page, PageProt::Unmapped);
                }
                CapturedProt::Rw if byte_off >= mapped => {
                    space.prot.insert(host_page, PageProt::Rw);
                }
                CapturedProt::Rw | CapturedProt::Backed => {}
            }
        }
    }
}

impl GuestMem for Mem {
    fn read_bytes(&self, ptr: u64, len: u64) -> Option<Vec<u8>> {
        self.read_bytes_impl(ptr, len)
    }
    fn write_bytes(&mut self, ptr: u64, data: &[u8]) -> Option<()> {
        self.write_bytes_impl(ptr, data)
    }

    /// §3e op 0 `map`: (re)commit pages with `prot`, zero-filling them (a fresh commit). Works
    /// anywhere in the reserved window `[0, reserved)` — including **growth** into the reserved
    /// tail `[mapped, reserved)`, the §1a sparse-address-space capability. Out-of-range /
    /// misaligned → `-EINVAL`.
    fn map(&mut self, offset: u64, len: u64, prot: i32) -> i64 {
        let pages = match self.prot_pages(offset, len) {
            Ok(p) => p,
            Err(e) => return e,
        };
        {
            let mut space = self.space_write();
            for page in pages.clone() {
                self.set_prot(&mut space.prot, page, prot);
            }
        }
        for page in pages {
            self.back
                .zero(self.window.base() + page * self.page, self.page); // commit ⇒ fresh zeroed page
        }
        0
    }

    /// §3e op 1 `unmap`: decommit pages — any later access faults, and a re-`map` reads zero.
    fn unmap(&mut self, offset: u64, len: u64) -> i64 {
        let pages = match self.prot_pages(offset, len) {
            Ok(p) => p,
            Err(e) => return e,
        };
        {
            let mut space = self.space_write();
            for page in pages.clone() {
                space.prot.insert(page, PageProt::Unmapped);
            }
        }
        for page in pages {
            self.back
                .zero(self.window.base() + page * self.page, self.page);
        }
        0
    }

    /// §3e op 2 `protect`: change the protection of mapped pages without touching their backing
    /// (the D40 read-only const-segment mechanism: `protect(READ)` ⇒ later stores fault). A §13
    /// aliased page stays aliased — only its writability changes (or it `unmap`s if neither R nor W),
    /// so the shared bytes survive a `protect(READ)`.
    fn protect(&mut self, offset: u64, len: u64, prot: i32) -> i64 {
        let pages = match self.prot_pages(offset, len) {
            Ok(p) => p,
            Err(e) => return e,
        };
        let mut space = self.space_write();
        for page in pages {
            if let Some(PageProt::Backed {
                region, region_off, ..
            }) = space.prot.get(&page).copied()
            {
                if prot & (PROT_READ | PROT_WRITE) == 0 {
                    space.prot.insert(page, PageProt::Unmapped);
                } else {
                    space.prot.insert(
                        page,
                        PageProt::Backed {
                            region,
                            region_off,
                            writable: prot & PROT_WRITE != 0,
                        },
                    );
                }
            } else {
                self.set_prot(&mut space.prot, page, prot);
            }
        }
        0
    }

    /// §13 op 0 `map`: alias `backing`'s `[region_off, region_off+len)` into the window at
    /// `[win_off, win_off+len)`. Both window offsets and the region offset round to whole pages; the
    /// region span must fit the backing; the mapping must be at least readable. The aliased pages'
    /// bytes then live in the region (a prior anonymous page there is dropped), so a store at one
    /// alias is visible at every other mapping of the same region.
    fn map_region(
        &mut self,
        win_off: u64,
        region_off: u64,
        len: u64,
        prot: i32,
        region: u32,
        backing: RegionBacking,
    ) -> i64 {
        let pages: Vec<u64> = match self.prot_pages(win_off, len) {
            Ok(p) => p.collect(),
            Err(e) => return e,
        };
        if !region_off.is_multiple_of(self.page) || prot & PROT_READ == 0 {
            return EINVAL;
        }
        match region_off.checked_add(len) {
            Some(end) if end <= backing.size() => {}
            _ => return EINVAL,
        }
        let writable = prot & PROT_WRITE != 0;
        // A §13 alias now exists ⇒ the per-byte path must consult the address space from here on.
        self.has_regions.store(true, Ordering::Relaxed);
        {
            let mut space = self.space_write();
            space.regions.insert(region, backing);
            for (i, &page) in pages.iter().enumerate() {
                space.prot.insert(
                    page,
                    PageProt::Backed {
                        region,
                        region_off: region_off + i as u64 * self.page,
                        writable,
                    },
                );
            }
        }
        for &page in &pages {
            self.back
                .zero(self.window.base() + page * self.page, self.page); // bytes live in the region now, not anonymous
        }
        0
    }

    /// §3e op 3 `page_size`: the backing-store page granularity (`self.page`, the host page) — the
    /// unit `map`/`unmap`/`protect` round to. The JIT's `MprotectWindow` reports the same host page,
    /// so the two backends agree.
    fn page_size(&self) -> i64 {
        self.page as i64
    }
    fn async_counter(&self, counter_addr: u64) -> Option<Arc<dyn AsyncCounter>> {
        self.async_counter_impl(counter_addr)
    }
}

/// Turn `width` raw little-endian bytes into the loaded value, sign- or zero-
/// extending narrow integer loads into the i32/i64 result type.
fn decode_loaded(rty: ValType, width: u32, signed: bool, raw: u64) -> Value {
    match rty {
        ValType::F32 => Value::F32(f32::from_bits(raw as u32)),
        ValType::F64 => Value::F64(f64::from_bits(raw)),
        // `v128` never reaches here — its loads go through the dedicated 16-byte path, not
        // a `LoadOp` (whose widths are ≤8). Total arm for exhaustiveness only.
        ValType::V128 => Value::V128([0; 16]),
        ValType::I32 | ValType::I64 | ValType::Ref | ValType::Cap => {
            let bits = width * 8;
            let ext = if signed && bits < 64 {
                let shift = 64 - bits;
                (((raw << shift) as i64) >> shift) as u64 // arithmetic sign-extend
            } else {
                raw
            };
            match rty {
                ValType::I32 | ValType::Cap => Value::I32(ext as i32),
                ValType::Ref => Value::Ref(ext), // opaque, stored/loaded as an i64-width word
                _ => Value::I64(ext as i64),
            }
        }
    }
}

/// Scan a fiber's frames for candidate root words in `[lo, hi)` and insert them into `out` (§GC
/// `gc.roots`). Each word is first masked (`m = w & mask`); the **masked** value is what's range-
/// tested and inserted, so a guest with tagged pointers (tag in the top byte) recovers the bare
/// offset (`mask = !0` is the untagged case). Conservative: every SSA value whose masked bits land
/// in range is a candidate — the interpreter scans typed `Value`s (the JIT scans raw control-stack
/// words; both are sound over-approximations that may differ in false positives, GC.md §3.2).
/// `v128` contributes both of its 64-bit halves. `mask` is caller-validated to top-byte-strip only,
/// so a host pointer stays large and is excluded by the range test.
fn gc_scan_frames(
    frames: &[Frame],
    lo: u64,
    hi: u64,
    mask: u64,
    out: &mut std::collections::BTreeSet<u64>,
) {
    let mut consider = |w: u64| {
        let m = w & mask;
        if m >= lo && m < hi {
            out.insert(m);
        }
    };
    for f in frames {
        for v in &f.vals {
            // Untyped raw slots: scan both 64-bit words, like the JIT scans raw control-stack
            // words (a `v128` contributes both halves; a scalar's high half is 0, which the GC
            // heap range never contains, so `consider` filters it out). An `i32` slot is
            // sign-extended in `lo` — exactly the JIT slot ABI — so the two backends' candidate
            // words now align (still sound over-approximations, GC.md §3.2).
            consider(v.lo);
            consider(v.hi);
        }
    }
}

/// The low 64 bits of a value, for storing (the store width selects how many bytes).
fn store_bits(v: Value) -> u64 {
    match v {
        Value::I32(x) => x as u32 as u64,
        Value::I64(x) => x as u64,
        Value::F32(x) => x.to_bits() as u64,
        Value::F64(x) => x.to_bits(),
        // `v128` stores go through the dedicated 16-byte path; a scalar store never sees one.
        // Total arm: low 8 bytes (little-endian).
        Value::V128(b) => u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]),
        Value::Ref(x) => x, // opaque reference: its raw 64-bit word
    }
}

/// Access width in bytes of an atomic `ty` (§12) — also its natural-alignment requirement.
fn atomic_width(ty: IntTy) -> u32 {
    match ty {
        IntTy::I32 => 4,
        IntTy::I64 => 8,
    }
}

/// Map the IR's RMW op onto the memory substrate's (the substrate sits below `svm-ir`, so it carries
/// its own mirrored enum).
fn rmw_op(op: AtomicRmwOp) -> RmwOp {
    match op {
        AtomicRmwOp::Add => RmwOp::Add,
        AtomicRmwOp::Sub => RmwOp::Sub,
        AtomicRmwOp::And => RmwOp::And,
        AtomicRmwOp::Or => RmwOp::Or,
        AtomicRmwOp::Xor => RmwOp::Xor,
        AtomicRmwOp::Xchg => RmwOp::Xchg,
    }
}

/// Low-`width`-bytes mask (`width` ∈ {4, 8}).
fn width_mask(width: u32) -> u64 {
    if width >= 8 {
        u64::MAX
    } else {
        (1u64 << (width * 8)) - 1
    }
}

/// Decode the low `ty`-width bytes (zero-extended, as from [`Mem::read_le`]) into a [`Value`].
fn atomic_decode(ty: IntTy, raw: u64) -> Value {
    match ty {
        IntTy::I32 => Value::I32(raw as i32),
        IntTy::I64 => Value::I64(raw as i64),
    }
}

/// Apply an atomic RMW: `old`/`arg` are the low `ty`-width bytes; returns the new low-`width` value.
fn atomic_rmw_apply(ty: IntTy, op: AtomicRmwOp, old: u64, arg: u64) -> u64 {
    match ty {
        IntTy::I32 => {
            let (o, a) = (old as u32, arg as u32);
            let r = match op {
                AtomicRmwOp::Add => o.wrapping_add(a),
                AtomicRmwOp::Sub => o.wrapping_sub(a),
                AtomicRmwOp::And => o & a,
                AtomicRmwOp::Or => o | a,
                AtomicRmwOp::Xor => o ^ a,
                AtomicRmwOp::Xchg => a,
            };
            r as u64
        }
        IntTy::I64 => match op {
            AtomicRmwOp::Add => old.wrapping_add(arg),
            AtomicRmwOp::Sub => old.wrapping_sub(arg),
            AtomicRmwOp::And => old & arg,
            AtomicRmwOp::Or => old | arg,
            AtomicRmwOp::Xor => old ^ arg,
            AtomicRmwOp::Xchg => arg,
        },
    }
}

fn bin32(op: BinOp, a: i32, b: i32) -> Result<i32, Trap> {
    Ok(match op {
        BinOp::Add => a.wrapping_add(b),
        BinOp::Sub => a.wrapping_sub(b),
        BinOp::Mul => a.wrapping_mul(b),
        BinOp::DivS => {
            check_div(b == 0, a == i32::MIN && b == -1)?;
            a.wrapping_div(b)
        }
        BinOp::DivU => {
            check_div(b == 0, false)?;
            ((a as u32) / (b as u32)) as i32
        }
        BinOp::RemS => {
            // `rem_s` traps only on a zero divisor. `INT_MIN % -1 == 0` — a perfectly
            // representable result, so it does *not* trap: traps are for results with no
            // representable value (§3b), and only the *quotient* overflows here, not the
            // remainder. (`wrapping_rem` yields 0.) See `div_s`, which does trap.
            check_div(b == 0, false)?;
            a.wrapping_rem(b)
        }
        BinOp::RemU => {
            check_div(b == 0, false)?;
            ((a as u32) % (b as u32)) as i32
        }
        BinOp::And => a & b,
        BinOp::Or => a | b,
        BinOp::Xor => a ^ b,
        // Shift amount is taken mod bitwidth (`wrapping_sh*` masks rhs to 0..31).
        BinOp::Shl => a.wrapping_shl(b as u32),
        BinOp::ShrS => a.wrapping_shr(b as u32),
        BinOp::ShrU => ((a as u32).wrapping_shr(b as u32)) as i32,
        // Rotation amount is also mod bitwidth (`rotate_*` reduces it internally).
        BinOp::Rotl => a.rotate_left(b as u32),
        BinOp::Rotr => a.rotate_right(b as u32),
    })
}

fn intun32(op: IntUnOp, a: i32) -> i32 {
    match op {
        IntUnOp::Clz => (a as u32).leading_zeros() as i32,
        IntUnOp::Ctz => (a as u32).trailing_zeros() as i32,
        IntUnOp::Popcnt => (a as u32).count_ones() as i32,
        IntUnOp::Extend8S => (a as i8) as i32,
        IntUnOp::Extend16S => (a as i16) as i32,
        IntUnOp::Extend32S => a, // identity for i32
    }
}

fn intun64(op: IntUnOp, a: i64) -> i64 {
    match op {
        IntUnOp::Clz => (a as u64).leading_zeros() as i64,
        IntUnOp::Ctz => (a as u64).trailing_zeros() as i64,
        IntUnOp::Popcnt => (a as u64).count_ones() as i64,
        IntUnOp::Extend8S => (a as i8) as i64,
        IntUnOp::Extend16S => (a as i16) as i64,
        IntUnOp::Extend32S => (a as i32) as i64,
    }
}

fn bin64(op: BinOp, a: i64, b: i64) -> Result<i64, Trap> {
    Ok(match op {
        BinOp::Add => a.wrapping_add(b),
        BinOp::Sub => a.wrapping_sub(b),
        BinOp::Mul => a.wrapping_mul(b),
        BinOp::DivS => {
            check_div(b == 0, a == i64::MIN && b == -1)?;
            a.wrapping_div(b)
        }
        BinOp::DivU => {
            check_div(b == 0, false)?;
            ((a as u64) / (b as u64)) as i64
        }
        BinOp::RemS => {
            // Only a zero divisor traps; `INT_MIN % -1 == 0` is representable (only the
            // quotient overflows, not the remainder), so it returns 0 — see `bin32`.
            check_div(b == 0, false)?;
            a.wrapping_rem(b)
        }
        BinOp::RemU => {
            check_div(b == 0, false)?;
            ((a as u64) % (b as u64)) as i64
        }
        BinOp::And => a & b,
        BinOp::Or => a | b,
        BinOp::Xor => a ^ b,
        BinOp::Shl => a.wrapping_shl(b as u32),
        BinOp::ShrS => a.wrapping_shr(b as u32),
        BinOp::ShrU => ((a as u64).wrapping_shr(b as u32)) as i64,
        BinOp::Rotl => a.rotate_left(b as u32),
        BinOp::Rotr => a.rotate_right(b as u32),
    })
}

#[inline]
fn check_div(by_zero: bool, overflow: bool) -> Result<(), Trap> {
    if by_zero {
        Err(Trap::DivByZero)
    } else if overflow {
        Err(Trap::IntOverflow)
    } else {
        Ok(())
    }
}

fn cmp32(op: CmpOp, a: i32, b: i32) -> bool {
    match op {
        CmpOp::Eq => a == b,
        CmpOp::Ne => a != b,
        CmpOp::LtS => a < b,
        CmpOp::LtU => (a as u32) < (b as u32),
        CmpOp::LeS => a <= b,
        CmpOp::LeU => (a as u32) <= (b as u32),
        CmpOp::GtS => a > b,
        CmpOp::GtU => (a as u32) > (b as u32),
        CmpOp::GeS => a >= b,
        CmpOp::GeU => (a as u32) >= (b as u32),
    }
}

fn cmp64(op: CmpOp, a: i64, b: i64) -> bool {
    match op {
        CmpOp::Eq => a == b,
        CmpOp::Ne => a != b,
        CmpOp::LtS => a < b,
        CmpOp::LtU => (a as u64) < (b as u64),
        CmpOp::LeS => a <= b,
        CmpOp::LeU => (a as u64) <= (b as u64),
        CmpOp::GtS => a > b,
        CmpOp::GtU => (a as u64) > (b as u64),
        CmpOp::GeS => a >= b,
        CmpOp::GeU => (a as u64) >= (b as u64),
    }
}

#[inline]
fn step(fuel: &mut u64, kill: Option<&AtomicBool>) -> Result<(), Trap> {
    // PROCESS.md S3 `kill`: a §14 child (and its inherited-flag descendants) polls its parent-set
    // kill flag once per op. Set ⇒ the vCPU self-terminates (`ThreadFault`, `poll` → 2). `None` for
    // the root / top-level threads (a predictable branch, no atomic load) — the undebugged hot path.
    if let Some(k) = kill {
        if k.load(Ordering::Relaxed) {
            return Err(Trap::ThreadFault);
        }
    }
    *fuel = fuel.checked_sub(1).ok_or(Trap::OutOfFuel)?;
    Ok(())
}

#[inline]
fn get(vals: &[Reg], v: ValIdx) -> Result<Reg, Trap> {
    vals.get(v as usize).copied().ok_or(Trap::Malformed)
}

// Typed operand reads: pull the needed scalar out of the (untyped) value slot. The executing
// instruction's static type picks which getter to call, so the bit pattern is interpreted
// correctly without a per-value tag. `Malformed` on an out-of-range index, as before.
#[inline]
fn get_i32(vals: &[Reg], v: ValIdx) -> Result<i32, Trap> {
    Ok(get(vals, v)?.i32())
}

#[inline]
fn get_i64(vals: &[Reg], v: ValIdx) -> Result<i64, Trap> {
    Ok(get(vals, v)?.i64())
}

#[inline]
fn get_f32(vals: &[Reg], v: ValIdx) -> Result<f32, Trap> {
    Ok(get(vals, v)?.f32())
}

#[inline]
fn get_f64(vals: &[Reg], v: ValIdx) -> Result<f64, Trap> {
    Ok(get(vals, v)?.f64())
}

fn collect(vals: &[Reg], idxs: &[ValIdx]) -> Result<Vec<Reg>, Trap> {
    idxs.iter().map(|&v| get(vals, v)).collect()
}

/// Like [`collect`], but gather into a caller-owned buffer instead of a fresh `Vec`. The
/// hot-loop branch path (`Br`/`BrIf`/`BrTable`) fills a reusable scratch buffer and then
/// **swaps** it into the frame's value slot, so a taken edge allocates nothing in steady
/// state — the back-edge of a loop runs every iteration, where the per-branch `collect`
/// alloc/free dominated. `dst` is cleared first; on a bad index it returns `Malformed`
/// (leaving `dst` partially filled, which the caller discards as it traps), exactly as
/// `collect` would have.
fn collect_into(dst: &mut Vec<Reg>, vals: &[Reg], idxs: &[ValIdx]) -> Result<(), Trap> {
    dst.clear();
    dst.reserve(idxs.len());
    for &v in idxs {
        dst.push(get(vals, v)?);
    }
    Ok(())
}

#[inline]
/// Encode a value into its `i64` capability-ABI slot (scalars; `i32`/`f32` in the low
/// bits). Mirrors the JIT's marshalling so both drive the same slot-based dispatch.
fn val_to_slot(v: Value) -> i64 {
    match v {
        Value::I32(x) => x as i64,
        Value::I64(x) => x,
        Value::F32(x) => x.to_bits() as i64,
        Value::F64(x) => x.to_bits() as i64,
        // The cap ABI marshals scalars only; a `v128` arg/result is out of MVP scope (§17). Total
        // arm — its low 8 bytes — keeps the interpreter panic-free if a module declares one.
        Value::V128(b) => i64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]),
        Value::Ref(x) => x as i64, // opaque reference marshals as its i64-width word
    }
}

/// Decode a capability-ABI result slot back to a `Value` of the declared type.
fn slot_to_val(ty: ValType, s: i64) -> Value {
    match ty {
        ValType::I32 => Value::I32(s as i32),
        ValType::I64 => Value::I64(s),
        ValType::F32 => Value::F32(f32::from_bits(s as u32)),
        ValType::F64 => Value::F64(f64::from_bits(s as u64)),
        ValType::Ref => Value::Ref(s as u64), // opaque i64-width reference
        ValType::Cap => Value::I32(s as i32), // §3.5: i32-width handle marker
        // `v128` cap results are out of MVP scope; zero-extend the slot into the low lanes.
        ValType::V128 => {
            let mut b = [0u8; 16];
            b[..8].copy_from_slice(&s.to_le_bytes());
            Value::V128(b)
        }
    }
}

/// Whether a variable scoped to `var_func` is visible in a frame of function `frame_func`: its own
/// function, or a [`svm_ir::GLOBAL_SCOPE`] module-scoped global (visible in every frame).
fn var_in_scope(var_func: u32, frame_func: u32) -> bool {
    var_func == frame_func || var_func == svm_ir::GLOBAL_SCOPE
}

/// The source line at pc `(func, block, inst)` — the nearest-preceding `debug.loc` within the block
/// (the same semantics as [`Inspector::source_loc`]). `None` ⇒ no source mapping at this pc.
fn source_line_at(di: &DebugInfo, func: u32, block: usize, inst: usize) -> Option<u32> {
    di.locs
        .iter()
        .filter(|l| l.func == func && l.block as usize == block && l.inst as usize <= inst)
        .max_by_key(|l| l.inst)
        .map(|l| l.line)
}

/// Whether a variable's lexical `scope` covers source line `line`. A function-wide var (`None`)
/// always covers; a `Some((start, end))` covers `line ∈ [start, end]`. When the pc has no source
/// line (`line == None`), scopes can't be evaluated, so every var is treated as covering (the
/// function-wide back-compat behavior).
fn scope_covers(scope: Option<(u32, u32)>, line: Option<u32>) -> bool {
    match (scope, line) {
        (None, _) | (_, None) => true,
        (Some((s, e)), Some(l)) => s <= l && l <= e,
    }
}

/// Pick the source variable named `name` that is **innermost in scope** at the stopped pc, resolving
/// C shadowing (an inner-block redeclaration, or a local shadowing a global): among same-name
/// candidates visible in `frame_func`, keep those whose lexical scope covers the stopped source
/// line, and choose the most deeply nested (largest `start_line`; a function-wide `None` is the
/// outermost). Ties keep the first (declaration order).
fn pick_var<'a>(
    di: &'a DebugInfo,
    frame_func: u32,
    name: &str,
    block: usize,
    inst: usize,
) -> Option<&'a VarInfo> {
    let line = source_line_at(di, frame_func, block, inst);
    let depth = |x: &VarInfo| x.scope.map_or(0, |(s, _)| s);
    let mut best: Option<&VarInfo> = None;
    for x in di.vars.iter().filter(|x| {
        var_in_scope(x.func, frame_func) && x.name == name && scope_covers(x.scope, line)
    }) {
        if best.is_none_or(|b| depth(x) > depth(b)) {
            best = Some(x);
        }
    }
    best
}

/// Resolve a debug-info location list (`SsaList` / `WindowVia` base) at pc `(block, inst)`: the
/// covering entry is the largest `inst` at-or-before within the stopped block (nearest-preceding,
/// DWARF line-table semantics). `None` ⇒ the var isn't live at this pc.
fn loclist_value(locs: &[SsaLoc], block: usize, inst: usize) -> Option<u32> {
    locs.iter()
        .filter(|l| l.block as usize == block && l.inst as usize <= inst)
        .max_by_key(|l| l.inst)
        .map(|l| l.value)
}

#[cfg(test)]
mod region_minter_tests {
    //! The §4b `RegionMinter` ABI: an mmap-capable `HostFnRegion` handler is handed exactly one extra
    //! authority — minting a `SharedRegion` — and nothing else of the `Host`. These pin that the
    //! handler can mint a region and hand its handle back to the guest, and that the minted handle is
    //! a live `SharedRegion` (the delivery mechanism for the zero-copy file-mmap bridge).
    use super::*;

    #[test]
    fn host_fn_region_handler_mints_a_region_and_returns_a_live_handle() {
        let mut host = Host::new();
        // Op 0: mint a 64-byte region and hand back its handle — the shape an mmap-capable fs uses.
        let h = host.grant_host_fn_region(Box::new(|op, _args, _mem, minter| {
            if op == 0 {
                let backing: RegionBacking = Arc::new(VecBacking(Mutex::new(vec![7u8; 64])));
                Ok(vec![minter.grant_region(backing) as i64])
            } else {
                Ok(vec![-22])
            }
        }));
        assert!(
            h >= 0,
            "grant_host_fn_region should yield a handle under iface HOST_FN"
        );

        // Dispatch op 0 → the handler mints via the `RegionMinter` and returns the region handle.
        let out = host
            .cap_dispatch_slots(cap_id::HOST_FN, 0, h, &[], None)
            .expect("host_fn_region dispatch");
        let region_h = out[0];
        assert!(
            region_h >= 0,
            "the minted region handle must be valid: {region_h}"
        );

        // The returned handle really is a live `SharedRegion` (resolves under iface 4), and its
        // backing is the 64-byte object the handler minted.
        match host.resolve(region_h as i32, cap_id::SHARED_REGION) {
            Ok(Binding::SharedRegion(id)) => {
                assert_eq!(host.regions[id as usize].size(), 64, "minted region size");
            }
            other => panic!("minted handle should resolve as a SharedRegion, got {other:?}"),
        }
    }

    #[test]
    fn region_and_plain_host_fn_are_distinct_handles_under_the_same_iface() {
        let mut host = Host::new();
        let plain = host.grant_host_fn(Box::new(|_, _, _| Ok(vec![0])));
        let region = host.grant_host_fn_region(Box::new(|_, _, _, _| Ok(vec![0])));
        assert!(plain >= 0 && region >= 0 && plain != region);
    }
}

#[cfg(test)]
mod gen_recycle_tests {
    //! The handle-table generation counter is a full `u32`, but a packed handle only carries
    //! `GEN_BITS` of it. The compare must therefore mask, so a slot **recycles** when the counter
    //! wraps rather than dying permanently (previously a single-slot, fail-closed DoS reachable by a
    //! `Jit.compile`→`release` loop). These pin both the recycle and that a stale handle stays inert.
    use super::*;

    #[test]
    fn handle_slot_recycles_across_generation_wrap() {
        let mut h = Host::new();
        // Grant once (slot 0, generation 1) and confirm it resolves.
        let handle0 = h.try_grant(cap_id::EXIT, Binding::Exit).expect("grant");
        let slot = (handle0 as u32 as usize) & (CAP - 1);
        assert!(h.resolve(handle0, cap_id::EXIT).is_ok());

        // Force the counter to the last value before wrap, then close + regrant: the new handle is
        // issued at generation `2^GEN_BITS`, whose low `GEN_BITS` are 0 — exactly the case the old
        // `s.generation == gen` compare got wrong (the slot would have died forever).
        h.table[slot].generation = (1u32 << GEN_BITS) - 1;
        h.close(handle0);
        let handle_wrap = h.try_grant(cap_id::EXIT, Binding::Exit).expect("regrant");
        assert_eq!(
            (handle_wrap as u32 as usize) & (CAP - 1),
            slot,
            "same slot reused"
        );
        assert_eq!(
            h.table[slot].generation,
            1u32 << GEN_BITS,
            "counter wrapped past GEN_BITS"
        );
        assert!(
            h.resolve(handle_wrap, cap_id::EXIT).is_ok(),
            "slot must recycle cleanly when the generation counter wraps"
        );

        // The stale pre-wrap handle is still inert (a forged/stale index re-checks the generation and
        // faults), so recycling did not reopen a use-after-close hole.
        assert!(matches!(
            h.resolve(handle0, cap_id::EXIT),
            Err(Trap::CapFault)
        ));
    }
}

#[cfg(test)]
mod prot_tests {
    //! White-box tests for the guest-visible page-protection model (`map`/`unmap`/`protect`,
    //! §3e Memory cap / §4) — the reference semantics the JIT's `mprotect`-backed side is
    //! differential-tested against next. Granularity is the **host** page size (4 KiB / 16 KiB),
    //! same as `Mem`, so these pass on any host.
    use super::*;

    /// The host page size — the protection granularity these tests align to.
    fn page() -> u64 {
        host_page_size()
    }

    /// A fully-mapped 64 KiB window (`mapped == reserved`, 16 pages).
    fn mem64k() -> Mem {
        Mem::with_reservation(0, 16)
    }

    #[test]
    fn protect_read_only_faults_store_allows_load() {
        let mut m = mem64k();
        let v = Value::I64(0x1122_3344_5566_7788u64 as i64);
        assert!(m.store(0, 0, StoreOp::I64, v).is_ok());
        assert_eq!(m.protect(0, page(), PROT_READ), 0);
        // a store to the RO page faults; the value is still readable
        assert_eq!(
            m.store(0, 0, StoreOp::I64, Value::I64(1)),
            Err(Trap::MemoryFault)
        );
        assert_eq!(m.load(0, 0, LoadOp::I64), Ok(v));
        // an adjacent, unprotected page is unaffected
        assert!(m.store(page(), 0, StoreOp::I64, Value::I64(7)).is_ok());
    }

    #[test]
    fn protect_rw_restores_writability() {
        let mut m = mem64k();
        assert_eq!(m.protect(0, page(), PROT_READ), 0);
        assert_eq!(
            m.store(0, 0, StoreOp::I64, Value::I64(1)),
            Err(Trap::MemoryFault)
        );
        assert_eq!(m.protect(0, page(), PROT_READ | PROT_WRITE), 0);
        assert!(m.store(0, 0, StoreOp::I64, Value::I64(1)).is_ok());
    }

    #[test]
    fn unmap_faults_then_remap_zeroes() {
        let mut m = mem64k();
        assert!(m.store(0, 0, StoreOp::I64, Value::I64(0x42)).is_ok());
        assert_eq!(m.unmap(0, page()), 0);
        assert_eq!(m.load(0, 0, LoadOp::I64), Err(Trap::MemoryFault));
        assert_eq!(
            m.store(0, 0, StoreOp::I64, Value::I64(1)),
            Err(Trap::MemoryFault)
        );
        // re-commit ⇒ accessible again and zeroed
        assert_eq!(m.map(0, page(), PROT_READ | PROT_WRITE), 0);
        assert_eq!(m.load(0, 0, LoadOp::I64), Ok(Value::I64(0)));
        assert!(m.store(0, 0, StoreOp::I64, Value::I64(1)).is_ok());
    }

    /// §12 shared synchronized address space: a forked vCPU view (`thread.spawn`) sees `map`/`unmap`
    /// made by another vCPU *after* the fork — the address space is shared, not snapshotted.
    #[test]
    fn forked_vcpu_sees_post_fork_mappings() {
        // 128 KiB reserved, 64 KiB mapped ⇒ the page at 64 KiB starts in the unmapped tail.
        let mut parent = Mem::with_reservation(17, 16);
        let child = parent.fork_for_thread();
        let tail = 1u64 << 16;
        // Both views fault on the tail initially (unmapped).
        assert_eq!(child.load(tail, 0, LoadOp::I64), Err(Trap::MemoryFault));
        // Parent maps + writes the tail *after* the fork.
        assert_eq!(parent.map(tail, page(), PROT_READ | PROT_WRITE), 0);
        assert!(parent
            .store(tail, 0, StoreOp::I64, Value::I64(0xCAFE))
            .is_ok());
        // The child now sees both the mapping (shared prot) and the bytes (shared region).
        assert_eq!(child.load(tail, 0, LoadOp::I64), Ok(Value::I64(0xCAFE)));
        // An unmap by the parent is likewise visible to the child.
        assert_eq!(parent.unmap(tail, page()), 0);
        assert_eq!(child.load(tail, 0, LoadOp::I64), Err(Trap::MemoryFault));
    }

    /// §14 nesting (interp `Mem` plumbing) under **trap-confinement**: a sub-window child admits an
    /// access iff it lies within its own slice `[base, base + size)` of the parent backing and **faults**
    /// on any out-of-child address (no aliasing back into the slice); every parent byte outside the slice
    /// is therefore unreachable (stays zero). This is the interpreter half of running a guest in a nested
    /// child window.
    #[test]
    fn sub_window_child_confined_to_its_slice() {
        let base = 1u64 << 16; // child at 64 KiB
        let size_log2 = 12u8; // 4 KiB child
        let size = 1u64 << size_log2;
        let parent = 1u64 << 17; // 128 KiB parent backing
        let mut mem = Mem::sub_window(base, size_log2, parent);

        // A store at child offset 8 lands at absolute base+8; a far offset (size+8) now **faults**
        // (trap-confinement: no wrap), leaving the earlier write intact.
        assert!(mem.store(8, 0, StoreOp::I64, Value::I64(0x1111)).is_ok());
        assert_eq!(
            mem.store(size + 8, 0, StoreOp::I64, Value::I64(0x2222)),
            Err(Trap::MemoryFault),
            "an out-of-child store faults, it does not wrap"
        );
        assert_eq!(mem.load(8, 0, LoadOp::I64), Ok(Value::I64(0x1111))); // untouched by the faulted store
        assert_eq!(mem.confine_checked(8, 0, 8), Ok(base + 8)); // confined to the child's slice

        // In-child offsets confine to `[base, base+size)`; every out-of-child address **faults**
        // (never aliased back in).
        assert_eq!(mem.confine_checked(0, 0, 1), Ok(base));
        assert_eq!(mem.confine_checked(size - 1, 0, 1), Ok(base + size - 1));
        for &a in &[size, size * 1000, u64::MAX, base, parent] {
            assert_eq!(
                mem.confine_checked(a, 0, 1),
                Err(Trap::MemoryFault),
                "out-of-child address {a:#x} must fault, not alias into the slice"
            );
        }

        // Decisive: every parent byte *outside* the child's slice is untouched (unreachable).
        for i in 0..parent {
            if i < base || i >= base + size {
                assert_eq!(
                    mem.back.byte(i),
                    0,
                    "child wrote outside its slice at {i:#x}"
                );
            }
        }
    }

    /// §14 nesting: a child's `AddressSpace`-style `map`/`unmap` (page protection) now works on a
    /// sub-window `Mem`. The prot map is keyed window-relative, so the base folds out consistently —
    /// before the fix, `unmap` on a sub-window `-EINVAL`'d (its absolute address was bounded against
    /// the child's window-relative `reserved`). A page unmapped via its **absolute** (§14-shifted)
    /// address faults a later plain access; re-`map` recommits it zeroed; and an address below the
    /// child's base or past its top is out of range.
    #[test]
    fn sub_window_page_protection_is_window_relative() {
        let base = 1u64 << 16; // child at 64 KiB
        let size_log2 = 16u8; // 64 KiB child (≥ one host page, so a whole page fits)
        let parent = 1u64 << 18; // 256 KiB parent backing
        let p = page();
        let mut mem = Mem::sub_window(base, size_log2, parent);

        // Initially fully mapped: a store/load at child offset 0 works.
        assert!(mem.store(0, 0, StoreOp::I64, Value::I64(0xABCD)).is_ok());
        assert_eq!(mem.load(0, 0, LoadOp::I64), Ok(Value::I64(0xABCD)));

        // Unmap the child's first page via its **guest-relative** offset 0 (the whole `GuestMem`
        // surface speaks the zero-based window the guest sees; the page lands at `base` in the
        // shared parent backing).
        assert_eq!(mem.unmap(0, p), 0, "sub-window unmap should succeed");
        assert_eq!(
            mem.load(0, 0, LoadOp::I64),
            Err(Trap::MemoryFault),
            "an access to the child's unmapped page must fault"
        );
        // A different page still within the child is unaffected.
        assert!(mem.store(p, 0, StoreOp::I64, Value::I64(0x1234)).is_ok());

        // Re-map recommits the page, zeroed — and the backing byte that changed is the *parent's*
        // byte at `base` (the child's slice), not the parent's page 0.
        assert_eq!(mem.map(0, p, PROT_WRITE), 0);
        assert_eq!(mem.load(0, 0, LoadOp::I64), Ok(Value::I64(0)));

        // The child cannot name anything at/past its own window top — its reserved domain is
        // `[0, size)`, wherever that sits in an ancestor's window.
        assert_eq!(
            mem.unmap(1u64 << size_log2, p),
            EINVAL,
            "at/after the child's window top is out of range"
        );
    }

    #[test]
    fn bad_args_einval() {
        let mut m = mem64k();
        assert_eq!(m.protect(1, page(), PROT_READ), EINVAL); // misaligned offset
        assert_eq!(m.protect(0, 0, PROT_READ), EINVAL); // zero length
                                                        // mem64k is fully mapped (reserved == mapped == 64 KiB), so its tail is empty: a range
                                                        // at/past the reserved top is still out of range.
        assert_eq!(m.unmap(65536, page()), EINVAL); // offset == reserved ⇒ out of range
        assert_eq!(m.map(0, 1 << 20, PROT_WRITE), EINVAL); // len past reserved
    }

    /// A window whose reserved domain (`1 MiB`) is larger than the initial backed prefix
    /// (`64 KiB`): the tail `[64 KiB, 1 MiB)` is reserved-but-unmapped and the guest can grow into
    /// it. `Mem::with_reservation(reserved_log2=20, mapped_log2=16)`.
    fn mem_growable() -> Mem {
        Mem::with_reservation(20, 16)
    }

    #[test]
    fn tail_access_faults_until_mapped() {
        let mut m = mem_growable();
        let tail = 1u64 << 16; // first byte of the reserved tail (64 KiB)
                               // Untouched tail faults (any access) — it is reserved-but-unmapped.
        assert_eq!(m.load(tail, 0, LoadOp::I64), Err(Trap::MemoryFault));
        assert_eq!(
            m.store(tail, 0, StoreOp::I64, Value::I64(1)),
            Err(Trap::MemoryFault)
        );
        // Grow one page into the tail; now it is committed, zeroed, read-write.
        assert_eq!(m.map(tail, page(), PROT_READ | PROT_WRITE), 0);
        assert_eq!(m.load(tail, 0, LoadOp::I64), Ok(Value::I64(0)));
        assert!(m.store(tail, 0, StoreOp::I64, Value::I64(0x99)).is_ok());
        assert_eq!(m.load(tail, 0, LoadOp::I64), Ok(Value::I64(0x99)));
        // The next page up is still unmapped.
        assert_eq!(
            m.load(tail + page(), 0, LoadOp::I64),
            Err(Trap::MemoryFault)
        );
    }

    #[test]
    fn grow_then_unmap_faults_again() {
        let mut m = mem_growable();
        let tail = 1u64 << 16;
        assert_eq!(m.map(tail, page(), PROT_READ | PROT_WRITE), 0);
        assert!(m.store(tail, 0, StoreOp::I64, Value::I64(7)).is_ok());
        assert_eq!(m.unmap(tail, page()), 0);
        assert_eq!(m.load(tail, 0, LoadOp::I64), Err(Trap::MemoryFault));
        // Re-mapping zero-fills (the old contents are gone).
        assert_eq!(m.map(tail, page(), PROT_READ | PROT_WRITE), 0);
        assert_eq!(m.load(tail, 0, LoadOp::I64), Ok(Value::I64(0)));
    }

    #[test]
    fn grow_read_only_then_store_faults() {
        let mut m = mem_growable();
        let tail = 1u64 << 16;
        // Map a tail page read-only: reads of the (zeroed) page succeed, a store faults.
        assert_eq!(m.map(tail, page(), PROT_READ), 0);
        assert_eq!(m.load(tail, 0, LoadOp::I64), Ok(Value::I64(0)));
        assert_eq!(
            m.store(tail, 0, StoreOp::I64, Value::I64(1)),
            Err(Trap::MemoryFault)
        );
    }

    #[test]
    fn growth_bounds_are_reserved_not_mapped() {
        let mut m = mem_growable();
        let reserved = 1u64 << 20;
        // Mapping anywhere in the reserved tail is allowed now (was EINVAL pre-growth).
        assert_eq!(m.map(1 << 16, page(), PROT_READ | PROT_WRITE), 0);
        assert_eq!(m.map(reserved - page(), page(), PROT_READ | PROT_WRITE), 0);
        // At/past the reserved top is still out of range.
        assert_eq!(m.map(reserved, page(), PROT_WRITE), EINVAL);
        assert_eq!(m.unmap(reserved - page(), 2 * page()), EINVAL);
    }

    #[test]
    fn grown_tail_buffer_borrow_round_trips() {
        // A cap buffer (§7 borrow) in a grown tail region validates and round-trips; one in the
        // unmapped tail is rejected (-EFAULT ⇒ None).
        let mut m = mem_growable();
        let tail = 1u64 << 16;
        assert!(m.write_bytes_impl(tail, &[1, 2, 3, 4]).is_none()); // unmapped ⇒ EFAULT
        assert_eq!(m.map(tail, page(), PROT_READ | PROT_WRITE), 0);
        assert!(m.write_bytes_impl(tail, &[1, 2, 3, 4]).is_some());
        assert_eq!(m.read_bytes_impl(tail, 4), Some(vec![1, 2, 3, 4]));
        // A borrow straddling the committed/uncommitted page boundary is rejected.
        assert!(m.read_bytes_impl(tail + page() - 2, 4).is_none());
    }

    #[test]
    fn cross_page_store_faults_if_either_page_protected() {
        let mut m = mem64k();
        // page 1 read-only; an 8-byte store straddling the page-0/1 boundary touches page 1.
        assert_eq!(m.protect(page(), page(), PROT_READ), 0);
        assert_eq!(
            m.store(page() - 4, 0, StoreOp::I64, Value::I64(1)),
            Err(Trap::MemoryFault)
        );
        // fully within page 0 (still rw) is fine
        assert!(m.store(page() - 8, 0, StoreOp::I64, Value::I64(1)).is_ok());
    }

    #[test]
    fn unprotected_window_is_unrestricted() {
        // With an empty protection map, check_prot is a no-op: every in-window access works.
        let mut m = mem64k();
        for off in [0u64, 8, page(), 65536 - 8] {
            assert!(m.store(off, 0, StoreOp::I64, Value::I64(0x55)).is_ok());
            assert_eq!(m.load(off, 0, LoadOp::I64), Ok(Value::I64(0x55)));
        }
    }

    // ---- §13 SharedRegion: host-backed memory aliased into the window ----

    /// A §13 `SharedRegion` backing of `pages` whole host pages, zero-filled.
    fn region(pages: u64) -> RegionBacking {
        Arc::new(VecBacking(Mutex::new(vec![0u8; (pages * page()) as usize])))
    }

    #[test]
    fn shared_region_aliases_two_window_offsets() {
        // One region mapped at two window offsets names the *same* bytes: a store at one alias is
        // visible at the other (and vice versa) — the §13 zero-overhead aliasing primitive.
        let mut m = mem64k();
        let r = region(1);
        let (a, b) = (0, page());
        assert_eq!(
            m.map_region(a, 0, page(), PROT_READ | PROT_WRITE, 0, r.clone()),
            0
        );
        assert_eq!(m.map_region(b, 0, page(), PROT_READ | PROT_WRITE, 0, r), 0);
        let v = Value::I64(0x0123_4567_89ab_cdefu64 as i64);
        assert!(m.store(a, 0, StoreOp::I64, v).is_ok());
        assert_eq!(m.load(b, 0, LoadOp::I64), Ok(v), "A→B alias");
        let w = Value::I64(0x7777);
        assert!(m.store(b + 16, 0, StoreOp::I64, w).is_ok());
        assert_eq!(m.load(a + 16, 0, LoadOp::I64), Ok(w), "B→A alias");
    }

    #[test]
    fn shared_region_offsets_are_region_relative() {
        // Pointers are region-relative (§13): the same *region* offset at two window offsets aliases;
        // different region offsets are independent.
        let mut m = mem64k();
        let r = region(2);
        // window pages 0,1 ⇒ region pages 0,1.
        assert_eq!(
            m.map_region(0, 0, 2 * page(), PROT_READ | PROT_WRITE, 0, r.clone()),
            0
        );
        // a second mapping of *region page 1* at window page 2.
        assert_eq!(
            m.map_region(2 * page(), page(), page(), PROT_READ | PROT_WRITE, 0, r),
            0
        );
        let v = Value::I64(0xdead_beef);
        assert!(m.store(page(), 0, StoreOp::I64, v).is_ok()); // write region page 1 via window page 1
        assert_eq!(m.load(2 * page(), 0, LoadOp::I64), Ok(v)); // observe via window page 2
        assert_eq!(m.load(0, 0, LoadOp::I64), Ok(Value::I64(0))); // region page 0 independent
    }

    #[test]
    fn shared_region_read_only_alias_shares_reads_faults_stores() {
        let mut m = mem64k();
        let r = region(1);
        assert_eq!(
            m.map_region(0, 0, page(), PROT_READ | PROT_WRITE, 0, r.clone()),
            0
        );
        assert_eq!(m.map_region(page(), 0, page(), PROT_READ, 0, r), 0); // RO alias of same region
        let v = Value::I64(0x5151_5151);
        assert!(m.store(0, 0, StoreOp::I64, v).is_ok());
        assert_eq!(
            m.load(page(), 0, LoadOp::I64),
            Ok(v),
            "RO alias sees the write"
        );
        assert_eq!(
            m.store(page(), 0, StoreOp::I64, Value::I64(1)),
            Err(Trap::MemoryFault),
            "store to RO alias faults"
        );
        // protect(READ) on the RW alias keeps it aliased (shared bytes survive), now store-faulting.
        assert_eq!(m.protect(0, page(), PROT_READ), 0);
        assert_eq!(m.load(0, 0, LoadOp::I64), Ok(v));
        assert_eq!(
            m.store(0, 0, StoreOp::I64, Value::I64(2)),
            Err(Trap::MemoryFault)
        );
    }

    #[test]
    fn shared_region_unmap_drops_alias_and_map_replaces_anonymous() {
        let mut m = mem64k();
        // Aliasing over an already-written anonymous page redirects to the region (old bytes gone).
        assert!(m.store(0, 0, StoreOp::I64, Value::I64(0x4242)).is_ok());
        let r = region(1);
        assert_eq!(m.map_region(0, 0, page(), PROT_READ | PROT_WRITE, 0, r), 0);
        assert_eq!(
            m.load(0, 0, LoadOp::I64),
            Ok(Value::I64(0)),
            "region zero-fill"
        );
        assert!(m.store(0, 0, StoreOp::I64, Value::I64(9)).is_ok());
        // unmap drops the alias → faults.
        assert_eq!(m.unmap(0, page()), 0);
        assert_eq!(m.load(0, 0, LoadOp::I64), Err(Trap::MemoryFault));
    }

    #[test]
    fn shared_region_bad_args_einval() {
        let mut m = mem64k();
        let r = region(1); // one page
        assert_eq!(m.map_region(1, 0, page(), PROT_READ, 0, r.clone()), EINVAL); // misaligned window
        assert_eq!(m.map_region(0, 0, 0, PROT_READ, 0, r.clone()), EINVAL); // zero len
        assert_eq!(m.map_region(0, 1, page(), PROT_READ, 0, r.clone()), EINVAL); // misaligned region
        assert_eq!(
            m.map_region(0, page(), page(), PROT_READ, 0, r.clone()),
            EINVAL
        ); // region OOB
        assert_eq!(
            m.map_region(0, 0, 2 * page(), PROT_READ, 0, r.clone()),
            EINVAL
        ); // span > backing
        assert_eq!(m.map_region(0, 0, page(), PROT_WRITE, 0, r.clone()), EINVAL); // not readable
        assert_eq!(m.map_region(65536, 0, page(), PROT_READ, 0, r), EINVAL); // window past reserved
    }
}

#[cfg(test)]
mod domain_table_tests {
    //! The shared, live [`DomainTable`] (DESIGN.md §22): an `install` on one vCPU's view is visible to
    //! every vCPU sharing the `Arc` — the threaded-install / install-during-own-invocation
    //! faithfulness the reference interpreter previously lacked (a per-vCPU table snapshot hid a
    //! post-spawn install from a worker, diverging from the JIT's one shared `fn_table`). Slot reads
    //! are lock-free atomic loads; the units backing resolves via a lazily-resynced local clone.
    use super::*;
    use svm_ir::{Func, ValType};

    /// A one-function unit `(i32 × n) -> (i32)` — blocks are irrelevant to table mechanics.
    fn unit(n: usize) -> Arc<[Func]> {
        Arc::from(vec![Func {
            params: vec![ValType::I32; n],
            results: vec![ValType::I32],
            blocks: Vec::new(),
        }])
    }

    fn sig(n: usize) -> FuncType {
        FuncType {
            params: vec![ValType::I32; n],
            results: vec![ValType::I32],
        }
    }

    /// **The faithfulness property:** an `install` is visible to another vCPU sharing the domain's
    /// `Arc<DomainTable>` (the `thread.spawn` / `Jit.invoke` child) *immediately* — not snapshotted.
    #[test]
    fn install_is_visible_across_a_shared_view() {
        let parent: Arc<[Func]> = unit(1); // module 0: one (i32)->(i32) function
        let dt = Arc::new(DomainTable::new(&parent, 3)); // reserve 8 slots → padding at 1..8
        assert_eq!(dt.slot(0).module, 0);
        assert_eq!(dt.slot(1).module, TABLE_EMPTY);

        // A second view — a worker thread / invoke child sharing the same live table.
        let worker = Arc::clone(&dt);

        let slot = dt.install(unit(2)).expect("install"); // a (i32,i32)->(i32) unit
        assert_eq!(
            slot, 1,
            "first padding slot is just past the 1 module function"
        );

        // The worker observes the install at once (shared, not a stale snapshot).
        assert_eq!(worker.slot(1).module, 1, "module id = units[0]");
        assert_eq!(worker.units_snapshot().len(), 1);

        // Resolve it through the worker's *stale* (empty) local cache — it re-syncs and finds the
        // unit, with the right signature. This is the dispatch path a worker takes.
        let mut local: Vec<Arc<[Func]>> = Vec::new();
        let got = dispatch_indirect(&worker, &parent, &mut local, &None, 1, &sig(2));
        assert_eq!(got, Ok((1, 0)));
        assert_eq!(local.len(), 1, "the miss re-synced the local prefix");
        // A signature mismatch on the same slot traps fail-closed.
        assert_eq!(
            dispatch_indirect(&worker, &parent, &mut local, &None, 1, &sig(1)),
            Err(Trap::IndirectCallType)
        );
    }

    /// `uninstall` clears a slot back to trapping padding and guards real-function / out-of-range /
    /// already-empty slots; the change is likewise visible across a shared view.
    #[test]
    fn uninstall_clears_and_guards() {
        let parent: Arc<[Func]> = unit(1);
        let dt = Arc::new(DomainTable::new(&parent, 2)); // 4 slots, 1 real func
        let slot = dt.install(unit(1)).expect("install") as usize;
        assert_eq!(slot, 1);
        assert!(!dt.uninstall(0, 1), "slot 0 is the real function");
        assert!(!dt.uninstall(99, 1), "out of range");
        assert!(dt.uninstall(slot, 1), "installed padding slot clears");
        assert_eq!(dt.slot(slot).module, TABLE_EMPTY);
        assert!(!dt.uninstall(slot, 1), "already empty");
        // A stale `call_indirect` of the cleared slot now traps.
        let mut local: Vec<Arc<[Func]>> = Vec::new();
        assert_eq!(
            dispatch_indirect(&dt, &parent, &mut local, &None, slot as i32, &sig(1)),
            Err(Trap::IndirectCallType)
        );
    }

    /// `resolve_module` routes the three module classes: 0 → the program, `INVOKE_MODULE` → the
    /// invoked unit (kept out of the shared `units`, so it never collides with an install), and
    /// `k ≥ 1` → an installed unit (via the lazily-resynced local clone).
    #[test]
    fn resolve_module_routes_program_invoke_and_installed() {
        let parent: Arc<[Func]> = unit(1);
        let dt = Arc::new(DomainTable::new(&parent, 2));
        dt.install(unit(2)).expect("install"); // module 1
        let invoked = Some(unit(3));
        let mut local: Vec<Arc<[Func]>> = Vec::new();

        let m0 = resolve_module(&parent, &mut local, &None, &dt, 0).unwrap();
        assert_eq!(m0[0].params.len(), 1);
        let mi = resolve_module(&parent, &mut local, &invoked, &dt, INVOKE_MODULE).unwrap();
        assert_eq!(
            mi[0].params.len(),
            3,
            "the invoked unit, not an installed module"
        );
        let m1 = resolve_module(&parent, &mut local, &None, &dt, 1).unwrap();
        assert_eq!(
            m1[0].params.len(),
            2,
            "installed unit, resolved via a re-synced local cache"
        );
        // An out-of-range module (a forged/stale slot) yields None → the caller traps.
        assert!(resolve_module(&parent, &mut local, &None, &dt, 9).is_none());
    }
}
