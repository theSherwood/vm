//! The **debuggee backend seam** (DEBUGGING.md G3). `DapServer` was hard-wired to the tree-walking
//! `svm_interp::Inspector`; this trait lets the same server drive *either* engine — the tree-walker
//! (the reference oracle, full feature set) or the **bytecode VM** the browser playground actually
//! runs, over `svm_interp::bytecode::DebugRun`.
//!
//! The bytecode engine covers breakpoints, stepping, backtrace, scalar/aggregate inspection,
//! **reverse debugging** (`seek`/`step_back`/`reverseContinue`, by deterministic replay — the debug
//! run is pure compute, so seeking to an earlier op clock rebuilds a fresh `DebugRun` and replays to
//! that many ops), and **data breakpoints** (`set_watchpoint` — a per-op check of the effective
//! address, computed like the interpreter's `access_of`, against the watched ranges; the run stops
//! *before* an op that touches one). For a spawn-free guest `supports_reverse`/`supports_watch` are
//! both `true`.
//!
//! **Multithreading** lands on its *own* engine — a `thread.spawn` guest launches the
//! `svm_interp::bytecode::ScheduledDebugRun`, a deterministic cooperative multi-vCPU **debug
//! scheduler**: breakpoints fire in whichever thread reaches them, `threads`/`stopped_task`/
//! `select_task` serve per-thread stacks, stepping (in/over/out) drives the stopped thread,
//! **cross-thread watchpoints** fire in whichever thread touches the range, and **reverse debugging**
//! works by deterministic replay to a global scheduler `turn` — *not* delegated to the tree-walker
//! (the differential oracle only, far too slow for any user-facing path). Both engines report
//! `supports_reverse`/`supports_watch` = `true`; the single-vCPU coordinate is the op `clock`, the
//! multithreaded one the global `turn`. Correctness is guaranteed by `crates/svm/tests/debug_parity.rs`
//! and `bytecode_debug_threads.rs` (engine level, vs the tree-walker oracle) and `dap_over_bytecode_*`
//! (server level).

use svm_interp::bytecode::{self, DebugRun, SchedBreak, SchedStop, ScheduledDebugRun};
use svm_interp::{
    FrameInfo, Inspector, IrPc, SourceLoc, Stop, StopReason, Trap, Value, VarValue, WatchId,
    WatchKind,
};
use svm_ir::{FuncIdx, Module};

/// The ~20 `Inspector` operations `DapServer` drives, abstracted so a bytecode-backed session can
/// serve the same requests. Methods a backend can't honor (reverse/watch) are gated by
/// [`supports_reverse`](Debuggee::supports_reverse) / [`supports_watch`](Debuggee::supports_watch);
/// the server checks the gate before calling, so their bodies are dormant on such a backend.
pub trait Debuggee {
    // --- execution -------------------------------------------------------------------------------
    fn run_until_stop(&mut self) -> Stop;
    fn step(&mut self) -> Stop;
    fn step_over(&mut self) -> Stop;
    fn step_out(&mut self) -> Stop;
    fn step_back(&mut self) -> Stop;
    fn seek(&mut self, t: u64) -> Stop;

    // --- breakpoints / watchpoints ---------------------------------------------------------------
    fn set_breakpoint(&mut self, pc: IrPc);
    fn clear_breakpoint(&mut self, pc: IrPc) -> bool;
    /// `None` if the backend has no watchpoints (bytecode) — the server reports the data breakpoint
    /// unverified.
    fn set_watchpoint(&mut self, addr: u64, len: u64, kind: WatchKind) -> Option<WatchId>;
    fn clear_watchpoint(&mut self, id: WatchId) -> bool;

    // --- inspection ------------------------------------------------------------------------------
    fn backtrace(&self) -> Vec<FrameInfo>;
    fn func_name(&self, func: FuncIdx) -> Option<&str>;
    fn source_loc(&self, pc: IrPc) -> Option<SourceLoc>;
    fn read_var(&self, frame_from_top: usize, name: &str, width: usize) -> Option<VarValue>;
    fn var_addr(&self, frame_from_top: usize, name: &str) -> Option<u64>;
    fn read_window(&self, addr: u64, len: usize) -> Result<Vec<u8>, Trap>;

    // --- threads / time coordinate ---------------------------------------------------------------
    fn threads(&self) -> Vec<u64>;
    fn select_task(&mut self, id: u64) -> bool;
    fn stopped_task(&self) -> Option<u64>;
    fn turn(&self) -> u64;
    fn clock(&self) -> u64;

    // --- capability gates ------------------------------------------------------------------------
    /// Reverse debugging (`stepBack` / `reverseContinue`). Default `true` (the tree-walker).
    fn supports_reverse(&self) -> bool {
        true
    }
    /// Data breakpoints (`setDataBreakpoints` watchpoints). Default `true` (the tree-walker).
    fn supports_watch(&self) -> bool {
        true
    }
}

/// The tree-walker backend — the original, full-featured engine. Every method delegates to the
/// inherent `Inspector` method (an inherent method shadows the trait one, so no recursion).
impl Debuggee for Inspector {
    fn run_until_stop(&mut self) -> Stop {
        Inspector::run_until_stop(self)
    }
    fn step(&mut self) -> Stop {
        Inspector::step(self)
    }
    fn step_over(&mut self) -> Stop {
        Inspector::step_over(self)
    }
    fn step_out(&mut self) -> Stop {
        Inspector::step_out(self)
    }
    fn step_back(&mut self) -> Stop {
        Inspector::step_back(self)
    }
    fn seek(&mut self, t: u64) -> Stop {
        Inspector::seek(self, t)
    }
    fn set_breakpoint(&mut self, pc: IrPc) {
        Inspector::set_breakpoint(self, pc)
    }
    fn clear_breakpoint(&mut self, pc: IrPc) -> bool {
        Inspector::clear_breakpoint(self, pc)
    }
    fn set_watchpoint(&mut self, addr: u64, len: u64, kind: WatchKind) -> Option<WatchId> {
        Some(Inspector::set_watchpoint(self, addr, len, kind))
    }
    fn clear_watchpoint(&mut self, id: WatchId) -> bool {
        Inspector::clear_watchpoint(self, id)
    }
    fn backtrace(&self) -> Vec<FrameInfo> {
        Inspector::backtrace(self)
    }
    fn func_name(&self, func: FuncIdx) -> Option<&str> {
        Inspector::func_name(self, func)
    }
    fn source_loc(&self, pc: IrPc) -> Option<SourceLoc> {
        Inspector::source_loc(self, pc)
    }
    fn read_var(&self, frame_from_top: usize, name: &str, width: usize) -> Option<VarValue> {
        Inspector::read_var(self, frame_from_top, name, width)
    }
    fn var_addr(&self, frame_from_top: usize, name: &str) -> Option<u64> {
        Inspector::var_addr(self, frame_from_top, name)
    }
    fn read_window(&self, addr: u64, len: usize) -> Result<Vec<u8>, Trap> {
        Inspector::read_window(self, addr, len)
    }
    fn threads(&self) -> Vec<u64> {
        Inspector::threads(self)
    }
    fn select_task(&mut self, id: u64) -> bool {
        Inspector::select_task(self, id)
    }
    fn stopped_task(&self) -> Option<u64> {
        Inspector::stopped_task(self)
    }
    fn turn(&self) -> u64 {
        Inspector::turn(self)
    }
    fn clock(&self) -> u64 {
        Inspector::clock(self)
    }
}

/// The bytecode engine behind a [`BytecodeBackend`]: the single-vCPU [`DebugRun`] for a spawn-free
/// guest, or the multi-vCPU [`ScheduledDebugRun`] (a cooperative debug scheduler with per-thread
/// breakpoints, `select_task`, in/over/out stepping, cross-thread watchpoints, and reverse debugging)
/// for a `thread.spawn` guest. Chosen at launch by [`bytecode::module_spawns_threads`]; both are fully
/// forward + reverse + watch capable.
enum Engine {
    Single(DebugRun),
    Threaded(ScheduledDebugRun),
}

/// The **bytecode backend** — the resumable bytecode debug session ([`Engine`]) plus the persistent
/// breakpoint set `DapServer` expects, the module (for `source_loc`/`func_name`, which are
/// engine-neutral free functions keyed on the `IrPc`), and the launch `func`/`args` so reverse
/// debugging can rebuild a fresh run and replay to an earlier op clock.
pub struct BytecodeBackend {
    engine: Engine,
    module: Module,
    func: FuncIdx,
    args: Vec<Value>,
    breakpoints: Vec<IrPc>,
    /// Armed watchpoints with backend-owned stable ids (re-applied to the run after a `seek` rebuild).
    /// Single-vCPU only — the scheduled engine reports `supports_watch = false` this slice.
    watch_specs: Vec<(WatchId, u64, u64, WatchKind)>,
    next_watch: u32,
    fuel: u64,
}

impl BytecodeBackend {
    /// Open a bytecode debug session on `module`'s `func(args)`. A `thread.spawn` guest gets the
    /// multithreaded scheduled engine; a spawn-free one the single-vCPU engine. `None` if the module
    /// is outside the bytecode engine's subset (`compile_module` declines it).
    pub fn new(
        module: Module,
        func: FuncIdx,
        args: &[Value],
        fuel: u64,
    ) -> Option<BytecodeBackend> {
        let engine = if bytecode::module_spawns_threads(&module) {
            Engine::Threaded(ScheduledDebugRun::new(&module, func, args)?)
        } else {
            Engine::Single(DebugRun::new(&module, func, args)?)
        };
        Some(BytecodeBackend {
            engine,
            module,
            func,
            args: args.to_vec(),
            breakpoints: Vec::new(),
            watch_specs: Vec::new(),
            next_watch: 0,
            fuel,
        })
    }

    /// Whether this session runs on the multithreaded scheduled engine (so the DAP server uses the
    /// global `turn` as the reverse time coordinate, not the single-vCPU op `clock`).
    pub fn is_threaded(&self) -> bool {
        matches!(self.engine, Engine::Threaded(_))
    }

    /// Push the current watchpoint ranges into the live engine (after arming/clearing one, or re-arming
    /// a fresh single-vCPU run built by `seek`). Cross-thread on the scheduled engine.
    fn apply_watches(&mut self) {
        let ranges: Vec<_> = self
            .watch_specs
            .iter()
            .map(|(_, a, l, k)| (*a, *l, *k))
            .collect();
        match &mut self.engine {
            Engine::Single(run) => run.set_watchpoints(ranges),
            Engine::Threaded(run) => run.set_watchpoints(ranges),
        }
    }

    /// Map an engine completion (a resume/step returned no pc) to a `Stop`: the finished result (or
    /// trap) if the root is done, else `Blocked` (a concurrency seam that engine can't follow).
    fn finish_stop(&self) -> Stop {
        let result = match &self.engine {
            Engine::Single(run) => run.result().cloned(),
            Engine::Threaded(run) => run.result().cloned(),
        };
        match result {
            Some(r) => Stop::Finished(r),
            None => Stop::Blocked,
        }
    }

    /// A `Step` stop at the focused thread's current pc (or the finished result), for a resume/seek that
    /// didn't hit a breakpoint.
    fn step_stop(&self) -> Stop {
        let pc = match &self.engine {
            Engine::Single(run) => run.frame_pc(0),
            Engine::Threaded(run) => run.frame_pc(0),
        };
        match pc {
            Some(pc) => Stop::Break {
                reason: StopReason::Step,
                pc,
            },
            None => self.finish_stop(),
        }
    }

    /// Map a multithreaded [`SchedStop`] to the DAP [`Stop`] — the `SchedBreak` reason carries whether
    /// it was a breakpoint, a data breakpoint (with the confined address + read/write), or a step.
    fn sched_stop(s: SchedStop) -> Stop {
        match s {
            SchedStop::Break { pc, reason } => {
                let reason = match reason {
                    SchedBreak::Breakpoint => StopReason::Breakpoint,
                    SchedBreak::Watchpoint { addr, write } => {
                        StopReason::Watchpoint { addr, write }
                    }
                    SchedBreak::Step => StopReason::Step,
                };
                Stop::Break { reason, pc }
            }
            SchedStop::Finished(r) => Stop::Finished(r),
            // No runnable thread (deadlock/`wait`), or an op outside the scheduler's subset.
            SchedStop::Blocked | SchedStop::Declined => Stop::Blocked,
        }
    }
}

impl Debuggee for BytecodeBackend {
    fn run_until_stop(&mut self) -> Stop {
        // A fresh fuel budget per resume (debugging is interactive; the run replays from scratch on a
        // seek, so a shared decrementing counter would be inconsistent).
        let mut fuel = self.fuel;
        match &mut self.engine {
            Engine::Single(run) => match run.run_to(&self.breakpoints, &mut fuel) {
                // A stop is a watchpoint hit if the run flagged one before this op, else a breakpoint.
                Some(pc) => {
                    let reason = match run.take_watch_hit() {
                        Some((addr, write)) => StopReason::Watchpoint { addr, write },
                        None => StopReason::Breakpoint,
                    };
                    Stop::Break { reason, pc }
                }
                None => self.finish_stop(),
            },
            Engine::Threaded(run) => {
                run.set_breakpoints(self.breakpoints.clone());
                Self::sched_stop(run.run_until_stop(&mut fuel))
            }
        }
    }
    fn step(&mut self) -> Stop {
        let mut fuel = self.fuel;
        match &mut self.engine {
            Engine::Single(run) => match run.step(&mut fuel) {
                Some(pc) => Stop::Break {
                    reason: StopReason::Step,
                    pc,
                },
                None => self.finish_stop(),
            },
            Engine::Threaded(run) => Self::sched_stop(run.step(&mut fuel)),
        }
    }
    fn step_over(&mut self) -> Stop {
        let mut fuel = self.fuel;
        match &mut self.engine {
            Engine::Single(run) => match run.step_over(&mut fuel) {
                Some(pc) => Stop::Break {
                    reason: StopReason::Step,
                    pc,
                },
                None => self.finish_stop(),
            },
            Engine::Threaded(run) => Self::sched_stop(run.step_over(&mut fuel)),
        }
    }
    fn step_out(&mut self) -> Stop {
        let mut fuel = self.fuel;
        match &mut self.engine {
            Engine::Single(run) => match run.step_out(&mut fuel) {
                Some(pc) => Stop::Break {
                    reason: StopReason::Step,
                    pc,
                },
                None => self.finish_stop(),
            },
            Engine::Threaded(run) => Self::sched_stop(run.step_out(&mut fuel)),
        }
    }
    // Reverse debugging by **deterministic replay** (DEBUGGING.md W1): the debug run is pure compute
    // (single-vCPU, no capabilities), so seeking to an earlier op clock = rebuild a fresh run and
    // replay to that many ops. `step_back` = one op earlier. (A checkpoint ladder to bound the replay
    // cost is a future optimization; the debugged programs here are small.)
    fn step_back(&mut self) -> Stop {
        // Rewind to the previous op that sits at a real IR instruction (a stoppable position — not a
        // terminator slot, where there's nothing to inspect) strictly before now, then seek there. The
        // single-vCPU coordinate is the op `clock`; the multithreaded one is the global scheduler `turn`.
        let now = match &self.engine {
            Engine::Single(run) => run.op_clock(),
            Engine::Threaded(run) => run.op_turn(),
        };
        let mut fuel = self.fuel;
        let target = match &self.engine {
            Engine::Single(_) => {
                let Some(mut probe) = DebugRun::new(&self.module, self.func, &self.args) else {
                    return Stop::Blocked;
                };
                let mut target = 0;
                loop {
                    let c = probe.op_clock();
                    if c >= now {
                        break;
                    }
                    if probe.frame_pc(0).is_some() {
                        target = c;
                    }
                    if !probe.tick(&mut fuel) {
                        break;
                    }
                }
                target
            }
            Engine::Threaded(_) => {
                let Some(mut probe) = ScheduledDebugRun::new(&self.module, self.func, &self.args)
                else {
                    return Stop::Blocked;
                };
                let mut target = 0;
                loop {
                    let c = probe.op_turn();
                    if c >= now {
                        break;
                    }
                    probe.locate();
                    if probe.frame_pc(0).is_some() {
                        target = c;
                    }
                    if !probe.tick(&mut fuel) {
                        break;
                    }
                }
                target
            }
        };
        self.seek(target)
    }
    fn seek(&mut self, t: u64) -> Stop {
        let mut fuel = self.fuel;
        if self.is_threaded() {
            // Rebuild a fresh scheduled run and replay `t` turns — the schedule is deterministic, so
            // this reproduces the exact state at global turn `t` (DEBUGGING.md W1, multithreaded).
            let Some(mut run) = ScheduledDebugRun::new(&self.module, self.func, &self.args) else {
                return Stop::Blocked;
            };
            run.set_breakpoints(self.breakpoints.clone());
            run.set_watchpoints(
                self.watch_specs
                    .iter()
                    .map(|(_, a, l, k)| (*a, *l, *k))
                    .collect(),
            );
            while run.op_turn() < t && run.tick(&mut fuel) {}
            run.locate();
            if let Some(pc) = run.frame_pc(0) {
                if self.breakpoints.contains(&pc) {
                    run.arm_breakpoint_skip();
                }
            }
            self.engine = Engine::Threaded(run);
            return self.step_stop();
        }
        let Some(mut run) = DebugRun::new(&self.module, self.func, &self.args) else {
            return Stop::Blocked;
        };
        while run.op_clock() < t && run.tick(&mut fuel) {}
        self.engine = Engine::Single(run);
        self.apply_watches(); // re-arm the watchpoints on the fresh (replayed) run
                              // If the replay landed exactly on a breakpoint op, arm the skip so a forward `continue` from
                              // here makes progress instead of immediately re-reporting this stop.
        if let Engine::Single(run) = &mut self.engine {
            if let Some(pc) = run.frame_pc(0) {
                if self.breakpoints.contains(&pc) {
                    run.arm_breakpoint_skip();
                }
            }
        }
        self.step_stop()
    }
    fn set_breakpoint(&mut self, pc: IrPc) {
        if !self.breakpoints.contains(&pc) {
            self.breakpoints.push(pc);
        }
    }
    fn clear_breakpoint(&mut self, pc: IrPc) -> bool {
        let before = self.breakpoints.len();
        self.breakpoints.retain(|&b| b != pc);
        self.breakpoints.len() != before
    }
    // Data breakpoints: arm a window watchpoint (a backend-owned stable id, so it survives a `seek`).
    // Cross-thread on the scheduled engine (fires in whichever thread touches the range).
    fn set_watchpoint(&mut self, addr: u64, len: u64, kind: WatchKind) -> Option<WatchId> {
        let id = WatchId::from_raw(self.next_watch);
        self.next_watch += 1;
        self.watch_specs.push((id, addr, len, kind));
        self.apply_watches();
        Some(id)
    }
    fn clear_watchpoint(&mut self, id: WatchId) -> bool {
        let before = self.watch_specs.len();
        self.watch_specs.retain(|(w, ..)| *w != id);
        let removed = self.watch_specs.len() != before;
        if removed {
            self.apply_watches();
        }
        removed
    }
    fn backtrace(&self) -> Vec<FrameInfo> {
        let mut out = Vec::new();
        // The focused thread's stack (single-vCPU: the sole thread; scheduled: `select_task`'s pick).
        let depth = match &self.engine {
            Engine::Single(run) => run.depth(),
            Engine::Threaded(run) => run.depth(),
        };
        for d in 0..depth {
            let pc = match &self.engine {
                Engine::Single(run) => run.frame_pc(d),
                Engine::Threaded(run) => run.frame_pc(d),
            };
            if let Some(pc) = pc {
                let source = svm_interp::source_loc(&self.module, pc);
                out.push(FrameInfo {
                    pc,
                    vals: Vec::new(), // unused by DapServer (it reads via read_var)
                    source,
                });
            }
        }
        out
    }
    fn func_name(&self, func: FuncIdx) -> Option<&str> {
        svm_interp::func_name(&self.module, func)
    }
    fn source_loc(&self, pc: IrPc) -> Option<SourceLoc> {
        svm_interp::source_loc(&self.module, pc)
    }
    fn read_var(&self, frame_from_top: usize, name: &str, width: usize) -> Option<VarValue> {
        match &self.engine {
            Engine::Single(run) => run.read_var(frame_from_top, name, width),
            Engine::Threaded(run) => run.read_var(frame_from_top, name, width),
        }
    }
    fn var_addr(&self, frame_from_top: usize, name: &str) -> Option<u64> {
        match &self.engine {
            Engine::Single(run) => run.var_addr(frame_from_top, name),
            Engine::Threaded(run) => run.var_addr(frame_from_top, name),
        }
    }
    fn read_window(&self, addr: u64, len: usize) -> Result<Vec<u8>, Trap> {
        match &self.engine {
            Engine::Single(run) => run.read_window(addr, len),
            Engine::Threaded(run) => run.read_window(addr, len),
        }
    }
    fn threads(&self) -> Vec<u64> {
        match &self.engine {
            // Single-vCPU: the sole task is 0 (DAP thread 1).
            Engine::Single(_) => vec![0],
            Engine::Threaded(run) => run.threads(),
        }
    }
    fn select_task(&mut self, id: u64) -> bool {
        match &mut self.engine {
            Engine::Single(_) => id == 0,
            Engine::Threaded(run) => run.select_task(id),
        }
    }
    fn stopped_task(&self) -> Option<u64> {
        match &self.engine {
            Engine::Single(_) => Some(0),
            Engine::Threaded(run) => run.stopped_task(),
        }
    }
    fn turn(&self) -> u64 {
        match &self.engine {
            // Single-vCPU: no scheduler turns; the op `clock` is the time coordinate.
            Engine::Single(_) => 0,
            Engine::Threaded(run) => run.turn(),
        }
    }
    fn clock(&self) -> u64 {
        match &self.engine {
            Engine::Single(run) => run.op_clock(),
            Engine::Threaded(run) => run.turn(),
        }
    }
    // Both engines are fully reversible (deterministic replay) and watch-capable: the single-vCPU op
    // `clock` and the multithreaded global `turn` are the two time coordinates.
    fn supports_reverse(&self) -> bool {
        true
    }
    fn supports_watch(&self) -> bool {
        true
    }
}
