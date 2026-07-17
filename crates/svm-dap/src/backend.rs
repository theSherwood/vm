//! The **debuggee backend seam** (DEBUGGING.md G3). `DapServer` was hard-wired to the tree-walking
//! `svm_interp::Inspector`; this trait lets the same server drive *either* engine — the tree-walker
//! (the reference oracle, full feature set) or the **bytecode VM** the browser playground actually
//! runs, over `svm_interp::bytecode::DebugRun`.
//!
//! The bytecode engine covers the **forward-debug subset today** (breakpoints, stepping, backtrace,
//! scalar/aggregate inspection). Reverse debugging (`seek`/`step_back`), data breakpoints
//! (`set_watchpoint`), and multithreading are **not yet built on it** — gated off by `supports_*` so
//! `DapServer` refuses them cleanly rather than returning a wrong answer. They are *not* delegated to
//! the tree-walker: it is the differential oracle only (and far too slow to sit on any user-facing
//! path). The direction is to build all three **on the bytecode engine** — reverse via deterministic
//! replay (+ checkpoints), watchpoints via a per-op watched-range check in the debug-stepping loop,
//! and multithreading via a deterministic cooperative multi-vCPU *debug scheduler* (see DEBUGGING.md
//! G3). Correctness of the shared subset is guaranteed by `crates/svm/tests/debug_parity.rs`, which
//! proves the two engines report identical stop locations, per-frame locals, and results.

use svm_interp::bytecode::DebugRun;
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

/// The **bytecode backend** — a `DebugRun` (the resumable bytecode debug session) plus the persistent
/// breakpoint set and fuel `DapServer` expects, and the module (for `source_loc`/`func_name`, which
/// are engine-neutral free functions keyed on the `IrPc`). Forward-debug only.
pub struct BytecodeBackend {
    run: DebugRun,
    module: Module,
    breakpoints: Vec<IrPc>,
    fuel: u64,
}

impl BytecodeBackend {
    /// Open a bytecode debug session on `module`'s `func(args)`. `None` if the module is outside the
    /// bytecode engine's debug subset (e.g. multi-vCPU — `DebugRun::new` declines it).
    pub fn new(
        module: Module,
        func: FuncIdx,
        args: &[Value],
        fuel: u64,
    ) -> Option<BytecodeBackend> {
        let run = DebugRun::new(&module, func, args)?;
        Some(BytecodeBackend {
            run,
            module,
            breakpoints: Vec::new(),
            fuel,
        })
    }

    /// Map a `DebugRun` completion (`run_to`/`step` returned `None`) to a `Stop`: the finished result
    /// (or trap) if the run is done, else `Blocked` (a concurrency seam the bytecode debugger can't
    /// follow).
    fn finish_stop(&self) -> Stop {
        match self.run.result() {
            Some(r) => Stop::Finished(r.clone()),
            None => Stop::Blocked,
        }
    }
}

impl Debuggee for BytecodeBackend {
    fn run_until_stop(&mut self) -> Stop {
        match self.run.run_to(&self.breakpoints, &mut self.fuel) {
            Some(pc) => Stop::Break {
                reason: StopReason::Breakpoint,
                pc,
            },
            None => self.finish_stop(),
        }
    }
    fn step(&mut self) -> Stop {
        match self.run.step(&mut self.fuel) {
            Some(pc) => Stop::Break {
                reason: StopReason::Step,
                pc,
            },
            None => self.finish_stop(),
        }
    }
    fn step_over(&mut self) -> Stop {
        match self.run.step_over(&mut self.fuel) {
            Some(pc) => Stop::Break {
                reason: StopReason::Step,
                pc,
            },
            None => self.finish_stop(),
        }
    }
    fn step_out(&mut self) -> Stop {
        match self.run.step_out(&mut self.fuel) {
            Some(pc) => Stop::Break {
                reason: StopReason::Step,
                pc,
            },
            None => self.finish_stop(),
        }
    }
    // Reverse debugging is out of the bytecode engine's single-vCPU scope (gated off — never called).
    fn step_back(&mut self) -> Stop {
        Stop::Blocked
    }
    fn seek(&mut self, _t: u64) -> Stop {
        Stop::Blocked
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
    // No watchpoints on the bytecode engine (gated off) — `None` ⇒ the server reports unverified.
    fn set_watchpoint(&mut self, _addr: u64, _len: u64, _kind: WatchKind) -> Option<WatchId> {
        None
    }
    fn clear_watchpoint(&mut self, _id: WatchId) -> bool {
        false
    }
    fn backtrace(&self) -> Vec<FrameInfo> {
        let mut out = Vec::new();
        for depth in 0..self.run.depth() {
            if let Some(pc) = self.run.frame_pc(depth) {
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
        self.run.read_var(frame_from_top, name, width)
    }
    fn var_addr(&self, frame_from_top: usize, name: &str) -> Option<u64> {
        self.run.var_addr(frame_from_top, name)
    }
    fn read_window(&self, addr: u64, len: usize) -> Result<Vec<u8>, Trap> {
        self.run.read_window(addr, len)
    }
    // Single-vCPU: the sole task is 0 (DAP thread 1); `select_task` succeeds only for it.
    fn threads(&self) -> Vec<u64> {
        vec![0]
    }
    fn select_task(&mut self, id: u64) -> bool {
        id == 0
    }
    fn stopped_task(&self) -> Option<u64> {
        Some(0)
    }
    fn turn(&self) -> u64 {
        0
    }
    fn clock(&self) -> u64 {
        0
    }
    fn supports_reverse(&self) -> bool {
        false
    }
    fn supports_watch(&self) -> bool {
        false
    }
}
