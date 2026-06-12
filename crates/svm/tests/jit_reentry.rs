//! JIT.md Phase 2a: **mid-run re-entry** into a live [`CompiledModule`] — the engine of the
//! guest-driven `Jit` capability (Model A), exercised below the capability layer with a
//! custom cap thunk standing in for the Phase-2b/2c host plumbing.
//!
//! What is established here:
//! - A `cap.call` handler can `define_extra` **while the guest is executing** (suspended in
//!   its synchronous cap.call): the incremental `finalize_definitions` runs with parent code
//!   live on the stack below the handler and that code returns into correctly — the mid-run
//!   form of the Phase-1 W^X spike, the strongest one.
//! - The handler can `invoke_extra` the new code **over the live window**: the invoked code
//!   reads/writes the guest's own memory in place (observed in the run's final snapshot) and
//!   its results marshal back through the cap.call result slots.
//! - Trap semantics are **terminal for the domain**: an IR trap in invoked code propagates
//!   through the guest's trap cell (the guest's cap.call propagation check unwinds), and a
//!   memory fault in invoked code is caught by the **nested** detect-and-kill recovery,
//!   reported as `MemoryFault` — and the host survives both (the module stays reusable).
//! - `invoke_extra` outside an in-flight run is rejected fail-closed.

use core::cell::Cell;
use core::ffi::c_void;
use svm_ir::{Func, DEFAULT_RESERVED_LOG2};
use svm_jit::{CompiledModule, JitOutcome, Quota, TrapKind};
use svm_text::parse_module;
use svm_verify::verify_module;

/// The parent guest: entry `(a, b)` forwards both args through `cap.call` (iface 100, op 0)
/// and returns the capability's result. Declares the 64 KiB window the invoked code shares.
const PARENT: &str = "memory 16\nfunc (i32, i32) -> (i32) {\nblock0(v0: i32, v1: i32):\n  v2 = i32.const 0\n  v3 = cap.call 100 0 (i32, i32) -> (i32) v2 (v0, v1)\n  return v3\n}\n";

/// Test-side stand-in for the Phase-2 `Jit` binding: the thunk context carrying the live
/// module pointer (set after `compile`, same provenance as the `run_raw` pointer) and the
/// pre-parsed unit to define on first use.
struct TestCtx {
    cm: Cell<*mut CompiledModule>,
    funcs: Vec<Func>,
    /// Cached trampoline from the first mid-run `define_extra` (later calls reuse it).
    code: Cell<*const u8>,
}

/// The stand-in handler: on first call, `define_extra` the unit **mid-run**; then
/// `invoke_extra` it over the live window, marshalling the guest's cap.call args/results.
unsafe extern "C" fn reentry_thunk(
    ctx: *mut c_void,
    mem_base: *mut u8,
    _mem_size: u64,
    _mem_reserved: u64,
    _type_id: u32,
    _op: u32,
    _handle: i32,
    args: *const i64,
    n_args: u64,
    results: *mut i64,
    n_results: u64,
    trap_out: *mut i64,
) {
    let tc = &*(ctx as *const TestCtx);
    let cm = tc.cm.get();
    if tc.code.get().is_null() {
        // Re-entrant incremental compile: the guest is suspended on this thread; the parent's
        // code is on the stack below us while finalize_definitions mprotects the new pages.
        let ptrs = (*cm).define_extra(&tc.funcs).expect("define_extra mid-run");
        tc.code.set(ptrs[0].tramp);
    }
    let args = std::slice::from_raw_parts(args, n_args as usize);
    let results = std::slice::from_raw_parts_mut(results, n_results as usize);
    CompiledModule::invoke_extra(cm, tc.code.get(), args, results, mem_base, trap_out)
        .expect("invoke_extra during a live run");
}

/// Compile the parent with the re-entry thunk armed, wire the module pointer into the thunk
/// ctx, and return both (the ctx box must outlive the module's runs).
fn setup(extra_src: &str) -> (Box<TestCtx>, Box<CompiledModule>) {
    let parent = parse_module(PARENT).expect("parse parent");
    verify_module(&parent).expect("verify parent");
    let extra = parse_module(extra_src).expect("parse extra");
    verify_module(&extra).expect("verify extra");
    let ctx = Box::new(TestCtx {
        cm: Cell::new(core::ptr::null_mut()),
        funcs: extra.funcs,
        code: Cell::new(core::ptr::null()),
    });
    let cm = CompiledModule::compile(
        &parent,
        0,
        reentry_thunk,
        &*ctx as *const TestCtx as *mut c_void,
        DEFAULT_RESERVED_LOG2,
        None,
        None,
        None,
        None,
        Quota::default(),
        0,
    )
    .expect("compile parent");
    let mut cm = Box::new(cm);
    ctx.cm.set(&mut *cm);
    (ctx, cm)
}

/// The full Model A loop below the capability layer: guest cap.calls → handler defines the
/// unit mid-run → invokes it over the live window → result returns through the cap.call.
/// The invoked code's store lands in the guest's own window (visible in the run snapshot).
#[test]
fn mid_run_define_and_invoke_over_live_window() {
    // (a, b) -> a + b + 1000, plus a store of 0xAB at offset 64 of the SHARED window.
    let extra_src = "memory 16\nfunc (i32, i32) -> (i32) {\nblock0(v0: i32, v1: i32):\n  v2 = i64.const 64\n  v3 = i32.const 171\n  i32.store8 v2 v3\n  v4 = i32.add v0 v1\n  v5 = i32.const 1000\n  v6 = i32.add v4 v5\n  return v6\n}\n";
    let (ctx, _cm) = setup(extra_src);
    let cm_ptr: *mut CompiledModule = ctx.cm.get();
    let (out, final_mem) =
        unsafe { CompiledModule::run_raw(cm_ptr, &[7, 35], None, None, None) }.expect("run");
    assert!(
        matches!(out, JitOutcome::Returned(ref s) if s == &[1042]),
        "{out:?}"
    );
    assert_eq!(
        final_mem[64], 0xab,
        "invoked code must write the LIVE window"
    );

    // The module survives: run again — the unit is already defined (cached), invoke again.
    let (out, final_mem) =
        unsafe { CompiledModule::run_raw(cm_ptr, &[1, 1], None, None, None) }.expect("re-run");
    assert!(matches!(out, JitOutcome::Returned(ref s) if s == &[1002]));
    assert_eq!(final_mem[64], 0xab);
}

/// An IR trap inside invoked code is **terminal for the domain**: the trampoline writes the
/// guest's trap cell, the handler returns, and the guest's cap.call propagation check unwinds
/// the whole run as `Unreachable`. The host survives.
#[test]
fn trap_in_invoked_code_kills_the_domain() {
    let extra_src =
        "memory 16\nfunc (i32, i32) -> (i32) {\nblock0(v0: i32, v1: i32):\n  unreachable\n}\n";
    let (ctx, _cm) = setup(extra_src);
    let cm_ptr = ctx.cm.get();
    let (out, _) =
        unsafe { CompiledModule::run_raw(cm_ptr, &[1, 2], None, None, None) }.expect("run");
    assert!(
        matches!(out, JitOutcome::Trapped(TrapKind::Unreachable)),
        "{out:?}"
    );
}

/// A memory fault inside invoked code (store past the backed extent, into the reserved tail)
/// is caught by `invoke_extra`'s **nested** recovery, reported as `MemoryFault` through the
/// guest's trap cell, and the host survives — the module even runs again afterwards.
#[test]
fn memory_fault_in_invoked_code_is_caught_and_terminal() {
    let extra_src = "memory 16\nfunc (i32, i32) -> (i32) {\nblock0(v0: i32, v1: i32):\n  v2 = i64.const 1048584\n  v3 = i32.const 1\n  i32.store8 v2 v3\n  v4 = i32.const 0\n  return v4\n}\n";
    let (ctx, _cm) = setup(extra_src);
    let cm_ptr = ctx.cm.get();
    let (out, _) =
        unsafe { CompiledModule::run_raw(cm_ptr, &[1, 2], None, None, None) }.expect("run");
    assert!(
        matches!(out, JitOutcome::Trapped(TrapKind::MemoryFault)),
        "{out:?}"
    );
    // Recovery: the same module runs again on a fresh window and dies the same way — the
    // nested guard restored the outer recovery state correctly.
    let (out, _) =
        unsafe { CompiledModule::run_raw(cm_ptr, &[3, 4], None, None, None) }.expect("re-run");
    assert!(matches!(out, JitOutcome::Trapped(TrapKind::MemoryFault)));
}

/// `invoke_extra` with no run in flight is rejected fail-closed (there is no live window to
/// confine the code against).
#[test]
fn invoke_outside_a_run_is_rejected() {
    let extra_src = "memory 16\nfunc (i32, i32) -> (i32) {\nblock0(v0: i32, v1: i32):\n  v2 = i32.add v0 v1\n  return v2\n}\n";
    let (ctx, mut cm) = setup(extra_src);
    let code = cm.define_extra(&ctx.funcs).expect("define outside run")[0].tramp;
    let mut results = [0i64; 1];
    let mut trap = 0i64;
    let err = unsafe {
        CompiledModule::invoke_extra(
            ctx.cm.get(),
            code,
            &[1, 2],
            &mut results,
            core::ptr::null_mut(),
            &mut trap,
        )
    };
    assert!(err.is_err(), "invoke outside an in-flight run must fail");
}
