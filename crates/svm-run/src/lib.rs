//! `svm-run` — the **embedding runtime**: instantiate a verified module with the MVP powerbox
//! (§3e) and run it on the Cranelift JIT, returning its outcome and the bytes it wrote.
//!
//! This is the single, reusable host glue — the `cap.call` trampoline ([`cap_thunk`]) plus the
//! powerbox grant ([`run_powerbox`]) — that was previously copy-pasted across the JIT test
//! harnesses (`c_frontend.rs`, `jit_diff.rs`). The `svm-run` **CLI** is a thin wrapper over it.
//!
//! It is *not* escape-TCB: the verifier (run before this) is what makes a module safe to run;
//! this crate only wires the host capabilities a guest is granted. A guest that traps
//! (out-of-window fault, `unreachable`, …) is **detect-and-killed** (§5) — surfaced here as an
//! `Err`, never undefined behaviour in the host.

use core::ffi::c_void;

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};
use svm_interp::{
    iface, run_capture_reserved_with_host, run_with_host, run_with_host_fast, AsyncCounter,
    CapPageMap, GuestMem, Host, HostFn, RegionBacking, StreamRole, Trap,
};
// `SharedBacking` is implemented by the per-OS shared-mapping backing (unix `ShmBacking`, windows
// `WinShmBacking`) the JIT aliases into the window for §13.
#[cfg(any(unix, windows))]
use svm_interp::SharedBacking;
use svm_ir::{FuncIdx, Module, Resolved, ValType};

// Re-export the value type + the §15 spawn quota so embedders (and the CLI) need not also depend on
// `svm-interp`.
pub use svm_interp::{Quota, Value};
use svm_jit::{compile_and_run, CompiledModule, JitFrameLoc, JitOutcome, TrapKind, EXIT_CODE};
pub use svm_peval::{SpecArg, SpecConfig};

/// Render a JIT trap-time backtrace (§5 W3) for a kill message — `\n    #i file:line:col in <name>`
/// per frame, innermost first, where `<name>` is the `-g` function name or the synthesized `fn{N}`.
/// Empty string when there are no frames (the module carried no `-g`), so the kill message is
/// byte-identical to before in that case.
fn format_backtrace(frames: &[JitFrameLoc]) -> String {
    if frames.is_empty() {
        return String::new();
    }
    let mut s = String::from("\n  backtrace (innermost first):");
    for (i, f) in frames.iter().enumerate() {
        let name = f
            .func_name
            .clone()
            .unwrap_or_else(|| format!("fn{}", f.func));
        s.push_str(&format!(
            "\n    #{i} {}:{}:{} in {name}",
            f.file, f.line, f.col
        ));
    }
    s
}

/// Options for the CLI `--specialize` path — the §20c first Futamura projection driven from the
/// command line.
#[derive(Clone, Debug, Default)]
pub struct SpecializeOpts {
    /// Which function to specialize (the residual's entry, index 0). Default `0`.
    pub func: u32,
    /// Per-parameter binding (a static constant or `Dynamic`), in parameter order.
    pub args: Vec<SpecArg>,
    /// Window ranges `[lo, hi)` the caller promises are constant at specialization time.
    pub const_regions: Vec<(u64, u64)>,
    /// A private, zero-initialized rename region (the interpreter's value-stack / locals) to lift
    /// into SSA and elide (Stage 2).
    pub rename: Option<(u64, u64)>,
    /// Promise the rename region is private (lets a dynamic-address heap coexist with it).
    pub rename_private: bool,
    /// Run the generic cleanup optimizer (fold / DCE / block-merge) on the residual.
    pub optimize: bool,
    /// Outline calls into shared residual functions instead of inlining them (a multi-function
    /// residual). Bounds code growth and specializes dynamic-depth recursion; requires no rename
    /// region (see [`svm_peval::SpecConfig::outline_calls`]).
    pub outline: bool,
    /// Selective outlining: inline the leaves/structure and outline **only** unbounded-recursion
    /// back-edges — a tight recursive residual rather than one function per call site (see
    /// [`svm_peval::SpecConfig::selective_outline`]). Requires no rename region.
    pub selective_outline: bool,
}

/// Specialize `module`'s entry against `opts` and re-verify the residual. The specializer is
/// untrusted-for-escape (§20c) like any frontend output, so [`svm_verify::verify_module`] is the
/// gate: a specializer bug is a clean verify error here, never an escape. Returns the residual — a
/// single function (index 0) whose parameters are the dynamic args, in order.
pub fn specialize_module(module: &Module, opts: &SpecializeOpts) -> Result<Module, String> {
    let nparams = module
        .funcs
        .get(opts.func as usize)
        .ok_or(format!(
            "func {} is out of range ({} functions)",
            opts.func,
            module.funcs.len()
        ))?
        .params
        .len();
    if opts.args.len() > nparams {
        return Err(format!(
            "{} argument binding(s) given for a {nparams}-parameter function",
            opts.args.len()
        ));
    }
    // Parameters without an explicit binding default to dynamic.
    let mut args = opts.args.clone();
    args.resize(nparams, SpecArg::Dynamic);

    let cfg = SpecConfig {
        rename: opts.rename,
        const_regions: opts.const_regions.clone(),
        rename_is_private: opts.rename_private,
        outline_calls: opts.outline,
        selective_outline: opts.selective_outline,
        ..SpecConfig::default()
    };
    let residual = svm_peval::specialize_with_config(module, opts.func, &args, &cfg)
        .map_err(|e| format!("specialization failed: {e:?}"))?;
    svm_verify::verify_module(&residual)
        .map_err(|e| format!("specialized residual failed re-verification: {e:?}"))?;
    if !opts.optimize {
        return Ok(residual);
    }
    let opt = svm_peval::optimize_module(&residual);
    svm_verify::verify_module(&opt)
        .map_err(|e| format!("optimized residual failed re-verification: {e:?}"))?;
    Ok(opt)
}

/// Default `call_indirect` table reservation for the CLI powerbox (`2^10` = 1024 slots) so a
/// guest using the `Jit` capability can `install` units (DESIGN.md §22). Embedders pick their
/// own via [`grant_jit`] + the compile `table_reserve_log2`.
const CLI_JIT_TABLE_LOG2: u8 = 10;

/// The host trampoline bridging the JIT's [`svm_jit::CapThunk`] ABI (§9) to the reference
/// [`Host`]'s capability dispatch — the host code a real embedder supplies. One shared copy.
///
/// # Safety
/// Honours the `CapThunk` contract: `ctx` is a live `*mut Host`; `args`/`results` are valid for
/// `n_args`/`n_results`; `mem_base` (when non-null) is the guest window with `mem_size` backed
/// bytes inside a `mem_reserved` reservation; `trap_out` is writable. The trap cell is encoded as
/// the JIT expects: `0` = ok, a [`TrapKind`] for a fault, or `EXIT_CODE | (code << 32)` for `Exit`.
pub unsafe extern "C" fn cap_thunk(
    ctx: *mut c_void,
    mem_base: *mut u8,
    mem_size: u64,
    mem_reserved: u64,
    type_id: u32,
    op: u32,
    handle: i32,
    args: *const i64,
    n_args: u64,
    results: *mut i64,
    n_results: u64,
    trap_out: *mut i64,
) {
    let host = &mut *(ctx as *mut Host);
    // The JIT passes a null args/results pointer when the count is 0; `from_raw_parts` requires a
    // non-null (aligned) pointer even for an empty slice, so use `&[]` in that case (UB otherwise).
    let arg_slots = if n_args == 0 {
        &[][..]
    } else {
        std::slice::from_raw_parts(args, n_args as usize)
    };
    // The guest window with a real hardware-protected Memory capability (`map`/`unmap`/`protect`,
    // incl. growth into the reserved tail): `mprotect` on unix, `VirtualProtect`/`VirtualAlloc` on
    // windows — the same software-page-map model, only the syscalls differ. The page map is the
    // **per-run** one from the `Host` (keyed by window base), so growth committed in an earlier
    // `cap.call` is still seen committed here — a borrow of a guest-grown page doesn't fail-closed.
    #[cfg(any(unix, windows))]
    let pages = host.cap_window_pages(mem_base as usize);
    #[cfg(any(unix, windows))]
    let mut wm = MprotectWindow::new_shared(mem_base, mem_size, mem_reserved, pages);
    #[cfg(any(unix, windows))]
    let gm: Option<&mut dyn GuestMem> = if mem_base.is_null() {
        None
    } else {
        Some(&mut wm)
    };
    // Any other target has no window backend (the JIT, `svm-jit`, does not build there anyway).
    #[cfg(not(any(unix, windows)))]
    let gm: Option<&mut dyn GuestMem> = {
        let _ = (mem_base, mem_size, mem_reserved);
        None
    };
    // Guest-driven `Jit` (iface 11, DESIGN.md §22): serviced natively here, not in the generic
    // Host dispatch — `compile` must call into Cranelift (`define_extra` on the live
    // `CompiledModule`) and `invoke` must call the unit's trampoline over the live window,
    // neither of which `svm-interp` can (or should) reach. The interpreter backend services the
    // same iface in its eval loop; both share the Host-side state and validator, so they stay in
    // differential lockstep.
    if type_id == iface::JIT {
        jit_native_op(
            host, op, handle, arg_slots, results, n_results, trap_out, gm, mem_base,
        );
        return;
    }
    match host.cap_dispatch_slots(type_id, op, handle, arg_slots, gm) {
        Ok(res) => {
            if n_results != 0 {
                let out = std::slice::from_raw_parts_mut(results, n_results as usize);
                for (o, r) in out.iter_mut().zip(res) {
                    *o = r;
                }
            }
            *trap_out = 0;
        }
        Err(Trap::Exit(code)) => *trap_out = EXIT_CODE as i64 | ((code as i64) << 32),
        Err(_) => *trap_out = TrapKind::CapFault as i64,
    }
}

/// **Multi-threaded `cap.call` thunk** (DESIGN.md §22 threaded *compile*): the same dispatch as
/// [`cap_thunk`], but serialized through a per-domain [`Mutex<Host>`] so a guest whose worker
/// threads make concurrent `cap.call`s (notably `Jit.compile`, which mutates the `Host` unit
/// registry and the live `CompiledModule`) does not data-race. `ctx` is `*const Mutex<Host>`
/// (vs `cap_thunk`'s raw `*mut Host`), so single-threaded guests keep the unlocked `cap_thunk`
/// and pay nothing; only a concurrent guest's run bakes *this* thunk (see [`jit_cap_run`]).
///
/// **Re-entrancy** (the "running units compile more" case): `Jit.invoke` runs guest code that may
/// itself `cap.call` (e.g. compile more) on the same thread, so the lock must **not** be held
/// across it — invoke reads the unit under the lock, *releases*, then trampolines. Every other op
/// is host-side only (the §14 Instantiator / fibers re-enter via their own runtimes, never through
/// here), so holding the lock across a plain delegate to [`cap_thunk`] is deadlock-free.
///
/// # Safety
/// Same contract as [`cap_thunk`]; additionally `ctx` is a live `*const Mutex<Host>` whose `Host`
/// has the in-flight run's `Jit` native ctx registered, and the lock is uncontended-safe to take
/// from any of the run's vCPU threads.
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn cap_thunk_locked(
    ctx: *mut c_void,
    mem_base: *mut u8,
    mem_size: u64,
    mem_reserved: u64,
    type_id: u32,
    op: u32,
    handle: i32,
    args: *const i64,
    n_args: u64,
    results: *mut i64,
    n_results: u64,
    trap_out: *mut i64,
) {
    let m = &*(ctx as *const Mutex<Host>);
    // `Jit.invoke` (iface 11 op 1) re-enters guest code → never hold the lock across it.
    if type_id == iface::JIT && op == 1 {
        let arg_slots = if n_args == 0 {
            &[][..]
        } else {
            std::slice::from_raw_parts(args, n_args as usize)
        };
        jit_invoke_locked(m, handle, arg_slots, results, n_results, trap_out, mem_base);
        return;
    }
    // Everything else is host-side only: hold the lock across a plain delegate to the unlocked
    // thunk over the locked `Host`'s pointer (compile/install/uninstall/release mutate the unit
    // registry + the live module; the generic ops mutate `Host` state). The guard is released on
    // return.
    let mut guard = m.lock().unwrap_or_else(|e| e.into_inner());
    let host_ptr = &mut *guard as *mut Host as *mut c_void;
    cap_thunk(
        host_ptr,
        mem_base,
        mem_size,
        mem_reserved,
        type_id,
        op,
        handle,
        args,
        n_args,
        results,
        n_results,
        trap_out,
    );
}

/// `Jit.invoke` for the [`cap_thunk_locked`] path: resolve the unit **under the lock**, then
/// **release** before running its trampoline (`invoke_extra`), so the invoked unit may itself
/// `cap.call` (e.g. compile more) on this thread without self-deadlocking and other threads keep
/// making progress while it runs. Mirrors [`jit_native_op`]'s op-1 arm exactly, minus the lock
/// scope.
///
/// # Safety
/// As [`cap_thunk_locked`].
unsafe fn jit_invoke_locked(
    m: &Mutex<Host>,
    handle: i32,
    args: &[i64],
    results: *mut i64,
    n_results: u64,
    trap_out: *mut i64,
    mem_base: *mut u8,
) {
    let cap_fault = |trap_out: *mut i64| *trap_out = TrapKind::CapFault as i64;
    // Read the target unit + its module pointer under the lock, then drop the guard.
    let resolved: Option<(usize, usize)> = {
        let host = m.lock().unwrap_or_else(|e| e.into_inner());
        (|| {
            let domain = host.resolve_jit_domain(handle).ok()?;
            let &ch = args.first()?;
            let (cd, cu) = host.resolve_jit_code(ch as i32).ok()?;
            if cd != domain {
                return None;
            }
            let code = host.jit_unit_native(cd, cu);
            let cm = host.jit_native_ctx(cd);
            let funcs = host.jit_unit_funcs(cd, cu)?;
            if code == 0 || cm == 0 {
                return None;
            }
            let entry = &funcs[0];
            if args.len() - 1 != entry.params.len() || n_results as usize != entry.results.len() {
                return None;
            }
            Some((code, cm))
        })()
    };
    let Some((code, cm)) = resolved else {
        return cap_fault(trap_out);
    };
    let out: &mut [i64] = if n_results == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(results, n_results as usize)
    };
    // SAFETY: lock released; `cm` is the in-flight run's CompiledModule, `code` its unit's
    // finalized trampoline; arity checked above; a nested `cap.call` (e.g. compile) from the
    // invoked code re-acquires the lock on this thread.
    if CompiledModule::invoke_extra(
        cm as *mut CompiledModule,
        code as *const u8,
        &args[1..],
        out,
        mem_base,
        trap_out,
    )
    .is_err()
    {
        cap_fault(trap_out);
    }
}

/// The native (Cranelift) half of the guest-driven `Jit` capability (DESIGN.md §22), reached
/// from [`cap_thunk`]'s iface-11 intercept. Op semantics — including every fail-closed path —
/// mirror the interpreter reference (`svm-interp`'s `Binding::JitDomain` dispatch arm + its
/// eval-loop `invoke`) exactly, so the two backends agree on results, errnos, and traps:
/// - op 0 `compile(ptr, len)`: borrow the blob, run the shared `Host::jit_compile` (the
///   injected validator gate), then **additionally** compile the unit into the live
///   [`CompiledModule`] (`define_extra`) and register its trampoline. Any failure leaves
///   nothing installed.
/// - op 1 `invoke(code_handle, args…)`: strict-arity call of the unit's trampoline over the
///   **live window** (`invoke_extra`); a trap in the invoked code lands in `trap_out` (the
///   run's trap cell) — terminal for the domain.
/// - op 2 `release(code_handle)`: revoke the handle (non-fatal `-EINVAL` if forged/closed).
///
/// # Safety
/// Called from [`cap_thunk`] (same contract); additionally, when a `Jit` domain has a native
/// ctx registered it must be the `*mut CompiledModule` of the **in-flight run on this thread**
/// (see [`jit_cap_run`]), so the transient re-entry here aliases no live reference
/// (`CompiledModule::run_raw`'s contract).
#[allow(clippy::too_many_arguments)]
unsafe fn jit_native_op(
    host: &mut Host,
    op: u32,
    handle: i32,
    args: &[i64],
    results: *mut i64,
    n_results: u64,
    trap_out: *mut i64,
    mut gm: Option<&mut dyn GuestMem>,
    mem_base: *mut u8,
) {
    // Negative-errno results (§3e D42), matching `svm-interp`'s private consts.
    const EINVAL: i64 = -22;
    const EFAULT: i64 = -14;
    // One errno/handle result slot + a clean trap cell — the compile/release result shape.
    let put = |results: *mut i64, n_results: u64, v: i64, trap_out: *mut i64| {
        if n_results != 0 {
            *results = v;
        }
        *trap_out = 0;
    };
    let cap_fault = |trap_out: *mut i64| *trap_out = TrapKind::CapFault as i64;
    match op {
        0 => {
            // compile(ptr, len) -> code_handle | -errno.
            let Ok(domain) = host.resolve_jit_domain(handle) else {
                return cap_fault(trap_out);
            };
            let cm = host.jit_native_ctx(domain) as *mut CompiledModule;
            if cm.is_null() {
                // No live module registered (host wiring bug) — fail closed, non-fatally.
                return put(results, n_results, EINVAL, trap_out);
            }
            let ptr = *args.first().unwrap_or(&0) as u64;
            let len = *args.get(1).unwrap_or(&0) as u64;
            let Some(bytes) = gm.as_mut().and_then(|m| m.read_bytes(ptr, len)) else {
                return put(results, n_results, EFAULT, trap_out);
            };
            let compiled = match host.jit_compile(handle, &bytes) {
                Ok(Ok(c)) => c,
                Ok(Err(e)) => return put(results, n_results, e, trap_out),
                Err(_) => return cap_fault(trap_out),
            };
            let funcs = host
                .jit_unit_funcs(compiled.domain, compiled.unit)
                .expect("unit was just stored");
            // SAFETY: `cm` is the in-flight run's CompiledModule (jit_cap_run registered it);
            // the guest is suspended in this synchronous cap.call, so this transient re-entry
            // aliases no live reference (run_raw's contract).
            match (*cm).define_extra(&funcs) {
                Ok(defs) => {
                    // The unit entry (func 0): trampoline for `invoke`, natural code + type_id
                    // for B2 `install` into the call_indirect table.
                    let d = defs[0];
                    host.set_jit_unit_native(
                        compiled.domain,
                        compiled.unit,
                        d.tramp as usize,
                        d.code as usize,
                        d.type_id,
                    );
                    put(results, n_results, compiled.handle as i64, trap_out);
                }
                Err(_) => {
                    // Verified but not natively lowerable (a backend gap): revoke the minted
                    // handle so nothing half-installed is guest-reachable; non-fatal errno.
                    let _ = host.jit_release(compiled.handle);
                    put(results, n_results, EINVAL, trap_out);
                }
            }
        }
        1 => {
            // invoke(code_handle, args…) -> results.
            let Ok(domain) = host.resolve_jit_domain(handle) else {
                return cap_fault(trap_out);
            };
            let Some(&ch) = args.first() else {
                return cap_fault(trap_out);
            };
            let Ok((cd, cu)) = host.resolve_jit_code(ch as i32) else {
                return cap_fault(trap_out);
            };
            // A code handle is only valid on the domain that compiled it.
            if cd != domain {
                return cap_fault(trap_out);
            }
            let code = host.jit_unit_native(cd, cu);
            let cm = host.jit_native_ctx(cd) as *mut CompiledModule;
            let Some(funcs) = host.jit_unit_funcs(cd, cu) else {
                return cap_fault(trap_out);
            };
            if code == 0 || cm.is_null() {
                return cap_fault(trap_out);
            }
            // Strict arity vs the unit's entry (parity with the interp eval arm): the invoke
            // args are the cap.call args minus the code handle; results must match exactly.
            let entry = &funcs[0];
            if args.len() - 1 != entry.params.len() || n_results as usize != entry.results.len() {
                return cap_fault(trap_out);
            }
            let out: &mut [i64] = if n_results == 0 {
                &mut []
            } else {
                std::slice::from_raw_parts_mut(results, n_results as usize)
            };
            // SAFETY: an in-flight run on this thread (we are inside its cap.call); `code` is
            // the unit's finalized trampoline; arity checked above; `mem_base`/`trap_out` are
            // the live run's window base and trap cell. On a clean return the cell stays 0; on
            // a trap the trampoline / nested guard wrote it — either way it holds the truth.
            if CompiledModule::invoke_extra(
                cm,
                code as *const u8,
                &args[1..],
                out,
                mem_base,
                trap_out,
            )
            .is_err()
            {
                cap_fault(trap_out);
            }
        }
        2 => {
            // release(code_handle) -> 0 | -EINVAL (forged/double release is non-fatal: it is
            // guest-reachable in a loop and must not kill the domain).
            if host.resolve_jit_domain(handle).is_err() {
                return cap_fault(trap_out);
            }
            let ch = *args.first().unwrap_or(&0) as i32;
            let v = match host.jit_release(ch) {
                Ok(()) => 0,
                Err(_) => EINVAL,
            };
            put(results, n_results, v, trap_out);
        }
        3 => {
            // install(code_handle) -> slot_index | -errno (DESIGN.md §22): write the unit's
            // natural entry + interned type_id into the live fn_table's next padding slot. The
            // slot index agrees with the interpreter's (both fill from the parent's funcs count).
            const ENOSPC: i64 = -28;
            let Ok(domain) = host.resolve_jit_domain(handle) else {
                return cap_fault(trap_out);
            };
            let Some(&ch) = args.first() else {
                return cap_fault(trap_out);
            };
            let Ok((cd, cu)) = host.resolve_jit_code(ch as i32) else {
                return cap_fault(trap_out);
            };
            if cd != domain {
                return cap_fault(trap_out);
            }
            let cm = host.jit_native_ctx(cd) as *mut CompiledModule;
            let (code, type_id) = host.jit_unit_install(cd, cu);
            if cm.is_null() || code == 0 {
                return cap_fault(trap_out);
            }
            // SAFETY: `cm` is the in-flight run's CompiledModule (guest suspended in this
            // synchronous cap.call); `code` is a natural-ABI entry the JIT registered for this
            // unit. The slot write does not move the table base (pre-reserved at compile).
            let v = match (*cm).install(code as *const u8, type_id) {
                Some(slot) => slot as i64,
                None => ENOSPC,
            };
            put(results, n_results, v, trap_out);
        }
        4 => {
            // uninstall(slot) -> 0 | -EINVAL (DESIGN.md §22 reclaim): clear an installed slot
            // so the index is reusable and a stale call_indirect of it traps.
            let Ok(domain) = host.resolve_jit_domain(handle) else {
                return cap_fault(trap_out);
            };
            let cm = host.jit_native_ctx(domain) as *mut CompiledModule;
            if cm.is_null() {
                return put(results, n_results, EINVAL, trap_out);
            }
            let slot = *args.first().unwrap_or(&-1);
            // SAFETY: `cm` is the in-flight run's CompiledModule (guest suspended).
            let v = if slot >= 0 && (*cm).uninstall(slot as u32) {
                0
            } else {
                EINVAL
            };
            put(results, n_results, v, trap_out);
        }
        5 => {
            // compile_linked(ir_ptr, ir_len, symtab_ptr, symtab_len) -> code_handle | -errno
            // (DESIGN.md §22 host-assisted dynamic linking). Like op 0, but the unit may carry
            // unresolved §7 imports bound by name against the guest's symbol-table buffer before
            // verify+compile. Any failure leaves nothing installed.
            let Ok(domain) = host.resolve_jit_domain(handle) else {
                return cap_fault(trap_out);
            };
            let cm = host.jit_native_ctx(domain) as *mut CompiledModule;
            if cm.is_null() {
                return put(results, n_results, EINVAL, trap_out);
            }
            let ir_ptr = *args.first().unwrap_or(&0) as u64;
            let ir_len = *args.get(1).unwrap_or(&0) as u64;
            let st_ptr = *args.get(2).unwrap_or(&0) as u64;
            let st_len = *args.get(3).unwrap_or(&0) as u64;
            let Some(ir) = gm.as_mut().and_then(|m| m.read_bytes(ir_ptr, ir_len)) else {
                return put(results, n_results, EFAULT, trap_out);
            };
            let Some(symtab) = gm.as_mut().and_then(|m| m.read_bytes(st_ptr, st_len)) else {
                return put(results, n_results, EFAULT, trap_out);
            };
            let compiled = match host.jit_compile_linked(handle, &ir, &symtab) {
                Ok(Ok(c)) => c,
                Ok(Err(e)) => return put(results, n_results, e, trap_out),
                Err(_) => return cap_fault(trap_out),
            };
            let funcs = host
                .jit_unit_funcs(compiled.domain, compiled.unit)
                .expect("unit was just stored");
            // SAFETY: identical contract to op 0 — `cm` is the in-flight run's CompiledModule and
            // the guest is suspended in this synchronous cap.call, so the re-entry aliases nothing.
            match (*cm).define_extra(&funcs) {
                Ok(defs) => {
                    let d = defs[0];
                    host.set_jit_unit_native(
                        compiled.domain,
                        compiled.unit,
                        d.tramp as usize,
                        d.code as usize,
                        d.type_id,
                    );
                    put(results, n_results, compiled.handle as i64, trap_out);
                }
                Err(_) => {
                    let _ = host.jit_release(compiled.handle);
                    put(results, n_results, EINVAL, trap_out);
                }
            }
        }
        // An op-index outside the Jit interface's defined ops (0..=5) is out of range, so it
        // traps (`CapFault`) — matching the interpreter, where an unknown Jit op falls through
        // the explicit op arms to the generic dispatch and faults, and §3c (an out-of-range
        // op-index is a runtime trap, not a non-fatal errno). The defined ops above own their
        // own errno-vs-fault choices; only genuinely unknown ops land here.
        _ => cap_fault(trap_out),
    }
}

/// The canonical [`svm_interp::JitValidator`] — the **security hinge** of the guest-driven
/// `Jit` capability (DESIGN.md §22 "Security argument"): `decode_module` (untrusted-input-facing,
/// fail-closed) → `verify_module` (the escape-freedom gate) → the **memory-match
/// precondition** (declared memory must equal the parent window, so verified bounds and the
/// runtime mask agree) → reject data segments (they would overwrite live guest memory) and
/// §12 concurrency ops (the single-threaded MVP restriction). Install the *same* function on
/// the interpreter and JIT `Host`s of a differential pair ([`grant_jit`] does), so both
/// backends accept/reject identically. All failures are `-EINVAL` (guest-visible, non-fatal,
/// nothing installed).
pub fn jit_blob_validator(
    bytes: &[u8],
    mem_log2: Option<u8>,
    symtab: &[u8],
) -> Result<Arc<[svm_ir::Func]>, i64> {
    const EINVAL: i64 = -22;
    // Decode the guest's symbol table (empty for the closed `compile` op — every prior caller —
    // which then resolves nothing, so a unit with imports fails closed). A malformed table is
    // fail-closed, before any IR is touched.
    let Some(table) = decode_symbol_table(symtab) else {
        return Err(EINVAL);
    };
    jit_resolve_and_validate(bytes, mem_log2, |name| table.get(name).copied())
}

/// Decode the guest-provided **symbol table** for `compile_linked` (DESIGN.md §22): a `name →
/// [`Resolved`]` map the loader binds a unit's §7 imports against. Untrusted-input-facing and
/// fail-closed (`None` on any malformation) — but note the *values* are guest-chosen by design:
/// a forged slot confers no authority (the resolved `call_indirect` is masked + `type_id`-checked
/// at the call, exactly like a slot the guest already controls in its own code), and the whole
/// unit is re-verified after the rewrite. Wire form (LEB128, mirroring `svm-encode`):
/// `count`, then per entry `name` (uleb len + UTF-8 bytes), a `kind` byte, and its payload —
/// `0` = `Slot(uleb)`, `1` = `Cap(uleb type_id, uleb op)`.
fn decode_symbol_table(bytes: &[u8]) -> Option<HashMap<String, Resolved>> {
    // The closed-blob `compile` op passes no table at all (`&[]`); treat that as the empty table
    // (it resolves nothing, so a unit with imports fails closed). `[0]` — an explicit count of 0 —
    // is the same thing and is handled by the normal path below.
    if bytes.is_empty() {
        return Some(HashMap::new());
    }
    let mut c = SymCursor { bytes, pos: 0 };
    let count = c.uleb()?;
    let mut table = HashMap::new();
    for _ in 0..count {
        let name = c.string()?;
        let resolved = match c.byte()? {
            0 => Resolved::Slot(c.u32()?),
            1 => Resolved::Cap(svm_ir::ResolvedCap {
                type_id: c.u32()?,
                op: c.u32()?,
            }),
            _ => return None, // unknown kind
        };
        table.insert(name, resolved);
    }
    // Trailing bytes mean a length mismatch — reject rather than silently ignore (fail-closed).
    (c.pos == bytes.len()).then_some(table)
}

/// Encode a [`decode_symbol_table`] buffer — the producer side a guest loader (or a test) uses to
/// build the symbol table it hands `compile_linked`. Only `Slot`/`Cap` bindings are deliverable:
/// `Func` is the static-link (same-module-index) case, meaningless for a separately-compiled unit.
pub fn encode_symbol_table(entries: &[(&str, Resolved)]) -> Vec<u8> {
    let mut out = Vec::new();
    svm_encode::write_uleb(&mut out, entries.len() as u64);
    for (name, r) in entries {
        svm_encode::write_uleb(&mut out, name.len() as u64);
        out.extend_from_slice(name.as_bytes());
        match r {
            Resolved::Slot(slot) => {
                out.push(0);
                svm_encode::write_uleb(&mut out, *slot as u64);
            }
            Resolved::Cap(cap) => {
                out.push(1);
                svm_encode::write_uleb(&mut out, cap.type_id as u64);
                svm_encode::write_uleb(&mut out, cap.op as u64);
            }
            Resolved::Func(_) => panic!("Func is not deliverable via the guest symbol table"),
        }
    }
    out
}

/// A minimal fail-closed cursor for [`decode_symbol_table`] (the IR codec's `Cursor` is private to
/// `svm-encode`). Never panics/over-reads on arbitrary bytes.
struct SymCursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl SymCursor<'_> {
    fn byte(&mut self) -> Option<u8> {
        let b = *self.bytes.get(self.pos)?;
        self.pos += 1;
        Some(b)
    }

    /// Unsigned LEB128 → `u64` (max 10 bytes; rejects overflow / truncation).
    fn uleb(&mut self) -> Option<u64> {
        let (mut result, mut shift) = (0u64, 0u32);
        loop {
            let b = self.byte()?;
            if shift >= 64 || (shift == 63 && b & 0x7f > 1) {
                return None;
            }
            result |= ((b & 0x7f) as u64) << shift;
            if b & 0x80 == 0 {
                return Some(result);
            }
            shift += 7;
        }
    }

    fn u32(&mut self) -> Option<u32> {
        u32::try_from(self.uleb()?).ok()
    }

    /// Length-prefixed UTF-8 string; the length is bounded by the remaining bytes (anti-OOM).
    fn string(&mut self) -> Option<String> {
        let n = usize::try_from(self.uleb()?).ok()?;
        let end = self.pos.checked_add(n)?;
        let s = core::str::from_utf8(self.bytes.get(self.pos..end)?).ok()?;
        self.pos = end;
        Some(s.to_owned())
    }
}

/// Host-assisted dynamic-link resolve — the host-assisted half of in-window dynamic linking
/// (DESIGN.md §22). Decode
/// a serialized unit that may carry **unresolved §7 imports** (the v2 wire form), bind each import
/// name through `resolve` (a *guest-controlled* symbol table: name → a `call_indirect` table slot,
/// or a host capability), then run the **same** fail-closed gate as [`jit_blob_validator`]. Crucially
/// the resolve is a source-to-source rewrite that runs *before* `verify_module`, so a mis-link — an
/// unknown name, a wrong import signature, a non-const slot handle — is caught by re-verification and
/// never trusted (DESIGN.md §22 "rewrite-then-verify"; the symbol table stays guest-controlled, the
/// loader cannot forge a binding the verifier would reject). All failures are `-EINVAL` (guest-visible,
/// non-fatal, nothing installed).
pub fn jit_resolve_and_validate(
    bytes: &[u8],
    mem_log2: Option<u8>,
    resolve: impl FnMut(&str) -> Option<Resolved>,
) -> Result<Arc<[svm_ir::Func]>, i64> {
    const EINVAL: i64 = -22;
    let Ok(m) = svm_encode::decode_module(bytes) else {
        return Err(EINVAL);
    };
    // Bind every named import to a concrete `call`/`call_indirect`/`cap.call` (fail-closed on an
    // unresolved or ill-typed binding), yielding an import-free module the verifier accepts unchanged.
    let Ok(m) = svm_ir::resolve_imports_with(&m, resolve) else {
        return Err(EINVAL);
    };
    if svm_verify::verify_module(&m).is_err() {
        return Err(EINVAL);
    }
    if m.memory.map(|mc| mc.size_log2) != mem_log2 {
        return Err(EINVAL);
    }
    if !m.data.is_empty() {
        return Err(EINVAL);
    }
    if m.funcs.is_empty() || m.funcs.iter().any(|f| f.uses_concurrency()) {
        return Err(EINVAL);
    }
    // A submitted unit's `call_indirect` (the new→old path) is now allowed: on the JIT it
    // dispatches through the parent `fn_table`; the reference interpreter mirrors this with its
    // module-aware dispatch table (a unit runs as a module ≥ 1 whose indirect calls resolve into
    // module 0). Both backends therefore reach the original program's functions identically
    // (DESIGN.md §22 new→old).
    Ok(m.funcs.into())
}

/// Grant the guest-driven `Jit` capability (opt-in, like `Memory`): install the canonical
/// [`jit_blob_validator`] and mint the domain handle bound to `m`'s declared memory (the
/// memory-match precondition). Works for both backends — the interpreter services the iface
/// in its eval loop/dispatch; the JIT needs the module registered too (see [`jit_cap_run`]).
/// `table_log2` reserves the `call_indirect` table for B2 `install` (pass the **same** value as
/// the JIT compile's `table_reserve_log2`); `0` ⇒ no install room.
pub fn grant_jit(host: &mut Host, m: &Module, table_log2: u8) -> i32 {
    host.set_jit_validator(jit_blob_validator);
    host.grant_jit_with_table(m.memory.map(|mc| mc.size_log2), table_log2)
}

/// Run `m` on the **JIT** with the `Jit` capability live: the long-lived compile→run split
/// ([`CompiledModule`]), with the module pointer registered in `host` so [`cap_thunk`]'s
/// native `Jit` ops can re-enter it mid-run (`define_extra` / `invoke_extra` while the guest
/// is suspended in its synchronous `cap.call`). The interpreter counterpart is the plain
/// `run_capture_reserved_with_host` over the same `Host` setup ([`grant_jit`]) — drive both
/// with identical inputs for the differential.
#[allow(clippy::too_many_arguments)]
pub fn jit_cap_run(
    m: &Module,
    entry: u32,
    args: &[i64],
    init_mem: &[u8],
    reserved_log2: u8,
    table_reserve_log2: u8,
    host: &mut Host,
) -> Result<(JitOutcome, Vec<u8>), svm_jit::JitError> {
    // A guest whose workers make concurrent `cap.call`s (threaded `Jit.compile`, DESIGN.md §22) runs
    // the **serialized** thunk over a per-domain `Mutex<Host>`; a single-threaded guest keeps the
    // unlocked `cap_thunk` + raw `Host` path verbatim (zero lock cost). The guest-facing iface is
    // identical either way — the serialization is an internal detail that can be made finer-grained
    // later without changing guest software.
    if m.funcs.iter().any(|f| f.uses_concurrency()) {
        let host_mutex = Mutex::new(std::mem::take(host));
        let ctx = &host_mutex as *const Mutex<Host> as *mut c_void;
        let mut cm = CompiledModule::compile(
            m,
            entry,
            cap_thunk_locked,
            ctx,
            reserved_log2,
            None,
            None,
            None,
            None,
            svm_jit::Quota::default(),
            table_reserve_log2,
        )?;
        host_mutex
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .set_jit_native_ctx(&mut cm as *mut CompiledModule as usize);
        // SAFETY: `&mut cm` is the only pointer the thunk's handlers re-enter through (registered
        // above); all of the run's vCPU threads serialize their `cap.call`s through `host_mutex`.
        let r =
            unsafe { CompiledModule::run_raw(&mut cm, args, Some(init_mem), Some(1 << 18), None) };
        host_mutex
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .set_jit_native_ctx(0);
        *host = host_mutex.into_inner().unwrap_or_else(|e| e.into_inner());
        return r;
    }
    let mut cm = CompiledModule::compile(
        m,
        entry,
        cap_thunk,
        host as *mut Host as *mut c_void,
        reserved_log2,
        None,
        None,
        None,
        None,
        svm_jit::Quota::default(),
        table_reserve_log2,
    )?;
    let cm_ptr: *mut CompiledModule = &mut cm;
    host.set_jit_native_ctx(cm_ptr as usize);
    // Snapshot span: the low 256 KiB, matching the interp/JIT `SNAP_CAP` capture pairing.
    // SAFETY: `cm_ptr` is the only pointer used for this run (the same one the thunk's handlers
    // re-enter through, registered above); the run is single-threaded on this thread.
    let r = unsafe { CompiledModule::run_raw(cm_ptr, args, Some(init_mem), Some(1 << 18), None) };
    // The module dies with this call — leave no dangling registration behind.
    host.set_jit_native_ctx(0);
    r
}

/// **Code-memory compaction** for a guest-driven `Jit` domain (DESIGN.md §22): rebuild the domain's
/// *live* JIT code into a **fresh** [`CompiledModule`], reclaiming the old module's 256 MiB code
/// arena (cranelift-jit has no per-function free, so reclaim = whole-module recompaction — drop
/// the old module and its arena goes with it). Returns the fresh module for the caller to swap in
/// (register its pointer with [`Host::set_jit_native_ctx`] and run on it from now on); the old
/// module stays valid until the caller drops it.
///
/// **Quiescent-only.** The guest is suspended *inside* the module being compacted during any
/// `cap.call`, so this is **embedder-facing** — call it between runs (a REPL between prompts),
/// never from a `cap.call` handler. Asserts `!old.is_running()`.
///
/// **What is carried, and why it is transparent.** Every unit that is still reachable — either
/// occupying a `call_indirect` table slot ([`CompiledModule::installed_slots`]) or held through a
/// live `CompiledCode` handle ([`Host::jit_live_units`]) — is re-`define_extra`'d into the fresh
/// module, its `Host` unit→native pointers are remapped (so existing `CompiledCode` handles, which
/// name a `(domain, unit)` not a code address, keep invoking the right code), and any slot it
/// occupied is reproduced at the **exact** index with [`CompiledModule::install_at`] (so a funcref
/// old code already holds still resolves to it). A unit that is neither installed nor live-handled
/// is dead and is **not** carried — that is the reclaim. The handle table itself needs no edit
/// (handles are `(domain, unit)` indices, indirected through the remapped native pointers).
///
/// `base`/`entry`/`reserved_log2`/`table_reserve_log2` must be the **same** inputs the old module
/// was compiled with (the caller owns them — the same ones it passed to [`jit_cap_run`] /
/// [`CompiledModule::compile`]); the fresh module shares the parent's baked environment exactly.
pub fn recompact_jit(
    base: &Module,
    entry: u32,
    reserved_log2: u8,
    table_reserve_log2: u8,
    host: &mut Host,
    domain: u32,
    old: &CompiledModule,
) -> Result<CompiledModule, svm_jit::JitError> {
    assert!(
        !old.is_running(),
        "recompact_jit is quiescent-only: no run may be in flight on the old module"
    );
    // Single-threaded path: the fresh module bakes the unlocked `cap_thunk` + raw `Host`. A
    // *concurrent* guest must compact through [`JitSession`] (or replicate its pattern:
    // `cap_thunk_locked` over a stable `Mutex<Host>`, then [`recompact_into`]) — re-running a fresh
    // module that baked the unlocked thunk under threads would race (DESIGN.md §22).
    let mut fresh = CompiledModule::compile(
        base,
        entry,
        cap_thunk,
        host as *mut Host as *mut c_void,
        reserved_log2,
        None,
        None,
        None,
        None,
        svm_jit::Quota::default(),
        table_reserve_log2,
    )?;
    recompact_into(&mut fresh, host, domain, old)?;
    Ok(fresh)
}

/// Rebuild the domain's **live unit set** into an already-compiled `fresh` module: carry every unit
/// still reachable (held through a live `CompiledCode` handle, [`Host::jit_live_units`], **or**
/// occupying a `call_indirect` slot of `old`), re-`define_extra` it, remap the `Host` unit→native
/// record (so existing handles keep resolving), and reproduce occupied slots at their **exact**
/// index. The thunk/ctx baked into `fresh` (locked vs raw) is the caller's choice — this is the
/// thunk-agnostic half of compaction, shared by [`recompact_jit`] (raw) and [`JitSession`] (locked,
/// concurrency-capable). Quiescent-only.
pub fn recompact_into(
    fresh: &mut CompiledModule,
    host: &mut Host,
    domain: u32,
    old: &CompiledModule,
) -> Result<(), svm_jit::JitError> {
    // The set of live table slots in the OLD module, keyed by the unit's natural-entry code so we
    // can rejoin slot → owning unit below (a unit may, in principle, occupy more than one slot).
    let mut code_to_slots: std::collections::HashMap<u64, Vec<u32>> =
        std::collections::HashMap::new();
    for (slot, code, _type_id) in old.installed_slots() {
        code_to_slots.entry(code).or_default().push(slot);
    }
    // Carry every unit that is still reachable: live-handled OR occupying a slot. (A slot can be
    // occupied by a unit whose handle was already released — a redefinition survivor — so the two
    // sources are unioned, not either alone.)
    let mut keep: Vec<u32> = host.jit_live_units(domain);
    for unit in 0..host.jit_unit_count(domain) {
        let (install_code, _) = host.jit_unit_install(domain, unit);
        if install_code != 0
            && code_to_slots.contains_key(&(install_code as u64))
            && !keep.contains(&unit)
        {
            keep.push(unit);
        }
    }
    keep.sort_unstable();
    keep.dedup();

    for unit in keep {
        // The unit's OLD natural-entry pointer — used to find the slot(s) it occupied — read
        // *before* we overwrite the Host's record with the fresh pointers.
        let (old_install_code, _) = host.jit_unit_install(domain, unit);
        let Some(funcs) = host.jit_unit_funcs(domain, unit) else {
            continue; // no IR retained (cannot happen for a compiled unit) — skip defensively
        };
        let defs = fresh.define_extra(&funcs)?;
        let d = defs[0];
        host.set_jit_unit_native(domain, unit, d.tramp as usize, d.code as usize, d.type_id);
        if let Some(slots) = code_to_slots.get(&(old_install_code as u64)) {
            for &slot in slots {
                // Exact-slot reproduction: a funcref old code holds keeps resolving to this unit.
                fresh.install_at(slot, d.code, d.type_id);
            }
        }
    }
    Ok(())
}

/// A long-lived **guest-driven JIT REPL session** over one domain (DESIGN.md §22): the persistent
/// `CompiledModule` + window an embedder re-enters once per prompt, with **automatic compaction**
/// when code-arena occupancy crosses a watermark. This is the auto-trigger policy that turns the
/// [`recompact_jit`] primitive into a usable long-session story — the missing piece between
/// "compaction works if you call it" and "a REPL never exhausts the 256 MiB arena."
///
/// Each [`Self::run_prompt`] runs the guest entry over the **carried window** (the prior prompt's
/// final low bytes seed the next, so guest heap/global state persists across prompts) with the
/// module registered for the `Jit` capability's mid-run re-entry (`compile`/`invoke`/`install`,
/// exactly as [`jit_cap_run`]). After the prompt returns — a **quiescent** point, the guest no
/// longer suspended — if [`CompiledModule::extra_byte_count`] has reached `watermark` code bytes, the session
/// rebuilds the domain's live code into a fresh module ([`recompact_jit`]) and drops the old one,
/// reclaiming its arena. Because compaction is transparent (live slots + handles are preserved,
/// see `recompact_jit`), the guest never observes it.
///
/// **Concurrency.** The session **owns** the `Host` behind a boxed `Mutex` (stable address) and bakes
/// [`cap_thunk_locked`], so a **multi-threaded** guest's worker `cap.call`s (incl. threaded
/// `Jit.compile`, DESIGN.md §22) serialize correctly — and compaction (a quiescent, between-prompts
/// operation) rebuilds the module with the **same** locked thunk, so the next multi-threaded prompt
/// stays sound. A single-threaded guest pays only an uncontended lock per `cap.call`, negligible for
/// an interactive REPL driver (the perf-critical single-run path is [`jit_cap_run`], which stays
/// unlocked). Retrieve the host with [`Self::into_host`].
///
/// # Lifetime contract
/// The boxed `Mutex<Host>`'s heap address is baked into the compiled code as the `cap.call` ctx (at
/// construction and at every recompaction); the `Box` keeps it stable across `JitSession` moves.
pub struct JitSession {
    base: Module,
    entry: u32,
    reserved_log2: u8,
    table_reserve_log2: u8,
    domain: u32,
    /// Auto-compact once `cm.extra_byte_count() >= watermark` **code bytes** (checked after each
    /// prompt, at the quiescent point) — a byte-accurate trigger, so a few large units fire it the
    /// same as many tiny ones. `0` disables auto-compaction (the embedder may still call
    /// [`Self::compact`] by hand).
    watermark: usize,
    cm: CompiledModule,
    /// The session-owned powerbox, boxed so its address is stable (baked as the `cap.call` ctx) and
    /// behind a `Mutex` so a multi-threaded guest's concurrent `cap.call`s serialize.
    host: Box<Mutex<Host>>,
    /// The carried guest window (low `SNAP` bytes), seeding each prompt and updated from its result.
    window: Vec<u8>,
    /// How many times this session has auto-compacted (observability / tests).
    compactions: usize,
}

/// The window snapshot span carried across prompts — matches the interp/JIT `SNAP_CAP` pairing
/// ([`jit_cap_run`]).
const SESSION_SNAP: usize = 1 << 18;

impl JitSession {
    /// Build a session that **takes ownership** of `host`: compile `base` (entry `func`) long-lived
    /// with the `Jit` ctx baked to the session's boxed `Mutex<Host>`, ready to run prompts on
    /// `domain` (the [`grant_jit`]-returned domain). `watermark` is the auto-compaction threshold in
    /// **code bytes** ([`CompiledModule::extra_byte_count`]; `0` = manual only). Pass the **same**
    /// `reserved_log2`/`table_reserve_log2` you
    /// grant the cap with. Recover the host via [`Self::into_host`].
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        base: &Module,
        entry: u32,
        reserved_log2: u8,
        table_reserve_log2: u8,
        domain: u32,
        watermark: usize,
        host: Host,
    ) -> Result<JitSession, svm_jit::JitError> {
        let host = Box::new(Mutex::new(host));
        let ctx = &*host as *const Mutex<Host> as *mut c_void;
        let cm = CompiledModule::compile(
            base,
            entry,
            cap_thunk_locked,
            ctx,
            reserved_log2,
            None,
            None,
            None,
            None,
            svm_jit::Quota::default(),
            table_reserve_log2,
        )?;
        Ok(JitSession {
            base: base.clone(),
            entry,
            reserved_log2,
            table_reserve_log2,
            domain,
            watermark,
            cm,
            host,
            window: vec![0u8; SESSION_SNAP],
            compactions: 0,
        })
    }

    /// Run the guest entry on `args` over the carried window, then auto-compact if the watermark is
    /// reached. Returns the prompt's [`JitOutcome`]; the window snapshot is retained for the next
    /// prompt (read it via [`Self::window`]). `args` is the raw i64-slot ABI (e.g. the `Jit` handle
    /// followed by the guest's own arguments). The guest may spawn threads (its worker `cap.call`s
    /// serialize through the session's `Mutex<Host>`).
    pub fn run_prompt(&mut self, args: &[i64]) -> Result<JitOutcome, svm_jit::JitError> {
        let cm_ptr: *mut CompiledModule = &mut self.cm;
        self.host
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .set_jit_native_ctx(cm_ptr as usize);
        // SAFETY: `cm_ptr` is the only pointer used for this run and the one the thunk's handlers
        // re-enter through (registered above); the run's vCPU threads serialize their `cap.call`s
        // through the session's `Mutex<Host>`; `self.cm` is not moved during the call (we hold
        // `&mut self`, and `run_raw` keeps no live reference across the guarded call).
        let r = unsafe {
            CompiledModule::run_raw(cm_ptr, args, Some(&self.window), Some(SESSION_SNAP), None)
        };
        self.host
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .set_jit_native_ctx(0);
        let (out, mem) = r?;
        self.window = mem;
        if self.watermark != 0 && self.cm.extra_byte_count() >= self.watermark {
            self.compact()?;
        }
        Ok(out)
    }

    /// Force a compaction now (the embedder's manual trigger; [`Self::run_prompt`] calls it
    /// automatically at the watermark). Quiescent — only valid between prompts. The fresh module
    /// bakes the **same** locked thunk over the session's `Mutex<Host>`, so a subsequent
    /// multi-threaded prompt stays sound.
    pub fn compact(&mut self) -> Result<(), svm_jit::JitError> {
        assert!(
            !self.cm.is_running(),
            "JitSession::compact is quiescent-only: no prompt may be in flight"
        );
        let ctx = &*self.host as *const Mutex<Host> as *mut c_void;
        let mut fresh = CompiledModule::compile(
            &self.base,
            self.entry,
            cap_thunk_locked,
            ctx,
            self.reserved_log2,
            None,
            None,
            None,
            None,
            svm_jit::Quota::default(),
            self.table_reserve_log2,
        )?;
        {
            let mut host = self.host.lock().unwrap_or_else(|e| e.into_inner());
            recompact_into(&mut fresh, &mut host, self.domain, &self.cm)?;
        }
        self.cm = fresh;
        self.compactions += 1;
        Ok(())
    }

    /// Recover the owned `Host` (e.g. to read captured stdout) after the session ends.
    pub fn into_host(self) -> Host {
        (*self.host).into_inner().unwrap_or_else(|e| e.into_inner())
    }

    /// Seed `bytes` into the carried guest window at `off` before the next prompt — e.g. a
    /// submitted-IR blob the guest `cap.call compile`s, or argv/env/data a REPL hands the first
    /// prompt. Persists like any window state (each prompt seeds from, and writes back to, the
    /// carried window). Out-of-range writes are clamped to the window.
    pub fn seed_window(&mut self, off: usize, bytes: &[u8]) {
        let end = (off + bytes.len()).min(self.window.len());
        if off < end {
            self.window[off..end].copy_from_slice(&bytes[..end - off]);
        }
    }

    /// The carried guest window (low [`SESSION_SNAP`] bytes) as of the last prompt.
    pub fn window(&self) -> &[u8] {
        &self.window
    }

    /// Current code-arena occupancy in **code bytes** ([`CompiledModule::extra_byte_count`]) — the
    /// quantity the `watermark` is compared against.
    pub fn occupancy(&self) -> usize {
        self.cm.extra_byte_count()
    }

    /// How many auto/manual compactions have run over this session's life.
    pub fn compactions(&self) -> usize {
        self.compactions
    }
}

/// The §9/D45 **devirtualized `cap.call` fast-path resolver** for the production powerbox. It claims
/// only the **window-independent, authority-checked** hot ops — `Clock.now` and `Blocking.work` — so
/// they take the register-to-register fast path; every other op (all *window-touching* ones —
/// `Memory`/`Stream`/`SharedRegion`/`IoRing` — and any multi-result or arity-mismatched op) returns
/// `null`, so the generic [`cap_thunk`] handles it unchanged.
///
/// **Safety / authority is preserved by construction:** the specialized fns delegate to the *same*
/// [`Host::cap_dispatch_slots`] the generic thunk uses (with `gm = None`, since these ops never touch
/// the guest window), so the I2 authority check — a forged/closed/wrong-type handle is an inert
/// `CapFault` — and the op semantics are byte-identical to the generic path. The win is only the
/// leaner JIT→host boundary (args/result in registers, no stack marshalling, no runtime `(type_id,
/// op)` dispatch). The arity gate (`n_args`/`n_res`) prevents a C-ABI mismatch if a frontend emits a
/// `cap.call` to one of these ops with an unexpected signature.
///
/// Pass it to [`svm_jit::compile_and_run_with_host_fast`] /
/// [`svm_jit::compile_and_run_with_host_interruptible_fast`]; [`run_powerbox`] uses it automatically.
///
/// # Safety
/// Honours the [`svm_jit::FastCapResolver`] contract: `ctx` (passed to the returned fns) is a live
/// `*mut Host`; the returned fns gate on the supplied arity and stay valid for the run.
pub unsafe extern "C" fn fast_cap_resolver(
    type_id: u32,
    op: u32,
    n_args: u32,
    n_res: u32,
) -> *const c_void {
    use svm_interp::iface;
    match (type_id, op, n_args, n_res) {
        (iface::CLOCK, 0, 0, 1) => fast_clock_now as *const c_void,
        (iface::BLOCKING, 0, 1, 1) => fast_blocking_work as *const c_void,
        _ => std::ptr::null(),
    }
}

/// `Clock.now() -> i64` (iface 2, op 0, no args) on the fast path.
///
/// Uses [`svm_interp::Host::fast_clock_now`] — an authority-checked, **allocation-free** inline read
/// (ISSUES.md I12), so this is genuinely cheaper than the generic dispatch, not just a leaner JIT→host
/// boundary. Falls back to [`fast_dispatch`] (the full slot dispatch) when a W1 record/replay tape is
/// active, so the clock crossing is still taped/served faithfully.
///
/// # Safety
/// `ctx` is a live `*mut Host`; `trap_out` is writable — the [`svm_jit::FastCapResolver`] contract.
unsafe extern "C" fn fast_clock_now(
    ctx: *mut c_void,
    _mem_base: *mut u8,
    _mem_size: u64,
    handle: i32,
    trap_out: *mut i64,
) -> i64 {
    let host = &mut *(ctx as *mut Host);
    match host.fast_clock_now(handle) {
        Some(Ok(ns)) => {
            *trap_out = 0;
            ns
        }
        Some(Err(_)) => {
            *trap_out = TrapKind::CapFault as i64;
            0
        }
        // A W1 tape is active — take the full path so the input is recorded/replayed.
        None => fast_dispatch(ctx, svm_interp::iface::CLOCK, 0, handle, &[], trap_out),
    }
}

/// `Blocking.work(a0) -> i64` (iface 10, op 0, one arg) on the fast path.
///
/// # Safety
/// As [`fast_clock_now`].
unsafe extern "C" fn fast_blocking_work(
    ctx: *mut c_void,
    _mem_base: *mut u8,
    _mem_size: u64,
    handle: i32,
    trap_out: *mut i64,
    a0: i64,
) -> i64 {
    fast_dispatch(ctx, svm_interp::iface::BLOCKING, 0, handle, &[a0], trap_out)
}

/// Shared body for the fast-path fns: drive the **same** [`Host::cap_dispatch_slots`] the generic
/// thunk uses (so the authority check + semantics are identical), with no window (`gm = None` — these
/// ops never touch the guest window). The register args are already collected in `args`; the single
/// result is returned and the trap cell encoded exactly as [`cap_thunk`].
///
/// # Safety
/// `ctx` is a live `*mut Host`; `trap_out` is writable.
#[inline]
unsafe fn fast_dispatch(
    ctx: *mut c_void,
    type_id: u32,
    op: u32,
    handle: i32,
    args: &[i64],
    trap_out: *mut i64,
) -> i64 {
    let host = &mut *(ctx as *mut Host);
    match host.cap_dispatch_slots(type_id, op, handle, args, None) {
        Ok(res) => {
            *trap_out = 0;
            res.first().copied().unwrap_or(0)
        }
        Err(Trap::Exit(code)) => {
            *trap_out = EXIT_CODE as i64 | ((code as i64) << 32);
            0
        }
        Err(_) => {
            *trap_out = TrapKind::CapFault as i64;
            0
        }
    }
}

/// The §14 **module resolver** for the JIT's nesting runtime: resolve a guest's `Module` handle
/// (granted by [`Host::grant_module`]) to the module's code/data so the runtime can compile and spawn
/// a **separate-module child** (`instantiate_module` & friends). Pass it (with the same `Host` ctx as
/// [`cap_thunk`]) to `compile_and_run_capture_reserved_with_host_ex`. Deliberately not routed through
/// `cap.call` dispatch: it yields host pointers, which must never be guest-reachable.
///
/// # Safety
/// `ctx` is the live `*mut Host` (the same as the cap thunk's); `out` is a writable
/// [`svm_jit::ResolvedModule`]. The `Host` must outlive the run (it owns the resolved views).
pub unsafe extern "C" fn module_resolver(
    ctx: *mut c_void,
    handle: i32,
    out: *mut svm_jit::ResolvedModule,
) -> i32 {
    let host = &*(ctx as *const Host);
    match host.resolve_module_parts(handle) {
        Some((funcs, n_funcs, memory_log2, data, n_data)) => {
            *out = svm_jit::ResolvedModule {
                funcs,
                n_funcs,
                memory_log2,
                data,
                n_data,
            };
            1
        }
        None => 0,
    }
}

/// The **host** page size: the protection granularity for `map`/`unmap`/`protect`, matching the
/// interpreter (`svm_interp`) and the JIT (`svm-jit`) on the same host so all three agree
/// page-for-page (§4 "pin page size", host-page default). `sysconf(_SC_PAGESIZE)` on unix,
/// `GetSystemInfo` on windows.
#[cfg(unix)]
fn host_page_size() -> u64 {
    // SAFETY: sysconf is always safe; _SC_PAGESIZE is positive.
    match unsafe { libc::sysconf(libc::_SC_PAGESIZE) } {
        p if p > 0 => p as u64,
        _ => 4096,
    }
}
#[cfg(windows)]
fn host_page_size() -> u64 {
    use windows_sys::Win32::System::SystemInformation::{GetSystemInfo, SYSTEM_INFO};
    // SAFETY: GetSystemInfo only writes its out-param; always safe.
    let mut si: SYSTEM_INFO = unsafe { core::mem::zeroed() };
    unsafe { GetSystemInfo(&mut si) };
    match si.dwPageSize as u64 {
        0 => 4096,
        p => p,
    }
}

/// A [`GuestMem`] over the JIT's guest window whose `map`/`unmap`/`protect` (the Memory capability,
/// §3e) are backed by **real hardware page protection** on the window pages (`mprotect` on unix,
/// `VirtualAlloc`/`VirtualProtect` on windows), mirrored by a software page-state map. The mirror
/// lets cap-buffer borrows (§7) **fail closed** (`-EFAULT`) on an unmapped/RO page instead of
/// faulting the host outside the guarded call, and bounds growth to the reserved mask domain —
/// keeping this backend bit-identical to the interpreter's paged `Mem` (the §18 oracle, enforced by
/// `jit_diff`'s differential). The page-map model is portable; only the three hardware primitives
/// (`hw_commit_rw`/`hw_apply`/`hw_release_hint`) differ per OS.
///
/// # Safety
/// `base` must point at the JIT guest window: `[base, base+mapped)` initially RW and the whole
/// `[base, base+reserved)` a live inaccessible/RW reservation owned for the call's duration.
#[cfg(any(unix, windows))]
pub struct MprotectWindow {
    base: *mut u8,
    mapped: u64,
    reserved: u64,
    /// Host page size (`host_page_size()`), the protection granularity (matches `svm_interp`).
    page: u64,
    /// Page index ⇒ explicit state code (1=Rw, 2=Ro, 3=Unmapped); absent ⇒ region default (rw in
    /// `[0, mapped)`, unmapped in the reserved tail). Mirrors `svm_interp`'s page map so the two
    /// backends agree page-for-page. **Shared** ([`Arc<Mutex<…>>`]) so it persists across the run's
    /// `cap.call`s (the JIT rebuilds the window view per call): guest-grown pages stay borrowable. The
    /// persistent home is the `Host` ([`Host::cap_window_pages`]); a one-off [`MprotectWindow::new`]
    /// gets a private fresh map.
    prot: CapPageMap,
}

#[cfg(any(unix, windows))]
#[derive(Clone, Copy, PartialEq, Eq)]
enum PageState {
    Rw,
    Ro,
    Unmapped,
}

#[cfg(any(unix, windows))]
impl PageState {
    fn code(self) -> u8 {
        match self {
            PageState::Rw => 1,
            PageState::Ro => 2,
            PageState::Unmapped => 3,
        }
    }
    fn from_code(c: u8) -> Option<PageState> {
        match c {
            1 => Some(PageState::Rw),
            2 => Some(PageState::Ro),
            3 => Some(PageState::Unmapped),
            _ => None,
        }
    }
}

#[cfg(any(unix, windows))]
impl MprotectWindow {
    /// Wrap the JIT window `[base, base+mapped)` (backed) inside a `reserved` mask domain with a
    /// **private** fresh page map — for a one-off view. Most callers want [`MprotectWindow::new_shared`]
    /// (the `cap_thunk` path) so growth persists across the run's cap.calls.
    pub fn new(base: *mut u8, mapped: u64, reserved: u64) -> MprotectWindow {
        Self::new_shared(
            base,
            mapped,
            reserved,
            Arc::new(Mutex::new(BTreeMap::new())),
        )
    }

    /// Like [`MprotectWindow::new`], but with a **shared** page map (typically the per-run one from
    /// [`Host::cap_window_pages`]) so a guest-grown page committed in one `cap.call` is still seen
    /// committed by a later one — the cap-buffer borrow of grown heap memory no longer fail-closes.
    pub fn new_shared(
        base: *mut u8,
        mapped: u64,
        reserved: u64,
        prot: CapPageMap,
    ) -> MprotectWindow {
        MprotectWindow {
            base,
            mapped,
            reserved: reserved.max(mapped),
            page: host_page_size(),
            prot,
        }
    }

    /// Read one page's explicit state from the shared map (locks; `None` ⇒ absent / region default).
    fn prot_get(&self, page: u64) -> Option<PageState> {
        self.prot
            .lock()
            .unwrap()
            .get(&page)
            .copied()
            .and_then(PageState::from_code)
    }
    /// Set one page's explicit state in the shared map.
    fn prot_set(&self, page: u64, st: PageState) {
        self.prot.lock().unwrap().insert(page, st.code());
    }
    /// Clear one page back to the region default (absent).
    fn prot_clear(&self, page: u64) {
        self.prot.lock().unwrap().remove(&page);
    }

    /// One page's access state: `None` ⇒ faults (unmapped), `Some(writable)` ⇒ committed — the
    /// same default rule as the interpreter (`svm_interp::Mem::page_access`).
    fn page_access(&self, page: u64) -> Option<bool> {
        match self.prot_get(page) {
            Some(PageState::Rw) => Some(true),
            Some(PageState::Ro) => Some(false),
            Some(PageState::Unmapped) => None,
            None => (page * self.page < self.mapped).then_some(true),
        }
    }

    /// Every page of `[ptr, ptr+len)` is committed (and writable when `write`), within
    /// `[0, reserved)` — the §7 borrow check, mirroring `svm_interp`.
    fn range_committed(&self, ptr: u64, len: u64, write: bool) -> bool {
        let Some(end) = ptr.checked_add(len) else {
            return false;
        };
        if end > self.reserved {
            return false;
        }
        if len == 0 {
            return true;
        }
        (ptr / self.page..=(end - 1) / self.page)
            .all(|p| matches!(self.page_access(p), Some(w) if w || !write))
    }

    /// Validate a `map`/`unmap`/`protect` range and return its inclusive page-index span, or
    /// `-EINVAL` (page-aligned offset, non-zero len, within `[0, reserved)`) — matching the
    /// interpreter's `prot_pages` (growth into the reserved tail is allowed).
    fn prot_pages(&self, offset: u64, len: u64) -> Result<std::ops::RangeInclusive<u64>, i64> {
        const EINVAL: i64 = -22;
        if len == 0 || !offset.is_multiple_of(self.page) {
            return Err(EINVAL);
        }
        let end = offset.checked_add(len).ok_or(EINVAL)?;
        if end > self.reserved {
            return Err(EINVAL);
        }
        Ok((offset / self.page)..=((end - 1) / self.page))
    }

    /// Update one page's software state from cap `prot` bits, mirroring `svm_interp::set_prot`:
    /// a read-write page is left absent in the prefix, explicit `Rw` in the reserved tail.
    fn set_prot(&mut self, page: u64, prot: i32) {
        const PROT_READ: i32 = 1;
        const PROT_WRITE: i32 = 2;
        if prot & PROT_WRITE != 0 {
            if page * self.page < self.mapped {
                self.prot_clear(page);
            } else {
                self.prot_set(page, PageState::Rw);
            }
        } else if prot & PROT_READ != 0 {
            self.prot_set(page, PageState::Ro);
        } else {
            self.prot_set(page, PageState::Unmapped);
        }
    }

    // ---- the three hardware primitives (the only per-OS part) -----------------------------------
    // All take a **page-aligned** `[off, off+len)` already validated `⊆ reserved` by `prot_pages`.

    /// Make `[off, off+len)` **committed and read-write** (so a following zero-fill / protection
    /// change lands). On unix the reservation is `MAP_NORESERVE`, so `mprotect(RW)` suffices and the
    /// kernel demand-zeroes; on windows the tail is reserved-but-uncommitted, so `VirtualAlloc(
    /// MEM_COMMIT)` is required (it zero-fills only *newly* committed pages — callers zero explicitly
    /// when they need it).
    #[cfg(unix)]
    fn hw_commit_rw(&self, off: u64, len: u64) {
        // SAFETY: `[base+off, +len)` is within the reserved mapping (validated), owned for the call.
        unsafe {
            libc::mprotect(
                self.base.add(off as usize) as *mut c_void,
                len as usize,
                libc::PROT_READ | libc::PROT_WRITE,
            );
        }
    }
    #[cfg(windows)]
    fn hw_commit_rw(&self, off: u64, len: u64) {
        // The JIT window is a **placeholder** reservation (`svm-jit`'s `mem::pal`), so a plain
        // `VirtualAlloc(MEM_COMMIT)` cannot commit a tail page — it must split the placeholder and
        // replace-commit it. Reuse the JIT's own primitive so the two stay byte-for-byte identical;
        // it is idempotent (an already-committed page is just re-asserted RW, never re-zeroed).
        // SAFETY: `[base+off, +len)` is within the reservation that produced `self.base` (validated).
        unsafe { svm_jit::win_commit_rw(self.base.add(off as usize), len as usize) }
    }

    /// Apply cap `prot` bits (`0` none / `1` read / `3` read-write) to the committed `[off, off+len)`
    /// without touching its contents — `mprotect` on unix, `VirtualProtect` on windows. `none` maps
    /// to `PROT_NONE`/`PAGE_NOACCESS` (the page stays committed but faults on access).
    #[cfg(unix)]
    fn hw_apply(&self, off: u64, len: u64, prot: i32) {
        const PROT_READ: i32 = 1;
        const PROT_WRITE: i32 = 2;
        let hw = if prot & PROT_WRITE != 0 {
            libc::PROT_READ | libc::PROT_WRITE
        } else if prot & PROT_READ != 0 {
            libc::PROT_READ
        } else {
            libc::PROT_NONE
        };
        // SAFETY: `[base+off, +len)` is within the reserved mapping (validated), owned for the call.
        unsafe {
            libc::mprotect(self.base.add(off as usize) as *mut c_void, len as usize, hw);
        }
    }
    #[cfg(windows)]
    fn hw_apply(&self, off: u64, len: u64, prot: i32) {
        use windows_sys::Win32::System::Memory::{
            VirtualProtect, PAGE_NOACCESS, PAGE_PROTECTION_FLAGS, PAGE_READONLY, PAGE_READWRITE,
        };
        const PROT_READ: i32 = 1;
        const PROT_WRITE: i32 = 2;
        let flags: PAGE_PROTECTION_FLAGS = if prot & PROT_WRITE != 0 {
            PAGE_READWRITE
        } else if prot & PROT_READ != 0 {
            PAGE_READONLY
        } else {
            PAGE_NOACCESS
        };
        let mut old: PAGE_PROTECTION_FLAGS = 0;
        // SAFETY: `[base+off, +len)` is committed (callers `hw_commit_rw` first) and in-reservation.
        unsafe {
            VirtualProtect(
                self.base.add(off as usize) as *const c_void,
                len as usize,
                flags,
                &mut old,
            );
        }
    }

    /// Hint the OS to drop the physical backing of the now-inaccessible `[off, off+len)` (a pure
    /// memory-footprint optimization, *after* the range has been zeroed + protected `none`). `unmap`
    /// semantics ("re-`map` reads zero") are already guaranteed by the explicit zero, so this need
    /// not be exact: `MADV_DONTNEED` on unix; a no-op on windows (the pages stay committed-but-
    /// `NOACCESS`, which keeps the snapshot's `restore_rw` able to read the backed prefix).
    #[cfg(unix)]
    fn hw_release_hint(&self, off: u64, len: u64) {
        // SAFETY: `[base+off, +len)` is within the reserved mapping (validated), owned for the call.
        unsafe {
            libc::madvise(
                self.base.add(off as usize) as *mut c_void,
                len as usize,
                libc::MADV_DONTNEED,
            );
        }
    }
    #[cfg(windows)]
    fn hw_release_hint(&self, _off: u64, _len: u64) {}
}

#[cfg(any(unix, windows))]
impl GuestMem for MprotectWindow {
    fn read_bytes(&self, ptr: u64, len: u64) -> Option<Vec<u8>> {
        if !self.range_committed(ptr, len, false) {
            return None;
        }
        // SAFETY: range_committed proved every page mapped+readable and `[ptr,ptr+len) ⊆ reserved`.
        let w = unsafe { std::slice::from_raw_parts(self.base, self.reserved as usize) };
        Some(w[ptr as usize..(ptr + len) as usize].to_vec())
    }
    fn write_bytes(&mut self, ptr: u64, data: &[u8]) -> Option<()> {
        if !self.range_committed(ptr, data.len() as u64, true) {
            return None;
        }
        // SAFETY: range_committed proved every page mapped+writable and the range ⊆ reserved.
        let w = unsafe { std::slice::from_raw_parts_mut(self.base, self.reserved as usize) };
        w[ptr as usize..ptr as usize + data.len()].copy_from_slice(data);
        Some(())
    }
    /// §3e op 0 `map`: (re)commit the **whole pages** covering `[offset,offset+len)` with `prot`,
    /// zero-filled — including **growth** into the reserved tail. The commit/zero/protect span the
    /// page range, not the raw `[offset, len)`, so the zeroing is page-granular and matches the
    /// interpreter's per-page `Mem::map` on any host page size (on a 16 KiB host, `len` may be a
    /// fraction of a page).
    fn map(&mut self, offset: u64, len: u64, prot: i32) -> i64 {
        let pages = match self.prot_pages(offset, len) {
            Ok(p) => p,
            Err(e) => return e,
        };
        let start = *pages.start() * self.page;
        let plen = (*pages.end() + 1 - *pages.start()) * self.page;
        // Commit + make RW so the zero-fill lands, zero (a fresh commit reads zero), then apply the
        // requested protection.
        self.hw_commit_rw(start, plen);
        // SAFETY: the page range is RW and within the reserved mapping (validated).
        unsafe { std::ptr::write_bytes(self.base.add(start as usize), 0, plen as usize) };
        for page in pages {
            self.set_prot(page, prot);
        }
        self.hw_apply(start, plen, prot);
        0
    }
    /// §3e op 1 `unmap`: decommit the **whole pages** covering `[offset,offset+len)` — any access
    /// faults, and a re-`map` reads zero. Operates on the page range (page-granular work needs whole
    /// pages) to match `Mem::unmap`.
    ///
    /// We **explicitly zero** the range so a later re-`map` reads zero on every platform: on Linux
    /// `MADV_DONTNEED` alone would suffice (next fault returns a fresh zero page), but Darwin treats
    /// it as advisory (stale bytes survive) and windows keeps the page committed — so the zero is what
    /// makes them all agree, and `hw_release_hint` is then a pure footprint optimization.
    fn unmap(&mut self, offset: u64, len: u64) -> i64 {
        let pages = match self.prot_pages(offset, len) {
            Ok(p) => p,
            Err(e) => return e,
        };
        let start = *pages.start() * self.page;
        let plen = (*pages.end() + 1 - *pages.start()) * self.page;
        // Commit + make RW, zero it, hint the OS to drop the backing, then protect NONE so any later
        // access faults (detect-and-kill).
        self.hw_commit_rw(start, plen);
        // SAFETY: the page range is RW and within the reserved mapping (validated).
        unsafe { std::ptr::write_bytes(self.base.add(start as usize), 0, plen as usize) };
        self.hw_release_hint(start, plen);
        self.hw_apply(start, plen, 0 /* none */);
        for page in pages {
            self.prot_set(page, PageState::Unmapped);
        }
        0
    }
    /// §3e op 2 `protect`: change protection without touching backing (the D40 RO mechanism). The
    /// page is committed first (a no-op on already-committed pages; on windows it makes a never-mapped
    /// reserved tail page addressable, matching the interpreter's "absent page reads zero" model)
    /// **without** zeroing live contents, then the protection is applied.
    fn protect(&mut self, offset: u64, len: u64, prot: i32) -> i64 {
        let pages = match self.prot_pages(offset, len) {
            Ok(p) => p,
            Err(e) => return e,
        };
        let start = *pages.start() * self.page;
        let plen = (*pages.end() + 1 - *pages.start()) * self.page;
        self.hw_commit_rw(start, plen);
        for page in pages {
            self.set_prot(page, prot);
        }
        self.hw_apply(start, plen, prot);
        0
    }
    /// §3e op 3 `page_size`: the hardware protection granularity (`self.page` = the host page) —
    /// the unit `map`/`unmap`/`protect` round to, matching the interpreter's `Mem::page_size` on the
    /// same host so the two backends agree.
    fn page_size(&self) -> i64 {
        self.page as i64
    }

    /// §9/§12 async-ring completion counter. The JIT's `atomic.wait` parks on the confined **physical**
    /// address `phys = base + (addr & mask)`; an offload worker bumps the counter and `notify`s that
    /// same `phys`, so the handle keys on it (vs. the interpreter's window-relative offset). `Some` only
    /// for a 4-byte-aligned, committed, writable in-window address — the same gate as a guest atomic.
    fn async_counter(&self, counter_addr: u64) -> Option<Arc<dyn AsyncCounter>> {
        let off = counter_addr & (self.reserved - 1); // the §4 mask domain, matching the JIT lowering
        if !off.is_multiple_of(4) || !self.range_committed(off, 4, true) {
            return None;
        }
        Some(Arc::new(PhysCounter {
            phys: self.base as u64 + off,
        }))
    }

    /// §13 op 0 `map`: alias a `SharedRegion` into the window with a **real shared mapping** —
    /// `mmap(MAP_SHARED | MAP_FIXED)` of the region's `os_fd` over `[win_off, win_off+len)`, so two
    /// mappings of the same region (here, or in another window) name the *same* physical pages: true
    /// hardware aliasing with zero per-access overhead (§13). The mapping persists in the window's
    /// reservation across `cap.call`s — this `MprotectWindow` is rebuilt per call, but the OS mapping
    /// and the region fd (owned by the `Host`'s backing) are not. Validation mirrors the interpreter's
    /// `Mem::map_region`. Wired on Linux (`memfd`); macOS/windows are a follow-up (→ `-EINVAL`).
    fn map_region(
        &mut self,
        win_off: u64,
        region_off: u64,
        len: u64,
        prot: i32,
        _region: u32,
        backing: RegionBacking,
    ) -> i64 {
        const EINVAL: i64 = -22;
        #[cfg(unix)]
        {
            const PROT_READ: i32 = 1;
            const PROT_WRITE: i32 = 2;
            let pages = match self.prot_pages(win_off, len) {
                Ok(p) => p,
                Err(e) => return e,
            };
            if !region_off.is_multiple_of(self.page) || prot & PROT_READ == 0 {
                return EINVAL;
            }
            match region_off.checked_add(len) {
                Some(end) if end <= backing.size() => {}
                _ => return EINVAL,
            }
            let Some(fd) = backing.os_fd() else {
                return EINVAL;
            };
            let writable = prot & PROT_WRITE != 0;
            let start = *pages.start() * self.page;
            // Whole-page span covering `[win_off, win_off+len)`. The region fd is page-rounded ≥ this,
            // so `region_off + plen` never maps past EOF (no SIGBUS); bytes past the logical region
            // size read zero on both backends.
            let plen = (*pages.end() + 1 - *pages.start()) * self.page;
            let hw = if writable {
                libc::PROT_READ | libc::PROT_WRITE
            } else {
                libc::PROT_READ
            };
            // SAFETY: `[base+start, +plen) ⊆` the reserved window (validated by `prot_pages`).
            // `MAP_FIXED` replaces those reserved pages with a shared mapping of the region fd at
            // `region_off`; the fd outlives the run (held by the Host's backing).
            let p = unsafe {
                libc::mmap(
                    self.base.add(start as usize) as *mut c_void,
                    plen as usize,
                    hw,
                    libc::MAP_SHARED | libc::MAP_FIXED,
                    fd,
                    region_off as libc::off_t,
                )
            };
            if p == libc::MAP_FAILED {
                return EINVAL;
            }
            // Mirror the software page state (committed; RW or RO) for in-call §7 borrow checks.
            let state = if writable {
                PageState::Rw
            } else {
                PageState::Ro
            };
            for page in pages {
                self.prot_set(page, state);
            }
            0
        }
        // §13 windows (issue #1): real shared mappings via **placeholder reservations**. The JIT
        // window is one `MEM_RESERVE_PLACEHOLDER` reservation (`svm-jit::mem`); to alias a section at
        // a fixed sub-range we free that sub-range back to a placeholder (`MEM_PRESERVE_PLACEHOLDER`)
        // — whether it is currently committed (the backed prefix) or an untouched placeholder tail —
        // then replace it with a view of the section (`MapViewOfFile3(MEM_REPLACE_PLACEHOLDER)`). Two
        // mappings of the same section then name the same physical pages: true hardware aliasing,
        // zero per-access overhead, persisting across `cap.call`s (the OS view + the section handle
        // held by the `Host` backing outlive this per-call `MprotectWindow`). Mirrors the unix path,
        // but at **allocation-granularity** (64 KiB) — what `MapViewOfFile3` requires for the
        // placement address and the section offset (the guest aligns to `region_page_size`, which
        // reports this granularity on windows).
        #[cfg(windows)]
        {
            use windows_sys::Win32::Foundation::HANDLE;
            use windows_sys::Win32::System::Memory::{
                MapViewOfFile3, VirtualFree, MEM_PRESERVE_PLACEHOLDER, MEM_RELEASE,
                MEM_REPLACE_PLACEHOLDER, PAGE_READONLY, PAGE_READWRITE,
            };
            use windows_sys::Win32::System::Threading::GetCurrentProcess;
            const PROT_READ: i32 = 1;
            const PROT_WRITE: i32 = 2;
            // Validate the window range (page-granular, within `[0, reserved)`) like unix…
            let pages = match self.prot_pages(win_off, len) {
                Ok(p) => p,
                Err(e) => return e,
            };
            // …then add the windows-only allocation-granularity constraints `MapViewOfFile3` imposes.
            let gran = svm_interp::host_region_granularity();
            if prot & PROT_READ == 0
                || !win_off.is_multiple_of(gran)
                || !region_off.is_multiple_of(gran)
                || !len.is_multiple_of(gran)
            {
                return EINVAL;
            }
            match region_off.checked_add(len) {
                Some(end) if end <= backing.size() => {}
                _ => return EINVAL,
            }
            let Some(section) = backing.os_section() else {
                return EINVAL;
            };
            let section = section as HANDLE;
            let writable = prot & PROT_WRITE != 0;
            let flags = if writable {
                PAGE_READWRITE
            } else {
                PAGE_READONLY
            };
            // SAFETY: GetCurrentProcess returns the current-process pseudo-handle; always safe.
            let proc = unsafe { GetCurrentProcess() };
            // Map one allocation granule at a time so each free-to-placeholder targets a single,
            // self-contained sub-range (committed prefix granule *or* placeholder tail granule).
            for i in 0..(len / gran) {
                let addr = unsafe { self.base.add((win_off + i * gran) as usize) };
                let roff = region_off + i * gran;
                // SAFETY: `[addr, addr+gran) ⊆` the reserved window (validated by `prot_pages`).
                // Free-to-placeholder decommits whatever is there (committed or placeholder) leaving
                // an exact placeholder; `MapViewOfFile3(MEM_REPLACE_PLACEHOLDER)` then aliases the
                // section over it. The section (held by the `Host` backing) outlives the run.
                unsafe {
                    VirtualFree(
                        addr as *mut c_void,
                        gran as usize,
                        MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
                    );
                    let view = MapViewOfFile3(
                        section,
                        proc,
                        addr as *const c_void,
                        roff,
                        gran as usize,
                        MEM_REPLACE_PLACEHOLDER,
                        flags,
                        core::ptr::null_mut(),
                        0,
                    );
                    if view.Value.is_null() {
                        // Fold GetLastError into the return so a red CI run names the failing call.
                        return EINVAL - last_error_win();
                    }
                }
            }
            // Mirror the software page state (committed; RW or RO) for in-call §7 borrow checks.
            let state = if writable {
                PageState::Rw
            } else {
                PageState::Ro
            };
            for page in pages {
                self.prot_set(page, state);
            }
            0
        }
        #[cfg(not(any(unix, windows)))]
        {
            let _ = (win_off, region_off, len, prot, backing);
            EINVAL
        }
    }
}

/// §9/§12 the JIT's [`AsyncCounter`]: the futex completion counter is a raw window address `phys`, so
/// an offload worker bumps it with a real atomic — the same `phys` the JIT's `atomic.wait` value-check
/// reads and the futex `notify` keys on. The run is quiesced before the window is freed
/// ([`HostAsyncHooks::finish`]), so `phys` is live whenever a worker calls this.
#[cfg(any(unix, windows))]
struct PhysCounter {
    phys: u64,
}
// SAFETY: `phys` is a stable, validated, committed window address; it is only ever atomic-accessed,
// and the offload pool is drained before the window is unmapped (no use-after-free).
#[cfg(any(unix, windows))]
unsafe impl Send for PhysCounter {}
#[cfg(any(unix, windows))]
unsafe impl Sync for PhysCounter {}

#[cfg(any(unix, windows))]
impl AsyncCounter for PhysCounter {
    fn increment(&self, delta: u64) {
        use std::sync::atomic::{AtomicU32, Ordering};
        // SAFETY: `phys` points at a 4-byte-aligned committed window word (validated in
        // `async_counter`); the run drains the pool before freeing the window, so it stays live.
        let a = unsafe { &*(self.phys as *const AtomicU32) };
        a.fetch_add(delta as u32, Ordering::SeqCst);
    }
    fn key(&self) -> u64 {
        self.phys
    }
}

/// §9/§12 the `Host`-backed [`svm_jit::AsyncHostHooks`] for the asynchronous `IoRing.submit_async`:
/// installs this JIT run's futex `notify` into the `Host` (which owns the offload pool) so a worker can
/// wake a vCPU parked on a completion counter, and drains the pool at teardown. Construct it over the
/// **same** `Host` whose pointer is the run's `cap_ctx`, and pass it to
/// [`svm_jit::compile_and_run_capture_reserved_with_host_async`].
pub struct HostAsyncHooks {
    host: *mut Host,
}

impl HostAsyncHooks {
    /// # Safety
    /// `host` must point at the live `Host` used as the run's `cap_ctx`, and outlive the run.
    pub unsafe fn new(host: *mut Host) -> HostAsyncHooks {
        HostAsyncHooks { host }
    }
}

impl svm_jit::AsyncHostHooks for HostAsyncHooks {
    fn install_notify(&self, notify: Arc<dyn Fn(u64, u32) + Send + Sync>) {
        // SAFETY: `host` is the live cap-ctx `Host`; install runs on the run thread before any vCPU.
        unsafe { (*self.host).set_async_notify(notify) };
    }
    fn finish(&self) {
        // SAFETY: same `Host`; called on the run thread after every vCPU has joined.
        unsafe {
            (*self.host).quiesce_pool();
            (*self.host).clear_async_notify();
        }
    }
}

/// `GetLastError()` as a non-negative `i64`, for folding into a `-EINVAL`-shaped return so a failing
/// Win32 call is identifiable in CI logs (no debugger). Windows-only.
#[cfg(windows)]
fn last_error_win() -> i64 {
    use windows_sys::Win32::Foundation::GetLastError;
    // SAFETY: GetLastError reads thread-local state; always safe.
    unsafe { GetLastError() as i64 }
}

/// Create a fresh anonymous, `cap`-byte OS shared-memory fd: `memfd_create` on Linux, an immediately-
/// `shm_unlink`ed POSIX `shm_open` object on other unix (macOS). The fd keeps the (unlinked) object
/// alive; closing it reclaims the memory. Sized with `ftruncate` so a window `mmap` of whole pages
/// never faults past EOF.
#[cfg(unix)]
fn create_region_fd(cap: usize) -> std::io::Result<std::os::fd::OwnedFd> {
    use std::os::fd::{FromRawFd, OwnedFd};
    #[cfg(target_os = "linux")]
    // SAFETY: a valid NUL-terminated name; returns a fresh owned fd or -1.
    let raw = unsafe { libc::memfd_create(c"svm_region".as_ptr(), libc::MFD_CLOEXEC) };
    #[cfg(not(target_os = "linux"))]
    let raw = {
        use std::sync::atomic::{AtomicU64, Ordering};
        static SEQ: AtomicU64 = AtomicU64::new(0);
        // A short unique name (POSIX shm names are length-capped): "/svm<pid·seq in hex>".
        let uniq = ((std::process::id() as u64) << 24) ^ SEQ.fetch_add(1, Ordering::Relaxed);
        let name = format!("/svm{uniq:x}\0");
        // SAFETY: a valid NUL-terminated name; O_EXCL so we own a fresh object, or -1.
        let raw = unsafe {
            libc::shm_open(
                name.as_ptr() as *const libc::c_char,
                libc::O_RDWR | libc::O_CREAT | libc::O_EXCL,
                0o600 as libc::c_int,
            )
        };
        if raw >= 0 {
            // Unlink now: the open fd keeps the object usable; it's anonymous + auto-reclaimed on close.
            // SAFETY: `name` is the just-created object's NUL-terminated name.
            unsafe { libc::shm_unlink(name.as_ptr() as *const libc::c_char) };
        }
        raw
    };
    if raw < 0 {
        return Err(std::io::Error::last_os_error());
    }
    // SAFETY: `raw` is a fresh owned fd.
    let fd = unsafe { OwnedFd::from_raw_fd(raw) };
    // SAFETY: sizing the just-created object (before any mmap), per the once-only ftruncate rule.
    if unsafe { libc::ftruncate(raw, cap as libc::off_t) } < 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(fd)
}

/// A §13 `SharedRegion` backing over a real OS shared-memory object (`memfd`/`shm`), whose `os_fd` a
/// window `mmap`s `MAP_SHARED` for true hardware aliasing. The fd is also mapped once into the host
/// process so `read_byte`/`write_byte` work (e.g. if an interpreter `Mem` uses this backing); in the
/// JIT differential the guest's loads/stores go straight through the window's shared mapping. Unix
/// only; windows (`CreateFileMapping` + placeholder reservations) is a follow-up.
#[cfg(unix)]
struct ShmBacking {
    fd: std::os::fd::OwnedFd,
    ptr: *mut u8,
    cap: usize, // page-rounded mapping length (the fd size)
    len: usize, // logical region size the guest sees
}

// SAFETY: `ptr` is a `MAP_SHARED` mapping of `fd` — a process-wide shared object, not thread-local.
// A §13 region is shared across vCPU threads (§12); `read_byte`/`write_byte` go through that shared
// mapping, and concurrent access is the guest's own race, confined to the region (never an escape).
#[cfg(unix)]
unsafe impl Send for ShmBacking {}
#[cfg(unix)]
unsafe impl Sync for ShmBacking {}

#[cfg(unix)]
impl ShmBacking {
    fn new(len: usize) -> std::io::Result<ShmBacking> {
        use std::os::fd::AsRawFd;
        let page = host_page_size() as usize;
        let cap = len.max(1).div_ceil(page) * page;
        let fd = create_region_fd(cap)?;
        // SAFETY: map the whole object shared into the host (for `read_byte`/`write_byte`).
        let p = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                cap,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd.as_raw_fd(),
                0,
            )
        };
        if p == libc::MAP_FAILED {
            return Err(std::io::Error::last_os_error());
        }
        Ok(ShmBacking {
            fd,
            ptr: p as *mut u8,
            cap,
            len,
        })
    }
}

#[cfg(unix)]
impl SharedBacking for ShmBacking {
    fn size(&self) -> u64 {
        self.len as u64
    }
    fn read_byte(&self, off: u64) -> u8 {
        if (off as usize) < self.len {
            // SAFETY: off < len ≤ cap; `ptr` maps `[0, cap)` RW for `self`'s lifetime.
            unsafe { *self.ptr.add(off as usize) }
        } else {
            0
        }
    }
    fn write_byte(&self, off: u64, b: u8) {
        if (off as usize) < self.len {
            // SAFETY: off < len ≤ cap; `ptr` maps `[0, cap)` RW for `self`'s lifetime.
            unsafe { *self.ptr.add(off as usize) = b }
        }
    }
    fn os_fd(&self) -> Option<i32> {
        use std::os::fd::AsRawFd;
        Some(self.fd.as_raw_fd())
    }
}

#[cfg(unix)]
impl Drop for ShmBacking {
    fn drop(&mut self) {
        // SAFETY: `ptr`/`cap` are the host mapping from `new`; the fd is closed by `OwnedFd`.
        unsafe { libc::munmap(self.ptr as *mut c_void, self.cap) };
    }
}

/// Create a §13 `SharedRegion` backing over a fresh `len`-byte OS shared-memory object — install it
/// with [`svm_interp::Host::grant_shared_region_backed`] so the JIT can `mmap` it for real aliasing.
#[cfg(unix)]
pub fn new_shared_region(len: usize) -> RegionBacking {
    std::sync::Arc::new(ShmBacking::new(len).expect("create shared region"))
}

/// A §13 `SharedRegion` backing over a Windows **pagefile-backed section** (`CreateFileMappingW` with
/// `INVALID_HANDLE_VALUE`), whose section handle a window aliases via `MapViewOfFile3` for true
/// hardware aliasing. Like the unix `ShmBacking`, the section is also mapped once into the host
/// process so `read_byte`/`write_byte` work; in the JIT differential the guest's loads/stores go
/// straight through the window's mapped views. The section is sized to whole allocation granules so a
/// window view of whole granules never maps past its end.
#[cfg(windows)]
struct WinShmBacking {
    section: windows_sys::Win32::Foundation::HANDLE,
    ptr: *mut u8,
    len: usize, // logical region size the guest sees
}

// SAFETY: `ptr`/`section` name a process-wide file mapping, not thread-local state. A §13 region is
// shared across vCPU threads (§12); access goes through the shared mapping and a concurrent race is
// the guest's own, confined to the region (never an escape).
#[cfg(windows)]
unsafe impl Send for WinShmBacking {}
#[cfg(windows)]
unsafe impl Sync for WinShmBacking {}

#[cfg(windows)]
impl WinShmBacking {
    fn new(len: usize) -> std::io::Result<WinShmBacking> {
        use windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE;
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MapViewOfFile, FILE_MAP_ALL_ACCESS, PAGE_READWRITE,
        };
        let gran = svm_interp::host_region_granularity() as usize;
        let cap = len.max(1).div_ceil(gran) * gran;
        // SAFETY: `INVALID_HANDLE_VALUE` + `PAGE_READWRITE` makes an anonymous pagefile-backed section
        // of `cap` bytes; NULL attrs/name → an unnamed section owned by the returned handle.
        let section = unsafe {
            CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                core::ptr::null(),
                PAGE_READWRITE,
                (cap >> 32) as u32,
                (cap & 0xffff_ffff) as u32,
                core::ptr::null(),
            )
        };
        if section.is_null() {
            return Err(std::io::Error::last_os_error());
        }
        // SAFETY: map the whole section RW into the host for `read_byte`/`write_byte`.
        let view = unsafe { MapViewOfFile(section, FILE_MAP_ALL_ACCESS, 0, 0, cap) };
        if view.Value.is_null() {
            let e = std::io::Error::last_os_error();
            // SAFETY: `section` is the just-created handle; close it on the error path.
            unsafe { windows_sys::Win32::Foundation::CloseHandle(section) };
            return Err(e);
        }
        Ok(WinShmBacking {
            section,
            ptr: view.Value as *mut u8,
            len,
        })
    }
}

#[cfg(windows)]
impl SharedBacking for WinShmBacking {
    fn size(&self) -> u64 {
        self.len as u64
    }
    fn read_byte(&self, off: u64) -> u8 {
        if (off as usize) < self.len {
            // SAFETY: off < len ≤ cap; `ptr` maps `[0, cap)` RW for `self`'s lifetime.
            unsafe { *self.ptr.add(off as usize) }
        } else {
            0
        }
    }
    fn write_byte(&self, off: u64, b: u8) {
        if (off as usize) < self.len {
            // SAFETY: off < len ≤ cap; `ptr` maps `[0, cap)` RW for `self`'s lifetime.
            unsafe { *self.ptr.add(off as usize) = b }
        }
    }
    fn os_section(&self) -> Option<isize> {
        Some(self.section as isize)
    }
}

#[cfg(windows)]
impl Drop for WinShmBacking {
    fn drop(&mut self) {
        use windows_sys::Win32::System::Memory::{UnmapViewOfFile, MEMORY_MAPPED_VIEW_ADDRESS};
        // SAFETY: `ptr` is the host mapping from `new`; the section handle is closed after.
        unsafe {
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                Value: self.ptr as *mut c_void,
            });
            windows_sys::Win32::Foundation::CloseHandle(self.section);
        }
    }
}

/// Create a §13 `SharedRegion` backing over a fresh `len`-byte Windows section — install it with
/// [`svm_interp::Host::grant_shared_region_backed`] so the JIT can alias it via `MapViewOfFile3`.
#[cfg(windows)]
pub fn new_shared_region(len: usize) -> RegionBacking {
    std::sync::Arc::new(WinShmBacking::new(len).expect("create shared region"))
}

/// How a guest program ended: its entry returned values, or it invoked `Exit(code)` (§3e).
#[derive(Debug, Clone, PartialEq)]
pub enum Outcome {
    Returned(Vec<Value>),
    Exited(i32),
}

/// The result of running a program through the powerbox: how it ended, plus the bytes it wrote
/// to stdout/stderr via the `Stream` capabilities.
#[derive(Debug, Clone)]
pub struct Run {
    pub outcome: Outcome,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
}

/// The frontend's powerbox entry shape (function 0): the three `i32` handles
/// `_start(stdout, stdin, exit)`, or four `_start(stdout, stdin, exit, memory)` once the program
/// uses the Memory capability (a guest heap that grows via `map`, §3e/§4). A module whose entry
/// matches either is a runnable *program*; anything else is a bare kernel (run with [`run_kernel`]).
pub fn is_powerbox_entry(module: &Module) -> bool {
    // The powerbox entry imports 3–8 `i32` capability handles (stdout, stdin, exit, [memory],
    // [addrspace], [ioring], [blocking], [jit] — §3e/§9/§12/DESIGN.md §22; a chibicc `_start` always
    // imports the full 8). The runner grants exactly as many as the entry declares (see
    // `run_powerbox_with_deadline`).
    matches!(
        module.funcs.first().map(|f| f.params.as_slice()),
        Some(p) if (3..=8).contains(&p.len()) && p.iter().all(|t| matches!(t, ValType::I32))
    )
}

/// The reference host's capability-import name policy (§7 "Host-defined capabilities &
/// discoverability"): the standard `name → (type_id, op)` binding that a frontend's `extern`
/// capability names resolve to at load. This is the default "powerbox ABI" the bundled toolchain
/// agrees on; a *different* host is free to supply its own resolver to `svm_ir::resolve_imports`,
/// binding these (or entirely new) names to its own capabilities — that is the §7 late binding.
///
/// Names are the bare operation names (no `__vm_` prefix); the capability **handle** is supplied
/// by the call site (the frontend's powerbox stash), never by this policy — so two names can share
/// an interface and differ only by which handle the guest passes (e.g. `write`/`read` are both
/// `Stream`, distinguished by the stdout vs stdin handle).
pub fn default_cap_resolver(name: &str) -> Option<svm_ir::ResolvedCap> {
    use svm_interp::iface;
    let (type_id, op): (u32, u32) = match name {
        // Stream — the *handle* (stdout/stdin) selects the endpoint, not the name.
        "write" => (iface::STREAM, 1),
        "read" => (iface::STREAM, 0),
        // Exit (noreturn).
        "exit" => (iface::EXIT, 0),
        // Memory management (§3e/§4).
        "vm_map" => (iface::MEMORY, 0),
        "vm_unmap" => (iface::MEMORY, 1),
        "vm_protect" => (iface::MEMORY, 2),
        "vm_page_size" => (iface::MEMORY, 3),
        // AddressSpace / SharedRegion aliasing (§13/§14).
        "vm_region_create" => (iface::ADDRESS_SPACE, 5),
        "vm_region_map" => (iface::SHARED_REGION, 0),
        "vm_region_unmap" => (iface::SHARED_REGION, 1),
        "vm_region_page_size" => (iface::SHARED_REGION, 3),
        // IoRing submit/complete (§9/§12).
        "vm_io_submit_async" => (iface::IO_RING, 1),
        "vm_io_reap" => (iface::IO_RING, 2),
        // Guest-driven JIT (§22).
        "vm_jit_compile" => (iface::JIT, 0),
        "vm_jit_compile_linked" => (iface::JIT, 5),
        "vm_jit_invoke2" => (iface::JIT, 1),
        "vm_jit_release" => (iface::JIT, 2),
        "vm_jit_install" => (iface::JIT, 3),
        "vm_jit_uninstall" => (iface::JIT, 4),
        _ => return None,
    };
    Some(svm_ir::ResolvedCap { type_id, op })
}

/// Lower a module's §7 named capability imports to concrete `cap.call`s using the reference host
/// policy ([`default_cap_resolver`]), to be called **before** `verify_module` (the resolved module
/// is what the verifier and backends run). A module with no imports is returned unchanged, so this
/// is a no-op for the legacy inline-`cap.call` form. Fails closed on an unknown import name.
pub fn resolve_capability_imports(module: Module) -> Result<Module, String> {
    if module.imports.is_empty() {
        return Ok(module);
    }
    svm_ir::resolve_imports(&module, default_cap_resolver).map_err(|e| match e {
        svm_ir::ImportError::Unresolved(n) => {
            format!("unresolved capability import `{n}` (no binding in the host policy)")
        }
        svm_ir::ImportError::BadImportIndex(i) => {
            format!("call.import references out-of-range import index {i}")
        }
        // `resolve_imports` only ever resolves to capabilities (`Resolved::Cap`), never a slot.
        svm_ir::ImportError::SlotHandleNotConst => {
            "call.import resolved to a table slot with a non-constant handle".into()
        }
    })
}

fn typed(t: ValType, v: i64) -> Value {
    match t {
        ValType::I32 => Value::I32(v as i32),
        ValType::I64 => Value::I64(v),
        ValType::F32 => Value::F32(f32::from_bits(v as u32)),
        ValType::F64 => Value::F64(f64::from_bits(v as u64)),
        ValType::Ref => Value::Ref(v as u64), // opaque i64-width reference
        // CLI entry args are scalar `i64` slots; a `v128` entry param is out of scope. Total arm:
        // zero-extend the slot into the low lanes.
        ValType::V128 => {
            let mut bytes = [0u8; 16];
            bytes[..8].copy_from_slice(&v.to_le_bytes());
            Value::V128(bytes)
        }
    }
}

/// The raw result of a JIT compile→run: the outcome plus the §5 W3 trap diagnostics (source
/// backtrace + trapping fiber, both empty/`None` unless the guest trapped under a `-g` module) and the
/// captured low-window snapshot (`snapshot_cap` bytes; empty when no snapshot was requested).
struct JitRun {
    outcome: JitOutcome,
    backtrace: Vec<JitFrameLoc>,
    trap_fiber: Option<i64>,
    snapshot: Vec<u8>,
}

/// Compile `module`'s function `func`, register the live module for the `Jit` cap's mid-run re-entry,
/// and run it over `slots` under the §5 kill-path armed by `interrupt`, seeded with `init_mem` and
/// (when `snapshot_cap` is `Some`) snapshotting the low `snapshot_cap` window bytes. A **concurrent**
/// guest (`locked` is `Some`) runs the serialized [`cap_thunk_locked`] over a per-domain `Mutex<Host>`
/// — so worker threads can `cap.call` (incl. threaded `Jit.compile`) without racing — and forgoes the
/// single-threaded-only D45 fast path; a single-threaded guest keeps the unlocked [`cap_thunk`] + raw
/// `*mut Host` + [`fast_cap_resolver`] exactly as before (zero lock cost). Exactly one of `locked` /
/// `raw_host` is used. The single low-level JIT entry, shared by [`jit_run`].
///
/// # Safety
/// `raw_host` (when `locked` is `None`) is a live `*mut Host`; `interrupt` (when `Some`) outlives the
/// call; the same `cap_thunk`/ctx/resolver contracts as [`run_powerbox_with_deadline_and_quota`].
#[allow(clippy::too_many_arguments)]
unsafe fn powerbox_compile_run(
    module: &Module,
    func: FuncIdx,
    locked: Option<&Mutex<Host>>,
    raw_host: *mut Host,
    slots: &[i64],
    interrupt: Option<&std::sync::Arc<std::sync::atomic::AtomicU64>>,
    quota: svm_jit::Quota,
    init_mem: Option<&[u8]>,
    snapshot_cap: Option<usize>,
) -> Result<JitRun, svm_jit::JitError> {
    let interrupt_ptr = interrupt.map(std::sync::Arc::as_ptr);
    if let Some(m) = locked {
        let ctx = m as *const Mutex<Host> as *mut c_void;
        let mut cm = CompiledModule::compile(
            module,
            func,
            cap_thunk_locked,
            ctx,
            svm_ir::DEFAULT_RESERVED_LOG2,
            None,
            None,
            interrupt_ptr,
            None, // no D45 fast path: the fast fns deref a raw `*mut Host`, not a `Mutex<Host>`
            quota,
            CLI_JIT_TABLE_LOG2,
        )?;
        m.lock()
            .unwrap_or_else(|e| e.into_inner())
            .set_jit_native_ctx(&mut cm as *mut CompiledModule as usize);
        let r = CompiledModule::run_raw(&mut cm, slots, init_mem, snapshot_cap, None);
        m.lock()
            .unwrap_or_else(|e| e.into_inner())
            .set_jit_native_ctx(0);
        // §5 W3 — carry the trap-time source backtrace + trapping fiber (§23-D57) out (empty/`None`
        // unless the guest trapped and the module carried `-g`), so the kill message can name *which
        // fiber* was *where*.
        return r.map(|(outcome, snapshot)| JitRun {
            outcome,
            backtrace: cm.last_trap_backtrace().to_vec(),
            trap_fiber: cm.last_trap_fiber(),
            snapshot,
        });
    }
    let mut cm = CompiledModule::compile(
        module,
        func,
        cap_thunk,
        raw_host as *mut c_void,
        svm_ir::DEFAULT_RESERVED_LOG2,
        None,
        None,
        interrupt_ptr,
        Some(fast_cap_resolver),
        quota,
        CLI_JIT_TABLE_LOG2,
    )?;
    let host = &mut *raw_host;
    host.set_jit_native_ctx(&mut cm as *mut CompiledModule as usize);
    let r = CompiledModule::run_raw(&mut cm, slots, init_mem, snapshot_cap, None);
    host.set_jit_native_ctx(0);
    r.map(|(outcome, snapshot)| JitRun {
        outcome,
        backtrace: cm.last_trap_backtrace().to_vec(),
        trap_fiber: cm.last_trap_fiber(),
        snapshot,
    })
}

/// Run `module`'s entry (function 0) on the JIT under the MVP powerbox (§3e): a writable
/// `stdout`, a readable `stdin` seeded from `stdin`, and `Exit` — the three handles the
/// frontend's `_start` expects, granted in declared order. Returns the outcome and captured
/// output. `Err` if the (already-verified) module fails to JIT-compile, or if the guest
/// **traps** (detect-and-kill, §5) — the guest can never corrupt the host. Unbounded execution
/// (no §5 kill-path); use [`run_powerbox_with_deadline`] to bound a possibly-runaway guest.
pub fn run_powerbox(module: &Module, stdin: &[u8]) -> Result<Run, String> {
    run_powerbox_with_deadline(module, stdin, None)
}

/// Like [`run_powerbox`], but arm the §5 fuel/epoch kill-path with `deadline`: a watchdog thread
/// stops a **runaway** guest (infinite loop / unbounded recursion) `deadline` after it starts,
/// surfacing as an `Err` (detect-and-kill) instead of hanging the process. `None` ⇒ the ordinary
/// unbounded run. The watchdog wakes early the moment the run finishes, so a fast program is never
/// delayed. The `svm-run` CLI reads `SVM_DEADLINE_MS` and passes it here; an embedder supplies its
/// own policy (reading process env vars is the CLI's job, not the library's). Uses the default
/// (anti-bomb-ceiling) spawn quota — use [`run_powerbox_with_deadline_and_quota`] to tighten it.
pub fn run_powerbox_with_deadline(
    module: &Module,
    stdin: &[u8],
    deadline: Option<std::time::Duration>,
) -> Result<Run, String> {
    run_powerbox_with_deadline_and_quota(module, stdin, deadline, Quota::default())
}

/// [`run_powerbox_with_deadline`] + a §15 **spawn quota**: cap how many fibers (`cont.new`) and
/// concurrently-live vCPUs (`thread.spawn`) the guest may create, *below* the fixed anti-bomb
/// ceilings — DoS *containment* the embedder configures (the deadline bounds runaway *execution*; the
/// quota bounds runaway *spawning*). The quota binds **both** backends (here, the JIT; the same
/// [`Quota`] on a [`Host`] would bind the interpreter). Exceeding it detect-and-kills the guest
/// (`FiberFault`/`ThreadFault`). [`Quota::default`] = the ceilings (unbounded-ish). The `svm-run` CLI
/// reads `SVM_MAX_FIBERS`/`SVM_MAX_VCPUS` and passes them here.
pub fn run_powerbox_with_deadline_and_quota(
    module: &Module,
    stdin: &[u8],
    deadline: Option<std::time::Duration>,
    quota: Quota,
) -> Result<Run, String> {
    run_powerbox_inner(module, stdin, &[], &[], deadline, quota)
}

/// Build the §3e powerbox **args buffer** from `args` (the `argv` vector — `args[0]` is the program
/// name) and `env` (the `envp` vector, each `KEY=VALUE`), at the layout
/// `svm_ir::POWERBOX_ARGS_BASE` documents: `{ argc:u32-LE, envc:u32-LE }` then the packed
/// NUL-terminated strings. An argument containing an embedded NUL, or a blob that would reach
/// `POWERBOX_ARGS_END`, is rejected (the C `argv[]` model can't represent the former, and the latter
/// would collide with the program's data segments).
fn build_args_blob(args: &[&[u8]], env: &[&[u8]]) -> Result<Vec<u8>, String> {
    let mut blob = Vec::new();
    blob.extend_from_slice(&(args.len() as u32).to_le_bytes());
    blob.extend_from_slice(&(env.len() as u32).to_le_bytes());
    for s in args.iter().chain(env.iter()) {
        if s.contains(&0) {
            return Err("powerbox arg/env contains an embedded NUL".into());
        }
        blob.extend_from_slice(s);
        blob.push(0);
    }
    if svm_ir::POWERBOX_ARGS_BASE + blob.len() as u64 > svm_ir::POWERBOX_ARGS_END {
        return Err(format!(
            "powerbox args buffer ({} bytes) does not fit in the args region [{}, {})",
            blob.len(),
            svm_ir::POWERBOX_ARGS_BASE,
            svm_ir::POWERBOX_ARGS_END
        ));
    }
    Ok(blob)
}

/// Like [`run_powerbox`], but hand the guest a program-arguments vector (and environment): the
/// frontend's `_start` for a `main(int, char**)` parses these into `argc`/`argv` (§3e / D44). For a
/// `main(void)` program the buffer is simply unread. `args[0]` is conventionally the program name;
/// `env` entries are `KEY=VALUE`. See [`build_args_blob`] for the (bounded) layout.
pub fn run_powerbox_with_args(
    module: &Module,
    stdin: &[u8],
    args: &[&[u8]],
    env: &[&[u8]],
) -> Result<Run, String> {
    run_powerbox_inner(module, stdin, args, env, None, Quota::default())
}

/// The full powerbox entry: a program-arguments vector + environment (§3e args buffer) *and* the §5
/// kill-path `deadline` / §15 spawn `quota`. The `svm-run` CLI uses this to forward its post-`--`
/// arguments to the guest while still bounding a runaway/spawn-bomb guest.
pub fn run_powerbox_with_args_and_limits(
    module: &Module,
    stdin: &[u8],
    args: &[&[u8]],
    env: &[&[u8]],
    deadline: Option<std::time::Duration>,
    quota: Quota,
) -> Result<Run, String> {
    run_powerbox_inner(module, stdin, args, env, deadline, quota)
}

fn run_powerbox_inner(
    module: &Module,
    stdin: &[u8],
    args: &[&[u8]],
    env: &[&[u8]],
    deadline: Option<std::time::Duration>,
    quota: Quota,
) -> Result<Run, String> {
    // The fixed §3e powerbox preset, expressed over the converged [`Instance`] core (F1): the
    // arity-based grant ([`grant_powerbox_prefix`]) and the JIT compile→run + §5 watchdog
    // ([`run_jit`]) now live in exactly one place, shared with the frontend-independent embedding
    // API. The `Instance` is built directly (not via [`instantiate`]) so this path does **not**
    // re-resolve or re-verify — preserving its behaviour for already-validated frontend output
    // (chibicc / svm-llvm), which emits inline `cap.call`s and a func-0 `_start` and runs JIT-only.
    let inst = Instance {
        module: module.clone(),
        binding: None,
    };
    let config = RunConfig {
        limits: Limits {
            fuel: None,
            deadline,
            max_fibers: quota.max_fibers,
            max_vcpus: quota.max_vcpus,
        },
        stdin: stdin.to_vec(),
        memory_size_log2: None,
        args: args.iter().map(|s| s.to_vec()).collect(),
        env: env.iter().map(|s| s.to_vec()).collect(),
    };
    inst.run(Backend::Jit, &config)
}

/// Run a bare (non-powerbox) kernel — `module`'s entry on the JIT with `args` and no host
/// capabilities — returning its typed result values. For hand-written IR that is a pure
/// function rather than a program (e.g. the benchmark kernels). `Err` on compile failure,
/// a guest trap, or an `Exit` (a kernel should not call one).
pub fn run_kernel(module: &Module, args: &[i64]) -> Result<Vec<Value>, String> {
    match compile_and_run(module, 0, args).map_err(|e| format!("JIT compile failed: {e:?}"))? {
        JitOutcome::Returned(s) => {
            let results = &module.funcs[0].results;
            Ok(s.iter().zip(results).map(|(&v, t)| typed(*t, v)).collect())
        }
        JitOutcome::Exited(code) => Err(format!("kernel called Exit({code})")),
        JitOutcome::Trapped(kind) => Err(format!("kernel trapped ({kind:?})")),
    }
}

/// Pack a typed [`Value`] into the raw `i64` register slot the JIT entry takes (the inverse of the
/// reference-host [`typed`]). A `v128` arg keeps its low 8 bytes (CLI/entry args are scalar slots).
fn value_slot(v: Value) -> i64 {
    match v {
        Value::I32(x) => x as i64,
        Value::I64(x) => x,
        Value::F32(x) => x.to_bits() as i64,
        Value::F64(x) => x.to_bits() as i64,
        Value::Ref(x) => x as i64,
        Value::V128(b) => i64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]),
    }
}

/// Grant the MVP powerbox (§3e) — the **contiguous prefix** of `n_handles` of the eight fixed
/// `VM_CAP_*` capabilities, in the canonical order the synthesized `_start` expects (stdout, stdin,
/// exit, memory, addrspace, ioring, blocking, jit). Mirrors the grant order of [`run_powerbox_inner`]
/// and the C-frontend test harness so handle values are deterministic; granted identically on the
/// two backends' hosts, the values match (asserted by [`Instance::run_powerbox_diff`]).
fn grant_powerbox_prefix(h: &mut Host, n_handles: usize, win: u64) -> Vec<Value> {
    // Guest-minted §13/§14 regions need an OS-shared-memory backing so the JIT can `map` them; the
    // `Jit` cap (slot 7) needs the canonical blob validator. Both are inert if never used.
    h.set_region_factory(new_shared_region);
    h.set_jit_validator(jit_blob_validator);
    let mem_log2 = (win != 0).then(|| win.trailing_zeros() as u8);
    let mut v = Vec::with_capacity(n_handles);
    if n_handles >= 1 {
        v.push(Value::I32(h.grant_stream(StreamRole::Out)));
    }
    if n_handles >= 2 {
        v.push(Value::I32(h.grant_stream(StreamRole::In)));
    }
    if n_handles >= 3 {
        v.push(Value::I32(h.grant_exit()));
    }
    if n_handles >= 4 {
        v.push(Value::I32(h.grant_memory()));
    }
    if n_handles >= 5 {
        v.push(Value::I32(h.grant_address_space(0, win)));
    }
    if n_handles >= 6 {
        v.push(Value::I32(h.grant_io_ring()));
    }
    if n_handles >= 7 {
        v.push(Value::I32(
            h.grant_blocking(std::time::Duration::ZERO, None),
        ));
    }
    if n_handles >= 8 {
        // Reserve the `call_indirect` install table at `CLI_JIT_TABLE_LOG2` — the **same** value the
        // JIT compile uses (see [`powerbox_compile_run`]) — so a `Jit.install` guest has room. This
        // matches `run_powerbox_inner`'s grant exactly (the two paths are now one; see F1).
        v.push(Value::I32(
            h.grant_jit_with_table(mem_log2, CLI_JIT_TABLE_LOG2),
        ));
    }
    // §7 register the granted prefix under canonical names (F7) so a fixed-powerbox guest can also
    // `cap.self`-resolve its capabilities by name, not only by stash slot. Names parallel the grant
    // order above; only the `n_handles` actually granted are registered.
    for (name, slot) in POWERBOX_CAP_NAMES.iter().zip(&v) {
        if let Value::I32(handle) = slot {
            h.register_cap_name(name, *handle);
        }
    }
    v
}

/// The canonical names of the eight fixed §3e powerbox capabilities, in grant order — the vocabulary a
/// fixed-powerbox guest resolves against via `cap.self` (F7). A name-bound guest
/// ([`instantiate_with_imports`]) instead resolves its own import names.
const POWERBOX_CAP_NAMES: [&str; 8] = [
    "stdout",
    "stdin",
    "exit",
    "memory",
    "addrspace",
    "ioring",
    "blocking",
    "jit",
];

/// Reconcile the interpreter's `Result<Vec<Value>, Trap>` with the JIT's [`JitOutcome`] for an entry
/// whose results are `results`: assert the two backends agree (the differential oracle of
/// `run_c_full`) and fold them into one [`Outcome`]. `Err` if they diverge or the guest trapped.
fn diff_outcome(
    results: &[ValType],
    interp: Result<Vec<Value>, Trap>,
    jit: JitOutcome,
) -> Result<Outcome, String> {
    match (interp, jit) {
        (Ok(want), JitOutcome::Returned(got)) => {
            let got_typed: Vec<Value> = results
                .iter()
                .zip(&got)
                .map(|(t, &v)| typed(*t, v))
                .collect();
            if want != got_typed {
                return Err(format!(
                    "interp/JIT results diverge: interp={want:?} jit={got_typed:?}"
                ));
            }
            Ok(Outcome::Returned(want))
        }
        (Err(Trap::Exit(wi)), JitOutcome::Exited(gj)) => {
            if wi != gj {
                return Err(format!(
                    "interp/JIT exit codes diverge: interp={wi} jit={gj}"
                ));
            }
            Ok(Outcome::Exited(wi))
        }
        (Err(t), j) if !matches!(t, Trap::Exit(_)) => Err(format!(
            "guest trapped under the interpreter ({t:?}); jit={j:?}"
        )),
        (i, j) => Err(format!(
            "interp/JIT outcomes diverge: interp={i:?} jit={j:?}"
        )),
    }
}

/// Fold an interpreter result (`TreeWalk`/`Bytecode`) into an [`Outcome`]: a clean return, an
/// `Exit(code)`, or a trap (detect-and-kill, surfaced as `Err`). The interpreter already returns typed
/// [`Value`]s, so no result-type table is needed (unlike [`outcome_from_jit`]).
fn outcome_from_interp(r: Result<Vec<Value>, Trap>) -> Result<Outcome, String> {
    match r {
        Ok(v) => Ok(Outcome::Returned(v)),
        Err(Trap::Exit(code)) => Ok(Outcome::Exited(code)),
        Err(t) => Err(format!("guest trapped ({t:?}) — detect-and-kill (§5)")),
    }
}

/// Fold a JIT outcome into an [`Outcome`], typing the raw result slots. A `Trapped` is normally folded
/// to `Err` earlier (with the backtrace + trapping fiber) by [`run_jit`]; handled here for totality.
fn outcome_from_jit(results: &[ValType], jit: JitOutcome) -> Result<Outcome, String> {
    match jit {
        JitOutcome::Returned(s) => Ok(Outcome::Returned(
            results.iter().zip(&s).map(|(t, &v)| typed(*t, v)).collect(),
        )),
        JitOutcome::Exited(code) => Ok(Outcome::Exited(code)),
        JitOutcome::Trapped(kind) => Err(format!("guest trapped ({kind:?})")),
    }
}

/// The default per-op fuel budget for the interpreters when [`Limits::fuel`] is `None` — generous, but
/// finite so a non-terminating guest under the tree-walker can't hang the host (a runaway guest is
/// better bounded by a `deadline` on the JIT, which has no cheap per-op counter).
const DEFAULT_FUEL: u64 = 1 << 34;

/// Which execution backend a run targets. All three honour the same [`RunConfig`] where they support
/// it; the differential oracle ([`Instance::run_diff`]) cross-checks `TreeWalk` against `Jit`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    /// The reference tree-walking interpreter (the §18 differential oracle).
    TreeWalk,
    /// The bytecode engine (transparently falls back to the tree-walker for ops it doesn't lower).
    Bytecode,
    /// The Cranelift JIT.
    Jit,
}

/// Resource limits applied **uniformly across backends, where each supports them** — the consumer
/// sets these once regardless of backend. Two knobs are inherently backend-specific (and documented as
/// such): `fuel` is the interpreters' per-op budget (the JIT has no cheap per-op counter), and
/// `deadline` arms the JIT's §5 watchdog (the interpreters bound themselves with `fuel`). The spawn
/// quota (`max_fibers` / `max_vcpus`) and the window size (via [`RunConfig::memory_size_log2`]) bind
/// all three.
#[derive(Clone, Debug)]
pub struct Limits {
    /// Per-op budget for `TreeWalk`/`Bytecode` (`None` ⇒ [`DEFAULT_FUEL`]); ignored by the JIT.
    pub fuel: Option<u64>,
    /// Wall-clock deadline for the JIT's detect-and-kill watchdog (§5); ignored by the interpreters.
    pub deadline: Option<std::time::Duration>,
    /// §15 spawn quota — max fibers (`cont.new`) a run may create.
    pub max_fibers: usize,
    /// §15 spawn quota — max concurrently-live vCPUs (`thread.spawn`); the "CPUs available" cap.
    pub max_vcpus: usize,
}

impl Default for Limits {
    fn default() -> Limits {
        let q = Quota::default();
        Limits {
            fuel: None,
            deadline: None,
            max_fibers: q.max_fibers,
            max_vcpus: q.max_vcpus,
        }
    }
}

impl Limits {
    /// The §15 spawn quota these limits imply (the interpreter form; the JIT form has identical fields).
    fn quota(&self) -> Quota {
        Quota {
            max_fibers: self.max_fibers,
            max_vcpus: self.max_vcpus,
        }
    }
}

/// How to run a powerbox entry: the resource [`Limits`], the guest's stdin, and an optional override of
/// the module's declared window size (the "amount of memory available"). `Default` is the easy button —
/// default limits, empty stdin, the module's own window.
#[derive(Clone, Debug, Default)]
pub struct RunConfig {
    pub limits: Limits,
    pub stdin: Vec<u8>,
    /// Override the module's linear-memory window `size_log2` for this run (must be ≥ what the program
    /// needs, or the guest faults). `None` ⇒ the module's declared size.
    pub memory_size_log2: Option<u8>,
    /// The guest's program-arguments vector (`argv`): `args[0]` is conventionally the program name. A
    /// `main(int, char**)` `_start` parses these (§3e / D44); a `main(void)` program leaves them
    /// unread. Empty ⇒ no args buffer is seeded (identical to the legacy no-args run).
    pub args: Vec<Vec<u8>>,
    /// The guest's environment vector (`envp`), each entry `KEY=VALUE`. See [`RunConfig::args`].
    pub env: Vec<Vec<u8>>,
}

impl RunConfig {
    /// The §3e args buffer to seed the window's low bytes with (argv/env at `POWERBOX_ARGS_BASE`), or
    /// `None` when neither `args` nor `env` is set. Seeded *before* the module's data segments (which
    /// live at/above `POWERBOX_ARGS_END`), so the two never overlap. The single source for the
    /// powerbox args layout (shared by `Instance::run`/`run_diff` and the `run_powerbox*` wrappers).
    fn init_mem(&self) -> Result<Option<Vec<u8>>, String> {
        if self.args.is_empty() && self.env.is_empty() {
            return Ok(None);
        }
        let args: Vec<&[u8]> = self.args.iter().map(|v| v.as_slice()).collect();
        let env: Vec<&[u8]> = self.env.iter().map(|v| v.as_slice()).collect();
        let blob = build_args_blob(&args, &env)?;
        let mut buf = vec![0u8; svm_ir::POWERBOX_ARGS_BASE as usize + blob.len()];
        buf[svm_ir::POWERBOX_ARGS_BASE as usize..].copy_from_slice(&blob);
        Ok(Some(buf))
    }
}

/// Run `f` with the §5 kill-path armed when `deadline` is `Some`: a watchdog thread sets an interrupt
/// cell after `deadline` (the JIT polls it at back-edges → detect-and-kill), and wakes early when `f`
/// returns so a fast run is never delayed. `None` ⇒ no watchdog. Mirrors `run_powerbox_inner`'s arming.
fn with_deadline<T>(
    deadline: Option<std::time::Duration>,
    f: impl FnOnce(Option<&std::sync::Arc<std::sync::atomic::AtomicU64>>) -> T,
) -> T {
    use std::sync::atomic::{AtomicU64, Ordering};
    match deadline.filter(|d| !d.is_zero()) {
        None => f(None),
        Some(d) => {
            let interrupt = std::sync::Arc::new(AtomicU64::new(0));
            let (done_tx, done_rx) = std::sync::mpsc::channel::<()>();
            let wd = interrupt.clone();
            let handle = std::thread::spawn(move || {
                if done_rx.recv_timeout(d).is_err() {
                    wd.store(1, Ordering::SeqCst);
                }
            });
            let out = f(Some(&interrupt));
            let _ = done_tx.send(()); // run finished — wake the watchdog so it exits promptly
            let _ = handle.join();
            out
        }
    }
}

/// The single JIT compile→run path: run `func` on the JIT under `limits` (quota + optional deadline
/// watchdog), seeded with `init_mem` and (when `snapshot_cap` is `Some`) returning the low-window
/// snapshot, folding a guest trap into an `Err` (with the §5 W3 backtrace + trapping fiber). A
/// concurrent guest serializes the cap-thunk over a `Mutex<Host>`; a single-threaded guest keeps the
/// unlocked fast path. Backs both the run-once [`run_jit`] (`func` 0, no snapshot) and the reactor
/// per-call capture ([`run_capture_on`]'s `Jit` arm: an export `func`, `REACTOR_SNAP_CAP` snapshot).
fn jit_run(
    m: &Module,
    func: FuncIdx,
    slots: &[i64],
    host: &mut Host,
    limits: &Limits,
    init_mem: Option<&[u8]>,
    snapshot_cap: Option<usize>,
) -> Result<(JitOutcome, Vec<u8>), String> {
    // One shared `Quota` type now (F6) — no interp→JIT facade conversion; reuse `Limits`' quota directly.
    let quota = limits.quota();
    let concurrent = m.funcs.iter().any(|f| f.uses_concurrency());
    // SAFETY: `host` outlives the run; the watchdog interrupt (if armed) outlives it too (joined inside
    // `with_deadline`); `init_mem` (when `Some`) outlives the call; the thunk/ctx contracts hold.
    let run = with_deadline(limits.deadline, |interrupt| {
        if concurrent {
            let locked = Mutex::new(std::mem::take(host));
            let r = unsafe {
                powerbox_compile_run(
                    m,
                    func,
                    Some(&locked),
                    std::ptr::null_mut(),
                    slots,
                    interrupt,
                    quota,
                    init_mem,
                    snapshot_cap,
                )
            };
            *host = locked.into_inner().unwrap_or_else(|e| e.into_inner());
            r
        } else {
            unsafe {
                powerbox_compile_run(
                    m,
                    func,
                    None,
                    host,
                    slots,
                    interrupt,
                    quota,
                    init_mem,
                    snapshot_cap,
                )
            }
        }
    })
    .map_err(|e| format!("JIT compile failed: {e:?}"))?;
    if let JitOutcome::Trapped(kind) = run.outcome {
        let who = match run.trap_fiber {
            Some(h) if h >= 0 => format!(" [fiber {h}]"),
            _ => String::new(),
        };
        return Err(format!(
            "guest trapped ({kind:?}) — detect-and-kill (§5){who}{}",
            format_backtrace(&run.backtrace)
        ));
    }
    Ok((run.outcome, run.snapshot))
}

/// Compile + run function 0 on the JIT under `limits` (the run-once powerbox entry, no snapshot),
/// folding a guest trap into an `Err`. A thin wrapper over [`jit_run`]. Shared by the single-backend
/// [`Instance::run`] and the [`Instance::run_diff`] oracle.
fn run_jit(
    m: &Module,
    slots: &[i64],
    host: &mut Host,
    limits: &Limits,
    init_mem: Option<&[u8]>,
) -> Result<JitOutcome, String> {
    jit_run(m, 0, slots, host, limits, init_mem, None).map(|(outcome, _snap)| outcome)
}

/// Run an interpreter `backend` (`TreeWalk`/`Bytecode`) on `func`, seeding the window with `init_mem`
/// when present (the §3e argv/env buffer) and discarding the run-once snapshot. With no `init_mem` it
/// keeps the zero-overhead fast paths ([`run_with_host`] / [`run_with_host_fast`]); with one it routes
/// through the capture-reserved variants (the only interp entries that seed). `Bytecode` falls back to
/// the tree-walker for modules the engine doesn't lower (matching `TreeWalk` exactly there). Shared by
/// [`Instance::run`] and the [`Instance::run_diff`] oracle so both seed args identically.
fn run_interp(
    backend: Backend,
    m: &Module,
    func: FuncIdx,
    args: &[Value],
    fuel: &mut u64,
    init_mem: Option<&[u8]>,
    host: &mut Host,
) -> Result<Vec<Value>, Trap> {
    match (backend, init_mem) {
        (Backend::Jit, _) => unreachable!("run_interp is interpreter-only"),
        (Backend::TreeWalk, None) => run_with_host(m, func, args, fuel, host),
        (Backend::Bytecode, None) => run_with_host_fast(m, func, args, fuel, host),
        (Backend::TreeWalk, Some(mem)) => {
            run_capture_reserved_with_host(
                m,
                func,
                args,
                fuel,
                mem,
                svm_ir::DEFAULT_RESERVED_LOG2,
                host,
            )
            .0
        }
        (Backend::Bytecode, Some(mem)) => {
            match svm_interp::bytecode::compile_and_run_capture_reserved_with_host(
                m,
                func,
                args,
                fuel,
                mem,
                svm_ir::DEFAULT_RESERVED_LOG2,
                host,
            ) {
                Some((r, _snap)) => r,
                None => {
                    run_capture_reserved_with_host(
                        m,
                        func,
                        args,
                        fuel,
                        mem,
                        svm_ir::DEFAULT_RESERVED_LOG2,
                        host,
                    )
                    .0
                }
            }
        }
    }
}

/// A host capability offered to a module's named import (wasm-style import matching, §7). It carries
/// the `(type_id, op)` the guest's `call.import "<name>"` lowers to *and* a re-grantable action that
/// mints the backing handle on a [`Host`]. Re-grantable (a plain `Fn`, not `FnOnce`) because the
/// differential wrapper grants it on **two** hosts (interpreter + JIT) which must agree; grants are
/// deterministic, so granting in the same order on both yields the same handle value.
/// The re-grantable grant action a [`HostCap`] carries: `(host, window_size) -> handle`. The window
/// size serves window-scoped caps (e.g. `AddressSpace`); most ignore it. `Arc` + `Send`/`Sync` so a
/// `HostCap` is cheap to clone and the differential wrapper can grant it on either backend's host.
type GrantFn = Arc<dyn Fn(&mut Host, u64) -> i32 + Send + Sync>;

#[derive(Clone)]
pub struct HostCap {
    type_id: u32,
    op: u32,
    grant: GrantFn,
}

impl HostCap {
    /// A `Stream` write endpoint (stdout): `write(buf, len)` is op 1.
    pub fn stdout() -> HostCap {
        HostCap {
            type_id: iface::STREAM,
            op: 1,
            grant: Arc::new(|h, _| h.grant_stream(StreamRole::Out)),
        }
    }
    /// A `Stream` read endpoint (stdin): `read(buf, len)` is op 0.
    pub fn stdin() -> HostCap {
        HostCap {
            type_id: iface::STREAM,
            op: 0,
            grant: Arc::new(|h, _| h.grant_stream(StreamRole::In)),
        }
    }
    /// The `Exit` lifecycle capability: `exit(code)` (op 0, noreturn).
    pub fn exit() -> HostCap {
        HostCap {
            type_id: iface::EXIT,
            op: 0,
            grant: Arc::new(|h, _| h.grant_exit()),
        }
    }
    /// The `Clock` capability: `now(clock_id) -> i64` (op 0).
    pub fn clock() -> HostCap {
        HostCap {
            type_id: iface::CLOCK,
            op: 0,
            grant: Arc::new(|h, _| h.grant_clock()),
        }
    }
    /// A **host-defined** capability (iface [`iface::HOST_FN`]) — arbitrary semantics behind a named
    /// import, the wasm-like escape hatch. `op` is the operation this name selects; `make` builds a
    /// fresh handler per host (called once per backend, so it must be re-buildable). The handler is
    /// `(op, args, guest_mem) -> result slots | Trap`.
    pub fn host_fn(op: u32, make: impl Fn() -> HostFn + Send + Sync + 'static) -> HostCap {
        let make = Arc::new(make);
        HostCap {
            type_id: iface::HOST_FN,
            op,
            grant: Arc::new(move |h, _| h.grant_host_fn(make())),
        }
    }
    /// A fully custom binding: an explicit `(type_id, op)` and a re-grantable grant action. The escape
    /// hatch for any capability the named constructors don't cover (e.g. `Memory`, `AddressSpace`).
    pub fn custom(
        type_id: u32,
        op: u32,
        grant: impl Fn(&mut Host, u64) -> i32 + Send + Sync + 'static,
    ) -> HostCap {
        HostCap {
            type_id,
            op,
            grant: Arc::new(grant),
        }
    }
}

/// A name → [`HostCap`] registry: the capabilities a host offers a module's imports, matched **by
/// name** at [`instantiate_with_imports`] (wasm-style linking — arbitrary names, interfaces, and
/// counts). The fixed §3e powerbox is just one preset over this mechanism (see [`instantiate`]).
#[derive(Default, Clone)]
pub struct Imports {
    map: HashMap<String, HostCap>,
}

impl Imports {
    pub fn new() -> Imports {
        Imports::default()
    }
    /// Offer `cap` under `name`. Builder-style; last write wins.
    pub fn provide(mut self, name: impl Into<String>, cap: HostCap) -> Imports {
        self.map.insert(name.into(), cap);
        self
    }
}

/// The name-bound capability set captured at [`instantiate_with_imports`]: the registry plus the
/// module's import order (slot `i` of the powerbox stash ↔ import `i`), so grant order matches the
/// stash layout `svm_ir::synth_powerbox_start` lays down.
struct NamedBinding {
    imports: Imports,
    order: Vec<String>,
}

/// A resolved, verified program ready to run on **both** backends — the easy "instantiate &amp; run"
/// default over a frontend's IR (built by [`instantiate`] / [`instantiate_with_imports`]). This is the
/// [`run_powerbox`] / `run_c_full` experience **decoupled from any C frontend**: hand it a module whose
/// function 0 is a powerbox `_start` (e.g. produced by [`svm_ir::synth_powerbox_start`]) and
/// [`Instance::call`] grants the capabilities, runs the entry on the interpreter *and* the JIT under
/// identical capabilities, asserts they agree (interp == jit), and returns the captured output.
///
/// The handle / object-capability model remains the escape hatch: for a fully custom setup, grant on a
/// [`svm_interp::Host`] yourself and call [`svm_interp::run_with_host`] /
/// [`svm_jit::compile_and_run_with_host`] directly. This wrapper is the default for the common case.
pub struct Instance {
    module: Module,
    // `Some` when built via `instantiate_with_imports` (name-bound capabilities); `None` for the fixed
    // powerbox preset (`instantiate`).
    binding: Option<NamedBinding>,
}

/// Resolve `module`'s §7 named capability imports under the reference host policy
/// ([`resolve_capability_imports`]) and verify the result (the escape-freedom gate, §2a). Entry
/// points are reached by name through the module's first-class [`svm_ir::Module::exports`] table — a
/// `svm_ir::synth_powerbox_start` module registers its bootstrap as `"_start"`, and any frontend
/// export (e.g. `"main"`) survives there too.
///
/// Returns an `Err` (fail-closed) if an import is unbound or the module fails verification — exactly
/// the gates a frontend's output must pass before it can run.
pub fn instantiate(module: Module) -> Result<Instance, String> {
    let resolved = resolve_capability_imports(module)?;
    svm_verify::verify_module(&resolved)
        .map_err(|e| format!("verify failed (fail-closed): {e:?}"))?;
    Ok(Instance {
        module: resolved,
        binding: None,
    })
}

/// Instantiate `module` against a **name-keyed capability registry** (`imports`), wasm-style: each
/// `call.import "<name>"` is matched by name to a [`HostCap`], lowered to its `(type_id, op)`, and —
/// at [`Instance::call`] — granted in import order so the powerbox stash slot `i`
/// (`svm_ir::synth_powerbox_start`) holds the handle for import `i`. This is decision #2's *dynamic,
/// name-based* binding: arbitrary names, interfaces, and counts, with the fixed §3e powerbox
/// ([`instantiate`]) just one preset over the same machinery.
///
/// `module` is the post-`synth_powerbox_start` module (function 0 is the `_start`; the import table is
/// untouched by the prepend). Fails closed if an imported name has no binding in `imports`, or the
/// resolved module fails verification.
pub fn instantiate_with_imports(module: Module, imports: Imports) -> Result<Instance, String> {
    // Capture the import order *before* resolving (which clears the table). Slot i ↔ import i.
    let order: Vec<String> = module.imports.iter().map(|i| i.name.clone()).collect();
    // Resolve every name through the registry; an unbound name is fail-closed (no silent no-op).
    let resolved = svm_ir::resolve_imports(&module, |name| {
        imports.map.get(name).map(|c| svm_ir::ResolvedCap {
            type_id: c.type_id,
            op: c.op,
        })
    })
    .map_err(|e| match e {
        svm_ir::ImportError::Unresolved(n) => {
            format!("unbound capability import `{n}` (no binding in the host registry)")
        }
        other => format!("resolve imports: {other:?}"),
    })?;
    svm_verify::verify_module(&resolved)
        .map_err(|e| format!("verify failed (fail-closed): {e:?}"))?;
    Ok(Instance {
        module: resolved,
        binding: Some(NamedBinding { imports, order }),
    })
}

impl Instance {
    /// The resolved, verified module (function 0 is the powerbox `_start`).
    pub fn module(&self) -> &Module {
        &self.module
    }

    /// Run the named export and return its outcome plus captured stdout/stderr.
    ///
    /// For the powerbox entry (`"_start"`, function 0), the capabilities are auto-granted — the
    /// name-bound registry (from [`instantiate_with_imports`]) if present, else the fixed §3e powerbox
    /// ([`instantiate`]) — and `args` must be empty. This is the easy default: [`Instance::run_diff`]
    /// with [`RunConfig::default`] (interpreter == JIT enforced). Any other export runs as a **bare
    /// kernel** with `args` and **no host capabilities** (the escape hatch for pure functions).
    ///
    /// Why a non-`_start` export gets no capabilities (decision F3): without `_start` having run, the
    /// powerbox **handle stash** (window offset 0) is empty, so a granted handle would be unreachable
    /// by the export anyway — granting caps to a one-shot kernel call would be a footgun, not a feature.
    /// A cap-using export is meant to be reached through a [`Session`] ([`Instance::start`]): the
    /// reactor runs `_start` once to stash the handles, then calls exports against the live window. So
    /// the rule is: **pure function → `Instance::call`; cap-using export → `Session::call_export`.**
    ///
    /// For a single backend or non-default limits, use [`Instance::run`] / [`Instance::run_diff`].
    pub fn call(&self, export: &str, args: &[Value]) -> Result<Run, String> {
        let fidx = self
            .module
            .resolve_export(export)
            .ok_or_else(|| format!("no export named `{export}`"))?;
        let is_powerbox_func0 =
            fidx == 0 && (self.binding.is_some() || is_powerbox_entry(&self.module));
        if is_powerbox_func0 {
            if !args.is_empty() {
                return Err(
                    "the powerbox entry takes no caller args (the handles are auto-granted)".into(),
                );
            }
            self.run_diff(&RunConfig::default())
        } else {
            self.run_kernel_diff(fidx, args)
        }
    }

    /// Like [`Instance::call`] for the powerbox entry, but seed the guest's `Stream{In}` (stdin).
    pub fn call_with_stdin(&self, stdin: &[u8]) -> Result<Run, String> {
        self.run_diff(&RunConfig {
            stdin: stdin.to_vec(),
            ..RunConfig::default()
        })
    }

    /// Run the powerbox entry (function 0) on a **single** `backend` under `config`. Grants the
    /// name-bound registry (or the fixed powerbox) and applies the [`Limits`] each backend supports —
    /// the uniform "pick a backend, set the knobs, run" entry. Returns the outcome + captured output,
    /// or `Err` on a guest trap / compile failure.
    pub fn run(&self, backend: Backend, config: &RunConfig) -> Result<Run, String> {
        let owned = self.window_override(config);
        let m = owned.as_ref().unwrap_or(&self.module);
        let win = m.memory.map_or(0, |mc| 1u64 << mc.size_log2);
        let init_mem = config.init_mem()?;

        let mut host = Host::new();
        host.stdin = config.stdin.clone();
        host.set_quota(config.limits.quota());
        let args = self.grant_caps(&mut host, win);

        let outcome = match backend {
            Backend::TreeWalk | Backend::Bytecode => {
                let mut fuel = config.limits.fuel.unwrap_or(DEFAULT_FUEL);
                let r = run_interp(
                    backend,
                    m,
                    0,
                    &args,
                    &mut fuel,
                    init_mem.as_deref(),
                    &mut host,
                );
                outcome_from_interp(r)?
            }
            Backend::Jit => {
                let slots: Vec<i64> = args.iter().copied().map(value_slot).collect();
                let jit = run_jit(m, &slots, &mut host, &config.limits, init_mem.as_deref())?;
                outcome_from_jit(&m.funcs[0].results, jit)?
            }
        };
        Ok(Run {
            outcome,
            stdout: host.stdout,
            stderr: host.stderr,
        })
    }

    /// Run the powerbox entry on the tree-walker **and** the JIT under identical capabilities + `config`
    /// and assert they agree (the §18 interp == jit oracle), returning the shared outcome + output.
    /// `Err` on divergence, a guest trap, or compile failure.
    pub fn run_diff(&self, config: &RunConfig) -> Result<Run, String> {
        let owned = self.window_override(config);
        let m = owned.as_ref().unwrap_or(&self.module);
        let win = m.memory.map_or(0, |mc| 1u64 << mc.size_log2);
        let init_mem = config.init_mem()?;

        // Two hosts, granted identically (grants are deterministic, so the handle vectors match).
        let mut hi = Host::new();
        let mut hj = Host::new();
        hi.stdin = config.stdin.clone();
        hj.stdin = config.stdin.clone();
        hi.set_quota(config.limits.quota());
        hj.set_quota(config.limits.quota());
        let args = self.grant_caps(&mut hi, win);
        let args_j = self.grant_caps(&mut hj, win);
        debug_assert_eq!(args, args_j, "grants must be deterministic across backends");

        let mut fuel = config.limits.fuel.unwrap_or(DEFAULT_FUEL);
        let interp = run_interp(
            Backend::TreeWalk,
            m,
            0,
            &args,
            &mut fuel,
            init_mem.as_deref(),
            &mut hi,
        );

        let slots: Vec<i64> = args.iter().copied().map(value_slot).collect();
        let jit = run_jit(m, &slots, &mut hj, &config.limits, init_mem.as_deref())?;

        let outcome = diff_outcome(&m.funcs[0].results, interp, jit)?;
        if hi.stdout != hj.stdout {
            return Err("interp/JIT stdout diverge".into());
        }
        if hi.stderr != hj.stderr {
            return Err("interp/JIT stderr diverge".into());
        }
        Ok(Run {
            outcome,
            stdout: hi.stdout,
            stderr: hi.stderr,
        })
    }

    /// An owned copy of the module with its window resized, when `config` overrides it; else `None`
    /// (run against `self.module` directly — no clone in the common case).
    fn window_override(&self, config: &RunConfig) -> Option<Module> {
        config.memory_size_log2.map(|size_log2| {
            let mut m = self.module.clone();
            m.memory = Some(svm_ir::Memory { size_log2 });
            m
        })
    }

    /// Grant the powerbox capabilities on `h` for function 0, returning the handle vector in stash
    /// order: the name-bound registry (import order, slot i ↔ import i) when present, else the fixed
    /// §3e powerbox prefix.
    fn grant_caps(&self, h: &mut Host, win: u64) -> Vec<Value> {
        match &self.binding {
            Some(b) => {
                // Inert unless a granted cap needs them (region-backed / Jit caps).
                h.set_region_factory(new_shared_region);
                h.set_jit_validator(jit_blob_validator);
                // Grant in import order, and register each grant under the guest's own import name in
                // the §7 capability-name directory (F7) so the guest can `cap.self`-resolve it at
                // runtime — the same names it wrote as `call.import "<name>"`.
                b.order
                    .iter()
                    .map(|name| {
                        let handle = (b.imports.map[name].grant)(h, win);
                        h.register_cap_name(name, handle);
                        Value::I32(handle)
                    })
                    .collect()
            }
            None => grant_powerbox_prefix(h, self.module.funcs[0].params.len(), win),
        }
    }

    /// Run a bare (non-powerbox) export with `args` on both backends, assert they agree, and return
    /// the outcome (no host capabilities granted — the escape hatch for pure kernel functions).
    fn run_kernel_diff(&self, fidx: FuncIdx, args: &[Value]) -> Result<Run, String> {
        let m = &self.module;
        let mut h = Host::new();
        let mut fuel = 50_000_000u64;
        let interp = run_with_host(m, fidx, args, &mut fuel, &mut h);
        let slots: Vec<i64> = args.iter().copied().map(value_slot).collect();
        let jit =
            compile_and_run(m, fidx, &slots).map_err(|e| format!("JIT compile failed: {e:?}"))?;
        let outcome = diff_outcome(&m.funcs[fidx as usize].results, interp, jit)?;
        Ok(Run {
            outcome,
            stdout: h.stdout,
            stderr: h.stderr,
        })
    }
}

// ----------------------------------------------------------------------------
// Phase 6 — the reactor model: a live, stateful Session you call exports into
// ----------------------------------------------------------------------------

/// The window span the reactor persists between calls. **Must match `svm_interp`/`svm_jit`'s private
/// `SNAP_CAP`** — the tree-walker and bytecode engine snapshot exactly this span, and the JIT is told
/// to (`run_raw`'s `snapshot_cap`), so the three round-trip the same bytes. (The cross-backend
/// `Session` diff would fail loudly if these ever diverged.)
const REACTOR_SNAP_CAP: usize = 1 << 18; // 256 KiB

/// Run export `fidx` on `backend`, seeded from `init_mem`, returning its outcome and the new window
/// snapshot. `Bytecode` transparently falls back to the tree-walker for modules it doesn't support
/// (so that arm matches `TreeWalk` exactly there). The shared per-call primitive for [`Session`].
fn run_capture_on(
    backend: Backend,
    m: &Module,
    fidx: FuncIdx,
    args: &[Value],
    init_mem: &[u8],
    host: &mut Host,
    limits: &Limits,
) -> Result<(Outcome, Vec<u8>), String> {
    let treewalk = |host: &mut Host| {
        let mut fuel = limits.fuel.unwrap_or(DEFAULT_FUEL);
        run_capture_reserved_with_host(
            m,
            fidx,
            args,
            &mut fuel,
            init_mem,
            svm_ir::DEFAULT_RESERVED_LOG2,
            host,
        )
    };
    match backend {
        Backend::TreeWalk => {
            let (r, snap) = treewalk(host);
            Ok((outcome_from_interp(r)?, snap))
        }
        Backend::Bytecode => {
            let mut fuel = limits.fuel.unwrap_or(DEFAULT_FUEL);
            match svm_interp::bytecode::compile_and_run_capture_reserved_with_host(
                m,
                fidx,
                args,
                &mut fuel,
                init_mem,
                svm_ir::DEFAULT_RESERVED_LOG2,
                host,
            ) {
                Some((r, snap)) => Ok((outcome_from_interp(r)?, snap)),
                None => {
                    let (r, snap) = treewalk(host); // unsupported by the bytecode engine — fall back
                    Ok((outcome_from_interp(r)?, snap))
                }
            }
        }
        Backend::Jit => {
            let slots: Vec<i64> = args.iter().copied().map(value_slot).collect();
            // Snapshot the low `REACTOR_SNAP_CAP` window so the next call resumes this state. The
            // reactor is single-threaded (`start` rejects concurrent guests), so `jit_run` takes its
            // unlocked fast path.
            let (jo, snap) = jit_run(
                m,
                fidx,
                &slots,
                host,
                limits,
                Some(init_mem),
                Some(REACTOR_SNAP_CAP),
            )?;
            Ok((outcome_from_jit(&m.funcs[fidx as usize].results, jo)?, snap))
        }
    }
}

/// A **live, stateful instance** on one backend (the reactor model): the powerbox is granted once and
/// the guest window (globals, the handle stash, BSS) **persists** across [`Session::call_export`]
/// calls. Built by [`Instance::start`]. This is the "instantiate once, call exports many times" shape
/// (wasm reactor / component model) that run-once [`Instance::run`] doesn't provide.
///
/// **Slice-1 scope:** single-threaded guests only; persistence covers the low [`REACTOR_SNAP_CAP`]
/// window (globals/stash/BSS) — a `malloc` heap living in the reserved tail above the mapped window is
/// **not** persisted yet. Exports use the convention `(i64 sp, <args…>) -> <results…>`: `call_export`
/// supplies a fresh `sp` (the powerbox data-stack base) and appends the caller's `args`.
pub struct Session {
    module: Module,
    backend: Backend,
    host: Host,
    /// The persisted window image (low `REACTOR_SNAP_CAP` bytes), round-tripped each call.
    snap: Vec<u8>,
    entry_sp: u64,
    limits: Limits,
}

impl Session {
    /// Call exported function `name` with `args`, returning its results. The window (globals, stash,
    /// BSS) and granted capability handles persist from prior calls. `Err` on a missing export, a
    /// trap, or an `Exit`.
    pub fn call_export(&mut self, name: &str, args: &[Value]) -> Result<Vec<Value>, String> {
        let fidx = self
            .module
            .resolve_export(name)
            .ok_or_else(|| format!("no export named `{name}`"))?;
        // Reactor calling convention: export is `(i64 sp, <args…>)` — supply a fresh data-stack base
        // above the persistent globals, then the caller's args.
        let mut call_args = Vec::with_capacity(args.len() + 1);
        call_args.push(Value::I64(self.entry_sp as i64));
        call_args.extend_from_slice(args);
        let (outcome, snap) = run_capture_on(
            self.backend,
            &self.module,
            fidx,
            &call_args,
            &self.snap,
            &mut self.host,
            &self.limits,
        )?;
        self.snap = snap;
        match outcome {
            Outcome::Returned(v) => Ok(v),
            Outcome::Exited(code) => Err(format!("export `{name}` called Exit({code})")),
        }
    }

    /// The backend this session runs on.
    pub fn backend(&self) -> Backend {
        self.backend
    }
    /// Bytes the guest has written to stdout across all calls so far.
    pub fn stdout(&self) -> &[u8] {
        &self.host.stdout
    }
    /// Bytes the guest has written to stderr across all calls so far.
    pub fn stderr(&self) -> &[u8] {
        &self.host.stderr
    }
}

/// A reactor [`Session`] mirrored across **all three backends** (tree-walker, bytecode engine, JIT),
/// stepped in lockstep: every [`DiffSession::call_export`] runs the call on each and asserts they
/// agree on results, captured stdout/stderr, and the persistent window prefix — the §18 oracle
/// extended to a *stateful sequence* of calls (state desyncs are caught the moment they appear). This
/// is the powerbox layer's first direct exercise of the bytecode engine (Followup F10).
pub struct DiffSession {
    sessions: Vec<Session>, // one per backend, in [TreeWalk, Bytecode, Jit] order
    entry_sp: u64,
}

impl DiffSession {
    /// Call `name` with `args` on all three backends in lockstep; return the agreed results, or `Err`
    /// the moment any backend diverges (results, output, or persistent window state).
    pub fn call_export(&mut self, name: &str, args: &[Value]) -> Result<Vec<Value>, String> {
        // The persistent window prefix `[0, entry_sp)` (stash + globals + BSS) must match across
        // backends; the data stack above `entry_sp` is transient and backend-specific (the JIT's frame
        // layout differs from the interpreters'), so it is excluded from the comparison.
        let persist = (self.entry_sp as usize).min(REACTOR_SNAP_CAP);
        let mut agreed: Option<(Vec<Value>, Vec<u8>, Vec<u8>)> = None;
        for s in &mut self.sessions {
            let backend = s.backend;
            let results = s.call_export(name, args)?;
            let prefix = s
                .snap
                .get(..persist.min(s.snap.len()))
                .unwrap_or(&[])
                .to_vec();
            let stdout = s.host.stdout.clone();
            match &agreed {
                None => agreed = Some((results, prefix, stdout)),
                Some((r0, w0, o0)) => {
                    if *r0 != results {
                        return Err(format!(
                            "backend {backend:?} results diverge on `{name}`: {r0:?} vs {results:?}"
                        ));
                    }
                    if *w0 != prefix {
                        return Err(format!(
                            "backend {backend:?} persistent window diverges on `{name}`"
                        ));
                    }
                    if *o0 != stdout {
                        return Err(format!("backend {backend:?} stdout diverges on `{name}`"));
                    }
                }
            }
        }
        Ok(agreed.expect("at least one backend").0)
    }

    /// Captured stdout (identical across backends — asserted on every call).
    pub fn stdout(&self) -> &[u8] {
        self.sessions[0].stdout()
    }
}

impl Instance {
    /// Start a **reactor session** on `backend` under `config`: grant the powerbox once, run the
    /// bootstrap `_start` (function 0) to stash handles + run the initializer, and keep the window +
    /// host live for repeated [`Session::call_export`] calls. Slice 1 is single-threaded — a guest
    /// using §12 threads is rejected (use [`Instance::run`]/[`Instance::run_diff`] for those).
    pub fn start(&self, backend: Backend, config: &RunConfig) -> Result<Session, String> {
        if self.module.funcs.iter().any(|f| f.uses_concurrency()) {
            return Err(
                "reactor Session is single-threaded (slice 1); use run/run_diff for concurrent guests"
                    .into(),
            );
        }
        let module = self
            .window_override(config)
            .unwrap_or_else(|| self.module.clone());
        let win = module.memory.map_or(0, |mc| 1u64 << mc.size_log2);
        let entry_sp = svm_ir::powerbox_entry_sp(&module);

        let mut host = Host::new();
        host.stdin = config.stdin.clone();
        host.set_quota(config.limits.quota());
        let args = self.grant_caps(&mut host, win);

        // Run `_start` (func 0) once: stash the granted handles into the window and run the
        // initializer. Capture the resulting window image as the session's persistent state.
        let (_init, snap) =
            run_capture_on(backend, &module, 0, &args, &[], &mut host, &config.limits)?;
        Ok(Session {
            module,
            backend,
            host,
            snap,
            entry_sp,
            limits: config.limits.clone(),
        })
    }

    /// Start a reactor session mirrored across all three backends (the stateful differential oracle).
    /// Fails if the backends disagree on the post-`_start` window.
    pub fn start_diff(&self, config: &RunConfig) -> Result<DiffSession, String> {
        let entry_sp = svm_ir::powerbox_entry_sp(&self.module);
        let sessions = [Backend::TreeWalk, Backend::Bytecode, Backend::Jit]
            .into_iter()
            .map(|b| self.start(b, config))
            .collect::<Result<Vec<_>, _>>()?;
        // The post-`_start` persistent window must already agree across backends.
        let persist = (entry_sp as usize).min(REACTOR_SNAP_CAP);
        let prefix = |s: &Session| {
            s.snap
                .get(..persist.min(s.snap.len()))
                .unwrap_or(&[])
                .to_vec()
        };
        let base = prefix(&sessions[0]);
        for s in &sessions[1..] {
            if prefix(s) != base {
                return Err(format!(
                    "backend {:?} window diverges from {:?} at start",
                    s.backend, sessions[0].backend
                ));
            }
        }
        Ok(DiffSession { sessions, entry_sp })
    }
}

#[cfg(test)]
mod symtab_tests {
    //! The `compile_linked` symbol table is a **new untrusted-input surface** (guest-controlled
    //! bytes the host decodes). Like the IR decoder it must be fail-closed: never panic / over-read
    //! / hang on arbitrary bytes — only `Some(table)` or `None`. These tests pin the round-trip with
    //! the encoder and sweep adversarial bytes through the decoder.
    use super::*;

    #[test]
    fn encode_decode_round_trips() {
        let entries = [
            ("sq", Resolved::Slot(0)),
            ("a_longer_name", Resolved::Slot(1234)),
            (
                "io",
                Resolved::Cap(svm_ir::ResolvedCap { type_id: 3, op: 7 }),
            ),
            ("", Resolved::Slot(u32::MAX)), // empty name + boundary slot
        ];
        let bytes = encode_symbol_table(&entries);
        let table = decode_symbol_table(&bytes).expect("a well-formed table decodes");
        assert_eq!(table.len(), entries.len());
        for (name, want) in entries {
            assert_eq!(table.get(name), Some(&want), "entry {name:?} round-trips");
        }
    }

    #[test]
    fn empty_buffer_and_explicit_zero_count_are_both_the_empty_table() {
        // `&[]` (the closed `compile` op) and `[0]` (encode of no entries) both mean "no symbols".
        assert_eq!(decode_symbol_table(&[]).map(|t| t.len()), Some(0));
        assert_eq!(decode_symbol_table(&[0]).map(|t| t.len()), Some(0));
        assert_eq!(
            decode_symbol_table(&encode_symbol_table(&[])).map(|t| t.len()),
            Some(0)
        );
    }

    #[test]
    fn malformations_fail_closed_without_panicking() {
        // Each of these is structurally broken in a different way; all must be `None`, never a panic.
        let cases: &[&[u8]] = &[
            &[1],                      // count 1, but no entry bytes
            &[1, 1],                   // count 1, namelen 1, but no name byte
            &[1, 1, b'F'],             // name present, but no kind byte
            &[1, 1, b'F', 9],          // unknown kind 9
            &[1, 1, b'F', 0],          // Slot kind, but no slot value
            &[1, 1, 0xff, 0, 0],       // a non-UTF-8 name byte
            &[1, 0x80],                // a truncated LEB128 namelen
            &[0xff, 0xff, 0xff, 0xff], // a huge count (must fail fast as bytes exhaust, not hang)
            &[0, 0],                   // count 0 but a trailing byte (length mismatch)
            &[1, 1, b'F', 0, 0, 0],    // a valid entry plus trailing bytes
        ];
        for &c in cases {
            assert_eq!(
                decode_symbol_table(c),
                None,
                "malformed {c:?} must fail closed"
            );
        }
    }

    #[test]
    fn never_panics_on_arbitrary_bytes() {
        // A deterministic adversarial sweep: every byte string up to length 4, plus a pseudo-random
        // tail of longer inputs. The decoder must always *return* (Some or None), never panic/hang.
        for len in 0..=3usize {
            let mut buf = vec![0u8; len];
            loop {
                let _ = decode_symbol_table(&buf); // must not panic
                                                   // Odometer over [0,256)^len; stop after the most-significant digit wraps.
                let mut i = 0;
                while i < len {
                    if buf[i] == 255 {
                        buf[i] = 0;
                        i += 1;
                    } else {
                        buf[i] += 1;
                        break;
                    }
                }
                if i == len {
                    break; // wrapped around (or len == 0): done.
                }
            }
        }
        // Longer pseudo-random inputs (xorshift) for breadth past the exhaustive region.
        let mut state = 0x9e3779b97f4a7c15u64;
        for _ in 0..100_000 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let n = (state % 96) as usize;
            let mut buf = Vec::with_capacity(n);
            let mut s = state;
            for _ in 0..n {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                buf.push((s >> 33) as u8);
            }
            let _ = decode_symbol_table(&buf); // must not panic
        }
    }
}
