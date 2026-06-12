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

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use svm_interp::{
    iface, AsyncCounter, CapPageMap, GuestMem, Host, RegionBacking, StreamRole, Trap,
};
// `SharedBacking` is implemented by the per-OS shared-mapping backing (unix `ShmBacking`, windows
// `WinShmBacking`) the JIT aliases into the window for §13.
#[cfg(any(unix, windows))]
use svm_interp::SharedBacking;
use svm_ir::{Module, ValType};

// Re-export the value type + the §15 spawn quota so embedders (and the CLI) need not also depend on
// `svm-interp`.
pub use svm_interp::{Quota, Value};
use svm_jit::{compile_and_run, CompiledModule, JitOutcome, TrapKind, EXIT_CODE};

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
    // Guest-driven `Jit` (iface 11, JIT.md Model A): serviced natively here, not in the generic
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

/// The native (Cranelift) half of the guest-driven `Jit` capability (JIT.md Model A), reached
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
                Ok(ptrs) => {
                    host.set_jit_unit_native(compiled.domain, compiled.unit, ptrs[0] as usize);
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
        _ => put(results, n_results, EINVAL, trap_out),
    }
}

/// The canonical [`svm_interp::JitValidator`] — the **security hinge** of the guest-driven
/// `Jit` capability (JIT.md "Security argument"): `decode_module` (untrusted-input-facing,
/// fail-closed) → `verify_module` (the escape-freedom gate) → the **memory-match
/// precondition** (declared memory must equal the parent window, so verified bounds and the
/// runtime mask agree) → reject data segments (they would overwrite live guest memory) and
/// §12 concurrency ops (the single-threaded MVP restriction). Install the *same* function on
/// the interpreter and JIT `Host`s of a differential pair ([`grant_jit`] does), so both
/// backends accept/reject identically. All failures are `-EINVAL` (guest-visible, non-fatal,
/// nothing installed).
pub fn jit_blob_validator(bytes: &[u8], mem_log2: Option<u8>) -> Result<Arc<[svm_ir::Func]>, i64> {
    const EINVAL: i64 = -22;
    let Ok(m) = svm_encode::decode_module(bytes) else {
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
    // Reject `call_indirect` in a submitted unit (the new→old path). On the JIT it dispatches
    // through the parent table, but the reference interpreter cannot model a frame spanning the
    // parent and unit function spaces, so the backends would diverge. Model A MVP supports
    // old→new (`invoke`); uniform cross-unit dispatch is the B2 (table-install) feature. Gating
    // it in the shared validator keeps both backends provably in lockstep (see
    // [`svm_ir::Func::uses_indirect_call`]).
    if m.funcs.iter().any(|f| f.uses_indirect_call()) {
        return Err(EINVAL);
    }
    Ok(m.funcs.into())
}

/// Grant the guest-driven `Jit` capability (opt-in, like `Memory`): install the canonical
/// [`jit_blob_validator`] and mint the domain handle bound to `m`'s declared memory (the
/// memory-match precondition). Works for both backends — the interpreter services the iface
/// in its eval loop/dispatch; the JIT needs the module registered too (see [`jit_cap_run`]).
pub fn grant_jit(host: &mut Host, m: &Module) -> i32 {
    host.set_jit_validator(jit_blob_validator);
    host.grant_jit(m.memory.map(|mc| mc.size_log2))
}

/// Run `m` on the **JIT** with the `Jit` capability live: the long-lived compile→run split
/// ([`CompiledModule`]), with the module pointer registered in `host` so [`cap_thunk`]'s
/// native `Jit` ops can re-enter it mid-run (`define_extra` / `invoke_extra` while the guest
/// is suspended in its synchronous `cap.call`). The interpreter counterpart is the plain
/// `run_capture_reserved_with_host` over the same `Host` setup ([`grant_jit`]) — drive both
/// with identical inputs for the differential.
pub fn jit_cap_run(
    m: &Module,
    entry: u32,
    args: &[i64],
    init_mem: &[u8],
    reserved_log2: u8,
    host: &mut Host,
) -> Result<(JitOutcome, Vec<u8>), svm_jit::JitError> {
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
/// # Safety
/// `ctx` is a live `*mut Host`; `trap_out` is writable — the [`svm_jit::FastCapResolver`] contract.
unsafe extern "C" fn fast_clock_now(
    ctx: *mut c_void,
    _mem_base: *mut u8,
    _mem_size: u64,
    handle: i32,
    trap_out: *mut i64,
) -> i64 {
    fast_dispatch(ctx, svm_interp::iface::CLOCK, 0, handle, &[], trap_out)
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
    // [addrspace], [ioring], [blocking], [jit] — §3e/§9/§12/JIT.md; a chibicc `_start` always
    // imports the full 8). The runner grants exactly as many as the entry declares (see
    // `run_powerbox_with_deadline`).
    matches!(
        module.funcs.first().map(|f| f.params.as_slice()),
        Some(p) if (3..=8).contains(&p.len()) && p.iter().all(|t| matches!(t, ValType::I32))
    )
}

fn typed(t: ValType, v: i64) -> Value {
    match t {
        ValType::I32 => Value::I32(v as i32),
        ValType::I64 => Value::I64(v),
        ValType::F32 => Value::F32(f32::from_bits(v as u32)),
        ValType::F64 => Value::F64(f64::from_bits(v as u64)),
        // CLI entry args are scalar `i64` slots; a `v128` entry param is out of scope. Total arm:
        // zero-extend the slot into the low lanes.
        ValType::V128 => {
            let mut bytes = [0u8; 16];
            bytes[..8].copy_from_slice(&v.to_le_bytes());
            Value::V128(bytes)
        }
    }
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
    let mut host = Host::new();
    host.set_quota(quota);
    host.stdin = stdin.to_vec();
    // Guest-minted §13/§14 regions (`__vm_region_create` → `AddressSpace.create_region`) need an
    // OS-shared-memory backing so the JIT can `map` them for real aliasing; install the factory
    // unconditionally (inert if the guest never mints).
    host.set_region_factory(new_shared_region);
    // Grant in the powerbox's declared import order: stdout, stdin, exit, then Memory if the
    // entry takes a 4th handle (§3e / D44) — so a `map`-growing guest heap has a handle to call —
    // then an AddressSpace over the whole window if it takes a 5th (§14: the memory-management
    // authority `create_region` mints from; attenuable; the carve source for nesting).
    let arity = module.funcs.first().map_or(0, |f| f.params.len());
    let mut slots = vec![
        host.grant_stream(StreamRole::Out) as i64,
        host.grant_stream(StreamRole::In) as i64,
        host.grant_exit() as i64,
    ];
    if arity >= 4 {
        slots.push(host.grant_memory() as i64);
    }
    if arity >= 5 {
        let win = module.memory.map_or(0, |mc| 1u64 << mc.size_log2);
        slots.push(host.grant_address_space(0, win) as i64);
    }
    // §9/§12 the async I/O ring: the IoRing + Blocking handles a chibicc `_start` always imports (the
    // 6th/7th of the fixed 7-handle powerbox). The mock Blocking op is non-blocking here (`ZERO`).
    if arity >= 6 {
        slots.push(host.grant_io_ring() as i64);
    }
    if arity >= 7 {
        slots.push(host.grant_blocking(std::time::Duration::ZERO, None) as i64);
    }
    // The guest-driven `Jit` capability (iface 11, JIT.md) — the 8th of the fixed chibicc
    // powerbox. The canonical validator is the security hinge; the live `CompiledModule` is
    // registered below, once it exists.
    if arity >= 8 {
        slots.push(grant_jit(&mut host, module) as i64);
    }
    // §15: the powerbox's spawn quota (default = the anti-bomb ceilings) — the JIT enforces the same
    // fiber/vCPU caps as the interpreter would. (An embedder sets it on the `Host`; a `run_powerbox`
    // quota arg is a follow-up.)
    let hq = host.quota();
    let quota = svm_jit::Quota {
        max_fibers: hq.max_fibers,
        max_vcpus: hq.max_vcpus,
    };
    let ctx = &mut host as *mut Host as *mut c_void;
    // The long-lived compile→run split (JIT.md Phase 1): compile once, register the live module
    // for the `Jit` capability's mid-run re-entry (define_extra / invoke_extra from `cap_thunk`),
    // then run through the same caller-managed pointer (`run_raw`'s provenance contract).
    // Behavior-identical to the historical one-shot entry points for a guest that never uses the
    // `Jit` cap — they are thin wrappers over this same machinery.
    //
    // §5 fuel/epoch kill-path: when a `deadline` is given, arm the JIT's interrupt poll with a
    // watchdog so a runaway guest is stopped after the deadline instead of hanging the process.
    // `None` ⇒ the ordinary unbounded run. The watchdog wakes early when the run finishes, so a
    // fast program is never delayed.
    let jit = if let Some(d) = deadline.filter(|d| !d.is_zero()) {
        let interrupt = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        let (done_tx, done_rx) = std::sync::mpsc::channel::<()>();
        let wd = interrupt.clone();
        let watchdog = std::thread::spawn(move || {
            // Timed out (or the run dropped the sender) ⇒ request the kill.
            if done_rx.recv_timeout(d).is_err() {
                wd.store(1, std::sync::atomic::Ordering::SeqCst);
            }
        });
        // SAFETY: `interrupt` (an `Arc<AtomicU64>`) outlives the run — it is dropped only after the
        // watchdog is joined below; `cap_thunk`/`ctx`/`fast_cap_resolver` honour their contracts.
        let r = CompiledModule::compile(
            module,
            0,
            cap_thunk,
            ctx,
            svm_ir::DEFAULT_RESERVED_LOG2,
            None,
            None,
            Some(std::sync::Arc::as_ptr(&interrupt)),
            Some(fast_cap_resolver),
            quota,
        )
        .and_then(|mut cm| {
            let cm_ptr: *mut CompiledModule = &mut cm;
            host.set_jit_native_ctx(cm_ptr as usize);
            // SAFETY: `cm_ptr` is the single pointer for this run (registered above for the
            // thunk's `Jit` handlers); the run is on this thread; `init_mem`/snapshot unused.
            let r = unsafe { CompiledModule::run_raw(cm_ptr, &slots, None, None, None) };
            host.set_jit_native_ctx(0);
            r.map(|(out, _)| out)
        });
        let _ = done_tx.send(()); // run finished — wake the watchdog so it exits promptly
        let _ = watchdog.join();
        r
    } else {
        // The §9/D45 fast path for the hot window-independent ops (Clock/Blocking); everything else
        // falls back to `cap_thunk` inside the resolver, so the run is otherwise identical.
        CompiledModule::compile(
            module,
            0,
            cap_thunk,
            ctx,
            svm_ir::DEFAULT_RESERVED_LOG2,
            None,
            None,
            None,
            Some(fast_cap_resolver),
            quota,
        )
        .and_then(|mut cm| {
            let cm_ptr: *mut CompiledModule = &mut cm;
            host.set_jit_native_ctx(cm_ptr as usize);
            // SAFETY: as above — the single caller-managed pointer for this run.
            let r = unsafe { CompiledModule::run_raw(cm_ptr, &slots, None, None, None) };
            host.set_jit_native_ctx(0);
            r.map(|(out, _)| out)
        })
    }
    .map_err(|e| format!("JIT compile failed: {e:?}"))?;

    let outcome = match jit {
        JitOutcome::Returned(s) => {
            let results = &module.funcs[0].results;
            Outcome::Returned(s.iter().zip(results).map(|(&v, t)| typed(*t, v)).collect())
        }
        JitOutcome::Exited(code) => Outcome::Exited(code),
        JitOutcome::Trapped(kind) => {
            return Err(format!("guest trapped ({kind:?}) — detect-and-kill (§5)"))
        }
    };
    Ok(Run {
        outcome,
        stdout: host.stdout,
        stderr: host.stderr,
    })
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
