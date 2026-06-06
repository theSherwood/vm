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

use svm_interp::{GuestMem, Host, StreamRole, Trap};
use svm_ir::{Module, ValType};

// Re-export the value type so embedders (and the CLI) need not also depend on `svm-interp`.
pub use svm_interp::Value;
use svm_jit::{compile_and_run, compile_and_run_with_host, JitOutcome, TrapKind, EXIT_CODE};

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
    // The guest window with a real `mprotect`-backed Memory capability (`map`/`unmap`/`protect`,
    // incl. growth into the reserved tail). Unix-only — like the JIT itself.
    #[cfg(unix)]
    let mut wm = MprotectWindow::new(mem_base, mem_size, mem_reserved);
    #[cfg(unix)]
    let gm: Option<&mut dyn GuestMem> = if mem_base.is_null() {
        None
    } else {
        Some(&mut wm)
    };
    #[cfg(not(unix))]
    let gm: Option<&mut dyn GuestMem> = {
        let _ = (mem_base, mem_size, mem_reserved);
        None
    };
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

/// The pinned guest page size (§4 "pin page size"); the model granularity for `map`/`unmap`/
/// `protect`, matching the interpreter's reference `PAGE` so the two backends agree.
#[cfg(unix)]
const PAGE: u64 = 4096;

/// A [`GuestMem`] over the JIT's guest window whose `map`/`unmap`/`protect` (the Memory capability,
/// §3e) are backed by **real `mprotect`** on the window pages, mirrored by a software page-state
/// map. The mirror lets cap-buffer borrows (§7) **fail closed** (`-EFAULT`) on an unmapped/RO page
/// instead of faulting the host outside the guarded call, and bounds growth to the reserved mask
/// domain — keeping this backend bit-identical to the interpreter's paged `Mem` (the §18 oracle,
/// enforced by `jit_diff`'s differential). Unix-only, like the JIT's guard page itself.
///
/// # Safety
/// `base` must point at the JIT guest window: `[base, base+mapped)` initially RW and the whole
/// `[base, base+reserved)` a live `PROT_NONE`/RW reservation owned for the call's duration.
#[cfg(unix)]
pub struct MprotectWindow {
    base: *mut u8,
    mapped: u64,
    reserved: u64,
    /// Page index ⇒ explicit state; absent ⇒ region default (rw in `[0, mapped)`, unmapped in the
    /// reserved tail). Mirrors `svm_interp`'s page map so the two backends agree page-for-page.
    prot: std::collections::BTreeMap<u64, PageState>,
}

#[cfg(unix)]
#[derive(Clone, Copy, PartialEq, Eq)]
enum PageState {
    Rw,
    Ro,
    Unmapped,
}

#[cfg(unix)]
impl MprotectWindow {
    /// Wrap the JIT window `[base, base+mapped)` (backed) inside a `reserved` mask domain.
    pub fn new(base: *mut u8, mapped: u64, reserved: u64) -> MprotectWindow {
        MprotectWindow {
            base,
            mapped,
            reserved: reserved.max(mapped),
            prot: std::collections::BTreeMap::new(),
        }
    }

    /// One page's access state: `None` ⇒ faults (unmapped), `Some(writable)` ⇒ committed — the
    /// same default rule as the interpreter (`svm_interp::Mem::page_access`).
    fn page_access(&self, page: u64) -> Option<bool> {
        match self.prot.get(&page) {
            Some(PageState::Rw) => Some(true),
            Some(PageState::Ro) => Some(false),
            Some(PageState::Unmapped) => None,
            None => (page * PAGE < self.mapped).then_some(true),
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
        (ptr / PAGE..=(end - 1) / PAGE)
            .all(|p| matches!(self.page_access(p), Some(w) if w || !write))
    }

    /// Validate a `map`/`unmap`/`protect` range and return its inclusive page-index span, or
    /// `-EINVAL` (page-aligned offset, non-zero len, within `[0, reserved)`) — matching the
    /// interpreter's `prot_pages` (growth into the reserved tail is allowed).
    fn prot_pages(&self, offset: u64, len: u64) -> Result<std::ops::RangeInclusive<u64>, i64> {
        const EINVAL: i64 = -22;
        if len == 0 || !offset.is_multiple_of(PAGE) {
            return Err(EINVAL);
        }
        let end = offset.checked_add(len).ok_or(EINVAL)?;
        if end > self.reserved {
            return Err(EINVAL);
        }
        Ok((offset / PAGE)..=((end - 1) / PAGE))
    }

    /// Update one page's software state from cap `prot` bits, mirroring `svm_interp::set_prot`:
    /// a read-write page is left absent in the prefix, explicit `Rw` in the reserved tail.
    fn set_prot(&mut self, page: u64, prot: i32) {
        const PROT_READ: i32 = 1;
        const PROT_WRITE: i32 = 2;
        if prot & PROT_WRITE != 0 {
            if page * PAGE < self.mapped {
                self.prot.remove(&page);
            } else {
                self.prot.insert(page, PageState::Rw);
            }
        } else if prot & PROT_READ != 0 {
            self.prot.insert(page, PageState::Ro);
        } else {
            self.prot.insert(page, PageState::Unmapped);
        }
    }

    /// `mprotect [offset, offset+len)` (page-rounded) to cap `prot` bits. The caller has already
    /// validated the range, so this only translates + applies.
    fn hw_protect(&self, offset: u64, len: u64, prot: i32) {
        const PROT_READ: i32 = 1;
        const PROT_WRITE: i32 = 2;
        let hw = if prot & PROT_WRITE != 0 {
            libc::PROT_READ | libc::PROT_WRITE
        } else if prot & PROT_READ != 0 {
            libc::PROT_READ
        } else {
            libc::PROT_NONE
        };
        let start = (offset / PAGE) * PAGE;
        let end = offset + len;
        let rlen = (end.div_ceil(PAGE) * PAGE) - start;
        // SAFETY: `[base+start, +rlen)` is within the window's reserved mapping (validated:
        // end ≤ reserved), owned by the JIT for the call's duration.
        unsafe {
            libc::mprotect(
                self.base.add(start as usize) as *mut c_void,
                rlen as usize,
                hw,
            );
        }
    }
}

#[cfg(unix)]
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
    /// §3e op 0 `map`: (re)commit `[offset,offset+len)` with `prot`, zero-filled — including
    /// **growth** into the reserved tail. Mirrors `svm_interp::Mem::map`.
    fn map(&mut self, offset: u64, len: u64, prot: i32) -> i64 {
        let pages = match self.prot_pages(offset, len) {
            Ok(p) => p,
            Err(e) => return e,
        };
        // Make the pages RW so the zero-fill lands, then apply the requested protection.
        self.hw_protect(offset, len, 2 /* WRITE */);
        // SAFETY: the pages are now RW and within the reserved mapping (validated).
        unsafe { std::ptr::write_bytes(self.base.add(offset as usize), 0, len as usize) };
        for page in pages {
            self.set_prot(page, prot);
        }
        self.hw_protect(offset, len, prot);
        0
    }
    /// §3e op 1 `unmap`: decommit — any access faults, the physical pages are released
    /// (`MADV_DONTNEED`), so a re-`map` reads zero. Mirrors `svm_interp::Mem::unmap`.
    fn unmap(&mut self, offset: u64, len: u64) -> i64 {
        let pages = match self.prot_pages(offset, len) {
            Ok(p) => p,
            Err(e) => return e,
        };
        // SAFETY: validated in-range; drop the physical backing then protect PROT_NONE.
        unsafe {
            libc::madvise(
                self.base.add(offset as usize) as *mut c_void,
                len as usize,
                libc::MADV_DONTNEED,
            );
        }
        self.hw_protect(offset, len, 0 /* PROT_NONE */);
        for page in pages {
            self.prot.insert(page, PageState::Unmapped);
        }
        0
    }
    /// §3e op 2 `protect`: change protection without touching backing (the D40 RO mechanism).
    fn protect(&mut self, offset: u64, len: u64, prot: i32) -> i64 {
        let pages = match self.prot_pages(offset, len) {
            Ok(p) => p,
            Err(e) => return e,
        };
        for page in pages {
            self.set_prot(page, prot);
        }
        self.hw_protect(offset, len, prot);
        0
    }
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
    matches!(
        module.funcs.first().map(|f| f.params.as_slice()),
        Some([ValType::I32, ValType::I32, ValType::I32])
            | Some([ValType::I32, ValType::I32, ValType::I32, ValType::I32])
    )
}

fn typed(t: ValType, v: i64) -> Value {
    match t {
        ValType::I32 => Value::I32(v as i32),
        ValType::I64 => Value::I64(v),
        ValType::F32 => Value::F32(f32::from_bits(v as u32)),
        ValType::F64 => Value::F64(f64::from_bits(v as u64)),
    }
}

/// Run `module`'s entry (function 0) on the JIT under the MVP powerbox (§3e): a writable
/// `stdout`, a readable `stdin` seeded from `stdin`, and `Exit` — the three handles the
/// frontend's `_start` expects, granted in declared order. Returns the outcome and captured
/// output. `Err` if the (already-verified) module fails to JIT-compile, or if the guest
/// **traps** (detect-and-kill, §5) — the guest can never corrupt the host.
pub fn run_powerbox(module: &Module, stdin: &[u8]) -> Result<Run, String> {
    let mut host = Host::new();
    host.stdin = stdin.to_vec();
    // Grant in the powerbox's declared import order: stdout, stdin, exit, then Memory if the
    // entry takes a 4th handle (§3e / D44) — so a `map`-growing guest heap has a handle to call.
    let wants_memory = matches!(
        module.funcs.first().map(|f| f.params.len()),
        Some(n) if n >= 4
    );
    let mut slots = vec![
        host.grant_stream(StreamRole::Out) as i64,
        host.grant_stream(StreamRole::In) as i64,
        host.grant_exit() as i64,
    ];
    if wants_memory {
        slots.push(host.grant_memory() as i64);
    }
    let jit = compile_and_run_with_host(
        module,
        0,
        &slots,
        cap_thunk,
        &mut host as *mut Host as *mut c_void,
    )
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
