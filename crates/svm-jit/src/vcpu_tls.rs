//! Per-vCPU **thread-local register** (`vcpu.tls.get`/`vcpu.tls.set`, §12) for the JIT.
//!
//! One `i64` per OS thread. A vCPU is a 1:1 OS thread (`thread.spawn`), so a per-OS-thread word is
//! per-vCPU — and a fiber that migrated here (D57: any vCPU may resume any resumable fiber) reads
//! *this* thread's word, i.e. the current vCPU's value, exactly like the interpreter reads the running
//! `VCpu`'s `tls`. Deliberately **independent of the fiber substrate** (unlike `CURRENT_RT`), so a
//! plain non-fiber root run also has a TLS word. Seeded at vCPU entry to a dense id (root = 0, a
//! spawned vCPU to its `idx + 1`), guest-overwritable via the `set` thunk.

use std::cell::Cell;

thread_local! {
    /// This OS thread's (vCPU's) TLS word. Defaults to 0 (the root's seed); [`seed`] resets it at the
    /// start of every root/child run so a reused worker thread can't leak a prior run's `set`.
    static VCPU_TLS: Cell<i64> = const { Cell::new(0) };
}

/// Seed/reset the current OS thread's TLS word. Called at the start of a root run (→ 0) and at a
/// spawned vCPU's entry (→ its dense id), so the value is deterministic and never stale.
pub(crate) fn seed(v: i64) {
    VCPU_TLS.with(|c| c.set(v));
}

/// `vcpu.tls.get` thunk — the current vCPU's TLS word. A pure thread-local read; it cannot fault, so
/// it takes no window/trap context (unlike the `cap.call`/`gc.roots` thunks).
pub(crate) extern "C" fn get() -> i64 {
    VCPU_TLS.with(|c| c.get())
}

/// `vcpu.tls.set` thunk — set the current vCPU's TLS word.
pub(crate) extern "C" fn set(v: i64) {
    VCPU_TLS.with(|c| c.set(v));
}
