//! Per-OS-thread **durable shadow-region base** register (`durable.shadow_base`, DURABILITY.md §12.8
//! Phase 4 Slice A.5) for the JIT.
//!
//! One `u64` window byte offset per OS thread (a vCPU): the base of the shadow region the running
//! durable context spills into. The durable transform reads it (via the `durable.shadow_base` IR op)
//! to address *this* context's per-context shadow-SP word, so concurrent vCPUs each unwind into their
//! own region with **no shared SP word** — retiring the single `SHADOW_SP_OFF` word (and its
//! `workers > 1` lock).
//!
//! Like [`crate::vcpu_tls`] it is a baked thunk over a thread-local — substrate-independent and unable
//! to fault — but **runtime-private**: the runtime seeds it (per dispatch / per child) and there is no
//! guest write thunk, so a guest cannot redirect its own shadow stack (unlike the guest-overwritable
//! `vcpu.tls`). Seeded at vCPU entry to `shadow_region_base(ctx)` (root = `SHADOW_BASE`).

use std::cell::Cell;

/// Default: the root context's region base (`shadow_region_base(0)` = `SHADOW_BASE`). The runtime
/// re-seeds at every root entry / inline child / fiber switch before any instrumented code runs, so
/// this default is only a never-stale fallback.
const ROOT_SHADOW_BASE: u64 = 64;

thread_local! {
    /// This OS thread's (vCPU's) active durable shadow-SP **word address** — the base of the region
    /// the running context spills into (§12.8 4A.5). [`seed`] resets it per root entry / inline child /
    /// fiber switch so a reused worker thread can't leak a prior run's value.
    static DURABLE_SHADOW_BASE: Cell<u64> = const { Cell::new(ROOT_SHADOW_BASE) };
}

/// Seed/reset the current OS thread's durable shadow-SP word address. Called when the runtime makes a
/// context active: the root entry, each inline child, and both edges of a fiber resume swap.
pub(crate) fn seed(base: u64) {
    DURABLE_SHADOW_BASE.with(|c| c.set(base));
}

/// `durable.shadow_base` thunk — the current context's shadow-region base. A pure thread-local read;
/// it cannot fault, so it takes no window/trap context (unlike the `cap.call`/`gc.roots` thunks).
pub(crate) extern "C" fn get() -> u64 {
    DURABLE_SHADOW_BASE.with(|c| c.get())
}
