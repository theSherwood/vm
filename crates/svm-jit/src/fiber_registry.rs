//! Single-owner ownership protocol for **migratable fibers** (D57 / `SCHEDULING.md` "Design sketch"
//! #1) — the load-bearing, loom-verifiable core of stackful work-stealing.
//!
//! The dangerous invariant of stackful work-stealing is **"one native stack, exactly one thread":**
//! a fiber's saved stack + register context must never be resumed by two OS threads at once — that
//! would execute one call stack on two cores and corrupt it irrecoverably (the precise unsafe D56
//! removed when it deleted the VM-owned M:N executor). Today fibers are *thread-affine* (each vCPU
//! owns a private fiber table, see `fiber_rt`/the interp), so the invariant holds trivially. Making
//! fibers **migratable** trades that for a shared registry where any worker may *steal* a
//! voluntarily-suspended fiber — and the whole safety of the feature reduces to one atomic property:
//!
//! > a fiber in the steal pool is claimed (transitioned to `Running`) by **exactly one** thread.
//!
//! This module is that property, isolated. It is **pure atomics — it touches no real stack** — so it
//! is fully `loom`-model-checkable, exactly as the `wait`/`notify` futex core is. Per SCHEDULING.md's
//! "earn the risk with verification, not assume it" mandate and the demo roadmap (#3), the protocol
//! is proven here first; the runtime integration (a shared registry replacing the per-thread tables)
//! and the cross-thread asm resume (the `svm-fiber` switch, which barely changes — design sketch #3)
//! are the *next*, review-gated slices that build on this proof. Nothing in this module is wired into
//! the live runtime yet, hence `#[allow(dead_code)]`.
//!
//! ## States (`AtomicU8`)
//! - `OWNED` — owned by one worker, **not** running and **not** in the steal pool: a just-created
//!   fiber (owned by its creator), or one *pinned* to its thread (a §5/§14 fault-suspended fiber
//!   carrying thread-affine recovery state — `sigjmp_buf`/VEH `CONTEXT` — which design sketch #2
//!   excludes from stealing). Only its owner may run it.
//! - `RUNNABLE` — **voluntarily** suspended, published into the shared steal pool, owned by no
//!   thread. The only stealable state.
//! - `RUNNING` — a worker is executing the native stack. **Never** stealable.
//! - `FREE` — the fiber returned; the slot is reclaimable.
//!
//! ## Memory ordering
//! A suspend that publishes a fiber to the pool (`suspend_to_pool`, `pin`) is a **release** store; a
//! claim that takes ownership (`try_steal`, `begin_owned`) is an **acquire** CAS. So the winning
//! claimant *synchronizes-with* the suspending thread and observes the complete saved context — the
//! same publish/consume discipline as the futex word and the JIT's atomic `FnEntry` (DESIGN §22).

#[cfg(loom)]
use loom::sync::atomic::{AtomicU8, Ordering};
#[cfg(not(loom))]
use std::sync::atomic::{AtomicU8, Ordering};

/// Owned by one worker; not running, not stealable (just-created, or pinned/fault-suspended).
const OWNED: u8 = 1;
/// Voluntarily suspended in the shared steal pool; ownerless — the only stealable state.
const RUNNABLE: u8 = 2;
/// A worker is executing this fiber's native stack — never stealable.
const RUNNING: u8 = 3;
/// The fiber returned; the slot is reclaimable.
const FREE: u8 = 4;

/// The atomic ownership state of one fiber slot. One per fiber in the (future) shared registry; the
/// only synchronization a steal needs (the pool membership *is* `state == RUNNABLE`, so the
/// load-bearing race is this single CAS, not the pool container — a Chase-Lev deque or a mutex'd
/// vec is an orthogonal, non-unsafe choice layered on top).
#[allow(dead_code)] // staged: verified in isolation here; runtime integration is the next slice
pub(crate) struct Ownership {
    state: AtomicU8,
}

#[allow(dead_code)] // staged: see the module header
impl Ownership {
    /// A freshly created fiber, **owned by its creator** and not yet in the steal pool.
    pub(crate) fn new_owned() -> Ownership {
        Ownership {
            state: AtomicU8::new(OWNED),
        }
    }

    /// The owner begins running an `OWNED` fiber: `OWNED → RUNNING`. Returns `false` if the fiber was
    /// not in `OWNED` (a caller bug — only the owner runs an owned fiber, and it is not concurrently
    /// stealable, so this CAS cannot legitimately lose). **Acquire** on success: the owner observes
    /// whatever it published when it last `pin`ned the fiber.
    pub(crate) fn begin_owned(&self) -> bool {
        self.state
            .compare_exchange(OWNED, RUNNING, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
    }

    /// Voluntarily suspend the running fiber **into the steal pool**: `RUNNING → RUNNABLE`, a
    /// **release** store so a stealer's acquiring claim sees the complete saved context. After this
    /// the fiber is ownerless and any worker may [`try_steal`](Self::try_steal) it.
    pub(crate) fn suspend_to_pool(&self) {
        debug_assert_eq!(self.state.load(Ordering::Relaxed), RUNNING);
        self.state.store(RUNNABLE, Ordering::Release);
    }

    /// Attempt to steal a pooled fiber: CAS `RUNNABLE → RUNNING`. Returns `true` iff **this** thread
    /// won exclusive ownership — at most one caller ever does, which is the protocol's whole point.
    /// A loser (the slot was already stolen, is running, or was pinned) returns `false` and backs
    /// off to try another slot. **Acquire** on success: the winner synchronizes-with the suspending
    /// thread and observes the published context.
    pub(crate) fn try_steal(&self) -> bool {
        self.state
            .compare_exchange(RUNNABLE, RUNNING, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
    }

    /// **Pin** the running fiber to its current owner (not stealable): `RUNNING → OWNED`. Used for a
    /// fault-suspended fiber whose recovery state (`sigjmp_buf`/VEH `CONTEXT`) is thread-affine, so it
    /// must resume on the same thread (design sketch #2). A **release** store, symmetric with
    /// `suspend_to_pool`, so the eventual `begin_owned` sees the saved context.
    pub(crate) fn pin(&self) {
        debug_assert_eq!(self.state.load(Ordering::Relaxed), RUNNING);
        self.state.store(OWNED, Ordering::Release);
    }

    /// The running fiber returned: `RUNNING → FREE`. The slot is now reclaimable.
    pub(crate) fn finish(&self) {
        debug_assert_eq!(self.state.load(Ordering::Relaxed), RUNNING);
        self.state.store(FREE, Ordering::Release);
    }

    /// The current state (relaxed) — for observability / assertions.
    pub(crate) fn state(&self) -> u8 {
        self.state.load(Ordering::Relaxed)
    }
}

// Real-build (non-loom) unit tests: the single-threaded transition table. The *concurrent*
// single-owner property is the loom model below.
#[cfg(all(test, not(loom)))]
mod tests {
    use super::*;

    #[test]
    fn owned_run_suspend_steal_cycle() {
        let o = Ownership::new_owned();
        assert_eq!(o.state(), OWNED);
        assert!(o.begin_owned(), "the owner runs its owned fiber");
        assert_eq!(o.state(), RUNNING);
        // A running fiber is never stealable.
        assert!(!o.try_steal(), "a running fiber must not be stealable");
        // Voluntary suspend publishes it to the pool; now it is stealable exactly once.
        o.suspend_to_pool();
        assert_eq!(o.state(), RUNNABLE);
        assert!(o.try_steal(), "a pooled fiber is stealable");
        assert_eq!(o.state(), RUNNING);
        assert!(!o.try_steal(), "it cannot be stolen twice");
    }

    #[test]
    fn pinned_fiber_is_not_in_the_pool() {
        let o = Ownership::new_owned();
        assert!(o.begin_owned());
        o.pin(); // fault-suspended → stays with its owner, excluded from the pool
        assert_eq!(o.state(), OWNED);
        assert!(!o.try_steal(), "a pinned fiber must not be stealable");
        // Its owner resumes it.
        assert!(o.begin_owned());
        assert_eq!(o.state(), RUNNING);
    }

    #[test]
    fn finish_frees_the_slot() {
        let o = Ownership::new_owned();
        assert!(o.begin_owned());
        o.finish();
        assert_eq!(o.state(), FREE);
        assert!(!o.try_steal(), "a freed slot is not stealable");
        assert!(!o.begin_owned(), "a freed slot is not runnable");
    }
}

// The load-bearing invariant, model-checked: when several workers race to steal one pooled fiber,
// **exactly one wins**, and that winner observes the suspending thread's published context (the
// acquire/release pairing). This is the migratable-fiber feature's entire safety argument, isolated
// to pure atomics so loom can exhaust every interleaving — run with
// `RUSTFLAGS="--cfg loom" cargo test -p svm-jit --lib fiber_registry::loom`.
#[cfg(all(test, loom))]
mod loom_tests {
    use super::*;
    use loom::sync::atomic::{AtomicU64, Ordering as LoomOrdering};
    use loom::sync::Arc;

    /// `STEALERS` threads race to steal one runnable fiber → exactly one wins, and the winner reads
    /// the publisher's context (no torn/stale read). Two stealers is enough to exhibit every
    /// claim/claim interleaving; loom checks them all.
    #[test]
    fn loom_single_owner_steal_is_exclusive() {
        const STEALERS: usize = 2;
        loom::model(|| {
            let own = Arc::new(Ownership::new_owned());
            // The "saved context" the suspending worker publishes; a stealer that wins must see it.
            let ctx = Arc::new(AtomicU64::new(0));

            // The creating worker runs the fiber, writes its context, and suspends it into the pool.
            assert!(own.begin_owned());
            ctx.store(0xC0FFEE, LoomOrdering::Relaxed);
            own.suspend_to_pool(); // release-publishes RUNNABLE

            let handles: Vec<_> = (0..STEALERS)
                .map(|_| {
                    let own = own.clone();
                    let ctx = ctx.clone();
                    loom::thread::spawn(move || {
                        if own.try_steal() {
                            // Won exclusive ownership: the acquire CAS must have synchronized-with the
                            // release publish, so the context is visible.
                            assert_eq!(
                                ctx.load(LoomOrdering::Relaxed),
                                0xC0FFEE,
                                "the winning stealer must observe the published context"
                            );
                            1u32
                        } else {
                            0
                        }
                    })
                })
                .collect();

            let winners: u32 = handles.into_iter().map(|h| h.join().unwrap()).sum();
            assert_eq!(winners, 1, "exactly one thread may steal a runnable fiber");
            assert_eq!(own.state(), RUNNING, "the stolen fiber ends up Running");
        });
    }

    /// A worker holding the fiber `Running` and a would-be stealer race: the stealer can **never**
    /// claim a running fiber (only `RUNNABLE → RUNNING` succeeds), regardless of interleaving.
    #[test]
    fn loom_running_fiber_is_never_stealable() {
        loom::model(|| {
            let own = Arc::new(Ownership::new_owned());
            assert!(own.begin_owned()); // RUNNING, not published to the pool

            let thief = {
                let own = own.clone();
                loom::thread::spawn(move || own.try_steal())
            };
            let stolen = thief.join().unwrap();
            assert!(
                !stolen,
                "a running (unpublished) fiber must never be stolen"
            );
        });
    }
}
