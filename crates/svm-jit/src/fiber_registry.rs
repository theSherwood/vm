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
//! ## States (low 2 bits of an `AtomicU64`)
//! - `OWNED` — owned by one worker, **not** running and **not** in the steal pool: a just-created
//!   fiber (owned by its creator), or one *pinned* to its thread (a §5/§14 fault-suspended fiber
//!   carrying thread-affine recovery state — `sigjmp_buf`/VEH `CONTEXT` — which design sketch #2
//!   excludes from stealing). Only its owner may run it.
//! - `RUNNABLE` — **voluntarily** suspended, published into the shared steal pool, owned by no
//!   thread. The only stealable state.
//! - `RUNNING` — a worker is executing the native stack. **Never** stealable.
//! - `FREE` — the fiber returned; the slot is reclaimable for a new fiber.
//!
//! ## Generation tagging (ABA-safety)
//! A shared registry **reuses slots**: a finished fiber's slot is recycled for a new fiber. The high
//! bits of the word are a **generation** counter, bumped on `finish`, so a stealer that recorded a
//! pooled fiber as `(slot, gen)` and acts only later — after that slot was stolen, finished, and
//! reused for a *different* fiber — finds its `try_steal(gen)` CAS fail (the word now carries a newer
//! generation). Without this, the classic ABA hazard would let the stale stealer claim the *new*
//! fiber as if it were the old one. The whole `(generation, state)` pair lives in one word so a
//! single CAS arbitrates both.
//!
//! ## Memory ordering
//! A suspend that publishes a fiber to the pool (`suspend_to_pool`, `pin`) is a **release** store; a
//! claim that takes ownership (`try_steal`, `begin_owned`) is an **acquire** CAS. So the winning
//! claimant *synchronizes-with* the suspending thread and observes the complete saved context — the
//! same publish/consume discipline as the futex word and the JIT's atomic `FnEntry` (DESIGN §22).

#[cfg(loom)]
use loom::sync::atomic::{AtomicU64, Ordering};
#[cfg(not(loom))]
use std::sync::atomic::{AtomicU64, Ordering};

/// Owned by one worker; not running, not stealable (just-created, or pinned/fault-suspended).
const OWNED: u64 = 0;
/// Voluntarily suspended in the shared steal pool; ownerless — the only stealable state.
const RUNNABLE: u64 = 1;
/// A worker is executing this fiber's native stack — never stealable.
const RUNNING: u64 = 2;
/// The fiber returned; the slot is reclaimable for a new fiber.
const FREE: u64 = 3;

/// Low-bit mask selecting the state; the rest of the word is the generation.
const STATE_BITS: u64 = 2;
const STATE_MASK: u64 = (1 << STATE_BITS) - 1;

#[inline]
fn pack(generation: u64, state: u64) -> u64 {
    (generation << STATE_BITS) | state
}
#[inline]
fn state_of(word: u64) -> u64 {
    word & STATE_MASK
}
#[inline]
fn gen_of(word: u64) -> u64 {
    word >> STATE_BITS
}

/// The atomic `(generation, state)` of one fiber slot. One per fiber in the (future) shared registry;
/// the only synchronization a steal needs (pool membership *is* `state == RUNNABLE`, so the
/// load-bearing race is this single CAS, not the pool container — a Chase-Lev deque or a mutex'd vec
/// is an orthogonal, non-unsafe choice layered on top).
#[allow(dead_code)] // staged: verified in isolation here; runtime integration is the next slice
pub(crate) struct Ownership {
    word: AtomicU64,
}

#[allow(dead_code)] // staged: see the module header
impl Ownership {
    /// A freshly created fiber, **owned by its creator** and not yet in the steal pool (generation 0).
    pub(crate) fn new_owned() -> Ownership {
        Ownership {
            word: AtomicU64::new(pack(0, OWNED)),
        }
    }

    /// The owner begins running an `OWNED` fiber: `OWNED → RUNNING`, keeping the generation. Returns
    /// `false` if the slot is not `OWNED` (a caller bug — an owned fiber is not concurrently
    /// stealable, so only its owner transitions it, and this CAS cannot legitimately lose).
    /// **Acquire** on success: the owner observes whatever it published when it last `pin`ned it.
    pub(crate) fn begin_owned(&self) -> bool {
        let w = self.word.load(Ordering::Relaxed);
        if state_of(w) != OWNED {
            return false;
        }
        self.word
            .compare_exchange(
                w,
                pack(gen_of(w), RUNNING),
                Ordering::Acquire,
                Ordering::Relaxed,
            )
            .is_ok()
    }

    /// Voluntarily suspend the running fiber **into the steal pool**: `RUNNING → RUNNABLE` (same
    /// generation), a **release** store so a stealer's acquiring claim sees the complete saved
    /// context. Returns the slot's current **generation**, which the caller records alongside the
    /// slot index when it pushes the fiber onto the steal deque — a later `try_steal` must present it.
    pub(crate) fn suspend_to_pool(&self) -> u64 {
        let w = self.word.load(Ordering::Relaxed);
        debug_assert_eq!(state_of(w), RUNNING);
        let g = gen_of(w);
        self.word.store(pack(g, RUNNABLE), Ordering::Release);
        g
    }

    /// Attempt to steal the pooled fiber the caller recorded as `(this slot, generation)`: CAS
    /// `(generation, RUNNABLE) → (generation, RUNNING)`. Returns `true` iff **this** thread won
    /// exclusive ownership — at most one caller ever does. Fails (and the caller backs off to another
    /// slot) if the slot was already stolen, is running/pinned, or — the ABA guard — was **finished
    /// and reused** since the caller recorded it (its generation has advanced). **Acquire** on
    /// success: the winner synchronizes-with the suspending thread and observes the published context.
    pub(crate) fn try_steal(&self, generation: u64) -> bool {
        self.word
            .compare_exchange(
                pack(generation, RUNNABLE),
                pack(generation, RUNNING),
                Ordering::Acquire,
                Ordering::Relaxed,
            )
            .is_ok()
    }

    /// **Pin** the running fiber to its current owner (not stealable): `RUNNING → OWNED`, same
    /// generation. Used for a fault-suspended fiber whose recovery state (`sigjmp_buf`/VEH `CONTEXT`)
    /// is thread-affine, so it must resume on the same thread (design sketch #2). A **release** store,
    /// symmetric with `suspend_to_pool`, so the eventual `begin_owned` sees the saved context.
    pub(crate) fn pin(&self) {
        let w = self.word.load(Ordering::Relaxed);
        debug_assert_eq!(state_of(w), RUNNING);
        self.word.store(pack(gen_of(w), OWNED), Ordering::Release);
    }

    /// The running fiber returned: `RUNNING → FREE`, **bumping the generation** so any stealer still
    /// holding the old `(slot, generation)` can never claim the slot once it is reused. The slot is
    /// now reclaimable via [`recycle_owned`](Self::recycle_owned).
    pub(crate) fn finish(&self) {
        let w = self.word.load(Ordering::Relaxed);
        debug_assert_eq!(state_of(w), RUNNING);
        self.word
            .store(pack(gen_of(w).wrapping_add(1), FREE), Ordering::Release);
    }

    /// Reuse a `FREE` slot for a **new** fiber: `FREE → OWNED`, keeping the (already-bumped)
    /// generation so stale stealers of the *previous* occupant remain locked out (the ABA guard).
    pub(crate) fn recycle_owned(&self) {
        let w = self.word.load(Ordering::Relaxed);
        debug_assert_eq!(state_of(w), FREE);
        self.word.store(pack(gen_of(w), OWNED), Ordering::Release);
    }

    /// The current generation (relaxed) — for observability / assertions.
    pub(crate) fn generation(&self) -> u64 {
        gen_of(self.word.load(Ordering::Relaxed))
    }

    /// The current state, one of `OWNED`/`RUNNABLE`/`RUNNING`/`FREE` (relaxed) — for assertions.
    pub(crate) fn state(&self) -> u64 {
        state_of(self.word.load(Ordering::Relaxed))
    }
}

// Real-build (non-loom) unit tests: the single-threaded transition table + the deterministic ABA
// guard. The *concurrent* single-owner property is the loom model below.
#[cfg(all(test, not(loom)))]
mod tests {
    use super::*;

    #[test]
    fn owned_run_suspend_steal_cycle() {
        let o = Ownership::new_owned();
        assert_eq!(o.state(), OWNED);
        assert!(o.begin_owned(), "the owner runs its owned fiber");
        assert_eq!(o.state(), RUNNING);
        // A running fiber is never stealable (no generation matches a RUNNING slot).
        assert!(!o.try_steal(0), "a running fiber must not be stealable");
        // Voluntary suspend publishes it to the pool at the current generation.
        let g = o.suspend_to_pool();
        assert_eq!(o.state(), RUNNABLE);
        assert!(
            o.try_steal(g),
            "a pooled fiber is stealable at its generation"
        );
        assert_eq!(o.state(), RUNNING);
        assert!(!o.try_steal(g), "it cannot be stolen twice");
    }

    #[test]
    fn pinned_fiber_is_not_in_the_pool() {
        let o = Ownership::new_owned();
        assert!(o.begin_owned());
        o.pin(); // fault-suspended → stays with its owner, excluded from the pool
        assert_eq!(o.state(), OWNED);
        assert!(!o.try_steal(0), "a pinned fiber must not be stealable");
        // Its owner resumes it.
        assert!(o.begin_owned());
        assert_eq!(o.state(), RUNNING);
    }

    #[test]
    fn finish_bumps_generation_and_frees() {
        let o = Ownership::new_owned();
        assert_eq!(o.generation(), 0);
        assert!(o.begin_owned());
        o.finish();
        assert_eq!(o.state(), FREE);
        assert_eq!(o.generation(), 1, "finish bumps the generation");
        assert!(!o.try_steal(0), "a freed slot is not stealable");
        assert!(!o.begin_owned(), "a freed slot is not runnable");
    }

    /// **The ABA guard, deterministically.** A stealer records a pooled fiber as `(slot, gen0)`. Before
    /// it acts, that slot is fully recycled: stolen, finished (generation bumped), reused for a *new*
    /// fiber, and that new fiber published. The stale `try_steal(gen0)` must **fail** — otherwise the
    /// stealer would resume the new fiber's stack believing it claimed the old one (a single-owner
    /// violation across reuse). A correct stealer at the new generation succeeds.
    #[test]
    fn stale_generation_steal_fails_across_reuse() {
        let o = Ownership::new_owned(); // fiber A, gen 0
        assert!(o.begin_owned());
        let g0 = o.suspend_to_pool(); // A published at gen 0
        assert_eq!(g0, 0);

        // The slot is fully recycled by other workers while our stealer still holds (slot, g0):
        assert!(o.try_steal(g0)); //   A stolen
        o.finish(); //                 A done → FREE, gen → 1
        o.recycle_owned(); //          slot reused for fiber B, gen 1
        assert!(o.begin_owned());
        let g1 = o.suspend_to_pool(); // B published at gen 1
        assert_eq!(g1, 1);

        // The stale stealer finally acts with the old generation — must be locked out.
        assert!(
            !o.try_steal(g0),
            "a stale-generation steal must fail across slot reuse (ABA-safe)"
        );
        // The correct generation still works.
        assert!(o.try_steal(g1), "the current generation steals normally");
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
    use loom::sync::atomic::{AtomicU64 as LoomU64, Ordering as LoomOrdering};
    use loom::sync::Arc;

    /// `STEALERS` threads race to steal one runnable fiber at its generation → exactly one wins, and
    /// the winner reads the publisher's context (no torn/stale read). Two stealers is enough to
    /// exhibit every claim/claim interleaving; loom checks them all.
    #[test]
    fn loom_single_owner_steal_is_exclusive() {
        const STEALERS: usize = 2;
        loom::model(|| {
            let own = Arc::new(Ownership::new_owned());
            // The "saved context" the suspending worker publishes; a stealer that wins must see it.
            let ctx = Arc::new(LoomU64::new(0));

            // The creating worker runs the fiber, writes its context, and suspends it into the pool.
            assert!(own.begin_owned());
            ctx.store(0xC0FFEE, LoomOrdering::Relaxed);
            let g = own.suspend_to_pool(); // release-publishes (g, RUNNABLE)

            let handles: Vec<_> = (0..STEALERS)
                .map(|_| {
                    let own = own.clone();
                    let ctx = ctx.clone();
                    loom::thread::spawn(move || {
                        if own.try_steal(g) {
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
    /// claim a running fiber (no generation matches `RUNNING`), regardless of interleaving.
    #[test]
    fn loom_running_fiber_is_never_stealable() {
        loom::model(|| {
            let own = Arc::new(Ownership::new_owned());
            assert!(own.begin_owned()); // RUNNING, not published to the pool

            let thief = {
                let own = own.clone();
                loom::thread::spawn(move || own.try_steal(0))
            };
            let stolen = thief.join().unwrap();
            assert!(
                !stolen,
                "a running (unpublished) fiber must never be stolen"
            );
        });
    }
}
