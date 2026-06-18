//! Single-owner ownership protocol for **migratable fibers** (D57 / DESIGN.md §23
//! #1) — the load-bearing, loom-verifiable core of stackful work-stealing.
//!
//! The dangerous invariant of stackful work-stealing is **"one native stack, exactly one thread":**
//! a fiber's saved stack + register context must never be resumed by two OS threads at once — that
//! would execute one call stack on two cores and corrupt it irrecoverably (the precise unsafe D56
//! removed when it deleted the VM-owned M:N executor). Fibers are **migratable** (D57 3c): any
//! worker may claim a voluntarily-suspended fiber from the shared registry — and the whole safety
//! of the feature reduces to one atomic property:
//!
//! > a fiber in the steal pool is claimed (transitioned to `Running`) by **exactly one** thread.
//!
//! This module is that property, isolated. It is **pure atomics — it touches no real stack** — so it
//! is fully `loom`-model-checkable, exactly as the `wait`/`notify` futex core is. Per DESIGN.md §23's
//! "earn the risk with verification, not assume it" mandate and the demo roadmap (#3), the protocol
//! was proven here first; it is **wired into the live runtime** (D57 3b-ii/3c): each slot of the
//! domain-shared `fiber_rt::SharedFiberTable` carries one [`Ownership`] word — `cont.resume` on
//! **any** vCPU claims via [`Ownership::claim`] (fresh `OWNED` or pooled `RUNNABLE` → `RUNNING`;
//! exactly one racing claimant wins), a voluntary suspend publishes via
//! [`Ownership::suspend_to_pool`], and a return [`Ownership::finish`]es the slot. Still staged:
//! [`Ownership::recycle_owned`] (slot recycling, with generation-carrying handles on both backends
//! together) and [`Ownership::pin`] (fault-suspended thread-affine fibers, design sketch #2).
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

/// Generation bits a fiber **guest handle** carries (recycling step 1): the handle is
/// `(generation << 16) | slot` and a slot fits in the low 16 bits (`MAX_FIBERS = 1<<16`), so the
/// handle conveys only the low 16 bits of the slot's full word generation. [`Ownership::claim_gen`]
/// validates a handle against `gen_of(word) & FIBER_HANDLE_GEN_MASK`, so the cross-backend handle
/// namespace (shared with `svm_interp`, whose `FIBER_GEN_SHIFT` is likewise 16) matches.
const FIBER_HANDLE_GEN_MASK: u64 = (1 << 16) - 1;

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
pub(crate) struct Ownership {
    word: AtomicU64,
}

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
    #[allow(dead_code)] // the owner-path primitive; the live resume claims via `claim` (3c) — kept for pinned fibers + tests
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

    /// **Claim the fiber for a resume, from any thread (3c).** A `cont.resume` does not know (or
    /// care) whether the slot is `OWNED` (fresh — never started) or `RUNNABLE` (voluntarily
    /// suspended, possibly by *another* thread): both are resumable, so this claims either —
    /// one CAS `(gen, OWNED|RUNNABLE) → (gen, RUNNING)` — and **exactly one** racing claimant
    /// wins (the same single-owner argument as [`Self::try_steal`], to which this is the
    /// "resume-by-handle" front end: the generation comes from the *current* word rather than a
    /// caller-recorded `(slot, gen)`, which is equivalent while slots are not recycled).
    /// `RUNNING`/`FREE`, or losing the CAS, ⇒ `false` (the caller faults — the loser semantics).
    /// **Acquire** on success: the winner synchronizes-with the releasing suspend
    /// ([`Self::suspend_to_pool`]/[`Self::pin`]) and observes the complete saved stack context —
    /// the load-bearing edge that makes a *cross-thread* resume of the saved native stack sound.
    ///
    /// # Precondition (ABA): only sound while slots are **not recycled**
    /// Because the generation is taken from the *current* word, `claim` is **not** ABA-safe: a stale
    /// caller presents no recorded generation for the CAS to reject. This is fine **only** while a slot
    /// is never reused for a different fiber — the live arrangement today ([`SharedFiberTable`] is
    /// push-only; [`Self::recycle_owned`] is unwired). The instant slot recycling is enabled, the
    /// resume-by-handle path **must** switch to a generation-checked claim ([`Self::try_steal`]-style,
    /// with generation-carrying guest handles), or a stale handle could claim a *recycled* fiber. See
    /// [`Self::recycle_owned`] and the `claim_ignores_a_stale_generation` characterization test.
    #[allow(dead_code)] // recycling step 1: the live `cont.resume` path now uses the generation-checked
                        // `claim_gen`; this ungated variant is kept as the primitive + ABA characterization
    pub(crate) fn claim(&self) -> bool {
        let w = self.word.load(Ordering::Relaxed);
        if state_of(w) != OWNED && state_of(w) != RUNNABLE {
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

    /// **Generation-checked resume-by-handle (recycling step 1).** Like [`Self::claim`] — claims a
    /// fresh (`OWNED`) or suspended (`RUNNABLE`) fiber for a `cont.resume`, one acquire CAS, exactly
    /// one racing claimant wins — but the caller presents the **generation carried in the guest
    /// handle**, and the claim succeeds only if it matches the slot's current generation (low
    /// [`FIBER_HANDLE_GEN_MASK`] bits). This is the ABA-safe front end recycling requires: a stale
    /// handle to a slot's *former* occupant (whose generation has since advanced via [`Self::finish`])
    /// is rejected, where [`Self::claim`] would wrongly claim the new occupant. The CAS preserves the
    /// slot's full word generation. While slots are not recycled every generation is 0, so
    /// `claim_gen(0)` is exactly `claim()` — behavior-preserving (the guest handle is then just `slot`).
    pub(crate) fn claim_gen(&self, handle_generation: u64) -> bool {
        let w = self.word.load(Ordering::Relaxed);
        let st = state_of(w);
        if (st != OWNED && st != RUNNABLE)
            || (gen_of(w) & FIBER_HANDLE_GEN_MASK) != (handle_generation & FIBER_HANDLE_GEN_MASK)
        {
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
    #[allow(dead_code)] // staged: the recycling/deque path presents a recorded (slot, gen); `claim` covers resume-by-handle
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
    #[allow(dead_code)] // staged: fault-suspended (thread-affine) fibers — design sketch #2; tests exercise it
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
    ///
    /// **Wiring precondition:** enabling this on the live path is unsound while the resume-by-handle
    /// front end is [`Self::claim`] (which re-reads the generation and so cannot reject a stale
    /// handle to a recycled slot — see `claim`'s ABA precondition). Recycling must land *together*
    /// with generation-carrying handles and a generation-checked claim ([`Self::try_steal`]).
    #[allow(dead_code)] // staged: slot recycling lands with generation-carrying handles on both backends
    pub(crate) fn recycle_owned(&self) {
        let w = self.word.load(Ordering::Relaxed);
        debug_assert_eq!(state_of(w), FREE);
        self.word.store(pack(gen_of(w), OWNED), Ordering::Release);
    }

    /// The current generation (relaxed) — for observability / assertions.
    #[allow(dead_code)] // observability/assertions (tests)
    pub(crate) fn generation(&self) -> u64 {
        gen_of(self.word.load(Ordering::Relaxed))
    }

    /// The current state, one of `OWNED`/`RUNNABLE`/`RUNNING`/`FREE` (relaxed) — for assertions.
    #[allow(dead_code)] // observability/assertions (tests)
    pub(crate) fn state(&self) -> u64 {
        state_of(self.word.load(Ordering::Relaxed))
    }

    /// Whether a worker is currently executing this fiber's native stack (`RUNNING`) — for the
    /// `gc.roots` walker (`fiber_rt`) to pick a parked vs. running stack-scan extent (relaxed read;
    /// the scan happens at a safepoint with the chain quiescent / stop-the-world).
    pub(crate) fn is_running(&self) -> bool {
        state_of(self.word.load(Ordering::Relaxed)) == RUNNING
    }

    /// Whether the fiber is **voluntarily suspended** (`RUNNABLE`, in the steal pool) — the durable
    /// freeze driver (DURABILITY.md §12.8) flattens exactly these (a fresh `OWNED` fiber has no
    /// continuation to flatten; `RUNNING`/`FREE` are not parked). Relaxed: the freeze runs at a
    /// quiescent safepoint (the root has unwound, single-vCPU).
    pub(crate) fn is_runnable(&self) -> bool {
        state_of(self.word.load(Ordering::Relaxed)) == RUNNABLE
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

    /// The 3c `claim` front end's transition table: claims succeed exactly on `OWNED` (fresh) and
    /// `RUNNABLE` (suspended), never on `RUNNING`/`FREE`.
    #[test]
    fn claim_transition_table() {
        let o = Ownership::new_owned();
        assert!(o.claim(), "a fresh (OWNED) fiber is claimable"); // → RUNNING
        assert!(!o.claim(), "a running fiber is not claimable");
        o.suspend_to_pool(); // → RUNNABLE
        assert!(o.claim(), "a suspended (RUNNABLE) fiber is claimable"); // → RUNNING
        o.finish(); // → FREE
        assert!(!o.claim(), "a finished slot is not claimable");
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

    /// **Characterization of `claim`'s ABA precondition (finding C1).** Unlike [`Ownership::try_steal`],
    /// the resume-by-handle [`Ownership::claim`] takes the generation from the *current* word, so it
    /// does **not** reject a caller holding a stale view. This test pins that root-cause behaviour: once
    /// a slot is finished (generation bumped) and `recycle_owned`'d for a new fiber, `claim` succeeds on
    /// the *new* occupant with no regard for the old generation. That is exactly why `claim` is sound
    /// **only** while slots are push-only (the live arrangement), and why enabling `recycle_owned` on
    /// the live path must come with a generation-checked claim. If this assertion ever needs to change,
    /// it is because the resume path gained a generation check — update `claim`'s precondition doc too.
    #[test]
    fn claim_ignores_a_stale_generation() {
        let o = Ownership::new_owned(); // fiber A, gen 0
        assert!(o.begin_owned());
        o.finish(); // A done → FREE, gen → 1
        o.recycle_owned(); // slot reused for fiber B, gen 1 (the staged recycling primitive)
        assert_eq!(o.generation(), 1);
        // `claim` does not carry/record a generation, so it claims B regardless of any stale (gen-0)
        // view a caller might hold — the unsafety the precondition guards against once recycling lands.
        assert!(
            o.claim(),
            "claim takes the generation from the current word, so it claims the recycled fiber"
        );
        assert_eq!(o.state(), RUNNING);
    }

    /// Recycling step 1: the live resume path uses [`Ownership::claim_gen`], which **does** carry the
    /// handle's generation — so a stale handle to a recycled slot is rejected (the ABA guard that lets
    /// recycling land), while the current-generation handle claims it. The companion to
    /// `claim_ignores_a_stale_generation` for the generation-checked front end.
    #[test]
    fn claim_gen_rejects_a_stale_generation() {
        let o = Ownership::new_owned(); // fiber A, gen 0
        assert!(o.begin_owned());
        o.finish(); // A done → FREE, gen → 1
        o.recycle_owned(); // slot reused for fiber B, gen 1
        assert_eq!(o.generation(), 1);
        assert!(
            !o.claim_gen(0),
            "a stale (gen-0) handle must not claim the recycled (gen-1) fiber"
        );
        assert_eq!(
            o.state(),
            OWNED,
            "the recycled fiber is untouched by the stale claim"
        );
        assert!(
            o.claim_gen(1),
            "the current-generation handle claims the recycled fiber"
        );
        assert_eq!(o.state(), RUNNING);
    }

    /// `claim_gen(0)` on a never-recycled slot is exactly `claim()` (every generation is 0 today), so
    /// the behavior-preserving wiring holds: fresh `OWNED` and suspended `RUNNABLE` are claimable once.
    #[test]
    fn claim_gen_matches_claim_at_generation_zero() {
        let o = Ownership::new_owned();
        assert!(
            o.claim_gen(0),
            "a fresh (OWNED, gen 0) fiber is claimable by its handle"
        );
        assert!(!o.claim_gen(0), "a running fiber is not claimable");
        o.suspend_to_pool();
        assert!(
            o.claim_gen(0),
            "a suspended (RUNNABLE, gen 0) fiber is claimable"
        );
        o.finish();
        assert!(!o.claim_gen(0), "a finished (FREE) slot is not claimable");
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

    /// **The 3c resume arbiter:** several workers race to `claim` (resume) one fiber a suspending
    /// worker just published — exactly one wins, and the winner observes the published context
    /// (the acquire/release pairing that makes resuming the saved *native stack* from another
    /// thread sound). This is `loom_single_owner_steal_is_exclusive` through the `claim` front
    /// end the live `cont.resume` thunk actually calls.
    #[test]
    fn loom_claim_is_exclusive_across_threads() {
        const CLAIMANTS: usize = 2;
        loom::model(|| {
            let own = Arc::new(Ownership::new_owned());
            let ctx = Arc::new(LoomU64::new(0));

            // The creating worker runs the fiber, writes its saved context, suspends it to the pool.
            assert!(own.begin_owned());
            ctx.store(0xF1BE5, LoomOrdering::Relaxed);
            own.suspend_to_pool(); // release-publishes RUNNABLE

            let handles: Vec<_> = (0..CLAIMANTS)
                .map(|_| {
                    let own = own.clone();
                    let ctx = ctx.clone();
                    loom::thread::spawn(move || {
                        if own.claim() {
                            assert_eq!(
                                ctx.load(LoomOrdering::Relaxed),
                                0xF1BE5,
                                "the winning claimant must observe the published stack context"
                            );
                            1u32
                        } else {
                            0
                        }
                    })
                })
                .collect();

            let winners: u32 = handles.into_iter().map(|h| h.join().unwrap()).sum();
            assert_eq!(winners, 1, "exactly one thread may claim a resumable fiber");
            assert_eq!(own.state(), RUNNING);
        });
    }

    /// The **generation-checked** front end the live `cont.resume` thunk now calls (recycling step 1):
    /// several workers race to `claim_gen(g)` one fiber suspended at generation `g` — still exactly one
    /// wins, and the winner observes the published context. The generation check must not weaken the
    /// single-owner arbitration (it only *adds* a reject for a mismatched generation).
    #[test]
    fn loom_claim_gen_is_exclusive_across_threads() {
        const CLAIMANTS: usize = 2;
        loom::model(|| {
            let own = Arc::new(Ownership::new_owned());
            let ctx = Arc::new(LoomU64::new(0));

            assert!(own.begin_owned());
            ctx.store(0xF1BE5, LoomOrdering::Relaxed);
            let g = own.suspend_to_pool(); // release-publishes (g, RUNNABLE)

            let handles: Vec<_> = (0..CLAIMANTS)
                .map(|_| {
                    let own = own.clone();
                    let ctx = ctx.clone();
                    loom::thread::spawn(move || {
                        if own.claim_gen(g) {
                            assert_eq!(
                                ctx.load(LoomOrdering::Relaxed),
                                0xF1BE5,
                                "the winning claimant must observe the published stack context"
                            );
                            1u32
                        } else {
                            0
                        }
                    })
                })
                .collect();

            let winners: u32 = handles.into_iter().map(|h| h.join().unwrap()).sum();
            assert_eq!(
                winners, 1,
                "exactly one thread may claim_gen a resumable fiber"
            );
            assert_eq!(own.state(), RUNNING);
        });
    }

    /// `claim` races on a **fresh** (`OWNED`, never-started) fiber: still exactly one winner — the
    /// interp registry lets any vCPU first-resume a pending fiber, so the JIT arbiter must too.
    #[test]
    fn loom_claim_of_fresh_fiber_is_exclusive() {
        loom::model(|| {
            let own = Arc::new(Ownership::new_owned()); // OWNED, never run
            let handles: Vec<_> = (0..2)
                .map(|_| {
                    let own = own.clone();
                    loom::thread::spawn(move || u32::from(own.claim()))
                })
                .collect();
            let winners: u32 = handles.into_iter().map(|h| h.join().unwrap()).sum();
            assert_eq!(
                winners, 1,
                "exactly one thread may first-resume a fresh fiber"
            );
            assert_eq!(own.state(), RUNNING);
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
