//! A safe, RAII **asymmetric coroutine** over the raw [`jump`]/[`make`] primitive.
//!
//! A [`Fiber`] owns its control [`Stack`] and a boxed entry closure. [`Fiber::resume`] switches into
//! it (passing a `u64`); the body runs until it either [`Yielder::suspend`]s a value back (the fiber
//! stays alive, resumable) or returns (the fiber is finished). All communication rides through a
//! heap [`Control`] cell shared between the two sides; since an asymmetric coroutine and its resumer
//! never run *concurrently* (control is always on exactly one side), plain `Cell`s suffice.
//!
//! Panic safety: unwinding across a stack switch is undefined behavior, so the fiber body runs under
//! [`catch_unwind`] and a panic is turned into a process [`abort`] at the boundary.

use crate::stack::Stack;
use crate::switch::{jump, make, Transfer};
use std::cell::Cell;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::process::abort;
use std::sync::atomic::{AtomicBool, Ordering};

/// AddressSanitizer **fiber-switch annotations** (`feature = "asan"`). ASan tracks each thread's
/// current stack (shadow + fake-stack bookkeeping), so every manual stack switch must be bracketed:
/// `__sanitizer_start_switch_fiber` on the *outgoing* stack right before the jump, and
/// `__sanitizer_finish_switch_fiber` on the *incoming* stack right after arrival — otherwise
/// ASan-instrumented code running on a fiber stack crashes or misreports. This is what makes the
/// migratable-fiber empirical net's sanitizer layer (DESIGN.md §23) actually runnable: with the
/// feature on, the whole fiber suite runs under ASan, fiber stacks included. Zero-cost otherwise.
#[cfg(feature = "asan")]
mod asan {
    use core::ffi::c_void;
    extern "C" {
        /// Save the current (outgoing) context's fake stack into `*fake_stack_save` and announce
        /// the stack we are about to switch to. A null `fake_stack_save` means the outgoing
        /// context is **dying** — ASan releases its fake stack.
        pub fn __sanitizer_start_switch_fiber(
            fake_stack_save: *mut *mut c_void,
            bottom: *const c_void,
            size: usize,
        );
        /// Restore this (incoming) context's fake stack (`null` on first entry) and report the
        /// bounds of the stack the switch came *from* through the out-params (may be null).
        pub fn __sanitizer_finish_switch_fiber(
            fake_stack: *mut c_void,
            bottom_old: *mut *const c_void,
            size_old: *mut usize,
        );
    }
}

/// Per-fiber ASan bookkeeping (`feature = "asan"`): each side's saved fake-stack pointer plus the
/// bounds of whatever stack last switched *into* the fiber — re-captured on every switch-in, since
/// the resumer may be a **different OS thread** each time (the D57 3c migration case).
#[cfg(feature = "asan")]
struct AsanState {
    /// The resumer's fake stack, saved while the fiber runs (written by `resume`'s start, restored
    /// by its finish when control returns).
    resumer_fake: Cell<*mut core::ffi::c_void>,
    /// The fiber's fake stack, saved while it is suspended.
    fiber_fake: Cell<*mut core::ffi::c_void>,
    /// Bounds of the stack the last switch-in came from (the current resumer's stack), captured by
    /// the fiber side's `finish` and used when suspending back out.
    from_bottom: Cell<*const core::ffi::c_void>,
    from_size: Cell<usize>,
    /// The fiber's own usable stack bounds (constant for its lifetime).
    stack_bottom: *const core::ffi::c_void,
    stack_size: usize,
}

/// The outcome of a [`Fiber::resume`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum State {
    /// The fiber [`Yielder::suspend`]ed this value and can be resumed again.
    Yielded(u64),
    /// The fiber's body returned this value; it must not be resumed again.
    Complete(u64),
}

/// Shared single-threaded mailbox between a [`Fiber`] and its running body. Only one side touches it
/// at a time (the other is suspended inside a [`jump`]), so `Cell` is sound — no atomics needed.
struct Control {
    /// The value in flight: set by `resume` before switching in, read by the body; set by the body
    /// before suspending/finishing, read by `resume` after switching out.
    value: Cell<u64>,
    /// The resumer's current context, refreshed on every switch so [`Yielder::suspend`] can return.
    resumer: Cell<*mut u8>,
    /// Set by the body when it returns (vs. suspends).
    done: Cell<bool>,
    /// **Single-owner tripwire** (escape-TCB defense-in-depth): `true` while this fiber's native stack
    /// is executing under a [`Fiber::resume`]. The whole crate's `Cell`-based soundness rests on the
    /// caller never resuming one fiber from two threads at once (the §23 single-owner protocol the
    /// `svm-jit` arbiter enforces). This is the only cross-thread-visible field, so a violation of that
    /// contract — an aliased `*mut Fiber` resumed concurrently — is caught here and `abort`s loudly
    /// *before* any `Cell` is raced, instead of becoming silent UB. Costs one relaxed-ish atomic swap
    /// per resume.
    running: AtomicBool,
    /// Whether the entry closure has been moved into the running body yet (for drop-before-start).
    taken: Cell<bool>,
    /// The entry closure, type-erased as `*mut F`; reclaimed by the body (`take`) or by `Drop`.
    closure: Cell<*mut ()>,
    /// Monomorphized dropper for `closure`, used only if the fiber is dropped before it ever ran.
    drop_closure: unsafe fn(*mut ()),
    /// This control stack's lowest usable address ([`Stack::usable_low`]). Exposed to the body via
    /// [`Yielder::stack_low`] so a caller (the svm-jit software stack-overflow guard) can pass it as
    /// the running stack's limit. Set once at [`Fiber::new`]; the stack is address-stable for the
    /// fiber's life.
    usable_low: u64,
    /// ASan fiber-switch bookkeeping (see [`AsanState`]).
    #[cfg(feature = "asan")]
    asan: AsanState,
}

/// Handed to a fiber body so it can suspend back to whoever resumed it.
pub struct Yielder {
    control: *const Control,
}

impl Yielder {
    /// This fiber's control-stack low bound (`usable_low`) — the running native stack's limit, for a
    /// caller's software stack-overflow guard. Constant for the fiber's life.
    pub fn stack_low(&self) -> u64 {
        // SAFETY: `control` outlives the body (the `Fiber` owns the `Box<Control>`).
        unsafe { (*self.control).usable_low }
    }

    /// Suspend the fiber, handing `val` back to the resumer; returns the value passed to the next
    /// [`Fiber::resume`].
    pub fn suspend(&self, val: u64) -> u64 {
        // SAFETY: `control` outlives the body (the `Fiber` owns the `Box<Control>` and cannot be
        // dropped while suspended inside its own body). The resumer is parked, so we have exclusive
        // access.
        let c = unsafe { &*self.control };
        c.value.set(val);
        // ASan: leaving the fiber stack for the resumer's stack (whose bounds were captured at our
        // last switch-in).
        #[cfg(feature = "asan")]
        // SAFETY: bracketing the jump below per the sanitizer fiber-annotation contract.
        unsafe {
            asan::__sanitizer_start_switch_fiber(
                c.asan.fiber_fake.as_ptr(),
                c.asan.from_bottom.get(),
                c.asan.from_size.get(),
            );
        }
        // SAFETY: `resumer` is the live context that switched into us.
        let t = unsafe { jump(c.resumer.get(), 0) };
        // Back on the fiber stack — possibly resumed from a *different thread/stack* than the one
        // we suspended toward (migration), so re-capture where this resume came from.
        #[cfg(feature = "asan")]
        // SAFETY: the arrival half of the bracketing above.
        unsafe {
            asan::__sanitizer_finish_switch_fiber(
                c.asan.fiber_fake.get(),
                c.asan.from_bottom.as_ptr(),
                c.asan.from_size.as_ptr(),
            );
        }
        c.resumer.set(t.fctx); // the resumer may be a different context next time
        c.value.get()
    }

    /// The resumer's saved stack pointer — the context this fiber will `jump` back to when it
    /// suspends, i.e. the resumer's live low-water mark at the moment it switched in. Used by the
    /// `svm-jit` `gc.roots` walker as the conservative low bound for scanning the **root
    /// computation's** frames (the resume chain's non-fiber parent runs on the OS thread stack, so
    /// its live region is `[resumer_sp, entry_sp)`).
    pub fn resumer_sp(&self) -> *const u8 {
        // SAFETY: `control` outlives the running body (the `Fiber` owns the `Box<Control>` and
        // cannot be dropped while suspended inside its own body), and only this side is live.
        unsafe { (*self.control).resumer.get() as *const u8 }
    }

    /// A stable identity for this fiber's `Control` — the box address, equal to the owning
    /// [`Fiber::control_id`]. Lets a GC scanner correlate a running fiber (found via its `Fiber` in
    /// the shared table) with its live `Yielder` in the resume chain, to read a tight scan low bound
    /// (`resumer_sp`) instead of the whole usable stack. Pointer identity only — never dereferenced.
    pub fn control_id(&self) -> usize {
        self.control as usize
    }
}

/// A first-class suspendable computation running on its own native stack.
pub struct Fiber {
    _stack: Stack, // owns the control stack; freed on drop
    ctx: *mut u8,  // the fiber's saved context — where to `jump` to resume it
    done: bool,
    control: Box<Control>,
}

// SAFETY: a `Fiber` bundles an owned stack + control cell reachable only through itself; it is only
// ever resumed by one thread at a time (never concurrently with its own body), so it is sound to move
// between threads (e.g. a scheduler migrating a parked fiber between workers).
unsafe impl Send for Fiber {}

/// Monomorphized dropper: reconstruct and drop the `Box<F>` behind an erased pointer.
unsafe fn drop_closure_impl<F>(p: *mut ()) {
    // SAFETY: `p` came from `Box::into_raw(Box::<F>::new(..))` for this same `F`.
    drop(unsafe { Box::from_raw(p as *mut F) });
}

/// The monomorphized [`crate::imp::Entry`] for a fiber whose body is an `F`. The first [`jump`] in
/// passes the [`Control`] pointer as `Transfer::data` to bootstrap; thereafter values flow through
/// the control cell.
extern "C" fn fiber_entry<F>(t: Transfer) -> !
where
    F: FnOnce(&Yielder, u64) -> u64,
{
    let control = t.data as *const Control;
    // SAFETY: `control` was just handed to us by `resume`; the resumer is parked.
    let c = unsafe { &*control };
    // ASan: first arrival on this fiber's stack (no fake stack to restore yet); capture the
    // resumer's stack bounds for the eventual suspend/exit back out.
    #[cfg(feature = "asan")]
    // SAFETY: the arrival half of `resume`'s start_switch bracketing.
    unsafe {
        asan::__sanitizer_finish_switch_fiber(
            core::ptr::null_mut(),
            c.asan.from_bottom.as_ptr(),
            c.asan.from_size.as_ptr(),
        );
    }
    c.resumer.set(t.fctx);
    c.taken.set(true);
    // SAFETY: `closure` is a `Box<F>` for this exact `F` (set in `Fiber::new`).
    let f: F = *unsafe { Box::from_raw(c.closure.get() as *mut F) };
    let yielder = Yielder { control };
    let first = c.value.get();

    // A panic must not unwind across the stack switch below (UB) — abort if the body panics.
    let result = match catch_unwind(AssertUnwindSafe(|| f(&yielder, first))) {
        Ok(v) => v,
        Err(_) => abort(),
    };

    c.value.set(result);
    c.done.set(true);
    // ASan: this context is dying — a null save slot tells ASan to release its fake stack.
    #[cfg(feature = "asan")]
    // SAFETY: bracketing the final jump; the matching finish runs in `resume`.
    unsafe {
        asan::__sanitizer_start_switch_fiber(
            core::ptr::null_mut(),
            c.asan.from_bottom.get(),
            c.asan.from_size.get(),
        );
    }
    // SAFETY: final switch back to the resumer; this context is never resumed again.
    unsafe { jump(c.resumer.get(), 0) };
    unreachable!("fiber resumed after completion")
}

impl Fiber {
    /// Create a fiber with a `stack_size`-byte (rounded up) guard-paged control stack. The body
    /// receives a [`Yielder`] and the first resume value, and returns the fiber's final value.
    ///
    /// Returns `None` if the OS refuses the stack reservation — a **recoverable** condition the
    /// caller turns into a `FiberFault` rather than an abort, so a guest spawning many fibers can
    /// never crash the host (ISSUES.md I1).
    pub fn new<F>(stack_size: usize, f: F) -> Option<Fiber>
    where
        F: FnOnce(&Yielder, u64) -> u64 + 'static,
    {
        let stack = Stack::new(stack_size)?;
        #[cfg(feature = "asan")]
        let (asan_bottom, asan_size) = stack.usable();
        let control = Box::new(Control {
            value: Cell::new(0),
            resumer: Cell::new(std::ptr::null_mut()),
            done: Cell::new(false),
            taken: Cell::new(false),
            running: AtomicBool::new(false),
            closure: Cell::new(Box::into_raw(Box::new(f)) as *mut ()),
            drop_closure: drop_closure_impl::<F>,
            usable_low: stack.usable_low() as u64,
            #[cfg(feature = "asan")]
            asan: AsanState {
                resumer_fake: Cell::new(std::ptr::null_mut()),
                fiber_fake: Cell::new(std::ptr::null_mut()),
                from_bottom: Cell::new(std::ptr::null()),
                from_size: Cell::new(0),
                stack_bottom: asan_bottom as *const core::ffi::c_void,
                stack_size: asan_size,
            },
        });
        // SAFETY: fresh, live stack; `fiber_entry::<F>` matches the boxed closure's type.
        let ctx = unsafe { make(&stack, fiber_entry::<F>) };
        Some(Fiber {
            _stack: stack,
            ctx,
            done: false,
            control,
        })
    }

    /// Whether the fiber has finished (returned). A finished fiber must not be resumed.
    pub fn is_done(&self) -> bool {
        self.done
    }

    /// Conservative GC stack-scan bounds `[low, high)` for a **parked** (suspended or fresh) fiber:
    /// `[ctx, top)`. `ctx` is the saved-context pointer — the lowest live address, since the switch
    /// spilled the fiber's callee-saved registers (its in-register roots) there before suspending —
    /// so this is the *exact* live extent, and scanning it conservatively (every in-range word is a
    /// candidate root) cannot miss a root the suspended fiber holds.
    ///
    /// Must only be called on a fiber that is **not currently running** (parked): while a fiber
    /// runs, `ctx` is the stale pre-resume context and does not bound the live frames.
    pub fn parked_extent(&self) -> (*const u8, *const u8) {
        (self.ctx as *const u8, self._stack.top() as *const u8)
    }

    /// Conservative bounds `[usable_low, top)` for a **running** fiber whose exact live SP the
    /// scanner does not know (a resume-chain ancestor, or the fiber calling `gc.roots` itself):
    /// scan the *whole* usable stack — a sound superset of its live frames (the unused portion is
    /// untouched/zeroed reserved pages or stale non-root bytes, harmless to a conservative scan).
    pub fn full_extent(&self) -> (*const u8, *const u8) {
        (self._stack.usable_low(), self._stack.top() as *const u8)
    }

    /// The high end (`top`) of this fiber's usable stack — the upper bound of any scan extent.
    pub fn stack_top(&self) -> *const u8 {
        self._stack.top() as *const u8
    }

    /// A stable identity for this fiber's `Control` — the `Box<Control>` address, equal to the
    /// [`Yielder::control_id`] handed to its body. Used by the GC scanner to correlate a running
    /// fiber with its `Yielder` in the resume chain (see `Yielder::control_id`). Identity only.
    pub fn control_id(&self) -> usize {
        &*self.control as *const Control as usize
    }

    /// Resume the fiber, passing `val`; returns whether it yielded or completed.
    ///
    /// # Panics
    /// Panics if the fiber has already completed.
    pub fn resume(&mut self, val: u64) -> State {
        assert!(!self.done, "resumed a finished fiber");
        // Single-owner tripwire (see `Control::running`): claim exclusive execution of this fiber's
        // stack before touching any `Cell`. If another thread is already inside this fiber's body
        // (an aliased `*mut Fiber` resumed concurrently — the catastrophe the §23 protocol forbids),
        // this swap sees `true` and we abort loudly rather than race the `Cell`s into UB. `Acquire`
        // pairs with the `Release` clear below so a (buggy) second resumer observes the first's writes.
        if self.control.running.swap(true, Ordering::Acquire) {
            eprintln!("svm-fiber: fiber resumed while already running (single-owner violation)");
            std::process::abort();
        }
        self.control.value.set(val);
        let cptr = &*self.control as *const Control as u64;
        // ASan: leaving this (the resumer's) stack for the fiber's stack.
        #[cfg(feature = "asan")]
        // SAFETY: bracketing the jump below per the sanitizer fiber-annotation contract.
        unsafe {
            asan::__sanitizer_start_switch_fiber(
                self.control.asan.resumer_fake.as_ptr(),
                self.control.asan.stack_bottom,
                self.control.asan.stack_size,
            );
        }
        // SAFETY: `ctx` is the fiber's live suspended context (fresh from `make`, or refreshed below).
        let t = unsafe { jump(self.ctx, cptr) };
        // Control returned to this stack (the fiber suspended or completed): restore our fake stack.
        #[cfg(feature = "asan")]
        // SAFETY: the arrival half of the bracketing above.
        unsafe {
            asan::__sanitizer_finish_switch_fiber(
                self.control.asan.resumer_fake.get(),
                core::ptr::null_mut(),
                core::ptr::null_mut(),
            );
        }
        self.ctx = t.fctx; // the fiber's new suspended context
                           // Control is back on the resumer's stack: release the single-owner claim
                           // (the fiber's stack is no longer executing). `Release` publishes our writes
                           // to any subsequent claimant.
        self.control.running.store(false, Ordering::Release);
        if self.control.done.get() {
            self.done = true;
            State::Complete(self.control.value.get())
        } else {
            State::Yielded(self.control.value.get())
        }
    }
}

impl Drop for Fiber {
    fn drop(&mut self) {
        if !self.control.taken.get() {
            // The body never ran, so it still owns the boxed closure — reclaim it. (A fiber dropped
            // mid-suspend leaks the values live on its stack; scheduler fibers run to completion.)
            // SAFETY: `closure` is the untaken `Box<F>` and `drop_closure` its matching dropper.
            unsafe { (self.control.drop_closure)(self.control.closure.get()) }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::rc::Rc;

    #[test]
    fn yields_then_completes() {
        let mut f = Fiber::new(64 * 1024, |y, first| {
            let a = y.suspend(first + 1);
            let b = y.suspend(a + 1);
            b + 100
        })
        .unwrap();
        assert_eq!(f.resume(10), State::Yielded(11));
        assert_eq!(f.resume(20), State::Yielded(21));
        assert_eq!(f.resume(30), State::Complete(130));
        assert!(f.is_done());
    }

    #[test]
    #[should_panic(expected = "resumed a finished fiber")]
    fn resume_after_complete_panics() {
        let mut f = Fiber::new(64 * 1024, |_y, _| 0).unwrap();
        assert_eq!(f.resume(0), State::Complete(0));
        f.resume(0);
    }

    #[test]
    fn captures_environment() {
        // Body keeps a running `acc`, suspending it before each step; each resume adds 2.
        let mut f = Fiber::new(64 * 1024, move |y, _| {
            let mut acc = 0u64;
            for _ in 0..5 {
                acc += y.suspend(acc);
            }
            acc
        })
        .unwrap();
        let mut vals = Vec::new();
        loop {
            match f.resume(2) {
                State::Yielded(v) => vals.push(v),
                State::Complete(v) => {
                    vals.push(v);
                    break;
                }
            }
        }
        assert_eq!(vals, vec![0, 2, 4, 6, 8, 10]);
    }

    #[test]
    fn drop_before_start_runs_closure_drop() {
        // The closure captures an Rc; dropping the never-started fiber must drop it (refcount → 1).
        let marker = Rc::new(());
        let captured = Rc::clone(&marker);
        let f = Fiber::new(64 * 1024, move |_y, _| {
            let _hold = &captured;
            0
        })
        .unwrap();
        assert_eq!(Rc::strong_count(&marker), 2);
        drop(f);
        assert_eq!(Rc::strong_count(&marker), 1, "closure was not dropped");
    }

    #[test]
    fn many_fibers_keep_independent_state() {
        // Each fiber k folds the two values it is resumed with into k*1000 + a*10 + b; driving them
        // in interleaved rounds proves resume/suspend never cross fiber state.
        let mut fibers: Vec<Fiber> = (0..4)
            .map(|k| {
                Fiber::new(64 * 1024, move |y, _| {
                    let a = y.suspend(0);
                    let b = y.suspend(0);
                    (k as u64) * 1000 + a * 10 + b
                })
                .unwrap()
            })
            .collect();
        // Round 1: start each (Yielded 0).
        for f in fibers.iter_mut() {
            assert_eq!(f.resume(0), State::Yielded(0));
        }
        // Round 2: feed `a = k + 1` (Yielded 0).
        for (k, f) in fibers.iter_mut().enumerate() {
            assert_eq!(f.resume(k as u64 + 1), State::Yielded(0));
        }
        // Round 3: feed `b = k + 5`; each must complete with its own folded value.
        for (k, f) in fibers.iter_mut().enumerate() {
            let k = k as u64;
            assert_eq!(
                f.resume(k + 5),
                State::Complete(k * 1000 + (k + 1) * 10 + (k + 5))
            );
        }
    }

    /// Fuzz-style property test for the per-ABI switch (`jump`/`make`): many fibers driven in
    /// **random resume orders** must each keep independent register/stack state across the interleaved
    /// switches. Each fiber `i` is a folding generator that suspends `k_i` times — yielding its running
    /// sum — then returns `sum + 99`; a parallel model tracks each fiber's expected sum and every
    /// resume is checked. A register- or stack-save bug in the hand-written switch corrupts a fiber
    /// resumed *after* others ran on top of it, surfacing here as a mismatched yield. Deterministic
    /// (seeded xorshift), so any failure replays exactly.
    #[test]
    fn random_resume_orders_keep_fibers_independent() {
        fn xorshift(s: &mut u64) -> u64 {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
            *s
        }
        let mut rng = 0x9E37_79B9_7F4A_7C15u64;
        for _trial in 0..300 {
            let n = 2 + (xorshift(&mut rng) % 5) as usize; // 2..=6 fibers
            let k: Vec<u64> = (0..n).map(|_| 1 + xorshift(&mut rng) % 8).collect();
            let mut fibers: Vec<Fiber> = k
                .iter()
                .map(|&ki| {
                    Fiber::new(64 * 1024, move |y, first| {
                        let mut a = first;
                        for _ in 0..ki {
                            a = a.wrapping_add(y.suspend(a));
                        }
                        a.wrapping_add(99)
                    })
                    .unwrap()
                })
                .collect();
            let mut sum = vec![0u64; n];
            let mut count = vec![0u64; n];
            let mut done = vec![false; n];
            let mut remaining = n;
            while remaining > 0 {
                // Pick a random not-yet-finished fiber.
                let mut i = (xorshift(&mut rng) % n as u64) as usize;
                while done[i] {
                    i = (i + 1) % n;
                }
                let v = xorshift(&mut rng);
                sum[i] = sum[i].wrapping_add(v);
                count[i] += 1;
                match fibers[i].resume(v) {
                    State::Yielded(got) => {
                        assert!(count[i] <= k[i], "fiber {i} yielded past its suspend count");
                        assert_eq!(
                            got, sum[i],
                            "fiber {i} yield mismatch (switch state corruption?)"
                        );
                    }
                    State::Complete(got) => {
                        assert_eq!(count[i], k[i] + 1, "fiber {i} completed early/late");
                        assert_eq!(got, sum[i].wrapping_add(99), "fiber {i} final mismatch");
                        done[i] = true;
                        remaining -= 1;
                    }
                }
            }
        }
    }
}
