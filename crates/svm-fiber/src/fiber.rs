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
    /// Whether the entry closure has been moved into the running body yet (for drop-before-start).
    taken: Cell<bool>,
    /// The entry closure, type-erased as `*mut F`; reclaimed by the body (`take`) or by `Drop`.
    closure: Cell<*mut ()>,
    /// Monomorphized dropper for `closure`, used only if the fiber is dropped before it ever ran.
    drop_closure: unsafe fn(*mut ()),
}

/// Handed to a fiber body so it can suspend back to whoever resumed it.
pub struct Yielder {
    control: *const Control,
}

impl Yielder {
    /// Suspend the fiber, handing `val` back to the resumer; returns the value passed to the next
    /// [`Fiber::resume`].
    pub fn suspend(&self, val: u64) -> u64 {
        // SAFETY: `control` outlives the body (the `Fiber` owns the `Box<Control>` and cannot be
        // dropped while suspended inside its own body). The resumer is parked, so we have exclusive
        // access.
        let c = unsafe { &*self.control };
        c.value.set(val);
        // SAFETY: `resumer` is the live context that switched into us.
        let t = unsafe { jump(c.resumer.get(), 0) };
        c.resumer.set(t.fctx); // the resumer may be a different context next time
        c.value.get()
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
    // SAFETY: final switch back to the resumer; this context is never resumed again.
    unsafe { jump(c.resumer.get(), 0) };
    unreachable!("fiber resumed after completion")
}

impl Fiber {
    /// Create a fiber with a `stack_size`-byte (rounded up) guard-paged control stack. The body
    /// receives a [`Yielder`] and the first resume value, and returns the fiber's final value.
    pub fn new<F>(stack_size: usize, f: F) -> Fiber
    where
        F: FnOnce(&Yielder, u64) -> u64 + 'static,
    {
        let stack = Stack::new(stack_size);
        let control = Box::new(Control {
            value: Cell::new(0),
            resumer: Cell::new(std::ptr::null_mut()),
            done: Cell::new(false),
            taken: Cell::new(false),
            closure: Cell::new(Box::into_raw(Box::new(f)) as *mut ()),
            drop_closure: drop_closure_impl::<F>,
        });
        // SAFETY: fresh, live stack; `fiber_entry::<F>` matches the boxed closure's type.
        let ctx = unsafe { make(&stack, fiber_entry::<F>) };
        Fiber {
            _stack: stack,
            ctx,
            done: false,
            control,
        }
    }

    /// Whether the fiber has finished (returned). A finished fiber must not be resumed.
    pub fn is_done(&self) -> bool {
        self.done
    }

    /// Resume the fiber, passing `val`; returns whether it yielded or completed.
    ///
    /// # Panics
    /// Panics if the fiber has already completed.
    pub fn resume(&mut self, val: u64) -> State {
        assert!(!self.done, "resumed a finished fiber");
        self.control.value.set(val);
        let cptr = &*self.control as *const Control as u64;
        // SAFETY: `ctx` is the fiber's live suspended context (fresh from `make`, or refreshed below).
        let t = unsafe { jump(self.ctx, cptr) };
        self.ctx = t.fctx; // the fiber's new suspended context
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
        });
        assert_eq!(f.resume(10), State::Yielded(11));
        assert_eq!(f.resume(20), State::Yielded(21));
        assert_eq!(f.resume(30), State::Complete(130));
        assert!(f.is_done());
    }

    #[test]
    #[should_panic(expected = "resumed a finished fiber")]
    fn resume_after_complete_panics() {
        let mut f = Fiber::new(64 * 1024, |_y, _| 0);
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
        });
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
        });
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
}
