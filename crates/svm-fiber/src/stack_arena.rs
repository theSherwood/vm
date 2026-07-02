//! Arena-allocated control stacks — PROTOTYPE, feature `arena-stacks`, unix only.
//!
//! Instead of one `mmap` + `mprotect` guard page per fiber (`stack_unix.rs`, 2 VMAs/fiber, which caps
//! concurrency at the `vm.max_map_count` wall — see the measurement in the design discussion), this
//! sub-allocates fixed-size 256 KiB stack slots from a few large `mmap`'d arenas with a free-list. A
//! fiber then costs ~0 extra VMAs (arenas are contiguous same-prot mappings, so the kernel coalesces
//! them), and `cont.new`/finish become a free-list pop/push instead of two syscalls.
//!
//! **This drops the hardware overflow guard.** In a real deployment, overflow protection moves to a
//! software stack-limit check the JIT emits in each prologue (`svm-jit` feature `stack-check`); this
//! file exists to *benchmark the allocation cost* of the arena scheme in isolation, and must not ship
//! on its own. A reclaimed slot is NOT zeroed (zeroing 256 KiB/alloc would defeat the very cost we are
//! measuring); a conservative GC stack scan over a reused slot therefore over-approximates roots
//! (sound — a superset — but noted).

use std::sync::Mutex;

/// Per-stack slot size. Must be `>=` every requested control-stack size — both `svm-jit`'s
/// `FIBER_STACK` and `CORO_STACK` are `1 << 18`.
const SLOT: usize = 1 << 18; // 256 KiB
/// One arena reservation; carved into `ARENA / SLOT` slots. 1 GiB ⇒ 4096 slots/arena.
const ARENA: usize = 1 << 30;
const SLOTS_PER_ARENA: usize = ARENA / SLOT;

struct ArenaState {
    /// Free slot base pointers, ready to hand out.
    free: Vec<*mut u8>,
    /// Arena base pointers, kept mapped for the life of the process (never unmapped — a prototype).
    arenas: Vec<*mut u8>,
}
// SAFETY: the pointers are arena/slot bases owned by this global allocator; only ever produced and
// consumed under the `STATE` mutex, and a `Stack` (which is `Send`) is the only thing that escapes.
unsafe impl Send for ArenaState {}

static STATE: Mutex<ArenaState> = Mutex::new(ArenaState {
    free: Vec::new(),
    arenas: Vec::new(),
});

/// Pop a free slot, carving a fresh arena when the free-list is empty. `None` if the OS refuses a new
/// arena (preserves the recoverable-`FiberFault` contract — never abort).
fn alloc_slot() -> Option<*mut u8> {
    let mut st = STATE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(p) = st.free.pop() {
        return Some(p);
    }
    // Over-allocate by one SLOT so the usable region can be rounded up to a **SLOT boundary**: with
    // SLOT-aligned slots, a slot's low bound is `sp & !(SLOT-1)` for any sp in it — which is how the
    // per-vCPU software stack-limit check derives the limit from SP alone (STACK_GUARD.md §2b path A),
    // with no per-thread cell. SAFETY: a fresh anonymous lazy reservation; checked for MAP_FAILED.
    let raw = unsafe {
        libc::mmap(
            core::ptr::null_mut(),
            ARENA + SLOT,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANON | libc::MAP_NORESERVE,
            -1,
            0,
        )
    };
    if raw == libc::MAP_FAILED {
        return None;
    }
    let raw = raw as *mut u8;
    // Record the raw reservation (never unmapped — a prototype).
    st.arenas.push(raw);
    // Round up to the next SLOT boundary; the extra SLOT guarantees `[base, base + ARENA)` fits.
    let base = ((raw as usize + SLOT - 1) & !(SLOT - 1)) as *mut u8;
    // Hand out slot 0 now; push the rest onto the free-list. Every slot is SLOT-aligned.
    for i in 1..SLOTS_PER_ARENA {
        // SAFETY: `i * SLOT < ARENA`, within the (over-)reservation from the aligned base.
        st.free.push(unsafe { base.add(i * SLOT) });
    }
    Some(base)
}

/// Return a slot to the free-list (not unmapped — slots are recycled).
fn free_slot(p: *mut u8) {
    STATE.lock().unwrap_or_else(|e| e.into_inner()).free.push(p);
}

/// An arena-allocated control stack slot — same surface as `stack_unix::Stack`, minus the guard page.
pub struct Stack {
    base: *mut u8,
}

// SAFETY: as in `stack_unix` — an owned slot (a pointer); the bytes are only touched by whichever
// thread is currently running on it, and the slot returns to the free-list on drop.
unsafe impl Send for Stack {}

impl Stack {
    /// Allocate a slot (rounded up to `SLOT`). `None` when arenas can't grow — a recoverable
    /// `FiberFault`, never an abort.
    pub fn new(size: usize) -> Option<Stack> {
        debug_assert!(size <= SLOT, "arena slot is {SLOT}; requested {size}");
        Some(Stack {
            base: alloc_slot()?,
        })
    }

    /// The top of the stack (highest address, exclusive) — passed to `make`.
    pub fn top(&self) -> *mut u8 {
        // SAFETY: one-past-the-end of our own slot.
        unsafe { self.base.add(SLOT) }
    }

    /// The lowest usable address — with no guard page, the slot base itself.
    pub fn usable_low(&self) -> *const u8 {
        self.base as *const u8
    }

    /// The usable region `[low, len)` for ASan fiber-switch annotations.
    #[cfg(feature = "asan")]
    pub fn usable(&self) -> (*const u8, usize) {
        (self.base as *const u8, SLOT)
    }

    /// The usable address range, for tests asserting a fiber runs on this stack.
    #[cfg(test)]
    pub fn usable_range(&self) -> (usize, usize) {
        (self.base as usize, self.base as usize + SLOT)
    }
}

impl Drop for Stack {
    fn drop(&mut self) {
        free_slot(self.base);
    }
}
