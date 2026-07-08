//! Arena-allocated control stacks — PROTOTYPE, feature `arena-stacks` (unix + x86-64 Windows).
//!
//! Instead of one guarded reservation per fiber (`stack_unix.rs` = `mmap`+`mprotect`, 2 VMAs/fiber;
//! `stack_windows.rs` = `VirtualAlloc`+`VirtualProtect`), this sub-allocates fixed-size 256 KiB stack
//! slots from a few large reservations with a free-list, so `cont.new`/finish become a free-list
//! pop/push instead of syscalls and a fiber costs ~0 extra VMAs (unix). Overflow protection moves to
//! the JIT's software stack-limit check (`svm-jit` feature `stack-check`), so this **drops the
//! hardware guard page** and must not ship without that check. A reclaimed slot is not zeroed
//! (zeroing would defeat the alloc win); a conservative GC scan over a reused slot over-approximates
//! roots (sound superset — noted).
//!
//! Windows note: there is no `MAP_NORESERVE` analogue, so each arena is `MEM_COMMIT`ted up front (a
//! pagefile *commit* charge; physical pages stay demand-zero, like unix). Per-slot commit-on-demand
//! is a follow-up; for the prototype benchmark this is the simplest correct shape.

use std::sync::Mutex;

/// Per-stack slot size. Must be `>=` every requested control-stack size — both `svm-jit`'s
/// `FIBER_STACK` and `CORO_STACK` are `1 << 18`.
const SLOT: usize = 1 << 18; // 256 KiB
/// One arena reservation; carved into `ARENA / SLOT` slots. 1 GiB ⇒ 4096 slots/arena.
const ARENA: usize = 1 << 30;
const SLOTS_PER_ARENA: usize = ARENA / SLOT;

/// Reserve one arena (`ARENA + SLOT` bytes so the usable base can be rounded up to a SLOT boundary)
/// and return its raw base, or null on failure. Never aborts — the caller turns null into a
/// recoverable `FiberFault`.
#[cfg(unix)]
unsafe fn reserve_arena() -> *mut u8 {
    // Lazy anonymous reservation (`MAP_NORESERVE`): committed physical pages cost nothing until touched.
    let p = libc::mmap(
        core::ptr::null_mut(),
        ARENA + SLOT,
        libc::PROT_READ | libc::PROT_WRITE,
        libc::MAP_PRIVATE | libc::MAP_ANON | libc::MAP_NORESERVE,
        -1,
        0,
    );
    if p == libc::MAP_FAILED {
        core::ptr::null_mut()
    } else {
        p as *mut u8
    }
}

#[cfg(windows)]
unsafe fn reserve_arena() -> *mut u8 {
    use windows_sys::Win32::System::Memory::{
        VirtualAlloc, MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE,
    };
    // Reserve **and** commit (no `MAP_NORESERVE` equivalent): the commit is a pagefile charge, but
    // physical pages remain demand-zero until touched. `PAGE_READWRITE`, no per-slot guard page.
    VirtualAlloc(
        core::ptr::null(),
        ARENA + SLOT,
        MEM_RESERVE | MEM_COMMIT,
        PAGE_READWRITE,
    ) as *mut u8
}

struct ArenaState {
    /// Free slot base pointers, ready to hand out.
    free: Vec<*mut u8>,
    /// Arena base pointers, kept reserved for the life of the process (never released — a prototype).
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
    // SAFETY: a fresh reservation; null-checked, owned by `STATE.arenas`.
    let raw = unsafe { reserve_arena() };
    if raw.is_null() {
        return None;
    }
    // Record the raw reservation (never released — a prototype).
    st.arenas.push(raw);
    // Round the usable base up to a SLOT boundary; the extra SLOT guarantees `[base, base+ARENA)` fits.
    let base = ((raw as usize + SLOT - 1) & !(SLOT - 1)) as *mut u8;
    // Hand out slot 0 now; push the rest onto the free-list. Every slot is SLOT-aligned.
    for i in 1..SLOTS_PER_ARENA {
        // SAFETY: `i * SLOT < ARENA`, within the (over-)reservation from the aligned base.
        st.free.push(unsafe { base.add(i * SLOT) });
    }
    Some(base)
}

/// Return a slot to the free-list (not released — slots are recycled).
fn free_slot(p: *mut u8) {
    STATE.lock().unwrap_or_else(|e| e.into_inner()).free.push(p);
}

/// An arena-allocated control stack slot — same surface as `stack_unix`/`stack_windows`, minus the
/// guard page.
pub struct Stack {
    base: *mut u8,
}

// SAFETY: as in the guarded backends — an owned slot (a pointer); the bytes are only touched by
// whichever thread is currently running on it, and the slot returns to the free-list on drop.
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

    /// The top of the stack (highest address, exclusive) — passed to `make`; also TEB `StackBase`.
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

    /// TEB `StackLimit` — the lowest usable address (no guard page, so the slot base).
    #[cfg(windows)]
    pub fn limit_ptr(&self) -> *mut u8 {
        self.base
    }

    /// TEB `DeallocationStack` — the slot base (there is no separate reservation base per slot).
    #[cfg(windows)]
    pub fn base_ptr(&self) -> *mut u8 {
        self.base
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
