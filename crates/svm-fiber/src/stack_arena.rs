//! Arena-allocated control stacks — the **default** control-stack backend (unix + x86-64 Windows);
//! opt out to the guard-paged backend with `guard-page-stacks`.
//!
//! Instead of one guarded reservation per fiber (`stack_unix.rs` = `mmap`+`mprotect`, 2 VMAs/fiber;
//! `stack_windows.rs` = `VirtualAlloc`+`VirtualProtect`), this sub-allocates fixed-size 256 KiB stack
//! slots from a few large reservations with a free-list, so `cont.new`/finish become a free-list
//! pop/push instead of syscalls and a fiber costs ~0 extra VMAs (unix) — lifting concurrency past the
//! `vm.max_map_count` VMA wall. It **drops the hardware guard page**; overflow protection is the
//! always-on JIT software stack-limit check (`emit_stack_check`), which is why that check lives in the
//! always-on escape-TCB path (it is the *sole* overflow defense here). A reclaimed slot is not zeroed
//! (zeroing would defeat the alloc win); the conservative GC scan skips the unused region below each
//! running fiber's live SP (`fiber_rt::gc_roots`), so stale slot bytes are not read as roots.
//!
//! **Commit-on-demand.** An arena is only *reserved* (address space, no backing) up front; a slot is
//! *committed* the first time it is handed out and stays committed when freed (so recycling is a
//! free-list pop, never a re-commit). On unix the reservation is `mmap`+`MAP_NORESERVE` (lazy) and the
//! per-slot commit is a no-op — physical pages fault in on first touch. On Windows there is no
//! `MAP_NORESERVE` analogue, so the reservation is `MEM_RESERVE` (no commit/pagefile charge) and each
//! slot is `MEM_COMMIT`ted on first hand-out. Either way the *commit* footprint tracks the peak
//! concurrent fiber count, not the reserved arena size — so reserving a large arena is free.

use std::sync::Mutex;

/// Per-stack slot size. Must be `>=` every requested control-stack size — both `svm-jit`'s
/// `FIBER_STACK` and `CORO_STACK` are `1 << 18`.
const SLOT: usize = 1 << 18; // 256 KiB
/// One arena reservation; carved into `ARENA / SLOT` slots. 1 GiB ⇒ 4096 slots/arena. Only reserved
/// (address space), so a big arena is cheap; slots are committed on demand.
const ARENA: usize = 1 << 30;
const SLOTS_PER_ARENA: usize = ARENA / SLOT;

/// Reserve one arena (`ARENA + SLOT` bytes so the usable base can be rounded up to a SLOT boundary),
/// **address space only** — no commit/backing. Returns the raw base, or null on failure. Never aborts;
/// the caller turns null into a recoverable `FiberFault`.
#[cfg(unix)]
unsafe fn reserve_arena() -> *mut u8 {
    // `MAP_NORESERVE`: no swap/commit reservation; physical pages fault in on first touch.
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
    use windows_sys::Win32::System::Memory::{VirtualAlloc, MEM_RESERVE, PAGE_READWRITE};
    // Reserve address space only (no `MEM_COMMIT` ⇒ no pagefile charge for the whole arena); each slot
    // is committed on first hand-out by `commit_slot`.
    VirtualAlloc(core::ptr::null(), ARENA + SLOT, MEM_RESERVE, PAGE_READWRITE) as *mut u8
}

/// Commit one freshly-carved `SLOT`-sized, SLOT-aligned slot within a reserved arena before it is
/// handed out. Returns `true` on success. On unix the reservation is already usable, so this is a
/// no-op; on Windows it `MEM_COMMIT`s the slot's pages. `false` (Windows commit failure) is a
/// recoverable `FiberFault`, never an abort.
#[cfg(unix)]
unsafe fn commit_slot(_slot: *mut u8) -> bool {
    true
}

#[cfg(windows)]
unsafe fn commit_slot(slot: *mut u8) -> bool {
    use windows_sys::Win32::System::Memory::{VirtualAlloc, MEM_COMMIT, PAGE_READWRITE};
    // Committing a sub-range of a reserved region is valid; `slot` is page-aligned (SLOT ≫ page).
    !VirtualAlloc(
        slot as *const core::ffi::c_void,
        SLOT,
        MEM_COMMIT,
        PAGE_READWRITE,
    )
    .is_null()
}

struct ArenaState {
    /// Committed, freed slots ready for immediate reuse (a free-list pop, no re-commit).
    free: Vec<*mut u8>,
    /// The current arena being carved (`null` if none yet), and the index of its next *uncommitted*
    /// slot. When `next == SLOTS_PER_ARENA` the arena is spent and a fresh one is reserved.
    cur_base: *mut u8,
    cur_next: usize,
    /// Every reserved arena's raw base, kept for the process life (never released — a prototype).
    arenas: Vec<*mut u8>,
}
// SAFETY: the pointers are arena/slot bases owned by this global allocator; only ever produced and
// consumed under the `STATE` mutex, and a `Stack` (which is `Send`) is the only thing that escapes.
unsafe impl Send for ArenaState {}

static STATE: Mutex<ArenaState> = Mutex::new(ArenaState {
    free: Vec::new(),
    cur_base: core::ptr::null_mut(),
    cur_next: 0,
    arenas: Vec::new(),
});

/// Hand out a slot: reuse a freed (already-committed) one, else carve + commit the next slot from the
/// current arena, reserving a fresh arena first when the current one is spent. `None` if the OS
/// refuses a reservation or commit (preserves the recoverable-`FiberFault` contract — never abort).
fn alloc_slot() -> Option<*mut u8> {
    let mut st = STATE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(p) = st.free.pop() {
        return Some(p);
    }
    // Need a fresh slot. Reserve a new arena if the current one is exhausted (or none yet).
    if st.cur_base.is_null() || st.cur_next >= SLOTS_PER_ARENA {
        // SAFETY: a fresh address-space reservation; null-checked, recorded.
        let raw = unsafe { reserve_arena() };
        if raw.is_null() {
            return None;
        }
        st.arenas.push(raw);
        // Round the usable base up to a SLOT boundary; the extra SLOT guarantees `[base, base+ARENA)`
        // fits within the reservation, so every slot is SLOT-aligned and in-bounds.
        st.cur_base = ((raw as usize + SLOT - 1) & !(SLOT - 1)) as *mut u8;
        st.cur_next = 0;
    }
    // SAFETY: `cur_next < SLOTS_PER_ARENA`, so this slot is within `[cur_base, cur_base + ARENA)`.
    let slot = unsafe { st.cur_base.add(st.cur_next * SLOT) };
    st.cur_next += 1;
    // SAFETY: `slot` is a page-aligned SLOT-sized sub-range of the reserved arena.
    if !unsafe { commit_slot(slot) } {
        return None; // Windows commit refused — recoverable FiberFault (the slot index is simply skipped).
    }
    Some(slot)
}

/// Return a slot to the free-list (kept committed — reuse is a pop, not a re-commit).
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

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::collections::HashSet;

    /// Allocate more than one arena's worth of slots, holding them so none free — this drives the
    /// cursor past `SLOTS_PER_ARENA` and forces a **second arena reservation**, the new commit-on-demand
    /// path the Fiber-level tests never reach (they don't create 4096+ concurrent fibers). Every slot
    /// must be non-null, `SLOT`-aligned, distinct (no overlap across the arena boundary), and **usable
    /// memory** (writing its top exercises the per-slot commit). Unix only: `MAP_NORESERVE` keeps the
    /// ~1 GiB reservation VA-only/lazy (on Windows this would `MEM_COMMIT` it).
    #[test]
    fn crosses_arena_boundary_with_usable_slots() {
        let n = SLOTS_PER_ARENA + 8; // > one arena ⇒ a second `reserve_arena`
        let mut stacks = Vec::with_capacity(n);
        let mut bases = HashSet::new();
        for _ in 0..n {
            let s = Stack::new(SLOT).expect("arena slot");
            let base = s.base as usize;
            assert!(!s.base.is_null());
            assert_eq!(base % SLOT, 0, "slot base must be SLOT-aligned");
            assert!(base.checked_add(SLOT).is_some(), "slot must not wrap");
            assert!(
                bases.insert(base),
                "slots must be distinct across the arena boundary"
            );
            // The slot is committed usable memory: writing near its top must not fault.
            // SAFETY: `[base, base+SLOT)` is this slot's own committed region; `base+SLOT-8` is in it.
            unsafe {
                s.base
                    .add(SLOT - 8)
                    .cast::<u64>()
                    .write_volatile(base as u64)
            };
            stacks.push(s);
        }
    }
}
