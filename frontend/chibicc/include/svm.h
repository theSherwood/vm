// <svm.h> — the SVM capability surface for C: the low-level `__vm_*` builtins the frontend
// lowers directly to `cap.call`, exposed as a documented, discoverable header.
//
// DESIGN §13/§14: a guest holding an **AddressSpace** capability (the 5th powerbox handle the
// runtime grants `_start`) can **mint its own shareable regions** and `map` them anywhere in its
// window — including at two adjacent offsets, so a wrap-around access becomes one contiguous access
// (the *magic ring buffer*). A minted region is real shared memory under the JIT (the embedder
// installs the OS-shared-memory factory), so it can also be **granted into a child domain** for
// zero-copy parent↔child data exchange. There is no ambient authority: minting consumes the
// memory-management capability the host chose to grant.
//
// This shadows nothing in the system include path; it is sandbox-only. A native `cc` build of the
// same source has no `__vm_*` symbols, so guard SVM-specific code accordingly if it must also build
// natively.
#ifndef __SVM_H
#define __SVM_H

// --- §7 capability handles & arbitrary host capabilities (late binding) --------------------
//
// `__vm_cap(i)` returns the i-th powerbox capability **handle** the runtime granted `_start`.
// Pass it as the first argument of a capability call. The indices:
#define VM_CAP_STDOUT 0
#define VM_CAP_STDIN 1
#define VM_CAP_EXIT 2
#define VM_CAP_MEMORY 3
#define VM_CAP_ADDRSPACE 4
#define VM_CAP_IORING 5
#define VM_CAP_BLOCKING 6
#define VM_CAP_JIT 7
int __vm_cap(int i);
//
// **Using an arbitrary host capability** (§7 late binding): declare it as a plain `extern` whose
// FIRST parameter is the capability handle (an `int`), then call it. The frontend lowers any call
// to an undefined `extern` (that isn't one of the builtins in this header) to a named import; the
// host binds the name to a concrete interface operation at load (see `default_cap_resolver`). So a
// new capability needs no frontend change — just an `extern` and a host that knows the name. E.g.:
//
//     extern long now(int clock_handle, int clock_id);   // host maps "now" -> (Clock, op 0)
//     long t = now(__vm_cap(/* a granted Clock handle */), 0);
//
// An unknown name is a clean load error (fail-closed), never a silent no-op.

// --- §13/§14 SharedRegion (guest-minted, shareable, multi-offset-mappable) -----------------
//
// Mint a fresh zero-filled region of `len` bytes from the AddressSpace capability; returns a region
// handle (>= 0) or a negative errno (e.g. -EINVAL for a non-positive or over-cap length).
long __vm_region_create(long len);

// Map region bytes `[region_off, region_off+len)` into the window at `[win_off, win_off+len)` with
// `prot` (READ|WRITE = 3). Map the same region at two adjacent `win_off`s to alias it twice — the
// ring-buffer layout. `win_off`/`region_off`/`len` must be region-granule-aligned (see below).
// Returns 0 or a negative errno.
long __vm_region_map(int region, long win_off, long region_off, long len, int prot);

// Unmap `[win_off, win_off+len)` (the window pages revert to ordinary, un-aliased backing).
long __vm_region_unmap(int region, long win_off, long len);

// The region's map granularity — the alignment unit for `__vm_region_map` (host MMU page /
// allocation granularity: 4 KiB or 16 KiB on unix, 64 KiB on Windows). Query it; never assume.
long __vm_region_page_size(int region);

// --- Guest-driven JIT (iface 11, JIT.md Model A) --------------------------------------------
//
// Submit serialized SVM IR (the binary `svm-encode` format) built at runtime in this window;
// the host validates it (decode + verify + the memory-match precondition: the blob must declare
// the same `memory` as this module, no data segments, no fibers/threads) and compiles it into
// THIS domain — same window, same powerbox. Returns a code handle (> 0) or a negative errno
// (-22 invalid, -12 compile quota exhausted). Fail-closed: on any error nothing is installed.
long __vm_jit_compile(void *blob, long len);

// Call a compiled unit's entry, which must be exactly the **raw** shape `(i64, i64) -> (i64)`
// (no C frame — the args go straight to the entry's block params; the strict-arity MVP shape,
// anything else faults). The unit reads/writes this window directly (zero copy). A trap inside
// the unit is terminal for the whole domain (§5 detect-and-kill). Use this to accelerate a hot
// loop your own code drives; to expose a unit as a *callable C function*, install it instead.
long __vm_jit_invoke2(long code, long a, long b);

// Revoke a code handle (the code itself is not reclaimed yet). Returns 0, or -22 if the handle
// is forged/already released (non-fatal).
long __vm_jit_release(long code);

// Install a compiled unit into the `call_indirect` table (Model B2), returning its slot index
// — a funcref old code (or another unit) can call indirectly at native speed (old→new). Cast
// the slot to a function pointer and call it like any C function. Returns -28 (ENOSPC) if the
// table is full (the embedder sizes the reservation; the CLI reserves 1024 slots).
//
// **Calling convention for an installed unit.** A unit reached through a C function pointer
// `T (*fp)(A, B)` must follow the **guest ABI**: this frontend threads the data-stack pointer
// as every function's hidden *first* parameter, so the unit's entry signature must be
// `(i64 sp, <A>, <B>) -> <T>` — the leaf body simply ignores `sp`. (A unit shaped for
// `__vm_jit_invoke2` instead omits `sp`; calling that one via a pointer is a clean
// `IndirectCallType` trap, never an escape.)
long __vm_jit_install(long code);

// Reclaim an installed slot (Model B2): clear it so the index is reusable by a later install
// and a stale call of it traps. Returns 0, or -22 for an out-of-range / not-installed slot.
// (Reclaims the table *slot*; the unit's code memory is not freed.)
long __vm_jit_uninstall(long slot);

#endif // __SVM_H
