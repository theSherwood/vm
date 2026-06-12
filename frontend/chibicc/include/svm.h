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

// Call a compiled unit's entry, which must be exactly `(i64, i64) -> (i64)` (the strict-arity
// MVP shape; anything else faults). The unit reads/writes this window directly (zero copy). A
// trap inside the unit is terminal for the whole domain (§5 detect-and-kill).
long __vm_jit_invoke2(long code, long a, long b);

// Revoke a code handle (the code itself is not reclaimed yet). Returns 0, or -22 if the handle
// is forged/already released (non-fatal).
long __vm_jit_release(long code);

#endif // __SVM_H
