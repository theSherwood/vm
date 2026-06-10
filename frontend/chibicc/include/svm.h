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

#endif // __SVM_H
