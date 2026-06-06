/* A tiny guest `malloc`/`free`/`calloc` that **grows the window** via the Memory capability
 * (§3e/§4) — the §3d "malloc as guest C over `map`" design. The heap lives at a fixed high base
 * in the reserved tail (above any backed prefix); `malloc` bumps a break pointer and commits
 * fresh pages on demand with `__vm_map` (the frontend builtin → `cap.call` on the granted Memory
 * handle). `free` is a no-op (a bump allocator — the MVP, no reclamation). Sandbox build only
 * (`__chibicc__`); a native `cc` build uses the real libc instead.
 *
 * Fresh window pages are zero-filled by `map` and the bump allocator never reuses a byte, so the
 * memory `malloc` hands back is already zero (hence `calloc` == `malloc`). */
#ifndef VM_MALLOC_H
#define VM_MALLOC_H

long __vm_map(long off, long len, int prot); /* commit [off,off+len) RW (prot 3); 0 / -errno */

#define VM_HEAP_BASE 268435456L /* 256 MiB: above the (<= 64 MiB) backed prefix, in the tail */
#define VM_PAGE 4096L

static long __vm_brk = VM_HEAP_BASE;       /* next free byte */
static long __vm_committed = VM_HEAP_BASE; /* first byte past the committed region */

static void *malloc(unsigned long n) {
  n = (n + 15UL) & ~15UL; /* 16-byte align the request */
  long p = __vm_brk;
  long end = p + (long)n;
  if (end > __vm_committed) {
    /* Grow: commit whole pages covering the shortfall, RW (READ|WRITE = 3). */
    long need = (end - __vm_committed + (VM_PAGE - 1)) & ~(VM_PAGE - 1);
    if (__vm_map(__vm_committed, need, 3) != 0)
      return 0; /* out of memory */
    __vm_committed += need;
  }
  __vm_brk = end;
  return (void *)p;
}

static void free(void *p) { (void)p; } /* bump allocator: no reclamation (MVP, §3d) */

static void *calloc(unsigned long n, unsigned long sz) {
  return malloc(n * sz); /* fresh pages are already zeroed */
}

#endif /* VM_MALLOC_H */
