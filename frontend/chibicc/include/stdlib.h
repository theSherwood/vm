// Minimal <stdlib.h> for the SVM sandbox target (the whole-program guest libc, §3d).
//
// The headline piece is a real **`malloc`/`free`/`calloc`/`realloc`** built on the Memory
// capability (§3e/§4): the heap lives at a fixed high base in the *reserved tail* of the window
// and **grows on demand** by committing pages with `__vm_map` — the §1a sparse-address-space win,
// available to any program that just `#include <stdlib.h>` (no special prelude). `free` is a
// no-op (a bump allocator — the §3d MVP; no reclamation), with a per-allocation size header so
// `realloc` can copy. A native `cc` build of the same source uses the platform libc instead;
// this header shadows the system one only for the sandbox frontend (chibicc searches its bundled
// include dir first), so demos stay byte-identical to native.
//
// Deliberately small: the allocator, `exit`/`abort`, and the `EXIT_*`/`NULL`/`size_t` boilerplate
// real programs pull from <stdlib.h>. Anything else a program calls is a clean "undefined
// function" error (there is no libc to link — the whole program is the translation unit).
#ifndef __SVM_STDLIB_H
#define __SVM_STDLIB_H

#include <stddef.h> // size_t, NULL

#define EXIT_SUCCESS 0
#define EXIT_FAILURE 1

// `exit` is a powerbox builtin (§3e), intercepted by name; declaring it here is enough.
void exit(int code);

// The Memory-capability builtins (§3e/§4), lowered to `cap.call` on the granted Memory handle.
// `__vm_map` commits `[off, off+len)` with `prot` (READ|WRITE = 3), returning 0 or a negative errno.
// `__vm_page_size` returns the host MMU page granularity the window is managed in (the unit `map`
// rounds to), so the guest can align to the *real* page instead of assuming a fixed size.
long __vm_map(long off, long len, int prot);
long __vm_page_size(void);

static void abort(void) {
  exit(134); // 128 + SIGABRT, the conventional code
}

// --- the map-growing heap -------------------------------------------------------------------
#define __SVM_HEAP_BASE 268435456L // 256 MiB: above the (<= 64 MiB) backed prefix, in the tail
#define __SVM_HDR 16L // per-allocation header (holds the payload size; keeps 16-byte alignment)

static long __svm_brk = __SVM_HEAP_BASE;       // next free byte
static long __svm_committed = __SVM_HEAP_BASE; // first byte past the committed region
static long __svm_page = 0;                    // cached host page granularity (0 = not yet queried)

// Heap-growth granularity = the **host page**, queried once and cached. The runtime's `map` commits
// and zero-fills the whole host page(s) covering a request (host-page default: 4 KiB on x86-64,
// 16 KiB on Apple Silicon, …); growing in that unit means each growth covers fresh page(s) on any
// host (a smaller step could re-`map` — and so re-zero — a page already holding live allocations).
// `__SVM_HEAP_BASE` (256 MiB) is a multiple of every realistic page, so growth stays page-aligned.
static long __svm_pagesize(void) {
  if (__svm_page == 0) {
    long p = __vm_page_size();
    __svm_page = p > 0 ? p : 4096L; // defensive floor if the query is unavailable
  }
  return __svm_page;
}

static void *malloc(size_t n) {
  n = (n + 15UL) & ~15UL; // 16-byte align the payload
  long hdr = __svm_brk;
  long payload = hdr + __SVM_HDR;
  long end = payload + (long)n;
  if (end > __svm_committed) {
    // Grow: commit whole host pages covering the shortfall, read-write (READ|WRITE = 3).
    long pg = __svm_pagesize();
    long need = (end - __svm_committed + (pg - 1)) & ~(pg - 1);
    if (__vm_map(__svm_committed, need, 3) != 0)
      return NULL; // out of memory
    __svm_committed += need;
  }
  __svm_brk = end;
  *(size_t *)hdr = n; // remember the size for realloc
  return (void *)payload;
}

static void free(void *p) {
  (void)p; // bump allocator: no reclamation (MVP, §3d)
}

static void *calloc(size_t n, size_t sz) {
  // Fresh window pages are zero-filled by `map`, and the bump allocator never reuses a byte, so
  // the payload is already zero.
  return malloc(n * sz);
}

static void *realloc(void *p, size_t n) {
  if (!p)
    return malloc(n);
  size_t old = *(size_t *)((char *)p - __SVM_HDR);
  void *q = malloc(n);
  if (!q)
    return NULL;
  size_t c = old < n ? old : n;
  for (size_t i = 0; i < c; i++) // self-contained copy (no <string.h> dependency)
    ((char *)q)[i] = ((char *)p)[i];
  return q;
}

#endif // __SVM_STDLIB_H
