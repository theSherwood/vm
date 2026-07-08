/* Guest `malloc`/`free`/`realloc` for the T-library build: `ltests.c`'s tracking allocator
 * (`debug_realloc`) sits on plain libc malloc/free, so the guest provides them — the same
 * segregated power-of-two free lists over a static arena as the sweep harness's Lua allocator
 * (deterministic, OS-free; guest definitions shadow the on-ramp's bump-only `malloc`). Blocks carry
 * an 8-byte size-class header; freed blocks recycle onto their class list. */
#include <stddef.h>

static char arena[56 * 1024 * 1024]; /* one shared state runs all 14 files (all.lua model) — as much as fits the reference JIT window */
static unsigned long arena_off;
#define NCLASS 24
static void *freelist[NCLASS];

static int class_of(unsigned long n) {
  unsigned long want = n + 8;
  int c = 0;
  unsigned long sz = 16;
  while (sz < want) {
    sz <<= 1;
    c++;
  }
  return c;
}

void *malloc(unsigned long n) {
  int c = class_of(n);
  if (c >= NCLASS) return (void *)0;
  if (freelist[c]) {
    char *blk = (char *)freelist[c];
    freelist[c] = *(void **)blk;
    *(unsigned long *)blk = (unsigned long)c; /* the free-list link reused the header word */
    return blk + 8;
  }
  unsigned long sz = 16UL << c;
  unsigned long start = (arena_off + 15UL) & ~15UL;
  if (start + sz > sizeof arena) return (void *)0;
  char *blk = &arena[start];
  arena_off = start + sz;
  *(unsigned long *)blk = (unsigned long)c;
  return blk + 8;
}

void free(void *p) {
  if (!p) return;
  char *blk = (char *)p - 8;
  int c = (int)*(unsigned long *)blk;
  *(void **)blk = freelist[c];
  freelist[c] = blk;
}

void *realloc(void *p, unsigned long n) {
  if (!p) return malloc(n);
  if (n == 0) {
    free(p);
    return (void *)0;
  }
  {
    int c = (int)*(unsigned long *)((char *)p - 8);
    unsigned long cap = (16UL << c) - 8;
    if (n <= cap) return p;
    void *np = malloc(n);
    if (!np) return (void *)0;
    for (unsigned long i = 0; i < cap; i++) ((char *)np)[i] = ((char *)p)[i];
    free(p);
    return np;
  }
}

void *calloc(unsigned long nm, unsigned long sz) {
  unsigned long n = nm * sz;
  char *p = (char *)malloc(n);
  if (p)
    for (unsigned long i = 0; i < n; i++) p[i] = 0;
  return p;
}
