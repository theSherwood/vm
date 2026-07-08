/* Guest `malloc`/`free`/`realloc` for the T-library / package builds: `ltests.c`'s tracking
 * allocator (`debug_realloc`) sits on plain libc malloc/free, so the guest provides them.
 *
 * A **coalescing explicit-free-list allocator** (dlmalloc-lite) over a static arena, deterministic
 * and OS-free. The earlier power-of-two-size-class version fragmented badly — a freed block only
 * served a same-class request, so the bump cursor climbed to ~3x the true live set and the whole
 * `all.lua` suite in one `lua_State` (peak ~19 MiB live) exhausted the arena. Coalescing keeps the
 * high-water near the true peak, so the suite fits the reference JIT's window.
 *
 * Layout, all 16-byte aligned: each block is `[HDR 16 | payload | FTR 16]` where HDR/FTR both hold
 * `(block_size | free_bit)` (boundary tags for O(1) neighbour coalescing). A *free* block also
 * threads `fd`/`bk` doubly-linked-list pointers through the first 16 bytes of its payload. Malloc is
 * first-fit over the free list with block splitting; free coalesces with the physical prev/next
 * block via the boundary tags. Fresh memory is carved by bumping `top` toward `end`. */
#include <stddef.h>

#define ARENA_BYTES (56UL * 1024 * 1024)
static char arena[ARENA_BYTES] __attribute__((aligned(16)));

#define HDR 16u
#define FTR 16u
#define ALIGN 16u
#define FREE_BIT 1UL
/* smallest block: HDR + (fd,bk = 16) + FTR */
#define MIN_BLOCK (HDR + 16u + FTR)

static char *arena_base;  /* 16-aligned start */
static char *top;         /* first byte of never-yet-carved memory */
static char *arena_end;
static char *free_head;   /* explicit free list (unordered doubly-linked) */
static int initialized;

static unsigned long align_up(unsigned long n, unsigned long a) { return (n + a - 1) & ~(a - 1); }

/* header/footer accessors (block = pointer to HDR) */
static unsigned long blk_size(char *b) { return *(unsigned long *)b & ~FREE_BIT; }
static int blk_free(char *b) { return (int)(*(unsigned long *)b & FREE_BIT); }
static void set_tags(char *b, unsigned long size, int free) {
  unsigned long v = size | (free ? FREE_BIT : 0);
  *(unsigned long *)b = v;
  *(unsigned long *)(b + size - FTR) = v;
}
static char **fd_of(char *b) { return (char **)(b + HDR); }
static char **bk_of(char *b) { return (char **)(b + HDR + 8); }

static void fl_remove(char *b) {
  char *fd = *fd_of(b), *bk = *bk_of(b);
  if (bk) *fd_of(bk) = fd; else free_head = fd;
  if (fd) *bk_of(fd) = bk;
}
static void fl_push(char *b) {
  *fd_of(b) = free_head;
  *bk_of(b) = (void *)0;
  if (free_head) *bk_of(free_head) = b;
  free_head = b;
}

static void init(void) {
  /* Load the arena address through a `volatile` so the alignment arithmetic runs at *runtime*: on
   * the constant `&arena` it would fold to a constexpr `Add(PtrToInt(@arena), 15)`, a constant
   * shape the on-ramp doesn't lower. (`arena` is already `aligned(16)`, so this is a no-op in
   * practice, but the volatile keeps the fold from happening.) */
  char *volatile ap = arena;
  arena_base = (char *)(((unsigned long)ap + (ALIGN - 1)) & ~((unsigned long)ALIGN - 1));
  top = arena_base;
  arena_end = arena + ARENA_BYTES;
  free_head = (void *)0;
  initialized = 1;
}

void *malloc(unsigned long n) {
  if (!initialized) init();
  if (n == 0) n = 1;
  unsigned long payload = align_up(n, ALIGN);
  if (payload < 16) payload = 16; /* room for fd/bk when later freed */
  unsigned long need = HDR + payload + FTR;

  /* first-fit over the free list */
  for (char *b = free_head; b; b = *fd_of(b)) {
    unsigned long bs = blk_size(b);
    if (bs >= need) {
      fl_remove(b);
      if (bs - need >= MIN_BLOCK) { /* split: [need | rest] */
        set_tags(b, need, 0);
        char *rest = b + need;
        set_tags(rest, bs - need, 1);
        fl_push(rest);
      } else {
        set_tags(b, bs, 0);
      }
      return b + HDR;
    }
  }
  /* carve fresh memory */
  if (top + need > arena_end) return (void *)0;
  char *b = top;
  top += need;
  set_tags(b, need, 0);
  return b + HDR;
}

void free(void *p) {
  if (!p) return;
  char *b = (char *)p - HDR;
  unsigned long size = blk_size(b);
  /* coalesce with next physical block if free and within the carved region */
  char *next = b + size;
  if (next < top && blk_free(next)) {
    fl_remove(next);
    size += blk_size(next);
  }
  /* coalesce with previous physical block if free (read its footer just below our header) */
  if (b > arena_base) {
    unsigned long ptag = *(unsigned long *)(b - FTR);
    if (ptag & FREE_BIT) {
      unsigned long psize = ptag & ~FREE_BIT;
      char *prev = b - psize;
      fl_remove(prev);
      b = prev;
      size += psize;
    }
  }
  /* if this block is now the arena top, hand it back to the bump cursor (keeps the high-water tight) */
  if (b + size == top) {
    top = b;
    return;
  }
  set_tags(b, size, 1);
  fl_push(b);
}

void *realloc(void *p, unsigned long n) {
  if (!p) return malloc(n);
  if (n == 0) {
    free(p);
    return (void *)0;
  }
  char *b = (char *)p - HDR;
  unsigned long cur_payload = blk_size(b) - HDR - FTR;
  if (n <= cur_payload) return p; /* shrink/fit in place */
  void *np = malloc(n);
  if (!np) return (void *)0;
  for (unsigned long i = 0; i < cur_payload; i++) ((char *)np)[i] = ((char *)p)[i];
  free(p);
  return np;
}

void *calloc(unsigned long nm, unsigned long sz) {
  unsigned long n = nm * sz;
  char *p = (char *)malloc(n);
  if (p)
    for (unsigned long i = 0; i < n; i++) p[i] = 0;
  return p;
}
