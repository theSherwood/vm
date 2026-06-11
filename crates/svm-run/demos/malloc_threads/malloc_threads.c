/* Concurrent `malloc` from multiple vCPUs — exercising the **thread-safe** guest allocator
 * (`include/stdlib.h`). The MVP allocator is a bump pointer over the map-growing heap; making it
 * thread-safe (a lock-free atomic-bump fast path + a spinlock around the rare page-growth) means
 * worker threads can allocate concurrently without the heap corrupting — previously the demos had to
 * pre-allocate on the main thread to sidestep it (D56/§12 note).
 *
 * `NWORKERS` threads each `malloc` `NALLOC` blocks and fill every byte with a pattern unique to
 * `(worker, block, offset)`. After they join, main re-checks every byte of every block: if any two
 * allocations had **overlapped** (the race the old allocator allowed), one fill would have clobbered
 * the other and the re-check would find a wrong byte. The program prints the number of corrupt blocks
 * — **0** on a correct allocator, on both the interpreter (the M:N oracle) and the JIT (real OS
 * threads). Everything is guest code over the VM's primitives; nothing here is a VM concern. */
#include <pthread.h>
#include <stdlib.h>

int write(int fd, char *buf, long n);

#define NWORKERS 4
#define NALLOC 64  /* allocations per worker */
#define BLKSZ 200  /* bytes per allocation */

static char *g_blocks[NWORKERS * NALLOC]; /* each worker writes its own disjoint slice */

/* A byte pattern unique to (worker, block, offset) — distinct across blocks so an overlap shows.
 * Kept in 0..127 (one printable-ish range) so the check is a plain low-bit `char` compare. */
static char pat(long w, int k, int i) {
  return (char)((w * 31 + k * 7 + i) & 0x7F);
}

static void *worker(void *arg) {
  long me = (long)arg;
  for (int k = 0; k < NALLOC; k++) {
    char *b = (char *)malloc(BLKSZ);
    g_blocks[me * NALLOC + k] = b; /* disjoint index per (me,k); read by main after join */
    if (!b)
      continue;
    for (int i = 0; i < BLKSZ; i++)
      b[i] = pat(me, k, i);
  }
  return 0;
}

static void print_long(long v) {
  char buf[24];
  int n = 0;
  if (v == 0) buf[n++] = '0';
  char tmp[24];
  int t = 0;
  while (v > 0) { tmp[t++] = (char)('0' + (v % 10)); v /= 10; }
  while (t > 0) buf[n++] = tmp[--t];
  buf[n++] = '\n';
  write(1, buf, n);
}

int main(void) {
  pthread_t workers[NWORKERS];
  for (int i = 0; i < NWORKERS; i++)
    pthread_create(&workers[i], 0, worker, (void *)(long)i);
  for (int i = 0; i < NWORKERS; i++)
    pthread_join(workers[i], 0);

  /* Verify every byte of every block — a clobber from an overlapping allocation would show here. */
  long bad = 0;
  for (long me = 0; me < NWORKERS; me++)
    for (int k = 0; k < NALLOC; k++) {
      char *b = g_blocks[me * NALLOC + k];
      if (!b) { bad++; continue; }
      for (int i = 0; i < BLKSZ; i++)
        if (b[i] != pat(me, k, i)) { bad++; break; }
    }
  print_long(bad); /* 0 = no corruption: every concurrent allocation was disjoint */
  return 0;
}
