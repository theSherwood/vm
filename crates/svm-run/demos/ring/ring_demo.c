/* The **magic ring buffer** through the SharedRegion capability (MMAP_CAPABILITY.md §4e — the
 * two-window-offsets case of multi-mapping). The primitive: map ONE region of `CAP` bytes at TWO
 * adjacent window offsets `[base, base+CAP)` and `[base+CAP, base+2*CAP)`, both aliasing the same
 * physical pages. Then a byte written at `base+CAP+k` is the same byte as `base+k`, so a read or
 * write of a span that runs off the end of the ring **continues seamlessly at the start** — a single
 * `memcpy` handles the wrap with no split and no branch. This is exactly what `SharedRegion`'s
 * multi-offset aliasing was built for; here a guest reaches it through the granted fs capability's
 * zero-copy bridge (`FS_MAP_REGION` mints the region, `__vm_region_call` maps it — twice).
 *
 * Two builds from one file:
 *   - guest (`-DSVM_GUEST`): the region is minted over a file by the `host_fs_mmap` capability and
 *     double-mapped via `SharedRegion.map`;
 *   - native oracle: a `memfd` double-mapped with raw `mmap(MAP_SHARED|MAP_FIXED)` — the classic
 *     magic-ring-buffer setup.
 * Same driver, same operations → byte-identical stdout: the capability primitive matches the OS one.
 */

#ifndef SVM_GUEST
#define _GNU_SOURCE
#include <sys/mman.h>
#include <unistd.h>
#endif

extern int printf(const char *, ...);
extern void *memcpy(void *, const void *, unsigned long);
extern void *memset(void *, int, unsigned long);
extern void *malloc(unsigned long);

#define CAP 4096 /* ring capacity = one page (the SharedRegion map granularity) */
#define NMSG 200

#ifdef SVM_GUEST
/* ---- guest: a file-backed region, double-mapped through the fs capability -------------------- */
extern int __vm_cap_resolve(const char *name, long len);
extern long __vm_host_call(int h, int op, long a, long b, long c, long d);
extern long __vm_region_call(int h, int op, long a, long b, long c, long d);

enum { FS_OPEN = 0, FS_MAP_REGION = 13 };
enum { SR_MAP = 0 };
enum { CAP_O_READ = 1, CAP_O_WRITE = 2, CAP_O_CREATE = 16 };
enum { CAP_PROT_READ = 1, CAP_PROT_WRITE = 2 };

static long cstrlen(const char *s) {
  long n = 0;
  while (s[n]) n++;
  return n;
}

/* Reserve `2*CAP` page-aligned window bytes and alias the `CAP`-byte region over both halves. */
static char *ring_setup(unsigned cap) {
  int fs = __vm_cap_resolve("fs", 2);
  if (fs < 0) return 0;
  const char *path = "ring.dat";
  int fd = (int)__vm_host_call(fs, FS_OPEN, (long)path, cstrlen(path),
                               CAP_O_READ | CAP_O_WRITE | CAP_O_CREATE, 0);
  if (fd < 0) return 0;
  long region = __vm_host_call(fs, FS_MAP_REGION, fd, 0, cap, 0);
  if (region < 0) return 0;
  /* over-allocate by `cap` (a page) so we can round the base up to a page boundary. */
  char *raw = (char *)malloc((unsigned long)cap * 3);
  if (!raw) return 0;
  unsigned long a = ((unsigned long)raw + cap - 1) & ~((unsigned long)cap - 1);
  char *base = (char *)a;
  long prot = CAP_PROT_READ | CAP_PROT_WRITE;
  if (__vm_region_call((int)region, SR_MAP, (long)base, 0, cap, prot) != 0) return 0;
  if (__vm_region_call((int)region, SR_MAP, (long)(base + cap), 0, cap, prot) != 0) return 0;
  return base;
}
#else
/* ---- native oracle: a memfd double-mapped with raw mmap ------------------------------------- */
static char *ring_setup(unsigned cap) {
  int fd = memfd_create("ring", 0);
  if (fd < 0) return 0;
  if (ftruncate(fd, cap) != 0) return 0;
  /* reserve 2*cap, then replace each half with a shared view of the same fd at offset 0. */
  char *base = mmap(0, (unsigned long)cap * 2, PROT_NONE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
  if (base == MAP_FAILED) return 0;
  if (mmap(base, cap, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, fd, 0) == MAP_FAILED) return 0;
  if (mmap(base + cap, cap, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, fd, 0) == MAP_FAILED)
    return 0;
  return base;
}
#endif

int main(void) {
  char *base = ring_setup(CAP);
  if (!base) {
    printf("ring setup failed\n");
    return 1;
  }

  /* 1. Cross-boundary contiguous access: push then pop `NMSG` variable-length messages, each a
   *    SINGLE memcpy at `pos % CAP`. Whenever `pos % CAP + len > CAP` the copy runs off the end of
   *    the ring and continues at the start — correct only because the second mapping aliases the
   *    first. A wrong alias would corrupt the read-back and the checksum. */
  unsigned wpos = 0, rpos = 0;
  unsigned long long sum = 0;
  for (int i = 0; i < NMSG; i++) {
    int len = 20 + (i * 37) % 200; /* 20..219 bytes; sum ≫ CAP so it wraps many times */
    char msg[256];
    for (int j = 0; j < len; j++) msg[j] = (char)('A' + ((i * 7 + j * 3) % 26));
    memcpy(base + (wpos % CAP), msg, (unsigned long)len);
    wpos += len;
    char out[256];
    memcpy(out, base + (rpos % CAP), (unsigned long)len);
    rpos += len;
    for (int j = 0; j < len; j++) {
      if (out[j] != msg[j]) {
        printf("MISMATCH i=%d j=%d\n", i, j);
        return 1;
      }
      sum = sum * 131 + (unsigned char)out[j];
    }
  }
  printf("ring: %d messages, %u bytes, checksum %llu\n", NMSG, wpos, sum);

  /* 2. Explicit alias witness: a 12-byte write straddling the end reads back contiguously through
   *    the second mapping, and its 7-byte overflow is physically visible at the very start. */
  memset(base, 0, CAP);
  const char *sentinel = "WRAPWRAPWRAP";
  int slen = 12, at = CAP - 5;
  memcpy(base + at, sentinel, (unsigned long)slen);
  char rb[16];
  memcpy(rb, base + at, (unsigned long)slen);
  rb[slen] = 0;
  char head[8];
  memcpy(head, base, 7);
  head[7] = 0;
  printf("wrap-read: %s\n", rb);   /* "WRAPWRAPWRAP" — the full straddling span, contiguous */
  printf("wrap-alias: %s\n", head); /* "RAPWRAP" — the overflow, aliased to offset 0 */
  return 0;
}
