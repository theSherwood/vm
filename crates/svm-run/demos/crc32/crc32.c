/* Shakedown driver: CRC-32 over stdin + a big-endian u32 reader, to exercise `llvm.bswap`.
 * Reads stdin into a buffer, prints its CRC-32 (`%08x`), then reinterprets each aligned 4-byte
 * group as a big-endian u32 (`__builtin_bswap32` on a host-endian load) and prints their sum.
 * svm-run's output must match a native `cc` build byte-for-byte. `bswap` lowers to an inline
 * shift/mask byte reversal in the guest; `printf` is the guest-side format engine. */
#include <stddef.h>

long read(int fd, char *buf, long n);
int printf(const char *fmt, ...);

static unsigned crc32(const unsigned char *p, long n) {
  unsigned c = 0xffffffffu;
  for (long i = 0; i < n; i++) {
    c ^= p[i];
    for (int k = 0; k < 8; k++)
      c = (c >> 1) ^ (0xedb88320u & (unsigned)(-(int)(c & 1)));
  }
  return ~c;
}

int main(void) {
  static unsigned char buf[256];
  long n = 0, r;
  while (n < (long)sizeof(buf) && (r = read(0, (char *)buf + n, (long)sizeof(buf) - n)) > 0)
    n += r;
  printf("crc32=%08x len=%u\n", crc32(buf, n), (unsigned)n);
  unsigned long sum = 0;
  for (long i = 0; i + 4 <= n; i += 4) {
    unsigned le;
    __builtin_memcpy(&le, buf + i, 4);
    sum += __builtin_bswap32(le); /* big-endian interpretation */
  }
  printf("be32sum=%lx\n", sum);
  return 0;
}
