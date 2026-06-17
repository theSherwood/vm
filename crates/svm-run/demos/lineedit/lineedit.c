/* Shakedown driver: a tiny line editor doing overlapping shifts, to exercise variable-length
 * `memmove` (the one mem op the on-ramp left `Unsupported` — overlap needs a direction-aware copy).
 * Reads a line from stdin, wraps it in `[...]` (a shift-right: dst > src), then deletes the middle
 * char (a shift-left: dst < src), and writes the result. The runtime length keeps clang from
 * constant-folding the `memmove`s into inline copies. svm-run's output must match a native `cc`
 * build byte-for-byte; the direction-aware copy runs in the guest. */
#include <stddef.h>

long read(int fd, char *buf, long n);
long write(int fd, const char *buf, long n);
void *memmove(void *dst, const void *src, unsigned long n);

int main(void) {
  char buf[256];
  long n = read(0, buf, 200);
  if (n < 0) n = 0;
  if (n > 0 && buf[n - 1] == '\n') n--;             /* strip trailing newline */
  memmove(buf + 1, buf, (unsigned long)n);          /* shift right (dst > src → backward) */
  buf[0] = '[';
  buf[n + 1] = ']';
  n += 2;
  long mid = n / 2;
  memmove(buf + mid, buf + mid + 1, (unsigned long)(n - mid - 1)); /* shift left (dst < src) */
  n -= 1;
  buf[n] = '\n';
  write(1, buf, n + 1);
  return 0;
}
