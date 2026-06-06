/* Demo: a guest program **grows its own heap** far past its initial window by allocating
 * megabytes through `malloc` — which, in the sandbox build, commits reserved-tail pages on demand
 * via the Memory capability (`vm_malloc.h` → `__vm_map`, §3e/§4). The §1a "large/sparse programs
 * that fight wasm's flat linear memory" target, demonstrated end to end: the sandboxed output is
 * byte-identical to a native `cc` build (which uses the real `malloc`).
 *
 * The program allocates eight 128 KiB int blocks (1 MiB total — ~16× the 64 KiB initial window),
 * fills each with a deterministic pattern, sums it, frees it, and prints the running totals. */

#include <stdlib.h> /* sandbox: the map-growing guest malloc; native: the real one */

int write(int fd, char *buf, long n); /* sandbox: powerbox builtin; native: libc */

static void puts_(const char *s) {
  int n = 0;
  while (s[n]) n++;
  write(1, (char *)s, n);
}
static void putl(long v) {
  char buf[24];
  int i = sizeof(buf);
  unsigned long u = v < 0 ? (unsigned long)(-v) : (unsigned long)v;
  if (u == 0) buf[--i] = '0';
  while (u) {
    buf[--i] = (char)('0' + u % 10);
    u /= 10;
  }
  if (v < 0) buf[--i] = '-';
  write(1, &buf[i], (long)(sizeof(buf) - i));
}

#define BLOCKS 8
#define N 32768 /* ints per block: 128 KiB each */

int main(void) {
  long total = 0;
  for (int b = 0; b < BLOCKS; b++) {
    int *a = (int *)malloc((unsigned long)N * sizeof(int));
    if (!a) {
      puts_("OOM\n");
      return 1;
    }
    for (int i = 0; i < N; i++)
      a[i] = (i * 7 + b * 3) & 1023;
    long s = 0;
    for (int i = 0; i < N; i++)
      s += a[i];
    total += s;
    puts_("block ");
    putl(b);
    puts_(" sum ");
    putl(s);
    puts_("\n");
    free(a);
  }
  puts_("total ");
  putl(total);
  puts_("\n");
  return 0;
}
