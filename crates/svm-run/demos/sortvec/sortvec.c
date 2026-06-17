/* Shakedown driver: a growable int vector + insertion sort, to exercise `realloc` and signed
 * `printf("%d")`. Generates 50 pseudo-random signed ints into a `realloc`-doubling buffer
 * (starting from `realloc(NULL, …)` ≡ malloc), sorts them, and prints them 10 per line. svm-run's
 * output must match a native `cc` build byte-for-byte. `realloc`/`malloc` run in the guest (the
 * `vm_map`-growing bump allocator); `printf` is the guest-side format engine. */
#include <stddef.h>

int printf(const char *fmt, ...);
void *realloc(void *p, unsigned long n);

int main(void) {
  int cap = 4, n = 0;
  int *a = (int *)realloc(NULL, (unsigned long)cap * sizeof(int));
  unsigned x = 12345u;
  for (int i = 0; i < 50; i++) {
    if (n == cap) {
      cap *= 2;
      a = (int *)realloc(a, (unsigned long)cap * sizeof(int));
    }
    x = x * 1103515245u + 12345u;          /* LCG */
    a[n++] = (int)((x >> 16) % 1000) - 500; /* signed, in [-500, 499] */
  }
  for (int i = 1; i < n; i++) {            /* insertion sort */
    int v = a[i], j = i - 1;
    while (j >= 0 && a[j] > v) { a[j + 1] = a[j]; j--; }
    a[j + 1] = v;
  }
  for (int i = 0; i < n; i++)
    printf("%d%c", a[i], (i + 1) % 10 == 0 ? '\n' : ' ');
  printf("\n");
  return 0;
}
