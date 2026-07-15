/* Guest mem + str builtins — real definitions of the mem/string builtins whose address Postgres takes.
 *
 * The on-ramp synthesizes these for *direct* calls (and lowers the `llvm.mem*` intrinsics inline), so
 * the ~14 985-function module never needed a defined `memcpy` — until Postgres takes the **address** of
 * one and calls through it. A `dynahash` `HASH_BLOBS` table stores `hashp->keycopy = memcpy` and
 * `hashp->match = memcmp`, a `HASH_STRINGS` table its string comparator, and calls them indirectly
 * (`ProcessSyncRequests` → `hash_search_with_hash_value` → `match(...)`). An address-taken function that
 * is *only synthesized* (never defined) resolves to a fail-closed trap stub — calling through it traps.
 * Defining these makes the taken address point at a real function with the libc ABI; direct calls still
 * fast-path through the synthesizer/intrinsic, only the funcref uses this body.
 *
 * Compiled with `-fno-builtin-*` for each (see `link_shims.sh`) so clang's loop-idiom pass doesn't
 * rewrite the byte loops into self-calls. `#include`d under `-DSVM_GUEST`.
 */

#include <stddef.h>

void *memcpy(void *dst, const void *src, size_t n) {
  unsigned char *d = (unsigned char *)dst;
  const unsigned char *s = (const unsigned char *)src;
  for (size_t i = 0; i < n; i++) d[i] = s[i];
  return dst;
}

int memcmp(const void *a, const void *b, size_t n) {
  const unsigned char *x = (const unsigned char *)a;
  const unsigned char *y = (const unsigned char *)b;
  for (size_t i = 0; i < n; i++)
    if (x[i] != y[i]) return (int)x[i] - (int)y[i];
  return 0;
}

size_t strlen(const char *s) {
  size_t n = 0;
  while (s[n]) n++;
  return n;
}

int strcmp(const char *a, const char *b) {
  while (*a && *a == *b) {
    a++;
    b++;
  }
  return (int)(unsigned char)*a - (int)(unsigned char)*b;
}

int strncmp(const char *a, const char *b, size_t n) {
  for (size_t i = 0; i < n; i++) {
    unsigned char ca = (unsigned char)a[i], cb = (unsigned char)b[i];
    if (ca != cb) return (int)ca - (int)cb;
    if (!ca) break;
  }
  return 0;
}

void *memmove(void *dst, const void *src, size_t n) {
  unsigned char *d = (unsigned char *)dst;
  const unsigned char *s = (const unsigned char *)src;
  if (d < s) {
    for (size_t i = 0; i < n; i++) d[i] = s[i];
  } else {
    for (size_t i = n; i > 0; i--) d[i - 1] = s[i - 1];
  }
  return dst;
}

void *memset(void *dst, int c, size_t n) {
  unsigned char *d = (unsigned char *)dst;
  for (size_t i = 0; i < n; i++) d[i] = (unsigned char)c;
  return dst;
}
