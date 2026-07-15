/* Guest mem* — real `memcpy`/`memmove`/`memset` definitions.
 *
 * The on-ramp synthesizes these for *direct* calls (and lowers the `llvm.mem*` intrinsics inline), so
 * the ~14 985-function module never needed a defined `memcpy` — until Postgres takes the **address**
 * of one (a funcref into a dispatch/callback table, e.g. a sort/serialize thunk). An address-taken
 * undefined extern resolves to a trap stub (the funcref counterpart of call-site stubbing), so calling
 * through it traps `Unreachable`. Defining these makes the taken address point at a real function with
 * the libc ABI — direct calls still fast-path through the synthesizer/intrinsic; only the funcref uses
 * this body.
 *
 * Compiled with `-fno-builtin-memcpy -fno-builtin-memmove -fno-builtin-memset` (see `link_shims.sh`) so
 * clang's loop-idiom pass doesn't rewrite these byte loops into self-calls. `#include`d under
 * `-DSVM_GUEST`.
 */

#include <stddef.h>

void *memcpy(void *dst, const void *src, size_t n) {
  unsigned char *d = (unsigned char *)dst;
  const unsigned char *s = (const unsigned char *)src;
  for (size_t i = 0; i < n; i++) d[i] = s[i];
  return dst;
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
