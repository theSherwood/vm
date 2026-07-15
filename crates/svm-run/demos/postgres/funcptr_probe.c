/* funcptr_probe.c — differential for **address-taken** mem/string builtins (mem_shim.c).
 *
 * The on-ramp synthesizes `memcmp`/`memcpy`/`strlen`/… for *direct* calls, but taking their **address**
 * and calling indirectly (dynahash stores `hashp->match = memcmp` / `keycopy = memcpy` and calls them
 * through the pointer — `ProcessSyncRequests` → `hash_search_with_hash_value`) resolves to a fail-closed
 * trap stub unless the function is actually *defined*. `mem_shim.c` defines them; this probe stores each
 * in a `volatile` pointer (so clang can't devirtualize the call back to a direct/inline form) and calls
 * it, byte-matching native glibc. This is the exact bug that trapped Postgres' end-of-recovery checkpoint.
 */

#include <stdio.h>
#include <string.h>

#ifdef SVM_GUEST
#include "mem_shim.c"
#endif

typedef int (*cmpf)(const void *, const void *, unsigned long);
typedef void *(*cpyf)(void *, const void *, unsigned long);
typedef void *(*setf)(void *, int, unsigned long);
typedef unsigned long (*lenf)(const char *);
typedef int (*scmpf)(const char *, const char *);
typedef int (*sncmpf)(const char *, const char *, unsigned long);

static int sgn(int x) { return x < 0 ? -1 : (x > 0 ? 1 : 0); }

int main(void) {
  volatile cmpf mc = memcmp;
  volatile cpyf cp = memcpy;
  volatile setf st = memset;
  volatile lenf ln = strlen;
  volatile scmpf sc = strcmp;
  volatile sncmpf snc = strncmp;

  char dst[8];
  cp(dst, "hello", 6);      /* -> "hello\0" */
  st(dst + 5, '!', 1);      /* -> "hello!" (no NUL) */
  dst[6] = 0;
  int a = mc("abc", "abd", 3);
  int b = sc("foo", "fop");
  int c = snc("bar", "baz", 2); /* equal in first 2 */
  printf("%s %d %d %d %lu\n", dst, sgn(a), sgn(b), sgn(c), ln("hello"));
  return 0;
}
