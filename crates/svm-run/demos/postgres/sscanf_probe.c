/* sscanf_probe.c — byte-exact differential for the guest varargs scanf engine (slice CJ,
 * Postgres runtime gap #11l).
 *
 * Guest (`-DSVM_GUEST`, os_shim.c + stdio_shim.c + printf_shim.c + scanf_shim.c) vs native glibc.
 * Part 1 drives `sscanf` over embedded strings across the conversions Postgres parses config/version
 * values with — d/i/u/o/x/c/s/f/lf/g/[scanset]/n/%%, with width, assignment-suppression `*`, and
 * length modifiers — printing each parsed value *and* the return count (a scanf differential must
 * check the count, not just the values). Part 2 drives `fscanf` from `stdin` (the powerbox in-Stream)
 * to prove the FILE back end. All output is to stdout; the guest byte-matches native. */

#include <stdio.h>

#ifdef SVM_GUEST
#include "../strtod/strtod.c" /* real correctly-rounded strtod (the on-ramp's is a trap stub) */
#include "os_shim.c"
#include "stdio_shim.c"
#include "printf_shim.c"
#include "scanf_shim.c"
#endif

int main(void) {
  int a, b, c, n;
  unsigned u;
  long l;
  long long ll;
  double d, e;
  float f;
  char s1[32], s2[32], s3[32];

  /* --- integers: bases, sign, width, suppression, %n ------------------------------------------ */
  n = sscanf("42 -7 +13", "%d %d %d", &a, &b, &c);
  printf("ints: n=%d %d %d %d\n", n, a, b, c);

  n = sscanf("0x1a 075 99", "%x %o %u", &u, &a, &b);
  printf("bases: n=%d %u %d %d\n", n, u, a, b);

  n = sscanf("2147483648 100 200", "%ld %lld %d", &l, &ll, &a);
  printf("lengths: n=%d %ld %lld %d\n", n, l, ll, a);

  n = sscanf("0x2A 0b 55", "%i %i %i", &a, &b, &c);
  printf("auto-base: n=%d %d %d\n", n, a, b); /* "%i" auto-detects; "0b" -> 0, stops */

  int chars = 0;
  n = sscanf("123abc", "%d%n", &a, &chars);
  printf("with-n: n=%d val=%d consumed=%d\n", n, a, chars);

  n = sscanf("11 22 33", "%d %*d %d", &a, &b); /* the middle is suppressed */
  printf("suppress: n=%d %d %d\n", n, a, b);

  n = sscanf("12345", "%3d%2d", &a, &b); /* field width splits the run */
  printf("width: n=%d %d %d\n", n, a, b);

  /* --- floats via strtod ---------------------------------------------------------------------- */
  double d3;
  n = sscanf("3.14159 -2.5e3 .5", "%lf %lf %f", &d, &e, &f);
  printf("floats: n=%d %.5f %.1f %.2f\n", n, d, e, f);

  n = sscanf("1e10 42.0 6.022e23", "%lf %lf %lf", &d, &e, &d3);
  printf("float2: n=%d %.1e %.1f %.15g\n", n, d, e, d3);

  /* --- strings + scansets --------------------------------------------------------------------- */
  n = sscanf("  hello world  ", "%s %s", s1, s2);
  printf("strs: n=%d [%s] [%s]\n", n, s1, s2);

  n = sscanf("key=value;rest", "%[^=]=%[^;]", s1, s2);
  printf("scanset: n=%d [%s] [%s]\n", n, s1, s2);

  n = sscanf("abc123def", "%[a-z]%[0-9]%[a-z]", s1, s2, s3);
  printf("ranges: n=%d [%s] [%s] [%s]\n", n, s1, s2, s3); /* a-z / 0-9 range expansion */

  n = sscanf("R255G128B064", "R%dG%dB%d", &a, &b, &c); /* literals between conversions */
  printf("mixed: n=%d %d %d %d\n", n, a, b, c);

  /* --- matching failures (partial assignment / count semantics) -------------------------------- */
  n = sscanf("7 xyz 9", "%d %d %d", &a, &b, &c);
  printf("fail: n=%d first=%d\n", n, a); /* stops at "xyz": returns 1 */

  n = sscanf("", "%d", &a);
  printf("empty: n=%d\n", n); /* EOF */

  /* --- fscanf from the powerbox stdin --------------------------------------------------------- */
  n = fscanf(stdin, "%d %d %lf %s", &a, &b, &d, s1);
  printf("stdin: n=%d %d %d %.3f [%s]\n", n, a, b, d, s1);
  int rest = fscanf(stdin, "%d", &c);
  printf("stdin2: n=%d %d\n", rest, c);
  return 0;
}
