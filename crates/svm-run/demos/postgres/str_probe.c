/* str_probe.c — byte-exact differential for the guest string + integer-parsing shim (slice CC).
 *
 * Guest (`-DSVM_GUEST`, using libc_shim.c) vs native glibc, over strcat/strncpy/strnlen/strstr/
 * strchrnul/strdup/strlcpy/strlcat/strtok/strxfrm and strtol/strtoul/atoi. Integer results are
 * printed as `(int)` — the guest and native both compute the same 64-bit value, so the truncation
 * agrees even on the ERANGE-clamped cases, and everything stays inside the printf `%d`/`%s` surface.
 */

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef SVM_GUEST
#include "libc_shim.c"
#endif

static void tl(const char *s, int base) {
  char *e;
  errno = 0;
  long v = strtol(s, &e, base);
  printf("strtol(%s,%d)=%d end=%d errno=%d\n", s, base, (int)v, (int)(e - s), errno);
}
static void tul(const char *s, int base) {
  char *e;
  errno = 0;
  unsigned long v = strtoul(s, &e, base);
  printf("strtoul(%s,%d)=%d end=%d errno=%d\n", s, base, (int)v, (int)(e - s), errno);
}

int main(void) {
  /* integer parsing: sign, bases, prefixes, whitespace, endptr, and ERANGE overflow */
  tl("123", 10);
  tl("  -456xyz", 10);
  tl("+0x1F", 0);
  tl("0777", 0);
  tl("z", 36);
  tl("  0X2a", 16);
  tl("99999999999999999999999", 10); /* overflow → LONG_MAX, ERANGE */
  tl("-99999999999999999999999", 10);
  tl("nope", 10);
  tl("7f", 16);
  tul("4294967295", 10);
  tul("-1", 10); /* wraps to ULONG_MAX */
  tul("0xffffffffffffffff", 0);
  printf("atoi=%d %d\n", atoi("  -42abc"), atoi("nan"));

  /* strcat / strncpy (NUL-pad) / strnlen */
  char buf[32];
  buf[0] = 0;
  strcat(buf, "foo");
  strcat(buf, "bar");
  printf("strcat=%s\n", buf);
  char nc[8];
  strncpy(nc, "abc", 8);
  printf("strncpy=%s len4=%d pad=%d\n", nc, (int)strnlen("abcdef", 4), nc[6] == 0);
  strncpy(nc, "abcdefgh", 4);
  nc[4] = 0;
  printf("strncpy_trunc=%s\n", nc);

  /* strstr / strchrnul */
  const char *h = "the quick brown fox";
  printf("strstr=%d %d\n", (int)(strstr(h, "brown") - h), strstr(h, "cat") == NULL);
  printf("strchrnul=%d %d\n", (int)(strchrnul(h, 'q') - h), (int)(strchrnul(h, 'Z') - h));

  /* strdup */
  char *d = strdup("duplicate");
  printf("strdup=%s eq=%d\n", d, strcmp(d, "duplicate") == 0);
  free(d);

  /* strlcpy / strlcat (bounded, returns attempted length) */
  char lc[8];
  size_t r1 = strlcpy(lc, "hello world", sizeof lc);
  printf("strlcpy=%s ret=%d\n", lc, (int)r1);
  char la[8] = "ab";
  size_t r2 = strlcat(la, "cdefghij", sizeof la);
  printf("strlcat=%s ret=%d\n", la, (int)r2);

  /* strtok */
  char toks[] = "a,,b;c";
  char *t = strtok(toks, ",;");
  printf("strtok:");
  while (t) {
    printf(" %s", t);
    t = strtok(NULL, ",;");
  }
  printf("\n");

  /* strxfrm (C locale = bounded identity copy, returns source length) */
  char xf[16];
  size_t xr = strxfrm(xf, "collate", sizeof xf);
  printf("strxfrm=%s ret=%d\n", xf, (int)xr);

  return 0;
}
