/* Guest-libc addendum for the T-library build. `ltests.h` turns internal assertions on
 * (`LUAI_ASSERT` → `<assert.h>`), so a failing `lua_assert` reaches glibc's `__assert_fail`;
 * `ltests.c` also calls `abort` directly on invariant violations. Both are honest hard-stops:
 * report on stdout (through the guest stdio layer — runtime strings, so `fputs`, not `printf`)
 * and exit nonzero, so a broken invariant is a loud failing exit code, never silence. */
#include <stddef.h>

typedef struct FILE FILE;
extern FILE *stdout;
extern int fputs(const char *s, FILE *f);
extern void exit(int code);

void abort(void) {
  fputs("abort()\n", stdout);
  exit(134); /* 128 + SIGABRT, the conventional code */
}

void __assert_fail(const char *assertion, const char *file, unsigned line, const char *function) {
  (void)line;
  fputs("assertion failed: ", stdout);
  fputs(assertion, stdout);
  fputs(" in ", stdout);
  fputs(function ? function : "?", stdout);
  fputs(" (", stdout);
  fputs(file, stdout);
  fputs(")\n", stdout);
  exit(134);
}

/* `LUA_COMPAT_MATHLIB` (set by ltests.h) revives `math.sinh`/`cosh`/`tanh` — derive them from the
 * guest `exp` (the compat functions are exercised only incidentally; IEEE-exact parity with glibc
 * is not asserted by the suite). */
extern double exp(double x);
double sinh(double x) { return (exp(x) - exp(-x)) / 2.0; }
double cosh(double x) { return (exp(x) + exp(-x)) / 2.0; }
double tanh(double x) {
  double e = exp(2.0 * x);
  return (e - 1.0) / (e + 1.0);
}

/* `debug_realloc` reads a `MEMLIMIT` env var through `strtoul` (base 10; unset in our runs, but the
 * symbol must translate). Minimal decimal/hex parse, glibc `endptr` shape. */
unsigned long strtoul(const char *s, char **end, int base) {
  unsigned long v = 0;
  while (*s == ' ' || *s == '\t') s++;
  if (base == 16 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) s += 2;
  for (;; s++) {
    int d;
    if (*s >= '0' && *s <= '9') d = *s - '0';
    else if (base == 16 && *s >= 'a' && *s <= 'f') d = *s - 'a' + 10;
    else if (base == 16 && *s >= 'A' && *s <= 'F') d = *s - 'A' + 10;
    else break;
    v = v * (unsigned long)base + (unsigned long)d;
  }
  if (end) *end = (char *)s;
  return v;
}

/* `ltests.c`'s reports use printf conversions the on-ramp's constant-format lowering doesn't carry
 * (`%X` et al.), so the T build links a real guest `printf` over the guest vsnprintf + stdio layer —
 * a guest definition shadows the lowering, exactly like the rest of the bring-your-own-libc model. */
typedef __builtin_va_list va_list;
extern int vsnprintf(char *buf, unsigned long size, const char *fmt, va_list ap);
int printf(const char *fmt, ...) {
  char buf[512];
  va_list ap;
  __builtin_va_start(ap, fmt);
  int n = vsnprintf(buf, sizeof buf, fmt, ap);
  __builtin_va_end(ap);
  if (n > (int)sizeof buf - 1) n = (int)sizeof buf - 1;
  if (n > 0) {
    buf[n] = 0;
    fputs(buf, stdout);
  }
  return n;
}

/* `luaB_opentests` registers `atexit(checkfinalmem)` — the end-of-process all-memory-freed check.
 * The guest powerbox has no exit-hook point (main returns straight into `_start`), so this records
 * nothing; the **native oracle runs the same check for real**, which keeps the leak invariant
 * enforced on every fixture regeneration. */
int atexit(void (*fn)(void)) {
  (void)fn;
  return 0;
}

/* Plain string helpers referenced by loadlib.c/ltests.c path and buffer handling. */
extern unsigned long strlen(const char *s);
char *strcat(char *d, const char *s) {
  char *p = d;
  while (*p) p++;
  while ((*p++ = *s++)) {}
  return d;
}
char *strcpy(char *d, const char *s) {
  char *p = d;
  while ((*p++ = *s++)) {}
  return d;
}
char *strncpy(char *d, const char *s, unsigned long n) {
  unsigned long i = 0;
  for (; i < n && s[i]; i++) d[i] = s[i];
  for (; i < n; i++) d[i] = 0;
  return d;
}
int strncmp(const char *a, const char *b, unsigned long n) {
  for (unsigned long i = 0; i < n; i++) {
    unsigned char ca = (unsigned char)a[i], cb = (unsigned char)b[i];
    if (ca != cb) return ca < cb ? -1 : 1;
    if (!ca) return 0;
  }
  return 0;
}

/* `sprintf` (loadlib.c's LUA_DIRSEP substitution path) — bounded onto the guest vsnprintf. */
int sprintf(char *buf, const char *fmt, ...) {
  va_list ap;
  __builtin_va_start(ap, fmt);
  int n = vsnprintf(buf, 1u << 20 /* caller-guaranteed, like libc */, fmt, ap);
  __builtin_va_end(ap);
  return n;
}
