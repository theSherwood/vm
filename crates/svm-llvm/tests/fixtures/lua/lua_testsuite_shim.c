/* Guest libc shim for the Lua test-suite build (the "program brings its own libc" model). Same as the
 * stdlib fixture's shim, except the inverse-trig (`asin`/`acos`/`atan`/`atan2`) + `modf` are now real
 * fdlibm transcriptions in `lua_testsuite_trig.c` (the math test exercises them), so they are dropped
 * here. Keeps `tan`/`log10`/`log2` (derived from the guest sin/cos/log), `strstr`, and the
 * no-filesystem stdio surface (file opens fail → clean Lua error; print→fwrite(stdout) is handled by
 * the on-ramp itself). */
#include <stddef.h>

extern double log(double x);
extern double sin(double x);
extern double cos(double x);

double log10(double x) { return log(x) * 0.43429448190325182765; } /* 1/ln(10) */
double log2(double x)  { return log(x) * 1.44269504088896340736; } /* 1/ln(2)  */
double tan(double x)   { return sin(x) / cos(x); }

long clock(void) { return 0; }
char *strerror(int e) { (void)e; return "error"; }

/* `localeconv()->decimal_point` — Lua reads only `decimal_point[0]` (the first struct field) when it
 * appends ".0" to an integer-valued float in `tostring`. The real `struct lconv` is larger, but Lua
 * touches only the first `char *`, so a one-field stand-in returning "." is sufficient (the "C" locale
 * decimal point). Shadows the on-ramp's fail-closed `localeconv` trap. */
struct lconv { char *decimal_point; };
static char lc_dot[] = ".";
static struct lconv lc_C = { lc_dot };
void *localeconv(void) { return &lc_C; }

char *strstr(const char *h, const char *n) {
  if (!*n) return (char *)h;
  for (; *h; h++) {
    const char *a = h, *b = n;
    while (*a && *b && *a == *b) { a++; b++; }
    if (!*b) return (char *)h;
  }
  return NULL;
}

/* no-filesystem stdio (unreached by non-file scripts) */
typedef struct FILE FILE;
FILE *stdin;
FILE *stdout;
FILE *stderr;
FILE *fopen64(const char *p, const char *m) { (void)p; (void)m; return NULL; }
FILE *freopen64(const char *p, const char *m, FILE *f) { (void)p; (void)m; (void)f; return NULL; }
unsigned long fread(void *b, unsigned long s, unsigned long n, FILE *f) { (void)b;(void)s;(void)n;(void)f; return 0; }
int fclose(FILE *f) { (void)f; return 0; }
int feof(FILE *f) { (void)f; return 1; }
int ferror(FILE *f) { (void)f; return 0; }
int fflush(FILE *f) { (void)f; return 0; }
int getc(FILE *f) { (void)f; return -1; }
int fprintf(FILE *f, const char *fmt, ...) { (void)f; (void)fmt; return 0; }
