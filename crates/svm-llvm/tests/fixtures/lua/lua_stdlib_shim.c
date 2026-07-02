/* Guest libc/libm shim for the Lua stdlib build (the "program brings its own libc" model). Real
 * implementations where cheap (derived from the guest exp/log/sin/cos + a strstr loop); fail-closed
 * (abort → on-ramp trap) for the inverse-trig not yet ported; and a no-filesystem stdio surface: file
 * opens fail so a file-loading script gets a clean Lua error, while the reachable output path
 * (print → fwrite(stdout)) is handled by the on-ramp itself. */
#include <stddef.h>
extern void abort(void) __attribute__((noreturn));

extern double log(double x);
extern double sin(double x);
extern double cos(double x);

double log10(double x) { return log(x) * 0.43429448190325182765; } /* 1/ln(10) */
double log2(double x)  { return log(x) * 1.44269504088896340736; } /* 1/ln(2)  */
double tan(double x)   { return sin(x) / cos(x); }

/* Inverse trig not yet ported to the guest libm — unreached by the stdlib milestone script; a call
 * fails closed (abort lowers to a guest trap) rather than silently mis-computing. */
double acos(double x)  { (void)x; abort(); }
double asin(double x)  { (void)x; abort(); }
double atan2(double y, double x) { (void)y; (void)x; abort(); }

long clock(void) { return 0; }
char *strerror(int e) { (void)e; return "error"; }

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
