/* Guest libc shim for the io/os (`files.lua`) build — the non-stdio, non-time remainder. The real
 * stdio lives in `lua_files_stdio.c` (over the Fs capability) and time/date in `lua_files_time.c`;
 * this keeps the derived-math wrappers and locale/error odds and ends the other fixtures' shims
 * carry (see `lua_testsuite_shim.c` for the rationale on each). */
#include <stddef.h>

extern double log(double x);
extern double sin(double x);
extern double cos(double x);

double log10(double x) { return log(x) * 0.43429448190325182765; } /* 1/ln(10) */
double log2(double x)  { return log(x) * 1.44269504088896340736; } /* 1/ln(2)  */
double tan(double x)   { return sin(x) / cos(x); }

char *strerror(int e) { (void)e; return "error"; }

/* `localeconv()->decimal_point` — see lua_testsuite_shim.c. */
struct lconv { char *decimal_point; };
static char lc_dot[] = ".";
static struct lconv lc_C = { lc_dot };
void *localeconv(void) { return &lc_C; }

/* `os.setlocale` (loslib): only the "C" locale exists; setting it (or querying with NULL) answers
 * "C", anything else refuses — exactly a minimal ANSI implementation. */
static char lc_name[] = "C";
char *setlocale(int cat, const char *loc) {
  (void)cat;
  if (!loc || (loc[0] == 'C' && loc[1] == 0) || loc[0] == 0) return lc_name;
  return (char *)0;
}

/* `os.execute` (loslib `system`): no shell exists. `system(NULL)` answers 0 ("no shell available"),
 * a command fails with -1 — both shapes loslib maps to honest Lua results. */
int system(const char *cmd) { return cmd ? -1 : 0; }

char *strstr(const char *h, const char *n) {
  if (!*n) return (char *)h;
  for (; *h; h++) {
    const char *a = h, *b = n;
    while (*a && *b && *a == *b) { a++; b++; }
    if (!*b) return (char *)h;
  }
  return (char *)0;
}
