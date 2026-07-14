/* fprintf_probe.c — byte-exact differential for the guest varargs printf engine (slice CH,
 * Postgres runtime gap #11g).
 *
 * Guest (`-DSVM_GUEST`, os_shim.c + stdio_shim.c + printf_shim.c) vs native glibc. Exercises the
 * runtime format family (`printf`/`fprintf`/`vfprintf`/`snprintf`) across the conversions Postgres
 * uses — `%d`/`%u`/`%x`/`%c`/`%s`/`%f`/`%e`/`%g` with width/precision/flags — to **three** targets:
 * `stdout` (the powerbox out-Stream via the slice-CE fd-dispatch), a real **file** (the fs cap, whose
 * contents are read back and echoed to stdout), and `stderr` (the shared out-Stream). To make the
 * two console streams a clean differential the *native* build folds `stderr` into `stdout` (`dup2`)
 * and runs both **unbuffered**, so they interleave in program order exactly as the guest's single
 * write-through Stream does. Runs over `mem_fs` and `host_fs`. */

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#ifdef SVM_GUEST
#include "os_shim.c"
#include "stdio_shim.c"
#include "printf_shim.c"
#else
#include <unistd.h>
#endif

/* The `vfprintf` va_list entry point Postgres' `errmsg`/`appendStringInfoVA` reach. */
static int say(FILE *f, const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  int n = vfprintf(f, fmt, ap);
  va_end(ap);
  return n;
}

int main(void) {
#ifndef SVM_GUEST
  setvbuf(stdout, NULL, _IONBF, 0);
  setvbuf(stderr, NULL, _IONBF, 0);
  dup2(STDOUT_FILENO, STDERR_FILENO);
#endif

  /* --- the format engine, to stdout ------------------------------------------------------------ */
  printf("int: %d %5d %-5d|%05d %+d % d\n", 42, 42, 42, 42, 42, 42);
  printf("neg: %d %ld %lld\n", -7, -1234567L, -9876543210LL);
  printf("uns: %u %x %X %#x %o %#o\n", 4000000000u, 0xdeadbeefu, 0xabcU, 255u, 64u, 64u);
  printf("str: [%s] [%10s] [%-10s] [%.3s]\n", "abc", "abc", "abc", "abcdef");
  printf("chr: %c%c%c pct:%%\n", 'x', 'y', 'z');
  printf("flt: %f %.2f %+.3f %10.2f %-10.2f|\n", 3.14159, 3.14159, 3.14159, 3.14159, 3.14159);
  printf("sci: %e %.3e %g %.10g %G\n", 31415.926, 31415.926, 0.0001234, 1.0 / 3.0, 6.022e23);
  int n = say(stdout, "vfprintf: %d/%d = %.4f\n", 22, 7, 22.0 / 7.0);
  printf("vfprintf_ret=%d\n", n);
  char b[64];
  int m = snprintf(b, sizeof b, "snprintf[%d,%s,%.1f]", 9, "ok", 2.5);
  printf("snprintf=%s ret=%d\n", b, m);

  /* --- fprintf to a real file (fs cap), then read it back to stdout ----------------------------- */
  FILE *f = fopen("out.txt", "w");
  fprintf(f, "file: n=%d s=%s f=%.2f\n", 100, "row", 9.75);
  fprintf(f, "line2 hex=%x oct=%o\n", 0xbeef, 511);
  fclose(f);
  f = fopen("out.txt", "r");
  char fbuf[128];
  size_t rd = fread(fbuf, 1, sizeof fbuf - 1, f);
  fbuf[rd] = 0;
  fclose(f);
  printf("readback=[%s]", fbuf); /* fbuf keeps its own newlines */

  /* --- the stderr half (shared out-Stream in the guest; dup2'd in native) ---------------------- */
  fprintf(stderr, "stderr: code=%d msg=%s val=%.1f\n", 7, "boom", 1.5);
  say(stderr, "stderr-vfprintf %g\n", 2.5e-3);
  return 0;
}
