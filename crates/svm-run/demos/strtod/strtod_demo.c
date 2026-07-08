/* Differential driver for the guest `strtod` (this directory's `strtod.c`): parse a grid of decimal
 * strings and write each result's **raw little-endian f64 image** plus the `endptr` offset to stdout.
 * Comparing raw bytes makes the differential exact. Since the parser is guest code, native `cc`
 * compiles the same source, so svm-run's stdout must equal a native build byte-for-byte on the
 * tree-walker, the bytecode VM, and the JIT (`demo_strtod_vs_native`). */
#include <stddef.h>

long write(int fd, const void *buf, long n);

#include "strtod.c"

int main(void) {
  static const char *const t[] = {
      "0",          "0.0",        "-0.0",       "1",          "3.14",
      "0.5",        "2.5",        "100.0",      "-2.5",       "1e10",
      "1e-10",      "1e100",      "1e-100",     "1.5e3",      "123456789.123456789",
      "0.1",        "0.3",        "9007199254740992",        "9007199254740993",
      "1.7976931348623157e308",   "2.2250738585072014e-308", "5e-324",
      "4.9e-324",   "2.4703282292062327e-324",  "1e309",      "1e-400",
      "0.0000001",  "  42.0",     "  -3.25e2",  "1000000000000000000000",
      "0.30000000000000004",      "1e22",       "1e23",       "0.000244140625",
      "1.25e-1",    ".5",         "5.",         "+1.5",       "3.141592653589793",
      "0x7.4",      "0x.ABCDEFp+24", "0x0.51p+8", "0xa.aP4",   "0x4P-2",
      "0x0.7a7040a5a323c9d6",       "0x1.8p1",    "-0x1.8p1",  "0x1p-1074",
      "0x",         "0x.",        "0x3.3.3",
  };
  int n = (int)(sizeof t / sizeof t[0]);
  unsigned char out[16 * 64];
  int o = 0;
  for (int i = 0; i < n; i++) {
    char *end;
    strtod_du u;
    u.d = strtod(t[i], &end);
    for (int b = 0; b < 8; b++) out[o++] = (unsigned char)(u.u >> (8 * b));
    long off = (long)(end - t[i]);
    for (int b = 0; b < 8; b++) out[o++] = (unsigned char)((unsigned long long)off >> (8 * b));
  }
  write(1, out, o);
  return 0;
}
