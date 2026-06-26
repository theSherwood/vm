/* Differential driver for the bundled guest `libm` (this directory's `libm.c`): evaluate `exp` and
 * `log` over a grid of runtime (`volatile`) inputs and write each result's **raw little-endian f64
 * image** to stdout. Comparing the raw bytes (not a formatted number) makes the differential exact.
 *
 * Because the transcendentals are *guest code*, native `cc` compiles the very same `exp`/`log`, so
 * every value is bit-identical (the only machine ops are IEEE `+ - * /`, comparisons, and the union
 * word access — unfused on both lanes). svm-run's stdout must equal a native build byte-for-byte, on
 * the tree-walker, the bytecode VM, and the JIT (`demo_libm_exp_log_vs_native`). */
#include <stddef.h>

long write(int fd, const void *buf, long n);

#include "libm.c"

int main(void) {
  /* exp inputs: zero, the small-|x| fast path, the reduction range, near overflow/underflow. */
  volatile double xs[] = {0.0,   0.5,   1.0,    2.0,  3.14159265358979, 10.0,  -1.0,
                          -5.0,  100.0, 0.001,  709.0, -700.0,          1e-300, -0.0,
                          708.0, 0.25,  -0.25,  -708.0};
  /* log inputs: 1, e, powers of two, large/small magnitudes, a subnormal. */
  volatile double ls[] = {1.0,   2.0,    0.5,  2.718281828459045, 10.0,   1e10, 1e-10,
                          0.001, 1e300,  123456.789, 4.0, 0.125, 1e-300, 5e-324};
  int ne = (int)(sizeof xs / sizeof xs[0]);
  int nl = (int)(sizeof ls / sizeof ls[0]);
  unsigned char out[8 * 64];
  int o = 0;
  for (int i = 0; i < ne; i++) {
    libm_du u;
    u.d = exp(xs[i]);
    for (int b = 0; b < 8; b++) out[o++] = (unsigned char)(u.u >> (8 * b));
  }
  for (int i = 0; i < nl; i++) {
    libm_du u;
    u.d = log(ls[i]);
    for (int b = 0; b < 8; b++) out[o++] = (unsigned char)(u.u >> (8 * b));
  }
  write(1, out, o);

  /* pow over (base, exponent) pairs: positive/negative bases, integer (odd/even) and fractional
   * exponents, 0/1/2/0.5 special cases, and a large exponent. */
  volatile double pb[] = {2.0, 3.0,  -2.0, -2.0, 10.0,  0.5,  2.0,  9.0,
                          -1.0, 1.5, 2.0,  0.0,  -3.0,  100.0, 2.0};
  volatile double pe[] = {10.0, 3.0, 3.0, 4.0,  -2.0, 0.5,  0.5,  0.5,
                          0.0,  2.0, 2.5, 5.0,  3.0,  0.0,   1e10};
  int np = (int)(sizeof pb / sizeof pb[0]);
  o = 0;
  for (int i = 0; i < np; i++) {
    libm_du u;
    u.d = pow(pb[i], pe[i]);
    for (int b = 0; b < 8; b++) out[o++] = (unsigned char)(u.u >> (8 * b));
  }
  write(1, out, o);
  return 0;
}
