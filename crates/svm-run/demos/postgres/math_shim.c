/* Guest `pow` — the one transcendental the on-ramp leaves fail-closed.
 *
 * The on-ramp bundles openlibm's `exp`/`log`/`sin`/… but routes `pow` to a trap stub (bit-exact-vs-host
 * `pow` was deferred — the host-libm decision). Postgres hits it early: `InitializeGUCOptions` →
 * `check_timezone` parses an interval and scales fractional seconds by `pow(10, n)`. Define `pow` so
 * that recognizer (gated on `pow` being *undefined*) defers to this real function.
 *
 * Integer exponents (the `pow(10, n)` datetime/numeric case) go through exact binary exponentiation;
 * a general real exponent uses `exp(y*log(x))` over the bundled openlibm (positive base — Postgres'
 * uses are non-negative). Not bit-exact to a specific host `pow`, but Postgres rounds these results
 * (`rint`), and it is deterministic. Compiled with `-fno-builtin-pow` (see `link_shims.sh`) so clang
 * doesn't fold the body into a self-call. `#include`d under `-DSVM_GUEST`.
 */

#include <math.h>

double pow(double x, double y) {
  if (y == 0.0) return 1.0; /* x**0 == 1 for every x (incl. NaN), per C */
  if (x == 1.0) return 1.0;
  if (y == 1.0) return x;
  if (y == (double)(long)y) { /* exact integer exponent (handles x<0 and gives exact 10^n) */
    long n = (long)y;
    int neg = n < 0;
    unsigned long m = neg ? (unsigned long)(-n) : (unsigned long)n;
    double r = 1.0, b = x;
    while (m) {
      if (m & 1UL) r *= b;
      m >>= 1;
      if (m) b *= b;
    }
    return neg ? 1.0 / r : r;
  }
  if (x > 0.0) return exp(y * log(x)); /* general real exponent, positive base */
  if (x == 0.0) return 0.0;
  return NAN; /* negative base, non-integer exponent */
}
