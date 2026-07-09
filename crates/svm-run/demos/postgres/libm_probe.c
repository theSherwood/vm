/* libm_probe.c — exercise the **bundled guest libm** (openlibm's transcendentals) and emit a
 * bit-exact FNV-1a hash of every result. The differential (guest vs native, *both* built against the
 * same bundled openlibm) then tests the **math translation** itself — not float formatting — because
 * the hash rides on the raw IEEE bits and is printed as hex via `putchar`. This is the libm the
 * Postgres on-ramp llvm-links in (the SVM has no transcendental op; math stays guest code, §"libm").
 *
 * Built on BOTH sides against openlibm (not the system `-lm`), so any implementation-defined last-ulp
 * choice is identical on guest and native by construction — the test asserts the on-ramp reproduces
 * openlibm bit-for-bit across interp / JIT / native, over ~3600 evaluations spanning the input range.
 */
#include "openlibm.h"

typedef unsigned long long u64;

static u64 g_h;
static void mix(double x) {
    union {
        double d;
        u64 u;
    } v;
    v.d = x;
    g_h ^= v.u;
    g_h *= 1099511628211ULL; /* FNV-1a 64-bit */
}

extern int putchar(int);
static void emit_hex(u64 x) {
    for (int i = 60; i >= 0; i -= 4)
        putchar("0123456789abcdef"[(x >> i) & 0xF]);
    putchar('\n');
}

int main(void) {
    g_h = 1469598103934665603ULL; /* FNV offset basis */
    for (int i = 1; i <= 200; i++) {
        double x = (double)i * 0.5 - 25.0; /* -24.5 .. 75.0 */
        /* logs (i >= 1 so the domain is valid) */
        mix(log((double)i));
        mix(log10((double)i));
        mix(log2((double)i));
        /* exponentials */
        mix(exp(x * 0.1));
        mix(exp2(x * 0.1));
        mix(pow(1.0 + (double)i * 0.01, x * 0.05));
        /* trig */
        mix(sin(x));
        mix(cos(x));
        mix(tan(x));
        mix(asin(x / 100.0));
        mix(acos(x / 100.0));
        mix(atan(x));
        mix(atan2(x, 3.0));
        /* hyperbolics */
        mix(sinh(x * 0.1));
        mix(cosh(x * 0.1));
        mix(tanh(x * 0.1));
        /* misc */
        mix(cbrt(x));
        mix(fmod(x, 3.0));
    }
    emit_hex(g_h);
    return 0;
}
