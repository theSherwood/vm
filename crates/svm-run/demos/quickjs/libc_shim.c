/* QuickJS libc shim — the small surface the on-ramp neither synthesizes nor covers via the reused
 * Postgres printf engine (`printf_shim.c`) and the guest `strtod` (`demos/strtod/strtod.c`).
 *
 * Linked into the guest build (and the native oracle, so the differential stays honest). Everything
 * here is ordinary guest C; a guest definition shadows the on-ramp's would-be trap stub. See the
 * QuickJS README for the full gap inventory.
 */
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>

extern void exit(int);

/* --- FP rounding mode (fenv) -----------------------------------------------------------------
 * The SVM float ops are round-to-nearest-even only, with no rounding-mode primitive. QuickJS's
 * shortest-round-trip Number->string (`js_ecvt1`) toggles FE_DOWNWARD/FE_UPWARD to find the
 * shortest decimal; that directed-rounding path is a KNOWN FOLLOW-UP (needs an SVM rounding-mode
 * op or a directed-rounding-free dtoa). Fixed-precision paths (`toFixed`/`toPrecision`) only ever
 * use FE_TONEAREST, which these honor exactly. `<fenv.h>` on x86: FE_TONEAREST==0. */
int fegetround(void) { return 0 /* FE_TONEAREST */; }
int fesetround(int mode) { return mode == 0 ? 0 : -1 /* refuse directed modes, don't lie */; }

/* --- rounding to integer ---------------------------------------------------------------------- */
long lrint(double x) { return (long)__builtin_rint(x); }
long long llrint(double x) { return (long long)__builtin_rint(x); }

/* --- strtol (+ the C23-renamed alias glibc >=2.38 emits) -------------------------------------- */
long strtol(const char *s, char **end, int base) {
    const char *p = s;
    while (*p == ' ' || (*p >= '\t' && *p <= '\r')) p++;
    int neg = 0;
    if (*p == '+' || *p == '-') neg = (*p++ == '-');
    if ((base == 0 || base == 16) && p[0] == '0' && (p[1] == 'x' || p[1] == 'X')) {
        p += 2;
        base = 16;
    } else if (base == 0) {
        base = (p[0] == '0') ? 8 : 10;
    }
    unsigned long acc = 0;
    int any = 0;
    for (;; p++) {
        int c = (unsigned char)*p, d;
        if (c >= '0' && c <= '9') d = c - '0';
        else if (c >= 'a' && c <= 'z') d = c - 'a' + 10;
        else if (c >= 'A' && c <= 'Z') d = c - 'A' + 10;
        else break;
        if (d >= base) break;
        acc = acc * (unsigned long)base + (unsigned long)d;
        any = 1;
    }
    if (end) *end = (char *)(any ? p : s);
    long v = (long)acc;
    return neg ? -v : v;
}
long __isoc23_strtol(const char *s, char **end, int base) { return strtol(s, end, base); }

/* --- misc ------------------------------------------------------------------------------------- */
void abort(void) {
    exit(134); /* 128 + SIGABRT */
    for (;;) {
    }
}

/* QuickJS uses this only for GC memory accounting (js_malloc_usable_size) — not output-affecting.
 * The on-ramp's allocator keeps a private size header but exposes no getter, so report 0 (QuickJS
 * treats an underestimate conservatively: it may realloc sooner, never incorrectly). */
size_t malloc_usable_size(void *p) {
    (void)p;
    return 0;
}
