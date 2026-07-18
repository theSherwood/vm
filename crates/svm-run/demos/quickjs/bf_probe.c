/* libbf BigInt differential (ISSUES.md I25) — isolates QuickJS's BigInt path from the whole engine.
 *
 * QuickJS BigInt was miscompiled through the LLVM on-ramp: `(7n).toString()` printed garbage and
 * `6n*7n` hung. Root cause: the translator split a **large i128 constant** into `(lo, hi)` with the
 * high limb hardcoded to 0, so libbf's `udiv1norm` (which folds `2^126` as an i128 subtrahend) —
 * reached via `bf_div` ← `bf_mul_pow_radix` ← `bf_atof` (the `7n` literal parse) — produced wrong
 * quotients. This driver calls the *exact* libbf primitives QuickJS's BigInt uses (`bf_atof` to parse
 * a literal, `bf_add`/`bf_mul` at `BF_PREC_INF|BF_RNDZ`, `bf_ftoa` base-10 with the JS-quirks format),
 * on tiny integers, and prints each result — diffed byte-for-byte (guest on-ramp vs native `cc`), it
 * covers the fix end-to-end without the JS layer. Links `libbf.c` + `cutils.c` only.
 */
#include <stdio.h>
#include <stdlib.h>
#include "libbf.h"

static void *probe_realloc(void *opaque, void *ptr, size_t size) {
    (void)opaque;
    return realloc(ptr, size);
}

/* toString exactly as QuickJS's `js_bigint_to_string1` (radix 10, prec 0, JS-quirks fixed format). */
static void show(const char *label, const bf_t *a) {
    size_t len = 0;
    char *s = bf_ftoa(&len, a, 10, 0, BF_RNDZ | BF_FTOA_FORMAT_FRAC | BF_FTOA_JS_QUIRKS);
    printf("%s=%s\n", label, s ? s : "(null)");
    free(s);
}

/* Parse a decimal BigInt literal the way QuickJS does (bf_atof, BF_PREC_INF|BF_RNDZ). */
static void set_lit(bf_t *r, const char *digits) {
    bf_atof(r, digits, NULL, 10, BF_PREC_INF, BF_RNDZ);
}

int main(void) {
    bf_context_t ctx;
    bf_context_init(&ctx, probe_realloc, NULL);
    bf_t a, b, r;
    bf_init(&ctx, &a);
    bf_init(&ctx, &b);
    bf_init(&ctx, &r);

    /* literal toString — the reported `(7n).toString()` and a spread of magnitudes/signs */
    set_lit(&a, "7");                    show("seven", &a);
    set_lit(&a, "0");                    show("zero", &a);
    set_lit(&a, "-42");                  show("negforty2", &a);
    set_lit(&a, "123456789012345678901234567890"); show("thirtydigits", &a);

    /* 2n + 3n */
    set_lit(&a, "2"); set_lit(&b, "3");
    bf_add(&r, &a, &b, BF_PREC_INF, BF_RNDZ);
    show("two_plus_three", &r);

    /* 6n * 7n (the reported hang) */
    set_lit(&a, "6"); set_lit(&b, "7");
    bf_mul(&r, &a, &b, BF_PREC_INF, BF_RNDZ);
    show("six_times_seven", &r);

    /* a larger product to exercise multi-limb normalization */
    set_lit(&a, "1000000007"); set_lit(&b, "999999937");
    bf_mul(&r, &a, &b, BF_PREC_INF, BF_RNDZ);
    show("prime_product", &r);

    bf_delete(&a);
    bf_delete(&b);
    bf_delete(&r);
    bf_context_end(&ctx);
    return 0;
}
