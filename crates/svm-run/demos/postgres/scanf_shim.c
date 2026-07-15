/* Guest varargs scanf engine (slice CJ, Postgres runtime gap #11l) — the runtime
 * `sscanf`/`vsscanf`/`fscanf`/`vfscanf`/`scanf`/`vscanf` family Postgres parses config values,
 * version strings, and numeric input with (the input twin of `printf_shim.c`). A format is parsed at
 * runtime, so the on-ramp has no translate-time analog; this is guest code, matching glibc.
 *
 * One char-source abstraction serves both back ends: a string (sscanf) or a `FILE*` (fscanf, via
 * `fgetc`/`ungetc` from stdio_shim.c — so it composes with the stream/file fd-dispatch). A single
 * pushback slot is enough: scanf only ever un-reads the one char that terminated a conversion.
 * Integers are accumulated inline (sign/base/width, saturating); floats collect a token and hand it
 * to the on-ramp-synthesized `strtod`. Conversions: d/i/u/o/x/X, c, s, f/e/g/E/G/a, [scanset], n, p,
 * %%, with assignment-suppression `*`, field width, and h/hh/l/ll/L/j/z/t length modifiers.
 *
 * `#include`d into a driver under `-DSVM_GUEST`, after `os_shim.c` + `stdio_shim.c` (for `fgetc`/
 * `ungetc`). Return value: the number of input items assigned, or EOF on input failure before the
 * first conversion — glibc semantics. */
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>

extern double strtod(const char *s, char **end);

/* ---- the char source: a string or a FILE, with one pushback slot ------------------------------ */
typedef struct {
  const char *s; /* non-NULL for sscanf: the cursor */
  FILE *fp;      /* non-NULL for fscanf */
  long nread;    /* chars consumed so far (for %n and the "any input?" check) */
  int pushed;    /* -2 = empty, else a pushed-back char */
} ScanSrc;

static int sc_get(ScanSrc *S) {
  if (S->pushed != -2) {
    int c = S->pushed;
    S->pushed = -2;
    S->nread++;
    return c;
  }
  int c;
  if (S->s) {
    c = (unsigned char)*S->s;
    if (c == 0) return -1; /* EOF */
    S->s++;
  } else {
    c = fgetc(S->fp);
    if (c < 0) return -1;
  }
  S->nread++;
  return c;
}
static void sc_unget(ScanSrc *S, int c) {
  if (c >= 0) {
    S->pushed = c;
    S->nread--;
  }
}

static int sc_isspace(int c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v'; }

/* Scanset membership with range support: `a-z`/`0-9` expand to ranges (glibc), but a `-` that is
 * first or last in the set is a literal. */
static int sc_in_set(int c, const char *set, int setlen) {
  for (int i = 0; i < setlen; i++) {
    if (set[i] == '-' && i > 0 && i + 1 < setlen) {
      unsigned lo = (unsigned char)set[i - 1], hi = (unsigned char)set[i + 1];
      if ((unsigned)c >= lo && (unsigned)c <= hi) return 1;
      i++; /* consume the range's upper bound (also matched as a literal, harmlessly) */
    } else if ((unsigned char)set[i] == (unsigned)c) {
      return 1;
    }
  }
  return 0;
}
static int sc_digit(int c, int base) {
  int v;
  if (c >= '0' && c <= '9') v = c - '0';
  else if (c >= 'a' && c <= 'z') v = c - 'a' + 10;
  else if (c >= 'A' && c <= 'Z') v = c - 'A' + 10;
  else return -1;
  return v < base ? v : -1;
}

/* Skip input whitespace; returns the count skipped (unused) — leaves the next non-space pushed back. */
static void sc_skipws(ScanSrc *S) {
  int c;
  while ((c = sc_get(S)) >= 0 && sc_isspace(c)) {
  }
  sc_unget(S, c);
}

/* The integer store is inlined at the one call site (below) with `va_arg(ap, …)` directly: passing a
 * `va_list *` to a helper trips the on-ramp's varargs ABI, so — like `printf_shim.c` — every `va_arg`
 * happens in the function that owns the `va_list`. lm: -2=hh -1=h 0=int 1=l 2=ll/L/j/z/t. */

static int vsscanf_impl(ScanSrc *S, const char *fmt, va_list ap);

int vsscanf(const char *str, const char *fmt, va_list ap) {
  ScanSrc S = {str, NULL, 0, -2};
  return vsscanf_impl(&S, fmt, ap);
}
int sscanf(const char *str, const char *fmt, ...) {
  va_list ap; va_start(ap, fmt);
  ScanSrc S = {str, NULL, 0, -2};
  int r = vsscanf_impl(&S, fmt, ap);
  va_end(ap);
  return r;
}
int vfscanf(FILE *fp, const char *fmt, va_list ap) {
  ScanSrc S = {NULL, fp, 0, -2};
  return vsscanf_impl(&S, fmt, ap);
}
int fscanf(FILE *fp, const char *fmt, ...) {
  va_list ap; va_start(ap, fmt);
  ScanSrc S = {NULL, fp, 0, -2};
  int r = vsscanf_impl(&S, fmt, ap);
  va_end(ap);
  return r;
}
int vscanf(const char *fmt, va_list ap) { return vfscanf(stdin, fmt, ap); }
int scanf(const char *fmt, ...) {
  va_list ap; va_start(ap, fmt);
  int r = vfscanf(stdin, fmt, ap);
  va_end(ap);
  return r;
}

/* glibc's C23 build renames the scanf-family callers to `__isoc23_*` (the C23 `%b`/no-`%'` grouping
 * semantics — identical for the formats Postgres uses), so a whole-program build references those
 * names. Forward each to the plain implementation. Postgres' `ValidatePgVersion` reads `PG_VERSION`
 * with `__isoc23_fscanf`. */
int __isoc23_vsscanf(const char *str, const char *fmt, va_list ap) { return vsscanf(str, fmt, ap); }
int __isoc23_sscanf(const char *str, const char *fmt, ...) {
  va_list ap; va_start(ap, fmt);
  ScanSrc S = {str, NULL, 0, -2};
  int r = vsscanf_impl(&S, fmt, ap);
  va_end(ap);
  return r;
}
int __isoc23_vfscanf(FILE *fp, const char *fmt, va_list ap) { return vfscanf(fp, fmt, ap); }
int __isoc23_fscanf(FILE *fp, const char *fmt, ...) {
  va_list ap; va_start(ap, fmt);
  ScanSrc S = {NULL, fp, 0, -2};
  int r = vsscanf_impl(&S, fmt, ap);
  va_end(ap);
  return r;
}
int __isoc23_vscanf(const char *fmt, va_list ap) { return vfscanf(stdin, fmt, ap); }
int __isoc23_scanf(const char *fmt, ...) {
  va_list ap; va_start(ap, fmt);
  int r = vfscanf(stdin, fmt, ap);
  va_end(ap);
  return r;
}

static int vsscanf_impl(ScanSrc *S, const char *fmt, va_list ap) {
  int assigned = 0;   /* successful assignments (the return value) */
  int any_conv = 0;   /* did we reach at least one conversion? (EOF-vs-0 distinction) */
  for (const char *p = fmt; *p;) {
    if (sc_isspace((unsigned char)*p)) { /* whitespace: skip run of input whitespace */
      sc_skipws(S);
      p++;
      continue;
    }
    if (*p != '%') { /* literal: must match exactly */
      int c = sc_get(S);
      if (c != (unsigned char)*p) { sc_unget(S, c); return assigned ? assigned : (c < 0 && !any_conv ? EOF : assigned); }
      p++;
      continue;
    }
    /* --- a conversion ------------------------------------------------------------------------- */
    p++;
    if (*p == '%') { /* %% matches a literal % (after optional whitespace? no — exact) */
      int c = sc_get(S);
      if (c != '%') { sc_unget(S, c); return assigned; }
      p++;
      continue;
    }
    int suppress = 0;
    if (*p == '*') { suppress = 1; p++; }
    int width = 0, have_width = 0;
    while (*p >= '0' && *p <= '9') { width = width * 10 + (*p++ - '0'); have_width = 1; }
    int lm = 0;
    for (;;) {
      if (*p == 'h') { lm = (lm == -1) ? -2 : -1; p++; }
      else if (*p == 'l') { lm = (lm == 1) ? 2 : 1; p++; }
      else if (*p == 'L' || *p == 'j' || *p == 'z' || *p == 't') { lm = 2; p++; }
      else break;
    }
    char conv = *p ? *p++ : 0;
    any_conv = 1;

    if (conv == 'n') { /* chars consumed so far — no input, not counted */
      if (!suppress) { int *np = va_arg(ap, int *); *np = (int)S->nread; }
      continue;
    }

    /* c / [ do NOT skip leading whitespace; the numeric + s conversions do. */
    if (conv != 'c' && conv != '[') sc_skipws(S);

    if (conv == 'c') {
      int n = have_width ? width : 1;
      char *out = suppress ? NULL : va_arg(ap, char *);
      int got = 0, c;
      while (got < n && (c = sc_get(S)) >= 0) { if (out) out[got] = (char)c; got++; }
      if (got < n) { return assigned ? assigned : (got == 0 ? EOF : assigned); } /* short read = failure */
      if (!suppress) assigned++;
      continue;
    }

    if (conv == 's') {
      int n = have_width ? width : 0x7fffffff;
      char *out = suppress ? NULL : va_arg(ap, char *);
      int got = 0, c;
      while (got < n && (c = sc_get(S)) >= 0) {
        if (sc_isspace(c)) { sc_unget(S, c); break; }
        if (out) out[got] = (char)c;
        got++;
      }
      if (got == 0) return assigned ? assigned : EOF; /* no non-space char = input failure */
      if (out) out[got] = 0;
      if (!suppress) assigned++;
      continue;
    }

    if (conv == '[') { /* scanset: %[...] / %[^...] */
      int negate = 0;
      if (*p == '^') { negate = 1; p++; }
      const char *set = p;
      if (*p == ']') p++; /* a leading ] is a member */
      while (*p && *p != ']') p++;
      int setlen = (int)(p - set);
      if (*p == ']') p++;
      int n = have_width ? width : 0x7fffffff;
      char *out = suppress ? NULL : va_arg(ap, char *);
      int got = 0, c;
      while (got < n && (c = sc_get(S)) >= 0) {
        int inset = sc_in_set(c, set, setlen);
        if (inset == negate) { sc_unget(S, c); break; }
        if (out) out[got] = (char)c;
        got++;
      }
      if (got == 0) return assigned ? assigned : EOF;
      if (out) out[got] = 0;
      if (!suppress) assigned++;
      continue;
    }

    if (conv == 'd' || conv == 'i' || conv == 'u' || conv == 'o' || conv == 'x' || conv == 'X' || conv == 'p') {
      int base = conv == 'o' ? 8 : (conv == 'x' || conv == 'X' || conv == 'p') ? 16 : (conv == 'i' ? 0 : 10);
      int is_signed = (conv == 'd' || conv == 'i');
      int lim = have_width ? width : 0x7fffffff, used = 0, c;
      int neg = 0;
      c = sc_get(S); used++;
      if (c == '+' || c == '-') { neg = (c == '-'); if (used >= lim) { sc_unget(S, c); return assigned ? assigned : EOF; } c = sc_get(S); used++; }
      /* base 0 / hex: optional 0x prefix */
      if ((base == 0 || base == 16) && c == '0') {
        int c2 = (used < lim) ? sc_get(S) : -1;
        if (c2 == 'x' || c2 == 'X') { base = 16; used++; c = (used < lim) ? sc_get(S) : -1; used++; }
        else { if (base == 0) base = 8; sc_unget(S, c2); /* the leading 0 is a valid digit */ }
      }
      if (base == 0) base = 10;
      unsigned long long v = 0; int ndig = 0;
      while (c >= 0 && used <= lim) {
        int d = sc_digit(c, base);
        if (d < 0) break;
        v = v * (unsigned)base + (unsigned)d;
        ndig++;
        if (used >= lim) { c = -2; break; } /* width exhausted; don't over-read */
        c = sc_get(S); used++;
      }
      if (c != -2) sc_unget(S, c);
      if (ndig == 0) return assigned ? assigned : EOF; /* matching failure */
      if (neg) v = (unsigned long long)(-(long long)v);
      (void)is_signed;
      if (!suppress) {
        if (lm <= -2) { signed char *pp = va_arg(ap, signed char *); *pp = (signed char)v; }
        else if (lm == -1) { short *pp = va_arg(ap, short *); *pp = (short)v; }
        else if (lm == 0) { int *pp = va_arg(ap, int *); *pp = (int)v; }
        else if (lm == 1) { long *pp = va_arg(ap, long *); *pp = (long)v; }
        else { long long *pp = va_arg(ap, long long *); *pp = (long long)v; }
        assigned++;
      }
      continue;
    }

    if (conv == 'f' || conv == 'e' || conv == 'g' || conv == 'E' || conv == 'G' || conv == 'a' || conv == 'A') {
      /* Collect a float token, then hand it to strtod. */
      char tok[64]; int ti = 0, lim = have_width ? width : 62, used = 0, c;
      int seen_digit = 0, seen_dot = 0, seen_exp = 0;
      c = sc_get(S); used++;
      if (c == '+' || c == '-') { if (ti < 62) tok[ti++] = (char)c; c = (used < lim) ? sc_get(S) : -1; used++; }
      while (c >= 0 && used <= lim + 1 && ti < 62) {
        if (c >= '0' && c <= '9') { seen_digit = 1; }
        else if (c == '.' && !seen_dot && !seen_exp) { seen_dot = 1; }
        else if ((c == 'e' || c == 'E') && seen_digit && !seen_exp) {
          seen_exp = 1; tok[ti++] = (char)c;
          c = (used < lim) ? sc_get(S) : -1; used++;
          if (c == '+' || c == '-') { if (ti < 62) tok[ti++] = (char)c; c = (used < lim) ? sc_get(S) : -1; used++; }
          continue;
        } else break;
        tok[ti++] = (char)c;
        if (used >= lim) { c = -2; break; }
        c = sc_get(S); used++;
      }
      if (c != -2) sc_unget(S, c);
      if (!seen_digit) return assigned ? assigned : EOF;
      tok[ti] = 0;
      double d = strtod(tok, NULL);
      if (!suppress) {
        if (lm >= 1) { double *dp = va_arg(ap, double *); *dp = d; }
        else { float *fp = va_arg(ap, float *); *fp = (float)d; }
        assigned++;
      }
      continue;
    }

    /* unknown conversion: stop (glibc treats it as a format error) */
    return assigned;
  }
  return assigned;
}
