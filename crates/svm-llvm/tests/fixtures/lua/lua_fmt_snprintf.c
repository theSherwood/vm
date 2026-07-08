/* Guest snprintf/vsnprintf: a runtime printf that a program (Lua's string.format) links to shadow the
 * on-ramp's constant-format snprintf. Integers/strings/chars are formatted in C (matching glibc);
 * floats delegate to the on-ramp's correctly-rounded bignum dtoa via __vm_fmt_{fix,sci,gen}. Single
 * definition covers both the Lua core's constant formats (%lld/%.14g) and string.format's runtime ones. */
#include <stdarg.h>
#include <stddef.h>

extern int __vm_fmt_fix(char *o, double x, int prec, int width, int flags);
extern int __vm_fmt_sci(char *o, double x, int prec, int width, int flags);
extern int __vm_fmt_gen(char *o, double x, int prec, int width, int flags);

typedef struct { char *buf; size_t size; size_t total; } Sink;
static void put(Sink *s, char c) { if (s->total + 1 < s->size) s->buf[s->total] = c; s->total++; }
static void puts_n(Sink *s, const char *p, int n) { for (int i = 0; i < n; i++) put(s, p[i]); }
static void pad(Sink *s, char c, int n) { for (int i = 0; i < n; i++) put(s, c); }

/* format an unsigned magnitude in `base` into `tmp` (reversed then returned in-order); returns #digits */
static int utoa(unsigned long long v, int base, int upper, char *tmp) {
  const char *dig = upper ? "0123456789ABCDEF" : "0123456789abcdef";
  char rev[24]; int n = 0;
  do { rev[n++] = dig[v % base]; v /= base; } while (v);
  for (int i = 0; i < n; i++) tmp[i] = rev[n - 1 - i];
  return n;
}

int vsnprintf(char *buf, size_t size, const char *fmt, va_list ap) {
  Sink s = { buf, size, 0 };
  for (const char *p = fmt; *p;) {
    if (*p != '%') { put(&s, *p++); continue; }
    p++;
    if (*p == '%') { put(&s, '%'); p++; continue; }
    int fl_left = 0, fl_plus = 0, fl_space = 0, fl_alt = 0, fl_zero = 0;
    for (;; p++) {
      if (*p == '-') fl_left = 1; else if (*p == '+') fl_plus = 1;
      else if (*p == ' ') fl_space = 1; else if (*p == '#') fl_alt = 1;
      else if (*p == '0') fl_zero = 1; else break;
    }
    int width = 0;
    if (*p == '*') { width = va_arg(ap, int); p++; if (width < 0) { fl_left = 1; width = -width; } }
    else while (*p >= '0' && *p <= '9') width = width * 10 + (*p++ - '0');
    int prec = -1;
    if (*p == '.') { p++; prec = 0; if (*p == '*') { prec = va_arg(ap, int); p++; } else while (*p >= '0' && *p <= '9') prec = prec * 10 + (*p++ - '0'); }
    int lm = 0; /* long-ness: +1 per 'l', -1 per 'h' */
    for (;;) { if (*p == 'l') { lm++; p++; } else if (*p == 'h') { lm--; p++; } else if (*p == 'L' || *p == 'j' || *p == 'z' || *p == 't') { lm = 2; p++; } else break; }
    char conv = *p ? *p++ : 0;
    if (fl_left) fl_zero = 0;
    if (fl_plus) fl_space = 0;
    switch (conv) {
      case 'd': case 'i': {
        long long v = (lm >= 2) ? va_arg(ap, long long) : (lm == 1) ? va_arg(ap, long) : va_arg(ap, int);
        unsigned long long mag = v < 0 ? (unsigned long long)(-(v + 1)) + 1 : (unsigned long long)v;
        char tmp[24]; int nd = utoa(mag, 10, 0, tmp);
        if (prec == 0 && v == 0) nd = 0; /* ISO C: zero with zero precision prints no digits */
        char sign = v < 0 ? '-' : fl_plus ? '+' : fl_space ? ' ' : 0;
        if (prec >= 0) fl_zero = 0;
        int zpad = (prec > nd) ? prec - nd : 0;                 /* precision-driven zeros */
        int body = (sign ? 1 : 0) + zpad + nd;
        int spad = (width > body) ? width - body : 0;           /* field padding */
        if (!fl_left && !fl_zero) pad(&s, ' ', spad);
        if (sign) put(&s, sign);
        if (!fl_left && fl_zero) pad(&s, '0', spad);
        pad(&s, '0', zpad); puts_n(&s, tmp, nd);
        if (fl_left) pad(&s, ' ', spad);
        break;
      }
      case 'u': case 'o': case 'x': case 'X': {
        int base = conv == 'o' ? 8 : (conv == 'u' ? 10 : 16);
        unsigned long long v = (lm >= 2) ? va_arg(ap, unsigned long long) : (lm == 1) ? va_arg(ap, unsigned long) : va_arg(ap, unsigned int);
        char tmp[24]; int nd = utoa(v, base, conv == 'X', tmp);
        if (prec == 0 && v == 0) nd = 0; /* ISO C: zero with zero precision prints no digits */
        if (prec >= 0) fl_zero = 0;
        int zpad = (prec > nd) ? prec - nd : 0;
        char pfx[2]; int npfx = 0;
        if (fl_alt && conv == 'o' && (zpad + nd == 0 || tmp[0] != '0')) { pfx[npfx++] = '0'; }
        if (fl_alt && (conv == 'x' || conv == 'X') && v != 0) { pfx[npfx++] = '0'; pfx[npfx++] = conv == 'X' ? 'X' : 'x'; }
        int body = npfx + zpad + nd;
        int spad = (width > body) ? width - body : 0;
        if (!fl_left && !fl_zero) pad(&s, ' ', spad);
        puts_n(&s, pfx, npfx);
        if (!fl_left && fl_zero) pad(&s, '0', spad);
        pad(&s, '0', zpad); puts_n(&s, tmp, nd);
        if (fl_left) pad(&s, ' ', spad);
        break;
      }
      case 'c': {
        int ch = va_arg(ap, int);
        int spad = width > 1 ? width - 1 : 0;
        if (!fl_left) pad(&s, ' ', spad);
        put(&s, (char)ch);
        if (fl_left) pad(&s, ' ', spad);
        break;
      }
      case 's': {
        const char *str = va_arg(ap, const char *);
        if (!str) str = "(null)";
        int n = 0; while (str[n] && (prec < 0 || n < prec)) n++;
        int spad = width > n ? width - n : 0;
        if (!fl_left) pad(&s, ' ', spad);
        puts_n(&s, str, n);
        if (fl_left) pad(&s, ' ', spad);
        break;
      }
      case 'p': {
        /* glibc shape: "0x<hex>" (or "(nil)"), honoring width + '-' with space padding — Lua's
         * string.format("%90p" / "%-60p", …) (strings.lua) observes exactly that. */
        void *pt = va_arg(ap, void *);
        char tok[24]; int n = 0;
        if (!pt) {
          const char *nil = "(nil)";
          while (nil[n]) { tok[n] = nil[n]; n++; }
        } else {
          char tmp[20]; int nd = utoa((unsigned long long)(size_t)pt, 16, 0, tmp);
          tok[n++] = '0'; tok[n++] = 'x';
          for (int i = 0; i < nd; i++) tok[n++] = tmp[i];
        }
        int spad = width > n ? width - n : 0;
        if (!fl_left) pad(&s, ' ', spad);
        puts_n(&s, tok, n);
        if (fl_left) pad(&s, ' ', spad);
        break;
      }
      case 'a': case 'A': {
        /* C99 hex-float: sign (honoring `+`/space), "0x<lead>.<mantissa nibbles>p" + signed decimal
         * exponent. No precision → exact nibbles, trailing zeros trimmed (glibc); a precision `p`
         * rounds the mantissa to `p` nibbles, round-half-to-even, carry into the lead digit
         * (0x1.f8p+0 at .1 → 0x2.0p+0, like glibc). Subnormals lead "0x0." (exp -1022), zero is
         * "0x0p+0". Lua needs both shapes: `%q` floats (`lua_number2strx`, no precision, must
         * round-trip) and strings.lua's `%+.2A`-style modifier checks. */
        double x = va_arg(ap, double);
        union { double d; unsigned long long u; } pun;
        pun.d = x;
        unsigned long long bits = pun.u;
        int up = conv == 'A';
        const char *dig = up ? "0123456789ABCDEF" : "0123456789abcdef";
        if (bits >> 63) put(&s, '-');
        else if (fl_plus) put(&s, '+');
        else if (fl_space) put(&s, ' ');
        unsigned long long mant = bits & 0xfffffffffffffULL;
        int be = (int)((bits >> 52) & 0x7ff);
        if (be == 0x7ff) { /* inf/nan */
          const char *t = mant ? (up ? "NAN" : "nan") : (up ? "INF" : "inf");
          while (*t) put(&s, *t++);
          break;
        }
        put(&s, '0'); put(&s, up ? 'X' : 'x');
        int e2 = be == 0 ? (mant ? -1022 : 0) : be - 1023;
        unsigned long long lead = be == 0 ? 0 : 1;
        int nfrac; /* fraction nibbles to print */
        char nib[16];
        if (prec >= 0) {
          int keep = prec < 13 ? prec : 13;
          int drop = (13 - keep) * 4;
          unsigned long long full = (lead << 52) | mant;
          unsigned long long kept = drop ? full >> drop : full;
          if (drop) { /* round half to even on the dropped bits */
            unsigned long long rem = full & ((1ULL << drop) - 1);
            unsigned long long half = 1ULL << (drop - 1);
            if (rem > half || (rem == half && (kept & 1))) kept++;
          }
          lead = kept >> (4 * keep);
          for (int i = 0; i < keep; i++) nib[i] = dig[(kept >> (4 * (keep - 1 - i))) & 0xf];
          for (int i = keep; i < prec && i < 15; i++) nib[i] = '0'; /* pad past exact digits */
          nfrac = prec < 15 ? prec : 15;
        } else {
          nfrac = 0;
          if (mant) {
            for (int i = 0; i < 13; i++) nib[i] = dig[(mant >> (48 - 4 * i)) & 0xf];
            nfrac = 13;
            while (nfrac > 1 && nib[nfrac - 1] == '0') nfrac--;
          }
        }
        put(&s, dig[lead & 0xf]);
        if (nfrac) {
          put(&s, '.');
          puts_n(&s, nib, nfrac);
        }
        put(&s, up ? 'P' : 'p');
        put(&s, e2 < 0 ? '-' : '+');
        char tmp[8]; int nd = utoa((unsigned long long)(e2 < 0 ? -e2 : e2), 10, 0, tmp);
        puts_n(&s, tmp, nd);
        break;
      }
      case 'f': case 'F': case 'e': case 'E': case 'g': case 'G': {
        /* The bignum helpers produce the unpadded "[sign]digits…" content (width 0); the ISO flag
         * behavior — `0` zero-padding after the sign, `#` keeping the point at precision 0, field
         * width/justification — is applied here, where the flags are known. */
        double x = va_arg(ap, double);
        int flags = (fl_plus << 1) | (fl_space << 2) | ((conv >= 'A' && conv <= 'Z') << 3);
        int pr = prec < 0 ? 6 : prec;
        char tmp[512]; int n;
        if (conv == 'f' || conv == 'F') n = __vm_fmt_fix(tmp, x, pr, 0, flags);
        else if (conv == 'e' || conv == 'E') n = __vm_fmt_sci(tmp, x, pr, 0, flags);
        else n = __vm_fmt_gen(tmp, x, pr, 0, flags);
        if (fl_alt && pr == 0 && (conv == 'f' || conv == 'F') && n < (int)sizeof tmp)
          tmp[n++] = '.'; /* %#.0f keeps the point */
        int sign = (n > 0 && (tmp[0] == '-' || tmp[0] == '+' || tmp[0] == ' ')) ? 1 : 0;
        int spad = width > n ? width - n : 0;
        if (fl_left) {
          puts_n(&s, tmp, n);
          pad(&s, ' ', spad);
        } else if (fl_zero) {
          if (sign) put(&s, tmp[0]);
          pad(&s, '0', spad);
          puts_n(&s, tmp + sign, n - sign);
        } else {
          pad(&s, ' ', spad);
          puts_n(&s, tmp, n);
        }
        break;
      }
      default: put(&s, '%'); if (conv) put(&s, conv); break;
    }
  }
  if (size > 0) buf[s.total < size ? s.total : size - 1] = 0;
  return (int)s.total;
}
int snprintf(char *buf, size_t size, const char *fmt, ...) {
  va_list ap; va_start(ap, fmt);
  int r = vsnprintf(buf, size, fmt, ap);
  va_end(ap);
  return r;
}
