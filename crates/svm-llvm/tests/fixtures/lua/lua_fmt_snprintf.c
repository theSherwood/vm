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
        void *pt = va_arg(ap, void *);
        if (!pt) { const char *nil = "(nil)"; puts_n(&s, nil, 5); break; }
        char tmp[20]; int nd = utoa((unsigned long long)(size_t)pt, 16, 0, tmp);
        put(&s, '0'); put(&s, 'x'); puts_n(&s, tmp, nd);
        break;
      }
      case 'f': case 'F': case 'e': case 'E': case 'g': case 'G': {
        double x = va_arg(ap, double);
        int flags = fl_left | (fl_plus << 1) | (fl_space << 2) | ((conv >= 'A' && conv <= 'Z') << 3);
        int pr = prec < 0 ? 6 : prec;
        char tmp[512]; int n;
        if (conv == 'f' || conv == 'F') n = __vm_fmt_fix(tmp, x, pr, width, flags);
        else if (conv == 'e' || conv == 'E') n = __vm_fmt_sci(tmp, x, pr, width, flags);
        else n = __vm_fmt_gen(tmp, x, pr, width, flags);
        puts_n(&s, tmp, n);
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
