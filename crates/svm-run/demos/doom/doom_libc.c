/* Doom's libc shim (Doom slice 3b): the freestanding-libc functions the on-ramp doesn't synthesize
 * and the Lua `fs`/format shims don't already provide. Same "program brings its own libc" model as
 * Lua/SQLite: system headers declare these; this file defines them. The on-ramp already provides
 * malloc/calloc/realloc/free/memcpy/memset and the write/read/exit/abort/vm_map capabilities; the
 * reused lua_files_stdio.c provides the FILE layer over the `fs` capability (fopen/fread/fwrite/
 * fseek/ftell/fclose/fprintf/putc/__errno_location/std streams); lua_fmt_snprintf.c provides the
 * printf format engine (snprintf/vsnprintf). This file adds the rest Doom references. */
typedef __builtin_va_list va_list;
typedef struct FILE FILE; /* opaque — the real definition lives in lua_files_stdio.c */

extern long write(int fd, const void *buf, long len);          /* stdout Stream (on-ramp) */
extern void *malloc(unsigned long n);                          /* on-ramp bump allocator */
extern int vsnprintf(char *buf, unsigned long size, const char *fmt, va_list ap); /* lua_fmt */
extern unsigned long fwrite(const void *ptr, unsigned long size, unsigned long nmemb, FILE *f); /* lua_files */
extern FILE *stdout;
extern FILE *stderr;

/* ---- string.h ------------------------------------------------------------------------------------ */
unsigned long strlen(const char *s) { unsigned long n = 0; while (s[n]) n++; return n; }
int strcmp(const char *a, const char *b) {
  while (*a && *a == *b) { a++; b++; }
  return (int)(unsigned char)*a - (int)(unsigned char)*b;
}
int strncmp(const char *a, const char *b, unsigned long n) {
  for (unsigned long i = 0; i < n; i++) {
    unsigned char ca = a[i], cb = b[i];
    if (ca != cb) return (int)ca - (int)cb;
    if (!ca) return 0;
  }
  return 0;
}
char *strncpy(char *d, const char *s, unsigned long n) {
  unsigned long i = 0;
  for (; i < n && s[i]; i++) d[i] = s[i];
  for (; i < n; i++) d[i] = 0;
  return d;
}
char *strrchr(const char *s, int c) {
  const char *last = 0;
  do { if (*s == (char)c) last = s; } while (*s++);
  return (char *)last;
}
char *strchr(const char *s, int c) {
  for (;; s++) { if (*s == (char)c) return (char *)s; if (!*s) return 0; }
}
char *strstr(const char *h, const char *n) {
  if (!*n) return (char *)h;
  for (; *h; h++) {
    const char *a = h, *b = n;
    while (*a && *b && *a == *b) { a++; b++; }
    if (!*b) return (char *)h;
  }
  return 0;
}
void *memchr(const void *s, int c, unsigned long n) {
  const unsigned char *p = (const unsigned char *)s;
  for (unsigned long i = 0; i < n; i++) if (p[i] == (unsigned char)c) return (void *)(p + i);
  return 0;
}
int bcmp(const void *a, const void *b, unsigned long n) {
  const unsigned char *x = (const unsigned char *)a, *y = (const unsigned char *)b;
  for (unsigned long i = 0; i < n; i++) if (x[i] != y[i]) return 1;
  return 0;
}
static int lc(int c) { return (c >= 'A' && c <= 'Z') ? c + 32 : c; }
int strcasecmp(const char *a, const char *b) {
  while (*a && lc((unsigned char)*a) == lc((unsigned char)*b)) { a++; b++; }
  return lc((unsigned char)*a) - lc((unsigned char)*b);
}
int strncasecmp(const char *a, const char *b, unsigned long n) {
  for (unsigned long i = 0; i < n; i++) {
    int ca = lc((unsigned char)a[i]), cb = lc((unsigned char)b[i]);
    if (ca != cb) return ca - cb;
    if (!ca) return 0;
  }
  return 0;
}
char *strdup(const char *s) {
  unsigned long n = strlen(s) + 1;
  char *p = (char *)malloc(n);
  if (p) for (unsigned long i = 0; i < n; i++) p[i] = s[i];
  return p;
}

/* ---- ctype.h ------------------------------------------------------------------------------------- */
int toupper(int c) { return (c >= 'a' && c <= 'z') ? c - 32 : c; }
int tolower(int c) { return (c >= 'A' && c <= 'Z') ? c + 32 : c; }
/* glibc's `toupper(c)` macro expands to `(*__ctype_toupper_loc())[c]` with c in [-128, 255]. */
static int toupper_tab[384];
static int *toupper_ptr;
static int toupper_init;
int **__ctype_toupper_loc(void) {
  if (!toupper_init) {
    for (int i = 0; i < 384; i++) { int c = i - 128; toupper_tab[i] = (c >= 'a' && c <= 'z') ? c - 32 : c; }
    toupper_ptr = &toupper_tab[128];
    toupper_init = 1;
  }
  return &toupper_ptr;
}

/* ---- stdlib.h ------------------------------------------------------------------------------------ */
int abs(int x) { return x < 0 ? -x : x; }
long labs(long x) { return x < 0 ? -x : x; }
static int isspace_(int c) { return c == ' ' || (c >= '\t' && c <= '\r'); }
/* strtol with base 0/8/10/16 detection (Doom passes base 10 and 16). */
long strtol(const char *s, char **end, int base) {
  const char *p = s;
  while (isspace_(*p)) p++;
  int neg = 0;
  if (*p == '+' || *p == '-') { neg = (*p == '-'); p++; }
  if ((base == 0 || base == 16) && p[0] == '0' && (p[1] == 'x' || p[1] == 'X')) { p += 2; base = 16; }
  else if (base == 0 && p[0] == '0') { base = 8; }
  else if (base == 0) base = 10;
  long v = 0;
  for (;; p++) {
    int c = *p, d;
    if (c >= '0' && c <= '9') d = c - '0';
    else if (c >= 'a' && c <= 'z') d = c - 'a' + 10;
    else if (c >= 'A' && c <= 'Z') d = c - 'A' + 10;
    else break;
    if (d >= base) break;
    v = v * base + d;
  }
  if (end) *end = (char *)p;
  return neg ? -v : v;
}
int atoi(const char *s) { return (int)strtol(s, (char **)0, 10); }
/* strtod — Doom's fixed-point renderer is integer, but a few config/DEH paths touch doubles. Handles
 * an optional sign, integer + fractional parts, and a base-10 exponent (enough for Doom's inputs). */
double strtod(const char *s, char **end) {
  const char *p = s;
  while (isspace_(*p)) p++;
  int neg = 0;
  if (*p == '+' || *p == '-') { neg = (*p == '-'); p++; }
  double v = 0.0;
  for (; *p >= '0' && *p <= '9'; p++) v = v * 10.0 + (*p - '0');
  if (*p == '.') {
    p++;
    double f = 0.1;
    for (; *p >= '0' && *p <= '9'; p++) { v += (*p - '0') * f; f *= 0.1; }
  }
  if (*p == 'e' || *p == 'E') {
    p++;
    int eneg = 0;
    if (*p == '+' || *p == '-') { eneg = (*p == '-'); p++; }
    int e = 0;
    for (; *p >= '0' && *p <= '9'; p++) e = e * 10 + (*p - '0');
    double m = 1.0;
    while (e--) m *= 10.0;
    v = eneg ? v / m : v * m;
  }
  if (end) *end = (char *)p;
  return neg ? -v : v;
}
int system(const char *cmd) { (void)cmd; return -1; }  /* no shell in the sandbox */
int mkdir(const char *path, unsigned int mode) { (void)path; (void)mode; return 0; } /* no-op dir create */

/* ---- stdio.h (the pieces lua_files_stdio.c doesn't define) --------------------------------------- */
int printf(const char *fmt, ...) {
  char buf[1024];
  va_list ap; __builtin_va_start(ap, fmt);
  int n = vsnprintf(buf, sizeof buf, fmt, ap);
  __builtin_va_end(ap);
  if (n > 0) write(1, buf, n > (int)sizeof buf ? (int)sizeof buf : n);
  return n;
}
int puts(const char *s) { write(1, s, (long)strlen(s)); write(1, "\n", 1); return 0; }
int vfprintf(FILE *f, const char *fmt, va_list ap) {
  char buf[1024];
  int n = vsnprintf(buf, sizeof buf, fmt, ap);
  if (n > 0) fwrite(buf, 1, (unsigned long)(n > (int)sizeof buf ? (int)sizeof buf : n), f);
  return n;
}
int fputs(const char *s, FILE *f); /* defined in lua_files_stdio.c */

/* ---- sscanf: single-integer scans only (Doom's m_config.c: "%d"/"%i"/"%x"/" 0x%x"/" 0%o") -------- */
static int scan_int(const char **pp, int base, long *out) {
  const char *p = *pp;
  while (isspace_(*p)) p++;
  int neg = 0;
  if (*p == '+' || *p == '-') { neg = (*p == '-'); p++; }
  if ((base == 0 || base == 16) && p[0] == '0' && (p[1] == 'x' || p[1] == 'X')) { p += 2; base = 16; }
  else if (base == 0 && p[0] == '0') base = 8;
  else if (base == 0) base = 10;
  const char *start = p;
  long v = 0;
  for (;; p++) {
    int c = *p, d;
    if (c >= '0' && c <= '9') d = c - '0';
    else if (c >= 'a' && c <= 'f') d = c - 'a' + 10;
    else if (c >= 'A' && c <= 'F') d = c - 'A' + 10;
    else break;
    if (d >= base) break;
    v = v * base + d;
  }
  if (p == start) return 0; /* no digits consumed */
  *out = neg ? -v : v;
  *pp = p;
  return 1;
}
static int vsscanf_ints(const char *in, const char *fmt, va_list ap) {
  int assigned = 0;
  const char *ip = in;
  for (const char *fp = fmt; *fp; fp++) {
    if (isspace_(*fp)) { while (isspace_(*ip)) ip++; continue; }
    if (*fp == '%') {
      fp++;
      int base = 10;
      if (*fp == 'i') base = 0;
      else if (*fp == 'x' || *fp == 'X') base = 16;
      else if (*fp == 'o') base = 8;
      else if (*fp == 'u' || *fp == 'd') base = 10;
      else continue; /* unsupported conversion — skip (Doom only uses the integer set) */
      long v;
      if (!scan_int(&ip, base, &v)) return assigned;
      *__builtin_va_arg(ap, int *) = (int)v;
      assigned++;
    } else {
      /* a literal in the format: skip a matching input char (whitespace already handled) */
      if (*ip == *fp) ip++;
    }
  }
  return assigned;
}
int sscanf(const char *in, const char *fmt, ...) {
  va_list ap; __builtin_va_start(ap, fmt);
  int r = vsscanf_ints(in, fmt, ap);
  __builtin_va_end(ap);
  return r;
}
int __isoc99_sscanf(const char *in, const char *fmt, ...) {
  va_list ap; __builtin_va_start(ap, fmt);
  int r = vsscanf_ints(in, fmt, ap);
  __builtin_va_end(ap);
  return r;
}

/* ---- netgame globals Doom references but whose defining TUs are excluded (single-player) ---------- */
int drone = 0;
int net_client_connected = 0;
