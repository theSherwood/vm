/* Guest stdio (FILE) layer over the **configurable Fs capability** — the "program brings its own
 * libc" model, with the storage backend dependency-injected at the capability boundary: every real
 * file op funnels to `__vm_host_call(fs, op, …)` against whatever backend the embedder granted under
 * the name "fs" (svm-run's in-memory `fs::mem_fs` or the rooted real-directory `fs::host_fs` — same
 * protocol, so this layer is backend-agnostic and byte-identical across runs). The std streams stay
 * on the powerbox `Stream` capability: `stdout`/`stderr` writes route to the raw `write(buf, len)`
 * import, `stdin` reads to `read(buf, len)` — the same funnel the on-ramp's own `printf` lowering
 * uses, so `print` and `io.write` interleave correctly.
 *
 * FILE semantics kept: mode parsing (`r`/`w`/`a`/`+`/`b`), a 1-slot `ungetc` pushback, EOF/error
 * flags, `fseek`/`ftell` (64-bit variants included), **setvbuf-honoring write buffering** (full by
 * default like glibc, line/none on request — files.lua observes the visibility differences through
 * a second reader, so this is load-bearing), and a minimal `fprintf` (vsnprintf → fwrite; Lua's
 * `f:write(number)` path). Reads are unbuffered (exactly ordered through the capability). */
#include <stddef.h>

/* the raw powerbox Stream externs, POSIX-shaped: the on-ramp drops the `fd` (the granted handle
 * selects the endpoint) and lowers the rest to Stream write/read */
extern long write(int fd, const void *buf, long len);
extern long read(int fd, void *buf, long len);
extern unsigned long strlen(const char *s);
extern int __vm_cap_resolve(const char *name, long len);
extern long __vm_host_call(int h, int op, long a, long b, long c, long d);
typedef __builtin_va_list va_list;
extern int vsnprintf(char *buf, unsigned long size, const char *fmt, va_list ap);

enum { FS_OPEN = 0, FS_READ = 1, FS_WRITE = 2, FS_SEEK = 3, FS_CLOSE = 4, FS_REMOVE = 5, FS_RENAME = 6 };
enum { O_READ = 1, O_WRITE = 2, O_APPEND = 4, O_TRUNC = 8, O_CREATE = 16 };

/* errno: liolib/loslib read it via <errno.h>'s __errno_location() */
static int errno_cell;
int *__errno_location(void) { return &errno_cell; }

static int fs_cap = -2; /* -2 = unresolved; resolved lazily so pure-stdio programs need no fs cap */
static int fs(void) {
  if (fs_cap == -2) fs_cap = __vm_cap_resolve("fs", 2);
  return fs_cap;
}
static long hc(int op, long a, long b, long c, long d) {
  long r = __vm_host_call(fs(), op, a, b, c, d);
  if (r < 0) errno_cell = (int)-r;
  return r;
}

#define WBUF_CAP 2048
typedef struct FILE {
  int used;     /* pool slot in use */
  int fd;       /* capability fd, or -1/-2/-3 for stdin/stdout/stderr */
  int readable, writable;
  int err, eof;
  int ungot;             /* one pushed-back char, -1 = none */
  int vmode;             /* 0 = full, 1 = line, 2 = none (glibc _IOFBF/_IOLBF/_IONBF) */
  unsigned wcap, wcount; /* write buffer: capacity in use (<= WBUF_CAP) and fill */
  char wbuf[WBUF_CAP];
} FILE;

static FILE io_stdin = {1, -1, 1, 0, 0, 0, -1, 2, 0, 0, {0}};
static FILE io_stdout = {1, -2, 0, 1, 0, 0, -1, 2, 0, 0, {0}};
static FILE io_stderr = {1, -3, 0, 1, 0, 0, -1, 2, 0, 0, {0}};
FILE *stdin = &io_stdin;
FILE *stdout = &io_stdout;
FILE *stderr = &io_stderr;

#define NFILES 64
static FILE pool[NFILES];

static int is_std(FILE *f) { return f->fd < 0; }
static int wflush(FILE *f);

/* "r"/"w"/"a" [+ "+"] [+ "b" anywhere] → capability flags; -1 on a malformed mode (liolib checks the
 * mode shape itself first (`l_checkmode`), so this is belt-and-braces). */
static long parse_mode(const char *m) {
  long fl;
  if (*m == 'r') fl = O_READ;
  else if (*m == 'w') fl = O_WRITE | O_CREATE | O_TRUNC;
  else if (*m == 'a') fl = O_APPEND | O_CREATE;
  else return -1;
  m++;
  for (; *m; m++) {
    if (*m == 'b') continue;
    if (*m == '+') fl |= (fl & O_READ) ? O_WRITE : O_READ;
    else return -1;
  }
  return fl;
}

FILE *fopen(const char *name, const char *mode) {
  long fl = parse_mode(mode);
  if (fl < 0) {
    errno_cell = 22; /* EINVAL */
    return (FILE *)0;
  }
  long fd = hc(FS_OPEN, (long)name, (long)strlen(name), fl, 0);
  if (fd < 0) return (FILE *)0;
  for (int i = 0; i < NFILES; i++) {
    if (!pool[i].used) {
      pool[i].used = 1;
      pool[i].fd = (int)fd;
      pool[i].readable = (fl & O_READ) != 0;
      pool[i].writable = (fl & (O_WRITE | O_APPEND)) != 0;
      pool[i].err = pool[i].eof = 0;
      pool[i].ungot = -1;
      pool[i].vmode = 0; /* full-buffered by default, like glibc */
      pool[i].wcap = WBUF_CAP;
      pool[i].wcount = 0;
      return &pool[i];
    }
  }
  hc(FS_CLOSE, fd, 0, 0, 0);
  errno_cell = 24; /* EMFILE */
  return (FILE *)0;
}
FILE *fopen64(const char *name, const char *mode) { return fopen(name, mode); }

FILE *freopen(const char *name, const char *mode, FILE *f) {
  if (!is_std(f)) hc(FS_CLOSE, f->fd, 0, 0, 0);
  f->used = 0;
  return fopen(name, mode);
}
FILE *freopen64(const char *name, const char *mode, FILE *f) { return freopen(name, mode, f); }

int fclose(FILE *f) {
  if (is_std(f)) return -1; /* EOF: std streams don't close (liolib guards this anyway) */
  wflush(f); /* close flushes — the "full buffer" setvbuf test observes exactly this */
  long r = hc(FS_CLOSE, f->fd, 0, 0, 0);
  f->used = 0;
  return r == 0 ? 0 : -1;
}

static long tmp_counter;
FILE *tmpfile(void) {
  char name[32];
  name[0] = 't'; name[1] = 'm'; name[2] = 'p'; name[3] = 'f';
  long n = ++tmp_counter;
  int i = 4;
  do { name[i++] = (char)('0' + n % 10); n /= 10; } while (n);
  name[i] = 0;
  FILE *f = fopen(name, "w+b");
  /* POSIX tmpfile is created unlinked: remove the name immediately — the open fd stays valid on
   * both backends (MemFs keeps the data alive for open handles; a Unix host keeps the inode) — so
   * the file is anonymous and vanishes on close, leaving the granted root clean. */
  if (f) hc(FS_REMOVE, (long)name, (long)strlen(name), 0, 0);
  return f;
}
FILE *tmpfile64(void) { return tmpfile(); }

/* `tmpnam` (ANSI loslib `os.tmpname`): a fresh name per call, in the capability's flat namespace. */
char *tmpnam(char *s) {
  static char buf[32];
  if (!s) s = buf;
  s[0] = 't'; s[1] = 'm'; s[2] = 'p'; s[3] = '_';
  long n = ++tmp_counter;
  int i = 4;
  do { s[i++] = (char)('0' + n % 10); n /= 10; } while (n);
  s[i] = 0;
  return s;
}

int remove(const char *name) {
  return hc(FS_REMOVE, (long)name, (long)strlen(name), 0, 0) == 0 ? 0 : -1;
}
int rename(const char *from, const char *to) {
  return hc(FS_RENAME, (long)from, (long)strlen(from), (long)to, (long)strlen(to)) == 0 ? 0 : -1;
}

/* Flush the FILE's write buffer to the capability. The visible-file state moves only here (and on
 * unbuffered writes), which is exactly what files.lua's setvbuf tests observe through a second
 * reader of the same file. */
static int wflush(FILE *f) {
  if (f->wcount == 0) return 0;
  long n = hc(FS_WRITE, f->fd, (long)f->wbuf, (long)f->wcount, 0);
  int ok = n == (long)f->wcount;
  f->wcount = 0;
  if (!ok) f->err = 1;
  return ok ? 0 : -1;
}

unsigned long fread(void *ptr, unsigned long size, unsigned long nmemb, FILE *f) {
  unsigned long total = size * nmemb, got = 0;
  char *p = (char *)ptr;
  if (total == 0) return 0;
  if (!f->readable) { /* a write-only FILE reads as an error, not EOF (liolib: ferror + errno) */
    f->err = 1;
    errno_cell = 9; /* EBADF */
    return 0;
  }
  if (!is_std(f)) wflush(f); /* update-mode ("r+") read sees the FILE's own writes */
  if (f->ungot >= 0) {
    p[got++] = (char)f->ungot;
    f->ungot = -1;
  }
  while (got < total) {
    long n = is_std(f) ? read(0, p + got, (long)(total - got))
                       : hc(FS_READ, f->fd, (long)(p + got), (long)(total - got), 0);
    if (n < 0) { f->err = 1; break; }
    if (n == 0) { f->eof = 1; break; }
    got += (unsigned long)n;
  }
  return size ? got / size : 0;
}

unsigned long fwrite(const void *ptr, unsigned long size, unsigned long nmemb, FILE *f) {
  unsigned long total = size * nmemb;
  if (total == 0) return nmemb;
  if (is_std(f)) {
    write(1, ptr, (long)total); /* stdout & stderr share the granted out-Stream */
    return nmemb;
  }
  if (!f->writable) { /* symmetric: writing a read-only FILE is an error with an errno */
    f->err = 1;
    errno_cell = 9; /* EBADF */
    return 0;
  }
  if (f->vmode == 2 || total > f->wcap) { /* unbuffered, or larger than the buffer: write through */
    if (wflush(f) != 0) return 0;
    long n = hc(FS_WRITE, f->fd, (long)ptr, (long)total, 0);
    if (n < 0 || (unsigned long)n != total) { f->err = 1; return 0; }
    return nmemb;
  }
  if (f->wcount + total > f->wcap && wflush(f) != 0) return 0;
  const char *src = (const char *)ptr;
  int has_nl = 0;
  for (unsigned long i = 0; i < total; i++) {
    f->wbuf[f->wcount + i] = src[i];
    if (src[i] == '\n') has_nl = 1;
  }
  f->wcount += (unsigned)total;
  /* line-buffered: a newline flushes the whole buffer (glibc behavior) */
  if (f->vmode == 1 && has_nl && wflush(f) != 0) return 0;
  return nmemb;
}

int getc(FILE *f) {
  unsigned char c;
  if (f->ungot >= 0) {
    int r = f->ungot;
    f->ungot = -1;
    return r;
  }
  if (fread(&c, 1, 1, f) != 1) return -1; /* EOF */
  return c;
}
int fgetc(FILE *f) { return getc(f); }

int ungetc(int c, FILE *f) {
  if (c < 0 || f->ungot >= 0) return -1;
  f->ungot = c & 0xff;
  f->eof = 0;
  return f->ungot;
}

char *fgets(char *s, int n, FILE *f) {
  int i = 0;
  while (i < n - 1) {
    int c = getc(f);
    if (c < 0) break;
    s[i++] = (char)c;
    if (c == '\n') break;
  }
  if (i == 0) return (char *)0;
  s[i] = 0;
  return s;
}

int fputc(int c, FILE *f) {
  unsigned char b = (unsigned char)c;
  return fwrite(&b, 1, 1, f) == 1 ? c : -1;
}
int putc(int c, FILE *f) { return fputc(c, f); }
int putchar(int c) { return fputc(c, stdout); }
int fputs(const char *s, FILE *f) {
  unsigned long l = strlen(s);
  return fwrite(s, 1, l, f) == l ? 0 : -1;
}

int fseek(FILE *f, long off, int whence) {
  if (is_std(f)) return -1;
  wflush(f); /* ANSI: a seek flushes buffered output */
  /* an un-consumed pushback sits one byte before the capability cursor */
  if (f->ungot >= 0 && whence == 1) off -= 1;
  f->ungot = -1;
  f->eof = 0;
  return hc(FS_SEEK, f->fd, whence, off, 0) >= 0 ? 0 : -1;
}
int fseeko(FILE *f, long off, int whence) { return fseek(f, off, whence); }
int fseeko64(FILE *f, long off, int whence) { return fseek(f, off, whence); }

long ftell(FILE *f) {
  if (is_std(f)) return -1;
  long p = hc(FS_SEEK, f->fd, 1, 0, 0);
  if (p < 0) return -1;
  p += (long)f->wcount; /* logical position includes not-yet-flushed bytes */
  return f->ungot >= 0 ? p - 1 : p;
}
long ftello(FILE *f) { return ftell(f); }
long ftello64(FILE *f) { return ftell(f); }

int fflush(FILE *f) {
  if (!f || is_std(f)) return 0; /* std streams are write-through */
  return wflush(f);
}
int feof(FILE *f) { return f->eof; }
int ferror(FILE *f) { return f->err; }
void clearerr(FILE *f) { f->err = f->eof = 0; }
/* glibc _IOFBF=0 / _IOLBF=1 / _IONBF=2 — the constants liolib's f_setvbuf passes. The caller's
 * `buf` is ignored (we keep the internal buffer); `size` is honored up to WBUF_CAP. */
int setvbuf(FILE *f, char *buf, int mode, unsigned long size) {
  (void)buf;
  if (is_std(f)) return 0;
  wflush(f);
  f->vmode = mode;
  f->wcap = (mode == 2 || size == 0) ? 0 : (size < WBUF_CAP ? (unsigned)size : WBUF_CAP);
  if (mode == 2) f->wcap = 0;
  return 0;
}

/* Lua's `f:write(number)` writes through `fprintf(f, "%lld" / "%.14g", n)` — route it through the
 * guest vsnprintf (the string.format engine, correctly rounded) then fwrite. */
int fprintf(FILE *f, const char *fmt, ...) {
  char buf[256];
  va_list ap;
  __builtin_va_start(ap, fmt);
  int n = vsnprintf(buf, sizeof buf, fmt, ap);
  __builtin_va_end(ap);
  if (n <= 0) return n;
  if (n > (int)sizeof buf) n = (int)sizeof buf;
  return (int)fwrite(buf, 1, (unsigned long)n, f);
}
