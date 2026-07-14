/* Guest stdio shim — the buffered-file `FILE*` surface Postgres uses, over the `fs` capability
 * (slice CD, Postgres runtime gap #11e). Layered on `os_shim.c`: every `FILE` is just an fs-cap fd
 * plus a little state, so `fopen`/`fread`/`fgets`/… bottom out in the same `open`/`read`/`write`/
 * `lseek`/`close` the syscall shim already routes to the capability. No buffering of its own — the
 * cap is the buffer boundary — so `fflush`/`setvbuf` are no-ops.
 *
 * Scope: exactly the members Postgres declares (its config reader `guc-file.l` uses fopen/fgets;
 * relation/WAL I/O uses the raw syscalls in `os_shim.c`). Deliberately NOT here: `stdout`/`stderr`/
 * `stdin` as `FILE*` (they must reach the powerbox Stream cap, which needs an on-ramp fd-dispatch —
 * a later slice; the on-ramp already funnels `puts`/`printf` to stdout) and the varargs `fprintf`/
 * `fscanf` format engines.
 *
 * `#include`d into a driver under `-DSVM_GUEST`, after `os_shim.c` (whose `open`/`read`/… it calls).
 */

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* A guest FILE: the underlying fs-cap fd + EOF/error flags + a one-byte ungetc slot. Returned to the
 * caller as an opaque `FILE*` (glibc's FILE is opaque; callers only ever hand it back to us). */
typedef struct {
  int fd;
  int eof;
  int err;
  int readable;
  int writable;
  int unget; /* -1 = empty, else the pushed-back byte */
} ShimFile;

/* Map a C `fopen` mode string to `<fcntl.h>` O_* flags — the same flags `open(2)` takes (os_shim.c's
 * `open` re-maps them to the cap's FS_O_*). A '+' means read+write; other mode chars (b/e/m/x/t/c)
 * don't affect the open flags. */
static int shim_mode_flags(const char *mode) {
  int plus = strchr(mode, '+') != (char *)0;
  switch (mode[0]) {
    case 'r': return plus ? O_RDWR : O_RDONLY;
    case 'w': return (plus ? O_RDWR : O_WRONLY) | O_CREAT | O_TRUNC;
    case 'a': return (plus ? O_RDWR : O_WRONLY) | O_CREAT | O_APPEND;
    default: return -1;
  }
}

static ShimFile *shim_new(int fd, const char *mode) {
  ShimFile *f = (ShimFile *)malloc(sizeof *f);
  if (!f) return (ShimFile *)0;
  f->fd = fd;
  f->eof = 0;
  f->err = 0;
  f->readable = (mode[0] == 'r') || strchr(mode, '+') != (char *)0;
  f->writable = (mode[0] != 'r') || strchr(mode, '+') != (char *)0;
  f->unget = -1;
  return f;
}

FILE *fopen(const char *path, const char *mode) {
  int fl = shim_mode_flags(mode);
  if (fl < 0) return (FILE *)0;
  int fd = open(path, fl, 0644);
  if (fd < 0) return (FILE *)0;
  ShimFile *f = shim_new(fd, mode);
  if (!f) { close(fd); return (FILE *)0; }
  return (FILE *)f;
}
FILE *freopen(const char *path, const char *mode, FILE *stream) {
  ShimFile *f = (ShimFile *)stream;
  if (f) { close(f->fd); }
  int fl = shim_mode_flags(mode);
  if (fl < 0) { free(f); return (FILE *)0; }
  int fd = open(path, fl, 0644);
  if (fd < 0) { free(f); return (FILE *)0; }
  if (!f) return (FILE *)shim_new(fd, mode);
  f->fd = fd;
  f->eof = 0;
  f->err = 0;
  f->readable = (mode[0] == 'r') || strchr(mode, '+') != (char *)0;
  f->writable = (mode[0] != 'r') || strchr(mode, '+') != (char *)0;
  f->unget = -1;
  return (FILE *)f;
}
int fclose(FILE *stream) {
  ShimFile *f = (ShimFile *)stream;
  int rc = close(f->fd);
  free(f);
  return rc < 0 ? EOF : 0;
}

size_t fread(void *ptr, size_t size, size_t nmemb, FILE *stream) {
  ShimFile *f = (ShimFile *)stream;
  if (size == 0 || nmemb == 0) return 0;
  size_t total = size * nmemb, got = 0;
  char *out = (char *)ptr;
  if (f->unget >= 0 && total > 0) { /* drain the pushed-back byte first */
    out[got++] = (char)f->unget;
    f->unget = -1;
  }
  while (got < total) {
    long n = read(f->fd, out + got, total - got);
    if (n < 0) { f->err = 1; break; }
    if (n == 0) { f->eof = 1; break; }
    got += (size_t)n;
  }
  return got / size; /* whole elements read */
}
size_t fwrite(const void *ptr, size_t size, size_t nmemb, FILE *stream) {
  ShimFile *f = (ShimFile *)stream;
  if (size == 0 || nmemb == 0) return 0;
  size_t total = size * nmemb, put = 0;
  const char *in = (const char *)ptr;
  while (put < total) {
    long n = write(f->fd, in + put, total - put);
    if (n <= 0) { f->err = 1; break; }
    put += (size_t)n;
  }
  return put / size;
}
int fgetc(FILE *stream) {
  ShimFile *f = (ShimFile *)stream;
  if (f->unget >= 0) { int c = f->unget; f->unget = -1; return c; }
  unsigned char b;
  long n = read(f->fd, &b, 1);
  if (n < 0) { f->err = 1; return EOF; }
  if (n == 0) { f->eof = 1; return EOF; }
  return (int)b;
}
int getc(FILE *stream) { return fgetc(stream); }
int _IO_getc(FILE *stream) { return fgetc(stream); }
int ungetc(int c, FILE *stream) {
  ShimFile *f = (ShimFile *)stream;
  if (c == EOF || f->unget >= 0) return EOF;
  f->unget = c & 0xff;
  f->eof = 0;
  return c & 0xff;
}
char *fgets(char *s, int n, FILE *stream) {
  if (n <= 0) return (char *)0;
  int i = 0;
  while (i < n - 1) {
    int c = fgetc(stream);
    if (c == EOF) break;
    s[i++] = (char)c;
    if (c == '\n') break;
  }
  if (i == 0) return (char *)0; /* nothing read (EOF or error) */
  s[i] = 0;
  return s;
}
int fputc(int c, FILE *stream) {
  ShimFile *f = (ShimFile *)stream;
  unsigned char b = (unsigned char)c;
  long n = write(f->fd, &b, 1);
  if (n <= 0) { f->err = 1; return EOF; }
  return (int)b;
}

int fseek(FILE *stream, long off, int whence) {
  ShimFile *f = (ShimFile *)stream;
  f->unget = -1;
  long rc = lseek(f->fd, off, whence);
  if (rc < 0) { f->err = 1; return -1; }
  f->eof = 0;
  return 0;
}
int fseeko(FILE *stream, off_t off, int whence) { return fseek(stream, (long)off, whence); }
long ftell(FILE *stream) {
  ShimFile *f = (ShimFile *)stream;
  return lseek(f->fd, 0, SEEK_CUR);
}
off_t ftello(FILE *stream) { return (off_t)ftell(stream); }
void rewind(FILE *stream) { (void)fseek(stream, 0, SEEK_SET); ((ShimFile *)stream)->err = 0; }

int feof(FILE *stream) { return ((ShimFile *)stream)->eof; }
int ferror(FILE *stream) { return ((ShimFile *)stream)->err; }
void clearerr(FILE *stream) {
  ShimFile *f = (ShimFile *)stream;
  f->eof = 0;
  f->err = 0;
}
int fileno(FILE *stream) { return ((ShimFile *)stream)->fd; }

/* Unbuffered (the cap is the boundary): buffering controls are inert, and there is nothing to flush. */
int fflush(FILE *stream) { (void)stream; return 0; }
int setvbuf(FILE *stream, char *buf, int mode, size_t size) {
  (void)stream; (void)buf; (void)mode; (void)size;
  return 0;
}
void setbuf(FILE *stream, char *buf) { (void)stream; (void)buf; }
