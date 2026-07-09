/* Guest OS shim for the LMDB demo — the small remainder the on-ramp's synthesized libc
 * (malloc/free/calloc/realloc, the mem and str families, printf, strtod, ...) does NOT already provide. Two kinds:
 *
 *  1. The **fs-capability bridge** (the same pattern as Lua's `lua_files_stdio.c`): LMDB's raw
 *     file + mmap syscalls are routed to the embedder-granted `fs` capability via
 *     `__vm_cap_resolve("fs")` + `__vm_host_call`. This is the whole point — every byte of LMDB's
 *     memory-mapped data plane flows through granted authority, nothing ambient.
 *  2. **Single-thread / no-op stubs** for the OS odds-and-ends LMDB references but (under
 *     `MDB_NOLOCK`, single-process) never meaningfully exercises: the pthread suite, `sysconf`,
 *     `getpid`, `uname`, `sigaddset`/`sigemptyset`, `madvise`, `posix_memalign` (only the unreached
 *     hot-backup path).
 *
 * Compiled ONLY into the guest build (`-DSVM_GUEST`); the native oracle uses real glibc.
 */

#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/uio.h>
#include <sys/utsname.h>
#include <sys/vfs.h>
#include <unistd.h>

extern void *malloc(unsigned long);
extern void *memcpy(void *, const void *, unsigned long);
extern unsigned long strlen(const char *);
extern int fputs(const char *, FILE *);

/* ---- the fs capability ------------------------------------------------------------------------ */

extern int __vm_cap_resolve(const char *name, long len);
extern long __vm_host_call(int h, int op, long a, long b, long c, long d);

enum {
  FS_OPEN = 0, FS_READ, FS_WRITE, FS_SEEK, FS_CLOSE, FS_REMOVE, FS_RENAME,
  FS_TRUNCATE, FS_SYNC, FS_MMAP, FS_MSYNC, FS_MUNMAP
};
enum { CAP_O_READ = 1, CAP_O_WRITE = 2, CAP_O_APPEND = 4, CAP_O_TRUNC = 8, CAP_O_CREATE = 16 };

static int g_fs = -2; /* -2 = unresolved, -1 = unavailable, ≥0 = handle */
static int fs(void) {
  if (g_fs == -2) g_fs = __vm_cap_resolve("fs", 2);
  return g_fs;
}
static long hc(int op, long a, long b, long c, long d) { return __vm_host_call(fs(), op, a, b, c, d); }

static long cstrlen(const char *s) {
  long n = 0;
  while (s[n]) n++;
  return n;
}

/* ---- file ops → fs capability ----------------------------------------------------------------- */

int open(const char *path, int flags, ...) {
  long cf = 0;
  int acc = flags & O_ACCMODE;
  if (acc == O_RDONLY) cf |= CAP_O_READ;
  else if (acc == O_WRONLY) cf |= CAP_O_WRITE;
  else cf |= CAP_O_READ | CAP_O_WRITE; /* O_RDWR */
  if (flags & O_CREAT) cf |= CAP_O_CREATE;
  if (flags & O_TRUNC) cf |= CAP_O_TRUNC;
  if (flags & O_APPEND) cf |= CAP_O_APPEND;
  long r = hc(FS_OPEN, (long)path, cstrlen(path), cf, 0);
  return (int)r;
}

int close(int fd) { return (int)hc(FS_CLOSE, fd, 0, 0, 0); }

off_t lseek(int fd, off_t off, int whence) { return (off_t)hc(FS_SEEK, fd, whence, off, 0); }

ssize_t pread(int fd, void *buf, size_t n, off_t off) {
  if (hc(FS_SEEK, fd, SEEK_SET, off, 0) < 0) return -1;
  return (ssize_t)hc(FS_READ, fd, (long)buf, (long)n, 0);
}

ssize_t pwrite(int fd, const void *buf, size_t n, off_t off) {
  if (hc(FS_SEEK, fd, SEEK_SET, off, 0) < 0) return -1;
  return (ssize_t)hc(FS_WRITE, fd, (long)buf, (long)n, 0);
}

ssize_t write(int fd, const void *buf, size_t n) { return (ssize_t)hc(FS_WRITE, fd, (long)buf, (long)n, 0); }

ssize_t pwritev(int fd, const struct iovec *iov, int cnt, off_t off) {
  ssize_t total = 0;
  for (int i = 0; i < cnt; i++) {
    ssize_t r = pwrite(fd, iov[i].iov_base, iov[i].iov_len, off + total);
    if (r < 0) return r;
    total += r;
  }
  return total;
}

ssize_t writev(int fd, const struct iovec *iov, int cnt) {
  ssize_t total = 0;
  for (int i = 0; i < cnt; i++) {
    ssize_t r = write(fd, iov[i].iov_base, iov[i].iov_len);
    if (r < 0) return r;
    total += r;
  }
  return total;
}

int ftruncate(int fd, off_t len) { return (int)hc(FS_TRUNCATE, fd, len, 0, 0); }
int fsync(int fd) { return (int)hc(FS_SYNC, fd, 0, 0, 0); }
int fdatasync(int fd) { return (int)hc(FS_SYNC, fd, 0, 0, 0); }

int fstat(int fd, struct stat *st) {
  /* LMDB reads only st_size (fresh-file detection + geometry). Get it via seek-to-end, restoring
   * the cursor so a subsequent pread is unaffected. */
  long cur = hc(FS_SEEK, fd, SEEK_CUR, 0, 0);
  if (cur < 0) return -1;
  long end = hc(FS_SEEK, fd, SEEK_END, 0, 0);
  hc(FS_SEEK, fd, SEEK_SET, cur, 0);
  if (end < 0) return -1;
  __builtin_memset(st, 0, sizeof *st);
  st->st_size = end;
  st->st_mode = S_IFREG | 0644;
  st->st_blksize = 4096;
  return 0;
}

/* ---- mmap → fs capability --------------------------------------------------------------------- */

void *mmap(void *addr, size_t len, int prot, int flags, int fd, off_t off) {
  (void)addr;
  (void)prot;
  (void)flags;
  void *buf = __builtin_malloc(len);
  if (!buf) return MAP_FAILED;
  if (hc(FS_MMAP, fd, off, (long)len, (long)buf) != 0) {
    __builtin_free(buf);
    return MAP_FAILED;
  }
  return buf;
}

int munmap(void *addr, size_t len) {
  (void)len;
  hc(FS_MUNMAP, (long)addr, 0, 0, 0);
  __builtin_free(addr);
  return 0;
}

int msync(void *addr, size_t len, int flags) {
  (void)flags;
  return (int)hc(FS_MSYNC, (long)addr, (long)len, 0, 0);
}

int madvise(void *addr, size_t len, int advice) {
  (void)addr;
  (void)len;
  (void)advice;
  return 0;
}

/* ---- single-thread / no-op OS stubs ----------------------------------------------------------- */

long sysconf(int name) {
  (void)name;
  return 4096; /* _SC_PAGE_SIZE — the only sysconf LMDB queries */
}
pid_t getpid(void) { return 1; }
int uname(struct utsname *u) {
  __builtin_memset(u, 0, sizeof *u);
  return 0;
}
int sigemptyset(sigset_t *s) {
  __builtin_memset(s, 0, sizeof *s);
  return 0;
}
int sigaddset(sigset_t *s, int n) {
  (void)s;
  (void)n;
  return 0;
}

/* Never reached at runtime (only the hot-backup copy path); needs to link + return valid memory. */
int posix_memalign(void **out, size_t align, size_t size) {
  (void)align;
  void *p = __builtin_malloc(size);
  if (!p) return 12; /* ENOMEM */
  *out = p;
  return 0;
}

/* pthread — MDB_NOLOCK + single process: the reader-table and write mutex are never taken, so these
 * only need to link (and behave sanely if a benign init/self is called). */
int pthread_mutex_init(pthread_mutex_t *m, const pthread_mutexattr_t *a) { (void)m; (void)a; return 0; }
int pthread_mutex_destroy(pthread_mutex_t *m) { (void)m; return 0; }
int pthread_mutex_lock(pthread_mutex_t *m) { (void)m; return 0; }
int pthread_mutex_unlock(pthread_mutex_t *m) { (void)m; return 0; }
int pthread_mutex_consistent(pthread_mutex_t *m) { (void)m; return 0; }
int pthread_mutexattr_init(pthread_mutexattr_t *a) { (void)a; return 0; }
int pthread_mutexattr_destroy(pthread_mutexattr_t *a) { (void)a; return 0; }
int pthread_mutexattr_setpshared(pthread_mutexattr_t *a, int s) { (void)a; (void)s; return 0; }
int pthread_mutexattr_setrobust(pthread_mutexattr_t *a, int r) { (void)a; (void)r; return 0; }
int pthread_cond_init(pthread_cond_t *c, const pthread_condattr_t *a) { (void)c; (void)a; return 0; }
int pthread_cond_destroy(pthread_cond_t *c) { (void)c; return 0; }
int pthread_cond_signal(pthread_cond_t *c) { (void)c; return 0; }
int pthread_cond_wait(pthread_cond_t *c, pthread_mutex_t *m) { (void)c; (void)m; return 0; }
pthread_t pthread_self(void) { return 1; }
int pthread_sigmask(int how, const sigset_t *set, sigset_t *old) { (void)how; (void)set; (void)old; return 0; }
int pthread_key_create(pthread_key_t *k, void (*d)(void *)) { (void)k; (void)d; return 0; }
int pthread_key_delete(pthread_key_t k) { (void)k; return 0; }
void *pthread_getspecific(pthread_key_t k) { (void)k; return 0; }
int pthread_setspecific(pthread_key_t k, const void *v) { (void)k; (void)v; return 0; }

/* ---- error/rare-path libc (mostly unreached on the happy path) --------------------------------- */

/* atoi/strtol: LMDB parses the kernel version from uname().release. Our uname zeroes the struct, so
 * this only ever parses "" -> 0, but the symbol (glibc's C23 redirect) must resolve. Base-10. */
long strtol(const char *s, char **end, int base) {
  (void)base;
  long v = 0, sign = 1;
  while (*s == ' ' || *s == '\t') s++;
  if (*s == '-') { sign = -1; s++; } else if (*s == '+') s++;
  while (*s >= '0' && *s <= '9') v = v * 10 + (*s++ - '0');
  if (end) *end = (char *)s;
  return v * sign;
}
long __isoc23_strtol(const char *s, char **end, int base) { return strtol(s, end, base); }
int atoi(const char *s) { return (int)strtol(s, 0, 10); }

char *strerror(int e) {
  (void)e;
  return "error"; /* only the CHECK-macro error path prints this; happy path never calls it */
}

char *strdup(const char *s) {
  unsigned long n = strlen(s) + 1;
  char *p = malloc(n);
  if (p) memcpy(p, s, n);
  return p;
}

FILE *stderr = 0;
int fprintf(FILE *f, const char *fmt, ...) {
  (void)f;
  (void)fmt;
  return 0; /* LMDB uses fprintf only for debug/assert diagnostics */
}
int sprintf(char *buf, const char *fmt, ...) {
  /* Only mdb_strerror's unknown-code path (unreached on success). Copy the literal format so buf is
   * NUL-terminated; %-expansion is irrelevant to the happy-path differential. */
  unsigned long i = 0;
  while (fmt[i]) { buf[i] = fmt[i]; i++; }
  buf[i] = 0;
  return (int)i;
}

void abort(void) {
  fputs("abort()\n", stdout);
  __builtin_trap();
}

int fcntl(int fd, int cmd, ...) {
  (void)fd;
  (void)cmd;
  return 0; /* F_GETFD/SETFD (CLOEXEC), F_GETFL/SETFL (O_DIRECT), F_SETLK (skipped under NOLOCK) */
}
int fstatfs(int fd, struct statfs *b) {
  (void)fd;
  __builtin_memset(b, 0, sizeof *b);
  return 0; /* f_type = 0 (unknown) -> LMDB proceeds without the fs-specific workarounds */
}

/* async-copy thread path (mdb_env_copythr) — never reached by this demo; link-only. */
int pthread_create(pthread_t *t, const pthread_attr_t *a, void *(*f)(void *), void *arg) {
  (void)t;
  (void)a;
  (void)f;
  (void)arg;
  return 1; /* EPERM-ish: force LMDB's synchronous fallback if it ever tried */
}
int pthread_join(pthread_t t, void **r) {
  (void)t;
  (void)r;
  return 0;
}
