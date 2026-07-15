/* Guest OS-shim — the POSIX file/directory syscalls over the `fs` capability (slice CA, gap #11b).
 *
 * Postgres (unlike SQLite, which funnels all I/O through one `sqlite3_vfs` seam) calls the libc
 * syscall wrappers — `open`/`read`/`pread`/`write`/`pwrite`/`lseek`/`stat`/`fstat`/`opendir`/
 * `readdir`/`mkdir`/… — directly, all over `fd.c`/`md.c`/`xlog.c`. In the whole-program bitcode
 * those are *undefined externals* (the guest links no libc), so under `--stub-externs` each is a
 * trap-if-called stub. This file **defines** them for a guest build, bridging every one to
 * `__vm_cap_resolve("fs")` + `__vm_host_call` — the same embedder-granted `fs` capability SQLite
 * Phase B and LMDB use (`crates/svm-run/src/fs.rs`), now with the metadata + directory surface
 * (slice BZ: `stat`/`mkdir`/`rmdir`/`opendir`/`readdir`). No ambient authority: no `fs` cap, no
 * bytes.
 *
 * Scope: the file + directory surface, all of it exercised by the `os_probe.c` differential. The
 * proc/time/signal stubs (`getpid`/`clock_gettime`/`sigaction`/…) and the pure-libc surface
 * (stdio `FILE*`, locale, ctype, `strtod`/`snprintf`) are later slices.
 *
 * Built by being `#include`d into a driver under `-DSVM_GUEST` (like SQLite's cap VFS) — a single
 * translation unit, so the on-ramp sees these definitions satisfy the driver's libc calls. The
 * native oracle omits this file and uses the real glibc.
 */

#include <dirent.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

/* §7 host-defined-capability surface (the same ABI SQLite's cap VFS uses). */
extern int __vm_cap_resolve(const char *name, long len);
extern long __vm_host_call(int h, int op, long a, long b, long c, long d);
/* Direct powerbox-stream access (on-ramp builtins): the fds the sandbox owns — 0 = stdin, 1 = stdout,
 * 2 = stderr — reach the Stream cap, not the fs cap. `write`/`read` below fd-dispatch to these so the
 * shim can serve *file* fds through the capability while stdout/stderr/stdin stay the real streams. */
extern long __vm_stream_write(long buf, long len);
extern long __vm_stream_read(long buf, long len);

/* The `fs` op protocol — crates/svm-run/src/fs.rs. */
enum {
  FS_OPEN = 0, FS_READ, FS_WRITE, FS_SEEK, FS_CLOSE, FS_REMOVE, FS_RENAME, FS_TRUNCATE,
  FS_SYNC, FS_MMAP, FS_MSYNC, FS_MUNMAP, FS_CRASH_ARM, FS_MAP_REGION,
  FS_STAT, FS_MKDIR, FS_RMDIR, FS_OPENDIR, FS_READDIR, FS_CLOSEDIR
};
enum { FS_O_READ = 1, FS_O_WRITE = 2, FS_O_APPEND = 4, FS_O_TRUNC = 8, FS_O_CREATE = 16 };
#define FS_STATBUF_LEN 72

/* errno: glibc's `errno` macro expands to `*__errno_location()`. The accessor + storage live in a
 * shared, include-guarded header so a driver that pulls in several shims defines them exactly once. */
#include "shim_errno.h"

/* Resolve the `fs` cap once (lazily), then every call rides the cached handle. */
static int g_fs = -2; /* -2 = not yet resolved */
static long fscall(int op, long a, long b, long c, long d) {
  if (g_fs == -2) g_fs = __vm_cap_resolve("fs", 2);
  return __vm_host_call(g_fs, op, a, b, c, d);
}

/* Every wrapper turns a negative cap return (`-errno`) into the POSIX `-1` + `errno`. */
static long fail(long rc) { shim_errno = (int)(-rc); return -1; }

/* ---- little-endian readers for the 72-byte StatBuf the FS_STAT op fills ---------------------- */
static unsigned ld_u32(const unsigned char *p) {
  return (unsigned)p[0] | (unsigned)p[1] << 8 | (unsigned)p[2] << 16 | (unsigned)p[3] << 24;
}
static long ld_i64(const unsigned char *p) {
  long v = 0;
  for (int i = 0; i < 8; i++) v |= (long)p[i] << (8 * i);
  return v;
}
/* StatBuf layout (fs.rs::stat_bytes): 0:mode 4:nlink 8:size 16:mtime_s 24:mtime_ns 32:ino 40:dev
 * 48:uid 52:gid 56:blksize 64:blocks. Copy into the guest's glibc `struct stat` by field name. */
static void fill_stat(struct stat *st, const unsigned char *b) {
  memset(st, 0, sizeof *st);
  st->st_mode = ld_u32(b + 0);
  st->st_nlink = ld_u32(b + 4);
  st->st_size = ld_i64(b + 8);
  st->st_mtime = ld_i64(b + 16);
  st->st_ino = (unsigned long)ld_i64(b + 32);
  st->st_dev = (unsigned long)ld_i64(b + 40);
  st->st_uid = ld_u32(b + 48);
  st->st_gid = ld_u32(b + 52);
  st->st_blksize = ld_i64(b + 56);
  st->st_blocks = ld_i64(b + 64);
}

/* ---- files ----------------------------------------------------------------------------------- */
int open(const char *path, int flags, ...) {
  long f = 0;
  int acc = flags & O_ACCMODE;
  if (acc == O_RDONLY || acc == O_RDWR) f |= FS_O_READ;
  if (acc == O_WRONLY || acc == O_RDWR) f |= FS_O_WRITE;
  if (flags & O_APPEND) f |= FS_O_APPEND;
  if (flags & O_TRUNC) f |= FS_O_TRUNC;
  if (flags & O_CREAT) f |= FS_O_CREATE;
  long rc = fscall(FS_OPEN, (long)path, (long)strlen(path), f, 0);
  return rc < 0 ? (int)fail(rc) : (int)rc;
}
int openat(int dirfd, const char *path, int flags, ...) {
  /* Postgres opens with AT_FDCWD; the cap root *is* the working directory. */
  (void)dirfd;
  return open(path, flags);
}
ssize_t read(int fd, void *buf, size_t n) {
  if (fd == 0) return __vm_stream_read((long)buf, (long)n); /* stdin → powerbox Stream */
  long rc = fscall(FS_READ, fd, (long)buf, (long)n, 0);
  return rc < 0 ? fail(rc) : rc;
}
ssize_t write(int fd, const void *buf, size_t n) {
  if (fd == 1 || fd == 2) return __vm_stream_write((long)buf, (long)n); /* stdout/stderr → Stream */
  long rc = fscall(FS_WRITE, fd, (long)buf, (long)n, 0);
  return rc < 0 ? fail(rc) : rc;
}
off_t lseek(int fd, off_t off, int whence) {
  long rc = fscall(FS_SEEK, fd, whence, (long)off, 0); /* SEEK_SET/CUR/END == 0/1/2 == cap whence */
  return rc < 0 ? fail(rc) : rc;
}
ssize_t pread(int fd, void *buf, size_t n, off_t off) {
  long rc = fscall(FS_SEEK, fd, 0, (long)off, 0);
  if (rc < 0) return fail(rc);
  return read(fd, buf, n);
}
ssize_t pwrite(int fd, const void *buf, size_t n, off_t off) {
  long rc = fscall(FS_SEEK, fd, 0, (long)off, 0);
  if (rc < 0) return fail(rc);
  return write(fd, buf, n);
}
int close(int fd) {
  long rc = fscall(FS_CLOSE, fd, 0, 0, 0);
  return rc < 0 ? (int)fail(rc) : 0;
}
int fsync(int fd) {
  long rc = fscall(FS_SYNC, fd, 0, 0, 0);
  return rc < 0 ? (int)fail(rc) : 0;
}
int fdatasync(int fd) { return fsync(fd); }
int ftruncate(int fd, off_t len) {
  long rc = fscall(FS_TRUNCATE, fd, (long)len, 0, 0);
  return rc < 0 ? (int)fail(rc) : 0;
}
int unlink(const char *path) {
  long rc = fscall(FS_REMOVE, (long)path, (long)strlen(path), 0, 0);
  return rc < 0 ? (int)fail(rc) : 0;
}
int rename(const char *from, const char *to) {
  long rc = fscall(FS_RENAME, (long)from, (long)strlen(from), (long)to, (long)strlen(to));
  return rc < 0 ? (int)fail(rc) : 0;
}

/* ---- metadata -------------------------------------------------------------------------------- */
int stat(const char *path, struct stat *st) {
  unsigned char b[FS_STATBUF_LEN];
  long rc = fscall(FS_STAT, (long)path, (long)strlen(path), (long)b, FS_STATBUF_LEN);
  if (rc < 0) return (int)fail(rc);
  fill_stat(st, b);
  return 0;
}
/* The cap's FS_STAT already has lstat semantics (symlinks are not followed). */
int lstat(const char *path, struct stat *st) { return stat(path, st); }
int fstat(int fd, struct stat *st) {
  /* No fstat-by-fd op yet: an open fd is a regular file here, and its size is SEEK_END. */
  long cur = fscall(FS_SEEK, fd, 1, 0, 0);
  long end = fscall(FS_SEEK, fd, 2, 0, 0);
  if (end < 0) return (int)fail(end);
  fscall(FS_SEEK, fd, 0, cur, 0); /* restore the cursor */
  memset(st, 0, sizeof *st);
  st->st_mode = 0100644; /* S_IFREG | 0644 */
  st->st_nlink = 1;
  st->st_size = end;
  st->st_blksize = 4096;
  st->st_blocks = (end + 511) / 512;
  return 0;
}
int access(const char *path, int mode) {
  /* Existence only; the rooted cap grants R/W/X uniformly, so mode bits don't gate. */
  (void)mode;
  unsigned char b[FS_STATBUF_LEN];
  long rc = fscall(FS_STAT, (long)path, (long)strlen(path), (long)b, FS_STATBUF_LEN);
  return rc < 0 ? (int)fail(rc) : 0;
}

/* ---- directories ----------------------------------------------------------------------------- */
int mkdir(const char *path, mode_t mode) {
  (void)mode; /* the granted root's umask governs */
  long rc = fscall(FS_MKDIR, (long)path, (long)strlen(path), 0, 0);
  return rc < 0 ? (int)fail(rc) : 0;
}
int rmdir(const char *path) {
  long rc = fscall(FS_RMDIR, (long)path, (long)strlen(path), 0, 0);
  return rc < 0 ? (int)fail(rc) : 0;
}

/* A guest DIR is just the cap's dir handle plus a `dirent` scratch buffer for the last name. */
typedef struct {
  int dh;
  struct dirent de;
} ShimDir;

DIR *opendir(const char *path) {
  long dh = fscall(FS_OPENDIR, (long)path, (long)strlen(path), 0, 0);
  if (dh < 0) {
    fail(dh);
    return (DIR *)0;
  }
  ShimDir *d = (ShimDir *)malloc(sizeof *d);
  if (!d) return (DIR *)0;
  d->dh = (int)dh;
  return (DIR *)d;
}
struct dirent *readdir(DIR *dp) {
  ShimDir *d = (ShimDir *)dp;
  long n = fscall(FS_READDIR, d->dh, (long)d->de.d_name, (long)sizeof d->de.d_name - 1, 0);
  if (n <= 0) return (struct dirent *)0; /* 0 = exhausted; the cap omits "." and ".." */
  d->de.d_name[n] = 0;
  d->de.d_ino = 1;  /* nonzero: some callers treat 0 as a tombstone */
  d->de.d_type = 0; /* DT_UNKNOWN → the caller stats to learn the type */
  return &d->de;
}
int closedir(DIR *dp) {
  ShimDir *d = (ShimDir *)dp;
  long rc = fscall(FS_CLOSEDIR, d->dh, 0, 0, 0);
  free(d);
  return rc < 0 ? (int)fail(rc) : 0;
}

/* The cap root *is* the working directory: chdir into the data dir is a no-op success, and `.`
 * names the root. (Postgres chdir's to DataDir at startup, then uses paths relative to it.) */
int chdir(const char *path) {
  (void)path;
  return 0;
}
char *getcwd(char *buf, size_t size) {
  if (!buf || size < 2) {
    shim_errno = 34; /* ERANGE */
    return (char *)0;
  }
  buf[0] = '.';
  buf[1] = 0;
  return buf;
}

/* realpath(3), POSIX.1-2008 semantics (what Postgres' `pg_realpath` needs to resolve the executable
 * path in `find_my_exec`): return a canonical **absolute** path for an existing file. The cap root is
 * the working directory ("/"), the cap forbids `..`/absolute escapes, and FS_STAT is lstat with no
 * symlinks to chase — so "canonical" here is the input made absolute and lexically cleaned (strip a
 * leading "./", collapse "//" and "/./", drop a trailing "/"). Existence is required (realpath fails
 * ENOENT on a missing path), so we stat first through the cap. `resolved == NULL` mallocs the result
 * (the POSIX.1-2008 form Postgres calls first); a caller-provided buffer is assumed >= 4096. */
char *realpath(const char *path, char *resolved) {
  if (!path || !*path) {
    shim_errno = 2; /* ENOENT */
    return (char *)0;
  }
  unsigned char sb[FS_STATBUF_LEN];
  long rc = fscall(FS_STAT, (long)path, (long)strlen(path), (long)sb, FS_STATBUF_LEN);
  if (rc < 0) {
    fail(rc);
    return (char *)0;
  }
  char *out = resolved ? resolved : (char *)malloc(4096);
  if (!out) {
    shim_errno = 12; /* ENOMEM */
    return (char *)0;
  }
  const char *s = path;
  while (s[0] == '.' && s[1] == '/') s += 2; /* strip leading "./" segments */
  while (*s == '/') s++;                      /* and any leading slashes (already rooted) */
  size_t j = 0;
  out[j++] = '/'; /* absolute, rooted at the cap root */
  for (const char *p = s; *p && j < 4094;) {
    if (p[0] == '/' && p[1] == '/') { /* collapse "//" */
      p++;
      continue;
    }
    if (p[0] == '/' && p[1] == '.' && (p[2] == '/' || p[2] == 0)) { /* collapse "/./" and trailing "/." */
      p += 2;
      continue;
    }
    out[j++] = *p++;
  }
  if (j > 1 && out[j - 1] == '/') j--; /* drop a trailing slash (but keep the root "/") */
  out[j] = 0;
  return out;
}
