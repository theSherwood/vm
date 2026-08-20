/* bash libc shim — the bash-specific OS/libc surface the on-ramp neither synthesizes, resolves
 * to an svm-posix capability, nor covers via a reused shim (#802 slices 2-3). Ordinary guest C; a
 * guest definition shadows the on-ramp's would-be trap stub (the `tcl_shim.c` discipline).
 *
 * The reuse map (see README) — NOT redefined here:
 *   - printf family / vsnprintf → `../postgres/printf_shim.c`
 *   - strtod                    → `../strtod/strtod.c`
 *   - fnmatch / regcomp-regexec → `../posix_libc/{fnmatch,regex}.c` (the personality's own guest
 *     libc, linked into the TU; their `__px_malloc`/`__px_free` externs bridge below)
 *   - mem/str/qsort/malloc/ctype/getenv/llvm.* → on-ramp-synthesized
 *
 * What lives here is bash's own OS entanglement, in five bands:
 *   0. the svm-posix personality — the OS surface (open/read/write/stat/fork/signals/…): the
 *      embedder grants the personality under the name "posix" (`svm_run::posix::posix_cap`); this
 *      band resolves it once (`__vm_cap_resolve`) and defines the real libc entry points over
 *      `__vm_host_call` op dispatch (the `posix_cap.rs` / `fixtures/posix_shim.h` idiom). Each
 *      wrapper marshals C conventions (NUL-strings, glibc struct layouts) to the op ABI
 *      (`(ptr, len)` strings, compact LE structs — layouts from `crates/svm-posix/src/lib.rs`);
 *   1. the stdio `FILE*` band — bash passes `stderr`/`stdout` around as real objects (xtrace,
 *      `internal_error`); a thin fd-backed FILE (no buffering: the fd IS the boundary);
 *   2. identity/limits/locale/time — deterministic single-user stubs (the Tcl fixed-epoch move);
 *   3. minimal multibyte — MB_CUR_MAX = 1 (ASCII): the mb-/wc- entry points bash still calls
 *      unconditionally get 1-byte implementations, the rest stay behind MB_CUR_MAX > 1 guards;
 *   4. fd/process oddments composed over band 0 (eaccess, ioctl(TIOCGWINSZ), sigsets, …);
 *   5. bridges — `__px_*` wrappers over band 0, so posix_libc guest code (regex.c today, exec.c
 *      in the fork/exec slice) links unchanged on the on-ramp path.
 *
 * Anything not here and not reached by the current slice rides `SVM_STUB_EXTERNS` trap stubs —
 * a hit stub names itself in the trap backtrace (the §6 name waist), which is the walk's compass.
 */
#include <stdarg.h>
#include <stddef.h>

/* --- on-ramp-synthesized externs this shim composes ------------------------------------------- */
extern void *malloc(unsigned long n);
extern void free(void *p);
extern unsigned long strlen(const char *s);
extern int vsnprintf(char *s, unsigned long n, const char *fmt, va_list ap);

/* --- errno ------------------------------------------------------------------------------------ */
static int __bash_errno;
int *__errno_location(void) { return &__bash_errno; }

/* --- band 0: the svm-posix personality --------------------------------------------------------
 * Resolve the granted "posix" capability once; every op is `__vm_host_call(handle, OP, a,b,c,d)`.
 * Ops return >= 0 or -errno; `px_ret_` folds that into the C convention (errno + -1). Op numbers
 * and marshaling contracts track `crates/svm-posix/src/lib.rs` (the vtable is the ABI). */
extern int __vm_cap_resolve(const char *name, long len);
extern long __vm_host_call(int h, int op, long a, long b, long c, long d);

enum {
  PX_WRITE = 0, PX_READ = 1, PX_EXIT = 4, PX_OPEN = 5, PX_CLOSE = 6, PX_LSEEK = 7,
  PX_UNLINK = 8, PX_GETCWD = 9, PX_CHDIR = 10, PX_STAT = 13, PX_OPENDIR = 14, PX_READDIR = 15,
  PX_CLOSEDIR = 16, PX_DUP2 = 24, PX_DUP = 25, PX_FCNTL = 26, PX_WAITPID = 28,
  PX_SIGNAL = 30, PX_KILL = 31, PX_MKDIR = 37, PX_RENAME = 38, PX_RMDIR = 39,
  PX_SIGPROCMASK = 40, PX_SIGACTION = 41, PX_SIGALTSTACK = 42, PX_GETPID = 44, PX_SETPGID = 45,
  PX_GETPGID = 46, PX_TCGETPGRP = 47, PX_TCSETPGRP = 48, PX_ISATTY = 49, PX_GETPPID = 50,
  PX_FORK = 51, PX_PIPE_ADOPT = 52, PX_TCGETATTR = 54, PX_TCSETATTR = 55, PX_TCGETWINSIZE = 56,
};

static int px_handle_ = -1;
static int px_(void) {
  if (px_handle_ < 0) px_handle_ = __vm_cap_resolve("posix", 5);
  return px_handle_;
}
static long px_call_(int op, long a, long b, long c, long d) {
  return __vm_host_call(px_(), op, a, b, c, d);
}
static long px_ret_(long r) {
  if (r < 0) { __bash_errno = (int)-r; return -1; }
  return r;
}
/* #972 tag protocol: an op landing on a core pipe/terminal end returns PX_TAG_BASE - handle
 * (<= -(1<<20)); the wrapper re-issues the transfer on the core handle via the core-pipe
 * builtins, where empty-with-writers PARKS and writer-count 0 is true EOF (the `util.c` shape).
 * Real errnos stay > -4096 and pass through. */
extern long __vm_pipe(int *fds);
extern long __vm_read(int h, void *buf, long len);
extern long __vm_write(int h, const void *buf, long len);
extern int __vm_close(int h);
static long px_tag_(long r) { return r <= -1048576 ? -(r + 1048576) : -1; }

long write(int fd, const void *buf, unsigned long n) {
  long r = px_call_(PX_WRITE, fd, (long)buf, (long)n, 0);
  long h = px_tag_(r);
  if (h >= 0) r = __vm_write((int)h, buf, (long)n);
  if (r == -32) px_call_(PX_KILL, 0, 13, 0, 0); /* -EPIPE: raise SIGPIPE per disposition */
  return px_ret_(r);
}
long read(int fd, void *buf, unsigned long n) {
  long r = px_call_(PX_READ, fd, (long)buf, (long)n, 0);
  long h = px_tag_(r);
  if (h >= 0) r = __vm_read((int)h, buf, (long)n);
  return px_ret_(r);
}
/* fd → path, recorded at open so fstat can re-stat (the memfs op surface is path-keyed). */
#define PX_NFDPATH 64
#define PX_FDPATH_CAP 256
static char px_fdpath_[PX_NFDPATH][PX_FDPATH_CAP];
static void px_fdpath_set_(long fd, const char *path) {
  if (fd < 0 || fd >= PX_NFDPATH) return;
  unsigned long i, n = strlen(path);
  if (n >= PX_FDPATH_CAP) n = PX_FDPATH_CAP - 1;
  for (i = 0; i < n; i = i + 1) px_fdpath_[fd][i] = path[i];
  px_fdpath_[fd][n] = 0;
}
int open(const char *path, int flags, ...) {
  long r = px_call_(PX_OPEN, (long)path, (long)strlen(path), flags, 0);
  if (r >= 0) px_fdpath_set_(r, path);
  return (int)px_ret_(r);
}
int close(int fd) {
  if (fd >= 0 && fd < PX_NFDPATH) px_fdpath_[fd][0] = 0;
  long r = px_call_(PX_CLOSE, fd, 0, 0, 0);
  long h = px_tag_(r);
  if (h >= 0) { /* last close of an adopted core end: release the powerbox handle */
    __vm_close((int)h);
    return 0;
  }
  return (int)px_ret_(r);
}
long lseek(int fd, long off, int whence) { return px_ret_(px_call_(PX_LSEEK, fd, off, whence, 0)); }
int unlink(const char *p) { return (int)px_ret_(px_call_(PX_UNLINK, (long)p, (long)strlen(p), 0, 0)); }
int rename(const char *o, const char *n) {
  return (int)px_ret_(px_call_(PX_RENAME, (long)o, (long)strlen(o), (long)n, (long)strlen(n)));
}
int mkdir(const char *p, unsigned int mode) {
  return (int)px_ret_(px_call_(PX_MKDIR, (long)p, (long)strlen(p), mode, 0));
}
int rmdir(const char *p) { return (int)px_ret_(px_call_(PX_RMDIR, (long)p, (long)strlen(p), 0, 0)); }
char *getcwd(char *buf, unsigned long size) {
  if (!buf) { /* the glibc allocate-extension — bash's shell-init cwd probe uses it */
    if (size == 0) size = 4096;
    buf = (char *)malloc(size);
    if (!buf) return 0;
    if (px_call_(PX_GETCWD, (long)buf, (long)size, 0, 0) < 0) {
      free(buf);
      return 0;
    }
    return buf;
  }
  return px_call_(PX_GETCWD, (long)buf, (long)size, 0, 0) < 0 ? 0 : buf;
}
int chdir(const char *p) { return (int)px_ret_(px_call_(PX_CHDIR, (long)p, (long)strlen(p), 0, 0)); }
/* pipe — mint a CORE pipe (blocking cross-process semantics: empty-with-writers parks the reader,
 * writer-count 0 is EOF) and adopt the two powerbox ends into this process's fd table (#972 —
 * `pipe_adopt` writes the `int[2]` fds). The in-personality PX_PIPE (op 23) is the older
 * non-blocking lane; fork twins need the core one. */
int pipe(int *fds) {
  int h[2];
  long r = __vm_pipe(h);
  if (r != 0) return (int)px_ret_(r);
  return (int)px_ret_(px_call_(PX_PIPE_ADOPT, h[0], h[1], (long)fds, 0));
}
int dup(int fd) { return (int)px_ret_(px_call_(PX_DUP, fd, 0, 0, 0)); }
int dup2(int o, int n) { return (int)px_ret_(px_call_(PX_DUP2, o, n, 0, 0)); }
int fcntl(int fd, int cmd, ...) {
  long arg = 0;
  if (cmd == 0 || cmd == 2 || cmd == 4 || cmd == 1030) { /* F_DUPFD/F_SETFD/F_SETFL/F_DUPFD_CLOEXEC */
    va_list ap;
    va_start(ap, cmd);
    arg = va_arg(ap, long);
    va_end(ap);
  }
  return (int)px_ret_(px_call_(PX_FCNTL, fd, cmd, arg, 0));
}
int isatty(int fd) { return px_call_(PX_ISATTY, fd, 0, 0, 0) == 1; }
int getpid(void) { return (int)px_call_(PX_GETPID, 0, 0, 0, 0); }
int getppid(void) { return (int)px_call_(PX_GETPPID, 0, 0, 0, 0); }
int getpgid(int pid) { return (int)px_ret_(px_call_(PX_GETPGID, pid, 0, 0, 0)); }
int getpgrp(void) { return getpgid(0); }
int setpgid(int pid, int pgid) { return (int)px_ret_(px_call_(PX_SETPGID, pid, pgid, 0, 0)); }
int tcgetpgrp(int fd) { return (int)px_ret_(px_call_(PX_TCGETPGRP, fd, 0, 0, 0)); }
int tcsetpgrp(int fd, int pgid) { return (int)px_ret_(px_call_(PX_TCSETPGRP, fd, pgid, 0, 0)); }
int kill(int pid, int sig) { return (int)px_ret_(px_call_(PX_KILL, pid, sig, 0, 0)); }
int fork(void) { return (int)px_ret_(px_call_(PX_FORK, 0, 0, 0, 0)); }
int waitpid(int pid, int *status, int opts) {
  return (int)px_ret_(px_call_(PX_WAITPID, pid, (long)status, opts, 0));
}
int wait(int *status) { return waitpid(-1, status, 0); }
void exit(int s) {
  px_call_(PX_EXIT, s, 0, 0, 0);
  for (;;) {} /* the exit op does not return */
}
void _exit(int s) { exit(s); }

/* stat — the op writes the compact {mode: i64, size: i64}; expand it into the glibc x86-64
 * `struct stat` bash was compiled against (dev@0, ino@8, nlink@16, mode@24:u32, uid@28, gid@32,
 * rdev@40, size@48, blksize@56, blocks@64, atim@72, mtim@88, ctim@104; 144 bytes). st_ino is a
 * path hash so bash's same-file checks (dev+ino equality) distinguish distinct paths. */
static unsigned long px_pathhash_(const char *p) {
  unsigned long h = 1469598103934665603ul; /* FNV-1a */
  while (*p) { h = (h ^ (unsigned char)*p) * 1099511628211ul; p = p + 1; }
  return h | 1; /* never 0 (0 is "no file") */
}
static void px_fill_stat_(void *st, unsigned long ino, unsigned int mode, long size) {
  char *b = (char *)st;
  unsigned long i;
  for (i = 0; i < 144; i = i + 1) b[i] = 0;
  *(unsigned long *)(b + 0) = 1;            /* st_dev */
  *(unsigned long *)(b + 8) = ino;          /* st_ino */
  *(unsigned long *)(b + 16) = 1;           /* st_nlink */
  *(unsigned int *)(b + 24) = mode;         /* st_mode */
  *(long *)(b + 48) = size;                 /* st_size */
  *(long *)(b + 56) = 4096;                 /* st_blksize */
  *(long *)(b + 64) = (size + 511) / 512;   /* st_blocks */
}
int stat(const char *path, void *st) {
  long out[2];
  long r = px_call_(PX_STAT, (long)path, (long)strlen(path), (long)out, 0);
  if (r < 0) return (int)px_ret_(r);
  px_fill_stat_(st, px_pathhash_(path), (unsigned int)out[0], out[1]);
  return 0;
}
int lstat(const char *path, void *st) { return stat(path, st); } /* the memfs has no symlinks */
int fstat(int fd, void *st) {
  if (fd >= 0 && fd <= 2) { /* stdio: character device (terminal-shaped) */
    px_fill_stat_(st, (unsigned long)fd + 1, 0020620, 0);
    return 0;
  }
  if (fd >= 0 && fd < PX_NFDPATH && px_fdpath_[fd][0]) return stat(px_fdpath_[fd], st);
  px_fill_stat_(st, 0, 0010600, 0); /* an untracked fd: pipe-shaped */
  return 0;
}

/* opendir/readdir/closedir — the op's DIR is a small index (possibly 0); bias by 1 so DIR* is
 * never NULL. readdir marshals the op's bare name into a static glibc dirent
 * {ino@0, off@8, reclen@16:u16, type@18:u8, name@19}. */
static char px_dirent_[19 + 256];
void *opendir(const char *path) {
  long r = px_call_(PX_OPENDIR, (long)path, (long)strlen(path), 0, 0);
  if (r < 0) { px_ret_(r); return 0; }
  return (void *)(r + 1);
}
void *readdir(void *dir) {
  if (!dir) return 0;
  long r = px_call_(PX_READDIR, (long)dir - 1, (long)(px_dirent_ + 19), 256, 0);
  if (r <= 0) { if (r < 0) px_ret_(r); return 0; } /* 0 = end of stream */
  *(unsigned long *)(px_dirent_ + 0) = px_pathhash_(px_dirent_ + 19); /* d_ino: non-zero */
  *(unsigned short *)(px_dirent_ + 16) = 19 + 256;                    /* d_reclen */
  px_dirent_[18] = 0;                                                 /* d_type: DT_UNKNOWN */
  return px_dirent_;
}
int closedir(void *dir) {
  if (!dir) return -1;
  return (int)px_ret_(px_call_(PX_CLOSEDIR, (long)dir - 1, 0, 0, 0));
}

/* signals — dispositions/mask/pending live in the personality. glibc's sigset_t is 128 bytes but
 * only the low u64 carries signals 1..63; the ops read/write exactly that u64 (LE), so the
 * caller's sigset_t* pass through directly. sigaction marshals glibc's
 * {handler@0, mask@8(128B), flags@136:int} to the op's compact 24-byte {handler, mask, flags}. */
void *signal(int sig, void *handler) {
  long r = px_call_(PX_SIGNAL, sig, (long)handler, 0, 0);
  return r < 0 ? (void *)-1 : (void *)r; /* SIG_ERR on -errno; else the previous handler */
}
int sigprocmask(int how, const void *set, void *oldset) {
  return (int)px_ret_(px_call_(PX_SIGPROCMASK, how, (long)set, (long)oldset, 0));
}
int sigaction(int sig, const void *act, void *oldact) {
  long a[3], o[3];
  if (act) {
    const char *b = (const char *)act;
    a[0] = *(const long *)(b + 0);          /* sa_handler */
    a[1] = *(const long *)(b + 8);          /* sa_mask (low u64) */
    a[2] = *(const int *)(b + 136);         /* sa_flags */
  }
  long r = px_call_(PX_SIGACTION, sig, act ? (long)a : 0, oldact ? (long)o : 0, 0);
  if (r < 0) return (int)px_ret_(r);
  if (oldact) {
    char *b = (char *)oldact;
    unsigned long i;
    for (i = 0; i < 152; i = i + 1) b[i] = 0;
    *(long *)(b + 0) = o[0];
    *(long *)(b + 8) = o[1];
    *(int *)(b + 136) = (int)o[2];
  }
  return 0;
}
int sigaltstack(const void *ss, void *oss) {
  (void)oss;
  if (!ss) return 0;
  /* glibc stack_t: {ss_sp@0, ss_flags@8:int, ss_size@16}. */
  const char *b = (const char *)ss;
  return (int)px_ret_(px_call_(PX_SIGALTSTACK, *(const long *)(b + 0), *(const long *)(b + 16), 0, 0));
}
int sigsuspend(const void *mask) {
  (void)mask;
  __bash_errno = 4; /* EINTR — the only POSIX return; real suspension rides the signals slice */
  return -1;
}
/* Async-signal delivery (#796 L2) is gated on a REGISTERED handler stack — the interp's safepoint
 * redirect runs the C handler on a dedicated stack, never the interrupted one — and is poll-only
 * without it. bash never calls sigaltstack on this config, so register a static stack before
 * `main` (a ctor: the synthesized `_start` runs `llvm.global_ctors`). Fork twins inherit the
 * registration over their own private window copy (POSIX). */
static char px_sigstack_[16384];
__attribute__((constructor)) static void px_sig_init_(void) {
  px_call_(PX_SIGALTSTACK, (long)px_sigstack_, sizeof(px_sigstack_), 0, 0);
}

/* terminal — the op termios is the 32-byte personality layout {lflag: i64, cc[8], vmin: i64,
 * vtime: i64}; marshal to/from glibc termios {iflag,oflag,cflag,lflag: u32@0..16, line: u8@16,
 * cc[32]@17, ispeed@36, ospeed@40}. */
int tcgetattr(int fd, void *t) {
  long buf[4];
  long r = px_call_(PX_TCGETATTR, fd, (long)buf, 0, 0);
  if (r < 0) return (int)px_ret_(r);
  char *b = (char *)t;
  unsigned long i;
  for (i = 0; i < 60; i = i + 1) b[i] = 0;
  *(unsigned int *)(b + 12) = (unsigned int)buf[0]; /* c_lflag */
  for (i = 0; i < 8; i = i + 1) b[17 + i] = ((char *)&buf[1])[i]; /* cc[0..8) */
  b[17 + 6] = (char)buf[2]; /* VMIN */
  b[17 + 5] = (char)buf[3]; /* VTIME */
  return 0;
}
int tcsetattr(int fd, int actions, const void *t) {
  (void)actions;
  const char *b = (const char *)t;
  long buf[4];
  unsigned long i;
  buf[0] = *(const unsigned int *)(b + 12);
  buf[1] = 0;
  for (i = 0; i < 8; i = i + 1) ((char *)&buf[1])[i] = b[17 + i];
  buf[2] = (unsigned char)b[17 + 6];
  buf[3] = (unsigned char)b[17 + 5];
  return (int)px_ret_(px_call_(PX_TCSETATTR, fd, (long)buf, 0, 0));
}
int tcgetwinsize(int fd, void *ws) {
  return (int)px_ret_(px_call_(PX_TCGETWINSIZE, fd, (long)ws, 0, 0));
}

/* --- environ ----------------------------------------------------------------------------------
 * bash's `main(argc, argv, env)` receives the real environment via its third parameter (the
 * on-ramp's `_start` parses the powerbox args buffer); the `environ` GLOBAL only needs to exist
 * as a real empty vector so a direct walk never dereferences NULL (the Tcl #986 shape). */
static char *shim_environ[1] = {0};
char **environ = shim_environ;

/* --- band 1: the fd-backed FILE ----------------------------------------------------------------
 * bash treats FILE* as opaque (it only hands it back to stdio functions), so a FILE here is just
 * {fd, eof, err}. No userspace buffering: every write goes straight to the fd — the personality
 * fd table is the buffer boundary — so fflush/setvbuf are honest no-ops. stderr keeps its own
 * sink (fd 2), which is what the `xtrace_set(-1, stderr)` startup path needs. */
typedef struct {
  int fd;
  int eof;
  int err;
} ShimFile;
static ShimFile shim_std_[3] = {{0, 0, 0}, {1, 0, 0}, {2, 0, 0}};
void *stdin_ptr_unused_;
/* The libc-global stream objects. */
ShimFile *stdin = &shim_std_[0];
ShimFile *stdout = &shim_std_[1];
ShimFile *stderr = &shim_std_[2];

#define SHIM_NFILES 16
static ShimFile shim_files_[SHIM_NFILES];
static int shim_files_used_[SHIM_NFILES];

int fileno(ShimFile *f) { return f ? f->fd : -1; }
int fflush(ShimFile *f) { (void)f; return 0; }
int ferror(ShimFile *f) { return f ? f->err : 1; }
void clearerr(ShimFile *f) { if (f) { f->err = 0; f->eof = 0; } }
void __fpurge(ShimFile *f) { (void)f; }
int setvbuf(ShimFile *f, char *buf, int mode, unsigned long size) {
  (void)f; (void)buf; (void)mode; (void)size;
  return 0;
}
ShimFile *fdopen(int fd, const char *mode) {
  (void)mode;
  if (fd >= 0 && fd <= 2) return &shim_std_[fd];
  int i;
  for (i = 0; i < SHIM_NFILES; i = i + 1) {
    if (!shim_files_used_[i]) {
      shim_files_used_[i] = 1;
      shim_files_[i].fd = fd;
      shim_files_[i].eof = 0;
      shim_files_[i].err = 0;
      return &shim_files_[i];
    }
  }
  return 0;
}
int fclose(ShimFile *f) {
  if (!f) return -1;
  if (f >= shim_files_ && f < shim_files_ + SHIM_NFILES) {
    shim_files_used_[f - shim_files_] = 0;
    return close(f->fd);
  }
  return 0; /* closing a std stream: keep the fd (bash never really wants it gone) */
}
long fwrite(const void *p, unsigned long sz, unsigned long n, ShimFile *f) {
  if (!f || sz == 0 || n == 0) return 0;
  long w = write(f->fd, p, sz * n);
  if (w < 0) { f->err = 1; return 0; }
  return (unsigned long)w / sz;
}
int fputs(const char *s, ShimFile *f) {
  if (!f) return -1;
  unsigned long n = strlen(s);
  return write(f->fd, s, n) == (long)n ? 0 : (f->err = 1, -1);
}
int fputc(int c, ShimFile *f) {
  char b = (char)c;
  if (!f || write(f->fd, &b, 1) != 1) { if (f) f->err = 1; return -1; }
  return (unsigned char)c;
}
int putc_shimmed_(int c, ShimFile *f) { return fputc(c, f); }
int putc(int c, ShimFile *f) { return fputc(c, f); }
int putchar(int c) { return fputc(c, stdout); }
int puts(const char *s) {
  if (fputs(s, stdout) < 0) return -1;
  return fputc('\n', stdout);
}

/* asprintf — over the reused printf engine's runtime vsnprintf (two-pass sizing). */
int asprintf(char **strp, const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  char probe[1];
  int need = vsnprintf(probe, 0, fmt, ap);
  va_end(ap);
  if (need < 0) return -1;
  char *buf = (char *)malloc((unsigned long)need + 1);
  if (!buf) return -1;
  va_start(ap, fmt);
  int n = vsnprintf(buf, (unsigned long)need + 1, fmt, ap);
  va_end(ap);
  *strp = buf;
  return n;
}

/* --- band 2: identity / limits / locale / time (deterministic single-user) ------------------- */
unsigned int getuid(void) { return 0; }
unsigned int geteuid(void) { return 0; }
unsigned int getgid(void) { return 0; }
unsigned int getegid(void) { return 0; }
int getgroups(int n, unsigned int *list) { (void)n; (void)list; return 0; }
int setresuid(unsigned int a, unsigned int b, unsigned int c) { (void)a; (void)b; (void)c; return 0; }
int setresgid(unsigned int a, unsigned int b, unsigned int c) { (void)a; (void)b; (void)c; return 0; }
void endpwent(void) {}
/* glibc x86-64 struct passwd: {name, passwd, uid, gid, gecos, dir, shell}. */
struct shim_passwd {
  char *pw_name;
  char *pw_passwd;
  unsigned int pw_uid;
  unsigned int pw_gid;
  char *pw_gecos;
  char *pw_dir;
  char *pw_shell;
};
static struct shim_passwd shim_pw_ = {"root", "", 0, 0, "", "/", "/bin/bash"};
struct shim_passwd *getpwuid(unsigned int uid) { (void)uid; return &shim_pw_; }
struct shim_passwd *getpwnam(const char *n) { (void)n; return &shim_pw_; }
int gethostname(char *buf, unsigned long n) {
  const char *h = "svm";
  unsigned long i;
  for (i = 0; h[i] && i + 1 < n; i = i + 1) buf[i] = h[i];
  if (n) buf[i] = 0;
  return 0;
}
char *ttyname(int fd) { (void)fd; return "/dev/tty"; }
int getdtablesize(void) { return 256; }
long sysconf(int name) {
  if (name == 2) return 100;  /* _SC_CLK_TCK */
  if (name == 4) return 256;  /* _SC_OPEN_MAX */
  if (name == 1) return 128;  /* _SC_CHILD_MAX */
  return -1;
}
unsigned long confstr(int name, char *buf, unsigned long n) {
  (void)name;
  if (buf && n) buf[0] = 0;
  return 1;
}
long pathconf(const char *p, int name) { (void)p; (void)name; return -1; }
int umask(int m) { (void)m; return 022; }
/* struct rlimit = {u64 cur, u64 max}; RLIM_INFINITY on everything (single-user sandbox). */
int getrlimit(int res, unsigned long *rl) {
  (void)res;
  rl[0] = ~0ul;
  rl[1] = ~0ul;
  return 0;
}
int setrlimit(int res, const unsigned long *rl) { (void)res; (void)rl; return 0; }
int getrusage(int who, void *ru) {
  (void)who;
  char *p = (char *)ru;
  unsigned long i;
  for (i = 0; i < 144; i = i + 1) p[i] = 0; /* zeroed struct rusage */
  return 0;
}
char *setlocale(int cat, const char *loc) { (void)cat; (void)loc; return "C"; }
char *nl_langinfo(int item) { (void)item; return ""; }
/* localeconv: the on-ramp synthesizes a C-locale lconv when the symbol is undefined; bash links
 * this shim, so define the same shape (first two members are all bash reads: decimal_point,
 * thousands_sep). */
static char *shim_lconv_[2] = {".", ""};
void *localeconv(void) { return (void *)shim_lconv_; }
long iconv_open(const char *a, const char *b) { (void)a; (void)b; return -1; }
long iconv(long cd, char **in, unsigned long *inb, char **out, unsigned long *outb) {
  (void)cd; (void)in; (void)inb; (void)out; (void)outb;
  return -1;
}
int iconv_close(long cd) { (void)cd; return 0; }
/* Fixed epoch (the Tcl determinism move): differentials must not depend on the wall clock. */
long time(long *t) {
  if (t) *t = 0;
  return 0;
}
int gettimeofday(long *tv, void *tz) {
  (void)tz;
  if (tv) { tv[0] = 0; tv[1] = 0; }
  return 0;
}
struct shim_tm {
  int tm_sec, tm_min, tm_hour, tm_mday, tm_mon, tm_year, tm_wday, tm_yday, tm_isdst;
  long tm_gmtoff;
  const char *tm_zone;
};
static struct shim_tm shim_tm_ = {0, 0, 0, 1, 0, 70, 4, 0, 0, 0, "UTC"};
struct shim_tm *localtime(const long *t) { (void)t; return &shim_tm_; }
unsigned long strftime(char *s, unsigned long max, const char *fmt, const void *tm) {
  (void)fmt; (void)tm;
  const char *fix = "1970-01-01";
  unsigned long i;
  for (i = 0; fix[i] && i + 1 < max; i = i + 1) s[i] = fix[i];
  if (max) s[i] = 0;
  return i;
}
void tzset(void) {}
unsigned int alarm(unsigned int s) { (void)s; return 0; }
int setitimer(int which, const void *nv, void *ov) { (void)which; (void)nv; (void)ov; return 0; }
unsigned int sleep(unsigned int s) { (void)s; return 0; }
/* Deterministic "randomness" ($RANDOM seeds from it; a differential needs sameness, not entropy). */
static unsigned long shim_rng_ = 0x5eed;
long getrandom(void *buf, unsigned long n, unsigned int flags) {
  (void)flags;
  unsigned char *p = (unsigned char *)buf;
  unsigned long i;
  for (i = 0; i < n; i = i + 1) {
    shim_rng_ = shim_rng_ * 6364136223846793005ul + 1442695040888963407ul;
    p[i] = (unsigned char)(shim_rng_ >> 33);
  }
  return (long)n;
}
unsigned int arc4random(void) {
  shim_rng_ = shim_rng_ * 6364136223846793005ul + 1442695040888963407ul;
  return (unsigned int)(shim_rng_ >> 33);
}

/* --- band 3: minimal multibyte (MB_CUR_MAX = 1, ASCII) ---------------------------------------- */
unsigned long __ctype_get_mb_cur_max(void) { return 1; }
int mblen(const char *s, unsigned long n) {
  if (!s) return 0;
  if (n == 0) return -1;
  return *s ? 1 : 0;
}
int mbtowc(int *wc, const char *s, unsigned long n) {
  if (!s) return 0;
  if (n == 0) return -1;
  if (wc) *wc = (unsigned char)*s;
  return *s ? 1 : 0;
}
int wctomb(char *s, int wc) {
  if (!s) return 0;
  *s = (char)wc;
  return 1;
}
unsigned long mbrtowc(int *wc, const char *s, unsigned long n, void *ps) {
  (void)ps;
  if (!s) return 0;
  if (n == 0) return (unsigned long)-2;
  if (wc) *wc = (unsigned char)*s;
  return *s ? 1 : 0;
}
unsigned long wcrtomb(char *s, int wc, void *ps) {
  (void)ps;
  if (!s) return 1;
  *s = (char)wc;
  return 1;
}
int mbsinit(const void *ps) { (void)ps; return 1; }
unsigned long mbstowcs(int *dst, const char *src, unsigned long n) {
  unsigned long i = 0;
  while (src[i] && i < n) {
    if (dst) dst[i] = (unsigned char)src[i];
    i = i + 1;
  }
  if (dst && i < n) dst[i] = 0;
  return i;
}
/* mbsrtowcs/wcsrtombs — the restartable string converters (bash's xdupmbstowcs / wide glob).
 * ASCII 1:1; glibc contract: converting through the terminating NUL sets `*src = NULL` and the
 * NUL is not counted. `dst == NULL` sizes without consuming. */
unsigned long mbsrtowcs(int *dst, const char **src, unsigned long len, void *ps) {
  (void)ps;
  const char *s = *src;
  unsigned long i = 0;
  if (!dst) { while (s[i]) i = i + 1; return i; }
  while (i < len) {
    dst[i] = (unsigned char)s[i];
    if (!s[i]) { *src = 0; return i; }
    i = i + 1;
  }
  *src = s + i;
  return i;
}
unsigned long mbsnrtowcs(int *dst, const char **src, unsigned long nms, unsigned long len, void *ps) {
  (void)ps;
  const char *s = *src;
  unsigned long i = 0;
  if (!dst) { while (i < nms && s[i]) i = i + 1; return i; }
  while (i < len && i < nms) {
    dst[i] = (unsigned char)s[i];
    if (!s[i]) { *src = 0; return i; }
    i = i + 1;
  }
  *src = s + i;
  return i;
}
unsigned long wcsrtombs(char *dst, const int **src, unsigned long len, void *ps) {
  (void)ps;
  const int *s = *src;
  unsigned long i = 0;
  if (!dst) { while (s[i]) i = i + 1; return i; }
  while (i < len) {
    dst[i] = (char)s[i];
    if (!s[i]) { *src = 0; return i; }
    i = i + 1;
  }
  *src = s + i;
  return i;
}
/* wide-string ops over 4-byte wchar_t (ASCII payloads only on this path). */
unsigned long wcslen(const int *s) {
  unsigned long n = 0;
  while (s[n]) n = n + 1;
  return n;
}
int wcscmp(const int *a, const int *b) {
  while (*a && *a == *b) { a = a + 1; b = b + 1; }
  return *a - *b;
}
int wcsncmp(const int *a, const int *b, unsigned long n) {
  unsigned long i = 0;
  while (i < n && a[i] && a[i] == b[i]) i = i + 1;
  return i == n ? 0 : a[i] - b[i];
}
int wcscoll(const int *a, const int *b) { return wcscmp(a, b); }
int *wcschr(const int *s, int c) {
  for (;; s = s + 1) {
    if (*s == c) return (int *)s;
    if (!*s) return 0;
  }
}
int *wmemchr(const int *s, int c, unsigned long n) {
  unsigned long i;
  for (i = 0; i < n; i = i + 1)
    if (s[i] == c) return (int *)(s + i);
  return 0;
}
int *wcsdup(const int *s) {
  unsigned long n = wcslen(s) + 1, i;
  int *d = (int *)malloc(n * 4);
  if (!d) return 0;
  for (i = 0; i < n; i = i + 1) d[i] = s[i];
  return d;
}
int wctob(int wc) { return wc >= 0 && wc < 256 ? wc : -1; }
int wcwidth(int wc) { return wc == 0 ? 0 : (wc >= 32 && wc < 127 ? 1 : (wc < 32 ? -1 : 1)); }
int wcswidth(const int *s, unsigned long n) {
  unsigned long i;
  int w = 0;
  for (i = 0; i < n && s[i]; i = i + 1) {
    int c = wcwidth(s[i]);
    if (c < 0) return -1;
    w = w + c;
  }
  return w;
}
/* wctype/iswctype — small dense ids for the POSIX classes bash's pattern code asks for. */
static const char *shim_wctypes_[] = {"alnum", "alpha", "blank", "cntrl", "digit", "graph",
                                      "lower", "print", "punct", "space", "upper", "xdigit"};
unsigned long wctype(const char *name) {
  unsigned long i;
  for (i = 0; i < 12; i = i + 1) {
    const char *t = shim_wctypes_[i];
    unsigned long j = 0;
    while (t[j] && t[j] == name[j]) j = j + 1;
    if (!t[j] && !name[j]) return i + 1;
  }
  return 0;
}
static int isw_digit_(int c) { return c >= '0' && c <= '9'; }
static int isw_lower_(int c) { return c >= 'a' && c <= 'z'; }
static int isw_upper_(int c) { return c >= 'A' && c <= 'Z'; }
static int isw_alpha_(int c) { return isw_lower_(c) || isw_upper_(c); }
static int isw_print_(int c) { return c >= 32 && c < 127; }
int iswctype(int wc, unsigned long t) {
  switch (t) {
    case 1: return isw_alpha_(wc) || isw_digit_(wc);
    case 2: return isw_alpha_(wc);
    case 3: return wc == ' ' || wc == '\t';
    case 4: return (wc >= 0 && wc < 32) || wc == 127;
    case 5: return isw_digit_(wc);
    case 6: return isw_print_(wc) && wc != ' ';
    case 7: return isw_lower_(wc);
    case 8: return isw_print_(wc);
    case 9: return isw_print_(wc) && wc != ' ' && !isw_alpha_(wc) && !isw_digit_(wc);
    case 10: return wc == ' ' || (wc >= '\t' && wc <= '\r');
    case 11: return isw_upper_(wc);
    case 12: return isw_digit_(wc) || (wc >= 'a' && wc <= 'f') || (wc >= 'A' && wc <= 'F');
    default: return 0;
  }
}
int iswalnum(int wc) { return iswctype(wc, 1); }
int iswlower(int wc) { return iswctype(wc, 7); }
int iswupper(int wc) { return iswctype(wc, 11); }
int iswprint(int wc) { return iswctype(wc, 8); }
int towlower(int wc) { return isw_upper_(wc) ? wc + 32 : wc; }
int towupper(int wc) { return isw_lower_(wc) ? wc - 32 : wc; }

/* --- band 2b: string oddments the on-ramp doesn't synthesize ---------------------------------- */
extern int strcmp(const char *a, const char *b);
int strcoll(const char *a, const char *b) { return strcmp(a, b); }
static int lower_(int c);
unsigned long strnlen(const char *s, unsigned long n) {
  unsigned long i = 0;
  while (i < n && s[i]) i = i + 1;
  return i;
}
char *strdup(const char *s) {
  unsigned long n = strlen(s) + 1, i;
  char *d = (char *)malloc(n);
  if (!d) return 0;
  for (i = 0; i < n; i = i + 1) d[i] = s[i];
  return d;
}
char *strncpy(char *d, const char *s, unsigned long n) {
  unsigned long i = 0;
  while (i < n && s[i]) { d[i] = s[i]; i = i + 1; }
  while (i < n) { d[i] = 0; i = i + 1; } /* POSIX: pad to n */
  return d;
}
char *strcat(char *d, const char *s) {
  unsigned long i = strlen(d), j = 0;
  while (s[j]) { d[i + j] = s[j]; j = j + 1; }
  d[i + j] = 0;
  return d;
}
char *strchrnul(const char *s, int c) {
  while (*s && *s != (char)c) s = s + 1;
  return (char *)s;
}
char *strstr(const char *h, const char *n) {
  if (!*n) return (char *)h;
  for (; *h; h = h + 1) {
    unsigned long i = 0;
    while (n[i] && h[i] == n[i]) i = i + 1;
    if (!n[i]) return (char *)h;
  }
  return 0;
}
char *strcasestr(const char *h, const char *n) {
  if (!*n) return (char *)h;
  for (; *h; h = h + 1) {
    unsigned long i = 0;
    while (n[i] && lower_((unsigned char)h[i]) == lower_((unsigned char)n[i])) i = i + 1;
    if (!n[i]) return (char *)h;
  }
  return 0;
}
/* imaxdiv — bash's printf builtin divides intmax_t by the base with it. */
typedef struct { long quot; long rem; } shim_imaxdiv_t;
shim_imaxdiv_t imaxdiv(long num, long den) {
  shim_imaxdiv_t r;
  r.quot = num / den;
  r.rem = num % den;
  return r;
}
static int lower_(int c) { return c >= 'A' && c <= 'Z' ? c + 32 : c; }
int strcasecmp(const char *a, const char *b) {
  unsigned long i = 0;
  while (a[i] && lower_((unsigned char)a[i]) == lower_((unsigned char)b[i])) i = i + 1;
  return lower_((unsigned char)a[i]) - lower_((unsigned char)b[i]);
}
int strncasecmp(const char *a, const char *b, unsigned long n) {
  unsigned long i = 0;
  while (i < n && a[i] && lower_((unsigned char)a[i]) == lower_((unsigned char)b[i])) i = i + 1;
  if (i == n) return 0;
  return lower_((unsigned char)a[i]) - lower_((unsigned char)b[i]);
}
char *strsignal(int sig) { (void)sig; return "Signal"; }
/* strerror — bash prints these in its error messages ("bash: f: No such file or directory"), so
 * the strings must match the native oracle's glibc for the errnos the personality produces; the
 * long tail gets a generic string. */
char *strerror(int e) {
  switch (e) {
    case 1: return "Operation not permitted";
    case 2: return "No such file or directory";
    case 3: return "No such process";
    case 4: return "Interrupted system call";
    case 5: return "Input/output error";
    case 9: return "Bad file descriptor";
    case 10: return "No child processes";
    case 11: return "Resource temporarily unavailable";
    case 12: return "Cannot allocate memory";
    case 13: return "Permission denied";
    case 14: return "Bad address";
    case 17: return "File exists";
    case 20: return "Not a directory";
    case 21: return "Is a directory";
    case 22: return "Invalid argument";
    case 25: return "Inappropriate ioctl for device";
    case 28: return "No space left on device";
    case 29: return "Illegal seek";
    case 32: return "Broken pipe";
    case 34: return "Numerical result out of range";
    case 36: return "File name too long";
    default: return "Unknown error";
  }
}
/* qsort — pure-computation libc the on-ramp does NOT synthesize (the tcl_shim heapsort, verbatim:
 * O(n log n) worst case, no recursion, no scratch; comparator via `call_indirect`). bash sorts
 * word lists, completion matches, and the hash-table walks with it. */
static void bswap_(unsigned char *a, unsigned char *b, unsigned long n) {
  while (n--) {
    unsigned char t = *a;
    *a++ = *b;
    *b++ = t;
  }
}
void qsort(void *base, unsigned long n, unsigned long size,
           int (*cmp)(const void *, const void *)) {
  unsigned char *a = (unsigned char *)base;
  if (n < 2 || size == 0) return;
  for (unsigned long start = n / 2; start-- > 0;) {
    unsigned long root = start;
    for (;;) {
      unsigned long child = 2 * root + 1;
      if (child >= n) break;
      if (child + 1 < n && cmp(a + child * size, a + (child + 1) * size) < 0) child++;
      if (cmp(a + root * size, a + child * size) >= 0) break;
      bswap_(a + root * size, a + child * size, size);
      root = child;
    }
  }
  for (unsigned long end = n; end-- > 1;) {
    bswap_(a, a + end * size, size);
    unsigned long root = 0;
    for (;;) {
      unsigned long child = 2 * root + 1;
      if (child >= end) break;
      if (child + 1 < end && cmp(a + child * size, a + (child + 1) * size) < 0) child++;
      if (cmp(a + root * size, a + child * size) >= 0) break;
      bswap_(a + root * size, a + child * size, size);
      root = child;
    }
  }
}

/* --- band 4: fd/process oddments composed over band 0 ----------------------------------------- */
int killpg(int pgrp, int sig) { return kill(-pgrp, sig); }
/* eaccess/faccessat over the stat op (the memfs reports exec bits on registered executables —
 * the #801 contract PATH search needs). st layout: {mode, size} per POSIX.md. */
static int shim_access_(const char *path, int mode) {
  long st[4];
  if (stat(path, st) != 0) return -1;
  if ((mode & 1) && !(st[0] & 0111)) return -1; /* X_OK against the exec bits */
  return 0;
}
int eaccess(const char *path, int mode) { return shim_access_(path, mode); }
int faccessat(int dirfd, const char *path, int mode, int flags) {
  (void)dirfd; (void)flags;
  return shim_access_(path, mode);
}
/* ioctl: the one request bash uses on this path is TIOCGWINSZ (checkwinsize); route it to the
 * #797 termios op and convert to struct winsize {u16 row, col, xpix, ypix}. */
int ioctl(int fd, unsigned long req, void *arg) {
  if (req == 0x5413 && arg) { /* TIOCGWINSZ */
    int ws[2];
    if (tcgetwinsize(fd, ws) != 0) return -1;
    unsigned short *w = (unsigned short *)arg;
    w[0] = (unsigned short)ws[0];
    w[1] = (unsigned short)ws[1];
    w[2] = 0;
    w[3] = 0;
    return 0;
  }
  return -1;
}
/* sigset_t manipulation — pure guest bit ops over the LOW u64 of the (glibc-sized) sigset the
 * caller allocates; the personality's sigprocmask op reads exactly that u64 (op 40). */
int sigemptyset(unsigned long *set) { *set = 0; return 0; }
int sigaddset(unsigned long *set, int sig) { *set |= 1ul << sig; return 0; }
int sigdelset(unsigned long *set, int sig) { *set &= ~(1ul << sig); return 0; }
int sigismember(const unsigned long *set, int sig) { return (*set >> sig) & 1; }
int __libc_current_sigrtmin(void) { return 34; }
int __libc_current_sigrtmax(void) { return 63; }
/* Sockets / dlopen / select: no surface yet — clean failures, never escapes. */
int socket(int d, int t, int p) { (void)d; (void)t; (void)p; return -1; }
int connect(int fd, const void *sa, unsigned int len) { (void)fd; (void)sa; (void)len; return -1; }
int getaddrinfo(const char *n, const char *s, const void *h, void **res) {
  (void)n; (void)s; (void)h; (void)res;
  return -2; /* EAI_NONAME */
}
void freeaddrinfo(void *ai) { (void)ai; }
const char *gai_strerror(int e) { (void)e; return "name resolution unavailable"; }
int getpeername(int fd, void *sa, unsigned int *len) { (void)fd; (void)sa; (void)len; return -1; }
void *dlopen(const char *f, int flags) { (void)f; (void)flags; return 0; }
void *dlsym(void *h, const char *s) { (void)h; (void)s; return 0; }
int dlclose(void *h) { (void)h; return 0; }
char *dlerror(void) { return "dynamic loading unavailable"; }
int select(int n, void *r, void *w, void *e, void *tv) {
  (void)n; (void)r; (void)w; (void)e; (void)tv;
  return -1;
}
int pselect(int n, void *r, void *w, void *e, const void *ts, const void *mask) {
  (void)n; (void)r; (void)w; (void)e; (void)ts; (void)mask;
  return -1;
}
int chown(const char *p, unsigned int u, unsigned int g) { (void)p; (void)u; (void)g; return 0; }
int fchmod(int fd, unsigned int m) { (void)fd; (void)m; return 0; }
long readlink(const char *p, char *buf, unsigned long n) {
  (void)p; (void)buf; (void)n;
  return -1; /* the memfs has no symlinks */
}
int mkstemp(char *tmpl) { (void)tmpl; return -1; }
char *mktemp(char *tmpl) { if (tmpl) *tmpl = 0; return tmpl; }
char *mkdtemp(char *tmpl) { (void)tmpl; return 0; }

/* strtol family — the C23-mangled names clang emits. */
long __isoc23_strtol(const char *s, char **end, int base) {
  long v = 0, neg = 0;
  while (*s == ' ' || *s == '\t') s = s + 1;
  if (*s == '-') { neg = 1; s = s + 1; }
  else if (*s == '+') s = s + 1;
  if (base == 0) {
    if (s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) { base = 16; s = s + 2; }
    else if (s[0] == '0') base = 8;
    else base = 10;
  } else if (base == 16 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) {
    s = s + 2;
  }
  for (;;) {
    int c = (unsigned char)*s;
    int d;
    if (c >= '0' && c <= '9') d = c - '0';
    else if (c >= 'a' && c <= 'z') d = c - 'a' + 10;
    else if (c >= 'A' && c <= 'Z') d = c - 'A' + 10;
    else break;
    if (d >= base) break;
    v = v * base + d;
    s = s + 1;
  }
  if (end) *end = (char *)s;
  return neg ? -v : v;
}
unsigned long __isoc23_strtoumax(const char *s, char **end, int base) {
  return (unsigned long)__isoc23_strtol(s, end, base);
}

/* --- band 5: the `__px_` bridges — posix_libc guest code on the on-ramp path ------------------
 * The personality's own guest libc (`../posix_libc/regex.c` here; `exec.c` in the fork/exec
 * slice) declares its ops as `__px_<name>(int cap, …)`; on the on-ramp band 0 serves the same
 * surface. Bridge the two vocabularies with trivial wrappers — the `cap` dummy is dropped. */
long __px_malloc(int cap, long n) { (void)cap; return (long)malloc((unsigned long)n); }
long __px_free(int cap, long p) { (void)cap; free((void *)p); return 0; }
/* The `../posix_libc/exec.c` externs (#801 execve/execv/execvp, linked as guest code): forwards
 * to the personality ops. Its `__vm_exec_module` is a core builtin the on-ramp lowers directly. */
long __px_exec_resolve(int cap, long path, long len) {
  (void)cap;
  return px_call_(53, path, len, 0, 0); /* PX_EXEC_RESOLVE */
}
long __px_getenv(int cap, long name, long len) {
  (void)cap;
  return px_call_(11, name, len, 0, 0); /* PX_GETENV (the personality env, not main's envp) */
}
long __px_open(int cap, long path, long len, long flags) {
  (void)cap;
  return px_call_(PX_OPEN, path, len, flags, 0);
}
long __px_read(int cap, long fd, long buf, long len) {
  (void)cap;
  return px_call_(PX_READ, fd, buf, len, 0);
}
long __px_close(int cap, long fd) {
  (void)cap;
  return px_call_(PX_CLOSE, fd, 0, 0, 0);
}
