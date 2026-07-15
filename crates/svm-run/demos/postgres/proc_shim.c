/* Guest process/time/signal shim — the deterministic OS stubs Postgres calls that carry no data of
 * their own (slice CC, Postgres runtime gap #11d). A single-user `postgres --single` in the sandbox
 * has one process, one thread, a frozen clock, and no signal delivery, so these wrappers return
 * fixed, valid values rather than reaching a host: identity/credentials are constants, the clock is
 * a fixed epoch, signal masks are inert, and `sleep`s are no-ops. (`os_shim.c` covers the syscalls
 * that DO carry data — the `fs`-cap file/directory surface; `libc_shim.c` the pure-computation libc.)
 *
 * Determinism is the point: a guest that asks the time twice gets the same answer, and nothing here
 * depends on the host, so a run is reproducible. Credentials are deliberately **non-root**
 * (`geteuid()==1000`) so Postgres's "do not run as root" guard passes.
 *
 * `#include`d into a driver under `-DSVM_GUEST`, like the other shims.
 */

#include <pwd.h>
#include <signal.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

/* A fixed epoch for every clock the guest reads (2001-09-09T01:46:40Z — a round `time_t`). */
#define SHIM_EPOCH 1000000000L

/* ---- identity / credentials (constants; deliberately non-root) ------------------------------- */
pid_t getpid(void) { return 1; }
pid_t getppid(void) { return 0; }
uid_t getuid(void) { return 1000; }
uid_t geteuid(void) { return 1000; }
gid_t getgid(void) { return 1000; }
gid_t getegid(void) { return 1000; }
pid_t setsid(void) { return 1; }
pid_t getsid(pid_t p) { (void)p; return 1; }
mode_t umask(mode_t m) { (void)m; return 022; }
int setpgid(pid_t a, pid_t b) { (void)a; (void)b; return 0; }

/* Password-database lookup: Postgres resolves the OS user name from `getpwuid(geteuid())->pw_name`
 * (`GetUserName`, for the bootstrap superuser). One process, one fixed identity — return a single
 * static entry matching the non-root uid/gid above. `getpwnam` mirrors it by name. */
struct passwd *getpwuid(uid_t uid) {
  (void)uid;
  static struct passwd pw = {
      .pw_name = (char *)"postgres",
      .pw_passwd = (char *)"x",
      .pw_uid = 1000,
      .pw_gid = 1000,
      .pw_gecos = (char *)"",
      .pw_dir = (char *)"/",
      .pw_shell = (char *)"/bin/sh",
  };
  return &pw;
}
struct passwd *getpwnam(const char *name) { (void)name; return getpwuid(1000); }

/* ---- time (a frozen clock) ------------------------------------------------------------------- */
int gettimeofday(struct timeval *tv, void *tz) {
  (void)tz;
  tv->tv_sec = SHIM_EPOCH;
  tv->tv_usec = 0;
  return 0;
}
int clock_gettime(clockid_t clk, struct timespec *ts) {
  (void)clk;
  ts->tv_sec = SHIM_EPOCH;
  ts->tv_nsec = 0;
  return 0;
}
time_t time(time_t *t) {
  if (t) *t = SHIM_EPOCH;
  return SHIM_EPOCH;
}
int nanosleep(const struct timespec *req, struct timespec *rem) {
  (void)req;
  if (rem) {
    rem->tv_sec = 0;
    rem->tv_nsec = 0;
  }
  return 0; /* single-threaded: nothing to wait for */
}
unsigned int sleep(unsigned int s) { (void)s; return 0; }
int setitimer(int which, const struct itimerval *nv, struct itimerval *ov) {
  (void)which;
  (void)nv;
  (void)ov;
  return 0;
}

/* ---- resource limits / usage (unlimited / zero) ---------------------------------------------- */
int getrlimit(int res, struct rlimit *rl) {
  (void)res;
  if (rl) {
    rl->rlim_cur = RLIM_INFINITY;
    rl->rlim_max = RLIM_INFINITY;
  }
  return 0;
}
int setrlimit(int res, const struct rlimit *rl) {
  (void)res;
  (void)rl;
  return 0;
}
int getrusage(int who, struct rusage *ru) {
  (void)who;
  if (ru) {
    char *p = (char *)ru;
    for (unsigned i = 0; i < sizeof *ru; i++) p[i] = 0;
  }
  return 0;
}

/* ---- signals (inert: masks track nothing, delivery never happens) ---------------------------- */
int sigemptyset(sigset_t *s) {
  if (s) {
    char *p = (char *)s;
    for (unsigned i = 0; i < sizeof *s; i++) p[i] = 0;
  }
  return 0;
}
int sigfillset(sigset_t *s) {
  if (s) {
    char *p = (char *)s;
    for (unsigned i = 0; i < sizeof *s; i++) p[i] = (char)0xff;
  }
  return 0;
}
int sigaddset(sigset_t *s, int n) { (void)s; (void)n; return 0; }
int sigdelset(sigset_t *s, int n) { (void)s; (void)n; return 0; }
int sigismember(const sigset_t *s, int n) { (void)s; (void)n; return 0; }
int sigprocmask(int how, const sigset_t *set, sigset_t *old) {
  (void)how;
  (void)set;
  if (old) sigemptyset(old);
  return 0;
}
int sigaction(int sig, const struct sigaction *act, struct sigaction *old) {
  (void)sig;
  (void)act;
  if (old) {
    char *p = (char *)old;
    for (unsigned i = 0; i < sizeof *old; i++) p[i] = 0;
  }
  return 0;
}
int kill(pid_t pid, int sig) { (void)pid; (void)sig; return 0; }
int raise(int sig) { (void)sig; return 0; }

/* ---- fatal-exit paths ------------------------------------------------------------------------ */
int atexit(void (*fn)(void)) { (void)fn; return 0; } /* one-shot process: handlers never run */
void abort(void) {
  _exit(134); /* 128 + SIGABRT */
  __builtin_unreachable();
}
void __assert_fail(const char *assertion, const char *file, unsigned line, const char *func) {
  (void)assertion;
  (void)file;
  (void)line;
  (void)func;
  _exit(134);
  __builtin_unreachable();
}
