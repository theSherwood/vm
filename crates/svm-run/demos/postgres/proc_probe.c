/* proc_probe.c — the guest process/time/signal stubs return their fixed, deterministic values
 * (slice CC). These are NOT differential vs native (a native `getpid`/`time` is nondeterministic):
 * the test asserts the guest's stdout equals the fixed expected report. Runs on the bare powerbox.
 */

#include <signal.h>
#include <stdio.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#ifdef SVM_GUEST
#include "proc_shim.c"
#endif

int main(void) {
  printf("pid=%d ppid=%d\n", getpid(), getppid());
  printf("uid=%d euid=%d gid=%d egid=%d\n", getuid(), geteuid(), getgid(), getegid());
  printf("umask=%d setsid=%d\n", (int)umask(0), (int)setsid());

  struct timeval tv;
  int r = gettimeofday(&tv, 0);
  printf("gtod r=%d sec=%d usec=%d\n", r, (int)tv.tv_sec, (int)tv.tv_usec);
  struct timespec ts;
  r = clock_gettime(CLOCK_REALTIME, &ts);
  printf("clock r=%d sec=%d nsec=%d\n", r, (int)ts.tv_sec, (int)ts.tv_nsec);
  printf("time=%d\n", (int)time(0));

  struct timespec rq = {0, 0}, rm;
  printf("nanosleep=%d\n", nanosleep(&rq, &rm));

  struct rlimit rl;
  r = getrlimit(RLIMIT_NOFILE, &rl);
  printf("rlimit r=%d inf=%d\n", r, rl.rlim_cur == RLIM_INFINITY);

  sigset_t ss;
  struct sigaction sa;
  printf("sigempty=%d sigaction=%d\n", sigemptyset(&ss), sigaction(SIGINT, 0, &sa));
  printf("sigprocmask=%d kill=%d raise=%d\n", sigprocmask(SIG_BLOCK, 0, 0), kill(1, 0), raise(0));
  return 0;
}
