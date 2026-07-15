/* Guest event-loop / latch infrastructure — `signalfd`/`epoll`/`eventfd`.
 *
 * Postgres's `latch.c` was configured for the Linux `WAIT_USE_EPOLL` backend: `InitializeLatchSupport`
 * opens a `signalfd` for SIGURG and a `WaitEventSet` builds an `epoll` fd it `epoll_ctl`s the latch and
 * socket onto. In a single process reading one query synchronously from stdin, no signal is ever
 * delivered and the latch is never actually waited on to *block* — so these collapse to bookkeeping:
 * hand out distinct fake fds (well above the fs cap's file fds, which start at 3, so they never
 * collide) and accept every registration. `epoll_wait` returns "timed out, no events" — the latch flag
 * (checked by `WaitLatch` around the wait) carries the real state; there is nothing to deliver here.
 *
 * `#include`d into a driver under `-DSVM_GUEST`.
 */

#include <stddef.h>
#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <sys/signalfd.h>

/* Distinct fake descriptors, high enough never to collide with the fs cap's file fds (>= 3) or the
 * powerbox stream fds (0/1/2). Opaque to Postgres — passed back only to epoll_ctl / close. */
static int g_next_event_fd = 900;
static int alloc_event_fd(void) { return g_next_event_fd++; }

int signalfd(int fd, const sigset_t *mask, int flags) {
  (void)mask;
  (void)flags;
  return fd >= 0 ? fd : alloc_event_fd(); /* re-use the caller's fd if it passed one (SFD semantics) */
}
int eventfd(unsigned int initval, int flags) {
  (void)initval;
  (void)flags;
  return alloc_event_fd();
}

int epoll_create(int size) {
  (void)size;
  return alloc_event_fd();
}
int epoll_create1(int flags) {
  (void)flags;
  return alloc_event_fd();
}
int epoll_ctl(int epfd, int op, int fd, struct epoll_event *event) {
  (void)epfd;
  (void)op;
  (void)fd;
  (void)event;
  return 0; /* every add/mod/del "succeeds"; there is nothing to arm */
}
int epoll_wait(int epfd, struct epoll_event *events, int maxevents, int timeout) {
  (void)epfd;
  (void)events;
  (void)maxevents;
  (void)timeout;
  return 0; /* no events ready — the latch flag (checked by the caller) holds the real state */
}
