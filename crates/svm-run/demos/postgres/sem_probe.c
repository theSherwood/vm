/* sem_probe.c — differential for the guest POSIX **counting** semaphore (ipc_shim.c).
 *
 * The shim must carry a real count, not be a blanket no-op: Postgres' `PGSemaphoreReset` drains a
 * semaphore with `while (sem_trywait(s) >= 0) ;`, so a `sem_trywait` that always "succeeds" spins
 * forever (this exact bug hung the boot). A single-process unnamed semaphore behaves identically under
 * glibc, so this is a real vs-native differential: init to 2, drain with `sem_trywait` (fail EAGAIN at
 * zero), `sem_post`, drain again. Only `sem_trywait` is used — a `sem_wait` at zero would block native.
 */

#include <semaphore.h>
#include <stdio.h>

#ifdef SVM_GUEST
#include "ipc_shim.c"
#endif

int main(void) {
  sem_t s;
  sem_init(&s, 0, 2);
  int a = sem_trywait(&s); /*  0: 2 -> 1 */
  int b = sem_trywait(&s); /*  0: 1 -> 0 */
  int c = sem_trywait(&s); /* -1: empty (EAGAIN) */
  sem_post(&s);            /*     0 -> 1 */
  int d = sem_trywait(&s); /*  0: 1 -> 0 */
  int e = sem_trywait(&s); /* -1: empty */
  printf("%d %d %d %d %d\n", a, b, c, d, e);
  return 0;
}
