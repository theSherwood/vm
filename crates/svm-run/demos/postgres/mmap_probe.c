/* mmap_probe.c — differential for the guest anonymous-mmap shim (slice CI).
 *
 * `mmap(MAP_ANONYMOUS)` must return zero-filled, writable memory (one address space — there's no
 * sharing to do); the guest (`-DSVM_GUEST`, ipc_shim.c → malloc + zero) must behave the same as
 * native: freshly zeroed, then holds what's written, then `munmap` succeeds. Deterministic — runs on
 * the bare powerbox. (The `sem_*`/`shm*` collapses are no-ops with no observable output, exercised by
 * the boot, not here.)
 */

#include <stdio.h>
#include <string.h>
#include <sys/mman.h>

#ifdef SVM_GUEST
#include "ipc_shim.c"
#endif

int main(void) {
  size_t n = 4096;
  unsigned char *p = mmap((void *)0, n, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
  printf("mmap_ok=%d\n", p != MAP_FAILED);

  int zeroed = 1;
  for (size_t i = 0; i < n; i++)
    if (p[i] != 0) {
      zeroed = 0;
      break;
    }
  printf("zeroed=%d\n", zeroed);

  for (size_t i = 0; i < n; i++) p[i] = (unsigned char)(i * 7 + 3);
  int held = 1;
  for (size_t i = 0; i < n; i++)
    if (p[i] != (unsigned char)(i * 7 + 3)) {
      held = 0;
      break;
    }
  printf("held=%d\n", held);
  printf("munmap=%d\n", munmap(p, n));
  return 0;
}
