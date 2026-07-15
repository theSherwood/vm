/* Guest IPC/shared-memory shim — the single-process collapses of Postgres's multi-process IPC
 * (slice CI, gap #11i). `postgres --single` still sets up a shared-memory segment (the buffer pool,
 * lock tables, …) and semaphores in early startup, before it prints its banner — so these are the
 * silent trap right after `find_my_exec`. In a single process there is no *sharing* to do: shared
 * memory is just anonymous memory (one address space), and semaphores are uncontended no-ops.
 *
 * `#include`d into a driver under `-DSVM_GUEST`, after os_shim.c (it uses malloc/free/memset, which
 * the on-ramp synthesizes).
 */

#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/shm.h>

#include "shim_errno.h" /* shim_errno, shared with os_shim (include-guarded) */

/* ---- anonymous mmap == plain zeroed memory (one address space, nothing to map/share) --------- */
void *mmap(void *addr, size_t len, int prot, int flags, int fd, off_t off) {
  (void)addr;
  (void)prot;
  (void)fd;
  (void)off;
  if (!(flags & MAP_ANONYMOUS)) return MAP_FAILED; /* file-backed mmap goes through the fs cap, not here */
  void *p = malloc(len);
  if (!p) return MAP_FAILED;
  memset(p, 0, len); /* MAP_ANONYMOUS is zero-filled */
  return p;
}
int munmap(void *addr, size_t len) {
  (void)len;
  free(addr);
  return 0;
}
int madvise(void *addr, size_t len, int advice) {
  (void)addr;
  (void)len;
  (void)advice;
  return 0;
}
int mlock(const void *addr, size_t len) { (void)addr; (void)len; return 0; }
int munlock(const void *addr, size_t len) { (void)addr; (void)len; return 0; }
int posix_fadvise(int fd, off_t off, off_t len, int advice) {
  (void)fd;
  (void)off;
  (void)len;
  (void)advice;
  return 0;
}
int posix_fallocate(int fd, off_t off, off_t len) {
  (void)fd;
  (void)off;
  (void)len;
  return 0;
}

/* ---- POSIX unnamed semaphores: a real **counting** semaphore, single-process -------------------
 * These must carry a value, not be blanket no-ops: Postgres' `PGSemaphoreReset` *drains* a semaphore
 * with `while (sem_trywait(sem) >= 0) ;`, so a `sem_trywait` that always "succeeds" spins forever. Keep
 * the count in the `sem_t` itself (an int fits in its 32-byte storage). One process, so `sem_wait` never
 * has to *block* — Postgres' lock protocol is balanced, and a wait-at-zero (would-block) just succeeds
 * rather than deadlock; the key is that `sem_trywait` fails (EAGAIN) at zero so the drain terminates. */
#include <semaphore.h>
int sem_init(sem_t *s, int pshared, unsigned int value) {
  (void)pshared;
  *(volatile int *)s = (int)value;
  return 0;
}
int sem_destroy(sem_t *s) {
  (void)s;
  return 0;
}
int sem_post(sem_t *s) {
  (*(volatile int *)s)++;
  return 0;
}
int sem_wait(sem_t *s) {
  volatile int *v = (volatile int *)s;
  if (*v > 0) (*v)--; /* uncontended acquire; at zero, succeed anyway (single process never blocks) */
  return 0;
}
int sem_trywait(sem_t *s) {
  volatile int *v = (volatile int *)s;
  if (*v > 0) {
    (*v)--;
    return 0;
  }
  shim_errno = 11; /* EAGAIN — empty; lets PGSemaphoreReset's drain loop terminate */
  return -1;
}

/* ---- System V shared memory: in one process a segment is just private memory --------------------
 * `shared_memory_type = mmap` (the default) puts the *main* shared memory in an anonymous mmap (above);
 * Postgres still creates a **tiny** SysV segment as a startup interlock (the `PGShmemHeader`), and — with
 * `dynamic_shared_memory_type = sysv` (the demo config, since the single process needs no cross-process
 * POSIX DSM) — its dynamic-shared-memory segments come through here too. So several segments coexist:
 * track each by a small id → (size, addr) table, `malloc` on attach, `free` on detach/remove. */
enum { SHM_MAX = 64 };
static struct {
  size_t size;
  void *addr;
  int used;
} g_seg[SHM_MAX];
static int g_seg_next = 1; /* id 0 is never handed out (shmget failure sentinel is -1 anyway) */

int shmget(int key, size_t size, int flag) {
  (void)key;
  (void)flag;
  int id = g_seg_next++;
  if (id >= SHM_MAX) {
    shim_errno = 28; /* ENOSPC */
    return -1;
  }
  g_seg[id].size = size ? size : 4096;
  g_seg[id].addr = (void *)0;
  g_seg[id].used = 1;
  return id;
}
void *shmat(int id, const void *addr, int flag) {
  (void)addr;
  (void)flag;
  if (id <= 0 || id >= SHM_MAX || !g_seg[id].used) {
    shim_errno = 22; /* EINVAL */
    return (void *)-1;
  }
  void *p = malloc(g_seg[id].size);
  if (!p) return (void *)-1;
  memset(p, 0, g_seg[id].size);
  g_seg[id].addr = p;
  return p;
}
int shmdt(const void *addr) {
  free((void *)addr); /* single process: attach == the allocation */
  return 0;
}
int shmctl(int id, int cmd, struct shmid_ds *buf) {
  if (cmd == IPC_RMID) { /* mark removed; the attached mapping stays valid until detached (SysV) */
    if (id > 0 && id < SHM_MAX) g_seg[id].used = 0;
    return 0;
  }
  if (buf) memset(buf, 0, sizeof *buf); /* IPC_STAT: nattch = 0 (no other attacher) */
  return 0;
}

/* POSIX shm (`shm_open`) is unused — the demo config routes DSM through SysV above (single process).
 * Fail closed with ENOSYS so a caller that probes for it falls back cleanly rather than reading a
 * bogus fd (and Postgres' error text shows a real errno, not "Success"). */
int shm_open(const char *name, int flag, unsigned int mode) {
  (void)name;
  (void)flag;
  (void)mode;
  shim_errno = 38; /* ENOSYS */
  return -1;
}
int shm_unlink(const char *name) { (void)name; return 0; }
