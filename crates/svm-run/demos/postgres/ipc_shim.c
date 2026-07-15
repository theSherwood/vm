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

/* ---- POSIX unnamed semaphores: uncontended in one process (no waiting ever blocks) ----------- */
#include <semaphore.h>
int sem_init(sem_t *s, int pshared, unsigned int value) {
  (void)s;
  (void)pshared;
  (void)value;
  return 0;
}
int sem_destroy(sem_t *s) { (void)s; return 0; }
int sem_post(sem_t *s) { (void)s; return 0; }
int sem_wait(sem_t *s) { (void)s; return 0; }
int sem_trywait(sem_t *s) { (void)s; return 0; }

/* ---- System V shared memory / shm_open: a single process needs no cross-process segment ------ */
int shmget(int key, size_t size, int flag) { (void)key; (void)size; (void)flag; return 1; }
void *shmat(int id, const void *addr, int flag) {
  (void)id;
  (void)addr;
  (void)flag;
  return (void *)-1; /* force Postgres onto the anonymous-mmap path, which we do support */
}
int shmdt(const void *addr) { (void)addr; return 0; }
int shmctl(int id, int cmd, void *buf) { (void)id; (void)cmd; (void)buf; return 0; }
int shm_open(const char *name, int flag, unsigned int mode) {
  (void)name;
  (void)flag;
  (void)mode;
  return -1; /* no POSIX shm object store; the anonymous-mmap path is used instead */
}
int shm_unlink(const char *name) { (void)name; return 0; }
