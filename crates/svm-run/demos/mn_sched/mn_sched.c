/* A guest-built M:N green-thread scheduler — sharded, thread-per-core (SCHEDULING.md, D56/D57).
 *
 * The VM ships only primitives: vCPUs (`thread.spawn`, 1:1 OS threads), stackful fibers
 * (`cont.*`), and the futex + C11 atomics. *This program is the scheduler* — proof the
 * abstractions compose into a real M:N runtime with no scheduler baked into the VM.
 *
 * Shape: NWORKERS OS threads (vCPUs), each running its own cooperative round-robin over
 * TASKS_PER_WORKER green tasks (fibers). Tasks are pinned to their worker (fibers are
 * thread-affine by design, D57); the workers run in parallel and coordinate only through one
 * shared atomic counter. Each task does STEPS units of work, yielding cooperatively between each.
 * The grand total (NWORKERS * TASKS_PER_WORKER * STEPS) is interleaving-invariant, hence
 * deterministic — so it is checkable across the interpreter (the M:N oracle) and the JIT.
 *
 * NB: the shipped MVP `malloc` is a single-threaded bump allocator, so the fiber data stacks are
 * pre-allocated on the main thread (a thread-safe guest `malloc` is a libc follow-up, not a VM
 * concern). Allocation policy is the guest's job — the VM only supplies the window + the `map` cap. */
#include <pthread.h>
#include <stdlib.h>

/* VM fiber primitive (cont.* — a stackful coroutine; see SCHEDULING.md). */
int  __vm_fiber_new(long (*f)(long), void *stack);
long __vm_fiber_resume(int k, long arg, int *done);
long __vm_fiber_suspend(long value);
/* VM atomic primitive (the `<stdatomic.h>` macros are non-atomic stand-ins; this is the real one). */
long __vm_atomic_add(void *p, long v); /* lock-free fetch-add over the shared window; returns old */
int  write(int fd, char *buf, long n);

#define NWORKERS         4
#define TASKS_PER_WORKER 8
#define NTOTAL           (NWORKERS * TASKS_PER_WORKER)
#define STEPS            32
#define STACK            16384

static long  g_total = 0;
static char *g_stacks[NTOTAL]; /* one data stack per green task, malloc'd on main */

/* A green task: STEPS units of shared work, yielding to its scheduler between each. */
static long task(long arg) {
  (void)arg;
  for (int i = 0; i < STEPS; i++) {
    __vm_atomic_add(&g_total, 1);
    __vm_fiber_suspend(0); /* cooperative yield — the heart of the green-thread switch */
  }
  return 0;
}

/* One worker's cooperative scheduler: round-robin every runnable fiber until all have returned. */
static void run_scheduler(char **stacks, int ntasks) {
  int handles[TASKS_PER_WORKER];
  char done[TASKS_PER_WORKER];
  for (int i = 0; i < ntasks; i++) {
    done[i] = 0;
    handles[i] = __vm_fiber_new(task, stacks[i]); /* each fiber owns its own data stack */
  }
  int remaining = ntasks;
  while (remaining > 0) {
    for (int i = 0; i < ntasks; i++) {
      if (done[i]) continue;
      int d = 0;
      __vm_fiber_resume(handles[i], 0, &d);
      if (d) { done[i] = 1; remaining -= 1; }
    }
  }
}

static void *worker(void *arg) {
  long w = (long)arg;
  run_scheduler(&g_stacks[w * TASKS_PER_WORKER], TASKS_PER_WORKER);
  return 0;
}

static void print_long(long v) {
  char buf[24];
  int n = 0;
  if (v == 0) { buf[n++] = '0'; }
  char tmp[24];
  int t = 0;
  while (v > 0) { tmp[t++] = (char)('0' + (v % 10)); v /= 10; }
  while (t > 0) { buf[n++] = tmp[--t]; }
  buf[n++] = '\n';
  write(1, buf, n);
}

int main(void) {
  for (int i = 0; i < NTOTAL; i++) g_stacks[i] = malloc(STACK); /* single-threaded: no race */

  pthread_t workers[NWORKERS];
  for (int i = 0; i < NWORKERS; i++) pthread_create(&workers[i], 0, worker, (void *)(long)i);
  for (int i = 0; i < NWORKERS; i++) pthread_join(workers[i], 0);

  print_long(g_total); /* all workers joined → no race; 4 * 8 * 32 = 1024 */
  return 0;
}
