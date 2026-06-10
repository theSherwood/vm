/* A guest-built **work-stealing** M:N scheduler over **stackless** tasks (SCHEDULING.md, D56/D57).
 *
 * Contrast with demos/mn_sched (sharded, *stackful* fibers, tasks pinned per worker). Here a task
 * is a **state machine** — a plain struct (its resume state is the `i` field) — so it is just
 * *data* and moves freely between worker threads. That makes **work-stealing** possible with **no
 * VM change**: an idle worker steals a task from a busy sibling (or pulls from a global injector)
 * and resumes it on its own thread — a pointer hand-off, safe by construction (no native stack to
 * migrate). This is the tokio architecture: a global injector queue + per-worker deques + stealing.
 *
 * Everything here is guest code over the VM's primitives: vCPUs (`thread.spawn`), the futex (under
 * `pthread_mutex`), and C11 atomics. No fibers, no scheduler in the VM (D56). The grand total
 * (NTASKS * STEPS) is interleaving-invariant, so it is identical on the interpreter (the M:N
 * deterministic oracle) and the JIT (real OS threads), regardless of *which* worker ran each task. */
#include <pthread.h>

long __vm_atomic_add(void *p, long v);  /* fetch-add, returns old */
long __vm_atomic_load(void *p);
int  write(int fd, char *buf, long n);

#define NWORKERS 4
#define NTASKS   16
#define STEPS    16
#define CAP      (NTASKS + 1) /* a queue holds at most every task */

/* A stackless task: its entire resumable state is the struct (here, the step counter `i`). */
typedef struct {
  int i;
} task_t;

/* A mutex-protected LIFO deque of task pointers. (A real runtime uses a lock-free Chase-Lev deque;
 * a mutex — itself the futex — keeps the demo about *composition*, not lock-free wizardry.) */
typedef struct {
  pthread_mutex_t lock;
  task_t *items[CAP];
  int len;
} queue_t;

static task_t  g_tasks[NTASKS];
static queue_t g_injector;          /* all tasks start here (the global "inject" queue) */
static queue_t g_local[NWORKERS];   /* per-worker deques */
static long    g_total = 0;         /* work units done (atomic) */
static long    g_remaining = 0;     /* tasks not yet complete (atomic) */

static void q_push(queue_t *q, task_t *t) {
  pthread_mutex_lock(&q->lock);
  if (q->len < CAP) q->items[q->len++] = t;
  pthread_mutex_unlock(&q->lock);
}
static task_t *q_pop(queue_t *q) {
  pthread_mutex_lock(&q->lock);
  task_t *t = q->len > 0 ? q->items[--q->len] : 0;
  pthread_mutex_unlock(&q->lock);
  return t;
}

/* Advance the task one unit and report completion — the stackless "resume": one call, state in the
 * struct, returns whether it's done. No suspend across frames (that's the stackless trade-off). */
static int task_step(task_t *t) {
  if (t->i >= STEPS) return 1;
  __vm_atomic_add(&g_total, 1);
  t->i += 1;
  return t->i >= STEPS;
}

static void *worker(void *arg) {
  long me = (long)arg;
  while (__vm_atomic_load(&g_remaining) > 0) {
    task_t *t = q_pop(&g_local[me]);            /* 1. my own deque */
    for (int v = 0; v < NWORKERS && !t; v++)    /* 2. steal from a sibling */
      if (v != me) t = q_pop(&g_local[v]);
    if (!t) t = q_pop(&g_injector);             /* 3. pull from the global injector */
    if (!t) continue;                           /* nothing runnable now (a task is in-flight) */
    if (task_step(t)) __vm_atomic_add(&g_remaining, -1); /* completed */
    else q_push(&g_local[me], t);               /* not done — keep it on *my* deque (it migrated here) */
  }
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
  for (int i = 0; i < NTASKS; i++) { g_tasks[i].i = 0; q_push(&g_injector, &g_tasks[i]); }
  g_remaining = NTASKS;

  pthread_t workers[NWORKERS];
  for (int i = 0; i < NWORKERS; i++) pthread_create(&workers[i], 0, worker, (void *)(long)i);
  for (int i = 0; i < NWORKERS; i++) pthread_join(workers[i], 0);

  print_long(g_total); /* NTASKS * STEPS = 16 * 16 = 256 */
  return 0;
}
