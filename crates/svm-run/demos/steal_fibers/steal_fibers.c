/* Demo 3 (SCHEDULING.md): a guest-built **work-stealing** M:N scheduler over **stackful,
 * migratable fibers** — D57 complete, entirely as guest policy.
 *
 * The capstone of the migratable-fiber track. Contrast the two earlier flavors:
 *   - demos/mn_sched      — stackful fibers, but *sharded*: tasks pinned to their worker.
 *   - demos/work_stealing — work-stealing, but *stackless*: tasks are state-machine structs,
 *                           movable only because they are plain data (function-coloring: they can
 *                           suspend only at points in their own transformed body).
 * This demo is the strictly-stronger combination both were stepping stones toward: tasks are
 * **fibers** (whole native call stacks), and an idle worker **steals a suspended fiber and resumes
 * it on its own OS thread** — Go-class scheduling of arbitrary, unmodified code. The proof of
 * "unmodified" is in `step_in_callee`: the task yields from *inside a nested call frame*, which no
 * state-machine rewrite can express — the task's entire stack must (and does) migrate with it.
 *
 * The VM's role is exactly two things (D57, SCHEDULING.md "the VM owes a namespace + arbiter"):
 * the domain-wide fiber-handle namespace, and the single-owner claim under `cont.resume` (exactly
 * one racing resumer wins; a loser faults). *This program* owns every policy decision — the
 * injector queue, the per-worker deques, the steal order — in ordinary guest C over pthreads +
 * atomics. The mutex'd queues keep each handle exclusively owned between pop and push, so a resume
 * never loses a claim race; the VM's arbiter is the backstop, not the scheduler.
 *
 * Both printed totals are interleaving-invariant, so the interpreter (the deterministic M:N
 * oracle, whose registry migrates fibers as pure data) and the JIT (real OS threads, real
 * cross-thread stack switches) must agree byte-for-byte:
 *   g_total   = NTASKS * STEPS                       = 256
 *   g_returns = sum(id*1000 + 0+1+...+(STEPS-1))     = 120000 + 16*120 = 121920
 * g_returns is the stackful smoking gun: each task's return value depends on locals (`acc`, `id`)
 * held live across every yield — and every migration. A torn or restarted stack changes it. */
#include <pthread.h>
#include <stdlib.h>

/* VM fiber primitive (cont.* — a stackful coroutine; see SCHEDULING.md). */
int  __vm_fiber_new(long (*f)(long), void *stack);
long __vm_fiber_resume(int k, long arg, int *done);
long __vm_fiber_suspend(long value);
/* VM atomic primitives (lock-free over the shared window). */
long __vm_atomic_add(void *p, long v); /* fetch-add, returns old */
long __vm_atomic_load(void *p);
int  write(int fd, char *buf, long n);

#define NWORKERS 4
#define NTASKS   16
#define STEPS    16
#define STACK    16384
#define CAP      (NTASKS + 1) /* a queue holds at most every task */

/* A mutex-protected LIFO deque of task indices (the fiber handles live in g_handles). A real
 * runtime would use a lock-free Chase-Lev deque; the mutex — itself the VM futex — keeps the demo
 * about *composition*. The exclusivity it provides (one holder between pop and push) is also the
 * guest-side discipline that keeps resumes race-free: a worker only resumes handles it popped. */
typedef struct {
  pthread_mutex_t lock;
  int items[CAP];
  int len;
} queue_t;

static int     g_handles[NTASKS]; /* fiber handle of each task (created on main, read-only after) */
static queue_t g_injector;        /* all tasks start here (the global "inject" queue) */
static queue_t g_local[NWORKERS]; /* per-worker deques */
static long    g_total = 0;       /* work units done (atomic) */
static long    g_returns = 0;     /* sum of completed fibers' return values (atomic) */
static long    g_remaining = 0;   /* tasks not yet complete (atomic) */
static char   *g_stacks[NTASKS];  /* one guest data stack per fiber, malloc'd on main */

static void q_push(queue_t *q, int t) {
  pthread_mutex_lock(&q->lock);
  if (q->len < CAP) q->items[q->len++] = t;
  pthread_mutex_unlock(&q->lock);
}
static int q_pop(queue_t *q) {
  pthread_mutex_lock(&q->lock);
  int t = q->len > 0 ? q->items[--q->len] : -1;
  pthread_mutex_unlock(&q->lock);
  return t;
}

/* One unit of work, performed — and **suspended** — inside a nested, unmodified call frame. This
 * is what stackless tasks fundamentally cannot do (function coloring), and why stealing this task
 * means migrating its whole native stack: when a different worker resumes the fiber, execution
 * continues *here*, mid-callee, on the new OS thread. */
static long step_in_callee(long i) {
  __vm_atomic_add(&g_total, 1);
  __vm_fiber_suspend(0); /* park; whichever worker pops this task next continues right here */
  return i;
}

/* A green task: live local state (`acc`, `id`, `i`) held across every yield and migration. */
static long task(long id) {
  long acc = id * 1000;
  for (long i = 0; i < STEPS; i++) acc += step_in_callee(i);
  return acc; /* id*1000 + (0+1+...+STEPS-1) — wrong if the stack was ever torn or restarted */
}

static void *worker(void *arg) {
  long me = (long)arg;
  while (__vm_atomic_load(&g_remaining) > 0) {
    int t = q_pop(&g_local[me]);             /* 1. my own deque (locality) */
    for (int v = 0; v < NWORKERS && t < 0; v++) /* 2. steal a *suspended fiber* from a sibling */
      if (v != me) t = q_pop(&g_local[v]);
    if (t < 0) t = q_pop(&g_injector);       /* 3. pull from the global injector */
    if (t < 0) continue;                     /* nothing runnable now (tasks are in flight) */
    /* We exclusively own task t between pop and push, so this resume cannot lose a claim race.
     * If t last suspended on another worker, this resume migrates its stack to this thread. */
    int d = 0;
    long r = __vm_fiber_resume(g_handles[t], t, &d);
    if (d) {
      __vm_atomic_add(&g_returns, r);
      __vm_atomic_add(&g_remaining, -1);
    } else {
      q_push(&g_local[me], t); /* parked again — stays here unless someone steals it */
    }
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
  for (int i = 0; i < NTASKS; i++) g_stacks[i] = malloc(STACK);
  for (int i = 0; i < NTASKS; i++) g_handles[i] = __vm_fiber_new(task, g_stacks[i]);
  g_remaining = NTASKS;
  for (int i = 0; i < NTASKS; i++) q_push(&g_injector, i);

  pthread_t workers[NWORKERS];
  for (int i = 0; i < NWORKERS; i++) pthread_create(&workers[i], 0, worker, (void *)(long)i);
  for (int i = 0; i < NWORKERS; i++) pthread_join(workers[i], 0);

  print_long(g_total);   /* 16 * 16            = 256    */
  print_long(g_returns); /* 120000 + 16*120    = 121920 */
  return 0;
}
