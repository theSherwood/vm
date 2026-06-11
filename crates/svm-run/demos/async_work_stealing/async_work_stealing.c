/* An **async work-stealing M:N runtime** (DESIGN §12, increment 3c capstone) — the union of the
 * stackless work-stealing scheduler (demos/work_stealing) and the async submit/complete ring (B).
 *
 * NWORKERS vCPUs cooperatively drain NTASKS I/O-bound tasks, each issuing a blocking op through the
 * ring. The point of the ring is that a worker **never blocks on an I/O**: it `submit_async`s a task's
 * op onto the host offload pool (which runs it on K threads) and moves on; when nothing is runnable it
 * **parks** on the in-window completion counter, woken by a pool worker's `notify` (an I/O completion
 * is a futex notify, §12). Work-stealing and I/O overlap: while N ops are in flight on the pool, the
 * NWORKERS vCPUs are reaping, not blocked — "OS threads bounded by K, not by I/O concurrency".
 *
 * The SQ/CQ are **global** fixed shared ring buffers, like real io_uring — one ring shared by all
 * workers, not per-worker scratch. Its submit/reap `cap.call`s are serialized by a guest mutex —
 * exactly as a shared io_uring requires single-producer submission; the blocking work still overlaps
 * on the pool, and parking is lock-free. (Grown-heap buffers borrow fine across cap.calls on both
 * backends — the JIT persists its cap-path page map per run — so the choice here is purely the
 * shared-ring design, not a workaround.)
 *
 * Everything is guest code over the VM's primitives: vCPUs (`thread.spawn`), C11 atomics, the futex
 * (`__vm_wait32`), and the ring builtins. The printed total — the wrapping sum of the host's
 * deterministic per-op results — is completion-order- and interleaving-invariant, so it is identical
 * on the interpreter (the M:N oracle) and the JIT (real OS threads), regardless of which worker
 * submitted or reaped each task. */
#include <pthread.h>

long __vm_io_submit_async(void *sq, long n, void *counter);
long __vm_io_reap(void *cq, long max);
int  __vm_blocking_handle(void);
long __vm_atomic_add(void *p, long v);  /* fetch-add (i64) on an 8-byte word, returns old */
int  __vm_atomic_add32(void *p, int v); /* fetch-add (i32) on a 4-byte word, returns old */
int  __vm_atomic_load32(void *p);
int  __vm_wait32(void *p, int expected, long timeout_ns);
int  write(int fd, char *buf, long n);

#define NWORKERS 4
#define NTASKS   16
#define TIMEOUT  10000000000L /* 10 s — a working notify resumes in ms; only bounds a regression */

typedef struct {
  unsigned type_id, op;
  int      handle;
  unsigned n_args;
  long     args[4];
  long     user_data, pad;
} sqe_t; /* 64 B */

typedef struct {
  long user_data, result, status, pad;
} cqe_t; /* 32 B */

static int  g_next;    /* next task index to submit (claimed by atomic fetch-add) */
static int  g_counter; /* completions posted by the pool — the futex address */
static int  g_reaped;  /* completions reaped + accumulated */
static long g_total;   /* wrapping sum of results (atomic) */

static sqe_t           g_sq;          /* shared single-slot submission queue (submit is serialized) */
static cqe_t           g_cq[NTASKS];  /* shared completion queue */
static pthread_mutex_t g_ring_lock;   /* serializes the shared ring's submit/reap cap.calls */

static void *worker(void *arg) {
  (void)arg;
  int bh = __vm_blocking_handle();
  while (1) {
    /* 1. Claim the next unsubmitted task and fire its I/O onto the pool, then move on — never block
     *    waiting for it. */
    int t = __vm_atomic_add32(&g_next, 1);
    if (t < NTASKS) {
      pthread_mutex_lock(&g_ring_lock);
      g_sq.type_id = 10; /* Blocking */
      g_sq.op = 0;       /* work */
      g_sq.handle = bh;
      g_sq.n_args = 1;
      g_sq.args[0] = t;
      g_sq.user_data = t;
      __vm_io_submit_async(&g_sq, 1, &g_counter);
      pthread_mutex_unlock(&g_ring_lock);
      continue;
    }

    /* 2. All tasks submitted: reap completions (serialized; concurrent reaps split the ready set),
     *    accumulate them, and park when nothing is ready yet. */
    int reaped = __vm_atomic_load32(&g_reaped);
    if (reaped >= NTASKS) break; /* every completion accounted for */
    int avail = __vm_atomic_load32(&g_counter);
    if (reaped < avail) {
      pthread_mutex_lock(&g_ring_lock);
      long got = __vm_io_reap(g_cq, NTASKS);
      long sum = 0;
      for (long k = 0; k < got; k++) sum += g_cq[k].result; /* read the shared CQ under the lock */
      pthread_mutex_unlock(&g_ring_lock);
      if (got > 0) {
        __vm_atomic_add(&g_total, sum);         /* i64: the wrapping result sum */
        __vm_atomic_add32(&g_reaped, (int)got); /* i32: completions accounted for */
      }
    } else {
      /* reaped == avail < NTASKS: ops still in flight, nothing to reap — park until the counter
       * advances. The futex value-check makes this race-free vs. a completion landing right now. */
      __vm_wait32(&g_counter, avail, TIMEOUT);
    }
  }
  return 0;
}

static void print_ulong(unsigned long v) {
  char buf[24];
  int  n = 0;
  if (v == 0) buf[n++] = '0';
  char tmp[24];
  int  t = 0;
  while (v > 0) { tmp[t++] = (char)('0' + (v % 10)); v /= 10; }
  while (t > 0) buf[n++] = tmp[--t];
  buf[n++] = '\n';
  write(1, buf, n);
}

int main(void) {
  pthread_t workers[NWORKERS];
  for (int i = 0; i < NWORKERS; i++) pthread_create(&workers[i], 0, worker, 0);
  for (int i = 0; i < NWORKERS; i++) pthread_join(workers[i], 0);
  print_ulong((unsigned long)g_total); /* Σ of NTASKS deterministic results, order-invariant */
  return 0;
}
