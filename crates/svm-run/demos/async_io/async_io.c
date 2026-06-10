/* An async **event-loop runtime** over the §9/§12 submit/complete ring (DESIGN §12, increment 3c).
 *
 * One vCPU drives many concurrent I/Os: it `submit_async`s a whole batch of blocking ops onto the
 * host **offload pool** (which runs them on K threads), then parks on an in-window completion
 * **counter** and reaps completions as they land — resuming after each wake to process the next. The
 * vCPU is *never* blocked per-op (the "0 blocked vCPU threads" win): N blocking I/Os in flight cost one
 * parked vCPU + K host pool threads, not N. An I/O completion *is* a futex notify — a pool worker bumps
 * the counter and wakes the parked vCPU.
 *
 * Everything here is guest C over the VM's primitives: the ring builtins (`__vm_io_*`), the futex
 * (`__vm_wait32`/`__vm_atomic_load32`), and the Blocking capability. The printed total — the sum of
 * the host's deterministic per-op results — is **completion-order-invariant** (every result is added
 * exactly once), so it is identical on the interpreter (the M:N oracle) and the JIT (real OS threads),
 * regardless of which pool thread finished each op first. */

long          __vm_io_submit_async(void *sq, long n, void *counter); /* cap.call 9 1 -> submitted */
long          __vm_io_reap(void *cq, long max);                      /* cap.call 9 2 -> reaped */
int           __vm_blocking_handle(void);                            /* the Blocking cap handle */
int           __vm_atomic_load32(void *p);                           /* i32 atomic load */
int           __vm_wait32(void *p, int expected, long timeout_ns);   /* futex wait on an i32 */
int           write(int fd, char *buf, long n);

#define NTASKS  8
#define TIMEOUT 10000000000L /* 10 s — a working notify resumes in ms; this only bounds a regression */

/* SQE (64 B) / CQE (32 B) — the ring's in-window wire format (must match the host's layout). */
typedef struct {
  unsigned type_id;   /* 10 = Blocking */
  unsigned op;        /* 0  = work     */
  int      handle;    /* the Blocking cap handle */
  unsigned n_args;    /* 1 */
  long     args[4];   /* args[0] = the work input */
  long     user_data; /* echoed back in the CQE */
  long     pad;
} sqe_t;

typedef struct {
  long user_data;
  long result; /* the host's deterministic per-op result */
  long status; /* 0 = ok */
  long pad;
} cqe_t;

static sqe_t        g_sq[NTASKS];
static cqe_t        g_cq[NTASKS];
static volatile int g_counter; /* completions posted by the pool (the futex address) */

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
  int bh = __vm_blocking_handle();

  /* Build the batch: NTASKS blocking ops, work input = task index. */
  for (int i = 0; i < NTASKS; i++) {
    g_sq[i].type_id = 10;
    g_sq[i].op = 0;
    g_sq[i].handle = bh;
    g_sq[i].n_args = 1;
    g_sq[i].args[0] = i;
    g_sq[i].user_data = i;
  }

  /* Submit the whole batch onto the offload pool and return immediately — N I/Os now in flight. */
  __vm_io_submit_async(g_sq, NTASKS, (void *)&g_counter);

  /* Event loop: reap whatever has completed, and park on the counter when nothing is ready. The vCPU
   * resumes on each completion (a pool worker's notify), never blocking on any single I/O. */
  unsigned long total = 0;
  int done = 0;
  while (done < NTASKS) {
    int avail = __vm_atomic_load32((void *)&g_counter);
    if (done < avail) {
      long got = __vm_io_reap(g_cq, avail - done);
      for (long k = 0; k < got; k++) { total += (unsigned long)g_cq[k].result; done++; }
    } else {
      /* done == avail < NTASKS: nothing new to reap — park until the counter advances. */
      __vm_wait32((void *)&g_counter, done, TIMEOUT);
    }
  }

  print_ulong(total); /* Σ of the host's NTASKS deterministic results, order-invariant */
  return 0;
}
