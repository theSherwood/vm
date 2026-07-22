// Threaded guest-driven JIT (DESIGN.md §22): multiple guest **worker threads each Cranelift-compile
// their own unit concurrently**, entirely inside the sandbox.
//
// `jit_demo.c` is the single-threaded capstone (a guest that JITs itself). This is its threaded
// sibling: `NWORKERS` pthreads each build serialized SVM IR for a *distinct* unit at runtime, submit
// it through the `Jit` capability (`__vm_jit_compile`) — so several `Jit.compile`s are in flight at
// once — then `__vm_jit_invoke2` the freshly-native code and check it against a plain C reference on
// a grid of inputs. Each worker releases its handle when done.
//
// The host runs this through the JIT backend with the **per-domain serialized cap-thunk** (a
// `Mutex<Host>` engaged automatically because the guest `thread.spawn`s, DESIGN.md §22): a worker
// calling `Jit.compile` (`finalize_definitions`) while siblings run is sound because the lock
// serializes the *compiles* while execution stays fully parallel — cranelift-jit appends new
// functions to fresh arena pages and never modifies running code, so a finalize never disturbs a
// sibling's executing unit. The guest-facing iface 11 is unchanged; the serialization is an internal
// host detail.
//
// Each worker keeps its IR in its **own** stack buffer (distinct window addresses across the pthread
// stacks) and threads the emit cursor explicitly, so there is no shared-mutable-state race in the
// guest emitter either — the only concurrency the VM mediates is the cap.call into the host.
//
// The program prints the number of input mismatches across every worker — **0** when each worker's
// concurrently-JITed unit agrees with the C reference. Run it sandboxed:
//
//   cargo run -p svm-run -- crates/svm-run/demos/jit/jit_threads.c

#include <pthread.h>
#include <svm.h>

int write(int fd, char *buf, long n);

#define NWORKERS 4

static void puts1(char *s) {
  long n = 0;
  while (s[n])
    n++;
  write(1, s, n);
}

static void put_i64(long v) {
  char tmp[24];
  int i = 0;
  if (v < 0) {
    write(1, "-", 1);
    v = -v;
  }
  if (v == 0) {
    write(1, "0", 1);
    return;
  }
  while (v) {
    tmp[i++] = '0' + (v % 10);
    v /= 10;
  }
  while (i)
    write(1, &tmp[--i], 1);
}

// --- a reentrant serialized-IR emitter (no global cursor — each worker owns its `Emit`) -------
// The binary layout mirrors `crates/svm-encode` (LEB128 + one-byte opcodes), exactly as
// `jit_demo.c` builds it, but the output cursor is threaded through `Emit` so concurrent workers
// never clobber a shared `n_out`.
typedef struct {
  char *buf;
  int n;
} Emit;

static void eb(Emit *e, int v) { e->buf[e->n++] = (char)v; }

static void uleb(Emit *e, unsigned long v) {
  for (;;) {
    int b7 = v & 0x7f;
    v >>= 7;
    if (v) {
      eb(e, b7 | 0x80);
    } else {
      eb(e, b7);
      return;
    }
  }
}

static void sleb(Emit *e, long v) {
  for (;;) {
    int b7 = v & 0x7f;
    v >>= 7; // arithmetic shift: sign-extends
    int done = (v == 0 && !(b7 & 0x40)) || (v == -1 && (b7 & 0x40));
    eb(e, done ? b7 : (b7 | 0x80));
    if (done)
      return;
  }
}

// Emit a one-function unit `(i64 a, i64 b) -> (i64)` (the raw shape `__vm_jit_invoke2` calls)
// computing `a * k + b + w` as straight-line IR. `k`/`w` are worker-specific, so each worker JITs a
// genuinely distinct unit. Opcodes mirror svm-encode: 0x11 = i64.const (+ sleb), 0x40 = i64.add,
// 0x42 = i64.mul (+ uleb operand indices); 0x83 = return.
static long emit_unit(char *buf, long k, long w) {
  Emit e = {buf, 0};
  // Header: magic + version.
  eb(&e, 'S');
  eb(&e, 'V');
  eb(&e, 'M');
  eb(&e, 0);
  eb(&e, 8); // format v8 (single-string import names; call.sym link form)
  // Memory descriptor: present, size_log2 16 — must match this module's window (the validator's
  // memory-match precondition), which chibicc keeps at the 64 KiB default for a small program.
  eb(&e, 1);
  eb(&e, 16);
  eb(&e, 0); // no data segments
  eb(&e, 0); // no imports — self-contained unit (v2 import section)
  eb(&e, 0); // no exports — invoked by handle, not by name (v3 export section)
  eb(&e, 0); // no interfaces (v6 interface section)
  eb(&e, 0); // no impl exports (v5 impl-export section)
  eb(&e, 1); // one function
  // params (a, b) : i64, i64 ; results : i64 (type tag 1 = i64).
  eb(&e, 2);
  eb(&e, 1);
  eb(&e, 1);
  eb(&e, 1);
  eb(&e, 1);
  // One block whose params mirror the function's.
  eb(&e, 1);
  eb(&e, 2);
  eb(&e, 1);
  eb(&e, 1);
  // 5 instructions: const k, mul, add b, const w, add w.
  uleb(&e, 5);
  // v2 = i64.const k
  eb(&e, 0x11);
  sleb(&e, k);
  // v3 = i64.mul v0 v2   (a * k)
  eb(&e, 0x42);
  uleb(&e, 0);
  uleb(&e, 2);
  // v4 = i64.add v3 v1   (+ b)
  eb(&e, 0x40);
  uleb(&e, 3);
  uleb(&e, 1);
  // v5 = i64.const w
  eb(&e, 0x11);
  sleb(&e, w);
  // v6 = i64.add v4 v5   (+ w)
  eb(&e, 0x40);
  uleb(&e, 4);
  uleb(&e, 5);
  // return v6
  eb(&e, 0x83);
  uleb(&e, 1);
  uleb(&e, 6);
  return e.n;
}

// The C reference the JITed unit must match.
static long ref_unit(long a, long b, long k, long w) { return a * k + b + w; }

static long g_bad[NWORKERS]; // per-worker mismatch count (disjoint index, read by main after join)

// One worker: emit a distinct unit, compile it (concurrently with its siblings), invoke the native
// code across a grid, and tally any disagreement with the C reference.
static void *worker(void *arg) {
  long me = (long)arg;
  long k = 2 + me;     // a per-worker multiplier
  long w = me * 1000;  // and a per-worker bias — so every worker JITs a different function
  char buf[256];       // this worker's own blob, at its own pthread-stack address
  long n = emit_unit(buf, k, w);
  long code = __vm_jit_compile(buf, n);
  if (code < 0) {
    g_bad[me] = 1000 + (-code); // surface a compile failure as a large nonzero
    return 0;
  }
  long bad = 0;
  for (long a = -4; a <= 4; a++)
    for (long b = -4; b <= 4; b++)
      if (__vm_jit_invoke2(code, a, b) != ref_unit(a, b, k, w))
        bad++;
  __vm_jit_release(code);
  g_bad[me] = bad;
  return 0;
}

int main(void) {
  pthread_t workers[NWORKERS];
  for (long i = 0; i < NWORKERS; i++)
    pthread_create(&workers[i], 0, worker, (void *)i);
  for (int i = 0; i < NWORKERS; i++)
    pthread_join(workers[i], 0);

  long bad = 0;
  for (int i = 0; i < NWORKERS; i++)
    bad += g_bad[i];

  put_i64(bad); // 0 = every worker's concurrently-JITed unit agreed with the reference
  write(1, "\n", 1);
  return 0;
}
