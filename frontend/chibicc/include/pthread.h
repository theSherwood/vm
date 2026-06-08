// Minimal <pthread.h> for the SVM sandbox target — a **C-compatible 1:1 threading layer**.
//
// DESIGN §12/D56: the VM ships threading *primitives* (`thread.spawn`/`thread.join`, the
// `wait`/`notify` futex, C11 atomics) and **no scheduler**; the threading model is libc/runtime
// policy. This header is that policy for C: each `pthread_t` is **one vCPU = one real OS thread**
// (`thread.spawn`, 1:1 — real parallelism, native preemptive semantics, scheduled by the host OS).
// Mutexes and condition variables are built on the 32-bit atomics + the futex. No green-thread
// multiplexing here — a guest that wants M:N builds it over the fiber builtins instead.
//
// This shadows the system <pthread.h> only for the sandbox frontend (chibicc searches its bundled
// include dir first); a native `cc` build of the same source uses the platform pthreads + libpthread.
//
// Scope (MVP): create/join, mutex (init/destroy/lock/trylock/unlock), cond (init/destroy/wait/
// signal/broadcast). Out of scope for now: pthread_self/exit/once/cancel, attributes, rwlocks,
// barriers, TLS keys — add as programs demand them.
#ifndef __SVM_PTHREAD_H
#define __SVM_PTHREAD_H

#include <stddef.h> // size_t, NULL
#include <stdlib.h> // malloc — per-thread data stack + start record

// VM primitives (lowered by name in the frontend, codegen_ir.c).
int __vm_thread_spawn(long (*fn)(long), void *stack, long arg);
long __vm_thread_join(int handle);
int __vm_atomic_cas32(void *p, int expected, int desired);
int __vm_atomic_load32(void *p);
void __vm_atomic_store32(void *p, int v);
int __vm_atomic_add32(void *p, int v);
int __vm_wait32(void *p, int expected, long timeout_ns);
int __vm_notify(void *p, int count);

// ---- threads (1:1) -------------------------------------------------------------------------
typedef int pthread_t;
typedef struct {
  int __unused;
} pthread_attr_t;

// Per-thread data stack (§3d two-stack split): a thread owns its own in-window data stack, grown
// upward from this base. 256 KiB is ample for the MVP; overflow is guest self-corruption (a heap
// clobber, not an escape — §1), pending guard-paged thread stacks.
#define __PTHREAD_STACK 262144L

struct __pthread_rec {
  void *(*fn)(void *);
  void *arg;
};

// The fixed entry `thread.spawn` launches. `thread.spawn` resolves a *static* function index, so it
// needs a direct function name — but the real `start_routine` is a runtime pointer, so this
// trampoline reaches it via an ordinary indirect call (a funcref dispatch, §3c). The thread's i64
// result is `start_routine`'s return value, delivered back through `thread.join`.
static long __pthread_entry(long rec) {
  struct __pthread_rec *r = (struct __pthread_rec *)rec;
  return (long)r->fn(r->arg);
}

static int pthread_create(pthread_t *t, const pthread_attr_t *attr,
                          void *(*start_routine)(void *), void *arg) {
  (void)attr;
  struct __pthread_rec *r = (struct __pthread_rec *)malloc(sizeof(struct __pthread_rec));
  if (!r)
    return 11; // EAGAIN
  r->fn = start_routine;
  r->arg = arg;
  void *stack = malloc(__PTHREAD_STACK);
  if (!stack)
    return 11;
  int h = __vm_thread_spawn(__pthread_entry, stack, (long)r);
  if (h < 0)
    return 11;
  *t = h;
  return 0;
}

static int pthread_join(pthread_t t, void **retval) {
  long r = __vm_thread_join(t);
  if (retval)
    *retval = (void *)r;
  return 0;
}

// ---- mutexes -------------------------------------------------------------------------------
// `__state`: 0 = unlocked, 1 = locked. Futex-backed — a contended locker parks on the state word
// and is woken by `unlock`'s notify; `__vm_wait32` re-checks the word atomically, so the classic
// unlock-between-cas-and-wait race cannot lose a wakeup.
typedef struct {
  int __state;
} pthread_mutex_t;
#define PTHREAD_MUTEX_INITIALIZER {0}

static int pthread_mutex_init(pthread_mutex_t *m, const void *attr) {
  (void)attr;
  m->__state = 0;
  return 0;
}
static int pthread_mutex_destroy(pthread_mutex_t *m) {
  (void)m;
  return 0;
}
static int pthread_mutex_lock(pthread_mutex_t *m) {
  while (__vm_atomic_cas32(&m->__state, 0, 1) != 0)
    __vm_wait32(&m->__state, 1, -1L); // park while locked (forever)
  return 0;
}
static int pthread_mutex_trylock(pthread_mutex_t *m) {
  return __vm_atomic_cas32(&m->__state, 0, 1) == 0 ? 0 : 16; // EBUSY
}
static int pthread_mutex_unlock(pthread_mutex_t *m) {
  __vm_atomic_store32(&m->__state, 0);
  __vm_notify(&m->__state, 1);
  return 0;
}

// ---- condition variables -------------------------------------------------------------------
// `__seq` is a wakeup-sequence counter: `wait` snapshots it (under the held mutex), drops the mutex,
// and parks until it changes; `signal`/`broadcast` bump it (atomic) and notify. Spurious wakeups are
// allowed (callers must re-test their predicate in a loop, as POSIX requires).
typedef struct {
  int __seq;
} pthread_cond_t;
#define PTHREAD_COND_INITIALIZER {0}

static int pthread_cond_init(pthread_cond_t *c, const void *attr) {
  (void)attr;
  c->__seq = 0;
  return 0;
}
static int pthread_cond_destroy(pthread_cond_t *c) {
  (void)c;
  return 0;
}
static int pthread_cond_wait(pthread_cond_t *c, pthread_mutex_t *m) {
  int seq = __vm_atomic_load32(&c->__seq);
  pthread_mutex_unlock(m);
  __vm_wait32(&c->__seq, seq, -1L); // park until a signal/broadcast bumps __seq
  pthread_mutex_lock(m);
  return 0;
}
static int pthread_cond_signal(pthread_cond_t *c) {
  __vm_atomic_add32(&c->__seq, 1);
  __vm_notify(&c->__seq, 1);
  return 0;
}
static int pthread_cond_broadcast(pthread_cond_t *c) {
  __vm_atomic_add32(&c->__seq, 1);
  __vm_notify(&c->__seq, 0x7fffffff);
  return 0;
}

#endif // __SVM_PTHREAD_H
