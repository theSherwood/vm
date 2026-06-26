/* Detect-and-kill trap recovery for the JIT (DESIGN.md §4/§5), unix only.
 *
 * The guest window is bracketed by a PROT_NONE guard page (allocated in Rust); the masking
 * lowering confines every access to [0, size), so a fault in the guard region is a
 * width-overrun at the very top of the window or — defense-in-depth — a masking/elision
 * bug. We catch SIGSEGV/SIGBUS, and if the faulting address is inside the *currently armed*
 * window's guarded range we siglongjmp back out of the JIT call, reporting a memory fault to
 * the host instead of crashing it. Anything else chains to the previous disposition.
 *
 * setjmp/sigsetjmp are macros that need the compiler's `returns_twice` handling, so this
 * lives in C (calling them via raw FFI from Rust is unsound). The recovery state is
 * thread-local, so concurrent JIT runs on different threads are independent: the handler
 * runs on the faulting thread and reads that thread's state.
 */
/* Feature-test macros must precede every include. `_XOPEN_SOURCE` exposes the (macOS-deprecated)
 * ucontext routines + `ucontext_t` mcontext on Apple SDKs; `_GNU_SOURCE` gates glibc's REG_RIP/REG_RBP
 * in <sys/ucontext.h>. Both are harmless where unneeded. */
#define _XOPEN_SOURCE 700
#define _GNU_SOURCE
#include <setjmp.h>
#include <signal.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <ucontext.h>

static _Thread_local sigjmp_buf g_buf;
static _Thread_local volatile int g_armed = 0;
static _Thread_local volatile uintptr_t g_lo = 0;
static _Thread_local volatile uintptr_t g_hi = 0;

/* §14 fault-driven yield (demand paging): while a demand-paged coroutine child runs, its window's
 * committed range is registered here. A fault inside it is *recoverable*: the callback suspends the
 * child's fiber to its parent (which supplies the page and resumes); when it returns nonzero the
 * handler simply returns, re-executing the faulting access against the now-mapped page. A zero
 * return (no live coroutine) falls through to the armed-window detect-and-kill below. */
static _Thread_local volatile uintptr_t g_demand_lo = 0;
static _Thread_local volatile uintptr_t g_demand_hi = 0;
static _Thread_local int (*g_demand_cb)(uintptr_t, void *) = 0;
static _Thread_local void *g_demand_ctx = 0;

void svm_set_demand(uintptr_t lo, uintptr_t hi, int (*cb)(uintptr_t, void *), void *ctx) {
    g_demand_lo = lo;
    g_demand_hi = hi;
    g_demand_ctx = ctx;
    g_demand_cb = cb;
}

void svm_clear_demand(void) {
    g_demand_cb = 0;
    g_demand_lo = 0;
    g_demand_hi = 0;
    g_demand_ctx = 0;
}

/* The trap-time backtrace capture *state* and the frame-pointer walk live in `trap_capture.c` (shared
 * with the windows VEH and the explicit-trap helper). The signal handler below extracts the faulting
 * `(pc, fp)` from the ucontext and hands them to `svm_store_trap_frame`, which walks the chain and
 * stashes it in a thread-local for the host to symbolize (DEBUGGING.md §5 W3). The walk must happen
 * here, while the guest stack is intact: a siglongjmp unwinds back onto the same stack the post-fault
 * host code then reuses, so the frames would be gone by the time the host could walk them. */
extern void svm_store_trap_frame(uintptr_t pc, uintptr_t fp);

/* Memory-fault capture: extract the faulting (pc, fp) from the signal ucontext and hand off the walk.
 * Async-signal-safe: only the ucontext read + the (stack-reading, TLS-writing) store. */
static void svm_capture_frame(void *uc) {
    uintptr_t pc = 0, fp = 0;
#if defined(__linux__) && defined(__x86_64__)
    ucontext_t *c = (ucontext_t *)uc;
    pc = (uintptr_t)c->uc_mcontext.gregs[REG_RIP];
    fp = (uintptr_t)c->uc_mcontext.gregs[REG_RBP];
#elif defined(__linux__) && defined(__aarch64__)
    ucontext_t *c = (ucontext_t *)uc;
    pc = (uintptr_t)c->uc_mcontext.pc;
    fp = (uintptr_t)c->uc_mcontext.regs[29]; /* AAPCS64 frame pointer */
#elif defined(__APPLE__) && defined(__x86_64__)
    ucontext_t *c = (ucontext_t *)uc;
    pc = (uintptr_t)c->uc_mcontext->__ss.__rip;
    fp = (uintptr_t)c->uc_mcontext->__ss.__rbp;
#elif defined(__APPLE__) && defined(__aarch64__)
    ucontext_t *c = (ucontext_t *)uc;
    pc = (uintptr_t)c->uc_mcontext->__ss.__pc;
    fp = (uintptr_t)c->uc_mcontext->__ss.__fp;
#else
    (void)uc; /* an arch we don't decode: pc/fp stay 0 → the host yields an empty backtrace */
#endif
    svm_store_trap_frame(pc, fp); /* walk + stash in the shared (trap_capture.c) thread-local */
}

static struct sigaction g_old_segv;
static struct sigaction g_old_bus;

static void svm_chain(struct sigaction *old, int sig, siginfo_t *info, void *uc) {
    if (old->sa_flags & SA_SIGINFO) {
        if (old->sa_sigaction)
            old->sa_sigaction(sig, info, uc);
    } else if (old->sa_handler != SIG_DFL && old->sa_handler != SIG_IGN) {
        old->sa_handler(sig);
    } else {
        /* No useful previous handler: restore the default and re-raise so the process dies
         * with the usual diagnostics (this is a genuine host fault, not a guest one). */
        signal(sig, SIG_DFL);
        raise(sig);
    }
}

static void svm_handler(int sig, siginfo_t *info, void *uc) {
    uintptr_t addr = (uintptr_t)info->si_addr;
    /* Recoverable demand fault first (the demand range lies inside the armed child window, so this
     * check must precede detect-and-kill). The callback suspends the child's fiber to its parent
     * *from this handler frame* (on the child's fiber stack — it stays live across the suspension);
     * when the parent resumes the child, the callback returns here and the plain return re-executes
     * the faulting access against the freshly supplied page. */
    if (g_demand_cb && addr >= g_demand_lo && addr < g_demand_hi) {
        if (g_demand_cb(addr, g_demand_ctx))
            return;
    }
    if (g_armed && addr >= g_lo && addr < g_hi) {
        g_armed = 0;
        svm_capture_frame(uc); /* stash the faulting frame before the stack unwinds (§5 W3) */
        siglongjmp(g_buf, 1);  /* back to svm_run_guarded; does not return */
    }
    svm_chain(sig == SIGBUS ? &g_old_bus : &g_old_segv, sig, info, uc);
}

/* Install the handler once (idempotent enough for a std::sync::Once caller). */
void svm_install_trap_handler(void) {
    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_sigaction = svm_handler;
    /* SA_NODEFER: don't block SIGSEGV/SIGBUS while the handler runs. A demand fault suspends to the
     * parent *from inside this handler*, and the parent must keep its own fault recovery (its guard,
     * or a further demand fault) while the child sits suspended in its handler frame — a blocked
     * synchronous signal would kill the process instead. The handler never faults on its own
     * (it touches only thread-locals), so unblocked re-entry is not a recursion hazard. */
    sa.sa_flags = SA_SIGINFO | SA_ONSTACK | SA_NODEFER;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGSEGV, &sa, &g_old_segv);
    sigaction(SIGBUS, &sa, &g_old_bus);
}

/* Run `fn(a, r, m, t, tc)` (a JIT entry trampoline) with faults in [lo, hi) caught.
 * Returns 0 if it ran to completion, 1 if a guarded fault was caught and unwound.
 *
 * **Re-entrant** (§14 nesting): a guest may run a *child* guest (in its own window) from inside its
 * own guarded call. The recovery state (buf/lo/hi/armed) is saved on this C stack frame and restored
 * on exit, so the child's guard nests cleanly and the parent's recovery point is intact afterwards.
 * A child fault unwinds to *this* (the child's) frame; the parent's frame is untouched. */
int svm_run_guarded(void (*fn)(const long *, long *, unsigned char *, const void *, long *),
                    const long *a, long *r, unsigned char *m, const void *t, long *tc,
                    uintptr_t lo, uintptr_t hi) {
    /* Save the caller's (possibly-armed parent) recovery state to restore on the way out. */
    sigjmp_buf saved_buf;
    memcpy(&saved_buf, &g_buf, sizeof saved_buf);
    int saved_armed = g_armed;
    uintptr_t saved_lo = g_lo;
    uintptr_t saved_hi = g_hi;

    g_lo = lo;
    g_hi = hi;
    if (sigsetjmp(g_buf, 1)) {
        /* A guarded fault unwound back here — restore the parent's recovery state and report it. */
        memcpy(&g_buf, &saved_buf, sizeof g_buf);
        g_armed = saved_armed;
        g_lo = saved_lo;
        g_hi = saved_hi;
        return 1;
    }
    g_armed = 1;
    fn(a, r, m, t, tc);
    /* Ran to completion — restore the parent's recovery state (re-arming the parent's range). */
    memcpy(&g_buf, &saved_buf, sizeof g_buf);
    g_armed = saved_armed;
    g_lo = saved_lo;
    g_hi = saved_hi;
    return 0;
}

/* ---- Guard-state snapshots (§14 co-fibers) -------------------------------------------------
 *
 * A coroutine child runs its own guarded call on a *separate fiber stack*; a suspend switches
 * stacks from inside that call, leaving the thread-local recovery state armed for the child while
 * the parent runs. The parent therefore swaps the whole recovery state (jmp_buf + armed + range)
 * around every switch: save the child's state at suspend-return, restore the parent's; reinstall
 * the child's at the next resume. The state is an opaque heap blob (C-side, for sigjmp_buf size
 * and alignment); a freshly boxed state is all-zero = disarmed. Same-thread only: a sigjmp_buf
 * must be longjmp'd on the thread that captured it.
 */
typedef struct {
    sigjmp_buf buf;
    int armed;
    uintptr_t lo, hi;
} svm_guard_state;

void *svm_guard_box(void) { return calloc(1, sizeof(svm_guard_state)); }

void svm_guard_unbox(void *p) { free(p); }

void svm_guard_save(void *p) {
    svm_guard_state *s = (svm_guard_state *)p;
    memcpy(&s->buf, &g_buf, sizeof s->buf);
    s->armed = g_armed;
    s->lo = g_lo;
    s->hi = g_hi;
}

void svm_guard_restore(const void *p) {
    const svm_guard_state *s = (const svm_guard_state *)p;
    memcpy(&g_buf, &s->buf, sizeof g_buf);
    g_armed = s->armed;
    g_lo = s->lo;
    g_hi = s->hi;
}
