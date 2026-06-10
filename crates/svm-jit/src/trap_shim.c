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
#include <setjmp.h>
#include <signal.h>
#include <stdint.h>
#include <string.h>

static _Thread_local sigjmp_buf g_buf;
static _Thread_local volatile int g_armed = 0;
static _Thread_local volatile uintptr_t g_lo = 0;
static _Thread_local volatile uintptr_t g_hi = 0;

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
    if (g_armed && addr >= g_lo && addr < g_hi) {
        g_armed = 0;
        siglongjmp(g_buf, 1); /* back to svm_run_guarded; does not return */
    }
    svm_chain(sig == SIGBUS ? &g_old_bus : &g_old_segv, sig, info, uc);
}

/* Install the handler once (idempotent enough for a std::sync::Once caller). */
void svm_install_trap_handler(void) {
    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_sigaction = svm_handler;
    sa.sa_flags = SA_SIGINFO | SA_ONSTACK;
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
