/* Cross-platform trap-time backtrace capture (DEBUGGING.md §5 W3), compiled on every target that has
 * the JIT trap-backtrace feature (unix + windows). The platform-specific *trap detection* lives
 * elsewhere — the unix SIGSEGV/SIGBUS handler in `trap_shim.c`, the windows Vectored Exception Handler
 * in `mem.rs` — but the capture *state* and the frame-pointer walk are shared here so a div-by-zero /
 * `unreachable` / `OutOfFuel` / indirect-call-type trap is captured identically on both platforms.
 *
 * The host reads the captured `(pc, return addresses)` via `svm_take_trap_frame` and only *symbolizes*
 * them (pure arithmetic, no stack reads). Thread-local: the trap is attributed to the faulting worker,
 * and the host takes the capture before re-running anything on that thread.
 *
 * MSVC has no `__builtin_frame_address`/`__builtin_return_address`, so: the trapping frame pointer is
 * passed *in* (the JIT computes it with Cranelift `get_frame_pointer` and threads it as an argument),
 * and the trap-site return address comes from `_ReturnAddress()` (MSVC) / `__builtin_return_address(0)`
 * (GCC/Clang) — neither needs the helper to have its own frame pointer. */
#include <stdint.h>

#ifdef _MSC_VER
#include <intrin.h>
#pragma intrinsic(_ReturnAddress)
#define SVM_TLS __declspec(thread)
#define SVM_RETURN_ADDRESS() _ReturnAddress()
#else
#define SVM_TLS _Thread_local
#define SVM_RETURN_ADDRESS() __builtin_return_address(0)
#endif

#define SVM_TRAP_MAXFRAMES 64
static SVM_TLS volatile int g_trap_valid = 0;
static SVM_TLS uintptr_t g_trap_pc = 0;
static SVM_TLS uintptr_t g_trap_rets[SVM_TRAP_MAXFRAMES];
static SVM_TLS int g_trap_nrets = 0;

/* Per-fiber trap attribution (DEBUGGING.md §5 W3 / §23-D57). `g_current_fiber` is the guest handle of
 * the fiber executing on *this* OS thread right now, or `SVM_NO_FIBER` (root, no fiber). The fiber
 * runtime publishes it with stack discipline across the resume seam (`svm_set_current_fiber`), so under
 * work-stealing migration — where a fiber may resume on a different vCPU thread than it suspended on —
 * a trap is attributed to the fiber *running at the trap instant*, not inferred from the thread. The
 * capture functions copy it into `g_trap_fiber` (signal-safe: a plain TLS read), and the host reads it
 * back via `svm_take_trap_fiber`. */
#define SVM_NO_FIBER (-1)
static SVM_TLS int64_t g_current_fiber = SVM_NO_FIBER;
static SVM_TLS int64_t g_trap_fiber = SVM_NO_FIBER;

/* Publish the fiber now running on this thread; returns the previous value so the caller can restore it
 * when the resume returns (the same save/restore the durable shadow-SP swap uses). */
int64_t svm_set_current_fiber(int64_t handle) {
    int64_t prev = g_current_fiber;
    g_current_fiber = handle;
    return prev;
}


/* Walk the frame-pointer chain from `fp` toward the stack base, appending each frame's return address
 * to `rets[]` from index `n`; returns the new count. The JIT's `preserve_frame_pointers` gives every
 * guest frame a `{ saved_fp, ret_addr }` record: `*fp` is the caller's saved frame pointer, `*(fp+1)`
 * the return address. The walk moves *up* (increasing addresses, away from the low-address stack guard
 * a fault sits near) and is bounded — aligned, non-null, strictly-increasing links, within a generous
 * span, and the frame cap — so a corrupt chain terminates instead of looping or reading wild memory.
 * The host stops at the first return address that isn't guest code, so a few trailing host frames are
 * harmless. Reads-only. */
static int svm_walk_fp_chain(uintptr_t fp, uintptr_t *rets, int n) {
    uintptr_t cur = fp;
    const uintptr_t start = fp;
    const uintptr_t span = 8u * 1024 * 1024; /* don't chase a corrupt chain off the stack */
    while (n < SVM_TRAP_MAXFRAMES && cur != 0 && (cur & (sizeof(uintptr_t) - 1)) == 0 &&
           cur >= start && cur - start < span) {
        uintptr_t next = *(uintptr_t *)cur;
        uintptr_t ret = *(uintptr_t *)(cur + sizeof(uintptr_t));
        rets[n++] = ret;
        if (next <= cur) /* frame pointers grow toward the base; a non-increasing link is the end */
            break;
        cur = next;
    }
    return n;
}

/* Store a **memory-fault** capture: the trap detector (unix signal handler / windows VEH) extracts the
 * faulting `(pc, fp)` from its platform context and calls this to walk + stash. `pc` is the exact
 * faulting instruction (the host symbolizes it directly). Async-signal-safe: stack reads + TLS writes. */
void svm_store_trap_frame(uintptr_t pc, uintptr_t fp) {
    g_trap_pc = pc;
    g_trap_nrets = svm_walk_fp_chain(fp, g_trap_rets, 0);
    g_trap_fiber = g_current_fiber;
    g_trap_valid = 1;
}

/* Capture an **explicit-check** trap (§5 W3 Stage 2). The JIT calls this from a trap site (div-by-zero,
 * `unreachable`, `OutOfFuel`, indirect-call-type) *before* storing the trap kind and returning — those
 * returns unwind every guest frame, so the chain must be walked here, while it is live. `guest_fp` is
 * the trapping function's frame pointer (Cranelift `get_frame_pointer`, threaded in by the JIT); the
 * trap site is this helper's own return address (just past the `call`), recorded as `rets[0]`
 * (symbolized at `ret - 1`, like every caller) with `pc` left 0. */
void svm_capture_explicit_trap(uintptr_t guest_fp) {
    uintptr_t trap_site = (uintptr_t)SVM_RETURN_ADDRESS();
    g_trap_pc = 0;
    int n = 0;
    g_trap_rets[n++] = trap_site;
    g_trap_nrets = svm_walk_fp_chain(guest_fp, g_trap_rets, n);
    g_trap_fiber = g_current_fiber;
    g_trap_valid = 1;
}

/* Read and clear the captured trap stack (the host calls this after a guarded run reports a trap).
 * Fills `*pc` and up to `max` return addresses into `rets`; returns the number written, or -1 if
 * nothing was captured. */
int svm_take_trap_frame(uintptr_t *pc, uintptr_t *rets, int max) {
    if (!g_trap_valid)
        return -1;
    g_trap_valid = 0;
    *pc = g_trap_pc;
    int n = g_trap_nrets;
    if (n > max)
        n = max;
    for (int i = 0; i < n; i++)
        rets[i] = g_trap_rets[i];
    return n;
}

/* The guest fiber handle captured with the most recent trap (paired with `svm_take_trap_frame`), or
 * `SVM_NO_FIBER` when the root computation (no fiber) trapped. Not cleared — read it right after a
 * successful `svm_take_trap_frame`. */
int64_t svm_take_trap_fiber(void) {
    return g_trap_fiber;
}

/* Peek the fiber running on this thread *now* (not the captured one) — for the Windows VEH path, whose
 * memory-fault capture lives Rust-side and snapshots this at fault time. `SVM_NO_FIBER` = root. */
int64_t svm_current_fiber(void) {
    return g_current_fiber;
}
