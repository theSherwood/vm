/*
 * svm.h — C bindings for the SVM embedding runtime (POWERBOX.md Phase 5).
 *
 * Pipeline: parse a module (text or binary IR) -> svm_module_synth_powerbox_start ->
 * bind host capabilities by name (built-ins or your own C callbacks) -> svm_instantiate* ->
 * svm_instance_run / svm_instance_run_diff -> read the outcome and captured stdout/stderr.
 *
 * Conventions:
 *   - Handles are opaque pointers; functions that say "consumes" take ownership (do not free after).
 *   - A NULL return or a non-zero status means failure; call svm_last_error() for the message.
 *   - Panics never cross the boundary (they become a NULL/error return).
 *
 * Link against libsvm_capi.a (staticlib) or libsvm_capi.{so,dylib} (cdylib).
 */
#ifndef SVM_H
#define SVM_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Status codes. */
#define SVM_OK 0
#define SVM_ERR_NULL 1
#define SVM_ERR_FAILED 2
#define SVM_ERR_PANIC 3

/* Backend selectors for svm_instance_run. */
#define SVM_BACKEND_TREEWALK 0
#define SVM_BACKEND_BYTECODE 1
#define SVM_BACKEND_JIT 2

/* svm_run_outcome_kind values. */
#define SVM_OUTCOME_RETURNED 0
#define SVM_OUTCOME_EXITED 1

/* Opaque handles. */
typedef struct SvmModule SvmModule;
typedef struct SvmImports SvmImports;
typedef struct SvmInstance SvmInstance;
typedef struct SvmRun SvmRun;
/* The calling guest's linear-memory window, passed to a host-fn callback for that call only. */
typedef struct SvmGuestMem SvmGuestMem;

/* The last error message on this thread (or NULL). Valid until the next svm-capi call. */
const char *svm_last_error(void);

/* ---- Module ---- */
SvmModule *svm_module_parse_text(const char *ir);
SvmModule *svm_module_decode(const uint8_t *bytes, size_t len);
/* Prepend the powerbox _start for n_handles granted handles (slot i <-> import i); 0 = SVM_OK. */
int32_t svm_module_synth_powerbox_start(SvmModule *m, uint32_t entry, size_t n_handles,
                                        bool seed_heap);
void svm_module_free(SvmModule *m);

/* ---- Imports registry (wasm-style name -> capability) ---- */
SvmImports *svm_imports_new(void);
int32_t svm_imports_provide_stdout(SvmImports *i, const char *name);
int32_t svm_imports_provide_stdin(SvmImports *i, const char *name);
int32_t svm_imports_provide_exit(SvmImports *i, const char *name);
int32_t svm_imports_provide_clock(SvmImports *i, const char *name);

/*
 * A host-capability callback: compute up to results_cap outputs from n_args inputs for operation op.
 * Return the number of results written (>= 0), or a negative value to trap the capability call.
 * ctx is the opaque pointer registered alongside the callback. mem is the calling guest's window
 * (NULL if the module declares none), accessible via svm_guest_read/svm_guest_write for this call
 * only — do not retain it past the callback.
 */
typedef int32_t (*SvmHostFn)(void *ctx, uint32_t op, const int64_t *args, size_t n_args,
                             int64_t *results, size_t results_cap, SvmGuestMem *mem);
int32_t svm_imports_provide_host_fn(SvmImports *i, const char *name, uint32_t op, SvmHostFn fn,
                                    void *ctx);
void svm_imports_free(SvmImports *i);

/*
 * Read/write the guest window from inside a host-fn callback, bounds-checked (fail-closed): each
 * returns SVM_OK, or SVM_ERR_FAILED (nothing transferred) if mem/buf is NULL or [ptr, ptr+len) is
 * not wholly within the window (and, for write, writable). The same §7 confinement the built-ins get.
 */
int32_t svm_guest_read(const SvmGuestMem *mem, uint64_t ptr, uint8_t *dst, size_t len);
int32_t svm_guest_write(SvmGuestMem *mem, uint64_t ptr, const uint8_t *src, size_t len);

/* ---- Instantiate (consume the module / imports) ---- */
SvmInstance *svm_instantiate(SvmModule *m);                          /* fixed §3e powerbox */
SvmInstance *svm_instantiate_with_imports(SvmModule *m, SvmImports *imports); /* by name */
void svm_instance_free(SvmInstance *i);

/* ---- Run config ---- (a NULL pointer means all defaults; *_set flags select a field). */
typedef struct {
  uint64_t fuel;       /* per-op budget for the interpreters (if fuel_set); ignored by the JIT */
  int32_t fuel_set;
  uint64_t deadline_ms; /* JIT detect-and-kill deadline (if deadline_set); ignored by interps */
  int32_t deadline_set;
  size_t max_fibers; /* §15 spawn quota (0 = default) */
  size_t max_vcpus;  /* §15 vCPU cap / "CPUs available" (0 = default) */
  const uint8_t *stdin_bytes; /* guest stdin (NULL/0 = empty) */
  size_t stdin_len;
  uint8_t memory_size_log2; /* window override (if memory_set) */
  int32_t memory_set;
} SvmRunConfig;

/* ---- Run ---- */
SvmRun *svm_instance_run(SvmInstance *i, int32_t backend, const SvmRunConfig *config);
SvmRun *svm_instance_run_diff(SvmInstance *i, const SvmRunConfig *config);

/* ---- Reactor sessions (Phase 6): instantiate once, call exports repeatedly, state persists ---- */
typedef struct SvmSession SvmSession;
SvmSession *svm_instance_start(const SvmInstance *i, int32_t backend, const SvmRunConfig *config);
/* Call `name` with n_args i64 args; write up to results_cap i64 results + *n_results. 0 = SVM_OK. */
int32_t svm_session_call_export(SvmSession *s, const char *name, const int64_t *args, size_t n_args,
                                int64_t *results, size_t results_cap, size_t *n_results);
const uint8_t *svm_session_stdout(const SvmSession *s, size_t *len);
void svm_session_free(SvmSession *s);

/* ---- Run results ---- (pointers valid until svm_run_free). */
const uint8_t *svm_run_stdout(const SvmRun *r, size_t *len);
const uint8_t *svm_run_stderr(const SvmRun *r, size_t *len);
int32_t svm_run_outcome_kind(const SvmRun *r);
int32_t svm_run_exit_code(const SvmRun *r);
size_t svm_run_result_count(const SvmRun *r);
int64_t svm_run_result(const SvmRun *r, size_t idx);
void svm_run_free(SvmRun *r);

#ifdef __cplusplus
}
#endif

#endif /* SVM_H */
