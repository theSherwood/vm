# Postgres `--single` on the LLVM on-ramp — pipeline + gap inventory (slice BM, a SPIKE)

The ladder-#7 capstone (`LLVM.md` §"Suggested ladder"): the *single-user* Postgres backend
(`postgres --single`) — one process reading SQL on **stdin**, no postmaster (no fork-per-connection,
no SysV shmem across processes, no listening socket, no signal-driven concurrency). Its whole
`PG_TRY`/`ereport` error model is `sigsetjmp`/`siglongjmp` (already landed on all three engines), and
its only real OS need is a **File capability** for the data dir — "SQLite Phase B at 100×."

This directory is the **reproduction** for the feasibility spike. It is *fetched-not-vendored*
(PostgreSQL license): `build_bitcode.sh` downloads the release tarball, builds the native oracle,
emits per-TU bitcode, links the exact `postgres` object set into one module, and runs it through the
on-ramp to enumerate the translator gaps. Postgres is **not** a single amalgamation like SQLite, so
the pipeline links ~833 modules rather than compiling one `.c`.

## Reproduce

    # needs: clang-18, llvm-dis, llvm-link, flex, bison, perl, make, curl
    bash build_bitcode.sh          # ~10-15 min: fetch → configure → native build → bitcode → link → translate

Artifacts land in `$SVM_PG_CACHE` (default `/tmp/svm_pg_cache`):
`inst/bin/postgres` (native oracle), `postgres.linked.bc` / `.ll` (the whole-program module),
`translate.err` (the first on-ramp gap).

The native oracle refuses to run as root; `build_bitcode.sh` runs the smoke test under an
unprivileged user if invoked as root.

## What the spike established

- **Native oracle works.** Postgres 17.5 + clang-18 (minimal config); `postgres --single` returns
  `SELECT 1+1, upper('hi')` → `2` / `HI`. The differential target.
- **The reader scales.** 833 modules → 17.8 MB `.bc` → 78 MB / 1.59 M-line `.ll`, **14 563** defined
  functions. The in-house textual-`.ll` reader ingests it and **fail-closes cleanly** on the first
  unsupported construct (inline `asm`). Scale is not the blocker.
- **Confirmed non-blockers:** `invoke`/`landingpad`/`resume` = 0 (no C++ EH — `--single` is
  `sigsetjmp`-only), `x86_fp80`/`fp128` = 0, `thread_local` = 0, `llvm.stacksave` = 0.

## Gap inventory (the deliverable)

| # | Gap | Sites | Route |
|---|-----|------:|-------|
| 1 | **inline `asm`** (~9 distinct templates) | 921 | **DONE** (slices BN/BO): barriers/PAUSE → drop; `popcnt` → `Popcnt`; `xchg`/`xadd`/`cmpxchg` → the runtime atomic ops (genuinely atomic); `cpuid` → zeroed → software fallbacks |
| 2 | **`atomicrmw`/`cmpxchg` instrs** | 110 | already lowered by the on-ramp (the asm atomics route to the same ops) |
| 3 | **`i128`** (numeric/aggregate widening) | 252 | on-ramp already lowers i128 div/rem; general i128 arith is tier-3'd — verify on demand |
| 4 | **libm** (`log`/`exp`/`pow`/trig) | 18 | **DONE** (slice BQ): openlibm's double funcs bundled as guest code, llvm-linked; on-ramp reproduces them bit-for-bit vs native (`libm_bundled_vs_native`) |
| 5 | **file/OS syscalls** | ~30 | the **`fs` capability** shim: `open`/`pread`/`pwrite`/`fsync`/`stat`/`mkdir`/`opendir`/`mmap`/… |
| 6 | **proc/time/signal** | ~24 | deterministic stubs: `getpid`/`geteuid`/`clock_gettime`/`sigaction`/`fork`/`kill`/… |
| 7 | **other libc** | ~180 | `strtod`/`snprintf`/`qsort`/`setlocale`/`strftime`/`memmem`/… — some synthesized, many not yet |
| 8 | **data dir + runtime** | — | `initdb` natively → expose via the `fs` cap; storage manager, WAL, single-process shmem, catalog bootstrap |

**Where it stands:** the complete module (834 modules / 14 730 functions) translates past the **entire
921-site inline-asm surface** (BN/BO) and, with the **bundled openlibm** llvm-linked (BQ), past all 18
**libm** transcendentals. Translation now fail-closes at the next undefined external (`strchrnul`):
the remaining ~230 externals (file/OS syscalls, proc+signal, other libc) must each resolve —
synthesized helper, `fs` capability, bundled guest code, or stub — before the module translates, then
verify + the runtime (initdb data dir, storage manager, WAL, single-process shmem, catalog bootstrap)
must work. That external waist + runtime is the **multi-week bulk** of the capstone. See `LLVM.md`
slices BM–BQ.
