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
| 5 | **the whole external surface (~250)** — file/OS syscalls, proc/time/signal, other libc | ~250 | **DONE at translate time** (slice BR): opt-in `--stub-externs` lowers every undefined external to a trap-if-called stub, so the ~200 dead on the `--single` path don't block. Only the ~50 the query path *calls* need real impls (fs cap / stubs) for the **runtime** |
| 6 | **SIMD vector ops** (per-lane `<16 x i32>`/`<4 x i64>` shifts, …) | ~9+ | per-lane scalarization in the on-ramp, or a config lever compiling the SSE/AVX fast-paths out (dead under `cpuid`→0, still compiled) |
| 7 | **data dir + runtime** | — | `initdb` natively → expose via the `fs` cap; storage manager, WAL, single-process shmem, catalog bootstrap — the ~50 *live* externals resolve here |

**Where it stands:** the complete module (834 modules / 14 730 functions) translates past the **entire
921-site inline-asm surface** (BN/BO), all 18 **libm** transcendentals (bundled openlibm, BQ), and —
with `--stub-externs` (BR) — the **entire ~250-external OS/libc surface** (the ~200 dead on the query
path become trap-if-called stubs). Translation now stops at the **SIMD vector tail** (~9 per-lane
vector shifts from explicit SSE/AVX). Remaining before it fully translates + verifies: that vector
category; then the **runtime** (initdb data dir, storage manager, WAL, single-process shmem, catalog
bootstrap) with real impls for the ~50 externals the query path actually calls. See `LLVM.md` slices
BM–BR.
