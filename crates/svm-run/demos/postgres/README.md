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
| 1 | **inline `asm`** (~9 distinct templates) | 921 | recognize-and-lower table: empty/`lock;addl`/`rep;nop` → no-op (barriers/PAUSE); `xchgb`/`xaddl`/`cmpxchg` → single-threaded load-op-store; `cpuid`/`popcnt` → fixed-value / `Popcount` |
| 2 | **`atomicrmw`** (generic `__atomic` path) | 110 | single-threaded lowering to load-op-store (shares #1's lowering) |
| 3 | **`i128`** (numeric/aggregate widening) | 252 | config lever `#undef HAVE_INT128` (pure-64-bit fallback), or i128-as-`{i64,i64}` |
| 4 | **vectors `<N x …>`** (mostly `<16 x i8>` struct `memcpy`) | ~3638 | general lane-wise scalarize-vector-memory pass; width census first |
| 5 | **varargs** (`llvm.va_start`; 0 `va_arg` instrs) | 43 | confirm the `printf` varargs shape covers `elog`/`ereport` |
| 6 | **fs/syscall shim** (runtime, not IR) | — | bridge `open`/`pread`/`pwrite`/`fsync`/`stat`/… to the `fs` capability; stub `getpid`/`geteuid`/clock; gate the root check |
| 7 | **data dir** | — | `initdb` natively, expose read/write via the `fs` cap (SQLite Phase B pattern) |

**Staged plan:** (1) portable-atomics + no-`int128` config + the asm/`atomicrmw` recognizer → the
module translates; (2) the vector-memory scalarization pass; (3) the fs/syscall shim + pre-`initdb`
data dir → boot `postgres --single` on a fixed SQL script byte-identical to native; (4) a
`pg_regress` subset. See `LLVM.md` slice BM for the full write-up.
