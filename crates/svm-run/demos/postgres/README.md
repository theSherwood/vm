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
| 6 | **SIMD vector ops** | — | **DONE (slice BV) via two build-config levers.** Most of the "SIMD tail" was never real Postgres SIMD — it was clang **auto-vectorizing** scalar C loops (`<2 x i32>` loads → `<2 x ptr>` gather-GEPs). `emit_bc.py` passed `-fno-vectorize -fno-slp-vectorize` but *before* the recovered `-O2`, which re-enabled it (last flag wins); appending the flags after `-O2` disables it for real, and the whole auto-vectorized tail vanishes. The residual **explicit** SIMD (SSE4.2 `_mm_crc32`, 128-bit float vectors) already translates via slices Y/BS/BT/BU. **AVX-512** (`pg_popcount_avx512`, `<64 x i1>`) is dead under the `cpuid`→0 stub and is dropped at the source by the second lever: `configure` is told the AVX-512 popcount autodetect is "no", so `USE_AVX512_POPCNT_WITH_RUNTIME_CHECK` is never defined and `PG_POPCNT_OBJS` is empty |
| 7 | **indirect varargs call** (`manifest_process_version`) | — | **DONE (slice BW):** the on-ramp already marshaled a *direct* `(...)` call's variadic args into overflow scratch; three coordinated edits extend the exact same marshaling to an **indirect** `(...)` callee (a function pointer), which then lowers to `call_indirect` with a §3c type-id check against the `(sp, fixed-params…)` signature a defined `(...)` function uses. Test `varargs_indirect_call` (interp + bytecode + JIT vs native, incl. the `return_call_indirect` tail path) |
| 8 | **two missing i128 op lowerings** (`sqrt_var`, `int2_accum`) | — | **DONE (slice BW):** the reported `value … not available in block` was not a liveness bug — it was `lower_i128` missing two arms, so the *generic scalar* handler resolved an i128 (which lives as an `agg` `(lo,hi)` pair, not in `idx_of`) and failed. Added **`select i128`** (per-word `Select` on the pairs — numeric `sqrt_var`'s Newton loop) and **`store i128`** (two i64 stores, lo at base / hi at base+8, mirroring `load i128` — numeric `int2_accum`'s `sumX2`). Test `i128_select_and_store_roundtrip` (hand `.ll`; interp + JIT) |
| 9 | **vector `llvm.bswap`** (`pg_sha256_final`) | — | **DONE (slice BX):** a 128-bit vector byte-swap (`<4 x i32>`, SHA-256's big-endian digest write) — reverse the bytes *within each lane*, scalarized per lane through the existing scalar `emit_bswap` (mirrors vector `ctpop`). Test `vector_bswap_128` (hand `.ll`; interp + JIT). **This was the last translate gap** — the whole module (14 985 funcs) now translates end-to-end |
| 10 | **verify: aggregate fan-out in the sparse-`switch` chain** (`ExecRenameStmt`) | — | **DONE (slice BY):** the sole verify error across all 14 985 functions. `block_param_types` (which types a synthetic compare-chain block's params) fanned out **wide vectors** but not **aggregates**, while `block_params`/`branch_args` fan out both — so a by-value `{i64,i32}` struct threaded through `ExecRenameStmt`'s sparse `switch` contributed one placeholder type there vs two args, desyncing the `zip` and mistyping every value behind it. One-branch fix (fan aggregates out too). Test `switch_sparse_threads_aggregate` (hand `.ll`; translate + verify + interp) |
| 11a | **`fs` cap: the metadata + directory surface** | — | **DONE (slice BZ):** the runtime's first blocker — the `fs` capability could open/read/write/seek *files* but had no way to **walk a tree** (no `stat`/`mkdir`/`rmdir`/`opendir`/`readdir`), which a natively-`initdb`'d cluster needs pervasively. Added ops 14–19 (`svm-run/src/fs.rs`): `stat` fills a fixed 72-byte little-endian `StatBuf` (the `S_IF*` type bits + size + mtime + ino/dev) with **lstat** semantics (a symlink can't be used to probe a type outside the granted root); `mkdir`/`rmdir`; `opendir` snapshots a directory's entries and `readdir` yields them sorted, one per call. Both backends at protocol parity — `mem_fs` models directories over its flat name table (a path is a dir if `mkdir`'d or a strict prefix of a file), `host_fs` uses the real tree — so a differential runs identically on either. Tests `os_metadata_ops_parity_mem_vs_host` + `readdir_is_sorted_and_bounded` |
| 11b | **data dir + the rest of the runtime** | — | `initdb` natively → expose the dir through the (now tree-walkable) `fs` cap; a **guest OS-shim** binding the ~30 file syscalls + proc/time to the cap; the ~180 remaining pure-libc externs (stdio `FILE*`, locale, ctype tables, `strtod`/`snprintf`) byte-exact vs the native oracle; storage manager, WAL, single-process shmem, catalog bootstrap — the ~50 *live* externals resolve here |

**Where it stands:** ★★ **the complete module (832 modules / 14 985 functions) now translates AND
verifies** — past the **entire 921-site inline-asm surface** (BN/BO), all 18 **libm** transcendentals
(BQ), the **entire ~250-external OS/libc surface** with `--stub-externs` (BR), the **whole SIMD tail**
(BV), the **indirect varargs call** + two **i128** op lowerings (BW), the final **vector `llvm.bswap`**
(BX), and — after `resolve_imports` binds the 4 powerbox caps (`read`/`write`/`exit`/`vm_map` →
`cap.call`) — a **clean `svm-verify` pass** (BY fixed the one remaining `TypeMismatch`). What's left is
purely the **runtime**, and slice BZ starts it: the `fs` capability can now **walk a data tree**
(`stat`/`mkdir`/`rmdir`/`opendir`/`readdir` — gap #11a), the metadata surface a natively-`initdb`'d
cluster needs before Postgres can open a single relation. Still ahead (gap #11b): a guest OS-shim
binding the file syscalls + proc/time to that cap, the ~180 remaining pure-libc externs byte-exact vs
the native oracle, and then the storage manager, WAL, single-process shmem, and catalog bootstrap. See
`LLVM.md` slices BM–BZ.
