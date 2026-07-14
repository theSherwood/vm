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
| 11b | **guest OS-shim: the file + directory syscalls** | — | **DONE (slice CA):** Postgres calls the libc syscall wrappers directly (`open`/`read`/`pread`/`write`/`pwrite`/`lseek`/`stat`/`fstat`/`lstat`/`access`/`unlink`/`rename`/`mkdir`/`rmdir`/`ftruncate`/`fsync`/`opendir`/`readdir`/`closedir`/`chdir`/`getcwd`) — in the whole-program bitcode those are undefined externals. `os_shim.c` **defines** them for a guest build, bridging each to `__vm_cap_resolve("fs")` + `__vm_host_call` (the slice-BZ cap), filling glibc's `struct stat`/`dirent` by field. Differential `demo_pg_oscap_vs_native`: `os_probe.c`'s deterministic file+dir walk byte-matches the native glibc oracle over `mem_fs` *and* `host_fs` (self-cleaning: the granted root is empty afterward) |
| 11c | **guest pure-libc: ctype** | — | **DONE (slice CB):** glibc's `<ctype.h>` `isalpha`/`isdigit`/… macros index locale tables reached through `__ctype_b_loc`/`__ctype_tolower_loc`/`__ctype_toupper_loc` — undefined externals in the guest, and Postgres's SQL scanner/parser classify every input byte through them. `libc_shim.c` provides the **C/POSIX-locale** tables as static compile-time literals (no runtime init). Differential `demo_pg_ctype_vs_native`: `ctype_probe.c` prints all twelve classifications + case mapping for every byte 0..255 and the guest byte-matches the native glibc oracle over the whole range (pinning every bit of every table) |
| 11d | **guest libc: string + integer parsing + proc/time/signal** | — | **DONE (slice CC):** `libc_shim.c` adds the `<string.h>`/`<stdlib.h>` members the on-ramp doesn't already synthesize — `strcat`/`strncpy`/`strnlen`/`strstr`/`strchrnul`/`strdup`/`strlcpy`/`strlcat`/`strtok`/`strxfrm`/`strcoll_l` and `strtol`/`strtoul`/`atoi` (`__isoc23_*` aliases too; `strtod`/`snprintf`/`getenv` were already synthesized), plus a shared `errno` cell (`shim_errno.h`). `proc_shim.c` returns the deterministic process/time/signal values a single-user sandbox backend needs — constant non-root identity (so Postgres's root guard passes), a frozen clock, inert signal masks, no-op sleeps. Tests: `demo_pg_string_vs_native` (byte-exact over signs/bases/prefixes/endptr/ERANGE + bounded copies vs glibc) and `demo_pg_procstub` (guest stub values) |
| 11e | **guest libc: file stdio + time + wide-char** | — | **DONE (slice CD):** `stdio_shim.c` layers the buffered `FILE*` surface Postgres declares (`fopen`/`freopen`/`fclose`/`fread`/`fwrite`/`fgetc`/`getc`/`fgets`/`fputc`/`fseek`/`fseeko`/`ftell`/`feof`/`ferror`/`clearerr`/`fflush`/`fileno`/`setvbuf`/`ungetc`) on `os_shim.c`'s fs-cap syscalls — its config reader `guc-file.l` uses `fopen`/`fgets`. `time_shim.c` adds `gmtime`/`localtime` (UTC calendar math) + a `strftime` format engine; `libc_shim.c` the C-locale `mbstowcs`/`wcstombs`. Differentials `demo_pg_stdio_vs_native` (write→reopen→read-back over `mem_fs` + `host_fs`) and `demo_pg_time_vs_native` (epochs incl. leap days + wide-char round-trip) |
| 11f | **stream `FILE*` + fd-dispatch** | — | **DONE (slice CE):** `stdout`/`stderr`/`stdin` must reach the powerbox **Stream** cap while `fopen`'d files reach the **fs** cap — but a guest that defines `write`/`read` (the syscall shim) shadows the on-ramp's Stream recognizer. Two new on-ramp builtins `__vm_stream_write`/`__vm_stream_read` (a `cap_spec` with `drop_args: 0` + import registration) give the shim direct Stream access; `os_shim.c`'s `write`/`read` **fd-dispatch** fds 0/1/2 to them and everything else to the fs cap; and the `fs` cap now **reserves fds 0/1/2** (`alloc_fd`) so a file fd (≥3) never collides with a stream fd. `stdio_shim.c` defines the `stdin`/`stdout`/`stderr` `FILE*` globals. Differential `demo_pg_stream_vs_native` (write to `stdout` FILE* *and* a real file; byte-exact over `mem_fs` + `host_fs`) |
| 11g | **first boot: the module runs** | — | **DONE (slice CF):** with all the shims linked in (`link_shims.sh`), the whole Postgres module **translates, verifies, AND runs** — fast (~24 s incl. translate) — executing real backend startup. The first live fault was a `MemoryFault` in `save_ps_display_args` walking an undefined **`environ`**; defining it (empty) cleared it. Added the early-startup libc surface: `environ`, `setlocale`/`newlocale`/`localeconv`/`nl_langinfo` (C-locale constants), the wide-ctype `iswX`/`towX` family (differential-tested, `demo_pg_wctype_vs_native`), `getopt`, `strsignal`. Postgres now boots past `save_ps_display_args` into deeper startup. See "Booting" below |
| 11h | **boot diagnostic: output on trap** | — | **DONE (slice CG):** the `Unreachable`/`MemoryFault` traps carried no name, so the boot chase was blind. `run_with_caps` now folds the guest's captured stdout/stderr into the trap error (`trap_err_with_output`) — a trapped program names its own failure. First payoff: the boot trap now reads `LOG:  could not find a "postgres" to execute` (Postgres's `find_my_exec` can't resolve its own binary path — `readlink("/proc/self/exe")`/argv[0]). Test `trap_error_surfaces_guest_output` |
| 11i | **boot: past `find_my_exec` + the IPC collapses** | — | **DONE (slice CI):** `find_my_exec` succeeds with a slashed `argv[0]` (`./postgres`) + an executable `postgres` in the data dir (its `validate_exec` is `stat` + `access(X_OK)`, both shimmed), so Postgres advances into shared-memory/semaphore setup. `ipc_shim.c` gives the single-process collapses: `mmap(MAP_ANONYMOUS)` → zeroed writable memory (one address space — **differential-tested** `demo_pg_mmap_vs_native`), `munmap`/`madvise`/`posix_fadvise` no-ops, POSIX `sem_*` uncontended no-ops, System-V `shmget`/`shmat` forced onto the anonymous-mmap path. Postgres now runs *past* exec resolution into early init |
| 11j | **guest libc: varargs `printf` engine** | — | **DONE (slice CH):** `printf_shim.c` provides the runtime `printf`/`fprintf`/`vfprintf`/`vprintf`/`snprintf`/`sprintf` family Postgres formats results + `elog`/`ereport` log lines with (a query result builds its directives at runtime — the on-ramp's constant-format engine can't lower that). The byte-exact-vs-glibc formatter (integers/strings/chars/`%p`/`%a` in C; `%f`/`%e`/`%g` via the bignum `__vm_fmt_*` dtoa) formats into a buffer and `fwrite`s it, composing with #11f's stream/file fd-dispatch. Linked into the boot via `pg_shims.c`. Differential `demo_pg_fprintf_vs_native` (format family to `stdout` + a real file + `stderr`, byte-exact on all three engines over `mem_fs` + `host_fs`) |
| 11k | **boot: name the silent traps, then forward** | — | the boot now traps **silently** (no output) in early init — the CG output-diagnostic only helps once Postgres has *printed*. Make stub-traps **self-naming** (the trap error carries the missing extern's name) — an on-ramp diagnostic that ends the guessing — then drive the now-legible boot through the storage manager / WAL / catalog bootstrap to the first `SELECT`, plus `strerror_r` (`_GNU_SOURCE` TU) |
| 11l | **guest libc: varargs `scanf` engine + real `strtod`** | — | **DONE (slice CJ):** `scanf_shim.c` provides the runtime `sscanf`/`vsscanf`/`fscanf`/`vfscanf`/`scanf`/`vscanf` family Postgres parses config/version values with (the input twin of #11j) — d/i/u/o/x/c/s/f/`%[scanset]`/`%n`/`%%` with `*`-suppression, field width, and length modifiers, over one char-source abstraction (a string, or a `FILE*` via `fgetc`/`ungetc`). Its `%f`/`%lf` need a **real `strtod`** — the on-ramp's is a *trap stub* — so the correctly-rounded bignum `strtod.c` (shadows the stub; `float8in` needs it too) is linked in beside it via `pg_shims.c`. Differential `demo_pg_sscanf_vs_native` (the conversions + return-count semantics + an `fscanf`-from-`stdin` half, byte-exact on all three engines) |

**Where it stands:** ★★ **the complete module (832 modules / 14 985 functions) now translates AND
verifies** — past the **entire 921-site inline-asm surface** (BN/BO), all 18 **libm** transcendentals
(BQ), the **entire ~250-external OS/libc surface** with `--stub-externs` (BR), the **whole SIMD tail**
(BV), the **indirect varargs call** + two **i128** op lowerings (BW), the final **vector `llvm.bswap`**
(BX), and — after `resolve_imports` binds the 4 powerbox caps (`read`/`write`/`exit`/`vm_map` →
`cap.call`) — a **clean `svm-verify` pass** (BY fixed the one remaining `TypeMismatch`). What's left is
purely the **runtime**, and slice BZ starts it: the `fs` capability can now **walk a data tree**
(`stat`/`mkdir`/`rmdir`/`opendir`/`readdir` — gap #11a), and a guest **OS-shim** (`os_shim.c`, gap
#11b) bridges the file + directory syscalls Postgres calls onto that cap — differential-clean vs the
native glibc oracle — and `libc_shim.c`/`proc_shim.c` (gaps #11c–#11e) cover the pure-libc surface so
far: the C-locale **ctype** tables, the **string + integer-parsing** members, the deterministic
**process/time/signal** stubs, the file-backed **stdio `FILE*`**, **time + wide-char**, and the
**stream `FILE*`** (`stdout`/`stderr`/`stdin` → the powerbox Stream via the CE on-ramp fd-dispatch).
And **the module now boots** (gap #11g, slice CF): linked with all the shims it translates, verifies,
and *runs* real backend startup, past `save_ps_display_args` (the `environ` fix) into deeper init.
The remaining stub-traps are now **legible** (gap #11h, slice CG: `run_with_caps` folds the guest's
captured output into the trap error) — the first boot trap reads `LOG:  could not find a "postgres"
to execute`. Slice CI clears that (gap #11i): a slashed `argv[0]` (`./postgres`) + an executable
`postgres` in the data dir lets `find_my_exec`'s `stat`/`access(X_OK)` succeed, and the
**single-process IPC collapses** (`ipc_shim.c`) carry the boot into shared-memory/semaphore init —
anonymous `mmap` is just `malloc`+zero (differential-tested `demo_pg_mmap_vs_native`), `shmat` forces
that path, and the unnamed semaphores are uncontended no-ops. The runtime **varargs `printf` engine**
(gap #11j, slice CH: `printf_shim.c` — the byte-exact-vs-glibc formatter Postgres builds its result +
`elog` output with, composing with the #11f fd-dispatch) is in place for when the boot reaches its
first printed output. The boot now traps **silently** in early init (no printed output for CG to
surface), so still ahead (gap #11k): make stub-traps **self-naming** to name the silent trap, then
drive on through the storage manager, WAL, catalog bootstrap, and the varargs `scanf` **input** engine.
See `LLVM.md` slices BM–CI, and "Booting" below.

## Booting

`build_bitcode.sh` produces `postgres_libm.bc` (the whole-program module + openlibm). `link_shims.sh`
then `llvm-link`s the guest shims (`pg_shims.c` = os/libc/locale/time/proc/stdio) into it →
`postgres_shimmed.bc`. A driver `translate_bc_path_with_options(…, stub_unresolved_externs: true)` →
`instantiate` → `run_with_caps(Backend::Bytecode, {stdin: SQL, args: ["postgres","--single","-D",".",…]},
[("fs", host_fs(datadir))])` runs it. As of slice CF the module executes real startup and traps deeper
in init; driving it to the first `SELECT` is gap #11h.
