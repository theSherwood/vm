# SQLite — the north-star demos (LLVM.md §8, slices BI + BJ)

Two whole-program demos driving the **unmodified SQLite amalgamation** through the LLVM→SVM-IR
on-ramp, byte-identical to the same source built natively with `cc`.

The amalgamation itself is **not vendored** (public domain, ~9 MB): the tests fetch and cache it
(`fetch_sqlite_amalgamation` in `crates/svm-llvm/tests/translate.rs`, currently 3.50.2) and skip
cleanly offline. **SQLite ≥ 3.47 is required** — earlier releases carry `long double` literals in
`sqlite3FpDecode` (`x86_fp80` in the IR, outside the f64 on-ramp); 3.47+ replaced that path with
Dekker double-double arithmetic, so the build is f64-clean with no source patching.

## `sqlite_demo.c` — Phase A, in-memory (test: `demo_sqlite_vs_native`)

A `:memory:` database running a 29-statement breadth script (DDL + indexes, recursive-CTE inserts,
aggregates, window functions, string/CASE/NULL semantics, floats through SQLite's own `%!.15g`,
date/time, `random()`, transactions, `PRAGMA integrity_check`, a deliberate error). Everything
nondeterministic is pinned in a `SQLITE_OS_OTHER=1` VFS — fixed-seed PRNG, fixed clock, and an
`xOpen` that fail-closes, proving the in-memory build cannot even reach for a disk.

## `sqlite_cap_vfs.c` — Phase B, disk-backed via the Fs capability (test: `demo_sqlite_fs_cap_vs_native`)

The database is a real file (`test.db`); in the guest build every byte flows through the
embedder-granted `fs` capability (`svm-run`'s `mem_fs`/`host_fs` — see `crates/svm-run/src/fs.rs`
for the op protocol): a guest `sqlite3_vfs` bridges xOpen/xRead/xWrite/xTruncate/xSync/xFileSize/
xDelete/xAccess to `__vm_cap_resolve("fs")` + `__vm_host_call`, the same way Lua's `io` runs. One
source file, two builds:

- `-DSVM_GUEST` → `SQLITE_OS_OTHER=1` + the capability VFS (translated, run in the sandbox);
- plain `cc` → SQLite's stock unix VFS (the native oracle).

The test asserts three directions: guest (`mem_fs`) stdout byte-matches native over
create → close → reopen → verify; under `host_fs` the guest's `test.db` really lands on disk and
**native SQLite opens the guest-written file**; and the guest reads a native-written database.

## `sqlite_logictest.c` — SQLite's own test corpus (tests: `demo_sqlite_logictest`, `_full`)

A compact **sqllogictest** runner (https://sqlite.org/sqllogictest/): reads a script from stdin,
runs every record against the same in-memory build as Phase A, and checks results against the
expected values the corpus bakes in (including the `N values hashing to <md5>` form — an RFC-1321
MD5 is embedded, and value formatting matches the reference runner byte-for-byte). Doubly gated:
the summary must report `failed=0` (self-validation) and guest stdout must byte-match the native
build over the same stdin (differential). The scripts are fetched-with-cache from the stable
GitHub mirror; CI runs `select1.test` (1031 records), the `_full` `#[ignore]`d sweep covers all
seven fetched scripts (~46k records).

## Running by hand

```sh
# fetch once
curl -sO https://sqlite.org/2025/sqlite-amalgamation-3500200.zip && unzip sqlite-amalgamation-3500200.zip

# Phase A native oracle
cc -I sqlite-amalgamation-3500200 sqlite_demo.c -lm -o sqlite_a && ./sqlite_a

# Phase B native oracle (writes ./test.db; `./sqlite_b verify` re-reads it)
cc -I sqlite-amalgamation-3500200 sqlite_cap_vfs.c -lm -o sqlite_b && ./sqlite_b

# guest bitcode for either (the tests do this automatically)
clang -O2 -emit-llvm -c -fno-vectorize -fno-slp-vectorize [-DSVM_GUEST] \
  -I sqlite-amalgamation-3500200 <demo>.c -o demo.bc
```
