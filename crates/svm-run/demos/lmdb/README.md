# LMDB — an embedded memory-mapped B-tree in the sandbox (LLVM.md slice BL)

The storage ladder's **second shape**, after SQLite's read/write VFS (slices BI–BK): LMDB
(OpenLDAP's Lightning MDB — the original mmap'd B-tree that libmdbx later hardened) reads its B-tree
straight out of a **file-backed memory mapping**. The data plane *is* the map — no per-access
syscalls — so it exercises a capability shape the SQLite work never touched.

The mmap flows through the granted **Fs capability's mmap surface** (`FS_MMAP`/`FS_MSYNC`/`FS_MUNMAP`,
see `crates/svm-run/src/fs.rs`): a guest shim bridges LMDB's `mmap`/`msync`/`pread`/`open`/… to
`__vm_cap_resolve("fs")` + `__vm_host_call`. Zero ambient authority — no capability, no bytes.

LMDB itself (`mdb.c`/`midl.c`/`lmdb.h`/`midl.h`) is **not vendored** (OpenLDAP Public License): the
test fetches and caches it from the upstream GitHub mirror (`fetch_lmdb` in svm-llvm's
`translate.rs`) and skips cleanly offline.

## Files

- **`lmdb_demo.c`** — the driver: opens `MDB_NOSUBDIR | MDB_NOLOCK | MDB_WRITEMAP` (single file, no
  lock table, writable map = coherent with the copy-in/flush-out emulation), fills 500 keys in
  scrambled order with deletes, then close → reopen → point-lookups + a full ordered cursor scan
  (a running checksum over the in-order B-tree walk) + stat. `verify` mode re-reads an existing
  `data.mdb`. Compiled for both builds; the native oracle uses real glibc/`mmap`.
- **`lmdb_shim.c`** — guest-only (`-DSVM_GUEST`): the Fs-capability bridge for the file + mmap
  syscalls, plus single-thread no-op stubs (pthread/`sysconf`/`uname`/`fcntl`/`fstatfs`/…) for the OS
  odds-and-ends `MDB_NOLOCK` never exercises. Everything else (malloc/free, the mem/str families,
  printf, strtod, …) the on-ramp already synthesizes.

## Why WRITEMAP

In `MDB_WRITEMAP` mode LMDB writes **every** page — data and meta — through the map (no `pwrite`
path), so the map is the single source of truth. Our mmap emulation copies the file into a guest
buffer on `mmap`, serves all reads/writes from it, and flushes back on `msync` — coherent precisely
because nothing writes the file behind the buffer's back.

## Test: `demo_lmdb_mmap_cap_vs_native`

Three directions (mirroring SQLite Phase B):
1. **stdout differential** — guest (`mem_fs`) byte-matches the native oracle (real mmap in a temp dir);
2. **the capability story** — under `host_fs` the guest's `data.mdb` really lands on disk and
   **native LMDB opens the guest-written file** and verifies it byte-identically (cross-implementation
   on-disk-format proof, capability-written);
3. **the reverse** — the guest reads a native-written `data.mdb`.

## Running by hand

```sh
# fetch once
B=https://raw.githubusercontent.com/LMDB/lmdb/mdb.master/libraries/liblmdb
for f in mdb.c midl.c lmdb.h midl.h; do curl -sO $B/$f; done

# native oracle (writes ./data.mdb; `./lmdb verify` re-reads it)
cc -O2 -I. mdb.c midl.c lmdb_demo.c -lpthread -o lmdb && ./lmdb

# guest bitcode (the test does this automatically): compile each -DSVM_GUEST, llvm-link, translate
```
