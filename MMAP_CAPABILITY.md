# File-backed mmap in the sandbox — design & goals

**Status:** design (pre-implementation). The LMDB slice (LLVM.md BL) shipped a *working* file-mmap
over the existing `HostFn` fs capability by **emulation**; this doc decides what the *first-class*
story should be before we build more.

## 1. What we are actually trying to prove

The storage ladder's thesis is: **a sandboxed guest does real, durable I/O only through explicitly
granted authority, auditable at the powerbox boundary.** SQLite (VFS, read/write) proved the
positioned-I/O shape. LMDB proves the **memory-mapped** shape, where the data plane *is* the
mapping — a program reads structured data straight out of the map with ordinary loads, no per-access
host call.

Three distinct things could be meant by "mmap works in the sandbox," in increasing ambition:

1. **Functional** — an mmap-centric program (LMDB) produces correct results in the sandbox, with its
   mmap flowing through a granted capability. *Achieved* (slice BL, by emulation).
2. **Zero-copy** — the guest reads the file's bytes directly out of its own window with no copy and
   no host round-trip; the host aliases the file into the window once. Not yet — emulation copies.
3. **Durable / crash-safe** — the capability has a **durability contract** (what `msync` guarantees,
   what a crash loses), and we can *demonstrate* crash-consistency: kill mid-transaction, reopen,
   prove the database recovers to the last committed state. Not yet.

The open question this doc answers: **which of (2) and (3) are worth building, in what order, and as
what capability shape?**

## 2. Where we are: the emulation, and why it's coherent-but-limited

Slice BL added three ops to the `HostFn` **fs** capability (`crates/svm-run/src/fs.rs`):

- `FS_MMAP(fd, file_offset, len, win_buf)` — `pread` the file region **into a guest-owned buffer**,
  record `win_buf → (fd, file_offset, len)`.
- `FS_MSYNC(win_buf, len)` — `pwrite` a sub-range of the buffer back to the file.
- `FS_MUNMAP(win_buf)` — flush + drop.

The guest shim's `mmap()` does `malloc(len)` then `FS_MMAP`. Between map and sync the guest does
direct loads/stores — so the *data-access path* already has zero host calls (the (2) property on
reads, once loaded). It is **coherent** only because LMDB runs `MDB_WRITEMAP`: every page — data and
meta — is written *through the map*, so the buffer is the sole authority; nothing writes the file
behind the buffer's back.

Its limits, precisely:

- **Not zero-copy.** `mmap` copies the whole file in (1 MiB here); a large DB would copy the whole
  geometry. Fine for a demo, wrong for scale.
- **No sharing.** Two `mmap`s of the same region get two independent buffers. LMDB's single-mapping
  config never does this, but a second reader, or `MDB_WRITEMAP`-off mode (map + `pwrite`), would
  silently diverge.
- **No durability contract.** `msync` happens to `pwrite`, and `munmap` flushes — but nothing
  *specifies* what survives a crash, and there is no way to *inject* one. So we cannot claim
  crash-safety, only "it round-trips when nothing goes wrong."

## 3. The machinery already in the tree

We are not starting from zero. The interface registry (`svm-interp` `iface`) already has **real**
window-aliasing capabilities, and DESIGN.md §13/§14 is the frame:

- **`SharedRegion` (iface 4)** — a host memory object (`memfd`/Windows section) aliased into the
  window with a **real shared mapping** (`mmap(MAP_SHARED|MAP_FIXED)` of the region's `os_fd` over
  `[win_off, win_off+len)`, `svm-run` `map_region`). The *same* backing can map at multiple window
  offsets → hardware-coherent aliasing (the magic-ring-buffer primitive). This is ~90% of the host
  mechanism a zero-copy **file** mapping needs — it just aliases a *memfd*, not a real file fd.
- **`AddressSpace` (iface 5)** / **`Memory` (iface 3)** — `map`/`unmap`/`protect`/`page_size` within
  the window, attenuable to a power-of-two sub-range (`sub`). The page-management half.
- **`HostFn` (iface 13)** — the embedder-registered escape hatch the fs cap (and the BL emulation)
  rides. Semantics live in the embedder's closure, *outside* the VM's escape-TCB match.

The key architectural fact: a **first-class file mmap = `SharedRegion`-style real aliasing, but the
backing `os_fd` is a host-opened file instead of a memfd.** On Linux that is one `mmap` of the file
fd `MAP_SHARED|MAP_FIXED` over a window sub-range; writes land in the page cache; `msync`/`fsync`
persist. Zero-copy and coherent *for free* from the OS — no copy-in, no flush-out, no coherence
bookkeeping. The emulation exists only because the fs cap is a `HostFn` (no window-mapping authority);
a dedicated cap in the `SharedRegion` family gets the real thing.

## 4. Design axes

### 4a. Backing: emulation vs real file-into-window aliasing

| | copy-in/flush (BL) | real `MAP_SHARED` of the file fd |
|---|---|---|
| zero-copy | no | **yes** |
| coherence | manual, WRITEMAP-only | **OS page cache**, any mode |
| large DB | copies whole geometry | maps lazily (demand paging) |
| window cost | a heap buffer per map | a window sub-range reservation |
| portability | pure Rust, all OSes | needs the `map_region` FFI path (Linux done; macOS/Win follow) |
| escape-TCB | none (HostFn) | it's `MAP_FIXED` into the guest window — **same TCB as SharedRegion** |

Real aliasing is the right long-term answer and reuses `map_region`. The cost is it must live in a
window-mapping capability (not `HostFn`), and it inherits `SharedRegion`'s per-OS story.

### 4b. Capability shape: fs op-protocol vs a dedicated iface

- **Stay on the fs `HostFn`** (BL): `mmap`/`msync`/`munmap` are ops 9–11 on the fs handle. Cheap,
  already shipped, but can never do real window aliasing (a `HostFn` closure has no authority to
  `MAP_FIXED` into the window — that's escape-TCB, reserved for the built-in iface match).
- **A dedicated `FileMapping` iface (14)** in the `SharedRegion` family: `open`/`map`/`msync`/
  `unmap`/`len`/`sync`. Real aliasing, clean attenuation (a mapping handle *is* the authority to a
  file region), discoverable, composes with `AddressSpace.sub`. This is the DESIGN §13/§14-consistent
  home. Cost: a new type_id in the escape-TCB match + the host backing + a guest shim rewrite.

The emulation and the dedicated cap are **not** mutually exclusive: the fs-cap emulation stays valid
as the portable/hermetic `mem_fs` path (and for hosts without the FFI); the dedicated cap is the
zero-copy real-file path. The guest shim can pick by which capability the embedder granted.

### 4c. Durability contract (the part with no code today)

A capability that persists needs a *specified* crash model, independent of backend:

- **`msync(range)` is the barrier**: after it returns, that range is durable (survives a crash).
  Buffer/map writes **not** covered by a completed `msync` may be lost.
- **Ordering**: two `msync`s are ordered; the cap must not reorder them (a DB's meta-page commit
  depends on data pages being durable first — LMDB's whole double-buffered-meta scheme).
- **Atomicity granularity**: a single-page `msync` is all-or-nothing (torn writes are the classic DB
  hazard; do we promise page-atomic, or expose the tear and let the DB's checksum catch it?).
- **`fsync` vs `msync`**: LMDB uses both — `msync` the map, then `fdatasync` the fd. The contract
  must say whether they're distinct barriers or one.

This is the actual intellectual content of "first-class mmap" and is **backend-independent** — worth
writing down even if we keep the emulation.

### 4d. Crash injection & recovery proof

To *demonstrate* the contract (not just assert it), the cap needs a test-only **crash hook**: after
N host writes, or on an explicit "crash now" op, drop all un-`msync`'d state (and optionally reorder
/ tear the last write) and refuse further I/O. Then a `demo_lmdb_crash_recovery` test: fill → crash
mid-txn → reopen → assert the DB is consistent to the last *committed* transaction (LMDB guarantees
this by design — its meta pages are double-buffered and checksummed). This is the highest-narrative
slice and is **mostly independent** of 4a/4b — it can run on the emulation.

### 4e. Multi-mapping coherence

Real `MAP_SHARED` gives this for free (OS page cache). The emulation does not. LMDB's chosen config
(single WRITEMAP mapping, NOLOCK) never needs it, so it only matters if we target a program that maps
a file at two window offsets, or mixes map-reads with `pwrite`. Low priority unless a target demands
it; real aliasing (4a) dissolves it anyway.

## 5. Recommendation & sequencing

**Goal we're committing to:** make the mmap capability *durable and demonstrably crash-safe* first
(the thesis payoff), then *zero-copy real* (the performance/scale payoff) — and only pursue
multi-mapping if a concrete target needs it.

Proposed order:

1. **Durability contract + crash-torture (slice; on the emulation).** Write §4c into this doc as a
   normative contract; add a crash-injection hook to the fs cap (drop un-synced buffer state on a
   "crash" op); ship `demo_lmdb_crash_recovery` proving LMDB recovers to the last committed txn.
   Highest narrative value, self-contained, no FFI. *Recommended first.*
2. **Dedicated `FileMapping` capability (iface 14) with real `MAP_SHARED` aliasing.** Reuse
   `map_region`; the guest shim resolves `filemap` when granted, falls back to the fs-cap emulation
   otherwise. Zero-copy, OS-coherent; carries the durability contract from (1). Linux first;
   macOS/Windows follow `SharedRegion`'s existing per-OS path.
3. **Multi-mapping / mixed pwrite** — only if a target (e.g. LMDB without WRITEMAP, or a second
   reader) is chosen that needs it. Real aliasing from (2) largely provides it.

Rationale for (1) before (2): the durability *contract* is what makes "a database in the sandbox"
mean something; it's backend-independent, so writing it against the emulation is not throwaway — the
dedicated cap must honor the same contract. And crash-recovery is the single most compelling demo of
the capability model ("the guest can't corrupt the host's file even when it dies mid-write").

## 6. Open questions (to resolve before slice 1)

- **Crash granularity**: page-atomic `msync`, or expose torn writes and rely on the DB's checksum?
  (LMDB's meta pages are checksummed → exposing tears is a *stronger* test.)
- **Where does the crash hook live** so it's test-only and never in a shipping grant? (A wrapper
  cap? A `mem_fs`/`host_fs` constructor variant? A feature flag?)
- **Does the dedicated cap subsume the fs cap's file ops**, or stack on top (open via fs, map via
  filemap, sharing the fd)? Fewer caps vs cleaner separation.
- **Window budget**: real aliasing reserves a window sub-range per mapping — who owns that address
  space (the guest via `AddressSpace.sub`, or the cap picks)? Ties into §14.
