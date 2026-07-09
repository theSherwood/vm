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

The key architectural fact — and the whole bridge (§4b) — is that `map_region` (`svm-run`) already
aliases **any `os_fd`** into the window via `mmap(MAP_SHARED | MAP_FIXED, fd, …)`; today that `os_fd`
is a memfd, but **a real file's fd is an equally valid `os_fd` for `MAP_SHARED`.** So
`mmap(win_off, len, MAP_SHARED|MAP_FIXED, file_fd, file_off)` *is* zero-copy, OS-coherent,
file-into-window mapping — performed by code that is **already blessed** for the escape-TCB. A
first-class file mmap is therefore not a new escape-TCB primitive; it is the existing `SharedRegion`
aliasing with a file-backed `os_fd`. The emulation exists only because the fs cap is a `HostFn`
(no window-mapping authority) — but the fs `HostFn` doesn't need that authority: it only needs to
**mint the backing** and let the built-in `SharedRegion` do the mapping (§4b).

## 4. Design axes

### 4a. Backing: emulation vs real file-into-window aliasing

| | copy-in/flush (BL) | real `MAP_SHARED` of the file fd |
|---|---|---|
| zero-copy | no | **yes** |
| coherence | manual, WRITEMAP-only | **OS page cache**, any mode |
| large DB | copies whole geometry | maps lazily (demand paging) |
| window cost | a heap buffer per map | a window sub-range reservation |
| portability | pure Rust, all OSes | needs the `map_region` FFI path (Linux done; macOS/Win follow) |
| escape-TCB | none (HostFn) | reuses `SharedRegion`'s already-audited `MAP_FIXED` — no new TCB code |

Real aliasing is the right long-term answer and reuses `map_region` unchanged. It doesn't require a
new window-mapping *capability*: the fs `HostFn` mints the backing and the built-in `SharedRegion`
does the aliasing (§4b). The only inherited cost is `SharedRegion`'s per-OS story (Linux done;
macOS/Windows follow).

### 4b. The bridge: fs mints a `SharedRegion` backing; the built-in machinery does the aliasing

The tension is real and is the security boundary working: a `HostFn` closure runs **outside the
escape-TCB** — it reads/writes the window through a masked `GuestMem`, but it has **no authority to
change window page mappings**, because `MAP_FIXED`-ing host memory into the window *is* the escape
surface, reserved for the built-in iface match. Zero-copy mmap needs exactly that `MAP_FIXED`. So the
fs `HostFn` can never *itself* alias a file into the window. The resolution is not to promote fs into
the TCB, but to **split the roles**:

- **The fs `HostFn` stays thin and outside the TCB.** On an mmap-open it opens the real file (authority
  the embedder already granted it) and asks the host to **mint a `SharedRegion` backing over that
  file's fd**, returning the region handle to the guest. It never maps anything itself. Minting a
  *host-granted* region is already supported (DESIGN §13 — only *guest*-minted regions, `create`/
  `grant`, are the §14 follow-up), so this is a small, existing-shaped new power for the closure.
- **The built-in `SharedRegion.map` (iface 4) does the window aliasing** — real `MAP_SHARED|MAP_FIXED`
  of the file fd, zero-copy, hardware-coherent, over **unchanged, already-audited** escape-TCB code
  (`map_region`). The only new thing it sees is a backing whose `os_fd` is a file, not a memfd.
- **The guest shim** ties them: `open` via fs → receive a region handle → `SharedRegion.map(win_off,
  0, len)` into a window sub-range it owns → read/write real file pages directly; `msync`/`fsync`
  become flush ops. It falls back to the fs-cap **emulation** when only a `HostFn` fs cap was granted.

Why this is the chosen shape over a dedicated `FileMapping` iface (14):

- **Zero new escape-TCB code.** The one blessed operation — `MAP_FIXED` of an `os_fd` into the window
  — already exists; we only broaden its backing to a file fd. A dedicated iface would re-implement
  the same mapping under a new type_id.
- **No duplication of file I/O.** LMDB `pread`s the header before mapping and (non-WRITEMAP) `pwrite`s;
  a self-contained `FileMapping` would have to carry `pread`/`pwrite`/`ftruncate` too, duplicating fs.
  The bridge keeps *file I/O in fs* and *only the aliasing in `SharedRegion`*, sharing one host `File`.
- **fs stays a `HostFn`** — its semantics remain in the embedder's closure, outside the VM's match.
- **Attenuation falls out**: the region handle *is* the authority to that one file region — the same
  model as memfd sharing.

**Security check.** A `MAP_SHARED` mapping of a granted file into the window has the **same escape
surface as the existing memfd `SharedRegion`**: the guest can read/write exactly the file region the
embedder handed over — the granted authority, nothing more. No new escape class; this wants a review
note when it lands, not new machinery.

**The one genuinely new question** (see §6, §4e, and §5 step 2): the region maps into a **window
sub-range the guest must own** — via `AddressSpace.sub` or a powerbox-reserved region. That coupling
to §14 is shared with `SharedRegion` today and is the only unsettled mechanism in the bridge.

The emulation and the bridge are **not** mutually exclusive: the fs-cap emulation stays valid as the
portable/hermetic `mem_fs` path (and for hosts without the `map_region` FFI); the bridge is the
zero-copy real-file path. The guest shim picks by which capability the embedder granted.

### 4c. Durability contract (normative — **shipped in slice 1**)

The persistence model, backend-independent (holds on the emulation today and must hold on the bridge):

- **A durability barrier is `msync(range)` or `sync(fd)`.** After a barrier *completes*, every write it
  covers is durable (survives a crash). Writes **not** yet covered by a completed barrier may be lost.
- **Barriers are ordered and not reordered.** A write made durable by an earlier barrier stays durable
  regardless of what a later barrier does — this is what lets a DB sequence "data pages, *then* the
  meta page that references them" (LMDB's double-buffered-meta commit).
- **A crash loses exactly the un-barriered tail.** On power loss, the backing file equals its bytes as
  of the last *completed* barrier; everything after is gone. Post-crash writes never reach the file.
- **Torn writes are exposed, not hidden.** The contract does **not** promise page-atomic barriers; a
  crash *during* a barrier drops that barrier wholesale (the emulation's model) — and a store that
  wants safety past that must checksum its own critical pages (LMDB checksums its meta pages, so a
  torn meta is rejected and the reader falls back to the previous good one). This is the *stronger*
  test posture — the DB's own integrity machinery is what we verify, not a crutch under it.

This is the actual intellectual content of "a database in the sandbox," and writing it against the
emulation is not throwaway: the bridge (§4b) must honor the *same* contract.

### 4d. Crash injection & recovery proof (**shipped in slice 1**)

Implemented as a **test-only** crash hook: the `*_crashy` fs backends (`mem_fs_crashy`/`host_fs_crashy`,
`crates/svm-run/src/fs.rs`) add op `FS_CRASH_ARM(n)` — arm a simulated power loss after `n` further
durability barriers. When it trips, every persisting op (`msync`/`sync`/`munmap` flush/`write`/
`truncate`) silently drops its effect (the file freezes at the last completed barrier) while reads keep
working (a dead process's file is still readable on reopen). The default `mem_fs`/`host_fs` grants have
**no** crash controller, so `FS_CRASH_ARM` is an unknown op (`-EINVAL`) there — the hook cannot exist
on a shipping grant. (Resolves open questions §6.1 *expose torn writes* and §6.2 *crashy constructor
variant*.)

The proof is `demo_lmdb_crash_recovery` (`crates/svm-llvm/tests/translate.rs`): the guest commits
snapshot **v1** durably, arms the crash, commits snapshot **v2** (same keys, different values) whose
durability the crash may swallow, then reopens and prints the surviving scan. Sweeping the crash point
across *every* barrier of v2's commit, the recovered state must byte-match either the committed v1 or
the committed v2 at every point — **never a torn mix** (transaction atomicity under power loss) — and
the sweep asserts both outcomes actually occur, so the coverage isn't vacuous. This runs entirely on
the emulation, **independent of the bridge (§4b)**.

### 4e. Multi-mapping coherence

Real `MAP_SHARED` gives this for free (OS page cache). The emulation does not. Two sub-cases:

- **One file mapped at two window offsets** — the **magic-ring-buffer** primitive `SharedRegion` was
  built for: alias the same pages at adjacent offsets so a span running off the end of a ring
  continues at the start, and a single `memcpy` handles the wrap. ✅ **Proven** (slice 3):
  `demo_ring_buffer_magic_mapping_vs_native` has a guest double-map one file-backed region (two
  `SharedRegion.map`s via `__vm_region_call`) and byte-match a native `memfd` double-mapped with raw
  `mmap(MAP_SHARED|MAP_FIXED)`, wrap-alias included — on **both backends** (`native == interp == jit`):
  the JIT does the *real* hardware double-mapping over its window (`MprotectWindow::map_region`), the
  interpreter models it in software. (The `Mem`-level aliasing was already unit-tested; this proves a
  real guest program *uses* it end-to-end.)
- **Mixing map-reads with `pwrite`** — e.g. LMDB *without* WRITEMAP. Not yet exercised; low priority
  unless a target demands it. The bridge's real aliasing (§4a/§4b) makes it coherent for free (one
  page cache), so it would be a *proof*, not new plumbing.

## 5. Recommendation & sequencing

**Goal we're committing to:** make the mmap capability *durable and demonstrably crash-safe* first
(the thesis payoff), then *zero-copy real* (the performance/scale payoff) — and only pursue
multi-mapping if a concrete target needs it.

Proposed order:

1. **Durability contract + crash-torture (on the emulation).** ✅ **Shipped.** §4c is now normative;
   the `*_crashy` fs backends add `FS_CRASH_ARM` (§4d); `demo_lmdb_crash_recovery` sweeps the crash
   point across a transaction's commit and proves LMDB recovers to the last committed snapshot at
   every barrier — never a torn mix. Self-contained, no FFI.
2. **Zero-copy real aliasing via the bridge (§4b).** ✅ **Shipped.** Concretely:
   - `FileBacking` (`svm-run`) — a `SharedRegion` backing whose `os_fd` is a real host file; the
     interpreter/JIT aliasing paths are backing-agnostic, so it aliases the file into a window with
     **no new escape-TCB code** (`map_region` reused unchanged). `new_file_region(file, len)`.
   - `RegionMinter` + `HostFnRegion` (`svm-interp`) — the delivery mechanism: the narrow authority
     (mint a `SharedRegion`, nothing else of the `Host`) handed to an opt-in mmap-capable fs handler,
     so the closure can return a region handle. Resolves the "how does the fs `HostFn` deliver the
     handle" question below.
   - `host_fs_mmap` + `FS_MAP_REGION` (op 13) — mints a file-backed region over an open fd (dup'd, so
     the map and the fs cap's `pread`/`pwrite`/`fsync` share one file/page-cache) and returns its
     handle. The guest shim `SharedRegion.map`s it (via a new `__vm_region_call` intrinsic) over a
     page-aligned window buffer, **falling back to the copy-in emulation** when granted a plain `fs`.
   - Proof: `demo_lmdb_mmap_zerocopy_vs_native` — LMDB reads/writes its B-tree straight out of the
     file alias and produces a native-readable `data.mdb` (both directions), plus unit tests for the
     backing, the ABI, and the op.
3. **Multi-mapping / mixed pwrite** — only if a target (e.g. LMDB without WRITEMAP, or a second
   reader) is chosen that needs it. The bridge's real aliasing from (2) largely provides it.

Rationale for (1) before (2): the durability *contract* is what makes "a database in the sandbox"
mean something; it's backend-independent, so writing it against the emulation is not throwaway — the
bridge must honor the same contract. And crash-recovery is the single most compelling demo of the
capability model ("the guest can't corrupt the host's file even when it dies mid-write").

## 6. Open questions

Resolved in slice 1 (§4c/§4d): **crash granularity** — we expose torn writes (a crash drops the
whole in-flight barrier) and rely on the store's own checksums, the stronger posture; **where the
crash hook lives** — a `*_crashy` constructor variant, so the default grants have no such op.

Resolved in slice 2 (§4b, §5.2):

- **How the fs `HostFn` hands the region handle to the guest** — a `RegionMinter` threaded into an
  opt-in `HostFnRegion` handler (`grant_host_fn_region`); the handler returns the minted handle as
  the `FS_MAP_REGION` result. The escape hatch widens by exactly region-minting, only for opt-in
  handlers.
- **Window budget** — the guest reserves the window sub-range itself: the shim page-aligns an
  ordinary heap allocation and `SharedRegion.map`s the file over it (the same place the copy-in
  emulation got its buffer). No new §14 mechanism needed for the single-mapping case.
- **`FileBacking` lifetime** — the backing owns its **own** `dup`'d fd over the file, independent of
  the guest's fd, so it outlives the mapping and both share one OS file / page cache; it is dropped
  with the `Host` at run end (regions aren't reclaimed mid-run — matching `SharedRegion` today).

Still open (deferred, not blocking): reclaiming a region mid-run (today it lives until run end, a
minor leak on repeated map/unmap); a non-zero file offset in `FS_MAP_REGION` (v1 requires offset 0,
which is all LMDB needs); and the macOS/Windows `map_region` backends (Linux-only, following
`SharedRegion`'s existing per-OS story).

Bridge questions resolved earlier (§4b): the capability shape (fs mints a `SharedRegion` backing, not
a dedicated `FileMapping` iface — no fs duplication, no new escape-TCB code) and the escape-surface
review (a file-backed `MAP_SHARED` region is the same surface as a memfd one).
