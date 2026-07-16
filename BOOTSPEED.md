# Boot speed — cold-start cost of the Postgres demo, measured

The Postgres-in-the-sandbox demo (`crates/svm-run/demos/postgres`, `LLVM.md` slices BM–CO) boots
PostgreSQL 17.5 `--single` to a live `backend>` and runs real queries. For a **browser** demo the
question is start-up latency: how long from page load to a backend ready to take a query. This note
decomposes that cost with measured numbers (not the old "~100 s" folklore, which was a *debug* build
running WAL crash recovery) and states what each speed lever buys.

**Bottom line: it's a plumbing job, not a durability project.** Ship the module *pre-translated* and
*pre-resolved*, plus a *cleanly-shut-down* data image, and cold start is ~2.6 s native / ~4–5 s
in-browser — no snapshot/restore machinery required. Snapshot/restore (freeze/thaw) would take it to
near-instant, but the numbers say you don't need it to ship.

## The cost, decomposed

Module: the whole-program Postgres module — **~15 068 functions, 20 MB `.svmb`**. Native numbers are
release builds on a shared box (treat sub-second values as ±15 %); measured by
`crates/svm-run/examples/prep_svmb.rs` (module prep) and the boot harness (guest run).

| phase | translate at load | ship pre-translated `.svmb` |
|---|--:|--:|
| translate bitcode → SVM-IR | **14 s** (16 KiB pages) / **45 s** (64 KiB browser pages + serialize) | 0 — done at build time |
| decode `.svmb` | — | ~0.45 s |
| resolve capability imports | — | 0 if pre-resolved (else ~0.38 s) |
| **verify** (escape-freedom TCB gate — *never* skippable) | (in the translate path) | ~0.55 s |
| bytecode-compile (interpreter cold cost) | ~0.20 s | ~0.19 s |
| guest boot to `backend>` + a full round-trip (cleanly-shut-down cluster) | ~1.4 s | ~1.4 s |
| **total cold start** | **~16–47 s** | **~2.6 s** |

The dominant cost is **translation** (14–45 s), and pre-translating eliminates it — that's the whole
game. Everything left (decode + verify + compile + boot) is ~2.6 s.

Two caveats baked into the guest-boot number:
- **~1.4 s assumes no WAL recovery.** A freshly `initdb`'d cluster that never shut down cleanly pays
  recovery on first boot; ship a data image that was booted once and shut down cleanly.
- It runs on the **bytecode interpreter** (interpreter-first). A cold eager JIT of all ~15 k functions
  is *slower* to first result, not faster — lazy-JIT only hot query functions if steady-state warrants.

## The wasm tax (the browser reality)

In the browser the SVM itself is compiled to wasm, so both module-prep and guest-execution pay a
sandbox tax. Two measurements pin it down:

**Module prep inside wasm — measured.** `browser/bench_prep.mjs` drives the `svm-browser` cdylib's
`svm_prep_bench` (decode + verify + bytecode-compile) on V8 over the same 20 MB resolved module:

| | decode + verify + compile |
|---|--:|
| native (`prep_svmb`) | ~1.16 s |
| **in wasm on V8** (`bench_prep.mjs`) | **~1.03–1.11 s** |
| **tax** | **~1×** (indistinguishable — V8 JITs the prep work as well as native) |

**Guest execution inside wasm — from the committed cross-engine bench** (`bench/cross-engine`, the
`svm-bytecode` vs `svm-bytecode-wasm` rows — the *same* engine, native vs compiled-to-wasm, on
identical IR):

| workload | wasm/native |
|---|--:|
| pure compute (alu, xorshift, fma) | ~1.2–1.4× |
| dependent-load memory (chase) | ~1.9× |
| cache-missing pointer chase (chase_rand) | ~3.4× |

Compute is barely taxed; only serial pointer-chasing pays real cost. Postgres boot is pointer-heavy, so
blend ~2–2.5×.

**In-browser projection:** module prep ~1.1 s (measured) + guest boot ~1.4 s × ~2–2.5× (extrapolated) ≈
**~4–5 s** from page load to a queried backend. The prep half is measured; the guest-boot half is
extrapolated from the committed kernel taxes — directly measuring it means running the full
fs-cap-in-wasm boot, i.e. building the demo itself (see "What's left").

## The levers, ranked

1. **Ship pre-translated (`.svmb`), not bitcode.** Removes the 14–45 s translate from every load,
   leaving ~2.6 s. Biggest win, lowest effort — build-pipeline plumbing. `svm-llvm-translate … --binary
   --host-page 65536` emits the browser-target module; `prep_svmb in.svmb out.svmb` resolves + verifies
   + re-serializes it so load skips resolve too.
2. **Ship a cleanly-shut-down data image.** No WAL recovery on boot. Trivial (boot once, shut down,
   snapshot the data dir).
3. **Interpreter-first, lazy-JIT.** The interpreter starts instantly and boots in ~1.4 s; never cold-JIT
   the whole module up front.
4. **Snapshot/restore the post-boot state (freeze/thaw) — optional.** Run to `backend>` once, freeze the
   guest, ship the frozen image, thaw to a ready backend — skips prep *and* boot. The measured ~4–5 s
   in-browser start says this is **not required to ship**; defer it unless a demo proves ~4–5 s too slow.
   Its long pole is not the memory image (that snapshots trivially, and `--single` has no background
   workers or cross-process IPC) but **restoring host-side open file descriptors** — the WAL/data-file
   handles held through the `fs` cap — since those live outside the linear-memory snapshot.

## What's left (to validate the projection / build the demo)

- **✅ Milestone A — Postgres boots on a virtual (in-memory) filesystem.** `mem_fs_from_host_dir`
  (`crates/svm-run/src/fs.rs`) seeds an in-memory `fs` cap from a data-dir image; Postgres `--single`
  then runs the full `CREATE TABLE` / `INSERT` / `SELECT` / `ORDER BY` / aggregate round-trip on it and
  exits cleanly (`Exited(0)`) — **zero real filesystem**, the exact requirement of the browser path.
  (Getting there needed three `mem_fs` fixes: consistent path normalization across all file ops, a
  read-only *directory* open so Postgres can `fsync` dirs at checkpoint, and a `0700` data-dir mode.)
  The seed step (~40 MB image) takes ~35 ms; the guest run ~1.2 s natively.
- **✅ Data image — a self-contained, shippable filesystem blob.** `encode_image`/`decode_image` +
  `mem_fs_from_archive` (`crates/svm-run/src/fs.rs`) serialize a cluster into one `SVMFSIM1` byte blob
  that mounts on the `fs` cap with **no host filesystem** — the browser's data half. `build_image`
  (example) produces it from an on-disk cluster (Postgres' 39 MB `initdb` tree → a 41 MB image in ~3 s);
  Postgres `--single` boots from the mounted archive and runs the round-trip (`Exited(0)`). So the
  demo's two artifacts are now both buildable: `{postgres_resolved.svmb, pgdata.img}`.
- **✅ The in-memory `fs` cap is now wasm-reachable.** Extracted the pure protocol + `mem_fs` +
  data-image format into the **`svm-fs`** crate (depends only on `svm-interp`, builds for `wasm32`);
  `svm-run` keeps the real-filesystem `host_fs` + the `HostCap` wrappers and re-exports `svm-fs`, so
  `svm_run::fs::*` is unchanged.
- **Measure the guest boot in wasm directly** (the one number still extrapolated). Now unblocked: a
  `svm-browser` cdylib entry that grants `svm_fs::mem_fs_seeded_handler` (mounting `pgdata.img`) + streams
  stdin/stdout, then time the boot on V8.
- **The loader/page.** The browser loads → `decode → verify → run on the interpreter` (the
  `svm_run`/`svm_prep_bench` shape already in the cdylib) with stdin/stdout streamed and `pgdata.img`
  mounted on the `fs` cap.

## Reproducing the measurements

```
# 1. translate → browser-target .svmb (build-time), then resolve+verify+re-serialize + time each phase:
svm-llvm-translate postgres_shimmed.bc -o postgres.svmb --binary --host-page 65536 --stub-externs
cargo run --release -p svm-run --example prep_svmb -- postgres.svmb postgres_resolved.svmb

# 2. module-prep tax inside wasm (V8):
cd browser && cargo build --release --lib --target wasm32-unknown-unknown
node bench_prep.mjs target/wasm32-unknown-unknown/release/svm_browser.wasm /path/to/postgres_resolved.svmb

# 3. guest-execution tax (committed cross-engine bench, svm-bytecode vs svm-bytecode-wasm):
#    see bench/cross-engine/README.md
```
