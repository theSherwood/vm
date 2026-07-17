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

**In-browser — now measured end-to-end.** `browser/bench_pg.mjs` boots the real Postgres module inside
the `svm-browser` cdylib on V8 (mount the data image on the `fs` cap, run to a queried backend): decode
+ verify + compile + full boot + the `CREATE/INSERT/SELECT` round-trip lands at **~6–8 s** (V8-warmup
variance; the guest run dominates). That is higher than the ~4–5 s the kernel-tax extrapolation
projected — Postgres boot is *even more* pointer-chasing-heavy than the `chase_rand` kernel (double
memory indirection: SVM confinement **and** wasm bounds, every catalog/buffer load), so the guest-boot
tax runs ~4–5× rather than the ~2–2.5× blend. Shippable with a spinner; the lever to shrink it is
snapshot/restore of the post-boot state (deferred — see the levers above).

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
- **✅ Postgres boots in wasm — measured.** The `svm-browser` cdylib's `svm_run_pg` entry (decode +
  verify → grant `stdout/stdin/exit/memory` + an `svm_fs::mem_fs_seeded_handler` over `pgdata.img` →
  seed the `--single` argv → reserved-window bytecode run) boots the real database on V8 to a queried
  backend, ~6–8 s (`browser/bench_pg.mjs`). The reserved-memory path works in wasm; the module stays
  import-free (no graphical caps granted).
- **✅ In the playground.** Postgres is a first-class example in the SVM **playground**
  (`browser/web/play.html` / `play.js`, the "PostgreSQL (17.5 — write & run SQL)" example): the editor's
  SQL is fed as stdin, the pre-translated+resolved module + `pgdata.img` are fetched (staged into
  `web/assets/` by `browser/build-pg-assets.mjs` — gitignored, like the Lua/SQLite assets), and
  `svm_run_pg` boots the backend on the **threads** engine the playground already runs, reading the
  output back onto the page. `browser/browser-test.mjs` drives it in real Chromium via Playwright —
  selects the example, clicks Run, asserts the query result — alongside every other playground example
  (the check skips when the artifacts aren't staged). **The demo is done: a real PostgreSQL, in the
  browser, in the playground next to Lua and SQLite, sandboxed.**
- **✅ Persistent interactive session — per-query boot eliminated.** The playground now runs Postgres as
  a *live* backend (`svm_pg_open`/`_query`/`_close`), not a fresh boot per query. The enabling mechanism
  is a **blocking-stdin park** in the interpreter: with `Host::set_stdin_blocking`, a `read` on an
  exhausted stdin buffer suspends the resumable `bytecode::Vcpu` (`VcpuEvent::StdinPark`) instead of
  returning EOF (which makes `--single` exit), so the backend parks at its `backend>` prompt; the host
  pushes the next query and resumes. Measured in Chromium: the **first** query pays the ~9–13 s boot, but
  every query after runs in **~30–120 ms** and **state persists** across them (a `CREATE TABLE` is still
  there for the next `SELECT`) — a real `psql`-style session. This closes the *per-query* boot cost; the
  **one** remaining latency is the first boot, whose lever is snapshot/restore of the post-boot state
  (freeze/thaw — deferred; the session makes it much less pressing, since you pay it once per page).

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

# 4. boot Postgres in wasm end-to-end (mount the data image, run the round-trip, time it):
cargo run --release -p svm-run --example build_image -- /path/to/pgdata pgdata.img
cd browser && cargo build --release --lib --target wasm32-unknown-unknown
node bench_pg.mjs target/wasm32-unknown-unknown/release/svm_browser.wasm \
    /path/to/postgres_resolved.svmb pgdata.img
```
