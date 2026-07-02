# BROWSER.md — the interpreter as a wasm guest (run SVM in the browser)

Tracks the design and implementation of compiling the SVM **bytecode interpreter**
(`crates/svm-interp/src/bytecode.rs`) to a wasm target so an SVM guest can run **inside a browser**
(or any wasm host).

> **Not to be confused with `WASM.md`.** That doc is the *inbound* `svm-wasm` frontend
> (wasm bytes → SVM IR). This doc is the *outbound* direction: the interpreter itself compiled
> *to* wasm. The two never meet — a guest could even be wasm transpiled by `svm-wasm`, then run by
> this interpreter-in-wasm.

It is a living document: update the **Status** and the **Phase tracker** as work lands. Fold into
`DESIGN.md` and drop this file once the work closes (repo convention, cf. `WASM.md`).

---

## Why

The interpreter is a self-contained, `#![forbid(unsafe_code)]`, cooperatively-scheduled execution
engine whose only sandbox is SVM's own masking + guard-page confinement (`svm-mask`/`svm-mem`).
Compiling it to wasm makes it **embeddable in the browser** with zero native dependencies — ship one
`.wasm`, run SVM guests client-side. The payoff is portability/embeddability, **not** added
isolation: inside wasm you are double-sandboxing (the host's wasm sandbox over SVM's own).

## Target: `wasm64`, not `wasm32`

SVM addresses are `u64` end-to-end — `svm-mask` confines into `[0, 1<<reserved_log2)` and every
`svm-mem` offset is a `u64`. `wasm64` (memory64) is therefore the production target: `wasm32` would
force a `u64`→32-bit narrowing at every guest access. `wasm32-unknown-unknown` remains useful as a
quick, flag-free Node-runnable smoke target for **compute-only** guests (no/small guest memory),
where the address width is immaterial.

## Why the bytecode engine (not the tree-walker)

`run`/`run_with_host` (the tree-walker) is the reference **oracle**, scheduled by a production
`Scheduler` that uses real OS threads and wall-clock timers — hostile to wasm. The **bytecode
engine** is reached via `run_fast`/`run_with_host_fast` and is the right target:

- **No platform surface in `bytecode.rs`** — no `std::thread`, no `Instant`, no `page_size`.
- **Its own single-OS-thread cooperative scheduler** — `drive` (`bytecode.rs:2377`) multiplexes
  guest **threads** (`thread.spawn`/`join` + `memory.wait`/`notify`), **fibers** (`step_vcpu`,
  `bytecode.rs:1916`), **coroutines**, and §14 executor children cooperatively on one OS thread over
  one shared `Mem`. A `wait` with no runnable task fires the earliest timeout **deterministically**
  (`bytecode.rs:2471`) — no wall-clock.
- **A ready-made embedder entry** — `compile_and_run_capture_reserved_with_host`
  (`bytecode.rs:1202`) takes an embedder `&mut Host` + init-memory image + reservation, runs, and
  returns a `Capture` (results **and** the final memory snapshot). This is the browser entry point;
  no new public API is required.
- **Clean deps** — `svm-ir`, `svm-mask` (`#![no_std]`), `svm-mem` (non-unix `Paged` fallback),
  `svm-verify` (pure). `page_size` is native-only (wasm hard-codes the 64 KiB page), so it's not in
  the wasm dep graph. No `svm-fiber` (asm stack-switch — fibers here are continuation-based) and no
  `svm-jit` (Cranelift). Nothing architecture-specific is dragged in.

---

## Status

**Viability: PROVEN. Production entry: landed; runtime-validated on wasm32 (Node) and wasm64
(Wasmtime).** All reproduced (not argued):

1. **Compiles to `wasm64`.** Both `svm-interp` and the `svm-browser` entry `cdylib` build clean for
   `wasm64-unknown-unknown` via `-Z build-std`. The `std::thread`/`Instant`/`page_size` references
   *compile* on wasm (they exist as symbols); they are a **runtime** concern only, and the bytecode
   engine's cooperative `drive` never invokes real OS threads — so it sidesteps them. cfg-gating the
   tree-walker `Scheduler` for wasm is **dead-code cleanup, not a correctness blocker**.

2. **The production entry executes correctly in a wasm sandbox (wasm32).** The `browser/`
   (`svm-browser`) `cdylib` exports `svm_run`: the host `svm_alloc`s a buffer, writes an **encoded
   SVM IR module** into it, and calls `svm_run(ptr, len, arg)` which **decodes** it (`svm-encode`),
   runs function 0 on the **bytecode engine** with a **deny-all `Host`**, and returns its `i64`
   result — **fail-closed** (`None` from
   `compile_module` → `STATUS_UNSUPPORTED`, no tree-walker fallback). In Node/V8 with **zero host
   imports**: `svm_run(arg=0) == 0`, `svm_run(arg=1) == 1442695040888963407` (hand-derived anchors,
   exercising loops, i64 wrapping arithmetic, branches, SSA block-arg copies), and garbage bytes →
   `STATUS_DECODE_ERR` (no crash). The embedded `run_guest` smoke probe agrees.

3. **`wasm64` executes correctly at runtime (Wasmtime).** On Wasmtime 45 (`-W memory64=y`), the
   `wasm64` `cdylib` runs the compute probe (`run_guest(1) == 1442695040888963407`, and `0` /
   `-1097658151202642380` for `0` / `1000` — matching native + wasm32), the concurrency probe
   (`run_threads() == 4000` — the cooperative `drive` + `thread.spawn` + atomics on memory64), the
   full encode/decode/execute roundtrip (`run_roundtrip() == 1442695040888963407`, exercising the
   production `svm-encode` decode path `svm_run` depends on), the host powerbox
   (`run_powerbox() == 17` — `Stream.write` + capture and `Exit.exit(42)`), the seed→transform→snapshot
   capture (`run_capture() == 1007`), a confined nested child guest (`run_instantiate() == 42123` —
   `Instantiator.instantiate`/`join` over a sub-window), **and** cooperative continuations
   (`run_fiber() == 107`, `run_coroutine() == 1001329`). So the full stack — compute, concurrency,
   codec, capabilities, memory capture, sub-guest isolation, *and* fibers/coroutines — runs on the
   real production target.
   *Node/V8 22.x cannot yet load it:* Rust's `wasm64` target emits **64-bit tables** (`table64` —
   table limits flag `0x05`, i64 element-segment offsets), and V8 implements memory64 *memory* but
   not 64-bit *tables* (`--v8-options` shows `--experimental-wasm-memory64` on by default, no table64
   flag). A V8 maturity gap, external to SVM — `wasm64` runs today on a table64-capable runtime, and
   the browser path is just compute-only-on-wasm32 (above) until V8 ships table64.

### Reproduce

```sh
rustup toolchain install nightly -c rust-src
rustup target add wasm32-unknown-unknown
cd browser
cargo run --bin genfixture -- alu.svmbc                 # encode the test guest module

# wasm32 — full end-to-end runtime validation (Node, no flags)
cargo build --release --lib --target wasm32-unknown-unknown
node run.mjs target/wasm32-unknown-unknown/release/svm_browser.wasm alu.svmbc

# wasm64 — production target; runtime-validated on Wasmtime (memory64 + table64)
cargo +nightly build -Z build-std=std,panic_abort --release --lib \
  --target wasm64-unknown-unknown
W=target/wasm64-unknown-unknown/release/svm_browser.wasm
wasmtime run --invoke run_guest     -W memory64=y "$W" 1 # 1442695040888963407 (compute)
wasmtime run --invoke run_threads   -W memory64=y "$W"   # 4000 (8 vCPUs, cooperative drive)
wasmtime run --invoke run_roundtrip -W memory64=y "$W"   # 1442695040888963407 (encode→decode→run)
wasmtime run --invoke run_powerbox  -W memory64=y "$W"   # 17 (stream write + capture + exit(42))
wasmtime run --invoke run_capture   -W memory64=y "$W"   # 1007 (seed window → transform → snapshot)
wasmtime run --invoke run_instantiate -W memory64=y "$W" # 42123 (confined nested child + shared backing)
wasmtime run --invoke run_fiber     -W memory64=y "$W"   # 107 (cont.new/resume cooperative fiber)
wasmtime run --invoke run_coroutine -W memory64=y "$W"   # 1001329 (spawn_coroutine/resume/yield)
wasmtime run --invoke run_tailcall  -W memory64=y "$W"   # 120 (return_call tail recursion, O(1) state)
wasmtime run --invoke run_simd      -W memory64=y "$W"   # 42 (i64x2 splat/add/extract_lane)
wasmtime run --invoke run_gcroots  -W memory64=y "$W"   # 2 (gc.roots conservative root scan)
wasmtime run --invoke run_reflect   -W memory64=y "$W"   # 3 (cap.self.count over a 3-cap powerbox)
wasmtime run --invoke run_region    -W memory64=y "$W"   # ...cdef (SharedRegion two-offset alias)
wasmtime run --invoke run_jit       -W memory64=y "$W"   # 142 (guest installs + call_indirects a JIT unit)
wasmtime run --invoke run_dynlink   -W memory64=y "$W"   # 777 (compile_linked resolves a named import)
wasmtime run --invoke run_durable   -W memory64=y "$W"   # 2001 (durable NORMAL run; freeze/thaw differ in corpus)
wasmtime run --invoke run_float     -W memory64=y "$W"   # 4611686018427387904 (sqrt(4.0)=2.0, bit-exact)

# Full differential (14 feature families, 185 cases) vs native ground truth — byte-identical
cargo run --bin gencorpus                                # native ground truth → corpus.json
node corpus.mjs target/wasm32-unknown-unknown/release/svm_browser.wasm   # wasm32 (Node): 185/185
# wasm64 (Wasmtime embedding): the same 185 cases byte-fed through the production target
cargo run --manifest-path wt/Cargo.toml --release -- \
  target/wasm64-unknown-unknown/release/svm_browser.wasm                 # wasm64: 185/185

# Live host imports — guest console/clock bound to real wasm imports (default build is import-free)
cargo build --release --lib --target wasm32-unknown-unknown --features live
node live.mjs target/wasm32-unknown-unknown/release/svm_browser.wasm corpus/live.svmbc
```

`browser/` (`svm-browser`) is a detached `[workspace]` crate (kept out of the main workspace because
it needs `-Z build-std`, like `fuzz/`/`bench/`); build artifacts + the regenerable `*.svmbc` fixture
are git-ignored.

---

## Decisions

- **Fallback policy → fail-closed (v1).** When `compile_module` returns `None` (rare seams:
  instantiate-mixed-with-fibers, multi-fiber durable freeze), the wasm entry returns a clean
  `Unsupported`-style trap rather than dropping to the tree-walker's threaded `Scheduler`. So the
  tree-walker `Scheduler` is purely cfg-gated *out* of wasm — no cooperative-fallback porting.
  (Non-durable guest threads still run on the engine's cooperative `drive`; only *durable* `thread.*`
  is refused, by `compile_and_run_capture_reserved_with_host` itself.)
- **Host capabilities → compute-only first, then a buffer-marshalled powerbox.** `svm_run` still
  supplies a deny-all `Host`. `svm_run_pb` adds a real capability set — **stdin/stdout/stderr
  streams, a monotonic clock, and exit** — granted by entry arity (1 `Stream(Out)` · 2 `Stream(In)` ·
  3 `Exit` · 4 `Stream(Err)` · 5 `Clock`), so `hello.svm`'s `(out, in, exit)` shape works unchanged.
  The `Host` powerbox is already **deterministic and self-contained** (stream writes accumulate in
  `Host::stdout`/`stderr`, `read` draws from `Host::stdin`, `Clock.now` is a strictly-increasing
  counter), so I/O crosses the wasm boundary the *same way the module does* — through host-allocated
  memory (stdin in an `svm_alloc`ation the host passes to `svm_run_pb`; the captured streams returned
  as cdylib-managed allocations read via `svm_stdout_ptr`/`svm_stderr_ptr`/`svm_exit_code`). **The
  default cdylib stays import-free** (verified: `imports: 0`).
- **Memory ABI → `svm_alloc`/`svm_dealloc`, not fixed buffers.** The host reserves linear memory of
  any size for module bytes / stdin (the allocator grows memory as needed), passes `(ptr, len)` to a
  run entry, and frees it after — no 1 MiB scratch cap. Output streams come back as cdylib-managed
  allocations valid until the next run. Demonstrated by a **2 MiB echo** roundtrip in the
  differential. `svm_abi_is64()` tells a host whether the pointer/length ABI is `i32` or `i64`.
- **Live capabilities → a feature-gated variant.** Real host imports are mandatory at instantiation
  for *every* entry, so binding a capability to the live host (`svm_run_live`, bridging guest
  `cap.call`s to `svm_host.host_write`/`host_now_ns` via `grant_host_fn`) lives behind
  `--features live` — the default build stays import-free for the compute/powerbox path, and the
  live build adds exactly the two `svm_host` imports.

---

## Non-portable surface (all in `lib.rs`; bytecode path uses none of it)

Empirically, the linker already handles most of this. The `cdylib`'s exports (`run_guest`,
`run_threads`, `run_roundtrip`, `svm_run`/`svm_run0`) reach only `bytecode::*` + `Host`; none reach
the tree-walker, so `--gc-sections` strips the whole cluster from the `.wasm`. Confirmed on the
built wasm32 binary: **zero** symbols for `Scheduler` / `worker_loop` / `DetSched` /
`available_parallelism`, and **zero** imports (no host, no threads).

1. **Tree-walker production `Scheduler`** (`available_parallelism`, `JoinHandle`/`thread::spawn`,
   `Instant`) and **blocking-offload pool** (`OffloadPool` + `thread::sleep`) — *already absent from
   the binary via dead-code elimination*. Source-level `cfg` would not shrink the artifact; it'd only
   document the wasm boundary, at the cost of entangled surgery (`SchedRef`'s `Real` variant + match
   arms, `fresh_single_root`'s debug-attach use of `Real`). Deferred as not worth the churn.
2. **`page_size` crate** — *done.* `host_page_size()` / `host_region_granularity()` are gated to
   `cfg(not(target_family = "wasm"))`; wasm hard-codes the 64 KiB linear-memory page. The crate is
   now a `[target.'cfg(not(target_family = "wasm"))'.dependencies]` entry, so it is no longer
   compiled into the wasm dependency graph (verified via `cargo tree --target wasm32-...`).

(`svm-mem`/`svm-mask`/`svm-verify` need no work: Paged fallback; `#![no_std]`; pure logic.)

---

## Phase tracker

- [x] **Spike — viability.** wasm64 compile + Node execution of a guest, correctness anchors green.
- [x] **wasm entry crate (`browser/` = `svm-browser`).** A `cdylib` exporting `svm_run` over the
  bytecode engine (decode encoded IR → run → `i64`), deny-all `Host` (compute-only v1), fail-closed
  on `compile_module == None`. Builds for wasm32 **and** wasm64; runtime-validated end-to-end on
  wasm32 in Node (anchors + decode-error path green).
- [x] **wasm64 runtime validation.** On Wasmtime 45 (`-W memory64=y`): the 17 embedded `--invoke`
  probes (`run_guest`/`run_threads`/… → `run_durable`/`run_float`) all correct on memory64/table64.
  Node/V8 still lacks table64 (above) — the browser-via-V8 path stays wasm32 until then.
- [x] **wasm64 byte-feeding differential (`browser/wt/`).** The CLI `--invoke` can't write the
  `svm_alloc` buffers, so a small Wasmtime-embedding harness (`svm-wt`, deps `wasmtime` + `serde_json`,
  no SVM crates) loads the **wasm64** module, `svm_alloc`s + writes each corpus module/stdin/window,
  calls the exports, reads results/streams/snapshots back, and compares to the *same* `corpus.json`.
  **185/185 match on wasm64** — byte-identical to wasm32 and native, including the 128 KiB durability
  snapshots and the 2 MiB echo. So the full differential now runs on **both** targets, not probe-only
  on the production one.
- [x] **cfg-gate `lib.rs` for wasm.** `page_size` is now native-only (`target.'cfg(not(wasm))'`)
  with a 64 KiB wasm fallback in `host_page_size()`/`host_region_granularity()` — dropped from the
  wasm dep graph; native unchanged (full `svm-interp`/`svm-mem` suite green, workspace builds). The
  OS-thread machinery (`Scheduler`/`OffloadPool`/`std::thread`) needed no gating: it's unreachable
  from the `cdylib` exports and already stripped by `--gc-sections` (zero symbols, zero imports in
  the built binary). Source-level gating of it was deferred — pure churn for no artifact change.
- [x] **Differential check (14 feature families, wasm32 + wasm64).** `gencorpus` (host) encodes a
  corpus + computes the **native** result per case; `corpus.mjs` (wasm32, Node) and `browser/wt`
  (wasm64, Wasmtime) run the same modules through the wasm exports and compare. **185/185 match**, zero
  host imports.
  *Compute/concurrency* (37): i64 arith+branches, multi-function `call`, memory store/load,
  divide-by-zero → `STATUS_TRAP`, **and a `thread.spawn` kernel** (8 vCPUs × 500 `atomic.rmw.add` =
  **4000** on the cooperative `drive`). *Powerbox* (5): stdout greeting, stdin→stdout echo,
  monotonic-clock delta, `exit(42)` → `STATUS_EXIT`, and stderr role-routing. *Snapshot* (3): a window
  seeded with 16 i64 words, transformed in place (`+arg`), with the **final memory image** captured
  and compared byte-for-byte. *Nested children* (5): §14 confined sub-guests (shared backing, depth-2,
  attenuated AddressSpace, boundary rejection, trap propagation). Plus fibers/coroutines, tail calls,
  SIMD/v128, gc.roots, reflection, SharedRegion, guest-JIT, dynlink, durability. *Scalar floats* (65):
  f32/f64 `add`/`sub`/`mul`/`div`/`sqrt`/`min`/`max`/`copysign`/conversions/comparisons over **NaN /
  ±inf / ±0 / subnormal / rounding** bit patterns — reinterpreted to i64 bits and compared **exactly**,
  the one numeric corner where a backend could diverge (NaN-payload canonicalization, rounding); it
  doesn't. *Fail-closed* (1): a module the engine rejects → `STATUS_UNSUPPORTED` (the negative path
  beside `STATUS_DECODE_ERR`). *Scale* (1): a **2 MiB** stdin→stdout echo through `svm_alloc`ed
  buffers. `gencorpus` and the wasm entries share the *same* exec helpers (the crate is
  `cdylib`+`rlib`), so the check isolates wasm effects, not logic drift.
- [x] **Scalar floats + fail-closed path.** The one numeric family integer ops can't stand in for:
  f32/f64 `add`/`sub`/`mul`/`div`/`sqrt`/`min`/`max`/`copysign`, `i64↔f64` conversions (incl. saturating
  trunc + f32 demote/promote), and comparisons — each guest reinterprets the i64 arg to f64, computes,
  and reinterprets the result to **i64 bits**, swept over NaN / ±inf / ±0 / subnormal / rounding
  patterns and compared **exactly**. Proves NaN-payload canonicalization and rounding agree across
  native / wasm32 / wasm64 (65 cases). Plus the `unsup` guest — a module the engine rejects →
  `STATUS_UNSUPPORTED`, pinning the fail-closed boundary's second negative path. wasm64
  `run_float() == 4611686018427387904` (bits of `2.0`).
- [x] **Host powerbox (console + clock).** `svm_run_pb` grants streams/clock/exit; the cdylib stays
  import-free; validated on wasm32 (5-case differential above) and wasm64 (`run_powerbox() == 17`).
- [x] **Memory ABI (`svm_alloc`/`svm_dealloc`).** Replaced the fixed 1 MiB scratch buffers: the host
  reserves linear memory of any size for module/stdin and reads captured streams from cdylib-managed
  allocations; `svm_run`/`svm_run0`/`svm_run_pb`/`svm_run_live` all take `(ptr, len)`. Validated by
  the 2 MiB echo (wasm32) and a direct `svm_alloc` call on wasm64. `svm_abi_is64()` exposes the
  pointer width. Follow-up: an `alloc`-returning result struct so multi-value returns avoid statics.
- [x] **Memory-snapshot capture (`svm_run_capture`).** The "host hands in a buffer, the guest
  transforms it in place, the host reads it back" shape: seed the window with `[init_ptr, init_len)`,
  run, and return the **final window image** (via `compile_and_run_capture`) as a cdylib-managed
  allocation read through `svm_snapshot_ptr`/`svm_snapshot_len`. Validated wasm32 (3-case snapshot
  differential, byte-for-byte) and wasm64 (`run_capture() == 1007`). Closes the last output channel —
  return value ✓, streams ✓, **memory image ✓**.
- [x] **§14 nested child guests (`svm_run_nested`).** Function 0 gets an `Instantiator` (iface 6) over
  `[0, 128 KiB)` and `instantiate`/`join`s **confined child domains** over power-of-two sub-windows —
  each a fresh domain, masked to its slice, running on the cooperative executor and joinable through
  the §12 thread machinery. 5-case differential (lifted from `bytecode_instantiate.rs`, all matching
  native): shared-backing data plane (`42123`), depth-2 VM-in-VM (`77`), a two-arg child managing its
  own pages via an attenuated `AddressSpace` (`0`), an out-of-range carve rejected at the boundary
  (`-22`), and a child trap propagating through `join` (`STATUS_TRAP`). wasm64 `run_instantiate() ==
  42123`. So a guest can spin up isolated sub-guests inside the wasm sandbox.
- [x] **§12 fibers + §14 coroutines.** Cooperative continuation switching — the engine's signature.
  *Fibers* (`cont.new`/`cont.resume`/`suspend`, no powerbox → the plain `svm_run0` path): run-to-
  completion (`107`), suspend round-trip (`36`), multi-suspend loop (`19`), and forged-handle / root-
  suspend faults (`STATUS_TRAP`). *Coroutines* (`spawn_coroutine`/`resume` + `Yielder.yield`, on the
  `svm_run_nested` Instantiator path): a 3-resume yield round-trip (`1001329`) and a forged-resume
  fault. All 7 match native; wasm64 `run_fiber() == 107`, `run_coroutine() == 1001329`.
- [x] **Tail calls** (`return_call`/`return_call_indirect`, O(1) window reuse). Plain compute path:
  tail-recursive factorial (sweep) + indirect dispatch through the natural table (incl. out-of-range →
  `IndirectCall` trap). wasm64 `run_tailcall() == 120`.
- [x] **§17 SIMD / v128.** The bytecode engine delegates the v128 long tail to the reference; observed
  via `extract_lane` to fit the i64 slot. `i64x2`/`i32x4` splat+add (→ 2·arg), and a `v128.store`/
  `v128.load` memory round-trip — all swept and matching native. wasm64 `run_simd() == 42`.
- [x] **§GC `gc.roots`** (conservative root enumeration). Capture path: the guest scans its
  activation for in-range words, writes them to a buffer, returns the count; snapshot+count compared
  byte-identically (same engine wasm vs native). `gc_baseline`/`gc_tagged` (tag-masked) → 2 roots each.
  wasm64 `run_gcroots() == 2`.
- [x] **§7 reflection** (`cap.self.count`/`cap.self.get`). Over a fixed 3-cap powerbox (Stream(Out)
  t0, Exit t1, host-fn t13): count → 3, and `get(i)` → (handle, type_id) for i=0..2, out-of-range →
  trap. wasm64 `run_reflect() == 3`.
- [x] **§13 SharedRegion** (host-backed memory aliased into the window). A 64 KiB region mapped at
  two window offsets aliases the same backing — a store through one mapping reads back through the
  other (the magic-ring-buffer primitive); plus `len` (→ 65536). wasm64 `run_region() == 0x0123…cdef`.
- [x] **§22 guest-JIT** (interpreted — no native backend). The guest holds a `Jit` cap (iface 11),
  `install`s a host-compiled unit (`a*b+100`) into its dispatch table and `call_indirect`s it (→ 142);
  `uninstall` then call → freed-slot trap. The **security validator** (`decode_module` → `verify_module`
  → memory-match / no-data / no-concurrency preconditions) is a pure-Rust replica of svm-run's, so it
  runs in wasm with no Cranelift. wasm64 `run_jit() == 142`.
- [x] **§22 dynamic linking** (`compile_linked`). A separately-compiled unit's **named import**
  (`call.import "clock"`) is resolved by a guest-provided symbol table to a host capability (Clock,
  iface 2) *before* verify — lowering it to a real `cap.call 2 0` — so a plugin reaches a host service
  by name → 777; an empty table leaves the import unresolved and `compile_linked` fails closed. The
  symtab codec + resolution run in wasm (own minimal wire form). wasm64 `run_dynlink() == 777`.
- [x] **Durability** (freeze / thaw, single-fiber, IR-driven). The `svm-durable` transform instruments
  a program (two clock reads = unwind points); over a durable window the bytecode engine drives:
  a NORMAL run (→ 2001), an UNWINDING **freeze** (a byte-identical 128 KiB snapshot wasm vs native),
  and a REWINDING **thaw** fed that snapshot back (→ reproduces 2001, ends NORMAL). wasm64
  `run_durable() == 2001`. **✅ Every bytecode-engine feature is now proven in wasm.**
- [x] **Live host imports (`--features live`).** `svm_run_live` bridges guest capabilities to **real
  wasm imports** via `Host::grant_host_fn` (iface 13): a `(console, clock)` powerbox where
  `console.write` forwards the guest's bytes to the imported `svm_host.host_write` (live host console,
  *during* the run) and `clock.now` reads `svm_host.host_now_ns` (real host time). Feature-gated so
  the default artifact stays import-free; the live build declares exactly those two imports (verified
  on wasm32 **and** wasm64). `live.mjs` supplies the imports and asserts the round-trip — the host
  received `"live from wasm!\n"` and the guest returned the host clock value. wasm64 *runtime* of the
  live path needs a Wasmtime embedding to supply imports (the CLI can't); Node/wasm32 is the real
  browser path and passes. Follow-up: an `alloc`/`dealloc` ABI to replace the fixed scratch buffers.
- [x] **Real-browser validation (Chromium via Playwright).** Everything above runs on Node
  `worker_threads`; this proves it runs in an **actual browser**, which Node skips: a tiny COOP/COEP
  server (`serve.mjs`) makes the page **cross-origin isolated** so `SharedArrayBuffer` / shared
  `WebAssembly.Memory` are exposed, and `browser-test.mjs` drives the preinstalled Chromium against a
  page (`web/index.html` + `main.js` + `worker.js`) that (1) runs the **powerbox** (`svm_run_pb` →
  `"hello, powerbox!"`, single-threaded on the page) and (2) runs **one guest's `thread.spawn`ed vCPUs
  across real Web Workers** over the shared memory — the browser twin of `threads-spawn.mjs`: the page
  creates every Worker and never blocks (a browser bans main-thread `Atomics.wait`); each Worker sets
  its own stack/TLS and services `thread.join` via `Atomics.wait` and the futex via
  `Atomics.wait`/`notify`. Verified: **crossOriginIsolated**, powerbox PASS, and the 8-vCPU counter
  kernel → **4000 across 9 Workers** (1 root + 8 spawned), stable across repeats. So the genuinely
  multithreaded SVM-in-wasm runs end-to-end in a real browser, not just Node.
- [x] **Performance — the sandbox tax (cross-engine `svm-bytecode-wasm` row).** Everything above
  proves *correctness*; this measures *cost*. The cross-engine benchmark
  (`crates/svm-llvm/examples/cross_engine.rs`) now times the bytecode engine **compiled to wasm** (the
  `svm_run_bench` export, driven by `browser/bench.mjs` on V8) running the same LLVM-frontend IR as its
  native `svm-bytecode` row — so the ratio *is* the double-sandboxing overhead, and every result is
  cross-checked against native bytecode (a mismatch is a loud `MISCOMPILE`). Indicative: **~1.2–1.4× on
  pure-compute kernels** (V8 JITs the dispatch loop, so the engine's own work is barely taxed) but
  **~1.9× / ~3.4× on the `chase` / `chase_rand` dependent-load kernels** — each guest load pays *both*
  SVM's mask/guard confinement and wasm's linear-memory bounds, and the serial chain can't hide that
  latency. The honest browser-path cost: cheap for compute, real for pointer-chasing. (See
  `bench/cross-engine/README.md` § "SVM-in-wasm".)

## Verification

- **Builds:** the two `cargo build` lines under **Reproduce** (wasm64 via build-std; wasm32 smoke).
- **No semantic drift natively:** the bytecode↔tree-walker exact-equality harnesses
  (`crates/svm/tests/bytecode_diff.rs` + the `bytecode_{caps,fibers,threads,coroutines,instantiate,
  tailcall,debug,durable,dynlink}.rs` suite) must stay green after the cfg-gating — proving the port
  didn't disturb engine semantics.
- **Runs in a wasm host:** `node browser/run.mjs` (smoke), `node browser/corpus.mjs` (the 187/187
  differential vs native on wasm32), `browser/wt` (wasm64 via a Wasmtime embedding), and
  `node browser/live.mjs` (host-import demo, `--features live`). The 16 embedded `--invoke` probes under
  **Reproduce** spot-check each feature on wasm64 directly.
- **Runs in a real browser:** `node browser/browser-test.mjs` (Chromium via Playwright) — cross-origin
  isolated, the powerbox prints `"hello, powerbox!"`, and one guest's vCPUs run across real Web Workers
  → 4000. (Build the threads module + `gencorpus` first; see the header of `browser-test.mjs`.)
- **Confinement intact:** `svm-mask` property/fuzz tests compile and pass unchanged.
