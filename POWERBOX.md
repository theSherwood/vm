# Frontend-independent instantiation & embedding

Tracking doc for the work that turns the powerbox bootstrap — today baked into the C on-ramp
(`svm-llvm`) — into a **public, frontend-neutral, wasm-like instantiation layer** with a uniform
run interface across all three backends (tree-walker, bytecode, JIT) and **C bindings** for
embedders who don't want to drive Rust.

Branch: `claude/gifted-hopper-4cj3ll`. Started from PR #92.

---

## Goal

A non-C frontend (e.g. JACL's codegen) that emits SVM-IR and links it itself should get the same
"just works" experience the C frontend enjoys, *plus* the flexibility wasm embedders expect:

- **Arbitrary imports/exports** — any number, with arbitrary names, signatures, and interfaces;
  resolved by **name** (not by fixed position), like wasm's `(module, name)` import matching.
- **Arbitrary host capabilities** — a host can expose any interface with any semantics, reached by
  the guest through the object-capability handle model.
- **Uniform run config across backends** — the consumer sets fuel/deadline, vCPU/fiber quota, and
  memory once, and it binds the tree-walker, the bytecode engine, and the JIT identically.
- **C bindings** — the whole surface (instantiate, bind imports, call exports, set limits, grant
  capabilities) usable from C, since many consumers will be more comfortable there.

---

## Design decisions (locked)

1. **Exports are first-class.** Add `Module.exports: Vec<Export>` (name → funcidx) to the IR, with
   text-grammar + binary-encode + verifier support. Today export names live only in
   `svm_ir::LinkUnit` and are consumed by `link`; the runtime `Module` has no name table, so
   `Instance::call("name")` currently relies on an ad-hoc side map. First-class exports make
   name-addressable entry points a real IR concept (wasm parity).

2. **Binding is by name, dynamic — not positional.** The host provides a *named* set of
   capabilities; `instantiate` matches each `Module.imports[i].name` against it (fail-closed on a
   missing/incompatible name), exactly like wasm import resolution. The fixed `VM_CAP_*` slot order
   (stdout=0, stdin=4, …) becomes just *one convention* layered on the general mechanism, not a
   hard requirement. Handle *delivery* to the guest (the stash) stays an implementation detail the
   frontend and `synth_powerbox_start` agree on; the *binding* the embedder sees is by name.

3. **Host-defined interface type_ids (judgement call): reserve an open range above the builtins.**
   The functional capability already exists — `Host::grant_host_fn` exposes arbitrary semantics
   through `iface::HOST_FN` (13), distinguished by handle + op. Recommendation: keep that as the
   mechanism, but let a host **tag** each registered interface with a nominal id drawn from an open
   range `[HOST_IFACE_BASE, u32::MAX)` (e.g. `HOST_IFACE_BASE = 1 << 16`), carried for diagnostics
   and `cap.self.*` discovery and dispatched through the same generic seam. The fixed builtins
   (0–13) keep their ids. This is a refinement, not a blocker — lowest priority of the three.

---

## What already exists (the hard part is done)

Grounded in the current code, so the plan builds on reality:

- **One generic capability seam, shared by all backends.**
  `Host::cap_dispatch_slots(type_id, op, handle, args, mem)` (`svm-interp/src/lib.rs:9324`) is the
  single dispatch used by the tree-walker, the bytecode engine (`bytecode.rs:3726`), and the JIT
  (via `svm-run`'s `cap_thunk` → same call). Only `Jit` (iface 11) is special-cased (it must
  re-enter Cranelift). Everything else is already uniform.
- **Arbitrary host capabilities.** `Host::grant_host_fn(Box<dyn FnMut(op, &[i64],
  Option<&mut dyn GuestMem>) -> Result<Vec<i64>, Trap>>)` (`svm-interp/src/lib.rs:8920`, type at
  `:8325`): arbitrary op, arbitrary i64 args/results, guest-memory access, return-or-trap. Runs
  identically on interp and JIT through the seam above.
- **Arbitrary named imports.** `Module.imports: Vec<Import{name, sig}>` (unbounded) +
  `Inst::CallImport{import, sig, handle, args}` + `svm_ir::resolve_imports_with` (any name →
  `Resolved::Cap{type_id, op}` / `Func` / `Slot`). The number/name/signature are already free.
- **Runtime discoverability.** `cap.self.count` / `cap.self.get` (`CapSelfCount`/`CapSelfGet`) let a
  guest enumerate its own handles at runtime — an alternative/complement to a fixed-offset stash.
- **Parameterizable memory reservation.** `reserved_log2` is already threadable on the
  `*_reserved` entries (`compile_and_run_capture_reserved_with_host`, interp `run_capture_reserved`)
  — just not on the convenient `run_with_host`, which hardcodes `DEFAULT_RESERVED_LOG2`.

### The genuine asymmetry to design around

**Fuel is not uniform and can't cheaply be.** The tree-walker/bytecode decrement a `&mut u64` per
op; the JIT polls an interrupt `AtomicU64` cell at back-edges (the deadline-watchdog path). So the
uniform "bound this run" knob must be **interrupt/deadline-based** (binds all three), with per-op
`fuel` as an interp-only refinement. Model it honestly:

```
Limits {
    deadline: Option<Duration>,  // binds all backends (interrupt cell / time poll)
    fuel: Option<u64>,           // interp-only refinement; JIT approximates via the interrupt
    max_vcpus: usize,            // = "CPUs available"
    max_fibers: usize,
    reserved_log2: u8,           // fed to BOTH backends — also makes the differential more sound
}
```

Note: the JIT and interp must share `reserved_log2` to remain a sound differential oracle
(`svm-interp/src/lib.rs:1909`), so threading it through one config is strictly better than today.

---

## Proposed shape

`svm_run::Instance` (PR #92) is already a prototype of the uniform facade — it runs interp + JIT
through one call, converting `Value`↔slots. Generalize it:

```rust
enum Backend { TreeWalk, Bytecode, Jit }

struct RunConfig {
    limits: Limits,
    init_mem: Option<Vec<u8>>,
    stdin: Vec<u8>,
    memory_size_log2: Option<u8>,   // override the module's declared window
}

// Host capabilities offered to a module's imports, matched by name (decision #2).
struct Imports { /* name -> capability (grant closure / descriptor) */ }

fn instantiate(module, imports, host) -> Instance     // resolve-by-name + verify
impl Instance {
    fn call(&self, backend, export_name, args, &config) -> Outcome   // by name (decision #1)
    fn run_diff(&self, export_name, args, &config) -> Outcome        // interp == jit oracle
}
```

`Quota` is currently two structurally-identical types (`svm_interp::Quota`, `svm_jit::Quota`) —
unify into one (in `svm-ir` or `svm-interp`) consumed by both backends.

---

## Roadmap / checklist

### Phase 0 — foundations (done)
- [x] `svm_ir::synth_powerbox_start` — frontend-independent `_start` synthesis (PR #92).
- [x] `svm_run::instantiate` / `Instance::call` — the differential wrapper (PR #92).
- [x] Acceptance test: hand-written IR, no C, interp == jit (`crates/svm/tests/powerbox_instantiate.rs`).
- [x] **Dedup the layout constants** — `svm-llvm` now references `svm_ir::POWERBOX_*` with
  compile-time asserts pinning the C `_start` layout to the public ABI (one source of truth).

### Phase 1 — first-class exports (decision #1) — done
- [x] `Module.exports: Vec<Export{name, func}>` in `svm-ir` + `Module::resolve_export(name)`.
- [x] Text grammar (`svm-text`): `export "<name>" <funcidx>`, parse + print + round-trip.
- [x] Binary encode/decode (`svm-encode`): v3 export section (after imports), round-trip.
- [x] Verifier: export funcidx in range (`ExportFuncOutOfRange`), names unique (`DuplicateExport`).
- [x] `link` populates `Module.exports` from each unit's exports (declaration order, reindexed).
- [x] `synth_powerbox_start` registers `"_start"` and shifts existing exports +1 (`offset_func_indices`).
- [x] Producers populate exports: `svm-wasm` (wasm exports → table), `svm-llvm` (defined fns →
      name-addressable C modules), `optimize_module` (carried through; 1:1 per-func).
- [x] `Instance::call(name)` resolves through `Module::resolve_export` (the ad-hoc map is gone).
- [x] Guest-JIT demos updated to emit the v3 blob format (version byte + empty export section).

### Phase 2 — name-based dynamic import binding (decision #2) — done
- [x] `svm_run::Imports` — a name → `HostCap` registry; `HostCap::{stdout,stdin,exit,clock,host_fn,
      custom}` cover built-ins and arbitrary host-defined interfaces (`HOST_FN` + op).
- [x] `instantiate_with_imports(module, imports)` matches each `call.import "<name>"` by name →
      `(type_id, op)`, fail-closed on an unbound name; the fixed §3e powerbox is now one preset over
      the same machinery (`instantiate` stays the canonical-prefix path).
- [x] Generalized `synth_powerbox_start`: the `3..=8` cap is lifted — the stash may hold any N that
      fits the reserved low region (≤ 8 with a seeded heap, ≤ 32 without). Grant order = import order
      = stash slot order (slot `i` ↔ import `i`).
- [x] `Instance::call` routes the powerbox entry through the name-bound registry when present, else
      the fixed powerbox; both share one differential body (`run_entry0_diff`). Acceptance:
      `crates/svm/tests/powerbox_imports.rs` (arbitrary-named host-fn caps, unbound fail-closed,
      standard names as a preset), interp == jit.
- [ ] (Deferred) a runtime name→handle directory capability for true dynamic/dlopen-style lookup —
      compile-time name binding covers wasm parity; revisit if a consumer needs in-guest discovery.
- [ ] (Deferred) full dynamic stash sizing (heap base above an arbitrary-N stash) to lift the
      ≤8-with-heap / ≤32-without cap; not needed until a frontend wants >8 named caps *and* a heap.

### Phase 3 — uniform run config across backends — done
- [x] `Limits` is the single unified quota knob the consumer sets (`max_fibers`/`max_vcpus` =
      "CPUs available", `fuel`, `deadline`); the facade converts to `svm_interp::Quota` /
      `svm_jit::Quota` internally, so the two structurally-identical backend `Quota` types stay an
      impl detail the embedder never touches. (Deduping them into one shared type would churn both
      escape-TCB-adjacent crates for no consumer-visible gain — left as optional cleanup.)
- [x] `Limits` + `RunConfig` (fuel = interpreters' per-op budget, ignored by the JIT; deadline = the
      JIT's §5 watchdog, ignored by the interpreters; `memory_size_log2` overrides the window). The
      asymmetry is modeled honestly and documented per-knob rather than papered over.
- [x] `Backend` selector + `Instance::run(backend, &config)` single-backend facade; `Instance::run_diff`
      for the tree-walk == JIT oracle; `call`/`call_with_stdin` are now `run_diff` with a default/stdin
      config. Acceptance: `crates/svm/tests/powerbox_run.rs` (every backend under one config; fuel
      bounds the interpreters not the JIT; window override on every backend).
- [x] `memory_size_log2` threaded through all three backends (window override per run). `reserved_log2`
      deliberately **not** exposed: it's a guard-region policy, not "memory available", and both
      backends must share it for the differential oracle to stay sound — so it's pinned at the shared
      default rather than made a knob.

### Phase 4 — host-defined nominal interfaces (decision #3 — resolved: NOT doing type_ids)
Decision #3 settled: keep `HOST_FN` + handle as the mechanism. The handle is already an unforgeable,
more-expressive disambiguator than a nominal type_id, and an open host-assignable type_id range would
weaken the §3c type-check's closed-enum audit surface for only a diagnostics gain.
- [ ] (Optional, cosmetic) carry an optional human-readable **interface label** alongside a `HostFn`
      grant — untrusted, verifier-ignored, for diagnostics / `cap.self.*` only. Not a type_id.

### Phase 5 — C bindings — done
- [x] `svm-capi` crate (`rlib` + `cdylib` + `staticlib`) + hand-written `include/svm.h` over the whole
      surface: parse (text/binary), `synth_powerbox_start`, name-keyed imports (built-ins +
      C-callback host caps), `instantiate`/`instantiate_with_imports`, `run`(backend)/`run_diff`,
      `Limits`/`RunConfig` (fuel/deadline/quota/stdin/memory), and outcome + stdout/stderr readback.
- [x] Host-capability callback ABI: a C `(ctx, op, args*, n_args, results*, results_cap) -> i32`
      function pointer bridged to `HostFn` (return = result count, negative = trap). Compute-only this
      slice (no guest-memory arg yet — memory-backed I/O goes through the built-in `Stream` caps);
      a `GuestMem` shim for the callback is the documented follow-up.
- [x] Memory/error model: opaque handles with explicit `*_free`; `instantiate*` consume their inputs;
      status codes + a thread-local `svm_last_error()`; every entry point `catch_unwind`s so a panic
      never crosses into C.
- [x] Tests: a CI-portable Rust ABI test (`src/abi_tests.rs`) drives the `extern "C"` surface end to
      end incl. a function-pointer host callback, all three backends, fuel/memory config, and
      fail-closed paths. A real C program (`examples/hello.c` + `examples/README.md`) links the
      staticlib and runs a guest on the JIT (verified locally: `Hello from C!` + return `42`); it's
      the human-facing linkage proof, kept out of CI to avoid `cc`-link fragility.

---

## Where the interfaces stand (audit, post-Phase-5)

There is **not** one host interface — several overlapping run paths coexist, and they have diverged
rather than converged:

| Path | Backends | Caps | Used by |
|---|---|---|---|
| `Instance::{call,run,run_diff}` (this work) | all 3 | fixed or name-bound | new tests, C ABI |
| `run_powerbox*` → `run_powerbox_inner` | JIT | fixed 8-handle powerbox | **the CLI** (`svm-run` bin) |
| `run_kernel` | bare | none | kernels / bench |
| `run_c_full` (test) | interp+JIT | fixed | chibicc suite |
| `JitSession` | JIT | fixed | guest-JIT REPL |
| §14 nested (`Instantiator`/`Module`) | interp+JIT | object-cap | `instantiator.rs`, `separate_module.rs` |

Two consequences worth fixing (Phase 6 / followups):
- The new `Instance` layer **did not replace** `run_powerbox_inner`; the powerbox-grant logic is now
  **duplicated** (`grant_caps` vs the hardcoded grant in `run_powerbox_inner`), and the CLI still uses
  the old path.
- The §14 nested path is **func-0-only and ignores `Module.exports`** — a child/separate-module guest
  cannot be called by name.

**Test gaps:** we test the export *table* (`resolve_export`) and calling `_start` by name, but **not**
calling a named non-`_start` export with args and checking results (host *or* nested). And
`Instance::call("<name>", args)` for a non-`_start` export currently runs as a **bare kernel with no
capabilities** — so a named export that uses caps can't be called at all today.

---

## Phase 6 — the reactor model: a live instance you call into (spec)

**Problem.** First-class exports (Phase 1) made entry points *addressable*, but you still can't really
*call* them: every `call`/`run`/`run_diff` builds fresh hosts + a fresh window, runs `_start` once, and
discards all state. There is no **live, stateful instance** you invoke repeatedly — the wasm
"reactor"/component model (instantiate once → call exports N times → linear memory, heap, and
capability handles persist between calls). This is the missing half of "calling exports from the host,"
and it subsumes the test gap above. `JitSession` already proves persistence is possible (it keeps a
compiled module + window alive across `run_prompt` calls) — Phase 6 generalizes that into `Instance`,
across backends, with named export calls.

**Model.** Split today's run-once `Instance` into:
- a **command** mode (what exists): run `_start` once, fresh state, return outcome — keep as-is for the
  program use case;
- a **reactor** mode (new): `instantiate` → optionally run an initializer (`_start`/`_initialize`) once
  → then `call_export("name", args) -> results` **repeatedly**, with the window (globals, heap) and the
  granted capability handles **persisting** between calls.

### Proposed API (`svm-run`)

```rust
/// A live, stateful instance: capabilities granted once, window persists across calls.
pub struct Session { /* module, host, window/CompiledModule, granted handle vector, backend */ }

impl Instance {
    /// Start a reactor session on `backend` under `config`: grant the powerbox once, set up the
    /// persistent window, and (if present) run the `_start`/`_initialize` export once.
    pub fn start(&self, backend: Backend, config: &RunConfig) -> Result<Session, String>;
}

impl Session {
    /// Call an exported function by name with `args`, returning its results. The window (heap,
    /// globals) and capability handles persist from prior calls. Caps are reached the normal way
    /// (the export loads handles from the stash that `start` populated once).
    pub fn call_export(&mut self, name: &str, args: &[Value]) -> Result<Vec<Value>, String>;

    /// Captured output so far, and a handle to read/seed the window if needed.
    pub fn stdout(&self) -> &[u8];
    pub fn stderr(&self) -> &[u8];
}
```

### Design questions to resolve before coding
1. **Export calling convention.** A reactor export is `(i64 sp, <args…>) -> <results…>`? It needs the
   data-stack pointer (like `_start` passes `main`) *plus* its own args. Decide: does `call_export`
   synthesize the `sp` (a fresh frame above the persistent globals/heap each call), and append the
   caller's `args` after it? (Likely yes — mirrors how `_start` calls `main(sp)`.)
2. **Handle delivery across calls.** `_start`/`start` stashes handles once at window offsets `i*4`; a
   reactor export reaches caps by loading from the stash (already persistent in the window). So the
   stash *is* the persistence mechanism — no per-call regrant. Confirm the frontend emits exports that
   read handles from the stash, same as `main`.
3. **Backend persistence.** Tree-walker: keep the `Mem` (window) + `Host` alive across calls (today
   `run_capture_reserved_with_host` owns the `Mem` internally — needs an entry that takes a persistent
   `Mem`). JIT: keep the `CompiledModule` + window alive (à la `JitSession`) and call exports via a
   per-export trampoline (the JIT compiles function 0; calling an arbitrary export needs an
   entry-by-index run over the live window — check `CompiledModule` supports invoking a chosen funcidx,
   or extend it). This is the main implementation cost.
4. **Differential.** `call_export` differential (interp vs JIT) must run *both* sessions in lockstep
   across the *sequence* of calls (state diverges if they desync), not per-call in isolation.
5. **Concurrency / durability.** Out of scope for the first slice (single-threaded reactor); note the
   interaction with §12 threads and durability snapshots as later work.

### Acceptance (first slice)
A hand-written module that exports `add(i64 sp, i64 x) -> i64` accumulating into a persistent global,
called via `Session::call_export("add", …)` several times, returns the running total (proving window
state persists), with interp == jit across the call sequence. Plus a C-ABI mirror (`svm_session_*`).

### Scope notes
- Keep `command`-mode `Instance::run`/`run_diff` unchanged (the program use case).
- This is the natural place to also wire `argv`/env (Followup F4) since a reactor's `start` is where an
  initializer would consume them.

---

## Followups / known gaps (logged, not yet scheduled)

- **F1 — converge the host runners.** Fold `run_powerbox_inner` and `run_c_full` onto the `Instance`
  layer (or factor one shared powerbox-grant/run core) so the powerbox-grant logic isn't duplicated and
  the CLI/tests use one interface. Do this alongside or after Phase 6.
- **F2 — name-addressable nested guests.** Thread `Module.exports` through the §14 `Module`-capability
  path so a parent can call a child's export by name, not just func 0 — consistent with host-driven
  calls. (Today nested children are func-0-only.)
- **F3 — test the named-export call path.** Even before Phase 6, add a test for `Instance::call("<non-
  _start>", args)` returning results; decide whether a non-`_start` export should get the powerbox caps
  (it currently gets none — likely wrong once reactors exist).
- **F4 — `argv`/env through `Instance`/`RunConfig`.** `synth_powerbox_start` is `main(void)`-only; the
  older `run_powerbox_with_args` supports the §3e args buffer. Thread it through the new layer.
- **F5 — guest-memory access from C callbacks** (Phase 5 deferral): a bounds-checked `GuestMem` shim so
  a C `HostFn` can read/write the window, not just compute on scalars.
- **F6 — unify `Quota`** into one shared type (Phase 3 deferral): currently `svm_interp::Quota` and
  `svm_jit::Quota` are structurally identical and converted at the facade.
- **F7 — runtime name→handle directory** (Phase 2 deferral): in-guest dlopen-style discovery, if a
  consumer needs it beyond compile-time name binding.
- **F8 — full dynamic stash sizing** (Phase 2 deferral): lift the ≤8-with-heap / ≤32-without cap by
  placing the heap base above an arbitrary-N stash.
- **F9 — cosmetic interface labels** (Phase 4 deferral): an untrusted, verifier-ignored human-readable
  label alongside a `HostFn` grant for diagnostics / `cap.self.*`. Not a nominal type_id.

---

## Open questions

- Phase 2: do we want the **runtime** name→handle directory (true dynamic lookup the *guest* can
  query), or is compile-time name binding at `instantiate` enough? The latter covers wasm parity;
  the former enables dlopen-style discovery. Current lean: compile-time binding first, runtime
  directory as a follow-up if a consumer needs it.
- Phase 5: which C consumers drive the priority — JACL's runtime, or external embedders? That
  decides whether the C ABI leads or trails the Rust facade.
