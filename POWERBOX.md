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

### Phase 1 — first-class exports (decision #1)
- [ ] `Module.exports: Vec<Export{name, funcidx}>` in `svm-ir`.
- [ ] Text grammar (`svm-text`) + binary encode/decode (`svm-encode`).
- [ ] Verifier: export funcidx in range, names unique.
- [ ] `link` populates `Module.exports` from `LinkUnit` exports.
- [ ] `Instance::call(name)` resolves through `Module.exports` (drop the ad-hoc map).

### Phase 2 — name-based dynamic import binding (decision #2)
- [ ] An `Imports`/host-capability registry keyed by name (grant closures or descriptors).
- [ ] `instantiate` matches `Module.imports[i].name` → registry entry; fail-closed on miss/sig-mismatch.
- [ ] Generalize `synth_powerbox_start`: stash an arbitrary N handles (lift the `3..=8` cap), in the
      bound order; the fixed powerbox becomes a preset over this.
- [ ] (Optional) a runtime name→handle directory capability for true dynamic/dlopen-style lookup.

### Phase 3 — uniform run config across backends
- [ ] Unify `Quota` into one type.
- [ ] `Limits` + `RunConfig` (deadline binds all three; fuel interp-only; reserved_log2 threaded).
- [ ] `Backend` selector + `run(backend, …)` facade; `run_diff` for the interp==jit oracle.
- [ ] Thread `memory_size_log2` / `reserved_log2` overrides through all three backends.

### Phase 4 — host-defined nominal interfaces (decision #3, lowest priority)
- [ ] Reserve `[HOST_IFACE_BASE, u32::MAX)` for host-assigned interface type_ids.
- [ ] Carry the nominal id for diagnostics + `cap.self.*`; dispatch through the generic HostFn seam.

### Phase 5 — C bindings
- [ ] A C ABI crate (`svm-capi` / `cdylib` + generated `svm.h`) over the embedding surface:
      module load/parse, `instantiate`, bind imports (by name, via C function pointers as host
      capabilities), `call` exports, set `Limits`, read back stdout/stderr/outcome.
- [ ] Host-capability callback ABI: a C function pointer `(op, args*, n_args, results*, n_results,
      mem*, trap_out)` bridged to `HostFn` (mirror the existing `cap_thunk` C signature).
- [ ] Memory ownership + error model (out-params + status codes; no unwinding across the boundary).
- [ ] A C smoke test mirroring the Rust acceptance test (imports `write`, exports an entry).

---

## Open questions

- Phase 2: do we want the **runtime** name→handle directory (true dynamic lookup the *guest* can
  query), or is compile-time name binding at `instantiate` enough? The latter covers wasm parity;
  the former enables dlopen-style discovery. Current lean: compile-time binding first, runtime
  directory as a follow-up if a consumer needs it.
- Phase 5: which C consumers drive the priority — JACL's runtime, or external embedders? That
  decides whether the C ABI leads or trails the Rust facade.
