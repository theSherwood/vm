# In-window dynamic linking (`DYNLINK.md`)

Handoff + tracker for **dynamic linking in SVM**: loading separately-authored/compiled code units
and resolving cross-unit references (functions, data) by **name** — the foundation for plugins,
**dynamic class loading in GC'd-language runtimes**, a stateful REPL, and shared runtime libraries.

Branch: **`claude/dynamic-linking`** (PR #11). All work is `svm-ir` + tests; no TCB-backend changes.
Fold the settled parts into `DESIGN.md` and drop this file once the loader lands (repo convention,
cf. the former `SCHEDULING.md`/`AUDIT.md`).

---

## The mental model (read this first)

**Linking is a source-to-source rewrite, above the TCB.** A unit carries *named placeholders*
(`call.import "f"`, a data reference, a slot reference); the linker resolves each name against a
**symbol table** and rewrites the placeholder into a concrete instruction (`call`, `call_indirect`, a
constant address). By the time the verifier and both backends see the module it is an ordinary
**closed** module — indistinguishable from any frontend's output, and **re-verified** like it. There
is no runtime "linker" and no new IR execution semantics; a mis-link (wrong signature, missing
symbol) is caught by re-verification, never trusted.

Two flavors, both implemented:
- **Static** (`svm_ir::link`): merge separate units into **one module** before compilation — function
  symbols → **direct `call`**, data symbols → **constant addresses** in one shared window. Like `ld`.
- **Dynamic** (`Resolved::Slot` + the guest-JIT): a **separately-compiled** unit reaches a function it
  doesn't share an index space with, through the **shared `call_indirect` table** at a runtime-assigned
  slot. Like `dlopen`. Both directions work: plugin→host and (old/loaded code)→newly-installed.

---

## What's built (all on `origin/main..claude/dynamic-linking`, full workspace green)

### svm-ir — the linking primitives (`crates/svm-ir/src/lib.rs`)

- **`Resolved`** — what an import name binds to:
  - `Cap(ResolvedCap)` → lower to `cap.call` (the §7 capability case; pre-existing).
  - `Func(FuncIdx)` → lower to a **direct `call`** (static link, same merged module).
  - `Slot(u32)` → lower to **`call_indirect <slot>`** (dynamic link, shared table).
- **`resolve_imports`** (cap-only, unchanged signature — delegates) and **`resolve_imports_with`**
  (the general pass). Each `CallImport` rewrites **1:1** (no value renumbering):
  - `Cap`/`Func` are trivial in-place swaps.
  - `Slot` patches the import's **handle operand** — which must be a `ConstI32` placeholder the
    frontend emits for a dynamic import — to the slot value, then reuses it as the `call_indirect`
    index. To find that const, the pass builds a per-block **value→defining-instruction map**
    (`def_of`, via `Inst::result_count`). A non-const handle → `ImportError::SlotHandleNotConst`.
- **`link(units: &[LinkUnit]) -> Result<Module, LinkError>`** — the static linker:
  - assigns each unit a function base (cumulative func count) and a 16-byte-aligned **data base**;
  - builds function + data **symbol tables** from all units' exports (duplicate / cross-namespace
    collision → `DuplicateSymbol`);
  - per unit: relocates its data segments by its data base; applies its **`relocations`** (patch
    address consts); `offset_func_indices` (shift the 4 static `FuncIdx` sites — `call`, `ref.func`,
    `thread.spawn`, `return_call` — by the func base); `resolve_imports_with(Func)`; appends.
- **`LinkUnit { module, exports, data_exports, relocations }`** (derives `Default`).
  - `exports: Vec<(String, FuncIdx)>` — function symbols.
  - `data_exports: Vec<(String, u64)>` — data symbols → byte offset within the unit's data.
  - `relocations: Vec<DataReloc>`.
- **`DataReloc { func, block, inst, kind }`** + **`RelocKind { SelfData, DataSymbol(String) }`** — a
  relocation patches the `ConstI64`/`ConstI32` at `(func, block, inst)` by **adding** a base
  (`SelfData` = this unit's data base; `DataSymbol(name)` = a cross-unit data import's address), with
  the const's current value as the **addend** (so `&g + 4` works). No new IR instruction, no value
  renumbering — a relocation is just a constant edit. A reloc at a missing/non-const inst → `BadReloc`.
- **`ImportError`**: `Unresolved(String)`, `BadImportIndex(u32)`, `SlotHandleNotConst`.
- **`LinkError`**: `DuplicateSymbol`, `BadExport`, `Unresolved`, `BadImportIndex`, `BadReloc`.

### Tests

- **`crates/svm/tests/dynlink.rs`** (IR-level, interp==JIT differential) — 13 tests:
  - M0: `caller_links_to_add_by_name`, `unresolved_symbol_fails_closed`,
    `signature_mismatch_is_caught_by_reverify`, `capability_resolution_still_works`.
  - M1: `links_two_separate_units_into_one_program`, `links_across_a_reindexing_offset`,
    `link_unresolved_symbol_fails_closed`, `link_duplicate_symbol_fails_closed`.
  - M2: `cross_unit_data_symbol_is_relocated`, `self_data_is_relocated`, `bad_relocation_fails_closed`.
  - M3: `import_resolves_to_a_call_indirect_slot`, `slot_import_requires_a_const_handle`.
- **`crates/svm/tests/dynlink_runtime.rs`** (end-to-end runtime, on the guest-JIT) — 2 tests:
  - `plugin_calls_host_program_by_resolved_slot` — a plugin compiled at runtime
    (`CompiledModule::define_extra`) calls the host program's `F` through the shared table,
    resolved by name (`Resolved::Slot(0)`). The plugin→host direction.
  - `loaded_client_links_to_a_newly_installed_service_by_name` — a client links to a **newly
    `install`ed** service by name (resolve → install slot). The old/loaded→new direction with symbol
    resolution.

### Pre-existing (in `main`) — the guest-JIT install *mechanism* this builds on

These already prove old code reaching newly-loaded code; the dynlink work adds the **symbol layer**:
- `crates/svm/tests/jit_incremental.rs` — `CompiledModule::define_extra` (compile a unit at runtime
  against the parent's table) and `install_makes_unit_call_indirectable` (`Jit.install` → a reserved
  table slot; parent `call_indirect`s it; un-installed slots trap). §22 Models A/B2.
- `crates/svm/tests/jit_cap.rs` — the guest-driven `Jit` capability (`cap.call 11 …`: compile op 0,
  install op 3) + `call_indirect[slot]`.
- `crates/svm-run/demos/jit/jit_demo.c` — a C demo: `__vm_jit_compile` + `__vm_jit_install` + a C
  function pointer (Model B2, old→new). chibicc builtins `__vm_jit_*` in `frontend/chibicc/codegen_ir.c`.

---

## Roadmap

- [x] **M0** — function symbol → direct `call` (`resolve_imports_with` + `Resolved::Func`).
- [x] **M1** — static linker: concatenate separate units → one program (`link`, FuncIdx reindex).
- [x] **M2** — data symbols + per-unit data relocation (`DataReloc`, `RelocKind`, data symbol table).
- [x] **M3** — dynamic: symbol → table slot → `call_indirect` (`Resolved::Slot`), both runtime demos.
- [ ] **The capstone: a guest-side `dlopen`/`dlsym` loader** — see below.
- [ ] GOT / late-binding variant (so *old* code calls *not-yet-loaded* code by name without recompiling
      the caller: the caller does `call_indirect (load GOT[i])`; the loader writes the resolved slot
      into the GOT at load — pure data writes via `Memory` + `memory.init`, reusing M2's reloc). The
      current `Slot` lowering bakes the slot at link time, so it needs the dependency installed first.
- [ ] Merge per-unit `debug_info` into the linked module (today `link` drops it: `debug_info: None`).
- [ ] Frontend support: have chibicc/svm-wasm emit `call.import` for *defined-elsewhere* symbols +
      the `data_exports`/`relocations` metadata, so real separately-compiled C/Rust units link (today
      tests hand-build `LinkUnit`s).

---

## Next step in detail — the guest-side loader (`vm_dlopen`/`vm_dlsym`)

The remaining capstone: package the primitives as **guest code** so the loader runs *in* the sandbox
(the runtime tests are `vm_dlopen` done by hand in the harness). Guest-facing surface, over the `Jit`
+ `Memory` caps:

```c
void *vm_dlopen(const void *ir_bytes, long len);  // resolve → place data → compile → install
void *vm_dlsym(void *handle, const char *name);   // → a funcref slot (callable) or a data address
int   vm_dlclose(void *handle);                   // Jit.uninstall + free the data region
```

`vm_dlopen` composes existing pieces: resolve imports against the loader's **symbol table** (guest
data: `name → slot | address`) with `resolve_imports_with` (`Slot`/`Data`); place data via `Memory`
+ `memory.init` applying `DataReloc`s; `Jit.compile` (re-verifies + compiles into the same window);
`Jit.install` each export → slots; record new exports. `vm_dlsym` is a symbol-table lookup.

**Design question to settle:** where does the import-resolution rewrite run — guest-side (the guest
reimplements/loads a rewrite, then submits a *closed* blob to `Jit.compile`) or host-assisted
(`Jit.compile` takes a guest-provided symbol table and resolves before verify)? Host-assisted is
simpler and still safe (rewrite-then-verify); the symbol table stays guest-controlled. Recommend
starting host-assisted.

**Why SVM's `dlopen` is *better* than POSIX** (the selling point): the "shared object" is serialized
SVM IR; `dlsym` returns an **unforgeable funcref slot** (§3c-checked at the call), not a raw pointer;
**everything loaded is re-verified** (a malicious object can't escape — worst case it corrupts its own
window, a §1 non-goal); and **loading is capability-gated** (you need the `Jit` cap, and the bytes
arrive through the powerbox — no ambient "load any file"). Safe dynamic loading.

---

## How to pick up

- Primitives live in `crates/svm-ir/src/lib.rs` (`resolve_imports_with`, `link`, `Resolved`,
  `LinkUnit`, `DataReloc`). Read `dynlink.rs` for IR-level usage, `dynlink_runtime.rs` for the
  guest-JIT (`CompiledModule::compile`/`define_extra`/`install`/`run_extra`) usage.
- Run: `cargo test -p svm --test dynlink --test dynlink_runtime`. Full gate:
  `cargo test --workspace && cargo clippy --workspace --all-targets -- -D warnings && cargo fmt --all --check`.
- Text-IR string escapes are `\xHH` (not WAT's `\HH`) — bit us in the M2 data tests.
- `svm-llvm` is workspace-**excluded** (links libLLVM); `cargo build --workspace` skips it, so changes
  to `svm_ir::Module`'s shape must also update `crates/svm-llvm/src/lib.rs` (it has its own CI job).
