# In-window dynamic linking (`DYNLINK.md`)

Handoff + tracker for **dynamic linking in SVM**: loading separately-authored/compiled code units
and resolving cross-unit references (functions, data) by **name** — the foundation for plugins,
**dynamic class loading in GC'd-language runtimes**, a stateful REPL, and shared runtime libraries.

Status: M0–M3 **merged to `main`** (via PR #11, branch `claude/dynamic-linking`) — those are `svm-ir`
+ tests, no TCB-backend changes. **C1 (host-assisted resolve)** then extended the binary codec
(`svm-encode`, which *is* untrusted-input TCB — re-verification still gates it) and added the host
resolving-compile primitive in `svm-run`. Fold the settled parts into `DESIGN.md` and drop this file
once the loader lands (repo convention, cf. the former `SCHEDULING.md`/`AUDIT.md`).

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

## What's built (in `main`, full workspace green)

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
- [x] **C1 (capstone, host-assisted resolve)** — the wire format carries **unresolved** imports + the
      host resolves them before verify. Two parts:
  - **svm-encode v2**: the binary form now round-trips the §7 import section (name + op sig) and the
    `call.import` opcode (`0x7C`), so a separately-compiled unit can be **serialized with its symbols
    still unresolved** — a `.so` with undefined references (was: `encode` `unreachable!`d on imports,
    `decode` always returned `imports: []`). `VERSION` 1→2; the three guest-side C IR emitters
    (`demos/jit/*.c`) emit v2 (a `0` import-count byte; they ship self-contained units).
  - **`svm_run::jit_resolve_and_validate`**: decode → `resolve_imports_with(symtab)` → verify → the
    existing fail-closed gate. The resolve (rewrite) runs **before** verify, so a mis-link (unknown
    name, wrong sig) is caught by re-verification, never trusted. `jit_blob_validator` is now the
    empty-symtab special case (a closed blob with imports fails closed). Tests:
    `crates/svm/tests/dynlink_resolve.rs` (3 + a codec round-trip).
- [x] **Grounding demo — a stateful REPL with by-name definitions** (`crates/svm/tests/dynlink_repl.rs`).
      The flagship use case made concrete on the C1 primitives: a `Repl` keeps a growing symbol table
      (`name → installed slot`); each `define` resolves the unit's `call.import`s against it, compiles
      + installs, and registers the name; later definitions reach earlier ones by name (`sq` → `quad =
      sq(sq(x))` → `quad_plus = quad(x)+sq(x)`). The symbol table **is** the dlopen registry — this is
      the **executable spec** for the guest-side `vm_dlopen`/`vm_dlsym` surface (runs today, no cap op).
- [x] **C2 — guest-driven `compile_linked` (`Jit` cap op 5)** (`crates/svm/tests/dynlink_cap.rs`). The
      guest delivers a unit-with-unresolved-imports **plus a symbol-table buffer** from its own window;
      the host resolves by name, re-verifies, and compiles — all in-sandbox. Pieces:
  - **Symbol-table wire form** (`svm-run`, LEB128): `count`, then per entry `name` + `kind`
    (`0=Slot(uleb)`, `1=Cap(uleb type_id, uleb op)`). `encode_symbol_table` (producer) +
    `decode_symbol_table` (fail-closed; values are guest-chosen by design — re-verify + the masked,
    `type_id`-checked `call_indirect` carry safety). An empty buffer = the empty table (closed op 0).
  - **The `JitValidator` seam** (`svm-interp`) widened to `fn(&[u8], Option<u8>, &[u8])` — the third
    arg is the symtab bytes; resolve stays in `svm-run` (can't be a dep of `svm-interp`).
    `Host::jit_compile_linked` is the core; `jit_compile` calls it with `&[]`.
  - **Op 5 on both backends**: the interp generic `Binding::JitDomain` arm + the JIT `jit_native_op`,
    differential-tested (interp == JIT) — incl. the full REPL flow guest-side (compile service →
    install → build symtab from the install slot → `compile_linked` a unit importing it → invoke = 127).
- [ ] **C3 — the `vm_dlopen`/`vm_dlsym`/`vm_dlclose` C surface** over op 5 + `Memory` (data placement /
      `DataReloc`s) + a chibicc `__vm_jit_compile_linked` builtin → `(iface::JIT, 5)`; then evolve
      `demos/jit/jit_repl.c` into the real linking REPL. (C1 host half + C2 cap op are done; this is the
      guest-facing packaging — the last capstone step.)
- [ ] GOT / late-binding variant (so *old* code calls *not-yet-loaded* code by name without recompiling
      the caller: the caller does `call_indirect (load GOT[i])`; the loader writes the resolved slot
      into the GOT at load — pure data writes via `Memory` + `memory.init`, reusing M2's reloc). The
      current `Slot` lowering bakes the slot at link time, so it needs the dependency installed first.
- [ ] Merge per-unit `debug_info` into the linked module (today `link` drops it: `debug_info: None`).
- [ ] Frontend support: have chibicc/svm-wasm emit `call.import` for *defined-elsewhere* symbols +
      the `data_exports`/`relocations` metadata, so real separately-compiled C/Rust units link (today
      tests hand-build `LinkUnit`s).

### Who actually uses this (real consumer programs)

To keep the work grounded — the programs that motivate by-name linking, and what each needs:
- **Stateful REPL with persistent definitions** — definitions compose by name across prompts; the
  symbol table is the dlopen registry. *Demoed today* at the harness level (`dynlink_repl.rs`); the
  guest-C version (evolving `demos/jit/jit_repl.c`, which today JITs standalone throwaway units) needs
  the `compile_linked` cap op. **The chosen flagship.**
- **Plugin host** — a host with a stable API loads a separately-shipped plugin from the powerbox; the
  plugin calls host services by name and exports an entry the host `vm_dlsym`s. The clearest "this is
  `dlopen`" story + the security pitch (re-verified, capability-gated, unforgeable funcref). Needs the
  cap op + the C surface.
- **Shared runtime library** (`libm`-shape) — one copy resolved by name from many clients. The static
  case is `link()` today (M1/M2); the load-time-shared case needs the cap op.
- **Hot reload / live patching** — install a new version at a new slot, rebind the name; new callers
  resolve to it, in-flight calls finish on the old. The REPL demo already shows the redefinition shape.
- **Dynamic class loading in a managed runtime** (the headline motivation) — a guest mini-language VM
  whose `load_class` compiles methods that link to runtime support + superclass methods by name. The
  most ambitious; needs the cap op + C surface + GC integration.

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

**Design question — SETTLED (C1): host-assisted.** The import-resolution rewrite runs **host-side**:
`Jit.compile`/`compile_linked` takes a guest-provided symbol table and resolves *before* verify
(rewrite-then-verify), so the symbol table stays guest-controlled but a mis-link can't escape — it
fails verification. Simpler than a guest-side rewrite (no in-guest IR-rewriter to load/trust) and
equally safe. **Done:** the host primitive (`jit_resolve_and_validate`), the serializable import
section (svm-encode v2), **and the `compile_linked` cap op (op 5) on both backends (C2)** — a guest
delivers the IR + symbol-table bytes and the host resolves+verifies+compiles.

**Remaining for the capstone — C3, the guest C surface:**
- `vm_dlopen`/`vm_dlsym`/`vm_dlclose` over op 5 + `Memory` (for data placement / `DataReloc`s) — the
  guest builds the symbol-table buffer (the C2 wire form) from its own loaded-unit registry. The op
  is `cap.call 11 5 (i64,i64,i64,i64)->(i64)`; add a chibicc `__vm_jit_compile_linked` builtin →
  `(iface::JIT, 5)` (next to the existing `__vm_jit_*` at `frontend/chibicc/codegen_ir.c` /
  `svm-run/src/lib.rs` ~`"vm_jit_compile" => (iface::JIT, 0)`).
- Then evolve `demos/jit/jit_repl.c` into the real **linking** REPL (today it JITs throwaway units).
- (Later) data-symbol resolution — today resolution covers *function* imports → slots; data imports
  still go through M2's `DataReloc` at link time, not a runtime `vm_dlsym` address.

**Why SVM's `dlopen` is *better* than POSIX** (the selling point): the "shared object" is serialized
SVM IR; `dlsym` returns an **unforgeable funcref slot** (§3c-checked at the call), not a raw pointer;
**everything loaded is re-verified** (a malicious object can't escape — worst case it corrupts its own
window, a §1 non-goal); and **loading is capability-gated** (you need the `Jit` cap, and the bytes
arrive through the powerbox — no ambient "load any file"). Safe dynamic loading.

---

## How to pick up

- Primitives live in `crates/svm-ir/src/lib.rs` (`resolve_imports_with`, `link`, `Resolved`,
  `LinkUnit`, `DataReloc`). Read `dynlink.rs` for IR-level usage, `dynlink_runtime.rs` for the
  guest-JIT (`CompiledModule::compile`/`define_extra`/`install`/`run_extra`) usage, and
  `dynlink_resolve.rs` for the C1 host-assisted path (serialize an unresolved unit →
  `svm_run::jit_resolve_and_validate` → `define_extra`).
- The wire format (`crates/svm-encode/src/lib.rs`, v2) carries the import section + `call.import`
  (`op::CALL_IMPORT = 0x7C`); the three guest-side C emitters in `crates/svm-run/demos/jit/*.c`
  hand-write it, so a layout change there must be matched in all three (a `0` import count, etc.).
- Run: `cargo test -p svm --test dynlink --test dynlink_runtime`. Full gate:
  `cargo test --workspace && cargo clippy --workspace --all-targets -- -D warnings && cargo fmt --all --check`.
- Text-IR string escapes are `\xHH` (not WAT's `\HH`) — bit us in the M2 data tests.
- `svm-llvm` is workspace-**excluded** (links libLLVM); `cargo build --workspace` skips it, so changes
  to `svm_ir::Module`'s shape must also update `crates/svm-llvm/src/lib.rs` (it has its own CI job).
