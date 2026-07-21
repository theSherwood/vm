# Glossary

Concise definitions for the project's working vocabulary. One or two lines each; the
authoritative detail lives in `DESIGN.md`, `IMPORTS.md`, and `PROCESS.md` — this file is
for reading those without a decoder ring.

## The conceptual map (read this first)

The system is four ideas wearing many names:

1. **One format, two roles.** A *module* is the only code format. Instantiated with its
   own window and powerbox it is a *domain*; grafted into an existing domain (guest-JIT
   submission) it is a *unit*. Same bytes, same verifier, different role.
2. **Two call paths.** Every call is either *capability-addressed* (through the
   host-owned handle table, use-site checked: `call.import`, dispatch-form `cap.call`,
   `Jit.invoke`) or *funcref-addressed* (through the domain's function table:
   `call_indirect`, installed units). Invoke/install are not new concepts — they are
   compiled code entering these two existing paths.
3. **Two binding times.** Authority is connected either at *spawn/wiring time* (the
   embedder registry, a parent binding a child's manifest, accepting an offer) or at
   *runtime* (`import.attach` into a rebindable slot — the capability "global", written
   by the guest from what it already holds). Declaration never moves authority; only
   binding acts do.
4. **Structural types all the way down.** A *type* is a function signature; an
   *interface* is a tuple of them. Identity is always by shape, never by name (D59) —
   which is what makes interposition, testing, and virtualization typewise invisible,
   with provenance as the one honest bit on top.

## Core runtime

- **domain** — one isolated unit of execution: a module's code + its window + its
  powerbox. The root program is a domain; every §14 child and every provider instance is
  its own domain.
- **guest** — code running inside a domain, under the verifier's rules and the
  confinement masking. Untrusted by default.
- **host** — the trusted embedder side. Also the name of the concrete struct (`Host`)
  that owns a domain's handle table and capability state.
- **window** — a domain's linear memory: one contiguous range, every access masked or
  proven inside `[0, size)`. A §14 child's window is a sub-range ("carve") of its
  parent's.
- **powerbox** — a domain's capability state as a whole: the handle table plus the
  host-side objects behind it. "Granting into the powerbox" = making a capability
  reachable from that domain.
- **capability / cap** — the authority to invoke an interface on some object (a stream,
  a clock, a child, a wired implementation). Exists only as a handle-table entry; guests
  never hold pointers, only handles.
- **handle** — the guest-visible name of a capability: a packed `(generation, slot)`
  `i32`. Forgeable as data, inert if forged — every use re-checks slot, generation, and
  type_id (the §3c use-site check).
- **slot** (handle table) — one entry position in the host-owned handle table. Also used
  for *import* slots (below); context disambiguates.
- **generation** — a per-slot counter bumped on each (re)grant, packed into the handle.
  Makes a closed handle's value permanently stale (use-after-close is a clean fault, D37).
- **grant** — the host-side act of installing a capability into a domain's table
  (`grant_stream`, `grant_module`, `regrant_into_child`, …). Authority moves only by
  grants.
- **type_id** — the runtime identity of an interface: a `u32` stored in each table entry
  and re-checked at every use. Small constants for the built-ins (`iface::STREAM = 0` …);
  interned per-host for guest-implemented interfaces.
- **`iface`** — two related uses: (1) the `svm_interp::iface` module of built-in type_id
  constants; (2) the `ImplExport::iface` field — an index into the module's type section
  naming the interface entry the export implements.
- **intern (structural)** — the map from an interface's *shape* (its op-signature list)
  to its runtime `type_id`, maintained per-host: structurally identical shapes get the
  same id (D59: id-equality ≡ structural equality). Why interfaces need no names.
- **dispatch (generic)** — the one host entry (`cap_dispatch_slots`) every capability
  call funnels through, on all three backends. Where the use-site check and the per-
  interface behavior live.

## Imports (the consumer side)

- **import** — a named capability requirement a module declares: `import 0 "write"
  (i64, i64) -> (i64)`. Says "bind me something implementing this signature under this
  name"; confers nothing by itself.
- **manifest** — a module's import list as a whole: the up-front, fail-closed statement
  of what it needs. Bounds *requirements*, not *reach* (reach is bounded by grants).
- **import slot** — the per-instance binding position behind import `i`: filled at
  instantiation with `(type_id, op, handle)`. The module's bytes are never rewritten;
  the slot table is host-side state.
- **`required` / `rebindable`** — an import's binding mode. `required`: bound at
  instantiation or the spawn fails; immutable for the instance's life (always safe to
  devirtualize). `rebindable`: declared and typed, may start empty, filled at runtime
  by `import.attach`.
- **`call.import`** — the one capability-call convention. Static mode: slot immediate,
  types from the manifest, verifier-checked at load. Dynamic mode (designed; today via
  the `cap.call` wire form): object from a runtime handle value, checked at the use site.
- **`import.attach`** — fill (or refill) a `rebindable` slot with a capability the
  domain already holds, type-checked fail-closed. The "reflect, decide, attach once,
  then ordinary calls" pattern.
- **grouped import** — (designed: IMPORTS.md §3.5) an import slot binding a *whole
  interface* rather than one op: `import 0 "fs" iface 2`, called as
  `call.import 0.<op>`. Today's flat named import is the one-op case.
- **`cap.call`** — the wire form of dynamic-mode dispatch: `(type_id, op, sig)`
  immediates plus a runtime handle. Retired as a *concept* (it's just dynamic mode);
  kept as the encoding and the escape hatch for undeclared grants.
- **manifest-complete** — a verifier-computed per-module bit: no dynamic-mode dispatch
  anywhere, so the manifest is the complete list of interfaces the module can ever
  drive. Reflection (`cap.self.*`, including its reserved-id dispatch form) is exempt —
  it confers nothing.
- **discovery / reflection (`cap.self.*`)** — authority-neutral ops reporting what this
  domain already holds: count, get `(handle, type_id)`, resolve-by-name, attest,
  provenance. Never a grant.

## Exports & offers (the provider side)

- **export** — a named function entry point: `export "main" 0`. Lets the host (or
  linker) call a function by name. Nothing capability-flavored about it.
- **type section** — the one place a module declares shapes: `type (params) -> (results)`
  entries are function signatures; `interface { 0, 1 }` entries are tuples of indices to
  them. One index space, each signature written once. Declarations only — no code, no
  authority, no nominal identity (two modules declaring the same shape mean the same
  type).
- **interface** — a type-section entry that is a tuple of function-signature entries: a
  capability's shape, op by op. Not a separate concept from types — the composite case.
- **impl export / offer** — a declaration that this module *implements* an interface:
  `export "adder" impl 0 : 3 4` — "my funcs 3 and 4 implement interface #0," verifier-
  checked exactly. An *offer* because declaring it confers nothing: it is an
  advertisement, callable by no one until wired.
- **wire / wiring** — the authority-moving act that accepts an offer: someone holding
  both ends (embedder registry, parent at spawn, `Host::wire_impl*`) mints the table
  entry that makes the offer's functions callable from a consumer's domain, after a
  structural fail-closed signature check.
- **pure offer** — a wired offer with no state: each op call is a fresh reference run
  over the offer's functions with no window and an empty powerbox — arguments in,
  results out.
- **instanced offer / provider instance** — a wired offer with its own **provider
  domain**: a persistent window (seeded from the provider module's memory + data
  declarations) and powerbox that every op runs over, so state survives across calls.
  Shared (aliased) across re-grants, like a pipe's backing.
- **provider-pays** — the §5.3 metering rule: an instanced provider funds its own
  dispatch compute from a drainable, wirer-priced fuel reserve; a dry reserve is a
  probeable fault until topped up.

## Guest JIT (§22)

- **`Jit` capability** — the interface a guest uses to compile code at runtime: submit
  serialized IR from its own window, get back a code handle. Submissions pass the same
  decode + verify gate as any module, plus §22 preconditions.
- **unit** — one compiled submission. Runs over the *same* window, capabilities, and
  live function table as the submitting module (same domain).
- **invoke** (Model A) — call a unit through the `Jit` capability, code handle as an
  argument: a nested, signature-checked run over the caller's own world. No shared-table
  entry; the mask never moves.
- **install** (Model B2) — put a unit into the pre-sized shared function table as a
  **funcref**; existing code then reaches it via ordinary `call_indirect` at native
  speed. The funcref is plain guest data.
- **funcref** — an `i32` equal to a function's index in the domain's table; the currency
  of indirect calls. Type-checked at dispatch via the interned signature id.
- **jit_link / `compile_linked`** — compile a submission whose *named imports* resolve
  against previously installed units through a guest-built symbol table: guest-side
  dynamic linking (`vm_dlopen`/`vm_dlsym` in guest C).

## Trust & identity

- **attest (`cap.self.attest`)** — the one non-interposable report: platform-vouched
  facts about *this domain's* exposure (isolation tier, window-exposed, freeze-exposed).
  The trust anchor no parent can fake.
- **provenance (`cap.self.provenance`)** — the per-*binding* honest bit:
  `0` = **platform-terminated** (host-native implementation), `d ≥ 1` =
  **ancestor-terminated** (a wired guest implementation, `d` re-grant hops up). A parent
  can interpose anything but cannot hide that it did.
- **interposition** — a parent supplying its own implementation behind a child's import
  (§3.3 wrap/override). Typewise invisible (structural identity — deliberately), honest
  via provenance.
- **forward / wrap / override / withhold** — the four per-slot policies a parent picks
  when binding a child's manifest: alias its own entry; interpose with its own impl;
  substitute any other impl; bind nothing (`required` ⇒ spawn fails, `rebindable` ⇒
  empty slot).
