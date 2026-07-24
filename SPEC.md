# Executable ISA spec & generated conformance tests (`svm-spec`)

> Status: **slice 1 landed** — `crates/svm-spec` (the op table: 80 scalar rows with
> reference `eval` closures, the exhaustive [`coverage`] walk over all of `Inst`) +
> `crates/svm/tests/spec_vectors.rs` (suite 1: ~48k boundary vectors × three backends,
> <5 s). First findings, both fixed with the slice: (1) DESIGN.md §3b prose claimed
> `rem_s` traps on INT_MIN/−1 — both backends (correctly, wasm-identically) return 0;
> the prose is corrected. (2) The JIT had **no lowering for `ptr.add`/`ptr.to_int`/
> `ptr.from_int`** — and because the differential harness skips `Unsupported` modules,
> every `irgen`-generated module containing a ptr op was silently dropped from the
> interp↔JIT differential; the (trivial, pure-arithmetic) lowering is added and the
> ptr ops now ride both the spec vectors and the 4000-seed differential.
>
> **Slice 2 landed** — the scalar float rows (70: consts, `FBin`/`FUn`/`Fma`/`FCmp`,
> saturating + trapping float→int with the exact boundary lattice, int→float, float
> `select`), taking the table to 150 rows / ~100k vectors × three backends (<8 s).
> No backend findings this time; the yield is precision — the §3b prose was silent on
> `min`/`max` NaN/zero semantics, `nearest` tie-breaking, and the trunc trap bounds,
> which are now pinned executable definitions (prose clarified in place).
>
> **Slices 3 + 4 landed** — `spec_encode.rs` pins every row's opcode byte against the
> spec's independently-restated byte map (explicit per-op bytes, not `base+index()`)
> plus per-op `decode∘encode` identity; `svm_spec::verify` is the **reference
> verifier** (an independent full second implementation of the §3b/§3c rules, all 86
> `Inst` variants), and `spec_verify.rs` holds it in agreement with `svm-verify` over
> every row module, ~300 generic per-row mutations (wrong operand type / undefined
> operand, each pinned to its `VerifyError` variant), ~20 directed per-rule rejects,
> and an `irgen` sweep (300 modules × 6 structural mutations, accept/reject
> agreement). The verifier's accept direction now has an independent check.
>
> **Slice 5 landed** — the spec **window model** (trap-confinement restated from the
> `svm-mask` contract: whole span in `[0, mapped)` computed without wraparound — a
> wrapping effective address faults, never aliases; zero-length bulk ops inert at wild
> pointers; a faulting access mutates nothing) + 26 memory rows (14 loads, 9 stores,
> the 3 bulk ops) with window-boundary vector lattices, run on all three backends with
> **model-computed final-window comparison** (`spec_mem.rs`) — plus their encoding
> pins and per-row verifier rejects (no-window / i32-address / undefined-operand).
> **Finding: ISSUES.md I21** — the JIT's D62 bulk lowering bounds its span check
> against `reserved` and relies on the libcall touching the guard for the
> `(mapped, reserved]` part, so (1) `mem.copy`/`mem.move` with `dst == src` and an
> oversized span **lose the trap entirely** (libc short-circuits the self-copy) —
> both interpreters trap — and (2) faulting bulk ops leave partial writes where the
> interpreter faults before any write. Not an escape (everything stays inside the
> reservation), but a §3 parity break. The suite pins interp + bytecode fully and
> skips only the JIT leg of the guard-hole trap vectors (grep I21) until the fix —
> which touches the confinement hinge and needs a reviewed design decision.
> A second, smaller precision catch (macOS CI): the `mapped` boundary is
> **host-page-granular** — a 4 KiB window on a 16 KiB-page host (macOS ARM) is backed
> by one whole page, so accesses just past `mapped` succeed there. The byte-exact
> boundary the model pins therefore holds only at page-aligned window sizes; the spec
> window is 64 KiB (the same choice `irgen` already made for the same reason), and
> the constraint is now recorded on `MEM_LOG2` as part of the executable definition.
>
> **Slice 6 landed** — the SIMD rows (`svm_spec::simd`, ~250 concrete op×shape rows
> covering every §17 `v128` value op) with independently-written lane semantics from
> the `svm-ir` op documentation, run on all three backends (`spec_simd.rs`, ~10k
> vectors, <4 s). Since `v128` can't cross the JIT entry ABI, inputs are baked as
> `v128.const` and results observed as two `i64x2.extract_lane`s, batched
> many-per-module to amortize compiles. `v128.load`/`store` get the 16-byte
> window-boundary lattice (the one escape-TCB delta SIMD adds, D58). Lane-NaN policy
> per D58: computed float lanes compare NaN-class, masks and moves bit-exact. One
> carve-out honored (not a finding): `i64x2.{min,max}_{s,u}` are a **documented** JIT
> `Unsupported` bail (no legalizable Cranelift lowering; wasm never emits them) — the
> interpreters stay fully pinned there. All SIMD rows also ride the encoding suite
> (the `0xFE` prefix + sub-opcode pins) and both verifiers.
>
> **Slice 7 landed — the plan is complete.** The completeness closure
> (`svm_spec::structural`, 36 typing+encoding rows: the 4 atomics + fence + 2 `v128`
> memory ops, the calls + `ref.func`, the 6 host ops, the 7 concurrency ops, the misc
> control ops, and all 7 terminators) homes every remaining op — each with a minimal
> **verifiable witness module** (accepted by both verifiers, except `call_import`, the
> un-verifiable pre-resolution import form), `decode∘encode` round-trip, and an opcode
> byte pin (`spec_structural.rs`). These carry no `eval` (host/interleaving-dependent —
> the SPEC.md scope fence). A new exhaustive `row_home()` match (the third forcing
> function, with `coverage()` and the reference verifier's `check_inst`) maps **every**
> one of the 86 `Inst` variants to its owning slice, so adding any op is a compile
> error until the spec homes it. The executable spec now covers the entire ISA.
>
> **Nightly coverage-guided fuzzing wired (post-plan).** The deterministic boundary
> lattices now have an unbounded counterpart: two libFuzzer targets (`fuzz/spec_ops`,
> `fuzz/spec_verify`) driven by a shared `specfuzz` driver — `spec_ops` feeds random
> operand values through each scalar/float row and checks all three backends against
> the spec `eval`; `spec_verify` holds `svm-verify` and the reference verifier in
> accept/reject agreement over generated + mutated modules. Both ride the scheduled
> `cargo-fuzz` CI matrix and are mirrored on stable by `spec_fuzz_smoke.rs` (so they
> gate every PR and can't rot), the same nightly-target + stable-mirror pattern as
> `diff`/`jit_fuzz`.

**Goal.** One **machine-readable description of the ISA** — typing rules, binary
encoding, and (for the deterministic core) semantics — that lives in a **test-tier
crate** and *generates* three conformance suites: per-op semantic vectors run on all
three backends, rule-keyed verifier accept/reject pairs, and encoding conformance.
Plus a tiny independent reference verifier differentially tested against
`svm-verify`. The spec is a *redundant, executable statement of intent*: any
disagreement between it and a backend is, by construction, a bug in one of them —
the same epistemics as the existing interp↔JIT differential (§18), extended to the
verifier and the encoding.

**Why.** Three gaps in an otherwise strong test story:

1. **The ISA has no single machine-readable definition.** `Inst` (~87 variants +
   ~30 operand sub-enums, `crates/svm-ir/src/lib.rs`) is defined once, but the
   byte map is a hand-maintained table in `svm-encode` (`mod op`), the typing
   rules are ~630 lines of imperative Rust in `svm-verify::check_inst`, the
   semantics live in the interpreter, and the human spec is prose
   (`DESIGN.md` §3b). Rust's exhaustive-match check keeps these structurally in
   sync; nothing cross-checks them *semantically*.
2. **The verifier is untested in the accept direction.** `fuzz/decode_verify`
   exercises fail-closed on garbage; `irgen` exercises "valid modules verify."
   But "the verifier accepts *exactly* what the typing rules allow" has no
   independent check — an accept-direction bug in the TCB's contract crate only
   surfaces if it happens to crash a backend downstream.
3. **No per-op test vectors.** The generative differential explores op
   *combinations*; there is no systematic "for `i32.div_s`: `(INT_MIN, −1)` →
   trap, `(x, 0)` → trap, boundary matrix → expected values" suite derived from
   each op's definition — the equivalent of wasm's spec test suite.

**Non-goals (scope fence).**

- **No mechanized proof.** §2a's bar is "as secure as Wasmtime"; a
  Coq/Lean/Isabelle treatment stays the explicit post-MVP workstream. If that
  ever opens, the op table built here is its natural input.
- **No codegen into the TCB.** The spec crate never becomes a build dependency
  of `svm-ir`/`svm-encode`/`svm-verify`/`svm-interp`/`svm-jit`, and no TCB code
  is generated from it. Those crates stay boring, hand-written, auditable
  (AGENTS.md prime directive). The spec *cross-examines* them in CI; it does not
  *produce* them. Generating the verifier from the spec would also destroy the
  point — two independent statements must agree.
- **No semantics for host / concurrency ops.** `cap.call`, `cont.*`,
  `thread.*`, `wait`/`notify`, `GcRoots`, `SetJmp`/`LongJmp` etc. get typing +
  encoding rows only. Their semantics are host- and interleaving-dependent and
  already have better-suited harnesses (`explore_all`, dpor, loom, the
  differentials).
- **Does not replace `irgen`/`diff`/`fuzz`.** Those explore the composition
  space; this pins per-op definitions. Complementary, not competing.

## Design

### Crate placement & dependency rule

New workspace member `crates/svm-spec`:

- The **library** depends on `svm-ir` only (it needs the `Inst`/sub-enum types
  to be exhaustive over them, and builds `Module`s programmatically the way
  `irgen.rs` does). Dependency-free beyond that.
- The **conformance tests** that drive backends live where cross-crate harnesses
  already live: `crates/svm/tests/spec_*.rs` (the umbrella crate already
  dev-depends on text/encode/verify/interp/jit). `svm-spec` itself never
  appears in any runtime dependency graph.
- Rides `cargo test --workspace` — deterministic, seconds-fast, gating (no
  fuzz-style time budgets).

### The op table

The heart of the crate: one **row per concrete op** (variant × sub-op × type —
e.g. `IntBin{I32, DivS}` is one row, roughly 200 scalar + 80 SIMD rows total):

| field | contents |
|---|---|
| `id` | mnemonic, matching `svm-text` exactly (e.g. `i32.div_s`) |
| `typing` | operand `ValType`s → result `ValType`s, as data; ops whose rule isn't a fixed signature (`select`'s polymorphism, calls, `br_table`, `cap.call`) are flagged `Bespoke` and handled by the reference verifier in code |
| `encoding` | expected opcode byte(s), re-stating `svm-encode`'s `mod op` map (family base + `index()`) as checked data |
| `class` | `Pure` / `Trapping` / `Memory` / `Control` / `Host` / `Concurrency` |
| `eval` | for `Pure`/`Trapping` scalar+SIMD rows: `fn(&[Val]) -> Result<Val, TrapKind>` — the reference semantics |

**The `eval` closures are written fresh from `DESIGN.md` §3b prose, not imported
from `svm-interp`.** This is the independence rule that keeps the suite from
being a tautology: closures copied from the interpreter would rubber-stamp it.
(For ops where the only sane implementation is identical — `wrapping_add` — the
redundancy is admittedly thin; the value there is pinning the *prose* — trap
conditions, shift-mod-bitwidth, saturation, rounding, sign-extension — as
executable expectations that all backends must meet.)

**Completeness is compiler-enforced**, the same mechanism that syncs the
backends today: the table constructor contains an exhaustive `match` over `Inst`
and over every sub-enum (`BinOp`, `LoadOp`, `VIntBinOp`, …) — adding an op
without a spec row is a compile error, not a review catch. A unit test
additionally walks every sub-enum's `index()` range and asserts a row exists per
concrete op.

### Generated suite 1 — per-op semantic vectors (`spec_vectors.rs`)

For each `Pure`/`Trapping` row:

- **Input classes per type**, biased to boundaries (extending `irgen`'s list):
  ints — `0, ±1, INT_MIN/MAX, 2^k ± 1`; floats — `±0, ±1, NaN, ±inf`, denormals,
  max-finite, values straddling every int-conversion bound. Full cross-product
  for unary/binary ops, deterministically capped where it explodes.
- **Expected result or expected trap kind** computed by the row's `eval`.
- **Vehicle:** modules built programmatically from `svm-ir` structs (as `irgen`
  does), batching ~64 vectors per generated function so the JIT compile cost is
  amortized — one compile per op, not per vector. Optional `SVM_SPEC_DUMP=dir`
  writes the batch as `.svm` text for debugging a red vector.
- **Run on all three backends** — tree-walk interpreter, bytecode interpreter,
  JIT — asserting each matches the spec expectation (value bit-exact, or trap
  kind). This is stronger than the existing differential shape: today backends
  are checked against *each other*; here all three are checked against a
  *definition*, so a shared misreading of the prose can no longer hide.
- **NaN policy:** same carve-out as the existing interp↔JIT differential — NaN
  bit patterns are host-defined in default mode (§3b), so a NaN expectation
  asserts "is NaN," not bits; everything else is bit-exact.

### Generated suite 2 — verifier conformance (`spec_verify.rs`)

Two parts:

- **Rule-keyed accept/reject pairs.** For each typing row, generate (a) a
  minimal module using the op correctly — must verify; (b) systematic mutations
  each violating *exactly one* rule — wrong operand type, out-of-range value
  index, wrong result arity, bad branch-arg types, bad lane/shape/ordering,
  out-of-range indices — each asserted to be rejected. Assert the specific
  `VerifyError` variant (`TypeMismatch`, `ValueOutOfRange`, `ArgCountMismatch`,
  `BadSimdLane`, …) where the mapping is unambiguous; assert reject-only where
  pinning the variant would be brittle.
- **A reference verifier.** A few hundred lines in `svm-spec` interpreting the
  typing table (plus bespoke code for the flagged rows). Differential:
  `svm_spec::verify(m).is_ok() == svm_verify::verify_module(m).is_ok()` over
  (a) every suite-2 module, (b) an `irgen` seed sweep, (c) the mutation corpus
  applied to `irgen` output. This closes gap 2: the accept direction now has an
  independent second implementation that must agree.

### Generated suite 3 — encoding conformance (`spec_encode.rs`)

For each row: build a single-op module, `encode` it, assert the instruction's
opcode byte equals the row's `encoding` and that `decode` returns the identical
IR (the existing `roundtrip` fuzz property, but *directed and exhaustive per op*,
and now tied to an expected byte — turning the `mod op` comment table in
`svm-encode` into a checked artifact). Encoding drift (renumbering, family-base
moves) becomes a red test instead of a silent format break.

### Memory-op semantics (the one place suites 1 and the escape oracle meet)

`Load*`/`Store*`/`MemCopy`/`MemMove`/`MemFill` rows carry `eval` against a tiny
spec window model: `mapped` little-endian bytes; an access
`[addr+offset, addr+offset+width)` succeeds iff it lies within `[0, mapped)`,
else `MemoryFault` — trap-confinement semantics exactly as specified in
`DESIGN.md` §4 (D63), including the overflow-free span check for the bulk ops
(D62). Vectors include the boundary lattice (`end == mapped`, `end == mapped+1`,
`addr` near `u64::MAX` so `addr+offset` wraps, zero-length bulk ops at wild
pointers). This does not replace `fuzz/mask` or `escape_oracle.rs` — those prove
the *lowering* and the *mechanism*; this pins the *definition* the mechanism
must implement, at the same three-backend level as every other vector.

## Implementation plan

Slices, each landing green with its tests (AGENTS.md: tests from the first
commit). Ordered so every slice delivers a standing suite:

1. **Skeleton + scalar integers** — **done** (see Status). `crates/svm-spec` with
   the table schema, rows + `eval` for consts, `IntBin`/`IntCmp`/`IntUn`/`Eqz`/
   `Convert`/`Select`, `Cast`, `PtrAdd`/`PtrCast`; suite-1 harness running all
   three backends.
   *Exit: every i32/i64 op has passing boundary vectors on interp, bytecode, JIT.* ✅
2. **Floats + conversions** — **done** (see Status). `FBin`/`FUn`/`Fma`/`FCmp`,
   `FToISat`/`FToITrap`/`IToFConv`, reinterpret casts; NaN policy wired. *Exit: the
   trapping and saturating conversion boundary lattices pass on all backends.* ✅
3. **Encoding conformance** — **done** (see Status). Suite 3 over all rows so far;
   the completeness walk is the exhaustive per-op encoding matches themselves (a
   new sub-op is a compile error until it gets a conscious byte assignment).
   *Exit: every specced op's byte is pinned; adding an op without a row fails the
   build.* ✅
4. **Verifier conformance** — **done** (see Status). The reference verifier +
   suite 2 (accept/reject mutation pairs keyed to `VerifyError`) + the
   accept/reject differential over an `irgen` sweep. *Exit: `svm-verify` and
   `svm-spec` agree on every module in the corpus; each typing rule has a
   directed reject test.* ✅
5. **Memory ops** — **done** (see Status; one open carve-out). The spec window
   model + `Load`/`Store`/bulk rows + the OOB boundary lattice. *Exit:
   trap-confinement boundary vectors pass on all three backends.* ✅ (JIT leg of
   the bulk guard-hole trap vectors excepted — ISSUES.md I21.)
6. **SIMD** — **done** (see Status). The `v128` families (lane typing already
   total, §17/D58); `eval` per lane op is mechanical. *Exit: parity with slices
   1–2 for vector ops.* ✅ (`i64x2` min/max JIT leg excepted — a documented
   backend bail, not a spec gap.)
7. **Coverage closure** — **done** (see Status). Typing + encoding rows for the remaining control /
   host / concurrency ops (no `eval`), plus the exhaustive `row_home()` walk.
   *Exit: the completeness walk covers all of `Inst`.* ✅

Per §18's taxonomy this is agent-fast volume work (data entry + harness); the
risk concentrates exactly where it should — writing `eval` closures honestly
from the prose. A closure↔backend divergence is the desired signal: either the
prose, the closure, or the backend is wrong, and CI now says so.

## Risks & honest notes

- **Tautology risk.** The independence rule (`eval` from prose, never from
  `svm-interp`) is discipline, not mechanism. Review for it; where an
  implementation is forced to be identical, the suite still buys pinned trap
  conditions and cross-backend agreement against a fixed expectation.
- **A third statement of the ISA to maintain.** Accepted: the cost is data-entry
  shaped, omission is a compile error (exhaustive matches), and staleness is a
  red test — strictly better failure modes than the prose spec's silent drift.
- **Prime-directive tension.** Contained by the scope fence: no proc-macros, no
  generated TCB code, no runtime dependency on `svm-spec`, table-as-plain-data.
  If a change here makes `svm-verify` or `svm-encode` harder to read, it's
  wrong — the spec adapts to the TCB, never the reverse.
- **Error-variant brittleness.** Suite 2 pins `VerifyError` variants only where
  the rule→error mapping is one-to-one; otherwise accept/reject agreement
  suffices. Refactoring the verifier's error taxonomy should not require
  re-deriving the suite.
- **CI time.** Bounded by construction: batched vectors (~one JIT compile per
  op), deterministic counts, no wall-clock budgets. If it ever exceeds a few
  seconds, cut vector counts, not backends.
