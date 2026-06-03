# Sandbox VM

[![CI](https://github.com/thesherwood/vm/actions/workflows/ci.yml/badge.svg)](https://github.com/thesherwood/vm/actions/workflows/ci.yml)

A compilation target and sandbox VM: as secure (for the host) as WebAssembly,
faster than wasm on the interface / 64-bit-memory / startup axes, with a simpler and
more flexible interface, and real virtual memory.

The full design lives in [`DESIGN.md`](DESIGN.md); the working agreement (keep it
simple, commit to `main`, fuzz/test/bench early, data-oriented design) is in
[`AGENTS.md`](AGENTS.md).

> Status: **Phase 1** — the core loop is in place. The full **scalar IR** (integer /
> float ops, linear memory with confinement masking, direct / indirect / tail calls +
> the function table, `select`, `br_table`, `unreachable`) plus **capabilities**
> (`cap.call` over a host-owned handle table, and the MVP powerbox — `Stream` / `Exit`
> / `Clock` / `Memory`, §3c/§3e) flow through text ⇄ binary ⇄ verifier ⇄ reference
> interpreter, with the masking unit isolated and fuzzed. Still ahead: the JIT, atomics,
> SIMD, and capability extras (attenuation, async, revocation, a module-level interface
> section). This is a research build; "appears to work" is reachable, "is certified
> secure" is an explicit post-MVP workstream (see `DESIGN.md` §2a/§18).

## Layout

| Crate | Role | TCB? |
|---|---|---|
| `svm-ir` | Core IR: block-local typed SSA over a CFG (§3a/§3b) | escape-TCB |
| `svm-mask` | Confinement masking — the isolated, separately-fuzzed unit (§4, I1) | escape-TCB |
| `svm-encode` | Binary encode + **decode** (untrusted-input-facing) (§3a) | escape-TCB |
| `svm-verify` | The verifier — single linear pass, fail-closed (§2a I2/I3/I4; §3b) | escape-TCB |
| `svm-interp` | Reference interpreter — the differential oracle (§18) | — |
| `svm-text` | Text format ⇄ IR (dev/debug; 1:1 with binary) (§3a) | — |
| `svm` | Umbrella: pipeline (`assemble`/`load`/`run`) + tests + bench | — |
| `fuzz/` | cargo-fuzz targets (nightly); mirror the stable smoke fuzz | — |

The escape-TCB crates are deliberately **dependency-free** (small, fast to compile,
auditable). The host is Rust; the (future) frontend is C; codegen will lower to
Cranelift (`DESIGN.md` D49 / D36).

## Build & test

```sh
cargo build --workspace
cargo test  --workspace          # pipeline + differential + 250k-iter smoke fuzz
cargo fmt   --all --check
cargo clippy --workspace --all-targets
cargo run --release --bin svm-bench   # decode / verify / interp throughput
```

## Fuzzing

Stable CI runs the smoke fuzz as ordinary tests (`crates/svm/tests/fuzz_smoke.rs`).
For coverage-guided fuzzing (nightly):

```sh
cargo install cargo-fuzz
cargo +nightly fuzz run decode_verify   # decode/verify/interp never crash
cargo +nightly fuzz run mask            # the confinement-masking invariant (I1)
cargo +nightly fuzz run roundtrip       # binary + text round-trip identity
```

Invariants under test (the security hinge, §2a/§4): on arbitrary bytes, `decode`
fails closed (never panics/OOMs/hangs), `verify` never panics, any *verified* module
is safe to interpret, the masking unit confines every access into its window, and
the formats round-trip without changing the IR.

## Example IR (text form)

```text
func (i32) -> (i32) {
block0(v0: i32):
  v1 = i32.const 0
  br block1(v0, v1)
block1(v2: i32, v3: i32):     ; v2 = i, v3 = sum
  v4 = i32.add v3 v2
  v5 = i32.const -1
  v6 = i32.add v2 v5
  br_if v6 block1(v6, v4) block2(v4)
block2(v7: i32):
  return v7                   ; sum of 1..=N
}
```
