# Sandbox VM

[![CI](https://github.com/thesherwood/vm/actions/workflows/ci.yml/badge.svg)](https://github.com/thesherwood/vm/actions/workflows/ci.yml)

A compilation target and sandbox VM: as secure (for the host) as WebAssembly,
faster than wasm on the interface / 64-bit-memory / startup axes, with a simpler and
more flexible interface, and real virtual memory.

The full design lives in [`DESIGN.md`](DESIGN.md); the working agreement (keep it
simple, commit to `main`, fuzz/test/bench early, data-oriented design) is in
[`AGENTS.md`](AGENTS.md).

> Status: **Phase 2 complete, into Phase 3** — the core loop, the Cranelift JIT, and the
> C frontend are all in place. The full **scalar IR** (integer /
> float ops, linear memory with confinement masking, direct / indirect / tail calls +
> the function table, `select`, `br_table`, `unreachable`) plus **capabilities**
> (`cap.call` over a host-owned handle table, and the MVP powerbox — `Stream` / `Exit`
> / `Clock` / `Memory`, §3c/§3e) flow through text ⇄ binary ⇄ verifier ⇄ reference
> interpreter ⇄ JIT, with the masking unit isolated and fuzzed, and a **generative
> interpreter-vs-JIT differential fuzzer** (a verifier-valid IR generator → both backends
> must agree on result + trap; stable-CI seed loop + a libFuzzer `diff` target). The
> **Cranelift JIT** (§9)
> now lowers the **entire IR** — integer/float ops, conversions, the §4 memory
> **masking lowering** (I1), function-table **indirect-call dispatch** (I2), direct/
> indirect/tail calls, trap detection, and **`cap.call` through a host thunk** — all
> **differential-tested against the interpreter** oracle (§18), including trap kinds and
> host side effects. A **C frontend** (`frontend/chibicc`, a vendored chibicc fork with an
> `--emit-ir` backend, §3d) compiles a broad C subset — ints/longs/floats, locals,
> pointers, arrays, structs/unions, globals & string literals (incl. **pointer
> initializers / relocations** — `char *p = "..."`, `&global`, function-pointer tables),
> the full operator set
> incl. short-circuit `&&`/`||`/`?:`, `if`/`while`/`for`/`do`/`switch` with
> `break`/`continue` and **general `goto`/labels**, functions and **recursion** (via a threaded data-stack pointer),
> **function pointers** (a function designator decays to its `ref.func` index; `fp(args)`
> lowers to `call_indirect` through the function table, I2), **by-value structs/unions**
> (passed by hidden pointer, returned via `sret`, D39; whole-aggregate assignment),
> **varargs + a guest-C `printf`** over the powerbox, and **`malloc`/`free`** (a guest
> bump allocator) — all of which **verify and run identically on the interpreter and the
> JIT**, hello-world and a heap-allocated linked list included (the §18 Phase-2 "it works"
> milestone). The §3d **SSA-promotion pass** lifts non-address-taken scalar locals out of
> memory into SSA values (threaded as block params), so the JIT register-allocates them — a
> hot loop body drops from ~22 load/store ops to zero. **Production trap-catching** is in
> (unix): an `mmap`'d window with a `PROT_NONE` guard page + a SIGSEGV/SIGBUS handler turns
> an out-of-window fault into a clean `MemoryFault` (§4/§5 detect-and-kill), and the large
> reserved-window model is the default; **read-only data segments** (§3a/D40) and a real
> **Memory capability** (`map`/`unmap`/`protect`) exist. Still ahead:
> narrow-scalar promotion, demand paging, atomics + the rest of the
> concurrency model (Phase 4), SIMD, and capability extras. This
> is a research build; "appears to work" is reachable, "is certified secure" is an explicit
> post-MVP workstream (see `DESIGN.md` §2a/§18).

## Layout

| Crate | Role | TCB? |
|---|---|---|
| `svm-ir` | Core IR: block-local typed SSA over a CFG (§3a/§3b) | escape-TCB |
| `svm-mask` | Confinement masking — the isolated, separately-fuzzed unit (§4, I1) | escape-TCB |
| `svm-encode` | Binary encode + **decode** (untrusted-input-facing) (§3a) | escape-TCB |
| `svm-verify` | The verifier — single linear pass, fail-closed (§2a I2/I3/I4; §3b) | escape-TCB |
| `svm-interp` | Reference interpreter — the differential oracle (§18) | — |
| `svm-jit` | Cranelift JIT — CLIF lowering + the §4 masking lowering + guard page/signal (§9) | escape-TCB† |
| `svm-text` | Text format ⇄ IR (dev/debug; 1:1 with binary) (§3a) | — |
| `svm` | Umbrella: pipeline (`assemble`/`load`/`run`) + tests + bench | — |
| `svm-run` | Embedding runtime + **`svm-run` CLI**: instantiate with the powerbox, run on the JIT | — |
| `fuzz/` | cargo-fuzz targets (nightly); mirror the stable smoke fuzz | — |

†`svm-jit` is escape-TCB but, by design (§1), shares Wasmtime's codegen — so unlike
the other TCB crates it *does* take a dependency (Cranelift). The dependency-free rule
covers only the small audit-critical crates (`svm-ir`/`svm-mask`/`svm-encode`/`svm-verify`).

The escape-TCB crates are deliberately **dependency-free** (small, fast to compile,
auditable). The host is Rust; the frontend (`frontend/chibicc`) is C; codegen lowers to
Cranelift (`DESIGN.md` D49 / D36).

## Build & test

```sh
cargo build --workspace
cargo test  --workspace          # pipeline + differential + 250k-iter smoke fuzz
cargo fmt   --all --check
cargo clippy --workspace --all-targets
cargo run --release --bin svm-bench   # decode / verify / interp throughput
```

## Run a program in the sandbox

The `svm-run` CLI compiles (if needed), verifies, and runs a guest program on the JIT under
the MVP powerbox (§3e) — `stdout`/`stderr` go to the real streams and it exits with the
guest's code:

```sh
cargo run -p svm-run -- crates/svm-run/demos/hello.svm   # text IR → "hello, sandbox!"
cargo run -p svm-run -- crates/svm-run/demos/hello.c     # C source (via the chibicc frontend)
cargo run -p svm-run -- crates/svm-run/demos/calc.c      # a recursive-descent calculator
cargo run -p svm-run -- crates/svm-run/demos/rational.c  # exact-rational arithmetic
cargo run -p svm-run -- crates/svm-run/demos/clay/clay_demo.c  # the Clay UI layout library!
cargo run -p svm-run -- crates/svm-run/demos/jsmn/jsmn_demo.c  # the jsmn JSON tokenizer
cargo run -p svm-run -- crates/svm-run/demos/sha256/sha_demo.c # SHA-256 (known test vectors)
cargo run -p svm-run -- crates/svm-run/demos/xxhash/xxh_demo.c # xxHash (XXH32/XXH64)
cargo run -p svm-run -- crates/svm-run/demos/tinfl/tinfl_demo.c # miniz tinfl (DEFLATE inflate)
cargo run -p svm-run -- crates/svm-run/demos/perlin/perlin_demo.c # stb_perlin (3D Perlin noise, floats)
cargo run -p svm-run -- crates/svm-run/demos/regex/regex_demo.c   # tiny-regex-c (backtracking matcher)
echo 'int main(){ return 42; }' > /tmp/r.c
cargo run -p svm-run -- /tmp/r.c ; echo "exit $?"        # → exit 42
```

`calc.c` (recursion + a function-pointer dispatch table) and `rational.c` (by-value
struct args/returns through direct and indirect calls) are larger real programs, each
checked byte-for-byte against a native `cc` build in `svm-run`'s tests. **`clay/clay_demo.c`
runs the real-world [Clay](https://github.com/nicbarker/clay) UI layout library** (a ~5k-line
third-party C header, vendored) sandboxed: it compiles through the frontend to ~93k lines of
IR, verifies, and runs on the JIT, producing the same render commands as a native build.
Getting it to run drove a batch of frontend/IR/JIT fixes (anonymous-aggregate designated
inits, ternary-returns-struct, >16-byte struct returns, mixed-width shifts, program-sized
windows, a contiguous JIT code arena, gcc-parity packed-enum/struct layout) — see `HANDOFF.md`.
**`jsmn/jsmn_demo.c`** runs the [jsmn](https://github.com/zserge/jsmn) zero-allocation JSON
tokenizer — a different shape (char/state-machine string scanning) that ran identically to a
native build with no new fixes, validating string handling, escapes, nesting, and error paths.
**`sha256/sha_demo.c`** runs Brad Conte's public-domain SHA-256 — the pure integer/bit shape
(32-bit wrapping arithmetic, rotates-as-shifts, a round-key table) — matching the standard test
vectors; it flushed a `func_index` null-token crash on undefined-function calls (now a clean error).
**`xxhash/xxh_demo.c`** runs [xxHash](https://github.com/Cyan4973/xxHash)'s scalar XXH32/XXH64
against the standard vectors; it added `_Static_assert` (C11) support to the frontend.
**`tinfl/tinfl_demo.c`** runs [miniz](https://github.com/richgel999/miniz)'s `tinfl` DEFLATE/zlib
*inflate* engine — a coroutine-style state machine (a deeply nested `switch`, bit-buffer shifts,
Huffman tables, a 32 KiB LZ77 dictionary inside the decompressor struct); it inflates an embedded
zlib stream byte-identically to a native build, with no new fixes.
**`perlin/perlin_demo.c`** runs [stb_perlin](https://github.com/nothings/stb) (Sean Barrett's 3D
Perlin noise) — the first **floating-point-heavy** shakedown (dense f32 gradient dot products, the
quintic ease polynomial, trilinear lerps, int↔float conversion, octave multiply/accumulate); it
prints fixed-point-scaled noise so any f32 divergence would show in the digits, and it matches a
native build byte-for-byte.
**`regex/regex_demo.c`** runs [tiny-regex-c](https://github.com/kokke/tiny-regex-c) — a
Rob-Pike-style **backtracking** matcher (`re_match` recurses through
`matchpattern`/`matchstar`/`matchplus`, retrying on failure), a new control-flow shape that
exercises data-stack threading and goto/branch lowering; it matches a native build with no new
fixes.

Accepts `.svm` (text IR), `.svmb` (binary), or `.c` (compiled through `frontend/chibicc`,
located via `$SVM_CHIBICC` or the in-repo build). Embedders can call the same path directly —
`svm_run::run_powerbox(&module, stdin)` returns the outcome plus captured output; it is the one
reusable host glue (the `cap.call` trampoline + powerbox grant), not escape-TCB (the verifier,
run first, is what makes a module safe).

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
