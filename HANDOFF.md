# Handoff — C frontend (chibicc → SVM IR) + differential fuzzing

Pick-up notes for a fresh session. Written 2026-06-03, **last updated 2026-06-10**.
Branch: **`main`** (this work has been committing straight to `main`; the remote is
`theSherwood/vm`). Everything below is committed and CI-green.

**Current state.** Phases 1–3.5 are complete and **Phase 4 has started.**
This file's single source of truth for status + open work is **§10**; the concurrency design rationale
(the VM ships *mechanism, not a scheduler*) lives in **DESIGN.md D56 / §12**.

- **Core loop + frontend.** IR ⇄ text ⇄ binary ⇄ verifier ⇄ reference interpreter ⇄ Cranelift JIT, with
  a broad C subset through `frontend/chibicc`, all differential-tested (interp == JIT == native `cc`),
  and a generative interp↔JIT differential fuzzer guarding the JIT.
- **Memory (Phase 3 / 3.5).** The §4 *large* reserved window + Memory cap + guest-controlled growth,
  guard-page/signal **detect-and-kill** (cross-platform: SIGSEGV/SIGBUS on unix, a vectored-exception
  guard on Windows), RO data segments (§3a/D40), and §13 `SharedRegion` aliasing — green on **Linux +
  macOS + Windows**. SSA promotion + mask elision are the perf wins; the **escape-oracle** (verified ⇒
  in-window final memory) is the confinement guard.
- **Concurrency (Phase 4, primitives only — mechanism, no VM scheduler).** Fibers (`cont.*`), threads
  (`thread.spawn`/`join`), linear-memory **atomics** (+ the C11 ordering surface + `atomic.fence`), a
  **`wait`/`notify` futex**, and a guest **`<pthread.h>`** — through the whole pipeline and **both
  backends**. The **interpreter** is the M:N green-thread executor and the deterministic oracle
  (`run_scheduled` / `explore_all`); the **JIT** runs 1:1 OS-thread vCPUs (`os_thread_rt`) + fibers
  (`fiber_rt`) over the `svm-fiber` stack switch on **x86-64 unix, aarch64 unix (macOS), and x86-64
  Windows** — cross-platform parity, all CI-green. Full breakdown + open items: §10.

---

## 1. What this project is (30-second orientation)

A capability-safe VM: a small typed SSA **IR** that goes text ⇄ binary ⇄ **verifier** ⇄
**reference interpreter** ⇄ **Cranelift JIT**. Memory is a power-of-two **window** with
address **masking** (§4) so guest memory accesses are confined; the verifier is the TCB
that enforces escape-freedom (§2a). Capabilities are host-owned handles invoked via
`cap.call` (§3c). The full design is in **`DESIGN.md`** (section numbers like "§3d" below
refer to it). Status framing is in **`README.md`**.

Workspace crates (`crates/`):
- `svm-ir` — IR types (`Module`, `Func`, `Block`, `ValType`, ops).
- `svm-text` — text parser/printer (`parse_module`).
- `svm-encode` — binary format.
- `svm-verify` — the verifier (`verify_module`).
- `svm-interp` — reference interpreter (`run`); also the M:N green-thread executor + the
  deterministic `run_scheduled`/`explore_all` concurrency oracle (§12).
- `svm-jit` — Cranelift JIT (`compile_and_run`, `JitOutcome`); JIT fibers/threads/futex on the three
  `fiber_rt` targets (x86-64 unix, aarch64 unix, x86-64 Windows) via `fiber_rt.rs`, `os_thread_rt.rs`.
- `svm-mask` — the isolated masking unit (`fuzz/mask` is its dedicated fuzzer).
- `svm-mem` — the shared guest-memory substrate (§12/§13); owns the memory `unsafe` so the
  interpreter stays `forbid(unsafe_code)`. Differentially fuzzed (raw `Mapped` vs the `Paged` model)
  and miri-checked (provenance + races) via a `cfg(miri)` heap backing.
- `svm-fiber` — native stack-switch primitive for JIT fibers / green threads; a per-ABI `switch`
  (x86-64 SysV, aarch64 AAPCS64, x86-64 Windows MS-x64) + a per-OS guard-paged `stack`. Switch fuzzer
  in its own tests.
- `svm` — umbrella crate + integration tests (`crates/svm/tests/`).
- `fuzz/` — libFuzzer targets (out of workspace; nightly + `cargo-fuzz`).

Two big things exist beyond the core loop: (1) **the C frontend** (most of this doc), and
(2) **a generative interp↔JIT differential fuzzer** (see §8). Test crates:
`c_frontend.rs` (C, two tiers), `jit_diff.rs` (hand-written JIT diff), `jit_fuzz.rs`
(generative diff), `escape_oracle.rs`, `pipeline.rs`, `fuzz_smoke.rs`, the §12 concurrency suite
(`threads.rs`, `concurrent.rs`, `concurrent_fuzz.rs`, `jit_threads.rs`, `jit_fibers.rs`,
`fiber_fuzz.rs`), the **concurrent escape-oracle** (`concurrent_escape.rs` + `concurrent_escape_fuzz.rs`),
and `shared_region.rs` (§13).

---

## 2. The C frontend — what exists

A **vendored fork of chibicc** (Rui Ueyama's small C compiler, MIT) lives in
**`frontend/chibicc/`**. We added one file, **`codegen_ir.c`**, an alternative backend
that walks chibicc's typed AST and emits **our text IR** instead of x86-64 asm, plus a
`--emit-ir` flag. Everything else in `frontend/chibicc/` is upstream chibicc (don't
edit it unless you must; keep the diff small).

**Two upstream `parse.c` fixes** (the only edits outside `codegen_ir.c`), both genuine chibicc
bugs found by trying to compile the **Clay** layout library, both around designated
initializers into **anonymous** aggregates (very common in real C), each validated against a
gcc matrix + the full suite with zero regressions:
1. `struct_designator` special-cased only anonymous *structs*, so a designator targeting an
   anonymous *union* member dereferenced a NULL `mem->name` → **segfault**. Now matches the
   canonical `get_struct_member` idiom (`TY_STRUCT || TY_UNION`).
2. `struct_initializer2` skipped the separator comma only on non-first members, but it is also
   entered right after a *designated* member (tok at the comma) when that member lands in a
   nested anonymous aggregate — so a following designator (`{ .a = x, .b = y }`) failed to
   parse. Now skips a leading comma when present (handling both callers: designated
   continuation at a comma, and brace-elision at a value).

**Clay runs end-to-end (the capstone).** Iterating on the Clay shakedown to completion,
`demos/clay/clay_demo.c` now compiles (~93k lines of IR), verifies, and runs on the JIT,
producing the same render commands as a native `cc` build (`svm-run` test
`demo_clay_layout_runs`). The full set of fixes Clay drove, beyond the two `parse.c` ones above:
- **gen_cond** — a ternary `?:` returning an aggregate carries the selected arm's *address*
  (merge type `pass_irty` = i64), not `irty(struct)` which errored.
- **guest_params** — chibicc prepends a hidden return-buffer pointer to `fn->params` for
  struct returns > 16 bytes (SysV); our §3d ABI uses its own sret for every size, so skip
  chibicc's to avoid double-counting (the ≤16B test structs never hit it).
- **binop shift width** — a shift keeps its amount's own width (`uint64_t << int`), so widen/
  narrow the amount to the value's width before `iN.shl/shr`.
- **svm-text i32.const** — accept the full u32 range (`0xFFFFFFFF` = -1).
- **program-sized window** — the frontend sizes the window to globals/BSS + a stack reserve
  (Clay's ~250 KB arena needs `memory 21`); small programs keep 64 KB.
- **svm-jit `ArenaMemoryProvider`** — allocate code+rodata from one contiguous 256 MiB arena;
  the default separate mmaps let ASLR place code and float-constant rodata > 2 GiB apart,
  overflowing cranelift's 32-bit PC-relative relocations (an intermittent ~1/6
  `compiled_blob.rs` panic on large modules) — now 25/25 clean.

**Struct-layout parity with gcc (fixed).** Initially every Clay struct holding a small enum
was bigger on the VM (`Clay_MinMemorySize` ~254 KB vs ~246 KB native) — chibicc sized **every
`enum` as `int` (4 bytes)**, while gcc honours Clay's `enum __attribute__((packed))` (1 byte).
This matters for host↔guest data exchange (a host writing structured data into the window must
agree on layout; §3d pins x86-64-SysV). Two-part fix:
- `enum_specifier` (parse.c) now parses `__attribute__((packed))`/`__packed__` and sizes the
  enum to the smallest integer type holding its values (1/2/4/8 bytes), and `gen_load`/
  `gen_store` access a packed enum at that width (it was always an i32 load → it read adjacent
  bytes; caught by `c_matches_gcc_packed_enums`).
- ship a minimal `frontend/chibicc/include/stdint.h`. Without it, `#include <stdint.h>` pulled
  the system `<sys/cdefs.h>`, which — because chibicc isn't `__GNUC__` — `#define`s
  `__attribute__(x)` to nothing, **silently stripping the attribute** before the parser saw it.
After both, **all 80 Clay struct sizes and `Clay_MinMemorySize` match gcc exactly**, and Clay
still renders identically. All edits except the three `parse.c` ones + `stdint.h` live in our
own crates / `codegen_ir.c`.

**Second real library — jsmn (clean).** The [jsmn](https://github.com/zserge/jsmn) JSON
tokenizer (`demos/jsmn/`, MIT, vendored) — a deliberately *different* shape from Clay (pure
char/state-machine string scanning, zero allocations) — compiled and ran **byte-identical to
native cc on the first try**, including string escapes, `\u` unicode, deep nesting, the
`-2`/`-3` error codes, and `JSMN_STRICT` mode. No new fixes needed: after the Clay batch the
frontend is robust enough that a clean library just works. Test `demo_jsmn_matches_native`.
(Also fixed `assert_demo_matches_cc` to flatten `/` in subdir demo names — it was silently
skipping the comparison for `jsmn/jsmn_demo.c`.)

**Hash libraries — SHA-256 and xxHash (one fix each).** Two integer/bit-shape shakedowns:
B-Con's public-domain **SHA-256** (`demos/sha256/`) and Cyan4973's **xxHash** XXH32/XXH64
(`demos/xxhash/`, scalar: `XXH_INLINE_ALL` + `XXH_NO_XXH3` + `XXH_NO_STREAM`). Both match native
cc + the standard test vectors; each demo provides the one or two `mem*` functions its library
uses (no libc). Fixes they drove: (1) `func_index` no longer segfaults reporting an
undefined-function call (a libc declaration has no source token) — clean error now; (2) chibicc
now supports **`_Static_assert`** (C11) / `static_assert` (C23) at file and block scope
(`static_assertion` in parse.c) — it was parsed as a function call. Tests `demo_sha256_*` /
`demo_xxhash_*` and `c_matches_gcc_static_assert`.

**Fifth real library — tinfl / miniz inflate (clean).** miniz's standalone DEFLATE/zlib
*inflate* engine (`demos/tinfl/`, MIT, vendored) — a fresh shape: a coroutine-style state
machine (a deeply nested `switch` driven by `TINFL_CR_*` macros + a saved program counter),
bit-buffer shifts, Huffman fast/slow lookup tables, and a 32 KiB LZ77 dictionary carried inside
the `tinfl_decompressor` struct. `tinfl_demo.c` inflates an embedded zlib stream (`blob.inc`) and
writes the result; it ran **byte-identical to native cc with no new fixes** — good evidence the
goto/switch lowering and struct layout hold up under a gnarly real-world state machine. The one
vendoring edit: `miniz_tinfl.c`'s `#include "miniz.h"` → `#include "miniz_tinfl.h"` (so the
inflate path is self-contained, no deflate/zip headers). Test `demo_tinfl_matches_native`.

**Sixth real library — stb_perlin / the first float shakedown (clean).** Every earlier
shakedown was integer/pointer/struct shaped, so the IR's **f32 path** had differential-fuzz
coverage but no *real-program* coverage. [stb_perlin](https://github.com/nothings/stb) (Sean
Barrett, public domain, `demos/perlin/`, vendored unmodified) is dense f32 arithmetic — gradient
dot products, the quintic ease polynomial, trilinear lerps, int↔float `fastfloor`, and
multiply/accumulate chains over octaves (fbm/turbulence/ridge). `perlin_demo.c` provides the one
libc function the octave variants need (`fabs`, no libm) and prints each value as a **fixed-point
integer** rather than via float formatting — so any divergence in the actual f32 arithmetic
between native cc and our JIT would land in the digits. It matched **byte-for-byte with no new
fixes** — good first evidence the f32 lowering is sound on real code. Test
`demo_perlin_matches_native`.

**Seventh real library — tiny-regex-c / backtracking recursion (clean).**
[tiny-regex-c](https://github.com/kokke/tiny-regex-c) (kokke, public domain, `demos/regex/`) is a
Rob-Pike-style matcher whose `re_match` recurses through
`matchpattern` → `matchstar`/`matchplus`/`matchquestion` → `matchpattern`, **backtracking** on
failure — a new control-flow shape (a workout for the threaded data-stack pointer and general
goto/branch lowering). Vendored with one minimal edit: the libc `<stdio.h>`/`<ctype.h>` includes
and the printf-only `re_print` debug helper (not in `re.h`'s API) are guarded behind
`#ifndef RE_FREESTANDING`; the driver defines it and supplies `isdigit`/`isalpha`/`isspace`. A
table of (pattern, text) cases prints match index/length and matches native cc **byte-for-byte,
no new fixes**. Test `demo_regex_matches_native`.

### Invocation
```
frontend/chibicc/chibicc -cc1 --emit-ir -cc1-input a.c -cc1-output a.svm a.c
```
`-cc1` runs the compiler in-process (no gcc-style driver subprocess); `--emit-ir`
dispatches to `codegen_ir` (see `cc1()` in `main.c`, where the wiring lives). Build with
`make -C frontend/chibicc` (needs `make` + a C compiler; both present in CI). Build
artifacts (`*.o`, the `chibicc` binary) are git-ignored.

### Test harness (`crates/svm/tests/c_frontend.rs`, 67 tests, two tiers)
`make`s the fork once, compiles each C snippet to IR, **verifies it**, then:
- **Tier 1 (all tests):** runs `main` (function 0 = `_start`) on **both the interpreter
  and the JIT** under identical mock powerboxes and asserts they agree on result, trap,
  and captured stdout/exit. Every C test is also a JIT differential test.
- **Tier 2 (`c_matches_gcc_*`):** compiles the *same* C with native **`cc`** (real
  stdio/stdlib) and asserts identical exit code + stdout — a real-compiler oracle for C
  semantics. ~15 programs incl. recursion (Ackermann), floats, printf, bubble sort, sieve,
  linked list. Needs `cc` (already required to build the fork).
```
cargo test -p svm --test c_frontend
```

### What C is supported today (the agreed stopping point)
`int`/`long`/`char`/`short`/`_Bool`/`enum`, `float`/`double`; pointers, arrays,
structs/unions (`.`/`->`, indexing, initializers); globals + string literals; the full
operator set incl. short-circuit `&&`/`||`/`?:`; `if`/`else`/`while`/`for`/`do`/`switch`
with `break`/`continue` and **general `goto`/labels**; functions, parameters,
**recursion**, **function pointers**
(indirect calls via `call_indirect`, dispatch tables, callbacks, fn-ptr struct members),
**by-value structs/unions** (passed/returned by value, whole-aggregate assignment),
**varargs**; **`printf`** and `exit` over the powerbox; **`malloc`/`free`/`calloc`** (guest
bump allocator). All verify and run identically on interp + JIT, and match native `cc`.

**By-value aggregates (sret, §3d D39).** Every by-value struct/union goes by hidden
pointer (no SysV register classification). A **struct/union return** makes the IR function
`(i64 sp, i64 sret, params…) -> ()`: the caller passes the address of chibicc's
`ret_buffer` (an lvar in the caller frame) as a hidden first arg, the callee writes the
result through it, and the call's value is that buffer address (so `f(x).field` and `s =
f(x)` work — `gen_addr(ND_FUNCALL)` returns it). A **by-value struct/union arg** is passed
as the lvalue address (`pass_irty`=i64); the callee `gen_memcpy`s it into its own frame
slot in the prologue (by-value semantics). **Whole-aggregate assignment** is a
`gen_memcpy`. Two chibicc quirks handled: a same-type aggregate cast on an assignment rhs
(`gen_convert` no-ops when held by-address), and **union first-member init** — chibicc emits
`v.i = (int)expr`, an aggregate→scalar cast that `gen_convert` lowers as a *load* of the
member's bytes (only array/function decay returns the address). `irty(TY_FUNC)`/`is_agg`/
`pass_irty`/`gen_memcpy` are the new helpers.
- **sret pointer is stashed to a frame slot, not threaded (bug fix, surfaced by
  `demos/rational.c`).** The sret pointer is a function parameter, so it only lives as `v1`
  in the **entry block** — but a `return <aggregate>` can be in *any* block (inside a loop,
  after an `if`), where `v1` is rebound (e.g. to a loop counter). The original code did
  `gen_memcpy(sret_param, …)` with a fixed value index → it wrote through the wrong value and
  emitted IR that failed verification. Fix: `prepare_func` reserves a hidden 8-byte slot just
  below the spill scratch (`sret_slot = stack_size − SCRATCH_BYTES − 16`); the entry block
  stashes the incoming sret pointer there (like the varargs pointer), and an aggregate
  `return` reloads it from `sp + sret_slot` (the data-SP `v0` is threaded everywhere, so this
  works in any block). Regression-tested (`c_matches_gcc_aggregates`: struct return from a
  loop/after-`if`).

**General `goto`/labels.** Each C label maps to one IR block keyed by chibicc's resolved
`unique_label` (`label_block_of`, reset per function); the block number is allocated on
first reference — label *or* a forward `goto` — which is sound because svm-text resolves
block targets **by name**, not position (`labels: HashMap<String,u32>` over appearance
order). `ND_LABEL` falls into its block (if reachable) then `open_block`s it; `ND_GOTO`
(after the existing break/continue match) branches to the target block, threading the
data-SP + promoted locals via `cvals()` — identical to loops. The ND_BLOCK dead-code drop
now also keeps `ND_LABEL` (a goto target reopens a reachable block). *Limitation:* a label
buried inside a compound statement that is skipped as dead code after a terminator won't be
emitted (goto-into-nested-block); labels at block/function scope — the cleanup/retry/state-
machine idioms — work. With this, the **C ABI (§3d) is feature-complete** for the MVP
subset: indirect calls, by-value aggregates, and goto all land.

**Global pointer initializers / relocations.** A global initialized with a pointer
(`char *p = "..."`, `&global`, `&arr[k]`, function pointers, and arrays/structs of them)
carries a chibicc relocation chain (`g->rel`: `{offset, char **label, addend}`).
`emit_data_segments` now resolves each at compile time — every global's window offset
(`layout_globals`) and function's funcref index (`funcs[]`) is already assigned — and patches
the 8-byte little-endian value (`symbol_value(target) + addend`) into the data image, which
is emitted as an ordinary `data`/`data ro` segment. A function-pointer target resolves to its
funcref index (§3c), so global dispatch tables compose with `call_indirect`. No runtime
relocation step; nothing relocation-specific reaches the IR/verifier/JIT (it's just bytes).
Tests: interp↔JIT differential + native-`cc` oracle (pointer-to-global, array-element
addend, pointer-to-pointer, struct-with-pointer-member, global fn-ptr tables, string-literal
`char*`, array-of-`char*`).

**Fuzzing — data segments now generated.** The generative interp↔JIT differential
(`support/irgen.rs`, shared by the stable `jit_fuzz` test and the libFuzzer `diff` target)
previously emitted `data: Vec::new()`. It now generates 0–3 in-window `data` segments
(rarely `readonly`), so interp↔JIT **data-initialization agreement** is fuzzed — caught
strongly by the existing final-window byte compare — plus the RO-protect fault path (both
backends protect page-granularly, so they agree). This is exactly the surface globals lower
onto. `generator_covers_*` gained assertions that non-empty and read-only data segments are
actually produced (so the coverage can't silently regress).

**Indirect calls (function pointers).** A function designator decays to its `ref.func`
index (an i32 funcref, §3c) widened to the 8-byte C pointer rep (`irty(TY_FUNC)`=i64,
`by_address` true so a "load" is a no-op returning the funcref). A call through a value
lowers to `call_indirect (i64 sp, params…[, i64 va]) -> (ret) <i32-wrapped idx>(csp,
args…)`; the signature **must include the leading data-SP `i64`** so the runtime type-id
check (`table_lookup`) matches the target. A type-confused/forged index is inert — it
traps `IndirectCallType` on both backends (I2; see `c_function_pointer_signature_mismatch_traps`).
The JIT lowers `RefFunc` to an `iconst.i32` and was extended in `ensure_supported`.
(Former coverage gap — *now closed*: the generative `jit_fuzz` exercises `call_indirect` but
historically not `ref.func`, which is why this JIT gap once surfaced only via the C tests. `irgen`
now emits `ref.func k` (arm 24; any function index — the result is a plain i32 that never feeds
`call_indirect`, so the halting-by-construction forward-only call DAG is untouched), and
`generator_covers_*` asserts it is produced, so `ref.func` rides the 4000-seed interp↔JIT
differential. The deterministic pin `jit_diff::jit_matches_interp_ref_func_indirect` and the C-level
`__vm_region_unmap` builtin (`c_region_unmap_builtin`) round out the coverage.)

Anything unsupported is a **hard `error_tok`** (with the AST node kind), by design — we
never emit IR we can't stand behind. The frontend is outside the escape-TCB (§2a): the
verifier re-checks whatever it emits.

---

## 3. The lowering model (read this before extending `codegen_ir.c`)

**Everything-in-memory, with a threaded data-stack pointer** — *then* the SSA-promotion
pass lifts the easy locals back out. The base model is chibicc's own "allocate all locals
to memory first" (DESIGN §3d); promotion (the documented "reverse" pass that matters for
speed) now runs on top of it. **A promoted local is no longer in memory at all:** it is a
real SSA value threaded as a block parameter of every block, exactly like the data-SP (see
"SSA promotion" below). The memory model below still governs every *non*-promoted local
(address-taken, narrow, aggregate, `_Atomic`).

- **Locals live in the window data stack.** Each local gets a **frame-relative offset**
  (`assign_offsets`, from 0). A local is accessed at run time as `sp + offset` via typed
  `load`/`store` (`i32.load`/`store8`/etc. by C type).
- **The data-SP is an explicit IR value**, threaded as **parameter `v0` of every IR
  function and every IR block** (`#define SP "v0"`). DESIGN §3d ultimately wants it
  register-pinned in `vmctx`; threading it as a value is the simple stand-in.
- **A call gives the callee a fresh frame** at `sp + cur_frame` (the caller's frame
  size). This is *the* reason recursion is correct — each activation has its own frame,
  so a parent's locals survive across recursive calls. This was the key bug fixed when
  calls landed: fixed per-function offsets clobbered on recursion.
- **Because state lives in memory, no SSA value crosses a block boundary** — the only
  cross-block value is the data-SP, passed as each block's `v0`. `nv` (value counter)
  **resets per block**; `nb` numbers blocks; `term` tracks whether the current block is
  already terminated (to drop dead code / avoid double terminators).
- **Blocks resolve by label name** in `svm-text` (appearance order = index), so we emit
  blocks sequentially with **forward label references** (`br block7(v0)` before block 7
  exists) — no buffering needed. The **entry block must be first** (index 0).
- **Functions are ordered with `main` first** (so `main` is function index 0, what the
  harness runs); `call` targets a function by this index (`funcs[]` / `func_index`).
- **The harness passes the initial data-SP** (`SP0 = 16`) as `main`'s `v0`. The low
  `[0,16)` window bytes are reserved so `&local` (= `sp + offset ≥ 16`) is never `NULL`.

### SSA promotion (the §3d "reverse" pass — `prepare_func`/`scan`/`undo_compound` + threading)
- **Which locals promote:** a local that is a **full-width scalar** (`int`/`long`/`enum`/
  pointer/`float`/`double`), **never address-taken**, not `_Atomic`, not the hidden
  `__va_area__`/alloca object, and not a synthetic temp. Narrow types (`char`/`short`/
  `_Bool`) stay in memory so their **store truncation** keeps happening; aggregates are
  by-address. `prepare_func` decides this per function and records it by setting the local's
  `offset` to the sentinel **`-(slot+1)`** (a memory local keeps a `≥0` offset).
- **How a promoted local lives:** as a **block parameter of every block** (slot `s` ⇒ `v(s+1)`,
  right after the data-SP `v0`), with `curval[s]` tracking its current SSA value in the
  current block. A read returns `curval`; an assignment rebinds it; `ND_MEMZERO` binds a
  typed zero — **no load/store/memzero is emitted**. This is the same "thread it through
  every block" trick already used for the data-SP, so it is SSA-valid by construction (the
  block param *is* the φ) — no dominance/liveness analysis; Cranelift drops the dead ones.
  `cvals()`/`cparams()` build the arg/param suffixes; every branch site passes `cvals()`.
- **The compound-assignment catch:** chibicc lowers `A op= B` and `A++`/`A--` to
  `tmp = (T*)&A, *tmp = *tmp op B` — taking `&A`, which would block promotion of every loop
  counter/accumulator. `undo_compound` (run by the `rewrite` AST pass before analysis)
  recognizes that exact shape for a **plain-variable** `A` and rewrites it back to the direct
  `A = A op B` (no address). Other lvalues (`a[i] += …`, `s.f += …`, `*p += …`) keep
  chibicc's form — their `tmp` is just a normal (often itself-promoted) pointer.

### Known quirks / inefficiencies (correct, just not optimal — don't "fix" without need)
- **Redundant `memzero`/init for promoted scalars:** chibicc still emits `ND_MEMZERO` then
  the initializer, so `int x = 5;` lowers to a dead `i32.const 0` (the bind) followed by the
  real `5`. For a promoted local these are dead **SSA consts**, not stores, and Cranelift
  DCEs them; for a memory local it's the old store-0-then-store-5. Harmless either way.
- **Over-reserved frames:** every function frame includes chibicc's hidden
  `__alloca_size__` (8 B), and `int main()` (empty parens ⇒ chibicc treats it as
  variadic) also gets `__va_area__` (136 B) — hence `main`'s `cur_frame = 144`. Harmless
  over-reservation; we don't use alloca/varargs yet.
- **Fixed 64 KB window** (`memory 16`) emitted whenever any function has locals. Becomes
  program-driven once a real data-SP base / heap lands.

---

## 4. `codegen_ir.c` map (where to add things)

- `irty(Type*)` → `"i32"`/`"i64"` (LP64: int=i32, long/ptr=i64). Extend for floats.
- `gen_load` / `gen_store` — typed memory access by C type (narrow widths included).
- `gen_addr(node)` — lvalue address as i64. Handles `ND_VAR` (local → `sp+offset`),
  `ND_DEREF`, `ND_COMMA`. **Add `ND_MEMBER` here** for structs.
- `gen_expr(node)` — the big dispatch. Has: `ND_NUM`, arithmetic/bitwise/shift/compare,
  `ND_NEG/NOT/BITNOT`, `ND_CAST` (i32↔i64 only), `ND_COMMA`, `ND_VAR`, `ND_DEREF`,
  `ND_ADDR`, `ND_ASSIGN`, `ND_NULL_EXPR`, `ND_MEMZERO`, `ND_FUNCALL` (direct only).
- `gen_if` / `gen_for` (handles both `for` and `while`) — the block CFG.
- `gen_stmt` — `ND_BLOCK` (drops dead code after a terminator), `ND_EXPR_STMT`, `ND_IF`,
  `ND_FOR`, `ND_RETURN`.
- `gen_func` — signature (`func (i64 sp, params...) -> (ret)`), entry block, param spill
  (or curval bind for promoted params), fall-off-end default `return 0`.
- `prepare_func(fn)` — the per-function analysis: `rewrite` (un-desugar compound assign) →
  `scan` (collect address-taken locals) → classify + lay out (promoted slot sentinel vs
  memory offset) + `stack_size`. Run for each func in `codegen_ir` before `gen_func`.
- `open_block`/`open_merge` + `cvals()`/`cparams()` — block headers and branch args that
  carry the data-SP **and the promoted locals** (`MERGE_VAL = npromo+1` is the carried
  result/switch-value slot, after the promoted ones).
- `codegen_ir` — orders funcs (main first), runs `prepare_func`, emits `memory`, emits funcs.

**chibicc AST facts learned (save you time):**
- `Obj` = function or variable; `Node` = AST node; `Type` (`TypeKind`, `->kind`,
  `->size`, `->is_unsigned`, `->base`, `->return_ty`, `->params`). Enums/structs are in
  `chibicc.h`.
- A declaration `T x = init;` lowers to `ND_EXPR_STMT(ND_NULL_EXPR)` (a VLA-size no-op)
  **plus** `ND_EXPR_STMT(ND_COMMA(ND_MEMZERO, ND_ASSIGN))`. That's why both no-op nodes
  are handled.
- `fn->params` is in **declaration order** (the recursive `create_param_lvars` +
  prepend cancel out). Offsets come from `fn->locals` (which includes params + hidden
  locals). Both are the same `Obj`s, so offsets assigned via `locals` are seen via
  `params`.
- A direct call has `node->lhs->kind == ND_VAR` with `node->lhs->var->is_function`;
  `node->args` is the (already param-cast) arg list; `node->func_ty->return_ty` /
  `node->ty` is the return type. Args are pre-cast to param types by the parser.
- Comparison result type is always `int` (i32); the **op width** comes from the operand
  type (`node->lhs->ty`), so e.g. `i64.lt_s` → i32 result.

---

## 5. C-frontend roadmap — items 1–8 all DONE (the agreed stopping point)

The frontend was taken as far as needed for "a capable VM"; items 1–8 below are complete.
The once-"Still TODO" items have since landed too — by-value aggregate `sret` (D39), general
`goto`/labels, and a real read-only data segment (D40) — leaving only minor inline notes
(`fd`→stream mapping, `%`-width/precision in the mini-printf, narrow-scalar promotion), none of
which block "C runs." History order:

1. ~~**Short-circuit `&&` / `||` and ternary `?:`**~~ — **DONE** (commit after `0f03686`).
   Lowered with option (b): the merge block carries the result as a second block param
   `(sp, v1: ty)`. See `gen_logand`/`gen_logor`/`gen_cond` + `gen_truth`/`gen_expr_as`/
   `open_merge` in `codegen_ir.c`. Tested incl. short-circuit side effects + chained `?:`.
2. ~~**Arrays + structs/unions**~~ — **DONE** (member read/write, indexing, `->`, 2D,
   array-of-struct, initializers). `irty(TY_ARRAY)=i64` (decay); `ND_MEMBER` in
   `gen_addr`/`gen_expr`. **Still TODO here:** by-value aggregate args/returns → hidden
   pointer (`sret`, §3d D39) and whole-struct assignment (`s1 = s2` memcpy) — currently
   only *pointers* to aggregates pass/return. chibicc computes all layout/offsets.
3. ~~**Globals + string literals**~~ — **DONE** (scalar/array/struct globals, mutable
   globals, string literals). Laid out at fixed window offsets in a data region [16,
   `data_end`); a synthetic **`_start`** (function 0) sets up the data-SP and calls
   `main` with the initial data-SP (`data_end`). The harness runs function 0 with **no
   args**. **Update (now done):** globals are emitted as **real IR `data` segments**
   (`emit_data_segments`, replacing the old per-byte `_start` init stores), with string
   literals as page-isolated `data ro` (read-only) segments — the §3a/D40 work that was
   originally TODO here. See §10's "Real read-only data segment" item. **Still TODO:**
   globals holding pointers/relocations.
4. ~~**stdio via the powerbox**~~ — **DONE** (hello-world works). `write`/`read`/`exit`
   are recognized **builtins** in `gen_expr`'s `ND_FUNCALL` (a declared-only prototype is
   enough), lowered to `cap.call` on Stream/Exit. `_start` now takes the capability
   handles `(stdout, stdin, exit)` and stashes them in reserved window slots (offsets
   0/4/8) that the builtins load. The harness (`run_c_full`) grants the caps on two
   `Host`s and runs both backends with `cap_thunk`, asserting outcome **and** stdout/
   stderr agree. **Still TODO:** real `printf` (format parsing), `fd`→stream mapping
   (stderr is not yet distinguished from stdout — `write` always uses the stdout handle),
   and `malloc`/`free` (guest libc over the `map` cap, §3d).
   *Latent bug fixed here:* `ND_MEMZERO` was zeroing locals at their **absolute** offset
   instead of `sp + offset` (harmless until the handle slots occupied low memory).
5. ~~**Floats** (`float`/`double` = f32/f64)~~ — **DONE** (arithmetic, compares, `-`/`!`,
   literals via `node->fval`, locals/params/returns, and all int↔float / f32↔f64
   conversions; float→int is saturating `trunc_sat` for total semantics). `gen_convert`
   is the one place all numeric conversions live (used by casts and `?:` arms).
6. ~~**`break` / `continue` / `switch`**~~ — **DONE**. A `LoopCtx` stack maps a
   break/continue `ND_GOTO` (matched by `unique_label`) to the loop's end/cont block;
   `for`/`while` gained a `cont` block, plus `do`/`while` (`gen_do`). `switch` (`gen_switch`)
   is a dispatch chain threading the value through `(sp, val)` compare blocks, with a
   `case_block_of` map for the body's `ND_CASE` labels; supports fall-through, `case`
   ranges, mid-position `default`, and `continue` passing through to an enclosing loop.
   **Still TODO:** general `goto`/user labels (`ND_LABEL`/non-loop `ND_GOTO`) still error.
7a. ~~**Varargs / `printf`**~~ — **DONE**. Flat-buffer varargs ABI (§3d): a custom
   `include/stdarg.h` (`va_list` = a pointer; `va_arg` = load + bump 8); `__va_area__` is
   now a pointer (chibicc `parse.c` change); `gen_func` adds a hidden trailing buffer
   pointer on variadic functions; the call site marshals promoted args into a buffer
   between the caller/callee frames. `printf` is guest C over `write` (the `LIBC` prelude
   in the test). **Two important fixes landed here:** (a) expression-level control flow
   (`&&`/`||`/`?:`) opens blocks and *stranded* values computed earlier in the same C
   expression — now spilled to a per-frame scratch region (`eval2`/`spill`/`reload`,
   `has_branch`); (b) `if`/`for`/`do`/`while` conditions are normalized to an i32 truth
   via `gen_truth` (a `long`/pointer condition is i64, but `br_if` needs i32). Also: a
   cast to `void` now just discards. **Still TODO:** `fd`→stream mapping, float varargs
   beyond `double`, `%`-width/precision in the mini-printf.
7b. ~~**`malloc`/`free`**~~ — **DONE**, and it needed **no frontend changes**: it is
   ordinary guest C — a bump allocator over a big BSS-global window heap, `free` a no-op
   (the §3d MVP "fixed-size window" allocator). Lives in the test `LIBC` prelude alongside
   `printf`; `calloc` too. (Heap **growth via the `map` capability** has since landed in the
   shipped `frontend/chibicc/include/stdlib.h` `malloc` — see §10 / `demos/heapgrow`; free-list
   reclamation is still deferred.) Demonstrated with a heap-allocated linked list of structs.
8. ~~**(Perf) SSA-promotion pass**~~ — **DONE**. Non-address-taken full-width scalar locals
   are promoted from memory to real SSA values, threaded as block params (see the "SSA
   promotion" subsection in §3). Removes the per-access masked load/store and the redundant
   `memzero` (now dead consts Cranelift DCEs); a hot loop body dropped from ~22 memory ops
   to 0. **Still TODO here:** narrow scalars (`char`/`short`/`_Bool`) stay in memory (we
   don't re-emit store truncation on SSA assignment yet); `volatile` is not honored because
   chibicc discards the qualifier (no regression — the old memory path didn't honor it
   either); and there is no general copy-propagation/DCE beyond what Cranelift does.

---

## 6. Working conventions

- **Gate before every commit:** `cargo fmt --all && cargo fmt --all --check`,
  `cargo clippy --workspace --all-targets` (no warnings), `cargo test --workspace`
  (all green). `codegen_ir.c` is C, so fmt/clippy don't touch it — but
  `make -C frontend/chibicc` must build warning-clean.
- **Commit messages** explain *why*, not just *what*; end with the
  `https://claude.ai/code/session_…` trailer (matches existing history).
- **Don't open a PR** unless asked.
- After pushing, CI is `ci.yml`; it builds the fork + runs the workspace. Check via the
  GitHub MCP tools (`mcp__github__actions_list` / `_get`); the list payload is large, so
  fetch and parse the saved file with `python3 -c "import json; ..."`.
- Recent C-frontend commits for reference: `34d104e` (vendor + expressions), `078dd71`
  (locals/pointers), `ead1bb2` (control flow), `a0c39ad` (functions/recursion); SSA
  promotion is the most recent.

---

## 7. Sanity check to confirm the pickup works
```
make -C frontend/chibicc
printf 'int fib(int n){if(n<2)return n;return fib(n-1)+fib(n-2);} int main(){return fib(10);}\n' > /tmp/t.c
frontend/chibicc/chibicc -cc1 --emit-ir -cc1-input /tmp/t.c -cc1-output /tmp/t.svm /tmp/t.c
cat /tmp/t.svm            # func 0 = _start, func 1 = main calling func 2 = fib; n promotes to v1
cargo test -p svm --test c_frontend   # 64 tests, all green (interp == JIT, and == cc)
cargo test -p svm --test jit_fuzz     # 4000 generated modules, interp == JIT
```
If those pass, you're oriented.

---

## 8. Generative interp↔JIT differential fuzzer (§18 "interpreter-as-oracle")

The JIT is the only component emitting unsafe machine code, so it gets dedicated fuzzing.

- **`crates/svm/tests/support/irgen.rs`** — a generator of **verifier-valid** IR modules
  *by construction*: typed value pool (constants synthesized on demand), branch/return
  args matched to target param types, **forward-only call graph (a DAG)**, and a CFG that is
  forward-only *except* `gen_loop_func`'s one **counted loop** (a strictly-incrementing i32
  counter to a small bound ⇒ still halts by construction). `call_indirect` dispatches only
  forward or type-mismatch-traps. Constants biased to boundary values (0, ±1, INT_MIN/MAX,
  NaN, ±inf); covers the whole scalar op set. `fuzz_one(&mut Gen)` generates → verifies →
  runs interp + JIT → asserts agreement (values + final memory equal; NaN-insensitive; both
  trapping ⇒ agree, kind not pinned). `Gen::from_seed` (stable) / `Gen::from_bytes` (libFuzzer).
- **`crates/svm/tests/jit_fuzz.rs`** — stable-CI loop over 4000 seeds (~1.6s).
- **`fuzz/fuzz_targets/diff.rs`** — libFuzzer target (`cargo +nightly fuzz run diff`).

Found no divergences. **The escape-oracle now lives here too** (§18 *"verified ⇒ cannot
escape"*): for a float-free module with memory, `run_differential` byte-compares the **final
guest window** across interp + JIT (via `run_capture` / `compile_and_run_capture`, seeded
non-zero). When the interpreter — the §4 masking reference — runs to completion, every
access it made was in-window, so the JIT lowering the same masking must leave an identical
window; a mismatch is an access that escaped or was mis-masked. Pinned by
`tests/escape_oracle.rs` and verified non-vacuous (corrupting the JIT mask makes it fail).
Loops/back-edges, `call_indirect`, and `cap.call` — **both** inert/ungranted (⇒ both-`CapFault`)
**and** the success path (a granted Memory cap, valid `map`/`unmap`/`protect`, via the capture+host
wrappers over `svm_run::cap_thunk`, so the cap's window effects ride the escape-oracle) — are now
generated (the trap-kind is no longer asserted when both backends trap — see §10); out-of-
allocation accesses now fault into the guard page and are caught as `MemoryFault` (§4/§5).
Remaining: float-module memory coverage is **deliberately excluded** (NaN bits aren't pinned across
backends → arch-specific; the oracle is about addresses, which integer modules cover — see §10).

---

## 9. Where the project stands vs DESIGN.md (compliance, honest)

Largely compliant; simplifications are the ones the design *sanctions*, deferrals are
incompleteness not contradiction:
- **Phases 2, 3, and 3.5 complete; into Phase 4.** Real C on interp + JIT (Phase 2); the §4
  *large* reserved window + Memory cap + guest-controlled growth + guard-page/signal
  detect-and-kill + RO data (Phase 3); cross-platform parity on **Linux + macOS + Windows** (Phase
  3.5). `malloc` over `map` is the default libc and `SharedRegion` aliasing is done on all three
  OSes. **Phase 4 has started:** the concurrency *primitives* (fibers, 1:1 threads, atomics + the
  C11 ordering surface, futex, a `<pthread.h>`) run on interp (all platforms) + JIT (x86-64 unix,
  aarch64 unix, x86-64 Windows) as mechanism with
  **no VM scheduler** (D56/§12). **§14 nesting has landed on both backends** (sub-windows, the
  attenuable `AddressSpace`, the `Instantiator` incl. recursion, co-fiber children, and
  fault-driven yield — the parent-as-pager *content* supply, hardware faults on the JIT). The
  **Separate-module children** (the host-granted `Module` capability, the "plugin-in-plugin"
  story) are in on both backends, as is **cross-domain `SharedRegion` `create`/`grant`** (guest-
  minted regions, granted into coroutine children — the zero-copy data plane). The genuine
  remainders are Phase-4: honoring weak orderings in execution, isolation tiers, Spectre, SIMD,
  and the language on-ramp.
- **§2a escape-TCB intact:** the frontend is untrusted; all its output is re-verified;
  every memory access is masked, so even a buggy/hostile data-SP cannot escape (the
  data-SP is a plain value, not trusted). Making it an explicit value rather than a
  register-pinned `vmctx` slot is exactly the "lowering detail" §3d calls it.
- **§3d implemented as a documented subset:** everything-in-memory **plus the SSA-promotion
  reverse pass** (non-address-taken full-width scalars → SSA values; narrow scalars and
  address-taken/aggregate locals stay in memory), flat-buffer varargs, guest `malloc` over
  the window, LP64 + pinned `char`/`long double`. The promotion split (SSA value vs
  data-stack slot) is exactly the §3d "local classification" — minus the data-SP being
  register-pinned in `vmctx`, which is still a plain threaded value. **Since the early
  drafts, several once-deferred §3d features have landed:** by-value aggregate args/returns
  by hidden pointer (D39, the `sret` work — §2), a real IR `data` section with const/string
  globals as read-only segments via `protect` (D40 — §10), and general `goto`/labels. **Genuine
  remaining deferrals (incompleteness, not contradictions):** narrow-scalar (`char`/`short`/
  `_Bool`) promotion (they stay in memory for store-truncation), and the data-SP being a threaded
  value rather than register-pinned in `vmctx`. (`malloc` over the `map` cap is now the **default
  guest libc**: the powerbox grants the Memory handle, the `__vm_map`/`__vm_unmap`/`__vm_protect`
  frontend builtins expose it, and the shipped `frontend/chibicc/include/stdlib.h` provides a
  `malloc`/`free`/`calloc`/`realloc` that grows the heap into the reserved tail — any program that
  `#include <stdlib.h>` gets it, cc-identically; `demos/heapgrow` is the showcase.)
- **De-risking moves from §18 now in place:** interpreter-as-oracle differential fuzzing
  (§8), masking-unit fuzzing (`fuzz/mask`), Cranelift backend, **the verifier escape-oracle**
  (verified ⇒ in-window final memory, §8/§10), **and guard-page/signal detect-and-kill**
  (§4/§5, cross-platform — SIGSEGV/SIGBUS on unix, a vectored-exception guard on Windows) so a
  gross out-of-window access faults cleanly rather than corrupting the host.
- **The hard ceiling still holds:** "appears to work" is well-supported now (two-tier C
  diff + generative JIT diff); "is certified secure" remains the separate post-MVP
  workstream §2a/§18 describes — unchanged by this work.

---

## 10. Status & open-work tracker (phases, fuzzing, benchmarking)

A single trackable place for "where are we / what's left," anchored to DESIGN §18's phase
plan. Check items off as they land. (Mechanism details live in the sections referenced;
this is the index.)

### Phase status (DESIGN §18)
- [x] **Phase 1 — core loop:** IR + text/binary + verifier + interpreter.
- [x] **Phase 2 — compilability proof:** chibicc→IR; real C on interp + JIT, two-tier
  tested (interp == JIT == native `cc`); SSA promotion landed (§5 item 8, §3).
- [x] **Phase 3 — Solid MVP:** the MVP remainder below all landed — large reserved window +
  Memory cap + guest-controlled growth, guard-page/signal detect-and-kill, RO data segments, the
  verifier escape-oracle, by-value aggregates (`sret`) + general `goto`. (README/§9 call Phase 3
  complete; what follows in the per-item list is the evidence.)
- [x] **Phase 3.5 — Cross-platform parity (Linux + macOS + Windows all GREEN):** the full `cargo
  test --workspace` passes on `ubuntu-latest` (x86-64 / 4 KiB), `macos-latest` (ARM64 / 16 KiB), and
  `windows-latest` (x86-64 / 4 KiB) in CI. Confinement masking is portable (§16/D51); only the
  non-TCB PAL differs, and all three PALs now reserve/commit/protect + recover from a guard fault.
  The svm-run `MprotectWindow` Memory-cap backend (`map`/`unmap`/`protect`/`page_size`) is now
  **cross-platform** — `mprotect`/`madvise` on unix, `VirtualAlloc(MEM_COMMIT)`/`VirtualProtect` on
  windows, sharing one software page-state map; the 4000-seed interp/JIT differential grants the
  Memory cap on every runner, so guest-driven growth + RO isolation are exercised on Windows too.
  Remaining polish (not a blocker): drop `continue-on-error` from the now-green `cross-os` matrix
  legs and fold them into gating (a one-line, maintainer-applied workflow edit).
  - **macOS (ARM64 / 16 KiB pages) is GREEN** — `macos-latest` runs the **whole** `cargo test
    --workspace` clean, including the re-enabled `c_frontend` differential suite (interp == JIT ==
    native `cc`) and the `escape_oracle`/`jit_diff` parity oracles. This closed out DESIGN §4 "pin
    page size" via the **host-page-default**: backends query the host MMU granularity at runtime so
    they agree page-for-page on any host (4 KiB / 16 KiB / …):
    - `svm-jit/src/mem.rs` is a portable window model over a small **PAL** seam
      (reserve/commit/protect/release + install_guard/run_guarded); the unix impl queries the host
      page; a platform-agnostic guard conformance test drives the window+guard directly (no JIT).
    - `svm-interp`'s `Mem` replaced `const PAGE = 4096` with the host page via the *safe* `page_size`
      crate (keeps `#![forbid(unsafe_code)]`); `svm-run`'s `MprotectWindow` queries `sysconf` and
      operates on whole host-page ranges in `map`/`unmap`/`protect`.
    - `unmap` now **explicitly zeroes** the page range before `MADV_DONTNEED`: that syscall releases
      anonymous backing on Linux (re-read = 0) but is only advisory on Darwin (stale bytes survive),
      which diverged the escape-oracle on 16 KiB. The zero makes both platforms agree; the advise is
      then a pure footprint hint.
    - The chibicc frontend emits portable IR and can't know the host page, so it **pins its
      RO-isolation boundary (`DATA_PAGE`) and heap-growth granularity (`__SVM_PAGE`) to the largest
      common host page (16 KiB)** — a multiple of 4 KiB, so 4 KiB hosts are unaffected (just coarser)
      while on 16 KiB the RO segment never shares a host page with writable data (no over-protection
      fault) and `malloc` growth never re-zeroes a live 16 KiB page.
  - **Windows (x86_64 / 4 KiB) is GREEN.** The PAL is pure Rust via `windows-sys`
    (`VirtualAlloc(MEM_RESERVE/COMMIT)` + `VirtualProtect(PAGE_NOACCESS)` + an `AddVectored­Exception­
    Handler` guard with `RtlCaptureContext` as the longjmp-equivalent recovery — no C shim, so it
    stays check-able from Linux via `cargo check --target x86_64-pc-windows-gnu`). Two runtime bugs
    were found + fixed from CI alone: (a) the guard AV'd **inside `RtlCaptureContext`** because
    windows-sys types `CONTEXT` `#[repr(C)]` only, but x86-64 `CONTEXT` must be **16-byte aligned**
    (it embeds XMM `M128A` state stored with aligned `movaps`); a bare stack local landed 8-mod-16
    and faulted — fixed with a `#[repr(C, align(16))]` wrapper. (b) stdio produced **empty output**
    because `cap_thunk` passed `gm = None` on non-unix, so a `Stream` write had no view of the guest
    window — first fixed with a portable `WindowMem`, since **superseded** by the full Windows
    Memory-cap backend (placeholder-aware commit / `VirtualProtect`, sharing the unix path's
    software page map), so guest-driven `map`/`unmap`/`protect`/growth + RO isolation now work on
    Windows and are covered by the interp/JIT differential. §13 `SharedRegion` aliasing is wired on
    windows too now (`MapViewOfFile3` over a placeholder reservation — issue #1). Tier-1 MPK stays
    Linux-only (degrades to tier 0/3 elsewhere).
  - **CI matrix is live** (the maintainer applied the workflow — needs the `workflows` token scope):
    the gating ubuntu job also runs the windows cross-`check`+clippy, and a `cross-os` job
    builds+tests on `windows-latest` + `macos-latest` (still `continue-on-error` — now safe to make
    gating since both are green). Fixes it drove along the way: (a) `cc` was a `cfg(unix)` *build*-dep
    — that cfg matches the **host**, so a windows host never got the crate and `build.rs` failed (the
    linux cross-check can't catch a host-only issue); made it an unconditional `[build-dependencies]`
    (the C shim compile stays target-gated on `CARGO_CFG_UNIX`). (b) `c_frontend` needs a unix C
    toolchain (`make`+`cc`) → `#![cfg(unix)]` (runs on Linux + macOS; skipped on Windows).
- [ ] **Phase 4 — post-MVP (started):** the **concurrency primitives** have landed (fibers, 1:1
  threads, atomics + C11 ordering surface, futex, a `<pthread.h>` libc — interp on all platforms,
  JIT on x86-64/aarch64 unix + x86-64 Windows; see below), and **§14 nesting** has landed on both
  backends (sub-windows, `AddressSpace` + attenuation, the `Instantiator` incl. recursion,
  co-fibers, fault-driven yield; see §10). The rest (isolation tiers, Spectre, split-host, SIMD,
  GPU, the language on-ramp) is deferred, developed against the parity matrix.

### Phase 3 / MVP remainder (what's left to call it a "Solid MVP")
- [x] **Production trap-catching (memory)** — *done (unix)*: the JIT window is now `mmap`'d
  with a trailing `PROT_NONE` **guard page**, and the entry runs under a SIGSEGV/SIGBUS
  handler (`crates/svm-jit/src/{mem.rs,trap_shim.c}`, a small `cc`-built C shim for sound
  `sigsetjmp`/`siglongjmp`). A fault in the window's guarded range unwinds out of the call as
  `TrapKind::MemoryFault` — §5 **detect-and-kill**, host survives — instead of corrupting it.
  Confinement is still the masking lowering; the guard is the safety net (width-overrun at
  the top now faults cleanly, and a masking/elision bug faults locally instead of corrupting
  the host). `cfg(unix)` at the time; *since ported* — Windows has the same model via a
  `VirtualAlloc2` placeholder reservation + a Vectored Exception Handler (Phase 3.5 below).
  Verified non-vacuous by `escape_oracle::guard_page_fault_is_detect_and_kill`; whole suite +
  4000 fuzz seeds green (the handler is exercised by width-overruns). **Not yet:** the
  *perf*-unlocking guard-when-bounded (needs a large window — below); div/rem/trunc still use
  explicit in-code trap checks (correct; converting them to #DE faults is optional).
  - **Fixed — software-trap propagation across calls (found by the differential fuzzer):** a
    *software* trap (the host trap cell — `cap.call` CapFault/`Exit`, div-by-zero, int-overflow,
    bad float→int, `unreachable`, indirect-call type mismatch) sets the cell and `return`s zeros
    from *its* clif function. The caller did **not** re-check the cell after a `call`/`call_indirect`,
    so a trap raised in a **callee** was swallowed: the caller ran on with bogus zero results, and a
    later *successful* `cap.call` (which resets the cell to 0) could erase it — the JIT then returned
    where the interpreter stays trapped. Net: a guest could neutralize any trap (even `exit`) by
    wrapping it in a function call. Fix: `emit_trap_propagate` after every `call`/`call_indirect`
    (mirroring `cap.call`), so a callee trap unwinds the whole guest stack immediately. Pinned by
    `jit_diff::cap::jit_trap_in_callee_propagates_through_caller` + the 4000-seed differential (the
    generator now also emits the `page_size` query, which is what surfaced the cell-reset).
- [x] **Real window / Memory capability + growth** — *done*: page size is the **host MMU
  granularity** (§4 "pin page size" → host-page default; all backends query it so they agree
  page-for-page on 4 KiB / 16 KiB hosts), and the guest can **read it at runtime** — `Memory` op 3
  `page_size() -> i64` (the `__vm_page_size` builtin); the shipped `<stdlib.h>` `malloc` caches it
  for its growth granularity instead of a hardcoded constant, so a guest adapts to the real page.
  The
  *large* reserved window (`DEFAULT_RESERVED_LOG2 = 40`, mask `reserved - 1`), and real
  `map`/`unmap`/`protect` **including guest-controlled growth into the reserved tail** — the §1a
  "sparse address space / lazy page supply" capability. The interp `Mem` (reference) commits pages
  sparsely across all of `[0, reserved)`: confinement masks the final address into `[0, reserved)`
  while per-page committed-ness (the page map) is the functional bound, so a `map` past the initial
  prefix grows the window and an uncommitted access faults. The JIT side is a production
  `svm_run::MprotectWindow` — real `libc::mprotect` across the reserved range + `MADV_DONTNEED` on
  `unmap`, mirrored by a software page map so §7 cap-buffer borrows fail closed (`-EFAULT`) instead
  of faulting the host — wired into the production `cap_thunk` (was a no-op `WindowMem`) and driven
  by `jit_diff` (the cap-thunk ABI gained `mem_reserved`). Differentially fuzzed across the
  prefix+tail (`jit_cap_memory_protect_map_unmap_differential`, 800 seeds) with a concrete guest
  consumer (`jit_cap_memory_growth_round_trips`: map at 1 MiB, store/load round-trip,
  unmap→fault). **Physical demand paging is already free** (the JIT reserves `PROT_NONE` +
  `MAP_NORESERVE`; the kernel lazily zero-fills touched RW pages), so no fault-driven commit
  machinery was needed. The Memory cap is surfaced in the *main* irgen fuzzer (arm 19, now spanning
  prefix **and** reserved tail), and the `_with_host` escape-oracle snapshot was **extended to grown
  tail pages** (the low `SNAP_CAP` = 256 KiB, not just the backed prefix; both backends `commit` the
  span so a grown/`unmap`-ed page reads back instead of faulting). Because a *random* completing run
  rarely leaves non-zero tail content (verified: a corrupt-a-tail-byte probe didn't fire in 4000
  seeds), the non-vacuous pin is the deterministic, cross-platform
  `jit_diff::jit_cap_memory_escape_oracle_grown_tail` (grow a tail page, store a marker, assert both
  windows agree *and* hold the marker). **§13 SharedRegion — interp reference landed (slice 1):** a
  host-granted `SharedRegion` capability (`iface::SHARED_REGION = 4`; op 0 `map(win_off, region_off,
  len, prot)`, 1 `unmap`, 2 `len`, 3 `page_size`) aliases a shared host buffer into the window via a
  new `PageProt::Backed { region, region_off, writable }` — the access path is unchanged (loads/stores
  redirect where a page's bytes live, zero overhead), so the same region mapped at two window offsets
  names the same bytes (the magic-ring-buffer primitive). White-box tests in `prot_tests` +
  end-to-end `svm/tests/shared_region.rs` (with a non-vacuous control). **Slices 2–3a (JIT + unix)
  landed:** `MprotectWindow::map_region` aliases via a **real shared mapping** — `mmap(MAP_SHARED |
  MAP_FIXED)` of the region's `os_fd` over the window range, so two mappings name the same physical
  pages (true hardware aliasing; the mapping persists across `cap.call`s — the per-call window is
  rebuilt but the OS mapping + the region fd held by the `Host` backing are not). The backing is
  `svm_run::new_shared_region` over an anonymous fd — `memfd_create` on Linux, an `shm_unlink`ed
  `shm_open` object on macOS (`ShmBacking`); installed via `Host::grant_shared_region_backed`. The
  interp↔JIT differential `jit_diff::jit_cap_shared_region_aliases_differential` pins it
  non-vacuously. **§13 windows — DONE (issue #1).** `MprotectWindow::map_region` now aliases on
  windows via **placeholder reservations**: the JIT window is reserved as a `VirtualAlloc2(
  MEM_RESERVE_PLACEHOLDER)` placeholder (`svm-jit/src/mem.rs`), and `map_region` frees the target
  sub-range back to a placeholder (`VirtualFree(MEM_PRESERVE_PLACEHOLDER)`, whether it was the
  committed prefix or an untouched tail) then replaces it with a view of the section
  (`MapViewOfFile3(MEM_REPLACE_PLACEHOLDER)`) — true hardware aliasing, at the **64 KiB allocation
  granularity** `MapViewOfFile3` requires (the guest aligns to `region_page_size`, op 3, which now
  reports that granularity on windows). The backing is `svm_run::new_shared_region` over a
  `CreateFileMapping` section (`WinShmBacking`); the `SharedBacking` trait gained `os_section`. The
  placeholder rework also touched the **commit path** — a plain `VirtualAlloc(MEM_COMMIT)` cannot
  commit a placeholder, so `svm-jit::win_commit_rw` does an idempotent `VirtualQuery`-driven split +
  `MEM_REPLACE_PLACEHOLDER` commit (reused by `svm-run`'s growth path). The differential
  `jit_diff::jit_cap_shared_region_aliases_differential` is now `#[cfg(any(unix, windows))]` and the
  old `#[cfg(windows)]` `-EINVAL` pin is gone. **Validated locally** by cross-compiling to
  `x86_64-pc-windows-msvc` (`cargo-xwin`, MS SDK now fetchable in this environment) and running the
  whole suite under **wine** — escape_oracle, the 4000-seed `jit_fuzz`, the Memory-cap differential,
  and the §13 alias differential all green — **and confirmed on the real `windows-latest` CI** (PR #2,
  merged: the `build · test (windows-latest)` gate passed, all three OS legs green). The original
  playbook is preserved below as the design record.
  **Still left (Phase 4, not MVP blockers):** *(both since **landed**)* — fault-driven *content*
  supply (a parent as pager, `userfaultfd`-style/§14): `spawn_demand_coroutine` + fault-driven
  yield on both backends; and cross-domain `SharedRegion` `create`/`grant`: guest-minted regions
  (`AddressSpace.create_region`) granted into coroutine children (`SharedRegion.grant`). **`malloc` over `map` is the default guest libc** — the powerbox
  grants the Memory handle, the `__vm_map`/`__vm_unmap`/`__vm_protect` builtins expose it
  (codegen_ir.c), and the shipped `frontend/chibicc/include/stdlib.h` provides a map-growing
  `malloc`/`free`/`calloc`/`realloc` to any program that `#include <stdlib.h>`; `demos/heapgrow`
  grows a guest heap megabytes past the initial window cc-identically
  (`demo_heapgrow_matches_native`).

### §13 Windows — playbook (issue #1) — ✅ DONE (kept as the design record)

> **Done.** Implemented as described below, with one refinement the playbook didn't anticipate:
> `MapViewOfFile3` requires **64 KiB allocation-granularity** alignment (not the 4 KiB page) for both
> the placement address and the section offset — so `SharedRegion` op 3 (`region_page_size`) reports
> the allocation granularity on windows and the guest aligns to it (`memory 17` in the tests so two
> granules fit). **Local windows test loop (this environment):** `cargo install cargo-xwin`, then
> `WINEPREFIX=… CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_RUNNER=wine cargo xwin test --target
> x86_64-pc-windows-msvc -p svm …` cross-compiles under real MSVC and runs the test binaries under
> **wine** (apt `wine64`). Wine implements `VirtualAlloc2`/`MapViewOfFile3` placeholders *and*
> delivers access-violations to the VEH guard, so it exercises the real placeholder + view + guard
> paths — a fast inner loop that made CI a formality rather than the only validator.



**Goal:** wire the JIT zero-overhead `SharedRegion` mapping on Windows so
`MprotectWindow::map_region` aliases (today it returns `-EINVAL` there). Then un-gate
`jit_diff::jit_cap_shared_region_aliases_differential` (`#[cfg(unix)]` → `#[cfg(any(unix, windows))]`)
and delete the `#[cfg(windows)]` `-EINVAL` pin in `svm/tests/shared_region.rs`. The interp reference
+ all-unix JIT path are already done and green; this is the last platform leg.

**Why it stalled here (toolchain), and the agreed fix.** Windows needs **placeholder reservations**
(you cannot map a fixed-address view into a plain `VirtualAlloc(MEM_RESERVE)` range). That is runtime
behavior — compile-success ≠ correctness — and this environment has **no local Windows runtime**:
`cargo-xwin` (local `x86_64-pc-windows-msvc`) is **blocked by the network policy (HTTP 403 fetching
the MS SDK)**, and `windows-gnu` only compiles/links (no run). **Plan: do this work in an environment
with network access for `cargo-xwin`** (the user is provisioning one). There, `cargo xwin build/test
--target x86_64-pc-windows-msvc` gives a real local MSVC compile (and, with a Windows runner or
wine-msvc, possibly run); the gating runtime check remains the `cross-os` `windows-latest` (MSVC) CI
job, which runs the **full suite on every `pull_request`** — so develop on a branch and iterate via
PR CI with main untouched.

**APIs are available now.** `windows-sys 0.59` already declares `VirtualAlloc2`, `MapViewOfFile3`,
`UnmapViewOfFile2`, `CreateFileMappingW`, and the `MEM_{RESERVE,REPLACE,PRESERVE}_PLACEHOLDER` /
`MEM_COALESCE_PLACEHOLDERS` consts. Add the **`Win32_System_SystemServices`** feature (for
`MEM_COALESCE_PLACEHOLDERS`) to `crates/svm-jit/Cargo.toml` and `crates/svm-run/Cargo.toml`;
`Win32_System_Memory` (already present) covers the rest. `windows-sys` bundles import libs, so even
`windows-gnu` links these — local compile/link is checkable without msvc.

**The hard part — cross-layer placeholder state.** Two layers operate on the *same* window and both
must speak "placeholder":
- `crates/svm-jit/src/mem.rs` (`mod pal`, `#[cfg(windows)]`): `reserve` (currently
  `VirtualAlloc(MEM_RESERVE, PAGE_NOACCESS)`), `commit_rw`, `protect`, `release`, plus the guard page
  and the snapshot `restore_rw`/`read_low`.
- `crates/svm-run/src/lib.rs` (`MprotectWindow`, `#[cfg(any(unix, windows))]`): `map`/`unmap`/
  `protect` (hardware via `VirtualAlloc`/`VirtualProtect`) and the new `map_region`.

**Suggested two-PR split (each green on `windows-latest` before merge):**
1. **Placeholder allocator (no SharedRegion yet).** Change svm-jit's Windows `reserve` to
   `VirtualAlloc2(NULL, NULL, total, MEM_RESERVE | MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS, NULL, 0)`.
   Make `commit_rw` materialize private committed RW *inside* the placeholder — split to the exact
   sub-range with `VirtualFree(addr, size, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)` then
   `VirtualAlloc2(addr, size, MEM_RESERVE | MEM_COMMIT | MEM_REPLACE_PLACEHOLDER, PAGE_READWRITE,
   NULL, 0)` — and on the unmap/decommit path restore the placeholder (`VirtualFree(MEM_RELEASE |
   MEM_PRESERVE_PLACEHOLDER)`) and coalesce adjacent placeholders
   (`VirtualFree(MEM_RELEASE | MEM_COALESCE_PLACEHOLDERS)`). `release` stays
   `VirtualFree(base, 0, MEM_RELEASE)`. **Success = the existing Windows Memory-cap tests
   (`jit_diff` cap module, `jit_fuzz`, growth) stay green** — proving the rework is transparent to
   non-shared paths. This PR is the real de-risk; expect to iterate the split/replace/coalesce
   granularity (placeholders split/coalesce in *whole pages*, and `MEM_REPLACE_PLACEHOLDER` requires
   the target be a placeholder of *exactly* the requested range).
2. **`map_region` + region backing.** In `MprotectWindow::map_region` (Windows branch), split the
   target placeholder and `MapViewOfFile3(hSection, GetCurrentProcess()?/NULL, base+win_off,
   region_off, plen, MEM_REPLACE_PLACEHOLDER, PAGE_READWRITE|PAGE_READONLY, NULL, 0)`. Add a Windows
   `SharedBacking` (alongside unix `ShmBacking`) over `CreateFileMappingW(INVALID_HANDLE_VALUE, NULL,
   PAGE_READWRITE, sizehigh, sizelow, NULL)` (a pagefile-backed section); `os_fd`'s `i32` return is
   unix-shaped, so either widen the trait to carry an OS handle (e.g. `os_section(&self) ->
   Option<*mut c_void>` returning the `HANDLE`) or add a Windows-specific accessor — **prefer a small
   trait tweak** so `map_region` stays platform-clean. `read_byte`/`write_byte` map the section once
   via `MapViewOfFile`. Wire `new_shared_region` for Windows. Then un-gate the differential + drop the
   pin test.

**Debuggability (no debugger on CI):** thread `GetLastError()` into distinct return codes / panic
messages (e.g. `EINVAL - (err as i64)` or a logged step id) so a red `windows-latest` run names the
failing call + error code in the test output.

**Gotchas to expect:** `MapViewOfFile3`/`VirtualAlloc2` live in `api-ms-win-core-memory-l1-1-6.dll`
(Win10+; fine on `windows-latest`); offset/len must be page-granular (already true via `prot_pages`);
the section must be ≥ `region_off + plen` (size the `CreateFileMapping` page-rounded, mirroring unix
`ShmBacking`'s `cap`); on teardown the window's single `VirtualFree(MEM_RELEASE)` must still unwind
views + placeholders cleanly (may need explicit `UnmapViewOfFile2(.., MEM_PRESERVE_PLACEHOLDER)` per
mapped region before releasing — verify on CI). Also handle the latent **`unmap`-of-region** case
(unix has it too): unmapping a region-mapped page should restore an anonymous/placeholder page, not
leave a shared view — add a unix test for this alongside the Windows work.

- [x] **Verifier escape-oracle fuzzer** — *done*: the differential now byte-compares the
  final guest window across interp + JIT (verified ⇒ in-window), in the 4000 stable seeds
  (every push) and the `diff` libFuzzer target. See Fuzzing below.
- [x] **Real read-only data segment (§3a / D40) — *done*.** The IR has a `data [ro] <off> "<bytes>"`
  section (`svm_ir::Data`, text/encode/verify); both backends place segments at instantiation and
  map `readonly` ones RO (interp page-map / JIT `mprotect`); the chibicc frontend emits one `data`
  segment per global (string literals → `data ro`, page-isolated) and no longer byte-stores in
  `_start`. A C write to a string literal detect-and-kills on both backends
  (`c_frontend::c_write_to_string_literal_faults`).
- [ ] *(optional, deferred even within MVP — not blockers)* by-value aggregate args/returns
  (`sret`, D39); general `goto`.

> **Ceiling reminder (§18):** the MVP target is *"appears to work"* — well-evidenced now.
> *"Is certified secure"* is **not** an MVP deliverable; it's a separate, open-ended
> post-MVP workstream (expert review + audit). Green tests ≠ secure.

### Phase 4 / post-MVP (concurrency primitives landed; the rest deferred)
- [x] **Concurrency — primitives DONE (mechanism only, no VM scheduler — D56/§12).** Through the
  whole pipeline (IR / text / binary / verify) and **both backends**: fibers (`cont.new`/`resume`/
  `suspend`), threads (`thread.spawn`/`join`), linear-memory **atomics** (load/store/rmw×6/cmpxchg,
  i32/i64) with the full **C11 ordering** surface + `atomic.fence`, and a **`wait`/`notify` futex** —
  plus a guest **`<pthread.h>`** (create/join/mutex/cond) in the libc, so real multithreaded C runs
  end-to-end. Two execution models, reconciled by D56:
  - **Interpreter** — an **M:N green-thread executor** (`Scheduler`, bounded worker pool, parked
    continuations, `MAX_VCPUS = 1<<16`) that doubles as the **deterministic oracle**: `run_scheduled`
    (seeded interleaving sweep) + `explore_all` (exhaustive stateless model checker, now with **DPOR** —
    see below). All platforms.
  - **JIT** — fibers via `svm-jit/src/fiber_rt.rs` over the `svm-fiber` stack switch, threads via
    `svm-jit/src/os_thread_rt.rs` as **1:1 OS-thread vCPUs** (D56 *removed* an earlier JIT M:N
    executor — `thread_rt`/`par` — as a re-litigation of D22), and the condvar futex (loom-checked).
    Runs on **x86-64 unix, aarch64 unix (macOS), and x86-64 Windows** — three hand-written `svm-fiber`
    switches (SysV / AAPCS64 / MS-x64), all CI-green; other targets bail `Unsupported`. Differentially
    tested against the interp
    (`jit_threads.rs`, `jit_fibers.rs`) — TSan can't instrument JITted code, so JIT concurrency leans
    on the differential + invariant stress + loom on the glue, not TSan; concurrent C is verified both
    real-executor and seed-swept.
  - **The §5 fuel/epoch kill-path is DONE on the JIT** (it was the "mid-flight preemption kill-path"
    open item): the lowering polls a host-owned interrupt cell at loop back-edges + function entries,
    so a host watchdog stops a runaway guest with `OutOfFuel` — and it reaches **every JIT execution
    context** (root vCPU, sibling vCPUs incl. ones *parked* in a futex `wait`/`join`, and nested §14
    children, which poll the parent's cell). Opt-in + guest-undisableable; the CLI arms it via
    `SVM_DEADLINE_MS`. See §10's tracker (next-pickups item 3 tail) for the full write-up.
  - **Guest-built M:N — both flavors DONE (worked examples).** The design decision (two primitives;
    "stackless tasks" add none; the two M:N flavors; the *Proposed* migratable-fiber path for stackful
    work-stealing) is **D57** + `SCHEDULING.md`. Both schedulers are *entirely guest code* over the
    VM's primitives — proof D56's "primitives, not policy" composes — and run identically on the interp
    (M:N oracle) and JIT (real OS threads):
    - **Demo 1 — `demos/mn_sched`: sharded (thread-per-core), *stackful*.** 4 `thread.spawn` workers,
      each round-robining 8 `cont.*` fibers (yield + increment a shared atomic; `4·8·32 = 1024`). Tasks
      pinned per worker (fibers are thread-affine). `c_frontend::c_guest_mn_scheduler_demo` +
      `run::demo_mn_scheduler_runs`.
    - **Demo 2 — `demos/work_stealing`: work-stealing, *stackless*.** tokio-style — a global injector +
      per-worker deques + stealing; tasks are state-machine structs (just data) that migrate freely
      between threads (a pointer hand-off, safe by construction — **no VM change**, the D57
      migratable-fiber primitive is not needed for stackless). `16·16 = 256`, and the exact total
      proves no task was lost/double-run as they migrated. `c_frontend::c_guest_work_stealing_demo` +
      `run::demo_work_stealing_runs`.
    - *Finding surfaced by the demo (now **FIXED**):* the shipped MVP `malloc` (a bump allocator) was
      **not thread-safe** — concurrent `malloc` from worker threads corrupted the heap, so the demos
      pre-allocated on the main thread to sidestep it. `include/stdlib.h`'s `malloc` is now thread-safe:
      a **lock-free atomic-bump** fast path (`__vm_atomic_add` on the bump pointer claims a unique
      `[hdr, end)`, so concurrent callers never overlap) with the rare **page growth** serialized by a
      spinlock (`__vm_atomic_cas32`) — a page is mapped exactly once (re-mapping would re-zero live
      data) and `__svm_committed` is published only *after* the pages are mapped. A single-threaded
      caller pays only uncontended atomics and never pulls in the thread runtime (atomics don't mark a
      module threaded). Demo `crates/svm-run/demos/malloc_threads` (4 vCPUs × 64 allocs, per-block
      patterns, main re-checks every byte for an overlap clobber) + test
      `c_frontend::c_guest_thread_safe_malloc` (0 corrupt on both backends; the old racy bump scored 11
      under the same load — non-vacuous).
  - **Async submit/complete ring (§9/§12) — COMPLETE (increments 1–3c, mechanism + runtime, both backends).** An `IoRing` capability (iface 9,
    `Host::grant_io_ring`); `op 0 submit(sq_ptr, n, cq_ptr)` runs `n` **deferred `cap.call`s** (each a
    64-byte SQE in the window) through the *same* capability dispatch and writes 32-byte CQEs — so the
    JIT gets it for free (a generic `cap.call` through the thunk; `io_ring_submit` recursively dispatches
    via `cap_dispatch_slots`). One boundary crossing for `n` ops (the §1a amortization). Synchronous +
    in-order ⇒ deterministic ⇒ differentially tested (`io_ring.rs`: 8 batched `Clock.now` total 28 on
    both backends; the `completed` count).
    - **Increment 2 — the bounded blocking-offload pool (DONE).** `submit` now classifies each SQE:
      window-/`&mut Host`-touching ops (Clock, Memory, Stream, …) still run **inline** on the submit
      thread in SQE order, but **`Blocking` SQEs** (a new mock synchronous-only capability, iface 10 =
      `BLOCKING`, `Host::grant_blocking`; op 0 `work(arg) -> mix(arg)`, window-independent +
      `&mut Host`-free) are handed to a lazily-created **`OffloadPool`** of `OFFLOAD_POOL_THREADS = 4`
      long-lived worker threads and run **concurrently** (waves of K) — the §12 path-2 "0 blocked *vCPU*
      threads" win (the guest's one vCPU parks on the single `submit`; the host pool absorbs the
      blocking). Window reads (SQE parse) + writes (CQE) stay on the submit thread and each `Blocking`
      result is a deterministic pure transform, so the final window is **identical to running every op
      inline** — the interp↔JIT differential (the §18 oracle) is preserved (`io_ring.rs`:
      `offload_batch_matches_inline_on_both_backends`). Overlap is proven **deterministically** (no
      timing flakiness) via a width-K rendezvous `Barrier` baked into the mock op: submit exactly K
      blocking ops and assert each backend's pool reached `max_active == K`
      (`offload_pool_overlaps_blocking_ops_on_k_threads`). The op is also an ordinary inline `cap.call`
      (`blocking_direct_cap_call_runs_inline`) and a forged `Blocking` handle is inert on the offload
      path (`offload_forged_blocking_handle_is_inert`, the I2 check). Pool internals: per-worker
      channels (a shared `Mutex<Receiver>` would serialize the blocking `recv`s); `Drop` joins the
      workers. Implementation entirely in `svm-interp` (the shared `Host`), so both backends get it for
      free through `cap_thunk` — no JIT/`svm-run` change.
    - **Increment 3a — async submit + fiber parking, interp (DONE).** The asynchronous path: *an I/O
      completion is a futex notify* (DESIGN §12). Two new IoRing ops (op 0 `submit` unchanged): op 1
      **`submit_async(sq_ptr, n, counter_addr)`** kicks the offloadable (`Blocking`) SQEs onto the pool
      and **returns immediately** with the count submitted (inline SQEs still run on the submit thread);
      each completion posts its CQE to a host-side `RingState` and atomic-increments the 4-byte in-window
      futex **completion counter**, and an *offloaded* completion additionally `notify`s the counter key
      to **wake a vCPU parked in `wait`** on it. op 2 **`reap(cq_ptr, max)`** pops ready completions and
      writes CQEs into the window *on the vCPU thread* — so the single counter atomic is the only
      cross-thread window write. The guest parks with the existing `i32.atomic.wait`; the wake is
      race-free via the scheduler's existing **compare-under-lock** futex guard (worker writes the
      counter, visible to the park's value-check, *before* it notifies ⇒ no lost wakeup — the same
      protocol a guest `atomic.store; atomic.notify` uses, already battle-tested). Wiring: `GuestMem`
      gained `async_counter(addr) -> Option<(Arc<Region>, key)>` (the `Send+Sync` handle a worker bumps
      the counter through — the same path cross-vCPU atomics take — `Some` only for a normal aligned
      anonymous writable page; default `None` ⇒ no async support, `submit_async` returns `-EINVAL` and a
      guest falls back to the synchronous submit); `Host` gained an `async_notify` hook that `drive`
      installs as `Scheduler::notify` (and clears + quiesces the pool at run end so no worker still holds
      the window backing); `OffloadPool` gained fire-and-return `dispatch` + in-flight tracking +
      `quiesce`; `Binding::IoRing` now carries its `RingState` index. Tests (`io_ring.rs`):
      `async_submit_parks_then_pool_notify_wakes_and_reaps` (a vCPU parks, the pool overlaps 4 blocking
      ops `max_active == K` and wakes it via `notify`, reaps `Σ mix(i)`, resolving far under the 10 s
      wait timeout ⇒ the wake was notify-driven not the timeout fallback; 0/20 flake runs) and
      `async_submit_returns_submitted_count`.
    - **Increment 3b — JIT parity (DONE): true cross-thread fiber wake.** The same fiber-parking on the
      JIT: an offload worker wakes a JIT **OS-thread vCPU** genuinely parked in `atomic.wait` on the
      counter. The pool lives in the embedder's `Host`; the JIT futex lives in svm-jit's per-run
      `Domain`. The new **`svm_jit::AsyncHostHooks`** seam bridges them: 3a's interp-specific return is
      generalized to a backend-neutral **`svm_interp::AsyncCounter`** (`increment` atomic-bumps the
      counter; `key` is the parking key — a window offset on the interp via `Region`, the absolute
      window address `phys` on the JIT via a raw atomic, each what that backend's `wait`/`notify`
      value-check reads). `run_inner`, after the thread `Domain` is up, calls `hooks.install_notify`
      with a hook that invokes the `Domain`'s `thread_notify(phys, count)`, and after `join_all` (before
      the window/`Domain` are freed) calls `hooks.finish` to drain the pool + drop the hook (no
      use-after-free). svm-run provides `PhysCounter` (`MprotectWindow::async_counter`) + `HostAsyncHooks`
      (the `Host`-backed seam impl); new entry point
      `compile_and_run_capture_reserved_with_host_async`. Reuses the JIT futex's existing
      compare-under-lock guard, so the wake is race-free (worker bumps the counter before it notifies).
      Test: `async_submit_parks_and_reaps_on_both_backends` (interp + JIT both return `Σ mix(i)` and
      overlap on their pools `max_active == K`; 0/25 flake; loom + windows cross-check green). The CQE
      **byte layout is not** cross-backend-compared — async completion *order* is nondeterministic, so
      only the order-invariant reaped **sum** is an invariant (the synchronous `submit` keeps its
      full-window compare).
    - **Increment 3c — the async event-loop runtime in real C (DONE): the async ring (B) is complete.**
      `crates/svm-run/demos/async_io/async_io.c` — one vCPU `submit_async`s a batch of `Blocking` ops
      onto the offload pool, then parks on an in-window completion counter (`__vm_wait32`) and reaps
      completions as the pool delivers them (`__vm_io_reap`): the "submit, park, run another, resume on
      completion" loop, with the parked vCPU woken by a pool worker's `notify`. N=8 I/Os in flight cost
      one parked vCPU + K pool threads (the "0 blocked vCPU threads" win). C-frontend (`codegen_ir.c`):
      new builtins `__vm_io_submit_async`/`__vm_io_reap` (→ `cap.call 9 1`/`9 2` on the stashed IoRing
      handle) + `__vm_blocking_handle` (the Blocking handle for an SQE). The powerbox is a **fixed
      7-handle** set (stdout, stdin, exit, memory, addrspace, ioring, blocking) every `_start` imports —
      one entry shape, mirroring how the frontend already always imports Memory/AddressSpace; a guest
      that never touches the ring just leaves the two handles stashed and unused. (An earlier draft made
      the arity conditional on a usage scan — collapsed to a single arity; the c_frontend harnesses
      share one `powerbox(h, win, block_for)` helper and `svm_run::run` grants by the entry's declared
      arity, now 6→IoRing/7→Blocking.) Tests (`c_frontend.rs`): **`c_guest_async_io_runtime`** — a
      single-vCPU event loop (`demos/async_io`, N=8) — and **`c_guest_async_work_stealing`** — the
      capstone **async work-stealing M:N runtime** (`demos/async_work_stealing`, NWORKERS=4 vCPUs draining
      NTASKS=16 I/O-bound tasks: a worker `submit_async`s a task's op and moves on, parking on the
      counter only when nothing is runnable, woken by a pool `notify`; work-stealing + I/O overlap).
      Both run on interp (`run_with_host`→`drive`) + JIT (`..._with_host_async` + `HostAsyncHooks`,
      `reserved_log2 = DEFAULT_RESERVED_LOG2` for the malloc growth tail) and print the order-invariant
      `Σ mix(i)`; 0/30 flake; full c_frontend suite (69 tests) + workspace + clippy + windows
      cross-check green.
      - **Two real findings the capstone surfaced (worth knowing):** (1) a **shared** ring's
        submit/reap `cap.call`s **must be serialized by the guest** (a guest mutex, like a real shared
        io_uring's single-producer SQ) — the JIT `cap_thunk` takes `&mut *host` with no lock (the interp
        serializes via `Arc<Mutex<Host>>`), so concurrent dispatch from multiple vCPUs would race the
        Host. (2) **cap-buffer ops to a guest-*grown* heap page — FIXED.** Previously fail-closed on the
        JIT: `cap_thunk` rebuilt `MprotectWindow` per call with a fresh software page map, so it didn't
        know the guest grew the heap (the interp's `Mem` persists the map across cap.calls), and a
        `read_bytes`/`write_bytes` to a grown-tail address returned `-EFAULT`/wrote nothing. Now the page
        map is **persisted per run in the `Host`** (`Host::cap_window_pages(base)` → a shared
        `CapPageMap = Arc<Mutex<BTreeMap<u64,u8>>>`, keyed by window base so it resets for a new window);
        `cap_thunk` builds the window via `MprotectWindow::new_shared(.., pages)`, so growth committed in
        one cap.call is seen by later ones — a borrow of grown heap memory works on the JIT exactly like
        the interp. Regression guard: `c_frontend::c_grown_heap_buffer_is_borrowable` (malloc 128 KiB past
        the window, `write()` the grown buffer; interp == JIT, was non-vacuously failing before). The demo
        keeps its global (prefix) SQ/CQ buffers purely for the shared-ring design now, not as a workaround.
  - **DPOR — DONE.** `explore_all` is now a **dynamic partial-order reduction** model checker
    (Flanagan–Godefroid stateless form): each visible op records the confined byte range / futex key it
    touches (`MemAccess`, computed at the op's commit point via the existing `confine_checked`), and
    after every schedule the checker detects races (for each transition, the latest earlier
    *conflicting* transition — same bytes, one a write — by a different vCPU) and adds that vCPU to the
    earlier decision's backtrack set, exploring **both** orders only for genuinely dependent ops while
    keeping one order for independent ones — **plus sleep sets** (the full FG algorithm): a thread that
    became redundant after an independent sibling ran is held *asleep* down that subtree until a
    conflicting transition wakes it, pruning the residual cross-cluster redundancy that backtrack-only
    DPOR re-explores (a sleep-blocked prefix just stops and contributes no outcome). The reduction is
    **sound** (reordering independent ops can't change the terminal state), proven non-vacuously by
    `svm/tests/dpor.rs`: a differential vs the retained unreduced enumerator (`explore_all_bruteforce`,
    the oracle) shows **identical outcome sets** on racy programs whose outcome *multiplicity* reflects
    coverage (lost-update counter → {1,2}; store-buffering two-var → {1,2,3}; two independent racy
    clusters → {17,18,33,34}) at far fewer schedules — atomic-counter 2 vs 11, racy-counter 4 vs 71,
    store-buffer 6 vs 71, all-independent stores **1 vs 379**; and on two independent atomic clusters
    sleep sets cut **8 vs 12** (backtrack-only) **vs 1270** (unreduced). The existing `concurrent.rs`
    proofs (`exhaustive_*`) still pass (same outcomes, `complete`). *Not full optimal-DPOR (no source
    sets / wakeup trees), but independent work no longer multiplies the tree.*
  - **Spin-loop handling — DONE.** The classic pathology — a busy-wait spinlock where every `cmpxchg`
    retry is a fresh decision point, so the tree is unbounded *and* an unfair schedule starves the
    holder into a spurious `OutOfFuel` — is now collapsed in the memop explorer. After each turn the
    scheduler compares a 64-bit fingerprint of the vCPU's local configuration (`VCpu::local_fingerprint`
    — fibers + reified call stacks, *not* shared memory) against the pre-turn one, alongside a per-`Mem`
    write counter; a visible op that **changed no memory** and returned the vCPU to the **same
    configuration** is a pure busy-wait, so the vCPU is **parked** (a `SpinWaiter` keyed by the byte
    range it read) off the runnable set until another vCPU writes that range (`DetState::wake_spins`) —
    the exact semantics of the spin, with no redundant decision points and no starvation. Sound (a
    stuttering thread's future is fixed until shared memory it reads changes), verified in
    `svm/tests/spinloop.rs`: a 2-worker `cmpxchg`-spinlock counter is now **exhaustively verifiable**
    (12 schedules, outcome `{2}` — was non-terminating; the parent commit times out >60 s), and an
    *asymmetric* spinlock (`+1` vs `*3` under the lock) yields exactly `{1,3}`, proving the
    lock-acquisition-order nondeterminism survives the pruning. *Limitation: detection is intra-turn, so
    it catches single-visible-op spin bodies (the `cmpxchg`/flag-load spinlock); a multi-visible-op spin
    body falls back to bounded exploration (still sound). Gated on memop mode (the exhaustive/brute
    explorers); the seeded sweep is fuel-bounded and unaffected.*
  - **Still open (Phase 4):** honoring *weak* orderings in execution (both backends run seq-cst
    today), fiber/vCPU quota metering (the kill path exists; *metering*/quotas don't yet), the D57
    migratable-fiber primitive (stackful work-stealing), and the DPOR refinements (source sets /
    wakeup trees for full optimality; multi-op spin-body detection).

- [ ] **Nesting (§14)** + **shared memory + isolation tiers (§13)** + **real guest-visible
  virtual memory** — *most of the §1a differentiators live here.* Sub-window **confinement** is
  in (the masking unit `Window::sub` + a both-backends run path with an interp↔JIT escape-oracle),
  as is the **`AddressSpace` capability + attenuation** (iface 5: a power-of-two window sub-range
  with `map`/`unmap`/`protect` confined to it + a `sub` op that mints an attenuated child), and the
  **`Instantiator` capability** (iface 6, **both backends**): `instantiate`/`join` spawns a
  same-module child confined to a sub-window — the interp runs it as a vCPU on the §12 executor
  (shared backing; join parks only the calling fiber); the JIT re-compiles it over its own re-entrantly
  guarded window (nesting cost at setup) and copies back — proven equal by an interp↔JIT differential.
  **co-fiber resume/suspend** children are in too, on **both backends** (the `Yielder` cap +
  `spawn_coroutine`/`resume`; on the JIT a child is a suspended `svm-fiber` native continuation),
  including **fault-driven yield** (`spawn_demand_coroutine` — interp: prot-map faults; JIT: real
  hardware faults suspended from the SIGSEGV/VEH handler) — the §14 lazy-paging primitive, end to
  end. **Separate-module children** are in too (a host-granted `Module` capability + the
  Instantiator's module ops — the "plugin-in-plugin" story, data segments materialized into the
  carve, lazily for demand children), and **cross-domain `SharedRegion` `create`/`grant`** (a guest
  mints a region via its `AddressSpace` and grants it into a coroutine child — zero-copy shared
  bytes across the domain boundary, both directions). Remaining nesting work: grant to
  executor/JIT children, richer cap pass-through, and a non-blocking JIT `instantiate` child;
  then the isolation tiers.
- [ ] Spectre hardening (§9); split-host supervisor; monitoring.
- [ ] SIMD (§17); GPU; capability revocation; cross-domain channels (§7); exception /
  `setjmp` **unwinding mechanics** (the stack-switch primitive is settled; unwind tables
  are not).
- [ ] **Language on-ramp:** native **LLVM backend** (the differentiator vehicle) and/or an
  optional **wasm bridge** (compat). chibicc stays the MVP frontend; this is breadth work.

### Fuzzing — have vs. gaps
Have (✅ continuously, except where noted):
- [x] `decode_verify` (libFuzzer) + `fuzz_smoke` (stable, every push/PR): decode
  fail-closed; verify never panics; a *verified* module never **panics** the interp
  (fuel-bounded). **Robustness, not escape.**
- [x] `diff` (libFuzzer) + `jit_fuzz` (stable, 4000 seeds every push/PR): interp == JIT on
  generated verifier-valid modules (`irgen.rs`, §8).
- [x] **Escape-oracle** — `run_differential` now also byte-compares the **final guest
  window** across interp + JIT for float-free modules: when the interpreter (the masking
  reference) completes, every access was in-window, so the JIT's window must match exactly;
  a mismatch is an access that escaped/wasn't masked into `[0,size)` (§4/§18). Threaded via
  `run_capture` (interp) / `compile_and_run_capture` (JIT); seeded non-zero so a divergent
  *read* shows too. Float modules are excluded (NaN bits aren't pinned across backends).
  Plumbing pinned by `tests/escape_oracle.rs`; **verified non-vacuous** (corrupting the JIT
  mask makes the fuzzer fail). Runs in the 4000 stable seeds (every push) *and* the `diff`
  libFuzzer target (`cargo +nightly fuzz run diff`).
- [x] `fuzz/mask` (libFuzzer): the confinement-masking unit — masked address always in
  `[0,size)` (D38, the escape hinge).
- [x] `roundtrip` (libFuzzer): encode∘decode identity.
- [x] **Nightly CI matrix** runs `decode_verify` **+ `diff` (carries the escape-oracle) +
  `mask`** (`ci.yml`, `schedule`/`workflow_dispatch`), so all three get coverage-guided time.
- [x] **Loops + indirect calls in `irgen`** — `gen_loop_func` emits one **counted loop**
  (entry/header/body/exit, a strictly-incrementing i32 counter to a small bound ⇒ halts by
  construction, no JIT fuel needed; ~half of functions), and `gen_inst` emits `call_indirect`
  in two terminating flavors (forward-success / type-mismatch-trap = the I2 "forged index is
  inert" check). Loop bodies run loads/stores ≤15× ⇒ repeated/aliased stores deepen the
  escape-oracle. A coverage-guard test asserts both shapes are actually produced. Surfacing
  this also relaxed an over-strict harness rule: when **both** backends trap, the trap *kind*
  is no longer asserted (a trap is terminal; an eager interp vs an optimizing JIT may surface
  different ones among several reachable traps — e.g. a dead trapping float→int convert).
- [x] **Concurrency escape-TCB hardening (§12/§18).** The §18 "fuzz the hinge" discipline now reaches
  the surface the concurrency work grew (the two new `unsafe` units + the concurrent access path):
  - **Concurrent escape-oracle** (`concurrent_escape.rs` + `concurrent_escape_fuzz.rs`): a *spawned
    thread* accessing an **out-of-window** address must confine identically on both backends — hand-
    written (commutative atomic counter + disjoint plain stores) *and* generative (out-of-window
    commutative-atomic programs across seeds, byte-comparing the final window), plus a **tail-fault**
    case (`reserved > mapped` ⇒ a thread's out-of-*mapped* access detect-and-kills, not wraps).
  - **`svm-fiber` switch fuzzer** (in its own tests): random resume orders over many fibers stress the
    per-ABI register/stack save-restore (the riskiest unsafe, ×3 ABIs).
  - **`svm-mem` differential fuzzer**: the raw-atomics `Mapped` backing vs the safe `Paged` model, 20k
    mixed ops (atomic/plain, 4/8-byte, cross-page, out-of-range) — the interp-as-oracle discipline for
    the memory substrate.
  - **miri** on `svm-mem` (a `cfg(miri)` heap backing replaces mmap; weak-memory emulation off — its
    store buffer ICEs on the intentional mixed-width atomic/byte overlap): provenance + data-race
    checks on the raw atomics. The **nightly `miri` CI job is pending a maintainer apply** (needs the
    `workflow` token scope; snippet in the session / commit `60d4f3a`).
  Validated linux + wine (x86-64 Windows); aarch64 via macOS CI. The fiber/JIT asm + the real mmap
  path miri can't execute — those stay covered by these fuzzers + the sanitizers (loom/TSan).

Gaps (priority order):
- [x] **`cap.call` — both the inert (fault) *and* success paths are generated.** Arm 18 emits a
  forged-handle cap.call (inert ⇒ `CapFault` on both, the I2 check). Arm 19 (gated on `has_mem`)
  emits a **valid Memory cap.call** — granted handle (`MEMORY_HANDLE = 1<<8`, the first grant),
  page-aligned in-range `map`/`unmap`/`protect` — so the **success path** runs on both backends:
  the harness grants a Memory cap to interp + JIT via new capture+host run wrappers
  (`run_capture_reserved_with_host` / `compile_and_run_capture_reserved_with_host`) over the
  production `svm_run::cap_thunk`, so the cap's window effects ride the **escape-oracle**, not just
  outcome agreement, interleaved with the random CFG/loops. A coverage guard
  (`generator_covers_*`) asserts a `type_id==3` cap.call is produced; the dedicated
  `jit_cap_memory_escape_oracle_differential` (jit_diff) adds a focused full-window pass. The
  integration **caught two real bugs**: (a) `cap_thunk` did `slice::from_raw_parts(args, 0)` on the
  JIT's null pointer for a 0-arg/0-result cap.call (UB) — now guarded; (b) the differential's
  `(Err, Returned)` arm rejected *any* modelled interp trap while the JIT returned, but a
  **droppable** pure-op trap (div/rem-by-zero, int-overflow, bad float→int convert) whose result is
  dead may be DCE'd by the JIT — relaxed via `droppable_trap` (effectful/control traps stay strict).
  Loops are still a single counted shape (no nested/irreducible/data-dependent) — richer shapes need
  a JIT step-cap to stay terminating.
- [x] **Escape-oracle on float modules — evaluated, deliberately *not* enabled.** Including float
  modules in the final-window byte-compare **passes on x86-64** today (interp + JIT lower float ops
  to the same hardware, so NaN bits agree), but that agreement is **arch-specific**: a Phase-3.5
  aarch64/Windows port could legitimately produce a different NaN payload, turning the oracle into a
  false-positive escape. The escape-oracle is about **addresses** (integer modules exercise the
  masking fully), so the float gain is ~zero; the NaN-insensitive value-compare + the float-free
  memory oracle stay. (Re-enable only with a sound canonical-NaN/integer-store-only scheme if a real
  need appears.)
- [x] **Guard-page fault detection (unix)** — beyond the final-memory divergence check, a
  gross out-of-window access now faults into the `PROT_NONE` guard page and is caught as a
  clean `MemoryFault` (detect-and-kill, see the trap-catching item above) rather than relying
  on a wild-pointer crash. (The fuzzer could be extended to assert "verified ⇒ no guard
  fault" as a second escape signal.)

### Benchmarking — have vs. gaps
Have (✅):
- [x] `crates/svm/src/bin/bench.rs`: decode / verify / **interp** throughput on one
  hand-written loop (`sum 0..N`), ns/iter, dependency-free.
- [x] **`bench/` — JIT vs Wasmtime** (out-of-workspace, like `fuzz/`; pulls in Wasmtime).
  Each kernel is written once in our IR text and once in equivalent WAT (results
  cross-checked before timing); both lower via Cranelift, so it's a like-for-like §1a check.
  Measures steady-state **compute** (per-iteration, isolated by big-vs-small subtraction so
  compile cancels) and **cold start** (source → first result). The memory kernels are timed
  against **both wasm32 and wasm64** (`Config::wasm_memory64`). `cargo run --release` from
  `bench/`; `--csv` for a line per kernel. **NB: the representative numbers below predate the
  `opt_level=speed` switch** (see the "Cranelift `opt_level=speed`" item under *Gaps* — memsum/scatter
  now *beat* wasm32, locals_c beats wasm64); they're kept for the per-kernel *narrative*, but the
  current ratios are in that item. **Representative numbers** (ratio = svm ÷ wasm; `<1` = svm faster;
  machine-dependent — watch the *ratio*, not the absolute ns):
  - `alu` (tight i64 mul/add loop): compute **≈1.0–1.05×** (parity, as designed — shared
    backend); cold start **≈0.3–0.45×** (we're ~2–3× faster — "SSA on the wire, no SSA
    reconstruction", §1a). *Both theses confirmed.*
    Both memory kernels now exercise the **mask-elision** path (below): their `(i&K)*8`
    addresses are provably in-window, so the JIT drops the `& mask`.
  - `memsum` (store+load to the **same** address each iter): **wasm32 ~0.69 < svm ~0.94 <
    wasm64 ~1.25** ns/it → svm ~1.36× wasm32, **~0.72× (faster) than wasm64**. (Pre-elision
    svm was ~1.10; Wasmtime CSEs the same-address bounds check, which still helps it.)
  - `scatter` (store + load to **different, per-iter varying** slots — the realistic test):
    **wasm32 ~1.03 < svm ~1.27 < wasm64 ~2.0** ns/it → svm **~1.21× wasm32** (pre-elision
    ~1.53×) and **~0.62× = ~1.6× *faster* than wasm64**. Varied addresses defeat Wasmtime's
    bounds-check CSE, so wasm64 pays a full check per access while our (now-elided) mask
    wins big. Net: §1a's two memory claims both hold — we clearly **beat wasm64**, and the
    **wasm32 gap is now ~1.2–1.36×** (mask elision closed roughly half of it; the residual
    is wasm32's truly-free guard-page access, which needs real guard pages, §5).
- [x] **Interface / host-call kernels (`hostcall`, `hostbuf`) — the §1a "around-compute" axis.**
  Each times one guest→host→guest crossing per iteration (own `N_HOST_BIG`): SVM `cap.call`
  through the bench trampoline thunk vs a **Wasmtime imported host function** (a `Linker`), both
  via Cranelift, results cross-checked. `Mode::HostCall` on `Resolved` selects the cap-thunk SVM
  path + import-linked wasm path in `measure`. **Honest findings** (best-of-5, machine-dependent):
  - `hostcall` (scalar `x→x+1` round-trip): svm **~1.24× slower**. `cap.call` lowers to a
    *generic* indirect thunk that packs args into an i64 array; the **devirtualize-to-direct-call
    win (D45) is deferred**, so this is the honest baseline that optimization will move.
  - `hostbuf` (zero-copy `(ptr,len)` **borrow buffer**, 64 B, host sums in place — the §7 path):
    svm **~1.8× faster** — *even vs a fair cached-`Memory` wasm baseline* (the wasm host fn caches
    the exported memory in `Store` data to avoid a per-call `get_export` lookup — I fixed an
    initial strawman where the naive lookup inflated wasm to a fake ~6×). The real win is
    structural: SVM hands the host the window base for free; Wasmtime still pays `mem.data(&caller)`
    per call. **This substantiates §1a's strongest claim.** The *larger* §1a win (vs the component
    model's lift/lower marshalling, and async rings) is a heavier comparison, **not** attempted.
  Both are tracked in `baseline.txt` (appended rows, measured on the dev container — a maintainer
  may re-baseline all rows on a canonical machine for cross-row consistency).

Gaps (the weakest area vs. AGENTS.md "benchmark early · measured vs. wasm/Wasmtime · catch
regressions one commit old"):
- [x] **Over-time tracking — *done* (tool + non-gating CI).** `bench/` has
  **`--save-baseline FILE`** / **`--check FILE`**: the committed **`bench/baseline.txt`** records
  the per-kernel **ratios** (svm÷wasm — the machine-portable signal, not the absolute ns), and
  `--check` reruns (best-of-`--reps 5`) and **exits non-zero** if any ratio grew past `--tol`
  (default 25%, a band that absorbs runner noise — a real regression like losing mask-elision was
  +26%, losing SSA promotion far more). Verified non-vacuous (a tightened baseline trips it). A
  **non-gating** `bench` job in `ci.yml` (nightly/`workflow_dispatch`, `continue-on-error`, wide
  `--tol 0.4`) runs `--check` so a gross regression surfaces without blocking merges on shared-
  runner noise. **Still TODO
  (minor):** `crates/svm/src/bin/bench.rs` (the in-tree interp
  throughput bench) still just prints; over-time *storage* of the numbers (vs. recompute-and-compare)
  isn't kept — `--check` compares against the committed baseline, which is enough for "one commit old."
- [x] **C-frontend promotion guard — *done* (structural test + `alu_c` timing kernel).** The
  headline §3 SSA-promotion win (loop body ~22→0 memory ops) is pinned **deterministically** by
  `c_frontend::c_ssa_promotion_eliminates_loop_body_memory_ops`: it compiles promotable hot loops
  and asserts **zero** `Load`/`Store` outside each function's entry block (`loop_region_mem_ops`),
  with an address-taken control proving the metric isn't blind — a promotion regression fails the
  gating job one commit old, with no timing noise. The **wall-clock** win is now *also* tracked:
  the `bench/` **`alu_c`** kernel takes its IR from chibicc (same recurrence as `alu`, compiled
  from C) and times it — it sits at ≈parity with `alu` (compute ratio ~1.02× here); a loop body
  regressing to memory would drift it toward the memory-bound path.
- [x] **Mask elision (§1a "mask-when-not", D36–D38)** — *done*: a conservative upper-bound
  analysis in the JIT (`ub_of`/`in_window`) drops the `& mask` when the address is provably
  `< size`, closing ~half the wasm32 gap (memsum 1.6→1.36×, scatter 1.53→1.21×) and widening
  the wasm64 lead. Guarded by the escape-oracle (a wrong bound diverges final memory / faults;
  verified non-vacuous). Pinned by `escape_oracle::elided_bounded_address_confines`.
- [x] **Cranelift `opt_level=speed` — *done* (was the big residual-gap closer).** The JIT had been
  compiling at the default `opt_level=none` (no GVN/CSE, no constant materialization, no
  store-to-load forwarding) while Wasmtime runs `speed` — so the comparison was *unfair* and svm left
  a lot on the table. `locals_c` exposed it: the store/load addresses (identical) were computed twice
  and the mask/constants were rip-relative pool loads (~13-instruction hot loop). Enabling `speed`
  (both the top-level and §14 child compiles) is a broad, fair win that **closes the residual wasm32
  gap**: memsum **1.37→0.91×** and scatter **1.24→0.94×** now *beat* wasm32; locals_c **3.25→1.48×**
  (wasm32) and **1.84→0.83×** (wasm64, now faster); hostbuf 0.80→0.64×; hostcall 1.24→1.11×. Cold
  start regresses modestly (alu 0.40→0.48× of Wasmtime) but stays ahead — "SSA on the wire" keeps the
  lead even with the optimizer on. **Caught + fixed a latent kill-path bug it exposed:**
  `emit_epoch_check` polled the host-owned interrupt cell with a *plain* load (relying on `none` to not
  hoist it); under `speed`, Cranelift's alias analysis sees no *guest* store to the cell (the watchdog
  writes it cross-thread) and hoisted the load out of the loop ⇒ the poll fired once and a runaway was
  never killed (`jit_killpath` hung). Now an **atomic load** (a sync op the optimizer won't hoist; the
  cell is a host `AtomicU64`). Verified byte-identical: escape_oracle + jit_diff + 4000-seed jit_fuzz +
  full workspace green; windows + loom clean. *(`baseline.txt` still holds the pre-`speed` numbers —
  re-baseline on a canonical machine.)*
- [ ] **Remaining `locals_c` gap (now ~1.48× wasm32, but it *beats* wasm64).** With the optimizer on,
  the leftover gap vs wasm32 is the un-elidable `sp`-relative mask (the data-SP is an unbounded block
  param) plus the threaded-SP add — i.e. the 64-bit-confinement tax, paid where elision can't fire.
  Closing it needs the verifier to prove the data-SP bounded (the §3d register-pinned-`sp` direction),
  *not* 32-bit addressing (D50, rejected). Lower priority now that we beat wasm64 everywhere and tie/
  beat wasm32 on the elided kernels; `locals_c` is also a deliberate worst case (`volatile` +
  address-taken forces memory residence; normal locals promote to SSA and are free).

### Suggested next pickups (ranked)

> **▶ START HERE (next session) — current frontier as of the 2026-06-11 batch.** Everything below
> this block is the **build log** (history of landed work, kept for context); this block is the live
> "what's next."
>
> **How to work** (unchanged): commit straight to `main`; gate every commit with
> `cargo fmt --all && cargo clippy --workspace --all-targets && cargo test --workspace` (all green),
> the **windows cross-check** (`cargo check -p svm-jit -p svm-run --target x86_64-pc-windows-gnu`),
> and — when touching the futex/thread runtime — the **loom** model check
> (`RUSTFLAGS="--cfg loom" cargo test -p svm-jit --lib loom`). The container can reset mid-session →
> recover with `git fetch origin main && git reset --hard origin/main`. Push to `main`, keep branch
> `claude/hopeful-franklin-66kiL` force-synced. **Never** push `.github/workflows/*` (no `workflow`
> token scope). Key design artifacts: **`AUDIT.md`** (security audit register — all 8 findings closed),
> **`SCHEDULING.md`** + **DESIGN D56/D57** (the concurrency-primitives decision), **`DESIGN.md`** /
> **`README.md`**.
>
> **Just landed (this session): wasm transpiler — function imports / the host ABI, then
> `memory.size`/`memory.grow` (item 0 below).**
> **(1) Imports.** A wasm `(import "<module>" "<name>" (func …))` lowers to a `cap.call` by the
> convention `module` = decimal capability `type_id`, `name` = decimal `op`; the transpiler threads one
> capability handle (an `i32`) as the leading param of every function (the data-SP trick), and the
> embedder grants the cap + passes its handle as the entry's leading arg. Function-index remapping
> (imports first), `call_indirect` handle-threading through the §3c type check, clean errors for
> non-numeric names / non-func imports / multiple interfaces; 7 differential tests
> (`crates/svm-wasm/tests/imports.rs`) on **both** backends under one reference `Host`. **Bench
> `--from-wasm` now also transpiles the `hostcall`/`hostbuf` kernels** (cross-checked identical to
> Wasmtime) — the apples-to-apples comparison covers the §1a interface axis, not just compute.
> **(2) Linear-memory growth.** `memory.size`/`memory.grow` (pages, incl. memory64): when a module uses
> `memory.grow` the window reserves the memory's full growable span at offset 0 (up to its declared
> `maximum`, else a modest default `DEFAULT_MAX_GROW_PAGES = 256`) and puts globals/table *above* it; a
> runtime 8-byte **size cell** backs the ops (`grow` updates it branch-free via `select`, returning the
> old size or `-1`). A pre-scan means a non-growing module is byte-identical to before (tight window, no
> cell, `memory.size` a constant). 6 new differential tests in `transpile.rs` (31 total). No-import
> /no-grow modules unchanged. Full detail in item 0's sub-bullets. **Next:** passive data / element
> segments (then bulk memory `memory.fill`/`copy`/`init`).
>
> **Earlier (prior session): the async I/O ring (B) — COMPLETE, increments 2 + 3a + 3b + 3c,
> mechanism + runtime on BOTH backends.** Increment 2 — the **bounded blocking-offload pool**: `submit`
> overlaps `Blocking` SQEs (iface 10) on an `OFFLOAD_POOL_THREADS = 4` pool (waves of K) while inline
> ops run in SQE order, transparently. Increment 3a/3b — **async submit + true fiber parking on interp
> *and* JIT**: op 1 `submit_async` kicks the batch to the pool and returns; the guest parks on an
> in-window futex completion **counter** via `i32.atomic.wait`; each pool worker, on completing, posts
> its CQE host-side + atomic-bumps the counter + `notify`s it to **wake the genuinely-parked vCPU** (an
> I/O completion is a futex notify — DESIGN §12); op 2 `reap` flushes CQEs on the vCPU thread. The
> interp wakes via `Scheduler::notify` (installed in `drive`); the JIT wakes a parked OS-thread vCPU via
> its per-run `Domain`'s futex, bridged by the `svm_jit::AsyncHostHooks` seam (`svm_run::HostAsyncHooks`
> + `compile_and_run_capture_reserved_with_host_async`), over a backend-neutral
> `svm_interp::AsyncCounter`. Race-free via each futex's compare-under-lock guard. Increment 3c — the
> async runtime **in real C**: `demos/async_io` (single-vCPU event loop, N=8) and the capstone
> `demos/async_work_stealing` (**async work-stealing M:N**, 4 vCPUs draining 16 I/O-bound tasks: submit,
> park, steal/run another, resume on completion), via new `codegen_ir.c` ring builtins
> (`__vm_io_submit_async`/`__vm_io_reap`/`__vm_blocking_handle`) + a **fixed 7-handle** powerbox (one
> `_start` shape). See §10's ring tracker + `crates/svm/tests/io_ring.rs` (10 tests) +
> `c_frontend.rs::{c_guest_async_io_runtime,c_guest_async_work_stealing}` (0 flake; loom + windows
> cross-check green). **Two findings the capstone surfaced** (see §10): a shared ring must be guest-
> serialized (the JIT `cap_thunk` doesn't lock the `Host`), and JIT cap-buffer ops to a guest-*grown*
> heap page fail-closed (per-cap.call `MprotectWindow` doesn't persist growth) — a safe interp/JIT
> divergence; the demos use global (prefix) SQ/CQ buffers. *(Earlier: the escape-TCB audit
> (`AUDIT.md`); D57 + `SCHEDULING.md`; the `demos/mn_sched` + `demos/work_stealing` guest M:N
> schedulers; ring increment 1.)*
>
> **Immediate frontier, ranked** *(the async ring (B) is done — these are the next big rocks):*
> 0. **wasm → IR transpiler (`crates/svm-wasm`) — IN PROGRESS (numeric + control + if/else + memory +
>    grow + imports).**
>    A second frontend after chibicc, chosen *before* the LLVM on-ramp because it's smaller and directly
>    serves the §1a benchmark thesis: take *any* wasm and run it on SVM vs Wasmtime on the *same bytes*,
>    instead of hand-writing IR+WAT kernel pairs. The interesting part is the **stack→SSA reconstruction**
>    (wasm is a stack machine; our IR is SSA) — done by threading all locals + the surviving operand
>    stack as block params at every control-flow target, the same trick chibicc uses for the data-SP.
>    **Landed:** i32/i64 numeric + locals; the full structured control set incl. `if`/`else` (with
>    dead-code / else-resurrection handling); **linear memory** load/store (i32/i64, narrow + `memory64`);
>    direct **`call`** (multi-fn + recursion); **floats** (f32/f64 const/arith/unary/compare/load/store +
>    every int↔float conversion); active **data segments**; **globals** (`global.get`/`set` → a reserved
>    window region above the linear memory, init via data segments); **`call_indirect`** + tables/element
>    segments (the wasm table → an in-window i32 funcref-index array; the runtime load feeds our
>    `CallIndirect`'s §3c type-id check — a type-confused index traps, the I2 guarantee); **function
>    imports / the host ABI** (a wasm `call` to an import → a `cap.call` — see the import-ABI note below).
>    Window layout: `linear-memory | globals | function-table`, all inside the masked power-of-two window.
>    All differentially tested (`svm-wasm/tests/transpile.rs`, **25 tests**: WAT → transpile → verify →
>    interp==JIT vs a hand oracle — the real `alu`/`memsum`(32+64)/`scatter` bench kernels, br_table,
>    collatz, recursive fib, harmonic float loop, data/global tests, a 3-way call_indirect dispatch +
>    type-mismatch trap) — **plus the capstone `real_clang_wasm`: compiles C with `clang --target=wasm32`
>    (+`wasm-ld`) and runs the transpiled module** (fib/sumto/poly + a **function-pointer `dispatch`** →
>    real clang call_indirect/tables/elements), exercising LLVM-optimized control flow, the
>    `__stack_pointer` global, and indirect calls on genuine real-world wasm (skips if the clang/wasm
>    toolchain is absent, like the `cc` tests). Two bugs the differential caught: a `locals` vec not grown
>    for declared locals; SSA value-numbering that mis-counted `store` (no result) — now `next_val`
>    advances only for value-producing insts. **Bench wiring — DONE:** `bench/ --from-wasm` replaces each
>    compute kernel's hand-written SVM IR with IR *transpiled from its WAT* (the same bytes Wasmtime
>    runs) — the genuine apples-to-apples comparison. Result: transpiled IR ≈ hand-written (alu 1.02×
>    both, memsum 0.91× both / beats wasm32, scatter 0.94→1.00× — a ~6% transpiler overhead from the
>    i32→i64 address extend). **Bonus finding:** `locals_c` is 1.43× from chibicc IR but **0.92× from the
>    transpiled WAT**, confirming that gap is a chibicc `volatile`-array lowering artifact, not the VM.
>    - **Imports / host ABI — DONE (this session).** A wasm `(import "<module>" "<name>" (func …))` binds
>      to an SVM capability by a naming **convention**: `module` = decimal capability **`type_id`**, `name`
>      = decimal **`op`**. A wasm `call` to an import lowers to `cap.call type_id op sig handle args`; the
>      transpiler threads **one** capability **handle** (an `i32`, the forgeable index a `cap.call` takes)
>      as the leading param of every function/block (the data-SP trick), so any function reaches it and the
>      embedder grants one capability + passes its handle as the entry's leading arg. The transpiler stays
>      pure mechanism — it never interprets the host semantics. The wasm function-index space puts imports
>      first, so all `call`/`call_indirect`/element/export indices remap by `−n_imp`; `call_indirect`
>      prepends the handle to both its args **and** the §3c type-check signature (matching the defined
>      targets that now carry it); a funcref to an import is a clean error. v1 threads one handle ⇒ all
>      imports must share one `type_id` (methods by `op`); a non-numeric name, a table/memory/global
>      import, or imports spanning multiple interfaces is a clean `Unsupported` (real WASI's non-numeric
>      imports need a dedicated shim). No-import modules are byte-identical to before (all 25 transpile
>      tests unchanged). This is exactly the `cap.call 0 0` / `cap.call 0 1` shape the bench `hostcall`/
>      `hostbuf` kernels hand-write. Differentially tested in **`svm-wasm/tests/imports.rs` (7 tests)**:
>      a `Clock.now` loop-sum (no-arg op), a `Blocking.work` loop-sum (scalar arg + result = the `hostcall`
>      shape, deterministic `mix`), handle-threading through a defined→defined call and through a
>      `call_indirect` dispatch table, plus the three clean-error guards — each run on **both** backends
>      under one reference `Host` (interp `run_with_host`; JIT `compile_and_run_with_host` over the
>      production `svm_run::cap_thunk`, added as a dev-dep). **Bench wiring — DONE:** the `hostcall`/
>      `hostbuf` interface kernels now transpile from their WAT under `--from-wasm` (no longer
>      hand-written-only) — their imports use the convention (`"0"/"0"` → op 0 scalar `x+1`, `"0"/"1"` →
>      op 1 borrow-buffer sum) matching `bench_thunk`'s op dispatch and the Wasmtime linker; the
>      transpiled entry takes the threaded handle as its leading param (the stateless thunk ignores it,
>      so `lead_args = [0]`). The bench's pre-timing cross-check confirms the transpiled SVM IR returns
>      identical results to Wasmtime on the same bytes (hostcall ~1.18×, hostbuf ~0.50× = ~2× faster).
>      So `--from-wasm` now covers the §1a **interface** axis too, not just compute. **Still open on
>      imports:** multiple distinct capability interfaces (one handle each).
>    - **`memory.size` / `memory.grow` — DONE (this session).** Pages, incl. `memory64`. The linear
>      memory is at window offset 0; when a module uses `memory.grow` the window reserves its **full
>      growable span** at the bottom — up to a declared `maximum`, or `DEFAULT_MAX_GROW_PAGES = 256`
>      (16 MiB, bounded by `MAX_GROW_PAGES`) for unbounded memory — and puts the globals/table regions
>      *above* it, so growth never collides. A runtime 8-byte **size cell** just above the linear memory
>      (initialized to the initial page count via a data segment) holds the current size: `memory.size`
>      loads it, `memory.grow(delta)` updates it **branch-free** (i64 page math, then `select` to store
>      `new`/unchanged and return `old`/`-1`). Because SVM masks accesses into the window rather than
>      bounds-checking-and-trapping, a grown page is just reachable; the cell only governs the return
>      values (the documented confinement difference). A **pre-scan** for the `memory.grow` opcode means
>      a non-growing module — every existing kernel — is **byte-identical** to before (tight initial-
>      sized window, no cell, `memory.size` a constant). 6 differential tests in `transpile.rs` (size
>      constant; grow returns old + size reflects it; over-cap → `-1` + unchanged; declared `maximum`
>      honored; grown memory store/load past 64 KiB; the memory64 path). *Limitation: the growable
>      window is eagerly RW-committed (lazy-physical on Linux via `MAP_NORESERVE`), so the unbounded
>      default is modest; a program needing a larger heap declares a `maximum` (honored) — a lazy-commit
>      growable window is a future JIT enhancement.*
>    **Missing wasm features (the explicit note — what svm-wasm does NOT transpile yet):** (1) passive
>    data / element segments + **bulk memory** (`memory.fill`/`copy`/`init`, `table.*`). (2) imports
>    spanning multiple capability interfaces (one handle is threaded). (3) SIMD (v128). (4) reference
>    types beyond funcref tables; multi-memory / multi-table. **Next slice:** passive data / element
>    segments + bulk-memory ops (`memory.fill`/`copy`) — common in clang/wasi-libc output. The subset
>    already transpiles real clang-emitted wasm (control flow, `__stack_pointer`, function pointers,
>    host imports, heap growth) end to end and benches at hand-written-IR speed.
> 1. **Language on-ramp (LLVM-bitcode→IR)** — the big breadth play (D54). **Architecture decided: AOT**
>    — the translator links libLLVM at build/dev time and is *off the runtime path* (keeps the ~5 MiB
>    JIT binary lean). MVP: `clang -emit-llvm` → IR for the scalar+memory+call subset chibicc already
>    proves (aggregates via memory; hard-error on vectors/unsupported intrinsics), with a differential
>    harness running the existing C demos through *stock LLVM* and matching native `clang`. (LLVM 18 +
>    `libLLVM.so` confirmed present in the dev container.)
> 2. **Migratable-fiber primitive (D57)** — the maintainer's stated ideal (stackful work-stealing).
>    Feasible (Go is the proof) but re-accepts D56's cross-thread-migration unsafe as a *primitive*
>    (guest owns the stealing policy; VM enforces single-owner). **Gated on a loom-verified ownership
>    protocol + expert review.** Design + roadmap in `SCHEDULING.md`. Now unblocked: B has landed (and
>    its suspend/wake protocol — the futex park + completion notify — informs the fiber's).
> 3. **Smaller open items:** honor *weak* memory orderings (§12; both backends seq-cst today); fiber/vCPU
>    quota *metering* (§15; the kill path exists, quotas don't); the async-ring
>    pool could grow more offloadable ops. *(Done across recent batches: **DPOR + sleep sets** for
>    `explore_all` — prunes independent-op reorderings and the residual cross-cluster redundancy, sound
>    vs the retained brute-force oracle (`svm/tests/dpor.rs`); and **spin-loop handling** — a busy-wait
>    spinlock is now exhaustively verifiable (parked-until-written, not re-spun; `svm/tests/spinloop.rs`)
>    instead of unbounded. Remaining: full optimal-DPOR via source sets / wakeup trees; multi-op spin
>    bodies.)* **Deferred design decision — narrow integer types (the wasm
>    tradeoff):** `char`/`short`/`_Bool` are `i32` values (no `i8`/`i16` SSA types), so frontends must
>    lower narrowing casts explicitly and **narrow-width atomics (`_Atomic char/short`) have no IR form**.
>    Decision + recommendation written up in **DESIGN.md §3b "Narrow integer types"** — keep the model;
>    if it bites (likely the LLVM on-ramp, or a narrow-atomic workload), prefer the existing
>    `extend8_s`/`extend16_s`/`extend32_s` ops (now lowered on **both** backends — interp + JIT via
>    `ireduce`→`sextend`, ride the 4000-seed differential; `jit_diff::jit_matches_interp_sign_extend_ops`)
>    + a guest-libc CAS-loop for narrow atomics, *not* adding `i8`/`i16` (which would widen the
>    escape-TCB). *(Done this batch: the JIT cap-path page-map persistence
>    (`Host::cap_window_pages` + `MprotectWindow::new_shared`); the **thread-safe guest `malloc`**; and a
>    **chibicc narrowing-cast bug** found via the malloc demo — a value-level cast to `char`/`short`/
>    `_Bool` (which the IR all carry as `i32`) wasn't truncated, so `(char)200`/`(_Bool)200` kept the
>    wrong value (only the *store* width truncated, so `char c = (char)200` worked but an rvalue cast
>    didn't). Fixed in `codegen_ir.c`'s `gen_convert` (`narrow_to`: sign-extend low byte/halfword via
>    shifts, `& 0xFF`/`0xFFFF` for unsigned, `!= 0` for `_Bool` — only ops every backend lowers).
>    Untrusted-frontend (re-verified output was always safe); guard `c_matches_gcc_narrowing_casts`; the
>    byte-heavy demos (sha256/xxhash/jsmn/tinfl) still match `cc`.)*
> 4. **Maintainer one-liners** (need the `workflow` token scope I can't push): apply the nightly **miri**
>    CI job (snippet at commit `60d4f3a`); drop `continue-on-error` from the now-green `cross-os` matrix.

---

#### Build log (landed) — history & rationale

*(Everything below is **done** — Phases 1–3.5, §12 concurrency + its cross-platform port, the
concurrency escape-TCB hardening, the §14 nesting cluster, the §5 kill-path, the security audit, the
M:N demos, and the async-ring (B, increments 1–3c). §10 is the live tracker; §9 the honest-compliance view.)*

The build log, roughly in landing order:

**Nesting / the §14 Instantiator** — the big §1a differentiator: power-of-two sub-window grants +
   attenuated caps + quota, which then unlocks **cross-domain `SharedRegion` `create`/`grant`** (§13)
   and the isolation tiers. Most of the remaining §1a edges live here. *Foundation landed:*
   `svm_mask::Window::sub` (the masking unit, fuzzed) plus a **fully-confined sub-window run path on
   both backends** — `svm_interp::run_capture_sub` / `svm_jit::compile_and_run_capture_sub`, where the
   JIT masking lowering adds `+ base` (`base == 0` elided so top-level codegen is unchanged). It's
   covered by a hand-written + generative interp↔JIT **sub-window escape-oracle** (`escape_oracle.rs`,
   `jit_fuzz` pass 3) that byte-compares the *whole parent* and asserts the child never touched a byte
   outside its slice. *Also landed: the **`AddressSpace` capability + attenuation** (iface 5,
   `Host::grant_address_space`)* — a power-of-two window sub-range whose `map`/`unmap`/`protect` are
   confined to it and whose `sub(off,size_log2)` op **mints a further-attenuated child range** (a
   parent can only sub-allocate what it holds). It runs through the shared `cap_dispatch_slots`, so
   both backends get it for free; covered by an interp↔JIT differential + authority-confinement tests
   (`address_space.rs`). This is the **memory half of the Instantiator** and the project's first
   *attenuation* primitive. *Also landed (interp): the **`Instantiator` capability** itself (iface 6,
   `Host::grant_instantiator`)* — `instantiate(entry, off, size_log2, fuel) -> child_handle` spawns a
   child as a vCPU on the §12 M:N executor confined to a power-of-two sub-window (`Mem::nested_view`
   shares the parent's backing, so the parent sees the child's bytes; masking confines the child to
   its slice), with an attenuated powerbox (an `Instantiator` over the child's **own** window, so it
   can recurse — **confinement composes to any depth**, verified to depth 2) + a fuel quota;
   `join(child)` parks **only the calling fiber** (siblings run) and delivers the child's result/trap.
   Covered by `instantiator.rs` (confinement, depth-2 nesting, out-of-range carve → `-EINVAL`,
   child-trap propagation). This is the chosen first cut: **spawn + explicit join, same-module child,
   interp-first**. *Also landed: the **page-protection coordinate reconciliation*** — `Mem`'s prot map
   is now uniformly keyed **window-relative** (`prot_pages`/`byte`/`set_byte`/`is_backed`/`init_data_at`
   fold the window base out, matching `check_prot`/`page_access`; `map`/`unmap`/`map_region` zero the
   `back` at the base-shifted absolute offset). Identical for a top-level window (base 0); for a §14
   child it makes a sub-window `map`/`unmap`/`protect` actually work (it `-EINVAL`'d before) and also
   hardens the sub-window escape-oracle (RO data segments now fault consistently across backends).
   Covered by `sub_window_page_protection_is_window_relative`. *Also landed: the child now gets a
   **usable `AddressSpace`*** over its own window in its powerbox (its entry takes one or two starter
   handles — `Instantiator`, and optionally `AddressSpace`), and `nested_view` gives each child its
   **own** address-space view (shared bytes, private page protections — a shared map would alias the
   child's pages onto the parent's). Covered by `child_manages_its_own_pages_via_address_space`.
   *Also landed: the **JIT `Instantiator` path*** (interp/JIT parity) — `instantiate`/`join` lower to a
   per-run `Nursery` (`instantiator_rt`) baked into the iface-6 cap.call sites; `instantiate`
   **re-compiles** the child as a top-level guest over its **own** fresh guarded window (DESIGN's
   "nesting cost paid at setup"; reuses the fully-fuzzed top-level confinement — no new escape-TCB
   codegen), seeded from / copied back to the parent's sub-region (the §14 superset materialized at
   join). The detect-and-kill guard (`trap_shim`/VEH) was made **re-entrant** (save/restore the
   recovery state) so a child runs guarded inside the parent's guarded call; a child width-overrun is
   caught by its *own* guard page and propagates as the parent's trap. Authority is resolved through
   the run's `cap.call` thunk (a forged handle is an inert `CapFault`). Covered by `jit_instantiator.rs`
   (interp↔JIT differential: result + whole-window byte-equality, out-of-range carve → `-EINVAL`,
   `unreachable` + width-overrun child-trap propagation). *Also landed (interp): **co-fiber
   resume/suspend*** — the `Yielder` capability (iface 7) + `Instantiator.spawn_coroutine` (op 2) /
   `resume` (op 3). A guest spawns a child confined to a sub-window as a **suspended continuation**
   (its own frames/mem/host + a `Yielder` back to the parent) and drives it cooperatively: each
   `resume(child, v)` runs the child inline until it `yield`s (status SUSPENDED, handing back a value)
   or returns (RETURNED), delivering `v` as the child's yield result; values round-trip both ways,
   confinement holds across suspensions, and a child trap propagates to the parent. This is the §14
   parent-virtualized-fault / lazy-paging primitive (a child parks on a fault the parent services).
   Covered by `coroutine.rs`. *Also landed (interp): **fault-driven yield*** — the actual
   userfaultfd-style lazy-paging. `spawn_demand_coroutine` (Instantiator op 4) starts the child with
   its window **unmapped**; a recoverable in-window page fault (`check_prot`) on a coroutine
   (`fault_yields`) records the confined address (`Mem::last_fault`), rewinds the access, and suspends
   to the parent (`Inner::CoFault`, status FAULTED, value = fault address) instead of trapping. The
   parent supplies the page (writes its bytes into the shared window, then `resume`s — which
   `supply_page`s it, mapping RW without zeroing) and the rewound access re-executes. An *out-of-window*
   fault still traps (the `last_fault` sentinel distinguishes them). Covered by `coroutine.rs`
   (`..._faults_then_resumes`, `..._reports_fault_address`). *Also landed: the **JIT co-fiber path**
   incl. fault-driven yield* — interp/JIT nesting parity is now complete. A JIT coroutine child is a
   **suspended native continuation**: an `svm-fiber` stack (the §12 boost.context substrate) running
   the child's own compilation over its own guarded window, its `Yielder` baked as the child's
   `cap.call` thunk (handle minted as the reference Host's first-grant encoding — guest-visible
   lockstep). The detect-and-kill recovery state is **swappable** (`mem::GuardState`, a C-side
   sigjmp_buf blob / the VEH frame) and the parent swaps it around every switch, so the child's armed
   guard survives suspension. Fault-driven yield is **hardware**: a demand child's window starts
   uncommitted (`GuestWindow::new_uncommitted`); the SIGSEGV/VEH handler — now `SA_NODEFER`, with a
   per-thread registered *demand range* checked before detect-and-kill — suspends the child's fiber
   *from the handler frame*; the parent supplies the page (`commit_range` + committed-page sync) and
   the resume returns into the handler, re-executing the faulting access. Parent slice ↔ child window
   sync at every switch (committed pages) is the cooperative equivalent of the interp's live shared
   backing. Covered by `jit_coroutine.rs` (5 differential tests incl. hardware demand paging).
   *Also landed: **separate-module children*** — the "plugin-in-plugin" story, both backends. The host
   verifies a *different* module and grants a **`Module` capability** (iface 8,
   `Host::grant_module`); the parent passes it to the Instantiator's **module ops** (5
   `instantiate_module` / 6 `spawn_coroutine_module` / 7 `spawn_demand_coroutine_module` — same
   shapes as 0/2/4 with the Module handle prepended; `join`/`resume` unchanged). The child runs the
   foreign module confined to a carve that must **equal its declared memory** (§14 transparency: the
   plugin behaves exactly as standalone); its **data segments materialize into the carve at spawn**
   (so e.g. string literals work — and a demand child gets them **supplied lazily**, page by page);
   RO-segment protection is skipped for nested children (§1 self-corruption non-goal). On the JIT the
   module resolves via a dedicated host callback (`svm_jit::ModuleResolver` / `svm_run::
   module_resolver`, threaded through `compile_and_run_capture_reserved_with_host_ex`) — deliberately
   **not** the `cap.call` surface, so the host pointers it yields are never guest-reachable (the
   generic dispatch on a Module handle is an inert `CapFault`) — and the foreign module is compiled
   at `instantiate` ("nesting cost paid at setup"). Covered by `separate_module.rs` (interp) +
   `jit_separate_module.rs` (differential incl. the lazy-segment fault address byte-exact).
   *Also landed: **cross-domain `SharedRegion` `create`/`grant`*** — the zero-copy parent↔child data
   plane. `create_region(len)` (**`AddressSpace` op 5**) mints a guest-owned region (backing from the
   embedder's factory — `Host::set_region_factory(svm_run::new_shared_region)` under the JIT, so a
   JIT guest can `map` what it mints, proven by the minted-region alias differential; 256 MiB
   per-region anti-bomb cap, §15 quotas later). `grant(coro_child)` (**`SharedRegion` op 4**, eval-loop
   serviced) installs the *same* backing into a suspended coroutine child's powerbox and returns the
   child-side handle, which the parent delivers via the next `resume` value; the child `map`s the
   region into its own window and parent/child share bytes with **no copies** (both directions
   tested). Landing it forced the right coordinate model: the whole **`GuestMem` surface is
   guest-relative** (the zero-based window the guest sees; `Mem` translates to its backing) and
   `AddressSpace`/`Instantiator` bindings record **holder-relative** ranges (translated to
   backing-absolute via the holder's window base at use) — so every capability now composes at any
   nesting depth, not just the ones that pre-shifted. Covered by `region_grant.rs`.
   *Now reaches **stock C***: the powerbox grants `_start` a 5th handle — an `AddressSpace` over the
   whole window — and the libc ships `<svm.h>` (`__vm_region_create`/`map`/`unmap`/`page_size`,
   lowering to `cap.call 5 5` on the AddressSpace and `cap.call 4 {0,1,3}` on the region). `svm-run`'s
   powerbox installs the OS-shared-memory factory unconditionally, so a stock C guest mints a region
   and maps it at two adjacent offsets to build the **magic ring buffer** — a single straddling store
   wraps tail→head as one contiguous access. Verified end to end on both backends
   (`c_ring_buffer_via_minted_region`), plus the minted-region straddle differential
   (`jit_minted_ring_buffer_straddle_matches_interp`). NB: growing the reserved handle region from
   16→32 bytes shifted chibicc's global base (`RESERVED_BYTES`), so all C arg-builders now grant 5
   handles. **Remaining:** (1) `grant` to executor (`instantiate`) children and to **JIT** children
   (the JIT child's powerbox is a baked thunk holding only its Yielder; a JIT child using
   fibers/threads is `Unsupported`); (2) richer cap pass-through; (3) a non-blocking JIT `instantiate`
   child ("park only the calling fiber" — today synchronous; coroutines already interleave
   cooperatively).
   *Also landed: the §5 **fuel/epoch kill-path on the JIT***. The interpreter has always bounded a
   runaway guest via its per-step fuel counter; the JIT now matches it — the lowering polls a
   host-owned interrupt cell (`AtomicU64`) at every loop back-edge **and** function entry (so both
   infinite loops *and* unbounded tail recursion are caught) and traps `OutOfFuel` the moment the
   host sets it. It's **opt-in + guest-undisableable**: armed via
   `compile_and_run_with_host_interruptible` (un-armed compiles are byte-identical — `epoch_addr == 0`
   ⇒ no checks emitted, so the whole differential is unchanged), the guest can't turn the poll off,
   and `svm-run` exposes it on the CLI via `SVM_DEADLINE_MS` (a watchdog thread that wakes early when
   the run finishes, so fast programs aren't delayed). The embedding deadline is now an explicit
   `run_powerbox_with_deadline(module, stdin, Option<Duration>)` arg (the CLI reads the env var and
   passes it — env-reading is CLI policy, not library behaviour); pinned end to end in
   `svm-run/tests/run.rs`: a runaway powerbox guest is killed at the deadline, a fast guest isn't
   delayed, and the **`svm-run` binary** detect-and-kills a C `for(;;){}` (frontend → JIT → watchdog
   → non-zero exit). Differentially tested in `jit_killpath.rs`
   (infinite loop, infinite tail recursion → both backends `OutOfFuel`; armed-finite + unarmed runs
   complete normally). The kill now covers a **whole multithreaded domain**: every vCPU runs the same
   finalized code, so a *spinning* sibling polls the one baked cell on its own; a *parked* sibling
   (blocked in a futex `wait` or `thread.join`) re-checks the cell on a bounded interval
   (`KILL_RECHECK = 20 ms`, real-build only — the loom futex model is untouched) so it wakes and
   unwinds too, and `join_all` never hangs on it. Tested in `jit_killpath_threads.rs` (spinning
   sibling + a sibling parked in an *infinite* futex wait → both killed). And it reaches **nested JIT
   children**: a child (synchronous `instantiate` or a co-fiber `spawn_coroutine`) is compiled to poll
   the *parent's* interrupt cell (threaded through the `Nursery`), so a runaway child trips `OutOfFuel`
   instead of hanging the parent inside the `instantiate`/`resume` call where the parent's own checks
   can't fire — `join` then propagates the child trap and the parent unwinds (tested:
   `jit_killpath_stops_runaway_child`). **The kill-path is now closed across every JIT execution
   context** (root, sibling vCPUs, nested children).
4. **Concurrency loose ends** — the async submit/complete ring (§9/§12) *(done)*, fiber/vCPU quota
   metering, and DPOR to scale `explore_all` past lock-free shapes *(done — `explore_all` is now a
   DPOR checker, sound vs the retained `explore_all_bruteforce` oracle; `svm/tests/dpor.rs`)*.
5. **Language on-ramp** (§14/D54) — the LLVM-bitcode→IR translator (breadth, the differentiator
   vehicle) and/or an optional wasm→IR bridge (compat).

The hard ceiling is unchanged (§2a/§18): *"appears to work"* is well-evidenced; *"is certified
secure"* remains the separate expert-review/audit workstream — not a byproduct of this build.
