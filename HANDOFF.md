# Handoff — C frontend (chibicc → SVM IR) + differential fuzzing

Pick-up notes for a fresh session. Written 2026-06-03, **last updated 2026-06-04**.
Branch: **`main`** (this work has been committing straight to `main`; the remote is
`theSherwood/vm`). Everything below is committed and CI-green.

**Status in one line:** Phase 2 ("real C runs") is **complete** — the C frontend is at the
agreed stopping point (broad subset, two-tier tested) — and we're into Phase 3 (the JIT +
windowed memory + capabilities exist; a generative interp↔JIT differential fuzzer now
guards the JIT). The §3d **SSA-promotion perf pass now exists** (item 8 below): scalar
locals that are never address-taken are promoted to SSA values threaded as block params, so
the JIT register-allocates them — a hot loop body went from ~22 load/store ops to **0**. The
big Phase-3 remainder is production trap-catching (guard pages + signal handler, §4/§5); the
other open item is the §8 verifier escape-oracle fuzzer.

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
- `svm-interp` — reference interpreter (`run`).
- `svm-jit` — Cranelift JIT (`compile_and_run`, `JitOutcome`).
- `svm-mask` — the isolated masking unit.
- `svm` — umbrella crate + integration tests (`crates/svm/tests/`).
- `fuzz/` — libFuzzer targets (out of workspace; nightly + `cargo-fuzz`).

Two big things exist beyond the core loop: (1) **the C frontend** (most of this doc), and
(2) **a generative interp↔JIT differential fuzzer** (see §8). Test crates:
`c_frontend.rs` (C, two tiers), `jit_diff.rs` (hand-written JIT diff), `jit_fuzz.rs`
(generative diff), `pipeline.rs`, `fuzz_smoke.rs`.

---

## 2. The C frontend — what exists

A **vendored fork of chibicc** (Rui Ueyama's small C compiler, MIT) lives in
**`frontend/chibicc/`**. We added one file, **`codegen_ir.c`**, an alternative backend
that walks chibicc's typed AST and emits **our text IR** instead of x86-64 asm, plus a
`--emit-ir` flag. Everything else in `frontend/chibicc/` is upstream chibicc (don't
edit it unless you must; keep the diff small).

### Invocation
```
frontend/chibicc/chibicc -cc1 --emit-ir -cc1-input a.c -cc1-output a.svm a.c
```
`-cc1` runs the compiler in-process (no gcc-style driver subprocess); `--emit-ir`
dispatches to `codegen_ir` (see `cc1()` in `main.c`, where the wiring lives). Build with
`make -C frontend/chibicc` (needs `make` + a C compiler; both present in CI). Build
artifacts (`*.o`, the `chibicc` binary) are git-ignored.

### Test harness (`crates/svm/tests/c_frontend.rs`, 33 tests, two tiers)
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
with `break`/`continue`; functions, parameters, **recursion**, **varargs**; **`printf`**
and `exit` over the powerbox; **`malloc`/`free`/`calloc`** (guest bump allocator). All
verify and run identically on interp + JIT, and match native `cc`.

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

## 5. C-frontend roadmap — items 1–7 all DONE (the agreed stopping point)

The frontend was taken as far as needed for "a capable VM"; items 1–7 below are complete.
Only item 8 (a perf pass) and the inline "Still TODO" notes (by-value aggregate `sret`,
general `goto`, a real RO data segment, `fd`→stream mapping) remain, and none block "C
runs." History order:

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
   `data_end`); a synthetic **`_start`** (function 0) writes initializer bytes then calls
   `main` with the initial data-SP (`data_end`). The harness runs function 0 with **no
   args**. **Note:** uses per-byte init stores, not a real IR data segment — the §3a
   read-only data section (and globals holding pointers/relocations) is still TODO and
   would be a cross-cutting `svm-ir`/text/encode/verify/interp/jit change.
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
   `printf`; `calloc` too. (Real free-list reclamation / heap growth via the `map`
   capability is deferred.) Demonstrated with a heap-allocated linked list of structs.
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
cargo test -p svm --test c_frontend   # 34 tests, all green (interp == JIT, and == cc)
cargo test -p svm --test jit_fuzz     # 4000 generated modules, interp == JIT
```
If those pass, you're oriented.

---

## 8. Generative interp↔JIT differential fuzzer (§18 "interpreter-as-oracle")

The JIT is the only component emitting unsafe machine code, so it gets dedicated fuzzing.

- **`crates/svm/tests/support/irgen.rs`** — a generator of **verifier-valid** IR modules
  *by construction*: typed value pool (constants synthesized on demand), branch/return
  args matched to target param types, **forward-only CFG + call graph (DAGs)** so
  execution always halts, constants biased to boundary values (0, ±1, INT_MIN/MAX, NaN,
  ±inf). Covers the whole scalar op set. `fuzz_one(&mut Gen)` generates → verifies →
  runs interp + JIT → asserts agreement (value-equal or same trap kind; NaN-insensitive).
  `Gen::from_seed` (stable) / `Gen::from_bytes` (libFuzzer).
- **`crates/svm/tests/jit_fuzz.rs`** — stable-CI loop over 4000 seeds (~1.6s).
- **`fuzz/fuzz_targets/diff.rs`** — libFuzzer target (`cargo +nightly fuzz run diff`).

Found no divergences. Natural extensions (if you return to fuzzing): loops/back-edges
(needs a JIT step-cap or fuel), `call_indirect`/`cap.call`, and a **final-memory +
escape-freedom** assertion (no out-of-window access) — which would realize the §18
de-risking move *"fuzz the verifier: verified ⇒ cannot escape,"* the one
design-recommended validation still missing (today `fuzz_smoke`/`decode_verify` only
check "verify never panics / verified modules don't panic the interp," not escape).

---

## 9. Where the project stands vs DESIGN.md (compliance, honest)

Largely compliant; simplifications are the ones the design *sanctions*, deferrals are
incompleteness not contradiction:
- **Phase 2 complete** (real C on interp + JIT). Solidly into **Phase 3** (JIT + masked
  window + caps done). Phase-3 remainder = production trap-catching (guard pages + signal
  handler, §4 still ⬜/parked = "fixed-size window, eager mapping" MVP, which is what we
  do) and demand paging (deferred).
- **§2a escape-TCB intact:** the frontend is untrusted; all its output is re-verified;
  every memory access is masked, so even a buggy/hostile data-SP cannot escape (the
  data-SP is a plain value, not trusted). Making it an explicit value rather than a
  register-pinned `vmctx` slot is exactly the "lowering detail" §3d calls it.
- **§3d implemented as a documented subset:** everything-in-memory **plus the SSA-promotion
  reverse pass** (non-address-taken full-width scalars → SSA values; narrow scalars and
  address-taken/aggregate locals stay in memory), flat-buffer varargs, guest `malloc` over
  the window, LP64 + pinned `char`/`long double`. The promotion split (SSA value vs
  data-stack slot) is exactly the §3d "local classification" — minus the data-SP being
  register-pinned in `vmctx`, which is still a plain threaded value. **Deferred SETTLED
  features (not contradictions):** by-value aggregate args/returns by hidden pointer (D39),
  const→RO data segment via `protect` (D40), a real IR data section (we use `_start`
  byte-stores), and narrow-scalar promotion.
- **De-risking moves from §18 now in place:** interpreter-as-oracle differential fuzzing
  (§8), masking-unit fuzzing (`fuzz/mask`), Cranelift backend. **Still missing:** the
  verifier escape-oracle fuzzer (see §8).
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
- [ ] **Phase 3 — Solid MVP (in progress):** the MVP remainder below.
- [ ] **Phase 4 — post-MVP:** deferred (below).

### Phase 3 / MVP remainder (what's left to call it a "Solid MVP")
- [ ] **Production trap-catching** — guard pages + a signal handler → §5 detect-and-kill.
  *The big one.* Today: masking confines and the interp/JIT detect traps via in-code
  checks; there is **no hardware-fault path** (see `svm-jit` ~L133, marked as where it
  goes). Systems-fiddly, debug-heavy — §18's fat-tail phase.
- [ ] **Real window / Memory capability** — pin page size + masking constant + guard-page
  placement; make `map`/`unmap`/`protect` real. Today they are **no-op stubs**
  (`svm-interp` ~L765) over a fixed-size, eagerly-mapped window; `malloc` is a guest bump
  allocator, not backed by `map`. §4 is "parked" at the MVP simplification.
- [ ] **Verifier escape-oracle fuzzer** — see Fuzzing below; the one expert-free security
  validation still missing.
- [ ] *(optional, deferred even within MVP — not blockers)* by-value aggregate args/returns
  (`sret`, D39); a real RO data segment (§3a/D40, vs `_start` byte-stores); general `goto`.

> **Ceiling reminder (§18):** the MVP target is *"appears to work"* — well-evidenced now.
> *"Is certified secure"* is **not** an MVP deliverable; it's a separate, open-ended
> post-MVP workstream (expert review + audit). Green tests ≠ secure.

### Phase 4 / post-MVP (DESIGN-specified, none built)
- [ ] Concurrency: fibers / vCPUs / M:N green threads, atomics, the C11 memory model,
  real threads (§12).
- [ ] **Nesting (§14)** + **shared memory + isolation tiers (§13)** + **real guest-visible
  virtual memory** — *most of the §1a differentiators live here.*
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
- [x] `fuzz/mask` (libFuzzer): the confinement-masking unit — masked address always in
  `[0,size)` (D38, the escape hinge).
- [x] `roundtrip` (libFuzzer): encode∘decode identity.

Gaps (priority order):
- [ ] **Escape-oracle — highest value.** Assert *verified ⇒ every memory access stays
  in-window* (final-memory + no-out-of-window check on interp **and** JIT), by extending
  `irgen.rs`. Today only "doesn't panic" is checked, **not escape** — AGENTS.md invariant
  #1 / §18 "the one security validation that needs no expert in the loop." (Was the
  alternative to the SSA-promotion pickup this session.)
- [ ] **CI nightly only runs `decode_verify`.** The `mask` target (the escape hinge!) and
  `diff` are in no scheduled job — add them to the nightly matrix (`ci.yml`).
- [ ] **Generator coverage holes:** `irgen` emits forward-only DAG CFGs — **no
  loops/back-edges** (needs a JIT step-cap/fuel) and **no `call_indirect`/`cap.call`**. The
  generative differential never exercises them (only the hand-written C tests do).

### Benchmarking — have vs. gaps
Have (✅):
- [x] `crates/svm/src/bin/bench.rs`: decode / verify / **interp** throughput on one
  hand-written loop (`sum 0..N`), ns/iter, dependency-free.

Gaps (the weakest area vs. AGENTS.md "benchmark early · measured vs. wasm/Wasmtime · catch
regressions one commit old"):
- [ ] **No JIT benchmark** — only interp; the JIT *is* the perf story.
- [ ] **No wasm/Wasmtime baseline** — the §1a compute-parity / around-compute thesis is
  unvalidated.
- [ ] **No compile / JIT-latency measurement** — the "SSA on the wire" startup claim (§1a)
  is unmeasured.
- [ ] **No over-time tracking / no CI bench job** — it prints and forgets; a regression
  isn't caught "one commit old."
- [ ] **No C-frontend program benches** — e.g. the SSA-promotion win (loop body ~22→0
  memory ops) is uncaptured; nothing would flag it if promotion regressed.

### Suggested next pickups (ranked)
1. **Escape-oracle fuzzer** — security spine, well-scoped, builds on `irgen` (§8).
2. **JIT + Wasmtime comparison bench with regression tracking** — validates the parity
   thesis and would guard perf work like the promotion pass.
3. **Cheap wins** — wire `mask`/`diff` into nightly CI; add loops + indirect calls to the
   generator.
