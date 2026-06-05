# Handoff — C frontend (chibicc → SVM IR) + differential fuzzing

Pick-up notes for a fresh session. Written 2026-06-03, **last updated 2026-06-04**.
Branch: **`main`** (this work has been committing straight to `main`; the remote is
`theSherwood/vm`). Everything below is committed and CI-green.

**Status in one line:** Phase 2 ("real C runs") is **complete** — the C frontend is at the
agreed stopping point (broad subset, two-tier tested) — and we're into Phase 3 (the JIT +
windowed memory + capabilities exist; a generative interp↔JIT differential fuzzer now
guards the JIT). The §3d **SSA-promotion perf pass now exists** (item 8 below): scalar
locals that are never address-taken are promoted to SSA values threaded as block params, so
the JIT register-allocates them — a hot loop body went from ~22 load/store ops to **0**.
Memory **detect-and-kill** now exists too: an `mmap`'d window + `PROT_NONE` guard page + a
SIGSEGV/SIGBUS handler turn an out-of-window fault into a clean `MemoryFault` (§4/§5, unix).
The remaining Phase-3 memory work is the *large* reserved window (the §4 perf/VM model). The
§18 verifier escape-oracle now exists (the differential byte-compares the final guest window
across interp + JIT: verified ⇒ in-window) — see §8 / §10.

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
(Coverage gap noted: the generative `jit_fuzz` exercises `call_indirect` but not `ref.func`,
which is why this JIT gap surfaced only via the C tests — worth adding to the fuzzer.)

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
Loops/back-edges, `call_indirect`, and `cap.call` (inert/ungranted ⇒ both-`CapFault`) are now
generated (the trap-kind is no longer asserted when both backends trap — see §10); out-of-
allocation accesses now fault into the guard page and are caught as `MemoryFault` (§4/§5).
Remaining: the cap.call *success* path (a mock powerbox granted to both backends) and
float-module memory coverage (NaN bits aren't pinned across backends).

---

## 9. Where the project stands vs DESIGN.md (compliance, honest)

Largely compliant; simplifications are the ones the design *sanctions*, deferrals are
incompleteness not contradiction:
- **Phase 2 complete** (real C on interp + JIT). Solidly into **Phase 3** (JIT + masked
  window + caps + **guard-page/signal detect-and-kill** done). Phase-3 remainder = the §4
  *large* reserved window + demand paging (still the "fixed-size window, eager mapping" MVP
  today), which the new guard-page/signal foundation is built to extend.
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
  (§8), masking-unit fuzzing (`fuzz/mask`), Cranelift backend, **the verifier escape-oracle**
  (verified ⇒ in-window final memory, §8/§10), **and guard-page/signal detect-and-kill**
  (§4/§5, unix) so a gross out-of-window access faults cleanly rather than corrupting the host.
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
- [ ] **Phase 3.5 — Cross-platform parity:** port to **Windows** (PAL:
  `VirtualAlloc`/`VirtualProtect` + **VEH/SEH** detect-and-kill) and validate
  **macOS** (Mach-exception path, which can intercept ahead of BSD signals); stand
  up a gating **Linux/Windows/macOS CI matrix** so parity holds from here on.
  Confinement masking is already portable (§16/D51); only the non-TCB PAL differs.
  Starting point: the JIT `compile_error!`s off unix, the guard/detect-and-kill
  path is `cfg(unix)`, and CI is `ubuntu-latest` only — this phase makes all three
  OSes first-class. Tier-1 MPK stays Linux-only (degrades to tier 0/3 elsewhere).
- [ ] **Phase 4 — post-MVP:** deferred (below), developed against the parity matrix.

### Phase 3 / MVP remainder (what's left to call it a "Solid MVP")
- [x] **Production trap-catching (memory)** — *done (unix)*: the JIT window is now `mmap`'d
  with a trailing `PROT_NONE` **guard page**, and the entry runs under a SIGSEGV/SIGBUS
  handler (`crates/svm-jit/src/{mem.rs,trap_shim.c}`, a small `cc`-built C shim for sound
  `sigsetjmp`/`siglongjmp`). A fault in the window's guarded range unwinds out of the call as
  `TrapKind::MemoryFault` — §5 **detect-and-kill**, host survives — instead of corrupting it.
  Confinement is still the masking lowering; the guard is the safety net (width-overrun at
  the top now faults cleanly, and a masking/elision bug faults locally instead of corrupting
  the host). `cfg(unix)`; other targets fall back to the old heap window (no guard).
  Verified non-vacuous by `escape_oracle::guard_page_fault_is_detect_and_kill`; whole suite +
  4000 fuzz seeds green (the handler is exercised by width-overruns). **Not yet:** the
  *perf*-unlocking guard-when-bounded (needs a large window — below); div/rem/trunc still use
  explicit in-code trap checks (correct; converting them to #DE faults is optional).
- [ ] **Real window / Memory capability** — pin page size + masking constant + the *large*
  reserved window; make `map`/`unmap`/`protect` real. **Largely done now** (see suggested
  pickups #1 and #3): the large reserved-window model is the default (`DEFAULT_RESERVED_LOG2
  = 40`, masking constant `reserved - 1`), and `map`/`unmap`/`protect` are **real**, not
  stubs — the interp `Mem` enforces a per-page protection map (`svm-interp`, the `map`/
  `unmap`/`protect` impls + `check_prot`), and the JIT side uses real `libc::mprotect` on the
  window pages, differentially fuzzed. `malloc` is still a guest bump allocator, not backed by
  `map`. **Still left** (so this stays unchecked): **demand paging** on fault, **growth**
  into the reserved tail (sparse address space, §98), and surfacing the Memory cap in the
  *main* irgen fuzzer. The guard-page + signal foundation above is what demand paging builds on.
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

Gaps (priority order):
- [~] **`cap.call` — *inert (fault) path generated*, success path not; loops still one shape.**
  The generator now emits `cap.call` with an **ungranted** handle (`gen_inst` arm 18): the fuzzer
  grants no caps, so it is inert on both backends (interp empty `Host` / JIT `empty_cap_thunk` ⇒
  both `CapFault`, agreeing under the both-trap rule) — the I2 check for capabilities (§3c, "a
  forged handle is inert") and the first generative exercise of the JIT's cap.call lowering
  (handle marshalling + thunk ABI + trap plumbing). A coverage-guard test asserts it's produced.
  **Still TODO:** the *success* path needs a deterministic **mock powerbox** granted identically
  to both backends (a `run_capture`-with-host on interp + a `compile_and_run_capture`-with-thunk
  on JIT, both already nearly present) so a returning cap.call is differentially tested too.
  Loops are still a single counted shape (no nested/irreducible loops, no data-dependent trip
  counts); richer loop shapes need a JIT step-cap/fuel to stay terminating.
- [ ] **Escape-oracle excludes float modules** (NaN-payload nondeterminism). A canonical-NaN
  normalization, or comparing only integer-store bytes, would extend coverage to them.
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
  `bench/`; `--csv` for a line per kernel. **Representative numbers** (ratio = svm ÷ wasm;
  `<1` = svm faster; machine-dependent — watch the *ratio*, not the absolute ns):
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
- [ ] **Residual wasm32 gap (~1.2–1.36×)** needs the *full* guard-when-bounded: real **guard
  pages** so even addresses we *can't* prove bounded (and the common data-SP–relative C
  locals, where `sp` is an unbounded block param) get the wasm32 zero-instruction access.
  That ties into Phase-3 trap-catching (guard pages + signal handler, §5). Also: the elision
  is per-block (block params = unknown); proving the threaded data-SP bounded would extend it
  to C locals.

### Suggested next pickups (ranked)
1. ✅ **Large reserved window → guard-when-bounded** (§4) — **DONE** (Increments 2–4 below; the
   final SP-elision step was decided *against*, D50). The decoupled `reserved`/`mapped` model is
   the default: a large reserved range with only `mapped` backed, out-of-`mapped` → detect-and-kill.
   *Original framing:* a multi-GB reserved window so 32-bit-bounded indices fit under the guard and
   the JIT can elide the mask without a proof (the wasm32 fast path), closing the residual gap incl.
   data-SP–relative C
   locals. **Plan:** (1) ✅ a `bench/` **`locals_c`** kernel (address-taken `volatile` stack array
   ⇒ per-iter `sp + (i&255)*8`, `sp` an unbounded i64 block param ⇒ masked every access) now
   measures the case — it starts at **2.26× vs wasm32**, the worst kernel and the target metric
   (memsum/scatter are already pre-elided, so they don't show it). (2) ✅ decoupled `reserved`
   (mask domain) from `mapped` (fault bound) in `svm-mask`: `Window::with_mapped(reserved_log2,
   mapped)` + `reserved()`/`mapped()` accessors; `confine` masks into `[0, reserved)`, `checked`
   faults outside the backed `[0, mapped)`. `new` stays fully-mapped (`mapped == reserved`) and
   `size()` aliases `reserved()`, so **no behavior change** and no caller churn; a second property
   test + the `mask` fuzz target now drive the split (incl. the unmapped-tail fault). (3) ✅ both
   backends adopt the decoupled model in lockstep: JIT `GuestWindow::new(mapped, reserved)`
   reserves a **host-configured** large window (§4: "e.g. 2^40, host-configurable" — *not* a fixed
   2^32; capped at `MAX_JIT_RESERVED_LOG2 = 2^40`) as `PROT_NONE`+`MAP_NORESERVE` (a huge reserve
   costs only VA) + guard page, maps `mapped` RW; mask const = `reserved-1`; elision threshold →
   `reserved`. Interp `Mem::with_reservation` mirrors it. Out-of-`mapped` accesses now **fault**
   instead of wrapping (the I1 change). Reservation is host policy threaded through the `_reserved`
   capture entries (`run_capture_reserved` / `compile_and_run_capture_reserved`), **not** baked
   into `svm-mask` (still policy-free); default everywhere is fully-mapped (`reserved == mapped`),
   so existing callers are unchanged. Tested: `escape_oracle::reserved_tail_access_faults_identically`
   + `reserved_in_mapped_access_matches` pin the semantics, and the generative fuzzer
   (`run_differential`) runs a **second `reserved > mapped` pass** so the 4000 seeds + `diff`
   libFuzzer target exercise the large reservation, mask/elision-to-`reserved`, and interp↔JIT
   trap-agreement on tail faults. (3b) ✅ **flipped the production default** to the §4 large-reserved
   model: `svm_ir::DEFAULT_RESERVED_LOG2 = 40` (host policy, shared by both backends so they stay in
   lockstep), applied by the non-`_reserved` `run`/`compile_and_run` entries. Out-of-`mapped`
   accesses now **fault by default** (detect-and-kill, demand-paging-ready) — valid programs are
   unaffected (all c_frontend/jit_diff/pipeline tests pass; only one wrap-asserting test was updated
   to the fault model: `pipeline::confinement_faults_out_of_window_address`). Bench confirms it's
   perf-neutral (same instruction sequence; memsum/scatter still pre-elide since their ub `< 2^40`).
   (4) ❌ **decided NOT to pursue (D50)** — the remaining `locals_c` ~2.26× wasm32 gap (data-SP
   relative `sp + dyn_offset`, `sp` an unbounded `i64` block param) is an **accepted cost** of the
   64-bit model. **Key soundness finding (don't reopen the dead ends):** eliding needs the address
   *provably `< reserved`*. Masking `sp` alone does **not** work — `sp & (reserved-1)` leaves
   `sp+offset > reserved` (un-elidable), and `sp & (mapped-1)` **diverges from the interp** (which
   masks the *full* address to `reserved`, then faults outside `mapped`) for any `sp ≥ mapped` → a
   spec mismatch. The only **sound** elision is the wasm32 trick: compute window addresses in
   **32-bit arithmetic** so the address is `< 2^32` *by construction* (`ub_of(ExtendI32U)` already
   handles it) and the interp computes the same 32-bit value ⇒ no divergence. That caps the elided
   window at 4 GiB and reworks the frontend pointer model (`#define SP` is `i64`) for one benchmark
   — **not worth trading the clean 64-bit address space** (D50). `locals_c` stays a tracked metric
   (no further regression), and it still beats wasm64.
2. ~~**Over-time bench tracking**~~ — **DONE** (`bench/ --save-baseline`/`--check` vs committed
   `bench/baseline.txt`, ratio-based, non-vacuous; `alu_c` chibicc kernel tracks the SSA-promotion
   win end-to-end at ≈parity — see Benchmarking gaps); a non-gating nightly CI `bench` job runs `--check`.
3. **Real Memory capability** (`map`/`unmap`/`protect` beyond no-op stubs) — guest-visible
   virtual memory (§1a differentiator); also lets the fuzzer generate `cap.call`. Now in
   progress (unblocked by the reservation work). **Increment 1 ✅ (interp spec):** `Mem` carries a
   per-page protection map (`PageProt::Ro`/`Unmapped`, absent ⇒ rw); `load`/`store` enforce it
   (`check_prot`); `GuestMem` gained `map`/`unmap`/`protect` (default no-op; interp `Mem`
   implements them within `[0, mapped)` — `protect`→RO for D40, `unmap`→fault, `map`→re-commit
   zeroed; misaligned/out-of-range ⇒ `-EINVAL`); `cap_dispatch_slots`' Memory arm calls them.
   White-box `prot_tests` pin the semantics. **Increment 2 ✅ (JIT side + differential):** the
   `jit_diff` cap-thunk now wraps the window as `MprotectWindow` (a `GuestMem` whose
   `map`/`unmap`/`protect` call real `libc::mprotect` on the window pages; `read`/`write` like
   `WindowMem`) instead of the no-op `WindowMem` — so a `protect`ed page is genuinely RO and a
   store to it faults into the guard → `MemoryFault`. `jit_cap_memory_protect_read_only_faults_store`
   pins it: the interp (page-map) and JIT (mprotect+guard) both detect-and-kill on a post-`protect`
   store, non-vacuously (a no-op JIT `protect` would diverge). Added `libc` as an svm dev-dep.
   **Increment 3 ✅ (generative fuzzing + 2 bug fixes it surfaced):**
   `jit_cap_memory_protect_map_unmap_differential` generates 500 random map/unmap/protect + store/
   load sequences and asserts interp (page-map) == JIT (mprotect+guard) on result/trap. JIT-side
   `map` now zero-fills (parity with the interp), so map-after-unmap is covered. Two real bugs the
   fuzzer caught: **(a)** `run_inner` always snapshots `window.rw_mut()[..mapped]` after the run, so
   a guest-`unmap`ped (`PROT_NONE`) page made the snapshot read fault *outside* the guarded call and
   crash the host → fixed with `GuestWindow::restore_rw()` (mprotect the backed region RW before the
   snapshot). **(b)** the JIT passed `mem_size = reserved` (the mask domain, 2^40) to the cap thunk
   instead of the backed `mapped` extent, so buffer borrows / Memory-cap ops bounded against the
   wrong size → now threads `mapped` into `Lower` and passes it. **Deferred (increment 4+):** growth
   (`map` into the reserved tail = sparse address space, §98); demand paging on fault; surfacing the
   Memory cap in the *main* irgen fuzzer (capture path needs restore_rw, now in place) — **next is
   (1): a guest consumer**, D40 const→read-only data segment via `protect` at `_start`.

*(Done this session: SSA-promotion pass; the escape-oracle fuzzer (+ nightly `diff`/`mask`
CI, merged); the JIT-vs-Wasmtime bench harness; mask elision for provably-bounded accesses;
loops + indirect calls in the generative fuzzer; guard pages + signal-handler detect-and-kill;
**over-time bench regression tracking** (`bench/ --save-baseline`/`--check` vs a committed
ratio baseline, + an `alu_c` chibicc-compiled kernel tracking the SSA-promotion win end-to-end;
a non-gating nightly CI `bench` job running `--check`); **a structural SSA-promotion guard**
(`c_frontend` asserts zero loop-body memory ops on
promotable loops, so the promotion win can't silently regress).)*
