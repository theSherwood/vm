# FRONTEND.md — the C frontend (chibicc → SVM IR)

Implementation reference for the C frontend: what C is supported, how it lowers to
the IR, and where to add things in `codegen_ir.c`. The *design* of the C ABI lives in
`DESIGN.md` §3d (section numbers like "§3d" below refer to it); this doc is the
implementation map. The frontend is **outside the escape-TCB** (§2a): the verifier
re-checks whatever it emits, so a frontend bug is a clean error, never an escape.

> A second/third frontend exists alongside this one: `svm-wasm` (core wasm → IR,
> `WASM.md`) and `svm-llvm` (LLVM bitcode → IR, `LLVM.md`). This doc is the C/chibicc
> frontend only.

---

## 1. What this is

A **vendored fork of chibicc** (Rui Ueyama's small C compiler, MIT) lives in
**`frontend/chibicc/`**. We added one file, **`codegen_ir.c`**, an alternative backend
that walks chibicc's typed AST and emits **our text IR** instead of x86-64 asm, plus a
`--emit-ir` flag. Everything else in `frontend/chibicc/` is upstream chibicc (don't
edit it unless you must; keep the diff small).

**Upstream `parse.c` fixes** (the only edits outside `codegen_ir.c`), all genuine chibicc
bugs found by compiling real libraries, each validated against a gcc matrix + the full
suite with zero regressions:
1. `struct_designator` special-cased only anonymous *structs*, so a designator targeting an
   anonymous *union* member dereferenced a NULL `mem->name` → **segfault**. Now matches the
   canonical `get_struct_member` idiom (`TY_STRUCT || TY_UNION`).
2. `struct_initializer2` skipped the separator comma only on non-first members, but it is also
   entered right after a *designated* member (tok at the comma) when that member lands in a
   nested anonymous aggregate — so a following designator (`{ .a = x, .b = y }`) failed to
   parse. Now skips a leading comma when present (handling both callers: designated
   continuation at a comma, and brace-elision at a value).
3. `enum_specifier` parses `__attribute__((packed))`/`__packed__` and sizes the enum to the
   smallest integer type holding its values (1/2/4/8 bytes) — gcc parity (chibicc sized every
   `enum` as `int`); `gen_load`/`gen_store` access a packed enum at that width. This matters
   for host↔guest data exchange (a host writing structured data into the window must agree on
   layout; §3d pins x86-64-SysV). Guarded by `c_matches_gcc_packed_enums`.
4. `static_assertion` — `_Static_assert` (C11) / `static_assert` (C23) at file and block scope
   (was parsed as a function call). Guarded by `c_matches_gcc_static_assert`.

A minimal `frontend/chibicc/include/stdint.h` ships too: without it, `#include <stdint.h>`
pulled the system `<sys/cdefs.h>`, which — because chibicc isn't `__GNUC__` — `#define`s
`__attribute__(x)` to nothing, **silently stripping the attribute** before the parser saw it.

### Real libraries that run end-to-end
Each is a vendored third-party C library, compiled through the frontend, verified, and run on
**both backends**, matching a native `cc` build byte-for-byte (the tests are in `svm-run` /
`crates/svm/tests/c_frontend.rs`):

- **Clay** (UI layout, ~5k-line header; the capstone) — `demos/clay/`, ~93k lines of IR. Drove
  the bulk of the frontend/IR/JIT fixes: anonymous-aggregate designated inits (the two `parse.c`
  fixes above), ternary-returns-struct (`gen_cond` carries the arm's *address*, merge type i64),
  >16-byte struct returns (skip chibicc's SysV hidden return-buffer param — our §3d sret covers
  every size), mixed-width shifts (a shift keeps its amount's own width — widen/narrow to the
  value's width before `iN.shl/shr`), full-u32 `i32.const` in svm-text (`0xFFFFFFFF` = -1),
  program-sized windows (size the window to globals/BSS + a stack reserve; Clay's ~250 KB arena
  needs `memory 21`, small programs keep 64 KB), and a contiguous JIT code arena
  (`ArenaMemoryProvider`: code+rodata from one 256 MiB arena, so ASLR can't place them > 2 GiB
  apart and overflow cranelift's 32-bit PC-relative relocations). After the packed-enum fix, all
  80 Clay struct sizes and `Clay_MinMemorySize` match gcc exactly.
- **jsmn** (JSON tokenizer; char/state-machine scanning, zero alloc) — `demos/jsmn/`. Ran
  byte-identical on the first try, no new fixes — evidence the frontend is robust after Clay.
- **SHA-256** (B-Con) and **xxHash** (XXH32/XXH64) — `demos/sha256/`, `demos/xxhash/`. Integer/bit
  shapes; matched the standard vectors. Drove: `func_index` no longer segfaults on an
  undefined-function call (a libc decl has no source token) — clean error now; and the
  `_Static_assert` support above.
- **tinfl / miniz inflate** — `demos/tinfl/`. A coroutine-style state machine (deeply nested
  `switch` + saved PC, bit-buffer shifts, Huffman tables, a 32 KiB LZ77 dictionary in the struct).
  Byte-identical, no new fixes — the goto/switch lowering and struct layout hold under a gnarly
  real state machine.
- **stb_perlin** — `demos/perlin/`. The first **float-heavy** real program (dense f32 dot
  products, the quintic ease polynomial, trilinear lerps, int↔float, octave accumulate). Prints
  fixed-point so any f32 divergence shows in the digits; matched byte-for-byte, no new fixes.
- **tiny-regex-c** — `demos/regex/`. A Rob-Pike-style **backtracking** matcher (`re_match`
  recurses and retries) — a control-flow workout for data-stack threading and goto/branch
  lowering. Matched native, no new fixes.

### Invocation
```
frontend/chibicc/chibicc -cc1 --emit-ir -cc1-input a.c -cc1-output a.svm a.c
```
`-cc1` runs the compiler in-process (no gcc-style driver subprocess); `--emit-ir`
dispatches to `codegen_ir` (see `cc1()` in `main.c`, where the wiring lives). Two more
flags: `-g` also emits the SVM debug-info section (file/line table, function names,
structured types, `debug.var` — `DEBUGGING.md` §6), and `--child-entry` emits function 0
with the §14 child ABI (`(i64 starter) -> (i64 status)`) so a compiled-C command is
spawnable as an `instantiate_module` child — a shell "exec" (`STAGE1.md`;
`crates/svm/tests/stage1_exec_command.rs`). Build with
`make -C frontend/chibicc` (needs `make` + a C compiler; both present in CI). Build
artifacts (`*.o`, the `chibicc` binary) are git-ignored.

### Test harness (`crates/svm/tests/c_frontend.rs`, two tiers)
`make`s the fork once, compiles each C snippet to IR, **verifies it**, then:
- **Tier 1 (all tests):** runs `main` (function 0 = `_start`) on **both the interpreter
  and the JIT** under identical mock powerboxes and asserts they agree on result, trap,
  and captured stdout/exit. Every C test is also a JIT differential test (§3 parity invariant).
- **Tier 2 (`c_matches_gcc_*`):** compiles the *same* C with native **`cc`** (real
  stdio/stdlib) and asserts identical exit code + stdout — a real-compiler oracle for C
  semantics. Incl. recursion (Ackermann), floats, printf, bubble sort, sieve, linked list.
  Needs `cc` (already required to build the fork).
```
cargo test -p svm --test c_frontend
```

### What C is supported today
`int`/`long`/`char`/`short`/`_Bool`/`enum`, `float`/`double`; pointers, arrays,
structs/unions (`.`/`->`, indexing, initializers); globals + string literals (incl. pointer
initializers / relocations); the full operator set incl. short-circuit `&&`/`||`/`?:`;
`if`/`else`/`while`/`for`/`do`/`switch` with `break`/`continue` and **general `goto`/labels**;
functions, parameters, **recursion**, **function pointers** (indirect calls via
`call_indirect`, dispatch tables, callbacks, fn-ptr struct members), **by-value structs/unions**
(passed/returned by value, whole-aggregate assignment), **varargs**; **`printf`** and `exit`
over the powerbox; **`malloc`/`free`/`calloc`/`realloc`** (guest allocator, heap grows via the
Memory cap). All verify and run identically on interp + JIT, and match native `cc`.
An unmodified **`main(argc, argv)`** also runs: the synthetic `_start` parses the §3e args
buffer (`POWERBOX_ARGS_BASE`) into a real `argv[]` and calls it (`STAGE1.md`;
`crates/svm/tests/stage1_argv_main.rs`). And a guest **definition** of `write`/`read`/`exit`
shadows the builtin (the builtins apply only to declared-but-undefined names) — which is how
a personality libc owns those names over §7 imports; real compiled C runs on the POSIX
personality this way (PROCESS.md S15(b); `crates/svm/tests/c_posix.rs` / `c_shell.rs`,
`POSIX.md`).

Anything unsupported is a **hard `error_tok`** (with the AST node kind), by design — we never
emit IR we can't stand behind.

**Remaining minor gaps** (none block "C runs"): narrow-scalar (`char`/`short`/`_Bool`) SSA
promotion (they stay in memory so store-truncation keeps happening); `volatile` is not honored
(chibicc discards the qualifier — no regression vs the old memory path); `fd`→stream mapping
in the raw powerbox builtin (`write`/`read` still ignore the fd — always the std stream; the
POSIX-personality path does map fds, incl. a distinct stderr and an fd table — `POSIX.md`);
float varargs beyond `double`; `%`-width/precision
in the mini-printf; and free-list reclamation in the guest allocator.

---

## 2. The lowering model (read this before extending `codegen_ir.c`)

**Everything-in-memory, with a threaded data-stack pointer** — *then* the SSA-promotion
pass lifts the easy locals back out. The base model is chibicc's own "allocate all locals
to memory first" (DESIGN §3d); promotion (the documented "reverse" pass that matters for
speed) runs on top of it. **A promoted local is no longer in memory at all:** it is a
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
- **Functions are ordered with `main` first**, behind a synthetic **`_start`** (function 0,
  `emit_start`) when `main` exists — so real functions begin at `start_off` (1) and `main` is
  function index 1; `call` targets a function by this index (`funcs[]` / `func_index`).
- **The harness runs `_start` with no arguments** — the powerbox is bound **by name**, not
  positionally (`export "_start" 0`; PROCESS.md S15(c)): `_start` resolves only the caps the
  program actually uses via `cap.self.resolve` and stashes the handles in the reserved low
  window bytes (`RESERVED_BYTES` = 32). It then calls `main` with the initial data-SP baked
  to `data_end` (the end of globals/BSS), so `&local` (= `sp + offset`) is never `NULL`.

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

### By-value aggregates (sret, §3d D39)
Every by-value struct/union goes by hidden pointer (no SysV register classification). A
**struct/union return** makes the IR function `(i64 sp, i64 sret, params…) -> ()`: the caller
passes the address of chibicc's `ret_buffer` (an lvar in the caller frame) as a hidden first
arg, the callee writes the result through it, and the call's value is that buffer address (so
`f(x).field` and `s = f(x)` work — `gen_addr(ND_FUNCALL)` returns it). A **by-value struct/union
arg** is passed as the lvalue address (`pass_irty`=i64); the callee `gen_memcpy`s it into its own
frame slot in the prologue (by-value semantics). **Whole-aggregate assignment** is a `gen_memcpy`.
Two chibicc quirks handled: a same-type aggregate cast on an assignment rhs (`gen_convert`
no-ops when held by-address), and **union first-member init** — chibicc emits `v.i = (int)expr`,
an aggregate→scalar cast that `gen_convert` lowers as a *load* of the member's bytes (only
array/function decay returns the address). `irty(TY_FUNC)`/`is_agg`/`pass_irty`/`gen_memcpy` are
the helpers.
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

### General `goto`/labels
Each C label maps to one IR block keyed by chibicc's resolved `unique_label` (`label_block_of`,
reset per function); the block number is allocated on first reference — label *or* a forward
`goto` — which is sound because svm-text resolves block targets **by name**, not position
(`labels: HashMap<String,u32>` over appearance order). `ND_LABEL` falls into its block (if
reachable) then `open_block`s it; `ND_GOTO` (after the existing break/continue match) branches to
the target block, threading the data-SP + promoted locals via `cvals()` — identical to loops. The
ND_BLOCK dead-code drop also keeps `ND_LABEL` (a goto target reopens a reachable block).
*Limitation:* a label buried inside a compound statement that is skipped as dead code after a
terminator won't be emitted (goto-into-nested-block); labels at block/function scope — the
cleanup/retry/state-machine idioms — work.

### Global pointer initializers / relocations
A global initialized with a pointer (`char *p = "..."`, `&global`, `&arr[k]`, function pointers,
and arrays/structs of them) carries a chibicc relocation chain (`g->rel`: `{offset, char **label,
addend}`). `emit_data_segments` resolves each at compile time — every global's window offset
(`layout_globals`) and function's funcref index (`funcs[]`) is already assigned — and patches the
8-byte little-endian value (`symbol_value(target) + addend`) into the data image, emitted as an
ordinary `data`/`data ro` segment. A function-pointer target resolves to its funcref index (§3c),
so global dispatch tables compose with `call_indirect`. No runtime relocation step; nothing
relocation-specific reaches the IR/verifier/JIT (it's just bytes). Tests: interp↔JIT differential
+ native-`cc` oracle (pointer-to-global, array-element addend, pointer-to-pointer,
struct-with-pointer-member, global fn-ptr tables, string-literal `char*`, array-of-`char*`).

### Indirect calls (function pointers)
A function designator decays to its `ref.func` index (an i32 funcref, §3c) widened to the 8-byte
C pointer rep (`irty(TY_FUNC)`=i64, `by_address` true so a "load" is a no-op returning the
funcref). A call through a value lowers to `call_indirect (i64 sp, params…[, i64 va]) -> (ret)
<i32-wrapped idx>(csp, args…)`; the signature **must include the leading data-SP `i64`** so the
runtime type-id check (`table_lookup`) matches the target. A type-confused/forged index is inert —
it traps `IndirectCallType` on both backends (I2; see
`c_function_pointer_signature_mismatch_traps`). The JIT lowers `RefFunc` to an `iconst.i32`.

### Known quirks / inefficiencies (correct, just not optimal — don't "fix" without need)
- **Redundant `memzero`/init for promoted scalars:** chibicc still emits `ND_MEMZERO` then
  the initializer, so `int x = 5;` lowers to a dead `i32.const 0` (the bind) followed by the
  real `5`. For a promoted local these are dead **SSA consts**, not stores, and Cranelift
  DCEs them; for a memory local it's the old store-0-then-store-5. Harmless either way.
- **Over-reserved frames:** every function frame includes chibicc's hidden `__alloca_size__`
  (8 B), and `int main()` (empty parens ⇒ chibicc treats it as variadic) also gets
  `__va_area__` (136 B). Harmless over-reservation.

---

## 3. `codegen_ir.c` map (where to add things)

- `irty(Type*)` → `"i32"`/`"i64"`/float types (LP64: int=i32, long/ptr=i64).
- `gen_load` / `gen_store` — typed memory access by C type (narrow widths included).
- `gen_addr(node)` — lvalue address as i64. Handles `ND_VAR` (local → `sp+offset`),
  `ND_DEREF`, `ND_COMMA`, `ND_MEMBER` (structs).
- `gen_expr(node)` — the big dispatch. Arithmetic/bitwise/shift/compare, `ND_NEG/NOT/BITNOT`,
  `ND_CAST`, `ND_COMMA`, `ND_VAR`, `ND_DEREF`, `ND_ADDR`, `ND_ASSIGN`, `ND_NULL_EXPR`,
  `ND_MEMZERO`, `ND_FUNCALL` (direct), `ND_MEMBER`, etc.
- `gen_if` / `gen_for` (handles both `for` and `while`) — the block CFG.
- `gen_stmt` — `ND_BLOCK` (drops dead code after a terminator), `ND_EXPR_STMT`, `ND_IF`,
  `ND_FOR`, `ND_RETURN`, switch/goto/labels.
- `gen_func` — signature (`func (i64 sp, params...) -> (ret)`), entry block, param spill
  (or curval bind for promoted params), fall-off-end default `return 0`.
- `prepare_func(fn)` — the per-function analysis: `rewrite` (un-desugar compound assign) →
  `scan` (collect address-taken locals) → classify + lay out (promoted slot sentinel vs
  memory offset) + `stack_size`. Run for each func in `codegen_ir` before `gen_func`.
- `open_block`/`open_merge` + `cvals()`/`cparams()` — block headers and branch args that
  carry the data-SP **and the promoted locals** (`MERGE_VAL = npromo+1` is the carried
  result/switch-value slot, after the promoted ones).
- `codegen_ir` — orders funcs (main first, after the synthetic `_start`), runs `prepare_func`,
  emits `memory` + data segments, `emit_start` (by-name powerbox resolution, argv parsing,
  the `--child-entry` ABI), emits funcs, then the `-g` debug-info section.

**chibicc AST facts learned (save you time):**
- `Obj` = function or variable; `Node` = AST node; `Type` (`TypeKind`, `->kind`, `->size`,
  `->is_unsigned`, `->base`, `->return_ty`, `->params`). Enums/structs are in `chibicc.h`.
- A declaration `T x = init;` lowers to `ND_EXPR_STMT(ND_NULL_EXPR)` (a VLA-size no-op)
  **plus** `ND_EXPR_STMT(ND_COMMA(ND_MEMZERO, ND_ASSIGN))`. That's why both no-op nodes
  are handled.
- `fn->params` is in **declaration order** (the recursive `create_param_lvars` + prepend
  cancel out). Offsets come from `fn->locals` (which includes params + hidden locals). Both are
  the same `Obj`s, so offsets assigned via `locals` are seen via `params`.
- A direct call has `node->lhs->kind == ND_VAR` with `node->lhs->var->is_function`;
  `node->args` is the (already param-cast) arg list; `node->func_ty->return_ty` / `node->ty`
  is the return type. Args are pre-cast to param types by the parser.
- Comparison result type is always `int` (i32); the **op width** comes from the operand type
  (`node->lhs->ty`), so e.g. `i64.lt_s` → i32 result.

---

## 4. Conventions & sanity check

**Gate before every commit:** `cargo fmt --all --check`, `cargo clippy --workspace
--all-targets` (no warnings), `cargo test --workspace` (all green). `codegen_ir.c` is C, so
fmt/clippy don't touch it — but `make -C frontend/chibicc` must build warning-clean. General
working agreement is in `AGENTS.md`.

**Sanity check that a fresh checkout works:**
```
make -C frontend/chibicc
printf 'int fib(int n){if(n<2)return n;return fib(n-1)+fib(n-2);} int main(){return fib(10);}\n' > /tmp/t.c
frontend/chibicc/chibicc -cc1 --emit-ir -cc1-input /tmp/t.c -cc1-output /tmp/t.svm /tmp/t.c
cat /tmp/t.svm                        # func 0 = _start, func 1 = main calling func 2 = fib; n promotes to v1
cargo test -p svm --test c_frontend   # interp == JIT, and == cc
cargo test -p svm --test jit_fuzz     # 4000 generated modules, interp == JIT
```
