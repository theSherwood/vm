# Handoff — C frontend (chibicc → SVM IR)

Pick-up notes for continuing the C-frontend work in a fresh session. Written 2026-06-03.
Branch: **`main`** (this session has been committing straight to `main`; the remote is
`theSherwood/vm`). Everything below is committed and CI-green.

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

**Phase 2 ("it works") is underway**: real C compiles → verifies → runs identically on
interpreter and JIT. That C frontend is what this handoff is about.

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

### Test harness
**`crates/svm/tests/c_frontend.rs`** — `make`s the fork once, compiles each C snippet to
IR, **verifies it**, and runs `main` (function 0) on **both the interpreter and the
JIT**, asserting they agree. So every C test doubles as a JIT differential test. Run:
```
cargo test -p svm --test c_frontend
```

### What C is supported today (≈10 tests, all green)
`int` / `long` / `void` functions; integer expressions (constants, `+ - * / %`, bitwise,
shifts, comparisons, unary `- ! ~`, integer casts, comma); **scalar locals**, assignment,
`&` / `*`, pointers to locals; **control flow** `if`/`else`, `while`, `for`; **multiple
functions, parameters, direct calls, and recursion** (incl. mutual recursion). Validated
end-to-end: `fib(10)=55`, `fact(6)=720`, iterative Fibonacci, prime-divisor loops, etc.

Anything unsupported is a **hard `error_tok`** (with the AST node kind), by design — we
never emit IR we can't stand behind. The frontend is outside the escape-TCB (§2a): the
verifier re-checks whatever it emits.

---

## 3. The lowering model (read this before extending `codegen_ir.c`)

**Everything-in-memory, with a threaded data-stack pointer.** This is chibicc's own
"allocate all locals to memory first" model (DESIGN §3d), *without* the SSA-promotion
pass yet (that's the documented "reverse" pass that matters for speed — not done).

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

### Known quirks / inefficiencies (correct, just not optimal — don't "fix" without need)
- **Redundant `memzero`:** chibicc emits `ND_MEMZERO` (zero-init) before every
  initializer, even for a fully-initialized scalar, so `int x = 5;` stores 0 then 5. The
  SSA-promotion / optimization pass (deferred) is where this goes away.
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
- `gen_func` — signature (`func (i64 sp, params...) -> (ret)`), entry block, param spill,
  fall-off-end default `return 0`.
- `codegen_ir` — orders funcs (main first), assigns offsets, emits `memory`, emits funcs.

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

## 5. Roadmap (in suggested order — all incremental, no open design questions)

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
8. **(Perf, later) SSA-promotion pass** — promote non-address-taken, non-`volatile`
   scalars from memory to real SSA values (DESIGN §3d "the pass that matters for speed").
   This also removes the redundant `memzero` and most loads/stores.

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
  (locals/pointers), `ead1bb2` (control flow), `a0c39ad` (functions/recursion).

---

## 7. Sanity check to confirm the pickup works
```
make -C frontend/chibicc
printf 'int fib(int n){if(n<2)return n;return fib(n-1)+fib(n-2);} int main(){return fib(10);}\n' > /tmp/t.c
frontend/chibicc/chibicc -cc1 --emit-ir -cc1-input /tmp/t.c -cc1-output /tmp/t.svm /tmp/t.c
cat /tmp/t.svm            # should show func 0 = main calling func 1 = fib, with sp threading
cargo test -p svm --test c_frontend   # ~10 tests, all green (interp == JIT)
```
If those pass, you're oriented — continue at §5 item 1.
