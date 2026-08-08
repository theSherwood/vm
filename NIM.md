# NIM.md — running Nim (nimony) on SVM, and self-hosting it

Status: **scoping / design doc + phase-1 in progress**, written 2026-07-28. This is the
work-breakdown and the load-bearing decisions for targeting SVM from
[nimony](https://github.com/nim-lang/nimony) — the in-development next-generation Nim
compiler. It leans on the on-ramp (`LLVM.md`), the C-selfhost template (`SELFHOST_C.md`),
libc-as-capabilities (`POSIX.md`), and the frontend trust model (`FRONTEND.md` §1,
`DESIGN.md` §2a). This doc is the *what/how/when*; it does not restate those.

This doc stays its own file — it is **not** folded into `DESIGN.md`. `DESIGN.md` describes
SVM itself; NIM.md describes a *guest* project that runs **on** SVM (nimony, a consumer of the
substrate). The two never merge: svm's design and the design of something built atop it are
separate concerns, even where they touch the same seams.

## 0. TL;DR

- **Goal.** Compile Nim → SVM IR, and eventually **self-host nimony on SVM** — the same
  shape as chibicc-on-SVM (`SELFHOST_C.md`) and the Postgres/QuickJS guest assets.
- **Two phases, cheapest-first.**
  - **Phase 1 (this doc's active work): the C on-ramp path — zero nimony code.**
    nimony already emits C; SVM already ingests C→bitcode→SVM-IR via the proven LLVM
    on-ramp (`LLVM.md`). So `nim → nimony → Leng → lengc(C) → clang -O2 → svm-llvm-translate
    → prep_svmb`. This retires the real risk (does Nim-shaped codegen — ARC, error-flag
    exceptions, raw-pointer objects, the `system` runtime — survive the on-ramp and run
    correctly under confinement?) and delivers **nimony-on-SVM for free**, exactly as
    `chibicc.svmb` came for free from the same pipeline.
  - **Phase 2 (optional, if warranted): a native `Leng → SVM-IR` backend.** A
    `lengc`-style backend written in Nim, consuming Leng NIF and emitting SVM text — the
    supported multi-backend seam (C/C++/LLVM-IR/arkham already coexist behind Leng). The
    **"arkham-for-SVM"** play: drops the clang/LLVM build-time dependency and shapes SVM-IR
    directly from Leng.
- **The one thing that could scare you — Leng assumes flat C-ABI memory with raw
  pointers — is already solved.** It is the C frontend's situation exactly: the guest gets a
  flat `[0, size)` **window**, pointers are window offsets, and the **masking lowering**
  confines every access (INVARIANTS §2 — the security hinge, *not* the verifier). Nim/Leng
  raw pointers are no more dangerous than C's. Leng's `ptr`/`aptr` split is *cleaner* than C.

## 1. Background — nimony's pipeline and where a backend plugs in

(Findings from reading nimony `master`, repo v0.4.0, 2026-07-28. **"NIFC" was renamed
"Leng"**; the tool is `lengc`; the token/format library is "nifcore"; the interchange format
is **NIF** — "Nim Intermediate Format", `nim-lang/nifspec`.)

The compiler is a chain of tools passing **NIF token streams** (not trees) between phases:

```
Nim source
  → nifler      parse → NIF dialect (.p.nif)
  → nimony      sema: symbol/type resolution, template/macro expansion (.s.nif)
                (+ effect inference — NOT yet implemented; + deref/mutation checks)
  → hexer       lowering: iterator inlining, lambda lifting, ARC dup/copy + destructor
                injection, control-flow-expr → stmt, exception translation, CPS
  → hexer       emit  Leng                       ← the stable, documented codegen IR
  → lengc       Leng → C | C++ | LLVM IR         ← the backend seam
```

- **Leng** (`doc/leng-spec.md`, ~450-line grammar) is a **typed, C-like tree/statement IR**
  — *not* SSA, *not* a stack machine. Named locals; structured `if`/`while`/`case` plus
  low-level `lab`/`jmp`. Types: sized `(i N)`/`(u N)`/`(f N)`/`(c N)`, `bool`, `void`,
  `ptr`/`aptr`, value-semantic `array`, `object` (single inheritance = first child is the
  base), `union`, `enum`, bitfields, SIMD `vector`. Ops carry their result type
  (`add`/`sub`/…), `cast` (C cast) vs `conv` (value-preserving). Lvalues: `deref`/`addr`/`at`/
  `pat`/`dot`. Overflow is explicit (`keepovf`/`ovf`, GCC-`__builtin_*_overflow`-shaped).
  `emit` = verbatim C passthrough.
- **The backend seam is real and already multi-consumer.** `src/lengc/` ships C, C++, **and
  LLVM-IR** backends (`codegen.nim`, `llvmcodegen.nim`, dispatched by `lengc {c|cpp|llvm}`);
  a separate **arkham** tool compiles Leng → typed asm-NIF → native (`nifasm`); **shoggoth**
  is a NIF-level optimizer. A new backend consuming Leng is the *supported* extension. This
  is what makes Phase 2 tractable.
- **Runtime shape — favorable:**
  - **ARC/ORC only, no tracing GC.** Destructors/dups are injected by hexer as **ordinary
    Leng calls** → ordinary SVM-IR calls. No GC runtime to port.
  - **libc optional.** Default stdlib is **libc-free** (native allocator + raw-syscall IO);
    `-d:useLibc` opts into mimalloc. Raw syscalls → unresolved named imports the POSIX
    personality resolves at load (`POSIX.md`) — the exact self-host libc model.
  - **Exceptions:** error-flag + `goto` (`errv` bool + `onerr`), **no setjmp/longjmp**, or
    C++ `try/throw` in cpp mode. Fully lowered before Leng.
  - **Concurrency:** CPS / `.passive` procs → state machines over a minimal `system.nim`.
  - **Self-hosting today:** nimony is written in Nim, built by Nim 2.x, and boots through C
    to a **byte-identical stage2==stage3 fixpoint** (`doc/nifcore_migration.md`).

### Compatibility ledger (the "why this works")

| nimony/Leng concern | Lands on SVM as |
|---|---|
| Flat C-ABI memory, raw `ptr`/`aptr`, unions, casts | Window + masking lowering, as for C; §3d pins x86-64-SysV struct layout — matches Leng's ABI |
| ARC/ORC, destructors as calls | Ordinary SVM-IR calls; **no GC runtime** |
| libc-free / raw syscalls / mimalloc | Named imports → POSIX personality; allocator grows the window via the Memory cap |
| Exceptions: error-flag + goto | SVM-IR general goto/branch (C frontend proves gnarly state machines) |
| Bit-reinterpret casts | Already lowered to `copyMem` upstream — not arbitrary bit-punning at Leng |
| CPS/passive → state machines | Ordinary code over minimal `system.nim`; SVM threads (`THREADS.md`) |
| Leng is not SSA | SSA/block-params synthesized from named locals + goto — the on-ramp already does φ→block-args (`LLVM.md`); a native backend redoes this |

### Trust (both phases)

The Nim-derived module is an **untrusted frontend artifact** (`DESIGN.md` §2a): the verifier
re-checks everything at load, so a nimony/backend bug is a **clean error, never an escape**.
No self-hosting convenience may bypass verification (INVARIANTS §9).

## 2. Phase 1 — the C on-ramp path (active)

**Pipeline.** `nim → nimony/hexer → Leng → lengc c → clang-18 -O2 -emit-llvm →
svm-llvm-translate → prep_svmb (decode → verify → bytecode-compile gate)`, then run on
interpreter + JIT. This is the **`build-pg-assets.mjs` / `build_chibicc_svmb.sh` pattern**,
retargeted at nimony's C output. Use the **LLVM on-ramp, not the chibicc C frontend**: Nim's
C leans on compiler builtins (overflow — Leng's `keepovf`/`ovf`) that clang handles and
chibicc does not.

**Build order.**

1. **Retire the codegen-shape risk *before* a nimony bootstrap** — validate that
   **Nim-shaped C** survives the on-ramp and runs identically to native. A small probe
   program exercises the exact patterns nimony/Leng emit: ARC refcount inc/dec + a destructor
   call, an error-flag + goto raise/handler, a tagged `object` with an inheritance-style first
   field, and a heap `seq`/`string`-like struct over `malloc`. Run interp == JIT == native.
   **→ DONE 2026-07-28** — `crates/svm-run/demos/nimony/` (`arc_probe.c` + `build_probe.sh` +
   the `nimony_probe` runner example). The probe translates through the on-ramp to a **21-func,
   9.3 KB `.svmb`** that decodes / verifies / bytecode-compiles, and runs **byte-identical to
   native `clang -O2` on all three engines** (treewalk / bytecode / JIT), exit 0. Every call
   resolved to an on-ramp-recognized name with **no `--stub-externs`** — i.e. ARC destructor
   calls, the error-flag+goto unwind, first-member-base object dispatch, and `realloc`-grown
   seqs all lower and confine cleanly. The codegen-shape risk is retired; the remaining Phase-1
   work is toolchain (step 2) + real libc surface (step 3), not "does Nim's shape fit SVM".
2. **Real Nim codegen on SVM.** **→ PARTIALLY DONE 2026-07-28 via stock Nim as a stand-in.**
   `crates/svm-run/demos/nimony/list_seq.nim` (ARC `ref object` linked list + a `seq`) is
   compiled by the **stock Nim 2.2.10 ARC backend** (`--mm:arc -d:useMalloc -d:noSignalHandler`)
   and on-ramped by `build_nim.sh` — the same `nim → C → clang -O2 → svm-llvm-translate →
   prep_svmb` chain, then run on all three engines. **Result: byte-identical to a native `nim c`
   build (stdout `listSum=385 / seqSum=55`, exit 0) on treewalk / bytecode / JIT.** This is
   genuine Nim-runtime codegen (ARC destructors, heap `ref`, `realloc`-grown `seq`), not the
   hand-modeled `arc_probe.c` — and stock Nim and nimony share the ARC/ORC model + C-ABI shape.
   - **Measured libc surface (the chibicc A.5 stub-audit method): 10 undefined symbols, all
     on-ramp-recognized** — `malloc`/`free`/`realloc`, `fwrite`/`fflush`/`fputc`/`stdout`/`stderr`,
     `exit`, `strlen`. *Far* smaller than chibicc's 41; no libc fill needed for this corpus.
   - **One gotcha found:** Nim installs SIGSEGV/etc. handlers at startup (`signal()` →
     stubbed → `Unreachable` trap); `-d:noSignalHandler` avoids it. Recorded in `build_nim.sh`.
   - **Step 2 proper — genuine nimony output runs on SVM. → DONE 2026-07-28.** Built **Nim 2.3.1
     from source** (stable 2.2.10 can't compile `hastur`), bootstrapped **nimony** (`nim c -r
     src/hastur build all` → `bin/{nimony,hexer,lengc,nifler,…}`), and on-ramped nimony's *own*
     `lengc c` output. `crates/svm-run/demos/nimony/sum_sq_nimony.nim` (an ARC `seq[int]` + a
     `var object`) compiles via the real pipeline (nifler → nimony → hexer → **Leng** → lengc C),
     and `build_nimony.sh` on-ramps that C to a **95-func `.svmb`** that decodes / verifies /
     bytecode-compiles and runs **byte-identical to nimony's native run (`sum_sq=385 / count=10`,
     exit 0) on treewalk / bytecode / JIT.** This is authentic nimony Leng→C→SVM-IR, not a
     stand-in. **Two concrete on-ramp findings** (both normalized in `build_nimony.sh`, both
     recorded as follow-ups for a proper backend):
     - **nimony's runtime allocates via `mmap`, not `malloc`** (libc-free stdlib), plus a handful
       of syscalls (`getpid`/`kill`/`dlopen`/`dlsym`/`_exit`; `write`/`exit` are on-ramp-recognized).
       A 20-line page-aligned bump-allocator shim (`nimony_runtime_shim.c`) over the window covers
       it — the allocator masks addresses to 4096, so `mmap` must return **absolutely** page-aligned
       pointers (the load-bearing subtlety).
     - **TLS gap:** nimony marks the allocator/exception globals `__thread`; the on-ramp has no
       `llvm.threadlocal.address` lowering. For a single-threaded guest these are plain globals
       (stripped in the build). *A real Leng→SVM-IR backend (Phase 2) would map these onto SVM's
       own thread-local/global model instead; the on-ramp TLS gap is worth a `LLVM.md` follow-up.*
3. **On-ramp the real C**, `-mlong-double-64` if nimony emits `long double` (the chibicc F3
   lesson), `--host-page 65536` for a browser-targetable asset. Fill the libc bottom edge by
   **reusing the Postgres/chibicc guest-libc shims** (`SELFHOST_C.md` Appendix B) — the
   surface is nearly identical (stdio, `malloc`/`realloc`, `str*`, `%.17g` via `__vm_fmt_gen`).
4. **Differential-validate** each corpus program: guest stdout/exit byte-matches native
   `nim c` build (the `chibicc_run.rs` / `run_selfhost_diff.sh` two-tier pattern).
5. **nimony-on-SVM (the self-host payoff).** On-ramp nimony's *own* C output into a
   `nimony.svmb` guest — the same way `chibicc.svmb` is built. `nim → nimony.svmb → SVM-IR`
   is then a composition of proven pieces; no new substrate.

**Exit criteria.** (a) the Nim-shaped-C probe runs interp==JIT==native; (b) ≥1 real nimony C
program runs on SVM matching native; (c) the libc fill-list is measured (the `--stub-externs`
+ stub-audit method, `SELFHOST_C.md` A.5).

### Current-sandbox status (2026-07-28)

- ✅ `clang-18` present; ✅ `svm-llvm-translate` built; ✅ **step-1 probe green on all three
  engines** (above).
- ✅ Nim **2.2.10** (choosenim) + **Nim 2.3.1 built from source**; ✅ **nimony bootstrapped**
  (`bin/{nimony,hexer,lengc,…}`); ✅ **genuine nimony Leng→C output runs on SVM, all three
  engines, byte-identical to native nimony** (step 2 above); ✅ stock-Nim ARC program green too.
- The C-on-ramp path (Phase 1) is now **proven end-to-end with the real compiler.** Remaining
  Phase-1 breadth: wider Nim/nimony corpus (strings, exceptions, closures, floats), the `mmap`
  allocator shim → a real Memory-cap allocator, and the TLS follow-up above. Then Phase 2 is the
  open design choice.
- **Reproduce the nimony demo:** build Nim 2.3.1 (`git clone nim-lang/Nim && sh build_all.sh`),
  bootstrap nimony (`nim c -r src/hastur build all`), then
  `NIMONY_BIN=…/nimony/bin/nimony bash crates/svm-run/demos/nimony/build_nimony.sh`.

## 3. Phase 2 — native `Leng → SVM-IR` backend (started)

A translator that consumes **Leng NIF** and emits **SVM IR** directly, bypassing C/clang. It
drops the build-time clang/LLVM dependency and shapes SVM-IR straight from Leng, and it is the
supported extension pattern — C/C++/LLVM-IR/arkham already coexist behind Leng, plus the
shoggoth optimizer.

> **Capstone reached 2026-07-28: whole real modules verify.** `svm-leng` translates **entire**
> real `hexer` modules — every proc plus globals, type decls, and cross-module imports — for three
> real Nim programs (`addTwo`+`main`, `maxi`+`sumto`, `dot2`+`idx`), each **parsing and passing
> `svm-verify`**, and the user `main` (an intra-module call) **runs end-to-end on both engines**
> (`crates/svm-leng/tests/whole_real_module.rs`). Driving a whole module out turned the "what's
> left" list into a measured one — the last gaps that blocked real modules were small: `(true)`/
> `(false)`/`(nil)` literals, `cast`, and coercing a bare-literal `ret` to the proc's result type
> (an i32 `main` returning `0`). The remaining breadth (below) is genuinely optional for coverage,
> not structural.

**Placement decision (2026-07-28): a Rust crate `crates/svm-leng` in *this* repo** — the
**fourth SVM frontend**, beside `svm-wasm` and `svm-llvm` (both Rust, both untrusted, both
verifier-rechecked). Rationale: it matches the established frontend pattern, reuses
`svm-ir`/`svm-text`/`svm-verify` directly, and is **CI-testable with checked-in Leng fixtures,
no nimony toolchain at build time**. (A Nim backend inside nimony's `src/lengc/` — the arkham
analog — is the alternative; it is better for *eventual* pure self-hosting through Leng but
couples to the nimony build and can't live in this repo. Revisit once the Rust translator has
proven the mapping. The two aren't exclusive.) Like every frontend it is **outside the
escape-TCB** (DESIGN.md §2a): the verifier re-checks its output, so a bug is a clean error.

### Walking skeleton — DONE 2026-07-28

`crates/svm-leng` translates the **integer / arithmetic / local / direct-call** subset with
straight-line bodies and `ret`, and **fail-closes (`LengError::Unsupported`) on everything
else** (the `svm-wasm`/`svm-llvm` `unsup(...)` discipline — never a silent mistranslation). It
emits SVM text (chibicc's `codegen_ir.c` model) via `svm_text::parse_module`. Six end-to-end
tests translate hand-written Leng-NIF (faithful to `doc/leng-spec.md`) → verify → **run on both
the interpreter and the JIT with identical results** (§9 parity): constant arithmetic
(`3 + 4*2`), params+locals, `div`/`mod`, i32↔i64 `conv`, cross-proc `call`, and the fail-closed
float case. `src/nif.rs` is a real NIF reader (parens, atoms, `:symdefs`, string literals,
`@lineinfo` stripping, `.nif`/`.indexat` directives) so it grows toward *real* nimony Leng, not
just fixtures.

This proves the seam. The remaining work is adding grammar arms below (each a new match case),
not rearchitecting.

### Real nimony output — DONE 2026-07-28 (go deep)

The skeleton now consumes a **real Leng file emitted by nimony's own `hexer`**, not just
hand-written fixtures. `hexer c --isMain <mod>.s.nif` produces Leng for a module; the verbatim
output for `proc addTwo(a,b: int): int = result = a + b` is checked in at
`tests/fixtures/real_module.leng.nif`, and `translate_proc(real, "addTwo.0.")` translates that
proc out of the full module (which also carries `gvar`/`type`/`main`/`ini` constructs still
outside the subset) and **runs it on both engines** (`addTwo(20,22)=42`, etc.). This drove two
reader corrections against real bytes:
- **Line-info is pervasive and multi-form.** NIF attaches position info to *every* token — tags,
  symbols, *and* numbers — introduced by `@` (`add@4`, `20@7`) **or** `~` (`a.0~2`, `(i~6,~4 64)`).
  The reader strips from the first `@`/`~`; neither can occur in a semantic token (mangled
  symbols encode them away; integer literals use `-`, not `~`).
- **`(stmts …)` may omit the SCOPE marker.** The grammar is `(stmts SCOPE Stmt*)`, but real hexer
  output starts straight with a statement (always a list); the reader skips a *leading atom*
  scope but keeps a leading list.

To go from one proc to the whole module (`main`, `ini`, `` `main ``) needs the broadening arms:
`gvar`/`type`/`const` top-levels, `if`/`while`, cross-module `call` (`…sysvq0asl` suffixes) as
imports, `cast`/pointers. Deep-then-broaden: the real seam works; each construct is now additive.

**The work (bounded — comparable to chibicc's `codegen_ir.c`, which exists and is proven):**

- **✅ integer scalars, arithmetic (`add`/`sub`/`mul`/`div`/`mod`), `neg`, width `conv`,
  locals (`var`/`asgn`), direct `call`, `ret`** — the landed skeleton.
- **✅ control flow — DONE 2026-07-28.** `if`/`elif`/`else`, `while`, `scope`, nested `stmts`,
  and comparisons (`eq`/`neq`/`lt`/`le`), lowered to **multi-block SVM-IR with locals threaded as
  block parameters** (the chibicc/on-ramp φ model — no separate dominance analysis; a merge is
  just the successor's block param). Value numbers reset per block; the entry block carries only
  the function params (the ABI), successors carry every slot. Tested on hand fixtures (max, a
  `while` sum, an `elif` sign chain) **and the real nimony `maxi` if/else** — interp == JIT on all.
  `case`→`br_table` and `lab`/`jmp` (Leng's low-level jump family) remain.
- **Then, further out:**
  - **C-ABI struct/union/enum layout** → SVM §3d (x86-64-SysV already pinned — Leng assumes the
    same ABI, so this is a match, not a negotiation).
  - **Memory:** Leng `ptr`/`aptr`/`at`/`pat`/`dot`/`deref`/`addr` → window loads/stores +
    `ptr.add`; every access confined by the masking lowering (INVARIANTS §2).
    - **✅ pointer params + `deref`/`store` — DONE 2026-07-28.** Pointer-typed params/vars are
      `i64` window offsets; `(deref p)` loads and `(asgn (deref p) v)`/`(store v p)` store, at the
      pointee width tracked per pointer local. The module declares a `memory` window only when a
      load/store is actually emitted. Tested store→load round-trips on both engines. (No frame
      yet — the pointer is supplied by the caller as an offset.)
    - **✅ address-of-local + the data-stack frame — DONE 2026-07-28.** `(addr x)` demotes a local
      from an SSA slot to a byte offset in a per-call window frame; the proc gains a leading `$sp`
      stack-pointer param (slot 0), reads/writes the local via `load`/`store` at `sp+off`, and a
      call to a frame-needing proc passes `sp + frame_size` as the callee's frame. SSA and frame
      locals coexist (only address-taken ones are framed). Tested with the real nimony loop shape
      `inc(addr i)` (a frameless pointer helper called from a frame-needing counter) and a mixed
      SSA-accumulator/framed-counter sum, interp == JIT. Address-taken *params* and recursion
      depth beyond one frame are the remaining refinements.
    - **✅ `at`/`dot`/`pat` + type layouts — DONE 2026-07-28.** Named `(type … (object …))` /
      `(array Elem Count)` layouts are registered (with forward-ref resolution); a unified
      `lvalue_addr` (the `codegen_ir.c` `gen_addr`) resolves `dot` (field), `at` (array element),
      `pat` (pointer index), `deref`, and frame/aggregate symbols to `(address, type descriptor)`,
      then a scalar leaf loads/stores. Object params are passed by address; aggregate `var`s are
      frame-resident (default-zeroed). Tested on hand fixtures (object field set/get, array `at`,
      pointer `pat`, a framed local array) **and real nimony object bytes** (`dot2`, `p.x*p.x+…`),
      interp == JIT. Whole-aggregate copy/`oconstr`/`aconstr` and C-ABI (SysV) field offsets remain.
    - **✅ whole-module: globals + multi-proc — DONE 2026-07-28.** `gvar`/`tvar` module globals live
      at fixed window offsets (below the caller-passed stack) and are shared across calls; scalar
      `const`s inline; `gvar`/`const`/`type` top-levels are accepted, and a module's procs are
      emitted together so **intra-module calls** resolve by index. Tested end-to-end: a global
      counter + `const` step + a `main → bumpN → bump` call chain, interp == JIT. Non-zero global
      initializers fail-close (a `data`-segment init is the refinement).
    - **✅ cross-module `call` → SVM imports — DONE 2026-07-28.** A call to a callee not defined in
      the module becomes a declared `import N "name" (params) -> (ret)` + `call.import N`; the
      signature is fixed from the call site (param types from the args; return arity from position —
      a stmt-call is void, an expr-call returns a value), cached per symbol (inconsistent arity
      fail-closes). The runtime binds the import by name at instantiation, exactly like `write`.
      Tested: a cross-module call translates + verifies, **runs correctly on the interpreter with a
      bound host fn** (`use_ext(x)=ext_double(x)+1`), a stmt-call declares a void import, and — the
      payoff — **real nimony `sumto` now translates and verifies**: `while i<=n: (inc(addr i);
      result+=i)` composes the frame (address-taken counter), `while`, and the cross-module `inc`
      import all at once. Signature inference is call-site-based (not the `.idx` export map); wiring
      the real export sigs (and JIT-side import binding in tests) are refinements.

  **State:** `svm-leng` translates whole real-ish modules — integers, floats, control flow (incl.
  `break`/`continue` and `block` via `jmp`/`lab`), pointers, frames, objects/arrays (incl.
  constructors, copy, and **sret return**), **object-of-`RootObj` inheritance** (base-inlining +
  vtable header), enum/distinct scalars, **exceptions** (nimony's error-flag ABI), **seq/string**
  value layout + operations (as runtime imports), globals, intra- and cross-module calls —
  fail-closed on the rest, and is validated against genuine `hexer` bytes (`addTwo`, `maxi`, `dot2`,
  `sumto`, `classify`, `favg`, `mkSum`, `mk`, `firstHit`, `labeled`, `toNum`, `mayFail`, `guarded`,
  `counter`, `getAt`, `sumSeq`, `makeSeq`, `kindOf`, `mkDerived`). **W1 (Leng totality) is
  essentially closed** — what's left is genuinely runtime, not translation: dynamic method dispatch
  and value-object exception payloads fail-close cleanly (both need the vtable/`exc`-threadvar
  runtime), and the `jtrue`/`mflag`/`vflag` cfvar forms never reach us (hexer's `xelim` lowers them
  away before the final IR). The remaining lever is **W3** — binding the seq/string (and other
  stdlib) imports to a real runtime so the lowered code *runs*, not just verifies.
    - **✅ whole-aggregate copy + `oconstr`/`aconstr` — DONE 2026-07-28.** An aggregate destination
      (frame var, `deref`/`dot`/`at`, global) is dispatched by a non-emitting `lvalue_type` walk:
      `(oconstr T (kv F E)*)` and `(aconstr T E*)` construct field/element-by-element in place (with
      nested aggregates recursing), and any other rhs is a whole-aggregate `mem.copy` of the
      source's bytes. Aggregate `var`s initialize the same way. Tested: object construct-and-read,
      an array `aconstr`, a struct copy (`mem.copy`), and **real nimony `mkSum`** (`var p = Pt(x:a,
      y:b); p.x+p.y`), interp == JIT.
    - **✅ object-of-`RootObj` inheritance — DONE 2026-07-29.** An inheritable object carries a
      leading vtable/type-header pointer (the positional slot an `(oconstr T <vtable> …)` fills), then
      the base's fields, then its own — `resolve_type` inlines a local base's layout at the front, and
      an external inheritable root (`RootObj`) contributes a single 8-byte header. A `Type.vt` (`Rtti`)
      const gets a zeroed, addressable placeholder global so `(addr Type.vt)` resolves; the stored
      vtable pointer is opaque (only *dynamic dispatch* reads through it, and that fail-closes).
      Tested on hand fixtures (construct-and-read-back a `Derived` — base field before derived field,
      both past the header — and a base-field read through a pointer, both engines) and **real nimony
      `kindOf`** (reads `e.value` through a `ptr BaseError`, *runs*) + **`mkDerived`** (constructs the
      inherited object with its vtable — translates + verifies; running needs the ARC destructor
      imports, W3). Value-object exception payloads (an object punned into the error tuple's scalar
      `ErrorCode` slot) stay fail-closed.
    - **✅ seq/string (value layout + operations as imports) — DONE 2026-07-29.** nimony's `seq[T]`
      is a `{len, data*}` fat-pointer **object** (`string` analogous), so its value layout and element
      access already ride the object + pointer machinery — a hand-written seq summed over a
      caller-provided buffer *runs* on both engines. Its *operations* (`add`/`[]`/`len`/`toOpenArray`/
      `newSeq`) are stdlib procs that lower to **imports** (the **W3** runtime edge: they verify, and
      run once bound). Getting real seq bytes to lower needed four fixes: (1) import **names** escaped
      for svm-text (the `[]` operator mangles to `\5B\5D…`, whose bare backslash the lexer rejected);
      (2) aggregate **args** to imports passed by address; (3) aggregate-**returning** imports (sret
      imports, e.g. `toOpenArray`/`newSeqUninit`); (4) structured `break`/`continue` (a `for` lowers
      to `while (true) { … else break }`). Real nimony `getAt`/`firstLen` (index/len) and
      `sumSeq`/`makeSeq` (the full `for`-read and `add`-write paths) now **translate and verify**.
    - **✅ non-zero global initializers — DONE 2026-07-29.** A `gvar` with a non-zero scalar-int
      initializer becomes a module `data` segment (little-endian bytes at the global's window offset)
      — the window is otherwise zero, so a zero initializer stays a no-op, and a non-scalar/aggregate
      initializer fail-closes. Tested on hand fixtures (i32 + i64) and **real nimony `var counter:
      int = 42`** with `getCounter`/`addCounter`: the data segment seeds the window so `getCounter()`
      reads 42, interp == JIT.
    - **✅ enum/distinct scalars — DONE 2026-07-29.** A named type is an aggregate only when it's a
      locally-declared `(object …)`/`(array …)`; every other named type — an `(enum …)`, a `distinct`
      int, a `proctype`, or a type external to the module — is an integer scalar (its values are
      plain integers). `collect_types` records the aggregate names up front, so `tydesc` classifies
      as it resolves. Also hardened `if`-condition truthiness: a wide (`i64`) condition — e.g. an
      enum error code — reduces via `!= 0`, not an `i64→i32` wrap that would drop the high word.
      Tested on hand fixtures and **real nimony `toNum`** (enum compares) + **`roundtrip`** (enum
      passthrough, a scalar return, not sret).
    - **✅ exceptions (error-flag ABI) — DONE 2026-07-29.** nimony lowers exceptions with *no new
      node type*: a `.raises` proc returns an `(object (fld :fld.0 ErrorCode) (fld :fld.1 result))`
      tuple by **sret** (`fld.0` the error code — an enum, hence scalar — `fld.1` the real result);
      `raise E` is `ret (oconstr tuple (kv fld.0 <nonzero>) (kv fld.1 <default>))`; the normal return
      sets `fld.0 = 0`; and `try/except` is `var canRaise = call; if canRaise.fld.0: jmp exlab;
      result = canRaise.fld.1` with the handler under an `if (false) { lab exlab; … }` guard reached
      only via the `jmp`. So it falls straight out of sret + objects + `if` + `jmp`/`lab` +
      enum-scalar error codes — no translator change beyond the enum slice. Tested on a hand-written
      model and **real nimony `mayFail`/`guarded`** (a `distinct`-int raiser + its `try/except`
      caller), interp == JIT: the happy path doubles the input, the error path returns the handler's
      -1. Exception payloads carrying `object`-of-`RootObj` inheritance (vtables) stay fail-closed.
    - **✅ general `goto` (`jmp`/`lab`) — DONE 2026-07-28.** hexer keeps `if`/`while` structured and
      emits the low-level jump family only for `break`/`block`-`break`: `(jmp L)` an unconditional
      branch, `(lab :L)` a label. Both fall straight out of the block-parameter (slot-threading)
      model — labels are pre-scanned and each assigned a block id, a `(jmp L)` is a `br` to that
      block passing the live slot set, and a `(lab :L)` opens it (fall-through if the prior block is
      live, else reached only by jumps). Dead statements after a `jmp` are skipped until the next
      `lab` reopens a reachable block; forward and backward edges both work. Tested on hand fixtures
      and **real nimony `firstHit`** (`while`+`break`) and **`labeled`** (`block done:`/`break done`
      out of a nested loop), interp == JIT. The `jtrue`/`mflag`/`vflag` conditional-jump forms (not
      emitted by hexer's default lowering) stay fail-closed.
    - **✅ aggregate return (sret) — DONE 2026-07-28.** A proc whose return type is a named aggregate
      returns `void` and takes a hidden `$sret` pointer param (after `$sp`, before the Leng params);
      `(ret aggval)` constructs/copies the result into that pointer (composing with `oconstr`/copy).
      A caller assigning the call to an aggregate destination (`var`/`asgn`/`ret`) hands that
      destination's address down as `$sret` — the callee writes in place, no temporary; a scalar or
      discarded use of an aggregate-returning call fail-closes. Aggregate call *arguments* pass by
      address to match by-address params. Tested (both engines, incl. the window bytes the callee
      wrote): a direct sret build, a caller→callee round-trip, return-by-copy, and **real nimony
      `mk`/`mkSum`** — the genuine `var result; result = Pt(…); ret result` + `var p = mk(a,b)` bytes,
      lifted out together via the new multi-proc `translate_procs`.
    - **✅ floats — DONE 2026-07-28.** `(f 32)`/`(f 64)` types; float arithmetic
      (`fN.add/sub/mul/div`), `neg` (`fN.neg`), and comparisons (`fN.lt/le/eq/ne`); int↔float and
      f32↔f64 `conv`/`cast` (`convert_iN_s`/`trunc_fN_s`/`promote`/`demote`); float literals
      (`2.0`, `1e3`) and `(inf)`/`(neginf)`/`(nan)`; float loads/stores follow from the scalar
      type. Tested on hand fixtures and **real nimony `favg`** (`(a+b)/2.0`) + `toF` (`float(n)*1.5`),
      interp == JIT (bit-exact).
    - **✅ `case` → `br_table` — DONE 2026-07-28.** A dense-integer `case` (`(case Disc (of
      (ranges V+) Body)* (else Body)?)`) lowers to a normalized `br_table`: the discriminant is
      offset to the value span's minimum, a table entry per value maps to its covering branch, and
      an out-of-range index (negative or over-large) selects the `else`/continuation via the table
      default. Single values, multi-value `of`s, and `(range lo hi)` are handled; sparse/huge spans
      (>256) fail-close (a comparison-chain lowering is the refinement). Tested on hand fixtures and
      **real nimony `classify`** (`0 / 1,2 / 3 / else`), interp == JIT.
  - **Calls + ARC:** indirect calls; destructor/dup calls pass through as ordinary calls;
    `onerr`/`errv` → branch-on-flag.
  - **Overflow:** `keepovf`/`ovf` → SVM's trapping/checked arithmetic.
  - **Runtime bottom edge:** raw syscalls / allocator → POSIX personality named imports +
    Memory cap, same as Phase 1 — and mapping nimony's TLS onto SVM's own model (the on-ramp
    gap found in Phase 1).

**Risks:** nimony is v0.4.0, "heavy development" — the Leng grammar and C output are moving
targets (Phase 1 is insulated: it only needs "nimony emits compilable C"; Phase 2 couples to
the grammar). Effect inference (pipeline phase 3) and parts of CPS are not yet implemented
*in nimony itself* — a limit of the source compiler, not the backend.

## 3a. Self-hosting roadmap — Path 2a (no C compiler)

The end state we're building toward: **nimony compiles itself on SVM with no C compiler in the
loop.** nimony is written in Nim, and `svm-leng` is its Leng→svm-ir backend. So the loop closes
when both the nimony compiler *and* `svm-leng` itself run as svm modules. Two sub-questions —
"can we translate the Leng nimony emits?" and "can the translator itself run on svm?" — and Path
2a answers the second by bootstrapping the Rust `svm-leng` onto svm the same way any Rust program
reaches svm: **Rust → wasm → the svm-wasm on-ramp → svm-ir.** No C compiler anywhere.

**Why not 2b (a Nim backend inside nimony's `lengc`, an arkham analog).** It would let nimony
emit svm-ir directly, no separate translator. Ruled out: we have **no influence over the nimony
repo**, so a backend living upstream is not a lever we control. `svm-leng` as an *external* Rust
translator keeps the whole path in this tree.

**The mapping is largely proven** (§3 above): integers, floats, control flow, pointers, frames,
objects/arrays + constructors + copy, globals, intra-/cross-module calls — all validated against
genuine `hexer` bytes, interp == JIT. The remaining work is not "can it be done" but breadth +
plumbing. Five workstreams, roughly independent:

- **W1 — Leng totality.** Close the Leng subset so *every* construct a real nimony program emits
  translates or fail-closes cleanly. Load-bearing next slices: **sret (aggregate return)**, then
  general **`goto`** (the low-level `jmp`/`lab`/`jtrue`/`mflag`/`vflag` jump family), then
  **exceptions** (`try`/`onerr`/`raise` as an error-flag model), then **seq/string** (nimony's
  built-in containers). Non-zero global/data initializers land here too.
- **W2 — Linker (the long pole).** A real program is many modules; nimony emits one Leng file per
  module. W2 resolves cross-module symbols, merges globals/data, and lays out one svm module from N
  Leng inputs — the analog of what the C on-ramp gets from `clang`+`lld` for free. **✅ Core done
  (2026-07-29): `svm_leng::link_units` links real cross-module nimony code.** nimony references a
  proc `P.` defined in module `stem` from elsewhere as `P.<stem>`; `link_units` translates each
  module's procs, exports them under those global names, and resolves every unit's cross-module
  calls (named imports) against the exports via `svm_ir::link` — one merged, re-verified, import-free
  module. Proven on a genuine 2-module program (`moda` importing `pkg/modb`: `useit(5)` calls
  `modb`'s compiled `helper` → 16) and a transitive A→B→C chain, both engines (`tests/link.rs`).
  This is the same link mechanism the end-to-end shim used, now across *real translated modules* —
  the shim's stand-in replaced by compiled Nim.
  - **✅ Relocatable globals — DONE 2026-07-30 (aligned to the new `.svmo` object dialect).** The
    tree landed a binary link-unit format (`.svmo`) + `svm-run --link`, with first-class export
    tables and inline data-relocation instructions — `data.self <off>` (this unit's data),
    `data.sym "<name>" <addend>` (a cross-unit data symbol), `data.top` (top-of-data / heap start) —
    that `link` resolves to concrete addresses. This filled the data-symbol gap. `svm-leng` is now
    **dual-mode**: `translate`/`translate_procs` emit a directly *runnable* module (globals at fixed
    absolute offsets, `.svmb` shape), while `link_units` emits a *link unit* (globals addressed via
    `data.self`, `.svmo` shape) so `link` relocates each unit's data into disjoint window regions.
    This was load-bearing: an absolute-offset unit *silently aliased* under linking — two modules'
    globals both at offset 16, one clobbering the other — a fail-closed violation the `data.self`
    lowering fixes (regression test: two modules each read their *own* global, `tests/link.rs`).
  - **✅ Through the `.svmo` narrow waist — DONE 2026-07-30.** `svm-leng` now emits real binary link
    **objects**: `svm_leng::compile_object(unit)` → a `.svmo` with the unit's procs exported *in-band*
    (`Module::exports`, stem-suffixed) — the counterpart of `svm-llvm-translate -o out.svmo`.
    `link_units` routes through the format: compile each module to `.svmo`, `decode_unit` it back
    through the hardened firewall (a frontend is untrusted), pair a `LinkUnit` from its in-band export
    tables (the same conversion `svm-run --link` does), then `svm_ir::link`. The linker stays shared;
    the format is the only added seam. Proven cross-producer (`tests/object.rs`): a nimony `.svmo`
    (`sumSeq`) links against a **separately produced runtime `.svmo`** — both binary objects, joined
    only through the format — and runs (Σ = 60, both engines). That composition is the point: the
    runtime object is the stand-in the real compiled `system` module (or a C-runtime `.svmo`) will
    replace, and now they meet at a versioned, spec-pinned, fuzzed boundary rather than in-process.
  - **✅ Cross-module data symbols — DONE 2026-07-30.** A `gvar` referenced across the module
    boundary — hexer emits it as `counter.0.<defining-stem>`, in lvalue/rvalue position — now links.
    The *defining* unit exports each of its globals as a `data_export` (stem-suffixed name → data
    offset, via `Translator::global_exports`); the *referencing* unit, finding an atom that's not a
    local/own-global/const/literal, emits a relocatable `data.sym "<name>"` the linker binds to that
    export (an unresolved name is a fail-closed link error, never a wrong address). External data
    symbols are assumed `i64` scalars — the common `int`/pointer global. Tested: a hand-written
    writer/reader pair sharing a global defined in a third data-only unit, real nimony
    `bump`/`store` (`bump()` increments `store.counter` across the boundary → 1), and two
    fail-closed cases (`tests/link.rs`).
  - **✅ Short string literals (SSO) — DONE 2026-07-30.** nimony's `string` is a small-string
    object `{bytes: u64@0, more: ptr@8}` (confirmed from the system module's `basic_types.nim`): a
    *short* literal packs its chars into the inline `bytes` word with a nil `more`, so it's an
    ordinary `(oconstr string (kv bytes.0 <packed-u64>) (kv more.0 (nil)))` — no data segment.
    Lowering it took two things: unsigned literals (`122511465736197u`) now parse (bit-pattern
    preserved, `u64`-wide), and the external `string` type's layout must be available. That layout
    normally comes from the `system` module across the link (cross-module *type* resolution — the
    third symbol kind after funcs and data; Path A); until that's automatic, `translate_proc_with_types`
    supplies it as a type prelude. Tested: a hand-written SSO construct-and-read (*runs*), an
    over-`i64::MAX` unsigned literal, and **real nimony `greet(): string = "hello"`** (genuine SSO
    `oconstr` by sret — translates + verifies given the real `string` def; running needs the ARC
    `=wasMoved`/`=destroy` imports, W3).
  - **✅ Automatic cross-module type resolution — DONE 2026-07-30.** The third symbol kind after
    funcs and data. Proc/data symbols resolve at *link* time, but an aggregate **type**'s layout is
    needed at *translate* time (field offsets are baked into loads/stores) — so `link_units` first
    pools every unit's `(type …)` defs under their stem-suffixed global names
    (`Translator::export_types`, rewriting nested same-module field types to their suffixed forms
    too) and pre-registers the pool in each unit's translator (`import_types`) before translating
    any. A module constructing a `string.0.sysvq0asl` now gets the system module's layout
    automatically; `translate_proc_with_types` remains as the manual escape hatch for
    single-module entry points. A *standalone* `compile_object` of a unit with an external value
    type still fail-closes (the layout only exists across the link) — if that ever needs to work
    without siblings, types would ride in-band in `.svmo`, a format question for later. Tested:
    hand-written flat + *nested* external value types (both run, both engines), the standalone
    fail-closed case, and **real nimony `greet(): string = "hello"` running end-to-end** — linked
    against a stand-in system unit under the real stem, no prelude, the packed SSO word and nil
    `more` land in the sret slot identically on both engines (`tests/link.rs`).
  - **✅ Cross-module funcref globals — DONE 2026-07-31.** The fourth cross-module symbol kind. A
    stdlib `gvar` whose type is a `proctype` (the allocator's `oomHandler`, `gExitFlush`,
    `scheduler`) is a function-*pointer* **data** symbol, not a proc — so a sibling module's
    `(call oomHandler.0.<sys> …)` must lower to a `data.sym` load of the `i32` funcref +
    `call_indirect`, not a proc import (which fail-closes at link as `Unresolved`, since the owning
    module exports the name as *data*). Like aggregate **types**, the `call_indirect` signature is
    needed at *translate* time, so `link_selected` now also pools every unit's funcref gvars
    (`Translator::export_funcrefs`, name+stem → `FnPtrSig`) and pre-registers them
    (`import_funcrefs`); `lvalue_type`/`lvalue_addr` then treat a pooled name as an `FnPtr` global
    (address via `data.sym`), and the existing `indirect_callee` path lowers the call unchanged. With
    this, a real allocating program links with **zero unresolved imports** (`oomHandler` binds). The
    funcref value itself stays zero-until-`ini` (a symbol-initialized gvar; see the `gExitFlush`
    note) — this slice is purely the cross-module *call-site* lowering. Tested: two hand-written
    modules where `w` calls through a `proctype` gvar defined in `s` (setter stores `ref.func`, then
    dispatch), both engines (`tests/link.rs`).
  - **✅ Cross-module frame-needing calls — DONE 2026-07-31.** A proc that takes a local's address
    gets a leading `$sp` param and its callers pass `sp + frame_size`; within a module the translator
    knows each callee's frame need, but a **cross-module** call (`call alloc.1.<sys>`) lowers through
    `call_import`, which derives the import's signature from the *args* — blind to the hidden `$sp`
    (verify: `CallArgCountMismatch`, expected 2 found 1). Because frame-need **propagates
    transitively across module boundaries** (`program → newSeqUninit → alloc → alloc.0.`), the linker
    now runs a **whole-program frame fixpoint**: `Translator::proc_frame_nodes` yields each proc's
    `(global_name, own_frame_need, global_callees)`, `link_selected` seeds the set from the
    self-framing procs and closes it under "calls a framed proc," and the final set is pre-registered
    in every unit (`import_proc_frames`). `call_import` then prepends `sp + frame_size` for a pooled
    callee, and `propagate_frames`/`body_calls_framed` became ext-aware so a proc framed *only* by a
    cross-module callee still gains its own `$sp` — the two agree because they are the same fixpoint
    over the same edges. With this the real allocating program **links and verifies with zero
    imports**. Tested: `w`'s `drive` calls `s`'s frame-needing `count` (which takes `(addr i)`), and
    the fixpoint frames `drive` too so it hands `$sp` down — both engines (`tests/link.rs`).
  - **✅ Whole-module link objects — DONE 2026-07-30 (Path A shape).** `link_units` lifts a
    *hand-picked subset* of a module's procs (the "go deep" mode); a real program instead links
    *whole compiled modules* — every proc plus the scaffolding nimony emits (`ini`
    module-initializer, C `main`, exportc gvars). `svm_leng::compile_whole_object`/`link_whole_units`
    (over a `WholeModule{stem, src}`) do that: `Translator::module_with_names` translates every proc
    and returns their local names, which become the object's in-band export table (each proc under
    its global stem-suffixed name), so other modules' cross-module calls resolve. This is `lld` over
    object files versus the selective lift. Proven on **two genuine `hexer` modules linked in full**
    (`tests/whole.rs`): real `moda` (main module — `useit`, its `ini`, the C `main` + exportc gvars)
    + real `modb` (`helper` + `ini`), their cross-module edges resolving against each other
    (`useit`→`helper`, `moda.ini`→`modb.ini`) and their `sysvq0asl` system edges against a small
    stand-in system object. The whole program links, verifies, and runs on both engines — including
    moda's **real C `main` executing the whole init chain** end-to-end (guards, sub-inits, flush) and
    returning 0. The stand-in system object is the last stub between here and the real compiled
    `system` module.
  - **✅ Long string literals (`LongString` data) — DONE 2026-07-30.** A string too long to pack
    into the SSO `bytes` word: nimony emits it as a `const` `LongString` blob
    `{fullLen, rc, capImpl, data: uarray char}` and a `string` value whose `more` field points at it
    (`(addr strlit…)`). Lowering it took **constant-aggregate materialization**: a `const` whose
    value is an `(oconstr T (kv field val)*)` now materializes its exact little-endian bytes into a
    data segment (scalar-int fields at their offset; a **string-literal** field — the `uarray` /
    `UncheckedArray` flexible tail — as raw bytes past the fixed size) and registers the const as an
    addressable global, so `(addr strlit…)` resolves to it. `resolve_type` gained the `uarray`
    flexible-array tail (size-0, the object's fixed size stops there). The `LongString` layout comes
    across the link (cross-module type resolution). Tested: a self-contained `const` blob whose bytes
    are read back from the window (both engines), and **real nimony
    `greetLong(): string = "hello, this is a long string!"` running end-to-end** — linked against a
    stand-in `system` unit (real `string`/`LongString` defs + no-op ARC stubs), the returned
    `string`'s `more` points at the materialized blob (`fullLen` = 29, data = the literal),
    identically on both engines (`tests/strings.rs`).
  - **✅ `exportc` C-name exports — DONE 2026-07-30.** A whole-module object now exposes its
    `exportc` symbols under their **C** names (the C `main`, and the `cmdCount`/`cmdLine`/`nimEnviron`
    gvars), alongside the mangled Leng names — `Translator::exportc_exports` scans proc/gvar
    `(pragmas (exportc "cname") …)` and adds a func/data export under `cname`. So the program's C-ABI
    surface is findable: a host or `svm-run --link` can enter at `main`. Tested on real `moda`'s whole
    object (`tests/object.rs`).
  - **◑ Real `system` module — STARTED 2026-07-30.** The `sysvq0asl` edges now bind to code
    translated from the **actual nimony `system` module**, not a hand-written stub — first real
    stdlib code running on SVM (`tests/system.rs`): a driver's cross-module call binds to the real
    `=wasMoved` (string's ARC "moved-from" reset, `s.bytes = 0` through a `ptr string`, verbatim
    from `system/stringimpl.nim`), and it runs correctly on both engines. Getting there closed two
    translator gaps the real module surfaces: **gvar symbol/proc-pointer initializers** (e.g.
    `gExitFlush = nimNoopFlush` — the pointer value is opaque, an indirect call through it
    fail-closes, so the slot is reserved zero-initialized and the runtime's `ini` writes it) and
    **`suf` suffixed literals** (`255'i64`). With those, the *whole* system module's globals, 96
    types, and 163 `strlit` (`LongString`) consts all collect, and `=wasMoved`,
    `nimFlushStdStreams`, `nimNoopFlush`, `setExitFlush` translate.
  - **✅ Coverage grind — 290/324 system procs translate (2026-07-30).** The ARC set
    (`=wasMoved`/`=destroy`/`=copy`) plus ~180 more procs now lower, after a run of bounded translator
    slices: bitwise/shift/`not`/`bitnot`, char literals, computed-pointer `deref`/`pat`, `importc`
    externs → imports, bool `case` values, sub-word (`i8`/`i16`) locals, local scalar+aggregate
    consts, inline flexible arrays (`LongString.data`), const array tables, aggregate rvalues in
    argument position, `keepovf`/`ovf`, computed `deref`/`pat` in `lvalue_type` (the `s.more.data[i]`
    walk), transitive frame propagation, and scalar parameter spill. Whole-module translation of the
    entire `system` module now runs to a single tail (`[]=`'s call arity).
  - **✅ End-to-end against real `system` ARC — MET 2026-07-30 (Path A payoff).** A *real* nimony
    program runs against *real* compiled `system` code, no stub: `greetLong(): string = "hello, this
    is a long string!"` links against the **real** `=wasMoved`/`=destroy` (verbatim from
    `system/stringimpl.nim`) and runs end-to-end on both engines, returning the literal. The long
    string's `LongString` blob is a `const` in greetLong's data; the linker **relocates** the string's
    `more` pointer to its placed address. This needed **const-to-const data relocations**
    (`data.ptr <at> self <off>`, svm_ir D-LINK): a pointer stored *inside* one const's bytes to
    another const (a `string` literal's `more = (addr strlit)`) — placeholder bytes the linker
    overwrites (`tests/system.rs`, `tests/strings.rs`).
  - **✅ Whole `system` module compiles — MET 2026-07-30 (Path A capstone).** `compile_whole_object`
    lowers the **entire** real `system` module to a 129 KB `.svmo` link object — **297 functions**
    plus exactly the **25 bottom-edge C imports** (`mmap`, `c_memcpy`/`memset`/`memcmp`, the atomics,
    `bswap64`/`ctz64`/`clz64`, `cWriteErr`, `cExitSys`, `dlopen`/`dlsym`/`dlclose`). Getting the last
    procs to lower closed a run of translator gaps, each with a synthetic both-engines test
    (`tests/indirect.rs`): the **char-literal lexer** (a space char `' '` no longer splits, fixing
    the phantom `[]=` arity), **named pointer-alias types** (`ref` types like `RootRef = (ptr …)`),
    **indirect calls through function pointers** (`ref.func` + `call_indirect`, scalar and sret —
    coroutine scheduling: `trivialTick`/`advance`/`setScheduler`), **RTTI virtual dispatch** (the
    `o.vt.mt[i]` method-table walk, via cast-deref static typing + `i64` slot → `i32` funcref), and
    `baseobj` base-subobject upcasts. Linking the whole object end-to-end now needs only the 25
    imports bound (W3, below); the object itself is produced and structurally sound.
  - **✅ Real Nim source runs on SVM, toolchain-driven — MET 2026-07-31 (W3, the Path A payoff).**
    The 25 bottom-edge C imports bind to a **20-function SVM runtime shim** (a pure-IR link unit,
    `tests/fixtures/system_runtime.svm.txt`): `mmap`→a bump allocator (cursor at window offset 8),
    `c_memcpy`→`mem.copy`, `c_memset`→`mem.fill`, `c_memcmp`→a byte-compare loop, the GCC atomics→
    plain memory ops (the in-process model is single-threaded), `ctz64`/`clz64`→`i64.ctz`/`clz`,
    `bswap64`→a shift/or chain, `cWriteErr`→success, `munmap`/`dl*`/`_exit`→inert. `svm-leng` gained
    `link_whole_with_runtime` (program + `system` + runtime units in one link).
    The test is **end-to-end from Nim source, not a committed artifact** (`tests/nim_e2e.rs`): it
    drives the real toolchain (`nimony c` → nifler → nimony → hexer) on small `.nim` programs, links
    the emitted Leng with the shim into one verified import-free module, and runs on **both engines**
    (§9 parity) — `addTwo`, a `while`/`if` routine (`sumTo`/`maxOf`), and the real allocator
    (`osAllocPages`, page-advancing). Because the `.x.nif` is regenerated each run, the test can't rot
    against a frozen snapshot. The toolchain is built in a dedicated CI job
    (`scripts/ci/provision-nimony.sh`); when it's absent the tests **skip** (the translator's own
    logic stays covered by the fast, toolchain-free unit tests).
  - **Remaining toward a full program:** heap programs (`seq`/`string`) now **link with zero
    unresolved imports** (the cross-module funcref-gvar slice above bound `oomHandler`), and the
    generic instantiated `seq[T]` `oconstr` was a red herring — the type *is* registered as an
    aggregate; the earlier `Unsupported("expression oconstr")` was a harness bug (import discovery
    compiled a *program* module standalone, which can't resolve a cross-module aggregate type like
    `string.0.<sys>` — now discovered from the self-contained `system` module only). A real
    `@[]`/`add` program now **links, verifies, and runs end-to-end through the full C-`main`/init
    chain** (returns 0) — the cross-module funcref-gvar, frame-fixpoint, and funcref-initializer
    slices closed every gap between "compiles" and "runs."
  - **✅ Funcref-gvar static initializers (`data.funcref`) — DONE 2026-07-31.** `var oomHandler =
    continueAfterOutOfMem`, `var gExitFlush = nimNoopFlush` are funcref gvars whose value is a
    **static initializer** (a proc pointer). We used to zero-reserve the slot on the assumption the
    system `ini` writes it — but `ini` is minimal (it only resets an exception global); the
    initializer *is* the value, and nothing wrote it, so the first `call_indirect` through the gvar
    trapped (`IndirectCallType` through func 0 — surfacing deep in `cAbort`'s flush). Fixed with a
    **`data.funcref` relocation** (the funcref twin of `data.ptr`, svm_ir D-LINK): `svm-leng`'s
    `collect_globals` records a proctype gvar's proc initializer, `translate_object_module` emits a
    `DataFuncref{at, name}` under the initializer's stem-suffixed name, and `link` resolves it to the
    merged funcidx and writes it (4-byte `i32`, the value `ref.func` yields) into the gvar's data
    slot — then clears the list (a survivor is a fail-closed `UnlinkedDataFuncref` verify error, the
    twin of `UnlinkedDataPtr`). The real `@[]`/`add` program now runs to completion with **no
    hand-patching**. Tested: a hand-written funcref gvar with a static `= dbl` initializer, called
    cross-module with *no runtime setter* — the materialized slot dispatches to `dbl`, both engines
    (`tests/link.rs`).
  - **✅ First heap program runs correctly, from Nim source — MET 2026-07-31.** `tests/nim_e2e.rs`
    now drives the toolchain on a real allocating program (`var s: seq[int] = @[]; while … s.add(i*i)`;
    sum it), links it with the runtime shim, runs its C `main` through the **entire init chain**, and
    reads the module global `r` back — `sumSquares(4) == 14` on **both engines** (§9 parity). This is
    the first genuine heap allocation on SVM end-to-end: the allocator, `oomHandler`, the frame
    handoff, and the funcref-initializer materialization all exercised at once, from source.
  - **✅ Both earlier follow-ups fixed by adopting the powerbox memory model** (2026-07-31). The
    root of both was that the linked module had no disciplined window layout: nimony based its
    globals at offset 16 and the caller seeded `$sp = 0`, so the **data stack collided with the
    globals, the powerbox heap-brk words (offsets 32/40), and the seq the allocator was growing**.
    That corruption produced the `+20` phantom element in `for x in s` (an allocator chunk write
    stomping `s.len`) and the module-order sensitivity (the seq landing on live scratch depended on
    layout). The fix mirrors svm-llvm's C on-ramp: globals based at `POWERBOX_STACK_PAGE` (16384, so
    page 0 stays reserved for the heap-brk/args scratch), `$sp = powerbox_entry_sp` (64 KiB-aligned,
    above all globals), and the heap seeded above the 1 MiB stack reserve via `POWERBOX_HEAP_BRK`.
    Globals / data stack / heap are now disjoint by construction — no SVM changes, the model the C
    on-ramp already runs under. `for x in s` now sums correctly (`sumSquares(5)` via `for` == 30 on
    both engines), and `s.len` is exact for every element count.
    - **Also fixed (uncovered once the layout stopped masking it): unsigned comparison lowering.**
      `svm-leng` lowered `(u N)` comparisons as **signed** (`le_s`/`lt_s`); the TLSF allocator's
      `uint32` bitmaps set bit 31, which a signed compare mis-orders — faulting `msbit`/`lsbit`
      during any reallocation past 2 elements. `compare` now emits `lt_u`/`le_u` when an operand is
      unsigned-typed (a `(u N)` slot or a `u`-suffixed/`conv`-typed form).
  The ARC-heavy string path already links and runs against real `=wasMoved`/`=destroy`.
- **W3 — Runtime bottom edge** (scoped in detail in §3b). Raw syscalls / the allocator →
  POSIX-personality named imports + the Memory cap (same seam as Phase 1), and mapping nimony's TLS
  onto svm's model (the on-ramp gap Phase 1 already surfaced). ARC destructors/dup calls pass
  through as ordinary calls. **Key finding (§3b): the bottom edge is only ~15 C functions, and
  Phase 1 already binds them** — so the lever between "translates real nimony" and "runs it" is
  mostly W2 (linking the compiled `system` module), or a Phase-1-style host runtime shim.
- **W4 — Multi-binary architecture (the other long pole).** nimony is not one binary: `nifmake`
  spawns `nifler` → `nimony` → `hexer` → `lengc` as subprocesses. Running the compiler on svm
  means either driving those phases in-process or giving svm a subprocess/exec personality. This
  is an architecture question, not a translation one, and it's the biggest unknown.
- **W5 — Bootstrap + browser.** Compile the Rust `svm-leng` to wasm, on-ramp it to svm, and run
  the loop (nimony-on-svm + svm-leng-on-svm) — first headless, then as a playground demo.

**Near-term milestone — ✅ MET (2026-07-29, see §3b Path B): compile & run one real Nim program
end-to-end** — source → nimony → hexer → `svm-leng` → svm-ir → runs on both engines with the right
answer (a real seq build-and-sum returns `3`). That
exercises W1 (totality on a whole program) and forces the first slice of W2/W3, and is the
concrete "it works" we can point at before the long poles. Everything below `## 3a` (W2/W4
especially) is bounded but real; the backend mapping is the part that's no longer in doubt.

## 3b. W3 scope — the runtime bottom edge (W1 is done; this is the next lever)

With W1 closed, `svm-leng` **translates real nimony modules to verified svm-ir** — but the lowered
code doesn't yet *run*, because it calls procs that aren't defined in the one module we translate.
Scoping W3 means answering exactly *what* those calls are and *how* they bind. Two layers, and the
boundary between them is the whole story:

**Layer 1 — compiled Nim stdlib (this is W2, not W3).** The seq/string/ARC ops a program calls —
`newSeqUninit`, `add`, `[]`, `len`, `toOpenArray`, `=destroy`, `=wasMoved` — are **ordinary Nim
code** that nimony compiles into the `system` module's Leng (`sysvq0asl.x.nif`). They look like
"imports" to us only because we translate one module in isolation. In a whole-program build they're
*defined*, reached by **linking** the user module with the compiled `system` module — the `Func`/
`Slot` bindings of `svm_ir::resolve_imports_with`. That's W2 (the linker), and it's the bulk of the
gap.

**Layer 2 — the true bottom edge (this is W3).** What does the `system` module *itself* bottom out
at? Measured directly from `hexer`-compiled `sysvq0asl.x.nif`, the runtime's **entire** external
(`importc`) surface, minus pure C *type* names, is ~15 functions:

| Group | Symbols | SVM binding |
| --- | --- | --- |
| Allocator | `mmap`, `munmap` | the **Memory cap** (Phase 1 seam) |
| Syscalls / process | `write`, `_exit`, `getpid`, `kill` | the **POSIX personality** (Phase 1 seam) |
| libc mem | `memcpy`, `memset`, `memcmp` | host cap, or lower to `mem.copy`/`mem.fill` |
| Atomics | `__atomic_{load,store,add_fetch,sub_fetch,exchange,compare_exchange}_n` (+ `__ATOMIC_*` order consts) | single-threaded guest → plain loads/stores |
| Builtins | `__builtin_{bswap64,clzll,ctzll}` | direct svm ops (`bswap`/`clz`/`ctz`) |
| Dynamic linking | `dlopen`, `dlsym`, `dlclose`, `dlerror` | unused by a static program → stub / fail-closed |

**The key finding: W3's hard part is already retired.** This is the *same* C bottom edge Phase 1's
on-ramp already binds — `crates/svm-run/demos/nimony/` runs a nimony-shaped module on all three
engines today, with `write`/`mmap`/`_exit`/`memcpy` resolved through the POSIX personality + Memory
cap. So the bindings exist and are proven; W3 is *wiring*, not invention. `resolve_imports_with`
already lowers a named import to a host capability (`Cap`) — that's the seam.

**Two paths to the near-term milestone (run one real program):**

- **Path B — host runtime shim first (recommended, no linker).** Skip compiling Nim's `seqimpl`;
  bind the *high-level* ops (`newSeqUninit`/`add`/`[]`/`len`/`=destroy`/`=wasMoved` + `memcpy`/
  `memset`) directly to a small host implementation via capability bindings, exactly as Phase 1's
  `nimony_runtime_shim.c` did. Gets end-to-end *running* fast, decoupled from the W2 linker. The
  handful of ops is small and well-understood (a `{len,data*}`/`{len,cap,data}` bump/realloc
  allocator over the window).
- **Path A — link the real `system` module (fidelity, needs W2).** Merge `sysvq0asl.x.nif` into the
  user module so the stdlib ops resolve to *compiled Nim* (`Func`/`Slot`), and only the ~15 C
  primitives hit the host (`Cap`). Faithful, but gated on the W2 linker.

Recommendation: **Path B first** — mirror Phase 1 (shim → real) to hit "runs a real Nim program
end-to-end", then do W2 + Path A for fidelity. Remaining unknowns are small and known: nimony's TLS
model onto svm (now settled — §3d: single-threaded `tvar → plain global`), and confirming the ARC
destructor protocol runs correctly against a real allocator.

**✅ Path B — DONE 2026-07-29: the near-term milestone is met.** A real nimony seq program **runs
end-to-end on SVM**, both engines, §9 parity. `svm-leng` lowers genuine `hexer` bytes for
`sumSeq`/`makeSeq` to verified svm-ir with their stdlib ops as named imports; a tiny SVM **runtime
shim** (the pure `toOpenArray`/`len`/`[]`/`inc` ops + a bump/realloc allocator for `newSeqUninit`/
`add`, `=wasMoved`/`=destroy` as zero/no-op — eight functions, ~90 lines of svm-text) is **linked
in** via `svm_ir::link`, binding each named import to a shim function. So the whole path — real Nim
→ `nimony` → `hexer` → `svm-leng` → svm-ir → link → **run** — closes with the right answer:
`sumSeq([10,20,30]) = 60`, `makeSeq(3)` builds `[0,1,2]` through the allocator, and a driver chaining
`makeSeq(3)` → `sumSeq` returns `3` in one pass (`tests/end_to_end.rs`). Notably the shim is *SVM
code linked in*, not Rust host capabilities — so it stays inside the pure-IR / both-engines model
and rides the same verifier. This is the linking mechanism W2 generalizes (many units → one), and
the shim is the placeholder the real compiled `system` module (Path A) will replace.

## 3c. W4 scope — the multi-binary driver (the unknown is retired: svm already has both seams)

W4 asked (`## 3a` above): nimony is not one binary — `nifmake` spawns `nifler` → `nimony` →
`hexer` → `lengc` as OS subprocesses, so running the compiler on svm means "either driving those
phases in-process **or** giving svm a subprocess/exec personality," flagged as "the biggest
unknown." **Both halves of that dichotomy already exist as landed seams, and neither is new
substrate** (INVARIANTS §1/§4, §4 below). W4 is no longer an open architecture question; it is a
build-out on proven mechanism, exactly like W3 turned out to be.

**The two seams, measured:**

- **In-process — `svm_ir::link`** (`crates/svm-ir/src/lib.rs:3762`). Statically links N units into
  one import-free module: functions concatenated + reindexed, each unit's data placed in a
  host-page-aligned non-overlapping window region, cross-unit symbols resolved to direct calls.
  This is the **W2** seam, already proven on real nimony (`link_units`, §3 above). Its shape:
  **one** module, **one** window/powerbox, **one** flat export namespace (a collision is
  `LinkError::DuplicateSymbol`, `lib.rs:3835`).
- **Subprocess/exec personality — the `exec` capability** (`EXEC.md`, `crates/svm-run/src/exec.rs`).
  A guest imports one interface `"exec"` (resolved by name like `"fs"`); the wirer picks the
  backend. The load-bearing one is **`domain_exec`** (`exec.rs:61`, BUILT 2026-07-23): each spawn is
  **a fresh child svm domain — its own window, powerbox, and fuel** — `argv[0]` resolves through a
  program registry (a miss is `-EPERM`), the full argv rides the §3e args buffer so an ordinary
  `main(int, char**)` reads it standalone, wire `stdin` seeds the child, both output streams are
  captured, and the exit code is the child's entry result verbatim. It is **not** new substrate: it
  is a `HostCap` composed over the §14 Instantiator machinery svm already has (op 13
  `instantiate_module_named` + `join`), the same mold as `fs`. Proven byte-for-byte on all three
  engines (`crates/svm-run/tests/exec_cap.rs:141`), incl. stdin flowing into the child
  (`exec_cap.rs:219`).

**Recommendation: the phase toolchain is `exec`/`domain_exec` (the subprocess route), not `link`.**
The reason is granularity. nimony's phases are **separate whole programs** — each has its own
`main`, its own globals, its own heap, and its own copy of the compiled `system` runtime. Collapsing
the four into one module with `link` would (a) collide immediately on `DuplicateSymbol` (four
`main`s, four `system` modules) and (b) force all four to share one window / one powerbox / one
allocator arena — semantically wrong for processes that nimony deliberately isolates. `link` stays
the right tool at the **other** granularity — merging the modules *within a single program or phase*
(that *is* W2, and Path A merges the user program with `system` this way). So the two linkers sit at
two levels: **`link` = within-program (W2); `exec` = across-phases (W4).** They compose — each phase
is itself a `link`ed module, and the driver `exec`s the phases in sequence.

There is already a **working precedent at exactly W4's shape**: the compiled-C shell drives
`instantiate_module_named` (op 13) + `join`, resolving `argv[0]` against a name → `Module` registry,
running an unmodified `main(argc, argv)` child with inherited stdout and seeded argv, and threading
each command's exit status into `$?` (`crates/svm/tests/c_shell_exec.rs`,
`crates/svm/tests/stage1_exec_command.rs`). A `nifmake` driver is that shell with a fixed four-command
script.

**Passing intermediate files between phases.** nimony's phases hand `.p.nif`/`.s.nif`/`.nif`/`.c`
files down the chain. Two existing options, no new host op:
1. **stdout → stdin piping** — free with `domain_exec`: the driver drains phase N's captured output
   (`read_out`) and seeds it as phase N+1's `stdin` (the `CAT_CONSUMER` pattern, `exec_cap.rs:185`).
   Fits a streaming `hexer | lengc` shape.
2. **A shared memfs** — the file-based `nifmake` shape (phase N writes `x.nif`, phase N+1 reads it).
   This is the faithful hand-off. It has **two** parts, and they sit on opposite sides of the
   security boundary — worth stating precisely, because the memfs machinery mostly already exists:
   - (a) **A store shared across domains** — the *data* layer. `mem_fs_seeded_handler` re-seeds a
     *fresh* store per grant (isolated filesystems), and `mem_fs_seeded_shared` shares one store but
     only host↔handle (its `MemFsHandle` is host-side, built for browser-Postgres session snapshots;
     it's used single-guest in `crates/svm/tests/c_link.rs` to seed the cc1 memfs). Neither shares a
     store **guest↔guest**. That piece is now built and tested: **`mem_fs_shared_factory`**
     (`svm-fs/src/lib.rs`) mints N `HostProc`s over one `Arc<Mutex<MemFsState>>`, so a file one domain
     writes another reads — proven at the op level by `two_grants_from_the_factory_share_one_store`
     (phase A writes `x`, a separate grant reads it back). It's an ordinary additive `svm-fs` helper:
     it changes no existing grant and hands the *caller* the choice to share, so it is **not** on the
     security boundary. (`mem_fs_seeded_shared` now delegates to it — one grant from the factory.)
   - (b) **Granting that store to a spawned child** — the *authority* layer, and the real gate. This
     decides *what filesystem authority a spawned child inherits* — a security-shaped call (INVARIANTS
     §1/§4), so it was held as an owner-reviewed decision, not a nimony-lane edit. **Now made, and
     wired** (`domain_exec_with_fs`, `exec.rs`): plain `domain_exec` still runs each child via `run`
     with **only stdin/stdout** — the default is unchanged, a child gets **no `fs`** — and a child
     gains `fs` *only* when the embedder builds the backend with `domain_exec_with_fs`, which runs each
     child via `run_with_caps` granting the one shared memfs under the name `"fs"`. The grant lives at
     the parent's construction site; a child cannot widen it, and gets *only* that in-memory store (no
     host filesystem, no ambient authority). The confinement default is pinned by a test
     (`a_child_gets_fs_only_when_the_parent_grants_it`: a probe resolves `fs` → granted only under
     `domain_exec_with_fs`, refused under plain `domain_exec`). This is the same `run_with_caps` seam
     that would grant the real `system`'s bottom-edge caps to a child — one child-capability-inheritance
     mechanism covers both.

**v1 gaps to hold (all bounded, none blocking).** `domain_exec` v1 runs each child **blocking and
one-shot** — no concurrent pipeline (`exec.rs:59`). For a compiler driver this is *fine*: the phases
run strictly in sequence anyway. And the shared-memfs wiring (option 2) is deliberate, not the
default grant. Remaining real W4 work is therefore **build-out, not invention**: (i) compile each
phase (`nifler`/`nimony`/`hexer`/`lengc`) to an svm module — Phase 1's C on-ramp already does this
for one binary, so it is four applications of a proven step; (ii) write the driver module that
registers them and chains them with a shared memfs; (iii) confirming each phase's allocator/`system`
runtime boots cleanly as an isolated child. (The **TLS model** that Phase 1's on-ramp surfaced is now
settled — see §3d: single-threaded `tvar → plain global`, done and tested.)

**First slice — ✅ the mechanism, proven with stand-in phases.** Mirroring how Path B's shim proved
the runtime edge before the real `system` module: a driver module runs stand-in "phase" child
modules via the `exec` cap in sequence, **passing each phase's output as the next phase's input**,
and the final result is the composition — the `nifmake` orchestration on svm, decoupled from the real
toolchain (`crates/svm-run/tests/multibinary.rs`; the driver + stand-in phases are pure SVM modules
over the `exec` cap, so the proof lives with that seam, not in `svm-leng`). Three cases, all green on
all three engines: (1) a two-phase hand-off (content-sensitive — `a → aa → aa!` — so it witnesses the
data flow, not just that two children ran); (2) the **full four-phase depth** (nifler → nimony →
hexer → lengc), `a → ab → abc → abcd → abcde`; and (3) **run-and-check-exit abort** — the driver reads
each phase's exit status (`exec` op 3) and `br_if`s to a short-circuit block, so when a phase exits
non-zero the pipeline stops with that phase's status and the later phases never run (the identical
driver module, only the phase registry differs — so the abort is the driver reacting to status, the
real `nifmake` control flow). This retires the "can the driver shape even run on svm" question,
including its failure handling; what's left (above) is compiling the actual phases and — for the
file-based hand-off specifically — the shared-memfs infra measured under "passing intermediate files."

**Second slice — ✅ a real compiled program as an isolated `exec` child** (step (iii), on a real
binary). The first slice's phases are hand-written; this runs output from the actual `Leng → SVM-IR`
backend as a `domain_exec` child. A Leng module with real control flow (a counted `while` loop) is
lowered by `svm-leng`, **verified**, registered as a phase, and `exec`d by the same nifmake-shaped
driver; its `main()` returns 55 (sum 1..10), `domain_exec` maps that return to the child's exit code,
and the driver reads it back via the status op and re-exits with it — so **exit 55 witnesses the whole
path**: frontend output → verify → isolated child domain → correct compute → status to driver
(`crates/svm-run/tests/multibinary.rs::driver_runs_a_real_svm_leng_compiled_program_as_an_exec_child`,
all three engines; svm-leng is a test-only dep of svm-run, no cycle). This proves a *real* compiled
binary boots and computes correctly as an isolated child — the previously-open half of (iii).

The remaining half of (iii) is **heap/`system`-backed** phases, and it splits cleanly by what the
child needs at its window edge. A self-contained program (allocator over its own window, the Path-B
shim) needs only that the shim **self-seed its brk at startup** rather than rely on harness seeding —
an in-lane change to the shim, no infra. The mmap-backed real `system` additionally needs
**bottom-edge host caps granted to the child** (`mmap`/`memcpy`/atomics/`fs`). The *mechanism* for
that is now in place — `domain_exec_with_fs` runs children via `run_with_caps` (see the third slice
below), the same seam any bottom-edge cap would ride — so what's left is only *which* caps a
`system`-backed phase is granted, decided at the parent's construction site.

**Third slice — ✅ the faithful file-based hand-off, with child-fs confinement.** The earlier slices
hand off via stdout→stdin piping; real `nifmake` passes *files*. This closes that: `domain_exec_with_fs`
grants every phase child one shared in-memory filesystem (the `mem_fs_shared_factory` store), so a
phase writes `mid` and a later phase reads it — the data crosses the child boundary through the file,
not a pipe (`crates/svm-run/tests/multibinary.rs::driver_hands_off_a_file_between_phases_through_a_shared_memfs`,
all three engines: `gen` writes "OK" → `use` reads and echoes it → driver stdout "OK"). The
child-capability grant that §3c held as owner-reviewed is **made and wired** as an *explicit,
attenuated, parent-side opt-in*: the default `domain_exec` still gives a child only stdin/stdout, a
child gains `fs` only through `domain_exec_with_fs`, and it gets *only* the one seeded in-memory store
— no host filesystem, no ambient authority, no self-widening. The confinement default is pinned
(`a_child_gets_fs_only_when_the_parent_grants_it`). With this, W4 is **build-out-complete on the
mechanism**: driver shape, real compiled child, and the file hand-off all run on svm; what remains is
compiling the four actual phase binaries (four applications of the on-ramp) and, for a heap phase, the
self-seeding brk — no open architecture question.

## 3d. TLS model — nimony's thread-vars onto svm (single-threaded now, `vcpu.tls` later)

nimony marks its allocator and exception state `__thread` (thread-local); `hexer` emits these as Leng
**`tvar`** (thread-var, the sibling of `gvar`). Phase 1's C on-ramp had no `llvm.threadlocal.address`
lowering, so `demos/nimony/build_nimony.sh` **strips `__thread`** before clang (a `sed` pass with a
`grep` guard that fails the build if any survives) — valid because the guest is single-threaded. This
section commits the Phase-2 backend's model. It is a **two-tier** answer, and Tier 1 is done.

**What svm actually offers (measured).** svm has exactly **one** thread-local primitive: a single
per-vCPU `i64` register, the IR ops `vcpu.tls.get` / `vcpu.tls.set` (§12,
`crates/svm-ir/src/lib.rs:1935-1959`), seeded to the dense vCPU id (root 0, children distinct;
`crates/svm-interp/src/lib.rs:7810`) and read *at the execution point* so it tracks the current vCPU
across fiber migration (D57). It is **not** per-thread global storage — it is one word, meant to hold
a *thread pointer*. Globals are process-global: a global is just a `Data { offset, readonly, bytes }`
segment (`crates/svm-ir/src/lib.rs:4338`) in the **one shared window** every thread sees
(`crates/svm-interp/src/bytecode.rs:8880` — "a thread shares its spawner's window/powerbox"); there
is **no thread-local storage class** in svm-ir. A real `__thread` is therefore the guest's job:
allocate a per-CPU block, put its base in `vcpu.tls`, and index thread-locals off it — the native
fs/gs-base recipe. `DESIGN.md:949` lists `_Thread_local` (with threads) as deferred.

**Tier 1 — `tvar` → plain global (committed, done).** For a single-threaded guest a thread-local has
exactly one instance, so a plain global *is* that instance. svm-leng lowers `tvar` **identically to
`gvar`**: one zero-initialized global at a fixed window offset, exported/linked as an ordinary data
symbol (`translate.rs` `collect_globals`, the `gvar | tvar` arms). This mirrors the on-ramp's
`__thread`-stripping and needs no new IR. It rests on one **invariant**, stated so it can't rot:
*every guest we target runs single-threaded* — each nimony compiler phase is a batch process (W4 runs
them as separate single-threaded domains, §3c), each svm domain is single-threaded, and nimony's own
concurrency is **CPS/`.passive` → state machines** over a minimal `system.nim` (§1), not OS threads.
Under that invariant the collapse is exact. Pinned by `crates/svm-leng/tests/thread_var.rs`: a `tvar`
persists across calls (write-then-read-back), a non-zero `tvar` initializer seeds the window, and a
`tvar` **links cross-module** like a global (the shape of the real allocator's thread-vars in
`system`, referenced from user code) — all on both engines. This is also already exercised
end-to-end: the heap programs of W3/Path A run against the compiled `system` module, whose allocator
state is thread-vars, and they get the right answer.

**Tier 2 — real per-thread `__thread` over `vcpu.tls` (implemented).** For a genuinely multi-threaded
guest (spawns svm threads *and* relies on per-thread `tvar` state), Tier 1's plain global is wrong —
all vCPUs would share one copy. The faithful lowering: (i) each `tvar` gets a fixed offset in a
per-CPU **TLS block** instead of a window offset; (ii) at thread entry the runtime allocates a block
and `vcpu.tls.set`s its base (the root vCPU too); (iii) every `tvar` access lowers to
`vcpu.tls.get()` + the tvar's block offset, exactly as native code adds to the fs/gs base. svm
supplies the base register; the block layout and per-thread allocation are the backend/runtime's
work — no new substrate.

svm-leng implements (i) and (iii) — the backend's half — behind an opt-in `tls_mode`
(`translate_tls` / `Translator::with_tls`; the `tvar` arm of `collect_globals` assigns block offsets,
`lvalue_addr` emits `vcpu.tls.get() + off`). This holds **across modules**: `link_units_tls_with_runtime`
runs a linker pre-pass (`export_tls_vars`) that pools every unit's thread-vars into one **shared block
layout** and hands it to all (`import_tls_layout`), so a `tvar` defined in `system` and referenced from
user code bakes the same offset on both sides — the TLS analog of the `data.sym` relocation a
cross-module global gets, except the offset is fixed at translate time (a `vcpu.tls`-relative constant
the linker can't relocate later). Step (ii) is the runtime's job — the `vcpu.tls.set` at thread entry,
the same division as the C runtime's fs/gs-base setup — so it stays outside the translator (a threaded
guest's thread-start shim, the analog of Path B's allocator shim; `link_units_tls_with_runtime` takes
it as an extra link unit). Proven in `crates/svm-leng/tests/thread_var.rs`, both engines: the lowering
routes a `tvar` through `vcpu.tls`; the `tvar` is **isolated per `vcpu.tls` base** (a driver sets base
B0 and bumps +3, base B1 and bumps +5, reads each back as 3 and 5 — a shared global would read 8); and
a `tvar` **defined in one unit, written cross-module and read via the defining unit's local name, hits
one slot** (offset agreement, at a non-zero offset). Remaining bounded follow-ups, all additive:
non-zero `tvar` initializers (fail-closed now — the per-thread block is zeroed, so non-zero state needs
per-thread seeding by the runtime); cross-module `tvar`s wider than an `i64` scalar (a cross-module
reference is assumed scalar-`i64`, as a cross-module data symbol is); and wiring the actual
thread-start block-alloc/`vcpu.tls.set` shim for a real threaded guest. Tier 1's tests remain the
differential oracle Tier 2 must satisfy when a `tls_mode` program runs single-threaded with its base
set once.

**Status:** the "TLS follow-up" flagged throughout this doc (the Phase-1 on-ramp gap) is **resolved**.
Tier 1 (single-threaded `tvar → global`) is the operative model for the self-host goal — nimony's
compiler is single-threaded — and both tiers are implemented and tested: Tier 1 is the default, Tier 2
(`tls_mode`, single- and cross-module) is ready for when a threaded Nim guest appears, needing only the
additive follow-ups
above.

## 4. Invariants this must respect

- **Untrusted frontend, zero escape-TCB.** Same class as chibicc/`svm-wasm`/`svm-llvm`: the
  verifier re-checks the produced module; a bug is a clean error (INVARIANTS §9, §2a).
- **No new substrate.** Both phases close over existing seams — the on-ramp, the POSIX
  personality, the Memory cap, `prep_svmb`, the `fs` cap memfs. No new host ops. Host stays
  mechanism (INVARIANTS §1/§4).
- **Confinement is the masking lowering.** Nim raw pointers ride the same window+mask regime
  as C; no new emitted-code/window-access surface (INVARIANTS §2).
- **Code-coupled asset.** `nimony.svmb` (and any corpus `.svmb`) regenerate on IR/ABI/encoder
  change, gated in CI — the Postgres/chibicc asset-lane template.

## 5. Non-goals

- Matching every Nim 2 feature — nimony's own coverage at v0.4.0 is the ceiling; effect
  inference etc. are upstream gaps.
- A general Nim package/build tool on SVM (nimble, etc.). The unit is a compiled SVM module.
- Making the Phase-2 backend the *only* path — the on-ramp (Phase 1) stays the low-risk lane
  and the self-host shipping path, exactly as the LLVM-built `chibicc.svmb` ships (§3 there).

---

## Appendix — recorded toolchain commands (Phase 1 step 2, for when a nim toolchain is present)

```sh
# Nim 2.x (apt's 1.6 is too old for nimony)
curl -fsSL https://nim-lang.org/choosenim/init.sh | sh    # or: choosenim 2.2.0
export PATH="$HOME/.nimble/bin:$PATH"

# nimony
git clone https://github.com/nim-lang/nimony && cd nimony
nim c -r src/hastur build all          # bootstrap the toolchain
# emit Leng-derived C for a program (via the C backend):
#   the toolchain drives nifler → nimony → hexer → lengc c; capture the generated .c

# on-ramp the C (mirrors build_chibicc_svmb.sh)
clang-18 -O2 -emit-llvm -c -mlong-double-64 prog.c -o prog.bc
svm-llvm-translate prog.bc -o prog_raw.svmb --binary --host-page 65536 [--stub-externs]
cargo run --release -p svm-run --example prep_svmb -- prog_raw.svmb prog.svmb
```
