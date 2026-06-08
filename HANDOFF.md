# Handoff — C frontend (chibicc → SVM IR) + differential fuzzing

Pick-up notes for a fresh session. Written 2026-06-03, **last updated 2026-06-07**.
Branch: **`main`** (this work has been committing straight to `main`; the remote is
`theSherwood/vm`). Everything below is committed and CI-green.

**Latest (2026-06-08):** §12 **JIT concurrency — commit 3: fibers run in the JIT.** `cont.new`/
`cont.resume`/`suspend` now lower (x86-64 unix) to a host fiber runtime (`svm-jit/src/fiber_rt.rs`)
over the `svm-fiber` stack switch: a boxed `FiberRuntime` (address baked in like `CapEnv`) + three
`extern "C"` thunks the JIT calls via `call_indirect` (threading `mem_base`/`fn_table_base`/`trap_out`
like `cap.call`), plus one generated CLIF call-trampoline that bridges Rust → the guest `Tail` ABI
(`(i64 sp, i64 arg) -> i64`). A fiber body runs JITted guest code on its own native control stack; a
`suspend` deep inside switches the whole stack back (the §3d two-stack model). Reentrancy is sound:
no `&mut FiberRuntime` is held across a switch (only a `*mut Fiber` to an address-stable boxed fiber),
and a `chain` rejects re-entrant resume. **Verified by the interp↔JIT differential** (`jit_fibers.rs`,
6 tests incl. a 3-level nested resume chain and a data-stack+memory fiber) — fibers are deterministic,
so the strongest oracle applies. `TrapKind::FiberFault` added; the old "JIT bails on fibers" test is
now platform-gated. Full suite + clippy + fmt green. **Next (commit 4):** the native M:N thread
scheduler (`thread.spawn`/`join`/`wait`/`notify`) — extract the shared safe `svm-sched` crate and run
JIT vCPUs as green threads (`Fiber` per vCPU) under TSan.

**Commit 4, part 1 DONE — cooperative thread-scheduler core (`svm-jit/src/thread_rt.rs`).** The
algorithm behind `thread.spawn`/`thread.join`, built on `svm-fiber`: a vCPU is a `Fiber`; a `Sched`
keeps a runnable queue and drives one vCPU at a time (cooperative / single OS thread — true multi-core
workers are a later step), a vCPU runs until it blocks (`join` on an unfinished child) or finishes,
blocking suspends its fiber back to the scheduler loop and a child's completion re-enqueues its
waiters. Same reentrancy discipline as `fiber_rt` (no `&mut Sched` across a switch; only a `*mut Fiber`
to an address-stable boxed fiber crosses it). Backend-agnostic (a vCPU body is any closure), so it's
**unit-tested standalone** (4 tests: spawn/join sum, nested spawn, join-blocks + forged/re-join inert,
16 interleaved children) — the novel scheduling logic is de-risked before codegen wiring. clippy/fmt/
suite green. **Findings that shape the rest:** (a) detect-and-kill uses a C `setjmp`/`siglongjmp` shim
wrapped around a single `Entry` call (`mem.rs` `run_guarded`), so driving the scheduler under the guard
needs a scheduler-as-`Entry` shim (a fault `longjmp`s back, abandoning fiber stacks — fine, the domain
is being killed). (b) **Verification limit:** TSan only instruments Rust, *not* JITted machine code, so
it cannot see JITted guest memory accesses — JIT concurrency is verified by invariant **stress** tests
+ the interp↔JIT **differential** on interleaving-invariant programs (e.g. atomic counters), not TSan.
**Commit 4, part 2 DONE — the JIT runs threads.** `thread.spawn`/`thread.join` now lower (x86-64 unix)
to `thread_rt` thunks (`thread_spawn`/`thread_join`, addresses baked in like `fiber_rt`/`CapEnv`),
threading `mem_base`/`fn_table_base`/`trap_out` from the call site. A threaded module runs under a
**scheduler shim** (`sched_entry`, shaped as `Entry`) driven by `run_guarded`: the guest entry becomes
root vCPU 0 (via the buffer-ABI entry trampoline), spawned vCPUs run JITted thread entries through the
shared fiber call-trampoline, and `join` blocks by suspending back to the scheduler loop. All vCPUs
share the one `Arc<Region>`-backed window (same `mem_base`) → shared memory + hardware atomics for free.
A vCPU trap sets the cell and the scheduler stops (domain killed; a fault `longjmp`s out, abandoning
fiber stacks). `TrapKind::ThreadFault` added; fibers+threads in one module bail `Unsupported` for now
(needs per-vCPU fiber tables). **Verified by the interp↔JIT differential** (`jit_threads.rs`, 3 tests:
spawn/join sums results, a 4×100 atomic counter → 400, double-join → `ThreadFault`). Full suite +
clippy + fmt green.

**Commit 4, part 3 DONE — `wait`/`notify` in the JIT.** `<ty>.atomic.wait`/`atomic.notify` now lower
to `thread_rt::thread_wait`/`thread_notify`: a futex over the scheduler. `wait` confines+aligns the
address (the lowering's `mask_addr` + `guard_atomic_align`), reads the value in the thunk, returns
`NOT_EQUAL` if it changed, else parks the vCPU on `Block::Wait { key=phys, deadline }` (suspend to the
scheduler); `notify` wakes up to `count` waiters on the key (insertion order). The scheduler gained a
`wait_waiters` list + a **logical clock** (advanced only to fire the earliest deadline when nothing is
runnable → timeouts are schedule-deterministic, like the interpreter's `DetSched`) and delivers the
wake status (`WOKEN`/`TIMED_OUT`) via a per-vCPU `resume_val`. Unit tests: a waiter blocked then woken
by a notifier (`WOKEN`), and an un-notified waiter that times out (`TIMED_OUT`). **Differential**
(`jit_threads.rs`): a futex handoff (producer payload + flag + notify, consumer waits then reads) →
987654 on both backends. Full suite + clippy + fmt green. **Next (part 4):** true multi-core workers —
the scheduler is cooperative / single OS thread today (correct under every interleaving, just no
parallel speedup; TSan can't see JITted accesses, so that step leans on the same invariant/differential
oracle, not TSan).

Earlier — §12 **JIT concurrency commit 1: the `svm-fiber` stack-switch primitive.**
Starting the unified-M:N-for-the-JIT effort (the agreed full path, not 1:1 OS threads). Because
`svm-interp` is `#![forbid(unsafe_code)]`, the native stack-switching `unsafe` lives in a new dedicated
crate `svm-fiber` (mirroring how `svm-mem` isolates memory `unsafe`). It implements a `boost.context`
`fcontext`-style symmetric switch on **x86-64 unix** via stable naked functions: `jump(to, data) ->
Transfer{fctx, data}` (push the 6 callee-saved, swap `rsp`, pop, `ret`; the two transferred words ride
in `rax:rdx`), `make(stack_top, entry)` (lay out a fresh stack so the first jump lands in a trampoline
that calls `entry`), and a guard-paged `Stack` (mmap + `PROT_NONE` overflow guard, §5). 5 tests pass:
roundtrip accumulation, runs-on-the-fiber-stack, deep recursion (fib(25)), 100k switches stable, two
independent fibers. Other targets compile but `supported()` is `false` (JIT keeps bailing there).
**Commit 2 DONE:** a safe RAII `Fiber`/`Yielder` *asymmetric coroutine* over the raw primitive —
`Fiber::new(stack, |y, first| …)` runs a boxed closure on its own stack, `resume(val) -> State::{Yielded,
Complete}`, `y.suspend(val)` switches back; values ride through a single-threaded `Cell` mailbox
(`Control`). RAII frees the stack; a never-started fiber's closure is reclaimed on drop; a panic in the
body is caught and converted to `abort` (unwinding across a stack switch is UB). 10 tests pass (yield/
complete, env capture, drop-before-start drops the closure, independent state across interleaved fibers).
**Architecture decision (agreed):** the cleanest end state is *one* safe generic M:N scheduler crate
(`svm-sched`) shared by both backends — the `Fiber` surface is safe, so a worker driving
`fiber.resume()` is safe code, and both task kinds present the same `run(quantum) -> Step`. The
deterministic explorer / `explore_all` oracle stays interp-only (it's a single-thread *replay* model).
That extraction happens at the *thread* scheduler step (commit 4), under full-suite + TSan cover — **not**
now, since fibers need no scheduler. Order: **fibers first (commit 3), then threads (commit 4+).**

**Commit 3 — JIT lowering of `cont.new`/`cont.resume`/`suspend` (PLANNED, fully mapped, not yet built).**
Single-threaded cooperative fibers in the JIT, differentially testable vs. the interpreter. Mechanics
(all sites verified against `crates/svm-jit/src/lib.rs`):
- **Guest ABI:** `sig_from` (lib.rs:632) makes every guest fn `(mem_base, fn_table_base, trap_out,
  params…) -> results` with `CallConv::Tail`. A fiber entry is the unified `(i64 sp, i64 arg) -> i64`
  (§12), i.e. CLIF `(mem_base, fn_table_base, trap_out, sp, arg) -> i64`.
- **funcref** = i32 fn index (lib.rs:1126); `indirect_dispatch` (lib.rs:1257) masks into the §3c table
  (`base + (idx&mask)*16`, code ptr at +8) — reuse to resolve a funcref to a code pointer.
- **Rust can't call `Tail`-conv directly**, so build ONE generic CLIF **call-trampoline** (like
  `build_trampoline`, lib.rs:825): `extern "C" fn(code, mem_base, fn_table_base, trap_out, sp, arg) ->
  i64` that `call_indirect`s `code` with the Tail fiber sig. Finalize it; hand its address to the runtime.
- **`FiberRuntime`** (boxed, address baked as a constant like `CapEnv` lib.rs:707/502): `{ fibers:
  Vec<Option<svm_fiber::Fiber>>, yielders: Vec<*const Yielder>, call_tramp: extern "C" fn(...)->i64 }`.
- **Three `extern "C"` thunks** (lowered as `call_indirect` to baked addresses, passing `mem_base`/
  `fn_table_base`/`trap_out` from `lower.{mem,fn_table,trap}_var`, exactly like `lower_cap_call`
  lib.rs:1314):
  - `fiber_new(rt, mem_base, fnt, trap_out, funcref_idx, sp) -> i32`: resolve code via the table, make a
    `Fiber::new(|y,arg| { rt.yielders.push(&y); let r = (rt.call_tramp)(code, mem_base, fnt, trap_out,
    sp, arg); rt.yielders.pop(); r })`; return slot handle.
  - `fiber_resume(rt, handle, val, status_out:*mut i64) -> i64`: `match fibers[h].resume(val) {
    Yielded(v)=>{*status_out=0; v} Complete(v)=>{*status_out=1; v} }` (forged/finished handle → inert,
    matching interp).
  - `fiber_suspend(rt, val) -> i64`: `let y = rt.yielders.pop(); let r = (*y).suspend(val);
    rt.yielders.push(y); r` — pop-before-switch/push-after keeps the top correct under nested
    `cont.resume` (resumer must see *its* yielder while the callee is suspended).
- **Lowering:** add `ContNew/ContResume/Suspend` to `ensure_supported` (lib.rs:655) and arms in the
  block lowering; `cont.resume` returns `(status, value)` via a stack `status_out` slot + the i64 return.
- **Wiring:** create the `FiberRuntime` + call-trampoline in `compile_and_run`/`_with_host` before the
  entry call (lib.rs:565); keep it alive across `run_guarded`; reentrancy on `rt` is single-threaded but
  overlaps (resume holds `&mut fibers[h]` while guest calls suspend) → use raw-pointer/`UnsafeCell`
  interior access (OK in svm-jit). **Caveat:** a fiber control-stack overflow hits the svm-fiber guard
  page, which the JIT's window signal handler won't classify as a clean trap (deep fiber recursion may
  crash rather than `Trap` — acceptable for v1, revisit with the scheduler).
- **Test:** a differential interp↔JIT test on a small fiber program (cont.new a worker, resume/suspend a
  few times, observe values) — the JIT must match the interpreter's fiber semantics exactly.

Before that — §18 **exhaustive interleaving model checker** (`svm_interp::explore_all`):
a stateless (CHESS/`shuttle`-style) checker that enumerates *every* schedule of a small concurrent
program at memory-op granularity and reports the outcome set — turning the seed sweep (sampling) into
a *proof*. Proves the lock-free atomic counter and the wait/notify handoff are interleaving-invariant,
with a negative test (a racy non-atomic counter) confirming it finds the lost update. See the §18
entry under Phase 4. Also this session: a **generator-driven concurrent oracle** (`concurrent_fuzz.rs`,
256 generated commutative-atomic programs vs. an exact checksum on both the explorer and the real
executor). Before that: §12 **real multi-threaded C now runs end-to-end**. `thread.spawn` was
reshaped from `(func, arg)` to `(func: FuncIdx, sp: ValIdx, arg: ValIdx)` with the thread entry
type changed `(i64) -> i64` → `(i64 sp, i64 arg) -> i64`, **unifying threads with fibers** under
§3d's universal SP-first calling convention (param 0 of every function is the data-stack pointer).
The reshape threaded through ir/verify/text/encode/interp + `threads.rs`/`concurrent.rs`. chibicc
(`codegen_ir.c`) gained `__vm_thread_spawn`/`__vm_thread_join`, `__vm_atomic_add`/`_load`/`_store`/
`_cas32`, and `__vm_wait32`/`__vm_notify` builtins (intercepted in `ND_FUNCALL`, with a
`fn_designator`/`func_index` helper to resolve a function operand to its `FuncIdx`). So ordinary C
that spawns threads + hits atomics compiles → IR → runs on the M:N executor: `c_threads_atomic_counter`
(4×500 `__vm_atomic_add` → 2000) and `c_threads_deterministic_sweep` (the same program through
`run_scheduled`, 100 seeds, all 2000). Full suite + clippy + fmt green; threads/concurrent TSan-clean.
See the §12 entries under Phase 4. Before that: §12 **fibers reached real C**. The stack-switch IR ops
(`cont.new(funcref, sp)`/`cont.resume`/`suspend`, opcodes `0xCA..=0xCC`) exist across
IR/text/binary/verify with a **real reference-interpreter** implementation (asymmetric stackful
coroutines: a fiber's continuation *is* its reified `Vec<Frame>`, switched via a fiber table + resume
chain in `run_func`; forgeable i32 handles, masked + inert on forge → `Trap::FiberFault`; each fiber
owns its data stack via the `sp` operand, §3d two-stack split). chibicc lowers
`__vm_fiber_new`/`__vm_fiber_resume`/`__vm_fiber_suspend` builtins to them, so ordinary C
(`long f(long)` bodies) creates/resumes/suspends fibers and runs on the interpreter (interp-only — the
JIT bails `Unsupported`; the machine switch is step 4). Hardened by `fiber_fuzz.rs` (structured: never
panics, deterministic). See the §12 fibers entry under Phase 4 for the full design. Before
that: §13 `SharedRegion` aliasing is now wired on **Windows** (issue #1,
PR #2, merged) — `MapViewOfFile3` over a `VirtualAlloc2` placeholder reservation — so the
feature is complete on all three OS legs (Linux/macOS/Windows), green on `windows-latest` CI.
A fast local Windows loop exists now: `cargo-xwin` (real MSVC) + **wine** runs the test
binaries, incl. the placeholder/view + VEH-guard paths (see "§13 Windows — playbook" in §10).

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

### Test harness (`crates/svm/tests/c_frontend.rs`, 48 tests, two tiers)
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
cargo test -p svm --test c_frontend   # 48 tests, all green (interp == JIT, and == cc)
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
- **Phase 2 complete** (real C on interp + JIT). Solidly into **Phase 3** (JIT + masked
  window + caps + **guard-page/signal detect-and-kill** done; the §4 *large* reserved window is
  the default and the **Memory cap now supports guest-controlled growth** into the reserved tail,
  with the kernel providing physical demand paging for free). Phase-3 remainder is small: `malloc`
  over `map`, and the Phase-4 virtual-memory extras (fault-driven content supply, `SharedRegion`
  aliasing) which the guard-page/signal + sparse-commit foundation is built to extend.
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
  **Still left (Phase 4, not MVP blockers):** fault-driven *content* supply (a guest/parent as pager —
  `userfaultfd`/§14), and cross-domain `SharedRegion` `create`/`grant` (guest-minted regions — needs
  the §14 Instantiator). **`malloc` over `map` is the default guest libc** — the powerbox
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

### Phase 4 / post-MVP (DESIGN-specified, none built)
- [ ] Concurrency: fibers / vCPUs / M:N green threads, atomics, the C11 memory model,
  real threads (§12). **Atomics — first slice DONE:** linear-memory atomic ops across the whole
  pipeline — `iface`-free IR (`Inst::AtomicLoad`/`AtomicStore`/`AtomicRmw`/`AtomicCmpxchg`, `ty` ∈
  {i32, i64}; `AtomicRmwOp` = add/sub/and/or/xor/xchg), text (`<ty>.atomic.<op>` with an `offset=`
  memarg, no `align`), binary (opcodes `0xC6..=0xC9`), verify, interp reference, and JIT lowering to
  Cranelift `atomic_load`/`atomic_store`/`atomic_rmw`/`atomic_cas`. **Natural alignment is required**
  — a misaligned effective address traps (`MemoryFault`) on both backends (interp `check_align`; JIT
  a software `guard_atomic_align` before the hardware atomic, so it's portable, not the guard page).
  Differentially tested in `jit_diff` (rmw ×6 × i32/i64, cmpxchg hit/miss, atomic↔plain aliasing,
  unaligned-traps-both; non-vacuous — corrupting the JIT rmw map fails it) + a parse/print/encode/
  decode round-trip in `pipeline`. **Still to come (§12):** narrow widths
  (8/16/32), fibers/vCPUs/M:N scheduling. The atomics (with orderings) + `atomic.fence` **are now
  emitted by the `irgen` differential fuzzer** — naturally-aligned addresses so they exercise the real
  atomic path, generated across the interp↔JIT differential (4000 modules) + the escape-oracle, with a
  coverage guard asserting they appear; still not emitted by the chibicc frontend (focused tests cover
  that).
  **Parallel threads — Phase 1 DONE (shared-memory substrate + real interp atomics):** new escape-TCB
  crate **`svm-mem`** owns the guest anonymous-page backing as a `Region` — on unix one demand-zeroed,
  page-aligned anonymous `mmap` of the window's *reserved* extent (the shareable substrate multi-vCPU
  execution will run over), with a portable `BTreeMap`-paged fallback (non-unix / a reservation too
  large to map; single-threaded-only). All the `unsafe` (mmap + raw-pointer atomics) lives in
  `svm-mem`, so **`svm-interp` stays `#![forbid(unsafe_code)]`**. The interpreter's `Mem` now holds a
  `Region` instead of `pages: BTreeMap`; `byte`/`set_byte`/`map`/`unmap`/`map_region`/`snapshot_window`
  route through it, and the four atomic ops use the region's **real `AtomicU32`/`U64` seq-cst
  hardware atomics** for anonymous pages (§13 `Backed` pages keep the value-correct `read_le`/`write_le`
  path). **Behaviour-preserving** (the 46 `jit_diff` differential cases, 7 escape-oracle snapshots,
  §13 `shared_region`, and the C-frontend suite all unchanged; clippy `-D warnings` + windows-gnu
  green). Single-threaded the atomic *values* are identical; the win is the substrate + genuine atomic
  instructions, ready for **Phase 2**.
  **Parallel threads — Phase 2 step 1 DONE (`Region` is soundly shareable):** the substrate is now
  `Send + Sync`, so multiple OS-thread vCPUs can hold `&Region` and run over the *one* guest memory
  image. Every accessor takes `&self`; the `unsafe impl Send/Sync` sits on `Mapped` (the only
  raw-pointer holder) with a documented contract — `atomic_*` are real seq-cst hardware atomics
  (the sound shared primitive); single-byte `byte`/`set_byte` use **relaxed atomics** so even a
  same-byte race is *defined*, not UB (a plain `mov` on x86); bulk `zero`/`read_into` are control-plane
  (map/unmap/snapshot, not raced against live access). The `Paged` fallback moved behind a `Mutex`
  (correct-but-serialized; not the parallel path). **Validated under ThreadSanitizer** (`-Zsanitizer=
  thread`): the headline test — 8 threads × 20 000 `fetch_add` on one shared counter landing on the
  exact total — plus a disjoint-plain-write test report **zero data races**. `Region` adds no
  ordering/scheduling policy; that lives above it.
  **Parallel threads — Phase 2 step 2 DONE (`thread.spawn`/`thread.join`, real OS-thread vCPUs):**
  first-class IR ops threaded through the whole pipeline (`Inst::ThreadSpawn { func, arg }` → i32
  handle, `Inst::ThreadJoin { handle }` → i64; text `thread.spawn <funcidx> vN` / `thread.join vN`;
  opcodes `0xCD`/`0xCE`; verify checks the spawnee is the fixed thread-entry type `(i64)->i64` via new
  `VerifyError::ThreadEntrySignature`; the JIT auto-reports them `Unsupported` like fibers, so they're
  interp-only — no differential pairing). The interpreter runs a spawned vCPU on a **real OS thread**
  inside a per-run `std::thread::scope` (so the child can borrow the module's `&'a` funcs and
  stragglers are joined at run end); a run-wide `AtomicU32` budget (`MAX_THREADS=64`, total) makes a
  thread-bomb a clean `Trap::ThreadFault`. The child shares the **same** memory: `Mem.back` is now an
  `Arc<Region>` and §13 region backings moved `Rc`/`RefCell` → `Arc<dyn SharedBacking + Send + Sync>`
  / `Mutex` (so `Mem` is `Send`; `SharedBacking` gained the supertrait, svm-run's `ShmBacking`/
  `WinShmBacking` got justified `unsafe impl Send+Sync`). A spawned vCPU shares anonymous bytes + §13
  aliases live, **snapshots** the page-protection map (post-spawn `map`/`unmap` is thread-local — a
  documented step-2 limitation), and starts with an empty powerbox + its own fuel. Thread handles are
  masked + liveness-checked like fiber/capability handles, so a forged/double `thread.join` is inert
  (`ThreadFault`); a child trap propagates through `join`. Tests: `crates/svm/tests/threads.rs` (×9 —
  shared-memory visibility, 4-way concurrent `atomic.rmw.add` summing exactly, forge/double-join
  inert, child-trap propagation, capture reflects thread writes, verify-rejects-bad-sig, binary+text
  round-trip), **all green under ThreadSanitizer** (`-Zsanitizer=thread`, zero data races) + clippy
  `-D warnings` + windows-gnu.
  **Parallel threads — Phase 2 step 3 DONE (`wait`/`notify` futex):** first-class blocking sync, so a
  vCPU parks instead of busy-spinning. Ops: `Inst::MemoryWait { ty, addr, expected, timeout }` → i32
  status (`0` woken / `1` value-not-equal / `2` timed-out), `Inst::MemoryNotify { addr, count }` →
  i32 woken. Text `<ty>.atomic.wait vaddr vexp vtimeout` / `atomic.notify vaddr vcount`; opcodes
  `0xCF` (+ty byte) / `0xE8`; verify requires declared memory + types the operands; JIT
  auto-`Unsupported`. Runtime: a per-run **parking lot** (`Parking` = one mutex + condvar, generation
  + waiter-count maps keyed by confined address), passed by `&` to every vCPU like the thread budget.
  `wait` confines/aligns/prot-checks the address, then **under the parking lock** compares `*addr` to
  `expected` (atomic with `notify` — no lost wakeup) and blocks on the condvar until the address's
  generation moves or the (host-capped `MAX_WAIT=10s`) timeout fires; `notify` bumps the generation
  and wakes up to `count` waiters. Tests in `crates/svm/tests/threads.rs` (now ×14): not-equal,
  timeout, **cross-thread notify wakeup** (a worker parks, main vCPU notifies — woken status drives a
  100 result), round-trip, verify-needs-memory — all green under ThreadSanitizer + clippy + windows.
  **Parallel threads — Phase 2 step 4 DONE (shared *synchronized* address space):** lifts the step-2
  limitation — `map`/`unmap`/`protect` (and §13 aliases) by one vCPU are now live-visible to all the
  others. The page-protection map + §13 region table moved out of `Mem` into a shared
  `Arc<RwLock<AddrSpace>>`; `fork_for_thread` clones that `Arc` (was: snapshotted the maps), so every
  vCPU reads/writes one address space. Many readers (every `check_prot`) run concurrently under the
  read lock; `map`/`unmap`/`protect`/`map_region` take the brief write lock (mutate the maps, then
  zero pages after releasing). The per-byte hot path stays lock-free via a monotonic
  `Arc<AtomicBool> has_regions`: until a §13 region is aliased (the common case), `byte`/`set_byte` go
  straight to `back` without touching the lock. Lock order is always parking→space (never the
  reverse), so wait/notify + map can't deadlock. White-box test `forked_vcpu_sees_post_fork_mappings`
  (a forked view sees a post-fork `map` then `unmap`); whole suite (incl. §13 `shared_region`,
  `jit_diff`, `c_frontend`) unchanged, TSan-clean (the 4-thread atomic test now hammers the shared
  `RwLock` `check_prot`), clippy + windows-gnu green.
  **Parallel threads — Phase 2 step 5 DONE (C11 memory-ordering surface + `atomic.fence`):** the four
  atomic ops gained an `order: svm_ir::Ordering` field (relaxed / acquire / release / acqrel / seqcst)
  and there's a new `Inst::AtomicFence { order }`. Text: a `.<order>` suffix on the mnemonic, omitted
  for the default seqcst so existing atomics round-trip unchanged (`i32.atomic.load.acquire`,
  `i64.atomic.rmw.add.relaxed`, `atomic.fence`, `atomic.fence.acquire`); binary: an ordering byte per
  atomic + opcode `0xE9` for fence; verify rejects impossible pairs (a load with release / a store
  with acquire — `VerifyError::BadAtomicOrdering`). **Both backends execute every atomic seq-cst** —
  a sound strengthening that keeps the interp↔JIT oracle exact (Cranelift atomics are seq-cst only) —
  so the `order` is carried+validated but not yet weaker-honored; the one place ordering is observable
  is the fence, which the interpreter issues as a real `std::sync::atomic::fence` (Relaxed = no-op, as
  `std` panics on it). **The JIT now lowers `atomic.fence`** too (Cranelift `fence`, seq-cst), so
  fence programs aren't interp-only and get differential coverage. Tests in
  `crates/svm/tests/threads.rs` (now ×18): ordering+fence round-trip (binary+text, seqcst stays
  implicit), verify-rejects-release-load / -acquire-store, and an execute test (release-store /
  acquire-load / fence / relaxed-rmw, value-correct); plus `jit_diff`'s
  `jit_matches_interp_orderings_and_fence` confirming the JIT lowers the ordering suffixes + fence
  identically to the interpreter. Whole suite + clippy + windows-gnu + (the existing) TSan green.
  **Honoring weaker orderings in execution** awaits a backend that supports them + the
  concurrent-oracle story.
  **Parallel threads — Phase 2 step 6 DONE (concurrent-live vCPU budget):** the `thread.spawn` cap
  changed from **total-per-run** to **concurrent-live** — the slot is charged at spawn and **refunded
  when the vCPU finishes** (the refund lands before the handle's result is observable to a `join`, so
  a joiner's next spawn sees the slot free). A guest that spawns-and-joins in a loop can now create
  unboundedly many vCPUs over its lifetime (only simultaneous liveness is capped at 64); previously it
  `ThreadFault`ed at the 65th spawn ever. Test `sequential_spawns_exceed_concurrency_cap` runs 200
  spawn+join iterations (sum 0..199 = 19900). Whole suite + TSan (19 thread tests, zero races) +
  clippy + windows-gnu green. **Note on true M:N:** running *many* vCPUs over *few* OS threads (tasks
  ≫ threads) needs a work-stealing executor that **parks fiber continuations** on every blocking op
  (`join`/`wait`) — an async runtime over the reified-continuation interpreter, with real
  lifetime/deadlock/race design (a naive bounded pool deadlocks once blocked tasks exceed workers).
  That's the next, larger step; today's model is 1 OS thread per concurrent vCPU (now with a reusable
  slot budget) plus cooperative fibers within each.
  **Parallel threads — M:N executor, commits 1–2 DONE (foundation):** the substrate is being moved so
  a vCPU can run on a *pooled* OS thread and **park** its continuation on a blocking op (the payoff is
  scaling past the 64-vCPU cap — thousands of green threads on few cores). (1) `run_func` no longer
  uses `thread::scope`: a vCPU owns an `Arc<[Func]>` + `Arc` runtime state (budget, parking lot) and a
  `thread.spawn` uses a **detached** `std::thread` that publishes its result via `Arc<TaskState>`
  (`thread.join` blocks on it); a shared `Registry` of handles is joined at run end so nothing
  outlives the run. (2) `Frame` stores a `FuncIdx` (not a `&Func`), so a vCPU's reified state
  (`frames`/`fibers`) is **plain owned data** — movable between worker threads, the prerequisite for
  parking. (3) `Frame` stores a `FuncIdx` and `run_func`→`run_vcpu(&mut VCpu)` lifts the driver's
  locals into an owned, movable `VCpu`. **(4–6) DONE — the executor itself.** `VCpu::run` returns
  `Step::Done | Step::Park(Join|Wait)` instead of OS-blocking: on a `thread.join`/`atomic.wait` whose
  event isn't ready it records a `Pending` and **parks** (its owned continuation is set aside, freeing
  the worker); on resume it finishes the op from `pending` (no re-execution). A single-mutex
  `Scheduler` drives it: a run-queue of `Box<VCpu>`, `results`/`join_waiters` maps (a child's
  completion re-enqueues its joiner), `wait_waiters` keyed by confined address (`notify` re-enqueues;
  the value is re-checked **under the scheduler lock** for futex atomicity), and a `timers` min-heap
  drained by idle workers for `wait` timeouts. Worker threads are spawned **lazily** (the calling
  thread is worker 0; more added toward `min(live, MAX_WORKERS=32)` only as vCPUs spawn), so a
  single-threaded guest creates **zero** extra threads. The live-vCPU cap is now `MAX_VCPUS=1<<16`
  (a parked green thread costs only its continuation, not an OS thread). Validated: the full suite
  (jit_diff/c_frontend/escape_oracle/pipeline/run…) green on the executor, `threads.rs` ×20 incl.
  **1000 green threads on ≤32 workers** (`many_green_threads_on_a_small_pool` — impossible under the
  old 64 cap), all **TSan-clean** (`-Zsanitizer=thread`, zero data races), clippy `-D warnings` +
  windows-gnu green.
  **Concurrent verification — DONE (the §18 oracle for multi-threaded code).** The interp↔JIT
  differential can't check threaded runs (thread ops are interp-only; runs are nondeterministic), so
  concurrent code is verified by **property + interleaving exploration**. (a) A **deterministic,
  seeded explorer** — `svm_interp::run_scheduled(m, func, args, fuel, seed)` — runs the *same* vCPUs
  on a single OS thread via `DetSched`/`run_det`, choosing which runnable vCPU to step (and a
  `1..=MAX_QUANTUM` quantum) from a seeded PRNG and timing out `atomic.wait`s on a **logical** clock.
  So a run is a pure function of its seed: fully reproducible, and sweeping seeds enumerates distinct
  interleavings (loom/shuttle-style), with no data races (one thread → each seed is one valid
  sequential interleaving). Enabled by abstracting a vCPU's executor as `SchedRef::{Real,Det}` and
  adding `Step::Yield` + a per-run `quantum` to `VCpu::run` (the real pool passes `u64::MAX`). (b)
  Tests in `crates/svm/tests/concurrent.rs`: a cmpxchg **spinlock guarding a non-atomic counter**
  (8×100 → 800 iff mutual exclusion holds), an **atomic-RMW counter** (8×500 → 4000), and a
  **wait/notify futex handoff** — each run both as real-executor **stress** (×30, OS interleavings,
  TSan-clean) and as a deterministic **seed sweep** (×200, reproducible), plus a reproducibility
  check. A scheduling/lock bug surfaces as a replayable failing seed.
  **Frontend wiring DONE (real multi-threaded C, 2026-06-08):** `thread.spawn` reshaped to
  `(func, sp, arg)` / entry `(i64 sp, i64 arg) -> i64` (unified with fibers under the §3d SP-first
  convention); chibicc lowers `__vm_thread_spawn`/`__vm_thread_join`, `__vm_atomic_add`/`_load`/
  `_store`/`_cas32`, `__vm_wait32`/`__vm_notify` (a `fn_designator`/`func_index` helper resolves a
  C function operand to its `FuncIdx`). `c_threads_atomic_counter` (4 threads × 500 atomic adds → 2000)
  runs on the M:N executor; `c_threads_deterministic_sweep` runs that same C program through the
  deterministic explorer for seeds 0..100 (all 2000). So concurrent C is now verified by both the
  real-executor path *and* the seeded explorer.
  **Generator-driven oracle DONE (2026-06-08, `crates/svm/tests/concurrent_fuzz.rs`):** the explorer
  is now driven by a **structured program generator**, not just hand-written modules. Each program
  seed emits N (2..6) worker threads, each doing `iters` `i64.atomic.rmw.add amount` on one of a few
  shared cells; the worker unpacks its `(cell, amount, iters)` script from the spawn `arg`
  (`(cell<<32)|(amount<<16)|iters`), so one worker function covers all threads. Because atomic
  RMW-add is linearizable and integer add is commutative+associative, each cell's final value — and
  `main`'s weighted checksum `Σ_t (cell_t+1)·amount_t·iters_t` (the `c+1` weight makes a *misrouted*
  add also perturb the result) — is **interleaving-invariant by construction**, so the host computes
  the exact expected value. 256 generated programs are each checked on the deterministic explorer
  (12 scheduler seeds) and the real M:N executor (2 runs); failures are replayable from
  `(program_seed, scheduler_seed)`. Catches lost updates, misrouted stores, and explorer
  interleavings that aren't actually realizable. TSan-clean.
  **Exhaustive model checker DONE (2026-06-08, `svm_interp::explore_all`):** a *stateless* interleaving
  model checker (CHESS/`shuttle`-style) that enumerates **every** distinct schedule of a small
  concurrent program and returns the set of terminal outcomes — turning the seed *sweep* (sampling)
  into a *proof*. Two pieces: (1) **memory-op granularity** — a `memop` flag on `VCpu` makes the
  `quantum` budget count only *visible* ops (`is_visible`: linear-memory accesses + thread/futex ops;
  fences excluded since both backends are seq-cst), so the scheduling decision points are exactly the
  shared-state operations (the partial-order reduction that keeps the tree finite). (2) **stateless
  DFS** — each schedule is one fresh execution replayed from a planned sequence of scheduling choices
  (`Choices` records the branch factor at each `n>1` runnable set); after each run it backtracks to the
  deepest decision with an unexplored sibling. `run_det` was refactored to `run_with_policy(Policy::
  {Seeded,Exhaustive})` so the checker reuses the whole park/join/wait/timeout machinery. Returns
  `Exhaustive { outcomes, schedules, complete }`. Tests (`concurrent.rs`): `exhaustive_tiny_atomic_counter`
  (2 threads × 1 atomic add → proves total is *always* 2, `complete`), `exhaustive_futex_handoff`
  (wait/notify, all interleavings → payload), and a **negative** `exhaustive_finds_known_race` (a
  *non-atomic* load/add/store counter — the checker must find both 2 and the lost-update 1, proving it
  has teeth). **Scope:** stateless + no DPOR beyond memop granularity, so it's for bounded-sync /
  lock-free shapes; a **busy-wait spinlock** is the classic blow-up case (each failed `cmpxchg` retry
  is a fresh decision point) and stays covered by `stress`+`sweep` instead.
  **Next here:** DPOR (dynamic partial-order reduction) to tame contended-lock trees, per-schedule
  memory reuse (each schedule currently re-`mmap`s a fresh reservation), and driving `explore_all` from
  the `concurrent_fuzz` generator for exhaustive proofs of *generated* small programs.
  **Phase 2 still to come:** per-thread capability grants (spawned vCPUs still start with an empty
  powerbox) and honoring weak orderings in execution.
  **Fibers — step 1 DONE (explicit-stack interpreter):** the reference interpreter no longer recurses
  on the host stack for guest calls — the guest call stack is **reified** as an explicit `Vec<Frame>`
  in `run_func` (`svm-interp`), where `Frame = { f, block, inst, vals }`. A `call` pushes a frame, a
  `return` pops and resumes the caller past the call, a tail call replaces the top in place (still
  O(1) frames). This is a **behaviour-preserving refactor** (identical results/traps/fuel; whole suite
  + clippy green on linux & windows-gnu; `MAX_CALL_DEPTH=256` boundary unchanged; new
  `mutual_recursion_traps_not_overflows` test exercises cross-function frames). **Why:** a fiber's
  continuation is exactly its `Vec<Frame>`, so this is the prerequisite for `suspend`/`resume`.
  **Fibers — step 2 DONE (stack-switch IR ops + interp semantics, asymmetric stackful coroutines):**
  three call-clobbering control ops across the whole pipeline — `iface`-free IR
  (`Inst::ContNew`/`ContResume`/`Suspend`), text (`cont.new v{f}` → handle; `cont.resume v{k} v{arg}`
  → `(status, value)`; `suspend v{v}` → `value`), binary (opcodes `0xCA..=0xCC`), verify, and a **real
  reference interpreter** — but **no JIT** (the machine-level switch is step 4, so the JIT cleanly
  *bails* `Unsupported`; `jit_bails_unsupported_on_fiber_ops` asserts this, and the differential
  harness skips fiber modules). Model: a fiber is a first-class suspendable computation whose
  continuation **is** its `Vec<Frame>`; **`cont.new(funcref, sp)`** makes a `Pending` fiber (started
  lazily; the funcref resolved through the table as **`(i64 sp, i64 arg) -> (i64)`** at first resume,
  like `call_indirect`), where `sp` is the fiber's **own data-stack base** — a fiber owns a *stack
  pair* (§3d): its in-window data stack plus the out-of-band control stack (the `Vec<Frame>`). The
  guest allocates each fiber a distinct data stack (e.g. `malloc`/a static buffer). `cont.resume(k,
  arg)` switches in (first entry calls `func(sp, arg)`; later resumes deliver `arg` as the body's
  `suspend` result), `suspend(value)` switches back out (yielding `value`, `status` 0=suspended /
  1=returned). The `run_func` driver holds a **fiber table** + a **resume chain** (root = `fibers[0]`;
  the running fiber's frames live in a local `frames`, its slot held `Running`); the single-stack/no-
  fiber path is byte-identical to step 1 (same depth bound, now summed across the chain). A fiber
  **handle is a forgeable i32**, masked into the table + chain/state-checked at resume so a
  forged/dead/in-chain handle is **inert** (`Trap::FiberFault`, new) — never an escape; `MAX_FIBERS`
  bounds a fiber-bomb. Tested at three levels: focused interp tests in `pipeline.rs` (`fiber_*`:
  value-threading, generator loop, 3-level nested resume chain, resume-after-return / suspend-at-root /
  forged-handle all trap) + round-trip; a **structured robustness fuzzer** (`fiber_fuzz.rs`: random
  multi-function fiber programs never panic the interp + are deterministic); and **real C** (below).
  **Fibers — step 6 (partial) DONE: real C reaches fibers.** chibicc (`codegen_ir.c`) now intercepts
  three builtins — `int __vm_fiber_new(long(*)(long), void *stack)`, `long __vm_fiber_resume(int k,
  long arg, int *done)`, `long __vm_fiber_suspend(long value)` — lowering them to `cont.new` (funcptr
  wrapped to the i32 funcref + the guest stack), `cont.resume` (status stored through `done`, value
  returned), and `suspend`. A fiber body is an ordinary `long f(long)` (IR `(i64 sp, i64 arg)->(i64)`
  by the existing data-SP ABI — *that's why the entry sig carries `sp`*). Interp-only C tests in
  `c_frontend.rs` (`c_fiber_*`: a generator, two-way resume-arg round-trip, two independent fibers on
  distinct stacks interleaving without clobbering — the data-stack-per-fiber property) via a new
  `run_c_interp` helper (the differential `run_c_full` can't drive fibers since the JIT bails). Whole
  suite + clippy + fmt green. **Cooperative C *multithreading* already works on this** with **no new
  VM primitive** (`c_cooperative_threads_round_robin`): a round-robin scheduler written in plain guest
  C interleaves three worker "threads" (each yields via `__vm_fiber_suspend` mid-loop) to completion —
  DESIGN §12's model, where scheduling is runtime/guest policy. What's *not* there: preemption +
  parallelism (need fuel-yield points + vCPUs/the JIT), a `<pthread.h>`/`<threads.h>` shim wrapping
  this pattern, and real `_Atomic` (still plain ops → would race under true concurrency).
  **Plan for C threading on the fibers/vCPU model** (no architectural blocker — the determinism vs.
  threading tension is resolved by running fibers cooperatively on a *single* vCPU in the differential
  oracle; true multi-vCPU parallelism is a separate, non-bit-deterministic mode validated by other
  means). Remaining steps: (4) the JIT's machine-level control-stack
  switch (asm SP swap — the riskiest, escape-TCB-adjacent piece) so compiled fibers suspend mid-callstack;
  (5) `wait`/`notify` (futex over the window) as cooperative park/unpark, which needs a symmetric
  scheduler (a runnable queue) — today's model is asymmetric (explicit resume/suspend); (6, rest) the
  remaining C threading surface: real `_Atomic`/`<stdatomic.h>` lowering (today stubbed → silently
  races), a `<pthread.h>`/`<threads.h>` shim onto the fiber builtins + futex, and `_Thread_local` →
  fiber-local storage. The data-stack half of the two-stack split is already built (chibicc lowers
  address-taken locals to an in-window data stack via data-SP `v0`), so only the control-stack half is
  new work.
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
3. ~~**Real Memory capability + growth**~~ — **DONE** (`map`/`unmap`/`protect` + guest-controlled
   growth into the reserved tail = the §1a sparse-address-space differentiator). Increments 1–3
   below + the growth increments A–C (see the "Real window / Memory capability + growth" checkbox
   above). **Increment 1 ✅ (interp spec):** `Mem` carries a
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
   wrong size → now threads `mapped` into `Lower` and passes it. **Growth — increments A–C ✅:**
   (A) the interp model decouples confinement (mask into `[0, reserved)`) from committed-ness (a
   sparse per-page set over all of `[0, reserved)`), so `map`/`unmap`/`protect` work past the
   prefix and an uncommitted access faults; (B) a production `svm_run::MprotectWindow` (real
   `mprotect` across the reserved range + `MADV_DONTNEED` on `unmap`, a software page mirror so §7
   borrows fail closed) replaces the no-op `WindowMem` in the production `cap_thunk`, and the
   cap-thunk ABI gained `mem_reserved`; (C) the differential fuzzer spans prefix+tail (800 seeds)
   plus a concrete guest-consumer round-trip. Physical demand paging is free (kernel lazy-zero of
   `MAP_NORESERVE` pages). The Memory cap is surfaced in the *main* irgen fuzzer (arm 19, prefix +
   tail) and the `_with_host` escape-oracle snapshot now covers grown tail pages (low 256 KiB),
   pinned non-vacuously by `jit_cap_memory_escape_oracle_grown_tail`. **§13 SharedRegion landed on all
   platforms (incl. windows — issue #1 DONE):** `iface::SHARED_REGION = 4`, `PageProt::Backed` aliasing
   (interp reference, every platform); the JIT match via real shared mappings (`MprotectWindow::
   map_region`) — `mmap(MAP_SHARED|MAP_FIXED)` of a `memfd`/`shm` `ShmBacking` on unix, and on windows
   `MapViewOfFile3(MEM_REPLACE_PLACEHOLDER)` of a `CreateFileMapping` section over a `VirtualAlloc2`
   **placeholder** window reservation (`WinShmBacking`, `os_section`, the `win_commit_rw` placeholder
   commit). interp↔JIT differential `jit_cap_shared_region_aliases_differential` is now
   `#[cfg(any(unix, windows))]`; validated locally via `cargo-xwin` + wine and by `windows-latest` PR
   CI. **Deferred (Phase 4):** fault-driven *content* supply (guest/parent as pager, `userfaultfd`/§14);
   cross-domain region `create`/`grant` (guest-minted regions, needs the §14 Instantiator).

*(Done this session: SSA-promotion pass; the escape-oracle fuzzer (+ nightly `diff`/`mask`
CI, merged); the JIT-vs-Wasmtime bench harness; mask elision for provably-bounded accesses;
loops + indirect calls in the generative fuzzer; guard pages + signal-handler detect-and-kill;
**over-time bench regression tracking** (`bench/ --save-baseline`/`--check` vs a committed
ratio baseline, + an `alu_c` chibicc-compiled kernel tracking the SSA-promotion win end-to-end;
a non-gating nightly CI `bench` job running `--check`); **a structural SSA-promotion guard**
(`c_frontend` asserts zero loop-body memory ops on
promotable loops, so the promotion win can't silently regress); **guest-controlled memory growth
into the reserved tail** (the §1a sparse-address-space capability: interp sparse-commit model +
production `mprotect`-backed `MprotectWindow` wired into `cap_thunk` + `mem_reserved` in the
cap-thunk ABI + prefix/tail differential fuzz + a guest-consumer round-trip); plus a batch of
real-library shakedowns — Clay, jsmn, SHA-256, xxHash, tinfl, stb_perlin (first float), tiny-regex-c
(backtracking) — each vendored as a `demos/` + cc-oracle test; **map-growing `malloc` promoted to
the default `<stdlib.h>`** (`demos/heapgrow`); and **fuzzer hardening** — the Memory cap's success
path now rides the escape-oracle in both the dedicated `jit_cap_memory_escape_oracle_differential`
and the main 4000-seed differential (granted cap + valid `map`/`unmap`/`protect`), which caught two
real bugs (a null-pointer `from_raw_parts` in `cap_thunk`; an over-strict droppable-trap arm).)*
