# Lua first-light fixture

`lua_first_light.bc` is **Lua 5.4.7's core** (lexer, parser, code generator, GC, and the bytecode VM —
no standard libraries) plus a tiny C-API harness, compiled to a single LLVM-18 bitcode module. It is a
committed golden input for the `lua_first_light` translate test: a regression guard that the on-ramp
translates and runs real Lua identically on the tree-walker, bytecode, and JIT.

The harness drives `lua_newstate` / `lua_load` / `lua_pcall` with its own `realloc` allocator and a
string reader, runs this script, and returns the result as the program's exit value:

```lua
local function fib(n) if n<2 then return n else return fib(n-1)+fib(n-2) end end
local t = {}
for i=1,10 do t[i] = i*i end
local sum = 0
for i=1,10 do sum = sum + t[i] end
local str = 'lua language'
local function counter() local c=0; return function() c=c+1; return c end end
local cnt = counter()
cnt(); cnt(); cnt()
return fib(10) + sum + #str + cnt()   -- 55 + 385 + 12 + 4 = 456
```

So it exercises recursion, table create/index, numeric `for`, closures with upvalues, the `#` operator,
and multiple calls — all core VM features, needing none of the fail-closed libc stubs. Expected
result: **456**.

## Regenerating

With Lua 5.4.7 unpacked in `lua-5.4.7/` and clang 18 (see the build recipe in `LLVM.md` §"Lua first
light"):

```sh
CORE="lapi lcode lctype ldebug ldo ldump lfunc lgc llex lmem lobject lopcodes lparser \
      lstate lstring ltable ltm lundump lvm lzio"   # core only — not the lib*.c / lauxlib / lua.c
for f in $CORE harness; do clang -O2 -emit-llvm -c -Ilua-5.4.7/src $f.c -o $f.bc; done
llvm-link *.bc -o lua_first_light.bc
```

where `harness.c` is the C-API driver above. Run it with `cargo run --example run_lua -- <bc> [backend]`.

# Lua + floats fixture

`lua_floats.bc` is the same Lua 5.4.7 core, **linked with the bundled guest `libm` and guest `strtod`**
(`crates/svm-run/demos/libm/libm.c`, `crates/svm-run/demos/strtod/strtod.c`), running a float script
(committed as `lua_floats_harness.c`). It is the payoff fixture for the `lua_floats` test: a single run
exercises, end to end through the whole VM, the guest `strtod` (every numeric literal), the guest `pow`
(the `^` operator), and the synthesized `fmod` (the `%` operator) — plus `frexp`/`localeconv`/`snprintf`/
`setjmp` referenced by the core. The integer-cast result is **1506304** on all three engines, identical
to a native build of the same sources.

## Regenerating (floats)

With Lua 5.4.7 in `lua-5.4.7/`, clang 18, and this repo's guest sources, compile **with
`-fno-vectorize -fno-slp-vectorize`** (the float paths SLP-vectorize to `<2 x double>`, the v128 lane
outside the scalar on-ramp; exact IEEE/integer arithmetic is identical scalar-vs-vectorized) and
`-fno-builtin` on the guest libm/strtod (so clang doesn't rewrite the bodies in terms of the functions
they define):

```sh
NV="-fno-vectorize -fno-slp-vectorize"
CORE="lapi lcode lctype ldebug ldo ldump lfunc lgc llex lmem lobject lopcodes lparser \
      lstate lstring ltable ltm lundump lvm lzio"
for f in $CORE; do clang -O2 $NV -emit-llvm -c -Ilua-5.4.7/src lua-5.4.7/src/$f.c -o $f.bc; done
clang -O2 $NV            -emit-llvm -c -Ilua-5.4.7/src lua_floats_harness.c -o harness.bc
clang -O2 $NV -fno-builtin -emit-llvm -c .../demos/libm/libm.c            -o guest_libm.bc
clang -O2 $NV -fno-builtin -emit-llvm -c .../demos/strtod/strtod.c        -o guest_strtod.bc
llvm-link $CORE.bc harness.bc guest_libm.bc guest_strtod.bc -o lua_floats.bc   # (expand $CORE.bc)
```

The guest `pow`/`exp`/`log`/`sin`/`cos`/`strtod` definitions **shadow** the on-ramp's would-be trap
stubs; `fmod`/`frexp`/`localeconv`/`snprintf`/`sqrt`/`ldexp`/the string ops stay undefined and are
synthesized/recognized by the on-ramp. Pin `EXPECT` in `tests/lua_floats.rs` against a native build of
the identical sources (`cc … -lm`, our strong defs shadowing libm).

# Lua stdlib fixture

`lua_stdlib.bc` is the Lua 5.4.7 core **plus the base/`string`/`table`/`math` libraries**
(`lbaselib`/`lstrlib`/`ltablib`/`lmathlib` + `lauxlib`), linked with the guest libm and a small guest
libc shim (`lua_stdlib_shim.c`: the transcendentals the guest libm lacks, `strstr`, and a
no-filesystem `stdio` surface). The harness (`lua_stdlib_harness.c`) opens the four libraries via
`luaL_requiref` and runs a script that `print`s. It is the fixture for the `lua_stdlib` test, which
asserts the exact **stdout bytes** (through the `Stream.write` capability) on all three engines,
identical to native.

The script exercises `print`, `string.upper`/`rep`/`sub`/`#`, `table.sort`/`concat`/`insert`/`remove`,
`math.sqrt`/`pi`/`floor`/`max`/`abs`, `ipairs`, `pairs`, `type`, `tostring`. It deliberately avoids
`string.format`: that builds the per-directive format spec **at runtime** and calls `snprintf` with it,
which the on-ramp cannot lower at translate time (the format engine parses constant formats only) — so
a non-constant / unsupported-conversion (`%a`) format fail-closes to a trap (present but traps if
called). `print` of numbers uses Lua's *constant* `%lld`/`%.14g` formats, which the on-ramp handles.
The `lua_fmt` fixture below makes `string.format` itself work by linking a **guest** runtime `snprintf`.

## Regenerating (stdlib)

With Lua 5.4.7 in `lua-5.4.7/`, clang 18, and this repo's guest sources, `-fno-vectorize
-fno-slp-vectorize` on all, `-fno-builtin` on the guest libm/shim:

```sh
NV="-fno-vectorize -fno-slp-vectorize"
CORE="lapi lcode lctype ldebug ldo ldump lfunc lgc llex lmem lobject lopcodes lparser \
      lstate lstring ltable ltm lundump lvm lzio"
LIBS="lbaselib lstrlib ltablib lmathlib lauxlib"
for f in $CORE $LIBS; do clang -O2 $NV -emit-llvm -c -Ilua-5.4.7/src lua-5.4.7/src/$f.c -o $f.bc; done
clang -O2 $NV              -emit-llvm -c -Ilua-5.4.7/src lua_stdlib_harness.c -o harness.bc
clang -O2 $NV -fno-builtin -emit-llvm -c -Ilua-5.4.7/src lua_stdlib_shim.c   -o guest_shim.bc
clang -O2 $NV -fno-builtin -emit-llvm -c .../demos/libm/libm.c               -o guest_libm.bc
llvm-link $CORE.bc $LIBS.bc harness.bc guest_shim.bc guest_libm.bc -o lua_stdlib.bc  # (expand globs)
```

# Lua string.format fixture

`lua_fmt.bc` is the same Lua 5.4.7 core + base/`string`/`table`/`math` libraries + guest libc shim +
guest libm + guest `strtod`, **plus a guest runtime `snprintf`** (`lua_fmt_snprintf.c`), running a
harness (`lua_fmt_harness.c`) whose script uses `string.format` heavily — width/precision/flags across
`%d`/`%x`/`%X`/`%#x`/`%o`, `%5d`/`%-5d`/`%05d`/`%+d`, `%10s`/`%-10s`/`%.3s`, `%c`, `%.2f`/`%8.3f`/
`%+.1f`, `%.3e`/`%g`/`%.10g`, `%q`, and a literal `%%`. It is the fixture for the `lua_fmt` test, which
asserts the exact **stdout bytes** on all three engines, identical to native `string.format`.

Lua's `str_format` parses each `%`-directive itself and calls `snprintf` **once per directive** with a
spec built at runtime — the path the on-ramp's translate-time constant-format engine cannot lower. The
guest `snprintf` (`lua_fmt_snprintf.c`) shadows that fail-closed trap: it formats integers/strings/chars
in C (matching glibc) and delegates floats to the on-ramp's correctly-rounded bignum dtoa via
`extern int __vm_fmt_{fix,sci,gen}(char *, double, int prec, int width, int flags)` — three vm-builtins
recognized in `lower_vm_builtin` that call the `dtoa_fix`/`dtoa_sci`/`dtoa_gen` helpers and `memcpy` the
result out. A single definition covers both the core's constant formats (`%lld`/`%.14g`) and
`string.format`'s runtime ones. (Known edge: `%f` of an extreme magnitude like `1e300` can differ from
native by one digit — a pre-existing `dtoa_fix` limit, not the format bridge; the script avoids it.)

## Regenerating (string.format)

Same recipe as the stdlib fixture, plus the guest `strtod` (float literals in the script) and the guest
`snprintf` (both `-fno-builtin` so clang doesn't rewrite them in terms of themselves):

```sh
NV="-fno-vectorize -fno-slp-vectorize"
CORE="lapi lcode lctype ldebug ldo ldump lfunc lgc llex lmem lobject lopcodes lparser \
      lstate lstring ltable ltm lundump lvm lzio"
LIBS="lbaselib lstrlib ltablib lmathlib lauxlib"
for f in $CORE $LIBS; do clang -O2 $NV -emit-llvm -c -Ilua-5.4.7/src lua-5.4.7/src/$f.c -o $f.bc; done
clang -O2 $NV              -emit-llvm -c -Ilua-5.4.7/src lua_fmt_harness.c   -o harness.bc
clang -O2 $NV -fno-builtin -emit-llvm -c -Ilua-5.4.7/src lua_stdlib_shim.c   -o guest_shim.bc
clang -O2 $NV -fno-builtin -emit-llvm -c                 lua_fmt_snprintf.c  -o guest_snprintf.bc
clang -O2 $NV -fno-builtin -emit-llvm -c .../demos/libm/libm.c               -o guest_libm.bc
clang -O2 $NV -fno-builtin -emit-llvm -c .../demos/strtod/strtod.c           -o guest_strtod.bc
llvm-link $CORE.bc $LIBS.bc harness.bc guest_shim.bc guest_snprintf.bc guest_libm.bc \
          guest_strtod.bc -o lua_fmt.bc   # (expand globs)
```

The guest `snprintf` **shadows** the on-ramp's `snprintf_rt` fail-closed trap; the `__vm_fmt_{fix,sci,
gen}` floats stay undefined and are recognized by the on-ramp. Pin `EXPECT` in `tests/lua_fmt.rs`
against a native build of the identical sources (guest `snprintf`/shim/strtod shadowing libc, `-lm`).
```

# Lua test-suite fixture

`lua_testsuite.bc` runs **three unmodified files from the official Lua 5.4.7 distribution's own test
suite** (`testes/vararg.lua`, `testes/bwcoercion.lua`, `testes/pm.lua`, embedded verbatim in
`lua_testsuite_tests.c`) through the whole VM with the base/`string`/`table`/`math`/`utf8` libraries
open. The harness (`lua_testsuite_harness.c`) loads each file as its own chunk under `pcall`, one fresh
`lua_State` per file; a Lua test signals failure by raising (an `assert`), so a clean **exit 0** means
every `assert` in all three files held — identical to native Lua. Test `tests/lua_testsuite.rs` asserts
`Returned([I32(0)])` on the tree-walker, bytecode, and JIT.

The three files were chosen because they are **self-contained** (no `require`/`os`/`io`/`debug`/
`coroutine`, no internal `T` test library): `vararg` covers `...`/`select`/`table.unpack`; `bwcoercion`
the string↔number bitwise coercions with `_ENV = nil`; `pm` the full pattern-matching engine
(`find`/`match`/`gmatch`/`gsub`, captures, anchors, `%b`, `%f`). Two translator/library fixes this
forced: the guest `strtod` now parses **hex floats** (`0x1.8p3`, Lua's hex-float literals — see
`demos/strtod`), and the on-ramp's `fcmp` lowering is now **NaN-correct** for ordered vs unordered
predicates (`emit_fcmp`), which Lua's `luaV_flttointeger` relies on. The guest shim adds real fdlibm
`asin`/`acos`/`atan`/`atan2`/`modf` (`lua_testsuite_trig.c`) that the base libm lacks.

Everything self-contained in the suite is now covered — including `debug` + the official
`coroutine.lua` (see below) and **io/os via `files.lua`** (see the files.lua fixture, riding the
configurable Fs capability). Only `main.lua` (tests the standalone `lua` binary itself) and the
`T`-library C-API tests remain out of scope.

# Lua files.lua fixture (io/os over the configurable Fs capability)

`lua_files.bc` runs the official **`testes/files.lua`** (embedded in `lua_files_tests.c`) with
base/`string`/`table`/`math`/`coroutine`/`debug` + **`io`** + **`os`** open. The io library rides a
real guest **stdio (FILE) layer** (`lua_files_stdio.c`) and the os library a guest **time/date
layer** (`lua_files_time.c`); `lua_files_shim.c` carries the usual non-stdio odds and ends
(derived math, `strerror`, `localeconv`, `setlocale`, `system`). The harness
(`lua_files_harness.c`) sets the suite's own portability knobs — `_port = true` (skips
`io.popen`/`os.execute`/OS-specific sections) and `_soft = true` (skips the huge-data stress) — the
documented configuration the suite itself honors, so the file runs **byte-for-byte unmodified**.

**The filesystem is a configurable capability, not ambient authority.** The guest stdio layer
resolves an embedder-granted capability by name (`__vm_cap_resolve("fs")` → §7 `cap.self.resolve`)
and drives a 7-op protocol (open/read/write/seek/close/remove/rename) through `__vm_host_call`
(§7 host-defined capability — the wasm-import analogue; both builtins were added for this). The
fixed §3e powerbox is untouched: a run has **no** filesystem unless the embedder injects one via
`svm_run::Instance::run_with_caps`. Two interchangeable backends live in `svm_run::fs`:

- `mem_fs()` — deterministic in-memory fs (fresh per run); `tests/lua_files.rs` asserts
  `Returned([I32(0)])` on **all three engines** against it.
- `host_fs(root)` — the **real** filesystem attenuated to a root directory (relative paths only;
  `..`/absolute refused by protocol on *both* backends). The same unmodified guest runs against a
  temp root, and the test asserts host-side that the root is left clean (files.lua removes
  everything it creates; `io.tmpfile` is created unlinked, POSIX-style).

The C probe `tests/fixtures/fs_probe.c` / `tests/fs_cap.rs` covers the raw capability protocol
(incl. attenuation refusals and real-disk assertions) independently of Lua.

Guest-layer details that files.lua actually observes: **setvbuf-honoring write buffering** (full by
default like glibc; the test watches full/line/none visibility through a second reader of the same
file), a write-only read failing with `ferror`+`errno` (not EOF), `ungetc` pushback vs. seek/tell,
and a proleptic-Gregorian `gmtime`/`mktime`/`strftime` (UTC, C locale) whose round-trips
(`os.time(os.date("*t", t)) == t`) hold exactly. `time()` is a fixed synthetic epoch (no ambient
clock; the suite only needs internal consistency). `os.getenv` rides the synthesized env-blob
`getenv` (the test seeds `PATH`).

## Regenerating (files.lua)

Same as the test-suite fixture below, but `LIBS` adds `lcorolib ldblib liolib loslib`, the harness is
`lua_files_harness.c`, the tests embed is `lua_files_tests.c`, and the guest layers
`lua_files_stdio.c` + `lua_files_time.c` + `lua_files_shim.c` replace `lua_testsuite_shim.c` (keep
trig/snprintf/libm/strtod). The native oracle builds the same core+harness against **real libc**
(no guest layers) and runs in a scratch directory.

# Lua coroutine fixture

`lua_coroutine.bc` runs an **in-house coroutine differential** (`lua_coroutine.lua`, kept readable in
this directory and embedded verbatim as bytes into `lua_coroutine_tests.c`) with
base/`string`/`table`/`math`/`coroutine` open. It exercises the whole coroutine surface that does *not*
need the `debug` library: `create`/`resume`/`yield` with multi-value transfer both directions, the
`suspended`/`running`/`normal`/`dead` status transitions, `running`/`isyieldable` in the main thread
vs. inside a coroutine, `wrap` (incl. error re-raise), error propagation out of `resume` (string and
non-string error values), **yield across `pcall` and `xpcall`** (the yieldable-pcall / continuation
machinery), `coroutine.close` with `<close>` to-be-closed variables, and a producer/filter/consumer
pipeline. Test `tests/lua_coroutine.rs` asserts `Returned([I32(0)])` on all three engines; the same
harness+file built natively also exits 0 (the differential oracle).

**Why this needed no new machinery.** Lua 5.4 coroutines are *stackless* with respect to the C stack:
each coroutine is a `lua_State` with its own heap-allocated Lua stack, and resume/yield ride the same
`luaD_rawrunprotected` / `luaD_throw` (setjmp/longjmp) primitive `pcall` already uses (ldo.c) — there is
no `swapcontext`/`ucontext`/assembly anywhere in Lua's core. So the on-ramp's existing `SetJmp`/`LongJmp`
core ops (proven by every working `pcall`) carry coroutines too; **no fiber or native-stack switching is
involved**, and no translator or libc change was needed. The harness (`lua_coroutine_harness.c`) opens
`luaopen_coroutine` and reuses the minimal `require` from the utf8 fixture. The official
`testes/coroutine.lua` additionally hard-requires the `debug` library (hooks, `getinfo`,
`getlocal`/`setlocal` on suspended coroutines, `traceback`), which is the next slice.

## Regenerating (coroutine)

Edit `lua_coroutine.lua`, re-embed it into `lua_coroutine_tests.c` (same byte-array format as the other
`*_tests.c`), then build like the test-suite fixture below but with `lcorolib` in `LIBS` (no `lutf8lib`)
and `lua_coroutine_harness.c`. Validate the `.lua` against native Lua first (it is the oracle).

# Lua official coroutine.lua fixture

`lua_coroutine_official.bc` runs the **unmodified official `testes/coroutine.lua`** (embedded in
`lua_coroutine_official_tests.c`) with base/`string`/`table`/`math`/`coroutine`/`debug` open. Standalone
the internal `T` C-test library is absent, so the file's own `if not T`/`if T==nil` guards skip the
C-API sections; what remains still drives the coroutine + **debug** libraries hard: yields inside every
metamethod and inside `for` iterators, `coroutine.close` with `<close>` variables, C-stack-overflow
detection, and `debug.getinfo`/`getlocal`/`setlocal`/`setupvalue`/`sethook`/`traceback` (incl. debug on
a *suspended* coroutine). Test `tests/lua_coroutine_official.rs` asserts `Returned([I32(0)])` on all
three engines; the same harness+file built natively also exits 0 (the differential oracle). The harness
(`lua_coroutine_official_harness.c`) opens `luaopen_debug` alongside `luaopen_coroutine`;
`lua_coroutine_official_shim.c` adds the one libc gap the debug lib needs — `fgets`, referenced only by
`ldblib.c`'s interactive `debug.debug()`, which `coroutine.lua` never calls.

**One reference-oracle change this forced.** No translator, coroutine, or debug change was needed, but
the file's *"infinite recursion of coroutines"* case (`a = function(a) coroutine.wrap(a)(a) end;
assert(not pcall(a, a))`) probes Lua's own C-stack-overflow detection: it must raise a `pcall`-catchable
"C stack overflow" via `LUAI_MAXCCALLS`. The production engines (bytecode, JIT) reach that self-limit,
but the tree-walker reference oracle previously capped its reified call stack at `MAX_CALL_DEPTH = 256`
and tripped first as an uncatchable §5 kill. Raising the cap to `2048` (still well under the durable
shadow-reserve frame budget) lets the oracle observe the same catchable error the real engines do — see
`svm_interp::MAX_CALL_DEPTH`. Verified regression-free (durable + interp + `jit_diff` suites green).

## Regenerating (official coroutine.lua)

Same as the test-suite fixture below, but with `lcorolib` **and** `ldblib` in `LIBS`,
`lua_coroutine_official_harness.c` (opens coroutine + debug), and the `fgets` stub
`lua_coroutine_official_shim.c` linked in. Validate against native Lua first (it is the oracle).

# Lua utf8.lua fixture

`lua_utf8.bc` runs the official **`testes/utf8.lua`** (embedded in `lua_utf8_tests.c`) through the whole
VM with base/`string`/`table`/`math`/`utf8` open. It is the full `utf8` library workout:
`utf8.char`/`codepoint`/`len`/`offset`/`codes`/`charpattern`, strict vs. `nonstrict` decoding across
every sequence size (1–6 bytes, up to the original-UTF-8 `0x7FFFFFFF`), surrogate and overlong
rejection, `utf8.len` error positions, `utf8.codes` iteration errors, the `\u{…}` string escapes
(round-tripped through `load`), and `string.gmatch(s, utf8.charpattern)`. Test `tests/lua_utf8.rs`
asserts `Returned([I32(0)])` on all three engines; the same harness+file built natively also exits 0
(the differential oracle).

This is the first fixture to need **`require`**: `utf8.lua` opens with `local utf8 = require'utf8'`.
Rather than compile stock `loadlib.c` — whose file and C-library searchers need a filesystem/dynamic
loader the on-ramp does not have, and can never run for a preloaded module — the harness
(`lua_utf8_harness.c`) installs a **minimal `require`** that returns `_LOADED[name]` (where
`luaL_requiref` records every opened library) and raises otherwise. That is exactly stock `require`'s
fast path for an already-loaded module, so `utf8.lua` runs unmodified. No translator or libc change was
needed — the earlier `fcmp`/narrow-signed-op/hex-`strtod` fixes plus the existing shim cover it.

## Regenerating (utf8.lua)

Same as the test-suite fixture below, but with `lua_utf8_harness.c` (adds the `require`) and
`lua_utf8_tests.c` (embeds only `utf8.lua`, count 1); `lutf8lib` stays in `LIBS`.

# Lua math.lua fixture

`lua_math.bc` runs the official **`testes/math.lua`** (embedded in `lua_math_tests.c`) — the densest
single file in the suite — through the whole VM with base/`string`/`table`/`math` open, via the same
harness. It exercises integer/float arithmetic and conversions, `//`/`%`, float↔integer order (every
NaN corner), `math.type`/`tointeger`/`floor`/`ceil`/`fmod`/`ult`/`min`/`max`, the transcendentals,
`math.modf`, `string.format` number formatting, decimal **and hex** float literals (incl. a 1000-digit
fraction), and the `math.random` distribution tests. Test `tests/lua_math.rs` asserts
`Returned([I32(0)])` on all three engines (JIT ~1 min — the module is large).

Getting `math.lua` fully green drove two on-ramp fixes beyond the ones above:
- **Sign-extended narrow signed ops.** A `<i32` value loaded from memory is *zero-extended*
  (canonical), so its sign bit is buried at bit `N-1`; the on-ramp's `ashr`/`sdiv`/`srem` on an
  `i8`/`i16` now sign-extend the operand first. Previously `ashr i8 0x80,7` gave `+1` (should be `-1`)
  — and since Lua's `testMMMode` compiles to exactly `ashr i8 luaP_opmodes[op],7`, `findsetreg` skipped
  the wrong instruction and `getobjname` dropped the operand name in error messages (`number (field
  'huge') has no integer representation` → `number has …`). See `narrow_signed_shift_div_rem` in
  `tests/translate.rs`.
- **Hex `strtod` leading zeros.** A hex fraction's leading zeros no longer consume the
  significant-digit budget, so `0x.000…0074p4004` (1000 zeros) parses correctly.

## Regenerating (math.lua)

Same as the test-suite fixture below, but `lua_math_tests.c` embeds only `math.lua` (as
`lua_tests`/`lua_test_lens`/`lua_test_names`/`lua_test_count`, count 1), and no `lutf8lib` is needed.

## Regenerating (test suite)

Download the official tests (`curl -O https://www.lua.org/tests/lua-5.4.7-tests.tar.gz`), then embed the
chosen files as a C byte-array (`lua_testsuite_tests.c` exports `lua_tests`/`lua_test_lens`/
`lua_test_names`/`lua_test_count`) and build like the `string.format` fixture, adding `lutf8lib` and the
fdlibm trig:

```sh
NV="-fno-vectorize -fno-slp-vectorize"
CORE="lapi lcode lctype ldebug ldo ldump lfunc lgc llex lmem lobject lopcodes lparser \
      lstate lstring ltable ltm lundump lvm lzio"
LIBS="lbaselib lstrlib ltablib lmathlib lauxlib lutf8lib"
for f in $CORE $LIBS; do clang -O2 $NV -emit-llvm -c -Ilua-5.4.7/src lua-5.4.7/src/$f.c -o $f.bc; done
clang -O2 $NV              -emit-llvm -c -Ilua-5.4.7/src lua_testsuite_harness.c -o harness.bc
clang -O2 $NV              -emit-llvm -c -Ilua-5.4.7/src lua_testsuite_tests.c   -o tests.bc
clang -O2 $NV -fno-builtin -emit-llvm -c -Ilua-5.4.7/src lua_testsuite_shim.c    -o guest_shim.bc
clang -O2 $NV -fno-builtin -fno-strict-aliasing -emit-llvm -c lua_testsuite_trig.c -o guest_trig.bc
clang -O2 $NV -fno-builtin -emit-llvm -c                 lua_fmt_snprintf.c      -o guest_snprintf.bc
clang -O2 $NV -fno-builtin -emit-llvm -c .../demos/libm/libm.c                   -o guest_libm.bc
clang -O2 $NV -fno-builtin -emit-llvm -c .../demos/strtod/strtod.c              -o guest_strtod.bc
llvm-link $CORE.bc $LIBS.bc harness.bc tests.bc guest_shim.bc guest_trig.bc guest_snprintf.bc \
          guest_libm.bc guest_strtod.bc -o lua_testsuite.bc   # (expand globs)
```

`lua_testsuite_shim.c` keeps a 48 MiB arena (the JIT window bound) and resets it between files. The
`lua_testsuite_trig.c` word-access macros type-pun through `int` pointers, so it needs
`-fno-strict-aliasing` (exactly as fdlibm does natively).
```


# Lua suite-sweep fixture (21 more official files)

`lua_sweep.bc` runs **twenty-one more unmodified official test files** in one bundle — light-to-heavy:
`tracegc`, `verybig`, `big`, `gengc`, `goto`, `events`, `code`, `bitwise`, `closure`, `tpack`,
`literals`, `errors`, `nextvar`, `sort`, `db`, `constructs`, `locals`, `cstack`, `strings`, `gc`,
`calls` — each as its own chunk in a fresh `lua_State` under `pcall`, with every guest-available
library open. Test `tests/lua_sweep.rs` asserts `Returned([I32(0)])`; the JIT run gates CI, the
interpreter runs are `#[ignore]`d full-depth gates (long, like the extended fuzz). The same
harness+bundle built natively also exits 0 (the differential oracle). With this, **28 of the suite's
33 files run** — excluded are `api`/`main`/`all` (need the internal `T` C-test library / the
standalone `lua` binary), `attrib` (the real `package` searchers), and `heavy` (deliberate memory
exhaustion).

The harness (`lua_sweep_harness.c`) extends the files.lua one with what the wider suite needs:
- a **real allocator** — segregated power-of-two free lists over the 48 MiB static arena (the
  bump-only arena exhausted under `gc.lua`'s collector stress; freed blocks now recycle), reset per
  file. 48 MiB keeps the guest window inside the reference JIT's 64 MiB cap; to make `locals.lua`'s
  stack-overflow-with-`<close>` test fit (a 1M-slot Lua stack costs ~190 B/frame ≈ 93 MiB at
  overflow), the bundle's Lua is built with **`LUAI_MAXSTACK = 250000`** — an edit to `luaconf.h`,
  Lua's own embedder porting header (the tests only need *a* ceiling to overflow, not that specific
  one; native uses the identical header, so the differential is symmetric);
- **sibling-module `require`**: suite files require each other (`bitwise` → `bwcoercion`,
  `cstack`/`locals` → `tracegc`); the embedded module sources are pre-run into the registry `_LOADED`
  table, exactly where stock `require` caches them — plus the stock **preload searcher**
  (`package.preload[name]`, how `bitwise.lua` installs its `bit32` shim);
- a faithful **`package` global** (`loaded` = `_LOADED`, `preload` = `_PRELOAD`, and
  `_LOADED["package"]` set — `nextvar.lua`'s global sweep erases any global not in `package.loaded`);
- **`@name.lua` chunknames** (how `dofile` names file chunks) — `db.lua` asserts
  `debug.getinfo(...).source` matches `^@.*db%.lua$`.

**One real translator bug and three guest-snprintf gaps fell out** (each caught by `strings.lua`,
each verified against native):
- **`BIG_NLIMBS` 40 → 48.** The bignum float formatter's big integers were sized for a double's exact
  value (`2^1074` ≈ 34 limbs) — but a fixed format also scales by `10^prec`, and `%.99f` of a
  near-maximum double reaches `2^1023 · 10^99 ≈ 2^1352` (43 limbs): the 40-limb ceiling **silently
  truncated the digits** (388 chars instead of 410). 48 limbs (1536 bits) covers every finite double
  at C-cap precision; the scratch layout shifted accordingly (`FMT_*_O`, `FLOAT_SCRATCH_SIZE`).
- Guest `snprintf`: `%p` honors width/`-`; `%a`/`%A` hex-floats exist (exact trailing-zero-trimmed
  form for Lua's `%q` float round-trip, plus precision with round-half-to-even and carry — the
  `%+.2A` modifier checks); integer conversions print nothing for a zero value at zero precision
  (ISO); float conversions apply `0`-padding, `#` (point at precision 0), and field width in the
  guest layer (the bignum helpers now produce unpadded content).

## Regenerating (sweep)

Like the files.lua fixture but with `lua_sweep_harness.c`, and `lua_sweep_tests.c` generated from the
21 files (in the order above) plus `tracegc.lua` + `bwcoercion.lua` as the embedded sibling modules
(the `lua_modules` arrays). Build from a **copy of the Lua sources with `luaconf.h` edited** to
`LUAI_MAXSTACK 250000` — note that `-DLUAI_MAXSTACK` does *not* work (luaconf.h redefines it
unconditionally) and `-include`-ing an edited copy silently breaks (`l_likely` is `LUA_CORE`-gated);
the quoted `#include "luaconf.h"` resolves against the source directory, so the edited copy must sit
next to the `.c` files. Native oracle: same source copy against real libc.
