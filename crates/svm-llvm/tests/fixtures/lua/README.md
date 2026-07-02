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

