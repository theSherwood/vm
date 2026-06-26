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
