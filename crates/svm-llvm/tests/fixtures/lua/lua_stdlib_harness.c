/* Lua stdlib harness: open base+string+table+math, run a script that prints. Output goes to stdout
 * via lua_writestring -> fwrite -> Stream.write. Arena allocator (no host malloc). Avoids
 * string.format (which builds a runtime format string -> runtime-format snprintf, not yet supported);
 * print of numbers uses Lua's constant %lld/%.14g formats, which the on-ramp handles. */
#include "lua.h"
#include "lauxlib.h"
#include "lualib.h"

static char arena[48 * 1024 * 1024];
static unsigned long arena_off = 0;
static void *l_alloc(void *ud, void *ptr, size_t osize, size_t nsize) {
  (void)ud;
  if (nsize == 0) return (void *)0;
  unsigned long start = (arena_off + 15UL) & ~15UL;
  if (start + nsize > sizeof arena) return (void *)0;
  char *np = &arena[start]; arena_off = start + nsize;
  if (ptr && osize) { size_t n = osize<nsize?osize:nsize; for (size_t i=0;i<n;i++) np[i]=((char*)ptr)[i]; }
  return np;
}
static const char SCRIPT[] =
  "print('hello from lua on the on-ramp')\n"
  "print('2 + 3 =', 2 + 3)\n"
  "print('upper', string.upper('lua'), 'rep', string.rep('ab', 3))\n"
  "print('sub', string.sub('hello world', 1, 5), 'len', #'hello world')\n"
  "local t = {3, 1, 4, 1, 5, 9, 2, 6}\n"
  "table.sort(t)\n"
  "print('sorted', table.concat(t, ','))\n"
  "table.insert(t, 7); table.remove(t, 1)\n"
  "print('after ins/rem', #t, t[#t])\n"
  "print('sqrt2', math.sqrt(2))\n"
  "print('pi', math.pi)\n"
  "print('floor', math.floor(3.7), 'max', math.max(1, 5, 3), 'abs', math.abs(-42))\n"
  "local s = 0\n"
  "for i, v in ipairs({10, 20, 30, 40}) do s = s + v end\n"
  "print('ipairs sum', s)\n"
  "local kv = 0\n"
  "for k, v in pairs({a=1, b=2, c=3}) do kv = kv + v end\n"
  "print('pairs sum', kv, 'type', type(kv), 'tostring', tostring(true))\n";
int main(void) {
  lua_State *L = lua_newstate(l_alloc, (void *)0);
  if (!L) return 1;
  static const luaL_Reg libs[] = {
    {LUA_GNAME, luaopen_base}, {LUA_STRLIBNAME, luaopen_string},
    {LUA_TABLIBNAME, luaopen_table}, {LUA_MATHLIBNAME, luaopen_math}, {(void*)0, (void*)0},
  };
  for (const luaL_Reg *lib = libs; lib->func; lib++) { luaL_requiref(L, lib->name, lib->func, 1); lua_pop(L, 1); }
  if (luaL_dostring(L, SCRIPT) != LUA_OK) return 2;
  lua_close(L);
  return 0;
}
