#include "lua.h"
#include "lauxlib.h"
#include "lualib.h"
static char arena[48 * 1024 * 1024];
static unsigned long arena_off = 0;
static void *l_alloc(void *ud, void *ptr, size_t osize, size_t nsize) {
  (void)ud; if (nsize == 0) return (void *)0;
  unsigned long start = (arena_off + 15UL) & ~15UL;
  if (start + nsize > sizeof arena) return (void *)0;
  char *np = &arena[start]; arena_off = start + nsize;
  if (ptr && osize) { size_t n = osize<nsize?osize:nsize; for (size_t i=0;i<n;i++) np[i]=((char*)ptr)[i]; }
  return np;
}
static const char SCRIPT[] =
  "print(string.format('%d + %d = %d', 2, 3, 2+3))\n"
  "print(string.format('[%5d][%-5d][%05d][%+d]', 42, 42, 42, 42))\n"
  "print(string.format('hex %x %X %#x oct %o', 255, 255, 255, 64))\n"
  "print(string.format('str [%10s][%-10s][%.3s]', 'hi', 'hi', 'hello'))\n"
  "print(string.format('char %c%c%c', 76, 117, 97))\n"
  "print(string.format('float %.2f %8.3f %+.1f', 3.14159, 2.5, 7.0))\n"
  "print(string.format('sci %.3e general %g %.10g', 123456.789, 0.0001, math.pi))\n"
  "print(string.format('%q', [[he said \"hi\"]]))\n"
  "print(string.format('%d items, %.1f%% done', 7, 87.5))\n"
  "for i = 1, 3 do print(string.format('  row %d: %s = %d', i, 'x', i*i)) end\n";
int main(void) {
  lua_State *L = lua_newstate(l_alloc, (void *)0);
  if (!L) return 1;
  static const luaL_Reg libs[] = {
    {LUA_GNAME, luaopen_base}, {LUA_STRLIBNAME, luaopen_string},
    {LUA_TABLIBNAME, luaopen_table}, {LUA_MATHLIBNAME, luaopen_math}, {(void*)0, (void*)0},
  };
  for (const luaL_Reg *lib = libs; lib->func; lib++) { luaL_requiref(L, lib->name, lib->func, 1); lua_pop(L, 1); }
  if (luaL_dostring(L, SCRIPT) != LUA_OK) { return 2; }
  lua_close(L);
  return 0;
}
