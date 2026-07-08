/* Runs the **unmodified official** Lua 5.4.7 `testes/coroutine.lua` (embedded as bytes in
 * `lua_coroutine_official_tests.c`) with the base/string/table/math/coroutine/debug libraries open. Same
 * shape as the other Lua harnesses — one fresh `lua_State`, a bump arena for the guest allocator, and a
 * minimal `require` over the preloaded modules. `coroutine.lua` opens with `require'coroutine'` and
 * `require'debug'`; run standalone the internal `T` C-test library is absent, so the file's own
 * `if not T`/`if T==nil` guards skip the C-API sections and it exercises pure coroutine + debug behavior
 * (yields inside metamethods and `for` iterators, `coroutine.close`, `<close>` variables, stack-overflow
 * detection, and debug `getinfo`/`getlocal`/`setlocal`/`setupvalue`/`sethook`/`traceback`). */
#include "lua.h"
#include "lauxlib.h"
#include "lualib.h"

extern const unsigned char *const lua_tests[];
extern const unsigned int lua_test_lens[];
extern const char *const lua_test_names[];
extern const unsigned int lua_test_count;

static char arena[48 * 1024 * 1024];
static unsigned long arena_off = 0;
static void *l_alloc(void *ud, void *ptr, size_t osize, size_t nsize) {
  (void)ud;
  if (nsize == 0) return (void *)0;
  unsigned long start = (arena_off + 15UL) & ~15UL;
  if (start + nsize > sizeof arena) return (void *)0;
  char *np = &arena[start];
  arena_off = start + nsize;
  if (ptr && osize) {
    size_t n = osize < nsize ? osize : nsize;
    for (size_t i = 0; i < n; i++) np[i] = ((char *)ptr)[i];
  }
  return np;
}

/* Minimal `require`: return `_LOADED[name]` if the module is already open (every library opened via
 * `luaL_requiref` is recorded there), else raise like stock Lua. No file/C searchers — those need a
 * filesystem/dynamic loader the on-ramp does not provide, and are never reachable for a preloaded lib. */
static int l_require(lua_State *L) {
  const char *name = luaL_checkstring(L, 1);
  luaL_getsubtable(L, LUA_REGISTRYINDEX, LUA_LOADED_TABLE);
  lua_getfield(L, -1, name);
  if (lua_toboolean(L, -1)) return 1;
  return luaL_error(L, "module '%s' not found", name);
}

static int run_one(const unsigned char *src, unsigned len, const char *name) {
  arena_off = 0;
  lua_State *L = lua_newstate(l_alloc, (void *)0);
  if (!L) return 1;
  static const luaL_Reg libs[] = {
    {LUA_GNAME, luaopen_base},   {LUA_STRLIBNAME, luaopen_string},
    {LUA_TABLIBNAME, luaopen_table}, {LUA_MATHLIBNAME, luaopen_math},
    {LUA_COLIBNAME, luaopen_coroutine}, {LUA_DBLIBNAME, luaopen_debug},
    {(void *)0, (void *)0},
  };
  for (const luaL_Reg *lib = libs; lib->func; lib++) {
    luaL_requiref(L, lib->name, lib->func, 1);
    lua_pop(L, 1);
  }
  lua_register(L, "require", l_require);
  lua_newtable(L);
  lua_setglobal(L, "arg");

  int rc = 0;
  const char *msg;
  if (luaL_loadbuffer(L, (const char *)src, len, name) != LUA_OK) {
    msg = lua_tostring(L, -1);
    rc = 2;
  } else if (lua_pcall(L, 0, 0, 0) != LUA_OK) {
    msg = lua_tostring(L, -1);
    rc = 3;
  } else {
    msg = "ok";
  }
  printf("%s: %s\n", name, msg ? msg : "?");
  lua_close(L);
  return rc;
}

int main(void) {
  for (unsigned i = 0; i < lua_test_count; i++) {
    if (run_one(lua_tests[i], lua_test_lens[i], lua_test_names[i]) != 0)
      return (int)(i + 1);
  }
  return 0;
}
