/* Runs the in-house `coroutine.lua` differential (embedded as bytes in `lua_coroutine_tests.c`) with
 * the base/string/table/math/coroutine libraries open. Same shape as `lua_utf8_harness.c` — one fresh
 * `lua_State`, a bump arena for the guest allocator, a minimal `require` over the preloaded modules —
 * but opens `luaopen_coroutine` instead of `luaopen_utf8`. Lua 5.4 coroutines are stackless with
 * respect to the C stack: each coroutine is a `lua_State` with its own heap Lua stack, and resume/yield
 * ride the same `luaD_rawrunprotected`/`luaD_throw` (setjmp/longjmp) primitive `pcall` already uses — so
 * no fiber or C-stack switching is involved, and the on-ramp needs nothing beyond its existing
 * setjmp/longjmp support. */
#include "lua.h"
#include "lauxlib.h"
#include "lualib.h"

extern const unsigned char *const lua_tests[];
extern const unsigned int lua_test_lens[];
extern const char *const lua_test_names[];
extern const unsigned int lua_test_count;

/* Bump arena — the on-ramp powerbox has no OS allocator, so the guest brings its own. Freed blocks are
 * not reclaimed; sized for the file and reset by tearing down the `lua_State`. */
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
  lua_getfield(L, -1, name); /* _LOADED[name] */
  if (lua_toboolean(L, -1)) return 1;
  return luaL_error(L, "module '%s' not found", name);
}

static int run_one(const unsigned char *src, unsigned len, const char *name) {
  arena_off = 0; /* reset the bump arena for a fresh, isolated run */
  lua_State *L = lua_newstate(l_alloc, (void *)0);
  if (!L) return 1;
  static const luaL_Reg libs[] = {
    {LUA_GNAME, luaopen_base},   {LUA_STRLIBNAME, luaopen_string},
    {LUA_TABLIBNAME, luaopen_table}, {LUA_MATHLIBNAME, luaopen_math},
    {LUA_COLIBNAME, luaopen_coroutine}, {(void *)0, (void *)0},
  };
  for (const luaL_Reg *lib = libs; lib->func; lib++) {
    luaL_requiref(L, lib->name, lib->func, 1);
    lua_pop(L, 1);
  }
  lua_register(L, "require", l_require); /* global `require` over the preloaded modules */
  lua_newtable(L);                       /* some files reference a global `arg` */
  lua_setglobal(L, "arg");

  /* One constant format string for every outcome: distinct `printf` formats get tail-merged at -O2 into
   * a single call with a PHI'd (non-constant) format pointer, which the on-ramp cannot lower. */
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
      return (int)(i + 1); /* 1-based index of the first failing file */
  }
  return 0;
}
