/* Runs a set of unmodified official Lua 5.4.7 test files (`testes/*.lua`, embedded as bytes in
 * `lua_testsuite_tests.c`) with the base/string/table/math/utf8 libraries open — one fresh `lua_State`
 * per file so the bump arena is bounded per test. Each file is loaded as its own chunk and run under
 * `pcall`; a Lua test signals failure by raising (an `assert`), which `pcall` catches. Returns 0 iff
 * every file completes with no assertion firing; otherwise the 1-based index of the first failing
 * file (and the file name + error go to stdout so a failure is legible). */
#include "lua.h"
#include "lauxlib.h"
#include "lualib.h"

extern const unsigned char *const lua_tests[];
extern const unsigned int lua_test_lens[];
extern const char *const lua_test_names[];
extern const unsigned int lua_test_count;

/* Bump arena — the on-ramp powerbox has no OS allocator, so the guest brings its own. Freed blocks
 * are not reclaimed (Lua's GC still runs, but `free` is a no-op here), so it is sized for the heaviest
 * single file and reset by tearing down the `lua_State` between files. */
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

static int run_one(const unsigned char *src, unsigned len, const char *name) {
  arena_off = 0; /* reset the bump arena for a fresh, isolated run */
  lua_State *L = lua_newstate(l_alloc, (void *)0);
  if (!L) return 1;
  static const luaL_Reg libs[] = {
    {LUA_GNAME, luaopen_base},   {LUA_STRLIBNAME, luaopen_string},
    {LUA_TABLIBNAME, luaopen_table}, {LUA_MATHLIBNAME, luaopen_math},
    {LUA_UTF8LIBNAME, luaopen_utf8}, {(void *)0, (void *)0},
  };
  for (const luaL_Reg *lib = libs; lib->func; lib++) {
    luaL_requiref(L, lib->name, lib->func, 1);
    lua_pop(L, 1);
  }
  lua_newtable(L); /* some files reference a global `arg` */
  lua_setglobal(L, "arg");

  /* One constant format string for every outcome: distinct `printf` formats get tail-merged at -O2
   * into a single call with a PHI'd (non-constant) format pointer, which the on-ramp cannot lower. */
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
