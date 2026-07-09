/* Runs official Lua 5.4.7 test files with the **real** `package` library (`loadlib.c`'s
 * `luaopen_package`) over the configurable Fs capability — the endgame harness. Same T-library
 * config as `lua_tlib_harness.c` (whole core `-DLUA_USER_H='"ltests.h"'`, `debug_realloc` allocator,
 * one shared `lua_State`, the official `all.lua` model), but with three changes that retire the
 * minimal-`require` shim in favor of stock `require`:
 *
 * - `luaopen_package` is opened like any other library, so the guest gets the **stock** global
 *   `require`, `package.path`/`cpath`/`config`/`searchpath`/`loadlib`, and the real searcher chain
 *   (preload → Lua-file). The Lua-file searcher's `readable`/`luaL_loadfilex` run over the Fs
 *   capability (`fopen`/`getc`/`fread`), so `require "mod"` genuinely searches `package.path` and
 *   loads `mod.lua` from the (in-memory) filesystem. The C-library searcher is the ANSI `LUA_DL_NONE`
 *   stub — `package.loadlib` returns `(nil, msg, "absent")`, exactly the "cannot load dynamic
 *   library" branch the suite guards for.
 * - The embedded module files (`lua_modules`) are **seeded into the Fs** (via `fopen`+`fwrite`), not
 *   pre-run into `_LOADED`: the sibling modules the suite `require`s (and, for `all.lua`, the whole
 *   `testes/` tree) live on the (in-memory) disk where stock `require`/`dofile` find them.
 * - `package.path` is pointed at the seeded layout (`?.lua;libs/?.lua`).
 *
 * `_port`/`_soft` still gate the process-spawning / huge-data sections. */
#include <stdio.h> /* FILE/fopen/fwrite/fclose — provided at link by the guest stdio layer */
#include "lua.h"
#include "lauxlib.h"
#include "lualib.h"
#include "ltests.h"

extern const unsigned char *const lua_tests[];
extern const unsigned int lua_test_lens[];
extern const char *const lua_test_names[];
extern const unsigned int lua_test_count;
/* Files seeded onto the (in-memory) disk before the run — sibling modules / the whole suite tree. */
extern const unsigned char *const lua_modules[];
extern const unsigned int lua_module_lens[];
extern const char *const lua_module_names[];
extern const unsigned int lua_module_count;

static lua_State *GL;

/* Seed one file's bytes onto the Fs. Directory components (`libs/foo.lua`) are fine: the MemFs keys
 * on the whole path, so no directory entry is needed. Returns 0 on success. */
static int seed_file(const char *name, const unsigned char *bytes, unsigned len) {
  FILE *f = fopen(name, "wb");
  if (!f) return 1;
  if (len && fwrite(bytes, 1, len, f) != len) {
    fclose(f);
    return 2;
  }
  return fclose(f) == 0 ? 0 : 3;
}

static int run_one(const unsigned char *src, unsigned len, const char *name) {
  int rc = 0;
  const char *msg = "ok";
  /* Skip a leading `#!...` shebang line, as `luaL_loadfilex` does for a file chunk (`all.lua`
   * starts with `#!../lua`); `luaL_loadbuffer` alone would choke on the `#`. */
  if (len > 0 && src[0] == '#') {
    unsigned k = 0;
    while (k < len && src[k] != '\n') k++;
    src += k;
    len -= k;
  }
  if (luaL_loadbuffer(GL, (const char *)src, len, name) != LUA_OK) {
    msg = lua_tostring(GL, -1);
    rc = 2;
  } else if (lua_pcall(GL, 0, 0, 0) != LUA_OK) {
    msg = lua_tostring(GL, -1);
    rc = 3;
  }
  printf("%s: %s\n", name, msg ? msg : "?");
  return rc;
}

int main(void) {
  setvbuf(stdout, (void *)0, _IONBF, 0); /* unbuffered: survive ltests' atexit abort natively (the
                                          * guest stdio layer is unbuffered regardless) */
  lua_State *L = lua_newstate(debug_realloc, &l_memcontrol);
  if (!L) return 99;
  GL = L;
  static const luaL_Reg libs[] = {
    {LUA_GNAME, luaopen_base},          {LUA_LOADLIBNAME, luaopen_package},
    {LUA_STRLIBNAME, luaopen_string},   {LUA_TABLIBNAME, luaopen_table},
    {LUA_MATHLIBNAME, luaopen_math},    {LUA_COLIBNAME, luaopen_coroutine},
    {LUA_DBLIBNAME, luaopen_debug},     {LUA_IOLIBNAME, luaopen_io},
    {LUA_OSLIBNAME, luaopen_os},        {LUA_UTF8LIBNAME, luaopen_utf8},
    {(void *)0, (void *)0},
  };
  for (const luaL_Reg *lib = libs; lib->func; lib++) {
    luaL_requiref(L, lib->name, lib->func, 1);
    lua_pop(L, 1);
  }
  luaL_requiref(L, "T", luaB_opentests, 1);
  lua_pop(L, 1);

  /* Point `package.path` at the seeded layout; drop `cpath` to the ANSI stub's default (no C libs). */
  lua_getglobal(L, "package");
  lua_pushstring(L, "?.lua;libs/?.lua;libs/?.lc;libs/?");
  lua_setfield(L, -2, "path");
  lua_pop(L, 1);

  lua_newtable(L);
  lua_setglobal(L, "arg");
  lua_pushboolean(L, 1);
  lua_setglobal(L, "_port"); /* skip non-portable (process-spawning) sections — main.lua early-returns */
  lua_pushboolean(L, 1);
  lua_setglobal(L, "_soft"); /* skip long/huge-memory sections (big.lua, etc.) */
  lua_pushboolean(L, 1);
  lua_setglobal(L, "_nomsg"); /* quiet the "tests not performed" bookkeeping (all.lua) */

  /* Seed the sibling-module / suite files onto the (in-memory) disk. */
  for (unsigned m = 0; m < lua_module_count; m++) {
    if (seed_file(lua_module_names[m], lua_modules[m], lua_module_lens[m]) != 0) {
      printf("seed failed: %s\n", lua_module_names[m]);
      return 97;
    }
  }

  for (unsigned i = 0; i < lua_test_count; i++) {
    if (run_one(lua_tests[i], lua_test_lens[i], lua_test_names[i]) != 0)
      return (int)(i + 1);
  }
  lua_close(L);
  return 0;
}
