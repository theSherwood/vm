/* Runs unmodified official Lua 5.4.7 test files with the internal **T library** (`ltests.c`) active:
 * the whole core is compiled with `-DLUA_USER_H='"ltests.h"'` (internal assertions, the tracking
 * `debug_realloc` allocator with failure injection, small debug sizes, `LUAI_MAXSTACK 50000`), the
 * state is created exactly as ltests.h's own `lua_c` recipe does — `lua_newstate(debug_realloc,
 * &l_memcontrol)` — and `T` is opened via `luaL_requiref("T", luaB_opentests)`. `debug_realloc`
 * sits on plain `malloc`/`free`, provided by the guest free-list allocator (`lua_tlib_malloc.c`).
 * Everything else matches `lua_sweep_harness.c` (sibling-module require + preload searcher,
 * faithful `package`, `@` chunknames, `_port`/`_soft`) — except the execution model: **one shared
 * `lua_State` runs every file in `all.lua`'s order**, exactly the official driver's semantics.
 * (ltests' `warnf` keeps its mode in process statics; api.lua's warning tests depend on that state
 * lining up with the `_WARN` global, which only holds when state lifetime == process lifetime, as
 * in the real suite.)
 *
 * - A **real allocator** (segregated free lists over the static arena) instead of the bump-only
 *   arena: churn-heavy files (`gc.lua`'s collector stress, `locals.lua`, `nextvar.lua`) genuinely
 *   free, so a no-free arena exhausts. Deterministic and OS-free like the bump version.
 * - **Sibling-module `require`**: suite files require each other (`bitwise.lua` →
 *   `require'bwcoercion'`, `cstack.lua`/`locals.lua` → `require'tracegc'`). The embedded module
 *   sources (also unmodified suite files) are pre-run and recorded in the registry `_LOADED` table,
 *   exactly where stock `require` would cache them; the minimal `require` then finds them.
 * - A **`package` global** with `loaded` = the registry `_LOADED` table (its stock identity) —
 *   `nextvar.lua`'s global-table sweep consults `package.loaded` unguarded.
 * - Chunks are loaded under the **`@name.lua`** chunkname (how `dofile` names file chunks):
 *   `db.lua` asserts `debug.getinfo(...).source` matches `^@.*db%.lua$`.
 * - The suite's own portability knobs: `_port = true`, `_soft = true` (skip popen/execute/OS
 *   sections and the huge-data stress) — the documented configuration, files run unmodified. */
#include "lua.h"
#include "lauxlib.h"
#include "lualib.h"
#include "ltests.h"

extern const unsigned char *const lua_tests[];
extern const unsigned int lua_test_lens[];
extern const char *const lua_test_names[];
extern const unsigned int lua_test_count;
/* Sibling modules the test files `require` (unmodified suite files, pre-run into `_LOADED`). */
extern const unsigned char *const lua_modules[];
extern const unsigned int lua_module_lens[];
extern const char *const lua_module_names[];
extern const unsigned int lua_module_count;

/* Minimal `require`: the `_LOADED` cache plus the stock **preload searcher** — a loader stored in
 * `package.preload[name]` (the registry `_PRELOAD` table) is called and its result cached, exactly
 * stock `require`'s first searcher (`bitwise.lua` installs its `bit32` shim this way). No file/C
 * searchers — those need a filesystem/dynamic loader the harness deliberately doesn't wire. */
static int l_require(lua_State *L) {
  const char *name = luaL_checkstring(L, 1);
  luaL_getsubtable(L, LUA_REGISTRYINDEX, LUA_LOADED_TABLE);
  lua_getfield(L, -1, name); /* _LOADED[name] */
  if (lua_toboolean(L, -1)) return 1;
  lua_pop(L, 1);
  luaL_getsubtable(L, LUA_REGISTRYINDEX, LUA_PRELOAD_TABLE);
  lua_getfield(L, -1, name); /* package.preload[name] */
  if (lua_isnil(L, -1)) return luaL_error(L, "module '%s' not found", name);
  lua_pushstring(L, name);
  lua_call(L, 1, 1); /* loader(name) */
  if (lua_isnil(L, -1)) {
    lua_pop(L, 1);
    lua_pushboolean(L, 1); /* a loader returning nothing records `true`, like stock require */
  }
  lua_pushvalue(L, -1);
  lua_setfield(L, -4, name); /* _LOADED[name] = result */
  return 1;
}

static lua_State *GL;

static int run_one(const unsigned char *src, unsigned len, const char *name) {
  /* `name` arrives as the "@file.lua" chunkname the embed generator wrote. */
  int rc = 0;
  const char *msg = "ok";
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
  /* ltests.h's own `lua_c` recipe: the tracking, failure-injecting allocator + the T library. */
  lua_State *L = lua_newstate(debug_realloc, &l_memcontrol);
  if (!L) return 99;
  GL = L;
  static const luaL_Reg libs[] = {
    {LUA_GNAME, luaopen_base},          {LUA_STRLIBNAME, luaopen_string},
    {LUA_TABLIBNAME, luaopen_table},    {LUA_MATHLIBNAME, luaopen_math},
    {LUA_COLIBNAME, luaopen_coroutine}, {LUA_DBLIBNAME, luaopen_debug},
    {LUA_IOLIBNAME, luaopen_io},        {LUA_OSLIBNAME, luaopen_os},
    {LUA_UTF8LIBNAME, luaopen_utf8},    {(void *)0, (void *)0},
  };
  for (const luaL_Reg *lib = libs; lib->func; lib++) {
    luaL_requiref(L, lib->name, lib->func, 1);
    lua_pop(L, 1);
  }
  luaL_requiref(L, "T", luaB_opentests, 1); /* the internal C-test library */
  lua_pop(L, 1);
  lua_register(L, "require", l_require);
  lua_newtable(L);
  luaL_getsubtable(L, LUA_REGISTRYINDEX, LUA_LOADED_TABLE);
  lua_setfield(L, -2, "loaded");
  luaL_getsubtable(L, LUA_REGISTRYINDEX, LUA_PRELOAD_TABLE);
  lua_setfield(L, -2, "preload");
  luaL_getsubtable(L, LUA_REGISTRYINDEX, LUA_LOADED_TABLE);
  lua_pushvalue(L, -2);
  lua_setfield(L, -2, "package");
  lua_pop(L, 1);
  lua_setglobal(L, "package");
  lua_newtable(L);
  lua_setglobal(L, "arg");
  lua_pushboolean(L, 1);
  lua_setglobal(L, "_port");
  lua_pushboolean(L, 1);
  lua_setglobal(L, "_soft");

  /* Pre-run the sibling modules into `_LOADED` once. */
  for (unsigned m = 0; m < lua_module_count; m++) {
    char chunk[64];
    int i = 0;
    chunk[i++] = '@';
    for (const char *p = lua_module_names[m]; *p && i < 58; p++) chunk[i++] = *p;
    chunk[i++] = '.'; chunk[i++] = 'l'; chunk[i++] = 'u'; chunk[i++] = 'a';
    chunk[i] = 0;
    if (luaL_loadbuffer(L, (const char *)lua_modules[m], lua_module_lens[m], chunk) != LUA_OK ||
        lua_pcall(L, 0, 1, 0) != LUA_OK)
      return 98;
    if (lua_isnil(L, -1)) {
      lua_pop(L, 1);
      lua_pushboolean(L, 1);
    }
    luaL_getsubtable(L, LUA_REGISTRYINDEX, LUA_LOADED_TABLE);
    lua_insert(L, -2);
    lua_setfield(L, -2, lua_module_names[m]);
    lua_pop(L, 1);
  }

  for (unsigned i = 0; i < lua_test_count; i++) {
    if (run_one(lua_tests[i], lua_test_lens[i], lua_test_names[i]) != 0)
      return (int)(i + 1); /* 1-based index of the first failing file */
  }
  lua_close(L);
  return 0;
}
