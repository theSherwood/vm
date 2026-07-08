/* Runs a batch of unmodified official Lua 5.4.7 test files (embedded in `lua_sweep_tests.c`) with
 * every guest-available library open (base/string/table/math/coroutine/debug/io/os/utf8), one fresh
 * `lua_State` per file. Extends the `lua_files_harness.c` shape with what the wider suite needs:
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

extern const unsigned char *const lua_tests[];
extern const unsigned int lua_test_lens[];
extern const char *const lua_test_names[];
extern const unsigned int lua_test_count;
/* Sibling modules the test files `require` (unmodified suite files, pre-run into `_LOADED`). */
extern const unsigned char *const lua_modules[];
extern const unsigned int lua_module_lens[];
extern const char *const lua_module_names[];
extern const unsigned int lua_module_count;

/* ---- allocator: segregated power-of-two free lists over a static arena ------------------------
 * Lua's contract: alloc(NULL, tag, n) / realloc(p, o, n) / free(p, o, 0). Each block carries an
 * 8-byte size-class header. Freed blocks push onto their class list; allocation pops or bumps.
 * Reset per file by dropping every list + the bump cursor (teardown of the `lua_State` frees all). */
static char arena[48 * 1024 * 1024]; /* fits the reference JIT's 64 MiB window cap; recycling (not
                                       * size) is what the collector-stress files need */
static unsigned long arena_off;
#define NCLASS 24 /* 16 B .. 128 MiB, class i = 1 << (i + 4) */
static void *freelist[NCLASS];

static int class_of(unsigned long n) {
  unsigned long want = n + 8; /* header */
  int c = 0;
  unsigned long sz = 16;
  while (sz < want) {
    sz <<= 1;
    c++;
  }
  return c;
}

static void *fl_alloc(unsigned long n) {
  int c = class_of(n);
  if (c >= NCLASS) return (void *)0;
  if (freelist[c]) {
    char *blk = (char *)freelist[c];
    freelist[c] = *(void **)blk;
    *(unsigned long *)blk = (unsigned long)c; /* the free-list link reused the header word */
    return blk + 8;
  }
  unsigned long sz = 16UL << c;
  unsigned long start = (arena_off + 15UL) & ~15UL;
  if (start + sz > sizeof arena) return (void *)0;
  char *blk = &arena[start];
  arena_off = start + sz;
  *(unsigned long *)blk = (unsigned long)c;
  return blk + 8;
}

static void fl_free(void *p) {
  char *blk = (char *)p - 8;
  int c = (int)*(unsigned long *)blk;
  *(void **)blk = freelist[c];
  freelist[c] = blk;
}

static void *l_alloc(void *ud, void *ptr, size_t osize, size_t nsize) {
  (void)ud;
  if (nsize == 0) {
    if (ptr) fl_free(ptr);
    return (void *)0;
  }
  if (!ptr) return fl_alloc(nsize);
  {
    int c = (int)*(unsigned long *)((char *)ptr - 8);
    unsigned long cap = (16UL << c) - 8;
    if (nsize <= cap) return ptr; /* still fits its class */
    void *np = fl_alloc(nsize);
    if (!np) return (void *)0;
    unsigned long n = osize < nsize ? osize : nsize;
    for (unsigned long i = 0; i < n; i++) ((char *)np)[i] = ((char *)ptr)[i];
    fl_free(ptr);
    return np;
  }
}

static void alloc_reset(void) {
  arena_off = 0;
  for (int i = 0; i < NCLASS; i++) freelist[i] = (void *)0;
}

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

static int run_one(const unsigned char *src, unsigned len, const char *name) {
  alloc_reset(); /* fresh, isolated heap per file */
  lua_State *L = lua_newstate(l_alloc, (void *)0);
  if (!L) return 1;
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
  lua_register(L, "require", l_require);
  /* A faithful-shape `package` global: `loaded` is the registry `_LOADED` table and `preload` the
   * registry `_PRELOAD` table (their stock identities), and the package table itself is recorded in
   * `_LOADED["package"]` (as stock `require"package"` does) — `nextvar.lua`'s global-table sweep
   * erases any global *not* in `package.loaded`, so without that entry it would wipe `package`
   * mid-loop and then fail indexing it. */
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

  int rc = 0;
  const char *msg = "ok";
  /* Pre-run the sibling modules into `_LOADED` (chunk result, or `true` if it returns nothing) —
   * what stock `require` would record. `@`-prefixed chunknames, as file chunks are named. */
  for (unsigned m = 0; m < lua_module_count && rc == 0; m++) {
    char chunk[64];
    /* "@<name>.lua" */
    {
      int i = 0;
      chunk[i++] = '@';
      for (const char *p = lua_module_names[m]; *p && i < 58; p++) chunk[i++] = *p;
      chunk[i++] = '.'; chunk[i++] = 'l'; chunk[i++] = 'u'; chunk[i++] = 'a';
      chunk[i] = 0;
    }
    if (luaL_loadbuffer(L, (const char *)lua_modules[m], lua_module_lens[m], chunk) != LUA_OK ||
        lua_pcall(L, 0, 1, 0) != LUA_OK) {
      msg = lua_tostring(L, -1);
      rc = 4;
      break;
    }
    if (lua_isnil(L, -1)) {
      lua_pop(L, 1);
      lua_pushboolean(L, 1);
    }
    luaL_getsubtable(L, LUA_REGISTRYINDEX, LUA_LOADED_TABLE);
    lua_insert(L, -2);
    lua_setfield(L, -2, lua_module_names[m]);
    lua_pop(L, 1);
  }

  if (rc == 0) {
    if (luaL_loadbuffer(L, (const char *)src, len, name) != LUA_OK) {
      msg = lua_tostring(L, -1);
      rc = 2;
    } else if (lua_pcall(L, 0, 0, 0) != LUA_OK) {
      msg = lua_tostring(L, -1);
      rc = 3;
    }
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
