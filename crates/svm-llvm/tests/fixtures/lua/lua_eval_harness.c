/* Lua eval harness: read a Lua chunk from **stdin**, run it, print its output (and any error) to
 * stdout. This is the interactive-playground guest — the page pipes the editor's text in as stdin.
 * Opens base+string+table+math+coroutine+io+os over the lua_files guest layers (stdio/time/shim):
 * print()/io.write go to stdout via the Stream capability; os.time/date/clock ride the guest time
 * layer; coroutines are pure (setjmp/longjmp). File I/O (io.open) degrades gracefully — the stdio
 * layer resolves the `fs` capability lazily and this editor grants none, so io.open returns nil.
 * Arena allocator (no host malloc). string.format works (guest snprintf linked, like lua_fmt). */
#include "lua.h"
#include "lauxlib.h"
#include "lualib.h"

extern long read(int fd, void *buf, long n);
extern long write(int fd, const void *buf, long n);
static unsigned long slen(const char *s) { unsigned long n = 0; while (s[n]) n++; return n; }

static char arena[48 * 1024 * 1024];
static unsigned long arena_off = 0;
static void *l_alloc(void *ud, void *ptr, size_t osize, size_t nsize) {
  (void)ud;
  if (nsize == 0) return (void *)0;
  unsigned long start = (arena_off + 15UL) & ~15UL;
  if (start + nsize > sizeof arena) return (void *)0;
  char *np = &arena[start]; arena_off = start + nsize;
  if (ptr && osize) { size_t n = osize < nsize ? osize : nsize; for (size_t i = 0; i < n; i++) np[i] = ((char *)ptr)[i]; }
  return np;
}

static char src[1 << 20]; /* up to 1 MiB of editor text */
int main(void) {
  long len = 0;
  for (;;) {
    long r = read(0, src + len, (long)sizeof(src) - 1 - len);
    if (r <= 0) break;
    len += r;
    if (len >= (long)sizeof(src) - 1) break;
  }
  src[len] = 0;

  lua_State *L = lua_newstate(l_alloc, (void *)0);
  if (!L) return 1;
  static const luaL_Reg libs[] = {
    {LUA_GNAME, luaopen_base},          {LUA_STRLIBNAME, luaopen_string},
    {LUA_TABLIBNAME, luaopen_table},    {LUA_MATHLIBNAME, luaopen_math},
    {LUA_COLIBNAME, luaopen_coroutine}, {LUA_IOLIBNAME, luaopen_io},
    {LUA_OSLIBNAME, luaopen_os},        {(void *)0, (void *)0},
  };
  for (const luaL_Reg *lib = libs; lib->func; lib++) { luaL_requiref(L, lib->name, lib->func, 1); lua_pop(L, 1); }

  if (luaL_loadbuffer(L, src, (size_t)len, "=input") != LUA_OK || lua_pcall(L, 0, 0, 0) != LUA_OK) {
    const char *msg = lua_tostring(L, -1);
    if (msg) { write(1, "error: ", 7); write(1, msg, (long)slen(msg)); write(1, "\n", 1); }
    lua_close(L);
    return 2;
  }
  lua_close(L);
  return 0;
}
