/* Float-Lua harness: drive Lua 5.4.7 core with a static-arena allocator and run an embedded float
 * script, returning the (integer-cast) result. The script exercises the guest libc/libm linked
 * alongside: `strtod` (every numeric literal, at parse time), `pow` (the `^` operator, at runtime),
 * and `fmod` (the `%` operator, at runtime). Core-only build (no stdlib), so no `math` table — the
 * float ops are reached through the VM/lexer directly. */
#include "lua.h"

static char arena[16 * 1024 * 1024];
static unsigned long arena_off = 0;

static void *l_alloc(void *ud, void *ptr, size_t osize, size_t nsize) {
  (void)ud;
  if (nsize == 0) return (void *)0; /* free: no-op in a bump arena */
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

static const char SCRIPT[] =
    "local function f(x) return x ^ 0.5 end\n"   /* runtime pow */
    "local function g(x, y) return x % y end\n"  /* runtime fmod */
    "local a = 3.14\n"                           /* strtod */
    "local b = f(2.0)\n"                         /* pow(2.0, 0.5) = sqrt 2 */
    "local c = g(10.5, 3.0)\n"                   /* fmod(10.5, 3.0) = 1.5 */
    "local d = 1.5e3 + 0.25\n"                   /* strtod scientific + frac */
    "return (a + b + c + d) * 1000.0\n";

typedef struct {
  const char *s;
  size_t left;
} Reader;

static const char *reader(lua_State *L, void *data, size_t *size) {
  (void)L;
  Reader *r = (Reader *)data;
  if (r->left == 0) return (const char *)0;
  *size = r->left;
  r->left = 0;
  return r->s;
}

int main(void) {
  lua_State *L = lua_newstate(l_alloc, (void *)0);
  if (!L) return -1;
  Reader r;
  r.s = SCRIPT;
  r.left = sizeof(SCRIPT) - 1;
  if (lua_load(L, reader, &r, "=script", (const char *)0) != LUA_OK) return -2;
  if (lua_pcall(L, 0, 1, 0) != LUA_OK) return -3;
  int isnum = 0;
  lua_Number n = lua_tonumberx(L, -1, &isnum);
  if (!isnum) return -4;
  return (int)n;
}
