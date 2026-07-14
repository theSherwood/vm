/* ctype_probe.c — a byte-exact differential for the guest ctype tables (slice CB).
 *
 * For every byte value 0..255, print the twelve `<ctype.h>` classifications (as booleans) plus the
 * `tolower`/`toupper` mappings. The guest build (`-DSVM_GUEST`, using libc_shim.c's C-locale tables)
 * must byte-match the native glibc build over the whole range — which pins every bit of every table,
 * since each class is tested independently on each character.
 *
 * Classification uses the `isX` macros, which glibc expands to `(*__ctype_b_loc())[c] & _ISbit` — a
 * direct table index (no function call), exactly the form Postgres's scanner uses. Case mapping is
 * read the same way, straight from `__ctype_tolower_loc()`/`__ctype_toupper_loc()` — the tables
 * Postgres indexes — rather than the `tolower()`/`toupper()` *functions* (glibc's `tolower(c)` macro
 * calls a function for a non-constant `int`, a separate surface this slice doesn't cover).
 *
 * `isX(c)` etc. index with `c` in 0..255 (a valid, in-range index — never a negative `char`).
 */

#include <ctype.h>
#include <stdio.h>

#ifdef SVM_GUEST
#include "libc_shim.c"
#endif

int main(void) {
  const int *tl = *__ctype_tolower_loc();
  const int *tu = *__ctype_toupper_loc();
  for (int c = 0; c < 256; c++) {
    printf("%d %d%d%d%d%d%d%d%d%d%d%d%d %d %d\n", c,
           !!isalnum(c), !!isalpha(c), !!isdigit(c), !!isxdigit(c), !!islower(c), !!isupper(c),
           !!isspace(c), !!isblank(c), !!ispunct(c), !!isprint(c), !!isgraph(c), !!iscntrl(c),
           tl[c], tu[c]);
  }
  return 0;
}
