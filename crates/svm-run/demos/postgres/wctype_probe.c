/* wctype_probe.c — byte-exact differential for the guest wide-ctype shim (slice CF).
 *
 * The C/POSIX-locale `iswX`/`towX` family is ASCII classification (nothing above 127 is classed).
 * For every code point 0..255 the guest (`-DSVM_GUEST`, locale_shim.c) must byte-match native glibc
 * over the twelve `iswX` classes + `towlower`/`towupper`. Pure — runs on the bare powerbox. (The
 * `iswX_l` variants just forward to these, so testing the base family pins them too.)
 */

#include <stdio.h>
#include <wctype.h>

#ifdef SVM_GUEST
#include "libc_shim.c"
#include "locale_shim.c"
#endif

int main(void) {
  for (int c = 0; c < 256; c++) {
    printf("%d %d%d%d%d%d%d%d%d%d%d%d%d %d %d\n", c, !!iswalnum(c), !!iswalpha(c), !!iswdigit(c),
           !!iswxdigit(c), !!iswlower(c), !!iswupper(c), !!iswspace(c), !!iswblank(c), !!iswpunct(c),
           !!iswprint(c), !!iswgraph(c), !!iswcntrl(c), (int)towlower(c), (int)towupper(c));
  }
  return 0;
}
