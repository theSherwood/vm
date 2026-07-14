/* time_probe.c — byte-exact differential for the guest time + wide-char shims (slice CD).
 *
 * Guest (`-DSVM_GUEST`, time_shim.c + libc_shim.c) vs native glibc: gmtime + strftime over several
 * epochs (incl. a leap day), using only TZ-independent conversions (no %Z/%z — the sandbox is UTC),
 * and the C-locale mbstowcs/wcstombs identity map. Pure — runs on the bare powerbox.
 */

#include <stdio.h>
#include <stdlib.h> /* mbstowcs / wcstombs */
#include <time.h>
#include <wchar.h>

#ifdef SVM_GUEST
#include "libc_shim.c"
#include "time_shim.c"
#endif

int main(void) {
  /* epoch 0 (1970-01-01 Thu), a round 10^9, ~2023, 2000-02-29 (leap), 1972-02-29 (leap) */
  const time_t stamps[5] = {0, 1000000000, 1700000000, 951782400, 68169600};
  char buf[128];
  for (int i = 0; i < 5; i++) {
    struct tm tm;
    gmtime_r(&stamps[i], &tm);
    strftime(buf, sizeof buf, "%Y-%m-%d %H:%M:%S %A %d %B %j %w %p %I%%", &tm);
    printf("[%s]\n", buf);
  }
  /* mbstowcs / wcstombs round-trip + count-only mode */
  wchar_t wb[32];
  size_t wn = mbstowcs(wb, "Hello123", 32);
  char nb[32];
  size_t bn = wcstombs(nb, wb, 32);
  nb[bn] = 0;
  printf("mbstowcs=%d wcstombs=%d [%s]\n", (int)wn, (int)bn, nb);
  printf("mbslen=%d\n", (int)mbstowcs((wchar_t *)0, "abcdef", 0));
  return 0;
}
