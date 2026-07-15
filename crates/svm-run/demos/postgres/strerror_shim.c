/* Guest `strerror_r` — the **GNU** `char *` variant, in its own translation unit.
 *
 * The bitcode declares `char *strerror_r(int, char*, size_t)` (Postgres is built with `_GNU_SOURCE`).
 * That prototype can't be defined in the shared shim TU: without `_GNU_SOURCE`, `<string.h>` declares
 * the POSIX `int strerror_r(...)` (a hard conflict), and defining `_GNU_SOURCE` across the whole shim
 * TU perturbs unrelated declarations (`__isoc23_strtol`, `getrlimit`/`setitimer` signatures). So this
 * one function is compiled *alone* with `-D_GNU_SOURCE` (see `link_shims.sh`) and `llvm-link`ed in.
 *
 * The message text is reused from `locale_shim.c`'s C-locale table (`shim_errmsg`); the GNU contract
 * lets us fill the caller's buffer and return it. Compiled with `-DSVM_GUEST` like the rest.
 */

#define _GNU_SOURCE
#include <stddef.h>
#include <string.h>

/* The C-locale errno→message table lives in locale_shim.c (exported for this TU). */
const char *shim_errmsg(int e);

char *strerror_r(int errnum, char *buf, size_t buflen) {
  const char *m = shim_errmsg(errnum);
  char tmp[32];
  if (!m) { /* "Unknown error N" for codes outside the table */
    const char *pre = "Unknown error ";
    size_t k = 0;
    while (pre[k] && k < sizeof tmp - 12) {
      tmp[k] = pre[k];
      k++;
    }
    unsigned v = errnum < 0 ? (unsigned)(-errnum) : (unsigned)errnum;
    if (errnum < 0) tmp[k++] = '-';
    char d[12];
    int t = 0;
    do {
      d[t++] = (char)('0' + v % 10);
      v /= 10;
    } while (v && t < (int)sizeof d);
    while (t > 0) tmp[k++] = d[--t];
    tmp[k] = 0;
    m = tmp;
  }
  if (buf && buflen) { /* GNU: may fill buf and/or return a string; return the filled buffer */
    size_t i = 0;
    while (m[i] && i + 1 < buflen) {
      buf[i] = m[i];
      i++;
    }
    buf[i] = 0;
    return buf;
  }
  return (char *)m;
}
