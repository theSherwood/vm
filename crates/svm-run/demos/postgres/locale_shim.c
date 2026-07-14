/* Guest locale + wide-ctype shim — the C/POSIX-locale surface Postgres touches during early startup
 * (slice CF, gap #11g). `set_pglocale_pgservice` calls `setlocale`; the scanner/encoding paths use
 * `newlocale`/`uselocale`, `nl_langinfo`, `localeconv`, and the `iswX`/`towX` wide-ctype family.
 * The sandbox is fixed to the **C locale**, so these are constants / ASCII classification — no host,
 * no locale database. `#include`d into a driver under `-DSVM_GUEST`, after `libc_shim.c`.
 */

#include <ctype.h>
#include <langinfo.h>
#include <locale.h>
#include <wctype.h>

/* setlocale always reports the C locale (the only one the sandbox has). */
char *setlocale(int category, const char *locale) {
  (void)category;
  (void)locale;
  return (char *)"C";
}

/* newlocale/uselocale/…: a single opaque C-locale handle. The `*_l` functions below ignore it. */
locale_t newlocale(int mask, const char *locale, locale_t base) {
  (void)mask;
  (void)locale;
  (void)base;
  return (locale_t)1;
}
locale_t uselocale(locale_t loc) {
  (void)loc;
  return (locale_t)1;
}
locale_t duplocale(locale_t loc) {
  (void)loc;
  return (locale_t)1;
}
void freelocale(locale_t loc) { (void)loc; }

/* localeconv: the C-locale `struct lconv` (only decimal_point is non-"" in C). */
struct lconv *localeconv(void) {
  static char empty[] = "";
  static char dot[] = ".";
  static struct lconv lc;
  lc.decimal_point = dot;
  lc.thousands_sep = empty;
  lc.grouping = empty;
  lc.int_curr_symbol = empty;
  lc.currency_symbol = empty;
  lc.mon_decimal_point = empty;
  lc.mon_thousands_sep = empty;
  lc.mon_grouping = empty;
  lc.positive_sign = empty;
  lc.negative_sign = empty;
  lc.int_frac_digits = 127;
  lc.frac_digits = 127;
  lc.p_cs_precedes = 127;
  lc.p_sep_by_space = 127;
  lc.n_cs_precedes = 127;
  lc.n_sep_by_space = 127;
  lc.p_sign_posn = 127;
  lc.n_sign_posn = 127;
  return &lc;
}

char *nl_langinfo(nl_item item) {
  if (item == CODESET) return (char *)"ANSI_X3.4-1968"; /* the C-locale codeset name (US-ASCII) */
  return (char *)"";
}

/* ---- wide ctype (C locale = ASCII classification for 0..127, nothing above) ------------------ */
static int w_ascii(wint_t c) { return c >= 0 && c <= 127; }
int iswalnum(wint_t c) { return w_ascii(c) && isalnum((int)c); }
int iswalpha(wint_t c) { return w_ascii(c) && isalpha((int)c); }
int iswdigit(wint_t c) { return w_ascii(c) && isdigit((int)c); }
int iswxdigit(wint_t c) { return w_ascii(c) && isxdigit((int)c); }
int iswlower(wint_t c) { return w_ascii(c) && islower((int)c); }
int iswupper(wint_t c) { return w_ascii(c) && isupper((int)c); }
int iswspace(wint_t c) { return w_ascii(c) && isspace((int)c); }
int iswprint(wint_t c) { return w_ascii(c) && isprint((int)c); }
int iswgraph(wint_t c) { return w_ascii(c) && isgraph((int)c); }
int iswpunct(wint_t c) { return w_ascii(c) && ispunct((int)c); }
int iswcntrl(wint_t c) { return w_ascii(c) && iscntrl((int)c); }
int iswblank(wint_t c) { return c == '\t' || c == ' '; }
wint_t towlower(wint_t c) { return w_ascii(c) ? (wint_t)tolower((int)c) : c; }
wint_t towupper(wint_t c) { return w_ascii(c) ? (wint_t)toupper((int)c) : c; }
/* The `*_l` variants ignore the locale (there is only the C locale). */
int iswalnum_l(wint_t c, locale_t l) { (void)l; return iswalnum(c); }
int iswalpha_l(wint_t c, locale_t l) { (void)l; return iswalpha(c); }
int iswdigit_l(wint_t c, locale_t l) { (void)l; return iswdigit(c); }
int iswlower_l(wint_t c, locale_t l) { (void)l; return iswlower(c); }
int iswupper_l(wint_t c, locale_t l) { (void)l; return iswupper(c); }
int iswspace_l(wint_t c, locale_t l) { (void)l; return iswspace(c); }
int iswprint_l(wint_t c, locale_t l) { (void)l; return iswprint(c); }
int iswgraph_l(wint_t c, locale_t l) { (void)l; return iswgraph(c); }
int iswpunct_l(wint_t c, locale_t l) { (void)l; return iswpunct(c); }
wint_t towlower_l(wint_t c, locale_t l) { (void)l; return towlower(c); }
wint_t towupper_l(wint_t c, locale_t l) { (void)l; return towupper(c); }
