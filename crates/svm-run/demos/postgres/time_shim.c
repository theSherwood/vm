/* Guest time shim — `gmtime`/`localtime` (UTC) + `strftime` (slice CD, gap #11e). The sandbox has a
 * single fixed timezone (UTC), so `localtime` == `gmtime`; both are pure calendar math on a `time_t`
 * (no host, no /etc/localtime). `strftime` is a bounded format engine over a `struct tm` — no varargs.
 * Reproduces glibc's C-locale output exactly (the `time_probe.c` differential over the TZ-independent
 * conversions Postgres's log/timestamp paths use).
 *
 * `#include`d into a driver under `-DSVM_GUEST`, like the other shims.
 */

#include <stddef.h>
#include <time.h>

/* Civil date from a day count (Howard Hinnant's algorithm), then time-of-day + weekday + yearday. */
struct tm *gmtime_r(const time_t *tp, struct tm *out) {
  long t = (long)*tp;
  long days = t / 86400;
  long rem = t % 86400;
  if (rem < 0) {
    rem += 86400;
    days -= 1;
  }
  out->tm_hour = (int)(rem / 3600);
  out->tm_min = (int)(rem % 3600 / 60);
  out->tm_sec = (int)(rem % 60);
  long wday = (4 + days) % 7; /* 1970-01-01 was a Thursday (=4) */
  if (wday < 0) wday += 7;
  out->tm_wday = (int)wday;

  long z = days + 719468;
  long era = (z >= 0 ? z : z - 146096) / 146097;
  long doe = z - era * 146097;
  long yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
  long y = yoe + era * 400;
  long doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
  long mp = (5 * doy + 2) / 153;
  long d = doy - (153 * mp + 2) / 5 + 1;
  long m = mp < 10 ? mp + 3 : mp - 9;
  y += (m <= 2);
  out->tm_year = (int)(y - 1900);
  out->tm_mon = (int)(m - 1);
  out->tm_mday = (int)d;
  out->tm_isdst = 0;

  int leap = (y % 4 == 0 && (y % 100 != 0 || y % 400 == 0));
  static const int mdays[12] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
  int yday = (int)d - 1;
  for (int i = 0; i < out->tm_mon; i++) yday += mdays[i] + (i == 1 ? leap : 0);
  out->tm_yday = yday;
#ifdef __USE_MISC
  out->tm_gmtoff = 0;
  out->tm_zone = "GMT"; /* glibc's gmtime uses "GMT" */
#endif
  return out;
}
static struct tm shim_tm;
struct tm *gmtime(const time_t *tp) { return gmtime_r(tp, &shim_tm); }
/* The sandbox is UTC: localtime is gmtime. */
struct tm *localtime_r(const time_t *tp, struct tm *out) { return gmtime_r(tp, out); }
struct tm *localtime(const time_t *tp) { return gmtime_r(tp, &shim_tm); }

static const char *const shim_wday_ab[7] = {"Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"};
static const char *const shim_wday_full[7] = {"Sunday",   "Monday", "Tuesday", "Wednesday",
                                              "Thursday", "Friday", "Saturday"};
static const char *const shim_mon_ab[12] = {"Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"};
static const char *const shim_mon_full[12] = {"January", "February", "March",     "April",
                                              "May",     "June",     "July",      "August",
                                              "September", "October", "November", "December"};

/* A tiny bounded appender: write `s` into buf[pos..cap), tracking overflow. */
static size_t shim_put(char *buf, size_t pos, size_t cap, const char *s) {
  while (*s) {
    if (pos + 1 < cap) buf[pos] = *s;
    pos++;
    s++;
  }
  return pos;
}
static size_t shim_num(char *buf, size_t pos, size_t cap, long v, int width, char pad) {
  char tmp[24];
  int n = 0;
  int neg = v < 0;
  unsigned long u = neg ? (unsigned long)(-v) : (unsigned long)v;
  do {
    tmp[n++] = (char)('0' + u % 10);
    u /= 10;
  } while (u);
  if (neg) tmp[n++] = '-';
  while (n < width) tmp[n++] = pad;
  while (n > 0) {
    char c = tmp[--n];
    if (pos + 1 < cap) buf[pos] = c;
    pos++;
  }
  return pos;
}

size_t strftime(char *s, size_t max, const char *fmt, const struct tm *tm) {
  size_t pos = 0;
  for (const char *p = fmt; *p; p++) {
    if (*p != '%') {
      if (pos + 1 < max) s[pos] = *p;
      pos++;
      continue;
    }
    p++;
    switch (*p) {
      case 'Y': pos = shim_num(s, pos, max, tm->tm_year + 1900, 4, '0'); break;
      case 'y': pos = shim_num(s, pos, max, (tm->tm_year + 1900) % 100, 2, '0'); break;
      case 'C': pos = shim_num(s, pos, max, (tm->tm_year + 1900) / 100, 2, '0'); break;
      case 'm': pos = shim_num(s, pos, max, tm->tm_mon + 1, 2, '0'); break;
      case 'd': pos = shim_num(s, pos, max, tm->tm_mday, 2, '0'); break;
      case 'e': pos = shim_num(s, pos, max, tm->tm_mday, 2, ' '); break;
      case 'H': pos = shim_num(s, pos, max, tm->tm_hour, 2, '0'); break;
      case 'I': {
        int h = tm->tm_hour % 12;
        if (h == 0) h = 12;
        pos = shim_num(s, pos, max, h, 2, '0');
        break;
      }
      case 'M': pos = shim_num(s, pos, max, tm->tm_min, 2, '0'); break;
      case 'S': pos = shim_num(s, pos, max, tm->tm_sec, 2, '0'); break;
      case 'j': pos = shim_num(s, pos, max, tm->tm_yday + 1, 3, '0'); break;
      case 'w': pos = shim_num(s, pos, max, tm->tm_wday, 0, '0'); break;
      case 'u': pos = shim_num(s, pos, max, tm->tm_wday == 0 ? 7 : tm->tm_wday, 0, '0'); break;
      case 'p': pos = shim_put(s, pos, max, tm->tm_hour < 12 ? "AM" : "PM"); break;
      case 'a': pos = shim_put(s, pos, max, shim_wday_ab[tm->tm_wday & 7]); break;
      case 'A': pos = shim_put(s, pos, max, shim_wday_full[tm->tm_wday & 7]); break;
      case 'b':
      case 'h': pos = shim_put(s, pos, max, shim_mon_ab[tm->tm_mon % 12]); break;
      case 'B': pos = shim_put(s, pos, max, shim_mon_full[tm->tm_mon % 12]); break;
      case 'n': pos = shim_put(s, pos, max, "\n"); break;
      case 't': pos = shim_put(s, pos, max, "\t"); break;
      case '%': pos = shim_put(s, pos, max, "%"); break;
      case '\0': p--; break; /* trailing '%' — stop */
      default:
        /* an unhandled conversion is emitted literally as "%<c>", matching glibc's fallback */
        if (pos + 1 < max) s[pos] = '%';
        pos++;
        if (pos + 1 < max) s[pos] = *p;
        pos++;
        break;
    }
  }
  if (max > 0) s[pos < max ? pos : max - 1] = 0;
  return pos < max ? pos : 0; /* glibc: 0 if the result (incl. NUL) didn't fit */
}
