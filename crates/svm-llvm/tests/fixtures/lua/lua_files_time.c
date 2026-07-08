/* Guest time/date layer for the `os` library (ANSI path: `time`/`gmtime`/`localtime`/`mktime`/
 * `strftime`/`difftime`/`clock`). Pure computation over a proleptic-Gregorian calendar (Howard
 * Hinnant's days↔civil algorithms), UTC only: `localtime` == `gmtime` (`tm_isdst = 0`) and `mktime`
 * is the exact inverse, so every Lua round-trip (`os.time(os.date("*t", t)) == t`) holds. `time()`
 * returns a fixed synthetic epoch — the on-ramp grants no ambient clock, Lua's own tests only need
 * internal consistency, and a constant keeps runs deterministic. Compiled against <time.h> so
 * `struct tm` has the platform layout loslib was compiled with. */
#include <time.h>

static const time_t SYNTHETIC_NOW = 1717171717; /* 2024-05-31T15:28:37Z, arbitrary but fixed */

time_t time(time_t *t) {
  if (t) *t = SYNTHETIC_NOW;
  return SYNTHETIC_NOW;
}

double difftime(time_t a, time_t b) { return (double)a - (double)b; }

/* days since 1970-01-01 for a civil date (proleptic Gregorian; Hinnant's days_from_civil) */
static long long days_from_civil(long long y, int m, int d) {
  y -= m <= 2;
  long long era = (y >= 0 ? y : y - 399) / 400;
  long long yoe = y - era * 400;                                  /* [0, 399] */
  long long doy = (153LL * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1; /* [0, 365] */
  long long doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;          /* [0, 146096] */
  return era * 146097 + doe - 719468;
}

/* civil date for days since 1970-01-01 (Hinnant's civil_from_days) */
static void civil_from_days(long long z, long long *y, int *m, int *d) {
  z += 719468;
  long long era = (z >= 0 ? z : z - 146096) / 146097;
  long long doe = z - era * 146097;                                     /* [0, 146096] */
  long long yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365; /* [0, 399] */
  long long yy = yoe + era * 400;
  long long doy = doe - (365 * yoe + yoe / 4 - yoe / 100);              /* [0, 365] */
  long long mp = (5 * doy + 2) / 153;                                   /* [0, 11] */
  *d = (int)(doy - (153 * mp + 2) / 5 + 1);
  *m = (int)(mp + (mp < 10 ? 3 : -9));
  *y = yy + (*m <= 2);
}

static int is_leap(long long y) { return (y % 4 == 0 && y % 100 != 0) || y % 400 == 0; }

static struct tm tm_static;

static struct tm *to_tm(time_t t, struct tm *out) {
  long long days = t / 86400, rem = t % 86400;
  if (rem < 0) {
    rem += 86400;
    days -= 1;
  }
  long long y;
  int m, d;
  civil_from_days(days, &y, &m, &d);
  out->tm_year = (int)(y - 1900);
  out->tm_mon = m - 1;
  out->tm_mday = d;
  out->tm_hour = (int)(rem / 3600);
  out->tm_min = (int)(rem % 3600 / 60);
  out->tm_sec = (int)(rem % 60);
  /* 1970-01-01 was a Thursday (wday 4) */
  out->tm_wday = (int)((days % 7 + 11) % 7);
  out->tm_yday = (int)(days - days_from_civil(y, 1, 1));
  out->tm_isdst = 0; /* UTC, no DST — localtime == gmtime */
  return out;
}

struct tm *gmtime(const time_t *t) { return to_tm(*t, &tm_static); }
struct tm *localtime(const time_t *t) { return to_tm(*t, &tm_static); }

/* Inverse of `to_tm`, with C `mktime` field normalization (out-of-range mon/mday/hour/… carry), so
 * `os.time{year=y, month=m, day=d, …}` accepts denormalized tables exactly like a native mktime.
 * `(time_t)-1` on a result outside time_t (loslib turns that into its own error). */
time_t mktime(struct tm *tm) {
  long long y = (long long)tm->tm_year + 1900;
  long long mon = tm->tm_mon; /* 0-based, may be out of range */
  y += mon / 12;
  mon %= 12;
  if (mon < 0) {
    mon += 12;
    y -= 1;
  }
  /* overflow guard: loslib probes huge years for its out-of-bound checks */
  if (y < -100000000LL || y > 100000000LL) return (time_t)-1;
  long long days = days_from_civil(y, (int)mon + 1, 1) + (tm->tm_mday - 1);
  long long secs =
      ((days * 24 + tm->tm_hour) * 60 + tm->tm_min) * 60LL + tm->tm_sec;
  to_tm((time_t)secs, tm); /* normalize the caller's fields, like the real mktime */
  return (time_t)secs;
}

long clock(void) { return 0; } /* no ambient clock; os.clock() reads 0.0 deterministically */

/* ---- strftime: the ANSI/C89 set loslib admits (LUA_STRFTIMEOPTIONS "aAbBcdHIjmMpSUwWxXyYzZ%") ---- */

static const char *WDAY_AB[] = {"Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"};
static const char *WDAY[] = {"Sunday", "Monday",   "Tuesday", "Wednesday",
                             "Thursday", "Friday", "Saturday"};
static const char *MON_AB[] = {"Jan", "Feb", "Mar", "Apr", "May", "Jun",
                               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"};
static const char *MON[] = {"January", "February", "March",     "April",   "May",      "June",
                            "July",    "August",   "September", "October", "November", "December"};

typedef struct Out {
  char *s;
  unsigned long n, cap;
} Out;

static void oc(Out *o, char c) {
  if (o->n < o->cap) o->s[o->n] = c;
  o->n++;
}
static void os_(Out *o, const char *s) {
  for (; *s; s++) oc(o, *s);
}
static void o2(Out *o, int v, char pad) { /* 2-wide, zero- or space-padded */
  oc(o, v >= 10 ? (char)('0' + v / 10 % 10) : pad);
  oc(o, (char)('0' + v % 10));
}
static void onum(Out *o, long long v, int width) { /* zero-padded decimal */
  char tmp[24];
  int i = 0, neg = v < 0;
  unsigned long long u = neg ? (unsigned long long)(-v) : (unsigned long long)v;
  do { tmp[i++] = (char)('0' + u % 10); u /= 10; } while (u);
  if (neg) tmp[i++] = '-';
  while (i < width) tmp[i++] = '0';
  while (i) oc(o, tmp[--i]);
}

/* week number: %U (week starts Sunday) / %W (starts Monday) */
static int weeknum(const struct tm *t, int wstart) {
  int wday = (t->tm_wday - wstart + 7) % 7;
  return (t->tm_yday + 7 - wday) / 7;
}

static void one_spec(Out *o, char c, const struct tm *t) {
  switch (c) {
    case 'a': os_(o, WDAY_AB[t->tm_wday % 7]); break;
    case 'A': os_(o, WDAY[t->tm_wday % 7]); break;
    case 'b': os_(o, MON_AB[t->tm_mon % 12]); break;
    case 'B': os_(o, MON[t->tm_mon % 12]); break;
    case 'c': /* C locale: "%a %b %e %H:%M:%S %Y" */
      os_(o, WDAY_AB[t->tm_wday % 7]); oc(o, ' ');
      os_(o, MON_AB[t->tm_mon % 12]); oc(o, ' ');
      o2(o, t->tm_mday, ' '); oc(o, ' ');
      o2(o, t->tm_hour, '0'); oc(o, ':');
      o2(o, t->tm_min, '0'); oc(o, ':');
      o2(o, t->tm_sec, '0'); oc(o, ' ');
      onum(o, t->tm_year + 1900LL, 4);
      break;
    case 'd': o2(o, t->tm_mday, '0'); break;
    case 'H': o2(o, t->tm_hour, '0'); break;
    case 'I': o2(o, t->tm_hour % 12 == 0 ? 12 : t->tm_hour % 12, '0'); break;
    case 'j': onum(o, t->tm_yday + 1, 3); break;
    case 'm': o2(o, t->tm_mon + 1, '0'); break;
    case 'M': o2(o, t->tm_min, '0'); break;
    case 'p': os_(o, t->tm_hour < 12 ? "AM" : "PM"); break;
    case 'S': o2(o, t->tm_sec, '0'); break;
    case 'U': o2(o, weeknum(t, 0), '0'); break;
    case 'w': oc(o, (char)('0' + t->tm_wday % 7)); break;
    case 'W': o2(o, weeknum(t, 1), '0'); break;
    case 'x': /* C locale: "%m/%d/%y" */
      o2(o, t->tm_mon + 1, '0'); oc(o, '/');
      o2(o, t->tm_mday, '0'); oc(o, '/');
      o2(o, (t->tm_year + 1900) % 100, '0');
      break;
    case 'X': /* C locale: "%H:%M:%S" */
      o2(o, t->tm_hour, '0'); oc(o, ':');
      o2(o, t->tm_min, '0'); oc(o, ':');
      o2(o, t->tm_sec, '0');
      break;
    case 'y': o2(o, (t->tm_year + 1900) % 100, '0'); break;
    case 'Y': onum(o, t->tm_year + 1900LL, 1); break;
    case 'z': os_(o, "+0000"); break; /* UTC */
    case 'Z': os_(o, "UTC"); break;
    case '%': oc(o, '%'); break;
    default: /* loslib validated against LUA_STRFTIMEOPTIONS, so this is unreachable */
      oc(o, '%');
      oc(o, c);
  }
}

unsigned long strftime(char *s, unsigned long max, const char *fmt, const struct tm *t) {
  Out o = {s, 0, max};
  for (; *fmt; fmt++) {
    if (*fmt == '%' && fmt[1]) {
      fmt++;
      one_spec(&o, *fmt, t);
    } else {
      oc(&o, *fmt);
    }
  }
  if (o.n < max) {
    s[o.n] = 0;
    return o.n;
  }
  return 0; /* didn't fit — ANSI: return 0, contents undefined */
}
