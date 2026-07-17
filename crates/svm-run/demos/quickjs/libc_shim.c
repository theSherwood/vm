/* QuickJS libc shim — the small surface the on-ramp neither synthesizes nor covers via the reused
 * Postgres printf engine (`printf_shim.c`) and the guest `strtod` (`demos/strtod/strtod.c`).
 *
 * Linked into the guest build (and the native oracle, so the differential stays honest). Everything
 * here is ordinary guest C; a guest definition shadows the on-ramp's would-be trap stub. See the
 * QuickJS README for the full gap inventory.
 */
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>

extern void exit(int);

/* --- FP rounding mode (fenv) -----------------------------------------------------------------
 * The SVM float ops are round-to-nearest-even only, with no rounding-mode primitive. QuickJS's
 * shortest-round-trip Number->string (`js_ecvt1`) toggles FE_DOWNWARD/FE_UPWARD to find the
 * shortest decimal; that directed-rounding path is a KNOWN FOLLOW-UP (needs an SVM rounding-mode
 * op or a directed-rounding-free dtoa). Fixed-precision paths (`toFixed`/`toPrecision`) only ever
 * use FE_TONEAREST, which these honor exactly. `<fenv.h>` on x86: FE_TONEAREST==0. */
int fegetround(void) { return 0 /* FE_TONEAREST */; }
int fesetround(int mode) { return mode == 0 ? 0 : -1 /* refuse directed modes, don't lie */; }

/* --- rounding to integer ---------------------------------------------------------------------- */
long lrint(double x) { return (long)__builtin_rint(x); }
long long llrint(double x) { return (long long)__builtin_rint(x); }

/* --- strtol (+ the C23-renamed alias glibc >=2.38 emits) -------------------------------------- */
long strtol(const char *s, char **end, int base) {
    const char *p = s;
    while (*p == ' ' || (*p >= '\t' && *p <= '\r')) p++;
    int neg = 0;
    if (*p == '+' || *p == '-') neg = (*p++ == '-');
    if ((base == 0 || base == 16) && p[0] == '0' && (p[1] == 'x' || p[1] == 'X')) {
        p += 2;
        base = 16;
    } else if (base == 0) {
        base = (p[0] == '0') ? 8 : 10;
    }
    unsigned long acc = 0;
    int any = 0;
    for (;; p++) {
        int c = (unsigned char)*p, d;
        if (c >= '0' && c <= '9') d = c - '0';
        else if (c >= 'a' && c <= 'z') d = c - 'a' + 10;
        else if (c >= 'A' && c <= 'Z') d = c - 'A' + 10;
        else break;
        if (d >= base) break;
        acc = acc * (unsigned long)base + (unsigned long)d;
        any = 1;
    }
    if (end) *end = (char *)(any ? p : s);
    long v = (long)acc;
    return neg ? -v : v;
}
long __isoc23_strtol(const char *s, char **end, int base) { return strtol(s, end, base); }

/* --- string ops the on-ramp does not synthesize --------------------------------------------------
 * (it already provides strlen/strcpy/strcmp/strncmp/strchr/strrchr/strspn/strcspn/memchr/bcmp). */
char *strcat(char *dst, const char *src) {
    char *d = dst;
    while (*d) d++;
    while ((*d++ = *src++)) {
    }
    return dst;
}

/* --- time: deterministic stubs (fixed epoch) --------------------------------------------------
 * QuickJS's Date / timing reads the wall clock; a differential-vs-native demo must be deterministic
 * (like SQLite's fixed-clock OS_OTHER VFS), and the driver's output does not depend on the time. A
 * real clock would need a host time capability. struct layouts match the x86-64 Linux ABI. */
struct __shim_timeval {
    long tv_sec;
    long tv_usec;
};
struct __shim_timespec {
    long tv_sec;
    long tv_nsec;
};
int gettimeofday(struct __shim_timeval *tv, void *tz) {
    (void)tz;
    if (tv) {
        tv->tv_sec = 0;
        tv->tv_usec = 0;
    }
    return 0;
}
int clock_gettime(int clk, struct __shim_timespec *ts) {
    (void)clk;
    if (ts) {
        ts->tv_sec = 0;
        ts->tv_nsec = 0;
    }
    return 0;
}
struct __shim_tm {
    int tm_sec, tm_min, tm_hour, tm_mday, tm_mon, tm_year, tm_wday, tm_yday, tm_isdst;
    long tm_gmtoff;
    const char *tm_zone;
};
struct __shim_tm *localtime_r(const long *t, struct __shim_tm *out) {
    (void)t;
    if (out) {
        out->tm_sec = out->tm_min = out->tm_hour = 0;
        out->tm_mday = 1; /* the epoch: 1970-01-01 00:00:00 UTC, Thursday */
        out->tm_mon = 0;
        out->tm_year = 70;
        out->tm_wday = 4;
        out->tm_yday = 0;
        out->tm_isdst = 0;
        out->tm_gmtoff = 0;
        out->tm_zone = 0;
    }
    return out;
}

/* --- pthreads: single-threaded no-op stubs -----------------------------------------------------
 * Referenced by QuickJS's Atomics.wait/notify (js_atomics_*), which a single-threaded eval never
 * exercises; the symbols must exist for translation. A single guest vCPU means no contention. */
int pthread_mutex_lock(void *m) {
    (void)m;
    return 0;
}
int pthread_mutex_unlock(void *m) {
    (void)m;
    return 0;
}
int pthread_cond_init(void *c, void *a) {
    (void)c;
    (void)a;
    return 0;
}
int pthread_cond_destroy(void *c) {
    (void)c;
    return 0;
}
int pthread_cond_signal(void *c) {
    (void)c;
    return 0;
}
int pthread_cond_wait(void *c, void *m) {
    (void)c;
    (void)m;
    return 0;
}
int pthread_cond_timedwait(void *c, void *m, const void *t) {
    (void)c;
    (void)m;
    (void)t;
    return 0;
}

/* --- misc ------------------------------------------------------------------------------------- */
void abort(void) {
    exit(134); /* 128 + SIGABRT */
    for (;;) {
    }
}

/* QuickJS uses this only for GC memory accounting (js_malloc_usable_size) — not output-affecting.
 * The on-ramp's allocator keeps a private size header but exposes no getter, so report 0 (QuickJS
 * treats an underestimate conservatively: it may realloc sooner, never incorrectly). */
size_t malloc_usable_size(void *p) {
    (void)p;
    return 0;
}
