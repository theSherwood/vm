/* SQLite Phase A — the in-memory SQL-engine capstone (LLVM.md §8 ladder #5).
 *
 * Compiles the unmodified SQLite amalgamation (fetched at test time — public domain, ~9 MB, not
 * vendored; see `demo_sqlite_vs_native` in svm-llvm's translate.rs) into a self-contained guest
 * program: open a `:memory:` database, run a breadth SQL script (DDL, recursive CTE inserts,
 * indexes, joins, aggregates, window functions, string/CASE/NULL semantics, floats through
 * SQLite's own %!.15g formatter, date/time off the fixed VFS clock, transactions, PRAGMA
 * integrity_check, and a deliberate error), printing every row. The differential builds this same
 * file natively with `cc` and asserts byte-identical stdout — SQLite's ≈600:1 test-to-code ratio
 * makes agreement here the gold-standard whole-program shakedown.
 *
 * Everything nondeterministic is pinned in the `SQLITE_OS_OTHER=1` VFS below: xRandomness is a
 * fixed-seed LCG, xCurrentTime/Int64 a fixed instant, xSleep a no-op — so `random()` and
 * `datetime('now')` agree between the native and SVM runs. `:memory:` + `SQLITE_TEMP_STORE=3`
 * means no file I/O at all: xOpen fail-closes (SQLITE_CANTOPEN), proving no path sneaks to disk.
 */

/* ---- build configuration (single source of truth for BOTH the native and bitcode builds) ---- */
#define SQLITE_THREADSAFE 0          /* single-threaded guest — no pthread surface */
#define SQLITE_OMIT_LOAD_EXTENSION 1 /* no dlopen */
#define SQLITE_OMIT_DEPRECATED 1
#define SQLITE_OMIT_WAL 1            /* file-backed WAL/shm is Phase B */
#define SQLITE_OMIT_LOCALTIME 1      /* no host tz reads — UTC only, off the fixed VFS clock */
#define SQLITE_TEMP_STORE 3          /* temp tables/indices always in memory */
#define SQLITE_OS_OTHER 1            /* we provide sqlite3_os_init (the deterministic VFS below) */
#define SQLITE_DEFAULT_MEMSTATUS 0
#define SQLITE_MAX_MMAP_SIZE 0       /* no mmap code paths */
#define SQLITE_DQS 0
/* Needs SQLite >= 3.47: earlier amalgamations carried `long double` literals in sqlite3FpDecode
 * (x86_fp80 in the IR — outside the on-ramp's f64 world); 3.47+ replaced that path with Dekker
 * double-double arithmetic, so the whole build is f64-clean. */
#define NDEBUG 1

#include "sqlite3.c"

/* ---- deterministic minimal VFS (SQLITE_OS_OTHER) ------------------------------------------- */

/* xRandomness: a fixed-seed SplitMix64 — identical bytes native vs SVM, so random() agrees. */
static int demoRandomness(sqlite3_vfs *unused, int n, char *out) {
  static unsigned long long s = 0x9E3779B97F4A7C15ull;
  (void)unused;
  for (int i = 0; i < n; i++) {
    s += 0x9E3779B97F4A7C15ull;
    unsigned long long z = s;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    out[i] = (char)(z >> 33);
  }
  return n;
}

static int demoSleep(sqlite3_vfs *unused, int us) {
  (void)unused;
  return us;
}

/* A fixed instant: 2024-01-01 00:00:00 UTC (Julian day 2460310.5). */
static int demoCurrentTime(sqlite3_vfs *unused, double *t) {
  (void)unused;
  *t = 2460310.5;
  return SQLITE_OK;
}

static int demoCurrentTimeInt64(sqlite3_vfs *unused, sqlite3_int64 *t) {
  (void)unused;
  *t = 212570827200000LL; /* 2460310.5 julian days * 86400000 ms */
  return SQLITE_OK;
}

/* :memory: + TEMP_STORE=3 must never open a file — fail closed if SQLite tries. */
static int demoOpen(sqlite3_vfs *unused, const char *name, sqlite3_file *f, int flags, int *out) {
  (void)unused;
  (void)name;
  (void)f;
  (void)flags;
  (void)out;
  return SQLITE_CANTOPEN;
}

static int demoDelete(sqlite3_vfs *unused, const char *name, int sync) {
  (void)unused;
  (void)name;
  (void)sync;
  return SQLITE_OK;
}

static int demoAccess(sqlite3_vfs *unused, const char *name, int flags, int *out) {
  (void)unused;
  (void)name;
  (void)flags;
  *out = 0;
  return SQLITE_OK;
}

static int demoFullPathname(sqlite3_vfs *unused, const char *name, int n, char *out) {
  (void)unused;
  int i = 0;
  for (; name[i] && i < n - 1; i++) out[i] = name[i];
  out[i] = 0;
  return SQLITE_OK;
}

static int demoGetLastError(sqlite3_vfs *unused, int n, char *out) {
  (void)unused;
  (void)n;
  (void)out;
  return 0;
}

SQLITE_API int sqlite3_os_init(void) {
  static sqlite3_vfs demoVfs = {
      2,                    /* iVersion (through xCurrentTimeInt64) */
      sizeof(sqlite3_file), /* szOsFile (never actually opened) */
      512,                  /* mxPathname */
      0,                    /* pNext */
      "svm-demo",           /* zName */
      0,                    /* pAppData */
      demoOpen,
      demoDelete,
      demoAccess,
      demoFullPathname,
      0, /* xDlOpen (OMIT_LOAD_EXTENSION) */
      0, /* xDlError */
      0, /* xDlSym */
      0, /* xDlClose */
      demoRandomness,
      demoSleep,
      demoCurrentTime,
      demoGetLastError,
      demoCurrentTimeInt64,
      0, /* xSetSystemCall */
      0, /* xGetSystemCall */
      0, /* xNextSystemCall */
  };
  return sqlite3_vfs_register(&demoVfs, 1);
}

SQLITE_API int sqlite3_os_end(void) { return SQLITE_OK; }

/* ---- the driver ------------------------------------------------------------------------------ */

static int print_row(void *unused, int argc, char **argv, char **col) {
  (void)unused;
  (void)col;
  for (int i = 0; i < argc; i++) {
    printf("%s%s", i ? "|" : "", argv[i] ? argv[i] : "NULL");
  }
  printf("\n");
  return 0;
}

/* The breadth script. Every SELECT is ORDER'd (or single-row) so output is deterministic. */
static const char *SCRIPT[] = {
    "CREATE TABLE t(a INTEGER PRIMARY KEY, b TEXT, c REAL);",
    /* recursive-CTE fill: the VDBE loop + SQLite's own printf() SQL function */
    "INSERT INTO t SELECT n, printf('row%03d', n), n * 0.5 FROM"
    " (WITH RECURSIVE s(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM s WHERE n < 50)"
    "  SELECT n FROM s);",
    "CREATE INDEX ti ON t(b);",
    "SELECT count(*), sum(a), sum(c), avg(c), min(b), max(b) FROM t;",
    "SELECT a, b, c FROM t WHERE a BETWEEN 8 AND 12 ORDER BY a;",
    "SELECT a, b FROM t WHERE b LIKE 'row04%' ORDER BY b DESC;",
    "SELECT a % 7 AS g, count(*), sum(c) FROM t GROUP BY g HAVING count(*) > 6 ORDER BY g;",
    /* self-join through the index */
    "SELECT x.a, y.b FROM t x JOIN t y ON y.a = x.a + 40 WHERE x.a <= 5 ORDER BY x.a;",
    /* strings, CASE, NULL semantics */
    "SELECT upper(b), substr(b, 4), length(b) || '!', CASE WHEN a % 2 THEN 'odd' ELSE 'even' END"
    " FROM t WHERE a < 4 ORDER BY a;",
    "SELECT coalesce(NULL, NULL, 'fallback'), NULL IS NULL, typeof(NULL), typeof(a), typeof(c),"
    " typeof(b) FROM t LIMIT 1;",
    /* float formatting through SQLite's %!.15g + carries/rounding */
    "SELECT 1.0 / 3.0, round(2.675, 2), abs(-4.25), 1e300, 1.5e-8, 0.1 + 0.2;",
    "SELECT CAST('42abc' AS INTEGER), CAST(3.99 AS INTEGER), CAST(7 AS TEXT), CAST(x'414243' AS TEXT);",
    /* window function over the index order */
    "SELECT a, sum(a) OVER (ORDER BY a ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) FROM t"
    " WHERE a <= 6 ORDER BY a;",
    /* recursive CTE compute: fib */
    "WITH RECURSIVE fib(n, x, y) AS (SELECT 1, 0, 1 UNION ALL SELECT n + 1, y, x + y FROM fib"
    " WHERE n < 20) SELECT n, x FROM fib WHERE n % 5 = 0 ORDER BY n;",
    /* the fixed VFS clock: 'now' is pinned to 2024-01-01T00:00:00Z on both sides */
    "SELECT datetime('now'), date('now', '+1 month'), strftime('%Y-%m-%d %H:%M', 'now', '90 minutes');",
    "SELECT julianday('2024-01-01 12:00:00'), unixepoch('2024-01-01');",
    /* the fixed VFS randomness: identical PRNG stream on both sides */
    "SELECT random() % 1000, random() % 1000;",
    "SELECT hex(randomblob(8));",
    /* mutation + transactions */
    "UPDATE t SET c = c * 2, b = b || '+' WHERE a % 10 = 0;",
    "SELECT a, b, c FROM t WHERE a % 10 = 0 ORDER BY a;",
    "BEGIN; DELETE FROM t WHERE a > 45; ROLLBACK;",
    "SELECT count(*) FROM t;",
    "DELETE FROM t WHERE a > 45;",
    "SELECT count(*), max(a) FROM t;",
    /* blobs + quoting */
    "SELECT quote(b), quote(x'00ff'), quote(NULL), quote(-0.0) FROM t WHERE a = 1;",
    /* subqueries + EXISTS */
    "SELECT a FROM t WHERE a IN (SELECT a + 1 FROM t WHERE a < 4) ORDER BY a;",
    "SELECT EXISTS(SELECT 1 FROM t WHERE b = 'row007'), NOT EXISTS(SELECT 1 FROM t WHERE a = 999);",
    "PRAGMA integrity_check;",
    /* a deliberate error: the ERR path must byte-match too */
    "SELECT * FROM no_such_table;",
    "SELECT 'done', sqlite_version() LIKE '3.%';",
};

int main(void) {
  sqlite3 *db = 0;
  int rc = sqlite3_open(":memory:", &db);
  if (rc != SQLITE_OK) {
    printf("open failed: %d\n", rc);
    return 1;
  }
  for (unsigned i = 0; i < sizeof(SCRIPT) / sizeof(SCRIPT[0]); i++) {
    char *err = 0;
    printf("-- %u\n", i);
    rc = sqlite3_exec(db, SCRIPT[i], print_row, 0, &err);
    if (rc != SQLITE_OK) {
      printf("ERR %d: %s\n", rc, err ? err : "?");
      sqlite3_free(err);
    }
  }
  sqlite3_close(db);
  return 0;
}
