/* SQLite interactive editor guest — the browser-playground REPL. Reads a SQL script from **stdin**
 * (the Stream.read capability), runs it against an in-memory database, and prints each statement's
 * result table (column headers + rows), DML change counts, and errors to stdout. The playground pipes
 * the SQL editor's text in as stdin, so a user can write and run their own SQL client-side.
 *
 * Same unmodified SQLite amalgamation + deterministic SQLITE_OS_OTHER VFS as `sqlite_demo.c` (fetched
 * at build time, public domain). :memory: only — no file I/O; xOpen fail-closes. Each Run is a fresh
 * database (stateless), matching the "run the whole buffer" editor model.
 */

/* ---- build configuration (identical to sqlite_demo.c) --------------------------------------- */
#define SQLITE_THREADSAFE 0
#define SQLITE_OMIT_LOAD_EXTENSION 1
#define SQLITE_OMIT_DEPRECATED 1
#define SQLITE_OMIT_WAL 1
#define SQLITE_OMIT_LOCALTIME 1
#define SQLITE_TEMP_STORE 3
#define SQLITE_OS_OTHER 1
#define SQLITE_DEFAULT_MEMSTATUS 0
#define SQLITE_MAX_MMAP_SIZE 0
#define SQLITE_DQS 0
#define NDEBUG 1

#include "sqlite3.c"

extern long read(int fd, void *buf, long n);

/* ---- deterministic minimal VFS (SQLITE_OS_OTHER), as in sqlite_demo.c ------------------------ */

static int replRandomness(sqlite3_vfs *unused, int n, char *out) {
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
static int replSleep(sqlite3_vfs *unused, int us) { (void)unused; return us; }
static int replCurrentTime(sqlite3_vfs *unused, double *t) { (void)unused; *t = 2460310.5; return SQLITE_OK; }
static int replCurrentTimeInt64(sqlite3_vfs *unused, sqlite3_int64 *t) { (void)unused; *t = 212570827200000LL; return SQLITE_OK; }
static int replOpen(sqlite3_vfs *u, const char *n, sqlite3_file *f, int fl, int *o) {
  (void)u; (void)n; (void)f; (void)fl; (void)o; return SQLITE_CANTOPEN;
}
static int replDelete(sqlite3_vfs *u, const char *n, int s) { (void)u; (void)n; (void)s; return SQLITE_OK; }
static int replAccess(sqlite3_vfs *u, const char *n, int fl, int *o) { (void)u; (void)n; (void)fl; *o = 0; return SQLITE_OK; }
static int replFullPathname(sqlite3_vfs *u, const char *n, int cap, char *o) {
  (void)u;
  int i = 0;
  for (; n[i] && i < cap - 1; i++) o[i] = n[i];
  o[i] = 0;
  return SQLITE_OK;
}
static int replGetLastError(sqlite3_vfs *u, int n, char *o) { (void)u; (void)n; (void)o; return 0; }

SQLITE_API int sqlite3_os_init(void) {
  static sqlite3_vfs replVfs = {
      2, sizeof(sqlite3_file), 512, 0, "svm-repl", 0,
      replOpen, replDelete, replAccess, replFullPathname,
      0, 0, 0, 0,
      replRandomness, replSleep, replCurrentTime, replGetLastError, replCurrentTimeInt64,
      0, 0, 0,
  };
  return sqlite3_vfs_register(&replVfs, 1);
}
SQLITE_API int sqlite3_os_end(void) { return SQLITE_OK; }

/* ---- the REPL driver ------------------------------------------------------------------------- */

static char sql[1 << 20]; /* up to 1 MiB of editor text */

/* Print one statement's results: column headers + rows (` | `-separated), or a change count for
 * DML/DDL, or `(no rows)`. Errors are printed by the caller from sqlite3_errmsg. */
static void run_stmt(sqlite3_stmt *stmt) {
  int ncol = sqlite3_column_count(stmt);
  int printed_header = 0, rows = 0, rc;
  while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
    if (!printed_header) {
      for (int i = 0; i < ncol; i++) printf("%s%s", i ? " | " : "", sqlite3_column_name(stmt, i));
      printf("\n");
      printed_header = 1;
    }
    for (int i = 0; i < ncol; i++) {
      const unsigned char *v = sqlite3_column_text(stmt, i);
      printf("%s%s", i ? " | " : "", v ? (const char *)v : "NULL");
    }
    printf("\n");
    rows++;
  }
  if (ncol == 0) {
    int ch = sqlite3_changes(sqlite3_db_handle(stmt));
    printf("ok (%d row%s changed)\n", ch, ch == 1 ? "" : "s");
  } else if (rows == 0) {
    printf("(no rows)\n");
  }
}

int main(void) {
  long len = 0;
  for (;;) {
    long r = read(0, sql + len, (long)sizeof(sql) - 1 - len);
    if (r <= 0) break;
    len += r;
    if (len >= (long)sizeof(sql) - 1) break;
  }
  sql[len] = 0;

  sqlite3 *db = 0;
  if (sqlite3_open(":memory:", &db) != SQLITE_OK) {
    printf("open failed: %s\n", sqlite3_errmsg(db));
    return 1;
  }
  const char *tail = sql;
  while (tail && *tail) {
    sqlite3_stmt *stmt = 0;
    const char *next = 0;
    int rc = sqlite3_prepare_v2(db, tail, -1, &stmt, &next);
    if (rc != SQLITE_OK) {
      printf("error: %s\n", sqlite3_errmsg(db));
      break;
    }
    if (stmt) {
      run_stmt(stmt);
      if (sqlite3_finalize(stmt) != SQLITE_OK) printf("error: %s\n", sqlite3_errmsg(db));
    }
    tail = next; /* a trailing comment / whitespace prepares to NULL — skip it */
  }
  sqlite3_close(db);
  return 0;
}
