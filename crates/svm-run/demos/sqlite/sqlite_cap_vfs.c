/* SQLite Phase B — disk-backed persistence through the **Fs capability** (LLVM.md §8, the SQLite
 * north star's second half). The same unmodified amalgamation as Phase A, but the database is a
 * real *file* (`test.db`), and in the guest build every byte of it flows through the embedder-granted
 * `fs` capability: a guest-side `sqlite3_vfs` (the "guest VFS shim" LLVM.md planned) bridges
 * xOpen/xRead/xWrite/xTruncate/xSync/xFileSize/xDelete/xAccess to `__vm_cap_resolve("fs")` +
 * `__vm_host_call` — the §7 host-defined-capability surface, exactly how Lua's `io` runs. Zero
 * ambient authority: no capability, no bytes.
 *
 * Two builds from this one file:
 *  - **guest** (`-DSVM_GUEST`): `SQLITE_OS_OTHER=1` + the capability VFS below.
 *  - **native oracle**: SQLite's stock unix VFS writing `test.db` in the CWD.
 * Same driver, same SQL → byte-identical stdout; and because both write the same on-disk *format*,
 * a `test.db` created by the guest through `host_fs` is readable by the native build (the test's
 * cross-implementation proof).
 *
 * Driver modes: no args = **create** (build the DB, close, reopen, verify — persistence inside one
 * run); `verify` = open an **existing** `test.db` read-only-ish and run the same checks (used by
 * the native binary against a guest-written file).
 */

/* ---- build configuration (shared by both builds) -------------------------------------------- */
#define SQLITE_THREADSAFE 0
#define SQLITE_OMIT_LOAD_EXTENSION 1
#define SQLITE_OMIT_DEPRECATED 1
#define SQLITE_OMIT_WAL 1 /* WAL needs shared memory; the rollback journal covers Phase B */
#define SQLITE_OMIT_LOCALTIME 1
#define SQLITE_TEMP_STORE 3
#define SQLITE_DEFAULT_MEMSTATUS 0
#define SQLITE_MAX_MMAP_SIZE 0
#define SQLITE_DQS 0
#define NDEBUG 1
#ifdef SVM_GUEST
#define SQLITE_OS_OTHER 1
#endif

#include "sqlite3.c"

#ifdef SVM_GUEST
/* ---- the capability VFS ---------------------------------------------------------------------- */

extern int __vm_cap_resolve(const char *name, long len);
extern long __vm_host_call(int h, int op, long a, long b, long c, long d);

/* svm-run fs op protocol (crates/svm-run/src/fs.rs). */
enum { FS_OPEN = 0, FS_READ, FS_WRITE, FS_SEEK, FS_CLOSE, FS_REMOVE, FS_RENAME, FS_TRUNCATE, FS_SYNC };
enum { FS_O_READ = 1, FS_O_WRITE = 2, FS_O_APPEND = 4, FS_O_TRUNC = 8, FS_O_CREATE = 16 };
#define FS_ENOENT 2

static int g_fs = -1;
static long fscall(int op, long a, long b, long c, long d) {
  return __vm_host_call(g_fs, op, a, b, c, d);
}

static long cstrlen(const char *s) {
  long n = 0;
  while (s[n]) n++;
  return n;
}

typedef struct CapFile {
  sqlite3_file base;
  long fd;
} CapFile;

static int capClose(sqlite3_file *f) {
  CapFile *c = (CapFile *)f;
  return fscall(FS_CLOSE, c->fd, 0, 0, 0) == 0 ? SQLITE_OK : SQLITE_IOERR_CLOSE;
}

static int capRead(sqlite3_file *f, void *buf, int amt, sqlite3_int64 ofst) {
  CapFile *c = (CapFile *)f;
  if (fscall(FS_SEEK, c->fd, 0, (long)ofst, 0) != (long)ofst) return SQLITE_IOERR_READ;
  long got = 0;
  while (got < amt) {
    long n = fscall(FS_READ, c->fd, (long)buf + got, amt - got, 0);
    if (n < 0) return SQLITE_IOERR_READ;
    if (n == 0) break; /* EOF */
    got += n;
  }
  if (got < amt) {
    /* SQLite requires the unread tail zero-filled on a short read. */
    char *p = (char *)buf;
    for (long i = got; i < amt; i++) p[i] = 0;
    return SQLITE_IOERR_SHORT_READ;
  }
  return SQLITE_OK;
}

static int capWrite(sqlite3_file *f, const void *buf, int amt, sqlite3_int64 ofst) {
  CapFile *c = (CapFile *)f;
  if (fscall(FS_SEEK, c->fd, 0, (long)ofst, 0) != (long)ofst) return SQLITE_IOERR_WRITE;
  long put = 0;
  while (put < amt) {
    long n = fscall(FS_WRITE, c->fd, (long)buf + put, amt - put, 0);
    if (n <= 0) return SQLITE_IOERR_WRITE;
    put += n;
  }
  return SQLITE_OK;
}

static int capTruncate(sqlite3_file *f, sqlite3_int64 size) {
  CapFile *c = (CapFile *)f;
  return fscall(FS_TRUNCATE, c->fd, (long)size, 0, 0) == 0 ? SQLITE_OK : SQLITE_IOERR_TRUNCATE;
}

static int capSync(sqlite3_file *f, int flags) {
  CapFile *c = (CapFile *)f;
  (void)flags;
  return fscall(FS_SYNC, c->fd, 0, 0, 0) == 0 ? SQLITE_OK : SQLITE_IOERR_FSYNC;
}

static int capFileSize(sqlite3_file *f, sqlite3_int64 *pSize) {
  CapFile *c = (CapFile *)f;
  long end = fscall(FS_SEEK, c->fd, 2, 0, 0);
  if (end < 0) return SQLITE_IOERR_FSTAT;
  *pSize = end; /* xRead/xWrite always re-seek, so the moved cursor is harmless */
  return SQLITE_OK;
}

/* Single-connection Phase B: locking is vacuously satisfied. */
static int capLock(sqlite3_file *f, int level) {
  (void)f;
  (void)level;
  return SQLITE_OK;
}
static int capUnlock(sqlite3_file *f, int level) {
  (void)f;
  (void)level;
  return SQLITE_OK;
}
static int capCheckReservedLock(sqlite3_file *f, int *pResOut) {
  (void)f;
  *pResOut = 0;
  return SQLITE_OK;
}
static int capFileControl(sqlite3_file *f, int op, void *pArg) {
  (void)f;
  (void)op;
  (void)pArg;
  return SQLITE_NOTFOUND;
}
static int capSectorSize(sqlite3_file *f) {
  (void)f;
  return 4096; /* SQLITE_DEFAULT_SECTOR_SIZE — journal padding matches the stock unix VFS */
}
static int capDeviceCharacteristics(sqlite3_file *f) {
  (void)f;
  return 0;
}

static const sqlite3_io_methods cap_io = {
    1, /* iVersion (no shm/mmap — OMIT_WAL + MAX_MMAP_SIZE=0) */
    capClose,
    capRead,
    capWrite,
    capTruncate,
    capSync,
    capFileSize,
    capLock,
    capUnlock,
    capCheckReservedLock,
    capFileControl,
    capSectorSize,
    capDeviceCharacteristics,
    0, 0, 0, 0, 0, 0, /* v2/v3 methods absent */
};

static int capOpen(sqlite3_vfs *v, const char *zName, sqlite3_file *f, int flags, int *pOutFlags) {
  (void)v;
  CapFile *c = (CapFile *)f;
  c->base.pMethods = 0;
  if (!zName) return SQLITE_CANTOPEN; /* anonymous temp files can't exist (TEMP_STORE=3) */
  long fl = FS_O_READ;
  if (flags & SQLITE_OPEN_READWRITE) fl |= FS_O_WRITE;
  if (flags & SQLITE_OPEN_CREATE) fl |= FS_O_CREATE;
  long fd = fscall(FS_OPEN, (long)zName, cstrlen(zName), fl, 0);
  if (fd < 0) return SQLITE_CANTOPEN;
  c->fd = fd;
  c->base.pMethods = &cap_io;
  if (pOutFlags) *pOutFlags = flags;
  return SQLITE_OK;
}

static int capDelete(sqlite3_vfs *v, const char *zName, int syncDir) {
  (void)v;
  (void)syncDir;
  long rc = fscall(FS_REMOVE, (long)zName, cstrlen(zName), 0, 0);
  if (rc == 0) return SQLITE_OK;
  return rc == -FS_ENOENT ? SQLITE_IOERR_DELETE_NOENT : SQLITE_IOERR_DELETE;
}

static int capAccess(sqlite3_vfs *v, const char *zName, int flags, int *pResOut) {
  (void)v;
  (void)flags;
  /* Existence == openable-for-read; the capability has no separate stat surface. */
  long fd = fscall(FS_OPEN, (long)zName, cstrlen(zName), FS_O_READ, 0);
  if (fd >= 0) {
    fscall(FS_CLOSE, fd, 0, 0, 0);
    *pResOut = 1;
  } else {
    *pResOut = 0;
  }
  return SQLITE_OK;
}

static int capFullPathname(sqlite3_vfs *v, const char *zName, int nOut, char *zOut) {
  (void)v;
  int i = 0;
  for (; zName[i] && i < nOut - 1; i++) zOut[i] = zName[i];
  zOut[i] = 0;
  return SQLITE_OK;
}

/* The deterministic randomness/clock trio (same values as Phase A, so `random()` and
 * `datetime('now')` agree with the native oracle, whose VFS these cannot pin — the driver's SQL
 * below simply avoids 'now'/random() where the native unix VFS would diverge). */
static int capRandomness(sqlite3_vfs *v, int n, char *out) {
  static unsigned long long s = 0x9E3779B97F4A7C15ull;
  (void)v;
  for (int i = 0; i < n; i++) {
    s += 0x9E3779B97F4A7C15ull;
    unsigned long long z = s;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    out[i] = (char)(z >> 33);
  }
  return n;
}
static int capSleep(sqlite3_vfs *v, int us) {
  (void)v;
  return us;
}
static int capCurrentTime(sqlite3_vfs *v, double *t) {
  (void)v;
  *t = 2460310.5;
  return SQLITE_OK;
}
static int capCurrentTimeInt64(sqlite3_vfs *v, sqlite3_int64 *t) {
  (void)v;
  *t = 212570827200000LL;
  return SQLITE_OK;
}
static int capGetLastError(sqlite3_vfs *v, int n, char *out) {
  (void)v;
  (void)n;
  (void)out;
  return 0;
}

SQLITE_API int sqlite3_os_init(void) {
  static sqlite3_vfs capVfs = {
      2,               /* iVersion */
      sizeof(CapFile), /* szOsFile */
      512,             /* mxPathname */
      0,
      "svm-fs-cap",
      0,
      capOpen,
      capDelete,
      capAccess,
      capFullPathname,
      0, 0, 0, 0, /* no dlopen */
      capRandomness,
      capSleep,
      capCurrentTime,
      capGetLastError,
      capCurrentTimeInt64,
      0, 0, 0,
  };
  g_fs = __vm_cap_resolve("fs", 2);
  if (g_fs < 0) return SQLITE_ERROR; /* no capability granted → no filesystem, period */
  return sqlite3_vfs_register(&capVfs, 1);
}

SQLITE_API int sqlite3_os_end(void) { return SQLITE_OK; }
#endif /* SVM_GUEST */

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

static int run_script(sqlite3 *db, const char **stmts, unsigned n, unsigned base) {
  for (unsigned i = 0; i < n; i++) {
    char *err = 0;
    printf("-- %u\n", base + i);
    int rc = sqlite3_exec(db, stmts[i], print_row, 0, &err);
    if (rc != SQLITE_OK) {
      printf("ERR %d: %s\n", rc, err ? err : "?");
      sqlite3_free(err);
    }
  }
  return 0;
}

/* Stage 1: build the on-disk database. The journal (default DELETE mode) is created, written,
 * and deleted through the VFS on every transaction — including an explicit ROLLBACK, which
 * *replays* the journal back into the database file. */
static const char *CREATE_SCRIPT[] = {
    "CREATE TABLE kv(k INTEGER PRIMARY KEY, v TEXT, w REAL);",
    "INSERT INTO kv SELECT n, printf('value-%04d', n), n * 1.5 FROM"
    " (WITH RECURSIVE s(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM s WHERE n < 200)"
    "  SELECT n FROM s);",
    "CREATE INDEX kv_v ON kv(v);",
    "UPDATE kv SET v = v || '*' WHERE k % 17 = 0;",
    "DELETE FROM kv WHERE k % 31 = 0;",
    "BEGIN; DELETE FROM kv; ROLLBACK;", /* journal replay: the data must survive */
    "SELECT count(*), sum(k), sum(w) FROM kv;",
};

/* Stage 2 / verify: the read-back checks — run after reopening the persisted file. */
static const char *VERIFY_SCRIPT[] = {
    "SELECT count(*), sum(k), max(k) FROM kv;",
    "SELECT k, v, w FROM kv WHERE k IN (1, 17, 100, 199) ORDER BY k;",
    "SELECT v FROM kv WHERE v LIKE 'value-002%' ORDER BY v LIMIT 5;",
    "SELECT count(*) FROM kv WHERE v LIKE '%*';",
    "PRAGMA integrity_check;",
};

#define NSTMT(a) (sizeof(a) / sizeof((a)[0]))

int main(int argc, char **argv) {
  sqlite3 *db = 0;
  int verify_only = argc > 1 && argv[1][0] == 'v';
  if (!verify_only) {
    if (sqlite3_open("test.db", &db) != SQLITE_OK) {
      printf("open(create) failed\n");
      return 1;
    }
    run_script(db, CREATE_SCRIPT, NSTMT(CREATE_SCRIPT), 0);
    sqlite3_close(db);
    printf("-- reopen\n");
  }
  if (sqlite3_open("test.db", &db) != SQLITE_OK) {
    printf("open(verify) failed\n");
    return 1;
  }
  run_script(db, VERIFY_SCRIPT, NSTMT(VERIFY_SCRIPT), 100);
  sqlite3_close(db);
  return 0;
}
