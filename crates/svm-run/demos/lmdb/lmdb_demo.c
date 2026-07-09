/* LMDB — an embedded **memory-mapped** B-tree KV store in the sandbox (LLVM.md storage ladder, the
 * second storage shape after SQLite's read/write VFS).
 *
 * LMDB (OpenLDAP's Lightning MDB — the original mmap'd B-tree that libmdbx later hardened) is the
 * canonical "the data plane *is* the memory mapping" store: readers walk the B-tree straight out of
 * a file-backed mmap, no per-access syscalls. This demo drives it as a guest program whose mmap goes
 * through the granted **Fs capability's mmap surface** (`FS_MMAP`/`FS_MSYNC`/`FS_MUNMAP`,
 * `crates/svm-run/src/fs.rs`) — so the whole memory-mapped data plane flows through explicitly
 * granted authority, zero ambient access.
 *
 * The env is opened `MDB_NOSUBDIR | MDB_NOLOCK | MDB_WRITEMAP`:
 *   - NOSUBDIR: the database is a single file (`data.mdb`), not a directory + lock file;
 *   - NOLOCK:   single-process / single-threaded, so no reader-lock table (no second mapping, no
 *               robust pthread mutex) — the pthread surface degrades to link-time stubs;
 *   - WRITEMAP: the map is *writable* and every write lands in it, so the map is the single source
 *               of truth (coherent with our copy-in/flush-out mmap emulation — no separate pwrite
 *               path to keep consistent).
 *
 * Two builds from this one file:
 *   - guest (`-DSVM_GUEST`): the mmap/pthread/file shims below over the Fs capability;
 *   - native oracle: stock glibc — real mmap, writing `data.mdb` in the cwd.
 * Same driver, same operations → byte-identical stdout; and because LMDB's on-disk format is
 * identical either way, a `data.mdb` written by the guest is readable by native LMDB (the test's
 * cross-implementation proof).
 *
 * Driver modes: no args = create (fill, read back, cursor-scan, close, reopen, verify — persistence
 * inside one run); `verify` = open an existing `data.mdb` and run the read-back checks (used by the
 * native binary against a guest-written file, and vice versa).
 */

#include "lmdb.h"

extern int printf(const char *, ...);

#define CHECK(expr, msg)                                                                            \
  do {                                                                                             \
    int rc_ = (expr);                                                                              \
    if (rc_ != MDB_SUCCESS) {                                                                      \
      printf("ERR %s: %s\n", msg, mdb_strerror(rc_));                                              \
      return 2;                                                                                    \
    }                                                                                              \
  } while (0)

/* A deterministic key/value scheme: key = "k%05d", value = a length-varying blob derived from i. */
static void make_key(char *buf, int i) {
  const char *d = "0123456789";
  buf[0] = 'k';
  buf[1] = d[(i / 10000) % 10];
  buf[2] = d[(i / 1000) % 10];
  buf[3] = d[(i / 100) % 10];
  buf[4] = d[(i / 10) % 10];
  buf[5] = d[i % 10];
  buf[6] = 0;
}
static int make_val(char *buf, int i) {
  int n = 4 + (i % 29); /* 4..32 bytes */
  for (int j = 0; j < n; j++) buf[j] = (char)('A' + ((i * 7 + j * 3) % 26));
  return n;
}

#define NKEYS 500

static int fill(MDB_env *env) {
  MDB_txn *txn;
  MDB_dbi dbi;
  CHECK(mdb_txn_begin(env, 0, 0, &txn), "txn_begin(w)");
  CHECK(mdb_dbi_open(txn, 0, 0, &dbi), "dbi_open");
  char kb[8], vb[64];
  for (int i = 0; i < NKEYS; i++) {
    /* Insert in a scrambled order so the B-tree actually rebalances (not append-only). */
    int k = (i * 131 + 7) % NKEYS;
    make_key(kb, k);
    int vn = make_val(vb, k);
    MDB_val key = {6, kb}, val = {(size_t)vn, vb};
    CHECK(mdb_put(txn, dbi, &key, &val, 0), "put");
  }
  /* delete every 13th key so the tree has holes */
  for (int i = 0; i < NKEYS; i += 13) {
    make_key(kb, i);
    MDB_val key = {6, kb};
    int rc = mdb_del(txn, dbi, &key, 0);
    if (rc != MDB_SUCCESS && rc != MDB_NOTFOUND) {
      printf("ERR del: %s\n", mdb_strerror(rc));
      return 2;
    }
  }
  CHECK(mdb_txn_commit(txn), "txn_commit");
  return 0;
}

static int verify(MDB_env *env) {
  MDB_txn *txn;
  MDB_dbi dbi;
  CHECK(mdb_txn_begin(env, 0, MDB_RDONLY, &txn), "txn_begin(r)");
  CHECK(mdb_dbi_open(txn, 0, 0, &dbi), "dbi_open(r)");

  /* point lookups */
  char kb[8], vb[64];
  for (int i = 0; i < NKEYS; i += 50) {
    make_key(kb, i);
    MDB_val key = {6, kb}, val;
    int rc = mdb_get(txn, dbi, &key, &val);
    if (i % 13 == 0) {
      printf("get %s: %s\n", kb, rc == MDB_NOTFOUND ? "deleted" : "PRESENT?!");
    } else {
      int vn = make_val(vb, i);
      int ok = rc == MDB_SUCCESS && (int)val.mv_size == vn;
      for (int j = 0; ok && j < vn; j++)
        ok = ((char *)val.mv_data)[j] == vb[j];
      printf("get %s: %s (%d bytes)\n", kb, ok ? "ok" : "MISMATCH", (int)val.mv_size);
    }
  }

  /* full ordered cursor scan → running checksum + count (proves in-order B-tree walk) */
  MDB_cursor *cur;
  CHECK(mdb_cursor_open(txn, dbi, &cur), "cursor_open");
  MDB_val key, val;
  unsigned long long sum = 0;
  long count = 0;
  int rc;
  while ((rc = mdb_cursor_get(cur, &key, &val, MDB_NEXT)) == MDB_SUCCESS) {
    for (size_t j = 0; j < key.mv_size; j++) sum = sum * 131 + (unsigned char)((char *)key.mv_data)[j];
    for (size_t j = 0; j < val.mv_size; j++) sum = sum * 131 + (unsigned char)((char *)val.mv_data)[j];
    count++;
  }
  mdb_cursor_close(cur);
  printf("scan: %ld entries, checksum %llu\n", count, sum);

  MDB_stat st;
  mdb_stat(txn, dbi, &st);
  printf("stat: entries=%llu depth=%u leaf=%llu\n", (unsigned long long)st.ms_entries, st.ms_depth,
         (unsigned long long)st.ms_leaf_pages);
  mdb_txn_abort(txn);
  return 0;
}

static MDB_env *open_env(void) {
  MDB_env *env;
  if (mdb_env_create(&env) != MDB_SUCCESS) return 0;
  mdb_env_set_mapsize(env, (size_t)1024 * 1024); /* 1 MiB map (WRITEMAP sizes the file to this) */
  unsigned flags = MDB_NOSUBDIR | MDB_NOLOCK | MDB_WRITEMAP;
  if (mdb_env_open(env, "data.mdb", flags, 0664) != MDB_SUCCESS) {
    mdb_env_close(env);
    return 0;
  }
  return env;
}

int main(int argc, char **argv) {
  int verify_only = argc > 1 && argv[1][0] == 'v';
  MDB_env *env = open_env();
  if (!env) {
    printf("env open failed\n");
    return 1;
  }
  if (!verify_only) {
    int rc = fill(env);
    if (rc) return rc;
    printf("filled %d keys; reopening\n", NKEYS);
    mdb_env_close(env);
    env = open_env();
    if (!env) {
      printf("reopen failed\n");
      return 1;
    }
  }
  int rc = verify(env);
  mdb_env_close(env);
  return rc;
}
