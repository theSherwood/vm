//! **End-to-end proof that a Postgres session survives a snapshot/reboot** — the persistence hinge
//! behind the browser console's "survive a page reload" feature. Drives the exact shipping FFI surface
//! (`svm_pg_open` / `svm_pg_query` / `svm_pg_snapshot` / `svm_pg_close`) against the real 20 MB Postgres
//! module + 40 MB data image, exactly as `play.js` does over the wasm boundary:
//!
//!   1. boot `postgres --single` to its prompt,
//!   2. `CREATE TABLE` + `INSERT` a known row,
//!   3. `svm_pg_snapshot` the live data dir to an image, then `svm_pg_close`,
//!   4. `svm_pg_open` a **fresh** backend from that snapshot image,
//!   5. `SELECT` the row back — proving Postgres' startup recovery replays the snapshot and the
//!      committed row is still there.
//!
//! `#[ignore]` because it needs the staged artifacts (`browser/web/assets/*`, produced by
//! `node build-pg-assets.mjs`) and boots the backend twice (a few seconds each). Run explicitly:
//!
//! ```text
//! cd browser && cargo test --test pg_snapshot_roundtrip -- --ignored --nocapture
//! ```

use std::path::PathBuf;

fn asset(name: &str) -> Vec<u8> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("web/assets")
        .join(name);
    std::fs::read(&p).unwrap_or_else(|e| {
        panic!("missing staged artifact {p:?} ({e}); run `node build-pg-assets.mjs` first")
    })
}

/// Read back the stdout the last `svm_pg_*` call captured (the cdylib-managed `OUT` buffer).
fn engine_stdout() -> String {
    let ptr = svm_browser::svm_stdout_ptr();
    let len = svm_browser::svm_stdout_len();
    if ptr.is_null() || len == 0 {
        return String::new();
    }
    // SAFETY: `svm_stdout_ptr/_len` name a live cdylib allocation valid until the next capturing call.
    let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
    String::from_utf8_lossy(bytes).into_owned()
}

fn open(module: &[u8], image: &[u8]) -> i32 {
    svm_browser::svm_pg_open(module.as_ptr(), module.len(), image.as_ptr(), image.len())
}

fn query(sql: &str) -> (i32, String) {
    let rc = svm_browser::svm_pg_query(sql.as_ptr(), sql.len());
    (rc, engine_stdout())
}

#[test]
#[ignore = "needs staged Postgres artifacts + boots the backend twice (slow)"]
fn session_survives_snapshot_and_reboot() {
    let module = asset("postgres_resolved.svmb");
    let image = asset("pgdata.img");

    // 1) Boot a fresh backend from the pristine image.
    let rc = open(&module, &image);
    assert_eq!(
        rc,
        0,
        "initial boot failed: status {}",
        svm_browser::svm_status()
    );
    let banner = engine_stdout();
    assert!(
        banner.contains("backend>"),
        "no prompt after boot; got:\n{banner}"
    );

    // 2) Create a table and insert a sentinel row.
    let (rc, _) = query("CREATE TABLE persist_probe (x int);");
    assert_eq!(
        rc,
        0,
        "CREATE TABLE failed: status {}",
        svm_browser::svm_status()
    );
    let (rc, out) = query("INSERT INTO persist_probe VALUES (424242);");
    assert_eq!(rc, 0, "INSERT failed: status {}", svm_browser::svm_status());
    assert!(!out.contains("ERROR"), "INSERT reported an error:\n{out}");

    // 3) Snapshot the live data dir, then tear the backend down entirely.
    assert_eq!(svm_browser::svm_pg_snapshot(), 0, "snapshot failed");
    let sptr = svm_browser::svm_pg_snapshot_ptr();
    let slen = svm_browser::svm_pg_snapshot_len();
    assert!(!sptr.is_null() && slen > 0, "empty snapshot image");
    // SAFETY: the snapshot buffer is valid until the next snapshot; copy it out before reopening.
    let snapshot = unsafe { std::slice::from_raw_parts(sptr, slen) }.to_vec();
    svm_browser::svm_pg_close();

    // 4) Boot a brand-new backend from the snapshot image (recovery runs here).
    let rc = open(&module, &snapshot);
    assert_eq!(
        rc,
        0,
        "reboot from snapshot failed: status {}",
        svm_browser::svm_status()
    );

    // 5) The row survives the reboot.
    let (rc, out) = query("SELECT x FROM persist_probe;");
    assert_eq!(
        rc,
        0,
        "SELECT after reboot failed: status {}",
        svm_browser::svm_status()
    );
    assert!(
        !out.contains("ERROR"),
        "SELECT after reboot errored:\n{out}"
    );
    assert!(
        out.contains("424242"),
        "the inserted row did not survive the snapshot/reboot; SELECT output:\n{out}"
    );

    svm_browser::svm_pg_close();
}
