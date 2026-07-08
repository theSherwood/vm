//! **The suite sweep** — twenty-one more unmodified official Lua 5.4.7 test files through the whole
//! VM in one bundle (`lua_sweep.bc`, light-to-heavy: `tracegc`, `verybig`, `big`, `gengc`, `goto`,
//! `events`, `code`, `bitwise`, `closure`, `tpack`, `literals`, `errors`, `nextvar`, `sort`, `db`,
//! `constructs`, `locals`, `cstack`, `strings`, `gc`, `calls`), each as its own chunk in a fresh
//! `lua_State` under `pcall`. With the earlier slices' library surface (`require` + io/os over the
//! Fs capability + debug + coroutine + utf8) and the sweep harness's additions (a real free-list
//! allocator — the collector stress in `gc.lua` genuinely frees; sibling-module `require` — the
//! suite's files require each other; a faithful `package.loaded`/`preload`; `@`-style chunknames),
//! every file runs **byte-for-byte unmodified** under the suite's own `_port`/`_soft` knobs. Exit 0
//! means every assert in all 21 files held — identical to native Lua (the same harness+bundle built
//! against real libc also exits 0, the differential oracle).
//!
//! Getting here surfaced one real translator bug and three guest-snprintf gaps (see the fixtures
//! README): the bignum float formatter's **40-limb ceiling silently truncated** `%.99f` of
//! near-max doubles (`strings.lua`'s longest-number test; now 48 limbs), and `%p` width, `%a`
//! hex-floats (with precision/rounding), and the ISO zero-precision/`0`/`#` flag corners were
//! missing from the guest formatter.
//!
//! The JIT test runs in CI; the interpreter runs are `#[ignore]`d — correct but long (the bundle is
//! ~15 min on the bytecode engine, more on the tree-walker), suited to the scheduled extended run
//! (`cargo test -- --ignored`), the same treatment the long fuzz targets get.

use svm_run::{fs, Backend, Limits, Outcome, RunConfig, Value};

fn run(backend: Backend) -> svm_run::Run {
    let bc = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/lua/lua_sweep.bc"
    );
    let t = svm_llvm::translate_bc_path(bc).expect("translate Lua sweep bundle");
    let inst = svm_run::instantiate(t.module).expect("instantiate");
    let config = RunConfig {
        limits: Limits {
            // The heavy files (gc stress, deep calls) exceed the default fuel on the interpreters.
            fuel: Some(u64::MAX / 4),
            deadline: None,
            max_fibers: 0,
            max_vcpus: 0,
        },
        stdin: vec![],
        memory_size_log2: None,
        args: vec![],
        env: vec![b"PATH=/usr/bin".to_vec()],
    };
    inst.run_with_caps(backend, &config, &[("fs", fs::mem_fs())])
        .expect("run Lua sweep bundle through the powerbox")
}

/// `main` returns 0 iff every assert in all 21 files held; a failure exits with the 1-based index
/// of the first failing file (its name + error are on stdout, surfaced here).
fn check(backend: Backend) {
    let out = run(backend);
    assert_eq!(
        out.outcome,
        Outcome::Returned(vec![Value::I32(0)]),
        "{backend:?}: sweep bundle failed\nstdout:\n{}",
        String::from_utf8_lossy(&out.stdout),
    );
}

#[test]
fn lua_sweep_jit() {
    check(Backend::Jit);
}

/// Full-depth interpreter gates — correct but long; run via `--ignored` (nightly/manual).
#[test]
#[ignore = "~15 min on the bytecode engine; scheduled/manual full-depth gate"]
fn lua_sweep_bytecode() {
    check(Backend::Bytecode);
}

#[test]
#[ignore = "long on the tree-walker; scheduled/manual full-depth gate"]
fn lua_sweep_tree_walker() {
    check(Backend::TreeWalk);
}
