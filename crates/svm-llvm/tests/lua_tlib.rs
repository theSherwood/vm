//! **Lua's internal `T` C-test library** (`ltests.c`) on the on-ramp — the whole core compiled with
//! `-DLUA_USER_H='"ltests.h"'`: internal assertions live (`LUAI_ASSERT` — every `lua_assert` in the
//! VM runs), the tracking + failure-injecting `debug_realloc` allocator, debug sizes
//! (`LUAL_BUFFERSIZE 23`, tiny string tables), `LUAI_MAXSTACK 50000`, no jump tables. With `T`
//! active, the suite's `if T` sections **run instead of skipping**, and `api.lua` — the C-API test
//! proper, 1.5k lines of `T.testC` driving raw `lua_*` call sequences, allocation-failure injection
//! at every site, GC internals — runs byte-for-byte unmodified.
//!
//! Two bundles, each **one shared `lua_State`** (the official `all.lua` execution model — ltests'
//! `warnf` keeps its warning mode in process statics, which must line up with the per-state `_WARN`
//! global, and cumulative T-mode memory in one state must fit the reference JIT's 64 MiB window):
//! - `lua_tlib.bc`: `cstack`, `code`, `events`, `gengc`, `errors`, `nextvar`, `locals`, `coroutine`
//!   — the files with substantive `T` sections (yields inside hooks, GC-age probes, C-stack limits).
//! - `lua_tapi.bc`: `gc` + `api` — the two `warn`-using files, together in their own state.
//!
//! Exit 0 means every assert (including the VM's own internal ones) held — identical to native
//! (same harness+bundles on real libc, exit 0, plus the real `atexit` leak check). The JIT runs
//! gate CI; the interpreter runs are `#[ignore]`d full-depth gates (api.lua alone is ~7 min on the
//! bytecode engine).
//!
//! Getting here surfaced one more translator bug: a constexpr `ptrtoint (ptr @g to i32)` folded to
//! its raw `i64` address, feeding I64 into i32 arithmetic (ltests' `strchr(ops, op) - ops` as
//! `int`) — see `const_ptrtoint_i32_width` in `tests/translate.rs`.

use svm_run::{fs, Backend, Limits, Outcome, RunConfig, Value};

fn run(bc: &str, backend: Backend) -> svm_run::Run {
    let t = svm_llvm::translate_bc_path(bc).expect("translate T-library bundle");
    let inst = svm_run::instantiate(t.module).expect("instantiate");
    let config = RunConfig {
        limits: Limits {
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
        .expect("run T-library bundle through the powerbox")
}

/// Exit 0 iff every assert in every file held; a failure exits with the 1-based index of the first
/// failing file (its name + error on stdout, surfaced here).
fn check(bc: &str, backend: Backend) {
    let out = run(bc, backend);
    assert_eq!(
        out.outcome,
        Outcome::Returned(vec![Value::I32(0)]),
        "{backend:?}: T bundle {bc} failed\nstdout:\n{}",
        String::from_utf8_lossy(&out.stdout),
    );
}

const TLIB: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/lua/lua_tlib.bc"
);
const TAPI: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/lua/lua_tapi.bc"
);

#[test]
fn lua_tlib_jit() {
    check(TLIB, Backend::Jit);
}

#[test]
fn lua_tapi_jit() {
    check(TAPI, Backend::Jit);
}

/// Full-depth interpreter gates — correct but long; run via `--ignored` (scheduled/manual).
#[test]
#[ignore = "long on the interpreters; scheduled/manual full-depth gate"]
fn lua_tlib_bytecode() {
    check(TLIB, Backend::Bytecode);
}

#[test]
#[ignore = "api.lua alone is ~7 min on the bytecode engine; scheduled/manual full-depth gate"]
fn lua_tapi_bytecode() {
    check(TAPI, Backend::Bytecode);
}

#[test]
#[ignore = "long on the tree-walker; scheduled/manual full-depth gate"]
fn lua_tlib_tree_walker() {
    check(TLIB, Backend::TreeWalk);
}

#[test]
#[ignore = "long on the tree-walker; scheduled/manual full-depth gate"]
fn lua_tapi_tree_walker() {
    check(TAPI, Backend::TreeWalk);
}
