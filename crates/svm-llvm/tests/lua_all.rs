//! **Lua's official test suite under its own driver `all.lua`** — the capstone. The unmodified
//! `testes/all.lua` runs on the on-ramp exactly as the Lua distribution ships it: it `dofile`s each
//! test file (through `loadfile` → `string.dump` → `load`, the suite's own dump/undump round-trip),
//! tracks memory + timing, runs the warning-system tests, and ends at `print("final OK !!!")`. The
//! whole `testes/` tree is seeded onto the **in-memory Fs** (`lua_all_tests.c`), so `all.lua`'s
//! `loadfile`/`dofile`/`require` genuinely search `package.path` and load each file off the
//! (in-memory) disk via the **real** `luaopen_package` (`loadlib.c`) — the minimal-`require` shim is
//! gone. The T library (`ltests.c`) is active with internal assertions live, the failure-injecting
//! `debug_realloc` allocator (over a coalescing guest `malloc`), and one shared `lua_State` for the
//! entire suite, as the driver intends.
//!
//! With the suite's documented `_port`/`_soft`/`_nomsg` knobs set, `all.lua` runs **26 files**
//! (`main.lua` early-returns under `_port`; `big.lua` skips under `_soft`) — `db`, `calls`, `strings`,
//! `literals`, `tpack`, **`attrib`** (the real-`require`/`package.searchpath` test, asserting
//! `== 27`), `gengc`, `locals`, `constructs`, `code`, `cstack`, `nextvar`, `pm`, `utf8`, `api`,
//! `events`, `vararg`, `closure`, `coroutine`, `goto`, `errors`, `math`, `sort`, `bitwise`,
//! `verybig`, `files`, plus `gc`. Exit 0 (and `final OK !!!` on stdout) means the suite passed —
//! identical to native (same harness+tree on real libc, exit 0, incl. ltests' `atexit` leak check).
//!
//! The JIT run gates CI (~20 s); the interpreter runs are `#[ignore]`d full-depth gates (the whole
//! suite in the tree-walker is long, like the extended fuzz).

use svm_run::{fs, Backend, Limits, Outcome, RunConfig, Value};

fn run(backend: Backend) -> svm_run::Run {
    let bc = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/lua/lua_all.ll");
    let t = svm_llvm::translate_ll_path(bc).expect("translate Lua all.lua suite");
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
        .expect("run Lua all.lua through the powerbox")
}

/// `all.lua` ends at `print("final OK !!!")` and returns; exit 0 iff the whole suite passed.
fn check(backend: Backend) {
    let out = run(backend);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(
        out.outcome,
        Outcome::Returned(vec![Value::I32(0)]),
        "{backend:?}: all.lua suite failed\nstdout tail:\n{}",
        stdout
            .lines()
            .rev()
            .take(8)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<Vec<_>>()
            .join("\n"),
    );
    assert!(
        stdout.contains("final OK !!!"),
        "{backend:?}: all.lua did not reach `final OK !!!`",
    );
}

#[test]
fn lua_all_jit() {
    check(Backend::Jit);
}

#[test]
#[ignore = "the whole suite on the bytecode engine is long; scheduled/manual full-depth gate"]
fn lua_all_bytecode() {
    check(Backend::Bytecode);
}

#[test]
#[ignore = "the whole suite on the tree-walker is long; scheduled/manual full-depth gate"]
fn lua_all_tree_walker() {
    check(Backend::TreeWalk);
}
