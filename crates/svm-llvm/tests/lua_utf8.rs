//! **Lua's official `utf8.lua` test** on the on-ramp — the unmodified `testes/utf8.lua` from the Lua
//! 5.4.7 distribution, run through the whole VM with base/`string`/`table`/`math`/`utf8` open. It
//! exercises the full `utf8` library: `utf8.char`/`codepoint`/`len`/`offset`/`codes`/`charpattern`,
//! strict vs. `nonstrict` decoding across every sequence size (1–6 bytes, incl. the original UTF-8
//! range up to `0x7FFFFFFF`), surrogate and overlong rejection, error positions from `utf8.len`,
//! `utf8.codes` iteration errors, the `\u{…}` string escapes (round-tripped through `load`), and
//! `string.gmatch(s, utf8.charpattern)`. A Lua test raises on a failing `assert`, so a clean **exit 0**
//! means every assert held — identical to native Lua. Same outcome on the tree-walker, bytecode, and JIT.
//!
//! `utf8.lua` opens with `local utf8 = require'utf8'`, so this is the first fixture to need a working
//! `require`. The harness (`lua_utf8_harness.c`) installs a minimal global `require` that resolves a
//! preloaded module from the registry `_LOADED` table — exactly stock `require`'s fast path for a
//! `luaL_requiref`'d library, the only path reachable on the on-ramp (no filesystem / dynamic loader
//! for the file and C-library searchers). See the fixtures README.

use svm_run::{Backend, Limits, Outcome, RunConfig, Value};

fn run(backend: Backend) -> svm_run::Run {
    let bc = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/lua/lua_utf8.ll"
    );
    let t = svm_llvm::translate_ll_path(bc).expect("translate Lua utf8.lua bitcode");
    let inst = svm_run::instantiate(t.module).expect("instantiate");
    let config = RunConfig {
        limits: Limits {
            fuel: None,
            deadline: None,
            max_fibers: 0,
            max_vcpus: 0,
        },
        stdin: vec![],
        memory_size_log2: None,
        args: vec![],
        env: vec![],
    };
    inst.run(backend, &config)
        .expect("run Lua utf8.lua through the powerbox")
}

/// `main` returns 0 iff every assert in `utf8.lua` held (a failure raises → the harness returns a
/// nonzero code and prints `utf8.lua: <error>` to stdout, surfaced here).
fn check(backend: Backend) {
    let out = run(backend);
    assert_eq!(
        out.outcome,
        Outcome::Returned(vec![Value::I32(0)]),
        "{backend:?}: utf8.lua failed\nstdout:\n{}",
        String::from_utf8_lossy(&out.stdout),
    );
}

#[test]
fn lua_utf8_tree_walker() {
    check(Backend::TreeWalk);
}

#[test]
fn lua_utf8_bytecode() {
    check(Backend::Bytecode);
}

#[test]
fn lua_utf8_jit() {
    check(Backend::Jit);
}
