//! **Coroutine library** on the on-ramp — an in-house differential (`lua_coroutine.lua` in the fixtures
//! directory, embedded verbatim) run through the whole VM with base/`string`/`table`/`math`/`coroutine`
//! open. It exercises the coroutine surface that does not need the `debug` library (that is a separate
//! slice, gating the official `testes/coroutine.lua`): `create`/`resume`/`yield` with multi-value
//! transfer both directions, the `suspended`/`running`/`normal`/`dead` status transitions,
//! `running`/`isyieldable` in the main thread vs. inside a coroutine, `wrap` (incl. error re-raise),
//! error propagation out of `resume` (string and non-string error values), **yield across `pcall` and
//! `xpcall`** (the yieldable-pcall / continuation machinery), `coroutine.close` with `<close>`
//! to-be-closed variables, and a producer/filter/consumer pipeline. A failing assert raises, so a clean
//! **exit 0** means every assert held — identical to native Lua. Same outcome on all three engines.
//!
//! Lua 5.4 coroutines are **stackless** with respect to the C stack: each coroutine is a `lua_State`
//! with its own heap-allocated Lua stack, and resume/yield ride the same `luaD_rawrunprotected` /
//! `luaD_throw` (setjmp/longjmp) primitive `pcall` already uses. No fiber or native-stack switching is
//! involved — the on-ramp's existing `SetJmp`/`LongJmp` core ops (proven by every working `pcall`)
//! carry it, and no translator or libc change was needed here.

use svm_run::{Backend, Limits, Outcome, RunConfig, Value};

fn run(backend: Backend) -> svm_run::Run {
    let bc = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/lua/lua_coroutine.ll"
    );
    let t = svm_llvm::translate_ll_path(bc).expect("translate Lua coroutine bitcode");
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
        .expect("run Lua coroutine differential through the powerbox")
}

/// `main` returns 0 iff every assert held (a failure raises → the harness returns a nonzero code and
/// prints `coroutine.lua: <error>` to stdout, surfaced here).
fn check(backend: Backend) {
    let out = run(backend);
    assert_eq!(
        out.outcome,
        Outcome::Returned(vec![Value::I32(0)]),
        "{backend:?}: coroutine differential failed\nstdout:\n{}",
        String::from_utf8_lossy(&out.stdout),
    );
}

#[test]
fn lua_coroutine_tree_walker() {
    check(Backend::TreeWalk);
}

#[test]
fn lua_coroutine_bytecode() {
    check(Backend::Bytecode);
}

#[test]
fn lua_coroutine_jit() {
    check(Backend::Jit);
}
