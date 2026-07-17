//! **Lua's official `coroutine.lua` test** on the on-ramp — the unmodified `testes/coroutine.lua` from
//! the Lua 5.4.7 distribution, run through the whole VM with base/`string`/`table`/`math`/`coroutine`/
//! `debug` open. Run standalone the internal `T` C-test library is absent, so the file's own
//! `if not T`/`if T==nil` guards skip the C-API sections; what remains still exercises the coroutine +
//! **debug** libraries deeply: yields inside every metamethod and inside `for` iterators,
//! `coroutine.close` with `<close>` to-be-closed variables, C-stack-overflow detection, and
//! `debug.getinfo`/`getlocal`/`setlocal`/`setupvalue`/`sethook`/`traceback` (including debug on a
//! *suspended* coroutine). A failing assert raises, so a clean **exit 0** means every assert held —
//! identical to native Lua (the same harness+file built natively also exits 0, the differential oracle).
//! Same outcome on the tree-walker, bytecode, and JIT.
//!
//! Landing this on **all three** engines required raising the tree-walker's `MAX_CALL_DEPTH` (256 →
//! 2048): the file's "infinite recursion of coroutines" case relies on Lua's own `LUAI_MAXCCALLS`
//! detection raising a `pcall`-catchable "C stack overflow". The production engines (bytecode, JIT)
//! reach that self-limit; the tree-walker's reified-call-stack cap previously tripped first as an
//! uncatchable §5 kill. The bump lets the reference oracle observe the same catchable error the
//! production engines do — see `svm_interp::MAX_CALL_DEPTH`. No translator or coroutine/debug change was
//! needed; the coroutine machinery itself is the stackless setjmp/longjmp path (no fibers), proven by
//! the sibling `lua_coroutine` differential.

use svm_run::{Backend, Limits, Outcome, RunConfig, Value};

fn run(backend: Backend) -> svm_run::Run {
    let bc = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/lua/lua_coroutine_official.ll"
    );
    let t = svm_llvm::translate_ll_path(bc).expect("translate Lua coroutine.lua bitcode");
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
        .expect("run Lua coroutine.lua through the powerbox")
}

/// `main` returns 0 iff every assert in `coroutine.lua` held (a failure raises → the harness returns a
/// nonzero code and prints `coroutine.lua: <error>` to stdout, surfaced here).
fn check(backend: Backend) {
    let out = run(backend);
    assert_eq!(
        out.outcome,
        Outcome::Returned(vec![Value::I32(0)]),
        "{backend:?}: coroutine.lua failed\nstdout:\n{}",
        String::from_utf8_lossy(&out.stdout),
    );
}

#[test]
fn lua_coroutine_official_tree_walker() {
    check(Backend::TreeWalk);
}

#[test]
fn lua_coroutine_official_bytecode() {
    check(Backend::Bytecode);
}

#[test]
fn lua_coroutine_official_jit() {
    check(Backend::Jit);
}
