//! Lua **eval from stdin** — the interactive-playground guest. The on-ramp runs Lua 5.4.7 (core +
//! base/`string`/`table`/`math` + a guest `snprintf`) with a harness that reads a Lua chunk from
//! **stdin** (the `Stream.read` capability), evaluates it, and `print`s the result to **stdout** (the
//! `Stream.write` capability). This is what the browser playground pipes the editor's text into, so a
//! user can write and run their own Lua. Asserts the exact stdout bytes on the tree-walker, bytecode,
//! and JIT — identical to a native build of the same sources.
//!
//! The fixture (`tests/fixtures/lua/lua_eval.bc`, harness `lua_eval_harness.c` alongside) differs from
//! `lua_stdlib` only in the harness: the script arrives as stdin instead of being embedded, and the
//! guest `snprintf` (from the `lua_fmt` recipe) is linked so `string.format` works.

use svm_run::{Backend, Limits, RunConfig};

/// A small script covering the three things a first-time user reaches for: `print`, `string.format`
/// (runtime format — the `lua_fmt` guest snprintf), and a `table.sort`/`concat`.
const SCRIPT: &str = "print('eval works, ' .. _VERSION)\n\
print(string.format('%d %s %.2f', 6 * 7, 'ok', 1.5))\n\
local t = { 3, 1, 2 }\n\
table.sort(t)\n\
print(table.concat(t, ','))\n";

/// The exact stdout of a native build of the same core + libraries + harness over `SCRIPT`.
const EXPECT: &str = "eval works, Lua 5.4\n\
42 ok 1.50\n\
1,2,3\n";

fn stdout_of(backend: Backend) -> Vec<u8> {
    let bc = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/lua/lua_eval.bc"
    );
    let t = svm_llvm::translate_bc_path(bc).expect("translate Lua eval bitcode");
    let inst = svm_run::instantiate(t.module).expect("instantiate");
    let config = RunConfig {
        limits: Limits {
            fuel: None,
            deadline: None,
            max_fibers: 0,
            max_vcpus: 0,
        },
        stdin: SCRIPT.as_bytes().to_vec(), // the guest reads its program off stdin
        memory_size_log2: None,
        args: vec![],
        env: vec![],
    };
    inst.run(backend, &config)
        .expect("run Lua eval through the powerbox")
        .stdout
}

fn check(backend: Backend) {
    assert_eq!(
        String::from_utf8_lossy(&stdout_of(backend)),
        EXPECT,
        "{backend:?}: stdout mismatch",
    );
}

#[test]
fn lua_eval_tree_walker() {
    check(Backend::TreeWalk);
}

#[test]
fn lua_eval_bytecode() {
    check(Backend::Bytecode);
}

#[test]
fn lua_eval_jit() {
    check(Backend::Jit);
}
