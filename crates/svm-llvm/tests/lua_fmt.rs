//! Lua **`string.format`** through the runtime format engine — the on-ramp runs Lua 5.4.7 with the
//! base, `string`, `table`, and `math` libraries open, executing a script that formats with
//! `string.format` (width/precision/flags across `%d`/`%x`/`%o`/`%s`/`%c`/`%f`/`%e`/`%g`/`%q`), and
//! captures the bytes the guest writes to **stdout through the `Stream.write` capability** —
//! byte-identical on the tree-walker, bytecode, and JIT, and identical to a native build.
//!
//! Unlike the earlier `lua_stdlib` fixture (which deliberately avoids `string.format`), this exercises
//! the **runtime** format path: Lua's `str_format` parses each `%`-directive itself and calls
//! `snprintf` once per directive with a spec built at runtime. The fixture links a guest `snprintf`
//! (`tests/fixtures/lua/lua_fmt_snprintf.c`) that formats integers/strings/chars in C and delegates
//! floats to the on-ramp's correctly-rounded bignum dtoa via `__vm_fmt_{fix,sci,gen}` (recognized in
//! `lower_vm_builtin`). The result is byte-identical to native `string.format`.

use svm_run::{Backend, Limits, RunConfig};

/// The exact stdout of a native build of the same core + libraries + guest snprintf + script.
/// (Written with `concat!` so the leading spaces on the `row` lines survive — a `\`-continuation
/// in a Rust string literal strips leading whitespace on the next line.)
const EXPECT: &str = concat!(
    "2 + 3 = 5\n",
    "[   42][42   ][00042][+42]\n",
    "hex ff FF 0xff oct 100\n",
    "str [        hi][hi        ][hel]\n",
    "char Lua\n",
    "float 3.14    2.500 +7.0\n",
    "sci 1.235e+05 general 0.0001 3.141592654\n",
    "\"he said \\\"hi\\\"\"\n",
    "7 items, 87.5% done\n",
    "  row 1: x = 1\n",
    "  row 2: x = 4\n",
    "  row 3: x = 9\n",
);

fn stdout_of(backend: Backend) -> Vec<u8> {
    let bc = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/lua/lua_fmt.ll");
    let t = svm_llvm::translate_ll_path(bc).expect("translate Lua string.format bitcode");
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
        .expect("run Lua string.format through the powerbox")
        .stdout
}

fn check(backend: Backend) {
    let got = stdout_of(backend);
    assert_eq!(
        String::from_utf8_lossy(&got),
        EXPECT,
        "{backend:?}: stdout mismatch"
    );
}

#[test]
fn lua_fmt_tree_walker() {
    check(Backend::TreeWalk);
}

#[test]
fn lua_fmt_bytecode() {
    check(Backend::Bytecode);
}

#[test]
fn lua_fmt_jit() {
    check(Backend::Jit);
}
