//! Lua **standard library + real output** — the on-ramp runs Lua 5.4.7 with the base, `string`,
//! `table`, and `math` libraries open, executing a script that `print`s, and captures the bytes the
//! guest writes to **stdout through the `Stream.write` capability** — byte-identical on the
//! tree-walker, bytecode, and JIT, and identical to a native build of the same sources.
//!
//! The fixture (`tests/fixtures/lua/lua_stdlib.bc`, harness + guest-libc shim alongside) opens the
//! four libraries via `luaL_requiref` and runs a script exercising `print`, `string.upper`/`rep`/
//! `sub`/`#`, `table.sort`/`concat`/`insert`/`remove`, `math.sqrt`/`pi`/`floor`/`max`/`abs`, `ipairs`,
//! `pairs`, `type`, and `tostring`. `print` of numbers uses Lua's **constant** `%lld`/`%.14g` formats
//! (which the on-ramp's snprintf handles); `string.format`'s **runtime** format string is a separate,
//! not-yet-supported path (it fail-closes to a trap — see `snprintf_rt`), so the script avoids it.
//!
//! Unlike the earlier Lua tests (which check a return value), this asserts the exact **stdout bytes** —
//! the first end-to-end demonstration of Lua producing real output through the powerbox.

use svm_run::{Backend, Limits, RunConfig};

/// The exact stdout of a native build of the same core + libraries + script.
const EXPECT: &str = "hello from lua on the on-ramp\n\
2 + 3 =\t5\n\
upper\tLUA\trep\tababab\n\
sub\thello\tlen\t11\n\
sorted\t1,1,2,3,4,5,6,9\n\
after ins/rem\t8\t7\n\
sqrt2\t1.4142135623731\n\
pi\t3.1415926535898\n\
floor\t3\tmax\t5\tabs\t42\n\
ipairs sum\t100\n\
pairs sum\t6\ttype\tnumber\ttostring\ttrue\n";

fn stdout_of(backend: Backend) -> Vec<u8> {
    let bc = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/lua/lua_stdlib.bc"
    );
    let t = svm_llvm::translate_bc_path(bc).expect("translate Lua stdlib bitcode");
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
        .expect("run Lua stdlib through the powerbox")
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
fn lua_stdlib_tree_walker() {
    check(Backend::TreeWalk);
}

#[test]
fn lua_stdlib_bytecode() {
    check(Backend::Bytecode);
}

#[test]
fn lua_stdlib_jit() {
    check(Backend::Jit);
}
