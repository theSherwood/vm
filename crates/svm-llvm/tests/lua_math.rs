//! **Lua's official `math.lua` test** on the on-ramp — the unmodified `testes/math.lua` from the Lua
//! 5.4.7 distribution, run through the whole VM with base/`string`/`table`/`math` open. It is the
//! densest single file in the suite: integer/float arithmetic and conversions, the `//`/`%` operators,
//! float↔integer order (incl. every NaN corner), `math.type`/`tointeger`/`floor`/`ceil`/`fmod`/`ult`/
//! `min`/`max`, the transcendentals (`sin`/`cos`/`tan`/`asin`/`acos`/`atan`/`atan2`/`log`/`exp`/`sqrt`),
//! `math.modf`, `string.format` number formatting, decimal **and hex** float literals (`0x.…p…`,
//! including a 1000-digit fraction), and `math.random` distribution tests. A Lua test raises on a
//! failing `assert`, so a clean **exit 0** means every assert held — identical to native Lua.
//! Byte-for-byte the same outcome on the tree-walker, bytecode, and JIT.
//!
//! The fixture (`tests/fixtures/lua/lua_math.ll`) links the core + those libraries with the guest libc
//! shim, guest `libm`, guest `strtod` (incl. correctly-rounded **hex** floats), the guest runtime
//! `snprintf`, and fdlibm inverse-trig/`modf` (`lua_testsuite_trig.c`) — see the fixtures README.
//! Getting `math.lua` fully green drove two on-ramp fixes: **NaN-correct `fcmp`** (ordered/unordered)
//! and **sign-extended narrow signed ops** (`ashr`/`sdiv`/`srem` on `i8`/`i16` — the bug that had
//! dropped Lua's `getobjname` operand name in error messages).

use svm_run::{Backend, Limits, Outcome, RunConfig, Value};

fn run(backend: Backend) -> svm_run::Run {
    let bc = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/lua/lua_math.ll"
    );
    let t = svm_llvm::translate_ll_path(bc).expect("translate Lua math.lua bitcode");
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
        .expect("run Lua math.lua through the powerbox")
}

/// `main` returns 0 iff every assert in `math.lua` held (a failure raises → the harness returns 3 and
/// prints `math.lua: FAILED: <error>` to stdout, surfaced here).
fn check(backend: Backend) {
    let out = run(backend);
    assert_eq!(
        out.outcome,
        Outcome::Returned(vec![Value::I32(0)]),
        "{backend:?}: math.lua failed\nstdout:\n{}",
        String::from_utf8_lossy(&out.stdout),
    );
}

#[test]
fn lua_math_tree_walker() {
    check(Backend::TreeWalk);
}

#[test]
fn lua_math_bytecode() {
    check(Backend::Bytecode);
}

#[test]
fn lua_math_jit() {
    check(Backend::Jit);
}
