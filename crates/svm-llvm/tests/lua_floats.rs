//! Lua **with floats** — the payoff of the guest-libm + guest-`strtod` work. The on-ramp translates
//! and runs **real Lua 5.4.7 core** linked with the bundled guest `libm` (`demos/libm/libm.c`) and
//! guest `strtod` (`demos/strtod/strtod.c`), executing a script whose float arithmetic reaches every
//! new piece of this work, identically on all three engines and identical to a native build.
//!
//! The script (see `tests/fixtures/lua/lua_floats_harness.c`):
//! ```lua
//! local function f(x) return x ^ 0.5 end   -- runtime pow  (the guest libm)
//! local function g(x, y) return x % y end  -- runtime fmod (the synthesized helper)
//! local a = 3.14                           -- strtod (the guest parser)
//! local b = f(2.0)                         -- pow(2.0, 0.5)
//! local c = g(10.5, 3.0)                   -- fmod(10.5, 3.0) = 1.5
//! local d = 1.5e3 + 0.25                   -- strtod (scientific + fraction)
//! return (a + b + c + d) * 1000.0          -- (int) = 1506304
//! ```
//!
//! So a single run exercises, end to end through the whole Lua VM: the **guest `strtod`** (every
//! numeric literal, parsed in the lexer), the **guest `pow`** (the `^` operator), and the synthesized
//! **`fmod`** (the `%` operator) — plus `frexp`/`localeconv`/`snprintf`/`setjmp` referenced by the
//! core. The integer-cast result is **1506304**, byte-identical to the native build of the same
//! sources (the differential the guest-`libm`/`strtod` unit tests already pin per-function).
//!
//! Reproduce the fixture: see the fixtures README §"Regenerating (floats)".

use svm_run::{Backend, Limits, Outcome, RunConfig};

/// `(int)((3.14 + sqrt(2) + (10.5 % 3.0) + 1500.25) * 1000.0)` — pinned against a native build.
const EXPECT: i32 = 1506304;

fn run(backend: Backend) -> Outcome {
    let bc = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/lua/lua_floats.ll"
    );
    let t = svm_llvm::translate_ll_path(bc).expect("translate Lua+floats bitcode");
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
        .expect("run Lua+floats through the powerbox")
        .outcome
}

fn result_of(outcome: &Outcome) -> i32 {
    match outcome {
        Outcome::Returned(vs) => match vs.first() {
            Some(svm_interp::Value::I32(x)) => *x,
            other => panic!("unexpected return value {other:?}"),
        },
        Outcome::Exited(code) => *code,
    }
}

#[test]
fn lua_floats_tree_walker() {
    assert_eq!(result_of(&run(Backend::TreeWalk)), EXPECT);
}

#[test]
fn lua_floats_bytecode() {
    assert_eq!(result_of(&run(Backend::Bytecode)), EXPECT);
}

#[test]
fn lua_floats_jit() {
    assert_eq!(result_of(&run(Backend::Jit)), EXPECT);
}
