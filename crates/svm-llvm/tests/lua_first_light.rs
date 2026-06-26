//! Lua first light — the on-ramp translates and runs **real Lua 5.4.7** (its core: lexer, parser,
//! code generator, GC, and the computed-`goto` bytecode VM with `setjmp` error handling) identically
//! on all three engines. The committed bitcode fixture (`tests/fixtures/lua/lua_first_light.bc`, see
//! its README) embeds a script exercising recursion, tables, numeric `for`, closures with upvalues,
//! and the `#` operator — all core VM features, no fail-closed libc stubs on the executed path — and
//! returns 456. This guards the whole stack the milestone rests on: the varargs ABI, the synthesized
//! libc batch, exact `ldexp`, and the cross-block `<N x i1>` mask fix.

use svm_run::{Backend, Limits, Outcome, RunConfig};

/// The script's expected result (`fib(10)=55` + `sum(i*i,1..10)=385` + `#"lua language"=12` +
/// `counter()`'s 4th call `=4`).
const EXPECT: i32 = 456;

fn run(backend: Backend) -> Outcome {
    let bc = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/lua/lua_first_light.bc"
    );
    let t = svm_llvm::translate_bc_path(bc).expect("translate Lua core bitcode");
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
        .expect("run Lua through the powerbox")
        .outcome
}

/// The result a first-light run yields, whether `main` returns it or `_start` turns it into an exit.
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
fn lua_first_light_tree_walker() {
    assert_eq!(result_of(&run(Backend::TreeWalk)), EXPECT);
}

#[test]
fn lua_first_light_bytecode() {
    assert_eq!(result_of(&run(Backend::Bytecode)), EXPECT);
}

#[test]
fn lua_first_light_jit() {
    assert_eq!(result_of(&run(Backend::Jit)), EXPECT);
}
