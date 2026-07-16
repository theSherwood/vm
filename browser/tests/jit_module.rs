//! **Single-shot module wasm-JIT differential** (the run-to-completion twin of `jit_reactor.rs`). An
//! on-ramp module's whole program is func 0 (`_start`); [`JitOnrampRun`] emits it and runs
//! `f0(win, env, ...slots)` once on `wasmi` (playing the browser's JS host), with the cross-tier
//! helpers relaying to the interpreter over the shared window through `env.call_interp`. The captured
//! stdout + exit must match [`onramp_exec`] (the interpreter oracle) byte-for-byte — the JIT
//! correctness contract for the module run path (Lua / SQLite / hello). Timing is printed (informational).

use std::path::Path;
use std::time::Instant;

use svm_browser::{onramp_exec, JitOnrampRun};
use wasmi::{Caller, Engine, Linker, Memory, MemoryType, Module as WModule, Store, Val};

const WIN_LOG2: u8 = 25; // 32 MiB window (holds Lua/SQLite's heap; the emitted run can't grow it)
const WIN_SIZE: u64 = 1 << WIN_LOG2;
const WIN_BASE: u32 = 0x1_0000; // window starts at 64 KiB (the env cell lives below it)
const ENV_PTR: u32 = 1024;

/// One captured run: stdout, whether it `exit`ed, the exit/return code.
#[derive(PartialEq, Debug)]
struct Out {
    stdout: Vec<u8>,
    exited: bool,
    code: i32,
}

/// Interpreter oracle via `onramp_exec`.
fn interp(m: &svm_ir::Module, stdin: &[u8]) -> (Out, u128) {
    let t0 = Instant::now();
    let o = onramp_exec(m, stdin);
    let us = t0.elapsed().as_micros();
    (
        Out {
            stdout: o.stdout,
            exited: o.status == svm_browser::STATUS_EXIT,
            code: if o.status == svm_browser::STATUS_EXIT {
                o.exit_code
            } else {
                o.value as i32
            },
        },
        us,
    )
}

/// wasm-JIT run: emitted `f0` on `wasmi`, cross-tier helpers on the interpreter over the shared window.
fn jit(m: &svm_ir::Module, stdin: &[u8]) -> (Out, u128) {
    let engine = Engine::default();
    let pages = ((WIN_BASE as u64 + WIN_SIZE) / (64 * 1024)) as u32;
    let mut store: Store<Option<JitOnrampRun>> = Store::new(&engine, None);
    let memory =
        Memory::new(&mut store, MemoryType::new(pages, Some(pages))).expect("wasmi memory");

    let win_ptr = unsafe {
        memory
            .data_mut(&mut store)
            .as_mut_ptr()
            .add(WIN_BASE as usize)
    };
    // SAFETY: fixed-size memory ⇒ stable data pointer; the window lives inside it for the run.
    let run = unsafe {
        JitOnrampRun::open_shared_run(m, win_ptr, WIN_SIZE, WIN_LOG2, false, stdin.to_vec())
    }
    .expect("module emittable as a single-shot JIT run");

    let emitted_wasm = run.emitted_wasm().to_vec();
    let slots: Vec<i32> = run
        .slots()
        .iter()
        .map(|v| match v {
            svm_interp::Value::I32(x) => *x,
            svm_interp::Value::I64(x) => *x as i32,
            _ => 0,
        })
        .collect();
    let rtys: Vec<svm_ir::ValType> = run.func_sig(0).1.to_vec();
    let module = match WModule::new(&engine, &emitted_wasm) {
        Ok(m) => m,
        Err(e) => panic!(
            "emitted _start ({} B) failed to validate: {e}",
            emitted_wasm.len()
        ),
    };
    *store.data_mut() = Some(run);

    let mut linker: Linker<Option<JitOnrampRun>> = Linker::new(&engine);
    linker.define("env", "memory", memory).unwrap();
    linker
        .func_wrap("env", "trap", |_c: Caller<'_, _>, _code: i32| {})
        .unwrap();
    linker
        .func_wrap(
            "env",
            "call_interp",
            move |mut caller: Caller<'_, Option<JitOnrampRun>>,
                  func: i32,
                  args_ptr: i32|
                  -> Result<(), wasmi::Error> {
                let (params, results) = {
                    let r = caller.data().as_ref().unwrap();
                    let (p, rs) = r.func_sig(func as u32);
                    (p.to_vec(), rs.to_vec())
                };
                let args: Vec<svm_interp::Value> = {
                    let data = memory.data(&caller);
                    params
                        .iter()
                        .enumerate()
                        .map(|(i, t)| {
                            let o = args_ptr as usize + i * 8;
                            let raw = u64::from_le_bytes(data[o..o + 8].try_into().unwrap());
                            match t {
                                svm_ir::ValType::I32 => svm_interp::Value::I32(raw as i32),
                                _ => svm_interp::Value::I64(raw as i64),
                            }
                        })
                        .collect()
                };
                let outcome = caller
                    .data_mut()
                    .as_mut()
                    .unwrap()
                    .run_cross_tier(func as u32, &args);
                match outcome {
                    Ok(vals) => {
                        let data = memory.data_mut(&mut caller);
                        for (i, v) in vals.iter().enumerate() {
                            if i >= results.len() {
                                break;
                            }
                            let raw = match v {
                                svm_interp::Value::I32(x) => *x as u32 as u64,
                                svm_interp::Value::I64(x) => *x as u64,
                                _ => 0,
                            };
                            let o = args_ptr as usize + i * 8;
                            data[o..o + 8].copy_from_slice(&raw.to_le_bytes());
                        }
                        Ok(())
                    }
                    Err(t) => {
                        caller
                            .data_mut()
                            .as_mut()
                            .unwrap()
                            .set_last_trap(format!("{t:?}"));
                        Err(wasmi::Error::from(
                            wasmi::core::TrapCode::UnreachableCodeReached,
                        ))
                    }
                }
            },
        )
        .unwrap();

    let instance = linker
        .instantiate(&mut store, &module)
        .unwrap()
        .start(&mut store)
        .unwrap();
    let f0 = instance.get_func(&store, "f0").expect("emitted f0 export");

    // f0(win, env, ...slots): the powerbox handles are _start's params.
    let mut args: Vec<Val> = vec![Val::I32(WIN_BASE as i32), Val::I32(ENV_PTR as i32)];
    args.extend(slots.iter().map(|&h| Val::I32(h)));
    let mut results: Vec<Val> = rtys
        .iter()
        .map(|t| match t {
            svm_ir::ValType::I32 => Val::I32(0),
            _ => Val::I64(0),
        })
        .collect();

    let t0 = Instant::now();
    // Huge dispatcher-fuel budget (debited per emitted dispatcher iteration).
    memory
        .write(&mut store, ENV_PTR as usize, &(1i64 << 60).to_le_bytes())
        .unwrap();
    let call = f0.call(&mut store, &args, &mut results);
    let us = t0.elapsed().as_micros();

    let run = store.data().as_ref().unwrap();
    // A trap that is the guest's `exit` (unwinding f0 via call_interp) is expected; a trap without an
    // `exit` recorded is a real fault.
    if call.is_err() && !run.exited() {
        panic!(
            "emitted f0 trapped (not an exit): {} ({})",
            call.unwrap_err(),
            run.last_trap()
        );
    }
    let value = match results.first() {
        Some(Val::I32(x)) => *x,
        Some(Val::I64(x)) => *x as i32,
        _ => 0,
    };
    (
        Out {
            stdout: run.stdout().to_vec(),
            exited: run.exited(),
            code: if run.exited() { run.exit_code() } else { value },
        },
        us,
    )
}

fn asset(name: &str) -> Option<svm_ir::Module> {
    let p = format!("web/assets/{name}.svmb");
    if !Path::new(&p).exists() {
        return None;
    }
    Some(svm_encode::decode_module(&std::fs::read(&p).unwrap()).expect("decode"))
}

fn differential(name: &str, stdin: &[u8]) {
    let Some(m) = asset(name) else {
        eprintln!("skipping {name}: asset not present");
        return;
    };
    let (i, ius) = interp(&m, stdin);
    let (j, jus) = jit(&m, stdin);
    assert_eq!(
        j, i,
        "{name}: JIT run must match the interpreter (stdout/exit)"
    );
    eprintln!(
        "{name}: MATCH — stdout {}B, exit={} code={} · interp {ius}µs vs jit {jus}µs ({:.1}×)",
        i.stdout.len(),
        i.exited,
        i.code,
        ius as f64 / jus.max(1) as f64,
    );
}

#[test]
fn hello_c_jit_matches_interpreter() {
    differential("hello_c", b"");
}

// Lua / SQLite emit valid wasm that **V8** runs (proven byte-identical by `browser-jit-module-test.mjs`),
// but `wasmi`'s register-based compiler rejects their huge hot functions (`luaV_execute`,
// `sqlite3VdbeExec`) with "translation requires more registers than available" — a `wasmi` limit, not an
// emitter one. So these run in the browser, not this native harness; `#[ignore]`d (and asset-gated).
#[test]
#[ignore = "wasmi can't compile Lua/SQLite's giant functions; V8 does — see browser-jit-module-test.mjs"]
fn lua_jit_matches_interpreter() {
    differential(
        "lua_eval",
        b"local s=0; for i=1,200000 do s=s+i end; print(s)\n",
    );
}

#[test]
#[ignore = "wasmi can't compile Lua/SQLite's giant functions; V8 does — see browser-jit-module-test.mjs"]
fn sqlite_jit_matches_interpreter() {
    differential(
        "sqlite_repl",
        b"CREATE TABLE t(x); INSERT INTO t VALUES (1),(2),(3); SELECT sum(x) FROM t;\n",
    );
}
