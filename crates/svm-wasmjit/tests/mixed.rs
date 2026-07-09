//! Mixed-tier differential (BROWSER.md § "wasm-JIT tier", slice 3b). A guest with an integer
//! (JIT-eligible) caller and a **SIMD** leaf (out of the emitter's subset, but integer-signature and
//! memory-free): the caller is emitted to wasm; the leaf runs on the bytecode interpreter via
//! `env.call_interp`. (The leaves use `v128` rather than floats precisely because floats are now
//! in-subset — a float leaf would be JITted directly, not cross-tier.) This proves the cross-tier
//! call ABI — the emitted code marshals i64 arg/result slots through the env scratch and the engine
//! callback runs the leaf — by comparing the mixed run (emitted `f0` under `wasmi`, `call_interp`
//! wired to the real bytecode engine) against the full-interpreter oracle.

use std::sync::Arc;

use svm_interp::{bytecode, Value};
use svm_wasmjit::{compile_module_mixed, ENV_CELL_BYTES};
use wasmi::{Caller, Engine, Linker, Memory, MemoryType, Module as WModule, Store, Val};

const WIN_BASE: u32 = 0x1_0000;
const ENV_PTR: u32 = 1024;

fn parse(src: &str) -> svm_ir::Module {
    let m = svm_text::parse_module(src).expect("parse");
    svm_verify::verify_module(&m).expect("verify");
    m
}

/// Full-interpreter oracle: run the whole module's func 0 on the bytecode engine.
fn oracle(m: &svm_ir::Module, arg: i64) -> i64 {
    let mut fuel = u64::MAX;
    match bytecode::compile_and_run(m, 0, &[Value::I64(arg)], &mut fuel) {
        Some(Ok(vals)) => match vals.first() {
            Some(Value::I64(x)) => *x,
            Some(Value::I32(x)) => *x as i64,
            _ => panic!("oracle: unexpected result"),
        },
        other => panic!("oracle: {other:?}"),
    }
}

/// Mixed run: emit func 0 (and any other in-subset function) to wasm; wire `env.call_interp` to the
/// bytecode engine so an interp-leaf call runs on the interpreter. `Ok(result)`, or `Err` if the run
/// trapped (an interp leaf that traps makes the callback trap the wasm — it surfaces here).
fn mixed(m: &svm_ir::Module, arg: i64) -> Result<i64, ()> {
    let wasm = compile_module_mixed(m, false).expect("mixed-eligible");
    let engine = Engine::default();
    let module = WModule::new(&engine, &wasm).expect("emitted wasm validates");
    let mut store: Store<i32> = Store::new(&engine, 0);
    let memory = Memory::new(&mut store, MemoryType::new(2, None)).unwrap();
    // A large POSITIVE i64 fuel budget (the emitted fuel check traps when the i64 counter goes < 0).
    memory
        .write(
            &mut store,
            ENV_PTR as usize,
            &1_000_000_000i64.to_le_bytes(),
        )
        .unwrap();

    let mut linker: Linker<i32> = Linker::new(&engine);
    linker.define("env", "memory", memory).unwrap();
    linker
        .func_wrap::<_, ()>("env", "trap", |mut caller: Caller<'_, i32>, code: i32| {
            *caller.data_mut() = code;
        })
        .unwrap();

    // The cross-tier callback: read the leaf's i64 arg slots at `args_ptr`, run it on the bytecode
    // engine, write its i64 result slots back. On a trap, record it and trap the wasm (surfacing at
    // the top-level `f0` call, like the browser's JS import throwing).
    let mod_cb = Arc::new(m.clone());
    let mem = memory; // `Memory` is a Copy handle — capture it (the emitted module imports memory,
                      // so it isn't reachable via `caller.get_export`).
    linker
        .func_wrap(
            "env",
            "call_interp",
            move |mut caller: Caller<'_, i32>,
                  func: i32,
                  args_ptr: i32|
                  -> Result<(), wasmi::Error> {
                let callee = &mod_cb.funcs[func as usize];
                let slot = |data: &[u8], i: usize| -> u64 {
                    let o = args_ptr as usize + i * 8;
                    u64::from_le_bytes(data[o..o + 8].try_into().unwrap())
                };
                let args: Vec<Value> = {
                    let data = mem.data(&caller);
                    callee
                        .params
                        .iter()
                        .enumerate()
                        .map(|(i, t)| match t {
                            svm_ir::ValType::I32 => Value::I32(slot(data, i) as i32),
                            _ => Value::I64(slot(data, i) as i64),
                        })
                        .collect()
                };
                let mut fuel = u64::MAX;
                match bytecode::compile_and_run(&mod_cb, func as u32, &args, &mut fuel) {
                    Some(Ok(vals)) => {
                        let data = mem.data_mut(&mut caller);
                        for (i, v) in vals.iter().enumerate() {
                            let raw = match v {
                                Value::I32(x) => *x as u32 as u64,
                                Value::I64(x) => *x as u64,
                                _ => panic!("non-integer cross-tier result"),
                            };
                            let o = args_ptr as usize + i * 8;
                            data[o..o + 8].copy_from_slice(&raw.to_le_bytes());
                        }
                        Ok(())
                    }
                    Some(Err(_trap)) => {
                        // Record a nonzero marker and trap the wasm (kind parity is checked coarsely).
                        *caller.data_mut() = 99;
                        Err(wasmi::Error::from(
                            wasmi::core::TrapCode::UnreachableCodeReached,
                        ))
                    }
                    None => panic!("interp leaf unsupported by the engine"),
                }
            },
        )
        .unwrap();

    let instance = linker
        .instantiate(&mut store, &module)
        .unwrap()
        .start(&mut store)
        .unwrap();
    let f0 = instance.get_func(&store, "f0").expect("f0");
    let mut results = [Val::I64(0)];
    match f0.call(
        &mut store,
        &[
            Val::I32(WIN_BASE as i32),
            Val::I32(ENV_PTR as i32),
            Val::I64(arg),
        ],
        &mut results,
    ) {
        Ok(()) => Ok(match results[0] {
            Val::I64(x) => x,
            Val::I32(x) => x as i64,
            _ => panic!("unexpected result type"),
        }),
        Err(_) => Err(()),
    }
}

/// Whether the full-interpreter oracle traps running the whole module's func 0 with `arg`.
fn oracle_traps(m: &svm_ir::Module, arg: i64) -> bool {
    let mut fuel = u64::MAX;
    matches!(
        bytecode::compile_and_run(m, 0, &[Value::I64(arg)], &mut fuel),
        Some(Err(_))
    )
}

/// `env` cell must be at least `ENV_CELL_BYTES`; we put it at ENV_PTR with 2 pages of memory, so
/// assert the layout fits (ENV_PTR + ENV_CELL_BYTES < WIN_BASE, and the window fits in page 1).
#[test]
fn env_layout_fits() {
    assert!(ENV_PTR as usize + ENV_CELL_BYTES < WIN_BASE as usize);
}

const ARGS: &[i64] = &[0, 1, 2, 5, 20, 100, -1, -5, 1000];

/// Integer caller sums a **deferred-SIMD leaf** over `0..n`. The caller is emitted; the leaf (a
/// `i16x8.dot_i8x16_s` reduction — out of the emitter's subset, i64 signature, memory-free) runs via
/// `env.call_interp`. (The core v128 lane ops are now in-subset, so the cross-tier exemplar is the
/// deferred `dot` reduction — whatever it computes, the mixed run and the whole-interp oracle agree.)
const SUM_FLOAT_LEAF: &str = r#"
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const 0
  v2 = i64.const 0
  br block1(v0, v1, v2)
block1(v3: i64, v4: i64, v5: i64):
  v6 = i64.lt_s v5 v3
  br_if v6 block2(v3, v4, v5) block3(v4)
block2(v7: i64, v8: i64, v9: i64):
  v10 = call 1 (v9)
  v11 = i64.add v8 v10
  v12 = i64.const 1
  v13 = i64.add v9 v12
  br block1(v7, v11, v13)
block3(v14: i64):
  return v14
}
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64x2.splat v0
  v2 = i16x8.dot_i8x16_s v1 v1
  v3 = i64x2.extract_lane 0 v2
  return v3
}
"#;

#[test]
fn sum_float_leaf() {
    let m = parse(SUM_FLOAT_LEAF);
    for &arg in ARGS {
        assert_eq!(
            mixed(&m, arg).expect("no trap"),
            oracle(&m, arg),
            "mixed != oracle for arg {arg}"
        );
    }
}

/// Same shape but the leaf has an **i32 signature** — exercises the i32→i64 arg widening and the
/// i64→i32 result narrowing in the cross-tier marshalling.
const SUM_I32_LEAF: &str = r#"
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const 0
  v2 = i64.const 0
  br block1(v0, v1, v2)
block1(v3: i64, v4: i64, v5: i64):
  v6 = i64.lt_s v5 v3
  br_if v6 block2(v3, v4, v5) block3(v4)
block2(v7: i64, v8: i64, v9: i64):
  v10 = i32.wrap_i64 v9
  v11 = call 1 (v10)
  v12 = i64.extend_i32_s v11
  v13 = i64.add v8 v12
  v14 = i64.const 1
  v15 = i64.add v9 v14
  br block1(v7, v13, v15)
block3(v16: i64):
  return v16
}
func (i32) -> (i32) {
block0(v0: i32):
  v1 = i32x4.splat v0
  v2 = i16x8.dot_i8x16_s v1 v1
  v3 = i32x4.extract_lane 0 v2
  return v3
}
"#;

#[test]
fn sum_i32_leaf() {
    let m = parse(SUM_I32_LEAF);
    for &arg in ARGS {
        assert_eq!(
            mixed(&m, arg).expect("no trap"),
            oracle(&m, arg),
            "mixed != oracle for arg {arg}"
        );
    }
}

/// **Trap propagation across the tier boundary.** The leaf is deferred-SIMD-gated (a
/// `i16x8.dot_i8x16_s` keeps it out of subset → cross-tier) and computes `100 / arg`, which traps
/// (div-by-zero) exactly when `arg == 0` — the dot result is folded in *after* the divide, so the
/// trap is purely arg-driven. The mixed run must trap iff the full-interpreter oracle traps —
/// proving the `env.call_interp` callback's trap path (it traps the wasm, which unwinds to the
/// top-level `f0` caller, like the browser's JS import throwing).
const TRAPPING_LEAF: &str = r#"
func (i64) -> (i64) {
block0(v0: i64):
  v1 = call 1 (v0)
  return v1
}
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64x2.splat v0
  v2 = i16x8.dot_i8x16_s v1 v1
  v3 = i64x2.extract_lane 0 v1
  v4 = i64.const 100
  v5 = i64.div_s v4 v3
  v6 = i64x2.extract_lane 0 v2
  v7 = i64.add v5 v6
  return v7
}
"#;

#[test]
fn trapping_leaf_propagates() {
    let m = parse(TRAPPING_LEAF);
    for &arg in &[0i64, 1, 2, 5, -3, 100] {
        let want_trap = oracle_traps(&m, arg);
        let got_trap = mixed(&m, arg).is_err();
        assert_eq!(
            want_trap, got_trap,
            "cross-tier trap parity broke for arg {arg} (oracle_traps={want_trap})"
        );
        // At arg 0 the leaf must actually trap on both tiers (the test would be vacuous otherwise).
        if arg == 0 {
            assert!(want_trap, "expected the leaf to trap at arg 0");
        } else {
            assert_eq!(
                mixed(&m, arg).expect("no trap"),
                oracle(&m, arg),
                "non-trapping arg {arg} must still match"
            );
        }
    }
}
