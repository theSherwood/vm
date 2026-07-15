//! **Cross-tier calls over a shared window** (Doom-perf keystone). `compile_module_reactor` emits the
//! in-subset functions and routes a direct `Call` to any reachable, non-emitted, integer-signature
//! function — *not just* memory-free leaves — through `env.call_interp`. Such a callee may read and
//! write memory, so its `call_interp` callback must run it over the **same** window as the emitted
//! code. This test proves that: `f0` (emitted) writes `mem[8]`, calls the cross-tier `f1` (kept out of
//! subset by a `v128` op) which **reads `mem[8]` and writes `mem[100]`**, then `f0` reads `mem[100]`.
//! With a shared window the round trip yields `x+7`; a throwaway callee window would drop both stores
//! and yield `0` — so the differential against the full interpreter is exactly the shared-window proof.

use std::sync::Arc;

use svm_interp::{bytecode, Region, Value};
use svm_wasmjit::{compile_module_reactor, ENV_CELL_BYTES};
use wasmi::{Caller, Engine, Linker, Memory, MemoryType, Module as WModule, Store, Val};

const WIN_BASE: u32 = 0x1_0000; // the guest window starts at wasm offset 64 KiB (`memory 16`)
const WIN_SIZE: u64 = 1 << 16; // 64 KiB window
const ENV_PTR: u32 = 1024;

// f0 (emitted) writes mem[8]=x+7, calls f1, reads mem[100]. f1 (cross-tier: a `v128` op keeps it out
// of subset; it touches memory) reads mem[8] and writes it to mem[100]. So f0(x) = x+7 iff f1 ran over
// f0's window (both directions of sharing).
const SRC: &str = r#"
memory 16
func (i64) -> (i64) {
block0(v0: i64):
  v7 = i64.const 7
  vpre = i64.add v0 v7
  va8 = i64.const 8
  i64.store va8 vpre
  vr1 = call 1 (v0)
  vaddr = i64.const 100
  vr = i64.load vaddr
  return vr
}
func (i64) -> (i64) {
block0(v0: i64):
  va8 = i64.const 8
  vread = i64.load va8
  vaddr = i64.const 100
  i64.store vaddr vread
  vs = i64x2.splat v0
  vd = i16x8.dot_i8x16_s vs vs
  ve = i64x2.extract_lane 0 vd
  return ve
}
"#;

fn parse(src: &str) -> svm_ir::Module {
    let m = svm_text::parse_module(src).expect("parse");
    svm_verify::verify_module(&m).expect("verify");
    m
}

/// Full-interpreter oracle over func 0.
fn oracle(m: &svm_ir::Module, arg: i64) -> i64 {
    let mut fuel = u64::MAX;
    match bytecode::compile_and_run(m, 0, &[Value::I64(arg)], &mut fuel) {
        Some(Ok(v)) => match v.first() {
            Some(Value::I64(x)) => *x,
            Some(Value::I32(x)) => *x as i64,
            _ => panic!("oracle result"),
        },
        other => panic!("oracle: {other:?}"),
    }
}

/// Mixed run: `f0` on wasm (wasmi), the cross-tier `f1` on the interpreter **over the shared window**.
fn reactor_run(m: &svm_ir::Module, arg: i64) -> i64 {
    let (wasm, emitted) =
        compile_module_reactor(m, 0, false).expect("cross-tier reactor emittable");
    assert_eq!(emitted, vec![true, false], "f0 emits, f1 is cross-tier");

    let engine = Engine::default();
    let module = WModule::new(&engine, &wasm).expect("emitted wasm validates");
    let mut store: Store<i32> = Store::new(&engine, 0);
    let memory = Memory::new(&mut store, MemoryType::new(2, None)).unwrap();
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

    let mod_cb = Arc::new(m.clone());
    let mem = memory;
    linker
        .func_wrap(
            "env",
            "call_interp",
            move |mut caller: Caller<'_, i32>,
                  func: i32,
                  args_ptr: i32|
                  -> Result<(), wasmi::Error> {
                let callee = &mod_cb.funcs[func as usize];
                let args: Vec<Value> = {
                    let data = mem.data(&caller);
                    callee
                        .params
                        .iter()
                        .enumerate()
                        .map(|(i, t)| {
                            let o = args_ptr as usize + i * 8;
                            let raw = u64::from_le_bytes(data[o..o + 8].try_into().unwrap());
                            match t {
                                svm_ir::ValType::I32 => Value::I32(raw as i32),
                                _ => Value::I64(raw as i64),
                            }
                        })
                        .collect()
                };
                // The SHARED window: a Region over the wasm memory at the guest window base, so the
                // interpreter's stores land in the same bytes the emitted code reads.
                let base = mem.data_mut(&mut caller).as_mut_ptr();
                // SAFETY: single-threaded; the wasm memory (2 pages) outlives this call and does not
                // grow, so `base + WIN_BASE` addresses `WIN_SIZE` valid bytes for the interpreter run.
                let back =
                    Arc::new(unsafe { Region::shared(base.add(WIN_BASE as usize), WIN_SIZE) });
                let mut fuel = u64::MAX;
                match bytecode::compile_and_run_capture_over(
                    &mod_cb,
                    func as u32,
                    &args,
                    &mut fuel,
                    &[],
                    back,
                ) {
                    Some((Ok(vals), _)) => {
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
                    Some((Err(_), _)) => {
                        *caller.data_mut() = 99;
                        Err(wasmi::Error::from(
                            wasmi::core::TrapCode::UnreachableCodeReached,
                        ))
                    }
                    None => panic!("cross-tier callee unsupported by the engine"),
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
    f0.call(
        &mut store,
        &[
            Val::I32(WIN_BASE as i32),
            Val::I32(ENV_PTR as i32),
            Val::I64(arg),
        ],
        &mut results,
    )
    .expect("f0 runs");
    match results[0] {
        Val::I64(x) => x,
        Val::I32(x) => x as i64,
        _ => panic!("result type"),
    }
}

#[test]
fn env_layout_fits() {
    assert!(ENV_PTR as usize + ENV_CELL_BYTES < WIN_BASE as usize);
}

#[test]
fn cross_tier_shares_the_window() {
    let m = parse(SRC);
    for &arg in &[0i64, 1, 42, 1000, -5] {
        let got = reactor_run(&m, arg);
        assert_eq!(got, oracle(&m, arg), "mixed != oracle for arg {arg}");
        assert_eq!(
            got,
            arg + 7,
            "f1's reads+writes must land in f0's window (arg {arg})"
        );
    }
}

// Same shared-window round trip, but f0 reaches f1 through a **`call_indirect`** rather than a direct
// call: `ref.func 1` forms the funcref, then `call_indirect (i64)->(i64)` dispatches through the
// identity table. f1 is cross-tier (a `v128` op keeps it out of subset), so its table slot holds a
// **trampoline** that bounces to `env.call_interp` — proving indirect calls reach the interpreter over
// the same window. Using `call_indirect` also forces `has_indirect`, so the reactor path only stays
// emittable because the trampoline routes the cross-tier target (it would otherwise fall back whole).
const SRC_INDIRECT: &str = r#"
memory 16
func (i64) -> (i64) {
block0(v0: i64):
  v7 = i64.const 7
  vpre = i64.add v0 v7
  va8 = i64.const 8
  i64.store va8 vpre
  vf = ref.func 1
  vr1 = call_indirect (i64) -> (i64) vf (v0)
  vaddr = i64.const 100
  vr = i64.load vaddr
  return vr
}
func (i64) -> (i64) {
block0(v0: i64):
  va8 = i64.const 8
  vread = i64.load va8
  vaddr = i64.const 100
  i64.store vaddr vread
  vs = i64x2.splat v0
  vd = i16x8.dot_i8x16_s vs vs
  ve = i64x2.extract_lane 0 vd
  return ve
}
"#;

#[test]
fn cross_tier_indirect_trampoline() {
    let m = parse(SRC_INDIRECT);
    for &arg in &[0i64, 1, 42, 1000, -5] {
        let got = reactor_run(&m, arg);
        assert_eq!(
            got,
            oracle(&m, arg),
            "indirect mixed != oracle for arg {arg}"
        );
        assert_eq!(
            got,
            arg + 7,
            "the trampoline must run f1 over f0's window (arg {arg})"
        );
    }
}

// Same shared-window round trip, but f0 reaches f1 through a **`return_call`** (tail call) rather than
// a direct call: f0 writes mem[8]=x+7 then `return_call 1 (v0)` — its result *is* f1's. f1 is
// cross-tier (a `v128` op keeps it out of subset) and reads mem[8] back, so the tail call must marshal
// through `env.call_interp` and return the callee's result over the shared window. This exercises the
// emitter's cross-tier tail-call lowering (env scratch marshal → call_interp → load results → return).
const SRC_TAILCALL: &str = r#"
memory 16
func (i64) -> (i64) {
block0(v0: i64):
  v7 = i64.const 7
  vpre = i64.add v0 v7
  va8 = i64.const 8
  i64.store va8 vpre
  return_call 1 (v0)
}
func (i64) -> (i64) {
block0(v0: i64):
  va8 = i64.const 8
  vread = i64.load va8
  vs = i64x2.splat v0
  vd = i16x8.dot_i8x16_s vs vs
  ve = i64x2.extract_lane 0 vd
  vz = i64.const 0
  vfold = i64.mul ve vz
  vout = i64.add vread vfold
  return vout
}
"#;

#[test]
fn cross_tier_tail_call() {
    let m = parse(SRC_TAILCALL);
    for &arg in &[0i64, 1, 42, 1000, -5] {
        let got = reactor_run(&m, arg);
        assert_eq!(
            got,
            oracle(&m, arg),
            "tail-call mixed != oracle for arg {arg}"
        );
        assert_eq!(
            got,
            arg + 7,
            "the cross-tier tail call must run f1 over f0's window (arg {arg})"
        );
    }
}
