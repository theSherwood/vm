//! Coverage for the two capabilities the cross-engine bench needs (BROWSER.md § "wasm-JIT tier"):
//! **entry-rooting** (`compile_module_mixed_entry` — the JIT entry is an arbitrary function, not
//! func 0) and **data tolerance** (the emitter no longer rejects `data` segments; the host lays them
//! into the window before running). Both are exercised by emitting under `wasmi`, initializing the
//! window from `m.data`, and comparing to the bytecode-engine oracle.

use svm_wasmjit::{analyze_from, compile_module_mixed_entry};
use wasmi::{Caller, Engine, Linker, Memory, MemoryType, Module as WModule, Store, Val};

const WIN_BASE: u32 = 0x1_0000;
const ENV_PTR: u32 = 1024;

fn parse(src: &str) -> svm_ir::Module {
    let m = svm_text::parse_module(src).expect("parse");
    svm_verify::verify_module(&m).expect("verify");
    m
}

/// Run the emitted `f{entry}` under wasmi over a window seeded with `m.data`, returning its i64
/// result. `arg` is the single i64 param.
fn jit_run(m: &svm_ir::Module, entry: u32, arg: i64) -> i64 {
    let wasm = compile_module_mixed_entry(m, entry, false).expect("mixed-eligible");
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
    // Host window init: lay each data segment at WIN_BASE + offset (what the browser/bench linkers do).
    for seg in &m.data {
        memory
            .write(
                &mut store,
                WIN_BASE as usize + seg.offset as usize,
                &seg.bytes,
            )
            .unwrap();
    }
    let mut linker: Linker<i32> = Linker::new(&engine);
    linker.define("env", "memory", memory).unwrap();
    linker
        .func_wrap::<_, ()>("env", "trap", |mut c: Caller<'_, i32>, code: i32| {
            *c.data_mut() = code
        })
        .unwrap();
    linker
        .func_wrap::<_, ()>(
            "env",
            "call_interp",
            |_: Caller<'_, i32>, _: i32, _: i32| unreachable!("no interp leaf in these kernels"),
        )
        .unwrap();
    let instance = linker
        .instantiate(&mut store, &module)
        .unwrap()
        .start(&mut store)
        .unwrap();
    let f = instance
        .get_func(&store, &format!("f{entry}"))
        .expect("entry export");
    let mut out = [Val::I64(0)];
    f.call(
        &mut store,
        &[
            Val::I32(WIN_BASE as i32),
            Val::I32(ENV_PTR as i32),
            Val::I64(arg),
        ],
        &mut out,
    )
    .expect("no trap");
    match out[0] {
        Val::I64(x) => x,
        Val::I32(x) => x as i64,
        _ => panic!("unexpected result"),
    }
}

fn oracle(m: &svm_ir::Module, entry: u32, arg: i64) -> i64 {
    let mut fuel = u64::MAX;
    match bytecode_run(m, entry, arg, &mut fuel) {
        Some(x) => x,
        None => panic!("oracle failed"),
    }
}

fn bytecode_run(m: &svm_ir::Module, entry: u32, arg: i64, fuel: &mut u64) -> Option<i64> {
    match svm_interp::bytecode::compile_and_run(m, entry, &[svm_interp::Value::I64(arg)], fuel) {
        Some(Ok(v)) => Some(match v.first() {
            Some(svm_interp::Value::I64(x)) => *x,
            Some(svm_interp::Value::I32(x)) => *x as i64,
            _ => panic!("result"),
        }),
        _ => None,
    }
}

/// Two kernels; **func 1** (not func 0) is the JIT entry — proving entry-rooted eligibility and that
/// the emitted export is `f1`. Func 0 uses a **deferred SIMD** op (`i16x8.dot_i8x16_s` — the core
/// v128 lane ops are now in-subset, but the widening/reduction family isn't), so it's out of subset,
/// which entry-rooting at func 1 correctly ignores.
const ENTRY_ONE: &str = r#"
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64x2.splat v0
  v2 = i16x8.dot_i8x16_s v1 v1
  v3 = i64x2.extract_lane 0 v2
  return v3
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 3
  v2 = i64.mul v0 v1
  v3 = i64.const 7
  v4 = i64.add v2 v3
  return v4
  }
}
"#;

#[test]
fn entry_not_func_zero() {
    let m = parse(ENTRY_ONE);
    // func 0 is SIMD → not eligible as an entry; func 1 is pure integer → eligible.
    assert!(!analyze_from(&m, 0).mixed_ok);
    assert!(analyze_from(&m, 1).mixed_ok);
    for arg in [0i64, 1, 7, -5, 1000] {
        assert_eq!(jit_run(&m, 1, arg), oracle(&m, 1, arg), "entry-1 arg {arg}");
    }
}

/// A kernel that **reads a data segment**: the window is seeded with 8 little-endian bytes at
/// offset 0 (the i64 `0x0102030405060708`); the guest loads them and adds `arg`. Proves the emitter
/// tolerates `data` and the host window-init makes the load see the initialized bytes.
const DATA_READ: &str = r#"
memory 16
data 0 "\08\07\06\05\04\03\02\01"
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 0
  v2 = i64.load v1
  v3 = i64.add v2 v0
  return v3
  }
}
"#;

#[test]
fn data_segment_read() {
    let m = parse(DATA_READ);
    assert!(!m.data.is_empty(), "kernel must carry a data segment");
    for arg in [0i64, 1, 100, -7] {
        assert_eq!(
            jit_run(&m, 0, arg),
            oracle(&m, 0, arg),
            "data-read arg {arg}"
        );
    }
}
