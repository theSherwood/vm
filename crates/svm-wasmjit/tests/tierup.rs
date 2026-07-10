//! The threads-tier gate (`BROWSER.md` § "wasm-JIT tier", per-Worker JIT). Unlike `differential.rs`
//! (which JITs a whole func-0-rooted kernel), a threaded guest keeps running on the resumable
//! interpreter and *tiers up* a direct `Call` to any emitted function. [`compile_module_tierup`]
//! decides that emit set: every in-subset function whose calls all route, **regardless** of func-0
//! reachability — so a pure compute leaf reachable only through `thread.spawn` still emits. These
//! tests pin the eligibility bitmap and differential-run each emitted `f{i}` against the bytecode
//! interpreter, the same MISCOMPILE-grade contract the whole-module path holds.

use svm_interp::{bytecode, Trap, Value};
use svm_wasmjit::{compile_module_tierup, TRAP_MEMORY_FAULT, TRAP_OUT_OF_FUEL};
use wasmi::{Caller, Engine, Linker, Memory, MemoryType, Module as WModule, Store, Val};

const WIN_BASE: u32 = 0x1_0000;
const ENV_PTR: u32 = 1024;
const FUEL: u64 = 100_000_000;

#[derive(Debug, PartialEq)]
enum Outcome {
    Vals(Vec<Value>),
    Trap(TrapKind),
}
#[derive(Debug, PartialEq, Clone, Copy)]
enum TrapKind {
    DivByZero,
    OverflowOrConv,
    MemoryFault,
    Unreachable,
    OutOfFuel,
    Other,
}

/// Run SVM function `func` standalone on the bytecode interpreter (the oracle for what the emitted
/// `f{func}` region must compute).
fn oracle(m: &svm_ir::Module, func: u32, args: &[Value]) -> Outcome {
    let mut fuel = FUEL;
    match bytecode::compile_and_run(m, func, args, &mut fuel) {
        None => panic!("oracle: module unsupported by the bytecode engine"),
        Some(Ok(vals)) => Outcome::Vals(vals),
        Some(Err(t)) => Outcome::Trap(match t {
            Trap::DivByZero => TrapKind::DivByZero,
            Trap::IntOverflow | Trap::BadConversion => TrapKind::OverflowOrConv,
            Trap::MemoryFault => TrapKind::MemoryFault,
            Trap::Unreachable => TrapKind::Unreachable,
            Trap::OutOfFuel => TrapKind::OutOfFuel,
            _ => TrapKind::Other,
        }),
    }
}

/// Call the emitted `f{func}` under wasmi over the window at `WIN_BASE`. A cross-tier `call_interp`
/// runs the named leaf on the interpreter (so an emitted caller can reach an interp leaf), matching
/// the browser host's `svm_wasmjit_call_interp`.
fn wasm_run(m: &svm_ir::Module, wasm: &[u8], func: u32, args: &[Value]) -> Outcome {
    let engine = Engine::default();
    let module = WModule::new(&engine, wasm).expect("emitted wasm must validate");
    let mut store: Store<i32> = Store::new(&engine, 0);
    let pages = 2 + m.memory.map_or(0, |mc| (mc.size() >> 16) as u32);
    let memory = Memory::new(&mut store, MemoryType::new(pages, None)).unwrap();
    memory
        .write(&mut store, ENV_PTR as usize, &(FUEL as i64).to_le_bytes())
        .unwrap();
    let mut linker: Linker<i32> = Linker::new(&engine);
    linker.define("env", "memory", memory).unwrap();
    linker
        .func_wrap("env", "trap", |mut caller: Caller<'_, i32>, code: i32| {
            *caller.data_mut() = code;
        })
        .unwrap();
    // These tier-up leaves are pure (no cross-tier calls in the differential guests), so the import
    // is present but a call would be a bug — assert it never fires.
    linker
        .func_wrap::<_, ()>(
            "env",
            "call_interp",
            |_: Caller<'_, i32>, _f: i32, _a: i32| {
                unreachable!("no cross-tier call expected in these tier-up leaves");
            },
        )
        .unwrap();
    let instance = linker
        .instantiate(&mut store, &module)
        .unwrap()
        .start(&mut store)
        .unwrap();
    let f = instance
        .get_func(&store, &format!("f{func}"))
        .unwrap_or_else(|| panic!("f{func} not exported"));

    let sig = &m.funcs[func as usize];
    let mut params = vec![Val::I32(WIN_BASE as i32), Val::I32(ENV_PTR as i32)];
    for (t, a) in sig.params.iter().zip(args) {
        params.push(match (t, a) {
            (svm_ir::ValType::I32, Value::I32(v)) => Val::I32(*v),
            (svm_ir::ValType::I64, Value::I64(v)) => Val::I64(*v),
            _ => panic!("arg/type mismatch"),
        });
    }
    let mut results: Vec<Val> = sig
        .results
        .iter()
        .map(|t| match t {
            svm_ir::ValType::I32 => Val::I32(0),
            _ => Val::I64(0),
        })
        .collect();

    match f.call(&mut store, &params, &mut results) {
        Ok(()) => Outcome::Vals(
            results
                .iter()
                .map(|v| match v {
                    Val::I32(x) => Value::I32(*x),
                    Val::I64(x) => Value::I64(*x),
                    _ => panic!("non-integer result"),
                })
                .collect(),
        ),
        Err(e) => {
            let host_code = *store.data();
            let kind = if host_code == TRAP_OUT_OF_FUEL {
                TrapKind::OutOfFuel
            } else if host_code == TRAP_MEMORY_FAULT {
                TrapKind::MemoryFault
            } else {
                use wasmi::core::TrapCode;
                match e.as_trap_code() {
                    Some(TrapCode::IntegerDivisionByZero) => TrapKind::DivByZero,
                    Some(TrapCode::IntegerOverflow | TrapCode::BadConversionToInteger) => {
                        TrapKind::OverflowOrConv
                    }
                    Some(TrapCode::UnreachableCodeReached) => TrapKind::Unreachable,
                    other => panic!("unexpected wasm trap: {other:?} ({e})"),
                }
            };
            Outcome::Trap(kind)
        }
    }
}

fn build(src: &str) -> svm_ir::Module {
    let m = svm_text::parse_module(src).expect("parse");
    svm_verify::verify_module(&m).expect("verify");
    m
}

/// Differential-run every emitted `f{i}` against the interpreter across an i64 arg sweep.
fn diff_eligible(m: &svm_ir::Module, eligible: &[bool], wasm: &[u8], sweep: &[i64]) {
    let mut ran = 0;
    for (i, &e) in eligible.iter().enumerate() {
        if !e {
            continue;
        }
        let f = &m.funcs[i];
        // All-i64 leaves only (what the browser tier-up ABI marshals); skip any i32/float/v128 sigs.
        if !f
            .params
            .iter()
            .chain(&f.results)
            .all(|t| *t == svm_ir::ValType::I64)
        {
            continue;
        }
        let arity = f.params.len();
        // Fill every param with the sweep value (a spawned worker body is `(sp, arg)` — 2 i64
        // params); both engines get identical args, so the differential stays exact.
        let sweeps: Vec<Vec<Value>> = if arity == 0 {
            vec![vec![]]
        } else {
            sweep.iter().map(|a| vec![Value::I64(*a); arity]).collect()
        };
        for args in &sweeps {
            let want = oracle(m, i as u32, args);
            let got = wasm_run(m, wasm, i as u32, args);
            assert_eq!(want, got, "tier-up MISCOMPILE: f{i} args {args:?}");
            ran += 1;
        }
    }
    assert!(
        ran > 0,
        "vacuous: no eligible all-i64 function was differential-run"
    );
}

const SWEEP: &[i64] = &[0, 1, 2, 5, 64, 1000, -1, -1000, 100_000, i64::MIN, i64::MAX];

/// The flagship shape: func 0 spawns worker vCPUs (concurrency — never JITs), the worker func 1
/// increments a shared counter through an atomic (concurrency — never JITs) by an amount computed
/// in the pure leaf func 2. Only func 2 is emitted + eligible, though it is reachable **only**
/// through `thread.spawn` from func 0.
const SPAWN: &str = r#"
memory 16
func () -> (i64) {
block0():
  v0 = i64.const 500
  v1 = thread.spawn 1 v0 v0
  v2 = thread.join v1
  v3 = i64.const 0
  v4 = i64.atomic.load v3
  return v4
}
func (i64, i64) -> (i64) {
block0(vsp: i64, v0: i64):
  v1 = call 2 (v0)
  v2 = i64.const 0
  v3 = i64.atomic.rmw.add v2 v1
  v4 = i64.const 0
  return v4
}
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const 2
  v2 = i64.mul v0 v1
  v3 = i64.sub v2 v0
  return v3
}
"#;

#[test]
fn spawn_only_leaf_is_eligible_and_matches() {
    let m = build(SPAWN);
    let (wasm, eligible) = compile_module_tierup(&m, false).expect("tier-up emit");
    // func 0 (thread.spawn) and func 1 (atomic.rmw) are concurrency → not in-subset → not emitted;
    // func 2 is a pure i64 leaf reachable only via spawn → emitted + eligible.
    assert_eq!(eligible, vec![false, false, true], "eligibility bitmap");
    diff_eligible(&m, &eligible, &wasm, SWEEP);
}

/// A pure leaf that itself calls a deeper pure leaf: the fixpoint must keep BOTH (an emitted caller's
/// direct call routes to an emitted target), so the tier-up module can run either as an entry.
const TRANSITIVE: &str = r#"
func (i64) -> (i64) {
block0(v0: i64):
  v1 = thread.spawn 1 v0 v0
  v2 = thread.join v1
  return v2
}
func (i64, i64) -> (i64) {
block0(vsp: i64, v0: i64):
  v1 = call 2 (v0)
  return v1
}
func (i64) -> (i64) {
block0(v0: i64):
  v1 = call 3 (v0)
  v2 = i64.const 7
  v3 = i64.add v1 v2
  return v3
}
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const 3
  v2 = i64.mul v0 v1
  return v2
}
"#;

#[test]
fn transitive_pure_leaves_both_emitted() {
    let m = build(TRANSITIVE);
    let (wasm, eligible) = compile_module_tierup(&m, false).expect("tier-up emit");
    // func 0 (spawn) + func 1 (spawn worker via thread.spawn? no — func 1 is a plain worker body that
    // directly calls func 2). func 1 is in-subset (pure call), func 2 + func 3 pure. Only func 0 uses
    // concurrency. So funcs 1,2,3 are all emitted; func 0 is not.
    assert_eq!(
        eligible,
        vec![false, true, true, true],
        "eligibility bitmap"
    );
    diff_eligible(&m, &eligible, &wasm, SWEEP);
}

/// An in-subset function that calls a **non-leaf, non-subset** function (one using atomics) must be
/// dropped — its emitted body would carry an unroutable `Call`. The fixpoint removes it; the deeper
/// pure leaf still emits.
const UNROUTABLE_CALLER: &str = r#"
memory 16
func (i64) -> (i64) {
block0(v0: i64):
  v1 = call 1 (v0)
  return v1
}
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const 0
  v2 = i64.atomic.rmw.add v1 v0
  v3 = call 2 (v2)
  return v3
}
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const 11
  v2 = i64.add v0 v1
  return v2
}
"#;

#[test]
fn caller_of_nonleaf_is_dropped() {
    let m = build(UNROUTABLE_CALLER);
    let (wasm, eligible) = compile_module_tierup(&m, false).expect("tier-up emit");
    // func 1 uses atomics (not in-subset, not a leaf — it has memory + a call). func 0 is in-subset
    // but calls func 1 (non-emitted, non-leaf) → dropped by the fixpoint. func 2 is a pure leaf →
    // emitted. Result: only func 2 eligible.
    assert_eq!(eligible, vec![false, false, true], "eligibility bitmap");
    diff_eligible(&m, &eligible, &wasm, SWEEP);
}

/// A fully in-subset module still tier-ups every function (the degenerate case — same emit set as
/// the whole-module path), so enabling tier-up never regresses a JITtable guest.
const ALL_SUBSET: &str = r#"
func (i64) -> (i64) {
block0(v0: i64):
  v1 = call 1 (v0)
  v2 = i64.const 1
  v3 = i64.add v1 v2
  return v3
}
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const 10
  v2 = i64.mul v0 v1
  return v2
}
"#;

#[test]
fn all_in_subset_all_eligible() {
    let m = build(ALL_SUBSET);
    let (wasm, eligible) = compile_module_tierup(&m, false).expect("tier-up emit");
    assert_eq!(eligible, vec![true, true], "eligibility bitmap");
    diff_eligible(&m, &eligible, &wasm, SWEEP);
}
