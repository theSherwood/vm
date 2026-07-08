//! The slice-1 gate: every kernel runs on the **bytecode engine** (the oracle) and on the
//! **emitted wasm** under `wasmi`, comparing results *and trap kinds* across an arg sweep. A
//! mismatch is a `MISCOMPILE`-grade failure — the emitter is escape-TCB-adjacent (its output
//! confines guest addresses), so this differential is the correctness contract, exactly like the
//! `svm-bytecode-wasm` bench row's cross-check.

use svm_interp::{bytecode, Trap, Value};
use svm_wasmjit::{compile_module, TRAP_MEMORY_FAULT, TRAP_OUT_OF_FUEL};
use wasmi::{Caller, Engine, Linker, Memory, MemoryType, Module as WModule, Store, Val};

/// Where the guest window starts in the harness's linear memory (page 1 — page 0 holds the env
/// cell, so an emitter bug that misses the `win` rebase lands *outside* the window and diverges).
const WIN_BASE: u32 = 0x1_0000;
/// The engine-side env cell (the i64 fuel counter) — outside the window, in "engine memory".
const ENV_PTR: u32 = 1024;

/// What one engine produced: values, or a trap kind (the SVM taxonomy).
#[derive(Debug, PartialEq)]
enum Outcome {
    Vals(Vec<Value>),
    Trap(TrapKind),
}

/// The trap kinds both engines can express. wasm's own traps (div-by-zero, overflow,
/// unreachable) map from `wasmi::core::TrapCode`; the SVM-specific ones (guard fault, fuel)
/// arrive via the `env.trap` host call.
#[derive(Debug, PartialEq, Clone, Copy)]
enum TrapKind {
    DivByZero,
    IntOverflow,
    MemoryFault,
    Unreachable,
    OutOfFuel,
    Other,
}

fn oracle(m: &svm_ir::Module, args: &[Value], fuel: u64) -> Outcome {
    let mut fuel = fuel;
    match bytecode::compile_and_run(m, 0, args, &mut fuel) {
        None => panic!("oracle: module unsupported by the bytecode engine"),
        Some(Ok(vals)) => Outcome::Vals(vals),
        Some(Err(t)) => Outcome::Trap(match t {
            Trap::DivByZero => TrapKind::DivByZero,
            Trap::IntOverflow => TrapKind::IntOverflow,
            Trap::MemoryFault => TrapKind::MemoryFault,
            Trap::Unreachable => TrapKind::Unreachable,
            Trap::OutOfFuel => TrapKind::OutOfFuel,
            _ => TrapKind::Other,
        }),
    }
}

/// Run the emitted wasm's `f0` under wasmi with the window at `WIN_BASE` and `fuel` in the env
/// cell. The host state is the last `env.trap` code (0 = none).
fn wasm_run(m: &svm_ir::Module, wasm: &[u8], args: &[Value], fuel: u64) -> Outcome {
    let engine = Engine::default();
    let module = WModule::new(&engine, wasm).expect("emitted wasm must validate");
    let mut store: Store<i32> = Store::new(&engine, 0);
    // Window size ≤ 64 KiB in these kernels: 2 pages (env page + window page) always suffice.
    let pages = 2 + m.memory.map_or(0, |mc| (mc.size() >> 16) as u32);
    let memory = Memory::new(&mut store, MemoryType::new(pages, None).unwrap()).unwrap();
    memory
        .write(&mut store, ENV_PTR as usize, &(fuel as i64).to_le_bytes())
        .unwrap();
    let mut linker: Linker<i32> = Linker::new(&engine);
    linker.define("env", "memory", memory).unwrap();
    linker
        .func_wrap("env", "trap", |mut caller: Caller<'_, i32>, code: i32| {
            *caller.data_mut() = code;
        })
        .unwrap();
    let instance = linker
        .instantiate(&mut store, &module)
        .unwrap()
        .start(&mut store)
        .unwrap();
    let f = instance.get_func(&store, "f0").expect("f0 exported");

    let mut params = vec![Val::I32(WIN_BASE as i32), Val::I32(ENV_PTR as i32)];
    for (t, a) in m.funcs[0].params.iter().zip(args) {
        params.push(match (t, a) {
            (svm_ir::ValType::I32, Value::I32(v)) => Val::I32(*v),
            (svm_ir::ValType::I64, Value::I64(v)) => Val::I64(*v),
            _ => panic!("arg/type mismatch"),
        });
    }
    let mut results: Vec<Val> = m.funcs[0]
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
                    Some(TrapCode::IntegerOverflow) => TrapKind::IntOverflow,
                    Some(TrapCode::UnreachableCodeReached) => TrapKind::Unreachable,
                    other => panic!("unexpected wasm trap: {other:?} ({e})"),
                }
            };
            Outcome::Trap(kind)
        }
    }
}

/// Parse + verify a kernel, then differential-run it over `args` (each as the single i64 param
/// unless the kernel takes none). Plenty of fuel by default; `fuel` overrides for the fuel test.
fn diff(name: &str, src: &str, sweep: &[i64], fuel: u64) {
    let m = svm_text::parse_module(src).unwrap_or_else(|e| panic!("{name}: {e}"));
    svm_verify::verify_module(&m).unwrap_or_else(|e| panic!("{name}: verify: {e:?}"));
    let wasm = compile_module(&m).unwrap_or_else(|e| panic!("{name}: emit: {e}"));
    let arity = m.funcs[0].params.len();
    let sweeps: Vec<Vec<Value>> = if arity == 0 {
        vec![vec![]]
    } else {
        sweep.iter().map(|a| vec![Value::I64(*a)]).collect()
    };
    for args in &sweeps {
        let want = oracle(&m, args, fuel);
        let got = wasm_run(&m, &wasm, args, fuel);
        assert_eq!(want, got, "{name}: MISCOMPILE for args {args:?}");
    }
}

const ARGS: &[i64] = &[0, 1, 2, 5, 64, 1000, -1, -1000, 100_000, i64::MIN, i64::MAX];
const FUEL: u64 = 100_000_000;

/// The §ROI "alu" i64-LCG recurrence — loops, wrapping mul/add, branches, block args.
const ALU: &str = r#"
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const 0
  v2 = i64.const 0
  br block1(v0, v1, v2)
block1(v3: i64, v4: i64, v5: i64):
  v6 = i64.lt_s v5 v3
  br_if v6 block2(v3, v4, v5) block3(v4)
block2(v7: i64, v8: i64, v9: i64):
  v10 = i64.const 6364136223846793005
  v11 = i64.mul v8 v10
  v12 = i64.const 1442695040888963407
  v13 = i64.add v11 v12
  v14 = i64.add v13 v9
  v15 = i64.const 1
  v16 = i64.add v9 v15
  br block1(v7, v14, v16)
block3(v17: i64):
  return v17
}
"#;

#[test]
fn alu() {
    diff("alu", ALU, ARGS, FUEL);
}

/// Multi-function direct call in the loop body (the env threading + result plumbing).
const CALL: &str = r#"
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const 0
  v2 = i64.const 0
  br block1(v0, v1, v2)
block1(v3: i64, v4: i64, v5: i64):
  v6 = i64.lt_s v5 v3
  br_if v6 block2(v3, v4, v5) block3(v4)
block2(v7: i64, v8: i64, v9: i64):
  v10 = call 1 (v8, v9)
  v11 = i64.const 1
  v12 = i64.add v9 v11
  br block1(v7, v10, v12)
block3(v13: i64):
  return v13
}
func (i64, i64) -> (i64) {
block0(v0: i64, v1: i64):
  v2 = i64.add v0 v1
  return v2
}
"#;

#[test]
fn call() {
    diff("call", CALL, ARGS, FUEL);
}

/// Store→load through the confined window every iteration (mask + guard on the hot path).
const MEM: &str = r#"
memory 16
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const 0
  v2 = i64.const 0
  br block1(v0, v1, v2)
block1(v3: i64, v4: i64, v5: i64):
  v6 = i64.lt_s v5 v3
  br_if v6 block2(v3, v4, v5) block3(v4)
block2(v7: i64, v8: i64, v9: i64):
  v10 = i64.const 8
  i64.store v10 v8
  v11 = i64.load v10
  v12 = i64.add v11 v9
  v13 = i64.const 1
  v14 = i64.add v9 v13
  br block1(v7, v12, v14)
block3(v15: i64):
  return v15
}
"#;

#[test]
fn mem() {
    diff("mem", MEM, ARGS, FUEL);
}

/// Narrow stores + sign/zero-extending narrow loads (every width through the confinement).
const MEM_NARROW: &str = r#"
memory 16
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const 3
  i64.store8 v1 v0
  v2 = i64.const 40
  i64.store16 v2 v0
  v3 = i64.const 80
  i64.store32 v3 v0
  v4 = i64.load8_s v1
  v5 = i64.load8_u v1
  v6 = i64.load16_s v2
  v7 = i64.load16_u v2
  v8 = i64.load32_s v3
  v9 = i64.load32_u v3
  v10 = i64.add v4 v5
  v11 = i64.add v6 v7
  v12 = i64.add v8 v9
  v13 = i64.add v10 v11
  v14 = i64.add v13 v12
  return v14
}
"#;

#[test]
fn mem_narrow() {
    diff("mem_narrow", MEM_NARROW, ARGS, FUEL);
}

/// A store far past the window: the mask folds it into the reserved domain and the guard check
/// faults it — both engines must agree on WHICH addresses fault (the §4 contract).
const MEM_OOB: &str = r#"
memory 16
func (i64) -> (i64) {
block0(v0: i64):
  i64.store v0 v0
  v1 = i64.load v0
  return v1
}
"#;

#[test]
fn mem_oob() {
    // In-window, boundary-straddling (65536-7..65536-1), masked-back-in-window (2^40 + 8 → 8),
    // and far outside — the sweep crosses the guard on both sides.
    let probe: &[i64] = &[
        0,
        8,
        65528,
        65529,
        65535,
        65536,
        65537,
        1 << 20,
        (1i64 << 40) + 8,
        -1,
        -8,
        i64::MIN,
    ];
    diff("mem_oob", MEM_OOB, probe, FUEL);
}

/// div/rem for both signednesses: /0 traps, INT_MIN/-1 traps div_s but not rem_s.
const DIVREM: &str = r#"
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const -1
  v2 = i64.div_s v0 v1
  v3 = i64.const 7
  v4 = i64.rem_s v2 v3
  v5 = i64.div_u v4 v3
  v6 = i64.rem_u v0 v3
  v7 = i64.add v5 v6
  return v7
}
"#;

#[test]
fn divrem() {
    diff("divrem", DIVREM, ARGS, FUEL);
}

/// Division by an argument — zero and the overflow pair land in the sweep.
const DIVARG: &str = r#"
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const -9223372036854775808
  v2 = i64.div_s v1 v0
  return v2
}
"#;

#[test]
fn divarg() {
    diff("divarg", DIVARG, ARGS, FUEL);
}

/// Shifts/rotates (amounts mod bitwidth), bit ops, clz/ctz/popcnt, i32↔i64 conversions,
/// unsigned compares, select, eqz — the scalar long tail on both widths.
const BITS: &str = r#"
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const 13
  v2 = i64.shl v0 v1
  v3 = i64.const 17
  v4 = i64.shr_u v0 v3
  v5 = i64.xor v2 v4
  v6 = i64.const 71
  v7 = i64.rotl v5 v6
  v8 = i64.rotr v0 v1
  v9 = i64.or v7 v8
  v10 = i64.clz v9
  v11 = i64.ctz v9
  v12 = i64.popcnt v9
  v13 = i32.wrap_i64 v9
  v14 = i32.clz v13
  v15 = i32.const 3
  v16 = i32.shr_s v14 v15
  v17 = i64.extend_i32_u v16
  v18 = i64.extend_i32_s v13
  v19 = i64.lt_u v18 v0
  v20 = i64.extend_i32_u v19
  v21 = i64.eqz v9
  v22 = i64.extend_i32_s v21
  v23 = i64.add v10 v11
  v24 = i64.add v23 v12
  v25 = i64.add v24 v17
  v26 = i64.add v25 v20
  v27 = i64.add v26 v22
  v28 = i64.ge_u v27 v0
  v29 = select v28 v27 v18
  v30 = i64.extend8_s v27
  v31 = i64.extend16_s v27
  v32 = i64.extend32_s v27
  v33 = i64.add v30 v31
  v34 = i64.add v33 v32
  v35 = i64.add v34 v29
  return v35
}
"#;

#[test]
fn bits() {
    diff("bits", BITS, ARGS, FUEL);
}

/// br_table dispatch — each arm carries different block args (the per-edge landing blocks).
const BRTABLE: &str = r#"
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i32.wrap_i64 v0
  v2 = i64.const 100
  v3 = i64.const 200
  br_table v1 [block1(v2), block2(v3), block1(v3)] block3(v0, v0)
block1(v4: i64):
  v5 = i64.const 1
  v6 = i64.add v4 v5
  return v6
block2(v7: i64):
  v8 = i64.const 2
  v9 = i64.add v7 v8
  return v9
block3(v10: i64, v11: i64):
  v12 = i64.add v10 v11
  return v12
}
"#;

#[test]
fn brtable() {
    diff("brtable", BRTABLE, &[0, 1, 2, 3, 100, -1, i64::MIN], FUEL);
}

/// A self-branch that PERMUTES its own params — the reverse-pop edge protocol's regression test
/// (naive in-order local.set would compute fib wrong).
const SWAP: &str = r#"
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const 0
  v2 = i64.const 1
  br block1(v0, v1, v2)
block1(v3: i64, v4: i64, v5: i64):
  v6 = i64.const 0
  v7 = i64.gt_s v3 v6
  br_if v7 block2(v3, v4, v5) block3(v4)
block2(v8: i64, v9: i64, v10: i64):
  v11 = i64.const -1
  v12 = i64.add v8 v11
  v13 = i64.add v9 v10
  br block1(v12, v10, v13)
block3(v14: i64):
  return v14
}
"#;

#[test]
fn swap_params() {
    diff("swap_params", SWAP, &[0, 1, 2, 3, 10, 50, 90], FUEL);
}

/// Guest `unreachable` → the same trap kind on both engines.
const UNREACH: &str = r#"
func (i64) -> (i64) {
block0(v0: i64):
  unreachable
}
"#;

#[test]
fn unreachable_traps() {
    diff("unreachable", UNREACH, &[0], FUEL);
}

/// An infinite loop must exhaust fuel on both tiers. The wasm tier debits per dispatcher
/// iteration (coarser than the interpreter's per-op debit — a §5 bound, not an observable), so
/// both trap `OutOfFuel`, at different instruction counts.
const SPIN: &str = r#"
func (i64) -> (i64) {
block0(v0: i64):
  br block1(v0)
block1(v1: i64):
  v2 = i64.const 1
  v3 = i64.add v1 v2
  br block1(v3)
}
"#;

#[test]
fn out_of_fuel() {
    diff("out_of_fuel", SPIN, &[0], 100_000);
}

/// Multi-result returns end-to-end (wasm multi-value + reverse result pops at the call site).
const MULTI: &str = r#"
func (i64) -> (i64) {
block0(v0: i64):
  v1, v2 = call 1 (v0)
  v3 = i64.sub v1 v2
  return v3
}
func (i64) -> (i64, i64) {
block0(v0: i64):
  v1 = i64.const 3
  v2 = i64.mul v0 v1
  v3 = i64.const 10
  v4 = i64.add v0 v3
  return v2, v4
}
"#;

#[test]
fn multi_value() {
    diff("multi_value", MULTI, ARGS, FUEL);
}

/// i32-typed params/results across the call boundary (wrap/extend at the edges, i32 arithmetic).
const I32FN: &str = r#"
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i32.wrap_i64 v0
  v2 = call 1 (v1)
  v3 = i64.extend_i32_s v2
  return v3
}
func (i32) -> (i32) {
block0(v0: i32):
  v1 = i32.const 1103515245
  v2 = i32.mul v0 v1
  v3 = i32.const 12345
  v4 = i32.add v2 v3
  v5 = i32.const 16
  v6 = i32.shr_u v4 v5
  v7 = i32.xor v4 v6
  return v7
}
"#;

#[test]
fn i32_fn() {
    diff("i32_fn", I32FN, ARGS, FUEL);
}

/// Everything the tier must REFUSE (fail-closed): the op families slice 3+ will route to the
/// interpreter. A module containing any of them is Unsupported as a whole in slice 1.
#[test]
fn fail_closed() {
    for (name, src) in [
        (
            "float",
            "func (i64) -> (i64) {\nblock0(v0: i64):\n  v1 = f64.convert_i64_s v0\n  v2 = i64.trunc_sat_f64_s v1\n  return v2\n}\n",
        ),
        (
            "fiber",
            "func () -> (i64) {\nblock0():\n  v0 = ref.func 1\n  v1 = i64.const 0\n  v2 = cont.new v0 v1\n  v3 = i64.const 7\n  v4, v5 = cont.resume v2 v3\n  return v5\n}\nfunc (i64, i64) -> (i64) {\nblock0(vsp: i64, varg: i64):\n  return varg\n}\n",
        ),
        (
            "threads",
            "memory 16\nfunc () -> (i64) {\nblock0():\n  v0 = i64.const 0\n  v1 = thread.spawn 1 v0 v0\n  v2 = thread.join v1\n  return v2\n}\nfunc (i64, i64) -> (i64) {\nblock0(vsp: i64, v0: i64):\n  v1 = i64.const 0\n  return v1\n}\n",
        ),
        (
            "tailcall",
            "func (i64) -> (i64) {\nblock0(v0: i64):\n  return_call 1(v0)\n}\nfunc (i64) -> (i64) {\nblock0(v0: i64):\n  return v0\n}\n",
        ),
    ] {
        let m = svm_text::parse_module(src).unwrap_or_else(|e| panic!("{name}: {e}"));
        svm_verify::verify_module(&m).unwrap_or_else(|e| panic!("{name}: verify: {e:?}"));
        assert!(
            compile_module(&m).is_err(),
            "{name}: must be Unsupported on the wasm tier (slice 1)"
        );
    }
}
