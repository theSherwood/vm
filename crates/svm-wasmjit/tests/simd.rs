//! §17 SIMD (v128) differential gate: every kernel runs on the **bytecode engine** (the oracle) and
//! on the **emitted wasm** under `wasmi`, comparing the i64 result across an arg sweep. The emitter
//! is escape-TCB-adjacent (its output confines guest addresses, and v128 adds the one 16-byte
//! widened access), so this differential is the correctness contract for every SIMD opcode — a
//! wrong `0xFD` subopcode either fails wasmi validation or diverges from the oracle here.
//!
//! Each kernel is `(i64) -> (i64)`: it builds v128s from the scalar arg (splat / const), runs the
//! op family, and reduces back to an i64 (lane extract, or reinterpret of a float lane's bits) so
//! the comparison is exact — NaN payloads and rounding included.

use svm_interp::{bytecode, Value};
use svm_wasmjit::compile_module;
use wasmi::{Caller, Config, Engine, Linker, Memory, MemoryType, Module as WModule, Store, Val};

const WIN_BASE: u32 = 0x1_0000;
const ENV_PTR: u32 = 1024;

fn oracle(m: &svm_ir::Module, args: &[Value], fuel: u64) -> Vec<Value> {
    let mut fuel = fuel;
    match bytecode::compile_and_run(m, 0, args, &mut fuel) {
        Some(Ok(vals)) => vals,
        Some(Err(t)) => panic!("oracle trapped: {t:?}"),
        None => panic!("oracle: module unsupported by the bytecode engine"),
    }
}

fn wasm_run(m: &svm_ir::Module, wasm: &[u8], args: &[Value], fuel: u64) -> Vec<Value> {
    // wasmi gates the SIMD proposal behind the `simd` crate feature + this config toggle.
    let mut config = Config::default();
    config.wasm_simd(true);
    let engine = Engine::new(&config);
    let module = WModule::new(&engine, wasm).expect("emitted wasm must validate");
    let mut store: Store<i32> = Store::new(&engine, 0);
    let pages = 2 + m.memory.map_or(0, |mc| (mc.size() >> 16) as u32);
    let memory = Memory::new(&mut store, MemoryType::new(pages, None)).unwrap();
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
    linker
        .func_wrap::<_, ()>(
            "env",
            "call_interp",
            |_: Caller<'_, i32>, _: i32, _: i32| {
                unreachable!("no cross-tier call in a SIMD kernel");
            },
        )
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
    f.call(&mut store, &params, &mut results)
        .unwrap_or_else(|e| panic!("emitted SIMD kernel trapped: {e}"));
    results
        .iter()
        .map(|v| match v {
            Val::I32(x) => Value::I32(*x),
            Val::I64(x) => Value::I64(*x),
            _ => panic!("non-integer result"),
        })
        .collect()
}

fn diff(name: &str, src: &str, sweep: &[i64]) {
    let m = svm_text::parse_module(src).unwrap_or_else(|e| panic!("{name}: {e}"));
    svm_verify::verify_module(&m).unwrap_or_else(|e| panic!("{name}: verify: {e:?}"));
    let wasm = compile_module(&m).unwrap_or_else(|e| panic!("{name}: emit: {e}"));
    for &a in sweep {
        let args = vec![Value::I64(a)];
        assert_eq!(
            oracle(&m, &args, FUEL),
            wasm_run(&m, &wasm, &args, FUEL),
            "{name}: MISCOMPILE for arg {a}"
        );
    }
}

/// Like [`diff`] but the single i64 result is a **reinterpreted f64 lane's bits**: two NaN results
/// compare equal regardless of sign/payload. wasm leaves the sign and payload of a *generated* NaN
/// nondeterministic (§ "NaN propagation"), so the bytecode oracle (Rust `f64`) and wasmi legitimately
/// disagree on those bits for inf/NaN inputs — a divergence that is unobservable to a conforming
/// guest. Finite results still compare exactly (rounding included).
fn diff_f64bits(name: &str, src: &str, sweep: &[i64]) {
    let m = svm_text::parse_module(src).unwrap_or_else(|e| panic!("{name}: {e}"));
    svm_verify::verify_module(&m).unwrap_or_else(|e| panic!("{name}: verify: {e:?}"));
    let wasm = compile_module(&m).unwrap_or_else(|e| panic!("{name}: emit: {e}"));
    for &a in sweep {
        let args = vec![Value::I64(a)];
        let o = oracle(&m, &args, FUEL);
        let w = wasm_run(&m, &wasm, &args, FUEL);
        let (Value::I64(ob), Value::I64(wb)) = (o[0], w[0]) else {
            panic!("{name}: expected one i64 result")
        };
        let (of, wf) = (f64::from_bits(ob as u64), f64::from_bits(wb as u64));
        assert!(
            ob == wb || (of.is_nan() && wf.is_nan()),
            "{name}: MISCOMPILE for arg {a}: oracle {ob:#018x} != wasm {wb:#018x}"
        );
    }
}

const FUEL: u64 = 100_000_000;
const ARGS: &[i64] = &[
    0,
    1,
    2,
    7,
    255,
    256,
    -1,
    -7,
    1000,
    i64::MIN,
    i64::MAX,
    0x0102_0304_0506_0708,
];
// f64 bit patterns fed as the i64 arg, reinterpreted to f64 inside (±0/±1/±inf/NaN/subnormal/π/100).
const F64_BITS: &[i64] = &[
    0x0000_0000_0000_0000u64 as i64,
    0x8000_0000_0000_0000u64 as i64,
    0x3FF0_0000_0000_0000u64 as i64,
    0xBFF0_0000_0000_0000u64 as i64,
    0x7FF0_0000_0000_0000u64 as i64,
    0xFFF0_0000_0000_0000u64 as i64,
    0x7FF8_0000_0000_0000u64 as i64,
    0x0000_0000_0000_0001u64 as i64,
    0x4009_21FB_5444_2D18u64 as i64,
    0x4059_0000_0000_0000u64 as i64,
];

/// Integer lane arithmetic across shapes: splat/replace/extract, i32x4 add/sub/mul/min_s/max_u,
/// i64x2 add/sub/mul, i16x8 add/mul, i8x16 add/sub.
const INT_LANES: &str = r#"
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i32.wrap_i64 v0
  v2 = i32x4.splat v1
  v3 = i32.const 3
  v4 = i32x4.splat v3
  v5 = i32x4.add v2 v4
  v6 = i32x4.sub v5 v4
  v7 = i32x4.mul v6 v4
  v8 = i32x4.min_s v7 v2
  v9 = i32x4.max_u v8 v4
  v10 = i32.const 1
  v11 = i32x4.replace_lane 2 v9 v10
  v12 = i32x4.extract_lane 2 v11
  v13 = i64x2.splat v0
  v14 = i64x2.add v13 v13
  v15 = i64x2.mul v14 v13
  v16 = i64x2.extract_lane 0 v15
  v17 = i64.extend_i32_s v12
  v18 = i64.add v16 v17
  return v18
}
"#;

#[test]
fn int_lanes() {
    diff("int_lanes", INT_LANES, ARGS);
}

/// Lane compares → the i32-producing reductions (bitmask / all_true / any_true).
const INT_CMP_REDUCE: &str = r#"
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i32.wrap_i64 v0
  v2 = i32x4.splat v1
  v3 = i32.const 0
  v4 = i32x4.splat v3
  v5 = i32x4.gt_s v2 v4
  v6 = i32x4.eq v2 v4
  v7 = i8x16.bitmask v5
  v8 = i32x4.all_true v6
  v9 = v128.any_true v5
  v10 = i32.add v7 v8
  v11 = i32.add v10 v9
  v12 = i64.extend_i32_u v11
  return v12
}
"#;

#[test]
fn int_cmp_reduce() {
    diff("int_cmp_reduce", INT_CMP_REDUCE, ARGS);
}

/// Lane shifts (shl/shr_s/shr_u) across i8x16/i16x8/i32x4/i64x2.
const SHIFTS: &str = r#"
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i32.wrap_i64 v0
  v2 = i64x2.splat v0
  v3 = i32.const 3
  v4 = i64x2.shl v2 v3
  v5 = i64x2.shr_s v4 v3
  v6 = i64x2.shr_u v5 v3
  v7 = i32x4.splat v1
  v8 = i32x4.shl v7 v3
  v9 = i32x4.shr_s v8 v3
  v10 = i32x4.extract_lane 0 v9
  v11 = i64x2.extract_lane 0 v6
  v12 = i64.extend_i32_s v10
  v13 = i64.add v11 v12
  return v13
}
"#;

#[test]
fn shifts() {
    diff("shifts", SHIFTS, ARGS);
}

/// Unary integer (abs/neg), saturating add/sub, avgr_u, popcnt — the i8x16/i16x8 long tail.
const SAT_UNARY: &str = r#"
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i32.wrap_i64 v0
  v2 = i8x16.splat v1
  v3 = i8x16.neg v2
  v4 = i8x16.abs v3
  v5 = i8x16.add_sat_s v4 v2
  v6 = i8x16.sub_sat_u v5 v2
  v7 = i8x16.avgr_u v6 v2
  v8 = i8x16.popcnt v7
  v9 = i16x8.splat v1
  v10 = i16x8.add_sat_u v9 v9
  v11 = i8x16.extract_lane_u 0 v8
  v12 = i16x8.extract_lane_s 0 v10
  v13 = i32.add v11 v12
  v14 = i64.extend_i32_s v13
  return v14
}
"#;

#[test]
fn sat_unary() {
    diff("sat_unary", SAT_UNARY, ARGS);
}

/// Whole-vector bitwise: and/or/xor/andnot/not/bitselect.
const BITWISE: &str = r#"
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64x2.splat v0
  v2 = i64.const -1
  v3 = i64x2.splat v2
  v4 = v128.and v1 v3
  v5 = v128.or v4 v1
  v6 = v128.xor v5 v3
  v7 = v128.andnot v6 v1
  v8 = v128.not v7
  v9 = v128.bitselect v1 v8 v3
  v10 = i64x2.extract_lane 0 v9
  v11 = i64x2.extract_lane 1 v8
  v12 = i64.add v10 v11
  return v12
}
"#;

#[test]
fn bitwise() {
    diff("bitwise", BITWISE, ARGS);
}

/// f64x2 / f32x4 arithmetic + IEEE min/max + pmin/pmax, exact via reinterpret of lane bits.
const FLOAT_ARITH: &str = r#"
func (i64) -> (i64) {
block0(v0: i64):
  vf = f64.reinterpret_i64 v0
  v1 = f64x2.splat vf
  v2 = f64x2.add v1 v1
  v3 = f64x2.mul v2 v1
  v4 = f64x2.div v3 v1
  v5 = f64x2.sub v4 v1
  v6 = f64x2.min v5 v1
  v7 = f64x2.max v6 v1
  v8 = f64x2.pmin v7 v1
  v9 = f64x2.pmax v8 v1
  v10 = f64x2.extract_lane 0 v9
  v11 = i64.reinterpret_f64 v10
  return v11
}
"#;

#[test]
fn float_arith() {
    diff_f64bits("float_arith", FLOAT_ARITH, F64_BITS);
}

/// f64x2 unary: abs/neg/sqrt/ceil/floor/trunc/nearest, exact.
const FLOAT_UNARY: &str = r#"
func (i64) -> (i64) {
block0(v0: i64):
  vf = f64.reinterpret_i64 v0
  v1 = f64x2.splat vf
  v2 = f64x2.abs v1
  v3 = f64x2.neg v2
  v4 = f64x2.sqrt v3
  v5 = f64x2.ceil v4
  v6 = f64x2.floor v5
  v7 = f64x2.trunc v6
  v8 = f64x2.nearest v7
  v9 = f64x2.extract_lane 0 v8
  v10 = i64.reinterpret_f64 v9
  return v10
}
"#;

#[test]
fn float_unary() {
    diff_f64bits("float_unary", FLOAT_UNARY, F64_BITS);
}

/// Float lane compares reduced to deterministic bitmasks (a compare of NaN yields an all-zeros /
/// all-ones mask on both engines — no NaN *bits* are returned, so this stays exact even for
/// inf/NaN inputs). Covers eq/ne/lt/gt/le/ge across f64x2/f32x4.
const FLOAT_CMP: &str = r#"
func (i64) -> (i64) {
block0(v0: i64):
  vf = f64.reinterpret_i64 v0
  v1 = f64x2.splat vf
  v2 = f64x2.add v1 v1
  v3 = f64x2.lt v1 v2
  v4 = f64x2.eq v1 v1
  v5 = f64x2.ne v1 v1
  v6 = f64x2.ge v2 v1
  v7 = f64x2.gt v2 v1
  v8 = f64x2.le v1 v2
  v9 = i8x16.bitmask v3
  v10 = i8x16.bitmask v4
  v11 = i8x16.bitmask v5
  v12 = i8x16.bitmask v6
  v13 = i8x16.bitmask v7
  v14 = i8x16.bitmask v8
  v15 = i32.add v9 v10
  v16 = i32.add v15 v11
  v17 = i32.add v16 v12
  v18 = i32.add v17 v13
  v19 = i32.add v18 v14
  v20 = i64.extend_i32_u v19
  return v20
}
"#;

#[test]
fn float_cmp() {
    diff("float_cmp", FLOAT_CMP, F64_BITS);
}

/// Lane int↔float / float↔float conversions (trunc_sat is non-trapping — NaN→0, clamp).
const CONVERT: &str = r#"
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i32.wrap_i64 v0
  v2 = i32x4.splat v1
  v3 = f32x4.convert_i32x4_s v2
  v4 = f32x4.convert_i32x4_u v2
  v5 = f32x4.add v3 v4
  v6 = i32x4.trunc_sat_f32x4_s v5
  v7 = i32x4.trunc_sat_f32x4_u v5
  v8 = i32x4.add v6 v7
  vf = f64.reinterpret_i64 v0
  v9 = f64x2.splat vf
  v10 = f32x4.demote_f64x2_zero v9
  v11 = f64x2.promote_low_f32x4 v10
  v12 = f64x2.convert_low_i32x4_s v8
  v13 = i32x4.trunc_sat_f64x2_s_zero v12
  v14 = i32x4.add v8 v13
  v15 = i32x4.extract_lane 0 v14
  v16 = i64.extend_i32_s v15
  return v16
}
"#;

#[test]
fn convert() {
    diff("convert", CONVERT, ARGS);
}

/// i8x16.shuffle (constant byte permute) + i8x16.swizzle (dynamic byte select).
const SHUFFLE: &str = r#"
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64x2.splat v0
  v2 = i8x16.shuffle 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1 0 v1 v1
  v3 = i8x16.swizzle v1 v2
  v4 = i64x2.extract_lane 0 v3
  v5 = i64x2.extract_lane 1 v2
  v6 = i64.add v4 v5
  return v6
}
"#;

#[test]
fn shuffle_swizzle() {
    diff("shuffle_swizzle", SHUFFLE, ARGS);
}

/// v128.const materialization + a lane op + width probe (`simd.width_bytes` == 16).
const CONST_VEC: &str = r#"
func (i64) -> (i64) {
block0(v0: i64):
  v1 = v128.const 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
  v2 = i64x2.splat v0
  v3 = i64x2.add v1 v2
  v4 = i64x2.extract_lane 0 v3
  v5 = simd.width_bytes
  v6 = i64.extend_i32_u v5
  v7 = i64.add v4 v6
  return v7
}
"#;

#[test]
fn const_vec() {
    diff("const_vec", CONST_VEC, ARGS);
}

/// v128.load / v128.store through the confined window (the one 16-byte widened access — §17/D58).
const MEM: &str = r#"
memory 16
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64x2.splat v0
  v2 = i64.const 16
  v128.store v2 v1
  v3 = v128.load v2
  v4 = i64x2.add v3 v1
  v5 = i64x2.extract_lane 1 v4
  return v5
}
"#;

#[test]
fn mem() {
    diff("mem", MEM, ARGS);
}

/// The SIMD ops this slice defers (fail-closed → the module stays on the interpreter): the
/// widening / reduction family and relaxed SIMD. A module containing any is Unsupported as a whole.
#[test]
fn fail_closed_simd() {
    for (name, body) in [
        ("dot", "v1 = i64x2.splat v0\n  v2 = i32x4.dot_i16x8_s v1 v1\n  v3 = i64x2.extract_lane 0 v2\n  return v3"),
        ("widen", "v1 = i64x2.splat v0\n  v2 = i16x8.extend_low_s v1\n  v3 = i64x2.extract_lane 0 v2\n  return v3"),
        ("narrow", "v1 = i64x2.splat v0\n  v2 = i8x16.narrow_s v1 v1\n  v3 = i64x2.extract_lane 0 v2\n  return v3"),
        ("extmul", "v1 = i64x2.splat v0\n  v2 = i16x8.extmul_low_i8x16_s v1 v1\n  v3 = i64x2.extract_lane 0 v2\n  return v3"),
        ("q15", "v1 = i64x2.splat v0\n  v2 = i16x8.q15mulr_sat_s v1 v1\n  v3 = i64x2.extract_lane 0 v2\n  return v3"),
        ("extadd", "v1 = i64x2.splat v0\n  v2 = i16x8.extadd_pairwise_i8x16_s v1\n  v3 = i64x2.extract_lane 0 v2\n  return v3"),
    ] {
        let src = format!("func (i64) -> (i64) {{\nblock0(v0: i64):\n  {body}\n}}\n");
        let m = svm_text::parse_module(&src).unwrap_or_else(|e| panic!("{name}: {e}"));
        svm_verify::verify_module(&m).unwrap_or_else(|e| panic!("{name}: verify: {e:?}"));
        assert!(
            compile_module(&m).is_err(),
            "{name}: deferred SIMD op must be Unsupported on the wasm tier"
        );
    }
}
