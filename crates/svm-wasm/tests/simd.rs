//! §17 SIMD (D58) capstone: real `v128` kernels compiled to wasm, transpiled to our IR,
//! verified, and run on **both** backends — proving the wasm→IR SIMD bridge plus native
//! Cranelift vector codegen produce a result that is (a) correct and (b) byte-identical
//! across interp and JIT. This is the "auto-vectorized kernel runs at real SIMD" milestone
//! from the §17 plan, with the wasm here standing in for a clang `-msimd128` emission.

use svm_interp::Value;

/// Transpile WAT → IR, verify, run `entry(args)` on interp + JIT, assert they agree, return
/// the interp result.
fn eval(wat: &str, entry: &str, args: &[Value]) -> Value {
    let wasm = wat::parse_str(wat).expect("assemble wat");
    let t = svm_wasm::transpile(&wasm).expect("transpile");
    svm_verify::verify_module(&t.module).expect("verify transpiled IR");
    let idx = t
        .exports
        .iter()
        .find(|(n, _)| n == entry)
        .unwrap_or_else(|| panic!("no export {entry}"))
        .1;
    let results = &t.module.funcs[idx as usize].results;
    let mut fuel = 100_000_000u64;
    let interp = svm_interp::run(&t.module, idx, args, &mut fuel).expect("interp run");
    let jit_args: Vec<i64> = args
        .iter()
        .map(|v| match v {
            Value::I32(x) => *x as i64,
            Value::I64(x) => *x,
            Value::F32(x) => x.to_bits() as i64,
            Value::F64(x) => x.to_bits() as i64,
            Value::V128(b) => i64::from_le_bytes(b[..8].try_into().unwrap()),
            Value::Ref(x) => *x as i64,
        })
        .collect();
    let jit = match svm_jit::compile_and_run(&t.module, idx, &jit_args).expect("jit") {
        svm_jit::JitOutcome::Returned(v) => v,
        other => panic!("jit did not return: {other:?}"),
    };
    for (i, rt) in results.iter().enumerate() {
        let ok = match (rt, interp[i]) {
            (svm_ir::ValType::I32, Value::I32(x)) => x as u32 as u64 == jit[i] as u32 as u64,
            (svm_ir::ValType::F32, Value::F32(x)) => {
                let j = f32::from_bits(jit[i] as u32);
                x.to_bits() == j.to_bits() || (x.is_nan() && j.is_nan())
            }
            (svm_ir::ValType::F64, Value::F64(x)) => {
                let j = f64::from_bits(jit[i] as u64);
                x.to_bits() == j.to_bits() || (x.is_nan() && j.is_nan())
            }
            _ => panic!("result type/value mismatch at {i}"),
        };
        assert!(
            ok,
            "interp != jit at result {i}: {:?} vs {:#x}",
            interp[i], jit[i]
        );
    }
    interp[0]
}

fn f32(v: Value) -> f32 {
    match v {
        Value::F32(x) => x,
        other => panic!("expected f32, got {other:?}"),
    }
}

/// A 4-wide f32 dot product: `v128.load` two vectors, `f32x4.mul`, horizontal-sum via
/// `extract_lane`. x = [1,2,3,4], y = [4,3,2,1] ⇒ 1·4+2·3+3·2+4·1 = 20.
#[test]
fn dot4_f32() {
    let wat = r#"
    (module
      (memory 1)
      (data (i32.const 0)  "\00\00\80\3f\00\00\00\40\00\00\40\40\00\00\80\40")
      (data (i32.const 16) "\00\00\80\40\00\00\40\40\00\00\00\40\00\00\80\3f")
      (func (export "dot") (result f32)
        (local $p v128)
        (local.set $p
          (f32x4.mul (v128.load (i32.const 0)) (v128.load (i32.const 16))))
        (f32.add
          (f32.add (f32x4.extract_lane 0 (local.get $p)) (f32x4.extract_lane 1 (local.get $p)))
          (f32.add (f32x4.extract_lane 2 (local.get $p)) (f32x4.extract_lane 3 (local.get $p))))))
    "#;
    assert_eq!(f32(eval(wat, "dot", &[])), 20.0);
}

/// f32x4 saxpy `a*x + y` with `a` splatted, the result `v128.store`d, then the requested lane
/// loaded back as a scalar — exercising splat, load, mul, add, store, and a scalar reload.
/// a=10, x=[1,2,3,4], y=[4,3,2,1] ⇒ [14,23,32,41].
#[test]
fn saxpy_f32x4() {
    let wat = r#"
    (module
      (memory 1)
      (data (i32.const 0)  "\00\00\80\3f\00\00\00\40\00\00\40\40\00\00\80\40")
      (data (i32.const 16) "\00\00\80\40\00\00\40\40\00\00\00\40\00\00\80\3f")
      (func (export "saxpy") (param $a f32) (param $lane i32) (result f32)
        (v128.store (i32.const 32)
          (f32x4.add
            (f32x4.mul (f32x4.splat (local.get $a)) (v128.load (i32.const 0)))
            (v128.load (i32.const 16))))
        (f32.load (i32.add (i32.const 32) (i32.mul (local.get $lane) (i32.const 4))))))
    "#;
    let expect = [14.0f32, 23.0, 32.0, 41.0];
    for (lane, &want) in expect.iter().enumerate() {
        let got = f32(eval(
            wat,
            "saxpy",
            &[Value::F32(10.0), Value::I32(lane as i32)],
        ));
        assert_eq!(got, want, "saxpy lane {lane}");
    }
}

/// An i32x4 vector add-reduce: splat two scalars, `i32x4.add`, sum the lanes. Proves the
/// integer-lane path through the wasm bridge + JIT. (a+b)*4 since both are splatted.
#[test]
fn i32x4_add_reduce() {
    let wat = r#"
    (module
      (func (export "f") (param $a i32) (param $b i32) (result i32)
        (local $v v128)
        (local.set $v (i32x4.add (i32x4.splat (local.get $a)) (i32x4.splat (local.get $b))))
        (i32.add
          (i32.add (i32x4.extract_lane 0 (local.get $v)) (i32x4.extract_lane 1 (local.get $v)))
          (i32.add (i32x4.extract_lane 2 (local.get $v)) (i32x4.extract_lane 3 (local.get $v))))))
    "#;
    for (a, b) in [(1, 2), (10, -3), (100, 100)] {
        let got = match eval(wat, "f", &[Value::I32(a), Value::I32(b)]) {
            Value::I32(x) => x,
            _ => unreachable!(),
        };
        assert_eq!(got, (a + b) * 4);
    }
}

/// A `v128.const` + `i8x16.shuffle` reverse, observed via a byte lane — pins the constant
/// vector immediate and the shuffle immediate through the wasm bridge.
#[test]
fn v128_const_shuffle() {
    let wat = r#"
    (module
      (func (export "f") (result i32)
        (local $v v128)
        (local.set $v
          (i8x16.shuffle 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1 0
            (v128.const i8x16 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)
            (v128.const i8x16 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0)))
        (i8x16.extract_lane_u 0 (local.get $v))))
    "#;
    // After reversing [0..15], lane 0 = original byte 15 = 15.
    assert_eq!(eval(wat, "f", &[]), Value::I32(15));
}

/// **The clang capstone.** A real `clang --target=wasm32 -msimd128 -O2` saxpy kernel
/// (`*out = a*(*x) + (*y)` over `float __attribute__((vector_size(16)))`), linked with
/// `wasm-ld` and embedded as a fixture so the test stays hermetic (no clang at test time).
/// Clang auto-emits exactly `f32x4.splat`, `v128.load`, `f32x4.mul`, `f32x4.add`,
/// `v128.store`; this asserts the wasm→IR bridge maps that real compiler output to verified
/// SIMD IR carrying those ops — the "auto-vectorized kernel transpiles" milestone (§17/D58).
#[test]
fn clang_saxpy_transpiles_to_verified_simd_ir() {
    use svm_ir::{Inst, VFloatBinOp, VShape};

    let wasm = include_bytes!("fixtures/saxpy_clang.wasm");
    let t = svm_wasm::transpile(wasm).expect("transpile clang -msimd128 output");
    svm_verify::verify_module(&t.module).expect("verify transpiled clang SIMD IR");

    // Tally the SIMD ops the kernel lowered to, across all function bodies.
    let (mut splat, mut load, mut store, mut mul, mut add) = (0, 0, 0, 0, 0);
    for f in &t.module.funcs {
        for b in &f.blocks {
            for inst in &b.insts {
                match inst {
                    Inst::Splat {
                        shape: VShape::F32x4,
                        ..
                    } => splat += 1,
                    Inst::V128Load { .. } => load += 1,
                    Inst::V128Store { .. } => store += 1,
                    Inst::VFloatBin {
                        shape: VShape::F32x4,
                        op: VFloatBinOp::Mul,
                        ..
                    } => mul += 1,
                    Inst::VFloatBin {
                        shape: VShape::F32x4,
                        op: VFloatBinOp::Add,
                        ..
                    } => add += 1,
                    _ => {}
                }
            }
        }
    }
    assert!(splat >= 1, "expected an f32x4.splat from the scalar `a`");
    assert!(load >= 2, "expected two v128.loads (x and y), got {load}");
    assert_eq!(store, 1, "expected one v128.store (out)");
    assert_eq!(mul, 1, "expected one f32x4.mul (a*x)");
    assert_eq!(add, 1, "expected one f32x4.add (+y)");
}

/// Integer lane comparisons through the wasm bridge: `iNxM.<cmp>` produces a per-lane all-ones/
/// all-zeros mask, observed via `extract_lane`. Pins the signed/unsigned distinction (`lt_s` vs
/// `lt_u`) on the same bytes — exactly where real `-msimd128` mask generation lives.
#[test]
fn i32x4_lane_compare_masks() {
    let cmp = |op: &str| {
        format!(
            "(module (func (export \"f\") (param $a i32) (param $b i32) (result i32)
               (i32x4.extract_lane 0
                 ({op} (i32x4.splat (local.get $a)) (i32x4.splat (local.get $b))))))"
        )
    };
    // a = -1 (0xFFFFFFFF), b = 1: signed -1 < 1 (true → -1); unsigned 0xFFFFFFFF < 1 (false → 0).
    let args = [Value::I32(-1), Value::I32(1)];
    assert_eq!(eval(&cmp("i32x4.lt_s"), "f", &args), Value::I32(-1));
    assert_eq!(eval(&cmp("i32x4.lt_u"), "f", &args), Value::I32(0));
    assert_eq!(
        eval(&cmp("i32x4.eq"), "f", &[Value::I32(7), Value::I32(7)]),
        Value::I32(-1)
    );
    assert_eq!(
        eval(&cmp("i32x4.ne"), "f", &[Value::I32(7), Value::I32(7)]),
        Value::I32(0)
    );
    assert_eq!(eval(&cmp("i8x16.eq"), "f", &args), Value::I32(0)); // shape check: distinct opcode
}

/// A real SIMD idiom: lane-wise signed **max** as `bitselect(a>b, a, b)` — the compare feeds its
/// mask straight into `v128.bitselect`. Proves the mask interoperates with downstream vector ops,
/// byte-identical across interp and JIT.
#[test]
fn i32x4_max_via_compare_and_bitselect() {
    let wat = r#"
    (module
      (func (export "maxlane") (param $a i32) (param $b i32) (result i32)
        (i32x4.extract_lane 0
          (v128.bitselect
            (i32x4.splat (local.get $a))
            (i32x4.splat (local.get $b))
            (i32x4.gt_s (i32x4.splat (local.get $a)) (i32x4.splat (local.get $b)))))))
    "#;
    for (a, b) in [(3, 7), (9, 2), (-5, -1), (i32::MIN, i32::MAX), (4, 4)] {
        let got = match eval(wat, "maxlane", &[Value::I32(a), Value::I32(b)]) {
            Value::I32(x) => x,
            _ => unreachable!(),
        };
        assert_eq!(got, a.max(b), "max({a}, {b})");
    }
}

/// A real clamp kernel through the wasm bridge: `clamp(x, lo, hi) = max(lo, min(hi, x))` lane-wise,
/// using `i32x4.min_s`/`max_s`. Proves the new min/max ops compose and stay byte-identical interp vs JIT.
#[test]
fn i32x4_clamp_via_min_max() {
    let wat = r#"
    (module
      (func (export "clamp") (param $x i32) (param $lo i32) (param $hi i32) (result i32)
        (i32x4.extract_lane 0
          (i32x4.max_s (i32x4.splat (local.get $lo))
            (i32x4.min_s (i32x4.splat (local.get $hi)) (i32x4.splat (local.get $x)))))))
    "#;
    for (x, lo, hi) in [
        (5, 0, 10),
        (-3, 0, 10),
        (42, 0, 10),
        (7, 7, 7),
        (-100, -50, -10),
    ] {
        let got = match eval(
            wat,
            "clamp",
            &[Value::I32(x), Value::I32(lo), Value::I32(hi)],
        ) {
            Value::I32(v) => v,
            _ => unreachable!(),
        };
        assert_eq!(got, x.clamp(lo, hi), "clamp({x}, {lo}, {hi})");
    }
}

/// Float lane comparisons through the wasm bridge, incl. the NaN behaviour (`eq`/`lt`/… ordered →
/// false, `ne` unordered → true). The mask is read out with `i32x4.extract_lane`.
#[test]
fn f32x4_lane_compare_masks() {
    let cmp = |op: &str| {
        format!(
            "(module (func (export \"f\") (param $a f32) (param $b f32) (result i32)
               (i32x4.extract_lane 0
                 ({op} (f32x4.splat (local.get $a)) (f32x4.splat (local.get $b))))))"
        )
    };
    assert_eq!(
        eval(&cmp("f32x4.lt"), "f", &[Value::F32(1.0), Value::F32(2.0)]),
        Value::I32(-1)
    );
    assert_eq!(
        eval(&cmp("f32x4.eq"), "f", &[Value::F32(2.0), Value::F32(2.0)]),
        Value::I32(-1)
    );
    assert_eq!(
        eval(&cmp("f32x4.ge"), "f", &[Value::F32(2.0), Value::F32(5.0)]),
        Value::I32(0)
    );
    // NaN: ordered compares are false; `ne` is true.
    let nan = [Value::F32(f32::NAN), Value::F32(1.0)];
    assert_eq!(eval(&cmp("f32x4.eq"), "f", &nan), Value::I32(0));
    assert_eq!(eval(&cmp("f32x4.lt"), "f", &nan), Value::I32(0));
    assert_eq!(eval(&cmp("f32x4.ne"), "f", &nan), Value::I32(-1));
}

/// Integer lane shifts through the wasm bridge: one scalar amount (taken mod the lane width) shifts
/// every lane. Covers `shl`/`shr_s`/`shr_u` and an amount ≥ the lane width.
#[test]
fn i32x4_lane_shifts() {
    let sh = |op: &str| {
        format!(
            "(module (func (export \"f\") (param $x i32) (param $amt i32) (result i32)
               (i32x4.extract_lane 0 ({op} (i32x4.splat (local.get $x)) (local.get $amt)))))"
        )
    };
    let i32 = |w: &str, x: i32, amt: i32| match eval(&sh(w), "f", &[Value::I32(x), Value::I32(amt)])
    {
        Value::I32(v) => v,
        _ => unreachable!(),
    };
    for amt in [0, 1, 5, 31, 34] {
        let m = (amt & 31) as u32;
        assert_eq!(
            i32("i32x4.shl", 0x0011_2233, amt),
            0x0011_2233i32.wrapping_shl(m),
            "shl {amt}"
        );
        assert_eq!(
            i32("i32x4.shr_u", -1, amt),
            (0xFFFF_FFFFu32 >> m) as i32,
            "shr_u {amt}"
        );
        assert_eq!(i32("i32x4.shr_s", -16, amt), -16i32 >> m, "shr_s {amt}");
    }
}

/// Integer lane abs/neg through the wasm bridge (two's-complement: `abs(INT_MIN) == INT_MIN`).
#[test]
fn i32x4_abs_neg() {
    let un = |op: &str| {
        format!(
            "(module (func (export \"f\") (param $x i32) (result i32)
               (i32x4.extract_lane 0 ({op} (i32x4.splat (local.get $x))))))"
        )
    };
    let f = |w: &str, x: i32| match eval(&un(w), "f", &[Value::I32(x)]) {
        Value::I32(v) => v,
        _ => unreachable!(),
    };
    for x in [7, -7, 0, i32::MIN, i32::MAX] {
        assert_eq!(f("i32x4.abs", x), x.wrapping_abs(), "abs {x}");
        assert_eq!(f("i32x4.neg", x), x.wrapping_neg(), "neg {x}");
    }
}
