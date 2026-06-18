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

fn f64(v: Value) -> f64 {
    match v {
        Value::F64(x) => x,
        other => panic!("expected f64, got {other:?}"),
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

/// f64↔i32 lane-count-changing conversions through the wasm bridge: `f64x2.convert_low_i32x4_s`
/// (low 2 i32 → f64x2) then `i32x4.trunc_sat_f64x2_s_zero` back, with a `+0.5` in between so the
/// truncation is observable. The result must round-trip the integer (trunc of `x + 0.5` is `x` for
/// the small values here). `eval` enforces interp == JIT through the i64x2-intermediate lowering.
#[test]
fn f64x2_convert_low_and_trunc_sat_zero() {
    let wat = "(module (func (export \"f\") (param $x i32) (result i32)
                 (i32x4.extract_lane 0
                   (i32x4.trunc_sat_f64x2_s_zero
                     (f64x2.add
                       (f64x2.convert_low_i32x4_s (i32x4.splat (local.get $x)))
                       (f64x2.splat (f64.const 0.5)))))))";
    for x in [0, 1, -1, 7, -7, 1000, -1000] {
        let got = match eval(wat, "f", &[Value::I32(x)]) {
            Value::I32(v) => v,
            _ => unreachable!(),
        };
        // trunc(x + 0.5): toward zero, so x for x>=0, and x for x<0 (e.g. -7+0.5=-6.5 → -6).
        assert_eq!(got, (x as f64 + 0.5).trunc() as i32, "convert/trunc {x}");
    }
}

/// Pseudo-min/max through the wasm bridge. `pmin`/`pmax` are a one-sided compare-and-select
/// (`pmin(a,b)=b<a?b:a`, `pmax(a,b)=a<b?b:a`), so a NaN operand and signed zeros propagate by the
/// `<` rule rather than IEEE min/max canonicalization. `eval` enforces interp == JIT lane-for-lane.
#[test]
fn f32x4_pmin_pmax() {
    let pm = |op: &str| {
        format!(
            "(module (func (export \"f\") (param $a f32) (param $b f32) (result f32)
               (f32x4.extract_lane 0 ({op} (f32x4.splat (local.get $a)) (f32x4.splat (local.get $b))))))"
        )
    };
    let run = |op: &str, a: f32, b: f32| f32(eval(&pm(op), "f", &[Value::F32(a), Value::F32(b)]));
    for (a, b) in [(1.0f32, 2.0), (2.0, 1.0), (-0.0, 0.0), (3.5, -3.5)] {
        let pmin = if b < a { b } else { a };
        let pmax = if a < b { b } else { a };
        assert_eq!(
            run("f32x4.pmin", a, b).to_bits(),
            pmin.to_bits(),
            "pmin {a} {b}"
        );
        assert_eq!(
            run("f32x4.pmax", a, b).to_bits(),
            pmax.to_bits(),
            "pmax {a} {b}"
        );
    }
    // NaN second operand: every `<` is false, so both return the first operand `a`.
    assert!(run("f32x4.pmin", 1.0, f32::NAN).to_bits() == 1.0f32.to_bits());
    assert!(run("f32x4.pmax", 1.0, f32::NAN).to_bits() == 1.0f32.to_bits());
}

/// **Relaxed SIMD** through the wasm bridge — each op runs one spec-allowed deterministic behavior,
/// interp == JIT (enforced by `eval`). `relaxed_madd`/`nmadd` are a genuine fused FMA (one rounding,
/// matching Rust's `mul_add`); the rest alias to the deterministic op SVM already lowers. This is the
/// shape a real `clang -mrelaxed-simd` kernel emits.
#[test]
fn relaxed_simd_madd_and_friends() {
    // relaxed_madd(a,b,c) = a*b + c (fused). Splat three scalars, read lane 0.
    let madd = |op: &str, a: f32, b: f32, c: f32| {
        let wat = format!(
            "(module (func (export \"f\") (param $a f32) (param $b f32) (param $c f32) (result f32)
               (f32x4.extract_lane 0 ({op} (f32x4.splat (local.get $a))
                  (f32x4.splat (local.get $b)) (f32x4.splat (local.get $c))))))"
        );
        f32(eval(
            &wat,
            "f",
            &[Value::F32(a), Value::F32(b), Value::F32(c)],
        ))
    };
    for (a, b, c) in [(2.0f32, 3.0, 4.0), (1e20, 1e20, -1e30), (0.1, 0.2, 0.3)] {
        assert_eq!(
            madd("f32x4.relaxed_madd", a, b, c).to_bits(),
            a.mul_add(b, c).to_bits(),
            "relaxed_madd {a} {b} {c}"
        );
        assert_eq!(
            madd("f32x4.relaxed_nmadd", a, b, c).to_bits(),
            (-a).mul_add(b, c).to_bits(),
            "relaxed_nmadd {a} {b} {c}"
        );
    }

    // relaxed_min/max alias to the deterministic wasm min/max.
    let minmax = |op: &str, a: f32, b: f32| {
        let wat = format!(
            "(module (func (export \"f\") (param $a f32) (param $b f32) (result f32)
               (f32x4.extract_lane 0 ({op} (f32x4.splat (local.get $a)) (f32x4.splat (local.get $b))))))"
        );
        f32(eval(&wat, "f", &[Value::F32(a), Value::F32(b)]))
    };
    assert_eq!(minmax("f32x4.relaxed_min", 2.0, 5.0), 2.0);
    assert_eq!(minmax("f32x4.relaxed_max", 2.0, 5.0), 5.0);

    // relaxed_trunc aliases to trunc_sat (saturating, NaN→0).
    let trunc = |x: f32| {
        let wat = "(module (func (export \"f\") (param $x f32) (result i32)
               (i32x4.extract_lane 0 (i32x4.relaxed_trunc_f32x4_s (f32x4.splat (local.get $x))))))";
        match eval(wat, "f", &[Value::F32(x)]) {
            Value::I32(v) => v,
            _ => unreachable!(),
        }
    };
    assert_eq!(trunc(3.9), 3);
    assert_eq!(trunc(-3.9), -3);
    assert_eq!(trunc(f32::NAN), 0, "trunc_sat maps NaN→0");

    // relaxed_laneselect aliases to bitselect (mask all-1 lane ⇒ take a).
    let wat = "(module (func (export \"f\") (result i32)
        (i32x4.extract_lane 0 (i32x4.relaxed_laneselect
          (i32x4.splat (i32.const 0xAAAA)) (i32x4.splat (i32.const 0x5555))
          (i32x4.splat (i32.const 0xFFFFFFFF))))))";
    assert_eq!(
        eval(wat, "f", &[]),
        Value::I32(0xAAAA),
        "all-1 mask selects a"
    );
}

/// The two relaxed **i8×i7 dot** ops (the deterministic signed-i8 lowering): the i16 dot and the
/// i32 dot-with-accumulate. Splatted bytes a=3, b=4 → each i16 lane = 3·4+3·4 = 24; the i32 add
/// variant widen-pairwise-adds two i16 lanes (24+24) and adds the accumulator. `eval` pins interp==JIT.
#[test]
fn relaxed_simd_i8_dot() {
    // i16x8.relaxed_dot_i8x16_i7x16_s: lane = a·b + a·b = 2·(a·b).
    let dot = "(module (func (export \"f\") (param $a i32) (param $b i32) (result i32)
        (i16x8.extract_lane_s 0 (i16x8.relaxed_dot_i8x16_i7x16_s
          (i8x16.splat (local.get $a)) (i8x16.splat (local.get $b))))))";
    let got = match eval(dot, "f", &[Value::I32(3), Value::I32(4)]) {
        Value::I32(v) => v,
        _ => unreachable!(),
    };
    assert_eq!(got, 24, "3·4 + 3·4");

    // i32x4.relaxed_dot_i8x16_i7x16_add_s: extadd_pairwise(dot) + c. dot lanes are all 24, so each
    // i32 lane = 24 + 24 + c = 48 + c. Accumulator c splatted to 100 ⇒ 148.
    let dota = "(module (func (export \"f\") (param $a i32) (param $b i32) (param $c i32) (result i32)
        (i32x4.extract_lane 0 (i32x4.relaxed_dot_i8x16_i7x16_add_s
          (i8x16.splat (local.get $a)) (i8x16.splat (local.get $b)) (i32x4.splat (local.get $c))))))";
    let got = match eval(dota, "f", &[Value::I32(3), Value::I32(4), Value::I32(100)]) {
        Value::I32(v) => v,
        _ => unreachable!(),
    };
    assert_eq!(got, 148, "24 + 24 + 100");
}

/// SIMD memory variants — splat-load, load-zero, load-extend, load/store-lane (the shapes clang
/// `-msimd128` emits constantly). Each composes a scalar load/store with a lane op; `eval` runs both
/// backends. A real auto-vectorized loop with a broadcast load exercises exactly these.
#[test]
fn simd_memory_variants() {
    // load32_splat: broadcast a scalar to every lane (read lane 3 to prove it reached the top).
    let splat = r#"(module (memory 1) (data (i32.const 0) "\78\56\34\12")
      (func (export "f") (result i32)
        (i32x4.extract_lane 3 (v128.load32_splat (i32.const 0)))))"#;
    assert_eq!(eval(splat, "f", &[]), Value::I32(0x12345678u32 as i32));

    // load8_splat: broadcast a byte; lane 9 = the byte.
    let bsplat = r#"(module (memory 1) (data (i32.const 0) "\2a")
      (func (export "f") (result i32)
        (i8x16.extract_lane_u 9 (v128.load8_splat (i32.const 0)))))"#;
    assert_eq!(eval(bsplat, "f", &[]), Value::I32(42));

    // load32_zero: lane 0 = scalar (-1), lane 1 = 0 ⇒ sum −1.
    let zero = r#"(module (memory 1) (data (i32.const 0) "\ff\ff\ff\ff")
      (func (export "f") (result i32)
        (i32.add
          (i32x4.extract_lane 0 (v128.load32_zero (i32.const 0)))
          (i32x4.extract_lane 1 (v128.load32_zero (i32.const 0))))))"#;
    assert_eq!(eval(zero, "f", &[]), Value::I32(-1));

    // load8x8_u: load 8 bytes [10,20,30,…], zero-extend to i16x8; lane 2 = 30.
    let ext = r#"(module (memory 1) (data (i32.const 0) "\0a\14\1e\28\32\3c\46\50")
      (func (export "f") (result i32)
        (i16x8.extract_lane_u 2 (v128.load8x8_u (i32.const 0)))))"#;
    assert_eq!(eval(ext, "f", &[]), Value::I32(30));

    // load16x4_s: load 4 i16 incl. a negative one, sign-extend to i32x4; lane 0 = -1.
    let exts = r#"(module (memory 1) (data (i32.const 0) "\ff\ff\01\00\02\00\03\00")
      (func (export "f") (result i32)
        (i32x4.extract_lane 0 (v128.load16x4_s (i32.const 0)))))"#;
    assert_eq!(eval(exts, "f", &[]), Value::I32(-1));

    // load32_lane: splice a loaded scalar into lane 2 of a splatted vector.
    let llane = r#"(module (memory 1) (data (i32.const 0) "\ad\de\00\00")
      (func (export "f") (result i32)
        (i32x4.extract_lane 2
          (v128.load32_lane 2 (i32.const 0) (i32x4.splat (i32.const 7))))))"#;
    assert_eq!(eval(llane, "f", &[]), Value::I32(0xdead));

    // store32_lane: extract lane 1 of a const vector, store it, load it back.
    let slane = r#"(module (memory 1)
      (func (export "f") (result i32)
        (v128.store32_lane 1 (i32.const 16) (v128.const i32x4 100 200 300 400))
        (i32.load (i32.const 16))))"#;
    assert_eq!(eval(slane, "f", &[]), Value::I32(200));
}

/// SIMD float rounding — `f32x4`/`f64x2` `.ceil/.floor/.trunc/.nearest` (the gap the spec-conformance
/// pass surfaced). `nearest` is round-to-nearest-ties-to-even. `eval` pins interp == JIT.
#[test]
fn simd_float_rounding() {
    let round = |op: &str, x: f32| {
        let wat = format!(
            "(module (func (export \"f\") (param $x f32) (result f32)
               (f32x4.extract_lane 0 ({op} (f32x4.splat (local.get $x))))))"
        );
        f32(eval(&wat, "f", &[Value::F32(x)]))
    };
    assert_eq!(round("f32x4.ceil", 2.3), 3.0);
    assert_eq!(round("f32x4.floor", 2.7), 2.0);
    assert_eq!(round("f32x4.trunc", -2.7), -2.0);
    assert_eq!(round("f32x4.nearest", 2.5), 2.0, "ties to even");
    assert_eq!(round("f32x4.nearest", 3.5), 4.0, "ties to even");
    // f64x2 path.
    let wat = "(module (func (export \"f\") (param $x f64) (result f64)
        (f64x2.extract_lane 1 (f64x2.floor (f64x2.splat (local.get $x))))))";
    assert_eq!(f64(eval(wat, "f", &[Value::F64(-1.2)])), -2.0);
}

#[test]
fn f64x2_pmin_pmax() {
    let pm = |op: &str| {
        format!(
            "(module (func (export \"f\") (param $a f64) (param $b f64) (result f64)
               (f64x2.extract_lane 1 ({op} (f64x2.splat (local.get $a)) (f64x2.splat (local.get $b))))))"
        )
    };
    let run = |op: &str, a: f64, b: f64| f64(eval(&pm(op), "f", &[Value::F64(a), Value::F64(b)]));
    for (a, b) in [(1.0f64, 2.0), (2.0, 1.0), (-0.0, 0.0), (3.5, -3.5)] {
        let pmin = if b < a { b } else { a };
        let pmax = if a < b { b } else { a };
        assert_eq!(
            run("f64x2.pmin", a, b).to_bits(),
            pmin.to_bits(),
            "pmin {a} {b}"
        );
        assert_eq!(
            run("f64x2.pmax", a, b).to_bits(),
            pmax.to_bits(),
            "pmax {a} {b}"
        );
    }
}

/// `i8x16.popcnt` through the wasm bridge: splat a byte across all lanes, popcount, read lane 0.
/// Oracle = Rust's `count_ones`; `eval` enforces interp == JIT.
#[test]
fn i8x16_popcnt() {
    let wat = "(module (func (export \"f\") (param $x i32) (result i32)
                 (i8x16.extract_lane_u 0 (i8x16.popcnt (i8x16.splat (local.get $x))))))";
    for byte in [0x00u8, 0xFF, 0x01, 0x80, 0xAA, 0x42] {
        let got = match eval(wat, "f", &[Value::I32(byte as i32)]) {
            Value::I32(x) => x,
            _ => unreachable!(),
        };
        assert_eq!(got, byte.count_ones() as i32, "popcnt 0x{byte:02x}");
    }
}

/// `i32x4.dot_i16x8_s` through the wasm bridge — the signed pairwise dot product that DSP/ML inner
/// loops emit. Two i16x8 vectors from `data`, dot, sum the four i32 lanes. With a=[1..8], b=[8..1]:
/// lanes = [1·8+2·7, 3·6+4·5, 5·4+6·3, 7·2+8·1] = [22, 38, 38, 22], total 120.
#[test]
fn i32x4_dot_i16x8_s() {
    let wat = r#"
    (module
      (memory 1)
      (data (i32.const 0)  "\01\00\02\00\03\00\04\00\05\00\06\00\07\00\08\00")
      (data (i32.const 16) "\08\00\07\00\06\00\05\00\04\00\03\00\02\00\01\00")
      (func (export "dot") (result i32)
        (local $p v128)
        (local.set $p
          (i32x4.dot_i16x8_s (v128.load (i32.const 0)) (v128.load (i32.const 16))))
        (i32.add
          (i32.add (i32x4.extract_lane 0 (local.get $p)) (i32x4.extract_lane 1 (local.get $p)))
          (i32.add (i32x4.extract_lane 2 (local.get $p)) (i32x4.extract_lane 3 (local.get $p))))))
    "#;
    assert_eq!(eval(wat, "dot", &[]), Value::I32(120));
}

/// Extended multiply + extadd_pairwise through the wasm bridge — the widening multiply-accumulate
/// that DSP/ML int8 kernels emit. Compute `i16x8.extmul_low_i8x16_s` of two byte vectors, then
/// `i32x4.extadd_pairwise_i16x8_s` to widen-and-sum — and read lane 0. With both operands = splat(3)
/// over the low 8 bytes: each extmul lane = 9, pairwise add of two = 18.
#[test]
fn extmul_extadd_pairwise_macc() {
    let wat = "(module (func (export \"f\") (param $x i32) (result i32)
                 (i32x4.extract_lane 0
                   (i32x4.extadd_pairwise_i16x8_s
                     (i16x8.extmul_low_i8x16_s
                       (i8x16.splat (local.get $x)) (i8x16.splat (local.get $x)))))))";
    for x in [3, 5, 10, -4] {
        let got = match eval(wat, "f", &[Value::I32(x)]) {
            Value::I32(v) => v,
            _ => unreachable!(),
        };
        // extmul_low_s squares the (sign-extended) low byte; pairwise add sums two adjacent.
        let sq = (x as i8 as i32).pow(2);
        assert_eq!(got, sq * 2, "macc {x}");
    }
}

/// `i16x8.q15mulr_sat_s` through the wasm bridge — Q15 fixed-point multiply (the saturating
/// rounding multiply audio/DSP code uses). Splat two i16 values, read lane 0; oracle = the
/// rounding-saturating formula. Includes the `-1.0 * -1.0` corner that saturates to i16::MAX.
#[test]
fn i16x8_q15mulr_sat_s() {
    let wat = "(module (func (export \"f\") (param $a i32) (param $b i32) (result i32)
                 (i16x8.extract_lane_s 0
                   (i16x8.q15mulr_sat_s (i16x8.splat (local.get $a)) (i16x8.splat (local.get $b))))))";
    for (a, b) in [
        (16384i16, 16384i16),
        (-32768, -32768),
        (1000, -2000),
        (0, 12345),
    ] {
        let got = match eval(wat, "f", &[Value::I32(a as i32), Value::I32(b as i32)]) {
            Value::I32(v) => v,
            _ => unreachable!(),
        };
        let want = ((a as i64 * b as i64 + 0x4000) >> 15).clamp(i16::MIN as i64, i16::MAX as i64);
        assert_eq!(got, want as i32, "q15mulr {a} {b}");
    }
}

/// `i8x16.avgr_u` through the wasm bridge — the unsigned rounding average that image/blend kernels
/// emit for "blend two pixels". Splat two byte values, read lane 0; oracle = `(a+b+1)>>1`.
#[test]
fn i8x16_avgr_u() {
    let wat = "(module (func (export \"f\") (param $a i32) (param $b i32) (result i32)
                 (i8x16.extract_lane_u 0
                   (i8x16.avgr_u (i8x16.splat (local.get $a)) (i8x16.splat (local.get $b))))))";
    for (a, b) in [(0u8, 0u8), (255, 255), (3, 4), (255, 1), (100, 101)] {
        let got = match eval(wat, "f", &[Value::I32(a as i32), Value::I32(b as i32)]) {
            Value::I32(x) => x,
            _ => unreachable!(),
        };
        assert_eq!(
            got,
            ((a as u32 + b as u32 + 1) >> 1) as i32,
            "avgr_u {a} {b}"
        );
    }
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

/// A real "does this vector contain X?" idiom (the SIMD `memchr` shape): compare a constant vector
/// against a splatted needle, then `v128.any_true`. Exercises a reduction fed by a lane compare.
#[test]
fn i8x16_any_match() {
    let wat = r#"
    (module
      (func (export "has") (param $needle i32) (result i32)
        (v128.any_true
          (i8x16.eq
            (v128.const i8x16 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16)
            (i8x16.splat (local.get $needle))))))
    "#;
    let has = |n: i32| match eval(wat, "has", &[Value::I32(n)]) {
        Value::I32(v) => v,
        _ => unreachable!(),
    };
    assert_eq!(has(5), 1, "5 is present");
    assert_eq!(has(16), 1, "16 is present");
    assert_eq!(has(0), 0, "0 is absent");
    assert_eq!(has(99), 0, "99 is absent");
}

/// `i32x4.bitmask` through the wasm bridge: the sign bit of each lane gathered into an i32 — the
/// move-mask used to branch on a lane compare. Compare [1,2,3,4] < splat(3) → lanes [T,T,F,F] → 0b0011.
#[test]
fn i32x4_bitmask_of_compare() {
    let wat = r#"
    (module
      (func (export "mask") (param $t i32) (result i32)
        (i32x4.bitmask
          (i32x4.lt_s (v128.const i32x4 1 2 3 4) (i32x4.splat (local.get $t))))))
    "#;
    let mask = |t: i32| match eval(wat, "mask", &[Value::I32(t)]) {
        Value::I32(v) => v,
        _ => unreachable!(),
    };
    assert_eq!(mask(3), 0b0011, "lanes 1 and 2 are < 3");
    assert_eq!(mask(5), 0b1111, "all < 5");
    assert_eq!(mask(0), 0b0000, "none < 0");
}

/// Saturating add through the wasm bridge — the classic "blend pixels without overflow" idiom:
/// `i8x16.add_sat_u` of two byte vectors clamps each lane at 255 instead of wrapping.
#[test]
fn i8x16_add_sat_u() {
    let wat = r#"
    (module
      (func (export "blend") (param $x i32) (param $y i32) (result i32)
        (i8x16.extract_lane_u 0
          (i8x16.add_sat_u (i8x16.splat (local.get $x)) (i8x16.splat (local.get $y))))))
    "#;
    let blend = |x: i32, y: i32| match eval(wat, "blend", &[Value::I32(x), Value::I32(y)]) {
        Value::I32(v) => v,
        _ => unreachable!(),
    };
    assert_eq!(blend(200, 100), 255, "200 + 100 saturates to 255");
    assert_eq!(blend(10, 20), 30, "10 + 20 = 30");
    assert_eq!(blend(255, 255), 255, "255 + 255 = 255");
}

/// Widen through the wasm bridge: sum the low 8 bytes of a vector by widening `i8x16`→`i16x8` (so
/// the adds don't overflow a byte), then horizontal-add a couple of lanes. Exercises the real
/// "widen before accumulate" pattern.
#[test]
fn i8x16_widen_low() {
    let wat = r#"
    (module
      (func (export "lane0") (param $x i32) (result i32)
        (i16x8.extract_lane_s 0
          (i16x8.extend_low_i8x16_s (i8x16.splat (local.get $x))))))
    "#;
    let lane0 = |x: i32| match eval(wat, "lane0", &[Value::I32(x)]) {
        Value::I32(v) => v,
        _ => unreachable!(),
    };
    assert_eq!(lane0(200), -56, "(i8)200 sign-extends to -56");
    assert_eq!(lane0(5), 5, "5 widens to 5");
    assert_eq!(lane0(0x80), -128, "(i8)0x80 = -128");
}

/// Narrow through the wasm bridge: clamp-pack two i16 vectors to u8 (the "convert 16-bit samples to
/// bytes with saturation" idiom). `i8x16.narrow_i16x8_u` of splat(x) — every lane clamps to [0,255].
#[test]
fn i8x16_narrow_u() {
    let wat = r#"
    (module
      (func (export "pack") (param $x i32) (result i32)
        (i8x16.extract_lane_u 0
          (i8x16.narrow_i16x8_u (i16x8.splat (local.get $x)) (i16x8.splat (local.get $x))))))
    "#;
    let pack = |x: i32| match eval(wat, "pack", &[Value::I32(x)]) {
        Value::I32(v) => v,
        _ => unreachable!(),
    };
    assert_eq!(pack(300), 255, "300 clamps to 255");
    assert_eq!(pack(-5), 0, "(i16)-5 clamps to 0");
    assert_eq!(pack(200), 200, "200 stays");
}

/// Int↔float conversions through the wasm bridge: convert an i32 vector to f32, scale, and
/// `trunc_sat` back — the "do float math on integer pixels/samples" pattern. Also pins the
/// saturating NaN/overflow behaviour.
#[test]
fn i32x4_convert_trunc_roundtrip() {
    // (i32) -> i32: round x through f32, multiply by 2.0, truncate back (saturating).
    let wat = r#"
    (module
      (func (export "f") (param $x i32) (result i32)
        (i32x4.extract_lane 0
          (i32x4.trunc_sat_f32x4_s
            (f32x4.mul
              (f32x4.convert_i32x4_s (i32x4.splat (local.get $x)))
              (f32x4.splat (f32.const 2.0)))))))
    "#;
    let f = |x: i32| match eval(wat, "f", &[Value::I32(x)]) {
        Value::I32(v) => v,
        _ => unreachable!(),
    };
    assert_eq!(f(10), 20, "10 → 20.0 → 20");
    assert_eq!(f(-7), -14, "-7 → -14");
    assert_eq!(
        f(2_000_000_000),
        i32::MAX,
        "2e9*2 overflows i32 → saturates to MAX"
    );
}
