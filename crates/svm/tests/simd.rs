//! §17 SIMD (fixed-128 `v128`, D58) — end-to-end tests across the pipeline:
//! text round-trip, binary encode round-trip, the interpreter reference semantics, and the
//! interpreter↔JIT differential (the escape-freedom oracle, §18/I4).
//!
//! Vectors are observed via scalar `extract_lane` so results fit the `i64`-slot JIT calling
//! convention; that is also the natural way a guest consumes a lane result.

use svm_encode::{decode_module, encode_module};
use svm_interp::{run, Value};
use svm_jit::{compile_and_run, JitOutcome};
use svm_text::{parse_module, print_module};
use svm_verify::verify_module;

/// Parse + verify, asserting both succeed.
fn build(src: &str) -> svm_ir::Module {
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");
    m
}

/// Run on the interpreter; expect a single returned value.
fn interp1(src: &str, args: &[Value]) -> Value {
    let m = build(src);
    let mut fuel = 1_000_000u64;
    let out = run(&m, 0, args, &mut fuel).expect("interp run");
    assert_eq!(out.len(), 1, "expected one result");
    out[0]
}

/// Assert the JIT and interpreter agree (single scalar result), returning that result.
fn diff1(src: &str, args: &[Value]) -> i64 {
    let m = build(src);
    let mut fuel = 1_000_000u64;
    let interp = run(&m, 0, args, &mut fuel).expect("interp");
    let slots: Vec<i64> = args
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
    let jit = match compile_and_run(&m, 0, &slots).expect("jit") {
        JitOutcome::Returned(v) => v,
        other => panic!("jit did not return: {other:?}"),
    };
    let want = match interp[0] {
        Value::I32(x) => x as u32 as i64,
        Value::I64(x) => x,
        Value::F32(x) => x.to_bits() as i64,
        Value::F64(x) => x.to_bits() as i64,
        Value::V128(b) => i64::from_le_bytes(b[..8].try_into().unwrap()),
        Value::Ref(x) => x as i64,
    };
    // JIT i32 results occupy the low 32 bits of the slot; compare width-appropriately.
    let got = jit[0];
    match interp[0] {
        Value::I32(_) => assert_eq!(got as i32 as i64, want as i32 as i64, "i32 mismatch"),
        _ => assert_eq!(got, want, "jit vs interp mismatch"),
    }
    want
}

// ---------------------------------------------------------------------------
// Text + binary round-trips
// ---------------------------------------------------------------------------

/// A module exercising every new SIMD op, round-tripped through text (parse→print→parse)
/// and binary (encode→decode), asserting structural identity both ways.
#[test]
fn simd_text_and_binary_roundtrip() {
    let src = "memory 16\n\
        func (i32, f32) -> (i32) {\n\
        block0(v0: i32, v1: f32):\n\
          v2 = v128.const 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16\n\
          v3 = i32x4.splat v0\n\
          v4 = f32x4.splat v1\n\
          v5 = i32x4.add v3 v3\n\
          v6 = i32x4.sub v3 v3\n\
          v7 = i32x4.mul v3 v3\n\
          v8 = f32x4.mul v4 v4\n\
          v9 = f32x4.div v4 v4\n\
          v10 = f32x4.min v4 v4\n\
          v11 = f32x4.max v4 v4\n\
          v12 = f32x4.sqrt v8\n\
          v13 = f32x4.abs v4\n\
          v14 = f32x4.neg v4\n\
          v15 = v128.and v2 v3\n\
          v16 = v128.or v2 v3\n\
          v17 = v128.xor v2 v3\n\
          v18 = v128.andnot v2 v3\n\
          v19 = v128.not v2\n\
          v20 = v128.bitselect v2 v3 v5\n\
          v21 = i8x16.shuffle 0 16 1 17 2 18 3 19 4 20 5 21 6 22 7 23 v2 v3\n\
          v22 = i8x16.swizzle v2 v3\n\
          v23 = i8x16.replace_lane 3 v2 v0\n\
          v24 = i16x8.splat v0\n\
          v25 = i64x2.add v3 v3\n\
          v26 = i8x16.extract_lane_s 3 v2\n\
          v27 = i8x16.extract_lane_u 3 v2\n\
          v28 = i32x4.extract_lane 0 v5\n\
          v29 = simd.width_bytes\n\
          v30 = i64.const 0\n\
          v128.store v30 v2 offset=0\n\
          v31 = v128.load v30\n\
          v32 = i32x4.extract_lane 1 v31\n\
          return v32\n\
        }\n";
    let m = build(src);

    // Text round-trip: print, reparse, structural equality.
    let printed = print_module(&m);
    let reparsed = parse_module(&printed)
        .unwrap_or_else(|e| panic!("reparse failed: {e}\n--- printed ---\n{printed}"));
    assert_eq!(m, reparsed, "text round-trip changed the module");

    // Binary round-trip: encode, decode, structural equality.
    let bytes = encode_module(&m);
    let decoded = decode_module(&bytes).expect("decode");
    assert_eq!(m, decoded, "binary round-trip changed the module");
}

// ---------------------------------------------------------------------------
// Interpreter reference semantics
// ---------------------------------------------------------------------------

#[test]
fn v128_const_and_i32x4_extract() {
    // i32x4 = [1, 2, 3, 4] (little-endian lane bytes).
    let src = "func () -> (i32) {\n\
        block0():\n\
          v0 = v128.const 1 0 0 0 2 0 0 0 3 0 0 0 4 0 0 0\n\
          v1 = i32x4.extract_lane 0 v0\n\
          v2 = i32x4.extract_lane 3 v0\n\
          v3 = i32.add v1 v2\n\
          return v3\n\
        }\n";
    assert_eq!(interp1(src, &[]), Value::I32(5)); // 1 + 4
}

#[test]
fn i8x16_extract_sign_vs_zero() {
    // Lane 0 byte = 0xFF; signed extract = -1, unsigned = 255.
    let s = "func () -> (i32) {\nblock0():\n\
        v0 = v128.const 255 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n\
        v1 = i8x16.extract_lane_s 0 v0\n  return v1\n}\n";
    assert_eq!(interp1(s, &[]), Value::I32(-1));
    let u = "func () -> (i32) {\nblock0():\n\
        v0 = v128.const 255 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n\
        v1 = i8x16.extract_lane_u 0 v0\n  return v1\n}\n";
    assert_eq!(interp1(u, &[]), Value::I32(255));
}

#[test]
fn i8x16_shuffle_interleaves() {
    // Interleave low bytes of a=[0..16] and b=[100..116]: result lane0=a0, lane1=b0, ...
    let s = "func () -> (i32) {\nblock0():\n\
        v0 = v128.const 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15\n\
        v1 = v128.const 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115\n\
        v2 = i8x16.shuffle 0 16 1 17 2 18 3 19 4 20 5 21 6 22 7 23 v0 v1\n\
        v3 = i8x16.extract_lane_u 1 v2\n  return v3\n}\n";
    // Result byte 1 = b[0] = 100.
    assert_eq!(interp1(s, &[]), Value::I32(100));
}

#[test]
fn i8x16_swizzle_indexes_and_zeroes() {
    // a[i]=i; indices select a[3] into lane0, and an out-of-range (200) → 0 into lane1.
    let s = "func () -> (i32) {\nblock0():\n\
        v0 = v128.const 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15\n\
        v1 = v128.const 3 200 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n\
        v2 = i8x16.swizzle v0 v1\n\
        v3 = i8x16.extract_lane_u 0 v2\n\
        v4 = i8x16.extract_lane_u 1 v2\n\
        v5 = i32.add v3 v4\n  return v5\n}\n";
    assert_eq!(interp1(s, &[]), Value::I32(3)); // a[3]=3, plus 0
}

#[test]
fn simd_width_is_fixed_128() {
    let s = "func () -> (i32) {\nblock0():\n  v0 = simd.width_bytes\n  return v0\n}\n";
    assert_eq!(interp1(s, &[]), Value::I32(16));
}

// ---------------------------------------------------------------------------
// Interpreter ↔ JIT differential
// ---------------------------------------------------------------------------

#[test]
fn diff_i32x4_splat_add_extract() {
    let s = "func (i32, i32) -> (i32) {\n\
        block0(v0: i32, v1: i32):\n\
          v2 = i32x4.splat v0\n\
          v3 = i32x4.splat v1\n\
          v4 = i32x4.add v2 v3\n\
          v5 = i32x4.extract_lane 2 v4\n\
          return v5\n}\n";
    for (a, b) in [(1, 2), (-5, 9), (i32::MAX, 1), (0, 0)] {
        let r = diff1(s, &[Value::I32(a), Value::I32(b)]);
        assert_eq!(r as i32, a.wrapping_add(b));
    }
}

#[test]
fn diff_i32x4_mul_and_sub() {
    let s = "func (i32, i32) -> (i32) {\n\
        block0(v0: i32, v1: i32):\n\
          v2 = i32x4.splat v0\n\
          v3 = i32x4.splat v1\n\
          v4 = i32x4.mul v2 v3\n\
          v5 = i32x4.sub v4 v3\n\
          v6 = i32x4.extract_lane 0 v5\n  return v6\n}\n";
    for (a, b) in [(3, 4), (-2, 7), (1000, 1000)] {
        let r = diff1(s, &[Value::I32(a), Value::I32(b)]);
        assert_eq!(r as i32, a.wrapping_mul(b).wrapping_sub(b));
    }
}

#[test]
fn diff_f32x4_arith() {
    // (x*x + x) / 2 lanewise, observed at lane 0.
    let s = "func (f32) -> (f32) {\n\
        block0(v0: f32):\n\
          v1 = f32x4.splat v0\n\
          v2 = f32x4.mul v1 v1\n\
          v3 = f32x4.add v2 v1\n\
          v4 = f32x4.extract_lane 0 v3\n  return v4\n}\n";
    for x in [0.0f32, 1.5, -3.25, 1e9, 0.1] {
        let r = diff1(s, &[Value::F32(x)]);
        let want = (x * x + x).to_bits() as i64;
        assert_eq!(r as u32, want as u32);
    }
}

#[test]
fn diff_f64x2_min_max_sqrt() {
    let s = "func (f64, f64) -> (f64) {\n\
        block0(v0: f64, v1: f64):\n\
          v2 = f64x2.splat v0\n\
          v3 = f64x2.splat v1\n\
          v4 = f64x2.max v2 v3\n\
          v5 = f64x2.min v4 v3\n\
          v6 = f64x2.sqrt v5\n\
          v7 = f64x2.extract_lane 1 v6\n  return v7\n}\n";
    for (a, b) in [(4.0f64, 9.0), (16.0, 2.0), (1.0, 100.0)] {
        diff1(s, &[Value::F64(a), Value::F64(b)]);
    }
}

#[test]
fn diff_bitwise_and_bitselect() {
    let s = "func (i32, i32) -> (i32) {\n\
        block0(v0: i32, v1: i32):\n\
          v2 = i32x4.splat v0\n\
          v3 = i32x4.splat v1\n\
          v4 = v128.and v2 v3\n\
          v5 = v128.or v4 v3\n\
          v6 = v128.xor v5 v2\n\
          v7 = v128.not v6\n\
          v8 = v128.andnot v7 v2\n\
          v9 = v128.bitselect v2 v3 v8\n\
          v10 = i32x4.extract_lane 0 v9\n  return v10\n}\n";
    for (a, b) in [
        (0xF0F0_F0F0u32 as i32, 0x0FF0_0FF0),
        (0, -1),
        (12345, 67890),
    ] {
        diff1(s, &[Value::I32(a), Value::I32(b)]);
    }
}

#[test]
fn diff_shuffle_swizzle() {
    let s = "func (i32) -> (i32) {\n\
        block0(v0: i32):\n\
          v1 = i32x4.splat v0\n\
          v2 = v128.const 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15\n\
          v3 = i8x16.shuffle 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1 0 v1 v2\n\
          v4 = i8x16.swizzle v3 v2\n\
          v5 = i8x16.extract_lane_u 0 v4\n  return v5\n}\n";
    for v in [0, 1, -1, 0x01020304] {
        diff1(s, &[Value::I32(v)]);
    }
}

#[cfg(unix)]
#[test]
fn diff_v128_load_store_roundtrip() {
    // Store a splatted vector to the window, load it back, extract a lane — same on both backends.
    let s = "memory 16\n\
        func (i32) -> (i32) {\n\
        block0(v0: i32):\n\
          v1 = i64.const 32\n\
          v2 = i32x4.splat v0\n\
          v128.store v1 v2\n\
          v3 = v128.load v1\n\
          v4 = i32x4.extract_lane 3 v3\n  return v4\n}\n";
    for v in [0, 42, -7, i32::MIN] {
        let r = diff1(s, &[Value::I32(v)]);
        assert_eq!(r as i32, v);
    }
}

#[cfg(unix)]
#[test]
fn diff_replace_lane() {
    let s = "func (i32, i32) -> (i32) {\n\
        block0(v0: i32, v1: i32):\n\
          v2 = i32x4.splat v0\n\
          v3 = i32x4.replace_lane 2 v2 v1\n\
          v4 = i32x4.extract_lane 2 v3\n\
          v5 = i32x4.extract_lane 1 v3\n\
          v6 = i32.add v4 v5\n  return v6\n}\n";
    for (a, b) in [(10, 20), (-1, 5), (0, 0)] {
        let r = diff1(s, &[Value::I32(a), Value::I32(b)]);
        assert_eq!(r as i32, b.wrapping_add(a)); // lane2 replaced with b, lane1 still a
    }
}

// ---------------------------------------------------------------------------
// Integer lane comparisons (VIntCmp) — a per-lane all-ones/all-zeros mask.
// ---------------------------------------------------------------------------

/// Text + binary round-trip of every comparison family across the integer shapes.
#[test]
fn lane_compare_roundtrip() {
    let src = "func (i32) -> (i32) {\n\
        block0(v0: i32):\n\
          v1 = i8x16.splat v0\n\
          v2 = i16x8.splat v0\n\
          v3 = i32x4.splat v0\n\
          v4 = i8x16.eq v1 v1\n\
          v5 = i8x16.ne v1 v1\n\
          v6 = i8x16.lt_s v1 v1\n\
          v7 = i8x16.lt_u v1 v1\n\
          v8 = i8x16.gt_s v1 v1\n\
          v9 = i8x16.gt_u v1 v1\n\
          v10 = i8x16.le_s v1 v1\n\
          v11 = i8x16.le_u v1 v1\n\
          v12 = i8x16.ge_s v1 v1\n\
          v13 = i8x16.ge_u v1 v1\n\
          v14 = i16x8.eq v2 v2\n\
          v15 = i16x8.lt_u v2 v2\n\
          v16 = i32x4.ne v3 v3\n\
          v17 = i32x4.ge_s v3 v3\n\
          v18 = i64x2.eq v3 v3\n\
          v19 = i64x2.lt_s v3 v3\n\
          v20 = i32x4.extract_lane 0 v16\n\
          return v20\n\
        }\n";
    let m = build(src);
    let reparsed = parse_module(&print_module(&m)).expect("reparse");
    assert_eq!(m, reparsed, "text round-trip changed the module");
    let decoded = decode_module(&encode_module(&m)).expect("decode");
    assert_eq!(m, decoded, "binary round-trip changed the module");
}

/// `i32x4.<cmp>` of two splatted scalars, the lane-0 mask read back as a signed i32 (`-1` true /
/// `0` false). `diff1` asserts interp == JIT; the expectation is Rust's own comparison (the oracle).
fn i32x4_cmp(op: &str, a: i32, b: i32) -> i32 {
    let s = format!(
        "func (i32, i32) -> (i32) {{\nblock0(v0: i32, v1: i32):\n\
         \x20 v2 = i32x4.splat v0\n  v3 = i32x4.splat v1\n  v4 = i32x4.{op} v2 v3\n\
         \x20 v5 = i32x4.extract_lane 0 v4\n  return v5\n}}\n"
    );
    diff1(&s, &[Value::I32(a), Value::I32(b)]) as i32
}

#[test]
fn diff_i32x4_lane_compares() {
    let mask = |t: bool| if t { -1 } else { 0 };
    for (a, b) in [
        (5, 5),
        (5, 7),
        (7, 5),
        (-1, 1),
        (1, -1),
        (i32::MIN, i32::MAX),
    ] {
        let (ua, ub) = (a as u32, b as u32);
        assert_eq!(i32x4_cmp("eq", a, b), mask(a == b), "eq {a} {b}");
        assert_eq!(i32x4_cmp("ne", a, b), mask(a != b), "ne {a} {b}");
        assert_eq!(i32x4_cmp("lt_s", a, b), mask(a < b), "lt_s {a} {b}");
        assert_eq!(i32x4_cmp("gt_s", a, b), mask(a > b), "gt_s {a} {b}");
        assert_eq!(i32x4_cmp("le_s", a, b), mask(a <= b), "le_s {a} {b}");
        assert_eq!(i32x4_cmp("ge_s", a, b), mask(a >= b), "ge_s {a} {b}");
        assert_eq!(i32x4_cmp("lt_u", a, b), mask(ua < ub), "lt_u {a} {b}");
        assert_eq!(i32x4_cmp("gt_u", a, b), mask(ua > ub), "gt_u {a} {b}");
        assert_eq!(i32x4_cmp("le_u", a, b), mask(ua <= ub), "le_u {a} {b}");
        assert_eq!(i32x4_cmp("ge_u", a, b), mask(ua >= ub), "ge_u {a} {b}");
    }
}

/// Narrow lanes: `i8x16.splat` broadcasts the low byte, so this compares `a`/`b` as bytes. The
/// `0xFF`-vs-`1` case pins the signed/unsigned distinction (`-1 <_s 1` true, `255 <_u 1` false).
fn i8x16_cmp(op: &str, a: i32, b: i32) -> i32 {
    let s = format!(
        "func (i32, i32) -> (i32) {{\nblock0(v0: i32, v1: i32):\n\
         \x20 v2 = i8x16.splat v0\n  v3 = i8x16.splat v1\n  v4 = i8x16.{op} v2 v3\n\
         \x20 v5 = i8x16.extract_lane_s 0 v4\n  return v5\n}}\n"
    );
    diff1(&s, &[Value::I32(a), Value::I32(b)]) as i32
}

#[test]
fn diff_i8x16_lane_compares_signedness() {
    assert_eq!(i8x16_cmp("eq", 0x102, 2), -1, "low byte 2 == 2");
    assert_eq!(i8x16_cmp("ne", 0xFF, 1), -1, "255 != 1");
    assert_eq!(i8x16_cmp("lt_s", 0xFF, 1), -1, "(-1) <_s 1");
    assert_eq!(i8x16_cmp("lt_u", 0xFF, 1), 0, "255 <_u 1 is false");
    assert_eq!(i8x16_cmp("gt_s", 0xFF, 1), 0, "(-1) >_s 1 is false");
    assert_eq!(i8x16_cmp("gt_u", 0xFF, 1), -1, "255 >_u 1");
    assert_eq!(i8x16_cmp("ge_u", 1, 1), -1, "1 >=_u 1");
}

/// `i16x8` and `i64x2` shapes (i64x2 is signed-only in the wasm spec). Observed via the
/// shape-appropriate `extract_lane`.
#[test]
fn diff_i16x8_and_i64x2_lane_compares() {
    let i16x8 = |op: &str, a: i32, b: i32| -> i32 {
        let s = format!(
            "func (i32, i32) -> (i32) {{\nblock0(v0: i32, v1: i32):\n\
             \x20 v2 = i16x8.splat v0\n  v3 = i16x8.splat v1\n  v4 = i16x8.{op} v2 v3\n\
             \x20 v5 = i16x8.extract_lane_s 0 v4\n  return v5\n}}\n"
        );
        diff1(&s, &[Value::I32(a), Value::I32(b)]) as i32
    };
    // Low halfword: 0xFFFF → -1 signed / 65535 unsigned.
    assert_eq!(i16x8("lt_s", 0xFFFF, 1), -1, "(-1) <_s 1");
    assert_eq!(i16x8("lt_u", 0xFFFF, 1), 0, "65535 <_u 1 is false");
    assert_eq!(i16x8("eq", 0x10003, 3), -1, "low halfword 3 == 3");

    let i64x2 = |op: &str, a: i64, b: i64| -> i64 {
        let s = format!(
            "func (i64, i64) -> (i64) {{\nblock0(v0: i64, v1: i64):\n\
             \x20 v2 = i64x2.splat v0\n  v3 = i64x2.splat v1\n  v4 = i64x2.{op} v2 v3\n\
             \x20 v5 = i64x2.extract_lane 0 v4\n  return v5\n}}\n"
        );
        diff1(&s, &[Value::I64(a), Value::I64(b)])
    };
    let mask = |t: bool| if t { -1i64 } else { 0 };
    for (a, b) in [(5i64, 5), (5, 7), (-1, 1), (i64::MIN, i64::MAX)] {
        assert_eq!(i64x2("eq", a, b), mask(a == b), "i64x2.eq {a} {b}");
        assert_eq!(i64x2("ne", a, b), mask(a != b), "i64x2.ne {a} {b}");
        assert_eq!(i64x2("lt_s", a, b), mask(a < b), "i64x2.lt_s {a} {b}");
        assert_eq!(i64x2("gt_s", a, b), mask(a > b), "i64x2.gt_s {a} {b}");
        assert_eq!(i64x2("le_s", a, b), mask(a <= b), "i64x2.le_s {a} {b}");
        assert_eq!(i64x2("ge_s", a, b), mask(a >= b), "i64x2.ge_s {a} {b}");
    }
}

// ---------------------------------------------------------------------------
// Integer lane min/max (signed + unsigned) — VIntBinOp::Min*/Max*.
// ---------------------------------------------------------------------------

/// `i32x4.{min,max}_{s,u}` of two splatted scalars, lane 0 read back. `diff1` asserts interp == JIT;
/// the expectation is Rust's own `min`/`max` (the oracle).
fn i32x4_minmax(op: &str, a: i32, b: i32) -> i32 {
    let s = format!(
        "func (i32, i32) -> (i32) {{\nblock0(v0: i32, v1: i32):\n\
         \x20 v2 = i32x4.splat v0\n  v3 = i32x4.splat v1\n  v4 = i32x4.{op} v2 v3\n\
         \x20 v5 = i32x4.extract_lane 0 v4\n  return v5\n}}\n"
    );
    diff1(&s, &[Value::I32(a), Value::I32(b)]) as i32
}

#[test]
fn diff_i32x4_min_max() {
    for (a, b) in [(3, 7), (-5, 2), (i32::MIN, i32::MAX), (4, 4), (-1, -100)] {
        let (ua, ub) = (a as u32, b as u32);
        assert_eq!(i32x4_minmax("min_s", a, b), a.min(b), "min_s {a} {b}");
        assert_eq!(i32x4_minmax("max_s", a, b), a.max(b), "max_s {a} {b}");
        assert_eq!(
            i32x4_minmax("min_u", a, b),
            ua.min(ub) as i32,
            "min_u {a} {b}"
        );
        assert_eq!(
            i32x4_minmax("max_u", a, b),
            ua.max(ub) as i32,
            "max_u {a} {b}"
        );
    }
}

/// Narrow lanes: `i8x16.splat` broadcasts the low byte; the result lane 0 is read as a signed i8.
/// `a = 0xFF` (−1 signed / 255 unsigned) vs `b = 1` pins the signed/unsigned split.
#[test]
fn diff_i8x16_and_i16x8_min_max() {
    let i8 = |op: &str, a: i32, b: i32| -> i32 {
        let s = format!(
            "func (i32, i32) -> (i32) {{\nblock0(v0: i32, v1: i32):\n\
             \x20 v2 = i8x16.splat v0\n  v3 = i8x16.splat v1\n  v4 = i8x16.{op} v2 v3\n\
             \x20 v5 = i8x16.extract_lane_s 0 v4\n  return v5\n}}\n"
        );
        diff1(&s, &[Value::I32(a), Value::I32(b)]) as i32
    };
    assert_eq!(i8("min_s", 0xFF, 1), -1, "min_s(-1, 1)");
    assert_eq!(i8("max_s", 0xFF, 1), 1, "max_s(-1, 1)");
    assert_eq!(i8("min_u", 0xFF, 1), 1, "min_u(255, 1)");
    assert_eq!(i8("max_u", 0xFF, 1), -1, "max_u(255, 1) = 255 → -1 as i8");

    let i16 = |op: &str, a: i32, b: i32| -> i32 {
        let s = format!(
            "func (i32, i32) -> (i32) {{\nblock0(v0: i32, v1: i32):\n\
             \x20 v2 = i16x8.splat v0\n  v3 = i16x8.splat v1\n  v4 = i16x8.{op} v2 v3\n\
             \x20 v5 = i16x8.extract_lane_s 0 v4\n  return v5\n}}\n"
        );
        diff1(&s, &[Value::I32(a), Value::I32(b)]) as i32
    };
    assert_eq!(i16("min_s", 0xFFFF, 1), -1, "min_s(-1, 1)");
    assert_eq!(
        i16("max_u", 0xFFFF, 1),
        -1,
        "max_u(65535, 1) = 65535 → -1 as i16"
    );
    assert_eq!(i16("min_u", 0xFFFF, 1), 1, "min_u(65535, 1)");
}

// ---------------------------------------------------------------------------
// Float lane comparisons (VFloatCmp) — ordered (ne unordered) → mask.
// ---------------------------------------------------------------------------

/// Float lane compares round-trip through text + binary.
#[test]
fn float_lane_compare_roundtrip() {
    let src = "func (f32, f64) -> (i32) {\n\
        block0(v0: f32, v1: f64):\n\
          v2 = f32x4.splat v0\n\
          v3 = f64x2.splat v1\n\
          v4 = f32x4.eq v2 v2\n\
          v5 = f32x4.ne v2 v2\n\
          v6 = f32x4.lt v2 v2\n\
          v7 = f32x4.ge v2 v2\n\
          v8 = f64x2.le v3 v3\n\
          v9 = f64x2.gt v3 v3\n\
          v10 = i32x4.extract_lane 0 v4\n\
          return v10\n\
        }\n";
    let m = build(src);
    assert_eq!(
        parse_module(&print_module(&m)).expect("reparse"),
        m,
        "text round-trip"
    );
    assert_eq!(
        decode_module(&encode_module(&m)).expect("decode"),
        m,
        "binary round-trip"
    );
}

/// `f32x4.<cmp>` of two splatted scalars; the lane-0 mask read back via `i32x4.extract_lane` (`-1`
/// true / `0` false). Oracle = Rust's float operators (which share wasm's ordered/`ne`-unordered rule).
fn f32x4_cmp(op: &str, a: f32, b: f32) -> i32 {
    let s = format!(
        "func (f32, f32) -> (i32) {{\nblock0(v0: f32, v1: f32):\n\
         \x20 v2 = f32x4.splat v0\n  v3 = f32x4.splat v1\n  v4 = f32x4.{op} v2 v3\n\
         \x20 v5 = i32x4.extract_lane 0 v4\n  return v5\n}}\n"
    );
    diff1(&s, &[Value::F32(a), Value::F32(b)]) as i32
}

#[test]
fn diff_f32x4_lane_compares() {
    let mask = |t: bool| if t { -1 } else { 0 };
    let nan = f32::NAN;
    for (a, b) in [
        (1.0, 2.0),
        (2.0, 2.0),
        (3.0, 1.0),
        (nan, 1.0),
        (1.0, nan),
        (nan, nan),
        (-0.0, 0.0),
    ] {
        assert_eq!(f32x4_cmp("eq", a, b), mask(a == b), "eq {a} {b}");
        assert_eq!(f32x4_cmp("ne", a, b), mask(a != b), "ne {a} {b}");
        assert_eq!(f32x4_cmp("lt", a, b), mask(a < b), "lt {a} {b}");
        assert_eq!(f32x4_cmp("gt", a, b), mask(a > b), "gt {a} {b}");
        assert_eq!(f32x4_cmp("le", a, b), mask(a <= b), "le {a} {b}");
        assert_eq!(f32x4_cmp("ge", a, b), mask(a >= b), "ge {a} {b}");
    }
}

#[test]
fn diff_f64x2_lane_compares() {
    let f64x2 = |op: &str, a: f64, b: f64| -> i64 {
        let s = format!(
            "func (f64, f64) -> (i64) {{\nblock0(v0: f64, v1: f64):\n\
             \x20 v2 = f64x2.splat v0\n  v3 = f64x2.splat v1\n  v4 = f64x2.{op} v2 v3\n\
             \x20 v5 = i64x2.extract_lane 0 v4\n  return v5\n}}\n"
        );
        diff1(&s, &[Value::F64(a), Value::F64(b)])
    };
    let mask = |t: bool| if t { -1i64 } else { 0 };
    let nan = f64::NAN;
    for (a, b) in [(1.0, 2.0), (2.0, 2.0), (nan, 1.0), (nan, nan)] {
        assert_eq!(f64x2("eq", a, b), mask(a == b), "eq {a} {b}");
        assert_eq!(f64x2("ne", a, b), mask(a != b), "ne {a} {b}");
        assert_eq!(f64x2("lt", a, b), mask(a < b), "lt {a} {b}");
        assert_eq!(f64x2("ge", a, b), mask(a >= b), "ge {a} {b}");
    }
}

// ---------------------------------------------------------------------------
// Integer lane shifts (VShift) — one scalar amount, mod the lane bit-width.
// ---------------------------------------------------------------------------

/// `<shape>.<shift>` of a splatted scalar by an i32 amount; lane 0 read back. Oracle = the scalar
/// shift in Rust at the lane width.
fn vshift(shape: &str, op: &str, ext: &str, val: i64, amt: i32) -> i64 {
    let (ity, splat, vty) = if shape == "i64x2" {
        ("i64", "i64x2.splat", "i64")
    } else {
        ("i32", &format!("{shape}.splat")[..], "i32")
    };
    let s = format!(
        "func ({ity}, i32) -> ({vty}) {{\nblock0(v0: {ity}, v1: i32):\n\
         \x20 v2 = {splat} v0\n  v3 = {shape}.{op} v2 v1\n  v4 = {shape}.{ext} v3\n  return v4\n}}\n"
    );
    let arg0 = if shape == "i64x2" {
        Value::I64(val)
    } else {
        Value::I32(val as i32)
    };
    diff1(&s, &[arg0, Value::I32(amt)])
}

#[test]
fn diff_lane_shifts() {
    // i32x4: shl/shr_s/shr_u, incl. amount ≥ lane width (mod 32) and a negative (sign) value.
    for amt in [0, 1, 3, 31, 33, 40] {
        let m = amt & 31;
        assert_eq!(
            vshift("i32x4", "shl", "extract_lane 0", 0x1234_5678, amt) as i32,
            0x1234_5678i32.wrapping_shl(m as u32),
            "i32 shl {amt}"
        );
        assert_eq!(
            vshift("i32x4", "shr_u", "extract_lane 0", -1, amt) as i32,
            (0xFFFF_FFFFu32 >> m) as i32,
            "i32 shr_u {amt}"
        );
        assert_eq!(
            vshift("i32x4", "shr_s", "extract_lane 0", -8, amt) as i32,
            (-8i32) >> m,
            "i32 shr_s {amt}"
        );
    }
    // i16x8 / i8x16: observe via signed extract; mod 16 / mod 8.
    assert_eq!(
        vshift("i16x8", "shr_s", "extract_lane_s 0", 0xFFF0, 4) as i32,
        ((0xFFF0u16 as i16) >> 4) as i32,
        "i16 shr_s 4"
    );
    assert_eq!(
        vshift("i16x8", "shr_u", "extract_lane_s 0", 0xFF00, 4) as i32,
        ((0xFF00u16 >> 4) as i16) as i32,
        "i16 shr_u 4"
    );
    assert_eq!(
        vshift("i8x16", "shl", "extract_lane_s 0", 0x03, 2) as i32,
        ((0x03u8 << 2) as i8) as i32,
        "i8 shl 2"
    );
    assert_eq!(
        vshift("i8x16", "shr_s", "extract_lane_s 0", 0x80, 1) as i32,
        ((0x80u8 as i8) >> 1) as i32,
        "i8 shr_s 1"
    );
    assert_eq!(
        vshift("i8x16", "shr_u", "extract_lane_s 0", 0x80, 1) as i32,
        ((0x80u8 >> 1) as i8) as i32,
        "i8 shr_u 1"
    );
    // i64x2.
    assert_eq!(
        vshift("i64x2", "shl", "extract_lane 0", 1, 40),
        1i64 << 40,
        "i64 shl 40"
    );
    assert_eq!(
        vshift("i64x2", "shr_s", "extract_lane 0", -256, 4),
        -256i64 >> 4,
        "i64 shr_s 4"
    );
}

// ---------------------------------------------------------------------------
// Integer lane abs/neg (VIntUn) — two's-complement, wrapping at INT_MIN.
// ---------------------------------------------------------------------------

#[test]
fn diff_lane_abs_neg() {
    let i32x4 = |op: &str, x: i32| -> i32 {
        let s = format!(
            "func (i32) -> (i32) {{\nblock0(v0: i32):\n\
             \x20 v1 = i32x4.splat v0\n  v2 = i32x4.{op} v1\n  v3 = i32x4.extract_lane 0 v2\n  return v3\n}}\n"
        );
        diff1(&s, &[Value::I32(x)]) as i32
    };
    for x in [0, 5, -5, i32::MIN, i32::MAX, -1] {
        assert_eq!(i32x4("abs", x), x.wrapping_abs(), "i32 abs {x}");
        assert_eq!(i32x4("neg", x), x.wrapping_neg(), "i32 neg {x}");
    }

    // i8x16: low byte, observed via signed extract. `abs(-128)` wraps to -128 (two's complement).
    let i8 = |op: &str, x: i32| -> i32 {
        let s = format!(
            "func (i32) -> (i32) {{\nblock0(v0: i32):\n\
             \x20 v1 = i8x16.splat v0\n  v2 = i8x16.{op} v1\n  v3 = i8x16.extract_lane_s 0 v2\n  return v3\n}}\n"
        );
        diff1(&s, &[Value::I32(x)]) as i32
    };
    assert_eq!(i8("abs", 0x80), -128, "i8 abs(-128) wraps");
    assert_eq!(i8("abs", 0xFB), 5, "i8 abs(-5)");
    assert_eq!(i8("neg", 5), -5, "i8 neg(5)");

    // i64x2 (also probes i64x2.abs JIT legalization — diff1 runs both backends).
    let i64 = |op: &str, x: i64| -> i64 {
        let s = format!(
            "func (i64) -> (i64) {{\nblock0(v0: i64):\n\
             \x20 v1 = i64x2.splat v0\n  v2 = i64x2.{op} v1\n  v3 = i64x2.extract_lane 0 v2\n  return v3\n}}\n"
        );
        diff1(&s, &[Value::I64(x)])
    };
    assert_eq!(i64("abs", -1000), 1000, "i64 abs(-1000)");
    assert_eq!(i64("abs", i64::MIN), i64::MIN, "i64 abs(MIN) wraps");
    assert_eq!(i64("neg", 7), -7, "i64 neg(7)");
}

// ---------------------------------------------------------------------------
// Boolean reductions (VAnyTrue / VAllTrue / VBitmask) — v128 → i32.
// ---------------------------------------------------------------------------

/// Build `() -> i32` from a body and run it on both backends (interp == JIT), returning the result.
fn reduce_i32(body: &str) -> i32 {
    diff1(
        &format!("func () -> (i32) {{\nblock0():\n{body}\n}}\n"),
        &[],
    ) as i32
}

#[test]
fn diff_bool_reductions() {
    // i32x4.bitmask: lanes 0 and 2 have the sign bit set (byte 3 = 0x80) → bits 0 and 2 → 0b0101.
    assert_eq!(
        reduce_i32("  v0 = v128.const 0 0 0 128 0 0 0 0 0 0 0 128 0 0 0 0\n  v1 = i32x4.bitmask v0\n  return v1\n"),
        0b0101
    );
    // i8x16.bitmask: high bit set at byte indices 0, 1, 7, 15.
    assert_eq!(
        reduce_i32("  v0 = v128.const 128 128 0 0 0 0 0 128 0 0 0 0 0 0 0 128\n  v1 = i8x16.bitmask v0\n  return v1\n"),
        (1 << 0) | (1 << 1) | (1 << 7) | (1 << 15)
    );
    // i32x4.all_true: every lane non-zero → 1; a zero lane → 0.
    assert_eq!(
        reduce_i32("  v0 = v128.const 1 0 0 0 2 0 0 0 3 0 0 0 4 0 0 0\n  v1 = i32x4.all_true v0\n  return v1\n"),
        1
    );
    assert_eq!(
        reduce_i32("  v0 = v128.const 1 0 0 0 0 0 0 0 3 0 0 0 4 0 0 0\n  v1 = i32x4.all_true v0\n  return v1\n"),
        0
    );
    // v128.any_true: any bit set → 1; all-zero → 0.
    assert_eq!(
        reduce_i32("  v0 = v128.const 0 0 0 0 0 0 0 0 0 0 0 0 7 0 0 0\n  v1 = v128.any_true v0\n  return v1\n"),
        1
    );
    assert_eq!(
        reduce_i32("  v0 = v128.const 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n  v1 = v128.any_true v0\n  return v1\n"),
        0
    );
    // i64x2.all_true / bitmask.
    assert_eq!(
        reduce_i32("  v0 = v128.const 0 0 0 0 0 0 0 128 0 0 0 0 0 0 0 0\n  v1 = i64x2.bitmask v0\n  return v1\n"),
        0b01
    );
}

/// `bitmask` round-trips through text + binary; `all_true`/`any_true` too.
#[test]
fn reduction_roundtrip() {
    let src = "func () -> (i32) {\nblock0():\n\
        v0 = v128.const 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16\n\
        v1 = v128.any_true v0\n\
        v2 = i8x16.all_true v0\n\
        v3 = i16x8.bitmask v0\n\
        v4 = i32x4.all_true v0\n\
        v5 = i64x2.bitmask v0\n\
        v6 = i32.add v1 v3\n  return v6\n}\n";
    let m = build(src);
    assert_eq!(
        parse_module(&print_module(&m)).expect("reparse"),
        m,
        "text round-trip"
    );
    assert_eq!(
        decode_module(&encode_module(&m)).expect("decode"),
        m,
        "binary round-trip"
    );
}

// ---------------------------------------------------------------------------
// Saturating add/sub (VSatBin) — i8x16/i16x8 only; clamps instead of wrapping.
// ---------------------------------------------------------------------------

#[test]
fn diff_saturating_add_sub() {
    // i8x16, signed result via extract_lane_s.
    let i8s = |op: &str, a: i32, b: i32| -> i32 {
        let s = format!(
            "func (i32, i32) -> (i32) {{\nblock0(v0: i32, v1: i32):\n\
             \x20 v2 = i8x16.splat v0\n  v3 = i8x16.splat v1\n  v4 = i8x16.{op} v2 v3\n\
             \x20 v5 = i8x16.extract_lane_s 0 v4\n  return v5\n}}\n"
        );
        diff1(&s, &[Value::I32(a), Value::I32(b)]) as i32
    };
    // i8x16, unsigned result via extract_lane_u.
    let i8u = |op: &str, a: i32, b: i32| -> i32 {
        let s = format!(
            "func (i32, i32) -> (i32) {{\nblock0(v0: i32, v1: i32):\n\
             \x20 v2 = i8x16.splat v0\n  v3 = i8x16.splat v1\n  v4 = i8x16.{op} v2 v3\n\
             \x20 v5 = i8x16.extract_lane_u 0 v4\n  return v5\n}}\n"
        );
        diff1(&s, &[Value::I32(a), Value::I32(b)]) as i32
    };
    for (a, b) in [
        (100, 50),
        (-100, -50),
        (127, 1),
        (-128, -1),
        (0, 0),
        (50, -50),
    ] {
        assert_eq!(
            i8s("add_sat_s", a, b),
            (a as i8).saturating_add(b as i8) as i32,
            "add_sat_s {a} {b}"
        );
        assert_eq!(
            i8s("sub_sat_s", a, b),
            (a as i8).saturating_sub(b as i8) as i32,
            "sub_sat_s {a} {b}"
        );
    }
    for (a, b) in [(200, 100), (50, 100), (255, 1), (0, 1), (128, 128)] {
        assert_eq!(
            i8u("add_sat_u", a, b),
            (a as u8).saturating_add(b as u8) as i32,
            "add_sat_u {a} {b}"
        );
        assert_eq!(
            i8u("sub_sat_u", a, b),
            (a as u8).saturating_sub(b as u8) as i32,
            "sub_sat_u {a} {b}"
        );
    }

    // i16x8 spot checks.
    let i16s = |op: &str, a: i32, b: i32| -> i32 {
        let s = format!(
            "func (i32, i32) -> (i32) {{\nblock0(v0: i32, v1: i32):\n\
             \x20 v2 = i16x8.splat v0\n  v3 = i16x8.splat v1\n  v4 = i16x8.{op} v2 v3\n\
             \x20 v5 = i16x8.extract_lane_s 0 v4\n  return v5\n}}\n"
        );
        diff1(&s, &[Value::I32(a), Value::I32(b)]) as i32
    };
    assert_eq!(
        i16s("add_sat_s", 30000, 5000),
        (30000i16).saturating_add(5000) as i32,
        "i16 add_sat_s"
    );
    assert_eq!(
        i16s("sub_sat_s", -30000, 5000),
        (-30000i16).saturating_sub(5000) as i32,
        "i16 sub_sat_s"
    );
}

/// Saturating ops round-trip through text + binary; the verifier rejects a wide shape.
#[test]
fn saturating_roundtrip_and_shape_reject() {
    let src = "func () -> (i32) {\nblock0():\n\
        v0 = v128.const 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16\n\
        v1 = i8x16.add_sat_s v0 v0\n\
        v2 = i8x16.sub_sat_u v0 v0\n\
        v3 = i16x8.add_sat_u v0 v0\n\
        v4 = i8x16.extract_lane_s 0 v1\n  return v4\n}\n";
    let m = build(src);
    assert_eq!(
        parse_module(&print_module(&m)).expect("reparse"),
        m,
        "text round-trip"
    );
    assert_eq!(
        decode_module(&encode_module(&m)).expect("decode"),
        m,
        "binary round-trip"
    );

    // i32x4 saturating add is not a wasm op and the verifier rejects it.
    let bad = parse_module(
        "func () -> () {\nblock0():\n\
         v0 = v128.const 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n\
         v1 = i32x4.add_sat_s v0 v0\n  return\n}\n",
    )
    .expect("parses (shape check is at verify)");
    assert!(
        verify_module(&bad).is_err(),
        "i32x4 saturating add must fail verification"
    );
}

// ---------------------------------------------------------------------------
// Lane widening / extend (VWiden) — low/high half, sign/zero-extend.
// ---------------------------------------------------------------------------

#[test]
fn diff_widen_extend() {
    // i16x8 ← i8x16. Source bytes: low half [128,1,..], high half [200,9,..].
    let src8 = "128 1 2 3 4 5 6 7 200 9 10 11 12 13 14 15";
    let w16 = |op: &str, ext: &str, lane: u8| -> i32 {
        let s = format!(
            "func () -> (i32) {{\nblock0():\n  v0 = v128.const {src8}\n\
             \x20 v1 = i16x8.{op} v0\n  v2 = i16x8.{ext} {lane} v1\n  return v2\n}}\n"
        );
        diff1(&s, &[]) as i32
    };
    assert_eq!(
        w16("extend_low_s", "extract_lane_s", 0),
        -128,
        "low_s lane0 = (i8)128 = -128"
    );
    assert_eq!(
        w16("extend_low_u", "extract_lane_u", 0),
        128,
        "low_u lane0 = (u8)128"
    );
    assert_eq!(w16("extend_low_s", "extract_lane_s", 1), 1, "low_s lane1");
    assert_eq!(
        w16("extend_high_s", "extract_lane_s", 0),
        -56,
        "high_s lane0 = (i8)200 = -56"
    );
    assert_eq!(
        w16("extend_high_u", "extract_lane_u", 0),
        200,
        "high_u lane0 = (u8)200"
    );

    // i32x4 ← i16x8. i16 lane0 = 0xFFFF (-1), lane1 = 1, lane4 (high half) = 7.
    let src16 = "255 255 1 0 2 0 3 0 7 0 8 0 9 0 10 0";
    let w32 = |op: &str, lane: u8| -> i32 {
        let s = format!(
            "func () -> (i32) {{\nblock0():\n  v0 = v128.const {src16}\n\
             \x20 v1 = i32x4.{op} v0\n  v2 = i32x4.extract_lane {lane} v1\n  return v2\n}}\n"
        );
        diff1(&s, &[]) as i32
    };
    assert_eq!(w32("extend_low_s", 0), -1, "i32 low_s lane0 = (i16)0xFFFF");
    assert_eq!(
        w32("extend_low_u", 0),
        0xFFFF,
        "i32 low_u lane0 = (u16)0xFFFF"
    );
    assert_eq!(w32("extend_high_s", 0), 7, "i32 high_s lane0 = i16 lane4");

    // i64x2 ← i32x4. i32 lane0 = 0xFFFFFFFF (-1), lane2 (high half) = 9.
    let src32 = "255 255 255 255 5 0 0 0 9 0 0 0 11 0 0 0";
    let w64 = |op: &str, lane: u8| -> i64 {
        let s = format!(
            "func () -> (i64) {{\nblock0():\n  v0 = v128.const {src32}\n\
             \x20 v1 = i64x2.{op} v0\n  v2 = i64x2.extract_lane {lane} v1\n  return v2\n}}\n"
        );
        diff1(&s, &[])
    };
    assert_eq!(w64("extend_low_s", 0), -1, "i64 low_s lane0");
    assert_eq!(w64("extend_low_u", 0), 0xFFFF_FFFF, "i64 low_u lane0");
    assert_eq!(w64("extend_high_s", 0), 9, "i64 high_s lane0 = i32 lane2");
}

/// Widen round-trips; the verifier rejects widening to `i8x16` (no narrower source).
#[test]
fn widen_roundtrip_and_shape_reject() {
    let src = "func () -> (i32) {\nblock0():\n\
        v0 = v128.const 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16\n\
        v1 = i16x8.extend_low_s v0\n\
        v2 = i32x4.extend_high_u v0\n\
        v3 = i64x2.extend_low_s v0\n\
        v4 = i16x8.extract_lane_s 0 v1\n  return v4\n}\n";
    let m = build(src);
    assert_eq!(
        parse_module(&print_module(&m)).expect("reparse"),
        m,
        "text round-trip"
    );
    assert_eq!(
        decode_module(&encode_module(&m)).expect("decode"),
        m,
        "binary round-trip"
    );

    let bad = parse_module(
        "func () -> () {\nblock0():\n\
         v0 = v128.const 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n\
         v1 = i8x16.extend_low_s v0\n  return\n}\n",
    )
    .expect("parses");
    assert!(
        verify_module(&bad).is_err(),
        "widening to i8x16 has no source — must reject"
    );
}

// ---------------------------------------------------------------------------
// Lane narrowing (VNarrow) — saturate two wide vectors into one narrow vector.
// ---------------------------------------------------------------------------

#[test]
fn diff_narrow() {
    // i8x16.narrow_i16x8: a lane0=300, lane1=-200; b lane0=5. Result = [sat(a..), sat(b..)].
    let a = "44 1 56 255 0 0 0 0 0 0 0 0 0 0 0 0"; // i16 [300, -200, 0, ...]
    let b = "5 0 6 0 0 0 0 0 0 0 0 0 0 0 0 0"; // i16 [5, 6, 0, ...]
    let n8 = |op: &str, ext: &str, lane: u8| -> i32 {
        let s = format!(
            "func () -> (i32) {{\nblock0():\n  v0 = v128.const {a}\n  v1 = v128.const {b}\n\
             \x20 v2 = i8x16.{op} v0 v1\n  v3 = i8x16.{ext} {lane} v2\n  return v3\n}}\n"
        );
        diff1(&s, &[]) as i32
    };
    assert_eq!(n8("narrow_s", "extract_lane_s", 0), 127, "300 →_s 127");
    assert_eq!(n8("narrow_s", "extract_lane_s", 1), -128, "-200 →_s -128");
    assert_eq!(
        n8("narrow_s", "extract_lane_s", 8),
        5,
        "b lane0 lands at result lane 8"
    );
    assert_eq!(n8("narrow_u", "extract_lane_u", 0), 255, "300 →_u 255");
    assert_eq!(n8("narrow_u", "extract_lane_u", 1), 0, "-200 →_u 0");

    // i16x8.narrow_i32x4: a lane0 = 100000 → sat_s 32767; narrow_u of -1 → 0.
    let a32 = "160 134 1 0 0 0 0 0 0 0 0 0 0 0 0 0"; // i32 [100000, 0, ...]
    let neg = "255 255 255 255 0 0 0 0 0 0 0 0 0 0 0 0"; // i32 [-1, 0, ...]
    let n16 = |aa: &str, op: &str, ext: &str| -> i32 {
        let s = format!(
            "func () -> (i32) {{\nblock0():\n  v0 = v128.const {aa}\n  v1 = v128.const {aa}\n\
             \x20 v2 = i16x8.{op} v0 v1\n  v3 = i16x8.{ext} 0 v2\n  return v3\n}}\n"
        );
        diff1(&s, &[]) as i32
    };
    assert_eq!(
        n16(a32, "narrow_s", "extract_lane_s"),
        32767,
        "100000 →_s i16 max"
    );
    assert_eq!(n16(neg, "narrow_u", "extract_lane_u"), 0, "-1 →_u 0");
}

/// Narrow round-trips; the verifier rejects narrowing to `i32x4`.
#[test]
fn narrow_roundtrip_and_shape_reject() {
    let src = "func () -> (i32) {\nblock0():\n\
        v0 = v128.const 1 0 2 0 3 0 4 0 5 0 6 0 7 0 8 0\n\
        v1 = i8x16.narrow_s v0 v0\n\
        v2 = i16x8.narrow_u v0 v0\n\
        v3 = i8x16.extract_lane_s 0 v1\n  return v3\n}\n";
    let m = build(src);
    assert_eq!(
        parse_module(&print_module(&m)).expect("reparse"),
        m,
        "text round-trip"
    );
    assert_eq!(
        decode_module(&encode_module(&m)).expect("decode"),
        m,
        "binary round-trip"
    );

    let bad = parse_module(
        "func () -> () {\nblock0():\n\
         v0 = v128.const 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n\
         v1 = i32x4.narrow_s v0 v0\n  return\n}\n",
    )
    .expect("parses");
    assert!(
        verify_module(&bad).is_err(),
        "narrowing to i32x4 is not a wasm op — must reject"
    );
}
