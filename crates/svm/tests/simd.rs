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
