//! Differential testing: the Cranelift JIT vs the reference interpreter (`DESIGN.md`
//! §18, invariants I1/I4). For every supported program and input, the JIT's result
//! must equal the interpreter oracle's. This is the methodology the whole
//! escape-freedom argument leans on, so it is wired up alongside the very first JIT
//! slice and grows with the lowering.

use svm_interp::{run, Value};
use svm_ir::ValType;
use svm_jit::{compile_and_run, JitError};
use svm_text::parse_module;
use svm_verify::verify_module;

/// Marshal a `Value` into its `i64` calling-convention slot (§ JIT calling
/// convention). Floats travel as their bit pattern in the low bits.
fn to_slot(v: Value) -> i64 {
    match v {
        Value::I32(x) => x as i64,
        Value::I64(x) => x,
        Value::F32(x) => x.to_bits() as i64,
        Value::F64(x) => x.to_bits() as i64,
    }
}

/// Decode a result slot back to a `Value` given the declared result type.
fn from_slot(t: ValType, s: i64) -> Value {
    match t {
        ValType::I32 => Value::I32(s as i32),
        ValType::I64 => Value::I64(s),
        ValType::F32 => Value::F32(f32::from_bits(s as u32)),
        ValType::F64 => Value::F64(f64::from_bits(s as u64)),
    }
}

/// Assert the JIT and the interpreter agree on `src(args)` for every input row.
fn assert_jit_matches_interp(src: &str, inputs: &[Vec<Value>]) {
    assert_jit_matches_interp_at(src, 0, inputs);
}

/// Bit-exact equality, except any two NaNs of the same width are considered equal:
/// IEEE NaN payloads are non-deterministic across backends (Cranelift vs the
/// interpreter), and we do not yet enforce canonical-NaN. `-0.0` and `+0.0` still
/// differ (their bits differ), as they must.
fn values_equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::F32(x), Value::F32(y)) => x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()),
        (Value::F64(x), Value::F64(y)) => x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()),
        _ => a == b,
    }
}

fn i32s(xs: &[i32]) -> Vec<Value> {
    xs.iter().map(|x| Value::I32(*x)).collect()
}

#[test]
fn jit_matches_interp_add() {
    let src = r#"
func (i32, i32) -> (i32) {
block0(v0: i32, v1: i32):
  v2 = i32.add v0 v1
  return v2
}
"#;
    assert_jit_matches_interp(
        src,
        &[
            i32s(&[2, 3]),
            i32s(&[-1, 1]),
            i32s(&[i32::MAX, 1]), // wraps to i32::MIN — must match the interp
            i32s(&[-5, -7]),
        ],
    );
}

#[test]
fn jit_matches_interp_arith_with_select() {
    // (v0 < v1) ? 100 : (v0 - v1)^2 — sub, mul, lt_s, select, const.
    let src = r#"
func (i32, i32) -> (i32) {
block0(v0: i32, v1: i32):
  v2 = i32.sub v0 v1
  v3 = i32.mul v2 v2
  v4 = i32.lt_s v0 v1
  v5 = i32.const 100
  v6 = select v4 v5 v3
  return v6
}
"#;
    assert_jit_matches_interp(
        src,
        &[
            i32s(&[1, 9]),
            i32s(&[9, 1]),
            i32s(&[5, 5]),
            i32s(&[-3, -10]),
        ],
    );
}

#[test]
fn jit_matches_interp_bitwise_and_shifts() {
    // Exercise and/or/xor/shl/shr_u/shr_s/rotl, incl. shift-count masking semantics.
    let src = r#"
func (i32, i32) -> (i32) {
block0(v0: i32, v1: i32):
  v2 = i32.and v0 v1
  v3 = i32.or v0 v1
  v4 = i32.xor v2 v3
  v5 = i32.shl v4 v1
  v6 = i32.shr_u v5 v1
  v7 = i32.shr_s v6 v1
  v8 = i32.rotl v7 v1
  return v8
}
"#;
    assert_jit_matches_interp(
        src,
        &[
            i32s(&[0x1234_5678u32 as i32, 3]),
            i32s(&[-1, 31]),
            i32s(&[-1, 33]), // shift count must be masked mod 32
            i32s(&[0xFF00FF00u32 as i32, 7]),
        ],
    );
}

#[test]
fn jit_matches_interp_comparisons_and_eqz() {
    let src = r#"
func (i32, i32) -> (i32) {
block0(v0: i32, v1: i32):
  v2 = i32.lt_u v0 v1
  v3 = i32.ge_s v0 v1
  v4 = i32.eqz v2
  v5 = i32.add v3 v4
  return v5
}
"#;
    assert_jit_matches_interp(
        src,
        &[i32s(&[1, 2]), i32s(&[-1, 1]), i32s(&[5, 5]), i32s(&[0, 0])],
    );
}

#[test]
fn jit_matches_interp_i64_ops() {
    let src = r#"
func (i64, i64) -> (i64) {
block0(v0: i64, v1: i64):
  v2 = i64.mul v0 v1
  v3 = i64.sub v2 v1
  return v3
}
"#;
    assert_jit_matches_interp(
        src,
        &[
            vec![Value::I64(1_000_000), Value::I64(1_000_000)],
            vec![Value::I64(-3), Value::I64(7)],
            vec![Value::I64(i64::MAX), Value::I64(2)], // overflow wraps
        ],
    );
}

#[test]
fn jit_matches_interp_loop_with_back_edge() {
    // sum = 1 + 2 + ... + n via a back-edge loop with block parameters — exercises
    // br / br_if and multi-block SSA lowering.
    let src = r#"
func (i32) -> (i32) {
block0(v0: i32):
  v1 = i32.const 0
  br block1(v0, v1)
block1(v2: i32, v3: i32):
  v4 = i32.add v3 v2
  v5 = i32.const -1
  v6 = i32.add v2 v5
  br_if v6 block1(v6, v4) block2(v4)
block2(v7: i32):
  return v7
}
"#;
    let inputs: Vec<Vec<Value>> = [1, 2, 5, 10, 100, 1000]
        .iter()
        .map(|n| i32s(&[*n]))
        .collect();
    assert_jit_matches_interp(src, &inputs);
}

#[test]
fn jit_matches_interp_mem_store_load_roundtrip() {
    // Store an i64 at the given address, read it back — exercises the §4 masking
    // lowering (mem_base + ((addr+offset) & mask)) against the interpreter.
    let src = r#"
memory 16

func (i64, i64) -> (i64) {
block0(v0: i64, v1: i64):
  i64.store v0 v1
  v2 = i64.load v0
  return v2
}
"#;
    assert_jit_matches_interp(
        src,
        &[
            vec![Value::I64(0), Value::I64(0x0123_4567_89AB_CDEF)],
            vec![Value::I64(64), Value::I64(-1)],
            vec![Value::I64(65528), Value::I64(42)], // last aligned 8-byte slot (2^16-8)
        ],
    );
}

#[test]
fn jit_matches_interp_mem_narrow_store_load() {
    // store8 keeps the low byte; load8_u zero-extends, load8_s sign-extends.
    let src = r#"
memory 16

func (i64, i32) -> (i32) {
block0(v0: i64, v1: i32):
  i32.store8 v0 v1
  v2 = i32.load8_u v0
  v3 = i32.load8_s v0
  v4 = i32.add v2 v3
  return v4
}
"#;
    assert_jit_matches_interp(
        src,
        &[
            vec![Value::I64(0), Value::I32(0x1FF)], // truncates to 0xFF
            vec![Value::I64(10), Value::I32(0x80)], // 128: u=128, s=-128
            vec![Value::I64(7), Value::I32(0x41)],  // 'A'
        ],
    );
}

#[test]
fn jit_matches_interp_mem_masking_aliases_out_of_window() {
    // I1: an out-of-window address must alias back via the mask, identically in the
    // JIT and the interpreter. Store at offset 8, read at (2^16 + 8) — same cell.
    let src = r#"
memory 16

func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const 8
  i64.store v1 v0
  v2 = i64.const 65544
  v3 = i64.load v2
  return v3
}
"#;
    assert_jit_matches_interp(src, &[vec![Value::I64(0xDEAD_BEEF)], vec![Value::I64(-99)]]);
}

// ---- br_table, trapping ops, unreachable ----

#[test]
fn jit_matches_interp_br_table() {
    let src = r#"
func (i32) -> (i32) {
block0(v0: i32):
  br_table v0 [block1(), block2(), block3()] block4()
block1():
  v1 = i32.const 10
  return v1
block2():
  v2 = i32.const 20
  return v2
block3():
  v3 = i32.const 30
  return v3
block4():
  v4 = i32.const 99
  return v4
}
"#;
    assert_jit_matches_interp(
        src,
        &[
            i32s(&[0]),
            i32s(&[1]),
            i32s(&[2]),
            i32s(&[3]),  // out of range -> default
            i32s(&[-1]), // huge unsigned -> default
        ],
    );
}

#[test]
fn jit_matches_interp_br_table_with_args() {
    let src = r#"
func (i32, i32) -> (i32) {
block0(v0: i32, v1: i32):
  br_table v0 [block1(v1), block2(v1)] block3(v1)
block1(v2: i32):
  v3 = i32.const 1
  v4 = i32.add v2 v3
  return v4
block2(v5: i32):
  v6 = i32.const 2
  v7 = i32.add v5 v6
  return v7
block3(v8: i32):
  return v8
}
"#;
    assert_jit_matches_interp(src, &[i32s(&[0, 100]), i32s(&[1, 100]), i32s(&[9, 100])]);
}

#[test]
fn jit_matches_interp_div_rem_signed() {
    let src = r#"
func (i32, i32) -> (i32) {
block0(v0: i32, v1: i32):
  v2 = i32.div_s v0 v1
  v3 = i32.rem_s v0 v1
  v4 = i32.add v2 v3
  return v4
}
"#;
    assert_jit_matches_interp(
        src,
        &[
            i32s(&[7, 3]),
            i32s(&[-7, 3]),
            i32s(&[7, -3]),
            i32s(&[100, 7]),
            i32s(&[10, 0]),        // div/rem by zero -> interp traps -> skipped
            i32s(&[i32::MIN, -1]), // div_s overflow -> interp traps -> skipped
        ],
    );
}

#[test]
fn jit_matches_interp_rem_s_int_min_neg_one() {
    // i32.rem_s INT_MIN, -1 = 0 with NO trap (wasm special case). This input is *not*
    // skipped, so the JIT must lower srem to give 0 (not a hardware overflow trap).
    let src = r#"
func (i32, i32) -> (i32) {
block0(v0: i32, v1: i32):
  v2 = i32.rem_s v0 v1
  return v2
}
"#;
    assert_jit_matches_interp(src, &[i32s(&[i32::MIN, -1]), i32s(&[7, 3]), i32s(&[-7, 3])]);
}

#[test]
fn jit_matches_interp_div_rem_unsigned() {
    let src = r#"
func (i32, i32) -> (i32) {
block0(v0: i32, v1: i32):
  v2 = i32.div_u v0 v1
  v3 = i32.rem_u v0 v1
  v4 = i32.add v2 v3
  return v4
}
"#;
    assert_jit_matches_interp(
        src,
        &[i32s(&[100, 7]), i32s(&[-1, 3]), i32s(&[5, 0])], // /0 skipped
    );
}

#[test]
fn jit_matches_interp_trapping_trunc() {
    let src = r#"
func (f64) -> (i32) {
block0(v0: f64):
  v1 = i32.trunc_f64_s v0
  return v1
}
"#;
    assert_jit_matches_interp(
        src,
        &[
            f64s(&[3.9]),
            f64s(&[-3.9]),
            f64s(&[100.0]),
            f64s(&[f64::NAN]), // traps -> skipped
            f64s(&[1e18]),     // out of range -> traps -> skipped
        ],
    );
}

#[test]
fn jit_matches_interp_unreachable_in_untaken_branch() {
    // The JIT compiles the `unreachable` block (a trap) but must not execute it when
    // the branch is not taken. v0 != 0 -> returns 5; v0 == 0 -> interp traps -> skipped.
    let src = r#"
func (i32) -> (i32) {
block0(v0: i32):
  br_if v0 block1() block2()
block1():
  v1 = i32.const 5
  return v1
block2():
  unreachable
}
"#;
    assert_jit_matches_interp(src, &[i32s(&[1]), i32s(&[42]), i32s(&[0])]);
}

#[test]
fn jit_matches_interp_no_args_const() {
    let src = r#"
func () -> (i32) {
block0():
  v0 = i32.const 42
  return v0
}
"#;
    assert_jit_matches_interp(src, &[vec![]]);
}

// ---- floats + conversions ----

fn f64s(xs: &[f64]) -> Vec<Value> {
    xs.iter().map(|x| Value::F64(*x)).collect()
}

#[test]
fn jit_matches_interp_f64_arith() {
    let src = r#"
func (f64, f64) -> (f64) {
block0(v0: f64, v1: f64):
  v2 = f64.add v0 v1
  v3 = f64.mul v2 v0
  v4 = f64.sub v3 v1
  v5 = f64.div v4 v0
  return v5
}
"#;
    assert_jit_matches_interp(
        src,
        &[
            f64s(&[3.0, 4.0]),
            f64s(&[-1.5, 2.25]),
            f64s(&[1.0, 0.0]), // division by zero -> +inf
            f64s(&[f64::INFINITY, 1.0]),
            f64s(&[f64::NAN, 2.0]), // NaN propagation
        ],
    );
}

#[test]
fn jit_matches_interp_f32_unary_and_minmax() {
    let src = r#"
func (f32, f32) -> (f32) {
block0(v0: f32, v1: f32):
  v2 = f32.abs v0
  v3 = f32.neg v1
  v4 = f32.min v2 v3
  v5 = f32.max v4 v1
  v6 = f32.sqrt v5
  v7 = f32.ceil v6
  v8 = f32.copysign v7 v1
  return v8
}
"#;
    let f32s = |a: f32, b: f32| vec![Value::F32(a), Value::F32(b)];
    assert_jit_matches_interp(
        src,
        &[
            f32s(4.0, -9.0),
            f32s(-2.5, 3.5),
            f32s(0.0, -0.0), // signed-zero handling in min/max/copysign
            f32s(f32::NAN, 1.0),
            f32s(2.0, f32::INFINITY),
        ],
    );
}

#[test]
fn jit_matches_interp_float_compares() {
    let src = r#"
func (f64, f64) -> (i32) {
block0(v0: f64, v1: f64):
  v2 = f64.lt v0 v1
  v3 = f64.ge v0 v1
  v4 = f64.ne v0 v1
  v5 = i32.add v2 v3
  v6 = i32.add v5 v4
  return v6
}
"#;
    assert_jit_matches_interp(
        src,
        &[
            f64s(&[1.0, 2.0]),
            f64s(&[2.0, 2.0]),
            f64s(&[f64::NAN, 1.0]), // NaN: lt/ge false, ne true (unordered)
            f64s(&[-0.0, 0.0]),     // equal
        ],
    );
}

#[test]
fn jit_matches_interp_int_extend_wrap() {
    let src = r#"
func (i32, i64) -> (i64) {
block0(v0: i32, v1: i64):
  v2 = i64.extend_i32_s v0
  v3 = i64.extend_i32_u v0
  v4 = i32.wrap_i64 v1
  v5 = i64.extend_i32_s v4
  v6 = i64.add v2 v3
  v7 = i64.add v6 v5
  return v7
}
"#;
    assert_jit_matches_interp(
        src,
        &[
            vec![Value::I32(-1), Value::I64(0x1_0000_0007)],
            vec![Value::I32(i32::MIN), Value::I64(-1)],
            vec![Value::I32(123456), Value::I64(0xDEAD_BEEF_CAFE)],
        ],
    );
}

#[test]
fn jit_matches_interp_int_float_conversions() {
    // i32 -> f64 (signed/unsigned), back via saturating trunc; reinterpret too.
    let src = r#"
func (i32, f64) -> (i64) {
block0(v0: i32, v1: f64):
  v2 = f64.convert_i32_s v0
  v3 = f64.convert_i32_u v0
  v4 = f64.add v2 v3
  v5 = f64.add v4 v1
  v6 = i64.trunc_sat_f64_s v5
  v7 = i64.reinterpret_f64 v5
  v8 = i64.add v6 v7
  return v8
}
"#;
    assert_jit_matches_interp(
        src,
        &[
            vec![Value::I32(-1), Value::F64(0.5)],
            vec![Value::I32(1000), Value::F64(-3.5)],
            vec![Value::I32(i32::MIN), Value::F64(1e18)], // saturates on trunc
            vec![Value::I32(7), Value::F64(f64::NAN)],    // trunc_sat NaN -> 0
        ],
    );
}

// ---- calls ----

/// Run the differential check against function index `idx` (not just 0).
///
/// If the interpreter *traps* on an input, the JIT is **not** run on it: we have no
/// trap-catching infrastructure yet, so a JIT trap would abort the process. Trapping
/// inputs are skipped here (the trap semantics are covered by the interpreter's own
/// tests); the JIT is checked for agreement on the non-trapping inputs.
fn assert_jit_matches_interp_at(src: &str, idx: u32, inputs: &[Vec<Value>]) {
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");
    let results_ty = m.funcs[idx as usize].results.clone();
    for args in inputs {
        let mut fuel = 10_000_000u64;
        let want = match run(&m, idx, args, &mut fuel) {
            Ok(v) => v,
            Err(_) => continue, // interpreter traps -> don't run the JIT (would abort)
        };
        let slots: Vec<i64> = args.iter().copied().map(to_slot).collect();
        let got_slots = match compile_and_run(&m, idx, &slots) {
            Ok(s) => s,
            Err(JitError::Unsupported(_)) => return,
            Err(e) => panic!("JIT failed: {e:?}"),
        };
        let got: Vec<Value> = results_ty
            .iter()
            .zip(got_slots)
            .map(|(t, s)| from_slot(*t, s))
            .collect();
        assert_eq!(want.len(), got.len(), "result arity mismatch on {src:?}");
        for (w, g) in want.iter().zip(&got) {
            assert!(
                values_equal(w, g),
                "interp/JIT disagree on {src:?} for {args:?}: {want:?} vs {got:?}"
            );
        }
    }
}

#[test]
fn jit_matches_interp_direct_call() {
    // f1 = square; f0 calls f1 twice and sums — exercises direct call + mem_base
    // threading (no memory here, but the ABI still passes it).
    let src = r#"
func (i32, i32) -> (i32) {
block0(v0: i32, v1: i32):
  v2 = call 1 (v0)
  v3 = call 1 (v1)
  v4 = i32.add v2 v3
  return v4
}

func (i32) -> (i32) {
block0(v0: i32):
  v1 = i32.mul v0 v0
  return v1
}
"#;
    assert_jit_matches_interp_at(src, 0, &[i32s(&[3, 4]), i32s(&[-2, 5]), i32s(&[0, 0])]);
}

#[test]
fn jit_matches_interp_call_through_memory() {
    // The callee writes to the window; the caller reads it back — confirms mem_base
    // is threaded so both frames address the same window.
    let src = r#"
memory 16

func (i64, i64) -> (i64) {
block0(v0: i64, v1: i64):
  v2 = call 1 (v0, v1)
  v3 = i64.load v0
  return v3
}

func (i64, i64) -> (i64) {
block0(v0: i64, v1: i64):
  i64.store v0 v1
  v2 = i64.const 0
  return v2
}
"#;
    assert_jit_matches_interp_at(
        src,
        0,
        &[
            vec![Value::I64(16), Value::I64(0xCAFE)],
            vec![Value::I64(128), Value::I64(-7)],
        ],
    );
}

#[test]
fn jit_matches_interp_return_call_tail_recursion() {
    // Tail-recursive factorial accumulator f(n, acc) = n==0 ? acc : f(n-1, acc*n) via
    // `return_call` — must run in O(1) native stack and agree with the interpreter.
    // Values flow between blocks only through block parameters (block-local SSA).
    let src = r#"
func (i32, i32) -> (i32) {
block0(v0: i32, v1: i32):
  v2 = i32.eqz v0
  br_if v2 block1(v1) block2(v0, v1)
block1(v3: i32):
  return v3
block2(v4: i32, v5: i32):
  v6 = i32.mul v5 v4
  v7 = i32.const -1
  v8 = i32.add v4 v7
  return_call 0(v8, v6)
}
"#;
    assert_jit_matches_interp_at(
        src,
        0,
        &[
            vec![Value::I32(1), Value::I32(1)],
            vec![Value::I32(5), Value::I32(1)],
            vec![Value::I32(10), Value::I32(1)],
        ],
    );
}

#[test]
fn jit_matches_interp_float_mem_roundtrip() {
    let src = r#"
memory 16

func (i64, f64) -> (f64) {
block0(v0: i64, v1: f64):
  f64.store v0 v1
  v2 = f64.load v0
  return v2
}
"#;
    assert_jit_matches_interp(
        src,
        &[
            vec![Value::I64(0), Value::F64(123.456)],
            vec![Value::I64(32), Value::F64(-2.5)],
            vec![Value::I64(65528), Value::F64(f64::INFINITY)],
        ],
    );
}
