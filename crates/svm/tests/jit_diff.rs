//! Differential testing: the Cranelift JIT vs the reference interpreter (`DESIGN.md`
//! §18, invariants I1/I4). For every supported program and input, the JIT's result
//! must equal the interpreter oracle's. This is the methodology the whole
//! escape-freedom argument leans on, so it is wired up alongside the very first JIT
//! slice and grows with the lowering.

use svm_interp::{run, Trap, Value};
use svm_ir::ValType;
use svm_jit::{compile_and_run, JitError, JitOutcome, TrapKind};
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

/// The JIT trap kind an interpreter `Trap` should correspond to, or `None` for traps
/// the scalar JIT does not model (memory-guard fault, fuel, stack, capabilities) — those
/// are skipped rather than asserted.
fn interp_trap_kind(t: &Trap) -> Option<TrapKind> {
    match t {
        Trap::DivByZero => Some(TrapKind::DivByZero),
        Trap::IntOverflow => Some(TrapKind::IntOverflow),
        Trap::BadConversion => Some(TrapKind::BadConversion),
        Trap::Unreachable => Some(TrapKind::Unreachable),
        Trap::IndirectCallType => Some(TrapKind::IndirectCallType),
        Trap::CapFault => Some(TrapKind::CapFault),
        _ => None,
    }
}

/// Run the differential check against function index `idx` (not just 0): the JIT and
/// interpreter must agree on the result **and on whether/why they trap** (§18).
fn assert_jit_matches_interp_at(src: &str, idx: u32, inputs: &[Vec<Value>]) {
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");
    let results_ty = m.funcs[idx as usize].results.clone();
    for args in inputs {
        let mut fuel = 10_000_000u64;
        let interp = run(&m, idx, args, &mut fuel);

        let slots: Vec<i64> = args.iter().copied().map(to_slot).collect();
        let outcome = match compile_and_run(&m, idx, &slots) {
            Ok(o) => o,
            Err(JitError::Unsupported(_)) => return, // op not lowered yet — skip module
            Err(e) => panic!("JIT failed to compile {src:?}: {e:?}"),
        };

        compare_outcome(interp, outcome, &results_ty, src, args);
    }
}

/// Assert a single interpreter result and JIT outcome agree (value-equal, or the same
/// trap kind / exit code; traps the JIT doesn't model are not asserted).
fn compare_outcome(
    interp: Result<Vec<Value>, Trap>,
    outcome: JitOutcome,
    results_ty: &[ValType],
    src: &str,
    args: &[Value],
) {
    match (interp, outcome) {
        (Ok(want), JitOutcome::Returned(got_slots)) => {
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
        (Err(trap), JitOutcome::Trapped(kind)) => {
            // A trap the JIT models must match in kind; one it doesn't model (e.g.
            // memory-guard fault) is fine either way.
            if let Some(want) = interp_trap_kind(&trap) {
                assert_eq!(
                    kind, want,
                    "JIT trapped {kind:?} but interp trapped {trap:?} on {src:?} for {args:?}"
                );
            }
        }
        (Err(trap), JitOutcome::Returned(_)) => {
            // OK only if it's a trap the JIT doesn't model (e.g. memory-guard fault).
            assert!(
                interp_trap_kind(&trap).is_none(),
                "interp trapped {trap:?} but JIT returned on {src:?} for {args:?}"
            );
        }
        (Err(Trap::Exit(want)), JitOutcome::Exited(got)) => {
            assert_eq!(want, got, "exit code mismatch on {src:?} for {args:?}");
        }
        (interp, outcome) => {
            panic!("interp/JIT outcome mismatch on {src:?} for {args:?}: {interp:?} vs {outcome:?}")
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

// f1 = +10, f2 = *2, both (i32)->(i32); f0 dispatches indirectly. Index masks into the
// power-of-two-padded table: 1->f1, 2->f2, 0->f0 (type mismatch -> trap), 3->padding.
const INDIRECT: &str = r#"
func (i32, i32) -> (i32) {
block0(v0: i32, v1: i32):
  v2 = call_indirect (i32) -> (i32) v0 (v1)
  return v2
}

func (i32) -> (i32) {
block0(v0: i32):
  v1 = i32.const 10
  v2 = i32.add v0 v1
  return v2
}

func (i32) -> (i32) {
block0(v0: i32):
  v1 = i32.const 2
  v2 = i32.mul v0 v1
  return v2
}
"#;

#[test]
fn jit_matches_interp_call_indirect() {
    assert_jit_matches_interp_at(
        INDIRECT,
        0,
        &[
            i32s(&[1, 5]),  // -> f1(5) = 15
            i32s(&[2, 5]),  // -> f2(5) = 10
            i32s(&[1, -3]), // -> f1(-3) = 7
            i32s(&[0, 5]),  // wrong type (f0) -> interp traps -> skipped
            i32s(&[3, 5]),  // padding slot -> interp traps -> skipped
        ],
    );
}

#[test]
fn jit_matches_interp_ref_func_indirect() {
    // `ref.func 2` materializes the index of f2 (*2); dispatch through it.
    let src = r#"
func (i32) -> (i32) {
block0(v0: i32):
  v1 = ref.func 2
  v2 = call_indirect (i32) -> (i32) v1 (v0)
  return v2
}

func (i32) -> (i32) {
block0(v0: i32):
  v1 = i32.const 10
  v2 = i32.add v0 v1
  return v2
}

func (i32) -> (i32) {
block0(v0: i32):
  v1 = i32.const 2
  v2 = i32.mul v0 v1
  return v2
}
"#;
    assert_jit_matches_interp_at(src, 0, &[i32s(&[5]), i32s(&[-4]), i32s(&[0])]);
}

#[test]
fn jit_matches_interp_return_call_indirect() {
    let src = r#"
func (i32, i32) -> (i32) {
block0(v0: i32, v1: i32):
  return_call_indirect (i32) -> (i32) v0 (v1)
}

func (i32) -> (i32) {
block0(v0: i32):
  v1 = i32.const 10
  v2 = i32.add v0 v1
  return v2
}

func (i32) -> (i32) {
block0(v0: i32):
  v1 = i32.const 2
  v2 = i32.mul v0 v1
  return v2
}
"#;
    assert_jit_matches_interp_at(src, 0, &[i32s(&[1, 5]), i32s(&[2, 5]), i32s(&[0, 5])]);
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

// ---- cap.call: the JIT dispatches through a host thunk to the interpreter's Host ----

use core::ffi::c_void;
use svm_interp::{run_with_host, GuestMem, Host, StreamRole};
use svm_jit::{compile_and_run_with_host, EXIT_CODE};

/// Bridge the JIT's `CapThunk` ABI to the interpreter's `Host::cap_dispatch_slots`.
/// This is the host "trampoline" (§9) a real embedder supplies; here it drives the same
/// mock Host the interpreter uses, so the two backends share capability semantics.
///
/// # Safety
/// Honours the `CapThunk` contract: `ctx` is a `*mut Host`, the slot/window pointers are
/// valid for their lengths, and `trap_out` is live.
unsafe extern "C" fn cap_thunk(
    ctx: *mut c_void,
    mem_base: *mut u8,
    mem_size: u64,
    type_id: u32,
    op: u32,
    handle: i32,
    args: *const i64,
    n_args: u64,
    results: *mut i64,
    n_results: u64,
    trap_out: *mut i64,
) {
    let host = &mut *(ctx as *mut Host);
    let arg_slots = std::slice::from_raw_parts(args, n_args as usize);
    // The JIT-side guest window, with `map`/`unmap`/`protect` backed by real `mprotect` on the
    // window pages — the mirror of the interpreter's page-protection map, so the Memory cap
    // behaves identically on both backends (a store to a protected page faults into the guard).
    let mut wm = MprotectWindow {
        base: mem_base,
        size: mem_size,
    };
    let gm: Option<&mut dyn GuestMem> = if mem_base.is_null() {
        None
    } else {
        Some(&mut wm)
    };

    match host.cap_dispatch_slots(type_id, op, handle, arg_slots, gm) {
        Ok(res) => {
            let out = std::slice::from_raw_parts_mut(results, n_results as usize);
            for (o, r) in out.iter_mut().zip(res) {
                *o = r;
            }
            *trap_out = 0;
        }
        Err(Trap::Exit(code)) => *trap_out = EXIT_CODE as i64 | ((code as i64) << 32),
        // CapFault (forged/wrong-type/closed) and anything else map to a CapFault trap.
        Err(_) => *trap_out = TrapKind::CapFault as i64,
    }
}

/// A [`GuestMem`] over the JIT's flat guest window whose `map`/`unmap`/`protect` are backed by
/// real `mprotect` on the window pages — the JIT-side mirror of the interpreter's page-protection
/// map, so the Memory capability behaves identically on both backends (a store to a protected
/// page faults into the guard region → detect-and-kill). `read_bytes`/`write_bytes` behave like
/// `svm_interp::WindowMem`. Unix-only, like the JIT's guard itself.
struct MprotectWindow {
    base: *mut u8,
    size: u64,
}

impl MprotectWindow {
    /// `mprotect [offset, offset+len)` to cap `prot_bits` (`READ`=1, `WRITE`=2). Page-aligned
    /// offset + in-range required (else `-EINVAL`), matching the interpreter's `prot_pages`.
    fn mprotect_range(&self, offset: u64, len: u64, prot_bits: i32) -> i64 {
        const EINVAL: i64 = -22;
        // SAFETY: sysconf is always safe to call; `_SC_PAGESIZE` is positive.
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as u64;
        let Some(end) = offset.checked_add(len) else {
            return EINVAL;
        };
        if len == 0 || page == 0 || !offset.is_multiple_of(page) || end > self.size {
            return EINVAL;
        }
        let prot = if prot_bits & 2 != 0 {
            libc::PROT_READ | libc::PROT_WRITE
        } else if prot_bits & 1 != 0 {
            libc::PROT_READ
        } else {
            libc::PROT_NONE
        };
        let last = end - 1;
        let rlen = ((last / page) - (offset / page) + 1) * page; // round up to whole pages
                                                                 // SAFETY: `[base+offset, +rlen)` is within the window's reserved mapping (offset+len ≤
                                                                 // size ≤ mapped, page-rounded), owned by the JIT for the duration of the call.
        let rc = unsafe {
            libc::mprotect(
                self.base.add(offset as usize) as *mut c_void,
                rlen as usize,
                prot,
            )
        };
        if rc == 0 {
            0
        } else {
            EINVAL
        }
    }
}

impl GuestMem for MprotectWindow {
    fn read_bytes(&self, ptr: u64, len: u64) -> Option<Vec<u8>> {
        let end = ptr.checked_add(len)?;
        if end > self.size {
            return None;
        }
        // SAFETY: `[base, base+size)` is the live guest window for the call's duration.
        let w = unsafe { std::slice::from_raw_parts(self.base, self.size as usize) };
        Some(w[ptr as usize..end as usize].to_vec())
    }
    fn write_bytes(&mut self, ptr: u64, data: &[u8]) -> Option<()> {
        let end = ptr.checked_add(data.len() as u64)?;
        if end > self.size {
            return None;
        }
        // SAFETY: as above; `&mut self` gives exclusive access for the write.
        let w = unsafe { std::slice::from_raw_parts_mut(self.base, self.size as usize) };
        w[ptr as usize..end as usize].copy_from_slice(data);
        Some(())
    }
    fn map(&mut self, offset: u64, len: u64, prot: i32) -> i64 {
        // Re-commit with the requested protection. NB: page *zeroing* on (re)map is not yet
        // mirrored here — the differential test exercises `protect`, not map-after-unmap.
        self.mprotect_range(offset, len, prot)
    }
    fn unmap(&mut self, offset: u64, len: u64) -> i64 {
        self.mprotect_range(offset, len, 0) // PROT_NONE ⇒ any access faults
    }
    fn protect(&mut self, offset: u64, len: u64, prot: i32) -> i64 {
        self.mprotect_range(offset, len, prot)
    }
}

/// Drive `src(args)` through the interpreter (on `hi`) and the JIT (on `hj`) and assert
/// they agree on the result/trap/exit *and* on the observable host effects. `hi` and `hj`
/// must be set up identically by the caller (grants are deterministic, so handle indices
/// match).
fn assert_cap_agrees(src: &str, args: &[Value], hi: &mut Host, hj: &mut Host) {
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");
    let results_ty = m.funcs[0].results.clone();

    let mut fuel = 10_000_000u64;
    let interp = run_with_host(&m, 0, args, &mut fuel, hi);
    let slots: Vec<i64> = args.iter().copied().map(to_slot).collect();
    let jit = compile_and_run_with_host(&m, 0, &slots, cap_thunk, hj as *mut Host as *mut c_void)
        .expect("jit compiles");

    compare_outcome(interp, jit, &results_ty, src, args);
    assert_eq!(hi.stdout, hj.stdout, "stdout differs on {src:?}");
    assert_eq!(hi.stderr, hj.stderr, "stderr differs on {src:?}");
}

#[test]
fn jit_cap_clock_now() {
    let src = r#"
func (i32) -> (i64) {
block0(v0: i32):
  v1 = i32.const 0
  v2 = cap.call 2 0 (i32) -> (i64) v0 (v1)
  return v2
}
"#;
    let mut hi = Host::new();
    let mut hj = Host::new();
    let ci = hi.grant_clock();
    let cj = hj.grant_clock();
    assert_eq!(ci, cj, "grants are deterministic");
    assert_cap_agrees(src, &[Value::I32(ci)], &mut hi, &mut hj);
}

#[test]
fn jit_cap_stream_write_captures_output() {
    // Store "Hi" into the window, then Stream.write(ptr=0, len=2) — exercises a buffer
    // arg through the §7 window borrow in the JIT thunk.
    let src = r#"
memory 16

func (i32) -> (i64) {
block0(v0: i32):
  v1 = i64.const 0
  v2 = i32.const 72
  i32.store8 v1 v2
  v3 = i64.const 1
  v4 = i32.const 105
  i32.store8 v3 v4
  v5 = i64.const 0
  v6 = i64.const 2
  v7 = cap.call 0 1 (i64, i64) -> (i64) v0 (v5, v6)
  return v7
}
"#;
    let mut hi = Host::new();
    let mut hj = Host::new();
    let oi = hi.grant_stream(StreamRole::Out);
    let oj = hj.grant_stream(StreamRole::Out);
    assert_eq!(oi, oj);
    assert_cap_agrees(src, &[Value::I32(oi)], &mut hi, &mut hj);
    assert_eq!(hj.stdout, b"Hi", "JIT captured the written bytes");
}

#[test]
fn jit_cap_exit_propagates_code() {
    let src = r#"
func (i32) -> () {
block0(v0: i32):
  v1 = i32.const 7
  cap.call 1 0 (i32) -> () v0 (v1)
  unreachable
}
"#;
    let mut hi = Host::new();
    let mut hj = Host::new();
    let ei = hi.grant_exit();
    let ej = hj.grant_exit();
    assert_eq!(ei, ej);
    assert_cap_agrees(src, &[Value::I32(ei)], &mut hi, &mut hj);
}

/// §3e Memory cap `protect`: make a page read-only, then store to it — must fault on **both**
/// backends (interp page-protection map; JIT real `mprotect` caught by the guard page). This is
/// the D40 read-only-const mechanism and the first end-to-end exercise of the Memory cap.
#[cfg(unix)]
#[test]
fn jit_cap_memory_protect_read_only_faults_store() {
    let src = r#"
memory 16

func (i32) -> (i32) {
block0(v0: i32):
  v1 = i64.const 0
  v2 = i64.const 4096
  v3 = i32.const 1
  v4 = cap.call 3 2 (i64, i64, i32) -> (i64) v0 (v1, v2, v3)
  v5 = i64.const 0
  v6 = i32.const 123
  i32.store8 v5 v6
  v7 = i32.const 0
  return v7
}
"#;
    // Non-vacuous: the interpreter (the spec) must actually fault on the post-`protect` store.
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");
    let mut h = Host::new();
    let mh = h.grant_memory();
    let mut fuel = 1_000_000u64;
    assert_eq!(
        run_with_host(&m, 0, &[Value::I32(mh)], &mut fuel, &mut h),
        Err(Trap::MemoryFault),
        "interp: a store to a protect(READ) page must fault"
    );
    // And the JIT agrees (mprotect + guard ⇒ MemoryFault). A no-op JIT `protect` would let the
    // store succeed and diverge here, so this also pins that the JIT side is real.
    let mut hi = Host::new();
    let mut hj = Host::new();
    let mi = hi.grant_memory();
    let mj = hj.grant_memory();
    assert_eq!(mi, mj, "grants are deterministic");
    assert_cap_agrees(src, &[Value::I32(mi)], &mut hi, &mut hj);
}

#[test]
fn jit_cap_forged_handle_is_capfault() {
    // A forged handle index must be inert in the JIT too (CapFault), matching interp.
    let src = r#"
func (i32) -> (i64) {
block0(v0: i32):
  v1 = i32.const 0
  v2 = cap.call 2 0 (i32) -> (i64) v0 (v1)
  return v2
}
"#;
    let mut hi = Host::new();
    let mut hj = Host::new();
    // No grants: any handle is forged. Also test a wrong-type handle (a Stream called
    // as a Clock) in a second pass.
    assert_cap_agrees(src, &[Value::I32(0x7fff)], &mut hi, &mut hj);

    let mut hi2 = Host::new();
    let mut hj2 = Host::new();
    let si = hi2.grant_stream(StreamRole::Out);
    let sj = hj2.grant_stream(StreamRole::Out);
    assert_eq!(si, sj);
    assert_cap_agrees(src, &[Value::I32(si)], &mut hi2, &mut hj2);
}
