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

/// §3a/D40 read-only data segment: the runtime copies the bytes in and maps the segment RO at
/// instantiation, so a guest **store** to it faults (detect-and-kill) while a **load** reads the
/// initialized byte — identically on the interpreter (page-protection map) and the JIT (real
/// `mprotect`, fault caught by the guard).
#[cfg(unix)]
#[test]
fn data_readonly_segment_write_faults_load_reads() {
    // memory 13 = 8 KiB (2 pages); a RO segment starts at page 1 (offset 4096).
    let store = "data ro 4096 \"\\xab\\xcd\"\nmemory 13\n\
        func () -> (i32) {\nblock0():\n  v0 = i64.const 4096\n  v1 = i32.const 1\n  \
        i32.store8 v0 v1\n  v2 = i32.const 0\n  return v2\n}\n";
    let m = parse_module(store).expect("parse");
    verify_module(&m).expect("verify");
    let mut fuel = 100_000u64;
    assert_eq!(
        run(&m, 0, &[], &mut fuel),
        Err(Trap::MemoryFault),
        "interp: store to RO data must fault"
    );
    assert!(
        matches!(
            compile_and_run(&m, 0, &[]).expect("jit"),
            JitOutcome::Trapped(TrapKind::MemoryFault)
        ),
        "jit: store to RO data must detect-and-kill"
    );

    // A load of the same RO byte succeeds and reads the initialized value (0xab) on both.
    let load = "data ro 4096 \"\\xab\\xcd\"\nmemory 13\n\
        func () -> (i32) {\nblock0():\n  v0 = i64.const 4096\n  v1 = i32.load8_u v0\n  return v1\n}\n";
    let m = parse_module(load).expect("parse");
    verify_module(&m).expect("verify");
    let mut fuel = 100_000u64;
    assert_eq!(run(&m, 0, &[], &mut fuel), Ok(vec![Value::I32(0xab)]));
    assert!(matches!(
        compile_and_run(&m, 0, &[]).expect("jit"),
        JitOutcome::Returned(ref s) if s == &[0xab]
    ));
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

// ---- §12 atomics: interp (reference) == JIT (hardware atomics) ----

#[test]
fn jit_matches_interp_atomic_rmw() {
    // Each RMW op at i32 and i64: seed mem[8], rmw it with the arg, return old + new — pinning both
    // the returned *old* value and the stored *new* value against the interpreter.
    let mk_i32 = (|x: i64| Value::I32(x as i32)) as fn(i64) -> Value;
    let mk_i64 = (|x: i64| Value::I64(x)) as fn(i64) -> Value;
    for (ty, mk) in [("i32", mk_i32), ("i64", mk_i64)] {
        for op in ["add", "sub", "and", "or", "xor", "xchg"] {
            let src = format!(
                "memory 16\n\
                 func ({ty}, {ty}) -> ({ty}) {{\n\
                 block0(v0: {ty}, v1: {ty}):\n\
                 \x20 v2 = i64.const 8\n\
                 \x20 {ty}.atomic.store v2 v0\n\
                 \x20 v3 = {ty}.atomic.rmw.{op} v2 v1\n\
                 \x20 v4 = {ty}.atomic.load v2\n\
                 \x20 v5 = {ty}.add v3 v4\n\
                 \x20 return v5\n\
                 }}\n"
            );
            assert_jit_matches_interp(
                &src,
                &[
                    vec![mk(0x12), mk(0x34)],
                    vec![mk(-1), mk(7)],
                    vec![mk(0x0F0F_0F0F), mk(0x00FF_00FF)],
                ],
            );
        }
    }
}

#[test]
fn jit_matches_interp_atomic_cmpxchg() {
    // Compare-exchange: on a match the replacement is stored; on a mismatch memory is unchanged;
    // either way the *old* value is returned. Return old + new to pin both.
    let src = "memory 16\n\
        func (i64, i64, i64) -> (i64) {\n\
        block0(v0: i64, v1: i64, v2: i64):\n\
        \x20 v3 = i64.const 16\n\
        \x20 i64.atomic.store v3 v0\n\
        \x20 v4 = i64.atomic.cmpxchg v3 v1 v2\n\
        \x20 v5 = i64.atomic.load v3\n\
        \x20 v6 = i64.add v4 v5\n\
        \x20 return v6\n\
        }\n";
    assert_jit_matches_interp(
        src,
        &[
            vec![Value::I64(5), Value::I64(5), Value::I64(99)], // match → writes 99
            vec![Value::I64(5), Value::I64(6), Value::I64(99)], // mismatch → unchanged
            vec![Value::I64(-1), Value::I64(-1), Value::I64(0)],
        ],
    );
}

#[test]
fn jit_matches_interp_atomic_aliases_plain_memory() {
    // Atomics and plain loads/stores are the *same* linear memory: an atomic store is seen by a
    // plain load, and an atomic load sees a plain store.
    let src = "memory 16\n\
        func (i64) -> (i64) {\n\
        block0(v0: i64):\n\
        \x20 v1 = i64.const 24\n\
        \x20 i64.atomic.store v1 v0\n\
        \x20 v2 = i64.load v1\n\
        \x20 i64.store v1 v2\n\
        \x20 v3 = i64.atomic.load v1\n\
        \x20 return v3\n\
        }\n";
    assert_jit_matches_interp(src, &[vec![Value::I64(0xDEAD_BEEF)], vec![Value::I64(-5)]]);
}

#[test]
fn jit_atomic_unaligned_traps_both() {
    // A misaligned atomic effective address traps (MemoryFault) identically on both backends — the
    // §12 natural-alignment requirement (offset 4 is 4- but not 8-aligned ⇒ an i64 atomic traps).
    // The JIT trap is the software alignment guard (not the hardware guard page), so it is portable.
    let src = "memory 16\n\
        func () -> (i64) {\n\
        block0():\n\
        \x20 v0 = i64.const 4\n\
        \x20 v1 = i64.const 1\n\
        \x20 v2 = i64.atomic.rmw.add v0 v1\n\
        \x20 return v2\n\
        }\n";
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");
    let mut fuel = 100_000u64;
    assert_eq!(
        run(&m, 0, &[], &mut fuel),
        Err(Trap::MemoryFault),
        "interp: unaligned atomic must trap"
    );
    assert!(
        matches!(
            compile_and_run(&m, 0, &[]).expect("jit"),
            JitOutcome::Trapped(TrapKind::MemoryFault)
        ),
        "jit: unaligned atomic must detect-and-kill"
    );
}

#[test]
fn jit_matches_interp_orderings_and_fence() {
    // The C11 ordering suffixes and `atomic.fence` lower on the JIT (all seq-cst, a sound
    // strengthening) and match the interpreter exactly — release-store 5, acquire-load it, fence,
    // relaxed rmw +3, then read 8; returns 5 + 8 = 13.
    let src = r#"
memory 16
func () -> (i64) {
block0():
  v0 = i64.const 0
  v1 = i64.const 5
  i64.atomic.store.release v0 v1
  v2 = i64.atomic.load.acquire v0
  atomic.fence
  atomic.fence.acquire
  v3 = i64.const 3
  v4 = i64.atomic.rmw.add.relaxed v0 v3
  v5 = i64.atomic.load v0
  v6 = i64.add v2 v5
  return v6
}
"#;
    assert_jit_matches_interp(src, &[vec![]]);
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

/// §12 fibers are **platform-gated** in the JIT: where the `svm-fiber` stack-switch substrate exists
/// the JIT lowers `cont.*` to its host fiber runtime (the happy path is covered by the interp↔JIT
/// differential in `jit_fibers.rs`); everywhere else it must cleanly **bail** `Unsupported` so the
/// differential harness skips fiber modules rather than mis-compiling them. The expectation is the JIT's
/// own `fiber_supported()` (the `fiber_rt` cfg) — the single source of truth, not a re-derived target set.
#[test]
fn jit_fiber_support_is_platform_gated() {
    let supported = svm_jit::fiber_supported();
    let srcs = [
        // cont.new + cont.resume
        "func () -> (i64) {\n\
         block0():\n\
         \x20 v0 = ref.func 1\n\
         \x20 v1 = i64.const 4096\n\
         \x20 v2 = cont.new v0 v1\n\
         \x20 v3 = i64.const 0\n\
         \x20 v4, v5 = cont.resume v2 v3\n\
         \x20 return v5\n\
         }\n\
         func (i64, i64) -> (i64) {\nblock0(v0: i64, v1: i64):\n  return v1\n}\n",
        // suspend
        "func (i64, i64) -> (i64) {\n\
         block0(v0: i64, v1: i64):\n\
         \x20 v2 = suspend v1\n\
         \x20 return v2\n\
         }\n",
    ];
    for src in srcs {
        let m = parse_module(src).expect("parse");
        verify_module(&m).expect("verify");
        let bailed = matches!(compile_and_run(&m, 0, &[0]), Err(JitError::Unsupported(_)));
        assert_eq!(
            bailed, !supported,
            "fiber lowering gating mismatch (supported={supported}): {src:?}"
        );
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

/// Escape-oracle over the **§1a growth path**, deterministic and cross-platform: grow a *reserved-
/// tail* page (above the backed prefix) via the Memory cap, store a known marker into it, and assert
/// both backends leave a byte-identical window whose tail page actually holds the marker. The
/// `_with_host` capture now snapshots the low 256 KiB (`SNAP_CAP`), tail included, so a grown page
/// that the JIT mis-grew / mis-masked / failed to commit-for-snapshot would diverge here. Unlike the
/// random `gen_memory_program` (whose completing runs rarely leave non-zero *tail* content — so its
/// tail comparison is largely vacuous), this pins the path non-vacuously, on unix **and** windows
/// (the JIT + Memory cap exist on both). `OFF` is a multiple of 16 KiB, so the single mapped page is
/// page-aligned on 4 KiB and 16 KiB hosts alike.
#[cfg(any(unix, windows))]
#[test]
fn jit_cap_memory_escape_oracle_grown_tail() {
    use core::ffi::c_void;
    use svm_interp::{run_capture_reserved_with_host, Host, Value};
    use svm_jit::compile_and_run_capture_reserved_with_host;

    const OFF: u64 = 80 * 1024; // 81920 = 5 * 16 KiB: in the tail (prefix is 64 KiB), page-aligned
    const MARKER: i64 = 0x0123_4567_89ab_cdefu64 as i64;
    let src = format!(
        "memory 16\n\
         func (i32) -> (i64) {{\n\
         block0(v0: i32):\n\
         \x20 v1 = i64.const {OFF}\n\
         \x20 v2 = i64.const 4096\n\
         \x20 v3 = i32.const 3\n\
         \x20 v4 = cap.call 3 0 (i64, i64, i32) -> (i64) v0 (v1, v2, v3)\n\
         \x20 v5 = i64.const {OFF}\n\
         \x20 v6 = i64.const {MARKER}\n\
         \x20 i64.store v5 v6\n\
         \x20 v7 = i64.load v5\n\
         \x20 return v7\n\
         }}\n"
    );
    let m = parse_module(&src).expect("parse");
    verify_module(&m).expect("verify");

    let mut hi = Host::new();
    let mut hj = Host::new();
    let mi = hi.grant_memory();
    let mj = hj.grant_memory();
    assert_eq!(mi, mj, "grants are deterministic");
    let init = vec![0u8; 1 << 16]; // 64 KiB prefix seed; the tail marker is what this test pins
    let mut fuel = 1_000_000u64;
    // reserved_log2 = 18 (256 KiB) ⇒ a reserved tail above the 64 KiB prefix; OFF is in it.
    let (interp, imem) =
        run_capture_reserved_with_host(&m, 0, &[Value::I32(mi)], &mut fuel, &init, 18, &mut hi);
    let (jit, jmem) = compile_and_run_capture_reserved_with_host(
        &m,
        0,
        &[mj as i64],
        &init,
        18,
        svm_run::cap_thunk,
        &mut hj as *mut Host as *mut c_void,
    )
    .expect("jit compiles");

    // Both grow the page and load the marker back.
    assert_eq!(
        interp.expect("interp run"),
        vec![Value::I64(MARKER)],
        "interp must load the marker from the grown tail page"
    );
    assert!(
        matches!(jit, JitOutcome::Returned(ref s) if s == &[MARKER]),
        "jit must load the marker from the grown tail page: {jit:?}"
    );
    // The windows must agree, the snapshot must reach the tail, and it must hold the marker there.
    assert_eq!(imem.len(), jmem.len(), "snapshot length differs");
    assert!(
        imem.len() as u64 >= OFF + 8,
        "snapshot ({}) must cover the grown tail page at {OFF}",
        imem.len()
    );
    assert_eq!(
        imem, jmem,
        "escape-oracle: interp/JIT grown-tail windows diverge"
    );
    assert_eq!(
        &imem[OFF as usize..OFF as usize + 8],
        &MARKER.to_le_bytes(),
        "grown tail page must hold the stored marker (non-vacuous)"
    );
}

/// §13 SharedRegion differential (all platforms): one region mapped at *two* window offsets must
/// alias on **both** backends — the interpreter (`VecBacking`, software aliasing) and the JIT (a
/// real OS shared section mapped at both offsets through `svm_run::cap_thunk`). A store at offset 0
/// is read back at the region `page_size` (the second mapping), and the final windows are
/// byte-identical. The guest queries the region's `page_size`, so it is granularity-agnostic: 4 KiB /
/// 16 KiB on unix (Linux `memfd` / macOS `shm` + `mmap(MAP_SHARED | MAP_FIXED)`), the 64 KiB
/// allocation granularity on Windows (`CreateFileMapping` + `MapViewOfFile3` over a placeholder
/// reservation). `memory 17` (128 KiB) so two whole granules fit even at 64 KiB. (Top-level, unlike
/// the broad `mod cap` Memory-cap differential which is still unix-gated — this uses the production
/// `cap_thunk`, so it runs on every platform.)
#[cfg(any(unix, windows))]
#[test]
fn jit_cap_shared_region_aliases_differential() {
    use core::ffi::c_void;
    use svm_interp::{run_capture_reserved_with_host, Host, Value};
    use svm_jit::compile_and_run_capture_reserved_with_host;

    const MARKER: i64 = 0x0123_4567_89ab_cdefu64 as i64;
    let src = format!(
        "memory 17\n\
         func (i32) -> (i64) {{\n\
         block0(v0: i32):\n\
         \x20 v1 = cap.call 4 3 () -> (i64) v0 ()\n\
         \x20 v2 = i64.const 0\n\
         \x20 v3 = i32.const 3\n\
         \x20 v4 = cap.call 4 0 (i64, i64, i64, i32) -> (i64) v0 (v2, v2, v1, v3)\n\
         \x20 v5 = cap.call 4 0 (i64, i64, i64, i32) -> (i64) v0 (v1, v2, v1, v3)\n\
         \x20 v6 = i64.const {MARKER}\n\
         \x20 i64.store v2 v6\n\
         \x20 v7 = i64.load v1\n\
         \x20 return v7\n\
         }}\n"
    );
    let m = parse_module(&src).expect("parse");
    verify_module(&m).expect("verify");

    // Region ≥ the largest granule (64 KiB on Windows), mapped whole at both offsets.
    let region_len = 1usize << 16;
    let mut hi = Host::new();
    let mut hj = Host::new();
    let ri = hi.grant_shared_region(region_len); // interp: pure-Rust VecBacking
    let rj = hj.grant_shared_region_backed(svm_run::new_shared_region(region_len)); // JIT: OS section
    assert_eq!(ri, rj, "grants are deterministic");

    let init = vec![0u8; 1 << 17];
    let mut fuel = 1_000_000u64;
    let (interp, imem) =
        run_capture_reserved_with_host(&m, 0, &[Value::I32(ri)], &mut fuel, &init, 0, &mut hi);
    let (jit, jmem) = compile_and_run_capture_reserved_with_host(
        &m,
        0,
        &[rj as i64],
        &init,
        0,
        svm_run::cap_thunk,
        &mut hj as *mut Host as *mut c_void,
    )
    .expect("jit compiles");

    assert_eq!(
        interp.expect("interp runs"),
        vec![Value::I64(MARKER)],
        "interp: store at 0 must be visible at the page_size alias"
    );
    assert!(
        matches!(jit, JitOutcome::Returned(ref s) if s == &[MARKER]),
        "jit: store at 0 must be visible at the page_size alias: {jit:?}"
    );
    assert_eq!(imem, jmem, "interp/JIT shared-region windows diverge");
    assert_eq!(
        &imem[0..8],
        &MARKER.to_le_bytes(),
        "the marker must be present at window offset 0 (non-vacuous)"
    );
}

// ---- cap.call: the JIT dispatches through a host thunk to the interpreter's Host ----
//
// This module's broad Memory-cap differential is still unix-gated; the cross-platform `_with_host`
// capture path (incl. the windows `VirtualProtect` Memory cap) is covered by `jit_fuzz` and by
// `jit_cap_memory_escape_oracle_grown_tail` above. Re-gating the whole module to windows is a
// follow-up. Gated as a module so windows CI still compiles + runs the rest of the suite.
#[cfg(unix)]
mod cap {
    use super::*;
    use core::ffi::c_void;
    use svm_interp::{
        run_capture_reserved_with_host, run_with_host, GuestMem, Host, StreamRole, Trap, Value,
    };
    use svm_jit::{
        compile_and_run_capture_reserved_with_host, compile_and_run_with_host, TrapKind, EXIT_CODE,
    };
    use svm_run::MprotectWindow;
    use svm_text::parse_module;
    use svm_verify::verify_module;

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
        mem_reserved: u64,
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
        // Null args pointer when there are 0 args; `from_raw_parts` needs non-null even for len 0.
        let arg_slots = if n_args == 0 {
            &[][..]
        } else {
            std::slice::from_raw_parts(args, n_args as usize)
        };
        // The production JIT-side guest window: `map`/`unmap`/`protect` backed by real `mprotect`
        // (incl. growth into the reserved tail), mirrored by a software page map so the Memory cap is
        // bit-identical to the interpreter's paged `Mem` — exactly what this differential checks.
        let mut wm = MprotectWindow::new(mem_base, mem_size, mem_reserved);
        let gm: Option<&mut dyn GuestMem> = if mem_base.is_null() {
            None
        } else {
            Some(&mut wm)
        };

        match host.cap_dispatch_slots(type_id, op, handle, arg_slots, gm) {
            Ok(res) => {
                if n_results != 0 {
                    let out = std::slice::from_raw_parts_mut(results, n_results as usize);
                    for (o, r) in out.iter_mut().zip(res) {
                        *o = r;
                    }
                }
                *trap_out = 0;
            }
            Err(Trap::Exit(code)) => *trap_out = EXIT_CODE as i64 | ((code as i64) << 32),
            // CapFault (forged/wrong-type/closed) and anything else map to a CapFault trap.
            Err(_) => *trap_out = TrapKind::CapFault as i64,
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
        let jit =
            compile_and_run_with_host(&m, 0, &slots, cap_thunk, hj as *mut Host as *mut c_void)
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

    /// Regression (trap propagation across calls): a trap raised in a **callee** must unwind the
    /// whole guest stack, not just the callee's own frame. A helper here `cap.call`s with a wrong
    /// `type_id` on the granted Memory handle (→ `CapFault`); the entry then makes a *successful*
    /// `page_size` `cap.call`, which resets the host trap cell. Before the JIT re-checked the cell
    /// after a `call`, the callee's trap merely returned zeros, the entry ran on, `page_size` reset
    /// the cell, and the JIT **returned** where the interpreter (correctly) stays trapped — which
    /// would let a guest neutralize *any* trap (even `exit`) by wrapping it in a function call. Both
    /// backends must detect-and-kill.
    #[cfg(unix)]
    #[test]
    fn jit_trap_in_callee_propagates_through_caller() {
        let src = r#"
memory 16

func (i32) -> (i32) {
block0(v0: i32):
  v1 = call 1(v0)
  v2 = cap.call 3 3 () -> (i64) v0 ()
  return v1
}

func (i32) -> (i32) {
block0(v0: i32):
  v1 = cap.call 99 0 () -> (i64) v0 ()
  v2 = i32.wrap_i64 v1
  return v2
}
"#;
        // Non-vacuous: the interpreter (the spec) stays trapped *through* the caller.
        let m = parse_module(src).expect("parse");
        verify_module(&m).expect("verify");
        let mut h = Host::new();
        let mh = h.grant_memory();
        let mut fuel = 1_000_000u64;
        assert_eq!(
            run_with_host(&m, 0, &[Value::I32(mh)], &mut fuel, &mut h),
            Err(Trap::CapFault),
            "interp: a callee's CapFault must propagate through the caller"
        );
        let mut hi = Host::new();
        let mut hj = Host::new();
        let mi = hi.grant_memory();
        let mj = hj.grant_memory();
        assert_eq!(mi, mj, "grants are deterministic");
        assert_cap_agrees(src, &[Value::I32(mi)], &mut hi, &mut hj);
    }

    /// Generate a random straight-line program that drives the `Memory` capability (handle in `v0`):
    /// a mix of `protect`/`unmap`/`map` on whole pages, interleaved with 8-byte stores and loads
    /// (loads accumulate into the returned sum). Page selectors span **0..32** while the window's
    /// backed prefix is only 16 pages (`memory 16`, 64 KiB) inside the large `DEFAULT_RESERVED_LOG2`
    /// reservation — so pages 16..32 are the **reserved tail**: a store/load there faults until a
    /// `map` *grows* into it, and an `unmap` decommits it again. Page-aligned, in-range args, so every
    /// op succeeds or faults deterministically and identically on both backends (the §1a growth path).
    /// Returns IR text. Deterministic in `seed` (SplitMix64).
    #[cfg(unix)]
    fn gen_memory_program(seed: u64) -> String {
        let mut state = seed;
        let mut next = move || {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        let mut body = String::new();
        let mut nv = 1u32; // v0 is the Memory handle (entry param); defined values start at v1
        let mut fresh = |body: &mut String, line: String| {
            let v = nv;
            nv += 1;
            body.push_str(&format!("  v{v} = {line}\n"));
            v
        };
        // Pages 0..16 are the backed prefix (`memory 16`); 16..32 are the reserved tail reachable
        // only after a `map` grows into them. Addresses span both so the tail's fault/grow path runs.
        const PAGES: u64 = 32;
        const SPAN: u64 = PAGES * 4096;
        let acc0 = fresh(&mut body, "i64.const 0".into()); // running sum of loads
        let mut acc = acc0;
        let nops = 3 + next() % 8; // 3..=10 ops
        for _ in 0..nops {
            match next() % 5 {
                0 | 1 => {
                    // protect(page, prot) — prot 0 (none/unmapped), 1 (RO), 3 (RW)
                    let page = next() % PAGES;
                    let prot = [0i32, 1, 3][(next() % 3) as usize];
                    let off = fresh(&mut body, format!("i64.const {}", page * 4096));
                    let len = fresh(&mut body, "i64.const 4096".into());
                    let pr = fresh(&mut body, format!("i32.const {prot}"));
                    fresh(
                        &mut body,
                        format!("cap.call 3 2 (i64, i64, i32) -> (i64) v0 (v{off}, v{len}, v{pr})"),
                    );
                }
                2 => {
                    // unmap(page)
                    let page = next() % PAGES;
                    let off = fresh(&mut body, format!("i64.const {}", page * 4096));
                    let len = fresh(&mut body, "i64.const 4096".into());
                    fresh(
                        &mut body,
                        format!("cap.call 3 1 (i64, i64) -> (i64) v0 (v{off}, v{len})"),
                    );
                }
                3 => {
                    // map(page, prot) — (re)commit readable (RO or RW); grows the tail when page ≥ 16
                    let page = next() % PAGES;
                    let prot = [1i32, 3][(next() % 2) as usize];
                    let off = fresh(&mut body, format!("i64.const {}", page * 4096));
                    let len = fresh(&mut body, "i64.const 4096".into());
                    let pr = fresh(&mut body, format!("i32.const {prot}"));
                    fresh(
                        &mut body,
                        format!("cap.call 3 0 (i64, i64, i32) -> (i64) v0 (v{off}, v{len}, v{pr})"),
                    );
                }
                4 => {
                    // store an 8-byte value at an aligned address (may land in the unmapped tail)
                    let addr = (next() % SPAN) & !7;
                    let val = next() as i64;
                    let a = fresh(&mut body, format!("i64.const {addr}"));
                    let v = fresh(&mut body, format!("i64.const {val}"));
                    body.push_str(&format!("  i64.store v{a} v{v}\n"));
                }
                _ => {
                    // load + accumulate
                    let addr = (next() % SPAN) & !7;
                    let a = fresh(&mut body, format!("i64.const {addr}"));
                    let ld = fresh(&mut body, format!("i64.load v{a}"));
                    acc = fresh(&mut body, format!("i64.add v{acc} v{ld}"));
                }
            }
        }
        format!("memory 16\nfunc (i32) -> (i64) {{\nblock0(v0: i32):\n{body}  return v{acc}\n}}\n")
    }

    /// Generative differential coverage of the `Memory` capability **including growth**: random
    /// `map`/`unmap`/`protect` sequences interleaved with stores/loads, with page selectors spanning
    /// the backed prefix *and* the reserved tail (see `gen_memory_program`), must produce the **same**
    /// result/trap on the interpreter (page-protection map) and the JIT (real `mprotect`, faults
    /// caught by the guard). A protected/unmapped/never-grown page makes a store or load fault on
    /// both; a `map` that grows into the tail makes it accessible on both; a re-`map` zero-fills.
    #[cfg(unix)]
    #[test]
    fn jit_cap_memory_protect_map_unmap_differential() {
        for seed in 0..800u64 {
            let src = gen_memory_program(seed);
            let m =
                parse_module(&src).unwrap_or_else(|e| panic!("parse seed {seed}: {e:?}\n{src}"));
            verify_module(&m).unwrap_or_else(|e| panic!("verify seed {seed}: {e:?}\n{src}"));
            let mut hi = Host::new();
            let mut hj = Host::new();
            let mi = hi.grant_memory();
            let mj = hj.grant_memory();
            assert_eq!(mi, mj);
            assert_cap_agrees(&src, &[Value::I32(mi)], &mut hi, &mut hj);
        }
    }

    /// A concrete guest consumer of **growth** (§1a sparse address space): a `memory 16` program
    /// (64 KiB backed) `map`s a page deep in the reserved tail at offset 1 MiB, stores a value there,
    /// reads it back, and returns it — then a second variant `unmap`s it and faults on the next load.
    /// Both the value round-trip and the post-`unmap` fault must agree on interp + JIT, proving the
    /// reserved tail is genuinely reachable after a grow and genuinely gone after a decommit. This is
    /// the end-to-end "a guest grows its own address space" path, not just a unit of the page map.
    #[cfg(unix)]
    #[test]
    fn jit_cap_memory_growth_round_trips() {
        // map(0x100000, 4096, RW); store 0xABCD at 0x100000; load it back; return it.
        let grow_and_read = r#"
memory 16
func (i32) -> (i64) {
block0(v0: i32):
  v1 = i64.const 1048576
  v2 = i64.const 4096
  v3 = i32.const 3
  v4 = cap.call 3 0 (i64, i64, i32) -> (i64) v0 (v1, v2, v3)
  v5 = i64.const 1048576
  v6 = i64.const 43981
  i64.store v5 v6
  v7 = i64.load v5
  return v7
}
"#;
        // Non-vacuous: the interpreter (the spec) returns the grown-page value, not a fault.
        let m = parse_module(grow_and_read).expect("parse");
        verify_module(&m).expect("verify");
        let mut hcheck = Host::new();
        let mc = hcheck.grant_memory();
        let mut fuel = 1_000_000u64;
        assert_eq!(
            run_with_host(&m, 0, &[Value::I32(mc)], &mut fuel, &mut hcheck),
            Ok(vec![Value::I64(43981)]),
            "interp: a load from a freshly-grown tail page must read back the stored value"
        );
        let mut hi = Host::new();
        let mut hj = Host::new();
        let mi = hi.grant_memory();
        let mj = hj.grant_memory();
        assert_eq!(mi, mj);
        assert_cap_agrees(grow_and_read, &[Value::I32(mi)], &mut hi, &mut hj);

        // map then unmap the tail page, then load it → MemoryFault on both backends.
        let grow_then_unmap = r#"
memory 16
func (i32) -> (i64) {
block0(v0: i32):
  v1 = i64.const 1048576
  v2 = i64.const 4096
  v3 = i32.const 3
  v4 = cap.call 3 0 (i64, i64, i32) -> (i64) v0 (v1, v2, v3)
  v5 = cap.call 3 1 (i64, i64) -> (i64) v0 (v1, v2)
  v6 = i64.load v1
  return v6
}
"#;
        let mut hi2 = Host::new();
        let mut hj2 = Host::new();
        let mi2 = hi2.grant_memory();
        let mj2 = hj2.grant_memory();
        assert_eq!(mi2, mj2);
        assert_cap_agrees(grow_then_unmap, &[Value::I32(mi2)], &mut hi2, &mut hj2);
    }

    /// **Escape-oracle for the Memory capability** (§18): run the same generated `map`/`unmap`/
    /// `protect` + store/load programs through the *capture + host* path and byte-compare the final
    /// guest window across the interpreter (page-protection map) and the JIT (real `mprotect` +
    /// guard). The outcome differential above checks return values / traps; this also checks the
    /// **window itself** is identical after the cap's success-path effects — a JIT store/protect/unmap
    /// landing on the wrong page (an escape) would diverge here, as would a snapshot/`restore_rw` bug.
    /// Fully-mapped (`reserved == mapped`, `reserved_log2 = 0` is raised to `size_log2`), so the whole
    /// 64 KiB window is comparable; the generator's tail ops/addresses become in-range no-ops/wraps,
    /// agreeing on both sides.
    #[cfg(unix)]
    #[test]
    fn jit_cap_memory_escape_oracle_differential() {
        // 300 seeds (each a JIT compile + a 64 KiB window snapshot/compare) keeps the stable suite
        // snappy; the 800-seed outcome differential above already covers the same generator cheaply.
        for seed in 0..300u64 {
            let src = gen_memory_program(seed);
            let m =
                parse_module(&src).unwrap_or_else(|e| panic!("parse seed {seed}: {e:?}\n{src}"));
            verify_module(&m).unwrap_or_else(|e| panic!("verify seed {seed}: {e:?}\n{src}"));
            let results_ty = m.funcs[0].results.clone();
            // A varied non-zero window seed, so a divergent (e.g. mis-masked) *read* also shows up.
            let init: Vec<u8> = (0..1usize << 16)
                .map(|i| (i as u8).wrapping_mul(31) ^ 0xa5)
                .collect();
            let mut hi = Host::new();
            let mut hj = Host::new();
            let mi = hi.grant_memory();
            let mj = hj.grant_memory();
            assert_eq!(mi, mj, "grants are deterministic");
            let args = [Value::I32(mi)];
            let mut fuel = 10_000_000u64;
            let (interp, imem) =
                run_capture_reserved_with_host(&m, 0, &args, &mut fuel, &init, 0, &mut hi);
            let (jit, jmem) = compile_and_run_capture_reserved_with_host(
                &m,
                0,
                &[mi as i64],
                &init,
                0,
                cap_thunk,
                &mut hj as *mut Host as *mut c_void,
            )
            .expect("jit compiles");
            compare_outcome(interp, jit, &results_ty, &src, &args);
            assert_eq!(imem.len(), jmem.len(), "window length differ (seed {seed})");
            if let Some(i) = imem.iter().zip(&jmem).position(|(a, b)| a != b) {
                panic!(
                    "escape-oracle: interp/JIT final window differ at byte {i} \
                 (interp={:#04x} jit={:#04x}) on seed {seed}\n{src}",
                    imem[i], jmem[i]
                );
            }
        }
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
}
