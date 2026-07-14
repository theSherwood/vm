//! SPEC.md suite 1 — per-op semantic vectors. For every op row in the executable spec
//! (`svm-spec`), run its boundary-value input vectors on **all three backends** — the
//! tree-walk interpreter, the bytecode interpreter, and the Cranelift JIT — and assert
//! each matches the spec's `eval` expectation (value bit-exact, or the same trap kind).
//!
//! This is deliberately stronger than the interp↔JIT differential (`jit_fuzz.rs`):
//! there the backends are checked against *each other*; here all three are checked
//! against an independent *definition* (written from the DESIGN.md §3b prose), so a
//! shared misreading of the prose cannot hide.
//!
//! NaN policy (§3b): NaN bit patterns are host-defined in default mode, so a NaN
//! expectation asserts "is a NaN" only; everything else compares bit-exact.

use svm_interp::{bytecode, run, Trap, Value};
use svm_jit::{compile, CompiledModule, JitOutcome, TrapKind};
use svm_spec::{module_for, scalar_rows, vectors_for, OpRow, Shape, SpecTrap, SpecVal};

fn to_value(v: SpecVal) -> Value {
    match v {
        SpecVal::I32(x) => Value::I32(x),
        SpecVal::I64(x) => Value::I64(x),
        SpecVal::F32(b) => Value::F32(f32::from_bits(b)),
        SpecVal::F64(b) => Value::F64(f64::from_bits(b)),
    }
}

/// The JIT trampoline ABI: one i64 slot per arg, floats as raw bits (zero-extended for
/// f32), matching `irgen`'s `to_slot`.
fn to_slot(v: SpecVal) -> i64 {
    match v {
        SpecVal::I32(x) => x as i64,
        SpecVal::I64(x) => x,
        SpecVal::F32(b) => b as i64,
        SpecVal::F64(b) => b as i64,
    }
}

fn interp_trap(t: SpecTrap) -> Trap {
    match t {
        SpecTrap::DivByZero => Trap::DivByZero,
        SpecTrap::IntOverflow => Trap::IntOverflow,
        SpecTrap::BadConversion => Trap::BadConversion,
    }
}

fn jit_trap(t: SpecTrap) -> TrapKind {
    match t {
        SpecTrap::DivByZero => TrapKind::DivByZero,
        SpecTrap::IntOverflow => TrapKind::IntOverflow,
        SpecTrap::BadConversion => TrapKind::BadConversion,
    }
}

/// Bit-exact match, except a NaN expectation accepts any NaN (§3b: NaN bits unpinned).
fn value_matches(expected: SpecVal, got: &Value) -> bool {
    match (expected, got) {
        (SpecVal::I32(e), Value::I32(g)) => e == *g,
        (SpecVal::I64(e), Value::I64(g)) => e == *g,
        (SpecVal::F32(e), Value::F32(g)) => {
            e == g.to_bits() || (f32::from_bits(e).is_nan() && g.is_nan())
        }
        (SpecVal::F64(e), Value::F64(g)) => {
            e == g.to_bits() || (f64::from_bits(e).is_nan() && g.is_nan())
        }
        _ => false,
    }
}

/// Decode a JIT result slot at the row's result type (the trampoline widens everything
/// to i64; an i32 result occupies the low 32 bits — compare there, like `irgen`).
fn slot_matches(expected: SpecVal, slot: i64) -> bool {
    match expected {
        SpecVal::I32(e) => e == slot as i32,
        SpecVal::I64(e) => e == slot,
        SpecVal::F32(e) => {
            e == slot as u32 || (f32::from_bits(e).is_nan() && f32::from_bits(slot as u32).is_nan())
        }
        SpecVal::F64(e) => {
            e == slot as u64 || (f64::from_bits(e).is_nan() && f64::from_bits(slot as u64).is_nan())
        }
    }
}

/// One vector on all three backends against the spec expectation. `cm` is the row's
/// pre-compiled JIT module (compiled once per row for `Operands` rows).
fn check_vector(row: &OpRow, m: &svm_ir::Module, cm: &mut CompiledModule, vector: &[SpecVal]) {
    let expected = (row.eval)(vector);
    let ctx = |backend: &str, got: &dyn std::fmt::Debug| {
        format!(
            "spec divergence [{backend}] op={} vector={vector:?}\n expected={expected:?}\n got={got:?}\n module:\n{}",
            row.id,
            svm::text::print_module(m)
        )
    };
    let args: Vec<Value> = match row.shape {
        Shape::Operands => vector.iter().copied().map(to_value).collect(),
        Shape::Immediate => Vec::new(),
    };

    // Tree-walk interpreter (the runtime oracle — here itself under the spec's oracle).
    let mut fuel = 10_000u64;
    let interp = run(m, 0, &args, &mut fuel);
    let ok = match (&expected, &interp) {
        (Ok(e), Ok(vs)) => vs.len() == 1 && value_matches(*e, &vs[0]),
        (Err(t), Err(tr)) => *tr == interp_trap(*t),
        _ => false,
    };
    assert!(ok, "{}", ctx("interp", &interp));

    // Bytecode interpreter. The scalar core must be supported (`None` = unsupported
    // module shape, which for these single-op modules would itself be a finding).
    let mut fuel = 10_000u64;
    let bc = bytecode::compile_and_run(m, 0, &args, &mut fuel)
        .unwrap_or_else(|| panic!("{}", ctx("bytecode", &"unsupported module")));
    let ok = match (&expected, &bc) {
        (Ok(e), Ok(vs)) => vs.len() == 1 && value_matches(*e, &vs[0]),
        (Err(t), Err(tr)) => *tr == interp_trap(*t),
        _ => false,
    };
    assert!(ok, "{}", ctx("bytecode", &bc));

    // Cranelift JIT.
    let slots: Vec<i64> = match row.shape {
        Shape::Operands => vector.iter().copied().map(to_slot).collect(),
        Shape::Immediate => Vec::new(),
    };
    let (out, _mem) = cm
        .run(&slots, None, None, None)
        .unwrap_or_else(|e| panic!("{}", ctx("jit", &e)));
    let ok = match (&expected, &out) {
        (Ok(e), JitOutcome::Returned(rs)) => rs.len() == 1 && slot_matches(*e, rs[0]),
        (Err(t), JitOutcome::Trapped(k)) => *k == jit_trap(*t),
        _ => false,
    };
    assert!(ok, "{}", ctx("jit", &out));
}

#[test]
fn spec_vectors_match_all_backends() {
    let mut vectors_run = 0usize;
    for row in scalar_rows() {
        match row.shape {
            Shape::Operands => {
                // Input-independent module: verify + JIT-compile once, run every vector.
                let m = module_for(&row, &[]);
                svm::verify::verify_module(&m)
                    .unwrap_or_else(|e| panic!("spec module for {} fails verify: {e:?}", row.id));
                let mut cm = compile(&m, 0).unwrap_or_else(|e| {
                    panic!("spec module for {} fails JIT compile: {e:?}", row.id)
                });
                for vector in vectors_for(&row) {
                    check_vector(&row, &m, &mut cm, &vector);
                    vectors_run += 1;
                }
            }
            Shape::Immediate => {
                // The input is the baked immediate: one module (and compile) per vector.
                for vector in vectors_for(&row) {
                    let m = module_for(&row, &vector);
                    svm::verify::verify_module(&m).unwrap_or_else(|e| {
                        panic!("spec module for {} fails verify: {e:?}", row.id)
                    });
                    let mut cm = compile(&m, 0).unwrap_or_else(|e| {
                        panic!("spec module for {} fails JIT compile: {e:?}", row.id)
                    });
                    check_vector(&row, &m, &mut cm, &vector);
                    vectors_run += 1;
                }
            }
        }
    }
    // Coverage canary: the suite actually exercised the expected order of magnitude
    // (all unary/binary rows take their full cross product — see svm-spec's
    // `no_striding_below_ternary`).
    assert!(
        vectors_run > 30_000,
        "suspiciously few spec vectors ran: {vectors_run}"
    );
}
