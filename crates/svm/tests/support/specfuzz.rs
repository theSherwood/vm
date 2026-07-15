//! Shared spec-differential fuzz drivers (SPEC.md) — the coverage-guided counterpart to
//! the deterministic spec suites. Two independent drivers, each reachable from a nightly
//! libFuzzer target (`fuzz/fuzz_targets/spec_{ops,verify}.rs`) and mirrored on stable by
//! `crates/svm/tests/spec_fuzz_smoke.rs`, exactly as `irgen.rs` is shared by `diff`/`jit_fuzz`:
//!
//! - [`ops_one`]: pick a scalar/float spec row, feed it **coverage-guided random** operand
//!   values, and assert all three backends (tree-walk interp, bytecode interp, JIT) match the
//!   spec's `eval` — the unbounded exploration the fixed boundary lattice (`spec_vectors`)
//!   can't reach.
//! - [`verify_one`]: generate a verifier-valid module (`irgen`), apply a random structural
//!   mutation, and assert `svm-verify` and the reference verifier (`svm_spec::verify`) agree
//!   on accept/reject — the coverage-guided version of `spec_verify`'s `irgen` sweep.
//!
//! A crash in either is a real finding: a backend diverging from the spec definition, or the
//! two verifiers disagreeing (an accept-direction verifier bug).

#![allow(dead_code)] // each includer (fuzz target / smoke test) uses one driver

#[path = "irgen.rs"]
mod irgen;

use svm_interp::{bytecode, run, Trap, Value};
use svm_jit::{compile, JitOutcome, TrapKind};
use svm_spec::{all_rows, module_for, OpRow, Shape, SpecTrap, SpecVal};

// ---------------------------------------------------------------------------------------------
// Entropy: a tiny byte cursor over the libFuzzer input, falling back to a deterministic xorshift
// once the bytes run out (so a short input still exercises a full vector, and a seed reproduces).
// ---------------------------------------------------------------------------------------------

struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
    rng: u64,
}

impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Reader<'a> {
        let mut seed =
            0x9e37_79b9_7f4a_7c15u64 ^ (data.len() as u64).wrapping_mul(0x0100_0000_01b3);
        for &b in data.iter().take(8) {
            seed = seed.wrapping_mul(31).wrapping_add(b as u64);
        }
        Reader {
            data,
            pos: 0,
            rng: seed | 1,
        }
    }
    fn byte(&mut self) -> u8 {
        if self.pos < self.data.len() {
            let b = self.data[self.pos];
            self.pos += 1;
            b
        } else {
            let mut x = self.rng;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.rng = x;
            (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 33) as u8
        }
    }
    fn u32(&mut self) -> u32 {
        let mut v = 0u32;
        for _ in 0..4 {
            v = (v << 8) | self.byte() as u32;
        }
        v
    }
    fn u64(&mut self) -> u64 {
        ((self.u32() as u64) << 32) | self.u32() as u64
    }
    fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.u32() as usize) % n
        }
    }
    fn out_of_input(&self) -> bool {
        self.pos >= self.data.len()
    }
}

// ---------------------------------------------------------------------------------------------
// Driver A — per-op semantics vs the spec `eval`, on all three backends.
// ---------------------------------------------------------------------------------------------

fn read_val(r: &mut Reader, t: svm_ir::ValType) -> SpecVal {
    use svm_ir::ValType as V;
    match t {
        V::I32 => SpecVal::I32(r.u32() as i32),
        V::I64 => SpecVal::I64(r.u64() as i64),
        V::F32 => SpecVal::F32(r.u32()),
        V::F64 => SpecVal::F64(r.u64()),
        // `all_rows()` (scalar + float value ops) never takes a v128/ref operand.
        V::V128 | V::Ref => unreachable!("ops driver row took a {t:?} operand"),
    }
}

// The ops driver runs only `all_rows()` (scalar + float value ops), so a `V128` value never
// reaches these — it exists on `SpecVal` for the SIMD slice, which this driver does not cover.
fn to_value(v: SpecVal) -> Value {
    match v {
        SpecVal::I32(x) => Value::I32(x),
        SpecVal::I64(x) => Value::I64(x),
        SpecVal::F32(b) => Value::F32(f32::from_bits(b)),
        SpecVal::F64(b) => Value::F64(f64::from_bits(b)),
        SpecVal::V128(_) => unreachable!("ops driver is scalar/float only"),
    }
}
fn to_slot(v: SpecVal) -> i64 {
    match v {
        SpecVal::I32(x) => x as i64,
        SpecVal::I64(x) => x,
        SpecVal::F32(b) => b as i64,
        SpecVal::F64(b) => b as i64,
        SpecVal::V128(_) => unreachable!("ops driver is scalar/float only"),
    }
}
fn interp_trap(t: SpecTrap) -> Trap {
    match t {
        SpecTrap::DivByZero => Trap::DivByZero,
        SpecTrap::IntOverflow => Trap::IntOverflow,
        SpecTrap::BadConversion => Trap::BadConversion,
        SpecTrap::MemoryFault => Trap::MemoryFault,
    }
}
fn jit_trap(t: SpecTrap) -> TrapKind {
    match t {
        SpecTrap::DivByZero => TrapKind::DivByZero,
        SpecTrap::IntOverflow => TrapKind::IntOverflow,
        SpecTrap::BadConversion => TrapKind::BadConversion,
        SpecTrap::MemoryFault => TrapKind::MemoryFault,
    }
}
/// Bit-exact, except a NaN expectation accepts any NaN (§3b: NaN bits unpinned).
fn value_matches(e: SpecVal, g: &Value) -> bool {
    match (e, g) {
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
fn slot_matches(e: SpecVal, slot: i64) -> bool {
    match e {
        SpecVal::I32(e) => e == slot as i32,
        SpecVal::I64(e) => e == slot,
        SpecVal::F32(e) => {
            e == slot as u32 || (f32::from_bits(e).is_nan() && f32::from_bits(slot as u32).is_nan())
        }
        SpecVal::F64(e) => {
            e == slot as u64 || (f64::from_bits(e).is_nan() && f64::from_bits(slot as u64).is_nan())
        }
        SpecVal::V128(_) => unreachable!("ops driver is scalar/float only"),
    }
}

fn check_one(
    row: &OpRow,
    m: &svm_ir::Module,
    cm: &mut svm_jit::CompiledModule,
    vector: &[SpecVal],
) {
    let expected = (row.eval)(vector);
    let ctx = |backend: &str| {
        format!(
            "spec-fuzz divergence [{backend}] op={} vector={vector:?} expected={expected:?}",
            row.id
        )
    };
    let args: Vec<Value> = match row.shape {
        Shape::Operands => vector.iter().copied().map(to_value).collect(),
        Shape::Immediate => Vec::new(),
    };

    let mut fuel = 10_000u64;
    let ir = run(m, 0, &args, &mut fuel);
    let ok = match (&expected, &ir) {
        (Ok(e), Ok(vs)) => vs.len() == 1 && value_matches(*e, &vs[0]),
        (Err(t), Err(tr)) => *tr == interp_trap(*t),
        _ => false,
    };
    assert!(ok, "{} got={ir:?}", ctx("interp"));

    let mut fuel = 10_000u64;
    if let Some(bc) = bytecode::compile_and_run(m, 0, &args, &mut fuel) {
        let ok = match (&expected, &bc) {
            (Ok(e), Ok(vs)) => vs.len() == 1 && value_matches(*e, &vs[0]),
            (Err(t), Err(tr)) => *tr == interp_trap(*t),
            _ => false,
        };
        assert!(ok, "{} got={bc:?}", ctx("bytecode"));
    }

    let slots: Vec<i64> = match row.shape {
        Shape::Operands => vector.iter().copied().map(to_slot).collect(),
        Shape::Immediate => Vec::new(),
    };
    if let Ok((out, _)) = cm.run(&slots, None, None, None) {
        let ok = match (&expected, &out) {
            (Ok(e), JitOutcome::Returned(rs)) => rs.len() == 1 && slot_matches(*e, rs[0]),
            (Err(t), JitOutcome::Trapped(k)) => *k == jit_trap(*t),
            _ => false,
        };
        assert!(ok, "{} got={out:?}", ctx("jit"));
    }
}

/// Driver A: one fuzz input drives one op row over several random vectors (compile the JIT
/// module once, amortized). A divergence from the spec `eval` on any backend panics.
pub fn ops_one(data: &[u8]) {
    let rows = all_rows();
    let mut r = Reader::new(data);
    let row = &rows[r.below(rows.len())];

    // `Immediate` rows (consts) bake the input, so each vector is its own module; `Operands`
    // rows compile once and run every vector.
    match row.shape {
        Shape::Immediate => {
            for _ in 0..8 {
                let v = vec![read_val(&mut r, row.result)];
                let m = module_for(row, &v);
                if let Ok(mut cm) = compile(&m, 0) {
                    check_one(row, &m, &mut cm, &v);
                }
                if r.out_of_input() {
                    break;
                }
            }
        }
        Shape::Operands => {
            let m = module_for(row, &[]);
            let Ok(mut cm) = compile(&m, 0) else { return };
            for _ in 0..16 {
                let v: Vec<SpecVal> = row.operands.iter().map(|&t| read_val(&mut r, t)).collect();
                check_one(row, &m, &mut cm, &v);
                if r.out_of_input() {
                    break;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Driver B — verifier accept/reject agreement between svm-verify and the reference verifier.
// ---------------------------------------------------------------------------------------------

use irgen::Gen;
use svm_ir::{Inst, Module, Terminator, ValType};

/// A deterministic structural mutation of a verifier-valid module. A mutation may leave the
/// module valid — the assertion in [`verify_one`] is *agreement*, not rejection.
fn mutate(m: &mut Module, kind: u64) {
    match kind % 6 {
        0 => {
            for f in &mut m.funcs {
                for b in &mut f.blocks {
                    if let Terminator::Br { target, .. } = &mut b.term {
                        *target = 999;
                        return;
                    }
                }
            }
        }
        1 => {
            for f in &mut m.funcs {
                for b in &mut f.blocks {
                    if let Terminator::Return(vals) = &mut b.term {
                        if !vals.is_empty() {
                            vals.clear();
                            return;
                        }
                    }
                }
            }
        }
        2 => {
            for f in &mut m.funcs {
                for b in &mut f.blocks {
                    if let Some(p) = b.params.first_mut() {
                        *p = match *p {
                            ValType::I32 => ValType::I64,
                            ValType::I64 => ValType::F32,
                            ValType::F32 => ValType::F64,
                            ValType::F64 => ValType::V128,
                            ValType::V128 => ValType::Ref,
                            ValType::Ref => ValType::I32,
                        };
                        return;
                    }
                }
            }
        }
        3 => m.memory = None,
        4 => {
            if let Some(f) = m.funcs.first_mut() {
                f.results.push(ValType::I32);
            }
        }
        5 => {
            for f in &mut m.funcs {
                for b in &mut f.blocks {
                    if !b.insts.is_empty() {
                        b.insts.insert(0, Inst::ConstI32(0));
                        return;
                    }
                }
            }
        }
        _ => unreachable!(),
    }
}

/// Driver B: an `irgen` module (verifier-valid by construction) must be accepted by BOTH
/// verifiers; a random mutation of it must leave the two in accept/reject agreement. A
/// disagreement panics — the accept-direction gap the reference verifier exists to close.
pub fn verify_one(data: &[u8]) {
    let mut g = Gen::from_bytes(data);
    let m = irgen::gen_module(&mut g);

    // The generator's output is valid by construction: both verifiers must accept it.
    let prod_ok = svm::verify::verify_module(&m).is_ok();
    let ref_ok = svm_spec::verify::verify(&m).is_ok();
    assert!(
        prod_ok && ref_ok,
        "irgen module rejected: production={prod_ok} reference={ref_ok}\n{}",
        svm::text::print_module(&m)
    );

    // Mutate and require agreement (a mutation may leave it valid — that's fine).
    let mut mm = m.clone();
    mutate(&mut mm, Reader::new(data).u64());
    let prod = svm::verify::verify_module(&mm).is_ok();
    let refv = svm_spec::verify::verify(&mm).is_ok();
    assert_eq!(
        prod,
        refv,
        "verifier disagreement: production={prod} reference={refv}\n{}",
        svm::text::print_module(&mm)
    );
}
