//! Equality harness for the Phase-1b bytecode engine (see `INTERP_PERF.md`).
//!
//! The bytecode engine (`svm_interp::bytecode`) must be a *bug-for-bug* match of the reference
//! interpreter on every program it supports — it reuses the same semantic helpers, so unlike the
//! interp-vs-JIT differential this is **exact** (results and trap kind identical, floats/NaN bits
//! included). This is the gate that must stay green before the bytecode path is ever made the
//! default. Two parts:
//!   1. Generative: thousands of verifier-valid modules from the shared generator; for each the
//!      bytecode engine supports, assert it agrees with `svm_interp::run`.
//!   2. The `interp_perf` kernels: assert equality and report the bytecode speedup over the
//!      tree-walker (the production counterpart of the throwaway `bytecode_spike`).

#[path = "support/irgen.rs"]
mod irgen;

use irgen::{gen_args, gen_module, Gen};
use std::hint::black_box;
use std::time::Instant;
use svm_interp::{bytecode, run, run_fast, run_traced, Host, Trap, Value};

/// Bit-wise value equality — `Value`'s derived `PartialEq` uses IEEE float `==`, where `NaN != NaN`,
/// so two bit-identical NaN results would compare unequal. Both engines share exact semantics
/// (including NaN bits), so compare by the raw bit pattern.
fn value_bits(v: &Value) -> u128 {
    match v {
        Value::I32(x) => *x as u32 as u128,
        Value::I64(x) => *x as u64 as u128,
        Value::F32(x) => x.to_bits() as u128,
        Value::F64(x) => x.to_bits() as u128,
        Value::Ref(x) => *x as u128,
        Value::V128(b) => u128::from_le_bytes(*b),
    }
}

fn results_eq(a: &Result<Vec<Value>, Trap>, b: &Result<Vec<Value>, Trap>) -> bool {
    match (a, b) {
        (Ok(va), Ok(vb)) => {
            va.len() == vb.len()
                && va
                    .iter()
                    .zip(vb)
                    .all(|(x, y)| value_bits(x) == value_bits(y))
        }
        (Err(ta), Err(tb)) => ta == tb,
        _ => false,
    }
}

/// Run `m`'s entry on both engines with equal fuel and assert equality — unless the bytecode engine
/// doesn't support the module (`None` → skip) or either side runs out of fuel (per-op fuel
/// accounting can differ by a hair near the limit; not a semantic divergence). Returns `true` if
/// the bytecode engine supported (and matched) the module.
fn check(m: &svm_ir::Module, args: &[Value], seed: u64) -> bool {
    if m.funcs.is_empty() {
        return false;
    }
    let mut fi = 2_000_000u64;
    let interp = run(m, 0, args, &mut fi);
    let mut fb = 2_000_000u64;
    let Some(bc) = bytecode::compile_and_run(m, 0, args, &mut fb) else {
        return false;
    };
    if matches!(interp, Err(Trap::OutOfFuel)) || matches!(bc, Err(Trap::OutOfFuel)) {
        return false;
    }
    if !results_eq(&bc, &interp) {
        panic!(
            "bytecode disagrees with interpreter\n seed={seed}\n args={args:?}\n interp={interp:?}\n bc    ={bc:?}\n module:\n{}",
            svm::text::print_module(m)
        );
    }
    // Trap-time backtrace parity (the bytecode mirror of `run_traced`): on a trap, both engines must
    // reify the *same* call stack (innermost frame first, as raw `IrPc`s — no `-g` info needed). The
    // single-stepping traced path returns `None` only on a concurrency seam (out of its single-vCPU
    // scope); skip those, the result equality above already covers them.
    if interp.is_err() {
        let mut ft = 2_000_000u64;
        let (tw_res, tw_bt, _) = run_traced(m, 0, args, &mut ft);
        let mut fbt = 2_000_000u64;
        if let Some((bc_res, bc_bt, _)) =
            bytecode::compile_and_run_with_host_traced(m, 0, args, &mut fbt, &mut Host::new())
        {
            if !matches!(tw_res, Err(Trap::OutOfFuel)) && !matches!(bc_res, Err(Trap::OutOfFuel)) {
                assert_eq!(
                    tw_bt,
                    bc_bt,
                    "trap backtrace disagrees\n seed={seed}\n args={args:?}\n tw_res={tw_res:?} bc_res={bc_res:?}\n module:\n{}",
                    svm::text::print_module(m)
                );
            }
        }
    }
    true
}

#[test]
fn bytecode_matches_interp_on_generated_modules() {
    let mut supported = 0u32;
    let total = 4000u32;
    for seed in 0..total as u64 {
        let mut g = Gen::from_seed(seed);
        let m = gen_module(&mut g);
        let args = gen_args(&mut g, &m.funcs[0].params);
        if check(&m, &args, seed) {
            supported += 1;
        }
    }
    // Sanity: the generator must actually exercise the bytecode path on a meaningful share of
    // modules (else the harness is vacuous). The subset is scalar + memory + direct calls.
    println!("bytecode supported {supported}/{total} generated modules");
    assert!(
        supported > total / 20,
        "bytecode path exercised on too few modules ({supported}/{total}) — harness near-vacuous"
    );
}

#[test]
fn bytecode_suspend_resume_preserves_result() {
    // Slice 1c-2: slicing a run at op boundaries (suspend, persist the `Vm`, resume) must be
    // bit-identical to running straight through, for any slice size — that is what proves the
    // reified continuation is exact. A subset of the corpus with a bounded fuel cap keeps the
    // slice=1 (suspend after *every* op) case cheap.
    let cap = 100_000u64;
    let mut checked = 0u32;
    for seed in (0..4000u64).step_by(16) {
        let mut g = Gen::from_seed(seed);
        let m = gen_module(&mut g);
        if m.funcs.is_empty() {
            continue;
        }
        let args = gen_args(&mut g, &m.funcs[0].params);
        let mut fw = cap;
        let Some(whole) = bytecode::compile_and_run(&m, 0, &args, &mut fw) else {
            continue;
        };
        if matches!(whole, Err(Trap::OutOfFuel)) {
            continue;
        }
        for slice in [1u64, 3, 17] {
            let mut fs = cap;
            let sliced = bytecode::compile_and_run_sliced(&m, 0, &args, &mut fs, slice)
                .expect("supported (the whole run above compiled)");
            assert!(
                results_eq(&sliced, &whole),
                "suspend/resume changed the result\n seed={seed} slice={slice}\n whole ={whole:?}\n sliced={sliced:?}\n module:\n{}",
                svm::text::print_module(&m)
            );
        }
        checked += 1;
    }
    println!("suspend/resume equality verified on {checked} modules");
    assert!(checked > 10, "too few modules exercised ({checked})");
}

#[test]
fn run_fast_matches_run_on_generated_modules() {
    // Slice 1c-4: `run_fast` routes eligible modules through the bytecode engine and falls back to
    // the tree-walker `run` otherwise; either way it must equal `run`. (Equivalence on the eligible
    // set is already gated above; this also covers the fallback path and the routing wrapper.)
    let total = 4000u64;
    for seed in 0..total {
        let mut g = Gen::from_seed(seed);
        let m = gen_module(&mut g);
        if m.funcs.is_empty() {
            continue;
        }
        let args = gen_args(&mut g, &m.funcs[0].params);
        let mut f1 = 2_000_000u64;
        let slow = run(&m, 0, &args, &mut f1);
        let mut f2 = 2_000_000u64;
        let fast = run_fast(&m, 0, &args, &mut f2);
        // Near the fuel limit the two engines' per-op accounting can differ by a hair (not a
        // semantic divergence) — skip those, like the main harness.
        if matches!(slow, Err(Trap::OutOfFuel)) || matches!(fast, Err(Trap::OutOfFuel)) {
            continue;
        }
        assert!(
            results_eq(&fast, &slow),
            "run_fast disagrees with run\n seed={seed}\n slow={slow:?}\n fast={fast:?}\n module:\n{}",
            svm::text::print_module(&m)
        );
    }
}

// ---- kernels: equality + perf (production counterpart of bytecode_spike) -----------------------

const ALU: &str = r#"
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 0
  v2 = i64.const 0
  br 1(v0, v1, v2)
}
block 1 (v3: i64, v4: i64, v5: i64) {
  v6 = i64.lt_s v5 v3
  br_if v6 2(v3, v4, v5) 3(v4)
}
block 2 (v7: i64, v8: i64, v9: i64) {
  v10 = i64.const 6364136223846793005
  v11 = i64.mul v8 v10
  v12 = i64.const 1442695040888963407
  v13 = i64.add v11 v12
  v14 = i64.add v13 v9
  v15 = i64.const 1
  v16 = i64.add v9 v15
  br 1(v7, v14, v16)
}
block 3 (v17: i64) {
  return v17
  }
}
"#;

const CALL: &str = r#"
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 0
  v2 = i64.const 0
  br 1(v0, v1, v2)
}
block 1 (v3: i64, v4: i64, v5: i64) {
  v6 = i64.lt_s v5 v3
  br_if v6 2(v3, v4, v5) 3(v4)
}
block 2 (v7: i64, v8: i64, v9: i64) {
  v10 = call 1 (v8, v9)
  v11 = i64.const 1
  v12 = i64.add v9 v11
  br 1(v7, v10, v12)
}
block 3 (v13: i64) {
  return v13
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  v2 = i64.add v0 v1
  return v2
  }
}
"#;

const MEM: &str = r#"
memory 16
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 0
  v2 = i64.const 0
  br 1(v0, v1, v2)
}
block 1 (v3: i64, v4: i64, v5: i64) {
  v6 = i64.lt_s v5 v3
  br_if v6 2(v3, v4, v5) 3(v4)
}
block 2 (v7: i64, v8: i64, v9: i64) {
  v10 = i64.const 8
  i64.store v10 v8
  v11 = i64.load v10
  v12 = i64.add v11 v9
  v13 = i64.const 1
  v14 = i64.add v9 v13
  br 1(v7, v12, v14)
}
block 3 (v15: i64) {
  return v15
  }
}
"#;

fn interp_call(m: &svm_ir::Module, n: i64) -> i64 {
    let mut fuel = u64::MAX;
    let v = run(m, 0, &[Value::I64(n)], &mut fuel).expect("interp");
    match v[0] {
        Value::I64(x) => x,
        o => panic!("{o:?}"),
    }
}

fn bc_call(m: &svm_ir::Module, n: i64) -> i64 {
    let mut fuel = u64::MAX;
    let v = bytecode::compile_and_run(m, 0, &[Value::I64(n)], &mut fuel)
        .expect("bytecode supports kernel")
        .expect("bytecode run");
    match v[0] {
        Value::I64(x) => x,
        o => panic!("{o:?}"),
    }
}

fn per_call(it: u32, mut f: impl FnMut()) -> f64 {
    for _ in 0..(it / 4).max(1) {
        f();
    }
    let t = Instant::now();
    for _ in 0..it {
        f();
    }
    t.elapsed().as_secs_f64() / it as f64
}

fn ns_per_iter(reps: u32, big: i64, small: i64, mut call: impl FnMut(i64) -> i64) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let b = per_call(10, || {
            black_box(call(big));
        });
        let s = per_call(10, || {
            black_box(call(small));
        });
        best = best.min((b - s) / (big - small) as f64 * 1e9);
    }
    best
}

#[test]
fn bytecode_matches_interp_on_kernels() {
    for (name, src) in [("alu", ALU), ("call", CALL), ("mem", MEM)] {
        let m = svm::text::parse_module(src).expect("parse");
        svm::verify::verify_module(&m).expect("verify");
        for n in [0i64, 1, 7, 1000, 123_456] {
            assert_eq!(bc_call(&m, n), interp_call(&m, n), "{name} n={n}");
        }
    }
}

#[test]
#[ignore = "benchmark; run explicitly with --nocapture --ignored"]
fn bytecode_kernel_perf() {
    for (name, src) in [("alu", ALU), ("call", CALL), ("mem", MEM)] {
        let m = svm::text::parse_module(src).expect("parse");
        svm::verify::verify_module(&m).expect("verify");
        let i = ns_per_iter(5, 200_000, 1_000, |n| interp_call(&m, n));
        let b = ns_per_iter(5, 200_000, 1_000, |n| bc_call(&m, n));
        println!(
            "\n{name} (ns/iter): interp {i:>8.3}  bytecode {b:>8.3}  ({:.2}x)",
            i / b
        );
    }
}
