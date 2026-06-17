//! Interpreter throughput baseline — the SAME hand-written IR run through the reference
//! interpreter and the JIT, so the ratio isolates interpreter overhead against the shared
//! semantics. Opt-in (it's a benchmark, not a gate):
//!   cargo test -p svm --release --test interp_perf -- --nocapture --ignored

use std::hint::black_box;
use std::time::Instant;

use svm_interp::Value;
use svm_jit::{compile_and_run, JitOutcome};

// acc = acc * C1 + C2 + i, for i in 0..n  (the bench's `alu` recurrence, hand-written).
const ALU: &str = r#"
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const 0
  v2 = i64.const 0
  br block1(v0, v1, v2)
block1(v3: i64, v4: i64, v5: i64):
  v6 = i64.lt_s v5 v3
  br_if v6 block2(v3, v4, v5) block3(v4)
block2(v7: i64, v8: i64, v9: i64):
  v10 = i64.const 6364136223846793005
  v11 = i64.mul v8 v10
  v12 = i64.const 1442695040888963407
  v13 = i64.add v11 v12
  v14 = i64.add v13 v9
  v15 = i64.const 1
  v16 = i64.add v9 v15
  br block1(v7, v14, v16)
block3(v17: i64):
  return v17
}
"#;

fn interp_call(m: &svm_ir::Module, n: i64) -> i64 {
    let mut fuel = u64::MAX;
    match svm_interp::run(m, 0, &[Value::I64(n)], &mut fuel) {
        Ok(v) => match v[0] {
            Value::I64(x) => x,
            other => panic!("{other:?}"),
        },
        Err(e) => panic!("interp trapped: {e:?}"),
    }
}

fn jit_call(m: &svm_ir::Module, n: i64) -> i64 {
    match compile_and_run(m, 0, &[n]) {
        Ok(JitOutcome::Returned(s)) => s[0],
        other => panic!("{other:?}"),
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

/// Per-iteration steady-state ns, isolated by subtraction (big - small) / delta.
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
#[ignore = "benchmark; run explicitly with --nocapture --ignored"]
fn interp_vs_jit_throughput() {
    let m = svm::text::parse_module(ALU).expect("parse");
    svm::verify::verify_module(&m).expect("verify");
    assert_eq!(
        interp_call(&m, 1000),
        jit_call(&m, 1000),
        "backends disagree"
    );

    // Interp is ~100x slower, so give it a smaller "big" to keep the test snappy.
    let i = ns_per_iter(5, 200_000, 1_000, |n| interp_call(&m, n));
    let j = ns_per_iter(5, 5_000_000, 1_000, |n| jit_call(&m, n));
    println!("\ninterp vs JIT (same IR, ns/iter of the alu recurrence):");
    println!("  interp : {i:>9.3} ns/iter");
    println!("  jit    : {j:>9.3} ns/iter");
    println!("  ratio  : {:>9.1}x  (interp / jit)", i / j);
}
