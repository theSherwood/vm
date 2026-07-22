//! Interpreter throughput baseline — the SAME hand-written IR run through the reference
//! interpreter and the JIT, so the ratio isolates interpreter overhead against the shared
//! semantics. Opt-in (it's a benchmark, not a gate):
//!   cargo test -p svm --release --test interp_perf -- --nocapture --ignored

use std::hint::black_box;
use std::process::Command;
use std::time::Instant;

use svm_interp::Value;
use svm_jit::{compile_and_run, JitOutcome};

// acc = acc * C1 + C2 + i, for i in 0..n  (the bench's `alu` recurrence, hand-written).
// Exercises the per-instruction dispatch / operand-read hot path with no calls or memory.
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

// acc += leaf(acc, i), for i in 0..n, where leaf(a, b) = a + b in a separate function.
// Each iteration is dominated by a direct call + return, so this kernel exercises the
// frame push/pop and argument/result marshalling rather than the ALU dispatch.
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

/// Steady-state ns/iter for the matching CPython kernel (`interp_perf.py`) — a calibration point
/// so the comparison isn't just interp-vs-JIT but interp-vs-(CPython, JIT). Best-effort: returns
/// `None` if `python3` isn't on PATH or the script errors, so the benchmark still runs without it.
fn python_ns(py_key: &str, big: i64) -> Option<f64> {
    let script = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/interp_perf.py");
    let out = Command::new("python3")
        .args([script, py_key, &big.to_string()])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    String::from_utf8(out.stdout).ok()?.trim().parse().ok()
}

/// Run one hand-written kernel through both backends and print the per-iteration ns and the
/// interp/JIT ratio (plus a CPython reference for the same computation). `interp_big`/`jit_big`
/// are the "big" loop counts for the subtraction method — the interp is much slower than the JIT,
/// so it gets a smaller `big` to keep the test snappy. `py_key` selects the matching CPython kernel.
fn bench(name: &str, src: &str, py_key: &str, interp_big: i64, jit_big: i64) {
    let m = svm::text::parse_module(src).expect("parse");
    svm::verify::verify_module(&m).expect("verify");
    assert_eq!(
        interp_call(&m, 1000),
        jit_call(&m, 1000),
        "backends disagree ({name})"
    );

    let i = ns_per_iter(5, interp_big, 1_000, |n| interp_call(&m, n));
    let j = ns_per_iter(5, jit_big, 1_000, |n| jit_call(&m, n));
    println!("\n{name} (ns/iter):");
    println!("  interp : {i:>9.3}");
    println!("  jit    : {j:>9.3}   (interp/jit {:>7.1}x)", i / j);
    match python_ns(py_key, interp_big) {
        Some(p) => println!("  python : {p:>9.3}   (interp/python {:>5.2}x)", i / p),
        None => println!("  python :         -   (python3 unavailable)"),
    }
}

// acc = load(load(acc) stored at a fixed address) + i, for i in 0..n. Each iteration does an
// i64 store + i64 load against linear memory, so this kernel exercises the memory-op path
// (confinement + page-protection check + the `Mem` wrappers) rather than pure ALU dispatch.
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

#[test]
#[ignore = "benchmark; run explicitly with --nocapture --ignored"]
fn interp_vs_jit_throughput() {
    bench("alu recurrence", ALU, "alu", 200_000, 5_000_000);
    bench("call/return loop", CALL, "call", 100_000, 5_000_000);
    bench("memory load/store loop", MEM, "mem", 100_000, 5_000_000);
}
