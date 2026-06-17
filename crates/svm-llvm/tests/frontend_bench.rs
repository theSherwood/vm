//! One-off: compare the SVM JIT running the SAME C through two frontends — chibicc (naive) vs
//! `clang -O2` + svm-llvm (optimized). Same backend, so the ratio isolates frontend IR quality.
//! Run: cargo test -p svm-llvm --release --test frontend_bench -- --nocapture --ignored
use std::hint::black_box;
use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

use svm_jit::{compile_and_run, JitOutcome};

// The bench's C kernel sources (run + main, exactly as in bench/src/main.rs).
const ALU: &str = "long run(long n){\n  long acc = 0;\n  for (long i = 0; i < n; i++)\n    \
    acc = acc * 6364136223846793005L + 1442695040888963407L + i;\n  return acc;\n}\n\
    int main(){ return (int)run(0); }\n";
const LOCALS: &str = "long run(long n){\n  volatile long a[256];\n  long acc = 0;\n  \
    for (long i = 0; i < n; i++) { a[i & 255] = i; acc += a[i & 255]; }\n  return acc;\n}\n\
    int main(){ return (int)run(0); }\n";
const IRRED: &str =
    "long long run(long long n){\n  long long a=0,b=0,i=0;\n  if (n & 1) goto odd;\n  \
    while (i < n) {\n    a += i; i++;\n  odd:\n    b += i*3; i++;\n  }\n  return a + b;\n}\n\
    int main(){ return (int)run(0); }\n";
// A complex kernel: the loop calls a small pure helper each iteration. `clang -O2` INLINES `mix`
// into the loop; chibicc (no inlining) emits a real `call` per iteration, and svm-jit does no
// cross-function inlining — so this is where LLVM's mid-end actually buys something a tight loop
// doesn't. An FNV-style mixing step.
const HASH: &str = "static long mix(long h, long x){\n  h ^= x;\n  h *= 1099511628211L;\n  \
    h ^= h >> 27;\n  return h;\n}\n\
    long run(long n){\n  long h = -3750763034362895579L;\n  \
    for (long i = 0; i < n; i++) h = mix(h, i);\n  return h;\n}\n\
    int main(){ return (int)run(0); }\n";

fn chibicc_ir(name: &str, src: &str) -> (svm_ir::Module, u32, Vec<i64>) {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    let dir = root.join("frontend/chibicc");
    assert!(Command::new("make")
        .arg("-s")
        .current_dir(&dir)
        .status()
        .unwrap()
        .success());
    let base = std::env::temp_dir().join(format!("fb_chi_{name}"));
    let cf = base.with_extension("c");
    let irf = base.with_extension("svm");
    std::fs::write(&cf, src).unwrap();
    assert!(Command::new(dir.join("chibicc"))
        .args(["-cc1", "--emit-ir", "-cc1-input"])
        .arg(&cf)
        .arg("-cc1-output")
        .arg(&irf)
        .arg(&cf)
        .status()
        .unwrap()
        .success());
    let ir = std::fs::read_to_string(&irf).unwrap();
    let m = svm_text::parse_module(&ir).unwrap();
    let e = run_entry(&m);
    (m, e, vec![0]) // chibicc sp lead = 0
}

fn llvm_ir(name: &str, src: &str) -> (svm_ir::Module, u32, Vec<i64>) {
    let base = std::env::temp_dir().join(format!("fb_llvm_{name}"));
    let cf = base.with_extension("c");
    let bc = base.with_extension("bc");
    std::fs::write(&cf, src).unwrap();
    assert!(Command::new("clang")
        .args([
            "-O2",
            "-emit-llvm",
            "-c",
            "-fno-vectorize",
            "-fno-slp-vectorize"
        ])
        .arg(&cf)
        .arg("-o")
        .arg(&bc)
        .status()
        .unwrap()
        .success());
    let t = svm_llvm::translate_bc_path(&bc).expect("translate");
    let e = run_entry(&t.module);
    (t.module, e, vec![t.entry_sp as i64]) // llvm sp lead = entry_sp
}

fn run_entry(m: &svm_ir::Module) -> u32 {
    m.funcs
        .iter()
        .position(|f| {
            f.params == [svm_ir::ValType::I64, svm_ir::ValType::I64]
                && f.results == [svm_ir::ValType::I64]
        })
        .expect("run(i64,i64)->i64 entry") as u32
}

fn call(m: &svm_ir::Module, e: u32, lead: &[i64], n: i64) -> i64 {
    let mut a = lead.to_vec();
    a.push(n);
    match compile_and_run(m, e, &a) {
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
fn ns(m: &svm_ir::Module, e: u32, lead: &[i64]) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..6 {
        let b = per_call(25, || {
            black_box(call(m, e, lead, 2_000_000));
        });
        let s = per_call(25, || {
            black_box(call(m, e, lead, 1_000));
        });
        best = best.min((b - s) * 1e9 / (2_000_000 - 1_000) as f64);
    }
    best
}

#[test]
#[ignore = "benchmark; run explicitly with --nocapture --ignored"]
fn chibicc_vs_llvm_jit() {
    println!("\nSVM JIT, same C, two frontends (ns/iter; ratio = chibicc / llvm):");
    println!(
        "{:<12} {:>10} {:>10} {:>8}",
        "kernel", "chibicc", "llvm", "ratio"
    );
    for (name, src) in [
        ("alu", ALU),
        ("locals", LOCALS),
        ("irreducible", IRRED),
        ("hash", HASH),
    ] {
        let (cm, ce, cl) = chibicc_ir(name, src);
        let (lm, le, ll) = llvm_ir(name, src);
        // sanity: both frontends agree on the result
        assert_eq!(
            call(&cm, ce, &cl, 1000),
            call(&lm, le, &ll, 1000),
            "{name}: frontends disagree"
        );
        let c = ns(&cm, ce, &cl);
        let l = ns(&lm, le, &ll);
        println!("{name:<12} {c:>10.3} {l:>10.3} {:>7.2}x", c / l);
    }
}
