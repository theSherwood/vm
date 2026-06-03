//! Dependency-free benchmark harness (`DESIGN.md` §18, AGENTS.md "benchmark early").
//!
//! Measures throughput of the escape-TCB hot paths — decode, verify, interp — so we
//! can watch them over time and catch regressions when they're one commit old. Uses
//! only `std`. (A statistical harness like criterion can be swapped in later if the
//! extra build cost is judged worth it.)
//!
//! Run: `cargo run --release --bin svm-bench`

use std::time::Instant;

use svm::{encode, ir, verify};
use svm_interp::Value;

fn main() {
    // A small loop program: sum 0..N via a back-edge with block params.
    let src = r#"
func (i32) -> (i32) {
block0(v0: i32):
  v1 = i32.const 0
  v2 = i32.const 0
  br block1(v1, v2)
block1(v3: i32, v4: i32):
  v5 = i32.add v3 v4
  v6 = i32.const 1
  v7 = i32.add v4 v6
  br_if v7 block1(v5, v7) block2(v5)
block2(v8: i32):
  return v8
}
"#;

    let module = ir_from_text(src);
    let bytes = encode::encode_module(&module);
    println!(
        "module: {} funcs, {} encoded bytes",
        module.funcs.len(),
        bytes.len()
    );

    bench("decode", 200_000, || {
        let m = encode::decode_module(&bytes).expect("decode");
        std::hint::black_box(&m);
    });

    bench("verify", 200_000, || {
        let m = encode::decode_module(&bytes).unwrap();
        verify::verify_module(&m).expect("verify");
        std::hint::black_box(&m);
    });

    // Interp with bounded fuel; the loop runs a fixed number of iterations.
    bench("interp", 50_000, || {
        let mut fuel = 1_000u64;
        let r = svm_interp::run(&module, 0, &[Value::I32(0)], &mut fuel);
        std::hint::black_box(&r);
    });
}

fn ir_from_text(src: &str) -> ir::Module {
    svm::text::parse_module(src).expect("corpus program must parse")
}

fn bench(name: &str, iters: u64, mut f: impl FnMut()) {
    // Warm up.
    for _ in 0..(iters / 10).max(1) {
        f();
    }
    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    let elapsed = start.elapsed();
    let per = elapsed.as_nanos() as f64 / iters as f64;
    println!("{name:>8}: {iters} iters in {elapsed:?}  ({per:.1} ns/iter)");
}
