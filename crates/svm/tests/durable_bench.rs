//! Perf probe for the durable transform (run on demand, not in CI):
//!
//!   cargo test -p svm --test durable_bench -- --ignored --nocapture
//!
//! Reports instrumented-IR size, JIT compile+run time, and interp freeze+thaw time, on a
//! "dead-heavy" guest — a prefix chain whose intermediates are dead across the `cap.call`
//! — so the current over-capture (spill *every* value visible at the op) vs. a future
//! minimal live-set (spill only values used after the op) shows its largest delta. Capture
//! the numbers here before/after the live-set change to see the effect on JIT/compile time.

use std::time::Instant;

use svm_durable::{
    init_durable_window, transform_module_assume_confined, write_state, STATE_REWINDING,
    STATE_UNWINDING,
};
use svm_interp::{run_capture_reserved_with_host, Host, Value};
use svm_ir::{Memory, Module};

const SIZE_LOG2: u8 = 18;
const WINDOW: usize = 1 << SIZE_LOG2;

/// `k` i64 values chained so each feeds only the next (intermediates die before the call),
/// then a `cap.call`, then a tail using just the last chain value + the call result. Over-
/// capture spills all `k`+; minimal spills ~2.
fn dead_heavy_src(k: usize) -> String {
    let mut s = String::from("func (i32) -> (i64) {\nblock0(v0: i32):\n  v1 = i64.const 1\n");
    for i in 2..=k {
        s += &format!("  v{i} = i64.add v{} v{}\n", i - 1, i - 1);
    }
    let arg = k + 1;
    let cap = k + 2;
    let res = k + 3;
    s += &format!("  v{arg} = i32.const 0\n");
    s += &format!("  v{cap} = cap.call 2 0 (i32) -> (i64) v0 (v{arg})\n");
    s += &format!("  v{res} = i64.add v{k} v{cap}\n");
    s += &format!("  return v{res}\n}}\n");
    s
}

fn instrument(src: &str) -> Module {
    let mut m = svm_text::parse_module(src).expect("parse");
    m.memory = Some(Memory {
        size_log2: SIZE_LOG2,
    });
    let inst = transform_module_assume_confined(&m).expect("transform");
    svm_verify::verify_module(&inst).expect("verify");
    inst
}

fn inst_count(m: &Module) -> usize {
    m.funcs
        .iter()
        .flat_map(|f| f.blocks.iter())
        .map(|b| b.insts.len())
        .sum()
}

fn time<F: FnMut()>(iters: u32, mut f: F) -> f64 {
    for _ in 0..(iters / 10).max(1) {
        f();
    }
    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    start.elapsed().as_nanos() as f64 / iters as f64
}

#[test]
#[ignore = "perf probe; run with --ignored --nocapture"]
fn durable_overhead_probe() {
    for &k in &[8usize, 24, 64] {
        let src = dead_heavy_src(k);

        // transform time (the svm-durable pass itself)
        let mut m = svm_text::parse_module(&src).unwrap();
        m.memory = Some(Memory {
            size_log2: SIZE_LOG2,
        });
        let t_transform = time(2000, || {
            let _ = std::hint::black_box(transform_module_assume_confined(&m).unwrap());
        });

        let inst = instrument(&src);
        let blocks = inst.funcs[0].blocks.len();
        let insts = inst_count(&inst);
        let bytes = svm::encode::encode_module(&inst).len();

        // JIT compile+run (NORMAL window). For a tiny guest, compile dominates.
        let t_jit = time(300, || {
            let mut h = Host::new();
            let clk = h.grant_clock();
            let slots = [clk as i64];
            let win = init_durable_window(WINDOW);
            let r = svm_jit::compile_and_run_capture_reserved_with_host(
                &inst,
                0,
                &slots,
                &win,
                SIZE_LOG2,
                svm_run::cap_thunk,
                &mut h as *mut Host as *mut core::ffi::c_void,
            );
            std::hint::black_box(r.unwrap());
        });

        // interp freeze (UNWINDING) + thaw (REWINDING) round-trip.
        let t_rt = time(20_000, || {
            let mut h = Host::new();
            h.clock_ns = 42;
            let clk = h.grant_clock();
            let mut win = init_durable_window(WINDOW);
            write_state(&mut win, STATE_UNWINDING);
            let mut fuel = 1_000_000u64;
            let (_, snap) = run_capture_reserved_with_host(
                &inst,
                0,
                &[Value::I32(clk)],
                &mut fuel,
                &win,
                SIZE_LOG2,
                &mut h,
            );
            let mut win = snap;
            write_state(&mut win, STATE_REWINDING);
            let mut h2 = Host::new();
            h2.clock_ns = h.clock_ns;
            let clk2 = h2.grant_clock();
            let mut fuel = 1_000_000u64;
            let r = run_capture_reserved_with_host(
                &inst,
                0,
                &[Value::I32(clk2)],
                &mut fuel,
                &win,
                SIZE_LOG2,
                &mut h2,
            );
            let _ = std::hint::black_box(r);
        });

        println!(
            "k={k:<3} blocks={blocks:<3} insts={insts:<4} bytes={bytes:<5} | transform={t_transform:>7.0}ns  jit(compile+run)={t_jit:>9.0}ns  interp_roundtrip={t_rt:>8.0}ns"
        );
    }
}
