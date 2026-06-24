//! **Capability-call (host-boundary) overhead.** `cap.call` is how a guest reaches the host — every
//! I/O, clock, spawn, and the durable safepoint story flows through it — yet it's invisible in the
//! compute benchmarks. This driver measures the per-`cap.call` cost across the three engines by
//! timing a loop whose body is a single `cap.call` to the cheapest host capability (the clock read),
//! minus an identical loop with no `cap.call` (so only the host-boundary crossing remains).
//!
//! The clock cap is serviced the same way each engine reaches the host: the tree-walker and bytecode
//! engine dispatch through the in-process `Host`; the JIT is timed twice — through the generic
//! `svm_run::cap_thunk` (§9 trampoline: marshalling + indirect dispatch) and through the §9/D45
//! devirtualized **fast resolver** that `run_powerbox` wires by default for known caps. Comparing the
//! two isolates how much the fast path actually buys (spoiler: for a cheap 0-arg cap, ~nothing — both
//! share `Host::cap_dispatch_slots`; see ISSUES.md I12).
//!
//! Run: cd crates/svm-llvm && cargo run --release --example cap_call

use std::ffi::c_void;
use std::hint::black_box;
use std::time::Instant;

use svm_interp::{bytecode, Host, Value};
use svm_ir::Module;

// `(i32 clk, i32 n) -> i64`: loop `n` times, each calling the clock cap (type_id 2, op 0) and folding
// the result. The clock handle and `n` are threaded as block params (cross-block values must be).
const CAPCALL_SRC: &str = r#"
func (i32, i32) -> (i64) {
block0(v0: i32, v1: i32):
  v2 = i64.const 0
  v3 = i32.const 0
  br block1(v3, v2, v0, v1)
block1(v4: i32, v5: i64, v6: i32, v7: i32):
  v8 = i32.const 0
  v9 = cap.call 2 0 (i32) -> (i64) v6 (v8)
  v10 = i64.add v5 v9
  v11 = i32.const 1
  v12 = i32.add v4 v11
  v13 = i32.lt_s v12 v7
  br_if v13 block1(v12, v10, v6, v7) block2(v10)
block2(v14: i64):
  return v14
}
"#;

// The same loop with the `cap.call` replaced by one loop-carried `i64.add` (acc = acc+acc) — an
// equally cheap, fold-resistant body — so subtracting its per-iter leaves just the host-boundary cost.
const BASELINE_SRC: &str = r#"
func (i32) -> (i64) {
block0(v0: i32):
  v1 = i64.const 1
  v2 = i32.const 0
  br block1(v2, v1, v0)
block1(v3: i32, v4: i64, v5: i32):
  v6 = i64.add v4 v4
  v7 = i32.const 1
  v8 = i32.add v3 v7
  v9 = i32.lt_s v8 v5
  br_if v9 block1(v8, v6, v5) block2(v6)
block2(v10: i64):
  return v10
}
"#;

const SMALL: i32 = 1_000;
const LARGE: i32 = 2_000_000;

fn parse(src: &str) -> Module {
    let m = svm_text::parse_module(src).expect("parse");
    svm_verify::verify_module(&m).expect("verify");
    m
}

/// Per-iteration ns via the two-point `(T(LARGE) − T(SMALL)) / Δn` min-of-reps fit.
fn per_iter(mut run_n: impl FnMut(i32)) -> f64 {
    let mut m = |n: i32| {
        run_n(n);
        let mut best = f64::MAX;
        for _ in 0..9 {
            let t = Instant::now();
            run_n(n);
            best = best.min(t.elapsed().as_nanos() as f64);
        }
        best
    };
    (m(LARGE) - m(SMALL)) / (LARGE - SMALL) as f64
}

/// Time a module that takes `(clk, n)` (cap version) or `(n)` (baseline) on all engines.
/// `has_cap` selects the arg shape and whether a clock is granted. Returns
/// `(tree-walk, bytecode, jit-generic, jit-fast)` per-iter ns — the JIT both through the generic
/// `cap_thunk` and through the §9/D45 devirtualized **fast resolver** (`run_powerbox`'s production path).
fn time_engines(m: &Module, has_cap: bool) -> (f64, f64, f64, f64) {
    let args_iv = |clk: i32, n: i32| -> Vec<Value> {
        if has_cap {
            vec![Value::I32(clk), Value::I32(n)]
        } else {
            vec![Value::I32(n)]
        }
    };
    let args_i64 = |clk: i32, n: i32| -> Vec<i64> {
        if has_cap {
            vec![clk as i64, n as i64]
        } else {
            vec![n as i64]
        }
    };

    let tw = per_iter(|n| {
        let mut host = Host::new();
        let clk = host.grant_clock();
        let mut fuel = u64::MAX;
        let r = svm_interp::run_with_host(m, 0, &args_iv(clk, n), &mut fuel, &mut host);
        black_box(&r);
    });
    let bc = per_iter(|n| {
        let mut host = Host::new();
        let clk = host.grant_clock();
        let mut fuel = u64::MAX;
        let r = bytecode::compile_and_run_with_host(m, 0, &args_iv(clk, n), &mut fuel, &mut host);
        black_box(&r);
    });
    let jit = per_iter(|n| {
        // Generic path: every `cap.call` goes through the runtime `cap_thunk` (marshalling + dispatch).
        let mut host = Host::new();
        let clk = host.grant_clock();
        let ctx = &mut host as *mut Host as *mut c_void;
        let r =
            svm_jit::compile_and_run_with_host(m, 0, &args_i64(clk, n), svm_run::cap_thunk, ctx);
        black_box(&r);
    });
    let jit_fast = per_iter(|n| {
        // Fast path: the §9/D45 resolver devirtualizes known caps (the clock is one) to a
        // register-to-register call — what `run_powerbox` actually uses in production.
        let mut host = Host::new();
        let clk = host.grant_clock();
        let ctx = &mut host as *mut Host as *mut c_void;
        let r = svm_jit::compile_and_run_with_host_fast(
            m,
            0,
            &args_i64(clk, n),
            svm_run::cap_thunk,
            ctx,
            svm_run::fast_cap_resolver,
            svm_jit::Quota::default(),
        );
        black_box(&r);
    });
    (tw, bc, jit, jit_fast)
}

fn main() {
    let cap = parse(CAPCALL_SRC);
    let base = parse(BASELINE_SRC);

    let (tw_c, bc_c, jit_c, jitf_c) = time_engines(&cap, true);
    let (tw_b, bc_b, jit_b, jitf_b) = time_engines(&base, false);

    println!(
        "{:<14} {:>12} {:>12} {:>14} {:>14}",
        "engine", "cap loop", "baseline", "cap.call cost", "vs baseline"
    );
    println!(
        "{:<14} {:>12} {:>12} {:>14} {:>14}",
        "", "(ns/iter)", "(ns/iter)", "(ns/call)", ""
    );
    for (name, c, b) in [
        ("tree-walk", tw_c, tw_b),
        ("bytecode", bc_c, bc_b),
        ("jit (generic)", jit_c, jit_b),
        ("jit (fast/D45)", jitf_c, jitf_b),
    ] {
        let cost = c - b;
        println!(
            "{name:<14} {c:>12.2} {b:>12.2} {cost:>14.2} {:>13.1}x",
            c / b.max(0.001)
        );
    }
    println!(
        "\n(cap.call cost = cap-loop per-iter − a no-cap loop of the same shape, so it isolates the\n \
         host-boundary crossing. `jit (generic)` routes through the runtime `cap_thunk` (marshalling +\n \
         indirect dispatch); `jit (fast/D45)` uses the devirtualized resolver `run_powerbox` wires by\n \
         default for known caps (clock/blocking) — register-to-register, the production cost. The\n \
         interpreters dispatch the clock cap through the in-process Host. The clock read is the cheapest\n \
         cap — window-touching caps (I/O, spawn) stay on the generic path and add their own work.)"
    );
}
