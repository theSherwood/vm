//! Browser wasm-JIT **threads** slice — native proof of the `TierUp` seam on the resumable `Vcpu`.
//!
//! When a vCPU carries a JIT-eligibility bitmap ([`Vcpu::with_jit_eligible`]), a direct `Call` to an
//! eligible module-0 function surfaces as [`VcpuEvent::TierUp`] instead of interpreting the callee —
//! the browser host then runs the emitted `f{func}` region on its Worker and delivers the results
//! back ([`Vcpu::deliver_tierup`]). This test stands in for that host **without any wasm**: it
//! services each `TierUp` by computing the callee on a standalone bytecode run (exactly what the
//! emitted region computes), and asserts the whole-vCPU result is **identical** to a pure-interpreter
//! run of the same guest. So the tier-up marshalling (args in, results out, resume) is exact.

use std::sync::Arc;
use svm_interp::{bytecode, Region, Trap, Value};
use svm_text::parse_module;

// func 0 sums `f1(i)` over `0..n`; func 1 is a pure-compute leaf `f(x) = x*3 + 7` — the tier-up
// target. No memory, all-i64 signature: exactly the shape the browser tiers up.
const SRC: &str = r#"
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
  v10 = call 1 (v9)
  v11 = i64.add v8 v10
  v12 = i64.const 1
  v13 = i64.add v9 v12
  br 1(v7, v11, v13)
}
block 3 (v14: i64) {
  return v14
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 3
  v2 = i64.mul v0 v1
  v3 = i64.const 7
  v4 = i64.add v2 v3
  return v4
  }
}
"#;

/// A fresh 64 KiB shared window for a root vCPU (this guest touches no memory, but `new_root` still
/// wants a backing region). Leaked for the test's lifetime.
fn window() -> Arc<Region> {
    let size = 1usize << 16;
    let layout = std::alloc::Layout::from_size_align(size, 8).unwrap();
    // SAFETY: non-zero 8-aligned layout; leaked for the process — never freed, so no aliasing.
    let base = unsafe { std::alloc::alloc_zeroed(layout) };
    assert!(!base.is_null());
    // SAFETY: `size` valid 8-aligned bytes, owned here and never freed.
    Arc::new(unsafe { Region::shared(base, size as u64) })
}

fn pure_interp(m: &svm_ir::Module, n: i64) -> Result<Vec<Value>, Trap> {
    let mut fuel = u64::MAX;
    bytecode::compile_and_run(m, 0, &[Value::I64(n)], &mut fuel).expect("supported")
}

/// Drive the root vCPU with tier-up enabled, emulating the browser host: each `TierUp(func, argv)`
/// is serviced by running `func` standalone (what the emitted region computes) and delivering its
/// i64 results back. Returns the whole-vCPU result.
fn tierup_run(
    m: &svm_ir::Module,
    prog: &bytecode::VcpuProgram,
    n: i64,
) -> (Result<Vec<Value>, Trap>, u32) {
    let back = window();
    // Only func 1 is eligible (func 0 is the interp-driven caller).
    let eligible: Arc<[bool]> = Arc::from(vec![false, true]);
    let mut vcpu = bytecode::Vcpu::new_root(prog, 0, &[Value::I64(n)], back, &[])
        .expect("root")
        .with_jit_eligible(eligible);
    let mut tierups = 0u32;
    loop {
        match vcpu.run() {
            bytecode::VcpuEvent::Done(vals) => return (Ok(vals), tierups),
            bytecode::VcpuEvent::Trapped(t) => return (Err(t), tierups),
            bytecode::VcpuEvent::TierUp { func, argv } => {
                tierups += 1;
                // Emulate `f{func}(win, env, ...argv)`: run the callee standalone over its i64 args.
                let args: Vec<Value> = argv.iter().map(|&s| Value::I64(s)).collect();
                let mut fuel = u64::MAX;
                match bytecode::compile_and_run(m, func, &args, &mut fuel).expect("supported") {
                    Ok(vals) => {
                        let slots: Vec<i64> = vals
                            .iter()
                            .map(|v| match v {
                                Value::I64(x) => *x,
                                Value::I32(x) => *x as i64,
                                _ => panic!("non-integer tier-up result"),
                            })
                            .collect();
                        vcpu.deliver_tierup(&slots);
                    }
                    Err(t) => vcpu.deliver_tierup_trap(t),
                }
            }
            _ => panic!("unexpected event (this guest only tiers up)"),
        }
    }
}

#[test]
fn tierup_matches_pure_interp() {
    let m = parse_module(SRC).unwrap();
    svm_verify::verify_module(&m).expect("verify");
    let prog = bytecode::VcpuProgram::compile(&m).expect("compile");
    for n in [0i64, 1, 2, 5, 10, 100, 1000] {
        let want = pure_interp(&m, n);
        let (got, tierups) = tierup_run(&m, &prog, n);
        assert_eq!(want, got, "tier-up run diverged from pure interp at n={n}");
        // Non-vacuity: the loop must have tiered up exactly once per iteration (func 1 is eligible).
        assert_eq!(
            tierups as i64,
            n.max(0),
            "expected {n} tier-ups, got {tierups}"
        );
    }
    // Spot-check the actual value: sum_{i=0}^{n-1} (3i + 7) for n=5 = 3*(0+1+2+3+4) + 7*5 = 30+35 = 65.
    assert_eq!(pure_interp(&m, 5), Ok(vec![Value::I64(65)]));
}

/// A tier-up region that **traps** must surface exactly where the interpreter would: func 1 here
/// divides by `(x - 3)`, trapping at `x == 3`. The tier-up run must trap iff the pure-interp run does.
const SRC_TRAP: &str = r#"
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = call 1 (v0)
  return v1
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 3
  v2 = i64.sub v0 v1
  v3 = i64.const 100
  v4 = i64.div_s v3 v2
  return v4
  }
}
"#;

#[test]
fn tierup_trap_parity() {
    let m = parse_module(SRC_TRAP).unwrap();
    svm_verify::verify_module(&m).expect("verify");
    let prog = bytecode::VcpuProgram::compile(&m).expect("compile");
    for x in [0i64, 1, 3, 4, 7] {
        let want = pure_interp(&m, x);
        let (got, _) = tierup_run(&m, &prog, x);
        assert_eq!(
            want.is_err(),
            got.is_err(),
            "tier-up trap parity broke at x={x} (interp err={:?})",
            want.is_err()
        );
        assert_eq!(want, got, "tier-up value/trap diverged at x={x}");
        if x == 3 {
            assert!(want.is_err(), "x=3 must trap (div by zero)");
        }
    }
}
