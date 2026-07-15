//! [`bytecode::SharedProgram`] — a module compiled **once** for repeated cross-tier runs over a
//! caller-provided shared window (the browser wasm-JIT reactor's per-frame `env.call_interp` path,
//! where recompiling the whole module per bounce otherwise dominates the frame). This gates two
//! properties: (1) a cached `run_over` gives the **same** result as the one-shot
//! `compile_and_run_over_shared_with_host`, and (2) repeated calls share one live window — a value
//! written by one call is read back by the next (proving the cache doesn't reset the window and the
//! `Arc`-cloned dispatch source stays correct across calls).

use std::sync::Arc;

use svm_interp::{bytecode, Host, Region, Value};
use svm_text::parse_module;

// func 0 `store(x)`: writes x to mem[0], returns x. func 1 `load()`: returns mem[0].
const SRC: &str = r#"
memory 16
func (i64) -> (i64) {
block0(v0: i64):
  va = i64.const 0
  i64.store va v0
  return v0
}
func () -> (i64) {
block0():
  va = i64.const 0
  vr = i64.load va
  return vr
}
"#;

/// A fresh zeroed shared window of `size` bytes (leaked — lives for the test).
fn window(size: usize) -> Arc<Region> {
    let layout = std::alloc::Layout::from_size_align(size, 8).unwrap();
    // SAFETY: non-zero layout; `base` is `size` valid 8-aligned bytes, leaked so it outlives `back`.
    let base = unsafe { std::alloc::alloc_zeroed(layout) };
    assert!(!base.is_null());
    Arc::new(unsafe { Region::shared(base, size as u64) })
}

#[test]
fn cached_run_persists_across_calls() {
    let m = parse_module(SRC).expect("parse");
    let prog = bytecode::SharedProgram::compile(&m).expect("compile");
    let back = window(1 << 16);
    let mut host = Host::new();
    let mut fuel = u64::MAX;

    // Call 0: store(42) over the shared window.
    let r = prog
        .run_over(
            0,
            &[Value::I64(42)],
            &mut fuel,
            back.clone(),
            &mut host,
            false,
        )
        .expect("store runs");
    assert_eq!(r.first(), Some(&Value::I64(42)));

    // Call 1: load() — must read the 42 the previous call wrote (same live window, no recompile/reset).
    let r = prog
        .run_over(1, &[], &mut fuel, back.clone(), &mut host, false)
        .expect("load runs");
    assert_eq!(
        r.first(),
        Some(&Value::I64(42)),
        "the cached run must read back the prior call's write over the shared window"
    );

    // A third call overwrites, and a fourth reads the new value — the cache is reusable indefinitely.
    prog.run_over(
        0,
        &[Value::I64(-7)],
        &mut fuel,
        back.clone(),
        &mut host,
        false,
    )
    .expect("store runs");
    let r = prog
        .run_over(1, &[], &mut fuel, back.clone(), &mut host, false)
        .expect("load runs");
    assert_eq!(r.first(), Some(&Value::I64(-7)));
}

#[test]
fn cached_run_matches_the_one_shot() {
    let m = parse_module(SRC).expect("parse");
    let prog = bytecode::SharedProgram::compile(&m).expect("compile");

    for &arg in &[0i64, 1, 42, -5, 1000] {
        // Cached: store(arg) then load(), over one shared window.
        let cached = {
            let back = window(1 << 16);
            let mut host = Host::new();
            let mut fuel = u64::MAX;
            prog.run_over(
                0,
                &[Value::I64(arg)],
                &mut fuel,
                back.clone(),
                &mut host,
                false,
            )
            .unwrap();
            prog.run_over(1, &[], &mut fuel, back.clone(), &mut host, false)
                .unwrap()
        };
        // One-shot: the same two calls over a fresh window (recompiling each time).
        let oneshot = {
            let back = window(1 << 16);
            let mut host = Host::new();
            let mut fuel = u64::MAX;
            bytecode::compile_and_run_over_shared_with_host(
                &m,
                0,
                &[Value::I64(arg)],
                &mut fuel,
                back.clone(),
                &mut host,
                false,
            )
            .unwrap()
            .unwrap();
            bytecode::compile_and_run_over_shared_with_host(
                &m,
                1,
                &[],
                &mut fuel,
                back.clone(),
                &mut host,
                false,
            )
            .unwrap()
            .unwrap()
        };
        assert_eq!(cached, oneshot, "cached run != one-shot for arg {arg}");
    }
}
