//! Regression for the §9/D45 **allocation-free clock fast path** (ISSUES.md I12). `run_powerbox`
//! wires `fast_cap_resolver`, which devirtualizes `Clock.now()` to `Host::fast_clock_now` — an
//! authority-checked, `Vec`-free inline read. This pins that the fast path produces the **same**
//! results as the generic `cap_thunk` path and the tree-walker (so interp == JIT still holds), and
//! that a forged handle still faults.

use svm_interp::{run_with_host, Host, Value};
use svm_jit::{compile_and_run_with_host, compile_and_run_with_host_fast, JitOutcome, Quota};
use svm_run::{cap_thunk, fast_cap_resolver};

// `(i32 clk) -> i64`: call `Clock.now()` (0-arg, so the fast resolver `(CLOCK,0,0,1)` engages) twice
// and return the delta. The mock clock starts at 0 and increments once per read, so this is 1.
const SRC: &str = r#"
func (i32) -> (i64) {
block0(v0: i32):
  v1 = cap.call 2 0 () -> (i64) v0 ()
  v2 = cap.call 2 0 () -> (i64) v0 ()
  v3 = i64.sub v2 v1
  return v3
}
"#;

fn module() -> svm_ir::Module {
    let m = svm_text::parse_module(SRC).expect("parse");
    svm_verify::verify_module(&m).expect("verify");
    m
}

#[test]
fn clock_fast_path_matches_generic_and_interp() {
    let m = module();

    // Tree-walker (full Host dispatch).
    let mut host = Host::new();
    let clk = host.grant_clock();
    let mut fuel = u64::MAX;
    let tw = run_with_host(&m, 0, &[Value::I32(clk)], &mut fuel, &mut host).expect("interp");
    assert_eq!(tw, vec![Value::I64(1)], "tree-walk clock delta");

    // JIT, generic cap_thunk.
    let mut host = Host::new();
    let clk = host.grant_clock();
    let ctx = &mut host as *mut Host as *mut core::ffi::c_void;
    let gen = compile_and_run_with_host(&m, 0, &[clk as i64], cap_thunk, ctx).expect("jit");
    assert_eq!(
        gen,
        JitOutcome::Returned(vec![1]),
        "jit generic clock delta"
    );

    // JIT, D45 fast resolver — the inline allocation-free path.
    let mut host = Host::new();
    let clk = host.grant_clock();
    let ctx = &mut host as *mut Host as *mut core::ffi::c_void;
    let fast = compile_and_run_with_host_fast(
        &m,
        0,
        &[clk as i64],
        cap_thunk,
        ctx,
        fast_cap_resolver,
        Quota::default(),
    )
    .expect("jit fast");
    assert_eq!(fast, JitOutcome::Returned(vec![1]), "jit fast clock delta");
}

#[test]
fn clock_fast_path_faults_on_forged_handle() {
    // A handle that was never granted must fault on the fast path exactly like the generic one
    // (authority is preserved: `resolve` rejects a wrong generation/type/closed slot).
    let m = module();
    let mut host = Host::new();
    let _real = host.grant_clock();
    let forged = 0x7fff_fffe; // not a valid (slot, generation) for a CLOCK binding
    let ctx = &mut host as *mut Host as *mut core::ffi::c_void;
    let r = compile_and_run_with_host_fast(
        &m,
        0,
        &[forged],
        cap_thunk,
        ctx,
        fast_cap_resolver,
        Quota::default(),
    );
    // A CapFault surfaces as a trap (Err) or a Trapped outcome — either way, not a clean return.
    let faulted = !matches!(r, Ok(JitOutcome::Returned(_)));
    assert!(
        faulted,
        "forged clock handle must not return cleanly: {r:?}"
    );
}
