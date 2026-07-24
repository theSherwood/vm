//! ISSUES.md I36 slice 3 — the **JIT serve loop**: `svc.poll`/`svc.wait` run natively on the
//! JIT (the cap thunk's `serve_native` arm invoking compiled handler trampolines over the live
//! window) instead of `module_serves` folding the whole run to the tree-walk oracle.
//! Differential: every scenario runs on the tree-walker and the JIT with identical `Host`
//! setups and must agree exactly — results, completion cells, drain-once semantics, and the
//! final window bytes (the escape-oracle, as `jit_cap.rs`).
//!
//! The routing veto is pinned too: `bytecode::serve_qualifies` admits the pure serving module
//! (so the JIT genuinely serves — pre-slice, `svc.poll` answered `-EINVAL` here and the fold
//! hid it) and rejects the park-seam module (which keeps folding to the oracle).

use std::sync::Arc;

use svm_interp::{bytecode, run_capture_reserved_with_host, Host, Value};
use svm_ir::DEFAULT_RESERVED_LOG2;
use svm_jit::{JitOutcome, TrapKind};
use svm_run::jit_cap_run;
use svm_text::parse_module;
use svm_verify::verify_module;

/// The serving domain from the §3.6 corpus (as `svm-interp/tests/bytecode_svc.rs`): offer
/// "counter" op 0 = func 1 `bump(x) -> old + x` over the LIVE value at mem[0]; `main` seeds
/// mem[0] = 7, `svc.poll`s, and returns `served * 1000 + mem[0]`.
const SERVER: &str = r#"
memory 16
type 0 func (i64) -> (i64)
type 1 interface { bump: 0 }
export 0 interface "counter" 1 { bump: 1 }

func () -> (i64) {
block 0 () {
  va = i64.const 0
  vseed = i64.const 7
  i64.store va vseed
  vz = i32.const 0
  vn = cap.call 4294967295 9 () -> (i64) vz ()
  vafter = i64.load va
  vk = i64.const 1000
  vm = i64.mul vn vk
  vr = i64.add vm vafter
  return vr
  }
}

func (i64) -> (i64) {
block 0 (vx: i64) {
  va = i64.const 0
  vold = i64.load va
  vnew = i64.add vold vx
  i64.store va vnew
  return vold
  }
}
"#;

/// As `SERVER`, plus an unreachable third function containing a futex wait — a park-capable
/// seam that must fail the serve qualification (module-wide scan), keeping the oracle fold.
const SERVER_WITH_PARK_SEAM: &str = r#"
memory 16
type 0 func (i64) -> (i64)
type 1 interface { bump: 0 }
export 0 interface "counter" 1 { bump: 1 }

func () -> (i64) {
block 0 () {
  va = i64.const 0
  vseed = i64.const 7
  i64.store va vseed
  vz = i32.const 0
  vn = cap.call 4294967295 9 () -> (i64) vz ()
  vafter = i64.load va
  vk = i64.const 1000
  vm = i64.mul vn vk
  vr = i64.add vm vafter
  return vr
  }
}

func (i64) -> (i64) {
block 0 (vx: i64) {
  va = i64.const 0
  vold = i64.load va
  vnew = i64.add vold vx
  i64.store va vnew
  return vold
  }
}

func () -> (i32) {
block 0 () {
  va = i64.const 8
  vexp = i32.const 0
  vto = i64.const 1000
  vst = i32.atomic.wait va vexp vto
  return vst
  }
}
"#;

fn module(src: &str) -> Arc<svm_ir::Module> {
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");
    Arc::new(m)
}

/// Run one scenario on both backends — pre-enqueue each `(export, op, args)` dispatch on a
/// fresh, identically-configured `Host`, run entry 0, and assert result values, completion
/// cells, drain-once semantics, and final low-window bytes all agree. Returns the shared
/// `(result values, completion cells)` for the caller's pins.
#[allow(clippy::type_complexity)]
fn diff_serve(
    m: &Arc<svm_ir::Module>,
    dispatches: &[(u32, u32, Vec<i64>)],
) -> (Vec<i64>, Vec<Option<i64>>) {
    // Tree-walk oracle.
    let mut host_i = Host::new();
    host_i.set_self_module(m);
    let tickets_i: Vec<Option<u64>> = dispatches
        .iter()
        .map(|(e, o, a)| host_i.svc_enqueue(*e, *o, a.clone()))
        .collect();
    let mut fuel = 50_000_000u64;
    let (ires, imem) = run_capture_reserved_with_host(
        m,
        0,
        &[],
        &mut fuel,
        &[],
        DEFAULT_RESERVED_LOG2,
        &mut host_i,
    );
    let ivals: Vec<i64> = ires
        .expect("oracle run")
        .iter()
        .map(|v| match v {
            Value::I32(x) => *x as i64,
            Value::I64(x) => *x,
            other => panic!("scalar result expected, got {other:?}"),
        })
        .collect();
    let icells: Vec<Option<i64>> = tickets_i
        .iter()
        .map(|t| t.and_then(|t| host_i.svc_result(t)))
        .collect();
    let iagain: Vec<Option<i64>> = tickets_i
        .iter()
        .map(|t| t.and_then(|t| host_i.svc_result(t)))
        .collect();

    // JIT — a fresh Host configured identically (enqueues are deterministic, so tickets match).
    let mut host_j = Host::new();
    host_j.set_self_module(m);
    let tickets_j: Vec<Option<u64>> = dispatches
        .iter()
        .map(|(e, o, a)| host_j.svc_enqueue(*e, *o, a.clone()))
        .collect();
    assert_eq!(
        tickets_i, tickets_j,
        "identical setups mint identical tickets"
    );
    let (jout, jmem) =
        jit_cap_run(m, 0, &[], &[], DEFAULT_RESERVED_LOG2, 0, &mut host_j).expect("jit run");
    let jvals = match jout {
        JitOutcome::Returned(slots) => slots,
        other => panic!("JIT run must return cleanly, got {other:?}"),
    };
    let jcells: Vec<Option<i64>> = tickets_j
        .iter()
        .map(|t| t.and_then(|t| host_j.svc_result(t)))
        .collect();
    let jagain: Vec<Option<i64>> = tickets_j
        .iter()
        .map(|t| t.and_then(|t| host_j.svc_result(t)))
        .collect();

    assert_eq!(ivals, jvals, "run result must match for {dispatches:?}");
    assert_eq!(
        icells, jcells,
        "completion cells must match for {dispatches:?}"
    );
    assert_eq!(iagain, jagain, "cells drain once on both backends");
    assert_eq!(imem, jmem, "final memory must be byte-identical");
    (jvals, jcells)
}

/// Every core serve scenario agrees exactly across the backends, and the serving module
/// qualifies for native JIT serving (no oracle fold — the point of the slice: pre-slice,
/// this exact program's `svc.poll` answered `-EINVAL` on a raw JIT run).
#[test]
fn jit_serve_loop_matches_the_tree_walker() {
    let m = module(SERVER);
    assert!(
        bytecode::serve_qualifies(&m.funcs),
        "the pure serving module must qualify — otherwise this differential only re-tests the fold"
    );
    let cases: &[&[(u32, u32, Vec<i64>)]] = &[
        &[],                                       // empty poll → 0 served, seed untouched
        &[(0, 0, vec![5]), (0, 0, vec![30])],      // two bumps, ordered
        &[(0, 0, vec![1, 2, 3]), (0, 0, vec![5])], // arity mismatch errnos, serving continues
    ];
    for dispatches in cases {
        diff_serve(&m, dispatches);
    }
    // Pin the headline values too (not just cross-backend equality).
    let (vals, cells) = diff_serve(&m, &[(0, 0, vec![5]), (0, 0, vec![30])]);
    assert_eq!(vals, vec![2042]);
    assert_eq!(cells, vec![Some(7), Some(12)]);
}

/// `svc.wait` with work already queued behaves like `svc.poll` (progress ⇒ deliver the count,
/// no park) — natively on the JIT, equal to the tree-walker.
#[test]
fn a_jit_svc_wait_with_queued_work_serves_and_returns() {
    let src = SERVER.replace(
        "vn = cap.call 4294967295 9 () -> (i64) vz ()",
        "vn = cap.call 4294967295 10 () -> (i64) vz ()",
    );
    let m = module(&src);
    assert!(bytecode::serve_qualifies(&m.funcs));
    let (vals, cells) = diff_serve(&m, &[(0, 0, vec![5]), (0, 0, vec![30])]);
    assert_eq!(vals, vec![2042]);
    assert_eq!(cells, vec![Some(7), Some(12)]);
}

/// `svc.wait` with an empty queue and no progress **fails closed** on the JIT (`ThreadFault`):
/// caller-side parking is not yet native (the op-14 fold stands), so no enqueuer can exist
/// mid-run and the park could never be woken — the deterministic-deadlock answer, the same the
/// bytecode drive gives. JIT-only pin (the oracle's scheduler would park the domain instead;
/// the differential never runs hang cases — the accepted fail-closed divergence, ISSUES.md I36).
#[test]
fn a_jit_svc_wait_with_an_empty_queue_fails_closed() {
    let src = SERVER.replace(
        "vn = cap.call 4294967295 9 () -> (i64) vz ()",
        "vn = cap.call 4294967295 10 () -> (i64) vz ()",
    );
    let m = module(&src);
    let mut host = Host::new();
    host.set_self_module(&m);
    let (jout, _) =
        jit_cap_run(&m, 0, &[], &[], DEFAULT_RESERVED_LOG2, 0, &mut host).expect("jit run");
    assert_eq!(
        jout,
        JitOutcome::Trapped(TrapKind::ThreadFault),
        "an unwakeable svc.wait park is a deterministic deadlock — fail closed"
    );
}

/// The routing veto: a park-capable seam anywhere in a serving module (here a futex wait in a
/// function nothing calls) fails the qualification, so svm-run keeps folding it to the oracle —
/// the same module-wide predicate the bytecode engine's compile veto applies (one definition).
#[test]
fn a_park_seam_fails_the_serve_qualification() {
    let m = module(SERVER_WITH_PARK_SEAM);
    assert!(
        !bytecode::serve_qualifies(&m.funcs),
        "a serving module with a futex wait must keep the oracle fold (module-wide veto)"
    );
    // And the pure-compute corpus without any service point doesn't qualify either — there is
    // nothing to serve natively (`module_serves` never routes it here).
    let plain = module(
        "memory 12\nfunc () -> (i64) {\nblock 0 () {\n  v = i64.const 7\n  return v\n  }\n}\n",
    );
    assert!(!bytecode::serve_qualifies(&plain.funcs));
}
