//! ISSUES.md I36 slices 1 + 2 — the **bytecode serve loop**: `svc.poll`/`svc.wait` run natively
//! on the bytecode engine (handlers as rewind-linked activations over the one world) instead of
//! folding the whole module back to the tree-walk oracle, and the caller side rides along —
//! `child_offer` mints, live calls enqueue + park on their ticket, the callee's serve settles
//! them awake. Differential: every scenario runs on both entries and must agree exactly —
//! results, completion cells, drain-once semantics.
//!
//! The qualification veto is pinned too: a serving module with any park-capable seam (here a
//! futex wait; the corpus' handler-park modules are the richer cases) must *decline* the compile
//! — `compile_module` returns `None` — and the fast entry then falls back to the tree-walker,
//! which serves.

use std::sync::Arc;
use svm_interp::{bytecode, run_with_host, run_with_host_fast, Host, Value, SVC_QUEUE_CAP};

/// The serving domain from the §3.6 slice-2 corpus, verbatim: offer "counter" op 0 = func 1
/// `bump(x) -> old + x` over the LIVE value at mem[0]; `main` seeds mem[0] = 7, `svc.poll`s, and
/// returns `served * 1000 + mem[0]`.
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

/// As `SERVER`, plus an unreachable third function containing a futex wait — a park-capable seam
/// that must veto the native serve compile (module-wide scan, so even dead code counts).
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
    let m = svm_text::parse_module(src).expect("parse");
    svm_verify::verify_module(&m).expect("verify");
    Arc::new(m)
}

/// Run one scenario on a given entry: enqueue each `(export, op, args)` dispatch, run, and return
/// `(run result, per-ticket completion cells, a second drain of the same tickets)`.
type Entry = fn(
    &svm_ir::Module,
    svm_ir::FuncIdx,
    &[Value],
    &mut u64,
    &mut Host,
) -> Result<Vec<Value>, svm_interp::Trap>;

#[allow(clippy::type_complexity)]
fn scenario(
    m: &Arc<svm_ir::Module>,
    dispatches: &[(u32, u32, Vec<i64>)],
    entry: Entry,
) -> (
    Result<Vec<Value>, svm_interp::Trap>,
    Vec<Option<i64>>,
    Vec<Option<i64>>,
) {
    let mut host = Host::new();
    host.set_self_module(m);
    let tickets: Vec<Option<u64>> = dispatches
        .iter()
        .map(|(e, o, a)| host.svc_enqueue(*e, *o, a.clone()))
        .collect();
    let mut fuel = u64::MAX;
    let r = entry(m, 0, &[], &mut fuel, &mut host);
    let cells: Vec<Option<i64>> = tickets
        .iter()
        .map(|t| t.and_then(|t| host.svc_result(t)))
        .collect();
    let again: Vec<Option<i64>> = tickets
        .iter()
        .map(|t| t.and_then(|t| host.svc_result(t)))
        .collect();
    (r, cells, again)
}

/// Every core serve scenario agrees exactly across the two entries, and the serving module
/// compiles natively (no oracle fallback — the point of the slice).
#[test]
fn bytecode_serve_loop_matches_the_tree_walker() {
    let m = module(SERVER);
    assert!(
        bytecode::compile_module(&m.funcs).is_some(),
        "the pure serving module must be admitted natively — otherwise this differential \
         only re-tests the fallback"
    );
    let cases: &[&[(u32, u32, Vec<i64>)]] = &[
        &[],                                       // empty poll → 0 served, seed untouched
        &[(0, 0, vec![5]), (0, 0, vec![30])],      // two bumps, ordered
        &[(0, 0, vec![1, 2, 3]), (0, 0, vec![5])], // arity mismatch errnos, serving continues
    ];
    for dispatches in cases {
        let (ri, ci, ai) = scenario(&m, dispatches, run_with_host);
        let (rf, cf, af) = scenario(&m, dispatches, run_with_host_fast);
        assert_eq!(ri, rf, "run result must match for {dispatches:?}");
        assert_eq!(ci, cf, "completion cells must match for {dispatches:?}");
        assert_eq!(
            ai, af,
            "cells drain once on both entries for {dispatches:?}"
        );
    }
    // Pin the headline values too (not just cross-entry equality): served*1000 + live counter.
    let (rf, cf, _) = scenario(&m, &[(0, 0, vec![5]), (0, 0, vec![30])], run_with_host_fast);
    assert_eq!(rf, Ok(vec![Value::I64(2042)]));
    assert_eq!(cf, vec![Some(7), Some(12)]);
}

/// A full queue refuses at enqueue on the shared Host — engine-independent, but run the drain on
/// the fast entry to prove a maximal queue serves natively end to end.
#[test]
fn bytecode_serves_a_full_queue() {
    let m = module(SERVER);
    let mut host = Host::new();
    host.set_self_module(&m);
    for i in 0..SVC_QUEUE_CAP {
        assert!(
            host.svc_enqueue(0, 0, vec![i as i64]).is_some(),
            "under cap"
        );
    }
    assert_eq!(host.svc_enqueue(0, 0, vec![0]), None, "full queue refuses");
    let mut fuel = u64::MAX;
    let r = run_with_host_fast(&m, 0, &[], &mut fuel, &mut host).expect("run");
    let sum: i64 = (0..SVC_QUEUE_CAP as i64).sum();
    assert_eq!(
        r,
        vec![Value::I64(SVC_QUEUE_CAP as i64 * 1000 + 7 + sum)],
        "all {SVC_QUEUE_CAP} dispatches served natively, in order, over the one world"
    );
}

/// The qualification veto: a park-capable seam anywhere in a serving module (here a futex wait in
/// a function nothing calls) declines the native compile, and the fast entry falls back to the
/// tree-walker — which serves identically.
#[test]
fn a_park_seam_vetoes_the_native_serve_and_falls_back() {
    let m = module(SERVER_WITH_PARK_SEAM);
    assert!(
        bytecode::compile_module(&m.funcs).is_none(),
        "a serving module with a futex wait must decline (module-wide veto)"
    );
    let dispatches: &[(u32, u32, Vec<i64>)] = &[(0, 0, vec![5]), (0, 0, vec![30])];
    let (ri, ci, _) = scenario(&m, dispatches, run_with_host);
    let (rf, cf, _) = scenario(&m, dispatches, run_with_host_fast);
    assert_eq!(ri, rf, "fallback serves identically");
    assert_eq!(ci, cf);
    assert_eq!(rf, Ok(vec![Value::I64(2042)]));
}

/// The §3.6 separate-module corpus, verbatim: the parent spawns a serving child from a granted
/// module (op 5), mints a live offer over its export (op 14), calls through it (enqueue + park),
/// and joins — 142 = join(served=1)*100 + add(40,2).
const SEP_CALLER: &str = r#"
memory 17

func (i32, i32) -> (i64) {
block 0 (v0: i32, v1: i32) {
  vmh = i64.extend_i32_u v1
  ventry = i64.const 0
  voff = i64.const 65536
  vlog = i64.const 12
  vq = i64.const 0
  v5 = cap.call 6 5 (i64, i64, i64, i64, i64) -> (i32) v0 (vmh, ventry, voff, vlog, vq)
  v6 = i64.const 0
  v7 = cap.call 6 14 (i32, i64) -> (i32) v0 (v5, v6)
  va = i64.const 40
  vb = i64.const 2
  vr = cap.call 268435456 0 (i64, i64) -> (i64) v7 (va, vb)
  vj = cap.call 6 1 (i32) -> (i64) v0 (v5)
  vk = i64.const 100
  vm = i64.mul vj vk
  vs = i64.add vm vr
  return vs
  }
}
"#;

const SEP_SERVER: &str = r#"
memory 12
type 0 func (i64, i64) -> (i64)
type 1 interface { add: 0 }
export 0 interface "adder" 1 { add: 1 }

func (i64) -> (i64) {
block 0 (v0: i64) {
  vz = i32.const 0
  vn = svc.wait vz
  return vn
  }
}

func (i64, i64) -> (i64) {
block 0 (va: i64, vb: i64) {
  vs = i64.add va vb
  return vs
  }
}
"#;

/// I36 slice 2 — the whole caller ↔ servicer round-trip runs natively: op-5 spawn, op-14 offer
/// mint, live-call enqueue + `BlockedTicket` park, the child's `svc.wait` park + enqueue wake,
/// handler dispatch, settle-wake of the caller, join. Both modules must compile natively (the
/// caller has no svc ops; the serving child qualifies), and the result matches the tree-walker.
#[test]
fn a_native_caller_parks_on_a_native_serving_child_and_wakes_with_the_reply() {
    let a = module(SEP_CALLER);
    let b = svm_text::parse_module(SEP_SERVER).expect("parse server");
    svm_verify::verify_module(&b).expect("verify server");
    assert!(
        bytecode::compile_module(&a.funcs).is_some(),
        "the caller (op 5 + op 14 + a live call) must be admitted natively"
    );
    assert!(
        bytecode::compile_module(&b.funcs).is_some(),
        "the serving child (svc.wait + a pure handler) must be admitted natively"
    );
    let run = |entry: Entry| {
        let mut host = Host::new();
        let hi = host.grant_instantiator(0, 1u64 << 17);
        let hm = host.grant_module(&b);
        let mut fuel = 5_000_000u64;
        entry(
            &a,
            0,
            &[Value::I32(hi), Value::I32(hm)],
            &mut fuel,
            &mut host,
        )
    };
    let ri = run(run_with_host);
    let rf = run(run_with_host_fast);
    assert_eq!(
        ri, rf,
        "caller/servicer round-trip must match the tree-walker"
    );
    assert_eq!(
        rf,
        Ok(vec![Value::I64(142)]),
        "join(served=1)*100 + add(40,2) — a foreign program served the native live call"
    );
}

/// `svc.wait` with work already queued behaves like `svc.poll` (progress ⇒ deliver the count, no
/// park) — natively, against pre-enqueued embedder dispatches, equal to the tree-walker.
#[test]
fn a_native_svc_wait_with_queued_work_serves_and_returns() {
    let src = SERVER.replace(
        "vn = cap.call 4294967295 9 () -> (i64) vz ()",
        "vn = cap.call 4294967295 10 () -> (i64) vz ()",
    );
    let m = module(&src);
    assert!(
        bytecode::compile_module(&m.funcs).is_some(),
        "the svc.wait server must be admitted natively"
    );
    let dispatches: &[(u32, u32, Vec<i64>)] = &[(0, 0, vec![5]), (0, 0, vec![30])];
    let (ri, ci, _) = scenario(&m, dispatches, run_with_host);
    let (rf, cf, _) = scenario(&m, dispatches, run_with_host_fast);
    assert_eq!(ri, rf, "svc.wait-with-work must match the tree-walker");
    assert_eq!(ci, cf);
    assert_eq!(rf, Ok(vec![Value::I64(2042)]));
}
