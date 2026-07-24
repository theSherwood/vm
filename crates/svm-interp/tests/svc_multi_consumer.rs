//! ISSUES.md I39 (top rung) — **multi-consumer `svc.wait`**: N spawned server vCPUs drain one
//! domain queue. The substrate change is in the scheduler's `svc_waiters` map — a domain may
//! now park **many** vCPUs in `svc.wait` (previously a second parker displaced the first,
//! dropping a live vCPU: a hang), and a wake re-admits **all** of them (the wake path knows
//! only the domain key, never which vCPU owns a parked handler, so wake-all is the
//! obviously-correct form: admission is race-free under the powerbox lock, and a consumer
//! that finds nothing runnable re-parks).
//!
//! What multi-consumer deliberately gives up: handler serialization (I39's documented
//! trade) — two handlers may now run concurrently over the one world, so a domain opting in
//! accepts the threading discipline (atomics/futexes for shared state; the handlers here are
//! pure or disjoint). The fast backends' serve veto already declines svc+thread modules —
//! pinned below — so the oracle is the only backend that runs these shapes, by design.

use std::sync::Arc;
use svm_interp::{bytecode, run_with_host, Host, Value};

fn module(text: &str) -> Arc<svm_ir::Module> {
    let m = svm_text::parse_module(text).expect("parse");
    svm_verify::verify_module(&m).expect("verify");
    Arc::new(m)
}

/// Two worker threads each `svc.poll` the SAME domain queue concurrently (main spawns them
/// with the queue pre-filled and joins both). The handler is pure (`double(x) = 2x`), so
/// every completion cell is deterministic; the racing pops split the queue arbitrarily but
/// the counts must SUM to the number of dispatches.
const TWO_POLLERS: &str = r#"
memory 16
type 0 func (i64) -> (i64)
type 1 interface { double: 0 }
export 0 interface "svc" 1 { double: 1 }

func () -> (i64) {
block 0 () {
  vsp = i64.const 0
  vz = i64.const 0
  vt1 = thread.spawn 2 vsp vz
  vt2 = thread.spawn 2 vsp vz
  vj1 = thread.join vt1
  vj2 = thread.join vt2
  vs = i64.add vj1 vj2
  return vs
  }
}

func (i64) -> (i64) {
block 0 (vx: i64) {
  vtwo = i64.const 2
  vr = i64.mul vx vtwo
  return vr
  }
}

func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vz = i32.const 0
  vn = cap.call 4294967295 9 () -> (i64) vz ()
  return vn
  }
}
"#;

#[test]
fn two_pollers_split_one_queue_and_their_counts_sum() {
    let m = module(TWO_POLLERS);
    assert!(
        bytecode::compile_module(&m.funcs).is_none(),
        "svc + threads must keep declining on the fast backends (the serve veto) — the oracle \
         is the only backend that runs multi-consumer shapes"
    );
    let mut host = Host::new();
    host.set_self_module(&m);
    let args: &[i64] = &[3, 5, 11];
    let tickets: Vec<u64> = args
        .iter()
        .map(|&a| host.svc_enqueue(0, 0, vec![a]).expect("enqueue"))
        .collect();
    let mut fuel = u64::MAX;
    let r = run_with_host(&m, 0, &[], &mut fuel, &mut host).expect("no trap, no hang");
    assert_eq!(
        r,
        vec![Value::I64(args.len() as i64)],
        "the two pollers' served counts sum to the dispatch count, however the race split them"
    );
    for (t, a) in tickets.iter().zip(args) {
        assert_eq!(host.svc_result(*t), Some(a * 2), "pure handler, exact cell");
    }
}

/// A **pure timeout**: the I38 timed `svc.wait` form (op 10 with the optional timeout arg, in
/// ns) parks with a deadline and returns `0` when it fires with nothing served — the
/// multi-consumer wind-down primitive (a spare consumer could otherwise never exit: any
/// sibling may work-steal every dispatch). Oracle-only: the fast entry must decline the module
/// (`serve_qualifies` treats the timed form as a park seam) and fall back — pinned by running
/// both entries.
const TIMED_WAIT_TIMES_OUT: &str = r#"
memory 12
type 0 func (i64) -> (i64)
type 1 interface { noop: 0 }
export 0 interface "svc" 1 { noop: 1 }

func () -> (i64) {
block 0 () {
  vz = i32.const 0
  vt = i64.const 10000000
  vn = cap.call 4294967295 10 (i64) -> (i64) vz (vt)
  return vn
  }
}

func (i64) -> (i64) {
block 0 (vx: i64) {
  return vx
  }
}
"#;

#[test]
fn a_timed_svc_wait_with_no_work_returns_zero() {
    let m = module(TIMED_WAIT_TIMES_OUT);
    assert!(
        !bytecode::serve_qualifies(&m.funcs),
        "the timed form is oracle-only (deadline machinery lives in the scheduler)"
    );
    assert!(bytecode::compile_module(&m.funcs).is_none());
    let mut host = Host::new();
    host.set_self_module(&m);
    let mut fuel = u64::MAX;
    let r = run_with_host(&m, 0, &[], &mut fuel, &mut host).expect("no trap, no hang");
    assert_eq!(r, vec![Value::I64(0)], "deadline fired, nothing served");
    // The fast entry declines and falls back to the tree-walker — identical answer.
    let mut host = Host::new();
    host.set_self_module(&m);
    let mut fuel = u64::MAX;
    let rf = svm_interp::run_with_host_fast(&m, 0, &[], &mut fuel, &mut host).expect("fallback");
    assert_eq!(rf, vec![Value::I64(0)]);
}

/// The serving CHILD spawns two internal worker threads that both consume the domain queue,
/// while the parent makes three sequential live calls through a `child_offer`-minted handle:
/// `add(40,2)`, `add(1,2)`, then `finish()` (whose handler sets the done flag at mem[8]).
/// Consumers **work-steal** — either worker may serve any subset, including all of it — so
/// each worker loops on a TIMED `svc.wait` (100ms) and exits when the flag is set: a spare
/// consumer parked when the last dispatch lands is re-admitted by its deadline at the latest,
/// sees the flag, and exits (the wind-down protocol; without the timed form it would park
/// forever and strand the child's `thread.join` — the hang the first draft of this test
/// found). Worker totals sum to the dispatch count however the race splits them; the child
/// joins both and returns the sum (3), and the parent packs
/// `join*100 + add(40,2) + add(1,2) + finish()` = 300 + 42 + 3 + 0 = 345. Every scheduler
/// interleaving must complete — including both-workers-parked, the ordering the old
/// single-slot `svc_waiters` map dropped a vCPU on.
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
  vr1 = cap.call 268435456 0 (i64, i64) -> (i64) v7 (va, vb)
  vc = i64.const 1
  vd = i64.const 2
  vr2 = cap.call 268435456 0 (i64, i64) -> (i64) v7 (vc, vd)
  vr3 = cap.call 268435456 1 () -> (i64) v7 ()
  vj = cap.call 6 1 (i32) -> (i64) v0 (v5)
  vk = i64.const 100
  vm = i64.mul vj vk
  vs1 = i64.add vr1 vr2
  vs2 = i64.add vs1 vr3
  vs = i64.add vm vs2
  return vs
  }
}
"#;

const SEP_SERVER_TWO_WORKERS: &str = r#"
memory 12
type 0 func (i64, i64) -> (i64)
type 1 func () -> (i64)
type 2 interface { add: 0, finish: 1 }
export 0 interface "adder" 2 { add: 1, finish: 3 }

func (i64) -> (i64) {
block 0 (v0: i64) {
  vsp = i64.const 0
  vz = i64.const 0
  vt1 = thread.spawn 2 vsp vz
  vt2 = thread.spawn 2 vsp vz
  vj1 = thread.join vt1
  vj2 = thread.join vt2
  vs = i64.add vj1 vj2
  return vs
  }
}

func (i64, i64) -> (i64) {
block 0 (va: i64, vb: i64) {
  vs = i64.add va vb
  return vs
  }
}

func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vz0 = i64.const 0
  br 1(vz0)
  }
block 1 (vtotal: i64) {
  vz = i32.const 0
  vt = i64.const 100000000
  vn = cap.call 4294967295 10 (i64) -> (i64) vz (vt)
  vtot = i64.add vtotal vn
  vfa = i64.const 8
  vf = i64.load vfa
  vzero = i64.const 0
  vdone = i64.ne vf vzero
  br_if vdone 2(vtot) 1(vtot)
  }
block 2 (vres: i64) {
  return vres
  }
}

func () -> (i64) {
block 0 () {
  vfa = i64.const 8
  vone = i64.const 1
  i64.store vfa vone
  vz = i64.const 0
  return vz
  }
}
"#;

#[test]
fn two_svc_wait_consumers_serve_a_live_caller_across_every_interleaving() {
    let a = module(SEP_CALLER);
    let b = module(SEP_SERVER_TWO_WORKERS);
    // Race coverage: the park orderings (0, 1, or 2 workers parked when each enqueue lands)
    // are scheduler-dependent, so run the scenario repeatedly — every interleaving must
    // complete with the same packed result.
    for i in 0..10 {
        let mut host = Host::new();
        let hi = host.grant_instantiator(0, 1u64 << 17);
        let hm = host.grant_module(&b);
        let mut fuel = u64::MAX;
        let r = run_with_host(
            &a,
            0,
            &[Value::I32(hi), Value::I32(hm)],
            &mut fuel,
            &mut host,
        );
        assert_eq!(
            r,
            Ok(vec![Value::I64(345)]),
            "iteration {i}: join(worker totals = 3)*100 + 42 + 3 + 0"
        );
    }
}
