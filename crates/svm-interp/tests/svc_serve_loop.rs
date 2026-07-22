//! §3.6 slice 2 — the **serve-loop core**: a domain's offers served as handlers over its
//! **one world** (IMPORTS.md §3.6). Dispatches queue on the domain's bounded inbound queue
//! (embedder-enqueued this slice; the cross-domain caller is the caller-parking slice) and are
//! admitted at the guest's `svc.poll` service point (`cap.call CAP_SELF_TYPE_ID 9` — riding
//! the reserved self-namespace dispatch, no wire change). A handler runs over the SAME live
//! window and powerbox as `main` — what `main` writes, handlers read, and vice versa. There
//! is no second state: the passive instance's two-world split is what §3.6 dissolves.

use std::sync::Arc;
use svm_interp::{run_with_host, Host, Value, CAP_SELF_SVC_POLL, SVC_QUEUE_CAP};

/// The serving domain. Offer "counter" op 0 = func 1 `bump(x) -> old + x`, where `old` is the
/// LIVE value at mem[0] — the handler both reads and writes main's memory (one world).
/// `main` (func 0) seeds mem[0] = 7, `svc.poll`s (serving everything queued), then returns
/// `served * 1000 + mem[0]` — so the return value proves both the served count and that the
/// handlers' writes landed in main's own window.
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

fn server_module() -> Arc<svm_ir::Module> {
    let m = svm_text::parse_module(SERVER).expect("parse");
    svm_verify::verify_module(&m).expect("verify");
    Arc::new(m)
}

#[test]
fn queued_dispatches_run_as_handlers_over_the_one_world() {
    let m = server_module();
    let mut host = Host::new();
    host.set_self_module(&m);
    // Two dispatches queued before the run: bump(5) then bump(30).
    let t1 = host.svc_enqueue(0, 0, vec![5]).expect("enqueue 1");
    let t2 = host.svc_enqueue(0, 0, vec![30]).expect("enqueue 2");
    let mut fuel = u64::MAX;
    let r = run_with_host(&m, 0, &[], &mut fuel, &mut host).expect("run");
    // served=2; mem[0] went 7 → 12 → 42 (handlers mutated MAIN's window: one world).
    assert_eq!(
        r,
        vec![Value::I64(2042)],
        "served*1000 + final live counter"
    );
    // Completion cells: each handler returned the counter's value BEFORE its bump — 7 and 12 —
    // proving the handlers observed main's seed and each other's effects, serialized in order.
    assert_eq!(host.svc_result(t1), Some(7));
    assert_eq!(host.svc_result(t2), Some(12));
    assert_eq!(host.svc_result(t1), None, "a completion cell drains once");
}

#[test]
fn an_empty_poll_serves_nothing_and_returns_zero() {
    let m = server_module();
    let mut host = Host::new();
    host.set_self_module(&m);
    let mut fuel = u64::MAX;
    let r = run_with_host(&m, 0, &[], &mut fuel, &mut host).expect("run");
    assert_eq!(r, vec![Value::I64(7)], "0 served, seed untouched");
}

#[test]
fn the_queue_is_bounded_and_refuses_fail_closed() {
    let m = server_module();
    let mut host = Host::new();
    host.set_self_module(&m);
    for i in 0..SVC_QUEUE_CAP {
        assert!(
            host.svc_enqueue(0, 0, vec![i as i64]).is_some(),
            "under cap"
        );
    }
    assert_eq!(
        host.svc_enqueue(0, 0, vec![0]),
        None,
        "a full queue refuses the enqueue — backpressure at the enqueuer, never buffering"
    );
    // An unservable target (no such offer/op) also refuses at enqueue: the queue only ever
    // holds dispatches the domain can actually serve.
    assert_eq!(host.svc_enqueue(9, 0, vec![0]), None, "unknown export");
    assert_eq!(host.svc_enqueue(0, 7, vec![0]), None, "unknown op");
}

#[test]
fn an_arity_mismatched_dispatch_gets_a_probeable_errno_and_serving_continues() {
    let m = server_module();
    let mut host = Host::new();
    host.set_self_module(&m);
    let bad = host.svc_enqueue(0, 0, vec![1, 2, 3]).expect("enqueues"); // bump takes 1 arg
    let good = host.svc_enqueue(0, 0, vec![5]).expect("enqueues");
    let mut fuel = u64::MAX;
    let r = run_with_host(&m, 0, &[], &mut fuel, &mut host).expect("run");
    // Only the good dispatch served (7 → 12); the bad one errnos in its cell.
    assert_eq!(r, vec![Value::I64(1012)]);
    assert_eq!(host.svc_result(bad), Some(-22), "-EINVAL, probeable");
    assert_eq!(host.svc_result(good), Some(7));
}

/// The self-namespace op number is part of the public surface (jacl will emit it): pin it.
#[test]
fn the_svc_poll_op_number_is_pinned() {
    assert_eq!(CAP_SELF_SVC_POLL, 9);
}
