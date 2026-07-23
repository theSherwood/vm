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
    assert_eq!(svm_interp::CAP_SELF_SVC_WAIT, 10);
}

/// §3.6 parity — **the bytecode entry serves via the oracle fallback**: `run_with_host_fast`
/// (the `Backend::Bytecode` entry) declines to compile the svc ops and runs the whole module
/// on the tree-walker, so a serving domain behaves **identically** through either entry.
/// Fallback is the same free-correctness path the Instantiator ops already ride.
#[test]
fn the_bytecode_entry_serves_identically_via_the_oracle_fallback() {
    let m = server_module();
    let mut host = Host::new();
    host.set_self_module(&m);
    let t1 = host.svc_enqueue(0, 0, vec![5]).expect("enqueue 1");
    let t2 = host.svc_enqueue(0, 0, vec![30]).expect("enqueue 2");
    let mut fuel = u64::MAX;
    let r = svm_interp::run_with_host_fast(&m, 0, &[], &mut fuel, &mut host).expect("run");
    assert_eq!(r, vec![Value::I64(2042)], "identical to the tree-walk run");
    assert_eq!(host.svc_result(t1), Some(7));
    assert_eq!(host.svc_result(t2), Some(12));
}

/// §3.6 parity — a backend tier **without** eval-loop servicing answers both svc ops with a
/// probeable `-EINVAL` from the one shared host dispatch (the JIT's route): refusal, never a
/// trap, never a wrong answer — pinned directly at the shared entry.
#[test]
fn a_non_serving_tier_refuses_both_svc_ops_probeably() {
    let mut host = Host::new();
    for op in [CAP_SELF_SVC_POLL, svm_interp::CAP_SELF_SVC_WAIT] {
        let r = host
            .cap_dispatch_slots(svm_ir::CAP_SELF_TYPE_ID, op, 0, &[], None)
            .expect("refusal, not a trap");
        assert_eq!(r, vec![-22], "probeable -EINVAL for svc op {op}");
    }
}

/// §3.6 slice 3 — **caller-side parking, end to end**: a parent spawns a serving child
/// (§14 same-module), mints a live-callee offer over the child's export
/// (`Instantiator.child_offer`, op 14), and calls through it. The call enqueues on the
/// child's queue and parks the parent; the child's `svc.wait` (op 10) wakes on the enqueue
/// (or finds the work already queued — both orders are correct), serves `add(40, 2)` as a
/// handler, and the reply wakes the parent with 42. The child returns its served count,
/// which the parent reads back through `join` — proving the whole caller ↔ servicer
/// round-trip parked and woke rather than deadlocked. The offer's structural type id is
/// the first guest intern (`GUEST_IMPL_BASE` = 268435456), pinned by D59 determinism.
const CALLER_PARKING: &str = r#"
memory 17
type 0 func (i64, i64) -> (i64)
type 1 interface { add: 0 }
export 0 interface "adder" 1 { add: 2 }

func (i32) -> (i64) {
block 0 (v0: i32) {
  v1 = i64.const 1
  v2 = i64.const 65536
  v3 = i64.const 12
  v4 = i64.const 0
  v5 = cap.call 6 0 (i64, i64, i64, i64) -> (i32) v0 (v1, v2, v3, v4)
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

func (i64) -> (i64) {
block 0 (v0: i64) {
  vz = i32.const 0
  vn = cap.call 4294967295 10 () -> (i64) vz ()
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

/// §3.6 slice 4 — the **slot route**: the same round-trip as the direct form, but the caller
/// attaches the live-callee cap into a rebindable import slot and calls `call.import 0` — the
/// discovery-then-attach pattern over a live domain. Same enqueue/park/reply machinery.
const SLOT_CALLER: &str = r#"
memory 17
type 0 func (i64, i64) -> (i64)
type 1 interface { add: 0 }
export 0 interface "adder" 1 { add: 2 }
import 0 "svc.add" (i64, i64) -> (i64) rebindable

func (i32) -> (i64) {
block 0 (v0: i32) {
  v1 = i64.const 1
  v2 = i64.const 65536
  v3 = i64.const 12
  v4 = i64.const 0
  v5 = cap.call 6 0 (i64, i64, i64, i64) -> (i32) v0 (v1, v2, v3, v4)
  v6 = i64.const 0
  v7 = cap.call 6 14 (i32, i64) -> (i32) v0 (v5, v6)
  vst = import.attach 0 v7
  va = i64.const 40
  vb = i64.const 2
  vr = call.import 0 (va, vb)
  vj = cap.call 6 1 (i32) -> (i64) v0 (v5)
  vk = i64.const 100
  vm = i64.mul vj vk
  vs = i64.add vm vr
  return vs
  }
}

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

#[test]
fn a_slot_attached_live_call_parks_and_wakes_like_the_direct_form() {
    let m = Arc::new({
        let m = svm_text::parse_module(SLOT_CALLER).expect("parse");
        svm_verify::verify_module(&m).expect("verify");
        m
    });
    let mut host = Host::new();
    host.set_self_module(&m);
    // The rebindable slot's template: typed to the (first-interned) offer interface, unbound.
    host.set_import_bindings(vec![svm_interp::BoundImport {
        type_id: 268435456, // GUEST_IMPL_BASE — the offer's structural intern (D59-deterministic)
        op: 0,
        handle: 0,
        bound: false,
        rebindable: true,
    }]);
    let h = host.grant_instantiator(0, 1u64 << 17);
    let mut fuel = 5_000_000u64;
    let r = run_with_host(&m, 0, &[Value::I32(h)], &mut fuel, &mut host).expect("run");
    assert_eq!(
        r,
        vec![Value::I64(142)],
        "attach → call.import through a live callee: park, serve (svc.wait sugar), reply, join"
    );
}

/// §3.6 slice 4 — the `svc.*` sugar round-trips: `svc.wait v0` in SLOT_CALLER above already
/// proves parse; this pins print→parse stability and the desugared identity.
#[test]
fn svc_sugar_round_trips_and_desugars_to_the_reserved_dispatch() {
    let m = svm_text::parse_module(SLOT_CALLER).expect("parse");
    let printed = svm_text::print_module(&m);
    assert!(
        printed.contains("svc.wait v"),
        "the printer emits the greppable sugar"
    );
    let m2 = svm_text::parse_module(&printed).expect("reparse");
    assert_eq!(m, m2, "text round-trip");
    let m3 = svm_encode::decode_module(&svm_encode::encode_module(&m)).expect("decode");
    assert_eq!(
        m, m3,
        "wire round-trip (sugar is pure spelling — no wire change)"
    );
}

/// §3.6 — **separate-module serving children**: the child domain runs its OWN module
/// (`instantiate_module`, op 5) with its own offers, and the parent wires a live offer over
/// the child's export via the same `child_offer` (op 14). The offer's shape is the CHILD
/// module's export — the parent registers no self module at all, pinning that the wirer's
/// own program is irrelevant to the wire — interned structurally into the parent's table
/// (D59: first guest intern = `GUEST_IMPL_BASE`, same as the same-module form). Same
/// enqueue/park/`svc.wait`-serve/reply/join round-trip: join(served=1)*100 + add(40,2) = 142.
const SEPARATE_MODULE_CALLER: &str = r#"
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

/// The child's own program: its own memory declaration (the carve must equal it — §14
/// transparency), its own offer, its own serve loop. Entry = func 0 (`svc.wait`, return the
/// served count to the joiner); `add` = func 1.
const SEPARATE_MODULE_SERVER: &str = r#"
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

#[test]
fn a_separate_module_child_serves_its_own_offers() {
    let a = Arc::new({
        let m = svm_text::parse_module(SEPARATE_MODULE_CALLER).expect("parse caller");
        svm_verify::verify_module(&m).expect("verify caller");
        m
    });
    let b = svm_text::parse_module(SEPARATE_MODULE_SERVER).expect("parse server");
    svm_verify::verify_module(&b).expect("verify server");
    let mut host = Host::new();
    // Deliberately NO set_self_module on the parent: the offer's shape is the child's.
    let hi = host.grant_instantiator(0, 1u64 << 17);
    let hm = host.grant_module(&b);
    let mut fuel = 5_000_000u64;
    let r = run_with_host(
        &a,
        0,
        &[Value::I32(hi), Value::I32(hm)],
        &mut fuel,
        &mut host,
    )
    .expect("run");
    assert_eq!(
        r,
        vec![Value::I64(142)],
        "join(served=1)*100 + add(40,2) — a foreign program served the parent's live call"
    );
    let _ = a;
}

/// A `child_offer` naming an export the child's module doesn't have refuses with a probeable
/// `-EINVAL` — resolved against the CHILD's module (which has export 0 only), never the
/// wirer's. The child here polls-and-returns (nothing to serve), so the parent's `join`
/// completes the run cleanly after the refused wire.
const SEPARATE_MODULE_BAD_EXPORT: &str = r#"
memory 17

func (i32, i32) -> (i64) {
block 0 (v0: i32, v1: i32) {
  vmh = i64.extend_i32_u v1
  ventry = i64.const 0
  voff = i64.const 65536
  vlog = i64.const 12
  vq = i64.const 0
  v5 = cap.call 6 5 (i64, i64, i64, i64, i64) -> (i32) v0 (vmh, ventry, voff, vlog, vq)
  v6 = i64.const 9
  v7 = cap.call 6 14 (i32, i64) -> (i32) v0 (v5, v6)
  vj = cap.call 6 1 (i32) -> (i64) v0 (v5)
  vr = i64.extend_i32_s v7
  vs = i64.add vr vj
  return vs
  }
}
"#;

/// The bad-export test's child: same offer surface, but the entry `svc.poll`s (serving the
/// nothing that's queued) and returns 0 — so it completes without a caller.
const SEPARATE_MODULE_POLL_SERVER: &str = r#"
memory 12
type 0 func (i64, i64) -> (i64)
type 1 interface { add: 0 }
export 0 interface "adder" 1 { add: 1 }

func (i64) -> (i64) {
block 0 (v0: i64) {
  vz = i32.const 0
  vn = svc.poll vz
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

#[test]
fn a_bad_export_on_a_separate_module_child_refuses_probeably() {
    let a = svm_text::parse_module(SEPARATE_MODULE_BAD_EXPORT).expect("parse");
    svm_verify::verify_module(&a).expect("verify");
    let b = svm_text::parse_module(SEPARATE_MODULE_POLL_SERVER).expect("parse server");
    svm_verify::verify_module(&b).expect("verify server");
    let mut host = Host::new();
    let hi = host.grant_instantiator(0, 1u64 << 17);
    let hm = host.grant_module(&b);
    let mut fuel = 5_000_000u64;
    let r = run_with_host(
        &a,
        0,
        &[Value::I32(hi), Value::I32(hm)],
        &mut fuel,
        &mut host,
    )
    .expect("run");
    assert_eq!(
        r,
        vec![Value::I64(-22)],
        "-EINVAL (plus join(0)), probeable — the wire refused, the run completed"
    );
}

#[test]
fn a_caller_parks_on_a_live_child_and_wakes_with_the_reply() {
    let m = Arc::new({
        let m = svm_text::parse_module(CALLER_PARKING).expect("parse");
        svm_verify::verify_module(&m).expect("verify");
        m
    });
    let mut host = Host::new();
    host.set_self_module(&m);
    let h = host.grant_instantiator(0, 1u64 << 17);
    let mut fuel = 5_000_000u64;
    let r = run_with_host(&m, 0, &[Value::I32(h)], &mut fuel, &mut host).expect("run");
    assert_eq!(
        r,
        vec![Value::I64(142)],
        "join(served=1)*100 + add(40,2) — the parked caller woke with the handler's reply"
    );
}
