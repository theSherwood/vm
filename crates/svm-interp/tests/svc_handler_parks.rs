//! §3.6 slice 5b — **handler-fiber parking at the serve loop**: a dispatch handler runs as a
//! fiber of the serving vCPU, so a handler that hits a blocking point parks the *dispatch*,
//! never the domain — the serve loop treats `FIBER_PARKED` as a completed-but-not-replied
//! dispatch (its caller stays parked in `ticket_waiters` / its completion cell stays empty),
//! moves on to the next queued dispatch, and re-claims parked handlers on every re-execution.
//! A parked handler's wake also `svc_wake`s the domain, so a `svc.wait`-parked serve loop
//! resumes it. All single-vCPU: the point is that the domain keeps serving.

use std::sync::Arc;
use svm_interp::{run_with_host, Host, StreamRole, Value};

fn module(text: &str) -> Arc<svm_ir::Module> {
    let m = svm_text::parse_module(text).expect("parse");
    svm_verify::verify_module(&m).expect("verify");
    Arc::new(m)
}

/// Two dispatches: `park` (op 0) futex-waits on cell 0 (forever) — parking its HANDLER, not
/// the domain — and `wake` (op 1), served *while `park` is parked*, stores 5 at cell 8 and
/// notifies cell 0. The woken `park` handler then resumes, reads cell 8, and returns
/// `status*100 + mem[8]` = 0*100 + 5 — proving it continued past its park AFTER `wake`'s
/// store (one world, ordered). `svc.poll` completes both: main returns 2.
const PARK_THEN_WAKE: &str = r#"
memory 16
type 0 func () -> (i64)
type 1 interface { park: 0, wake: 0 }
export 0 interface "svc" 1 { park: 1, wake: 2 }

func () -> (i64) {
block 0 () {
  vz = i32.const 0
  vn = svc.poll vz
  return vn
  }
}

func () -> (i64) {
block 0 () {
  vaddr = i64.const 0
  vexp = i32.const 0
  vto = i64.const -1
  vst = i32.atomic.wait vaddr vexp vto
  vst64 = i64.extend_i32_s vst
  vk = i64.const 100
  va = i64.mul vst64 vk
  vm8a = i64.const 8
  vm8 = i64.load vm8a
  vr = i64.add va vm8
  return vr
  }
}

func () -> (i64) {
block 0 () {
  vm8a = i64.const 8
  vfive = i64.const 5
  i64.store vm8a vfive
  vaddr = i64.const 0
  vcnt = i32.const 1
  vw = atomic.notify vaddr vcnt
  vw64 = i64.extend_i32_s vw
  return vw64
  }
}
"#;

#[test]
fn a_parked_handler_blocks_its_dispatch_not_the_serve_loop() {
    let m = module(PARK_THEN_WAKE);
    let mut host = Host::new();
    host.set_self_module(&m);
    let t_park = host.svc_enqueue(0, 0, vec![]).expect("enqueue park");
    let t_wake = host.svc_enqueue(0, 1, vec![]).expect("enqueue wake");
    let mut fuel = u64::MAX;
    let r = run_with_host(&m, 0, &[], &mut fuel, &mut host).expect("no trap, no hang");
    assert_eq!(r, vec![Value::I64(2)], "both dispatches completed");
    assert_eq!(
        host.svc_result(t_park),
        Some(5),
        "WAIT_WOKEN*100 + the value `wake` stored AFTER `park` parked"
    );
    assert_eq!(host.svc_result(t_wake), Some(1), "notify woke exactly 1");
}

/// The racing-handlers pattern: `read` (op 0) blocks reading empty blocking stdin — its
/// handler parks — and `close` (op 1), served while `read` is parked, revokes the handle.
/// The revocation completes the parked read with the probeable errno (`-EBADF`), which the
/// woken handler returns into its completion cell. D37 turned inward, entirely inside one
/// serve loop on one vCPU.
const REVOKE_INTO_HANDLER: &str = r#"
memory 16
type 0 func (i64) -> (i64)
type 1 interface { a_read: 0, b_close: 0 }
export 0 interface "svc" 1 { a_read: 1, b_close: 2 }

func () -> (i64) {
block 0 () {
  vz = i32.const 0
  vn = svc.poll vz
  return vn
  }
}

func (i64) -> (i64) {
block 0 (varg: i64) {
  vh = i32.wrap_i64 varg
  vbuf = i64.const 8
  vcap = i64.const 4
  vr = cap.call 0 0 (i64, i64) -> (i64) vh (vbuf, vcap)
  return vr
  }
}

func (i64) -> (i64) {
block 0 (varg: i64) {
  vh = i32.wrap_i64 varg
  vc = cap.call 0 2 () -> (i64) vh ()
  return vc
  }
}
"#;

#[test]
fn a_sibling_dispatch_revokes_the_handle_a_parked_handler_reads() {
    let m = module(REVOKE_INTO_HANDLER);
    let mut host = Host::new();
    host.set_self_module(&m);
    let h = host.grant_stream(StreamRole::In);
    host.set_stdin_blocking(true);
    let t_read = host
        .svc_enqueue(0, 0, vec![h as i64])
        .expect("enqueue read");
    let t_close = host
        .svc_enqueue(0, 1, vec![h as i64])
        .expect("enqueue close");
    let mut fuel = u64::MAX;
    let r = run_with_host(&m, 0, &[], &mut fuel, &mut host).expect("no trap");
    assert_eq!(r, vec![Value::I64(2)], "both dispatches completed");
    assert_eq!(host.svc_result(t_read), Some(-9), "-EBADF, probeable");
    assert_eq!(host.svc_result(t_close), Some(0), "close succeeded");
}

/// A `svc.wait`-parked serve loop is re-admitted by an in-flight handler's wake, not just by
/// an enqueue: `sleeper`'s handler futex-waits with a 50ms timeout and parks; the queue is
/// empty and nothing has completed, so `svc.wait` parks the domain. The TIMER wake delivers
/// `WAIT_TIMED_OUT` into the handler fiber *and* re-admits the serve loop (the waiter's
/// domain key), which resumes the handler to completion — `svc.wait` returns 1.
const TIMER_REARMS_SVC_WAIT: &str = r#"
memory 16
type 0 func () -> (i64)
type 1 interface { sleeper: 0 }
export 0 interface "svc" 1 { sleeper: 1 }

func () -> (i64) {
block 0 () {
  vz = i32.const 0
  vn = svc.wait vz
  return vn
  }
}

func () -> (i64) {
block 0 () {
  vaddr = i64.const 0
  vexp = i32.const 0
  vto = i64.const 50000000
  vst = i32.atomic.wait vaddr vexp vto
  vst64 = i64.extend_i32_s vst
  vk = i64.const 200
  vr = i64.add vst64 vk
  return vr
  }
}
"#;

#[test]
fn a_handler_wake_readmits_a_svc_wait_parked_serve_loop() {
    let m = module(TIMER_REARMS_SVC_WAIT);
    let mut host = Host::new();
    host.set_self_module(&m);
    let t = host.svc_enqueue(0, 0, vec![]).expect("enqueue sleeper");
    let mut fuel = u64::MAX;
    let r = run_with_host(&m, 0, &[], &mut fuel, &mut host).expect("no trap, no hang");
    assert_eq!(
        r,
        vec![Value::I64(1)],
        "svc.wait woke on the handler's completion, not an enqueue"
    );
    assert_eq!(host.svc_result(t), Some(202), "200 + WAIT_TIMED_OUT");
}

/// A nested `svc.*` from under a running handler is refused with a probeable `-EINVAL`: the
/// serve loop is the domain's outermost dispatcher (re-entry into a domain is a fresh
/// dispatch, never a nested drain). The handler returns the refusal into its cell; the
/// domain keeps serving.
const NESTED_SERVE_REFUSED: &str = r#"
memory 16
type 0 func () -> (i64)
type 1 interface { nested: 0 }
export 0 interface "svc" 1 { nested: 1 }

func () -> (i64) {
block 0 () {
  vz = i32.const 0
  vn = svc.poll vz
  return vn
  }
}

func () -> (i64) {
block 0 () {
  vz = i32.const 0
  vn = svc.poll vz
  return vn
  }
}
"#;

#[test]
fn a_nested_svc_poll_under_a_handler_is_refused_probeably() {
    let m = module(NESTED_SERVE_REFUSED);
    let mut host = Host::new();
    host.set_self_module(&m);
    let t = host.svc_enqueue(0, 0, vec![]).expect("enqueue");
    let mut fuel = u64::MAX;
    let r = run_with_host(&m, 0, &[], &mut fuel, &mut host).expect("run");
    assert_eq!(r, vec![Value::I64(1)], "the dispatch itself completed");
    assert_eq!(host.svc_result(t), Some(-22), "-EINVAL, probeable");
}

/// A handler that `suspend`s has no resumer to receive its yield — the serve loop is not a
/// `cont.resume` site. Terminal for the one world: `FiberFault`, same family as the root
/// suspending.
const HANDLER_SUSPENDS: &str = r#"
memory 16
type 0 func () -> (i64)
type 1 interface { bad: 0 }
export 0 interface "svc" 1 { bad: 1 }

func () -> (i64) {
block 0 () {
  vz = i32.const 0
  vn = svc.poll vz
  return vn
  }
}

func () -> (i64) {
block 0 () {
  vz = i64.const 0
  vs = suspend vz
  return vs
  }
}
"#;

#[test]
fn a_handler_that_suspends_faults_the_domain() {
    let m = module(HANDLER_SUSPENDS);
    let mut host = Host::new();
    host.set_self_module(&m);
    host.svc_enqueue(0, 0, vec![]).expect("enqueue");
    let mut fuel = u64::MAX;
    let r = run_with_host(&m, 0, &[], &mut fuel, &mut host);
    assert!(r.is_err(), "a suspend with no resumer is a fiber fault");
}
