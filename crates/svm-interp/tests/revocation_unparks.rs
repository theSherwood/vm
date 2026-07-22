//! §3.6 slice 1 — **revocation-unparks** (IMPORTS.md §3.6 consumer pinning; PROCESS.md §4):
//! the racing-fibers escape hatch. A fiber parked inside a capability call — a blocking
//! stream read with no data — is woken by a *sibling* fiber revoking the handle it is parked
//! through (`Stream.close`, D37 turned inward). The parked call completes with a probeable
//! negative errno (`-EBADF`); the fiber is never killed and nothing traps — cancellation is
//! a returned value the fiber handles on its own error path.
//!
//! Also pins the revocation act itself: `Stream.close` is now real (the slot entry is
//! cleared), so a *fresh* call on the closed handle is the clean D37 use-after-close
//! `CapFault` the docs always promised — while the in-flight parked call gets the errno,
//! per D42 (errors return; traps stay for escape/fatal).

use svm_interp::{run_with_host, Host, StreamRole, Trap, Value};

/// The §3.6 revocation completion status: `-EBADF`.
const CAP_REVOKED: i64 = -9;

/// func 0 (root): spawn the closer, then block reading the (empty, blocking) stdin stream —
/// parking this fiber keyed by the handle. When the closer revokes it, the read completes
/// with the errno, which the root returns after joining the closer.
/// func 1 (closer): park ~100ms on a futex nobody notifies (deterministically ordering the
/// close after the root's park), then `Stream.close` the shared handle.
const RACING_FIBERS: &str = r#"
memory 16
func (i32) -> (i64) {
block 0 (v0: i32) {
  vh64 = i64.extend_i32_u v0
  vt = thread.spawn 1 vh64 vh64
  vbuf = i64.const 8
  vcap = i64.const 4
  vr = cap.call 0 0 (i64, i64) -> (i64) v0 (vbuf, vcap)
  vj = thread.join vt
  return vr
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, vharg: i64) {
  vaddr = i64.const 0
  vexp = i32.const 0
  vto = i64.const 100000000
  vw = i32.atomic.wait vaddr vexp vto
  vh = i32.wrap_i64 vharg
  vc = cap.call 0 2 () -> (i64) vh ()
  return vc
  }
}
"#;

#[test]
fn a_sibling_fiber_revokes_a_parked_read_which_completes_with_an_errno() {
    let m = svm_text::parse_module(RACING_FIBERS).expect("parse");
    svm_verify::verify_module(&m).expect("verify");
    let mut host = Host::new();
    let h = host.grant_stream(StreamRole::In);
    host.set_stdin_blocking(true); // empty + blocking: the read parks instead of returning EOF
    let mut fuel = u64::MAX;
    let r = run_with_host(&m, 0, &[Value::I32(h)], &mut fuel, &mut host).expect("no trap");
    assert_eq!(
        r,
        vec![Value::I64(CAP_REVOKED)],
        "the parked read completed with the probeable errno — woken, not killed, not trapped"
    );
}

/// The revocation act alone: `close` returns 0 and a **fresh** call on the closed handle is
/// the D37 use-after-close `CapFault` (fail-closed at the use site) — distinct from the
/// in-flight errno above, per D42.
const CLOSE_THEN_USE: &str = r#"
memory 16
func (i32) -> (i64) {
block 0 (v0: i32) {
  vc = cap.call 0 2 () -> (i64) v0 ()
  vbuf = i64.const 8
  vcap = i64.const 4
  vr = cap.call 0 0 (i64, i64) -> (i64) v0 (vbuf, vcap)
  return vr
  }
}
"#;

#[test]
fn a_fresh_call_on_a_closed_stream_is_the_d37_use_after_close_fault() {
    let m = svm_text::parse_module(CLOSE_THEN_USE).expect("parse");
    svm_verify::verify_module(&m).expect("verify");
    let mut host = Host::new();
    let h = host.grant_stream(StreamRole::In);
    let mut fuel = u64::MAX;
    let r = run_with_host(&m, 0, &[Value::I32(h)], &mut fuel, &mut host);
    assert_eq!(
        r,
        Err(Trap::CapFault),
        "use-after-close faults clean at the use site (D37)"
    );
}

/// Close is idempotent-shaped at the op level: closing an already-closed handle is itself the
/// D37 fault (the handle no longer resolves) — nothing dangles, nothing double-frees.
#[test]
fn close_of_a_closed_handle_faults_rather_than_dangling() {
    let m = svm_text::parse_module(
        r#"
memory 16
func (i32) -> (i64) {
block 0 (v0: i32) {
  vc = cap.call 0 2 () -> (i64) v0 ()
  vd = cap.call 0 2 () -> (i64) v0 ()
  return vd
  }
}
"#,
    )
    .expect("parse");
    svm_verify::verify_module(&m).expect("verify");
    let mut host = Host::new();
    let h = host.grant_stream(StreamRole::Out);
    let mut fuel = u64::MAX;
    let r = run_with_host(&m, 0, &[Value::I32(h)], &mut fuel, &mut host);
    assert_eq!(r, Err(Trap::CapFault));
}
