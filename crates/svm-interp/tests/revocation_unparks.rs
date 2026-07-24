//! §3.6 slice 1 — **revocation-unparks** (IMPORTS.md §3.6 consumer pinning; PROCESS.md §4):
//! the racing-fibers escape hatch. A fiber parked inside a capability call — a blocking
//! stream read with no data — is woken by a *sibling* fiber revoking the handle it is parked
//! through (`Stream.close`, D37 turned inward). The parked call completes with a probeable
//! negative errno (`-EBADF`); the fiber is never killed and nothing traps — cancellation is
//! a returned value the fiber handles on its own error path.
//!
//! Also pins the revocation act itself: `Stream.close` is real (the slot entry is cleared),
//! and — since I41 (graceful revocation) — a *fresh* call on the closed handle completes with
//! the **same probeable errno** as the in-flight parked call: cancellation is a value whether
//! you were mid-call or call a moment later, per D42 (errors return; traps stay for
//! escape/fatal — i.e. for *forged* handles, which D37 still faults).

use svm_interp::{run_with_host, Host, StreamRole, Value};

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

/// The revocation act alone: `close` returns 0 and a **fresh** call on the closed handle
/// completes with the same probeable errno as the in-flight one above (I41 graceful
/// revocation — the once-issued generation is its own tombstone; only a forgery traps).
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
fn a_fresh_call_on_a_closed_stream_completes_with_the_revocation_errno() {
    let m = svm_text::parse_module(CLOSE_THEN_USE).expect("parse");
    svm_verify::verify_module(&m).expect("verify");
    let mut host = Host::new();
    let h = host.grant_stream(StreamRole::In);
    let mut fuel = u64::MAX;
    let r = run_with_host(&m, 0, &[Value::I32(h)], &mut fuel, &mut host);
    assert_eq!(
        r,
        Ok(vec![Value::I64(CAP_REVOKED)]),
        "use-after-close is the same probeable errno as the unpark (I41) — no trap"
    );
}

/// §3.6 slice 4 — **rebind revokes the outgoing connection** (the pinned trigger: "closing/
/// REBINDING the client handle"): fiber A parks reading through the handle that import slot 0
/// is bound to; fiber B `import.attach`es a different stream into the slot; A's parked call
/// completes with the revocation errno. Same wake as a close — rebind is a hang-up.
const REBIND_RACERS: &str = r#"
memory 16
import 0 "in" (i64, i64) -> (i64) rebindable
func (i32, i32) -> (i64) {
block 0 (v0: i32, v1: i32) {
  vh64 = i64.extend_i32_u v1
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
  vst = import.attach 0 vh
  vz64 = i64.extend_i32_u vst
  return vz64
  }
}
"#;

#[test]
fn a_rebind_of_the_slot_wakes_the_fiber_parked_through_its_old_binding() {
    let m = svm_text::parse_module(REBIND_RACERS).expect("parse");
    svm_verify::verify_module(&m).expect("verify");
    let mut host = Host::new();
    let stdin_h = host.grant_stream(StreamRole::In);
    let other_h = host.grant_stream(StreamRole::Out);
    host.set_stdin_blocking(true);
    host.set_import_bindings(vec![svm_interp::BoundImport {
        type_id: 0, // STREAM
        op: 0,
        handle: stdin_h,
        bound: true,
        rebindable: true,
    }]);
    let mut fuel = u64::MAX;
    let r = run_with_host(
        &m,
        0,
        &[Value::I32(stdin_h), Value::I32(other_h)],
        &mut fuel,
        &mut host,
    )
    .expect("no trap");
    assert_eq!(
        r,
        vec![Value::I64(CAP_REVOKED)],
        "the rebind hung up the old connection; the parked read completed with the errno"
    );
}

/// Close is idempotent-shaped at the op level: closing an already-closed handle answers the
/// same revocation errno (I41 — the handle no longer resolves, and it is a once-valid one) —
/// nothing dangles, nothing double-frees, nothing traps.
#[test]
fn close_of_a_closed_handle_errnos_rather_than_dangling() {
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
    assert_eq!(r, Ok(vec![Value::I64(CAP_REVOKED)]));
}
