//! §3.6 slice 5a — **fiber-level park routing**: a park inside a fiber parks the FIBER, not
//! the vCPU (DESIGN.md "blocks the fiber, never the domain"). The resumer observes a third
//! `cont.resume` status, `FIBER_PARKED` (3) — the suspend the guest didn't write — and keeps
//! running; re-resuming while blocked reports it again (the cooperative poll); after the
//! event fires, a resume continues the fiber past its park with the event's result delivered.
//! All single-vCPU: the whole point is that the domain never stops.

use svm_interp::{run_with_host, Host, StreamRole, Value};

/// Futex park: the fiber `atomic.wait`s on a zero cell (forever). The root sees
/// `(FIBER_PARKED, 0)` twice (park, then poll), `atomic.notify`s the cell (waking exactly 1
/// waiter — the fiber), and the third resume runs the fiber to completion with the wait's
/// `WAIT_WOKEN` (0) status as its return. Composite: s1*100_000 + s2*10_000 + woken*1_000 +
/// s3*100 + value = 3*100_000 + 3*10_000 + 1*1_000 + 1*100 + 0 = 331_100.
const FUTEX_FIBER_PARK: &str = r#"
memory 16
func () -> (i64) {
block 0 () {
  v0 = ref.func 1
  v1 = i64.const 0
  v2 = cont.new v0 v1
  v3 = i64.const 0
  vs1, vv1 = cont.resume v2 v3
  vs2, vv2 = cont.resume v2 v3
  vaddr = i64.const 0
  vcnt = i32.const 1
  vw = atomic.notify vaddr vcnt
  vs3, vv3 = cont.resume v2 v3
  vk1 = i64.const 100000
  vs1e = i64.extend_i32_s vs1
  va = i64.mul vs1e vk1
  vk2 = i64.const 10000
  vs2e = i64.extend_i32_s vs2
  vb = i64.mul vs2e vk2
  vk3 = i64.const 1000
  vwe = i64.extend_i32_s vw
  vc = i64.mul vwe vk3
  vk4 = i64.const 100
  vs3e = i64.extend_i32_s vs3
  vd = i64.mul vs3e vk4
  vab = i64.add va vb
  vcd = i64.add vc vd
  vabcd = i64.add vab vcd
  vr = i64.add vabcd vv3
  return vr
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vaddr = i64.const 0
  vexp = i32.const 0
  vto = i64.const -1
  vst = i32.atomic.wait vaddr vexp vto
  vst64 = i64.extend_i32_s vst
  return vst64
  }
}
"#;

#[test]
fn a_fiber_futex_park_parks_the_fiber_not_the_vcpu() {
    let m = svm_text::parse_module(FUTEX_FIBER_PARK).expect("parse");
    svm_verify::verify_module(&m).expect("verify");
    let mut host = Host::new();
    let mut fuel = u64::MAX;
    let r = run_with_host(&m, 0, &[], &mut fuel, &mut host).expect("no trap, no hang");
    assert_eq!(
        r,
        vec![Value::I64(331_100)],
        "park(3) → poll(3) → notify wakes 1 → resume returns (1, WAIT_WOKEN=0)"
    );
}

/// The racing-fibers pattern on ONE vCPU: the fiber blocks reading empty blocking stdin
/// (fiber-parks; root sees `FIBER_PARKED`), the ROOT — still running, that's the point —
/// closes the handle, and the next resume completes the fiber's read with the revocation
/// errno (`-EBADF`), which it returns. Composite: s1*10_000 + s2*100 + (-value) =
/// 3*10_000 + 1*100 + 9 = 30_109.
const REVOKE_INTO_FIBER: &str = r#"
memory 16
func (i32) -> (i64) {
block 0 (v0: i32) {
  vf = ref.func 1
  vsp = i64.const 0
  vk = cont.new vf vsp
  vh64 = i64.extend_i32_u v0
  vs1, vv1 = cont.resume vk vh64
  vc = cap.call 0 2 () -> (i64) v0 ()
  vs2, vv2 = cont.resume vk vh64
  vk1 = i64.const 10000
  vs1e = i64.extend_i32_s vs1
  va = i64.mul vs1e vk1
  vk2 = i64.const 100
  vs2e = i64.extend_i32_s vs2
  vb = i64.mul vs2e vk2
  vz = i64.const 0
  vneg = i64.sub vz vv2
  vab = i64.add va vb
  vr = i64.add vab vneg
  return vr
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vh = i32.wrap_i64 varg
  vbuf = i64.const 8
  vcap = i64.const 4
  vr = cap.call 0 0 (i64, i64) -> (i64) vh (vbuf, vcap)
  return vr
  }
}
"#;

#[test]
fn the_root_revokes_a_read_its_own_fiber_is_parked_in() {
    let m = svm_text::parse_module(REVOKE_INTO_FIBER).expect("parse");
    svm_verify::verify_module(&m).expect("verify");
    let mut host = Host::new();
    let h = host.grant_stream(StreamRole::In);
    host.set_stdin_blocking(true);
    let mut fuel = u64::MAX;
    let r = run_with_host(&m, 0, &[Value::I32(h)], &mut fuel, &mut host).expect("no trap");
    assert_eq!(
        r,
        vec![Value::I64(30_109)],
        "fiber parked (3), root closed, fiber returned -EBADF (status 1, value -9)"
    );
}

/// The park-vs-event race, closed: the cell already differs from `expected` when the fiber
/// waits, so the register-then-recheck wakes it immediately — the resumer sees one
/// `FIBER_PARKED` (the transient set-aside), and the next resume completes the wait with
/// `WAIT_NOT_EQUAL` (1). Composite: 3*10_000 + 1*100 + 1 = 30_101.
const NOT_EQUAL_INSTA_WAKE: &str = r#"
memory 16
func () -> (i64) {
block 0 () {
  vaddr = i64.const 8
  vfive = i64.const 5
  i64.store vaddr vfive
  vf = ref.func 1
  vsp = i64.const 0
  vk = cont.new vf vsp
  vz = i64.const 0
  vs1, vv1 = cont.resume vk vz
  vs2, vv2 = cont.resume vk vz
  vk1 = i64.const 10000
  vs1e = i64.extend_i32_s vs1
  va = i64.mul vs1e vk1
  vk2 = i64.const 100
  vs2e = i64.extend_i32_s vs2
  vb = i64.mul vs2e vk2
  vab = i64.add va vb
  vr = i64.add vab vv2
  return vr
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vaddr = i64.const 8
  vexp = i32.const 0
  vto = i64.const -1
  vst = i32.atomic.wait vaddr vexp vto
  vst64 = i64.extend_i32_s vst
  return vst64
  }
}
"#;

#[test]
fn a_prechanged_cell_wakes_the_parking_fiber_immediately() {
    let m = svm_text::parse_module(NOT_EQUAL_INSTA_WAKE).expect("parse");
    svm_verify::verify_module(&m).expect("verify");
    let mut host = Host::new();
    let mut fuel = u64::MAX;
    let r = run_with_host(&m, 0, &[], &mut fuel, &mut host).expect("no trap, no hang");
    assert_eq!(
        r,
        vec![Value::I64(30_101)],
        "one transient FIBER_PARKED, then WAIT_NOT_EQUAL delivered"
    );
}
