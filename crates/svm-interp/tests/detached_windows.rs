//! PROCESS.md §5 — **detached windows**: a child spawned through a `WindowMinter` capability
//! (`Instantiator.instantiate_detached`, op 15) runs in a fresh platform window *outside* its
//! spawner's — no ancestor below the platform holds read authority, and the child attests
//! `window_exposed = false` (the jacl distrust-spawner trust anchor). Detachment severs READ,
//! not lifecycle (the spawner keeps kill/join) and not coordination (live offers work — the
//! linkage is the powerbox, not the window). The minter's byte quota is host-enforced at each
//! mint; misses refuse probeably.

use std::sync::Arc;
use svm_interp::{run_with_host, Host, Value};

fn module(text: &str) -> Arc<svm_ir::Module> {
    let m = svm_text::parse_module(text).expect("parse");
    svm_verify::verify_module(&m).expect("verify");
    Arc::new(m)
}

/// The detached serving child: its own module, own offer, own serve loop — same shape as the
/// separate-module (nested) server, now in a window the parent cannot see.
const DETACHED_SERVER: &str = r#"
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

/// The parent: spawns the server DETACHED (op 15 — minter, module, no grants), wires its live
/// offer (`child_offer` — identical to the nested form: the linkage is the powerbox Arc), calls
/// `add(40, 2)` through it (park, serve, reply), joins. Composite: join(1)*100 + 42 = 142.
const DETACHED_CALLER: &str = r#"
memory 17

func (i32, i32, i32) -> (i64) {
block 0 (v0: i32, v1: i32, v2: i32) {
  vmh = i64.extend_i32_u v1
  vmin = i64.extend_i32_u v2
  vz = i64.const 0
  ve = i64.const 0
  vlog = i64.const 12
  vq = i64.const 0
  vB = cap.call 6 15 (i64, i64, i64, i64, i64, i64, i64) -> (i32) v0 (vmin, vmh, vz, vz, ve, vlog, vq)
  vex = i64.const 0
  vcap = cap.call 6 14 (i32, i64) -> (i32) v0 (vB, vex)
  va = i64.const 40
  vb = i64.const 2
  vr = cap.call 268435456 0 (i64, i64) -> (i64) vcap (va, vb)
  vj = cap.call 6 1 (i32) -> (i64) v0 (vB)
  vk = i64.const 100
  vm = i64.mul vj vk
  vs = i64.add vm vr
  return vs
  }
}
"#;

#[test]
fn a_detached_child_serves_live_calls_from_a_window_its_parent_cannot_see() {
    let a = module(DETACHED_CALLER);
    let b = module(DETACHED_SERVER);
    let mut host = Host::new();
    let hi = host.grant_instantiator(0, 1u64 << 17);
    let hm = host.grant_module(&b);
    let hw = host.grant_window_minter(1 << 12);
    let mut fuel = 5_000_000u64;
    let r = run_with_host(
        &a,
        0,
        &[Value::I32(hi), Value::I32(hm), Value::I32(hw)],
        &mut fuel,
        &mut host,
    )
    .expect("run");
    assert_eq!(
        r,
        vec![Value::I64(142)],
        "detached spawn → child_offer → park/serve/reply → join: detachment severs read, not coordination"
    );
}

/// A child that reports its own `cap.self.attest` — the non-interposable trust anchor.
const ATTEST_MOD: &str = r#"
memory 12

func (i64) -> (i64) {
block 0 (v0: i64) {
  vz = i32.const 0
  va = cap.call 4294967295 4 () -> (i64) vz ()
  return va
  }
}
"#;

/// The trust anchor, side by side: the SAME module spawned **nested** (op 5, a carve of the
/// parent's window) attests `tier 1 | window_exposed` = 257; spawned **detached** (op 15) it
/// attests `tier 1` alone = 1 — the distrust-spawner report, platform-vouched. Composite:
/// nested*1000 + detached = 257_001.
const ATTEST_BOTH: &str = r#"
memory 17

func (i32, i32, i32) -> (i64) {
block 0 (v0: i32, v1: i32, v2: i32) {
  vmh = i64.extend_i32_u v1
  vmin = i64.extend_i32_u v2
  ve = i64.const 0
  voff = i64.const 65536
  vlog = i64.const 12
  vq = i64.const 0
  vN = cap.call 6 5 (i64, i64, i64, i64, i64) -> (i32) v0 (vmh, ve, voff, vlog, vq)
  vz = i64.const 0
  vD = cap.call 6 15 (i64, i64, i64, i64, i64, i64, i64) -> (i32) v0 (vmin, vmh, vz, vz, ve, vlog, vq)
  vjN = cap.call 6 1 (i32) -> (i64) v0 (vN)
  vjD = cap.call 6 1 (i32) -> (i64) v0 (vD)
  vk = i64.const 1000
  vm = i64.mul vjN vk
  vs = i64.add vm vjD
  return vs
  }
}
"#;

#[test]
fn a_detached_child_attests_window_unexposed_where_a_nested_one_attests_exposed() {
    let a = module(ATTEST_BOTH);
    let b = module(ATTEST_MOD);
    let mut host = Host::new();
    let hi = host.grant_instantiator(0, 1u64 << 17);
    let hm = host.grant_module(&b);
    let hw = host.grant_window_minter(1 << 12);
    let mut fuel = 5_000_000u64;
    let r = run_with_host(
        &a,
        0,
        &[Value::I32(hi), Value::I32(hm), Value::I32(hw)],
        &mut fuel,
        &mut host,
    )
    .expect("run");
    assert_eq!(
        r,
        vec![Value::I64(257_001)],
        "nested attests tier|exposed (257); detached attests tier alone (1)"
    );
}

/// The minter's quota is the attenuation: with exactly one window's worth (4096 bytes), the
/// first detached spawn succeeds and the second refuses probeably (`-EINVAL`, nothing
/// charged) — a numeric quota, host-enforced at mint. Composite: first_failed*10 +
/// second_failed = 0*10 + 1 = 1.
const QUOTA_EXHAUSTS: &str = r#"
memory 17

func (i32, i32, i32) -> (i64) {
block 0 (v0: i32, v1: i32, v2: i32) {
  vmh = i64.extend_i32_u v1
  vmin = i64.extend_i32_u v2
  vz = i64.const 0
  ve = i64.const 0
  vlog = i64.const 12
  vq = i64.const 0
  vfst = cap.call 6 15 (i64, i64, i64, i64, i64, i64, i64) -> (i32) v0 (vmin, vmh, vz, vz, ve, vlog, vq)
  vsnd = cap.call 6 15 (i64, i64, i64, i64, i64, i64, i64) -> (i32) v0 (vmin, vmh, vz, vz, ve, vlog, vq)
  vj = cap.call 6 1 (i32) -> (i64) v0 (vfst)
  vzero = i32.const 0
  vf1 = i32.lt_s vfst vzero
  vf2 = i32.lt_s vsnd vzero
  vten = i32.const 10
  vm = i32.mul vf1 vten
  vs = i32.add vm vf2
  vr = i64.extend_i32_s vs
  return vr
  }
}
"#;

#[test]
fn the_minter_quota_bounds_detached_mints() {
    let a = module(QUOTA_EXHAUSTS);
    let b = module(ATTEST_MOD);
    let mut host = Host::new();
    let hi = host.grant_instantiator(0, 1u64 << 17);
    let hm = host.grant_module(&b);
    let hw = host.grant_window_minter(1 << 12); // exactly one 2^12 window
    let mut fuel = 5_000_000u64;
    let r = run_with_host(
        &a,
        0,
        &[Value::I32(hi), Value::I32(hm), Value::I32(hw)],
        &mut fuel,
        &mut host,
    )
    .expect("run");
    assert_eq!(
        r,
        vec![Value::I64(1)],
        "first mint fits the quota, second refuses probeably"
    );
}

/// A forged minter handle (the Instantiator handle itself, wrong type) refuses probeably —
/// the minter is spawn evidence, and no evidence means no detached window, never a trap.
const FORGED_MINTER: &str = r#"
memory 17

func (i32, i32) -> (i64) {
block 0 (v0: i32, v1: i32) {
  vmh = i64.extend_i32_u v1
  vmin = i64.extend_i32_u v0
  vz = i64.const 0
  ve = i64.const 0
  vlog = i64.const 12
  vq = i64.const 0
  vs = cap.call 6 15 (i64, i64, i64, i64, i64, i64, i64) -> (i32) v0 (vmin, vmh, vz, vz, ve, vlog, vq)
  vr = i64.extend_i32_s vs
  return vr
  }
}
"#;

#[test]
fn a_forged_minter_refuses_probeably() {
    let a = module(FORGED_MINTER);
    let b = module(ATTEST_MOD);
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
    .expect("run — refusal, not a trap");
    assert_eq!(r, vec![Value::I64(-22)], "-EINVAL, probeable");
}
