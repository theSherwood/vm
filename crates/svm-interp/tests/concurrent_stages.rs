//! STAGE1.md item 6 / PROCESS.md §4 — **concurrent stages**: two children running
//! CONCURRENTLY, piped through a granted `SharedRegion` + canonical-key futex. The parent
//! creates a region, spawns a producer and a consumer (named-grant spawns, each re-granted
//! the region), and joins both; the children map the region into their own windows and move
//! four items through a **one-slot bounded ring** (flag at region byte 0, datum at byte 8),
//! parking on the flag and waking each other by `notify`.
//!
//! This is the shape sequential spawn/wait cannot run at all: with a 1-slot ring and 4
//! items, the producer MUST park mid-stream and be woken by the consumer (and vice versa) —
//! run-to-completion order deadlocks. It also pins the S1c futex-key residue closed: each
//! child maps the region in its OWN window (its own address space, its own per-window region
//! id), so wait/notify only rendezvous if the key is the **backing identity** — with
//! per-domain keys every wake misses. A regression surfaces loudly, not as a hang: waits
//! carry a 30 s timeout and each child folds its TIMED_OUT count into its result ×1000, so
//! missed wakes turn 410 into a big wrong number.

use std::sync::Arc;
use svm_interp::{run_with_host, Host, Value};

/// func 0 — the parent: mint a region (AddressSpace op 5), build one named-grant record
/// (`"ring"` → the region handle, stored at runtime), spawn producer (entry 1) and consumer
/// (entry 2) as 32 KiB carves, join both. Composite: join(producer=4)*100 + join(consumer=10)
/// = 410.
///
/// funcs 1/2 — the stages: resolve `"ring"`, query the map granule, map the region at window
/// offset 0, then run the ring protocol. Producer publishes 1..=4 (park while full);
/// consumer sums them (park while empty) → 10.
const PIPELINE: &str = r#"
memory 17
data 200 "ring"

func (i32, i32) -> (i64) {
block 0 (v0: i32, v1: i32) {
  vlen = i64.const 32768
  vrh64 = cap.call 5 5 (i64) -> (i64) v1 (vlen)
  vrh = i32.wrap_i64 vrh64
  va1 = i64.const 256
  vv1 = i32.const 200
  i32.store va1 vv1
  va2 = i64.const 260
  vv2 = i32.const 4
  i32.store va2 vv2
  va3 = i64.const 264
  i32.store va3 vrh
  vgp = i64.const 256
  vgn = i64.const 1
  ve1 = i64.const 1
  voffp = i64.const 65536
  vlog = i64.const 15
  vq = i64.const 0
  vp = cap.call 6 11 (i64, i64, i64, i64, i64, i64) -> (i32) v0 (vgp, vgn, ve1, voffp, vlog, vq)
  ve2 = i64.const 2
  voffc = i64.const 98304
  vc = cap.call 6 11 (i64, i64, i64, i64, i64, i64) -> (i32) v0 (vgp, vgn, ve2, voffc, vlog, vq)
  vjp = cap.call 6 1 (i32) -> (i64) v0 (vp)
  vjc = cap.call 6 1 (i32) -> (i64) v0 (vc)
  vk = i64.const 100
  vm = i64.mul vjp vk
  vs = i64.add vm vjc
  return vs
  }
}

func (i64) -> (i64) {
block 0 (v0: i64) {
  vnm = i64.const 1735289202
  vz = i64.const 0
  i64.store vz vnm
  vp = i64.const 0
  vl = i64.const 4
  vh = cap.self.resolve vp vl
  vg = cap.call 4 3 () -> (i64) vh ()
  vroff = i64.const 0
  vprot = i32.const 3
  vm = cap.call 4 0 (i64, i64, i64, i32) -> (i64) vh (vroff, vroff, vg, vprot)
  vone = i64.const 1
  br 1(vone, vroff)
  }
block 1 (vi: i64, vtos: i64) {
  vfour = i64.const 4
  vdone = i64.lt_s vfour vi
  br_if vdone 5(vtos) 2(vi, vtos)
  }
block 2 (vi: i64, vtos: i64) {
  vfa = i64.const 0
  vf = i32.load vfa
  br_if vf 3(vi, vtos) 4(vi, vtos)
  }
block 3 (vi: i64, vtos: i64) {
  vfa = i64.const 0
  vexp = i32.const 1
  vto = i64.const 30000000000
  vst = i32.atomic.wait vfa vexp vto
  vtwo = i32.const 2
  vis = i32.eq vst vtwo
  vis64 = i64.extend_i32_u vis
  vtos2 = i64.add vtos vis64
  br 2(vi, vtos2)
  }
block 4 (vi: i64, vtos: i64) {
  vda = i64.const 8
  i64.store vda vi
  vfa = i64.const 0
  vfull = i32.const 1
  i32.store vfa vfull
  vcnt = i32.const 1
  vw = atomic.notify vfa vcnt
  vone = i64.const 1
  vni = i64.add vi vone
  br 1(vni, vtos)
  }
block 5 (vtos: i64) {
  vk = i64.const 1000
  vm = i64.mul vtos vk
  vfour = i64.const 4
  vr = i64.add vm vfour
  return vr
  }
}

func (i64) -> (i64) {
block 0 (v0: i64) {
  vnm = i64.const 1735289202
  vz = i64.const 0
  i64.store vz vnm
  vp = i64.const 0
  vl = i64.const 4
  vh = cap.self.resolve vp vl
  vg = cap.call 4 3 () -> (i64) vh ()
  vroff = i64.const 0
  vprot = i32.const 3
  vm = cap.call 4 0 (i64, i64, i64, i32) -> (i64) vh (vroff, vroff, vg, vprot)
  vone = i64.const 1
  br 1(vone, vroff, vroff)
  }
block 1 (vn: i64, vsum: i64, vtos: i64) {
  vfour = i64.const 4
  vdone = i64.lt_s vfour vn
  br_if vdone 5(vsum, vtos) 2(vn, vsum, vtos)
  }
block 2 (vn: i64, vsum: i64, vtos: i64) {
  vfa = i64.const 0
  vf = i32.load vfa
  br_if vf 4(vn, vsum, vtos) 3(vn, vsum, vtos)
  }
block 3 (vn: i64, vsum: i64, vtos: i64) {
  vfa = i64.const 0
  vexp = i32.const 0
  vto = i64.const 30000000000
  vst = i32.atomic.wait vfa vexp vto
  vtwo = i32.const 2
  vis = i32.eq vst vtwo
  vis64 = i64.extend_i32_u vis
  vtos2 = i64.add vtos vis64
  br 2(vn, vsum, vtos2)
  }
block 4 (vn: i64, vsum: i64, vtos: i64) {
  vda = i64.const 8
  vd = i64.load vda
  vsum2 = i64.add vsum vd
  vfa = i64.const 0
  vempty = i32.const 0
  i32.store vfa vempty
  vcnt = i32.const 1
  vw = atomic.notify vfa vcnt
  vone = i64.const 1
  vnn = i64.add vn vone
  br 1(vnn, vsum2, vtos)
  }
block 5 (vsum: i64, vtos: i64) {
  vk = i64.const 1000
  vm = i64.mul vtos vk
  vr = i64.add vm vsum
  return vr
  }
}
"#;

#[test]
fn two_concurrent_children_pipe_through_a_shared_region_ring() {
    let m = svm_text::parse_module(PIPELINE).expect("parse");
    svm_verify::verify_module(&m).expect("verify");
    let m = Arc::new(m);
    let mut host = Host::new();
    host.set_self_module(&m);
    let hi = host.grant_instantiator(0, 1u64 << 17);
    let ha = host.grant_address_space(0, 1u64 << 17);
    let mut fuel = 50_000_000u64;
    let r = run_with_host(
        &m,
        0,
        &[Value::I32(hi), Value::I32(ha)],
        &mut fuel,
        &mut host,
    )
    .expect("no trap, no hang");
    assert_eq!(
        r,
        vec![Value::I64(410)],
        "producer published 4 (park while full), consumer summed 10 (park while empty), \
         zero timeouts — a 1-slot ring across two live domains"
    );
}
