//! STAGE1.md item 6 — the **JIT pipeline**: two granted children running CONCURRENTLY on their own
//! OS threads, piped through a granted `SharedRegion` + canonical-key futex — the fast-backend twin
//! of `svm-interp/tests/concurrent_stages.rs`. The parent mints a region (the run's backing factory
//! is `svm_run::new_shared_region`, so it is a real OS shared-memory object), spawns producer and
//! consumer via `instantiate_named` (op 11, each re-granted the region by name), and joins both.
//!
//! What this pins, JIT-specifically:
//! - **op-11 children are async** (S1c): each runs on its own OS thread in its own guarded window.
//!   With a 1-slot ring and 4 items, run-to-completion order deadlocks — the producer MUST park
//!   mid-stream and be woken by the consumer, so the old synchronous spawn cannot pass this at all.
//! - **real aliasing into separate child windows**: each child `map`s the region into its OWN
//!   window (`MprotectWindow::map_region` — `mmap(MAP_SHARED|MAP_FIXED)` of the region's memfd on
//!   unix, placeholder + `MapViewOfFile3` on windows), so parent-minted bytes are the same physical
//!   pages in both children.
//! - **canonical futex keys across windows**: each child's first `cap.call` installs the region-canon
//!   hook over its own `mem_base`, so `atomic.wait`/`notify` in different windows key on the backing
//!   identity `(os_fd, offset)` and rendezvous. With per-window keys every wake misses — and the
//!   regression surfaces loudly, not as a hang: waits carry a 5 s timeout, each child folds its
//!   TIMED_OUT count into its result ×1000, and a child that times out more than 6 times bails.
//!
//! (Windows sizing: the §13 map granule is the 64 KiB allocation granularity there, so child windows
//! are 128 KiB — `memory 17` carves — and the children map `len = granule` queried at run time.)

use std::sync::Arc;
use svm_interp::{run_with_host, Host, Value};
use svm_jit::{compile_and_run_capture_reserved_with_host_ex, GrantChildHooks, JitOutcome};
use svm_text::parse_module;
use svm_verify::verify_module;

/// Byte-identical to the interpreter test's `PIPELINE` module, so the two backends run the same
/// program: func 0 (parent) mints a 64 KiB region, spawns producer (func 1) and consumer (func 2)
/// as 128 KiB carves granting `"ring"` → region, joins both → join(producer=4)*100 +
/// join(consumer=10) = 410. The stages resolve `"ring"`, query the map granule, map the region at
/// window offset 0, and move 1..=4 through the one-slot ring (flag at region byte 0, datum at 8).
const PIPELINE: &str = r#"
memory 19
data 200 "ring"

func (i32, i32) -> (i64) {
block 0 (v0: i32, v1: i32) {
  vlen = i64.const 65536
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
  voffp = i64.const 131072
  vlog = i64.const 17
  vq = i64.const 0
  vp = cap.call 6 11 (i64, i64, i64, i64, i64, i64) -> (i32) v0 (vgp, vgn, ve1, voffp, vlog, vq)
  ve2 = i64.const 2
  voffc = i64.const 262144
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
  vto = i64.const 5000000000
  vst = i32.atomic.wait vfa vexp vto
  vtwo = i32.const 2
  vis = i32.eq vst vtwo
  vis64 = i64.extend_i32_u vis
  vtos2 = i64.add vtos vis64
  vsix = i64.const 6
  vbail = i64.lt_s vsix vtos2
  br_if vbail 5(vtos2) 2(vi, vtos2)
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
  vto = i64.const 5000000000
  vst = i32.atomic.wait vfa vexp vto
  vtwo = i32.const 2
  vis = i32.eq vst vtwo
  vis64 = i64.extend_i32_u vis
  vtos2 = i64.add vtos vis64
  vsix = i64.const 6
  vbail = i64.lt_s vsix vtos2
  br_if vbail 5(vsum, vtos2) 2(vn, vsum, vtos2)
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

fn grant_hooks() -> GrantChildHooks {
    GrantChildHooks {
        build: svm_run::grant_child_build,
        build_named: svm_run::grant_named_child_build,
        bind_imports: svm_run::child_bind_imports,
        release: svm_run::grant_child_release,
    }
}

/// The interpreter reference: the same source, same 410.
fn run_interp() -> Vec<Value> {
    let m = parse_module(PIPELINE).expect("parse");
    verify_module(&m).expect("verify");
    let m = Arc::new(m);
    let mut host = Host::new();
    host.set_self_module(&m);
    let hi = host.grant_instantiator(0, 1u64 << 19);
    let ha = host.grant_address_space(0, 1u64 << 19);
    let mut fuel = 50_000_000u64;
    run_with_host(
        &m,
        0,
        &[Value::I32(hi), Value::I32(ha)],
        &mut fuel,
        &mut host,
    )
    .expect("interp: no trap, no hang")
}

#[test]
fn two_concurrent_jit_children_pipe_through_a_shared_region_ring() {
    // Nesting requires the child runner (`fiber_rt`); where unsupported the JIT declines child
    // spawns, so there is nothing to pin — the interpreter remains the only backend there.
    if !svm_jit::fiber_supported() {
        return;
    }
    let ir = run_interp();
    assert_eq!(ir, vec![Value::I64(410)], "interp reference");

    let m = parse_module(PIPELINE).expect("parse");
    verify_module(&m).expect("verify");
    let mut host = Host::new();
    // Regions minted by this run are real OS shared-memory objects (memfd / section), so the JIT
    // children can `map` them for hardware aliasing.
    host.set_region_factory(svm_run::new_shared_region);
    let hi = host.grant_instantiator(0, 1u64 << 19);
    let ha = host.grant_address_space(0, 1u64 << 19);
    let (jo, _jmem) = compile_and_run_capture_reserved_with_host_ex(
        &m,
        0,
        &[hi as i64, ha as i64],
        &vec![0u8; 1 << 19],
        0,
        svm_run::cap_thunk,
        &mut host as *mut Host as *mut core::ffi::c_void,
        None,
        Some(grant_hooks()),
    )
    .expect("jit");
    assert!(
        matches!(jo, JitOutcome::Returned(ref s) if s == &[410]),
        "jit: producer published 4 (park while full), consumer summed 10 (park while empty), \
         zero timeouts — a 1-slot ring across two child OS threads, aliased into both windows; \
         got {jo:?}"
    );
}
