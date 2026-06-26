//! THREADS.md 4c-domain — §14 `Instantiator.instantiate` (same-module confined executor children)
//! under the **parallel** driver. Each `instantiate` spawns a child on its own OS thread, confined to
//! a power-of-two sub-window (`nested_view` of the shared backing, own page-prot map), with an
//! attenuated powerbox (`Instantiator` + `AddressSpace` over its own window), its own natural dispatch
//! table (no parent install slots), and a quota sub-allocated from the parent's fuel — a **nested
//! confined parallel run**, joinable through the registry exactly like a `thread.spawn` child.
//!
//! Both kernels are schedule-independent: every child computes the same pure value, folded by `join`,
//! so the result is byte-identical to the **cooperative** single-threaded oracle no matter how the
//! children's threads interleave. The fan-out kernel proves the basic confined child + join; the
//! nested kernel proves confinement **composes** under parallelism (a child itself instantiates a
//! grandchild on a further nested scope).
//!
//! The `unsafe` (borrowing host memory via `Region::shared`) lives in this embedder/test, mirroring
//! `bytecode_parallel_caps.rs`.

use std::sync::Arc;
use svm_interp::{bytecode, Host, Region, Value};
use svm_text::parse_module;

/// Root `(instantiator) -> sum`: instantiate 8 children (func 1), each in its own 4 KiB sub-window at
/// `64 KiB + i*4 KiB`, store the handles at `mem[16 + i*4]`, then `join` each and sum the results.
/// Each child returns 5 ⇒ 8 × 5 = 40, regardless of how the child threads interleave.
const FANOUT: &str = r#"memory 17
func (i32) -> (i64) {
block0(v0: i32):
  vi0 = i64.const 0
  br block1(vi0, v0)
block1(vi: i64, vinst: i32):
  vn = i64.const 8
  vlt = i64.lt_u vi vn
  br_if vlt block2(vi, vinst) block3(vinst)
block2(vi2: i64, vinst2: i32):
  v4096 = i64.const 4096
  vofflo = i64.mul vi2 v4096
  v64k = i64.const 65536
  voff = i64.add v64k vofflo
  ventry = i64.const 1
  vslog = i64.const 12
  vquota = i64.const 0
  vh = cap.call 6 0 (i64, i64, i64, i64) -> (i32) vinst2 (ventry, voff, vslog, vquota)
  v4 = i64.const 4
  vholo = i64.mul vi2 v4
  v16 = i64.const 16
  vhoff = i64.add v16 vholo
  i32.store vhoff vh
  v1 = i64.const 1
  vinext = i64.add vi2 v1
  br block1(vinext, vinst2)
block3(vinst3: i32):
  vj0 = i64.const 0
  vs0 = i64.const 0
  br block4(vj0, vs0, vinst3)
block4(vj: i64, vs: i64, vinst4: i32):
  vn2 = i64.const 8
  vlt2 = i64.lt_u vj vn2
  br_if vlt2 block5(vj, vs, vinst4) block6(vs)
block5(vj2: i64, vs2: i64, vinst5: i32):
  v4b = i64.const 4
  vjlo = i64.mul vj2 v4b
  v16b = i64.const 16
  vjoff = i64.add v16b vjlo
  vhh = i32.load vjoff
  vr = cap.call 6 1 (i32) -> (i64) vinst5 (vhh)
  vsn = i64.add vs2 vr
  v1b = i64.const 1
  vjn = i64.add vj2 v1b
  br block4(vjn, vsn, vinst5)
block6(vs3: i64):
  return vs3
}
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const 5
  return v1
}
"#;

/// Same fan-out, but each child (func 1, handed its own `Instantiator`) itself instantiates a
/// grandchild (func 2) in a 1 KiB sub-window of its own window, joins it, and returns its value —
/// confinement composing to depth 2 under parallelism. Each grandchild returns 9 ⇒ 8 × 9 = 72.
const NESTED: &str = r#"memory 17
func (i32) -> (i64) {
block0(v0: i32):
  vi0 = i64.const 0
  br block1(vi0, v0)
block1(vi: i64, vinst: i32):
  vn = i64.const 8
  vlt = i64.lt_u vi vn
  br_if vlt block2(vi, vinst) block3(vinst)
block2(vi2: i64, vinst2: i32):
  v4096 = i64.const 4096
  vofflo = i64.mul vi2 v4096
  v64k = i64.const 65536
  voff = i64.add v64k vofflo
  ventry = i64.const 1
  vslog = i64.const 12
  vquota = i64.const 0
  vh = cap.call 6 0 (i64, i64, i64, i64) -> (i32) vinst2 (ventry, voff, vslog, vquota)
  v4 = i64.const 4
  vholo = i64.mul vi2 v4
  v16 = i64.const 16
  vhoff = i64.add v16 vholo
  i32.store vhoff vh
  v1 = i64.const 1
  vinext = i64.add vi2 v1
  br block1(vinext, vinst2)
block3(vinst3: i32):
  vj0 = i64.const 0
  vs0 = i64.const 0
  br block4(vj0, vs0, vinst3)
block4(vj: i64, vs: i64, vinst4: i32):
  vn2 = i64.const 8
  vlt2 = i64.lt_u vj vn2
  br_if vlt2 block5(vj, vs, vinst4) block6(vs)
block5(vj2: i64, vs2: i64, vinst5: i32):
  v4b = i64.const 4
  vjlo = i64.mul vj2 v4b
  v16b = i64.const 16
  vjoff = i64.add v16b vjlo
  vhh = i32.load vjoff
  vr = cap.call 6 1 (i32) -> (i64) vinst5 (vhh)
  vsn = i64.add vs2 vr
  v1b = i64.const 1
  vjn = i64.add vj2 v1b
  br block4(vjn, vsn, vinst5)
block6(vs3: i64):
  return vs3
}
func (i64) -> (i64) {
block0(v0: i64):
  vinst = i32.wrap_i64 v0
  ventry = i64.const 2
  voff = i64.const 0
  vslog = i64.const 10
  vquota = i64.const 0
  vgh = cap.call 6 0 (i64, i64, i64, i64) -> (i32) vinst (ventry, voff, vslog, vquota)
  vgr = cap.call 6 1 (i32) -> (i64) vinst (vgh)
  return vgr
}
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const 9
  return v1
}
"#;

/// An 8-aligned zeroed buffer + a `Region::shared` over it; caller frees via the returned layout.
fn shared_window(size: usize) -> (Arc<Region>, *mut u8, std::alloc::Layout) {
    let layout = std::alloc::Layout::from_size_align(size, 8).unwrap();
    // SAFETY: non-zero layout; `size` valid 8-aligned bytes owned here, used only as this window.
    let base = unsafe { std::alloc::alloc_zeroed(layout) };
    assert!(!base.is_null());
    // SAFETY: `base` is `size` valid 8-aligned bytes, exclusively this window's, freed only after.
    let back = Arc::new(unsafe { Region::shared(base, size as u64) });
    (back, base, layout)
}

/// Cooperative oracle: one shared host, all children multiplexed on one thread (deterministic).
fn run_cooperative(src: &str) -> Result<Vec<Value>, svm_interp::Trap> {
    let m = parse_module(src).unwrap();
    let mut host = Host::new();
    let inst = host.grant_instantiator(0, 128 << 10); // authority over the whole 128 KiB window
    let mut f = 50_000_000u64;
    bytecode::compile_and_run_with_host(&m, 0, &[Value::I32(inst)], &mut f, &mut host)
        .expect("bytecode engine drives §14 instantiate (cooperative)")
}

/// Parallel: each `instantiate` runs its confined child on a real OS thread over the shared backing.
fn run_parallel(src: &str) -> Result<Vec<Value>, svm_interp::Trap> {
    let m = parse_module(src).unwrap();
    let mut host = Host::new();
    let inst = host.grant_instantiator(0, 128 << 10);
    let (back, base, layout) = shared_window(1 << 17);
    let mut f = 50_000_000u64;
    let r = bytecode::compile_and_run_capture_over_parallel_with_host(
        &m,
        0,
        &[Value::I32(inst)],
        &mut f,
        &[],
        Arc::clone(&back),
        &mut host,
    )
    .expect("bytecode engine drives §14 instantiate (parallel)")
    .0;
    drop(back);
    // SAFETY: same layout; the region (and all borrows) are gone (the scope joined every vCPU).
    unsafe { std::alloc::dealloc(base, layout) };
    r
}

/// 8 confined executor children on real threads, joined and summed — matches the cooperative oracle.
#[test]
fn parallel_instantiate_fanout_matches_oracle() {
    let want = run_cooperative(FANOUT);
    assert_eq!(want, Ok(vec![Value::I64(40)]), "oracle: 8 × child(5) = 40");
    for i in 0..50 {
        assert_eq!(
            run_parallel(FANOUT),
            want,
            "parallel instantiate != oracle (run {i})"
        );
    }
}

/// Depth-2: each confined child instantiates a grandchild on a further nested scope — confinement
/// composes under genuine parallelism; the folded result still matches the oracle.
#[test]
fn parallel_instantiate_nested_matches_oracle() {
    let want = run_cooperative(NESTED);
    assert_eq!(
        want,
        Ok(vec![Value::I64(72)]),
        "oracle: 8 × grandchild(9) = 72"
    );
    for i in 0..50 {
        assert_eq!(
            run_parallel(NESTED),
            want,
            "parallel nested instantiate != oracle (run {i})"
        );
    }
}
