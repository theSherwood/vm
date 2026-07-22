//! THREADS.md 4c-domain — **Miri** verification of §14 `Instantiator.instantiate` under the
//! **parallel** driver. The differential test in `svm/tests/bytecode_parallel_instantiate.rs` proves
//! the parallel confined-child path agrees with the cooperative oracle; this proves the genuinely-
//! parallel machinery — each child running on its own thread over a `nested_view` sub-window of the
//! **one** shared `Region::shared` backing (own page-prot map), its writes handed back to the parent
//! across the join — is **free of data races / UB / provenance errors**. Miri's checker, not the
//! iteration count, is the point, so the kernel (2 children) is small.
//!
//! Run: `cargo +nightly miri test -p svm-interp --test parallel_instantiate_miri`

use std::sync::Arc;
use svm_interp::{bytecode, Host, Region, Value};
use svm_text::parse_module;

// Root (instantiator) instantiates 2 confined children, each in its own 4 KiB sub-window at 64 KiB /
// 68 KiB; each child writes the marker 21 at its own offset 0 (→ the shared backing) and returns 5.
// The parent joins both (the join is the happens-before that publishes the children's writes), reads
// both markers back through the shared window, and returns 5 + 5 + 21 + 21 = 52 — exercising real
// cross-thread non-atomic access through the confined sub-windows.
const SRC: &str = r#"memory 17
func (i32) -> (i64) {
block 0 (v0: i32) {
  vo0 = i64.const 65536
  ve = i64.const 1
  vsl = i64.const 12
  vq = i64.const 0
  vh0 = cap.call 6 0 (i64, i64, i64, i64) -> (i32) v0 (ve, vo0, vsl, vq)
  vo1 = i64.const 69632
  vh1 = cap.call 6 0 (i64, i64, i64, i64) -> (i32) v0 (ve, vo1, vsl, vq)
  vr0 = cap.call 6 1 (i32) -> (i64) v0 (vh0)
  vr1 = cap.call 6 1 (i32) -> (i64) v0 (vh1)
  vm0a = i64.const 65536
  vm0 = i32.load8_u vm0a
  vm0e = i64.extend_i32_u vm0
  vm1a = i64.const 69632
  vm1 = i32.load8_u vm1a
  vm1e = i64.extend_i32_u vm1
  vs1 = i64.add vr0 vr1
  vs2 = i64.add vs1 vm0e
  vs3 = i64.add vs2 vm1e
  return vs3
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  vaddr = i64.const 0
  v21 = i32.const 21
  i32.store8 vaddr v21
  v5 = i64.const 5
  return v5
  }
}
"#;

fn run_parallel(src: &str) -> Result<Vec<Value>, svm_interp::Trap> {
    let m = parse_module(src).unwrap();
    let mut host = Host::new();
    let inst = host.grant_instantiator(0, 128 << 10);
    let size = 1usize << 17;
    let layout = std::alloc::Layout::from_size_align(size, 8).unwrap();
    // SAFETY: non-zero layout; `size` valid 8-aligned bytes owned here until freed below.
    let base = unsafe { std::alloc::alloc_zeroed(layout) };
    assert!(!base.is_null());
    // SAFETY: `base` is `size` valid 8-aligned bytes, exclusively this window's, freed only after.
    let back = Arc::new(unsafe { Region::shared(base, size as u64) });
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

#[test]
fn parallel_instantiate_race_free_under_miri() {
    assert_eq!(run_parallel(SRC), Ok(vec![Value::I64(52)])); // 5 + 5 + 21 + 21
}

// §14-B `instantiate_module`: the granted "plugin" module (4 KiB window, a data segment `"K"` = 75 at
// offset 0) reads its own data byte, writes the marker 30 at offset 1, and returns 75. The root
// instantiates it into 2 confined children on real threads, joins both, then reads both markers back —
// exercising data-segment materialization + cross-thread non-atomic access through the pushed module.
const MODULE_CHILD: &str = r#"memory 12
data 0 "K"
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 0
  v2 = i32.load8_u v1
  vm = i64.const 1
  v30 = i32.const 30
  i32.store8 vm v30
  v3 = i64.extend_i32_u v2
  return v3
  }
}
"#;

const MODULE_ROOT: &str = r#"memory 17
func (i32, i32) -> (i64) {
block 0 (vinst: i32, vmod: i32) {
  vmod64 = i64.extend_i32_s vmod
  ve = i64.const 0
  vsl = i64.const 12
  vq = i64.const 0
  vo0 = i64.const 65536
  vh0 = cap.call 6 5 (i64, i64, i64, i64, i64) -> (i32) vinst (vmod64, ve, vo0, vsl, vq)
  vo1 = i64.const 69632
  vh1 = cap.call 6 5 (i64, i64, i64, i64, i64) -> (i32) vinst (vmod64, ve, vo1, vsl, vq)
  vr0 = cap.call 6 1 (i32) -> (i64) vinst (vh0)
  vr1 = cap.call 6 1 (i32) -> (i64) vinst (vh1)
  va0 = i64.const 65537
  vm0 = i32.load8_u va0
  vm0e = i64.extend_i32_u vm0
  va1 = i64.const 69633
  vm1 = i32.load8_u va1
  vm1e = i64.extend_i32_u vm1
  vs1 = i64.add vr0 vr1
  vs2 = i64.add vs1 vm0e
  vs3 = i64.add vs2 vm1e
  return vs3
  }
}
"#;

fn run_parallel_mod() -> Result<Vec<Value>, svm_interp::Trap> {
    let m = parse_module(MODULE_ROOT).unwrap();
    let child = parse_module(MODULE_CHILD).unwrap();
    let mut host = Host::new();
    let inst = host.grant_instantiator(0, 128 << 10);
    let mh = host.grant_module(&child);
    let size = 1usize << 17;
    let layout = std::alloc::Layout::from_size_align(size, 8).unwrap();
    // SAFETY: non-zero layout; `size` valid 8-aligned bytes owned here until freed below.
    let base = unsafe { std::alloc::alloc_zeroed(layout) };
    assert!(!base.is_null());
    // SAFETY: `base` is `size` valid 8-aligned bytes, exclusively this window's, freed only after.
    let back = Arc::new(unsafe { Region::shared(base, size as u64) });
    let mut f = 50_000_000u64;
    let r = bytecode::compile_and_run_capture_over_parallel_with_host(
        &m,
        0,
        &[Value::I32(inst), Value::I32(mh)],
        &mut f,
        &[],
        Arc::clone(&back),
        &mut host,
    )
    .expect("bytecode engine drives §14 instantiate_module (parallel)")
    .0;
    drop(back);
    // SAFETY: same layout; the region (and all borrows) are gone (the scope joined every vCPU).
    unsafe { std::alloc::dealloc(base, layout) };
    r
}

#[test]
fn parallel_instantiate_module_race_free_under_miri() {
    assert_eq!(run_parallel_mod(), Ok(vec![Value::I64(210)])); // 75 + 75 + 30 + 30
}
