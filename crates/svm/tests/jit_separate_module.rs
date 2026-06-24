//! §14 **separate-module children on the JIT**, differentially vs. the interpreter — the
//! "plugin-in-plugin" story end to end. The host verifies a *different* module, grants both backends
//! an identical `Module` capability (iface 8), and the parent guest spawns a child domain running
//! that module via the `Instantiator`'s module ops (5/6/7). On the JIT the child module is resolved
//! through `svm_run::module_resolver` (a host callback, never guest-reachable) and **compiled at
//! `instantiate`** — §14's "nesting cost paid at setup" for foreign code. Both backends must agree on
//! the child's result, its data segments materializing into the carve, the lazy (fault-driven)
//! supply of those segments for demand children, validation (`-EINVAL` / `CapFault`), and the final
//! parent window bytes.

use svm_interp::{run_capture_reserved_with_host, Host, Trap, Value};
use svm_jit::{compile_and_run_capture_reserved_with_host_ex, JitOutcome};
use svm_text::parse_module;
use svm_verify::verify_module;

const RETURNED: i64 = 1;
const FAULTED: i64 = 2;

type BothOut = (Result<Vec<Value>, Trap>, Vec<u8>, JitOutcome, Vec<u8>);

/// The child ("plugin") module — see `separate_module.rs`: 64 KiB window, `data 100 "VM"`, an entry
/// that reads its data byte, stores a marker at 0, and returns `byte + 1000`.
fn child_src() -> &'static str {
    "memory 16
data 100 \"VM\"
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const 100
  v2 = i32.load8_u v1
  v3 = i64.const 0
  v4 = i32.const 7
  i32.store8 v3 v4
  v5 = i64.extend_i32_u v2
  v6 = i64.const 1000
  v7 = i64.add v5 v6
  return v7
}
"
}

/// Run `parent_src` on both backends with identical grants: an `Instantiator` over the whole 128 KiB
/// window and a `Module` capability for `child_src` (their handles passed as the entry's two args).
fn both(parent_src: &str) -> BothOut {
    let parent = parse_module(parent_src).expect("parse parent");
    verify_module(&parent).expect("verify parent");
    let child = parse_module(child_src()).expect("parse child");
    verify_module(&child).expect("verify child");
    let init: Vec<u8> = (0..(128u64 << 10))
        .map(|i| (i as u8).wrapping_mul(31) ^ 0xa5)
        .collect();

    let mut hi = Host::new();
    let ii = hi.grant_instantiator(0, 128 << 10);
    let mi = hi.grant_module(&child);
    let mut hj = Host::new();
    let ij = hj.grant_instantiator(0, 128 << 10);
    let mj = hj.grant_module(&child);
    assert_eq!((ii, mi), (ij, mj), "grants must encode identically");

    let mut fuel = 5_000_000u64;
    let (ir, imem) = run_capture_reserved_with_host(
        &parent,
        0,
        &[Value::I32(ii), Value::I32(mi)],
        &mut fuel,
        &init,
        0,
        &mut hi,
    );
    let (jo, jmem) = compile_and_run_capture_reserved_with_host_ex(
        &parent,
        0,
        &[ij as i64, mj as i64],
        &init,
        0,
        svm_run::cap_thunk,
        &mut hj as *mut Host as *mut core::ffi::c_void,
        Some(svm_run::module_resolver),
    )
    .expect("jit");
    (ir, imem, jo, jmem)
}

/// A child ("plugin") with two exported entries — `"alpha"` (func 0) → `byte+1000`, `"beta"` (func 1)
/// → `byte+2000` — to prove name→funcidx resolution selects the non-default entry identically on both
/// backends (see `separate_module.rs`).
fn named_child_src() -> &'static str {
    "memory 16
data 100 \"VM\"
export \"alpha\" 0
export \"beta\" 1
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const 100
  v2 = i32.load8_u v1
  v3 = i64.extend_i32_u v2
  v4 = i64.const 1000
  v5 = i64.add v3 v4
  return v5
}
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const 100
  v2 = i32.load8_u v1
  v3 = i64.extend_i32_u v2
  v4 = i64.const 2000
  v5 = i64.add v3 v4
  return v5
}
"
}

/// Like [`both`], but grants [`named_child_src`] (the two-export child) so a parent can resolve a child
/// entry by name on both backends.
fn both_named(parent_src: &str) -> BothOut {
    let parent = parse_module(parent_src).expect("parse parent");
    verify_module(&parent).expect("verify parent");
    let child = parse_module(named_child_src()).expect("parse child");
    verify_module(&child).expect("verify child");
    let init: Vec<u8> = (0..(128u64 << 10))
        .map(|i| (i as u8).wrapping_mul(31) ^ 0xa5)
        .collect();

    let mut hi = Host::new();
    let ii = hi.grant_instantiator(0, 128 << 10);
    let mi = hi.grant_module(&child);
    let mut hj = Host::new();
    let ij = hj.grant_instantiator(0, 128 << 10);
    let mj = hj.grant_module(&child);
    assert_eq!((ii, mi), (ij, mj), "grants must encode identically");

    let mut fuel = 5_000_000u64;
    let (ir, imem) = run_capture_reserved_with_host(
        &parent,
        0,
        &[Value::I32(ii), Value::I32(mi)],
        &mut fuel,
        &init,
        0,
        &mut hi,
    );
    let (jo, jmem) = compile_and_run_capture_reserved_with_host_ex(
        &parent,
        0,
        &[ij as i64, mj as i64],
        &init,
        0,
        svm_run::cap_thunk,
        &mut hj as *mut Host as *mut core::ffi::c_void,
        Some(svm_run::module_resolver),
    )
    .expect("jit");
    (ir, imem, jo, jmem)
}

#[test]
fn jit_module_child_by_export_name_matches_interp() {
    if !svm_jit::fiber_supported() {
        return; // no JIT nesting runtime on this target
    }
    // The parent resolves "beta" (func 1) via Module op 0 — routed through `cap_thunk` →
    // `cap_dispatch_slots` on the JIT, same as the interpreter — then instantiate_module's it → join.
    let parent = "memory 17
data 200 \"beta\"
func (i32, i32) -> (i64) {
block0(v0: i32, v1: i32):
  v2 = i64.const 200
  v3 = i64.const 4
  v4 = cap.call 8 0 (i64, i64) -> (i64) v1 (v2, v3)
  v5 = i64.extend_i32_s v1
  v6 = i64.const 0
  v7 = i64.const 65536
  v8 = i64.const 16
  v9 = cap.call 6 5 (i64, i64, i64, i64, i64) -> (i32) v0 (v5, v4, v7, v8, v6)
  v10 = cap.call 6 1 (i32) -> (i64) v0 (v9)
  return v10
}
";
    let (ir, imem, jo, jmem) = both_named(parent);
    let ival = ir.expect("interp ran ok").pop().expect("one result");
    assert_eq!(ival, Value::I64(86 + 2000), "interp name-addressed child");
    assert!(
        matches!(jo, JitOutcome::Returned(ref s) if s == &[86 + 2000]),
        "jit name-addressed child: {jo:?}"
    );
    assert_eq!(imem, jmem, "interp/JIT parent windows diverge");
}

#[test]
fn jit_module_child_matches_interp() {
    if !svm_jit::fiber_supported() {
        return; // no JIT nesting runtime on this target
    }
    // instantiate_module(module, entry 0, off 64 KiB, size 2^16, fuel 0) → join → child's result.
    let parent = "memory 17
func (i32, i32) -> (i64) {
block0(v0: i32, v1: i32):
  v2 = i64.extend_i32_s v1
  v3 = i64.const 0
  v4 = i64.const 65536
  v5 = i64.const 16
  v6 = cap.call 6 5 (i64, i64, i64, i64, i64) -> (i32) v0 (v2, v3, v4, v5, v3)
  v7 = cap.call 6 1 (i32) -> (i64) v0 (v6)
  return v7
}
";
    let (ir, imem, jo, jmem) = both(parent);
    let ival = ir.expect("interp ran ok").pop().expect("one result");
    assert_eq!(ival, Value::I64(86 + 1000), "interp module-child result");
    assert!(
        matches!(jo, JitOutcome::Returned(ref s) if s == &[86 + 1000]),
        "jit: {jo:?}"
    );
    // The escape-oracle over a foreign module: byte-identical parent windows — its data segment and
    // marker in the carve, everything outside it exactly as seeded.
    assert_eq!(imem, jmem, "interp/JIT parent windows diverge");
    const CHILD: u64 = 64 << 10;
    assert_eq!(&jmem[(CHILD + 100) as usize..(CHILD + 102) as usize], b"VM");
    assert_eq!(jmem[CHILD as usize], 7, "child marker missing on the JIT");
    let init: Vec<u8> = (0..(128u64 << 10))
        .map(|i| (i as u8).wrapping_mul(31) ^ 0xa5)
        .collect();
    for i in 0..CHILD {
        assert_eq!(
            jmem[i as usize], init[i as usize],
            "module child escaped to parent byte {i}"
        );
    }
}

/// Demand-paged module child: its data segments live in the carve, its pages start unmapped — the
/// first touch FAULTs to the parent (the reported address must match across backends: the segment's
/// page), the parent resumes without writing anything, and the child reads its lazily supplied
/// segment byte. Lazy plugin loading, identical on both backends.
#[test]
fn jit_demand_module_child_matches_interp() {
    if !svm_jit::fiber_supported() {
        return;
    }
    let parent = "memory 17
func (i32, i32) -> (i64) {
block0(v0: i32, v1: i32):
  v2 = i64.extend_i32_s v1
  v3 = i64.const 0
  v4 = i64.const 65536
  v5 = i64.const 16
  v6 = cap.call 6 7 (i64, i64, i64, i64, i64) -> (i32) v0 (v2, v3, v4, v5, v3)
  v7, v8 = cap.call 6 3 (i32, i64) -> (i32, i64) v0 (v6, v3)
  v9, v10 = cap.call 6 3 (i32, i64) -> (i32, i64) v0 (v6, v3)
  v11 = i64.extend_i32_s v7
  v12 = i64.const 1000000000
  v13 = i64.mul v11 v12
  v14 = i64.extend_i32_s v9
  v15 = i64.const 100000000
  v16 = i64.mul v14 v15
  v17 = i64.const 1000000
  v18 = i64.mul v8 v17
  v19 = i64.add v10 v13
  v20 = i64.add v19 v16
  v21 = i64.add v20 v18
  return v21
}
";
    let (ir, imem, jo, jmem) = both(parent);
    // status1 = FAULTED at the child's first touched page (its data read at child offset 100 →
    // parent address 64 KiB + 100 = 65636), status2 = RETURNED with 1086 ('V' + 1000).
    let want = 1086 + FAULTED * 1_000_000_000 + RETURNED * 100_000_000 + 65636 * 1_000_000;
    let ival = ir.expect("interp ran ok").pop().expect("one result");
    assert_eq!(ival, Value::I64(want), "interp lazy module-child");
    assert!(
        matches!(jo, JitOutcome::Returned(ref s) if s == &[want]),
        "jit: {jo:?}"
    );
    assert_eq!(imem, jmem, "interp/JIT parent windows diverge");
}

/// Validation parity: a carve that doesn't equal the module's declared memory is `-EINVAL`, and a
/// forged module handle (the Instantiator handle passed as a Module handle) is a `CapFault` — on
/// both backends.
#[test]
fn jit_module_child_validation_matches_interp() {
    if !svm_jit::fiber_supported() {
        return;
    }
    // (a) wrong carve size → -EINVAL.
    let parent = "memory 17
func (i32, i32) -> (i64) {
block0(v0: i32, v1: i32):
  v2 = i64.extend_i32_s v1
  v3 = i64.const 0
  v4 = i64.const 65536
  v5 = i64.const 12
  v6 = cap.call 6 5 (i64, i64, i64, i64, i64) -> (i32) v0 (v2, v3, v4, v5, v3)
  v7 = i64.extend_i32_s v6
  return v7
}
";
    let (ir, _i, jo, _j) = both(parent);
    let ival = ir.expect("interp ran ok").pop().expect("one result");
    assert_eq!(ival, Value::I64(-22), "interp: carve ≠ declared memory");
    assert!(
        matches!(jo, JitOutcome::Returned(ref s) if s == &[-22]),
        "jit: {jo:?}"
    );

    // (b) forged module handle → CapFault.
    let parent = "memory 17
func (i32, i32) -> (i64) {
block0(v0: i32, v1: i32):
  v2 = i64.extend_i32_s v0
  v3 = i64.const 0
  v4 = i64.const 65536
  v5 = i64.const 16
  v6 = cap.call 6 5 (i64, i64, i64, i64, i64) -> (i32) v0 (v2, v3, v4, v5, v3)
  v7 = i64.extend_i32_s v6
  return v7
}
";
    let (ir, _i, jo, _j) = both(parent);
    assert!(
        matches!(ir, Err(Trap::CapFault)),
        "interp: forged module handle must CapFault, got {ir:?}"
    );
    assert!(matches!(jo, JitOutcome::Trapped(_)), "jit: {jo:?}");
}
