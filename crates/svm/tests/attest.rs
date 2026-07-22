//! PROCESS.md §6 — `cap.self.attest`: the non-interposable **trust anchor**. A domain reads its own
//! platform-vouched provenance (isolation tier + whether an ancestor can read/snapshot it) as a packed
//! `i32` — `tier | (window_exposed << 8) | (freeze_exposed << 9)`. Being a D46 `cap.self` intrinsic
//! (runtime-resolved, never a handle), no nested host can interpose and forge it.
//!
//! `attest` routes through the same `Host::self_dispatch` on every backend (like `cap.self.count`), so
//! the **root** report is a cross-backend differential. A §14 nested child's report (window-exposed)
//! is interpreter-first: a plain JIT child has an empty powerbox, so `cap.self.*` there is a follow-up.

use svm_interp::{run_capture_reserved_with_host, Attestation, Host, Value};
use svm_jit::{compile_and_run_capture_reserved_with_host, JitOutcome};
use svm_text::parse_module;
use svm_verify::verify_module;

/// func 0 `(i32) -> (i64)` (arg ignored): return `cap.self.attest` zero-extended to `i64`.
const ATTEST_SELF: &str = "memory 17\n\
func (i32) -> (i64) {\n\
block 0 (v0: i32) {\n\
  va = cap.self.attest\n\
  vr = i64.extend_i32_u va\n\
  return vr\n\
  }\n\
}\n";

fn run_interp(src: &str, host: &mut Host) -> Result<Vec<Value>, svm_interp::Trap> {
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");
    let mut fuel = 5_000_000u64;
    run_capture_reserved_with_host(
        &m,
        0,
        &[Value::I32(0)],
        &mut fuel,
        &[0u8; 128 << 10],
        0,
        host,
    )
    .0
}

fn run_jit(src: &str, host: &mut Host) -> JitOutcome {
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");
    compile_and_run_capture_reserved_with_host(
        &m,
        0,
        &[0i64],
        &[0u8; 128 << 10],
        0,
        svm_run::cap_thunk,
        host as *mut Host as *mut core::ffi::c_void,
    )
    .expect("jit")
    .0
}

/// The root's `attest` report packs `tier | (exposed << 8) | (freeze << 9)` and is byte-identical on
/// both backends (`self_dispatch` is shared). Checked across a few provenance configs.
fn both_root(att: Attestation) -> (Result<Vec<Value>, svm_interp::Trap>, JitOutcome) {
    let mut ih = Host::new();
    ih.set_attestation(att);
    let ir = run_interp(ATTEST_SELF, &mut ih);
    let mut jh = Host::new();
    jh.set_attestation(att);
    let jo = run_jit(ATTEST_SELF, &mut jh);
    (ir, jo)
}

fn expect_both(att: Attestation, packed: i64) {
    let (ir, jo) = both_root(att);
    assert_eq!(ir, Ok(vec![Value::I64(packed)]), "interp: attest packing");
    assert!(
        matches!(jo, JitOutcome::Returned(ref s) if s == &[packed]),
        "jit: attest must match interp ({packed}), got {jo:?}"
    );
}

#[test]
fn root_attest_default_is_tier1_unexposed() {
    // A fresh Host is a root domain: tier 1, no ancestor read, not ancestor-freezable → packed 1.
    expect_both(Attestation::default(), 1);
}

#[test]
fn root_attest_packs_tier_and_exposure_bits() {
    // tier 3 (separate-process): packed 3.
    expect_both(
        Attestation {
            tier: 3,
            window_exposed: false,
            freeze_exposed: false,
        },
        3,
    );
    // window-exposed sets bit 8; freeze-exposed sets bit 9. tier 1 + both → 1 | 256 | 512 = 769.
    expect_both(
        Attestation {
            tier: 1,
            window_exposed: true,
            freeze_exposed: true,
        },
        769,
    );
}

/// A §14 nested child (interpreter): the parent `instantiate`s a child that returns its own `attest`,
/// `join`s it, and returns `child_attest * 1000 + parent_attest`. The parent is the root (packed `1`);
/// the child is **window-exposed** (its carve is a superset the parent reads) and non-durable, so its
/// report is `1 | (1 << 8)` = `257`. Result `257 * 1000 + 1` = `257001`.
const NESTED_ATTEST: &str = "memory 17\n\
func (i32) -> (i64) {\n\
block 0 (vinst: i32) {\n\
  ventry = i64.const 1\n\
  voff = i64.const 0\n\
  vsl = i64.const 12\n\
  vq = i64.const 0\n\
  vch = cap.call 6 0 (i64, i64, i64, i64) -> (i32) vinst (ventry, voff, vsl, vq)\n\
  vcr = cap.call 6 1 (i32) -> (i64) vinst (vch)\n\
  vpa = cap.self.attest\n\
  vpa64 = i64.extend_i32_u vpa\n\
  vk = i64.const 1000\n\
  vt = i64.mul vcr vk\n\
  vsum = i64.add vt vpa64\n\
  return vsum\n\
  }\n\
}\n\
func (i64) -> (i64) {\n\
block 0 (vci: i64) {\n\
  va = cap.self.attest\n\
  vr = i64.extend_i32_u va\n\
  return vr\n\
  }\n\
}\n";

#[test]
fn nested_child_reports_window_exposed_interp() {
    let m = parse_module(NESTED_ATTEST).expect("parse");
    verify_module(&m).expect("verify");
    let mut host = Host::new();
    let ih = host.grant_instantiator(0, 128 << 10);
    let mut fuel = 50_000_000u64;
    let r = run_capture_reserved_with_host(
        &m,
        0,
        &[Value::I32(ih)],
        &mut fuel,
        &[0u8; 128 << 10],
        0,
        &mut host,
    )
    .0;
    // child (exposed) = 257, parent (root) = 1 → 257001. The child *learns it is exposed* — the whole
    // point of the trust anchor.
    assert_eq!(
        r,
        Ok(vec![Value::I64(257_001)]),
        "nested child must see window_exposed (257); root does not (1)"
    );
}
