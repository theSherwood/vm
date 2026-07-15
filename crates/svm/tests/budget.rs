//! PROCESS.md §5 / §15 — `Budget` (iface 14): a passable, **splittable** resource-quota vector
//! `(fuel, mem, spawn)`. §15's principle — "every meterable resource is already a capability with a
//! quota" — promoted to an object a domain can `split` (attenuate a sub-budget) and `read` (monitor).
//!
//! `Budget` is an ordinary capability (it dispatches through the generic `cap.call` path, not the
//! eval-loop-serviced `Instantiator`), so the interpreter and the JIT service it through the **same**
//! `Host::cap_dispatch_slots` — these tests run each program on both backends and assert identical
//! results (parity for free, like `Stream`/`Clock`). Charging a domain's live consumption against its
//! budget (the `create(module, window, budget)` accounting) is the follow-up; this pins the object.

use svm_interp::{run_capture_reserved_with_host, Host, Value};
use svm_jit::{compile_and_run_capture_reserved_with_host, JitOutcome};
use svm_text::parse_module;
use svm_verify::verify_module;

/// Run `src`'s func 0 on the interpreter with `bh` (a Budget handle) as its single `i32` arg.
fn run_interp(src: &str, host: &mut Host, bh: i32) -> Result<Vec<Value>, svm_interp::Trap> {
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");
    let mut fuel = 5_000_000u64;
    run_capture_reserved_with_host(
        &m,
        0,
        &[Value::I32(bh)],
        &mut fuel,
        &[0u8; 128 << 10],
        0,
        host,
    )
    .0
}

/// Same program + Budget handle on the JIT (`bh` widened into the entry's `i64` arg slot).
fn run_jit(src: &str, host: &mut Host, bh: i32) -> JitOutcome {
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");
    compile_and_run_capture_reserved_with_host(
        &m,
        0,
        &[bh as i64],
        &[0u8; 128 << 10],
        0,
        svm_run::cap_thunk,
        host as *mut Host as *mut core::ffi::c_void,
    )
    .expect("jit")
    .0
}

/// `split(300, 200, 3)` out of a `(1000, 500, 10)` budget, then read the parent's fuel remaining and
/// the child's whole vector; encode all four as `((parent_fuel*1000 + child_fuel)*1000 +
/// child_mem)*1000 + child_spawn` = `((700*1000 + 300)*1000 + 200)*1000 + 3` = `700300200003`.
const SPLIT_AND_READ: &str = "memory 17\n\
func (i32) -> (i64) {\n\
block0(vb: i32):\n\
  f300 = i64.const 300\n\
  m200 = i64.const 200\n\
  s3 = i64.const 3\n\
  vsub = cap.call 14 0 (i64, i64, i64) -> (i32) vb (f300, m200, s3)\n\
  fld0 = i64.const 0\n\
  fld1 = i64.const 1\n\
  fld2 = i64.const 2\n\
  vpf = cap.call 14 1 (i64) -> (i64) vb (fld0)\n\
  vcf = cap.call 14 1 (i64) -> (i64) vsub (fld0)\n\
  vcm = cap.call 14 1 (i64) -> (i64) vsub (fld1)\n\
  vcs = cap.call 14 1 (i64) -> (i64) vsub (fld2)\n\
  k1000 = i64.const 1000\n\
  t0 = i64.mul vpf k1000\n\
  t1 = i64.add t0 vcf\n\
  t2 = i64.mul t1 k1000\n\
  t3 = i64.add t2 vcm\n\
  t4 = i64.mul t3 k1000\n\
  t5 = i64.add t4 vcs\n\
  return t5\n\
}\n";

/// `split(2000, 0, 0)` out of a `(1000, …)` budget over-asks the bounded fuel field, so the whole
/// split fails closed with `-EINVAL` (`-22`); nothing is deducted.
const OVER_SPLIT: &str = "memory 17\n\
func (i32) -> (i64) {\n\
block0(vb: i32):\n\
  big = i64.const 2000\n\
  z = i64.const 0\n\
  vsub = cap.call 14 0 (i64, i64, i64) -> (i32) vb (big, z, z)\n\
  vr = i64.extend_i32_s vsub\n\
  return vr\n\
}\n";

/// `split(-1, -1, -1)` takes **all remaining** of every field; the parent is left at `(0, 0, 0)`.
/// Encode `((child_fuel*1000 + parent_fuel)*1000 + parent_spawn)` = `((1000*1000 + 0)*1000 + 0)` =
/// `1000000000`.
const SPLIT_ALL: &str = "memory 17\n\
func (i32) -> (i64) {\n\
block0(vb: i32):\n\
  all = i64.const -1\n\
  vsub = cap.call 14 0 (i64, i64, i64) -> (i32) vb (all, all, all)\n\
  fld0 = i64.const 0\n\
  fld2 = i64.const 2\n\
  vcf = cap.call 14 1 (i64) -> (i64) vsub (fld0)\n\
  vpf = cap.call 14 1 (i64) -> (i64) vb (fld0)\n\
  vps = cap.call 14 1 (i64) -> (i64) vb (fld2)\n\
  k1000 = i64.const 1000\n\
  t0 = i64.mul vcf k1000\n\
  t1 = i64.add t0 vpf\n\
  t2 = i64.mul t1 k1000\n\
  t3 = i64.add t2 vps\n\
  return t3\n\
}\n";

fn both(src: &str, budget: (i64, i64, i64)) -> (Result<Vec<Value>, svm_interp::Trap>, JitOutcome) {
    let mut ih = Host::new();
    let ibh = ih.grant_budget(budget.0, budget.1, budget.2);
    let ir = run_interp(src, &mut ih, ibh);
    let mut jh = Host::new();
    let jbh = jh.grant_budget(budget.0, budget.1, budget.2);
    let jo = run_jit(src, &mut jh, jbh);
    (ir, jo)
}

#[test]
fn split_and_read_matches_across_backends() {
    let (ir, jo) = both(SPLIT_AND_READ, (1000, 500, 10));
    assert_eq!(
        ir,
        Ok(vec![Value::I64(700_300_200_003)]),
        "interp: parent 700 fuel left; child holds (300, 200, 3)"
    );
    assert!(
        matches!(jo, JitOutcome::Returned(ref s) if s == &[700_300_200_003]),
        "jit: must match interp, got {jo:?}"
    );
}

#[test]
fn over_split_is_einval_on_both() {
    let (ir, jo) = both(OVER_SPLIT, (1000, 500, 10));
    assert_eq!(
        ir,
        Ok(vec![Value::I64(-22)]),
        "interp: over-split -> -EINVAL"
    );
    assert!(
        matches!(jo, JitOutcome::Returned(ref s) if s == &[-22]),
        "jit: over-split must be -EINVAL, got {jo:?}"
    );
}

#[test]
fn split_all_drains_parent_on_both() {
    let (ir, jo) = both(SPLIT_ALL, (1000, 500, 10));
    assert_eq!(
        ir,
        Ok(vec![Value::I64(1_000_000_000)]),
        "interp: child took all 1000 fuel; parent left at 0"
    );
    assert!(
        matches!(jo, JitOutcome::Returned(ref s) if s == &[1_000_000_000]),
        "jit: must match interp, got {jo:?}"
    );
}
