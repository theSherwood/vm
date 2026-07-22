//! PROCESS.md S1b — **canonical-key futex** over a §13 `SharedRegion`. Two mappings of one region at
//! *different* window offsets alias the same bytes; the futex wait-queue must key on the region's
//! canonical `(backing, offset)` identity, not the window-absolute address, so a `notify` through one
//! alias wakes a waiter parked through the other. (Linux draws the same line: `FUTEX_PRIVATE` keys on
//! the virtual address, a shared futex on the backing page.)
//!
//! This isolates the **notify path** specifically — not the value path, which already works for
//! aliases (a store through one alias is visible through the other). The parent parks on the region
//! byte via **alias B** with the value left at its expected `0`, so *only* a `notify` can wake it; a
//! spawned child spin-`notify`s the *same region byte* via **alias A**. With canonical keys the two
//! aliases produce the same futex key and the parent is woken (`atomic.wait` → status `0`); without
//! them the notify misses, the parent never wakes, and the child spins to a fuel trap — so a
//! regression turns this from `Ok(I64(0))` into a trap. Ordering is safe on any worker count: on a
//! single worker the parent parks (freeing the worker) before the child runs, so the child's first
//! notify lands; on several workers the child spins until the parent has parked.

use svm_interp::{run_capture_reserved_with_host, Host, Value};
use svm_text::parse_module;
use svm_verify::verify_module;

/// func 0 (parent): query the region granule `G` (op 3), map the region at window offset `0`
/// (alias A) and again at `G` (alias B) — both covering region byte 0 — spawn the notifier child,
/// then `atomic.wait` on **alias B** (window `G`, region byte 0), expected `0`, no timeout. Returns
/// the wait status: `0` iff a `notify` woke it.
///
/// func 1 (child): spin-`notify` **alias A** (window `0`, region byte 0) until it reports a waiter
/// woken, then return `7`. Unbounded on purpose — if the keys don't match this never succeeds and the
/// child fuel-traps, which fails the test loudly instead of hanging.
const SRC: &str = "memory 17\n\
func (i32) -> (i64) {\n\
block 0 (v0: i32) {\n\
  vps = cap.call 4 3 () -> (i64) v0 ()\n\
  vz = i64.const 0\n\
  vprot = i32.const 3\n\
  vm1 = cap.call 4 0 (i64, i64, i64, i32) -> (i64) v0 (vz, vz, vps, vprot)\n\
  vm2 = cap.call 4 0 (i64, i64, i64, i32) -> (i64) v0 (vps, vz, vps, vprot)\n\
  vchild = thread.spawn 1 vz vz\n\
  vexp = i32.const 0\n\
  vto = i64.const -1\n\
  vst = i32.atomic.wait vps vexp vto\n\
  vjr = thread.join vchild\n\
  vst64 = i64.extend_i32_u vst\n\
  return vst64\n\
  }\n\
}\n\
func (i64, i64) -> (i64) {\n\
block 0 (vsp: i64, varg: i64) {\n\
  br 1()\n\
}\n\
block 1 () {\n\
  v0 = i64.const 0\n\
  v1 = i32.const 1\n\
  vw = atomic.notify v0 v1\n\
  vzero = i32.const 0\n\
  vgt = i32.lt_u vzero vw\n\
  br_if vgt 2() 1()\n\
}\n\
block 2 () {\n\
  v7 = i64.const 7\n\
  return v7\n\
  }\n\
}\n";

#[test]
fn notify_through_alias_wakes_waiter_on_the_other() {
    let m = parse_module(SRC).expect("parse");
    verify_module(&m).expect("verify");

    // Run several times to exercise different real-pool interleavings; every run must wake (status 0).
    for iter in 0..16 {
        let mut host = Host::new();
        let h = host.grant_shared_region(1 << 16); // 64 KiB region ≥ any host granule
        let mut fuel = 50_000_000u64;
        let (res, _snap) =
            run_capture_reserved_with_host(&m, 0, &[Value::I32(h)], &mut fuel, &[], 0, &mut host);
        assert_eq!(
            res,
            Ok(vec![Value::I64(0)]),
            "iter {iter}: a notify through alias A must wake the waiter parked through alias B \
             (status 0 = woken); a trap/other status means the aliases keyed differently"
        );
    }
}
