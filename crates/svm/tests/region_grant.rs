//! §13/§14 **cross-domain `SharedRegion`** — guest-minted regions (`create`) and granting them into
//! a child domain (`grant`): the zero-copy parent↔child data plane.
//!
//! - **`create`** (`AddressSpace` op 5, `create_region(len) -> handle`): the memory-management
//!   authority mints a fresh shareable region. Rides the generic dispatch, so it works on **both
//!   backends** (the JIT host installs `svm_run::new_shared_region` as the backing factory, so a JIT
//!   guest can `map` what it mints for real hardware aliasing) — covered differentially below.
//! - **`grant`** (`SharedRegion` op 4, `grant(coro_child) -> child_handle`): installs the *same*
//!   backing into a suspended coroutine child's powerbox; the parent hands the returned handle to
//!   the child via the next `resume` value, and the child `map`s the region into its own window —
//!   parent and child then share bytes through the region with no copies. Interp-first (the JIT
//!   child's powerbox is a baked thunk — a follow-up, like the Instantiator's own JIT port was).

use svm_interp::{run_capture_reserved_with_host, Host, Trap, Value};
use svm_text::parse_module;
use svm_verify::verify_module;

const RETURNED: i64 = 1;

/// Run `src`'s entry on the interpreter with `(Instantiator over the whole window, AddressSpace over
/// the whole window)` as its two args, over a 128 KiB window.
fn run_interp(src: &str) -> (Result<Vec<Value>, Trap>, Vec<u8>) {
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");
    let mut host = Host::new();
    let ih = host.grant_instantiator(0, 128 << 10);
    let ah = host.grant_address_space(0, 128 << 10);
    let mut fuel = 5_000_000u64;
    run_capture_reserved_with_host(
        &m,
        0,
        &[Value::I32(ih), Value::I32(ah)],
        &mut fuel,
        &[0u8; 128 << 10],
        0,
        &mut host,
    )
}

/// The headline: a **guest-minted region shared across domains, zero-copy**. The parent
/// `create_region`s a 64 KiB region, maps it at its own window offset 0, and stores `204` there
/// (→ region byte 0). It spawns a coroutine child (64 KiB at 64 KiB), `grant`s it the region, and
/// delivers the child-side handle via the resume value. The child maps the region into *its* window
/// and reads byte 0 — the parent's write arrives through the shared backing, no copies, across the
/// domain boundary.
#[test]
fn minted_region_granted_to_child_shares_bytes() {
    let src = "memory 17
func (i32, i32) -> (i64) {
block0(v0: i32, v1: i32):
  v2 = i64.const 65536
  v3 = cap.call 5 5 (i64) -> (i64) v1 (v2)
  v4 = i32.wrap_i64 v3
  v5 = i64.const 0
  v6 = i32.const 3
  v7 = cap.call 4 0 (i64, i64, i64, i32) -> (i64) v4 (v5, v5, v2, v6)
  v8 = i32.const 204
  i32.store8 v5 v8
  v9 = i64.const 1
  v10 = i64.const 16
  v11 = cap.call 6 2 (i64, i64, i64, i64) -> (i32) v0 (v9, v2, v10, v5)
  v12, v13 = cap.call 6 3 (i32, i64) -> (i32, i64) v0 (v11, v5)
  v14 = cap.call 4 4 (i32) -> (i64) v4 (v11)
  v15, v16 = cap.call 6 3 (i32, i64) -> (i32, i64) v0 (v11, v14)
  v17 = i64.extend_i32_s v15
  v18 = i64.const 1000
  v19 = i64.mul v17 v18
  v20 = i64.add v16 v19
  return v20
}
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i32.wrap_i64 v0
  v2 = i64.const 0
  v3 = cap.call 7 0 (i64) -> (i64) v1 (v2)
  v4 = i32.wrap_i64 v3
  v5 = i64.const 65536
  v6 = i32.const 3
  v7 = cap.call 4 0 (i64, i64, i64, i32) -> (i64) v4 (v2, v2, v5, v6)
  v8 = i32.load8_u v2
  v9 = i64.extend_i32_u v8
  return v9
}
";
    let (res, _mem) = run_interp(src);
    // The final resume RETURNs (status 1) the byte the parent wrote into the shared region: 204.
    let want = 204 + RETURNED * 1000;
    assert_eq!(
        res.expect("run ok"),
        vec![Value::I64(want)],
        "the child must read the parent's write through the granted region (zero-copy)"
    );
}

/// And the reverse direction: the **child writes**, the parent reads — after the child returns, the
/// parent loads its own region-mapped window byte and sees the child's value (the same shared
/// backing, no copy-back machinery involved).
#[test]
fn child_write_via_granted_region_reaches_parent() {
    let src = "memory 17
func (i32, i32) -> (i64) {
block0(v0: i32, v1: i32):
  v2 = i64.const 65536
  v3 = cap.call 5 5 (i64) -> (i64) v1 (v2)
  v4 = i32.wrap_i64 v3
  v5 = i64.const 0
  v6 = i32.const 3
  v7 = cap.call 4 0 (i64, i64, i64, i32) -> (i64) v4 (v5, v5, v2, v6)
  v8 = i64.const 1
  v9 = i64.const 16
  v10 = cap.call 6 2 (i64, i64, i64, i64) -> (i32) v0 (v8, v2, v9, v5)
  v11, v12 = cap.call 6 3 (i32, i64) -> (i32, i64) v0 (v10, v5)
  v13 = cap.call 4 4 (i32) -> (i64) v4 (v10)
  v14, v15 = cap.call 6 3 (i32, i64) -> (i32, i64) v0 (v10, v13)
  v16 = i32.load8_u v5
  v17 = i64.extend_i32_u v16
  return v17
}
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i32.wrap_i64 v0
  v2 = i64.const 0
  v3 = cap.call 7 0 (i64) -> (i64) v1 (v2)
  v4 = i32.wrap_i64 v3
  v5 = i64.const 65536
  v6 = i32.const 3
  v7 = cap.call 4 0 (i64, i64, i64, i32) -> (i64) v4 (v2, v2, v5, v6)
  v8 = i32.const 99
  i32.store8 v2 v8
  v9 = i64.const 0
  return v9
}
";
    let (res, _mem) = run_interp(src);
    // The parent's load at its region-mapped offset 0 sees the child's 99.
    assert_eq!(
        res.expect("run ok"),
        vec![Value::I64(99)],
        "the parent must read the child's write through the shared region"
    );
}

/// `create_region` validation: a zero or over-cap length is `-EINVAL`; `grant` of a real region to a
/// nonexistent coroutine child is an inert `CapFault`.
#[test]
fn create_and_grant_validation() {
    // len = 0 → -EINVAL.
    let src = "memory 17
func (i32, i32) -> (i64) {
block0(v0: i32, v1: i32):
  v2 = i64.const 0
  v3 = cap.call 5 5 (i64) -> (i64) v1 (v2)
  return v3
}
";
    let (res, _m) = run_interp(src);
    assert_eq!(res.expect("run ok"), vec![Value::I64(-22)], "len 0");

    // len > the 256 MiB per-region cap → -EINVAL.
    let src = "memory 17
func (i32, i32) -> (i64) {
block0(v0: i32, v1: i32):
  v2 = i64.const 268435457
  v3 = cap.call 5 5 (i64) -> (i64) v1 (v2)
  return v3
}
";
    let (res, _m) = run_interp(src);
    assert_eq!(res.expect("run ok"), vec![Value::I64(-22)], "len > cap");

    // grant to a nonexistent coroutine child → CapFault.
    let src = "memory 17
func (i32, i32) -> (i64) {
block0(v0: i32, v1: i32):
  v2 = i64.const 65536
  v3 = cap.call 5 5 (i64) -> (i64) v1 (v2)
  v4 = i32.wrap_i64 v3
  v5 = i32.const 7
  v6 = cap.call 4 4 (i32) -> (i64) v4 (v5)
  return v6
}
";
    let (res, _m) = run_interp(src);
    assert!(
        matches!(res, Err(Trap::CapFault)),
        "grant to a bogus child must CapFault, got {res:?}"
    );
}

/// **`create_region` differentially on both backends**: a guest mints a region via its
/// `AddressSpace` and maps it at two window offsets — a store through one alias must read back
/// through the other, identically on the interpreter and the JIT (whose host installs the
/// OS-shared-memory factory so the minted region is genuinely `mmap`-able). The §13 alias
/// differential, now over a guest-minted region.
#[test]
fn jit_minted_region_aliases_match_interp() {
    use svm_jit::{compile_and_run_capture_reserved_with_host, JitOutcome};

    // Query the region granularity (op 3), mint a 64 KiB region, map it at window offsets 0 and
    // `granule`, store 171 at 0, load at `granule` — the alias must read it back.
    let src = "memory 17
func (i32) -> (i64) {
block0(v0: i32):
  v1 = i64.const 65536
  v2 = cap.call 5 5 (i64) -> (i64) v0 (v1)
  v3 = i32.wrap_i64 v2
  v4 = cap.call 4 3 () -> (i64) v3 ()
  v5 = i64.const 0
  v6 = i32.const 3
  v7 = cap.call 4 0 (i64, i64, i64, i32) -> (i64) v3 (v5, v5, v4, v6)
  v8 = cap.call 4 0 (i64, i64, i64, i32) -> (i64) v3 (v4, v5, v4, v6)
  v9 = i32.const 171
  i32.store8 v5 v9
  v10 = i32.load8_u v4
  v11 = i64.extend_i32_u v10
  return v11
}
";
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");

    let mut hi = Host::new();
    hi.set_region_factory(svm_run::new_shared_region);
    let ai = hi.grant_address_space(0, 128 << 10);
    let mut hj = Host::new();
    hj.set_region_factory(svm_run::new_shared_region);
    let aj = hj.grant_address_space(0, 128 << 10);
    assert_eq!(ai, aj, "grants must encode identically");

    let mut fuel = 1_000_000u64;
    let (ir, imem) = run_capture_reserved_with_host(
        &m,
        0,
        &[Value::I32(ai)],
        &mut fuel,
        &[0u8; 128 << 10],
        0,
        &mut hi,
    );
    let (jo, jmem) = compile_and_run_capture_reserved_with_host(
        &m,
        0,
        &[aj as i64],
        &[0u8; 128 << 10],
        0,
        svm_run::cap_thunk,
        &mut hj as *mut Host as *mut core::ffi::c_void,
    )
    .expect("jit");

    let ival = ir.expect("interp ran ok").pop().expect("one result");
    assert_eq!(ival, Value::I64(171), "interp: minted-region alias");
    assert!(
        matches!(jo, JitOutcome::Returned(ref s) if s == &[171]),
        "jit: {jo:?}"
    );
    assert_eq!(
        imem, jmem,
        "interp/JIT windows diverge over a minted region"
    );
}
