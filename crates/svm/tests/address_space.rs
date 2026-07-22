//! §14 `AddressSpace` capability + **attenuation** (iface 5), end-to-end through the real
//! `cap.call` dispatch on **both** backends. An `AddressSpace` is the memory half of the
//! `Instantiator`: a power-of-two window sub-range whose `map`/`unmap`/`protect` are confined to
//! it, and whose `sub` op mints a *further-attenuated* child range (a parent can only sub-allocate
//! what it holds). These tests pin two properties: the attenuated ops take effect (and agree
//! interp↔JIT, the §18 oracle) *and* an op reaching outside the holder's sub-range is rejected
//! (`-EINVAL`) without touching a byte the holder doesn't own — confinement of the authority, not
//! just of the access.

use svm_interp::{cap_id, run_capture_reserved_with_host, Host, Value};
use svm_jit::{compile_and_run_capture_reserved_with_host, JitOutcome};
use svm_text::parse_module;
use svm_verify::verify_module;

const EINVAL: i64 = -22;

/// Parse+verify `src`, grant **both** an interp `Host` and a JIT `Host` an identical root
/// `AddressSpace` over the whole window `[0, 1<<size_log2)`, run the entry (the granted handle is
/// `v0`) on each over a fully-mapped window seeded with `init`, and return both results and both
/// final-window snapshots. The two `Host`s grant the root identically, so the minted child handles
/// (slot+generation) match across backends and the snapshots are directly comparable.
fn both(src: &str, size_log2: u8, init: &[u8]) -> (Value, Vec<u8>, JitOutcome, Vec<u8>) {
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");
    let win = 1u64 << size_log2;

    let mut hi = Host::new();
    let hi_handle = hi.grant_address_space(0, win);
    let mut hj = Host::new();
    let hj_handle = hj.grant_address_space(0, win);
    assert_eq!(hi_handle, hj_handle, "root grants must encode identically");

    let mut fuel = 1_000_000u64;
    let (ir, imem) = run_capture_reserved_with_host(
        &m,
        0,
        &[Value::I32(hi_handle)],
        &mut fuel,
        init,
        0,
        &mut hi,
    );
    let (jo, jmem) = compile_and_run_capture_reserved_with_host(
        &m,
        0,
        &[hj_handle as i64],
        init,
        0,
        svm_run::cap_thunk,
        &mut hj as *mut Host as *mut core::ffi::c_void,
    )
    .expect("jit");
    let ival = ir.expect("interp ran ok").pop().expect("one result");
    (ival, imem, jo, jmem)
}

/// A 128 KiB window with a non-zero seed so an `unmap` (which makes a page read back as zero) is
/// observable against the seed, and an *escaped* effect would perturb a byte the seed pins.
fn seed_128k() -> Vec<u8> {
    (0..(128u64 << 10))
        .map(|i| (i as u8).wrapping_mul(31) ^ 0xa5)
        .collect()
}

/// Attenuate the root `AddressSpace` to a 64 KiB child at window offset 64 KiB, then through the
/// **child** unmap its first 16 KiB (child-relative `[0, 16 KiB)` → window `[64 KiB, 80 KiB)`). A
/// 16 KiB span is a whole multiple of every host page size (4 KiB / 16 KiB), so the unmap fully
/// decommits it on either backend. It must succeed (return 0) and zero exactly that span, leaving
/// the rest as seeded — so the attenuated, base-shifted op took effect identically interp↔JIT.
#[test]
fn attenuated_unmap_takes_effect_and_matches() {
    let src = "\
memory 17
func (i32) -> (i64) {
block 0 (v0: i32) {
  v2 = i64.const 65536
  v3 = i64.const 16
  v4 = cap.call 5 4 (i64, i64) -> (i32) v0 (v2, v3)
  v5 = i64.const 0
  v7 = i64.const 16384
  v6 = cap.call 5 1 (i64, i64) -> (i64) v4 (v5, v7)
  return v6
  }
}
";
    let init = seed_128k();
    let (ival, imem, jo, jmem) = both(src, 17, &init);
    assert_eq!(ival, Value::I64(0), "attenuated unmap should succeed");
    assert!(
        matches!(jo, JitOutcome::Returned(ref s) if s == &[0]),
        "jit: {jo:?}"
    );
    assert_eq!(
        imem, jmem,
        "interp/JIT windows diverge after attenuated unmap"
    );

    let (base, span) = (64u64 << 10, 16u64 << 10);
    for i in 0..(128u64 << 10) {
        if (base..base + span).contains(&i) {
            assert_eq!(imem[i as usize], 0, "child's unmapped byte {i} not zeroed");
        } else {
            assert_eq!(
                imem[i as usize], init[i as usize],
                "byte {i} outside the unmap changed"
            );
        }
    }
}

/// Confinement of the *authority*: a child attenuated to the low 4 KiB cannot reach a window page
/// outside its sub-range. Through that child, `unmap(4096, 16384)` (child-relative — starting just
/// past its 4 KiB range) must be rejected `-EINVAL` and leave the window *entirely* as seeded, even
/// though those underlying window pages exist and the root holder could have unmapped them.
#[test]
fn attenuated_op_outside_subrange_is_rejected() {
    let src = "\
memory 17
func (i32) -> (i64) {
block 0 (v0: i32) {
  v2 = i64.const 0
  v3 = i64.const 12
  v4 = cap.call 5 4 (i64, i64) -> (i32) v0 (v2, v3)
  v5 = i64.const 4096
  v7 = i64.const 16384
  v6 = cap.call 5 1 (i64, i64) -> (i64) v4 (v5, v7)
  return v6
  }
}
";
    let init = seed_128k();
    let (ival, imem, jo, jmem) = both(src, 17, &init);
    assert_eq!(
        ival,
        Value::I64(EINVAL),
        "out-of-subrange unmap must be -EINVAL"
    );
    assert!(
        matches!(jo, JitOutcome::Returned(ref s) if s == &[EINVAL]),
        "jit: {jo:?}"
    );
    assert_eq!(imem, jmem, "interp/JIT windows diverge");
    // A rejected op touches nothing — the whole window stays exactly as seeded.
    assert_eq!(imem, init, "a rejected attenuated op perturbed the window");
}

/// Attenuation is itself confined: `sub` only carves a power-of-two-aligned sub-range that lies
/// **within** the holder's range. A child over the low 4 KiB cannot mint a grandchild that is
/// larger than it, unaligned, or spilling past its end — each is `-EINVAL`. Driven directly on the
/// host so the bounds logic is pinned independently of a particular backend.
#[test]
fn sub_rejects_ranges_outside_the_holder() {
    let mut h = Host::new();
    let root = h.grant_address_space(0, 64 << 10); // a 64 KiB holder

    // A helper to call `sub(off, size_log2)` on `handle` and get the raw result.
    let sub = |h: &mut Host, handle: i32, off: i64, size_log2: i64| -> i64 {
        h.cap_dispatch_slots(cap_id::ADDRESS_SPACE, 4, handle, &[off, size_log2], None)
            .expect("dispatch")[0]
    };

    // A legal 4 KiB child at offset 4 KiB → a fresh handle (>= 0), distinct from the root.
    let child = sub(&mut h, root, 4 << 10, 12);
    assert!(
        child >= 0 && child != root as i64,
        "legal sub should mint a child handle"
    );

    // Too large (128 KiB > the 64 KiB holder), unaligned (offset 1), and spilling past the end
    // (offset 32 KiB + a 64 KiB child) are all rejected.
    assert_eq!(sub(&mut h, root, 0, 17), EINVAL, "child larger than holder");
    assert_eq!(sub(&mut h, root, 1, 12), EINVAL, "unaligned child offset");
    assert_eq!(
        sub(&mut h, root, 32 << 10, 16),
        EINVAL,
        "child spills past holder end"
    );
    assert_eq!(sub(&mut h, root, 0, 64), EINVAL, "out-of-range size_log2");

    // The minted child can attenuate again, but only within *its* 4 KiB — a 64 KiB grandchild fails.
    let child = child as i32;
    assert_eq!(
        sub(&mut h, child, 0, 16),
        EINVAL,
        "grandchild can't exceed its parent"
    );
    assert!(
        sub(&mut h, child, 0, 11) >= 0,
        "a 2 KiB grandchild within the child is fine"
    );
}

/// Audit #1 — a guest must not be able to crash the host by exhausting the 256-slot handle table.
/// `AddressSpace.sub` mints a handle each call; this loops it 300 times (past `CAP = 256`). Once the
/// table fills, `sub` must return **-EMFILE** (-24), and the run must complete normally on **both**
/// backends — in particular the JIT path routes `cap.call` through the `extern "C"` thunk, where a
/// panic (the pre-fix `.expect("handle table full")`) would unwind across the FFI boundary and
/// **abort the host process**. Reaching `block2` with -24 proves the table-full path is a clean
/// errno, not a crash.
#[test]
fn minting_past_table_capacity_returns_emfile_not_panic() {
    const EMFILE: i64 = -24;
    let src = "\
memory 17
func (i32) -> (i64) {
block 0 (v0: i32) {
  v1 = i64.const 300
  v2 = i64.const 0
  br 1(v0, v1, v2)
}
block 1 (v3: i32, v4: i64, v5: i64) {
  v6 = i64.const 0
  v7 = i64.const 12
  v8 = cap.call 5 4 (i64, i64) -> (i32) v3 (v6, v7)
  v9 = i64.extend_i32_s v8
  v10 = i64.const -1
  v11 = i64.add v4 v10
  v12 = i64.const 0
  v13 = i64.gt_s v11 v12
  br_if v13 1(v3, v11, v9) 2(v9)
}
block 2 (v14: i64) {
  return v14
  }
}
";
    let (ival, _imem, jo, _jmem) = both(src, 17, &seed_128k());
    assert_eq!(
        ival,
        Value::I64(EMFILE),
        "interp: an over-capacity mint must return -EMFILE"
    );
    assert!(
        matches!(jo, JitOutcome::Returned(ref s) if s == &[EMFILE]),
        "jit: an over-capacity mint must return -EMFILE (not abort across the cap thunk): {jo:?}"
    );
}
