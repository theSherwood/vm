//! §13 SharedRegion — end-to-end through the real `cap.call` dispatch (iface 4): a host-granted
//! shared region mapped into the window at *two* offsets aliases the same bytes, so a store at one
//! offset is visible at a load from the other — including the **magic ring buffer** (adjacent
//! mappings + a single access *straddling the seam*, which wraps tail→head as one contiguous
//! access; tested differentially below). The basic-alias interp↔JIT differential lives in
//! `jit_diff.rs` (`jit_cap_shared_region_aliases_differential`, all platforms). The guest queries
//! the region's `page_size` (op 3) and works in whole granules, so it is host-agnostic (4 KiB /
//! 16 KiB on unix, the 64 KiB allocation granularity on Windows — what `MapViewOfFile3` requires).

use svm_interp::{run_capture_reserved_with_host, Host, Value};
use svm_text::parse_module;
use svm_verify::verify_module;

const MARKER: i64 = 0x0123_4567_89ab_cdef;

/// Map the region at window offset 0 and again at window offset `page_size`, store `MARKER` at 0,
/// then return the i64 loaded from `page_size` — which must read back `MARKER` *iff* the two
/// mappings alias the same backing.
fn alias_probe_src() -> String {
    format!(
        // `memory 17` (128 KiB) so two whole granules fit at offsets 0 and `page_size` — on Windows
        // that granule is the 64 KiB allocation granularity, so the second alias lands at 64 KiB.
        "memory 17\n\
         func (i32) -> (i64) {{\n\
         block 0 (v0: i32) {{\n\
         \x20 v1 = cap.call 4 3 () -> (i64) v0 ()\n\
         \x20 v2 = i64.const 0\n\
         \x20 v3 = i32.const 3\n\
         \x20 v4 = cap.call 4 0 (i64, i64, i64, i32) -> (i64) v0 (v2, v2, v1, v3)\n\
         \x20 v5 = cap.call 4 0 (i64, i64, i64, i32) -> (i64) v0 (v1, v2, v1, v3)\n\
         \x20 v6 = i64.const {MARKER}\n\
         \x20 i64.store v2 v6\n\
         \x20 v7 = i64.load v1\n\
         \x20 return v7\n\
           }}\n\
         }}\n"
    )
}

#[test]
fn shared_region_two_offsets_alias_through_cap_call() {
    let src = alias_probe_src();
    let m = parse_module(&src).expect("parse");
    verify_module(&m).expect("verify");

    let mut host = Host::new();
    // A region comfortably larger than any host page (covers 16 KiB pages too).
    let h = host.grant_shared_region(1 << 16);
    let mut fuel = 1_000_000u64;
    let (res, _snap) =
        run_capture_reserved_with_host(&m, 0, &[Value::I32(h)], &mut fuel, &[], 0, &mut host);

    assert_eq!(
        res.expect("run ok"),
        vec![Value::I64(MARKER)],
        "the second mapping must alias the first: a store at offset 0 must be visible at page_size"
    );
}

#[test]
fn shared_region_without_second_mapping_is_not_aliased() {
    // Control: map the region only at offset 0, store MARKER there, and read at page_size — which is
    // an ordinary (unmapped/zero) window page, *not* aliased. Proves the positive test is non-vacuous.
    let src = format!(
        "memory 17\n\
         func (i32) -> (i64) {{\n\
         block 0 (v0: i32) {{\n\
         \x20 v1 = cap.call 4 3 () -> (i64) v0 ()\n\
         \x20 v2 = i64.const 0\n\
         \x20 v3 = i32.const 3\n\
         \x20 v4 = cap.call 4 0 (i64, i64, i64, i32) -> (i64) v0 (v2, v2, v1, v3)\n\
         \x20 v6 = i64.const {MARKER}\n\
         \x20 i64.store v2 v6\n\
         \x20 v7 = i64.load v1\n\
         \x20 return v7\n\
           }}\n\
         }}\n"
    );
    let m = parse_module(&src).expect("parse");
    verify_module(&m).expect("verify");

    let mut host = Host::new();
    let h = host.grant_shared_region(1 << 16);
    let mut fuel = 1_000_000u64;
    let (res, _snap) =
        run_capture_reserved_with_host(&m, 0, &[Value::I32(h)], &mut fuel, &[], 0, &mut host);

    assert_eq!(
        res.expect("run ok"),
        vec![Value::I64(0)],
        "an un-aliased window page must read zero, not the marker"
    );
}

/// The **magic ring buffer** (§13's headline use of multi-offset aliasing), differentially on both
/// backends: map the region at two *adjacent* window offsets `[0, g)` and `[g, 2g)` (both aliasing
/// region `[0, g)`), then issue a single 8-byte store at `g - 4` — **straddling the seam**. Its low
/// 4 bytes land at the region's *tail* (via mapping 1) and its high 4 bytes wrap to the region's
/// *head* (via mapping 2): a wrap-around write as one contiguous access, the whole point of the
/// layout. The guest then recombines the marker from the *other* mapping's views — an i32 load at
/// window 0 (the head, via mapping 1) and at `2g - 4` (the tail, via mapping 2) — and adds a
/// round-trip check of the straddling load itself; the result equals the marker **iff** the wrap is
/// exact. Decisive differentially: the JIT does the straddling access as one raw hardware store
/// across the alias boundary, while the interpreter's software model resolves it per byte through
/// the page map — a model that resolved a whole access by its first byte's page would diverge here.
#[test]
fn ring_buffer_straddling_access_wraps_differential() {
    use svm_jit::{compile_and_run_capture_reserved_with_host, JitOutcome};

    const RING_MARKER: i64 = 0x1122_3344_5566_7788;
    let src = format!(
        "memory 17\n\
         func (i32) -> (i64) {{\n\
         block 0 (v0: i32) {{\n\
         \x20 v1 = cap.call 4 3 () -> (i64) v0 ()\n\
         \x20 v2 = i64.const 0\n\
         \x20 v3 = i32.const 3\n\
         \x20 v4 = cap.call 4 0 (i64, i64, i64, i32) -> (i64) v0 (v2, v2, v1, v3)\n\
         \x20 v5 = cap.call 4 0 (i64, i64, i64, i32) -> (i64) v0 (v1, v2, v1, v3)\n\
         \x20 v6 = i64.const 4\n\
         \x20 v7 = i64.sub v1 v6\n\
         \x20 v8 = i64.const {RING_MARKER}\n\
         \x20 i64.store v7 v8\n\
         \x20 v9 = i64.load v7\n\
         \x20 v10 = i32.load v2\n\
         \x20 v11 = i64.const 2\n\
         \x20 v12 = i64.mul v1 v11\n\
         \x20 v13 = i64.sub v12 v6\n\
         \x20 v14 = i32.load v13\n\
         \x20 v15 = i64.extend_i32_u v10\n\
         \x20 v16 = i64.const 4294967296\n\
         \x20 v17 = i64.mul v15 v16\n\
         \x20 v18 = i64.extend_i32_u v14\n\
         \x20 v19 = i64.add v17 v18\n\
         \x20 v20 = i64.sub v9 v8\n\
         \x20 v21 = i64.add v19 v20\n\
         \x20 return v21\n\
           }}\n\
         }}\n"
    );
    let m = parse_module(&src).expect("parse");
    verify_module(&m).expect("verify");

    // Region ≥ the largest granule (64 KiB on Windows); each mapping is one granule.
    let region_len = 1usize << 16;
    let mut hi = Host::new();
    let mut hj = Host::new();
    let ri = hi.grant_shared_region(region_len); // interp: pure-Rust VecBacking (per-byte model)
    let rj = hj.grant_shared_region_backed(svm_run::new_shared_region(region_len)); // JIT: OS memory
    assert_eq!(ri, rj, "grants are deterministic");

    let init = vec![0u8; 1 << 17];
    let mut fuel = 1_000_000u64;
    let (interp, imem) =
        run_capture_reserved_with_host(&m, 0, &[Value::I32(ri)], &mut fuel, &init, 0, &mut hi);
    let (jit, jmem) = compile_and_run_capture_reserved_with_host(
        &m,
        0,
        &[rj as i64],
        &init,
        0,
        svm_run::cap_thunk,
        &mut hj as *mut Host as *mut core::ffi::c_void,
    )
    .expect("jit compiles");

    // head (0x11223344) << 32 | tail (0x55667788) recombines the marker, +0 from the round-trip
    // check — exactly the marker iff the straddling store wrapped into the region correctly.
    let ival = interp.expect("interp ran ok").pop().expect("one result");
    assert_eq!(
        ival,
        Value::I64(RING_MARKER),
        "interp: the straddling store must wrap tail→head through the ring mappings"
    );
    assert!(
        matches!(jit, JitOutcome::Returned(ref s) if s == &[RING_MARKER]),
        "jit: {jit:?}"
    );
    assert_eq!(
        imem, jmem,
        "interp/JIT windows diverge across the ring seam"
    );
}
