//! Phase-2 page protections, end to end on the interpreter:
//!
//! * **capture** — the interpreter reports real per-page protections, so a D40 `readonly` data
//!   segment is captured as `Ro` and survives the §12 codec (byte image *and* protection), where
//!   Phase-1's flat all-`Rw` image would have lost it;
//! * **re-establish** — restoring that protection map and seeding a thawed run with it makes a
//!   write to a restored `Ro` page fault exactly as the frozen guest would (vs. succeeding when
//!   the page comes back `Rw`) — on **both** the interpreter and the JIT (real `mprotect` /
//!   `VirtualProtect`), so the two backends agree on a thawed protected window.

use svm_interp::{run_capture_reserved_with_host_prots, CapturedProt, Host, Value};
use svm_ir::Memory;
use svm_snapshot::{freeze_with_prots, restore_with_prots, PageProt, PAGE};

const SIZE_LOG2: u8 = 17;
const WINDOW: usize = 1 << SIZE_LOG2;
const RO_OFF: usize = 5 * PAGE; // a read-only data segment lands on page 5

// A read-only data segment + a trivial entry (it doesn't touch memory; the segment alone marks
// its page `Ro` at instantiation, D40).
const SRC: &str = r#"
data ro 20480 "ABCD"
func (i32) -> (i64) {
block0(v0: i32):
  v1 = i64.const 7
  return v1
}
"#;

/// Map the interpreter's captured protections to the codec's, refusing a §13 shared-region page
/// (D-region: a durable freeze must reject those — there are none here).
fn to_codec_prots(caps: &[CapturedProt]) -> Vec<PageProt> {
    caps.iter()
        .map(|c| match c {
            CapturedProt::Rw => PageProt::Rw,
            CapturedProt::Ro => PageProt::Ro,
            CapturedProt::Unmapped => PageProt::Unmapped,
            CapturedProt::Backed => {
                panic!("freeze must refuse a §13 shared-region page (D-region)")
            }
        })
        .collect()
}

#[test]
fn readonly_data_segment_is_captured_and_survives_the_codec() {
    assert_eq!(RO_OFF, 20480);
    let mut m = svm_text::parse_module(SRC).expect("parse");
    m.memory = Some(Memory {
        size_log2: SIZE_LOG2,
    });

    let mut host = Host::new();
    let _ = host.grant_clock(); // a durable handle so the freeze has a non-empty table

    let init = vec![0u8; WINDOW];
    let mut fuel = 100_000u64;
    let (r, window, caps) = run_capture_reserved_with_host_prots(
        &m,
        0,
        &[Value::I32(0)],
        &mut fuel,
        &init,
        None, // capture only
        SIZE_LOG2,
        &mut host,
    );
    assert_eq!(r, Ok(vec![Value::I64(7)]));

    // Capture: the readonly segment's page is `Ro`; an ordinary committed page is `Rw`.
    let ro_page = RO_OFF / PAGE;
    assert_eq!(
        caps[ro_page],
        CapturedProt::Ro,
        "readonly segment page captured Ro"
    );
    assert_eq!(
        caps[0],
        CapturedProt::Rw,
        "an ordinary committed page is Rw"
    );
    assert_eq!(
        &window[RO_OFF..RO_OFF + 4],
        b"ABCD",
        "segment bytes landed in the window"
    );

    // Through the §12 codec: the protection is recorded and recovered (Phase-1 would have lost it).
    let art = freeze_with_prots(&m, &window, &to_codec_prots(&caps), &host).expect("freeze");
    let mut rhost = Host::new();
    let (rwin, rprots) = restore_with_prots(&art, &m, &mut rhost).expect("restore");
    assert_eq!(
        rprots[ro_page],
        PageProt::Ro,
        "Ro survives serialize/restore"
    );
    assert_eq!(rprots[0], PageProt::Rw);
    assert_eq!(&rwin[RO_OFF..RO_OFF + 4], b"ABCD", "Ro page bytes survive");
}

/// Map the codec's protections back to the interpreter's, for seeding a thawed run.
fn to_captured(prots: &[PageProt]) -> Vec<CapturedProt> {
    prots
        .iter()
        .map(|p| match p {
            PageProt::Rw => CapturedProt::Rw,
            PageProt::Ro => CapturedProt::Ro,
            PageProt::Unmapped => CapturedProt::Unmapped,
        })
        .collect()
}

// Stores to page 5 (`RO_OFF`), then returns 0. With that page restored `Ro` the store faults;
// with it `Rw` the store succeeds.
const STORE_SRC: &str = r#"
func (i32) -> (i64) {
block0(v0: i32):
  v1 = i64.const 20480
  v2 = i64.const 99
  i64.store v1 v2
  v3 = i64.const 0
  return v3
}
"#;

#[test]
fn restore_re_establishes_ro_so_a_thawed_write_faults() {
    let mut m = svm_text::parse_module(STORE_SRC).expect("parse");
    m.memory = Some(Memory {
        size_log2: SIZE_LOG2,
    });
    let mut host = Host::new();
    let _ = host.grant_clock();

    // Freeze a window whose page 5 is read-only, then restore it.
    let window = vec![0u8; WINDOW];
    let mut prots = vec![PageProt::Rw; WINDOW / PAGE];
    prots[RO_OFF / PAGE] = PageProt::Ro;
    let art = freeze_with_prots(&m, &window, &prots, &host).expect("freeze");
    let mut rhost = Host::new();
    let (rwin, rprots) = restore_with_prots(&art, &m, &mut rhost).expect("restore");

    // Thaw with the restored protections: the store into the Ro page faults.
    let mut fuel = 100_000u64;
    let (faulted, _, _) = run_capture_reserved_with_host_prots(
        &m,
        0,
        &[Value::I32(0)],
        &mut fuel,
        &rwin,
        Some(&to_captured(&rprots)),
        SIZE_LOG2,
        &mut rhost,
    );
    assert!(faulted.is_err(), "a store to a restored Ro page must fault");

    // Control: the same window/guest with no protections re-established — the store succeeds.
    let mut fuel = 100_000u64;
    let mut host2 = Host::new();
    let (ok, _, _) = run_capture_reserved_with_host_prots(
        &m,
        0,
        &[Value::I32(0)],
        &mut fuel,
        &rwin,
        None,
        SIZE_LOG2,
        &mut host2,
    );
    assert_eq!(
        ok,
        Ok(vec![Value::I64(0)]),
        "without the Ro protection the store succeeds"
    );
}

#[test]
fn jit_re_establishes_ro_so_a_thawed_write_faults() {
    use core::ffi::c_void;
    use svm_jit::{
        compile_and_run_capture_reserved_with_host_prots, JitOutcome, TrapKind, WindowProt,
    };

    let mut m = svm_text::parse_module(STORE_SRC).expect("parse");
    m.memory = Some(Memory {
        size_log2: SIZE_LOG2,
    });
    let mut host = Host::new();
    let _ = host.grant_clock();

    // Freeze a window whose page 5 is read-only, then restore it.
    let window = vec![0u8; WINDOW];
    let mut prots = vec![PageProt::Rw; WINDOW / PAGE];
    prots[RO_OFF / PAGE] = PageProt::Ro;
    let art = freeze_with_prots(&m, &window, &prots, &host).expect("freeze");
    let mut rhost = Host::new();
    let (rwin, rprots) = restore_with_prots(&art, &m, &mut rhost).expect("restore");
    let jit_prots: Vec<WindowProt> = rprots
        .iter()
        .map(|p| match p {
            PageProt::Rw => WindowProt::Rw,
            PageProt::Ro => WindowProt::Ro,
            PageProt::Unmapped => WindowProt::Unmapped,
        })
        .collect();

    // Thaw on the JIT with the restored protections: the store into the Ro page faults (real
    // mprotect/VirtualProtect → guard → MemoryFault), exactly as on the interpreter above.
    let mut h = Host::new();
    let (faulted, _) = compile_and_run_capture_reserved_with_host_prots(
        &m,
        0,
        &[0i64],
        &rwin,
        &jit_prots,
        SIZE_LOG2,
        svm_run::cap_thunk,
        &mut h as *mut Host as *mut c_void,
    )
    .expect("jit compiles");
    assert!(
        matches!(faulted, JitOutcome::Trapped(TrapKind::MemoryFault)),
        "a JIT store to a restored Ro page must fault, got {faulted:?}"
    );

    // Control: no protections re-established — the same store succeeds on the JIT.
    let mut h2 = Host::new();
    let (ok, _) = compile_and_run_capture_reserved_with_host_prots(
        &m,
        0,
        &[0i64],
        &rwin,
        &[], // all Rw
        SIZE_LOG2,
        svm_run::cap_thunk,
        &mut h2 as *mut Host as *mut c_void,
    )
    .expect("jit compiles");
    assert!(
        matches!(ok, JitOutcome::Returned(_)),
        "without the Ro protection the JIT store succeeds, got {ok:?}"
    );
}

#[test]
fn jit_capture_matches_interp_for_a_readonly_segment() {
    use core::ffi::c_void;
    use svm_jit::compile_and_run_capture_reserved_with_host_prots;

    let mut m = svm_text::parse_module(SRC).expect("parse"); // `data ro 20480 "ABCD"`
    m.memory = Some(Memory {
        size_log2: SIZE_LOG2,
    });
    let npages = WINDOW / PAGE;
    let init = vec![0u8; WINDOW];

    // Interpreter capture (reads its software page map).
    let mut hi = Host::new();
    let _ = hi.grant_clock();
    let mut fuel = 100_000u64;
    let (_, _, caps_i) = run_capture_reserved_with_host_prots(
        &m,
        0,
        &[Value::I32(0)],
        &mut fuel,
        &init,
        None,
        SIZE_LOG2,
        &mut hi,
    );

    // JIT capture: run, then reconstruct the protections from the host (data segments + cap map).
    let mut hj = Host::new();
    let _ = hj.grant_clock();
    let _ = compile_and_run_capture_reserved_with_host_prots(
        &m,
        0,
        &[0i64],
        &init,
        &[],
        SIZE_LOG2,
        svm_run::cap_thunk,
        &mut hj as *mut Host as *mut c_void,
    )
    .expect("jit compiles");
    let caps_j = hj.capture_window_prots(&m.data, WINDOW as u64, npages);

    assert_eq!(caps_i[RO_OFF / PAGE], CapturedProt::Ro);
    assert_eq!(
        caps_j[RO_OFF / PAGE],
        CapturedProt::Ro,
        "JIT capture reports the readonly segment as Ro"
    );
    assert_eq!(
        caps_i, caps_j,
        "interp and JIT capture the same protection map"
    );
}

#[test]
fn jit_capture_overlays_runtime_protect_over_the_default() {
    // The runtime page-state map (`cap_pages`, populated by Memory-cap map/unmap/protect)
    // overrides the default — exercised directly here (page 0 → Unmapped) so the merge is
    // covered without a Memory-cap-using guest. Page 0 maps to host page 0 on any host page size.
    let mut h = Host::new();
    let map = h.cap_window_pages(0);
    map.lock().unwrap().insert(0, 3); // code 3 = Unmapped
    let caps = h.capture_window_prots(&[], WINDOW as u64, WINDOW / PAGE);
    assert_eq!(
        caps[0],
        CapturedProt::Unmapped,
        "a runtime cap_pages entry overrides the page default"
    );
    assert_eq!(
        caps[WINDOW / PAGE - 1],
        CapturedProt::Rw,
        "untouched pages stay Rw"
    );
}
