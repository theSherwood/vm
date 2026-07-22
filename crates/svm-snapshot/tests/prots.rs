//! Per-page protections round-trip through the §12.3 window image (Phase-2 slice 1): `Ro` and
//! `Unmapped` pages are carried in the artifact and recovered on restore, while zero `Rw` pages
//! stay elided. The next slice feeds these from / applies them to a running backend's window.

use svm_interp::{Host, StreamRole};
use svm_ir::{Memory, Module};
use svm_snapshot::{freeze, freeze_with_prots, restore_with_prots, FreezeError, PageProt};

const SIZE_LOG2: u8 = 17; // 128 KiB
const WINDOW: usize = 1 << SIZE_LOG2;
const PAGE: usize = 4096;
const NPAGES: usize = WINDOW / PAGE; // 32

// A minimal module that just declares the window: freeze digests its encoded bytes and restore
// checks the geometry against `memory.size_log2`.
const SRC: &str = r#"
func (i32) -> (i64) {
block 0 (v0: i32) {
  v1 = i64.const 7
  return v1
  }
}
"#;

fn module() -> Module {
    let mut m = svm_text::parse_module(SRC).expect("parse");
    m.memory = Some(Memory {
        size_log2: SIZE_LOG2,
    });
    m
}

fn host_with_durable_handles() -> Host {
    let mut h = Host::new();
    h.grant_clock();
    let _ = h.grant_stream(StreamRole::Out);
    h
}

#[test]
fn page_protections_round_trip_through_the_window_image() {
    let m = module();
    let host = host_with_durable_handles();

    let mut window = vec![0u8; WINDOW];
    let mut prots = vec![PageProt::Rw; NPAGES];
    window[0] = 0xAB; // page 0: Rw, non-zero
    window[PAGE - 1] = 0xCD;
    // page 3 left zero `Rw` → elided, restores as zero.
    window[5 * PAGE + 10] = 0x11; // page 5: Ro, non-zero
    prots[5] = PageProt::Ro;
    prots[6] = PageProt::Ro; // page 6: Ro, all-zero → must still come back Ro
    window[9 * PAGE + 2] = 0x99; // page 9: Unmapped — content is NOT stored
    prots[9] = PageProt::Unmapped;

    let art = freeze_with_prots(&m, &window, &prots, &host).expect("freeze");

    let mut rhost = Host::new();
    let (rwin, rprots) = restore_with_prots(&art, &m, &mut rhost).expect("restore");

    // Protections recovered exactly.
    assert_eq!(rprots.len(), NPAGES);
    assert_eq!(rprots[0], PageProt::Rw);
    assert_eq!(rprots[3], PageProt::Rw, "elided page defaults to Rw");
    assert_eq!(rprots[5], PageProt::Ro);
    assert_eq!(rprots[6], PageProt::Ro, "a zero Ro page is still preserved");
    assert_eq!(rprots[9], PageProt::Unmapped);

    // Bytes recovered for Rw/Ro; an Unmapped page restores zero (its content is never stored).
    assert_eq!(rwin[0], 0xAB);
    assert_eq!(rwin[PAGE - 1], 0xCD);
    assert_eq!(rwin[5 * PAGE + 10], 0x11);
    assert!(
        rwin[6 * PAGE..7 * PAGE].iter().all(|&b| b == 0),
        "zero Ro page restores zero"
    );
    assert!(
        rwin[9 * PAGE..10 * PAGE].iter().all(|&b| b == 0),
        "Unmapped page drops its pre-freeze content"
    );

    // §12.6 canonical: re-serializing the restored image at the same safepoint is byte-identical.
    assert_eq!(
        freeze_with_prots(&m, &rwin, &rprots, &host).expect("re-freeze"),
        art,
        "restore → re-serialize reproduces the artifact"
    );
}

#[test]
fn freeze_rejects_a_wrong_length_prot_map() {
    let m = module();
    let host = host_with_durable_handles();
    let window = vec![0u8; WINDOW];
    let prots = vec![PageProt::Rw; NPAGES - 1]; // one short
    assert!(matches!(
        freeze_with_prots(&m, &window, &prots, &host),
        Err(FreezeError::ProtCount { pages: NPAGES, prots: p }) if p == NPAGES - 1
    ));
}

#[test]
fn flat_freeze_equals_an_all_rw_prot_map() {
    // The flat convenience must equal an explicit all-`Rw` map — the back-compat the
    // cross-backend (`durable_jit`) path relies on.
    let m = module();
    let host = host_with_durable_handles();
    let mut window = vec![0u8; WINDOW];
    window[100] = 0x42;
    let all_rw = [PageProt::Rw; NPAGES];
    assert_eq!(
        freeze(&m, &window, &host).expect("flat"),
        freeze_with_prots(&m, &window, &all_rw, &host).expect("explicit"),
    );
}
