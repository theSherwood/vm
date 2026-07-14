//! The reactor's **`fs` capability** (Doom slice 4) — the WAD read path. Unlike a single-shot
//! on-ramp guest (which reads its input from stdin), a graphical reactor guest reads a served file:
//! Doom's `_start` (`doomgeneric_Create`) opens its IWAD through `fs` during init. [`OnrampReactor::
//! open_with_fs`] serves one read-only in-memory file behind that cap, using the same open/read/seek
//! op protocol as the native `doom_diff` differential's WAD server.
//!
//! Fixture `fixtures/fsread.svmb` (`display/fsread.c`, `clang -O2` + `svm-llvm-translate --host-page
//! 65536`): `_start` opens "data.bin", seeks to END for its size, seeks back, reads the bytes; each
//! `tick` renders those bytes as a 16×16 grayscale frame (pixel i's R=G=B is byte i, or 0 past the
//! end). So a served blob round-trips through the `fs` cap into guest memory and back out as pixels —
//! the differential anchor for the reactor's file-serving plumbing.

use svm_browser::{Frame, OnrampReactor, STATUS_OK};

const W: u32 = 16;
const H: u32 = 16;

/// Open the fsread reactor with `blob` served as "data.bin" through the `fs` capability.
fn open_with(blob: Vec<u8>) -> OnrampReactor {
    let bytes = include_bytes!("fixtures/fsread.svmb");
    let m = svm_encode::decode_module(bytes).expect("decode fsread.svmb");
    OnrampReactor::open_with_fs(&m, "data.bin".to_string(), blob).expect("open the fsread reactor")
}

/// Run one frame and return the presented frame (panicking if the guest presented none or errored).
fn step(r: &mut OnrampReactor) -> Frame {
    let (status, _stdout) = r.frame();
    assert_eq!(status, STATUS_OK, "tick should keep going");
    r.take_frame().expect("tick presented a frame")
}

/// The red channel of pixel `i` in a 16×16 RGBA frame.
fn red(f: &Frame, i: usize) -> u8 {
    f.rgba[i * 4]
}

#[test]
fn reactor_reads_a_served_file_through_fs() {
    // A known blob: byte i = i (0..64). The guest reads it via `fs` at init and renders each byte as a
    // grayscale pixel, so every read byte reappears as that pixel's red channel — proving open + seek
    // (to size) + read delivered the file into the guest's window across the cap boundary.
    let blob: Vec<u8> = (0..64u32).map(|i| i as u8).collect();
    let mut r = open_with(blob.clone());
    let f = step(&mut r);
    assert_eq!((f.width, f.height), (W, H), "16×16 frame");
    for (i, &b) in blob.iter().enumerate() {
        assert_eq!(
            red(&f, i),
            b,
            "byte {i} of the served file round-tripped to pixel {i}"
        );
    }
    // Past the file's end the guest renders 0 (its `len` bounds the read) — so nothing beyond the
    // served bytes leaks in.
    for i in blob.len()..(W * H) as usize {
        assert_eq!(red(&f, i), 0, "pixel {i} is past the file end → 0");
    }
}

#[test]
fn reactor_frame_persists_across_ticks() {
    // The file is read once at `_start`; the reactor keeps the guest instance alive, so a later frame
    // renders the same bytes (state persistence — the property the per-frame Doom loop relies on).
    let blob: Vec<u8> = (0..32u32).map(|i| (i * 7) as u8).collect();
    let mut r = open_with(blob.clone());
    let f1 = step(&mut r);
    let f5 = {
        for _ in 0..3 {
            step(&mut r);
        }
        step(&mut r)
    };
    assert_eq!(
        f1.rgba, f5.rgba,
        "the served file renders identically frame to frame"
    );
    assert_eq!(red(&f1, 5), blob[5], "byte 5 = 35 present in frame 1");
}

#[test]
fn reactor_without_fs_open_still_works() {
    // A blob that never matches: the guest asks for "data.bin", we serve "other.bin", so `open`
    // returns ENOENT and the guest reads nothing (len stays 0). The reactor still runs — an absent
    // file is not a trap — and every pixel is 0.
    let bytes = include_bytes!("fixtures/fsread.svmb");
    let m = svm_encode::decode_module(bytes).expect("decode fsread.svmb");
    let mut r = OnrampReactor::open_with_fs(&m, "other.bin".to_string(), vec![1, 2, 3])
        .expect("open the fsread reactor");
    let f = step(&mut r);
    for i in 0..(W * H) as usize {
        assert_eq!(red(&f, i), 0, "no file matched → pixel {i} is 0");
    }
}
