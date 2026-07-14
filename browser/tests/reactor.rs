//! The **reactor** run model + `keyboard` capability (Doom slice 2) — the interactive/graphical path.
//! Unlike single-shot `onramp_exec`, an [`OnrampReactor`] instantiates once and calls the guest's
//! exported `tick` per frame, persisting globals/BSS between frames, presenting a frame through
//! `display` and draining input through `keyboard`. The browser drives it in a requestAnimationFrame
//! loop; the wasm `svm_onramp_{open,frame,key,close}` exports wrap these same methods.
//!
//! Two fixtures, both `clang -O2 -emit-llvm` + `svm-llvm-translate --host-page 65536`:
//! - `fixtures/bounce.svmb` (`display/bounce.c`) — the box's motion is a pure function of its initial
//!   state + injected key events, asserted to the pixel (animation, input steering, state persistence).
//! - `fixtures/life.svmb` (`display/life.c`) — Conway's Game of Life on a **malloc heap above the
//!   mapped window**; the glider only advances if the reactor persists the *whole* guest memory
//!   (heap included) between frames — the Doom heap-persistence proof (slice 3).

use svm_browser::{Frame, OnrampReactor, STATUS_OK};

const W: u32 = 160;
const H: u32 = 120;
const BOX: u32 = 8;
// JS keyCodes the guest steers on.
const LEFT: i32 = 37;
const RIGHT: i32 = 39;

fn open() -> OnrampReactor {
    let bytes = include_bytes!("fixtures/bounce.svmb");
    let m = svm_encode::decode_module(bytes).expect("decode bounce.svmb");
    OnrampReactor::open(&m).expect("open the bounce reactor")
}

/// Run one frame and return the presented frame (panicking if the guest presented none or errored).
fn step(r: &mut OnrampReactor) -> Frame {
    let (status, _stdout) = r.frame();
    assert_eq!(status, STATUS_OK, "tick should keep going");
    r.take_frame().expect("tick presented a frame")
}

/// Top-left corner of the amber box (its min x / min y over the bright pixels) and the bright-pixel
/// count — the box is the only `(255, 220, 40)` region, so this recovers its exact position.
fn box_pos(f: &Frame) -> (u32, u32, u32) {
    let (mut minx, mut miny, mut count) = (u32::MAX, u32::MAX, 0u32);
    for y in 0..f.height {
        for x in 0..f.width {
            let i = ((y * f.width + x) * 4) as usize;
            if f.rgba[i] == 255 && f.rgba[i + 1] == 220 && f.rgba[i + 2] == 40 {
                minx = minx.min(x);
                miny = miny.min(y);
                count += 1;
            }
        }
    }
    (minx, miny, count)
}

#[test]
fn reactor_presents_and_animates() {
    let mut r = open();

    // Frame 1: the box starts centered (bx=(160-8)/2=76, by=(120-8)/2=56) and moves down-right by
    // SPEED=2 before the first present → (78, 58). Full 8×8 box = 64 amber pixels.
    let f1 = step(&mut r);
    assert_eq!((f1.width, f1.height), (W, H));
    assert_eq!(box_pos(&f1), (78, 58, BOX * BOX), "box after frame 1");
    // A corner is the dark-blue background (the box is near the center).
    assert_eq!(&f1.rgba[0..4], &[16, 16, 40, 255], "corner is background");

    // Frame 2 (no input): the box advances to (80, 60) — the frame changed, i.e. it animates.
    let f2 = step(&mut r);
    assert_eq!(box_pos(&f2), (80, 60, BOX * BOX), "box after frame 2");
}

#[test]
fn reactor_responds_to_input() {
    let mut r = open();
    // Two frames moving down-right → box at (80, 60), heading +x.
    step(&mut r);
    let before = box_pos(&step(&mut r));
    assert_eq!(before.0, 80, "moving right before input");

    // Press Left: the guest polls it next frame and flips vx to -SPEED, so x now *decreases*.
    r.push_key(LEFT, 1);
    let after = box_pos(&step(&mut r));
    assert_eq!(after.0, 78, "box reversed left after the Left key");
    assert!(after.0 < before.0, "input steered the box left");

    // Press Right: back to +x.
    r.push_key(RIGHT, 1);
    let again = box_pos(&step(&mut r));
    assert_eq!(again.0, 80, "box reversed right after the Right key");
}

// ---- heap-persistence (Doom slice 3): Conway's Game of Life over a malloc heap -------------------

fn open_life() -> OnrampReactor {
    let bytes = include_bytes!("fixtures/life.svmb");
    let m = svm_encode::decode_module(bytes).expect("decode life.svmb");
    OnrampReactor::open(&m).expect("open the life reactor")
}

/// Live-cell count and the top-left of their bounding box (live = amber `(255,200,40)`), recovering
/// the glider's position from the 1-pixel-per-cell frame.
fn glider(f: &Frame) -> (u32, u32, u32) {
    let (mut count, mut minx, mut miny) = (0u32, u32::MAX, u32::MAX);
    for y in 0..f.height {
        for x in 0..f.width {
            let i = ((y * f.width + x) * 4) as usize;
            if f.rgba[i] == 255 && f.rgba[i + 1] == 200 && f.rgba[i + 2] == 40 {
                count += 1;
                minx = minx.min(x);
                miny = miny.min(y);
            }
        }
    }
    (count, minx, miny)
}

#[test]
fn reactor_persists_the_malloc_heap_across_frames() {
    let mut r = open_life();
    // The grids live in the malloc heap, above the mapped window. If that heap did NOT persist, every
    // `tick` would re-run from the seed → every frame would be an identical generation 1. It persists,
    // so the glider evolves: exactly 5 live cells throughout, translating (+1, +1) every 4 generations
    // (the classic glider on a torus) — a fully deterministic sequence, the differential anchor.
    let g1 = glider(&step(&mut r)); // generation 1
    assert_eq!(g1, (5, 2, 3), "glider at generation 1");
    for _ in 0..3 {
        assert_eq!(
            glider(&step(&mut r)).0,
            5,
            "a glider is always 5 live cells"
        );
    }
    let g5 = glider(&step(&mut r)); // generation 5 = one full glider period after g1
    assert_eq!(
        g5,
        (5, 3, 4),
        "glider translated (+1,+1) after 4 generations — the heap persisted"
    );
    // Had the heap reset each frame, g5 would equal g1 (stuck at generation 1). It advanced.
    assert_ne!(
        g5, g1,
        "the grid evolved between frames (persistence, not a frozen seed)"
    );

    for _ in 0..4 {
        assert_eq!(glider(&step(&mut r)).0, 5);
    }
    let g9 = glider(&step(&mut r)); // generation 9 = two periods after g1
    assert_eq!(
        g9,
        (5, 4, 5),
        "glider translated another (+1,+1) — deterministic across periods"
    );
}
