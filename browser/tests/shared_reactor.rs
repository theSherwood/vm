//! **Shared-window reactor** (BROWSER.md § "wasm-JIT tier", slice 5b). [`SharedOnrampReactor`] runs
//! the same on-ramp reactor model as [`OnrampReactor`], but over a guest window relocated into a
//! caller-provided region of this module's own linear memory (a `Region::shared`) instead of a window
//! the engine backs internally — the substrate the emitted `tick` (slice 5c) needs so the interpreter
//! and the JS-compiled wasm read/write one set of bytes. With no JIT eligibility set it interprets
//! every frame, so it must be a **byte-identical** substitute for `OnrampReactor`: this differential
//! steps both reactors frame-for-frame over the bounce (animation + input) and life (malloc-heap
//! persistence) fixtures and asserts the presented frames and stdout match exactly.

use svm_browser::{Frame, OnrampReactor, SharedOnrampReactor, STATUS_OK};

// A 32 MiB owned window — comfortably covers the bounce/life mapped window plus their grown heap.
const WIN_LOG2: u8 = 25;

// JS keyCodes the bounce guest steers on.
const LEFT: i32 = 37;
const RIGHT: i32 = 39;

fn decode(bytes: &[u8]) -> svm_ir::Module {
    svm_encode::decode_module(bytes).expect("decode fixture")
}

/// Assert two frames are byte-identical (dimensions + every RGBA byte).
fn assert_frames_eq(a: &Frame, b: &Frame, ctx: &str) {
    assert_eq!(
        (a.width, a.height),
        (b.width, b.height),
        "frame dims ({ctx})"
    );
    assert_eq!(a.rgba, b.rgba, "frame RGBA bytes differ ({ctx})");
}

/// Step both reactors once and assert the frame + status + stdout match exactly.
fn step_both(internal: &mut OnrampReactor, shared: &mut SharedOnrampReactor, ctx: &str) -> Frame {
    let (si, out_i) = internal.frame();
    let (ss, out_s) = shared.frame();
    assert_eq!(si, ss, "frame status ({ctx})");
    assert_eq!(si, STATUS_OK, "tick should keep going ({ctx})");
    assert_eq!(out_i, out_s, "stdout delta ({ctx})");
    let fi = internal.take_frame().expect("internal presented a frame");
    let fs = shared.take_frame().expect("shared presented a frame");
    assert_frames_eq(&fi, &fs, ctx);
    fs
}

#[test]
fn bounce_frames_match_internal_reactor() {
    let m = decode(include_bytes!("fixtures/bounce.svmb"));
    let mut internal = OnrampReactor::open(&m).expect("open internal bounce reactor");
    let mut shared =
        SharedOnrampReactor::open_owned(&m, WIN_LOG2).expect("open shared bounce reactor");

    // A dozen free-running frames: the animation must match pixel-for-pixel.
    for i in 0..12 {
        step_both(
            &mut internal,
            &mut shared,
            &format!("bounce free frame {i}"),
        );
    }

    // Now steer with identical input on both reactors — the input-driven divergence must also match.
    for (frame_i, (keycode, pressed)) in [(LEFT, 1), (RIGHT, 1), (LEFT, 1), (LEFT, 0)]
        .into_iter()
        .enumerate()
    {
        internal.push_key(keycode, pressed);
        shared.push_key(keycode, pressed);
        step_both(
            &mut internal,
            &mut shared,
            &format!("bounce input frame {frame_i}"),
        );
    }
}

#[test]
fn life_heap_persistence_matches_internal_reactor() {
    // Life keeps its grids on a malloc heap *above* the mapped window; a frame only matches across
    // reactors if the shared window persists the whole guest memory (heap included) between frames.
    let m = decode(include_bytes!("fixtures/life.svmb"));
    let mut internal = OnrampReactor::open(&m).expect("open internal life reactor");
    let mut shared =
        SharedOnrampReactor::open_owned(&m, WIN_LOG2).expect("open shared life reactor");

    let mut last = None;
    for gen in 0..12 {
        let f = step_both(
            &mut internal,
            &mut shared,
            &format!("life generation {gen}"),
        );
        last = Some(f);
    }
    // Non-vacuity: the final frame is not blank (the glider actually ran on both windows).
    let f = last.unwrap();
    assert!(
        f.rgba.iter().any(|&b| b != 0),
        "life presented a non-blank frame (the heap-backed grid advanced)"
    );
}
