//! The **`display` capability** — the framebuffer output waist (the path Doom rides). An on-ramp
//! guest presents an RGBA frame via `__vm_cap_resolve("display")` + `__vm_host_call(h, 0, ptr, w, h)`,
//! and [`onramp_exec`] captures it into `PbOutcome::framebuffer` for the browser to blit to a
//! `<canvas>`. The wasm `svm_run_onramp` export exposes the same bytes via `svm_framebuffer_*`.
//!
//! The fixture `fixtures/gradient.svmb` is `crates/svm-run/demos/display/gradient.c` compiled with
//! stock `clang -O2 -emit-llvm` and translated (`--host-page 65536`, the wasm page — the same asset
//! the playground fetches). Its image is a pure function of `(x, y)`, so the captured bytes are
//! deterministic and a native `cc` build produces the identical frame (the differential anchor).

use svm_browser::{onramp_exec, STATUS_OK};

const W: u32 = 128;
const H: u32 = 128;

/// Expected pixel per the guest: R ramps left→right, G ramps top→bottom, B is `((x^y)&63)*4`, A=255.
fn expect(x: u32, y: u32) -> [u8; 4] {
    [
        (x * 255 / (W - 1)) as u8,
        (y * 255 / (H - 1)) as u8,
        (((x ^ y) & 63) * 4) as u8,
        255,
    ]
}

fn pixel(rgba: &[u8], x: u32, y: u32) -> [u8; 4] {
    let i = ((y * W + x) * 4) as usize;
    [rgba[i], rgba[i + 1], rgba[i + 2], rgba[i + 3]]
}

#[test]
fn gradient_presents_a_captured_frame() {
    let bytes = include_bytes!("fixtures/gradient.svmb");
    let m = svm_encode::decode_module(bytes).expect("decode gradient.svmb");
    let out = onramp_exec(&m, b"");
    assert_eq!(out.status, STATUS_OK, "the guest should run cleanly");

    let fb = out
        .framebuffer
        .expect("the guest presented a frame via `display`");
    assert_eq!((fb.width, fb.height), (W, H), "dimensions round-trip");
    assert_eq!(
        fb.rgba.len(),
        (W * H * 4) as usize,
        "rgba is exactly width*height*4 bytes",
    );

    // Every pixel matches the analytic image — full-frame, not just corners, since it's cheap and
    // this is the security-sensitive read-out-of-guest-memory path.
    for y in 0..H {
        for x in 0..W {
            assert_eq!(
                pixel(&fb.rgba, x, y),
                expect(x, y),
                "pixel ({x},{y}) must match the guest's analytic gradient",
            );
        }
    }
}
