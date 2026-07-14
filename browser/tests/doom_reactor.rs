//! **Doom boots + renders in the browser reactor** (Doom slice 4) — the end-to-end proof that the
//! playground's run model (the wasm `svm_onramp_open_fs` → [`OnrampReactor::open_with_fs`] path)
//! drives real Doom: `_start` (`doomgeneric_Create`) reads the shareware IWAD through the reactor's
//! `fs` capability, then each `tick` (`doomgeneric_Tick`) renders one 640×400 frame over the
//! persistent window. This is the same module + WAD the page runs, exercised natively over the exact
//! Rust the wasm export wraps — the slice-3c `doom_diff` differential already proved the *pixels* are
//! byte-exact; this proves the *reactor wiring* (fs-served WAD in, per-frame `tick`, `display` out).
//!
//! `#[ignore]`d — it needs the built `doom.svmb` (`demos/doom/build.sh`) and the freely-distributable
//! shareware `doom1.wad`, neither vendored. Paths are overridable via `DOOM_SVMB` / `DOOM_WAD`
//! (defaults match the demo scripts' cache). Run:
//!   sh crates/svm-run/demos/doom/fetch.sh && sh crates/svm-run/demos/doom/build.sh
//!   cargo test -p svm-browser --test doom_reactor -- --ignored --nocapture

use svm_browser::{OnrampReactor, STATUS_OK};

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

#[test]
#[ignore = "needs demos/doom/{fetch,build}.sh artifacts + doom1.wad; slow"]
fn doom_boots_and_renders_in_the_reactor() {
    let svmb = std::fs::read(env_or("DOOM_SVMB", "/tmp/doomgeneric_cache/bc/doom.svmb"))
        .expect("built doom.svmb — run demos/doom/build.sh first");
    let wad = std::fs::read(env_or("DOOM_WAD", "/tmp/doomgeneric_cache/doom1.wad"))
        .expect("shareware doom1.wad");
    assert_eq!(&wad[..4], b"IWAD", "shareware IWAD");

    let m = svm_encode::decode_module(&svmb).expect("decode doom.svmb");
    // Open the reactor with the WAD served behind `fs` (the wasm `svm_onramp_open_fs` path). `_start`
    // runs Doom's whole init here — Z_Init, W_Init (reads the WAD through `fs`), R_Init, the lot.
    let mut r = OnrampReactor::open_with_fs(&m, "doom1.wad".to_string(), wad)
        .expect("Doom's _start (doomgeneric_Create) runs to completion over the fs-served WAD");

    // Drive frames; each `tick` is one `doomgeneric_Tick`. The title screen presents a 640x400 frame
    // (the doomgeneric resolution) through `display`; after a few seconds of ticks Doom auto-plays the
    // demo1 gameplay. `DOOM_FRAMES` (default 8) drives how far — 300 reaches demo playback.
    let frames: usize = env_or("DOOM_FRAMES", "8").parse().unwrap_or(8);
    let mut presented = 0;
    for i in 0..frames {
        if i % 20 == 0 {
            eprintln!("  frame {i}/{frames}…");
        }
        let (status, _stdout) = r.frame();
        assert_eq!(status, STATUS_OK, "tick {i} keeps going (of {frames})");
        if let Some(f) = r.take_frame() {
            assert_eq!((f.width, f.height), (640, 400), "doomgeneric 640x400 frame");
            assert_eq!(f.rgba.len(), 640 * 400 * 4, "RGBA framebuffer");
            assert!(f.rgba.iter().any(|&b| b != 0), "frame {i} is not all-black");
            presented += 1;
        }
    }
    assert!(
        presented > 0,
        "Doom presented at least one frame through display"
    );
    eprintln!(
        "Doom booted over the fs-served WAD and presented {presented}/{frames} frames (640x400)"
    );
}
