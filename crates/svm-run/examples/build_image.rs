//! **Build a demo data image.** Walk a host directory (e.g. a natively-`initdb`'d, cleanly-shut-down
//! Postgres cluster) into a single self-contained `SVMFSIM1` blob — the shippable filesystem image a
//! demo mounts on the `fs` cap with no host filesystem (`svm_run::fs::mem_fs_from_archive`). This is
//! the data half of the browser demo's artifacts (the code half is the resolved `.svmb`; see
//! `BOOTSPEED.md`).
//!
//!   cargo run --release -p svm-run --example build_image -- <dir> <out.img>

use std::time::Instant;

fn main() {
    let mut args = std::env::args().skip(1);
    let (Some(dir), Some(out)) = (args.next(), args.next()) else {
        eprintln!("usage: build_image <dir> <out.img>");
        std::process::exit(2);
    };
    let t = Instant::now();
    let (files, dirs) = svm_run::fs::read_host_dir(std::path::Path::new(&dir)).expect("walk dir");
    let img = svm_run::fs::encode_image(&files, &dirs);
    std::fs::write(&out, &img).expect("write image");
    let bytes: usize = files.iter().map(|(_, d)| d.len()).sum();
    println!(
        "{out}: {} files ({} MiB), {} dirs -> {} MiB image in {:.1?}",
        files.len(),
        bytes / (1 << 20),
        dirs.len(),
        img.len() / (1 << 20),
        t.elapsed(),
    );
    // Self-check: the image must round-trip (fail-closed before shipping).
    let _ = svm_run::fs::decode_image(&img).expect("image round-trips");
}
