//! WebGPU **compute → image → PNG** demo (LLVM.md WebGPU demo 3). A guest C program (translated
//! through the LLVM on-ramp) runs a WGSL Mandelbrot compute shader on the host GPU (lavapipe in CI),
//! reads the RGBA image back, self-validates it (exact top/bottom mirror symmetry — a
//! float-implementation-independent invariant — plus an in-set/escaped sanity pair), and writes the
//! raw pixels out through the granted `fs` capability. The test drives it on all three SVM engines,
//! asserts the guest's self-check, and encodes the pixels to a PNG (exercising the readback → image
//! path). Exercises the `webgpu` and `fs` capabilities together. Skips cleanly with no GPU adapter.

use std::process::Command;

const W: u32 = 320;
const H: u32 = 240;

#[test]
fn demo_webgpu_mandelbrot() {
    if !svm_webgpu::adapter_available() {
        eprintln!(
            "note: skipping webgpu mandelbrot (no wgpu adapter — install mesa-vulkan-drivers)"
        );
        return;
    }
    let demo = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../svm-run/demos/webgpu/mandelbrot.c");
    let bc = std::env::temp_dir().join(format!("svm_webgpu_mandel_{}.bc", std::process::id()));
    let built = Command::new("clang")
        .args([
            "-O2",
            "-emit-llvm",
            "-c",
            "-DSVM_GUEST",
            "-fno-vectorize",
            "-fno-slp-vectorize",
        ])
        .arg(&demo)
        .arg("-o")
        .arg(&bc)
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if !built {
        eprintln!("note: skipping webgpu mandelbrot (clang unavailable)");
        return;
    }

    let t = svm_llvm::translate_bc_path(&bc).expect("translate mandelbrot bitcode");
    let inst = svm_run::instantiate(t.module).expect("instantiate");
    let config = || svm_run::RunConfig {
        limits: svm_run::Limits {
            fuel: None,
            deadline: None,
            max_fibers: 0,
            max_vcpus: 0,
        },
        stdin: vec![],
        memory_size_log2: None,
        args: vec![],
        env: vec![],
    };

    for backend in [
        svm_run::Backend::TreeWalk,
        svm_run::Backend::Bytecode,
        svm_run::Backend::Jit,
    ] {
        let root = std::env::temp_dir().join(format!(
            "svm_webgpu_mandel_{}_{:?}",
            std::process::id(),
            backend
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).expect("out dir");
        let out = inst
            .run_with_caps(
                backend,
                &config(),
                &[
                    ("webgpu", svm_webgpu::webgpu_cap()),
                    ("fs", svm_run::fs::host_fs(root.clone())),
                ],
            )
            .unwrap_or_else(|e| panic!("mandelbrot run ({backend:?}): {e}"));
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(
            stdout.contains("symmetric=1 center_in=1 corner_out=1")
                && stdout.contains("webgpu mandelbrot: ALL OK"),
            "mandelbrot ({backend:?}) self-check failed; stdout:\n{stdout}"
        );

        // The guest wrote the raw RGBA through the fs cap; encode a PNG to prove the readback → image
        // path (and that the file landed on disk through the capability).
        let rgba = std::fs::read(root.join("mandel.rgba")).expect("guest-written mandel.rgba");
        assert_eq!(rgba.len(), (W * H * 4) as usize, "rgba size");
        let png = root.join("mandelbrot.png");
        image::save_buffer(&png, &rgba, W, H, image::ExtendedColorType::Rgba8).expect("encode png");
        assert!(std::fs::metadata(&png)
            .map(|m| m.len() > 0)
            .unwrap_or(false));
        let _ = std::fs::remove_dir_all(&root);
    }
}
