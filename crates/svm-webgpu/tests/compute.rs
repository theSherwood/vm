//! End-to-end WebGPU compute demo: a guest C program (translated through the LLVM on-ramp) uploads
//! a buffer, runs a WGSL compute shader on the host GPU (lavapipe in CI), reads it back, and checks
//! the result against a CPU reference **in-guest** — on all three SVM engines. The capability is
//! SVM-only (no native symbol), so the assertion is the guest's own self-check (`ALL MATCH cpu`),
//! like the async/JIT capability demos.

use std::process::Command;

fn build_guest_bc() -> Option<std::path::PathBuf> {
    let demo = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../svm-run/demos/webgpu/wgpu_demo.c");
    let bc = std::env::temp_dir().join(format!("svm_webgpu_demo_{}.ll", std::process::id()));
    let ok = Command::new("clang")
        .args([
            "-O2",
            "-emit-llvm",
            "-S",
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
    ok.then_some(bc)
}

#[test]
fn demo_webgpu_compute() {
    if !svm_webgpu::adapter_available() {
        eprintln!("note: skipping webgpu compute (no wgpu adapter — install mesa-vulkan-drivers / lavapipe)");
        return;
    }
    let Some(bc) = build_guest_bc() else {
        eprintln!("note: skipping webgpu compute (clang unavailable)");
        return;
    };

    let t = svm_llvm::translate_ll_path(&bc).expect("translate webgpu demo bitcode");
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
        let out = inst
            .run_with_caps(backend, &config(), &[("webgpu", svm_webgpu::webgpu_cap())])
            .unwrap_or_else(|e| panic!("webgpu demo run ({backend:?}): {e}"));
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(
            stdout.contains("compute1 (mul-add, 1 buffer): OK")
                && stdout.contains("compute2 (a*a+i, 2 buffers): OK")
                && stdout.contains("webgpu compute: ALL MATCH cpu"),
            "webgpu demo ({backend:?}) self-check failed; stdout:\n{stdout}"
        );
    }
}
