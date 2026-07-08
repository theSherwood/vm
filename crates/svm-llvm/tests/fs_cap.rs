//! The **configurable Fs capability** end to end: a C guest (`fixtures/fs_probe.c`) resolves the
//! embedder-granted capability by name (`__vm_cap_resolve` → §7 `cap.self.resolve`) and drives the
//! whole op protocol through `__vm_host_call` (§7 host-defined capability — the wasm-import
//! analogue): open/write/close, reopen/seek/read-back, rename, append, remove, truncate, sync,
//! EOF, read-only refusal, and the attenuation refusals (`..`/absolute paths). No filesystem authority exists
//! unless the test injects one via [`svm_run::Instance::run_with_caps`] — the fixed powerbox is
//! untouched.
//!
//! Two interchangeable backends behind the same protocol (dependency injection at the capability
//! boundary):
//! - [`svm_run::fs::mem_fs`] — deterministic in-memory fs, on **all three engines**;
//! - [`svm_run::fs::host_fs`] — the **real** filesystem attenuated to a temp root: the test
//!   pre-seeds `seed.txt`, the guest verifies it and leaves `out.txt` behind, and the test asserts
//!   the bytes really landed on disk.

use svm_run::{fs, Backend, Limits, Outcome, RunConfig, Value};

fn instance() -> svm_run::Instance {
    let bc = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/fs_probe.bc");
    let t = svm_llvm::translate_bc_path(bc).expect("translate fs probe");
    svm_run::instantiate(t.module).expect("instantiate")
}

fn config() -> RunConfig {
    RunConfig {
        limits: Limits {
            fuel: None,
            deadline: None,
            max_fibers: 0,
            max_vcpus: 0,
        },
        stdin: vec![],
        memory_size_log2: None,
        args: vec![],
        env: vec![],
    }
}

/// The probe exits 0 iff every step behaved; a failure exits with the step number (surfaced here).
fn check_mem_fs(backend: Backend) {
    let out = instance()
        .run_with_caps(backend, &config(), &[("fs", fs::mem_fs())])
        .expect("run fs probe");
    assert_eq!(
        out.outcome,
        Outcome::Returned(vec![Value::I32(0)]),
        "{backend:?}: fs probe failed (exit = failing step)\nstdout:\n{}",
        String::from_utf8_lossy(&out.stdout),
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "fs probe ok\n");
}

#[test]
fn fs_probe_mem_fs_tree_walker() {
    check_mem_fs(Backend::TreeWalk);
}

#[test]
fn fs_probe_mem_fs_bytecode() {
    check_mem_fs(Backend::Bytecode);
}

#[test]
fn fs_probe_mem_fs_jit() {
    check_mem_fs(Backend::Jit);
}

/// The same guest against the **real** filesystem, attenuated to a fresh temp root. Beyond the
/// guest's own exit-0, the *host side* is asserted: the pre-seeded `seed.txt` was readable by the
/// guest (phase B ran), the guest's `out.txt` really landed on disk with the right bytes, and the
/// probe's scratch files (`hello.txt`/`world.txt`) were genuinely created-then-removed.
#[test]
fn fs_probe_host_fs_all_backends() {
    for backend in [Backend::TreeWalk, Backend::Bytecode, Backend::Jit] {
        let root =
            std::env::temp_dir().join(format!("svm-fs-probe-{}-{backend:?}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).expect("create temp root");
        std::fs::write(root.join("seed.txt"), "SEED").expect("seed");

        let out = instance()
            .run_with_caps(backend, &config(), &[("fs", fs::host_fs(root.clone()))])
            .expect("run fs probe");
        assert_eq!(
            out.outcome,
            Outcome::Returned(vec![Value::I32(0)]),
            "{backend:?}: fs probe failed against host_fs (exit = failing step)\nstdout:\n{}",
            String::from_utf8_lossy(&out.stdout),
        );

        // The guest's write really reached the disk...
        let out_txt = std::fs::read_to_string(root.join("out.txt")).expect("out.txt on disk");
        assert_eq!(
            out_txt, "GUEST",
            "{backend:?}: guest bytes must land on disk"
        );
        // ...and the removed scratch files are really gone.
        assert!(!root.join("hello.txt").exists() && !root.join("world.txt").exists());

        std::fs::remove_dir_all(&root).expect("cleanup");
    }
}
