//! **Lua's official `files.lua` (io/os) test** on the on-ramp — the unmodified `testes/files.lua`
//! from the Lua 5.4.7 distribution, run through the whole VM with base/`string`/`table`/`math`/
//! `coroutine`/`debug`/**`io`**/**`os`** open. The io library rides the guest stdio layer
//! (`lua_files_stdio.c`: real `FILE` semantics — mode parsing, `ungetc` pushback, EOF/error flags,
//! seek/tell) over the **configurable Fs capability**, and the os library the guest time layer
//! (`lua_files_time.c`: proleptic-Gregorian `gmtime`/`mktime`/`strftime`, UTC). The harness sets the
//! suite's own portability knobs (`_port`/`_soft` — skipping `popen`/`os.execute`/huge-data, exactly
//! as the suite documents), so the file runs byte-for-byte unmodified. A failing assert raises → a
//! clean **exit 0** means every assert held, identical to native Lua (the same harness+file built
//! against real libc in a scratch directory also exits 0 — the differential oracle).
//!
//! **No filesystem authority is ambient.** The fixed powerbox is untouched; each test injects the
//! backend it wants at the capability boundary (`Instance::run_with_caps`):
//! - [`svm_run::fs::mem_fs`] — deterministic in-memory fs, asserted on **all three engines**;
//! - [`svm_run::fs::host_fs`] — the real filesystem attenuated to a fresh temp root, proving the
//!   same unmodified guest drives real disk I/O end to end (and leaves the root clean afterwards —
//!   files.lua removes everything it creates).

use svm_run::{fs, Backend, Limits, Outcome, RunConfig, Value};

fn instance() -> svm_run::Instance {
    let bc = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/lua/lua_files.bc"
    );
    let t = svm_llvm::translate_bc_path(bc).expect("translate Lua files.lua bitcode");
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
        // files.lua's first assert: `type(os.getenv"PATH") == "string"` (via the synthesized
        // env-blob getenv), so seed a PATH like any hosted process would see.
        env: vec![b"PATH=/usr/bin".to_vec()],
    }
}

/// `main` returns 0 iff every assert in `files.lua` held (a failure raises → the harness returns a
/// nonzero code and prints `files.lua: <error>` to stdout, surfaced here).
fn check_mem_fs(backend: Backend) {
    let out = instance()
        .run_with_caps(backend, &config(), &[("fs", fs::mem_fs())])
        .expect("run Lua files.lua through the powerbox");
    assert_eq!(
        out.outcome,
        Outcome::Returned(vec![Value::I32(0)]),
        "{backend:?}: files.lua failed\nstdout:\n{}",
        String::from_utf8_lossy(&out.stdout),
    );
}

#[test]
fn lua_files_mem_fs_tree_walker() {
    check_mem_fs(Backend::TreeWalk);
}

#[test]
fn lua_files_mem_fs_bytecode() {
    check_mem_fs(Backend::Bytecode);
}

#[test]
fn lua_files_mem_fs_jit() {
    check_mem_fs(Backend::Jit);
}

/// The same unmodified guest against the **real** filesystem (attenuated to a fresh temp root):
/// files.lua's tmpfiles really hit the disk, and because the file removes everything it creates,
/// the root must be empty again afterwards — asserted host-side.
#[test]
fn lua_files_host_fs() {
    let root = std::env::temp_dir().join(format!("svm-lua-files-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).expect("create temp root");

    let out = instance()
        .run_with_caps(
            Backend::Bytecode,
            &config(),
            &[("fs", fs::host_fs(root.clone()))],
        )
        .expect("run Lua files.lua against host_fs");
    assert_eq!(
        out.outcome,
        Outcome::Returned(vec![Value::I32(0)]),
        "files.lua failed against host_fs\nstdout:\n{}",
        String::from_utf8_lossy(&out.stdout),
    );

    let leftovers: Vec<_> = std::fs::read_dir(&root)
        .expect("read temp root")
        .map(|e| e.expect("dir entry").file_name())
        .collect();
    assert!(
        leftovers.is_empty(),
        "files.lua removes everything it creates; leftovers: {leftovers:?}"
    );
    std::fs::remove_dir_all(&root).expect("cleanup");
}
