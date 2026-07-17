//! Stage 1 (STAGE1.md) — **exec a compiled-C command as a child** and thread its exit status. This
//! is the exec-wiring core: a "shell" parent spawns a *separate*, unmodified `int main(int argc, char
//! **argv)` C program (compiled by chibicc with `--child-entry`) via `Instantiator.instantiate_module`
//! (op 5), delivering `argv` through the §3e args buffer seeded into the child's carve, and `join`s
//! (op 1) for `main`'s return — the value a shell records in `$?`.
//!
//! The command is compiled with `--child-entry`, so its `_start` has the §14 child ABI
//! (`(i64 starter) -> (i64 status)`) rather than the top-level powerbox entry — the one thing that
//! made a compiled program spawnable — while still parsing the args buffer into `main(argc, argv)`.
//! This is a **no-capability** command (pure status, no stdout): stdio-inheriting commands need a
//! module+grant primitive (a documented follow-up). The status is a function of both `argc` and
//! `argv`'s bytes, so it proves real argv delivery, differential interp==JIT.
//!
//! Gated `#![cfg(unix)]` (needs the chibicc toolchain, like `c_frontend.rs`).
#![cfg(unix)]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use svm_interp::{run_capture_reserved_with_host, Host, Trap, Value};
use svm_jit::{compile_and_run_capture_reserved_with_host_ex, JitOutcome};
use svm_text::parse_module;
use svm_verify::verify_module;

const WIN: usize = 256 << 10; // parent window: 256 KiB (holds the 128 KiB carve at 128 KiB)
const CARVE: u64 = 128 << 10; // the command's carve (its declared `memory 17` = 128 KiB)
const ARGS_BASE: u64 = 128; // svm_ir::POWERBOX_ARGS_BASE

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}

fn chibicc() -> &'static Path {
    static CC: OnceLock<PathBuf> = OnceLock::new();
    CC.get_or_init(|| {
        let dir = repo_root().join("frontend/chibicc");
        let status = Command::new("make")
            .arg("-s")
            .current_dir(&dir)
            .status()
            .expect("build chibicc");
        assert!(status.success(), "chibicc build failed");
        dir.join("chibicc")
    })
    .as_path()
}

/// Compile `src` to text IR with `--child-entry` (the spawnable §14 child ABI).
fn child_ir(src: &str) -> String {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static N: AtomicUsize = AtomicUsize::new(0);
    let id = N.fetch_add(1, Ordering::Relaxed);
    let base = std::env::temp_dir().join(format!("svm_exec_{}_{id}", std::process::id()));
    let cfile = base.with_extension("c");
    let irfile = base.with_extension("svm");
    std::fs::write(&cfile, src).unwrap();
    let status = Command::new(chibicc())
        .args([
            "-cc1",
            "--emit-ir",
            "--child-entry",
            "-cc1-input",
            cfile.to_str().unwrap(),
            "-cc1-output",
            irfile.to_str().unwrap(),
            cfile.to_str().unwrap(),
        ])
        .status()
        .expect("run chibicc");
    assert!(status.success(), "chibicc failed on:\n{src}");
    std::fs::read_to_string(&irfile).unwrap()
}

/// The §3e args blob: `{ argc:u32, envc:u32 }` then packed NUL-terminated argv strings (no env).
fn args_blob(argv: &[&[u8]]) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(&(argv.len() as u32).to_le_bytes());
    b.extend_from_slice(&0u32.to_le_bytes());
    for a in argv {
        b.extend_from_slice(a);
        b.push(0);
    }
    b
}

/// A hand-written "shell" parent: seed the args blob into the command's carve at its
/// `POWERBOX_ARGS_BASE`, `instantiate_module` the command (entry 0 = its `_start`, carve at `CARVE`,
/// size_log2 17), `join`, and return the command's status. `(Instantiator, Module)` are the args.
fn parent_src(argv: &[&[u8]]) -> String {
    let blob = args_blob(argv);
    let seed: String = blob
        .iter()
        .enumerate()
        .map(|(i, &b)| {
            let addr = CARVE + ARGS_BASE + i as u64;
            format!("  a{i} = i64.const {addr}\n  b{i} = i32.const {b}\n  i32.store8 a{i} b{i}\n")
        })
        .collect();
    format!(
        "memory 18
func (i32, i32) -> (i64) {{
block0(vinst: i32, vmod: i32):
{seed}  me = i64.extend_i32_s vmod
  ent = i64.const 0
  off = i64.const {CARVE}
  sl = i64.const 17
  q = i64.const 0
  ch = cap.call 6 5 (i64, i64, i64, i64, i64) -> (i32) vinst (me, ent, off, sl, q)
  r = cap.call 6 1 (i32) -> (i64) vinst (ch)
  return r
}}
"
    )
}

/// The command: return `argc + argv[argc-1][0]` — a function of both the count and the last arg's
/// first byte, so the status proves real argv delivery.
const CMD: &str = "int main(int argc, char **argv){ return argc + argv[argc-1][0]; }";

fn expected(argv: &[&[u8]]) -> i64 {
    argv.len() as i64 + argv.last().unwrap()[0] as i64
}

fn run_interp(cmd: &svm_ir::Module, argv: &[&[u8]]) -> Result<Vec<Value>, Trap> {
    let parent = parse_module(&parent_src(argv)).expect("parse parent");
    verify_module(&parent).expect("verify parent");
    let mut host = Host::new();
    let ih = host.grant_instantiator(0, WIN as u64);
    let mh = host.grant_module(cmd);
    let mut fuel = 50_000_000u64;
    let (res, _) = run_capture_reserved_with_host(
        &parent,
        0,
        &[Value::I32(ih), Value::I32(mh)],
        &mut fuel,
        &vec![0u8; WIN],
        0,
        &mut host,
    );
    res
}

fn run_jit(cmd: &svm_ir::Module, argv: &[&[u8]]) -> JitOutcome {
    let parent = parse_module(&parent_src(argv)).expect("parse parent");
    verify_module(&parent).expect("verify parent");
    let mut host = Host::new();
    let ih = host.grant_instantiator(0, WIN as u64);
    let mh = host.grant_module(cmd);
    let (jo, _) = compile_and_run_capture_reserved_with_host_ex(
        &parent,
        0,
        &[ih as i64, mh as i64],
        &vec![0u8; WIN],
        0,
        svm_run::cap_thunk,
        &mut host as *mut Host as *mut core::ffi::c_void,
        Some(svm_run::module_resolver),
        None,
    )
    .expect("jit");
    jo
}

/// The shell spawns the compiled-C command as a child, delivers argv, and joins for `main`'s status —
/// identically on both backends, the status tracking argv (real delivery, not a constant).
#[test]
fn shell_execs_compiled_command_and_collects_status() {
    let cmd = svm_run::resolve_capability_imports(parse_module(&child_ir(CMD)).expect("parse cmd"))
        .expect("resolve");
    verify_module(&cmd).expect("verify cmd");
    for argv in [&[b"cmd".as_slice()][..], &[b"cmd".as_slice(), b"hi"][..]] {
        let want = expected(argv);
        let ir = run_interp(&cmd, argv);
        let jo = run_jit(&cmd, argv);
        assert_eq!(
            ir.expect("interp run ok"),
            vec![Value::I64(want)],
            "interp: exec status = argc + argv[last][0] for {argv:?}"
        );
        assert!(
            matches!(jo, JitOutcome::Returned(ref s) if s == &[want]),
            "jit: exec status must be {want} for {argv:?}, got {jo:?}"
        );
    }
}
