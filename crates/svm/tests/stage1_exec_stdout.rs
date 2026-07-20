//! Stage 1 (STAGE1.md) — **exec a compiled-C command with inherited stdout**: the full external
//! `echo`. A "shell" parent spawns a *separate*, unmodified `int main(int argc, char **argv)` C
//! program (chibicc `--child-entry`) via the new `Instantiator.instantiate_module_named` (op 13) —
//! the union of `instantiate_module` (op 5, run a foreign `Module`) and `instantiate_named` (op 11,
//! re-grant caps by name). The parent re-grants its own `stdout` into the child under the name
//! `"stdout"`; the command's `_start` resolves it by name and `write(1, …)` lands in the shell's sink.
//!
//! This closes the gap the earlier `stage1_exec_command.rs` documented: a compiled command can now do
//! real I/O, not just return a status. The output tracks argv (real delivery), differential
//! interp==JIT — op 13 lowered on both backends (interp dispatch + JIT `instantiate_module_named`
//! thunk), with the JIT given **both** the module resolver and the named-grant hooks.
//!
//! Gated `#![cfg(unix)]` (needs the chibicc toolchain, like `c_frontend.rs`).
#![cfg(unix)]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use svm_interp::{run_capture_reserved_with_host, Host, StreamRole, Trap, Value};
use svm_jit::{compile_and_run_capture_reserved_with_host_ex, GrantChildHooks, JitOutcome};
use svm_text::parse_module;
use svm_verify::verify_module;

const WIN: usize = 256 << 10;
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
            .unwrap();
        assert!(status.success(), "chibicc build failed");
        dir.join("chibicc")
    })
    .as_path()
}

/// Compile `src` with `--child-entry` (spawnable §14 child ABI).
fn child_ir(src: &str) -> String {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static N: AtomicUsize = AtomicUsize::new(0);
    let id = N.fetch_add(1, Ordering::Relaxed);
    let base = std::env::temp_dir().join(format!("svm_execout_{}_{id}", std::process::id()));
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
        .unwrap();
    assert!(status.success(), "chibicc failed on:\n{src}");
    std::fs::read_to_string(&irfile).unwrap()
}

/// A real command: echo every `argv[i]` on its own line (ambient `write(1, …)`), return `argc`.
const CMD: &str = r#"
long write(long fd, void *buf, long n);
static long slen(char *s){ long n=0; while(s[n]) n++; return n; }
int main(int argc, char **argv){
  for (int i = 0; i < argc; i++){ write(1, argv[i], slen(argv[i])); write(1, "\n", 1); }
  return argc;
}
"#;

/// The §3e args blob: `{ argc:u32, envc:u32 }` then packed NUL-terminated argv (no env).
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

/// The "shell" parent `(Instantiator, Module, stdout)`: lay a grant record naming `"stdout"`, seed the
/// args blob into the command's carge at `POWERBOX_ARGS_BASE`, `instantiate_module_named` (op 13) the
/// command re-granting `stdout` by name, `join`, and return the command's status.
///
/// `wide` selects the declared width of op 13's child-handle result (and the matching `join` arg): the
/// canonical contract shape is i32, but a chibicc-style frontend widens every scalar to an i64 slot, so
/// both backends must accept an i64-declared `cap.call 6 13`/`6 1` and coerce. `wide == true` exercises
/// that path with hand-IR (no chibicc toolchain needed), guarding the JIT's `slot_i32`/`result_as`
/// coercions against the interpreter's slot-width tolerance.
fn parent_src(argv: &[&[u8]], wide: bool) -> String {
    // Grant record at window 0: {name_off=100, name_len=6, handle=stdout, flags=0}; "stdout" at 100.
    let (chty, jarg) = if wide { ("i64", "i64") } else { ("i32", "i32") };
    let blob = args_blob(argv);
    let seed: String = blob
        .iter()
        .enumerate()
        .map(|(i, &b)| {
            let addr = CARVE + ARGS_BASE + i as u64;
            format!("  s{i} = i64.const {addr}\n  z{i} = i32.const {b}\n  i32.store8 s{i} z{i}\n")
        })
        .collect();
    format!(
        r#"memory 18
func (i32, i32, i32) -> (i64) {{
block0(vinst: i32, vmod: i32, vout: i32):
  r0 = i64.const 0
  n100 = i32.const 100
  i32.store r0 n100
  r4 = i64.const 4
  n6 = i32.const 6
  i32.store r4 n6
  r8 = i64.const 8
  i32.store r8 vout
  r12 = i64.const 12
  zf = i32.const 0
  i32.store r12 zf
  cS = i32.const 115
  cT = i32.const 116
  cD = i32.const 100
  cO = i32.const 111
  cU = i32.const 117
  q100 = i64.const 100
  i32.store8 q100 cS
  q101 = i64.const 101
  i32.store8 q101 cT
  q102 = i64.const 102
  i32.store8 q102 cD
  q103 = i64.const 103
  i32.store8 q103 cO
  q104 = i64.const 104
  i32.store8 q104 cU
  q105 = i64.const 105
  i32.store8 q105 cT
{seed}  me = i64.extend_i32_s vmod
  gp = i64.const 0
  gn = i64.const 1
  ent = i64.const 0
  off = i64.const {CARVE}
  sl = i64.const 17
  qz = i64.const 0
  ch = cap.call 6 13 (i64, i64, i64, i64, i64, i64, i64) -> ({chty}) vinst (me, gp, gn, ent, off, sl, qz)
  r = cap.call 6 1 ({jarg}) -> (i64) vinst (ch)
  return r
}}
"#
    )
}

fn grant_hooks() -> GrantChildHooks {
    GrantChildHooks {
        build: svm_run::grant_child_build,
        build_named: svm_run::grant_named_child_build,
        bind_imports: svm_run::child_bind_imports,
        release: svm_run::grant_child_release,
    }
}

fn run_interp(
    cmd: &svm_ir::Module,
    argv: &[&[u8]],
    wide: bool,
) -> (Result<Vec<Value>, Trap>, Vec<u8>) {
    let parent = parse_module(&parent_src(argv, wide)).expect("parse parent");
    verify_module(&parent).expect("verify parent");
    let mut host = Host::new();
    let ih = host.grant_instantiator(0, WIN as u64);
    let mh = host.grant_module(cmd);
    let oh = host.grant_stream(StreamRole::Out);
    let mut fuel = 50_000_000u64;
    let (res, _) = run_capture_reserved_with_host(
        &parent,
        0,
        &[Value::I32(ih), Value::I32(mh), Value::I32(oh)],
        &mut fuel,
        &vec![0u8; WIN],
        0,
        &mut host,
    );
    (res, host.stdout_bytes())
}

fn run_jit(cmd: &svm_ir::Module, argv: &[&[u8]], wide: bool) -> (JitOutcome, Vec<u8>) {
    let parent = parse_module(&parent_src(argv, wide)).expect("parse parent");
    verify_module(&parent).expect("verify parent");
    let mut host = Host::new();
    let ih = host.grant_instantiator(0, WIN as u64);
    let mh = host.grant_module(cmd);
    let oh = host.grant_stream(StreamRole::Out);
    let (jo, _) = compile_and_run_capture_reserved_with_host_ex(
        &parent,
        0,
        &[ih as i64, mh as i64, oh as i64],
        &vec![0u8; WIN],
        0,
        svm_run::cap_thunk,
        &mut host as *mut Host as *mut core::ffi::c_void,
        Some(svm_run::module_resolver),
        Some(grant_hooks()),
    )
    .expect("jit");
    (jo, host.stdout_bytes())
}

/// The shell execs the compiled command with inherited stdout: it echoes its argv to the shell's
/// sink and returns `argc`, identically on both backends. Output tracks argv (real delivery through
/// a re-granted capability) — the full external `echo`.
#[test]
fn shell_execs_command_with_inherited_stdout() {
    // Phase 3: keep the manifest — the op-13 spawn binds the child's slots.
    let cmd = parse_module(&child_ir(CMD)).expect("parse cmd");
    verify_module(&cmd).expect("verify cmd");
    // Each argv runs with both the canonical i32-declared op-13/join shape and the i64-widened shape a
    // chibicc frontend emits — both must agree interp==JIT.
    for argv in [
        &[b"echo".as_slice(), b"hi"][..],
        &[b"prog".as_slice(), b"a", b"bb"][..],
    ] {
        for wide in [false, true] {
            let (ir, iout) = run_interp(&cmd, argv, wide);
            let (jo, jout) = run_jit(&cmd, argv, wide);
            let expect: Vec<u8> = argv
                .iter()
                .flat_map(|a| [a, b"\n".as_slice()].concat())
                .collect();
            assert_eq!(
                ir.expect("interp run ok"),
                vec![Value::I64(argv.len() as i64)],
                "interp: exec status = argc for {argv:?} (wide={wide})"
            );
            assert_eq!(
                iout, expect,
                "interp: command echoed argv to inherited stdout (wide={wide})"
            );
            assert!(
                matches!(jo, JitOutcome::Returned(ref s) if s == &[argv.len() as i64]),
                "jit: exec status = argc for {argv:?} (wide={wide}), got {jo:?}"
            );
            assert_eq!(
                jout, iout,
                "jit: inherited-stdout bytes must match interp (wide={wide})"
            );
        }
    }
}
