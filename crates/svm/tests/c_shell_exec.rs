//! Stage 1 (STAGE1.md) — **op 13 wired into a chibicc shell**: a compiled-C shell (not hand-written
//! IR) drives `instantiate_module_named` (op 13) to `exec` a *separate*, unmodified compiled-C command
//! with inherited stdout, and collects its status. The shell parses its own `argv` (the powerbox args
//! buffer), looks the command up, seeds the command's `argv` into a carve, re-grants `stdout` by name,
//! spawns via op 13, and `join`s — the whole external-command path emitted by the frontend.
//!
//! Both the shell and the command are ordinary C. Capability wiring: `stdout` is a re-grantable
//! `Stream` (shared sink, so the command's output and any shell output unify); `exec_stdout`/
//! `exec_lookup` are a tiny host fn (the embedder's PATH → `Module` map); `__spawn`/`__join` bind to
//! `cap.call 6 13`/`6 1` with the `Instantiator` baked in (`Resolved::CapBound`). Differential
//! interp==JIT — the JIT is given the module resolver *and* the named-grant hooks op 13 needs.
//!
//! This is the frontend-drives-exec proof. Folding it into the full `c_shell.rs` builtin dispatch (its
//! personality-heap-at-`win/2` layout vs. a 128 KiB command carve) is the follow-up.
//!
//! Gated `#![cfg(unix)]` (needs the chibicc toolchain).
#![cfg(unix)]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use core::ffi::c_void;
use svm_interp::{run_capture_reserved_with_host, GuestMem, Host, StreamRole, Trap};
use svm_ir::{Resolved, ResolvedCap};
use svm_jit::{compile_and_run_capture_reserved_with_host_ex, GrantChildHooks, JitOutcome};
use svm_text::parse_module as parse_module_raw;
use svm_verify::verify_module;

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

/// Compile `src` to text IR; `child_entry` selects the §14 spawnable entry ABI.
fn c_to_ir(src: &str, child_entry: bool) -> String {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static N: AtomicUsize = AtomicUsize::new(0);
    let id = N.fetch_add(1, Ordering::Relaxed);
    let base = std::env::temp_dir().join(format!("svm_cshx_{}_{id}", std::process::id()));
    let cfile = base.with_extension("c");
    let irfile = base.with_extension("svm");
    std::fs::write(&cfile, src).unwrap();
    let mut args = vec!["-cc1", "--emit-ir"];
    if child_entry {
        args.push("--child-entry");
    }
    let cin = cfile.to_str().unwrap().to_string();
    let cout = irfile.to_str().unwrap().to_string();
    args.extend(["-cc1-input", &cin, "-cc1-output", &cout, &cin]);
    let status = Command::new(chibicc()).args(&args).status().unwrap();
    assert!(status.success(), "chibicc failed on:\n{src}");
    std::fs::read_to_string(&irfile).unwrap()
}

/// The command: echo every `argv[i]` on its own line (ambient `write(1, …)`), return `argc`.
const CMD: &str = r#"
long write(long fd, void *buf, long n);
static long slen(char *s){ long n=0; while(s[n]) n++; return n; }
int main(int argc, char **argv){
  for (int i = 0; i < argc; i++){ write(1, argv[i], slen(argv[i])); write(1, "\n", 1); }
  return argc;
}
"#;

/// The shell: `main(argc, argv)` — exec `argv[1]` as an external command, passing it `argv[1..]`.
/// `pool` (a big writable global) both forces a window large enough for the command's carve and holds
/// the grant record + the aligned carve. Handle wiring is via imports (see the resolver).
const SHELL: &str = r#"
/* Natural C prototypes: the child handle is a plain `long`, as a shell author would write it. chibicc
 * widens every scalar to an i64 slot, so `cap.call 6 13` is declared `(i64…) -> (i64)` even though the
 * Instantiator contract's canonical child handle is i32. Both backends reconcile that width: the interp
 * reads args as i64 slots and coerces the result to the declared type; the JIT's `lower_instantiator`
 * does the matching `slot_i64`/`slot_i32`/`result_as` coercions. (Before that fix the JIT CapFaulted on
 * the i64 result — no compiled-C program could drive the Instantiator on the JIT.) The handle operand
 * (`inst`) must stay `int`: `Resolved::CapBound` requires an i32 const placeholder there. */
long __spawn(int inst, long module, long gp, long gn, long entry, long off, long sl, long q);
long __join(int inst, long child);
long exec_stdout(int h);
long exec_lookup(int h, char *name, long len);
long stream_write(int h, void *buf, long n);
static long slen(char *s){ long n=0; while(s[n]) n++; return n; }
/* 384 KiB: room for the grant record/name low, a 128 KiB-aligned 128 KiB carve, all below the SP. */
static char pool[393216];
int main(int argc, char **argv){
  long out = exec_stdout(0);
  if (argc < 2) return 1;
  long mod = exec_lookup(0, argv[1], slen(argv[1]));
  if (mod < 0){ stream_write(out, "not found\n", 10); return 127; }
  long base = (long)pool;
  long carve = (base + 131071) & ~131071;
  /* grant record at base: {name_off, name_len, out, flags} ; "stdout" name follows at base+16 */
  int *rec = (int *)base;
  rec[0] = (int)(base + 16);
  rec[1] = 6;
  rec[2] = (int)out;
  rec[3] = 0;
  char *nm = (char *)(base + 16);
  nm[0]='s'; nm[1]='t'; nm[2]='d'; nm[3]='o'; nm[4]='u'; nm[5]='t';
  /* the command's args buffer at carve+128: {argc-1, envc=0} then packed argv[1..] */
  char *ab = (char *)(carve + 128);
  int *hdr = (int *)ab;
  hdr[0] = argc - 1;
  hdr[1] = 0;
  char *p = ab + 8;
  for (int i = 1; i < argc; i++){ char *s = argv[i]; long L = slen(s); for (long k=0;k<L;k++) *p++ = s[k]; *p++ = 0; }
  long child = __spawn(0, mod, base, 1, 0, carve, 17, 0);
  return __join(0, child);
}
"#;

/// The embedder's PATH → `Module` map + stdout handle, as one host fn (op 0 = stdout handle, op 1 =
/// look a command name up). Returns handle values valid in the shell's own cap table.
fn exec_host(out_h: i32, echo_h: i32) -> svm_interp::HostFn {
    Box::new(
        move |op: u32, args: &[i64], mem: Option<&mut dyn GuestMem>| match op {
            0 => Ok(vec![out_h as i64]),
            1 => {
                let mem = mem.ok_or(Trap::Malformed)?;
                let ptr = *args.first().ok_or(Trap::Malformed)? as u64;
                let len = (*args.get(1).ok_or(Trap::Malformed)?).max(0) as u64;
                let name = mem.read_bytes(ptr, len).ok_or(Trap::Malformed)?;
                Ok(vec![if name == b"echo" { echo_h as i64 } else { -1 }])
            }
            _ => Err(Trap::CapFault),
        },
    )
}

/// Bind the shell's imports: `stdout` handle via the call operand (`Cap`), the rest with a baked
/// handle (`CapBound`).
fn resolver(inst_h: i32, exec_h: i32) -> impl Fn(&str) -> Option<Resolved> {
    move |name| match name {
        "stream_write" => Some(Resolved::Cap(ResolvedCap { type_id: 0, op: 1 })),
        "__spawn" => Some(Resolved::CapBound {
            type_id: 6,
            op: 13,
            handle: inst_h,
        }),
        "__join" => Some(Resolved::CapBound {
            type_id: 6,
            op: 1,
            handle: inst_h,
        }),
        "exec_stdout" => Some(Resolved::CapBound {
            type_id: 13,
            op: 0,
            handle: exec_h,
        }),
        "exec_lookup" => Some(Resolved::CapBound {
            type_id: 13,
            op: 1,
            handle: exec_h,
        }),
        _ => None,
    }
}

/// The §3e args blob for `argv` (the shell's own args): `{argc, envc}` + packed NUL-terminated strings.
fn args_blob(argv: &[&str]) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(&(argv.len() as u32).to_le_bytes());
    b.extend_from_slice(&0u32.to_le_bytes());
    for a in argv {
        b.extend_from_slice(a.as_bytes());
        b.push(0);
    }
    b
}

fn grant_hooks() -> GrantChildHooks {
    GrantChildHooks {
        build: svm_run::grant_child_build,
        build_named: svm_run::grant_named_child_build,
        bind_imports: svm_run::child_bind_imports,
        release: svm_run::grant_child_release,
    }
}

/// Run the shell with powerbox `argv`; return (status, stdout).
fn run(shell: &svm_ir::Module, cmd: &svm_ir::Module, argv: &[&str], jit: bool) -> (i64, Vec<u8>) {
    let win = 1usize << shell.memory.expect("shell window").size_log2;
    let mut host = Host::new();
    let _sink = host.shared_stdout(); // route the stdout Stream + re-granted child streams to one sink
    let out_h = host.grant_stream(StreamRole::Out);
    let inst_h = host.grant_instantiator(0, win as u64);
    let echo_h = host.grant_module(cmd);
    let exec_h = host.grant_host_fn(exec_host(out_h, echo_h));
    // Resolve the shell's imports against this run's handles.
    let m = svm_ir::resolve_imports_with(shell, resolver(inst_h, exec_h)).expect("resolve");
    verify_module(&m).expect("verify shell");
    // Seed the shell's own args buffer at POWERBOX_ARGS_BASE.
    let mut init = vec![0u8; win];
    let blob = args_blob(argv);
    init[svm_ir::POWERBOX_ARGS_BASE as usize..svm_ir::POWERBOX_ARGS_BASE as usize + blob.len()]
        .copy_from_slice(&blob);

    if jit {
        let (jo, _) = compile_and_run_capture_reserved_with_host_ex(
            &m,
            0,
            &[],
            &init,
            0,
            svm_run::cap_thunk,
            &mut host as *mut Host as *mut c_void,
            Some(svm_run::module_resolver),
            Some(grant_hooks()),
        )
        .expect("jit");
        let code = match jo {
            JitOutcome::Returned(ref s) => s.first().copied().unwrap_or(0),
            JitOutcome::Exited(c) => c as i64,
            ref o => panic!("jit ended abnormally: {o:?}"),
        };
        (code, host.stdout_bytes())
    } else {
        let mut fuel = 200_000_000u64;
        let res = run_capture_reserved_with_host(&m, 0, &[], &mut fuel, &init, 0, &mut host);
        let code = match res.0 {
            Ok(ref v) => match v.first() {
                Some(svm_interp::Value::I32(x)) => *x as i64,
                Some(svm_interp::Value::I64(x)) => *x,
                _ => 0,
            },
            Err(Trap::Exit(c)) => c as i64,
            Err(e) => panic!("interp trapped: {e:?}"),
        };
        (code, host.stdout_bytes())
    }
}

/// The compiled shell execs the compiled `echo` command with inherited stdout, identically on both
/// backends: the command's argv reaches its stdout (the shell's sink) and its `argc` is the shell's
/// exit status. This is op 13 driven end to end by the frontend.
#[test]
fn compiled_shell_execs_command_via_op13() {
    // Parse the shell raw — its imports (`__spawn`/`exec_*`/`stream_write`) are resolved per-run by
    // `resolver` against that run's handles, so the names must survive parsing.
    let shell = parse_module_raw(&c_to_ir(SHELL, false)).expect("parse shell");
    // Phase 3: keep the manifest — the op-13 spawn binds the child's slots.
    let cmd = parse_module_raw(&c_to_ir(CMD, true)).expect("parse cmd");
    verify_module(&cmd).expect("verify cmd");

    for argv in [&["sh", "echo", "hi"][..], &["sh", "echo", "a", "bb"][..]] {
        let (ic, iout) = run(&shell, &cmd, argv, false);
        let (jc, jout) = run(&shell, &cmd, argv, true);
        // The command echoes its argv (`argv[1..]` of the shell) and returns that count.
        let cmd_argv = &argv[1..];
        let expect: Vec<u8> = cmd_argv
            .iter()
            .flat_map(|a| [a.as_bytes(), b"\n"].concat())
            .collect();
        assert_eq!(
            iout, expect,
            "interp: command echoed its argv to the shell's sink"
        );
        assert_eq!(
            ic,
            cmd_argv.len() as i64,
            "interp: shell status = command argc"
        );
        assert_eq!(jout, iout, "jit: exec output must match interp");
        assert_eq!(jc, ic, "jit: exec status must match interp");
    }
}
