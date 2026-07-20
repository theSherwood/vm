//! Stage 1 (STAGE1.md §5) — **`spawn` in the POSIX personality**: a compiled-C shell running on the
//! real `svm-posix` personality dispatches an *unknown* command to a spawned external child instead of
//! printing `<cmd>: not found`. This folds the op-13 exec path (`c_shell_exec.rs`) onto the personality:
//! the shell reads its own `argv` through the personality (`argc`/`argv`), writes its own output through
//! the personality (`write`), looks a command up in the personality's **PATH registry** (`exec_lookup`),
//! and re-grants the personality's forwardable stdout (`exec_stdout`) to the child — but the spawn
//! itself is the shell's own `Instantiator.instantiate_module_named` (op 13) + `join`, driven through
//! capability imports (`Resolved::Cap`, link-time symbol resolution; the guest discovers the
//! `Instantiator`/personality handles itself via `cap.self` reflection).
//!
//! The two stdout models are **unified**: the personality's fd-1 writes and the child's re-granted
//! `Stream` writes both land in the `Host`'s shared sink (`Host::shared_stdout` + `Posix::set_stdout_sink`),
//! so the shell's own output and the command's output interleave in one stream. Differential
//! interp==JIT (the JIT is given the module resolver *and* the named-grant hooks op 13 needs).
//!
//! Three paths exercised: a **builtin** (handled without spawning), an **external command** (spawned,
//! its argv echoed to the shared sink, its `argc` the shell's status), and a **not-found** command
//! (`exec_lookup` returns `-1` → status 127). *Follow-up:* the full `c_shell.rs` builtin dispatch (its
//! 128 KiB window / personality-heap-at-`win/2` layout vs. a 128 KiB command carve).
//!
//! Gated `#![cfg(unix)]` (needs the chibicc toolchain).
#![cfg(unix)]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use core::ffi::c_void;
use svm_interp::{run_capture_reserved_with_host, Host, StreamRole, Trap};
use svm_ir::Resolved;
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
    let base = std::env::temp_dir().join(format!("svm_pxspawn_{}_{id}", std::process::id()));
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

/// The external command: echo every `argv[i]` on its own line (ambient `write(1, …)`), return `argc`.
const CMD: &str = r#"
long write(long fd, void *buf, long n);
static long slen(char *s){ long n=0; while(s[n]) n++; return n; }
int main(int argc, char **argv){
  for (int i = 0; i < argc; i++){ write(1, argv[i], slen(argv[i])); write(1, "\n", 1); }
  return argc;
}
"#;

/// The shell, running on the `svm-posix` personality. It reads its own `argv` via the personality
/// (`__px_argc`/`__px_argv`), writes its own output via the personality (`__px_write`), and — for a
/// command that is not the `hi` builtin — looks it up (`__px_exec_lookup`) and either reports
/// `not found` or spawns it. The spawn re-grants the personality's forwardable stdout
/// (`__px_exec_stdout`) under the name `"stdout"` and drives `Instantiator` op 13 (`__spawn`) + `join`.
/// `pool` (a big writable global) forces a window large enough to hold a 128 KiB-aligned command carve.
const SHELL: &str = r#"
long __px_write(int h, long fd, void *buf, long n);
long __px_exec_lookup(int h, void *name, long len);
long __px_exec_stdout(int h);
int  __px_argc(int h);
long __px_argv(int h, long i, void *buf, long cap);
long __spawn(int inst, long module, long gp, long gn, long entry, long off, long sl, long q);
long __join(int inst, long child);
int __vm_cap_count(void);
int __vm_cap_at(int i, int *type_id_out);
static long slen(char *s){ long n=0; while(s[n]) n++; return n; }
/* Discover the handle of interface `want` from the domain's own capability table
   (cap.self reflection — the discovery tier IMPORTS.md keeps). */
static int __capof(int want) {
  int n = __vm_cap_count();
  int i = 0;
  while (i < n) {
    int t = 0;
    int h = __vm_cap_at(i, &t);
    if (t == want) return h;
    i = i + 1;
  }
  return -1;
}
static int __h_px = -1;
static int __px(void) { if (__h_px < 0) __h_px = __capof(13); return __h_px; }   /* HOST_FN = 13 */
static int __h_inst = -1;
static int __inst(void) { if (__h_inst < 0) __h_inst = __capof(6); return __h_inst; } /* Instantiator = 6 */
static int  seq(char *a, char *b){ long i=0; for(;;){ if(a[i]!=b[i]) return 0; if(!a[i]) return 1; i++; } }
/* 384 KiB: room for the grant record/name low, a 128 KiB-aligned 128 KiB carve, all below the SP. */
static char pool[393216];
int main(void){
  int n = __px_argc(__px());
  if (n < 2){ __px_write(__px(), 1, "usage\n", 6); return 2; }
  char cmd[256];
  __px_argv(__px(), 1, cmd, 256);            /* argv[1] = command name */
  if (seq(cmd, "hi")){ __px_write(__px(), 1, "hi from shell\n", 14); return 0; }  /* a builtin */
  long mod = __px_exec_lookup(__px(), cmd, slen(cmd));
  if (mod < 0){
    __px_write(__px(), 1, cmd, slen(cmd));
    __px_write(__px(), 1, ": not found\n", 12);
    return 127;
  }
  long out = __px_exec_stdout(__px());
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
  hdr[0] = n - 1;
  hdr[1] = 0;
  char *p = ab + 8;
  for (int i = 1; i < n; i++){
    char tmp[256];
    long L = __px_argv(__px(), i, tmp, 256);
    for (long k = 0; k < L; k++) *p++ = tmp[k];
    *p++ = 0;
  }
  long child = __spawn(__inst(), mod, base, 1, 0, carve, 17, 0);
  return __join(__inst(), child);
}
"#;

/// Link the shim's import names to their interfaces — link-time symbol resolution (the phase-4
/// linker-only `resolve_imports_with`; IMPORTS.md §2.5): `__px_*` names strip the prefix and map
/// through [`svm_posix::resolve`] to `(HOST_FN, op)`; `__spawn`/`__join` are the shell's own
/// `Instantiator` ops (13 / 1). No handle is baked at link: each lowered `cap.call` dispatches on
/// the guest's own handle operand, discovered at run time via `__vm_cap_count`/`__vm_cap_at`
/// reflection (§3c protection at the boundary, IMPORTS.md §2.3 dynamic mode).
fn link_shim(name: &str) -> Option<Resolved> {
    let cap = match name {
        "__spawn" => svm_ir::ResolvedCap { type_id: 6, op: 13 },
        "__join" => svm_ir::ResolvedCap { type_id: 6, op: 1 },
        n => svm_posix::resolve(n.strip_prefix("__px_")?)?,
    };
    Some(Resolved::Cap(cap))
}

fn grant_hooks() -> GrantChildHooks {
    GrantChildHooks {
        build: svm_run::grant_child_build,
        build_named: svm_run::grant_named_child_build,
        bind_imports: svm_run::child_bind_imports,
        release: svm_run::grant_child_release,
    }
}

/// Run the shell with `argv` (`argv[0]` = shell name, `argv[1]` = command, `argv[2..]` = its args);
/// return (status, unified stdout). Both the shell's own output and a spawned command's output land in
/// the one shared sink.
fn run(shell: &svm_ir::Module, cmd: &svm_ir::Module, argv: &[&str], jit: bool) -> (i64, Vec<u8>) {
    let win = 1usize << shell.memory.expect("shell window").size_log2;
    let mut host = Host::new();
    let sink = host.shared_stdout(); // the child's re-granted Stream writes here…
    let out_h = host.grant_stream(StreamRole::Out);
    let _inst_h = host.grant_instantiator(0, win as u64);
    let echo_h = host.grant_module(cmd);
    // The personality's heap sits in the top 64 KiB, clear of the shell's data/stack and the command
    // carve (a lean shell never `malloc`s, so this region stays untouched).
    let (_px_h, posix) =
        svm_posix::grant(&mut host, (win - (64 << 10)) as u64, win as u64, Vec::new());
    posix.set_stdout_sink(sink); // …and the shell's own fd-1 writes land in the same sink.
    posix.set_exec_stdout(out_h);
    posix.register_command("echo", echo_h);
    posix.set_args(argv);

    let m = svm_ir::resolve_imports_with(shell, link_shim).expect("resolve");
    verify_module(&m).expect("verify shell");
    let init = vec![0u8; win];

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
        (code, posix.stdout())
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
        (code, posix.stdout())
    }
}

/// A shell on the real POSIX personality dispatches an unknown command to a spawned external child
/// (echoing its argv to the shared sink, its `argc` the shell's status), handles a builtin without
/// spawning, and reports `not found` — identically on both backends. This is op 13 folded onto the
/// personality's `exec` surface, stdout unified through the `Host` sink.
#[test]
fn posix_shell_spawns_external_command() {
    let shell = parse_module_raw(&c_to_ir(SHELL, false)).expect("parse shell");
    // Phase 3: keep the manifest — the op-13 spawn binds the child's slots.
    let cmd = parse_module_raw(&c_to_ir(CMD, true)).expect("parse cmd");
    verify_module(&cmd).expect("verify cmd");

    // (argv, expected stdout, expected status)
    let cases: &[(&[&str], &[u8], i64)] = &[
        // builtin: handled in the shell, no spawn.
        (&["sh", "hi"], b"hi from shell\n", 0),
        // external command: spawned, echoes its argv (`argv[0]` = the command name), returns argc.
        (&["sh", "echo", "a", "bb"], b"echo\na\nbb\n", 3),
        (&["sh", "echo", "one"], b"echo\none\n", 2),
        // not found: exec_lookup returns -1.
        (&["sh", "nope"], b"nope: not found\n", 127),
    ];

    for &(argv, expect_out, expect_st) in cases {
        let (ic, iout) = run(&shell, &cmd, argv, false);
        let (jc, jout) = run(&shell, &cmd, argv, true);
        assert_eq!(iout, expect_out, "interp: stdout for {argv:?}");
        assert_eq!(ic, expect_st, "interp: status for {argv:?}");
        assert_eq!(jout, iout, "jit: stdout must match interp for {argv:?}");
        assert_eq!(jc, ic, "jit: status must match interp for {argv:?}");
    }
}
