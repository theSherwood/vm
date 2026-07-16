//! A minimal **Stage-0 shell** (PROCESS.md §10 / S7) compiled through chibicc onto the POSIX
//! personality — a real read-eval loop over stdin with builtin commands, no `fork`/`exec`.
//!
//! This is the playground target in miniature: it proves a genuine command interpreter runs end to
//! end on `svm-posix` (the libc-as-host-caps personality), and it's the scaffold BusyBox `ash` slots
//! into once the fork/exec surface lands. The shell's libc calls reach the personality **by name**:
//! `write`/`read`/`exit` are *defined* by the guest shim (shadowing chibicc's Stream/Exit builtins,
//! S15b) and forward — fd preserved — to `__px_`-prefixed generic imports; `getcwd`/`chdir`/`getenv`
//! are ordinary generic imports. The resolver binds each name to the granted personality handle
//! (`svm_ir::Resolved::CapBound`, S15a), so there is no positional powerbox anywhere.
//!
//! The shell runs either a **script from preloaded stdin** (the personality's `read(0, …)` drains it)
//! or a single `sh -c "<command>"` — its `argv` delivered by the personality's host-side argument
//! vector (`argc`/`argv`, the symmetric analogue of `getenv`). It reaches the fs surface too:
//! `ls` drives `opendir`/`readdir`. It runs on **both** backends under identical personalities,
//! asserting they agree on the captured stdout — a cross-backend differential.
#![cfg(unix)]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use core::ffi::c_void;
use svm_interp::{run_with_host, Host, Trap};
use svm_jit::{compile_and_run_with_host, JitOutcome};
use svm_run::cap_thunk;
use svm_text::parse_module as parse_module_raw;
use svm_verify::verify_module;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}

/// Build the chibicc fork once per test binary.
fn chibicc() -> &'static Path {
    static CC: OnceLock<PathBuf> = OnceLock::new();
    CC.get_or_init(|| {
        let dir = repo_root().join("frontend/chibicc");
        let status = Command::new("make")
            .arg("-s")
            .current_dir(&dir)
            .status()
            .expect("run `make` to build the chibicc fork");
        assert!(status.success(), "chibicc build failed");
        dir.join("chibicc")
    })
    .as_path()
}

/// Compile a C source string to text IR via the frontend.
fn c_to_ir(src: &str) -> String {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static N: AtomicUsize = AtomicUsize::new(0);
    let id = N.fetch_add(1, Ordering::Relaxed);
    let base = std::env::temp_dir().join(format!("svm_cshell_{}_{id}", std::process::id()));
    let cfile = base.with_extension("c");
    let irfile = base.with_extension("svm");
    std::fs::write(&cfile, src).unwrap();
    let status = Command::new(chibicc())
        .args([
            "-cc1",
            "--emit-ir",
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

/// Bind the shim's `__px_`-prefixed import names to the personality (strip the prefix, resolve the
/// bare libc name to `(HOST_FN, op)` + the granted handle).
fn resolver(handle: i32) -> impl Fn(&str) -> Option<svm_ir::Resolved> {
    let bound = svm_posix::resolve_bound(handle);
    move |name| bound(name.strip_prefix("__px_")?)
}

/// The guest libc shim (guest code): standard libc names, adapting C's NUL-terminated `char*` calls
/// to the personality's explicit-length `(ptr, len)` ABI (POSIX.md §4). `write`/`read`/`exit` are
/// *defined* here so their definitions shadow chibicc's builtins (S15b).
const SHIM: &str = r#"
long __px_write(int cap, long fd, long buf, long len);
long __px_read(int cap, long fd, long buf, long len);
long __px_getcwd(int cap, long buf, long size);
long __px_chdir(int cap, long path, long len);
long __px_getenv(int cap, long name, long len);
void __px_exit(int cap, int code);
long __px_opendir(int cap, long path, long len);
long __px_readdir(int cap, long dir, long namebuf, long namecap);
long __px_closedir(int cap, long dir);
long __px_argc(int cap);
long __px_argv(int cap, long i, long buf, long cap2);

static long slen(char *s) { long n = 0; while (s[n]) n = n + 1; return n; }

long write(long fd, void *buf, long n) { return __px_write(0, fd, (long)buf, n); }
long read(long fd, void *buf, long n) { return __px_read(0, fd, (long)buf, n); }
char *getcwd(char *buf, long size) { return __px_getcwd(0, (long)buf, size) > 0 ? buf : 0; }
long chdir(char *path) { return __px_chdir(0, (long)path, slen(path)); }
char *getenv(char *name) { return (char *)__px_getenv(0, (long)name, slen(name)); }
void exit(int code) { __px_exit(0, code); }
/* A DIR is just the personality's stream handle (a long); readdir writes the next name. */
long opendir(char *path) { return __px_opendir(0, (long)path, slen(path)); }
long readdir(long dir, char *namebuf, long cap) { return __px_readdir(0, dir, (long)namebuf, cap); }
long closedir(long dir) { return __px_closedir(0, dir); }
/* The host-side argument vector (personality extension): sh reads its own argv here. */
int argc_(void) { return (int)__px_argc(0); }
long getarg(int i, char *buf, long cap) { return __px_argv(0, i, (long)buf, cap); }
"#;

/// The Stage-0 shell itself (guest code). `run_line` executes one command line (builtins only —
/// `echo` with `$VAR`, `pwd`, `cd`, `ls`, `exit`; unknown → `<cmd>: not found`), tokenizing it into a
/// command and a single argument on the first space. `main` supports two invocations: `sh -c "…"`
/// (read via the personality's `argc`/`argv`) runs a single line; otherwise it's a read-eval loop
/// over stdin. `exit` calls the personality `exit` (terminal).
const SHELL_MAIN: &str = r#"
static char cwd[256];
static int streq(char *a, char *b) { int i = 0; while (a[i] && a[i] == b[i]) i++; return a[i] == 0 && b[i] == 0; }
static void puts_(char *s) { write(1, s, slen(s)); }

static void run_line(char *line) {
  int sp = 0; while (line[sp] && line[sp] != ' ') sp++;
  char *cmd = line, *arg = 0;
  if (line[sp] == ' ') { line[sp] = 0; arg = line + sp + 1; }
  if (line[0] == 0) return;
  if (streq(cmd, "echo")) {
    if (arg && arg[0] == '$') { char *v = getenv(arg + 1); if (v) puts_(v); }
    else if (arg) puts_(arg);
    puts_("\n");
  } else if (streq(cmd, "pwd")) {
    if (getcwd(cwd, 256)) puts_(cwd);
    puts_("\n");
  } else if (streq(cmd, "cd")) {
    if (arg) chdir(arg);
  } else if (streq(cmd, "ls")) {
    char *dir = arg ? arg : (getcwd(cwd, 256), cwd);
    long d = opendir(dir);
    if (d < 0) { puts_(dir); puts_(": not found\n"); }
    else {
      static char name[128];
      while (readdir(d, name, 128) > 0) { puts_(name); puts_("\n"); }
      closedir(d);
    }
  } else if (streq(cmd, "exit")) {
    exit(0);
  } else {
    puts_(cmd); puts_(": not found\n");
  }
}

int main(void) {
  static char cmd[256];
  /* `sh -c "<command>"` — a single command line delivered via argv. */
  if (argc_() >= 3) {
    static char flag[8];
    if (getarg(1, flag, 8) > 0 && streq(flag, "-c") && getarg(2, cmd, 256) > 0) {
      run_line(cmd);
      return 0;
    }
  }
  /* Otherwise: a read-eval loop over stdin. */
  static char line[256];
  for (;;) {
    int n = 0;
    for (;;) {
      char c;
      long r = read(0, &c, 1);
      if (r <= 0) { if (n == 0) return 0; break; }   /* EOF ends the shell */
      if (c == '\n') break;
      if (n < 255) line[n++] = c;
    }
    line[n] = 0;
    run_line(line);
  }
}
"#;

/// Compile the shell, grant the personality (with `stdin` preloaded as the script) on two identical
/// hosts, resolve libc by name, and run on **both** backends. `env` seeds the personality environment
/// and `files` seeds the memfs before the run. Returns each backend's captured stdout (asserted equal
/// for the differential).
fn run_shell(
    stdin: &[u8],
    env: &[(&str, &str)],
    files: &[&str],
    args: &[&str],
) -> (Vec<u8>, Vec<u8>) {
    let src = format!("{SHIM}\n{SHELL_MAIN}");
    let ir = c_to_ir(&src);
    let raw = parse_module_raw(&ir)
        .unwrap_or_else(|e| panic!("parse IR failed: {e:?}\n--- IR ---\n{ir}"));
    let win = 1u64 << raw.memory.expect("frontend declares a window").size_log2;

    let mut ih = Host::new();
    let (ipx, iposix) = svm_posix::grant(&mut ih, win / 2, win, stdin.to_vec());
    let mut jh = Host::new();
    let (jpx, jposix) = svm_posix::grant(&mut jh, win / 2, win, stdin.to_vec());
    assert_eq!(ipx, jpx, "identical grant order → identical handle");
    for (k, v) in env {
        iposix.set_env(k, v);
        jposix.set_env(k, v);
    }
    for path in files {
        iposix.write_file(path, b"");
        jposix.write_file(path, b"");
    }
    if !args.is_empty() {
        iposix.set_args(args);
        jposix.set_args(args);
    }

    let m = svm_ir::resolve_imports_with(&raw, resolver(ipx))
        .unwrap_or_else(|e| panic!("resolve imports: {e:?}\n--- IR ---\n{ir}"));
    verify_module(&m).unwrap_or_else(|e| panic!("verify failed: {e:?}\n--- IR ---\n{ir}"));

    // Interpreter: the shell loops to EOF and returns 0 (or `exit`s, a `Trap::Exit`).
    let mut fuel = 50_000_000u64;
    match run_with_host(&m, 0, &[], &mut fuel, &mut ih) {
        Ok(_) | Err(Trap::Exit(_)) => {}
        Err(e) => panic!("interp trapped: {e:?}\n--- IR ---\n{ir}"),
    }
    // JIT.
    let jout =
        compile_and_run_with_host(&m, 0, &[], cap_thunk, &mut jh as *mut Host as *mut c_void)
            .expect("jit compiles");
    assert!(
        matches!(jout, JitOutcome::Returned(_) | JitOutcome::Exited(_)),
        "jit ended abnormally: {jout:?}\n--- IR ---\n{ir}"
    );
    (iposix.stdout(), jposix.stdout())
}

/// The headline milestone: a real script runs through the shell loop end to end on the personality,
/// identically on both backends. `echo` (literal + `$VAR`), `pwd` after `cd`, and an unknown command
/// — every line's output is the personality's captured stdout.
#[test]
fn stage0_shell_runs_a_script() {
    let script = b"echo hello, shell\n\
                   echo $HOME\n\
                   cd /tmp\n\
                   pwd\n\
                   frobnicate\n\
                   exit\n";
    let (iout, jout) = run_shell(script, &[("HOME", "/root")], &[], &[]);
    assert_eq!(
        iout, b"hello, shell\n/root\n/tmp\nfrobnicate: not found\n",
        "interp: the shell ran the script (echo, $VAR, cd+pwd, unknown cmd)"
    );
    assert_eq!(jout, iout, "jit: shell output must match interp");
}

/// EOF (no trailing `exit`) cleanly ends the loop — the personality's `read(0, …)` returns `0` at the
/// end of the preloaded script, and `main` returns. Also checks a bare `pwd` at the default cwd `/`.
#[test]
fn stage0_shell_handles_eof_and_default_cwd() {
    let (iout, jout) = run_shell(b"pwd\necho done", &[], &[], &[]);
    assert_eq!(
        iout, b"/\ndone\n",
        "interp: default cwd is / then echo, then EOF ends it"
    );
    assert_eq!(jout, iout, "jit: must match interp");
}

/// The `ls` builtin drives the personality's `opendir`/`readdir`/`closedir` from compiled C: with a
/// memfs staged, `ls /tmp` lists the immediate children (files and the subdir once), sorted; `ls` of
/// a missing directory reports `not found`. Proves the fs-metadata surface (S7 item 2) end to end.
#[test]
fn stage0_shell_ls_lists_a_directory() {
    let (iout, jout) = run_shell(
        b"ls /tmp\nls /nope\n",
        &[],
        &["/tmp/a.txt", "/tmp/b.txt", "/tmp/sub/c"],
        &[],
    );
    assert_eq!(
        iout, b"a.txt\nb.txt\nsub\n/nope: not found\n",
        "interp: ls lists sorted children (subdir once), then a miss"
    );
    assert_eq!(jout, iout, "jit: ls output must match interp");
}

/// `sh -c "<command>"` — the standard non-interactive shell invocation, delivered through the
/// personality's host-side argument vector (`argc`/`argv`, S7 item 1). No stdin script; the command
/// comes from `argv[2]`. Runs one line (`echo $HOME`) and returns, differential on both backends.
#[test]
fn stage0_shell_dash_c_runs_one_command() {
    let (iout, jout) = run_shell(
        b"", // no stdin script — the command is in argv
        &[("HOME", "/home/user")],
        &[],
        &["sh", "-c", "echo $HOME"],
    );
    assert_eq!(iout, b"/home/user\n", "interp: sh -c ran the argv command");
    assert_eq!(jout, iout, "jit: sh -c output must match interp");
}
