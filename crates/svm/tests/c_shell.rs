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
long __px_open(int cap, long path, long len, long flags);
long __px_close(int cap, long fd);
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
long open(char *path, long flags) { return __px_open(0, (long)path, slen(path), flags); }
long close(long fd) { return __px_close(0, fd); }
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

/// The Stage-0 shell itself (guest code). `run_line` first strips `< file`, `> file`, and `>> file`
/// redirects (pointing globals `in_fd`/`out_fd` at the targets via `open`, restored after), then
/// `exec_line` runs one command line (builtins only — `echo` with `$VAR`, `pwd`, `cd`, `cat`, `wc`,
/// `ls`, `exit`; unknown → `<cmd>: not found`), tokenizing it into a command and a single argument
/// on the first space. `cat`/`wc` read a path arg or the redirected `in_fd`; together with `>`/`>>`
/// this exercises the real file surface (`open`/`read`/`write`/`close`). `main` supports two
/// invocations: `sh -c "…"` (read via the personality's `argc`/`argv`) runs a single line; otherwise
/// it's a read-eval loop over stdin. `exit` calls the personality `exit`.
const SHELL_MAIN: &str = r#"
#define O_WRONLY 1
#define O_CREAT  0100
#define O_TRUNC  01000
#define O_APPEND 02000
static char cwd[256];
/* Current stdio for the command in flight: `out_fd` is 1 unless a `>`/`>>` redirect is active,
   `in_fd` is 0 unless a `<` redirect is active. run_line points them at files and restores them. */
static long out_fd = 1;
static long in_fd = 0;
static int streq(char *a, char *b) { int i = 0; while (a[i] && a[i] == b[i]) i++; return a[i] == 0 && b[i] == 0; }
static void puts_(char *s) { write(out_fd, s, slen(s)); }

/* Emit a non-negative count in decimal (wc's output). */
static void put_num(long n) {
  static char b[24]; int i = 24;
  if (n == 0) { puts_("0"); return; }
  while (n > 0) { b[--i] = '0' + (int)(n % 10); n /= 10; }
  write(out_fd, b + i, 24 - i);
}

/* Read source for cat/wc: an explicit path arg (caller must close), else the current in_fd. */
static long src_fd(char *arg, int *close_it) {
  if (arg) { *close_it = 1; return open(arg, 0); }   /* O_RDONLY */
  *close_it = 0; return in_fd;
}

/* Execute one command after redirection has been stripped: `cmd` plus an optional single `arg`. */
static void exec_line(char *line) {
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
  } else if (streq(cmd, "cat")) {
    int ci; long fd = src_fd(arg, &ci);
    if (fd < 0) { puts_(arg); puts_(": not found\n"); }
    else {
      static char buf[256];
      long r;
      while ((r = read(fd, buf, 256)) > 0) write(out_fd, buf, r);
      if (ci) close(fd);
    }
  } else if (streq(cmd, "wc")) {
    int ci; long fd = src_fd(arg, &ci);
    if (fd < 0) { puts_(arg); puts_(": not found\n"); }
    else {
      static char buf[256];
      long r, lines = 0, words = 0, bytes = 0; int inword = 0;
      while ((r = read(fd, buf, 256)) > 0) {
        for (long i = 0; i < r; i++) {
          char c = buf[i]; bytes++;
          if (c == '\n') lines++;
          if (c == ' ' || c == '\n' || c == '\t') inword = 0;
          else { if (!inword) words++; inword = 1; }
        }
      }
      if (ci) close(fd);
      put_num(lines); puts_(" "); put_num(words); puts_(" "); put_num(bytes); puts_("\n");
    }
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

/* Strip `< file`, `> file`, and `>> file` redirects out of the line, pointing in_fd/out_fd at the
   targets for the duration of the command, then restore stdio. Absent any redirect, run straight to
   stdin/stdout. Multiple redirects are honored (e.g. `wc < in > out`). */
static void run_line(char *line) {
  long ifd = 0, ofd = 1, ci = -1, co = -1;
  int cmd_end = -1, i = 0;
  while (line[i]) {
    char op = line[i];
    if (op != '<' && op != '>') { i++; continue; }
    if (cmd_end < 0) cmd_end = i;
    line[i++] = 0;                       /* terminate the token to the left of the operator */
    int append = 0;
    if (op == '>' && line[i] == '>') { append = 1; i++; }
    while (line[i] == ' ') i++;          /* skip spaces before the filename */
    char *target = line + i;
    while (line[i] && line[i] != ' ' && line[i] != '<' && line[i] != '>') i++;
    char save = line[i]; line[i] = 0;    /* terminate the filename for open() */
    if (op == '<') {
      long fd = open(target, 0);         /* O_RDONLY */
      if (fd < 0) { puts_(target); puts_(": not found\n"); goto done; }
      ifd = fd; ci = fd;
    } else {
      long flags = append ? (O_WRONLY | O_CREAT | O_APPEND) : (O_WRONLY | O_CREAT | O_TRUNC);
      long fd = open(target, flags);
      if (fd < 0) { puts_(target); puts_(": cannot open\n"); goto done; }
      ofd = fd; co = fd;
    }
    line[i] = save;                      /* restore so scanning continues past the filename */
    if (save == 0) break;
  }
  if (cmd_end >= 0) { int e = cmd_end; while (e > 0 && line[e - 1] == ' ') line[--e] = 0; }
  in_fd = ifd; out_fd = ofd;
  exec_line(line);
done:
  if (ci >= 0) close(ci);
  if (co >= 0) close(co);
  in_fd = 0; out_fd = 1;
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

/// I/O redirection + `cat` end to end (S7 item 3): `echo … > f` opens/truncates a memfs file and
/// writes there instead of stdout (so the redirected lines are absent from captured stdout); `>>`
/// appends; `cat f` reads it back to stdout. Only the final `cat`s reach stdout, proving the
/// `open`/`write`/`read`/`close` round-trip through the personality on both backends.
#[test]
fn stage0_shell_redirection_and_cat() {
    let (iout, jout) = run_shell(
        b"echo first > /out\n\
          echo second >> /out\n\
          cat /out\n\
          echo only-stdout\n\
          cat /missing\n",
        &[],
        &[],
        &[],
    );
    assert_eq!(
        iout, b"first\nsecond\nonly-stdout\n/missing: not found\n",
        "interp: `>`/`>>` divert to the file; only the cats + bare echo hit stdout"
    );
    assert_eq!(jout, iout, "jit: redirection/cat output must match interp");
}

/// A truncating redirect (`>`) replaces the file's prior contents rather than appending: after two
/// separate `>` writes, `cat` sees only the second. Confirms `O_TRUNC` on re-open.
#[test]
fn stage0_shell_redirect_truncates() {
    let (iout, jout) = run_shell(
        b"echo one > /f\n\
          echo two > /f\n\
          cat /f\n",
        &[],
        &[],
        &[],
    );
    assert_eq!(
        iout, b"two\n",
        "interp: the second `>` truncated the first write"
    );
    assert_eq!(jout, iout, "jit: truncation output must match interp");
}

/// Input redirection (`<`) + `wc` (S7 item 3, cont.): write a two-line file, then `wc < /f` reads it
/// through the redirected `in_fd` and reports `lines words bytes`; `cat < /f` streams the same file
/// back to stdout. Proves `<` binds a file to the command's input and that arg-less `cat`/`wc`
/// consume it. `wc` with an explicit path arg matches the redirected form.
#[test]
fn stage0_shell_input_redirection_and_wc() {
    let (iout, jout) = run_shell(
        b"echo hello world > /f\n\
          echo again >> /f\n\
          wc < /f\n\
          wc /f\n\
          cat < /f\n",
        &[],
        &[],
        &[],
    );
    // "hello world\nagain\n" = 2 lines, 3 words, 18 bytes.
    assert_eq!(
        iout, b"2 3 18\n2 3 18\nhello world\nagain\n",
        "interp: `< /f` feeds wc/cat; path arg and redirect agree"
    );
    assert_eq!(
        jout, iout,
        "jit: input-redirection/wc output must match interp"
    );
}

/// Both redirections at once: `wc < in > out` reads one file and writes the counts to another, so
/// nothing reaches stdout; `cat out` then reveals the diverted result. Exercises the multi-redirect
/// path in `run_line`.
#[test]
fn stage0_shell_input_and_output_redirection() {
    let (iout, jout) = run_shell(
        b"echo a b c > /in\n\
          wc < /in > /out\n\
          cat /out\n",
        &[],
        &[],
        &[],
    );
    // "a b c\n" = 1 line, 3 words, 6 bytes; the wc line itself is diverted to /out.
    assert_eq!(
        iout, b"1 3 6\n",
        "interp: wc's output went to /out, surfaced by cat"
    );
    assert_eq!(
        jout, iout,
        "jit: combined-redirection output must match interp"
    );
}
