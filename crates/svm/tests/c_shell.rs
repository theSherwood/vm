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
long __px_unlink(int cap, long path, long len);
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
long unlink(char *path) { return __px_unlink(0, (long)path, slen(path)); }
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
/// `exec_line` tokenizes the remainder into `argv[]` and runs one builtin — `echo` (with `$VAR` and
/// `$?`), `pwd`, `cd`, `cat`, `wc`, `grep`, `head`/`tail` (`-n N`), `rm`, `ls`, `true`/`false`,
/// `test`/`[ … ]`, `exit`; unknown → `<cmd>: not found`. Every command yields an exit status
/// (`grep` no-match → 1, unknown → 127, `test` per its predicate); the last is kept in `last_status`
/// and surfaced as `$?`. The text filters (`cat`/`wc`/`grep`/`head`/`tail`) read a path arg or the
/// redirected `in_fd`; together with `>`/`>>` and `rm` (`unlink`) this exercises the real file
/// surface (`open`/`read`/`write`/`close`/`unlink`). `main` supports two
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
/* Exit status of the last command, surfaced as `$?` and consumed by `&&`/`||`. 0 = success. */
static int last_status = 0;
static int streq(char *a, char *b) { int i = 0; while (a[i] && a[i] == b[i]) i++; return a[i] == 0 && b[i] == 0; }
static void puts_(char *s) { write(out_fd, s, slen(s)); }

/* Emit a non-negative count in decimal (wc's output). */
static void put_num(long n) {
  static char b[24]; int i = 24;
  if (n == 0) { puts_("0"); return; }
  while (n > 0) { b[--i] = '0' + (int)(n % 10); n /= 10; }
  write(out_fd, b + i, 24 - i);
}

/* Read source for a filter: an explicit path (caller must close), else the current in_fd. */
static long src_fd(char *path, int *close_it) {
  if (path) { *close_it = 1; return open(path, 0); }   /* O_RDONLY */
  *close_it = 0; return in_fd;
}

/* Parse a non-negative decimal prefix (head/tail counts). */
static long atoi_(char *s) {
  long v = 0; int i = 0;
  while (s[i] >= '0' && s[i] <= '9') { v = v * 10 + (s[i] - '0'); i++; }
  return v;
}

/* Substring test: does `hay` contain `needle`? An empty needle matches. */
static int contains(char *hay, char *needle) {
  for (int i = 0; hay[i]; i++) {
    int j = 0; while (needle[j] && hay[i + j] == needle[j]) j++;
    if (needle[j] == 0) return 1;
  }
  return needle[0] == 0;
}

/* Read one newline-delimited line from fd into buf (NUL-terminated, newline dropped); returns its
   length, or -1 at EOF with nothing read. One byte per read keeps it correct across any source. */
static long read_line(long fd, char *buf, long lim) {
  long n = 0; char c;
  for (;;) {
    long r = read(fd, &c, 1);
    if (r <= 0) { if (n == 0) return -1; break; }
    if (c == '\n') break;
    if (n < lim - 1) buf[n++] = c;
  }
  buf[n] = 0;
  return n;
}

/* Split `line` into space-separated tokens (runs of spaces collapse), writing pointers into argv and
   returning the count (capped at MAXARGS). Mutates `line` in place with NUL terminators. */
#define MAXARGS 16
static int tokenize(char *line, char **argv) {
  int argc = 0, i = 0;
  for (;;) {
    while (line[i] == ' ') i++;
    if (line[i] == 0 || argc >= MAXARGS) break;
    argv[argc++] = line + i;
    while (line[i] && line[i] != ' ') i++;
    if (line[i] == ' ') line[i++] = 0;
  }
  return argc;
}

/* Evaluate `test`/`[ … ]` and return a shell status (0 = true). Supports: a lone non-empty string;
   unary `-f`/`-d`/`-e` (file / dir / either exists), `-z`/`-n` (empty / non-empty); and binary
   `=`/`!=` (string) and `-eq`/`-ne`/`-lt`/`-gt` (numeric). Anything else is false. */
static int do_test(int argc, char **argv) {
  int top = argc;
  if (streq(argv[0], "[") && top > 1 && streq(argv[top - 1], "]")) top--;   /* drop the closing `]` */
  char **a = argv + 1;
  int n = top - 1;
  if (n == 1) return a[0][0] ? 0 : 1;
  if (n == 2) {
    if (streq(a[0], "-f")) { long fd = open(a[1], 0); if (fd >= 0) { close(fd); return 0; } return 1; }
    if (streq(a[0], "-d")) { long d = opendir(a[1]); if (d >= 0) { closedir(d); return 0; } return 1; }
    if (streq(a[0], "-e")) {
      long fd = open(a[1], 0); if (fd >= 0) { close(fd); return 0; }
      long d = opendir(a[1]); if (d >= 0) { closedir(d); return 0; }
      return 1;
    }
    if (streq(a[0], "-z")) return a[1][0] ? 1 : 0;
    if (streq(a[0], "-n")) return a[1][0] ? 0 : 1;
    return 1;
  }
  if (n == 3) {
    if (streq(a[1], "=")) return streq(a[0], a[2]) ? 0 : 1;
    if (streq(a[1], "!=")) return streq(a[0], a[2]) ? 1 : 0;
    if (streq(a[1], "-eq")) return atoi_(a[0]) == atoi_(a[2]) ? 0 : 1;
    if (streq(a[1], "-ne")) return atoi_(a[0]) != atoi_(a[2]) ? 0 : 1;
    if (streq(a[1], "-lt")) return atoi_(a[0]) < atoi_(a[2]) ? 0 : 1;
    if (streq(a[1], "-gt")) return atoi_(a[0]) > atoi_(a[2]) ? 0 : 1;
    return 1;
  }
  return 1;
}

/* Execute one command after redirection has been stripped. `line` is tokenized into argv; builtins
   read their input from a path argument or, absent one, the (possibly redirected) in_fd. Returns the
   command's exit status (0 = success). */
static int exec_line(char *line) {
  char *argv[MAXARGS];
  int argc = tokenize(line, argv);
  if (argc == 0) return 0;
  char *cmd = argv[0];
  char *arg = argc > 1 ? argv[1] : 0;
  int st = 0;
  if (streq(cmd, "echo")) {
    for (int i = 1; i < argc; i++) {
      char *a = argv[i];
      if (streq(a, "$?")) put_num(last_status);
      else if (a[0] == '$') { char *v = getenv(a + 1); if (v) puts_(v); }
      else puts_(a);
      if (i + 1 < argc) puts_(" ");
    }
    puts_("\n");
  } else if (streq(cmd, "true")) {
    st = 0;
  } else if (streq(cmd, "false")) {
    st = 1;
  } else if (streq(cmd, "test") || streq(cmd, "[")) {
    st = do_test(argc, argv);
  } else if (streq(cmd, "pwd")) {
    if (getcwd(cwd, 256)) puts_(cwd);
    puts_("\n");
  } else if (streq(cmd, "cd")) {
    if (arg) chdir(arg);
  } else if (streq(cmd, "cat")) {
    int ci; long fd = src_fd(arg, &ci);
    if (fd < 0) { puts_(arg); puts_(": not found\n"); st = 1; }
    else {
      static char buf[256];
      long r;
      while ((r = read(fd, buf, 256)) > 0) write(out_fd, buf, r);
      if (ci) close(fd);
    }
  } else if (streq(cmd, "wc")) {
    int ci; long fd = src_fd(arg, &ci);
    if (fd < 0) { puts_(arg); puts_(": not found\n"); st = 1; }
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
  } else if (streq(cmd, "grep")) {
    int matched = 0;
    if (argc > 1) {
      char *pat = argv[1];
      int ci; long fd = src_fd(argc > 2 ? argv[2] : 0, &ci);
      if (fd < 0) { puts_(argv[2]); puts_(": not found\n"); st = 2; }
      else {
        static char lb[256];
        while (read_line(fd, lb, 256) >= 0)
          if (contains(lb, pat)) { puts_(lb); puts_("\n"); matched = 1; }
        if (ci) close(fd);
      }
    }
    if (st == 0 && !matched) st = 1;   /* grep: no match is exit 1 */
  } else if (streq(cmd, "head")) {
    int ai = 1; long n = 10;
    if (argc > ai && streq(argv[ai], "-n") && argc > ai + 1) { n = atoi_(argv[ai + 1]); ai += 2; }
    int ci; long fd = src_fd(argc > ai ? argv[ai] : 0, &ci);
    if (fd < 0) { puts_(argv[ai]); puts_(": not found\n"); st = 1; }
    else {
      static char lb[256];
      for (long k = 0; k < n && read_line(fd, lb, 256) >= 0; k++) { puts_(lb); puts_("\n"); }
      if (ci) close(fd);
    }
  } else if (streq(cmd, "tail")) {
    int ai = 1; long n = 10;
    if (argc > ai && streq(argv[ai], "-n") && argc > ai + 1) { n = atoi_(argv[ai + 1]); ai += 2; }
    if (n > 16) n = 16;   /* the ring holds at most 16 lines */
    int ci; long fd = src_fd(argc > ai ? argv[ai] : 0, &ci);
    if (fd < 0) { puts_(argv[ai]); puts_(": not found\n"); st = 1; }
    else {
      static char ring[16][256];
      long count = 0;
      while (read_line(fd, ring[count % 16], 256) >= 0) count++;
      if (ci) close(fd);
      long start = count > n ? count - n : 0;
      for (long k = start; k < count; k++) { puts_(ring[k % 16]); puts_("\n"); }
    }
  } else if (streq(cmd, "rm")) {
    for (int i = 1; i < argc; i++)
      if (unlink(argv[i]) < 0) { puts_(argv[i]); puts_(": not found\n"); st = 1; }
  } else if (streq(cmd, "ls")) {
    char *dir = arg ? arg : (getcwd(cwd, 256), cwd);
    long d = opendir(dir);
    if (d < 0) { puts_(dir); puts_(": not found\n"); st = 1; }
    else {
      static char name[128];
      while (readdir(d, name, 128) > 0) { puts_(name); puts_("\n"); }
      closedir(d);
    }
  } else if (streq(cmd, "exit")) {
    exit(argc > 1 ? (int)atoi_(argv[1]) : last_status);
  } else {
    puts_(cmd); puts_(": not found\n"); st = 127;
  }
  return st;
}

/* Strip `< file`, `> file`, and `>> file` redirects out of the line, pointing in_fd/out_fd at the
   targets for the duration of the command, then restore stdio. Absent any redirect, run straight to
   stdin/stdout. Multiple redirects are honored (e.g. `wc < in > out`). */
static int run_line(char *line) {
  long ifd = 0, ofd = 1, ci = -1, co = -1;
  int cmd_end = -1, i = 0, st = 1;
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
  st = exec_line(line);
done:
  if (ci >= 0) close(ci);
  if (co >= 0) close(co);
  in_fd = 0; out_fd = 1;
  return st;
}

int main(void) {
  static char cmd[256];
  /* `sh -c "<command>"` — a single command line delivered via argv. */
  if (argc_() >= 3) {
    static char flag[8];
    if (getarg(1, flag, 8) > 0 && streq(flag, "-c") && getarg(2, cmd, 256) > 0) {
      last_status = run_line(cmd);
      return last_status;
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
    last_status = run_line(line);
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

/// `grep` over a redirected file: only lines containing the pattern survive. Exercises the argv
/// tokenizer (pattern in argv[1], file via `<`) and line-buffered reading (`read_line`).
#[test]
fn stage0_shell_grep_filters_lines() {
    let (iout, jout) = run_shell(
        b"echo alpha > /f\n\
          echo beta >> /f\n\
          echo alps >> /f\n\
          grep al < /f\n",
        &[],
        &[],
        &[],
    );
    assert_eq!(
        iout, b"alpha\nalps\n",
        "interp: grep keeps lines containing `al`"
    );
    assert_eq!(jout, iout, "jit: grep output must match interp");
}

/// `head -n N` / `tail -n N` with an explicit path argument select the first / last N lines of a
/// six-line file. Exercises `-n` flag parsing (`atoi_`) and tail's ring buffer.
#[test]
fn stage0_shell_head_and_tail() {
    let (iout, jout) = run_shell(
        b"echo l1 > /f\n\
          echo l2 >> /f\n\
          echo l3 >> /f\n\
          echo l4 >> /f\n\
          echo l5 >> /f\n\
          echo l6 >> /f\n\
          head -n 2 /f\n\
          tail -n 2 /f\n",
        &[],
        &[],
        &[],
    );
    assert_eq!(
        iout, b"l1\nl2\nl5\nl6\n",
        "interp: head -n 2 → first two, tail -n 2 → last two"
    );
    assert_eq!(jout, iout, "jit: head/tail output must match interp");
}

/// `rm` removes a memfs file (`unlink`, op 8): after `rm /f`, `cat /f` reports not-found, and
/// removing an absent file reports not-found too. Multi-arg `rm` deletes each argument.
#[test]
fn stage0_shell_rm_removes_files() {
    let (iout, jout) = run_shell(
        b"echo x > /a\n\
          echo y > /b\n\
          rm /a /b\n\
          cat /a\n\
          rm /gone\n",
        &[],
        &[],
        &[],
    );
    assert_eq!(
        iout, b"/a: not found\n/gone: not found\n",
        "interp: rm unlinks; cat of a removed/absent file is not-found"
    );
    assert_eq!(jout, iout, "jit: rm output must match interp");
}

/// `echo` now joins multiple argv tokens with single spaces (argv tokenizer), collapsing the runs of
/// spaces in the source line. A `$VAR` token still expands mid-line.
#[test]
fn stage0_shell_echo_joins_argv() {
    let (iout, jout) = run_shell(
        b"echo  a   b    c\n\
          echo hi $WHO !\n",
        &[("WHO", "bob")],
        &[],
        &[],
    );
    assert_eq!(
        iout, b"a b c\nhi bob !\n",
        "interp: argv tokens rejoin with single spaces; $WHO expands"
    );
    assert_eq!(jout, iout, "jit: echo-join output must match interp");
}

/// Exit status via `$?`: `true`/`false` set 0/1, an unknown command sets 127, `grep` with no match
/// sets 1, and `echo $?` reports the previous command's status. Proves `exec_line` returns a status
/// that `main` threads into `last_status`.
#[test]
fn stage0_shell_exit_status() {
    let (iout, jout) = run_shell(
        b"true\n\
          echo $?\n\
          false\n\
          echo $?\n\
          nope\n\
          echo $?\n\
          echo hit > /f\n\
          grep zzz < /f\n\
          echo $?\n",
        &[],
        &[],
        &[],
    );
    assert_eq!(
        iout, b"0\n1\nnope: not found\n127\n1\n",
        "interp: $? tracks true/false/unknown/grep-miss"
    );
    assert_eq!(jout, iout, "jit: exit-status output must match interp");
}

/// `test` / `[ … ]`: string equality, numeric comparison, and file/dir predicates over the memfs.
/// Each result is read back through `$?`.
#[test]
fn stage0_shell_test_builtin() {
    let (iout, jout) = run_shell(
        b"test a = a\n\
          echo $?\n\
          [ 3 -gt 5 ]\n\
          echo $?\n\
          echo hi > /f\n\
          test -f /f\n\
          echo $?\n\
          test -d /nodir\n\
          echo $?\n\
          [ -n hello ]\n\
          echo $?\n",
        &[],
        &[],
        &[],
    );
    assert_eq!(
        iout, b"0\n1\n0\n1\n0\n",
        "interp: test string/numeric/-f/-d/-n predicates"
    );
    assert_eq!(jout, iout, "jit: test-builtin output must match interp");
}
