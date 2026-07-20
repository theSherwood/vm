//! A minimal **Stage-0 shell** (PROCESS.md §10 / S7) compiled through chibicc onto the POSIX
//! personality — a real read-eval loop over stdin with builtin commands, no `fork`/`exec`.
//!
//! This is the playground target in miniature: it proves a genuine command interpreter runs end to
//! end on `svm-posix` (the libc-as-host-caps personality), and it's the scaffold BusyBox `ash` slots
//! into once the fork/exec surface lands. The shell's libc calls reach the personality **by name**:
//! `write`/`read`/`exit` are *defined* by the guest shim (shadowing chibicc's Stream/Exit builtins,
//! S15b) and forward — fd preserved — to `__px_`-prefixed generic imports; `getcwd`/`chdir`/`getenv`
//! are ordinary generic imports. The linker maps each name to its interface `(HOST_FN, op)`
//! (`svm_ir::Resolved::Cap`, link-time symbol resolution); the guest discovers the granted handles
//! itself via `cap.self` reflection, so there is no positional powerbox anywhere.
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
use svm_interp::{run_capture_reserved_with_host, Host, StreamRole, Trap};
use svm_jit::{compile_and_run_capture_reserved_with_host_ex, GrantChildHooks, JitOutcome};
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

/// Compile a C source string to text IR with the `--child-entry` spawnable §14 child ABI — how an
/// external command the shell `exec`s (STAGE1.md §5) is built.
fn c_to_ir_child(src: &str) -> String {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static N: AtomicUsize = AtomicUsize::new(0);
    let id = N.fetch_add(1, Ordering::Relaxed);
    let base = std::env::temp_dir().join(format!("svm_cshcmd_{}_{id}", std::process::id()));
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
    assert!(status.success(), "chibicc --child-entry failed on:\n{src}");
    std::fs::read_to_string(&irfile).unwrap()
}

/// The op-13 named-grant hooks the JIT needs to spawn a separate-module child with a by-name powerbox.
fn grant_hooks() -> GrantChildHooks {
    GrantChildHooks {
        build: svm_run::grant_child_build,
        build_named: svm_run::grant_named_child_build,
        bind_imports: svm_run::child_bind_imports,
        release: svm_run::grant_child_release,
    }
}

/// Link the shim's import names to their interfaces — link-time symbol resolution (the phase-4
/// linker-only `resolve_imports_with`; IMPORTS.md §2.5): `__px_*` names strip the prefix and map
/// through [`svm_posix::resolve`] to `(HOST_FN, op)`; `__spawn`/`__join` are the shell's own
/// `Instantiator` ops (13 / 1, STAGE1.md §5). No handle is baked at link: each lowered `cap.call`
/// dispatches on the guest's own handle operand, discovered at run time via
/// `__vm_cap_count`/`__vm_cap_at` reflection (§3c protection at the boundary, IMPORTS.md §2.3
/// dynamic mode).
fn link_shim(name: &str) -> Option<svm_ir::Resolved> {
    let cap = match name {
        "__spawn" => svm_ir::ResolvedCap { type_id: 6, op: 13 },
        "__join" => svm_ir::ResolvedCap { type_id: 6, op: 1 },
        n => svm_posix::resolve(n.strip_prefix("__px_")?)?,
    };
    Some(svm_ir::Resolved::Cap(cap))
}

/// The guest libc shim (guest code): standard libc names, adapting C's NUL-terminated `char*` calls
/// to the personality's explicit-length `(ptr, len)` ABI (POSIX.md §4). `write`/`read`/`exit` are
/// *defined* here so their definitions shadow chibicc's builtins (S15b).
const SHIM: &str = r#"
int __vm_cap_count(void);
int __vm_cap_at(int i, int *type_id_out);

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

long __px_write(int cap, long fd, long buf, long len);
long __px_read(int cap, long fd, long buf, long len);
long __px_open(int cap, long path, long len, long flags);
long __px_close(int cap, long fd);
long __px_unlink(int cap, long path, long len);
long __px_getcwd(int cap, long buf, long size);
long __px_chdir(int cap, long path, long len);
long __px_getenv(int cap, long name, long len);
long __px_setenv(int cap, long name, long nlen, long val, long vlen, long overwrite);
void __px_exit(int cap, int code);
long __px_opendir(int cap, long path, long len);
long __px_readdir(int cap, long dir, long namebuf, long namecap);
long __px_closedir(int cap, long dir);
long __px_argc(int cap);
long __px_argv(int cap, long i, long buf, long cap2);
/* Personality `exec` surface (STAGE1.md §5): PATH lookup + the forwardable stdout handle. The spawn
   itself is the shell's own `Instantiator` cap.call — `__spawn` (op 13) / `__join` (op 1) — dispatched
   on the reflection-discovered `Instantiator` handle (`__inst()`), like every import here. */
long __px_exec_lookup(int cap, long name, long len);
long __px_exec_stdout(int cap);
long __spawn(int inst, long module, long gp, long gn, long entry, long off, long sl, long q);
long __join(int inst, long child);

static long slen(char *s) { long n = 0; while (s[n]) n = n + 1; return n; }

long write(long fd, void *buf, long n) { return __px_write(__px(), fd, (long)buf, n); }
long read(long fd, void *buf, long n) { return __px_read(__px(), fd, (long)buf, n); }
long open(char *path, long flags) { return __px_open(__px(), (long)path, slen(path), flags); }
long close(long fd) { return __px_close(__px(), fd); }
long unlink(char *path) { return __px_unlink(__px(), (long)path, slen(path)); }
char *getcwd(char *buf, long size) { return __px_getcwd(__px(), (long)buf, size) > 0 ? buf : 0; }
long chdir(char *path) { return __px_chdir(__px(), (long)path, slen(path)); }
char *getenv(char *name) { return (char *)__px_getenv(__px(), (long)name, slen(name)); }
long setenv_(char *name, char *val) { return __px_setenv(__px(), (long)name, slen(name), (long)val, slen(val), 1); }
void exit(int code) { __px_exit(__px(), code); }
/* A DIR is just the personality's stream handle (a long); readdir writes the next name. */
long opendir(char *path) { return __px_opendir(__px(), (long)path, slen(path)); }
long readdir(long dir, char *namebuf, long cap) { return __px_readdir(__px(), dir, (long)namebuf, cap); }
long closedir(long dir) { return __px_closedir(__px(), dir); }
/* The host-side argument vector (personality extension): sh reads its own argv here. */
int argc_(void) { return (int)__px_argc(__px()); }
long getarg(int i, char *buf, long cap) { return __px_argv(__px(), i, (long)buf, cap); }
"#;

/// The Stage-0 shell itself (guest code). `run_line` first strips `< file`, `> file`, and `>> file`
/// redirects (pointing globals `in_fd`/`out_fd` at the targets via `open`, restored after), then
/// `exec_line` tokenizes the remainder into `argv[]`, sets a shell variable for a lone `NAME=VALUE`,
/// then expands `$NAME`/`$?` tokens (shell vars shadow the environment) and glob tokens (`*`/`?`
/// matched against the memfs, `dir/name` results, literal if no match) before running one builtin —
/// `echo`, `export`, `pwd`, `cd`, `cat`, `wc`, `grep` (`-v`/`-c`), `head`/`tail` (`-n N`), `sort`, `uniq`, `rm`,
/// `ls`, `true`/`false`, `test`/`[ … ]`, `exit`; unknown → `<cmd>: not found`. Every command yields an exit status
/// (`grep` no-match → 1, unknown → 127, `test` per its predicate); the last is kept in `last_status`
/// and surfaced as `$?`. The text filters (`cat`/`wc`/`grep`/`head`/`tail`) read a path arg or the
/// redirected `in_fd`; together with `>`/`>>` and `rm` (`unlink`) this exercises the real file
/// surface (`open`/`read`/`write`/`close`/`unlink`). `run_list` (splitting on `;`/`&&`/`||`, short-
/// circuiting on `$?`) sits above `run_pipeline` (splitting on `|`, staging each stage's stdout
/// through a memfs temp the next stage reads as stdin) above `run_line`. `run_top` routes a line to
/// the single-line `if COND; then …; [else …;] fi` construct (`run_if`) or to a command list. `main`
/// supports two invocations: `sh -c "…"` (read via the personality's `argc`/`argv`) runs one line;
/// otherwise it's a read-eval loop over stdin. `exit` calls the personality `exit`.
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

/* Byte-wise string compare (for sort/uniq): <0, 0, >0. */
static int scmp(char *a, char *b) {
  int i = 0; while (a[i] && a[i] == b[i]) i++;
  return (int)(unsigned char)a[i] - (int)(unsigned char)b[i];
}

/* Bounded string copy (dst holds up to 255 bytes + NUL). */
static void scpy(char *d, char *s) {
  int i = 0; while (s[i] && i < 255) { d[i] = s[i]; i++; } d[i] = 0;
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
   returning the count (capped at MAXARGS). Mutates `line` in place with NUL terminators. The cap is
   generous so glob expansion (which grows argv) has room. */
#define MAXARGS 64
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

/* Shell variables (distinct from the personality environment until `export`ed). A tiny flat table:
   name/value pairs in fixed storage, linear-scanned. */
#define NVARS 32
static char var_name[NVARS][32];
static char var_val[NVARS][128];
static int nvars = 0;
static char *get_var(char *name) {
  for (int i = 0; i < nvars; i++)
    if (streq(var_name[i], name)) return var_val[i];
  return 0;
}
static void set_var(char *name, char *val) {
  int slot = -1;
  for (int i = 0; i < nvars; i++)
    if (streq(var_name[i], name)) { slot = i; break; }
  if (slot < 0) { if (nvars >= NVARS) return; slot = nvars++; }
  int i = 0; while (name[i] && i < 31) { var_name[slot][i] = name[i]; i++; } var_name[slot][i] = 0;
  int j = 0; while (val[j] && j < 127) { var_val[slot][j] = val[j]; j++; } var_val[slot][j] = 0;
}

/* Format a non-negative integer into one of a few rotating static buffers (for `$?` expansion). */
static char *itoa_(long n) {
  static char ring[4][24]; static int k = 0;
  char *b = ring[k]; k = (k + 1) & 3;
  char t[24]; int ti = 0;
  if (n == 0) t[ti++] = '0';
  while (n > 0) { t[ti++] = '0' + (int)(n % 10); n /= 10; }
  int bi = 0; while (ti > 0) b[bi++] = t[--ti]; b[bi] = 0;
  return b;
}

/* Expand one token: `$?` → last status, `$NAME` → shell var then environment (empty if unset), else
   the token unchanged. Returned pointers stay valid for the command's duration. */
static char *expand(char *tok) {
  if (tok[0] != '$') return tok;
  char *name = tok + 1;
  if (streq(name, "?")) return itoa_(last_status);
  char *v = get_var(name);
  if (v) return v;
  v = getenv(name);
  return v ? v : "";
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

/* Glob match with `*` (any run, incl. empty) and `?` (one char). Iterative with backtracking. */
static int fnmatch_(char *p, char *s) {
  char *star = 0, *ss = 0;
  while (*s) {
    if (*p == '?' || *p == *s) { p++; s++; }
    else if (*p == '*') { star = p++; ss = s; }
    else if (star) { p = star + 1; s = ++ss; }
    else return 0;
  }
  while (*p == '*') p++;
  return *p == 0;
}

/* Expand a glob token into matching absolute paths, appending pointers (into `store`) to `out`. The
   token is split at its last `/` into a directory (default: cwd) and a pattern; each directory entry
   matching the pattern yields `dir/name`. Returns the number of matches appended. */
static int glob_expand(char *tok, char **out, int *oc, char store[][256], int *sn, int cap) {
  int last = -1;
  for (int i = 0; tok[i]; i++) if (tok[i] == '/') last = i;
  static char dir[256]; char *pat;
  if (last < 0) { getcwd(dir, 256); pat = tok; }
  else if (last == 0) { dir[0] = '/'; dir[1] = 0; pat = tok + 1; }
  else { int k = 0; while (k < last && k < 255) { dir[k] = tok[k]; k++; } dir[k] = 0; pat = tok + last + 1; }
  long d = opendir(dir);
  if (d < 0) return 0;
  static char name[256]; int matched = 0;
  while (readdir(d, name, 256) > 0) {
    if (!fnmatch_(pat, name)) continue;
    if (*oc >= cap || *sn >= 64) break;
    char *g = store[*sn];
    int p = 0, k = 0;
    while (dir[k] && p < 254) g[p++] = dir[k++];
    if (p == 0 || g[p - 1] != '/') g[p++] = '/';
    k = 0; while (name[k] && p < 255) g[p++] = name[k++];
    g[p] = 0;
    out[(*oc)++] = g; (*sn)++; matched++;
  }
  closedir(d);
  return matched;
}

/* Execute one command after redirection has been stripped. `line` is tokenized into argv; builtins
   read their input from a path argument or, absent one, the (possibly redirected) in_fd. Returns the
   command's exit status (0 = success). */
/* Spawn an external command (STAGE1.md §5). `pool` (a big writable global) forces a window large
   enough to hold a 128 KiB-aligned 128 KiB command carve below the stack, and holds the grant record.
   The command's stdout is the personality's forwardable `Stream` (`exec_stdout`), re-granted by name so
   its `write(1, …)` reaches the shell's sink — a `>`/`|` redirect on an external command is not honored
   (that needs the Power-2 `Endpoint`, STAGE1.md); the command always writes to the terminal sink. */
static char pool[393216];
static int spawn_cmd(long mod, int argc, char **argv) {
  long out = __px_exec_stdout(__px());
  long base = (long)pool;
  long carve = (base + 131071) & ~131071;
  /* grant record at base: {name_off, name_len, out, flags}; "stdout" name follows at base+16 */
  int *rec = (int *)base;
  rec[0] = (int)(base + 16); rec[1] = 6; rec[2] = (int)out; rec[3] = 0;
  char *nm = (char *)(base + 16);
  nm[0]='s'; nm[1]='t'; nm[2]='d'; nm[3]='o'; nm[4]='u'; nm[5]='t';
  /* the command's args buffer at carve+128 (POWERBOX_ARGS_BASE): {argc, envc=0} then packed argv */
  char *ab = (char *)(carve + 128);
  int *hdr = (int *)ab;
  hdr[0] = argc; hdr[1] = 0;
  char *p = ab + 8;
  for (int i = 0; i < argc; i++) {
    char *s = argv[i]; long L = slen(s);
    for (long k = 0; k < L; k++) *p++ = s[k];
    *p++ = 0;
  }
  long child = __spawn(__inst(), mod, base, 1, 0, carve, 17, 0);
  return (int)__join(__inst(), child);
}

static int exec_line(char *line) {
  char *argv[MAXARGS];
  int argc = tokenize(line, argv);
  if (argc == 0) return 0;
  /* A lone `NAME=VALUE` (identifier before `=`) sets a shell variable, expanding the RHS. */
  if (argc == 1) {
    char *t = argv[0]; int eq = -1;
    for (int j = 0; t[j]; j++) {
      char c = t[j];
      if (c == '=') { eq = j; break; }
      if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_')) break;
    }
    if (eq > 0) { t[eq] = 0; set_var(t, expand(t + eq + 1)); return 0; }
  }
  /* Expand `$`-tokens in place before dispatch. */
  for (int i = 0; i < argc; i++) argv[i] = expand(argv[i]);
  /* Glob-expand any token containing `*`/`?` against the memfs; a token with no match stays literal
     (bash's default nullglob-off). Rebuild argv from the expansion. */
  {
    static char gstore[64][256];
    char *gbuf[MAXARGS];
    int gn = 0, ac2 = 0;
    for (int i = 0; i < argc && ac2 < MAXARGS; i++) {
      char *tok = argv[i];
      int has = 0;
      for (int j = 0; tok[j]; j++) if (tok[j] == '*' || tok[j] == '?') { has = 1; break; }
      if (!has) { gbuf[ac2++] = tok; continue; }
      int m = glob_expand(tok, gbuf, &ac2, gstore, &gn, MAXARGS);
      if (m == 0 && ac2 < MAXARGS) gbuf[ac2++] = tok;
    }
    for (int i = 0; i < ac2; i++) argv[i] = gbuf[i];
    argc = ac2;
  }
  if (argc == 0) return 0;
  char *cmd = argv[0];
  char *arg = argc > 1 ? argv[1] : 0;
  int st = 0;
  if (streq(cmd, "echo")) {
    for (int i = 1; i < argc; i++) {
      puts_(argv[i]);
      if (i + 1 < argc) puts_(" ");
    }
    puts_("\n");
  } else if (streq(cmd, "export")) {
    for (int i = 1; i < argc; i++) {
      char *e = argv[i]; int eq = -1;
      for (int j = 0; e[j]; j++) if (e[j] == '=') { eq = j; break; }
      if (eq >= 0) { e[eq] = 0; char *v = expand(e + eq + 1); set_var(e, v); setenv_(e, v); }
      else { char *v = get_var(e); setenv_(e, v ? v : ""); }
    }
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
    static char buf[256];
    long r;
    if (argc > 1) {
      for (int ai = 1; ai < argc; ai++) {          /* concatenate each file argument */
        long fd = open(argv[ai], 0);
        if (fd < 0) { puts_(argv[ai]); puts_(": not found\n"); st = 1; }
        else { while ((r = read(fd, buf, 256)) > 0) write(out_fd, buf, r); close(fd); }
      }
    } else {
      while ((r = read(in_fd, buf, 256)) > 0) write(out_fd, buf, r);   /* no args: stream in_fd */
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
    int ai = 1, inv = 0, cnt = 0;       /* -v: invert match; -c: print only the match count */
    while (ai < argc && argv[ai][0] == '-') {
      if (streq(argv[ai], "-v")) inv = 1;
      else if (streq(argv[ai], "-c")) cnt = 1;
      ai++;
    }
    long matches = 0;
    if (ai < argc) {
      char *pat = argv[ai++];
      int ci; long fd = src_fd(ai < argc ? argv[ai] : 0, &ci);
      if (fd < 0) { puts_(argv[ai]); puts_(": not found\n"); st = 2; }
      else {
        static char lb[256];
        while (read_line(fd, lb, 256) >= 0) {
          int m = contains(lb, pat);
          if (inv) m = !m;
          if (m) { matches++; if (!cnt) { puts_(lb); puts_("\n"); } }
        }
        if (ci) close(fd);
      }
    }
    if (cnt) { put_num(matches); puts_("\n"); }
    if (st == 0 && matches == 0) st = 1;   /* grep: no match is exit 1 */
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
  } else if (streq(cmd, "sort")) {
    int ci; long fd = src_fd(arg, &ci);
    if (fd < 0) { puts_(arg); puts_(": not found\n"); st = 1; }
    else {
      static char buf[64][256]; int n = 0;
      while (n < 64 && read_line(fd, buf[n], 256) >= 0) n++;
      if (ci) close(fd);
      for (int i = 1; i < n; i++) {          /* insertion sort (n <= 64) */
        static char key[256]; scpy(key, buf[i]);
        int j = i - 1;
        while (j >= 0 && scmp(buf[j], key) > 0) { scpy(buf[j + 1], buf[j]); j--; }
        scpy(buf[j + 1], key);
      }
      for (int i = 0; i < n; i++) { puts_(buf[i]); puts_("\n"); }
    }
  } else if (streq(cmd, "uniq")) {
    int ci; long fd = src_fd(arg, &ci);
    if (fd < 0) { puts_(arg); puts_(": not found\n"); st = 1; }
    else {
      static char cur[256], prev[256]; int have = 0;
      while (read_line(fd, cur, 256) >= 0)
        if (!have || scmp(cur, prev) != 0) { puts_(cur); puts_("\n"); scpy(prev, cur); have = 1; }
      if (ci) close(fd);
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
    /* Not a builtin: look the command up in the personality's PATH registry and, if found, spawn it as
       an external child (STAGE1.md §5); otherwise the classic `<cmd>: not found`. */
    long mod = __px_exec_lookup(__px(), (long)cmd, slen(cmd));
    if (mod < 0) { puts_(cmd); puts_(": not found\n"); st = 127; }
    else st = spawn_cmd(mod, argc, argv);
  }
  return st;
}

/* Strip `< file`, `> file`, and `>> file` redirects out of the line, pointing in_fd/out_fd at the
   targets (or the caller-supplied `def_in`/`def_out` when a stream is not explicitly redirected) for
   the duration of the command, then restore stdio. An explicit redirect wins over the default — so a
   pipe stage that also redirects behaves like bash. `def_in`/`def_out` fds are owned by the caller
   and are not closed here. Multiple redirects on one line are honored (`wc < in > out`). */
static int run_line_io(char *line, long def_in, long def_out) {
  long ifd = def_in, ofd = def_out, ci = -1, co = -1;
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

/* A command with its own redirects but default stdin/stdout. */
static int run_line(char *line) { return run_line_io(line, 0, 1); }

/* Run a pipeline `A | B | C`: each stage's stdout is staged into a fresh memfs temp file that the
   next stage reads as stdin. Not real concurrent processes — the playground has no fork yet — but it
   reproduces pipeline *semantics* (each stage sees the previous stage's full output) end to end on
   the personality's file surface. The exit status is the last stage's. A stage may still carry its
   own `<`/`>` redirects, which override the pipe. Temp files are unlinked when the pipeline ends. */
static int run_pipeline(char *seg) {
  char *stages[8];
  int ns = 0, i = 0, start = 0;
  for (;;) {
    char c = seg[i];
    if (c == '|' && ns < 7) { seg[i] = 0; stages[ns++] = seg + start; start = i + 1; i++; }
    else if (c == 0) { stages[ns++] = seg + start; break; }
    else i++;
  }
  if (ns == 1) return run_line(stages[0]);
  static char tmp[8];                    /* "/.pipeN" — one name per producing stage */
  tmp[0] = '/'; tmp[1] = '.'; tmp[2] = 'p'; tmp[3] = 'i'; tmp[4] = 'p'; tmp[5] = 'e'; tmp[7] = 0;
  long prev_in = 0;                      /* stage 0 reads real stdin */
  int st = 0;
  for (int s = 0; s < ns; s++) {
    long def_out = 1, tmpfd = -1;
    if (s + 1 < ns) {
      tmp[6] = '0' + s;
      tmpfd = open(tmp, O_WRONLY | O_CREAT | O_TRUNC);
      def_out = tmpfd;
    }
    st = run_line_io(stages[s], prev_in, def_out);
    if (tmpfd >= 0) close(tmpfd);
    if (prev_in > 0) close(prev_in);     /* done reading the previous stage's temp */
    if (s + 1 < ns) { tmp[6] = '0' + s; prev_in = open(tmp, 0); }   /* next stage reads it */
  }
  if (prev_in > 0) close(prev_in);
  for (int s = 0; s + 1 < ns; s++) { tmp[6] = '0' + s; unlink(tmp); }
  return st;
}

/* Run a list of commands joined by `;`, `&&`, `||`, short-circuiting on `last_status`: `&&` runs the
   next segment only after success (0), `||` only after failure, `;` always. A skipped segment leaves
   `last_status` unchanged so it propagates down a chain, matching bash. Returns the final status. */
static int run_list(char *line) {
  int i = 0, start = 0, skip = 0;
  for (;;) {
    char c = line[i];
    int is_semi = c == ';';
    int is_and = c == '&' && line[i + 1] == '&';
    int is_or = c == '|' && line[i + 1] == '|';
    if (c == 0 || is_semi || is_and || is_or) {
      char save = c; line[i] = 0;
      if (!skip) last_status = run_pipeline(line + start);
      if (save == 0) break;
      if (is_and) skip = last_status != 0;
      else if (is_or) skip = last_status == 0;
      else skip = 0;                     /* `;` starts a fresh short-circuit context */
      i += is_semi ? 1 : 2;
      start = i;
    } else {
      i++;
    }
  }
  return last_status;
}

static char *ltrim(char *s) { while (*s == ' ') s++; return s; }
/* Does `w` start with the whole word `kw` (followed by a space or end)? */
static int kw_is(char *w, char *kw) {
  int i = 0; while (kw[i]) { if (w[i] != kw[i]) return 0; i++; }
  return w[i] == 0 || w[i] == ' ';
}

/* Single-line `if COND; then BODY…; [else BODY…;] fi`. Split on `;` into segments: the first holds
   `if COND`, later ones begin with `then`/`else` (whose remainder is the first body command) or are
   further body commands, ending at `fi`. Evaluate COND with run_list, then run the taken branch's
   commands. Returns the branch's status (0 when no branch runs). */
static int run_if(char *line) {
  char *seg[32]; int ns = 0, i = 0, start = 0;
  for (;;) {
    char c = line[i];
    if (c == ';' || c == 0) {
      line[i] = 0;
      if (ns < 32) seg[ns++] = line + start;
      if (c == 0) break;
      start = i + 1;
    }
    i++;
  }
  char *cond = ltrim(ltrim(seg[0]) + 2);   /* drop leading "if" */
  char *thenb[16]; int nt = 0;
  char *elseb[16]; int ne = 0;
  int mode = 0;                            /* 1 = collecting then-body, 2 = else-body */
  for (int s = 1; s < ns; s++) {
    char *w = ltrim(seg[s]);
    if (kw_is(w, "fi")) break;
    if (kw_is(w, "then")) { mode = 1; char *b = ltrim(w + 4); if (b[0] && nt < 16) thenb[nt++] = b; }
    else if (kw_is(w, "else")) { mode = 2; char *b = ltrim(w + 4); if (b[0] && ne < 16) elseb[ne++] = b; }
    else if (mode == 1 && nt < 16) thenb[nt++] = w;
    else if (mode == 2 && ne < 16) elseb[ne++] = w;
  }
  int rc = 0;
  if (run_list(cond) == 0) { for (int k = 0; k < nt; k++) rc = run_list(thenb[k]); }
  else { for (int k = 0; k < ne; k++) rc = run_list(elseb[k]); }
  return rc;
}

/* Top-level line dispatch: an `if …` line runs the conditional construct, everything else is a
   command list. */
static int run_top(char *line) {
  char *t = ltrim(line);
  if (kw_is(t, "if")) return run_if(t);
  return run_list(line);
}

int main(void) {
  static char cmd[256];
  /* `sh -c "<command>"` — a single command line delivered via argv. */
  if (argc_() >= 3) {
    static char flag[8];
    if (getarg(1, flag, 8) > 0 && streq(flag, "-c") && getarg(2, cmd, 256) > 0) {
      return run_top(cmd);
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
    last_status = run_top(line);
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
    run_shell_ex(stdin, env, files, args, &[])
}

/// As [`run_shell`], plus a **PATH registry** of external commands `(name, C source)`: each is compiled
/// `--child-entry`, granted as a `Module`, and registered so an unknown command name in the script is
/// `exec`'d as an external child (STAGE1.md §5) instead of `<cmd>: not found`. With no `cmds` (the
/// [`run_shell`] case) `exec_lookup` always misses, so the `not found` path is unchanged.
fn run_shell_ex(
    stdin: &[u8],
    env: &[(&str, &str)],
    files: &[&str],
    args: &[&str],
    cmds: &[(&str, &str)],
) -> (Vec<u8>, Vec<u8>) {
    let src = format!("{SHIM}\n{SHELL_MAIN}");
    let ir = c_to_ir(&src);
    let raw = parse_module_raw(&ir)
        .unwrap_or_else(|e| panic!("parse IR failed: {e:?}\n--- IR ---\n{ir}"));
    let win = 1usize << raw.memory.expect("frontend declares a window").size_log2;

    // The external command `Module`s (shared by both hosts), compiled with the spawnable child ABI.
    let cmd_mods: Vec<(&str, svm_ir::Module)> = cmds
        .iter()
        .map(|&(name, csrc)| {
            // Phase 3: keep the manifest — the op-13 spawn binds the child's slots.
            let m = parse_module_raw(&c_to_ir_child(csrc)).expect("parse cmd");
            verify_module(&m).expect("verify cmd");
            (name, m)
        })
        .collect();

    // Grant a personality + the spawn caps on one host; identical grant order across the two hosts keeps
    // the handles equal (so the guest's reflection scan discovers the same handles on both, keeping the
    // differential exact). The `Instantiator` (over the whole window) and a
    // forwardable stdout `Stream` back the shell's `__spawn`/`exec_stdout`; the personality's fd-1 writes
    // route to the same shared sink as the child's re-granted `Stream`, unifying their output.
    let setup = |host: &mut Host| -> (svm_posix::Posix, i32, i32) {
        let sink = host.shared_stdout();
        let out_h = host.grant_stream(StreamRole::Out);
        let inst_h = host.grant_instantiator(0, win as u64);
        let cmd_handles: Vec<(&str, i32)> = cmd_mods
            .iter()
            .map(|(n, m)| (*n, host.grant_module(m)))
            .collect();
        // The shell never `malloc`s, so the personality heap (top 64 KiB) is never touched — it just
        // stays clear of the command carve (inside `pool`, low) and the shell's stack.
        let (px_h, posix) =
            svm_posix::grant(host, (win - (64 << 10)) as u64, win as u64, stdin.to_vec());
        posix.set_stdout_sink(sink);
        posix.set_exec_stdout(out_h);
        for (n, h) in &cmd_handles {
            posix.register_command(n, *h);
        }
        for (k, v) in env {
            posix.set_env(k, v);
        }
        for path in files {
            posix.write_file(path, b"");
        }
        if !args.is_empty() {
            posix.set_args(args);
        }
        (posix, px_h, inst_h)
    };

    let mut ih = Host::new();
    let (iposix, ipx, iinst) = setup(&mut ih);
    let mut jh = Host::new();
    let (jposix, jpx, jinst) = setup(&mut jh);
    assert_eq!(
        (ipx, iinst),
        (jpx, jinst),
        "identical grant order → identical handles"
    );

    let m = svm_ir::resolve_imports_with(&raw, link_shim)
        .unwrap_or_else(|e| panic!("resolve imports: {e:?}\n--- IR ---\n{ir}"));
    verify_module(&m).unwrap_or_else(|e| panic!("verify failed: {e:?}\n--- IR ---\n{ir}"));
    let init = vec![0u8; win];

    // Interpreter: the shell loops to EOF and returns 0 (or `exit`s, a `Trap::Exit`). The reserved
    // window backs the command carve op 13 spawns into.
    let mut fuel = 200_000_000u64;
    match run_capture_reserved_with_host(&m, 0, &[], &mut fuel, &init, 0, &mut ih).0 {
        Ok(_) | Err(Trap::Exit(_)) => {}
        Err(e) => panic!("interp trapped: {e:?}\n--- IR ---\n{ir}"),
    }
    // JIT — given the module resolver + named-grant hooks op 13 needs.
    let (jout, _) = compile_and_run_capture_reserved_with_host_ex(
        &m,
        0,
        &[],
        &init,
        0,
        cap_thunk,
        &mut jh as *mut Host as *mut c_void,
        Some(svm_run::module_resolver),
        Some(grant_hooks()),
    )
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

/// An external command: echo every `argv[i]` on its own line, return `argc` (a non-zero status that
/// tracks the argument count, so `$?` is observable).
const CMD_ECHO: &str = r#"
long write(long fd, void *buf, long n);
static long slen(char *s){ long n=0; while(s[n]) n++; return n; }
int main(int argc, char **argv){
  for (int i = 0; i < argc; i++){ write(1, argv[i], slen(argv[i])); write(1, "\n", 1); }
  return argc;
}
"#;

/// An external command that succeeds: print `ok\n`, return `0` — so `&&`/`||` see a success status.
const CMD_OK: &str = r#"
long write(long fd, void *buf, long n);
int main(int argc, char **argv){ write(1, "ok\n", 3); return 0; }
"#;

/// STAGE1.md §5 — the real Stage-0 shell **spawns an external command**. A command name that is not a
/// builtin is looked up in the personality's PATH registry and, if found, run as a separate compiled-C
/// child via `Instantiator` op 13 + `join`: its `argv` is delivered, its stdout interleaves with the
/// shell's own output in the one shared sink, and its status threads into `$?`. An unregistered name is
/// still `<cmd>: not found` (status 127). Differential interp==JIT.
#[test]
fn stage0_shell_spawns_external_command() {
    let script = b"echo start\n\
                   say hi there\n\
                   echo rc $?\n\
                   bogus\n\
                   echo rc $?\n";
    let (iout, jout) = run_shell_ex(script, &[], &[], &[], &[("say", CMD_ECHO)]);
    assert_eq!(
        iout,
        b"start\nsay\nhi\nthere\nrc 3\nbogus: not found\nrc 127\n".as_slice(),
        "interp: builtin + spawned external (argv echoed, status = argc) + not-found, all in one sink"
    );
    assert_eq!(jout, iout, "jit: shell output must match interp");
}

/// A spawned command's status participates in `&&`/`||` short-circuiting exactly like a builtin's, and
/// the PATH registry holds more than one command. `ok` returns 0 (success); `say` returns its argc
/// (non-zero, a failure). Differential interp==JIT.
#[test]
fn stage0_shell_external_command_status_in_control_flow() {
    let script = b"ok && echo yes\n\
                   say a || echo fallback\n\
                   ok || echo skipped\n";
    let (iout, jout) = run_shell_ex(script, &[], &[], &[], &[("say", CMD_ECHO), ("ok", CMD_OK)]);
    assert_eq!(
        iout,
        b"ok\nyes\nsay\na\nfallback\nok\n".as_slice(),
        "interp: `ok`(0)&&echo → yes; `say a`(2, fail)||echo → fallback; `ok`(0)||echo → skipped"
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

/// `grep -v` inverts the match and `grep -c` prints only the count. Both read the redirected file.
#[test]
fn stage0_shell_grep_flags() {
    let (iout, jout) = run_shell(
        b"echo alpha > /f\n\
          echo beta >> /f\n\
          echo alps >> /f\n\
          grep -v al < /f\n\
          grep -c al < /f\n",
        &[],
        &[],
        &[],
    );
    // -v al → lines without "al" → "beta"; -c al → count of matching lines → 2.
    assert_eq!(
        iout, b"beta\n2\n",
        "interp: grep -v inverts, grep -c counts"
    );
    assert_eq!(jout, iout, "jit: grep-flags output must match interp");
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

/// Command sequencing: `;` runs unconditionally; `&&` runs the next only after success; `||` only
/// after failure. Short-circuiting is driven by `$?` and threaded through `run_list`.
#[test]
fn stage0_shell_sequencing_and_short_circuit() {
    let (iout, jout) = run_shell(
        b"echo a ; echo b\n\
          true && echo yes\n\
          false && echo no\n\
          false || echo fallback\n\
          true || echo skip\n\
          false && echo x || echo y\n",
        &[],
        &[],
        &[],
    );
    assert_eq!(
        iout, b"a\nb\nyes\nfallback\ny\n",
        "interp: ; always, && on success, || on failure, with chaining"
    );
    assert_eq!(jout, iout, "jit: sequencing output must match interp");
}

/// Pipelines: a multi-stage `cat FILE | grep P | wc` streams each stage's full output into the next
/// via memfs temps, and the final stage's result reaches stdout. Also checks a per-stage redirect
/// inside a pipeline (`| grep P > out`) overrides the pipe. This is the shell's process-driven core
/// (emulated in-process, no fork yet).
#[test]
fn stage0_shell_pipelines() {
    let (iout, jout) = run_shell(
        b"echo apple > /f\n\
          echo apricot >> /f\n\
          echo banana >> /f\n\
          echo cherry >> /f\n\
          cat /f | grep ap | wc\n\
          cat /f | grep ap > /hits\n\
          cat /hits\n",
        &[],
        &[],
        &[],
    );
    // grep ap → "apple\napricot\n" (2 lines, 2 words, 14 bytes); the redirected pipeline writes the
    // same two lines to /hits, surfaced by cat.
    assert_eq!(
        iout, b"2 2 14\napple\napricot\n",
        "interp: pipeline stages chain; a stage redirect overrides the pipe"
    );
    assert_eq!(jout, iout, "jit: pipeline output must match interp");
}

/// A pipeline reading real (redirected) stdin at its head: `grep b < /f | wc -l`-style chain, here
/// `cat < /f | tail -n 1` — the first stage consumes the `<` file, the last emits to stdout.
#[test]
fn stage0_shell_pipeline_from_stdin_redirect() {
    let (iout, jout) = run_shell(
        b"echo one > /f\n\
          echo two >> /f\n\
          echo three >> /f\n\
          cat < /f | tail -n 1\n",
        &[],
        &[],
        &[],
    );
    assert_eq!(
        iout, b"three\n",
        "interp: `<` feeds stage 0; tail -n 1 ends the pipe"
    );
    assert_eq!(
        jout, iout,
        "jit: pipeline-from-stdin output must match interp"
    );
}

/// Shell variables: `NAME=VALUE` sets a shell var, `$NAME` (a whole token) expands it in any argument
/// position (not just echo), a shell var shadows an environment var of the same name, and a `$NAME`
/// RHS composes. An unset variable token expands to nothing (an empty line here).
#[test]
fn stage0_shell_variables() {
    let (iout, jout) = run_shell(
        b"X=hello\n\
          echo $X world\n\
          Y=$X\n\
          echo $Y\n\
          echo $UNSET\n\
          echo $X > /vf\n\
          cat /vf\n\
          WHO=shellvar\n\
          echo $WHO\n",
        &[("WHO", "envvar")],
        &[],
        &[],
    );
    assert_eq!(
        iout, b"hello world\nhello\n\nhello\nshellvar\n",
        "interp: assignment, expansion everywhere, shadowing, unset->empty"
    );
    assert_eq!(jout, iout, "jit: variable output must match interp");
}

/// `export` promotes a shell variable into the personality environment (`setenv`). Both
/// `export NAME=VALUE` and `export NAME` (of an existing shell var) make the value observable —
/// expansion confirms it round-trips.
#[test]
fn stage0_shell_export_to_env() {
    let (iout, jout) = run_shell(
        b"export FOO=fooval\n\
          echo $FOO\n\
          BAR=barval\n\
          export BAR\n\
          echo $BAR\n",
        &[],
        &[],
        &[],
    );
    assert_eq!(
        iout, b"fooval\nbarval\n",
        "interp: export NAME=VALUE and export NAME both reach env"
    );
    assert_eq!(jout, iout, "jit: export output must match interp");
}

/// `sort` and `uniq` as a pipeline: `cat f | sort | uniq` orders the lines and collapses adjacent
/// duplicates — the canonical Unix idiom, here proving three-stage piping of real filters.
#[test]
fn stage0_shell_sort_uniq_pipeline() {
    let (iout, jout) = run_shell(
        b"echo banana > /f\n\
          echo apple >> /f\n\
          echo cherry >> /f\n\
          echo apple >> /f\n\
          echo banana >> /f\n\
          cat /f | sort | uniq\n",
        &[],
        &[],
        &[],
    );
    // sorted: apple, apple, banana, banana, cherry → uniq → apple, banana, cherry.
    assert_eq!(
        iout, b"apple\nbanana\ncherry\n",
        "interp: sort orders, uniq collapses adjacent dups, over a 3-stage pipe"
    );
    assert_eq!(jout, iout, "jit: sort/uniq output must match interp");
}

/// Globbing: `*` expands against the memfs into sorted `dir/name` matches, feeding multi-file
/// builtins (`echo`, `cat`, `rm`); a pattern with no match stays literal (nullglob-off). Exercises
/// `fnmatch_` + `glob_expand` driving `opendir`/`readdir`.
#[test]
fn stage0_shell_globbing() {
    let (iout, jout) = run_shell(
        b"echo one > /a1\n\
          echo two > /a2\n\
          echo three > /b1\n\
          echo /a*\n\
          cat /a*\n\
          echo /z*\n\
          rm /a*\n\
          cat /a1\n",
        &[],
        &[],
        &[],
    );
    // `/a*` → /a1 /a2 (sorted); cat concatenates both; `/z*` has no match so stays literal; rm /a*
    // removes both, so the final cat misses.
    assert_eq!(
        iout, b"/a1 /a2\none\ntwo\n/z*\n/a1: not found\n",
        "interp: glob expands, feeds cat/rm, and is literal on no match"
    );
    assert_eq!(jout, iout, "jit: globbing output must match interp");
}

/// Single-line `if/then/else/fi`: the condition's exit status picks the branch, both the taken and
/// not-taken branches behave, and multiple body commands run. Uses `test -f` over the memfs and a
/// multi-command then-body.
#[test]
fn stage0_shell_if_then_else() {
    let (iout, jout) = run_shell(
        b"echo hi > /f\n\
          if test -f /f; then echo present; echo again; else echo absent; fi\n\
          if test -f /nope; then echo present; else echo absent; fi\n\
          if false; then echo t; fi\n\
          if true; then echo taken; fi\n",
        &[],
        &[],
        &[],
    );
    assert_eq!(
        iout, b"present\nagain\nabsent\ntaken\n",
        "interp: if picks the branch by $?, runs multi-command bodies, no-else is a no-op"
    );
    assert_eq!(jout, iout, "jit: if/then/else output must match interp");
}

/// `if` composes with the rest of the shell: the condition can be a pipeline (`grep` sets the status)
/// and a body command can redirect. Proves `run_if` delegates each part back through `run_list`.
#[test]
fn stage0_shell_if_with_pipeline_condition() {
    let (iout, jout) = run_shell(
        b"echo apple > /f\n\
          echo banana >> /f\n\
          if cat /f | grep ban; then echo found > /r; else echo missing > /r; fi\n\
          cat /r\n",
        &[],
        &[],
        &[],
    );
    // grep prints its match (to stdout) and succeeds, so the then-branch writes "found" to /r.
    assert_eq!(
        iout, b"banana\nfound\n",
        "interp: pipeline condition drives if; redirected body writes the result"
    );
    assert_eq!(jout, iout, "jit: if-with-pipeline output must match interp");
}
