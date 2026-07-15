//! End-to-end C linking against the **POSIX personality** (`svm-posix`) through §7 named imports.
//!
//! This is POSIX.md §6 step 2 — the "real linking path" — but with *real compiled C* instead of
//! hand-written IR. A tiny guest libc **shim** (guest code) declares each libc call as an undefined
//! `extern` whose first argument is a capability handle; chibicc lowers those to `call.import
//! "<name>"` on the handle (the generic §7 capability-import convention, `gen_builtin_import`). A
//! resolver binds each name to the personality's `(HOST_FN, op)`, and `resolve_imports` lowers every
//! `call.import` to a `cap.call` on the shared personality handle — exactly what a linker does for a
//! shell's libc imports, now driven by the frontend end to end.
//!
//! The personality handle is granted into powerbox **slot 7** (the guest-`Jit` slot this test does
//! not use), so the shim fetches it with `__vm_cap(7)`. Wiring the personality into a first-class
//! powerbox slot (a durable ABI decision) is a follow-up (POSIX.md §6); this proves the linking path
//! with zero frontend/runner change.
//!
//! Each program runs `_start` (function 0) on **both** the interpreter and the JIT under an identical
//! host, asserting they agree on the result *and* the observable personality state (captured stdout,
//! the memfs) — so it doubles as a cross-backend differential, capability effects included. The
//! personality's `HostFn` dispatches through the same `cap_dispatch_slots` the JIT's `cap.call` thunk
//! calls, so parity comes for free. Requires a unix C toolchain (`make` + `cc`) to build the chibicc
//! fork, so the suite is gated to `#![cfg(unix)]` (like `c_frontend.rs`).
#![cfg(unix)]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use core::ffi::c_void;
use svm_interp::{iface, run_with_host, Host, StreamRole, Value};
use svm_ir::ResolvedCap;
use svm_jit::{compile_and_run_with_host, JitOutcome};
use svm_posix::Posix;
use svm_run::cap_thunk;
use svm_text::parse_module as parse_module_raw;
use svm_verify::verify_module;

/// Powerbox slot the personality handle is stashed in (`__vm_cap(7)` in the shim). Slot 7 is the
/// guest-`Jit` handle in the standard powerbox, which none of these programs use.
const POSIX_SLOT: i32 = 7;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}

/// Build the chibicc fork once per test binary, returning the path to its binary.
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

/// Compile a C source string to our text IR via the frontend.
fn c_to_ir(src: &str) -> String {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static N: AtomicUsize = AtomicUsize::new(0);
    let id = N.fetch_add(1, Ordering::Relaxed);
    let base = std::env::temp_dir().join(format!("svm_cposix_{}_{id}", std::process::id()));
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

/// The §7 resolver binding the shim's libc-shaped import names to the POSIX personality's
/// `(HOST_FN, op)`. The names are the raw libc calls the shim declares as capability externs; each
/// binds to the matching `svm_posix::OP_*`. Unknown names fail closed (a shim/personality mismatch).
fn resolve(name: &str) -> Option<ResolvedCap> {
    let op = match name {
        "malloc" => svm_posix::OP_MALLOC,
        "open" => svm_posix::OP_OPEN,
        "px_write" => svm_posix::OP_WRITE,
        "px_read" => svm_posix::OP_READ,
        "lseek" => svm_posix::OP_LSEEK,
        _ => return None,
    };
    Some(ResolvedCap {
        type_id: iface::HOST_FN,
        op,
    })
}

/// Grant the fixed 8-handle powerbox on `host`, with the POSIX personality at [`POSIX_SLOT`] and a
/// window-heap region in the upper half of the guest window (clear of chibicc's low data image +
/// data stack). Returns the entry args and a [`Posix`] handle to the personality's captured state.
fn setup(host: &mut Host, win: u64) -> ([Value; 8], Posix) {
    host.set_region_factory(svm_run::new_shared_region);
    host.set_jit_validator(svm_run::jit_blob_validator);
    let (px, posix) = svm_posix::grant(host, win / 2, win, Vec::new());
    let mut args = [
        Value::I32(host.grant_stream(StreamRole::Out)),
        Value::I32(host.grant_stream(StreamRole::In)),
        Value::I32(host.grant_exit()),
        Value::I32(host.grant_memory()),
        Value::I32(host.grant_address_space(0, win)),
        Value::I32(host.grant_io_ring()),
        Value::I32(host.grant_blocking(std::time::Duration::ZERO, None)),
        Value::I32(0), // POSIX_SLOT, filled below
    ];
    args[POSIX_SLOT as usize] = Value::I32(px);
    (args, posix)
}

/// What a program did on one backend: `main`'s returned result values, plus the personality's
/// captured stdout and the memfs contents of file `"f"`.
struct Effects {
    result: Vec<Value>,
    stdout: Vec<u8>,
    file_f: Option<Vec<u8>>,
}

fn to_slot(v: Value) -> i64 {
    match v {
        Value::I32(x) => x as i64,
        Value::I64(x) => x,
        Value::Ref(x) => x as i64,
        other => panic!("unexpected entry-arg value {other:?}"),
    }
}

/// Compile + resolve (through [`resolve`]) + verify a C program, then run `_start` on **both**
/// backends under identical personalities and return each backend's observable effects for the
/// caller to compare. Panics with the IR on a parse/verify/trap so failures are legible.
fn run_both(src: &str) -> (Effects, Effects) {
    let ir = c_to_ir(src);
    let raw = parse_module_raw(&ir)
        .unwrap_or_else(|e| panic!("parse IR failed: {e:?}\n--- IR ---\n{ir}"));
    let m = svm_ir::resolve_imports(&raw, resolve)
        .unwrap_or_else(|e| panic!("resolve imports: {e:?}\n--- IR ---\n{ir}"));
    verify_module(&m).unwrap_or_else(|e| panic!("verify failed: {e:?}\n--- IR ---\n{ir}"));
    let win = 1u64 << m.memory.expect("the frontend declares a window").size_log2;

    // Interpreter.
    let mut ih = Host::new();
    let (iargs, iposix) = setup(&mut ih, win);
    let mut fuel = 50_000_000u64;
    let ires = run_with_host(&m, 0, &iargs, &mut fuel, &mut ih)
        .unwrap_or_else(|e| panic!("interp trapped: {e:?}\n--- IR ---\n{ir}"));
    let interp = Effects {
        result: ires,
        stdout: iposix.stdout(),
        file_f: iposix.read_file("f"),
    };

    // JIT.
    let mut jh = Host::new();
    let (jargs, jposix) = setup(&mut jh, win);
    let slots: Vec<i64> = jargs.iter().copied().map(to_slot).collect();
    let jout = compile_and_run_with_host(
        &m,
        0,
        &slots,
        cap_thunk,
        &mut jh as *mut Host as *mut c_void,
    )
    .expect("jit compiles");
    let jresult = match jout {
        JitOutcome::Returned(s) => s.iter().map(|&x| Value::I64(x)).collect(),
        other => panic!("jit did not return normally: {other:?}\n--- IR ---\n{ir}"),
    };
    let jit = Effects {
        result: jresult,
        stdout: jposix.stdout(),
        file_f: jposix.read_file("f"),
    };

    (interp, jit)
}

/// A tiny guest libc shim (guest code) binding C's libc calls to the POSIX personality. Each `extern`
/// takes the personality **handle** as its first argument (the §7 generic-import convention), which
/// `main` fetches once with `__vm_cap(POSIX_SLOT)`. The wrappers adapt C's NUL-terminated `char*`
/// convention to the personality's explicit-length `(ptr, len)` ABI (POSIX.md §4) — the adaptation is
/// guest code. `write` is a chibicc *builtin* (it lowers to a Stream call, not an import), so the
/// personality's write is imported under the distinct name `px_write`; `open`/`read`(as `px_read`)/
/// `lseek`/`malloc` are ordinary undefined externs the resolver binds.
const SHIM: &str = r#"
int __vm_cap(int i);
long malloc(int h, long size);
long open(int h, long path, long len, long flags);
long px_write(int h, long fd, long buf, long len);
long px_read(int h, long fd, long buf, long len);
long lseek(int h, long fd, long off, long whence);

static long slen(char *s) { long n = 0; while (s[n]) n = n + 1; return n; }
"#;

/// The full round-trip: `malloc` a buffer, write it to the personality's **stdout** (fd 1), `open`
/// a memfs file and write the same bytes there, then `lseek` to 0 and `read` them back into a second
/// buffer and echo *that* to stdout. Proves malloc, fd routing (stdout vs a file fd), open, write,
/// lseek, and read all reach the personality from compiled C — identically on both backends.
#[test]
fn c_links_libc_to_posix_personality_roundtrip() {
    let src = format!(
        "{SHIM}\n\
int main() {{\n\
  int h = __vm_cap({POSIX_SLOT});\n\
  char *msg = \"hi\\n\";\n\
  long n = slen(msg);\n\
  char *buf = (char *)malloc(h, 32);\n\
  for (long i = 0; i < n; i = i + 1) buf[i] = msg[i];\n\
  px_write(h, 1, (long)buf, n);            /* fd 1 -> captured stdout */\n\
  long fd = open(h, (long)\"f\", 1, 66);   /* O_CREAT|O_RDWR */\n\
  px_write(h, fd, (long)buf, n);           /* -> memfs file \"f\" */\n\
  lseek(h, fd, 0, 0);                      /* SEEK_SET 0 */\n\
  char *buf2 = (char *)malloc(h, 32);\n\
  long r = px_read(h, fd, (long)buf2, 32); /* read the file back */\n\
  px_write(h, 1, (long)buf2, r);           /* echo it to stdout again */\n\
  return (int)fd;                          /* the first file fd is 3 */\n\
}}\n"
    );
    let (interp, jit) = run_both(&src);

    // Interpreter reference: first file fd is 3; stdout got "hi\n" twice; the memfs file holds "hi\n".
    assert_eq!(
        interp.result,
        vec![Value::I32(3)],
        "interp: main returns fd 3"
    );
    assert_eq!(interp.stdout, b"hi\nhi\n", "interp: two writes to stdout");
    assert_eq!(
        interp.file_f.as_deref(),
        Some(&b"hi\n"[..]),
        "interp: the memfs file was written"
    );

    // JIT parity — same personality path, so identical result + effects (result slots are i64).
    assert_eq!(jit.result, vec![Value::I64(3)], "jit: fd must match interp");
    assert_eq!(jit.stdout, interp.stdout, "jit: stdout must match interp");
    assert_eq!(jit.file_f, interp.file_f, "jit: memfs must match interp");
}
