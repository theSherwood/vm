//! Stage 1 (STAGE1.md) — an **unmodified `int main(int argc, char **argv)` compiled by chibicc**,
//! run with a real `argv` and ambient stdio. This is the "as close to native as the security model
//! allows" milestone: the C source is ordinary (no capability threading, `write(1, …)` to an ambient
//! fd), and chibicc's synthesized `_start` parses the §3e powerbox args buffer into `argv[]` and calls
//! `main(argc, argv)` — the crt a real host provides. Argv delivery varies the output, and the exit
//! code is `argc`, checked on both backends (the powerbox seeds the args buffer identically).
//!
//! Gated `#![cfg(unix)]` (needs the chibicc toolchain, like `c_frontend.rs`).
#![cfg(unix)]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use svm_run::{Backend, Outcome, RunConfig, Value};

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
            .expect("run `make` to build the chibicc fork");
        assert!(status.success(), "chibicc build failed");
        dir.join("chibicc")
    })
    .as_path()
}

fn c_to_ir(src: &str) -> String {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static N: AtomicUsize = AtomicUsize::new(0);
    let id = N.fetch_add(1, Ordering::Relaxed);
    let base = std::env::temp_dir().join(format!("svm_argvmain_{}_{id}", std::process::id()));
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

/// An unmodified C program: echo each `argv[i]` on its own line, return `argc`. Uses `write(1, …)` —
/// ambient stdout, no capability threading in the source.
const PROG: &str = r#"
long write(long fd, void *buf, long n);
static long slen(char *s){ long n=0; while(s[n]) n++; return n; }
/* A writable global (initialized data): forces the globals-shift path — it must land above the args
   buffer [128, 16384), not collide with seeded argv. If the shift were wrong, argv would corrupt it. */
static char tag[4] = "ok:";
int main(int argc, char **argv){
  write(1, tag, slen(tag));
  write(1, "\n", 1);
  for (int i = 0; i < argc; i++){ write(1, argv[i], slen(argv[i])); write(1, "\n", 1); }
  return argc;
}
"#;

/// Compile the program and run it on `backend` with `args`; return (exit code, stdout).
fn run(backend: Backend, args: &[&[u8]]) -> (i64, Vec<u8>) {
    let ir = c_to_ir(PROG);
    let module = svm_text::parse_module(&ir).expect("parse chibicc IR");
    let inst = svm_run::instantiate(module).expect("instantiate (resolve + verify)");
    let cfg = RunConfig {
        limits: Default::default(),
        stdin: Vec::new(),
        memory_size_log2: None,
        args: args.iter().map(|s| s.to_vec()).collect(),
        env: Vec::new(),
    };
    let run = inst.run(backend, &cfg).expect("run");
    let code = match run.outcome {
        Outcome::Exited(c) => c as i64,
        Outcome::Returned(ref vals) => match vals.first() {
            Some(Value::I32(x)) => *x as i64,
            Some(Value::I64(x)) => *x,
            _ => 0,
        },
    };
    (code, run.stdout)
}

/// The chibicc-compiled `main(int, char**)` receives its argv and echoes it, identically on both
/// backends; the exit code is `argc`. Output varies with argv (real delivery, not a constant).
#[test]
fn argv_main_echoes_args_and_returns_argc() {
    for args in [
        &[b"prog".as_slice()][..],
        &[b"myprog".as_slice(), b"hello", b"world"][..],
    ] {
        let (ic, iout) = run(Backend::TreeWalk, args);
        let (jc, jout) = run(Backend::Jit, args);
        // The writable global "ok:" prints first (uncorrupted by the seeded argv), then each argv line.
        let mut expect: Vec<u8> = b"ok:\n".to_vec();
        expect.extend(args.iter().flat_map(|a| [a, b"\n".as_slice()].concat()));
        assert_eq!(ic, args.len() as i64, "interp: exit code = argc");
        assert_eq!(iout, expect, "interp: echoed argv");
        assert_eq!(jc, ic, "jit: exit code matches interp");
        assert_eq!(jout, iout, "jit: stdout matches interp");
    }
}
