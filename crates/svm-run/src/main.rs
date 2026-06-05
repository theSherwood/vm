//! `svm-run` — run a guest program in the sandbox from the command line.
//!
//! ```text
//! svm-run <file> [--stdin FILE]
//! ```
//!
//! `<file>` is `.svm` (text IR), `.svmb` (binary), or `.c` (C source — compiled through the
//! chibicc frontend, located via `$SVM_CHIBICC` or the in-repo build). The module is verified,
//! then run on the Cranelift JIT under the MVP powerbox (§3e): bytes it writes to `stdout`/
//! `stderr` go to the real streams, and it terminates with the guest's exit code (`Exit(code)`
//! or `main`'s return value). A bare kernel (a non-powerbox entry) is run with zero args and its
//! result printed. A guest that traps is detect-and-killed (§5) and reported on stderr.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::{env, fs, process};

use svm_ir::Module;
use svm_run::{is_powerbox_entry, run_kernel, run_powerbox, Outcome, Value};
use svm_verify::verify_module;

fn main() {
    if let Err(e) = try_main() {
        eprintln!("svm-run: {e}");
        process::exit(1);
    }
}

fn try_main() -> Result<(), String> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() || args.iter().any(|a| a == "-h" || a == "--help") {
        eprintln!(
            "usage: svm-run <file.svm|.svmb|.c> [--stdin FILE]\n\
             \n  Verifies the module, then runs it sandboxed on the JIT under the MVP powerbox\n\
             \n  (stdout/stderr → real streams, exit code = the guest's). `.c` is compiled via\n\
             \n  the chibicc frontend ($SVM_CHIBICC or the in-repo build)."
        );
        return Err("no input file".into());
    }
    let mut file: Option<String> = None;
    let mut stdin_path: Option<String> = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--stdin" => {
                stdin_path = Some(it.next().ok_or("--stdin needs a file argument")?.clone())
            }
            _ if a.starts_with('-') => return Err(format!("unknown flag `{a}`")),
            _ => {
                if file.replace(a.clone()).is_some() {
                    return Err("more than one input file given".into());
                }
            }
        }
    }
    let file = file.ok_or("no input file")?;
    let stdin = match stdin_path {
        Some(p) => fs::read(&p).map_err(|e| format!("read --stdin file `{p}`: {e}"))?,
        None => Vec::new(),
    };

    let module = load_module(Path::new(&file))?;
    verify_module(&module).map_err(|e| format!("verification failed (fail-closed): {e:?}"))?;

    if is_powerbox_entry(&module) {
        let run = run_powerbox(&module, &stdin)?;
        // Flush captured output to the real streams (process::exit skips destructors, so flush
        // explicitly), then terminate with the guest's exit code.
        let mut out = std::io::stdout().lock();
        out.write_all(&run.stdout).ok();
        out.flush().ok();
        let mut err = std::io::stderr().lock();
        err.write_all(&run.stderr).ok();
        err.flush().ok();
        process::exit(exit_code(&run.outcome));
    } else {
        // A bare kernel: run entry 0 with zero args and print its result values.
        let args = vec![0i64; module.funcs[0].params.len()];
        let results = run_kernel(&module, &args)?;
        println!("{results:?}");
        Ok(())
    }
}

/// The process exit code for a finished program: the `Exit(code)`, or `main`'s integer return
/// value (C convention), or 0.
fn exit_code(outcome: &Outcome) -> i32 {
    match outcome {
        Outcome::Exited(c) => *c,
        Outcome::Returned(v) => match v.first() {
            Some(Value::I32(n)) => *n,
            Some(Value::I64(n)) => *n as i32,
            _ => 0,
        },
    }
}

/// Load a module by file extension: `.svm`/`.ir` text IR, `.svmb`/`.bin` binary, or `.c` C
/// source (compiled through the frontend).
fn load_module(path: &Path) -> Result<Module, String> {
    match path.extension().and_then(|e| e.to_str()).unwrap_or("") {
        "svm" | "ir" | "txt" => {
            let text =
                fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
            svm_text::parse_module(&text).map_err(|e| format!("parse text IR: {e:?}"))
        }
        "svmb" | "bin" => {
            let bytes = fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
            svm_encode::decode_module(&bytes).map_err(|e| format!("decode binary module: {e:?}"))
        }
        "c" => {
            let ir = compile_c(path)?;
            svm_text::parse_module(&ir).map_err(|e| format!("parse frontend IR: {e:?}"))
        }
        other => Err(format!(
            "unknown extension `.{other}` — expected .svm, .svmb, or .c"
        )),
    }
}

/// Compile a C source file to text IR through the chibicc frontend (the same `--emit-ir`
/// backend the tests use). The guest is *untrusted for escape* (§2a) — the verifier re-checks
/// whatever the frontend emits.
fn compile_c(path: &Path) -> Result<String, String> {
    let chibicc = locate_chibicc()?;
    let ir_out = env::temp_dir().join(format!("svm_run_{}.svm", process::id()));
    let ok = Command::new(&chibicc)
        .args([
            "-cc1",
            "--emit-ir",
            "-cc1-input",
            path.to_str().ok_or("non-UTF-8 input path")?,
            "-cc1-output",
            ir_out.to_str().ok_or("non-UTF-8 temp path")?,
            path.to_str().unwrap(),
        ])
        .status()
        .map_err(|e| format!("run chibicc ({}): {e}", chibicc.display()))?
        .success();
    if !ok {
        return Err("chibicc failed to compile the C source".into());
    }
    fs::read_to_string(&ir_out).map_err(|e| format!("read frontend IR: {e}"))
}

/// Find the chibicc binary: `$SVM_CHIBICC`, else the in-repo `frontend/chibicc/chibicc` (built
/// on demand via `make`, when `svm-run` is run from its source tree).
fn locate_chibicc() -> Result<PathBuf, String> {
    if let Ok(p) = env::var("SVM_CHIBICC") {
        return Ok(PathBuf::from(p));
    }
    // `CARGO_MANIFEST_DIR` is `<repo>/crates/svm-run`; the frontend is `<repo>/frontend/chibicc`.
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .ok_or("cannot locate the repo root")?
        .join("frontend/chibicc");
    let _ = Command::new("make").arg("-s").current_dir(&dir).status();
    let bin = dir.join("chibicc");
    if bin.exists() {
        Ok(bin)
    } else {
        Err("cannot find chibicc — set $SVM_CHIBICC to the frontend binary, or pass a .svm/.svmb file".into())
    }
}
