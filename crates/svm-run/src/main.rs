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

use std::hash::{BuildHasher, RandomState};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};
use std::{env, fs, process};

use svm_ir::Module;
use svm_run::{
    is_powerbox_entry, run_kernel, run_powerbox_with_deadline_and_quota, Outcome, Quota, Value,
};
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
             \n  the chibicc frontend ($SVM_CHIBICC or the in-repo build).\n\
             \n  env: SVM_DEADLINE_MS (kill a runaway guest after N ms),\n\
             \n       SVM_MAX_FIBERS / SVM_MAX_VCPUS (§15 spawn quotas — kill a fiber/thread bomb)."
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
    // §7: lower any named capability imports to concrete `cap.call`s under the reference host
    // policy *before* verification (fail-closed on an unknown name). A no-op for modules that
    // inline their capability calls (the legacy form).
    let module = svm_run::resolve_capability_imports(module)?;
    verify_module(&module).map_err(|e| format!("verification failed (fail-closed): {e:?}"))?;

    if is_powerbox_entry(&module) {
        // §5 kill-path: `SVM_DEADLINE_MS` (CLI policy) bounds a possibly-runaway guest so it is
        // detect-and-killed after the deadline instead of hanging the process; unset ⇒ unbounded.
        let deadline = std::env::var("SVM_DEADLINE_MS")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .filter(|&ms| ms > 0)
            .map(std::time::Duration::from_millis);
        // §15 spawn quota (CLI policy): `SVM_MAX_FIBERS`/`SVM_MAX_VCPUS` cap fiber/vCPU spawning so a
        // spawn-bomb is detect-and-killed; unset ⇒ the default anti-bomb ceilings.
        let env_usize = |k: &str, dflt: usize| {
            std::env::var(k)
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .filter(|&n| n > 0)
                .unwrap_or(dflt)
        };
        let dq = Quota::default();
        let quota = Quota {
            max_fibers: env_usize("SVM_MAX_FIBERS", dq.max_fibers),
            max_vcpus: env_usize("SVM_MAX_VCPUS", dq.max_vcpus),
        };
        let run = run_powerbox_with_deadline_and_quota(&module, &stdin, deadline, quota)?;
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
    let ir_out = fresh_temp_ir()?;
    let result = (|| {
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
    })();
    // Don't leak the intermediate IR file regardless of outcome.
    let _ = fs::remove_file(&ir_out);
    result
}

/// Create a fresh, **unpredictably-named** temp file for the frontend's IR output and return its path.
/// The previous `svm_run_<pid>.svm` name was predictable, so a local attacker could pre-plant a symlink
/// there and redirect chibicc's write. The name now carries an OS-RNG-seeded suffix (via `RandomState`,
/// whose seed is process-secret), and the file is created with `create_new` (`O_EXCL`), which fails
/// rather than following a pre-existing path. `$SVM_CHIBICC`/`.c` compilation is a trusted-operator
/// path (the binary it runs is operator-chosen); this just removes the predictable-temp footgun.
fn fresh_temp_ir() -> Result<PathBuf, String> {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let rnd = RandomState::new().hash_one((process::id(), nanos));
    let out = env::temp_dir().join(format!("svm_run_{rnd:016x}.svm"));
    fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&out)
        .map_err(|e| format!("create temp IR file: {e}"))?;
    Ok(out)
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
