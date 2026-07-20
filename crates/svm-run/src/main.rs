//! `svm-run` — run a guest program in the sandbox from the command line.
//!
//! ```text
//! svm-run <file> [--stdin FILE] [-- <guest args>]
//! ```
//!
//! `<file>` is `.svm` (text IR), `.svmb` (binary), or `.c` (C source — compiled through the
//! chibicc frontend, located via `$SVM_CHIBICC` or the in-repo build). The module is verified,
//! then run on the Cranelift JIT under the MVP powerbox (§3e): bytes it writes to `stdout`/
//! `stderr` go to the real streams, and it terminates with the guest's exit code (`Exit(code)`
//! or `main`'s return value). A bare kernel (a non-powerbox entry) is run with zero args and its
//! result printed. A guest that traps is detect-and-killed (§5) and reported on stderr.
//!
//! `--specialize` instead partial-evaluates the entry (§20c first Futamura projection): bind some
//! parameters to constants (`--arg i64:N`) and/or declare window bytes constant (`--const-region`),
//! and the residual — re-verified — is written as a binary artifact (`-o`), printed as text IR
//! (`--emit-text`), or run as a kernel. See `--help`.

use std::hash::{BuildHasher, RandomState};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};
use std::{env, fs, process};

use svm_ir::Module;
use svm_run::{
    is_named_powerbox_entry, run_kernel, run_powerbox_with_args_and_limits, specialize_module,
    Outcome, Quota, SpecArg, SpecializeOpts, Value,
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
        print_usage();
        return Err("no input file".into());
    }
    let mut file: Option<String> = None;
    let mut stdin_path: Option<String> = None;
    // Everything after a `--` is the guest's own argument vector (the §3e args buffer): it becomes
    // `argv[1..]`, with `argv[0]` set to the input file name — so a `main(int, char**)` program sees
    // them exactly as a native invocation would.
    let mut guest_args: Vec<String> = Vec::new();
    // `--specialize` (§20c partial evaluation) options.
    let mut specialize = false;
    let mut func: u32 = 0;
    let mut spec_args: Vec<SpecArg> = Vec::new();
    let mut const_regions: Vec<(u64, u64)> = Vec::new();
    let mut rename: Option<(u64, u64)> = None;
    let mut rename_private = false;
    let mut optimize = true;
    let mut outline = false;
    let mut selective_outline = false;
    let mut out_path: Option<String> = None;
    let mut emit_text = false;
    let mut run_args: Vec<i64> = Vec::new();

    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--" => {
                guest_args.extend(it.by_ref().cloned());
                break;
            }
            "--stdin" => {
                stdin_path = Some(it.next().ok_or("--stdin needs a file argument")?.clone())
            }
            "--specialize" => specialize = true,
            "--func" => func = parse_u64v(it.next().ok_or("--func needs an index")?)? as u32,
            "--arg" => spec_args.push(parse_arg(it.next().ok_or("--arg needs a binding")?)?),
            "--const-region" => const_regions.push(parse_region(
                it.next().ok_or("--const-region needs lo:hi")?,
            )?),
            "--rename" => rename = Some(parse_region(it.next().ok_or("--rename needs lo:hi")?)?),
            "--rename-private" => rename_private = true,
            "--outline" => outline = true,
            "--selective" => selective_outline = true,
            "--no-optimize" => optimize = false,
            "-o" | "--out" => out_path = Some(it.next().ok_or("-o needs a file argument")?.clone()),
            "--emit-text" => emit_text = true,
            "--run-args" => {
                for p in it
                    .next()
                    .ok_or("--run-args needs a comma-separated list")?
                    .split(',')
                    .filter(|p| !p.is_empty())
                {
                    run_args.push(parse_i64v(p)?);
                }
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
    // IMPORTS.md phase 4: the runtime never rewrites. A powerbox entry keeps its manifest —
    // `run_powerbox` binds each slot at instantiation and `call.import` dispatches through the
    // bindings. A module that declares imports without the powerbox entry shape cannot run.
    if !module.imports.is_empty() && !svm_run::is_named_powerbox_entry(&module) {
        return Err(
            "module declares imports but has no powerbox entry (paramless exported `_start`) — \
             the runtime binds manifest slots, it does not rewrite (IMPORTS.md phase 4)"
                .into(),
        );
    }
    verify_module(&module).map_err(|e| format!("verification failed (fail-closed): {e:?}"))?;

    if specialize {
        return run_specialize(
            &module,
            func,
            spec_args,
            const_regions,
            rename,
            rename_private,
            optimize,
            outline,
            selective_outline,
            out_path,
            emit_text,
            run_args,
        );
    }

    if is_named_powerbox_entry(&module) {
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
        // The guest's `argv`: when the user passes `-- <args>`, the input file name is `argv[0]`
        // and the post-`--` tokens follow. When *no* `--` args are given, pass an empty vector so the
        // runner seeds **no** args buffer (`init_mem` stays `None`, byte-identical to a bare run) —
        // a program that wants `argc>=1` must be invoked with `--`. (Seeding the window for every run
        // is both pointless for `main(void)` programs and an unnecessary perturbation of the guest's
        // initial state.) The environment is deliberately empty — no ambient host env leaks in (§3e/§7).
        let argv: Vec<&[u8]> = if guest_args.is_empty() {
            Vec::new()
        } else {
            std::iter::once(file.as_bytes())
                .chain(guest_args.iter().map(|s| s.as_bytes()))
                .collect()
        };
        let run = run_powerbox_with_args_and_limits(&module, &stdin, &argv, &[], deadline, quota)?;
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

/// The `--specialize` path: partial-evaluate the entry against the declared static/dynamic binding
/// and constant-memory config (§20c), then emit the residual as a binary artifact (`-o`), print it
/// as text (`--emit-text`), or run it as a kernel and print the results.
#[allow(clippy::too_many_arguments)]
fn run_specialize(
    module: &Module,
    func: u32,
    spec_args: Vec<SpecArg>,
    const_regions: Vec<(u64, u64)>,
    rename: Option<(u64, u64)>,
    rename_private: bool,
    optimize: bool,
    outline: bool,
    selective_outline: bool,
    out_path: Option<String>,
    emit_text: bool,
    mut run_args: Vec<i64>,
) -> Result<(), String> {
    // `specialize_module` validates the arity and pads missing bindings with `Dynamic`.
    let opts = SpecializeOpts {
        func,
        args: spec_args,
        const_regions,
        rename,
        rename_private,
        optimize,
        outline,
        selective_outline,
    };
    let residual = specialize_module(module, &opts)?;

    if let Some(path) = out_path {
        let bytes = svm_encode::encode_module(&residual);
        fs::write(&path, &bytes).map_err(|e| format!("write {path}: {e}"))?;
        eprintln!(
            "svm-run: wrote specialized residual to {path} ({} bytes, {} block(s))",
            bytes.len(),
            residual.funcs[0].blocks.len()
        );
        Ok(())
    } else if emit_text {
        print!("{}", svm_text::print_module(&residual));
        Ok(())
    } else {
        // Run the residual as a kernel. Its parameters are the dynamic args, in order.
        let want = residual.funcs[0].params.len();
        if run_args.is_empty() {
            run_args = vec![0i64; want];
        }
        if run_args.len() != want {
            return Err(format!(
                "--run-args has {} value(s) but the residual takes {want} dynamic argument(s)",
                run_args.len()
            ));
        }
        let results = run_kernel(&residual, &run_args)?;
        println!("{results:?}");
        Ok(())
    }
}

/// Parse an unsigned address/size (decimal, or `0x`-prefixed hex).
fn parse_u64v(s: &str) -> Result<u64, String> {
    let s = s.trim();
    let r = if let Some(h) = s.strip_prefix("0x") {
        u64::from_str_radix(h, 16)
    } else {
        s.parse::<u64>()
    };
    r.map_err(|_| format!("invalid number `{s}`"))
}

/// Parse a signed 64-bit value (decimal with optional `-`, or `0x` hex bit-pattern).
fn parse_i64v(s: &str) -> Result<i64, String> {
    let s = s.trim();
    let r = if let Some(h) = s.strip_prefix("0x") {
        u64::from_str_radix(h, 16).map(|u| u as i64)
    } else {
        s.parse::<i64>()
    };
    r.map_err(|_| format!("invalid integer `{s}`"))
}

/// Parse a `lo:hi` window range.
fn parse_region(s: &str) -> Result<(u64, u64), String> {
    let (lo, hi) = s
        .split_once(':')
        .ok_or(format!("region `{s}` must be lo:hi"))?;
    let (lo, hi) = (parse_u64v(lo)?, parse_u64v(hi)?);
    if lo >= hi {
        return Err(format!("region `{s}`: lo must be < hi"));
    }
    Ok((lo, hi))
}

/// Parse one `--arg` parameter binding: `dyn`, `i32:N`, or `i64:N`.
fn parse_arg(s: &str) -> Result<SpecArg, String> {
    if s == "dyn" {
        return Ok(SpecArg::Dynamic);
    }
    let (ty, v) = s
        .split_once(':')
        .ok_or(format!("--arg `{s}` must be `dyn`, `i32:N`, or `i64:N`"))?;
    match ty {
        "i32" => Ok(SpecArg::ConstI32(parse_i64v(v)? as i32)),
        "i64" => Ok(SpecArg::ConstI64(parse_i64v(v)?)),
        _ => Err(format!("--arg type `{ty}` must be i32 or i64")),
    }
}

fn print_usage() {
    eprintln!(
        "usage: svm-run <file.svm|.svmb|.c> [--stdin FILE] [-- <guest args>]\n\
         \n  Verify a module, then run it sandboxed on the JIT under the MVP powerbox\n\
         \n  (stdout/stderr → real streams, exit code = the guest's). `.c` is compiled via\n\
         \n  the chibicc frontend ($SVM_CHIBICC or the in-repo build). Arguments after `--`\n\
         \n  are passed to the guest as argv[1..] (argv[0] = the file name; empty environment).\n\
         \n  env: SVM_DEADLINE_MS (kill a runaway guest after N ms),\n\
         \n       SVM_MAX_FIBERS / SVM_MAX_VCPUS (§15 spawn quotas — kill a fiber/thread bomb).\n\
         \n\
         \nspecialize (§20c first Futamura projection): turn an interpreter + a fixed program into\n\
         \nthe compiled residual.\n\
         \n  svm-run <file> --specialize [--func N] [--arg BIND]... [--const-region lo:hi]...\n\
         \n                 [--rename lo:hi] [--rename-private] [--outline] [--selective]\n\
         \n                 [--no-optimize]\n\
         \n                 [-o OUT.svmb | --emit-text | --run-args v,v,...]\n\
         \n  --arg BIND   per-parameter binding in order: `dyn`, `i32:N`, or `i64:N`\n\
         \n               (parameters without a binding default to `dyn`)\n\
         \n  --const-region lo:hi   promise window bytes [lo,hi) are constant at spec time\n\
         \n  --rename lo:hi         lift a private value-stack/locals range into SSA (Stage 2)\n\
         \n  --outline    specialize calls into shared residual functions (multi-function residual)\n\
         \n               instead of inlining — bounds size; specializes dynamic-depth recursion\n\
         \n  --selective  inline leaves/structure, outline only recursion back-edges (a tight\n\
         \n               recursive residual rather than one function per call site)\n\
         \n  -o OUT.svmb  write the (re-verified) residual as a binary artifact; else --emit-text\n\
         \n               prints it as text IR, else it is run as a kernel and its results printed.\n\
         \n  lo/hi/N accept decimal or 0x-hex."
    );
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
    let dir = fresh_temp_dir()?;
    let ir_out = dir.join("out.svm");
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
    // Don't leak the intermediate dir (and its IR file) regardless of outcome.
    let _ = fs::remove_dir_all(&dir);
    result
}

/// Create a fresh, **owner-only** temp *directory* to hold the frontend's IR output, returning its
/// path. chibicc's `-cc1-output` (and our read-back) reopen the IR **by path**, which follows
/// symlinks — so an unpredictable filename + `O_EXCL` on the *file* still left a swap window between
/// our create and chibicc's reopen. Containing the IR in a private directory closes it: the name is
/// OS-RNG-seeded (`RandomState`, process-secret seed), the directory is created **non-recursively**
/// (`mkdir(2)` — which returns `EEXIST` rather than following a pre-planted symlink at that path),
/// and on unix it is mode `0700`, so a different-uid attacker cannot traverse it to place or swap
/// `out.svm`. `$SVM_CHIBICC`/`.c` compilation is a trusted-operator path (the binary it runs is
/// operator-chosen); this removes the residual reopen-by-path TOCTOU on top of that.
fn fresh_temp_dir() -> Result<PathBuf, String> {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let rnd = RandomState::new().hash_one((process::id(), nanos));
    let dir = env::temp_dir().join(format!("svm_run_{rnd:016x}"));
    // `mut` is used only on unix (the `mode` call below); on other targets the cfg block is empty,
    // so silence the would-be `unused_mut` warning there rather than under `-D warnings` failing CI.
    #[cfg_attr(not(unix), allow(unused_mut))]
    let mut b = fs::DirBuilder::new();
    // Non-recursive create: fails (EEXIST) on any pre-existing path — including a planted symlink —
    // rather than following it. On unix, restrict to owner so the inner file can't be swapped.
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        b.mode(0o700);
    }
    b.create(&dir)
        .map_err(|e| format!("create temp IR dir: {e}"))?;
    Ok(dir)
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
