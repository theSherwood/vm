//! **rustc-bitcode miscompile probe** (ISSUES.md I23 follow-on). A fast *correctness-only* sweep: for
//! each `rustbench/probes/<name>.rs` workload (prelude + a `run(n)->i64`), compile it with the system
//! rustc, translate the textual IR through `svm-llvm`, run it on `svm-jit`, and compare `run(n)` to a
//! native build over a spread of `n` — a MISCOMPILE (a trap on an in-bounds program, or a wrong value)
//! is exactly the class of bug the rustbench harness caught for I23 (opaque-pointer / auto-vectorizer
//! patterns clang doesn't emit but rustc does). No timing, so it iterates in seconds over many probes.
//!
//! Run from `bench/`:  cargo run --release --bin rustprobe

use std::path::{Path, PathBuf};
use std::process::Command;

const NS: &[i64] = &[0, 1, 2, 3, 7, 10, 33, 64, 100];

fn rb_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("rustbench")
}
fn probes_dir() -> PathBuf {
    rb_dir().join("probes")
}
fn tmp(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("rustprobe_{}_{name}", std::process::id()))
}
fn rustc() -> Command {
    match std::env::var("SVM_RUSTBENCH_RUSTC") {
        Ok(t) if t.starts_with('+') => {
            let mut c = Command::new("rustc");
            c.arg(&t);
            c
        }
        Ok(t) => Command::new(t),
        Err(_) => Command::new("rustc"),
    }
}

/// prelude.rs + probes/<name>.rs — the full crate source for one probe.
fn compose(name: &str) -> Option<String> {
    let pre = std::fs::read_to_string(rb_dir().join("prelude.rs")).ok()?;
    let wl = std::fs::read_to_string(probes_dir().join(format!("{name}.rs"))).ok()?;
    Some(format!("{pre}\n{wl}"))
}

/// rustc `--emit=llvm-ir` → svm_llvm → svm_jit; returns a `run(n)` runner (Err on a trap).
fn svm_runner(src: &Path) -> Option<impl FnMut(i64) -> Result<i64, String>> {
    let ll = tmp("svm.ll");
    let ok = rustc()
        .args([
            "--edition",
            "2021",
            "-O",
            "-Cpanic=abort",
            "--emit=llvm-ir",
            "--crate-type=cdylib",
            "--target=x86_64-unknown-linux-gnu",
        ])
        .arg(src)
        .arg("-o")
        .arg(&ll)
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if !ok {
        return None;
    }
    let t = match svm_llvm::translate_ll_path(&ll) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("  translate error: {e:?}");
            return None;
        }
    };
    let sp = t.entry_sp as i64;
    let e = t.exports.iter().find(|(n, _)| n == "run")?.1;
    let mut cm = match svm_jit::compile(&t.module, e) {
        Ok(cm) => cm,
        Err(e) => {
            eprintln!("  svm_jit::compile error: {e:?}");
            return None;
        }
    };
    Some(move |n: i64| -> Result<i64, String> {
        match cm.run(&[sp, n], None, None, None) {
            Ok((svm_jit::JitOutcome::Returned(v), _)) => Ok(v[0]),
            Ok((o, _)) => Err(format!("{o:?}")),
            Err(e) => Err(format!("run err {e:?}")),
        }
    })
}

/// Native `rustc` staticlib + the confine C main → `run(n)` (via a zero-width timing call).
fn native_build(src: &Path) -> Option<PathBuf> {
    let lib = tmp("native.a");
    let ok = rustc()
        .args([
            "--edition",
            "2021",
            "-O",
            "-Cpanic=abort",
            "--crate-type=staticlib",
        ])
        .arg(src)
        .arg("-o")
        .arg(&lib)
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if !ok {
        return None;
    }
    let exe = tmp("native.exe");
    let main_c = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("confine/native_main.c");
    let ok = Command::new("clang")
        .args(["-O2", "-march=native"])
        .arg(&main_c)
        .arg(&lib)
        .arg("-o")
        .arg(&exe)
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    ok.then_some(exe)
}
fn native_run(exe: &Path, n: i64) -> Option<i64> {
    // native_main prints "<per_iter_ns>\n<run(small)>"; call with small=n so line 2 is run(n).
    let out = Command::new(exe)
        .args([n.to_string(), (n + 1).to_string(), "1".to_string()])
        .output()
        .ok()?;
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .nth(1)?
        .trim()
        .parse()
        .ok()
}

fn main() {
    let mut probes: Vec<String> = std::fs::read_dir(probes_dir())
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| {
            let p = e.path();
            (p.extension()?.to_str()? == "rs").then(|| p.file_stem()?.to_str().map(String::from))?
        })
        .collect();
    probes.sort();
    if probes.is_empty() {
        eprintln!("no probes in {}", probes_dir().display());
        std::process::exit(2);
    }
    let mut fails = 0usize;
    for name in &probes {
        let Some(text) = compose(name) else {
            eprintln!("{name}: compose failed — skip");
            continue;
        };
        let src = tmp(&format!("{name}.rs"));
        if std::fs::write(&src, &text).is_err() {
            continue;
        }
        let Some(mut svm) = svm_runner(&src) else {
            eprintln!("{name}: svm lane unavailable — skip");
            continue;
        };
        let Some(exe) = native_build(&src) else {
            eprintln!("{name}: native lane unavailable — skip");
            continue;
        };
        let mut bad: Vec<String> = Vec::new();
        for &n in NS {
            let want = native_run(&exe, n);
            let got = svm(n);
            match (want, &got) {
                (Some(w), Ok(g)) if *g == w => {}
                (Some(w), Ok(g)) => bad.push(format!("n={n}: svm={g} native={w}")),
                (Some(w), Err(e)) => bad.push(format!("n={n}: svm TRAP({e}) native={w}")),
                (None, _) => bad.push(format!("n={n}: native failed")),
            }
        }
        if bad.is_empty() {
            println!("{name:<24} OK");
        } else {
            fails += 1;
            println!("{name:<24} MISCOMPILE");
            for b in &bad {
                println!("    {b}");
            }
        }
    }
    println!("\n{}/{} probes clean", probes.len() - fails, probes.len());
    std::process::exit(if fails == 0 { 0 } else { 1 });
}
