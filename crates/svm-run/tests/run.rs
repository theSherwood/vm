//! End-to-end tests for the embedding runtime + CLI: hand-written sandboxed programs (text IR,
//! no frontend) exercised through `run_powerbox`/`run_kernel`, the shipped demo, and the `svm-run`
//! binary itself driving a `.svm` file with real stdout + exit code.

use std::path::PathBuf;
use std::process::Command;

use svm_ir::Module;
use svm_run::{is_powerbox_entry, run_kernel, run_powerbox, Outcome, Value};
use svm_text::parse_module;
use svm_verify::verify_module;

/// Parse + verify a text-IR program (verification is what makes it safe to run).
fn load(src: &str) -> Module {
    let m = parse_module(src).expect("parse text IR");
    verify_module(&m).expect("verify");
    m
}

#[test]
fn writes_to_stdout_and_returns() {
    // A powerbox program: write "hi\n" to stdout (Stream cap, type 0 op 1) on the granted
    // stdout handle (v0), then return 7.
    let m = load(
        "memory 16\n\
         data 16 \"hi\\n\"\n\
         func (i32, i32, i32) -> (i32) {\n\
         block0(v0: i32, v1: i32, v2: i32):\n\
         \x20 v3 = i64.const 16\n\
         \x20 v4 = i64.const 3\n\
         \x20 v5 = cap.call 0 1 (i64, i64) -> (i64) v0(v3, v4)\n\
         \x20 v6 = i32.const 7\n\
         \x20 return v6\n\
         }\n",
    );
    assert!(is_powerbox_entry(&m));
    let run = run_powerbox(&m, b"").expect("run");
    assert_eq!(run.stdout, b"hi\n");
    assert_eq!(run.outcome, Outcome::Returned(vec![Value::I32(7)]));
    assert!(run.stderr.is_empty());
}

#[test]
fn exit_capability_sets_code() {
    // The guest invokes Exit(5) (type 1 op 0) on the granted exit handle (v2) — terminal.
    let m = load(
        "func (i32, i32, i32) -> (i32) {\n\
         block0(v0: i32, v1: i32, v2: i32):\n\
         \x20 v3 = i32.const 5\n\
         \x20 cap.call 1 0 (i32) -> () v2(v3)\n\
         \x20 unreachable\n\
         }\n",
    );
    let run = run_powerbox(&m, b"").expect("run");
    assert_eq!(run.outcome, Outcome::Exited(5));
}

#[test]
fn echoes_stdin_to_stdout() {
    // read(stdin) into the window, then write that many bytes back out — a stdin round-trip.
    let m = load(
        "memory 16\n\
         func (i32, i32, i32) -> (i32) {\n\
         block0(v0: i32, v1: i32, v2: i32):\n\
         \x20 v3 = i64.const 0\n\
         \x20 v4 = i64.const 64\n\
         \x20 v5 = cap.call 0 0 (i64, i64) -> (i64) v1(v3, v4)\n\
         \x20 v6 = cap.call 0 1 (i64, i64) -> (i64) v0(v3, v5)\n\
         \x20 v7 = i32.const 0\n\
         \x20 return v7\n\
         }\n",
    );
    let run = run_powerbox(&m, b"ping").expect("run");
    assert_eq!(run.stdout, b"ping");
}

#[test]
fn bare_kernel_returns_value() {
    // A non-powerbox entry — a pure function (i64 x) -> (i64) returning x + 1.
    let m = load(
        "func (i64) -> (i64) {\n\
         block0(v0: i64):\n\
         \x20 v1 = i64.const 1\n\
         \x20 v2 = i64.add v0 v1\n\
         \x20 return v2\n\
         }\n",
    );
    assert!(!is_powerbox_entry(&m));
    let out = run_kernel(&m, &[41]).expect("run kernel");
    assert_eq!(out, vec![Value::I64(42)]);
}

fn demo(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("demos")
        .join(name)
}

#[test]
fn runs_shipped_demo() {
    let src = std::fs::read_to_string(demo("hello.svm")).expect("read hello.svm");
    let m = load(&src);
    let run = run_powerbox(&m, b"").expect("run");
    assert_eq!(run.stdout, b"hello, sandbox!\n");
    assert_eq!(run.outcome, Outcome::Returned(vec![Value::I32(0)]));
}

/// Drive the actual `svm-run` binary on the demo `.svm`: it must print the greeting to stdout
/// and exit 0 — the "a program runs in the sandbox from the command line" milestone.
#[test]
fn cli_runs_svm_file() {
    let out = Command::new(env!("CARGO_BIN_EXE_svm-run"))
        .arg(demo("hello.svm"))
        .output()
        .expect("spawn svm-run");
    assert!(
        out.status.success(),
        "exit: {:?}\nstderr: {}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(out.stdout, b"hello, sandbox!\n");
}

/// Compile a demo `.c` with native `cc`, run both it and `svm-run`, and assert identical
/// stdout — a real-program oracle (the whole stack vs. a real compiler). Skipped (not failed)
/// when `cc` or the frontend is unavailable.
fn assert_demo_matches_cc(name: &str) {
    let c = demo(name);
    let exe = std::env::temp_dir().join(format!(
        "svm_demo_{}_{}",
        std::process::id(),
        name.replace(['.', '/'], "_") // flatten subdirs (e.g. `jsmn/jsmn_demo.c`) into one name
    ));
    match Command::new("cc").arg(&c).arg("-o").arg(&exe).status() {
        Ok(s) if s.success() => {}
        _ => {
            eprintln!("note: skipping {name} (cc unavailable)");
            return;
        }
    }
    let native = Command::new(&exe).output().expect("run native build");
    let svm = Command::new(env!("CARGO_BIN_EXE_svm-run"))
        .arg(&c)
        .output()
        .expect("spawn svm-run");
    if !svm.status.success() {
        let err = String::from_utf8_lossy(&svm.stderr);
        if err.contains("chibicc") {
            eprintln!(
                "note: skipping {name} (frontend unavailable): {}",
                err.trim()
            );
            return;
        }
        panic!("svm-run on {name} failed: {err}");
    }
    assert_eq!(
        String::from_utf8_lossy(&svm.stdout),
        String::from_utf8_lossy(&native.stdout),
        "{name}: svm-run vs native cc stdout differ"
    );
}

/// A recursive-descent calculator (recursion, a global string table + a global struct-array of
/// function pointers, indirect dispatch) — sandboxed output must match native `cc`.
#[test]
fn demo_calc_matches_native() {
    assert_demo_matches_cc("calc.c");
}

/// **The capstone: a real third-party C library runs in the sandbox.** The Clay UI layout
/// library (`demos/clay/clay.h`, ~5k lines, zlib-licensed, vendored) compiles through the
/// frontend to ~93k lines of IR, verifies, and runs on the JIT — building a small layout and
/// printing its render commands, deterministically and identically to a native build. Skipped
/// (not failed) when the chibicc frontend is unavailable.
#[test]
fn demo_clay_layout_runs() {
    let out = Command::new(env!("CARGO_BIN_EXE_svm-run"))
        .arg(demo("clay/clay_demo.c"))
        .output()
        .expect("spawn svm-run");
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        if err.contains("chibicc") {
            eprintln!(
                "note: skipping Clay demo (frontend unavailable): {}",
                err.trim()
            );
            return;
        }
        panic!("svm-run on clay_demo.c failed: {err}");
    }
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "3 render commands:\n\
         \x20 cmd 1 bbox=(16,16 768x40)\n\
         \x20 cmd 3 bbox=(16,16 152x18)\n\
         \x20 cmd 1 bbox=(16,64 768x520)\n",
    );
}

/// Exact-rational arithmetic (by-value struct args/returns through direct *and* indirect calls,
/// recursion) — sandboxed output must match native `cc`. The program that surfaced the
/// sret-from-a-non-entry-block bug.
#[test]
fn demo_rational_matches_native() {
    assert_demo_matches_cc("rational.c");
}

/// The jsmn JSON tokenizer (`demos/jsmn/jsmn.h`, MIT, vendored) — a different shape from Clay:
/// pure char/state-machine string scanning, zero allocations. Tokenizes a JSON string sandboxed
/// and prints the token types/spans; output must match a native `cc` build. (It needed no new
/// fixes — a clean validation that string parsing, escapes, nesting, and error paths work.)
#[test]
fn demo_jsmn_matches_native() {
    assert_demo_matches_cc("jsmn/jsmn_demo.c");
}

/// SHA-256 (B-Con's `crypto-algorithms`, public domain, vendored) — a pure integer/bit shape
/// (32-bit wrapping arithmetic, rotates-as-shifts, a round-key table). Hashes a few strings
/// sandboxed and prints the hex digests; must match a native `cc` build (and the standard test
/// vectors). The shakedown turned a `func_index` null-token crash into a clean error.
#[test]
fn demo_sha256_matches_native() {
    assert_demo_matches_cc("sha256/sha_demo.c");
}

/// xxHash (Cyan4973/xxHash, BSD-2-Clause, vendored) — XXH32/XXH64 in a self-contained scalar
/// build. Another integer/bit shape (multiply/rotate hashing); output matches a native `cc`
/// build and the standard test vectors. The shakedown added `_Static_assert` support.
#[test]
fn demo_xxhash_matches_native() {
    assert_demo_matches_cc("xxhash/xxh_demo.c");
}

/// If the chibicc frontend is buildable, the CLI compiles and runs the C demo too — the same
/// greeting from C source. Skipped (not failed) when the toolchain is unavailable.
#[test]
fn cli_compiles_and_runs_c() {
    let out = Command::new(env!("CARGO_BIN_EXE_svm-run"))
        .arg(demo("hello.c"))
        .output()
        .expect("spawn svm-run");
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        if err.contains("chibicc") {
            eprintln!(
                "note: skipping C demo (frontend unavailable): {}",
                err.trim()
            );
            return;
        }
        panic!("svm-run on hello.c failed: {err}");
    }
    assert_eq!(out.stdout, b"hello, sandbox!\n");
}
