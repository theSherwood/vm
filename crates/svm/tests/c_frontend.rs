//! End-to-end C: the vendored chibicc fork (`frontend/chibicc`, `--emit-ir`) compiles C
//! to our text IR, which we verify and run on the reference interpreter. This is the
//! Phase-2 "it works" milestone (`DESIGN.md` §18) — real C through the whole pipeline.
//!
//! Requires a C toolchain (`make` + `cc`) to build the frontend; skipped-by-build only
//! if those are absent. The frontend is outside the escape-TCB (§2a): whatever IR it
//! emits still goes through the verifier before it runs.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use svm_interp::{run, Value};
use svm_ir::ValType;
use svm_jit::{compile_and_run, JitOutcome};
use svm_text::parse_module;
use svm_verify::verify_module;

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
    let base = std::env::temp_dir().join(format!("svm_cfe_{}_{id}", std::process::id()));
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

/// Compile + verify + run `main` (function 0) on **both** the interpreter and the JIT,
/// assert they agree, and return the results. So every C test is also a JIT diff test.
fn run_c(src: &str) -> Vec<Value> {
    let ir = c_to_ir(src);
    let m =
        parse_module(&ir).unwrap_or_else(|e| panic!("parse IR failed: {e:?}\n--- IR ---\n{ir}"));
    verify_module(&m).unwrap_or_else(|e| panic!("verify failed: {e:?}\n--- IR ---\n{ir}"));

    let mut fuel = 1_000_000u64;
    let want = run(&m, 0, &[], &mut fuel).expect("interp run");

    match compile_and_run(&m, 0, &[]).expect("jit compile") {
        JitOutcome::Returned(slots) => {
            let got: Vec<Value> = m.funcs[0]
                .results
                .iter()
                .zip(slots)
                .map(|(t, s)| match t {
                    ValType::I32 => Value::I32(s as i32),
                    ValType::I64 => Value::I64(s),
                    ValType::F32 => Value::F32(f32::from_bits(s as u32)),
                    ValType::F64 => Value::F64(f64::from_bits(s as u64)),
                })
                .collect();
            assert_eq!(
                want, got,
                "interp/JIT disagree for C:\n{src}\n--- IR ---\n{ir}"
            );
        }
        other => panic!("JIT did not return for C:\n{src}\ngot {other:?}"),
    }
    want
}

fn i32_of(src: &str) -> i32 {
    match run_c(src).as_slice() {
        [Value::I32(x)] => *x,
        other => panic!("expected one i32 result, got {other:?}"),
    }
}

#[test]
fn c_integer_arithmetic_end_to_end() {
    assert_eq!(i32_of("int main() { return 42; }"), 42);
    assert_eq!(i32_of("int main() { return 6 * 7 - 1; }"), 41);
    assert_eq!(i32_of("int main() { return (2 + 3) * 4; }"), 20);
    assert_eq!(i32_of("int main() { return 100 / 7; }"), 14);
    assert_eq!(i32_of("int main() { return 100 % 7; }"), 2);
    assert_eq!(i32_of("int main() { return -5 + 2; }"), -3);
    assert_eq!(i32_of("int main() { return ~0; }"), -1);
    assert_eq!(i32_of("int main() { return !0; }"), 1);
    assert_eq!(i32_of("int main() { return !5; }"), 0);
}

#[test]
fn c_bitwise_and_shifts_end_to_end() {
    assert_eq!(i32_of("int main() { return 0xff & 0x0f; }"), 0x0f);
    assert_eq!(i32_of("int main() { return 0xf0 | 0x0f; }"), 0xff);
    assert_eq!(i32_of("int main() { return 0xff ^ 0x0f; }"), 0xf0);
    assert_eq!(i32_of("int main() { return 1 << 4; }"), 16);
    assert_eq!(i32_of("int main() { return 256 >> 3; }"), 32);
}

#[test]
fn c_comparisons_end_to_end() {
    assert_eq!(i32_of("int main() { return 1 < 2; }"), 1);
    assert_eq!(i32_of("int main() { return 2 < 1; }"), 0);
    assert_eq!(i32_of("int main() { return 5 == 5; }"), 1);
    assert_eq!(i32_of("int main() { return 5 != 5; }"), 0);
    assert_eq!(i32_of("int main() { return 3 >= 3; }"), 1);
    assert_eq!(i32_of("int main() { return 2 > 3; }"), 0);
}

#[test]
fn c_long_arithmetic_end_to_end() {
    // `long` is i64 (LP64); the result is truncated to i32 by the `int main` return cast.
    assert_eq!(
        i32_of("int main() { return (int)(1000000L * 1000000L % 1000000007L); }"),
        { ((1_000_000i64 * 1_000_000) % 1_000_000_007) as i32 }
    );
}

#[test]
fn c_locals_and_assignment_end_to_end() {
    assert_eq!(i32_of("int main() { int x = 5; return x; }"), 5);
    assert_eq!(
        i32_of("int main() { int x = 5; int y = 7; return x * y + x; }"),
        40
    );
    assert_eq!(i32_of("int main() { int x = 5; x = x + 1; return x; }"), 6);
    assert_eq!(
        i32_of("int main() { int a = 2; int b = 3; int c = a * b; return c; }"),
        6
    );
    // y is an independent copy of x.
    assert_eq!(
        i32_of("int main() { int x = 10; int y = x; x = 20; return y; }"),
        10
    );
    // chained assignment returns the assigned value.
    assert_eq!(
        i32_of("int main() { int x; int y; y = x = 9; return x + y; }"),
        18
    );
}

#[test]
fn c_long_locals_end_to_end() {
    assert_eq!(
        i32_of("int main() { long x = 5000000000; return (int)(x / 1000000); }"),
        5000
    );
}

#[test]
fn c_pointers_to_locals_end_to_end() {
    // &x / *p over a data-stack local (the §3d masked-memory model).
    assert_eq!(
        i32_of("int main() { int x = 42; int *p = &x; return *p; }"),
        42
    );
    assert_eq!(
        i32_of("int main() { int x = 42; int *p = &x; *p = 99; return x; }"),
        99
    );
    assert_eq!(
        i32_of("int main() { long n = 7; long *p = &n; *p = *p + 1; return (int)n; }"),
        8
    );
}

#[test]
fn c_control_flow_end_to_end() {
    // if / else
    assert_eq!(
        i32_of("int main() { int x = 5; if (x > 3) return 1; else return 2; }"),
        1
    );
    assert_eq!(
        i32_of("int main() { int x = 1; if (x > 3) return 1; return 2; }"),
        2
    );
    // while: sum 1..=100
    assert_eq!(
        i32_of("int main() { int s = 0; int i = 1; while (i <= 100) { s = s + i; i = i + 1; } return s; }"),
        5050
    );
    // for: factorial of 5
    assert_eq!(
        i32_of("int main() { int f = 1; for (int i = 1; i <= 5; i = i + 1) f = f * i; return f; }"),
        120
    );
    // nested + early return: first divisor of 91 above 1
    assert_eq!(
        i32_of("int main() { for (int i = 2; i < 91; i = i + 1) { if (91 % i == 0) return i; } return 91; }"),
        7
    );
    // fibonacci(10) iteratively
    assert_eq!(
        i32_of("int main() { int a = 0; int b = 1; for (int i = 0; i < 10; i = i + 1) { int t = a + b; a = b; b = t; } return a; }"),
        55
    );
}
