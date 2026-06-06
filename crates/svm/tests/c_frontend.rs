//! End-to-end C: the vendored chibicc fork (`frontend/chibicc`, `--emit-ir`) compiles C
//! to our text IR, which we verify and run on the reference interpreter. This is the
//! Phase-2 "it works" milestone (`DESIGN.md` §18) — real C through the whole pipeline.
//!
//! Each test runs `main` on **both** the interpreter and the JIT and asserts they agree
//! (results + captured stdout/exit), so every test doubles as a JIT differential test. A
//! second tier (`c_matches_gcc_*`) compiles the *same* C with native `cc` and compares
//! exit code + stdout, validating C semantics against a real compiler.
//!
//! Requires a unix C toolchain (`make` + `cc`) to build the chibicc fork, and the frontend bakes
//! a **4 KiB** RO-data page-isolation at compile time — so this suite is **Linux-only for now**
//! (`#![cfg(target_os = "linux")]`). Windows lacks the toolchain; macOS-ARM (16 KiB pages) needs
//! the frontend RO-isolation pin + guest page-size exposure (Phase 3.5 part 2) before its programs
//! run without spurious RO over-protection faults. The frontend is outside the escape-TCB (§2a):
//! whatever IR it emits still goes through the verifier, and the JIT/PAL it exercises is validated
//! cross-platform by `jit_fuzz`/`escape_oracle` + the `svm-jit` PAL conformance test instead.
#![cfg(target_os = "linux")]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use core::ffi::c_void;
use svm_interp::{run_with_host, Host, StreamRole, Trap, Value};
use svm_ir::ValType;
use svm_jit::{compile_and_run_with_host, JitOutcome, TrapKind};
use svm_run::cap_thunk; // the shared JIT-CapThunk → reference-Host bridge (§9)
use svm_text::parse_module;
use svm_verify::verify_module;

fn to_slot(v: Value) -> i64 {
    match v {
        Value::I32(x) => x as i64,
        Value::I64(x) => x,
        Value::F32(x) => x.to_bits() as i64,
        Value::F64(x) => x.to_bits() as i64,
    }
}

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

/// Count `Load`/`Store` instructions that live **outside each function's entry block**
/// (`blocks[0]`). The frontend's one-time setup — `_start` writing globals/strings, and a
/// function's `__va_area__`/spill prologue — all lands in entry blocks, so this isolates
/// *steady-state* memory traffic: loop bodies and post-entry control flow. A fully
/// SSA-promoted scalar loop has **zero** here (its locals are block params, not memory) —
/// the §3 promotion win this guards against silently regressing.
fn loop_region_mem_ops(ir: &str) -> usize {
    let m = parse_module(ir).expect("frontend IR should parse");
    m.funcs
        .iter()
        .flat_map(|f| f.blocks.iter().skip(1)) // skip each function's entry block
        .flat_map(|b| b.insts.iter())
        .filter(|i| matches!(i, svm_ir::Inst::Load { .. } | svm_ir::Inst::Store { .. }))
        .count()
}

/// What a C program did: either `main` returned normally (with its result values) or it
/// called `exit(code)`. Plus the bytes it wrote to stdout/stderr through the powerbox.
#[derive(Debug, PartialEq)]
enum Outcome {
    Returned(Vec<Value>),
    Exited(i32),
}
struct CRun {
    outcome: Outcome,
    stdout: Vec<u8>,
}

/// Compile + verify + run function 0 (the synthetic `_start`) on **both** the interpreter
/// and the JIT under identical mock powerboxes, assert they agree on the outcome *and* the
/// observable host effects (stdout/stderr), and return both. So every C test is also a JIT
/// differential test, capability effects included.
fn run_c_full(src: &str) -> CRun {
    let ir = c_to_ir(src);
    let m =
        parse_module(&ir).unwrap_or_else(|e| panic!("parse IR failed: {e:?}\n--- IR ---\n{ir}"));
    verify_module(&m).unwrap_or_else(|e| panic!("verify failed: {e:?}\n--- IR ---\n{ir}"));

    // `_start(stdout, stdin, exit, memory)` takes the powerbox handles. Grant them identically on
    // both hosts (grants are deterministic, so the handle values match).
    let mut hi = Host::new();
    let mut hj = Host::new();
    let grant = |h: &mut Host| {
        [
            Value::I32(h.grant_stream(StreamRole::Out)),
            Value::I32(h.grant_stream(StreamRole::In)),
            Value::I32(h.grant_exit()),
            Value::I32(h.grant_memory()),
        ]
    };
    let args = grant(&mut hi);
    assert_eq!(args, grant(&mut hj), "grants are deterministic");

    let mut fuel = 50_000_000u64;
    let interp = run_with_host(&m, 0, &args, &mut fuel, &mut hi);
    let slots: Vec<i64> = args.iter().copied().map(to_slot).collect();
    let jit = compile_and_run_with_host(
        &m,
        0,
        &slots,
        cap_thunk,
        &mut hj as *mut Host as *mut c_void,
    )
    .expect("jit compiles");

    let typed = |s: &[i64]| -> Vec<Value> {
        m.funcs[0]
            .results
            .iter()
            .zip(s)
            .map(|(t, &v)| match t {
                ValType::I32 => Value::I32(v as i32),
                ValType::I64 => Value::I64(v),
                ValType::F32 => Value::F32(f32::from_bits(v as u32)),
                ValType::F64 => Value::F64(f64::from_bits(v as u64)),
            })
            .collect()
    };
    let outcome = match (interp, jit) {
        (Ok(want), JitOutcome::Returned(got)) => {
            assert_eq!(
                want,
                typed(&got),
                "interp/JIT result disagree:\n{src}\n{ir}"
            );
            Outcome::Returned(want)
        }
        (Err(Trap::Exit(want)), JitOutcome::Exited(got)) => {
            assert_eq!(want, got, "interp/JIT exit code disagree:\n{src}");
            Outcome::Exited(want)
        }
        (i, j) => panic!("interp/JIT outcome disagree for:\n{src}\ninterp={i:?} jit={j:?}\n{ir}"),
    };
    assert_eq!(hi.stdout, hj.stdout, "stdout differs:\n{src}");
    assert_eq!(hi.stderr, hj.stderr, "stderr differs:\n{src}");
    CRun {
        outcome,
        stdout: hi.stdout,
    }
}

/// Run a normally-returning program and return its result values.
fn run_c(src: &str) -> Vec<Value> {
    match run_c_full(src).outcome {
        Outcome::Returned(v) => v,
        Outcome::Exited(c) => panic!("expected a normal return, but the program exited({c})"),
    }
}

fn i32_of(src: &str) -> i32 {
    match run_c(src).as_slice() {
        [Value::I32(x)] => *x,
        other => panic!("expected one i32 result, got {other:?}"),
    }
}

/// D40 (§3a/§4): a string literal lives in a **read-only** data segment, so writing through a
/// pointer to it (UB in C) detect-and-kills on both backends instead of silently corrupting the
/// literal. The first real C consumer of the read-only data section.
#[cfg(unix)]
#[test]
fn c_write_to_string_literal_faults() {
    let src = "int main() { char *s = \"hi\"; s[0] = 'X'; return 0; }";
    let ir = c_to_ir(src);
    assert!(
        ir.contains("data ro "),
        "expected a read-only data segment:\n{ir}"
    );
    let m = parse_module(&ir).expect("parse");
    verify_module(&m).expect("verify");

    let mut hi = Host::new();
    let mut hj = Host::new();
    let grant = |h: &mut Host| {
        [
            Value::I32(h.grant_stream(StreamRole::Out)),
            Value::I32(h.grant_stream(StreamRole::In)),
            Value::I32(h.grant_exit()),
            Value::I32(h.grant_memory()),
        ]
    };
    let args = grant(&mut hi);
    grant(&mut hj);
    let mut fuel = 50_000_000u64;
    let interp = run_with_host(&m, 0, &args, &mut fuel, &mut hi);
    let slots: Vec<i64> = args.iter().copied().map(to_slot).collect();
    let jit = compile_and_run_with_host(
        &m,
        0,
        &slots,
        cap_thunk,
        &mut hj as *mut Host as *mut c_void,
    )
    .expect("jit compiles");

    assert_eq!(
        interp,
        Err(Trap::MemoryFault),
        "interp: write to a string literal must fault\n{ir}"
    );
    assert!(
        matches!(jit, JitOutcome::Trapped(TrapKind::MemoryFault)),
        "jit: write to a string literal must detect-and-kill, got {jit:?}\n{ir}"
    );
}

/// The guest **grows its own window** through the Memory capability: `__vm_map` (a frontend
/// builtin → `cap.call` on the granted Memory handle) commits a page at 256 MiB — deep in the
/// reserved tail, far above the backed prefix — then a store/load round-trips through it. Proves
/// the Memory cap is granted to compiled C programs and the builtin lowers correctly, on both
/// backends (interp page map + JIT real `mprotect`). The §1a sparse-address-space path from C.
#[test]
fn c_memory_cap_grows_into_reserved_tail() {
    let src = r#"
long __vm_map(long off, long len, int prot);
int main() {
  long base = 268435456;                        /* 256 MiB, in the reserved tail */
  if (__vm_map(base, 4096, 3) != 0) return -1;  /* commit one RW page (READ|WRITE) */
  int *p = (int *)base;
  p[0] = 43981;                                 /* 0xABCD */
  p[1] = p[0] + 1;
  return p[1];
}
"#;
    assert_eq!(run_c(src), vec![Value::I32(43982)]);
}

/// The shipped `<stdlib.h>` is a real guest libc: `malloc`/`calloc`/`realloc`/`free` that **grow
/// the window via the Memory cap** — available to any program that just `#include <stdlib.h>`, no
/// prelude. Allocates 400 KiB (well past the 64 KiB initial window, forcing growth), checks
/// `calloc` zeroes and `realloc` preserves contents, on both backends.
#[test]
fn c_default_stdlib_malloc_grows() {
    let src = r#"
#include <stdlib.h>
int main() {
  int n = 50000;                              /* 2 x 200 KiB > the 64 KiB initial window */
  int *a = (int *)malloc(n * sizeof(int));
  int *b = (int *)calloc(n, sizeof(int));     /* must be zero-filled */
  if (!a || !b) return -1;
  for (int i = 0; i < n; i++) a[i] = 1;
  long s = 0;
  for (int i = 0; i < n; i++) s += a[i] + b[i];   /* = n (a=1, b=0) */
  free(a); free(b);
  int *c = (int *)malloc(8 * sizeof(int));
  for (int i = 0; i < 8; i++) c[i] = i;
  c = (int *)realloc(c, 16 * sizeof(int));    /* preserves c[0..8] */
  long rs = 0;
  for (int i = 0; i < 8; i++) rs += c[i];     /* = 28 */
  return (int)(s + rs);                       /* 50000 + 28 */
}
"#;
    assert_eq!(run_c(src), vec![Value::I32(50028)]);
}

/// An access to an **un-grown** tail page faults (detect-and-kill) on both backends: the guest
/// must `map` before it can touch the reserved tail. The negative of the test above.
#[test]
fn c_ungrown_tail_access_faults() {
    let src = "int main() { int *p = (int *)268435456; return p[0]; }";
    let ir = c_to_ir(src);
    let m = parse_module(&ir).expect("parse");
    verify_module(&m).expect("verify");
    let grant = |h: &mut Host| {
        [
            Value::I32(h.grant_stream(StreamRole::Out)),
            Value::I32(h.grant_stream(StreamRole::In)),
            Value::I32(h.grant_exit()),
            Value::I32(h.grant_memory()),
        ]
    };
    let mut hi = Host::new();
    let mut hj = Host::new();
    let args = grant(&mut hi);
    grant(&mut hj);
    let mut fuel = 50_000_000u64;
    let interp = run_with_host(&m, 0, &args, &mut fuel, &mut hi);
    let slots: Vec<i64> = args.iter().copied().map(to_slot).collect();
    let jit = compile_and_run_with_host(
        &m,
        0,
        &slots,
        cap_thunk,
        &mut hj as *mut Host as *mut c_void,
    )
    .expect("jit compiles");
    assert_eq!(
        interp,
        Err(Trap::MemoryFault),
        "interp: ungrown tail must fault"
    );
    assert!(
        matches!(jit, JitOutcome::Trapped(TrapKind::MemoryFault)),
        "jit: ungrown tail must detect-and-kill, got {jit:?}"
    );
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

#[test]
fn c_function_calls_end_to_end() {
    // direct call with parameters
    assert_eq!(
        i32_of("int add(int a, int b) { return a + b; } int main() { return add(3, 4) * 2; }"),
        14
    );
    // a callee with its own locals must not clobber the caller's frame (fresh data-SP)
    assert_eq!(
        i32_of(
            "int sq(int x) { int t = x * x; return t; } \
             int main() { int a = sq(3); int b = sq(4); return a + b; }"
        ),
        25
    );
}

#[test]
fn c_recursion_end_to_end() {
    // Recursion is the real test of per-call frames (§3d data stack): each activation of
    // fib has its own `n`, so the parent's `n` survives across the recursive calls.
    assert_eq!(
        i32_of("int fib(int n) { if (n < 2) return n; return fib(n-1) + fib(n-2); } int main() { return fib(10); }"),
        55
    );
    assert_eq!(
        i32_of("int fact(int n) { if (n <= 1) return 1; return n * fact(n-1); } int main() { return fact(6); }"),
        720
    );
    // mutual recursion
    assert_eq!(
        i32_of(
            "int is_even(int n); \
             int is_odd(int n) { if (n == 0) return 0; return is_even(n - 1); } \
             int is_even(int n) { if (n == 0) return 1; return is_odd(n - 1); } \
             int main() { return is_even(10) + is_odd(7); }"
        ),
        2
    );
}

#[test]
fn c_short_circuit_and_ternary_end_to_end() {
    // basic truth values (&&, ||, ?:)
    assert_eq!(
        i32_of("int main() { int x = 5; return x > 0 && x < 10; }"),
        1
    );
    assert_eq!(
        i32_of("int main() { int x = 5; return x > 9 && x < 10; }"),
        0
    );
    assert_eq!(i32_of("int main() { return 0 || 3; }"), 1);
    assert_eq!(i32_of("int main() { return 0 || 0; }"), 0);
    assert_eq!(
        i32_of("int main() { int x = 5; return x > 3 ? 100 : 200; }"),
        100
    );
    assert_eq!(
        i32_of("int main() { int x = 1; return x > 3 ? 100 : 200; }"),
        200
    );
    // && and || normalize to 0/1 even for non-1 truthy operands
    assert_eq!(i32_of("int main() { return 7 && 4; }"), 1);
    // short-circuit must NOT evaluate the RHS side effect
    assert_eq!(
        i32_of("int main() { int x = 5; (x == 5) || (x = 99); return x; }"),
        5
    );
    assert_eq!(
        i32_of("int main() { int x = 5; (x != 5) && (x = 99); return x; }"),
        5
    );
    // ...but DOES evaluate it when not short-circuited
    assert_eq!(
        i32_of("int main() { int x = 5; (x == 0) || (x = 42); return x; }"),
        42
    );
    // ternary with mixed-width arms (result is long); chained ternary
    assert_eq!(
        i32_of("int main() { int g = 85; return g >= 90 ? 4 : g >= 80 ? 3 : g >= 70 ? 2 : 1; }"),
        3
    );
    // ternary only evaluates the taken arm
    assert_eq!(
        i32_of("int main() { int x = 0; int c = 1; c ? (x = 10) : (x = 20); return x; }"),
        10
    );
}

#[test]
fn c_arrays_end_to_end() {
    // index + store/load over a data-stack array
    assert_eq!(
        i32_of("int main() { int a[3]; a[0]=10; a[1]=20; a[2]=30; int s=0; for(int i=0;i<3;i=i+1) s=s+a[i]; return s; }"),
        60
    );
    // pointer walking: *(a+i)
    assert_eq!(
        i32_of("int main() { int a[4]; for(int i=0;i<4;i=i+1) *(a+i)=i*i; int *p=a; return p[0]+p[1]+p[2]+p[3]; }"),
        14
    );
    // array initializer (lowered to per-element stores) + reverse sum
    assert_eq!(
        i32_of("int main() { int a[5] = {1,2,3,4,5}; int s=0; for(int i=4;i>=0;i=i-1) s=s*10+a[i]; return s; }"),
        54321
    );
    // 2D array
    assert_eq!(
        i32_of("int main() { int m[2][2]; m[0][0]=1; m[0][1]=2; m[1][0]=3; m[1][1]=4; return m[0][0]+m[0][1]+m[1][0]+m[1][1]; }"),
        10
    );
}

#[test]
fn c_structs_end_to_end() {
    // member read/write
    assert_eq!(
        i32_of("struct P { int x; int y; }; int main() { struct P p; p.x=3; p.y=4; return p.x*p.x + p.y*p.y; }"),
        25
    );
    // struct initializer (member-wise) + mixed widths
    assert_eq!(
        i32_of("struct R { int lo; long hi; }; int main() { struct R r = {7, 5000000000}; return r.lo + (int)(r.hi / 1000000); }"),
        5007
    );
    // pointer to struct: p->field, and writing through it
    assert_eq!(
        i32_of(
            "struct P { int x; int y; }; \
             int sx(struct P *p) { return p->x; } \
             int main() { struct P p; p.x=11; p.y=22; struct P *q=&p; q->x = q->x + q->y; return sx(q); }"
        ),
        33
    );
    // array of structs
    assert_eq!(
        i32_of(
            "struct Pt { int x; int y; }; \
             int main() { struct Pt a[3]; for (int i=0;i<3;i=i+1) { a[i].x=i; a[i].y=i*i; } \
                          int s=0; for (int i=0;i<3;i=i+1) s += a[i].x + a[i].y; return s; }"
        ),
        8
    );
}

#[test]
fn c_globals_end_to_end() {
    // initialized scalar global
    assert_eq!(i32_of("int g = 42; int main() { return g; }"), 42);
    // mutable global persists across writes
    assert_eq!(
        i32_of(
            "int counter; int bump() { counter = counter + 1; return counter; } \
                int main() { bump(); bump(); return bump(); }"
        ),
        3
    );
    // global array initializer
    assert_eq!(
        i32_of("int arr[3] = {10, 20, 30}; int main() { return arr[0] + arr[1] + arr[2]; }"),
        60
    );
    // global + array + string literal together
    assert_eq!(
        i32_of(
            "int g = 42; int arr[3] = {10,20,30}; \
                int main() { char *s = \"AB\"; return g + arr[0]+arr[1]+arr[2] + s[0] + s[1]; }"
        ),
        233
    );
    // a global struct
    assert_eq!(
        i32_of(
            "struct P { int x; int y; }; struct P p = {3, 4}; \
                int main() { return p.x * p.x + p.y * p.y; }"
        ),
        25
    );
}

#[test]
fn c_string_literals_end_to_end() {
    // string literal indexing + a simple strlen loop over a data-stack copy
    assert_eq!(
        i32_of("int main() { char *s = \"hello\"; int n = 0; while (s[n]) n = n + 1; return n; }"),
        5
    );
    // sum of byte values of a string literal
    assert_eq!(
        i32_of("int main() { char *s = \"ABC\"; return s[0] + s[1] + s[2]; }"),
        65 + 66 + 67
    );
}

#[test]
fn c_hello_world_end_to_end() {
    // The milestone: real C writing to stdout through the powerbox (Stream.write), with
    // a guest strlen, run on both backends and checked against the captured bytes.
    let r = run_c_full(
        "int write(int fd, char *buf, int n); \
         int strlen(char *s) { int n = 0; while (s[n]) n = n + 1; return n; } \
         int main() { char *s = \"hello, world\\n\"; write(1, s, strlen(s)); return 0; }",
    );
    assert_eq!(r.outcome, Outcome::Returned(vec![Value::I32(0)]));
    assert_eq!(r.stdout, b"hello, world\n");
}

#[test]
fn c_stdout_from_loop_end_to_end() {
    // Build a buffer of digits on the data stack, then write it in one call.
    let r = run_c_full(
        "int write(int fd, char *buf, int n); \
         int main() { char buf[10]; for (int i = 0; i < 10; i = i + 1) buf[i] = '0' + i; \
                      write(1, buf, 10); return 0; }",
    );
    assert_eq!(r.stdout, b"0123456789");
}

#[test]
fn c_exit_code_end_to_end() {
    // exit(code) reaches the Exit capability; both backends report the same terminal code.
    let r = run_c_full(
        "void exit(int code); \
         int main() { exit(7); return 0; }",
    );
    assert_eq!(r.outcome, Outcome::Exited(7));
}

#[test]
fn c_write_after_partial_then_exit() {
    // Output is flushed before exit, and code after exit() is dead.
    let r = run_c_full(
        "int write(int fd, char *buf, int n); void exit(int code); \
         int main() { write(1, \"bye\\n\", 4); exit(3); return 99; }",
    );
    assert_eq!(r.outcome, Outcome::Exited(3));
    assert_eq!(r.stdout, b"bye\n");
}

#[test]
fn c_partial_initializer_zero_fill_end_to_end() {
    // A partial initializer relies on ND_MEMZERO zeroing the *rest* of the local at the
    // correct sp-relative address (regression test for the data-SP memzero fix).
    assert_eq!(
        i32_of("int main() { int a[5] = {1, 2}; return a[0]+a[1]+a[2]+a[3]+a[4]; }"),
        3
    );
    assert_eq!(
        i32_of("struct P { int x; int y; int z; }; int main() { struct P p = {7}; return p.x + p.y + p.z; }"),
        7
    );
}

#[test]
fn c_floats_end_to_end() {
    // double arithmetic, truncated back to int on return
    assert_eq!(
        i32_of("int main() { double x = 3.5; double y = 2.0; return (int)(x * y + 1.0); }"),
        8
    );
    // float (f32) arithmetic
    assert_eq!(
        i32_of("int main() { float a = 1.5f; float b = 0.5f; return (int)((a + b) * 4.0f); }"),
        8
    );
    // int -> double -> int conversions
    assert_eq!(
        i32_of("int main() { int n = 7; double d = n; d = d / 2.0; return (int)(d * 10); }"),
        35
    );
    // float comparisons (result is int 0/1)
    assert_eq!(
        i32_of("int main() { double x = 3.14; return x > 3.0 && x < 4.0; }"),
        1
    );
    assert_eq!(i32_of("int main() { double x = 3.14; return x < 3.0; }"), 0);
    // unary float negation
    assert_eq!(
        i32_of("int main() { double x = 5.0; return (int)(-x + 8.0); }"),
        3
    );
    // !x on a float
    assert_eq!(i32_of("int main() { double x = 0.0; return !x; }"), 1);
    // float parameter + return value through a call, and a float ternary
    assert_eq!(
        i32_of(
            "double sq(double x) { return x * x; } \
                int main() { double r = sq(3.0); return (int)(r > 8.0 ? r : 0.0); }"
        ),
        9
    );
    // f32 <-> f64 promotion/demotion
    assert_eq!(
        i32_of(
            "int main() { float f = 2.5f; double d = f; float g = d + 1.5; return (int)(g * 2); }"
        ),
        8
    );
}

#[test]
fn c_break_continue_end_to_end() {
    // continue skips 5, break stops at 10: sum 0..=9 minus 5 = 40
    assert_eq!(
        i32_of("int main() { int s=0; for (int i=0;i<20;i=i+1) { if (i==5) continue; if (i==10) break; s=s+i; } return s; }"),
        40
    );
    // break out of a while
    assert_eq!(
        i32_of("int main() { int i=0; while (1) { if (i>=7) break; i=i+1; } return i; }"),
        7
    );
    // do/while runs the body at least once
    assert_eq!(
        i32_of("int main() { int n=0; int i=0; do { n=n+i; i=i+1; } while (i<5); return n; }"),
        10
    );
    // continue in a while must still make progress (no infinite loop)
    assert_eq!(
        i32_of("int main() { int i=0; int s=0; while (i<10) { i=i+1; if (i%2==0) continue; s=s+i; } return s; }"),
        25
    );
    // nested loops: break only exits the inner loop
    assert_eq!(
        i32_of("int main() { int c=0; for (int i=0;i<3;i=i+1) for (int j=0;j<5;j=j+1) { if (j==2) break; c=c+1; } return c; }"),
        6
    );
}

#[test]
fn c_switch_end_to_end() {
    // basic dispatch + default
    let prog = "int classify(int n) { switch (n) { case 0: return 100; case 1: case 2: return 200; default: return 999; } } \
                int main() { return classify(0) + classify(1) + classify(2) + classify(5); }";
    assert_eq!(i32_of(prog), 100 + 200 + 200 + 999);
    // fall-through accumulation with break
    assert_eq!(
        i32_of("int main() { int x = 0; int n = 2; switch (n) { case 2: x = x + 2; case 1: x = x + 1; break; case 0: x = 99; } return x; }"),
        3
    );
    // break ends the switch; default in the middle
    assert_eq!(
        i32_of("int f(int n) { int r = 0; switch (n) { default: r = 7; break; case 1: r = 1; break; } return r; } \
                int main() { return f(1) * 10 + f(42); }"),
        17
    );
    // switch inside a loop: break exits the switch, not the loop
    assert_eq!(
        i32_of("int main() { int s = 0; for (int i = 0; i < 5; i = i + 1) { switch (i) { case 2: continue; case 4: break; } s = s + i; } return s; }"),
        8
    );
}

/// A tiny freestanding libc (`printf` family over the `write` builtin + varargs) prepended
/// to the printf tests — exercises the §3d flat-buffer varargs ABI through real code.
const LIBC: &str = r#"
#include <stdarg.h>
int write(int fd, char *buf, long n);

// A bump allocator over a pre-mapped window heap (§3d): the heap is just a big BSS
// global, malloc bumps within it, free is a no-op. (A real free-list / growth via the
// `map` capability is deferred — this is the MVP "fixed-size window" allocator.)
static char __heap[32768];
static long __heap_used = 0;
void *malloc(long n) {
  n = (n + 7) & ~7;
  if (n < 0 || __heap_used + n > 32768) return 0;
  char *p = __heap + __heap_used;
  __heap_used = __heap_used + n;
  return p;
}
void *calloc(long count, long size) {
  long n = count * size;
  char *p = malloc(n);
  if (p) for (long i = 0; i < n; i = i + 1) p[i] = 0;
  return p;
}
void free(void *p) {}

static void __putc(char c) { write(1, &c, 1); }
static void __puts(char *s) { while (*s) { __putc(*s); s = s + 1; } }
static void __putu(unsigned long v, int base) {
  char buf[24]; int i = 0;
  if (v == 0) { __putc('0'); return; }
  while (v) { int d = v % base; buf[i] = (d < 10 ? '0' + d : 'a' + d - 10); i = i + 1; v = v / base; }
  while (i) { i = i - 1; __putc(buf[i]); }
}
static void __putd(long v) {
  if (v < 0) { __putc('-'); __putu((unsigned long)(-v), 10); } else { __putu((unsigned long)v, 10); }
}
int printf(char *fmt, ...) {
  va_list ap; va_start(ap, fmt);
  int i = 0;
  while (fmt[i]) {
    if (fmt[i] != '%') { __putc(fmt[i]); i = i + 1; continue; }
    i = i + 1;
    int lng = 0;
    while (fmt[i] == 'l') { lng = 1; i = i + 1; }
    char c = fmt[i]; i = i + 1;
    if (c == 'd') { if (lng) __putd(va_arg(ap, long)); else __putd(va_arg(ap, int)); }
    else if (c == 'u') { if (lng) __putu(va_arg(ap, unsigned long), 10); else __putu(va_arg(ap, unsigned int), 10); }
    else if (c == 'x') { if (lng) __putu(va_arg(ap, unsigned long), 16); else __putu(va_arg(ap, unsigned int), 16); }
    else if (c == 'c') __putc((char)va_arg(ap, int));
    else if (c == 's') __puts(va_arg(ap, char*));
    else if (c == '%') __putc('%');
    else { __putc('%'); __putc(c); }
  }
  va_end(ap);
  return 0;
}
"#;

fn stdout_of(body: &str) -> String {
    let r = run_c_full(&format!("{LIBC}\n{body}"));
    String::from_utf8(r.stdout).expect("utf-8 stdout")
}

#[test]
fn c_varargs_end_to_end() {
    // a hand-rolled variadic function summing its args (the raw §3d ABI)
    assert_eq!(
        i32_of("#include <stdarg.h>\n\
                int sum(int n, ...) { va_list ap; va_start(ap, n); int s = 0; \
                  for (int i = 0; i < n; i = i + 1) s = s + va_arg(ap, int); va_end(ap); return s; } \
                int main() { return sum(4, 10, 20, 30, 40); }"),
        100
    );
    // mixed int/long/double through one va_list
    assert_eq!(
        i32_of("#include <stdarg.h>\n\
                int f(int n, ...) { va_list ap; va_start(ap, n); \
                  int a = va_arg(ap, int); long b = va_arg(ap, long); double c = va_arg(ap, double); \
                  va_end(ap); return a + (int)b + (int)c; } \
                int main() { return f(3, 100, 2000000000000L, 7.0); }"),
        100 + 2000000000000i64 as i32 + 7
    );
}

#[test]
fn c_function_pointers_end_to_end() {
    // A function pointer is a funcref index (§3c): assign a function, call through it.
    assert_eq!(
        i32_of(
            "int add(int a, int b){ return a + b; } \
             int main(){ int (*fp)(int,int) = add; return fp(3, 4); }"
        ),
        7
    );
    // Indirect call selected at runtime from an array of function pointers (a dispatch
    // table) — the common C idiom the function table exists for.
    assert_eq!(
        i32_of(
            "int add(int a,int b){return a+b;} int sub(int a,int b){return a-b;} \
             int mul(int a,int b){return a*b;} \
             int main(){ int (*ops[3])(int,int) = {add, sub, mul}; \
               int s = 0; for (int i=0;i<3;i++) s += ops[i](6, 2); return s; }"
        ),
        (6 + 2) + (6 - 2) + (6 * 2)
    );
    // A callback passed to another function (qsort-shaped: a fn taking a fn pointer),
    // and a nested indirect call.
    assert_eq!(
        i32_of(
            "int apply(int (*f)(int), int x){ return f(f(x)); } \
             int inc(int n){ return n + 1; } \
             int main(){ return apply(inc, 40); }"
        ),
        42
    );
    // A function pointer stored in a struct and called via `->`.
    assert_eq!(
        i32_of(
            "struct Op { int (*f)(int,int); int x, y; }; \
             int add(int a,int b){ return a + b; } \
             int main(){ struct Op o; o.f = add; o.x = 19; o.y = 23; \
               struct Op *p = &o; return p->f(p->x, p->y); }"
        ),
        42
    );
    // The explicit `(*fp)(...)` deref-call form, and a *void* function pointer whose only
    // effect is through a global.
    assert_eq!(
        i32_of(
            "int g; void set(int v){ g = v; } \
             int main(){ void (*fp)(int) = set; (*fp)(42); return g; }"
        ),
        42
    );
}

#[test]
fn c_goto_end_to_end() {
    // Forward goto: jump past the rest of a loop body to a label after it.
    assert_eq!(
        i32_of(
            "int main(){ int s=0; for(int i=0;i<5;i++){ if(i==3) goto done; s+=i; } \
                done: return s; }"
        ),
        1 + 2 // i = 0,1,2 before the goto at i==3
    );
    // Backward goto building a loop by hand (the label precedes the goto).
    assert_eq!(
        i32_of("int main(){ int i=0,s=0; loop: if(i<5){ s+=i; i++; goto loop; } return s; }"),
        1 + 2 + 3 + 4 // i = 0..4
    );
    // Multi-level loop exit — the classic reason goto survives in C.
    assert_eq!(
        i32_of(
            "int main(){ int s=0; for(int i=0;i<4;i++) for(int j=0;j<4;j++){ \
                if(i*j>=6) goto out; s++; } out: return s; }"
        ),
        // counts (i,j) pairs until i*j>=6: (0,*)4 (1,*)4 (2,0..2)3 → 11, then 2*3=6 stops
        11
    );
    // A `cleanup:` label reached by an early `goto` (the error-handling idiom), with a
    // promoted local threaded across the jump.
    assert_eq!(
        i32_of(
            "int main(){ int rc=0; int *p=0; if(!p){ rc=42; goto cleanup; } rc=1; \
                cleanup: return rc; }"
        ),
        42
    );
}

#[test]
fn c_global_pointer_relocations_end_to_end() {
    // A global pointer initialized with the address of another global (a relocation): the
    // frontend resolves the target's window offset at compile time into the data image.
    assert_eq!(
        i32_of("int x = 5; int *p = &x; int main(){ return *p; }"),
        5
    );
    // Pointer into an array element — exercises the relocation addend.
    assert_eq!(
        i32_of("int a[3] = {10, 20, 30}; int *p = &a[1]; int main(){ return *p; }"),
        20
    );
    // A pointer to a pointer (chained relocations).
    assert_eq!(
        i32_of("int x = 99; int *p = &x; int **pp = &p; int main(){ return **pp; }"),
        99
    );
    // A struct global mixing a pointer member (relocation) with a raw scalar.
    assert_eq!(
        i32_of(
            "struct S { int *p; int n; }; int v = 7; struct S s = {&v, 3}; \
             int main(){ return *s.p + s.n; }"
        ),
        10
    );
    // A global function-pointer table: each entry relocates to a funcref index (§3c), and an
    // indirect call dispatches through it — composes global relocations with part-1 calls.
    assert_eq!(
        i32_of(
            "int f(int x){ return x + 1; } int g(int x){ return x * 2; } \
             int (*tbl[2])(int) = {f, g}; \
             int main(){ return tbl[0](10) + tbl[1](10); }"
        ),
        11 + 20
    );
}

#[test]
fn c_by_value_aggregates_end_to_end() {
    // Struct passed by value (callee copies the caller's value into its own frame, §3d).
    assert_eq!(
        i32_of(
            "struct P { int x, y; }; \
             int sum(struct P p){ return p.x + p.y; } \
             int main(){ struct P p; p.x = 19; p.y = 23; return sum(p); }"
        ),
        42
    );
    // Struct *returned* by value (the sret ABI), then read back, assigned, and re-used.
    assert_eq!(
        i32_of(
            "struct P { int x, y; }; \
             struct P mk(int a, int b){ struct P p; p.x = a; p.y = b; return p; } \
             int main(){ struct P p = mk(30, 12); struct P q = p; /* whole-struct copy */ \
               return q.x + q.y; }"
        ),
        42
    );
    // A function taking *and* returning structs by value, plus a member of a call result.
    assert_eq!(
        i32_of(
            "struct P { int x, y; }; \
             struct P add(struct P a, struct P b){ struct P r; r.x=a.x+b.x; r.y=a.y+b.y; return r; } \
             struct P mk(int a,int b){ struct P p; p.x=a; p.y=b; return p; } \
             int main(){ struct P s = add(mk(1,2), mk(3,4)); return s.x*10 + s.y + mk(0,26).y; }"
        ),
        // s = (4, 6); 4*10 + 6 + 26 = 72
        72
    );
    // An odd-sized struct (13 bytes: int + char + int) exercises the 4/2/1 memcpy chunks,
    // and a struct >16 bytes (no register classification — always by pointer, §3d).
    assert_eq!(
        i32_of(
            "struct Q { int a; char b; int c; }; \
             struct Q bump(struct Q q){ q.a++; q.b++; q.c++; return q; } \
             int main(){ struct Q q; q.a=10; q.b=20; q.c=30; struct Q r = bump(q); \
               return r.a + r.b + r.c; }"
        ),
        63
    );
    // A union by value round-trips its active member.
    assert_eq!(
        i32_of(
            "union U { int i; char c[4]; }; \
             union U pass(union U u){ return u; } \
             int main(){ union U u; u.i = 0x41424344; union U v = pass(u); return v.c[0]; }"
        ),
        0x44
    );
}

/// I2 (§2a/§3c): an indirect call re-checks the selected function's signature at the use
/// site, so a function pointer cast to the wrong type — here a no-arg `int(*)(void)` aimed
/// at a 2-arg function — **traps** (the function-table type-id check), identically on both
/// backends, rather than calling with a mismatched frame. A type-confused index is inert,
/// never an escape.
#[test]
fn c_function_pointer_signature_mismatch_traps() {
    let src = "int two(int a, int b){ return a + b; } \
               int main(){ int (*w)(void) = (int(*)(void))two; return w(); }";
    let ir = c_to_ir(src);
    assert!(
        ir.contains("call_indirect"),
        "expected an indirect call:\n{ir}"
    );
    let m = parse_module(&ir).expect("parse");
    verify_module(&m).expect("verify");

    let mut hi = Host::new();
    let mut hj = Host::new();
    let grant = |h: &mut Host| {
        [
            Value::I32(h.grant_stream(StreamRole::Out)),
            Value::I32(h.grant_stream(StreamRole::In)),
            Value::I32(h.grant_exit()),
            Value::I32(h.grant_memory()),
        ]
    };
    let args = grant(&mut hi);
    grant(&mut hj);
    let mut fuel = 50_000_000u64;
    let interp = run_with_host(&m, 0, &args, &mut fuel, &mut hi);
    let slots: Vec<i64> = args.iter().copied().map(to_slot).collect();
    let jit = compile_and_run_with_host(
        &m,
        0,
        &slots,
        cap_thunk,
        &mut hj as *mut Host as *mut c_void,
    )
    .expect("jit compiles");
    assert_eq!(
        interp,
        Err(Trap::IndirectCallType),
        "interp must trap on the signature mismatch\n{ir}"
    );
    assert!(
        matches!(jit, JitOutcome::Trapped(TrapKind::IndirectCallType)),
        "JIT must trap on the signature mismatch, got {jit:?}\n{ir}"
    );
}

#[test]
fn c_printf_end_to_end() {
    assert_eq!(
        stdout_of("int main() { printf(\"hello, world\\n\"); return 0; }"),
        "hello, world\n"
    );
    assert_eq!(
        stdout_of("int main() { printf(\"%d + %d = %d\\n\", 2, 3, 2 + 3); return 0; }"),
        "2 + 3 = 5\n"
    );
    assert_eq!(
        stdout_of("int main() { printf(\"%s is %d\\n\", \"answer\", 42); return 0; }"),
        "answer is 42\n"
    );
    assert_eq!(
        stdout_of("int main() { printf(\"%c%c%c\", 'a', 'b', 'c'); return 0; }"),
        "abc"
    );
    assert_eq!(
        stdout_of("int main() { printf(\"%x %x\\n\", 255, 4096); return 0; }"),
        "ff 1000\n"
    );
    assert_eq!(
        stdout_of("int main() { printf(\"neg %d\\n\", -17); return 0; }"),
        "neg -17\n"
    );
    assert_eq!(
        stdout_of("int main() { printf(\"%ld\\n\", 5000000000L); return 0; }"),
        "5000000000\n"
    );
    assert_eq!(
        stdout_of("int main() { printf(\"100%%\\n\"); return 0; }"),
        "100%\n"
    );
    // printf driven by a loop (FizzBuzz-ish)
    assert_eq!(
        stdout_of(
            "int main() { for (int i = 1; i <= 5; i = i + 1) printf(\"%d \", i * i); return 0; }"
        ),
        "1 4 9 16 25 "
    );
}

#[test]
fn c_branching_operands_spill_regression() {
    // A value computed before a control-flow sub-expression must survive the branch it
    // opens (it is spilled to scratch and reloaded in the merge block). Earlier tests used
    // a *single* &&/||/?: as the whole expression and so missed this stranding bug.
    assert_eq!(
        i32_of(
            "int main() { int x = 5; return (x > 0 && x < 10) + (0 || 3) + (x > 3 ? 100 : 7); }"
        ),
        1 + 1 + 100
    );
    // a branching argument among several call arguments
    assert_eq!(
        i32_of(
            "int add3(int a, int b, int c) { return a + b + c; } \
                int main() { int x = 2; return add3(10, x > 0 ? 5 : 9, 100); }"
        ),
        115
    );
    // assignment whose rhs branches: the lhs address must survive
    assert_eq!(
        i32_of("int main() { int a[2]; a[0] = 0; a[1] = 0; int i = 1; a[i] = (i ? 42 : 7); return a[1]; }"),
        42
    );
}

/// Run `LIBC + body` and return the i32 result of `main` (for malloc tests etc.).
fn i32_libc(body: &str) -> i32 {
    match run_c_full(&format!("{LIBC}\n{body}")).outcome {
        Outcome::Returned(v) => match v.as_slice() {
            [Value::I32(x)] => *x,
            other => panic!("expected one i32, got {other:?}"),
        },
        Outcome::Exited(c) => panic!("unexpected exit({c})"),
    }
}

#[test]
fn c_malloc_end_to_end() {
    // allocate, fill, sum, free (free is a no-op bump allocator)
    assert_eq!(
        i32_libc(
            "int main() { int *a = malloc(5 * sizeof(int)); \
                  for (int i=0;i<5;i=i+1) a[i]=i*i; int s=0; for (int i=0;i<5;i=i+1) s=s+a[i]; \
                  free(a); return s; }"
        ),
        30
    );
    // two allocations are disjoint
    assert_eq!(
        i32_libc(
            "int main() { int *a = malloc(8); int *b = malloc(8); \
                  *a = 11; *b = 22; return *a + *b + (a == b ? 1000 : 0); }"
        ),
        33
    );
    // calloc zero-initializes
    assert_eq!(
        i32_libc(
            "int main() { int *a = calloc(4, sizeof(int)); int s = 0; \
                  for (int i=0;i<4;i=i+1) s=s+a[i]; a[2]=7; return s + a[2]; }"
        ),
        7
    );
    // a heap-allocated linked list
    assert_eq!(
        i32_libc("struct N { int v; struct N *next; }; \
                  int main() { struct N *head = 0; \
                    for (int i=1;i<=5;i=i+1) { struct N *n = malloc(sizeof(struct N)); n->v=i; n->next=head; head=n; } \
                    int s=0; for (struct N *p=head; p; p=p->next) s=s+p->v; return s; }"),
        15
    );
}

#[test]
fn c_malloc_with_printf_end_to_end() {
    // build a dynamic array and print it
    assert_eq!(
        stdout_of(
            "int main() { int n = 4; int *a = malloc(n * sizeof(int)); \
                   for (int i=0;i<n;i=i+1) a[i] = (i+1)*(i+1); \
                   for (int i=0;i<n;i=i+1) printf(\"%d \", a[i]); free(a); return 0; }"
        ),
        "1 4 9 16 "
    );
}

// ---- differential against native `cc`: same C source, two compilers, compare ----
//
// The VM runs `LIBC + body` (our printf/malloc over the powerbox); native `cc` runs
// `<real headers> + body` (the system printf/malloc). Identical observable behaviour
// (process exit code + stdout) validates our frontend's *C semantics* against a real
// compiler — a check that interp/JIT agreement alone cannot give.

/// Compile `src` with native `cc` and run it; return (exit code low byte, stdout).
fn native_run(src: &str) -> (u8, Vec<u8>) {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static N: AtomicUsize = AtomicUsize::new(0);
    let id = N.fetch_add(1, Ordering::Relaxed);
    let base = std::env::temp_dir().join(format!("svm_gcc_{}_{id}", std::process::id()));
    let cfile = base.with_extension("c");
    let exe = base.with_extension("exe");
    std::fs::write(&cfile, src).unwrap();
    let status = Command::new("cc")
        .args(["-w", "-O0", "-o"])
        .arg(&exe)
        .arg(&cfile)
        .status()
        .expect("run cc");
    assert!(status.success(), "native cc failed to compile:\n{src}");
    let out = Command::new(&exe).output().expect("run native exe");
    let code = out.status.code().unwrap_or(-1) as u8;
    (code, out.stdout)
}

/// Assert our VM and native `cc` agree on a program's exit code and stdout.
fn assert_matches_gcc(body: &str) {
    let vm = run_c_full(&format!("{LIBC}\n{body}"));
    let (vm_code, vm_out) = match vm.outcome {
        Outcome::Returned(ref v) => match v.as_slice() {
            [Value::I32(x)] => (*x as u8, vm.stdout.clone()),
            [Value::I64(x)] => (*x as u8, vm.stdout.clone()),
            other => panic!("unexpected result {other:?} for:\n{body}"),
        },
        Outcome::Exited(c) => (c as u8, vm.stdout.clone()),
    };
    let native_src = format!(
        "#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n#include <unistd.h>\n{body}"
    );
    let (g_code, g_out) = native_run(&native_src);
    assert_eq!(
        vm_code, g_code,
        "exit code: VM={vm_code} cc={g_code} for:\n{body}"
    );
    assert_eq!(
        String::from_utf8_lossy(&vm_out),
        String::from_utf8_lossy(&g_out),
        "stdout differs from cc for:\n{body}"
    );
}

#[test]
fn c_matches_gcc_arithmetic_and_control_flow() {
    assert_matches_gcc("int main() { return 6 * 7 - 1; }");
    assert_matches_gcc("int main() { return 100 / 7 + 100 % 7; }");
    assert_matches_gcc("int main() { return (0xff & 0x3c) | (1 << 5); }");
    assert_matches_gcc("int main() { int s=0; for (int i=1;i<=20;i++) s+=i; return s; }");
    assert_matches_gcc(
        "int main() { int x=5; return (x>0 && x<10) + (0 || 3) + (x>3 ? 100 : 7); }",
    );
    assert_matches_gcc("int main() { int s=0,i=0; do { s+=i; i++; } while (i<6); return s; }");
    assert_matches_gcc(
        "int main() { int s=0; for (int i=0;i<6;i++){ switch(i){case 2:continue;case 5:break;} s+=i; } return s; }",
    );
}

#[test]
fn c_matches_gcc_functions_and_recursion() {
    assert_matches_gcc(
        "int fib(int n){ if(n<2)return n; return fib(n-1)+fib(n-2);} int main(){ return fib(13); }",
    );
    assert_matches_gcc(
        "int gcd(int a,int b){ while(b){ int t=b; b=a%b; a=t; } return a; } int main(){ return gcd(1071, 462); }",
    );
    assert_matches_gcc(
        "int ack(int m,int n){ if(m==0)return n+1; if(n==0)return ack(m-1,1); return ack(m-1, ack(m,n-1)); } int main(){ return ack(2,3); }",
    );
}

#[test]
fn c_matches_gcc_function_pointers() {
    // A comparator-driven selection sort (qsort-shaped) plus printf output, validated
    // against native `cc` — function pointers through a real algorithm, both orderings.
    assert_matches_gcc(
        "int asc(int a,int b){ return a - b; } int desc(int a,int b){ return b - a; } \
         void sort(int *v, int n, int (*cmp)(int,int)){ \
           for (int i=0;i<n;i++) for (int j=i+1;j<n;j++) \
             if (cmp(v[j], v[i]) < 0){ int t=v[i]; v[i]=v[j]; v[j]=t; } } \
         int main(){ int a[5] = {5,3,1,4,2}; \
           sort(a, 5, asc);  for (int i=0;i<5;i++) printf(\"%d \", a[i]); printf(\"\\n\"); \
           sort(a, 5, desc); for (int i=0;i<5;i++) printf(\"%d \", a[i]); printf(\"\\n\"); \
           return a[0]; }",
    );
    // A runtime-indexed dispatch table.
    assert_matches_gcc(
        "int add(int a,int b){ return a + b; } int sub(int a,int b){ return a - b; } \
         int main(){ int (*ops[2])(int,int) = {add, sub}; int s = 0; \
           for (int i=0;i<2;i++) s += ops[i](10, 3); printf(\"%d\\n\", s); return s; }",
    );
}

#[test]
fn c_matches_gcc_goto() {
    // A hand-rolled state machine driven by goto, validated against native `cc`.
    assert_matches_gcc(
        "int main(){ int n=27, steps=0; \
         start: if(n==1) goto done; \
           if(n%2==0){ n=n/2; steps++; goto start; } \
           n=3*n+1; steps++; goto start; \
         done: printf(\"%d\\n\", steps); return steps; }",
    );
    // Forward goto skipping initialization, plus a backward goto retry loop.
    assert_matches_gcc(
        "int main(){ int tries=0; \
         retry: tries++; if(tries<3) goto retry; \
           int sum=0; for(int i=0;i<5;i++){ if(i==2) goto skip; sum+=i; skip:; } \
           printf(\"%d %d\\n\", tries, sum); return sum; }",
    );
}

#[test]
fn c_matches_gcc_global_relocations() {
    // A global char* to a string literal (the relocation targets read-only data), printed.
    assert_matches_gcc(
        "char *greeting = \"hello, globals\\n\"; \
         int main(){ printf(\"%s\", greeting); return 0; }",
    );
    // An array of string pointers — a classic relocation-heavy table.
    assert_matches_gcc(
        "char *days[3] = {\"Mon\", \"Tue\", \"Wed\"}; \
         int main(){ for(int i=0;i<3;i++) printf(\"%s \", days[i]); printf(\"\\n\"); return 0; }",
    );
    // A global dispatch table of function pointers, called in a loop.
    assert_matches_gcc(
        "int add(int a,int b){return a+b;} int mul(int a,int b){return a*b;} \
         int (*ops[2])(int,int) = {add, mul}; \
         int main(){ int s=0; for(int i=0;i<2;i++) s+=ops[i](6,7); printf(\"%d\\n\", s); return s; }",
    );
}

#[test]
fn c_matches_gcc_static_assert() {
    // C11 `_Static_assert(const-expr, msg);` — a compile-time check that emits nothing when the
    // expression is non-zero, at file *and* block scope. chibicc previously treated it as a
    // function call (`implicit declaration`). Found via xxHash. (Only the `_Static_assert`
    // keyword is exercised here — the C23 `static_assert` spelling needs <assert.h>/C23 on gcc.)
    assert_matches_gcc(
        "_Static_assert(sizeof(int) == 4, \"int is 4 bytes\"); \
         _Static_assert(sizeof(long) == 8, \"LP64 long\"); \
         int main(){ _Static_assert(1 + 1 == 2, \"arithmetic\"); \
           int n = 0; for (int i = 0; i < 5; i++) { _Static_assert(2 * 3 == 6, \"loop\"); n += i; } \
           printf(\"%d %d %d\\n\", (int)sizeof(int), (int)sizeof(long), n); return n; }",
    );
}

#[test]
fn c_matches_gcc_packed_enums() {
    // `enum __attribute__((packed))` sizes to the smallest integer type holding its values
    // (gcc semantics), so a struct containing small enums has the **same layout** as gcc —
    // which matters for host↔guest data exchange (§3d pins x86-64-SysV layout). chibicc
    // previously made every enum `int` (4 bytes); found via Clay. `sizeof`/offsets and the
    // exit code are all compared to native `cc`.
    assert_matches_gcc(
        "typedef enum __attribute__((__packed__)) { A, B, C } E; \
         struct S { E first; int x; E second; char c; }; \
         int main(){ struct S s = { .first=B, .x=7, .second=C, .c='z' }; \
           printf(\"%d %d %d %d %d\\n\", (int)sizeof(E), (int)sizeof(struct S), \
                  s.first, s.second, s.c); \
           return (int)sizeof(struct S); }",
    );
    // A packed enum forced wider by its value range (> 255 → 2 bytes), still matching gcc.
    assert_matches_gcc(
        "typedef enum __attribute__((packed)) { LO = 0, HI = 1000 } W; \
         int main(){ printf(\"%d\\n\", (int)sizeof(W)); return (int)sizeof(W); }",
    );
}

#[test]
fn c_matches_gcc_clay_fixes() {
    // Four frontend/IR fixes surfaced by compiling the Clay layout library, each validated
    // against native `cc`.
    // (a) ternary `?:` returning a struct by value (gen_cond aggregate result → i64 address):
    assert_matches_gcc(
        "struct P{int x,y;}; struct P pick(int c){ struct P a={1,2},b={3,4}; return c?a:b; } \
         int main(){ struct P p=pick(1); printf(\"%d %d\\n\",p.x,p.y); return p.x+p.y; }",
    );
    // (b) struct return > 16 bytes (chibicc prepends a hidden return-buffer param; no longer
    //     double-counted against our own sret):
    assert_matches_gcc(
        "struct Big{int a,b,c,d,e;}; struct Big mk(int x){ struct Big b={x,x+1,x+2,x+3,x+4}; return b; } \
         int main(){ struct Big b=mk(10); printf(\"%d\\n\", b.a+b.e); return b.a+b.e; }",
    );
    // (c) mixed-width shift `uint64_t << int` (shift amount widened to the value's width):
    assert_matches_gcc(
        "int main(){ unsigned long h=12345; h+=(h<<10); h^=(h>>6); printf(\"%lu\\n\",h); return (int)(h&255); }",
    );
    // (d) an unsigned 32-bit constant 0xFFFFFFFF (`i32.const` accepts the full u32 range):
    assert_matches_gcc(
        "int main(){ unsigned int m=0xFFFFFFFFu; printf(\"%u\\n\", m); return (int)(m & 7); }",
    );
}

#[test]
fn c_matches_gcc_aggregates() {
    // Two upstream chibicc parser bugs, found via the Clay layout library and fixed in
    // `frontend/chibicc/parse.c`, both around designated initializers into **anonymous**
    // aggregates (pervasive in real C). (1) `struct_designator` only special-cased anonymous
    // *structs*, so a designator targeting an anonymous *union* member NULL-derefed (segfault).
    // (2) `struct_initializer2` didn't skip the separator comma when a designated member landed
    // in a nested anonymous aggregate, so a following designator failed to parse. Validate
    // both against native `cc`.
    assert_matches_gcc(
        "struct S { union { int config; struct { int tc; int data; }; }; int flag; }; \
         int main(){ struct S a = { .tc = 5, .flag = 1 };      /* fix (1)+(2) */ \
                     struct S b = { .flag = 9, .config = 7 };  /* anonymous-union member */ \
                     printf(\"%d %d %d %d\\n\", a.tc, a.flag, b.config, b.flag); \
                     return a.tc + a.flag; }",
    );
    // Regression (demos/rational.c): a struct returned from a **non-entry block** — inside a
    // loop, and after it. The sret pointer is a parameter that only lives in the entry block,
    // so it is stashed to a frame slot and reloaded on return; earlier it was read from the
    // block's rebound parameter (the loop counter), emitting IR that failed verification.
    assert_matches_gcc(
        "struct P { int x, y; }; \
         struct P pick(int n){ struct P r; \
           for (int i=0;i<100;i++) if (i==n) { r.x=i; r.y=i*i; return r; } \
           r.x=-1; r.y=-1; return r; } \
         int main(){ struct P p = pick(7); printf(\"%d,%d\\n\", p.x, p.y); return p.x + p.y; }",
    );
    // Structs by value through args and returns (the sret ABI), with printf output,
    // validated against native `cc`.
    assert_matches_gcc(
        "struct V { int x, y, z; }; \
         struct V add(struct V a, struct V b){ \
           struct V r; r.x=a.x+b.x; r.y=a.y+b.y; r.z=a.z+b.z; return r; } \
         int dot(struct V a, struct V b){ return a.x*b.x + a.y*b.y + a.z*b.z; } \
         int main(){ struct V a = {1,2,3}, b = {4,5,6}; \
           struct V s = add(a, b); \
           printf(\"%d %d %d\\n\", s.x, s.y, s.z); \
           printf(\"%d\\n\", dot(a, b)); \
           return s.x + s.y + s.z; }",
    );
    // A struct returned from one call fed straight into another, plus a function pointer
    // whose signature passes/returns a struct by value (composes increment 1 + 2).
    assert_matches_gcc(
        "struct P { int x, y; }; \
         struct P mk(int a, int b){ struct P p; p.x=a; p.y=b; return p; } \
         struct P swap(struct P p){ struct P r; r.x=p.y; r.y=p.x; return r; } \
         int main(){ struct P (*f)(struct P) = swap; \
           struct P p = f(mk(7, 9)); \
           printf(\"%d,%d\\n\", p.x, p.y); return p.x - p.y; }",
    );
}

#[test]
fn c_matches_gcc_floats() {
    assert_matches_gcc("int main() { double x=3.5,y=2.25; return (int)(x*y*100); }");
    assert_matches_gcc(
        "int main() { float a=1.5f; double s=0; for(int i=0;i<10;i++) s+=a*i; return (int)s; }",
    );
    assert_matches_gcc(
        "int main() { double x=100; for(int i=0;i<5;i++) x=x/2 + 1; return (int)(x*16); }",
    );
}

#[test]
fn c_matches_gcc_printf() {
    assert_matches_gcc("int main(){ printf(\"%d %d %d\\n\", 1, -2, 30000); return 0; }");
    assert_matches_gcc("int main(){ printf(\"%s=%d, %x, %ld%%\\n\", \"k\", 42, 3735928559, 9000000000L); return 0; }");
    assert_matches_gcc(
        "int main(){ for(int i=1;i<=12;i++) printf(\"%d \", i*i); printf(\"\\n\"); return 0; }",
    );
    assert_matches_gcc(
        "int main(){ char *s=\"hello\"; for(int i=0;s[i];i++) printf(\"%c.\", s[i]); return 0; }",
    );
}

#[test]
fn c_matches_gcc_malloc_and_data_structures() {
    // bubble sort a malloc'd array, print it
    assert_matches_gcc(
        "int main(){ int n=8; int *a=malloc(n*sizeof(int)); int seed[8]={5,2,8,1,9,3,7,4}; \
         for(int i=0;i<n;i++) a[i]=seed[i]; \
         for(int i=0;i<n;i++) for(int j=0;j<n-1-i;j++) if(a[j]>a[j+1]){int t=a[j];a[j]=a[j+1];a[j+1]=t;} \
         for(int i=0;i<n;i++) printf(\"%d\", a[i]); free(a); return 0; }",
    );
    // sieve of Eratosthenes, count primes < 100
    assert_matches_gcc(
        "int main(){ int n=100; char *p=calloc(n,1); int c=0; \
         for(int i=2;i<n;i++) if(!p[i]){ c++; for(int j=i*2;j<n;j+=i) p[j]=1; } return c; }",
    );
    // linked list of structs
    assert_matches_gcc(
        "struct N{int v; struct N*next;}; int main(){ struct N*h=0; \
         for(int i=1;i<=6;i++){ struct N*x=malloc(sizeof(struct N)); x->v=i*i; x->next=h; h=x; } \
         int s=0; for(struct N*p=h;p;p=p->next) s+=p->v; return s; }",
    );
}

#[test]
fn c_matches_gcc_ssa_promotion() {
    // Exercise the SSA-promotion pass (DESIGN §3d): scalar locals that are never
    // address-taken become real SSA values threaded through block params. These programs
    // lean on the cases that pass relies on — and would break if promotion were wrong:
    // every compound-assignment / inc-dec flavour (which chibicc desugars through `&x`,
    // un-desugared by the frontend), promoted values crossing loop back-edges and
    // &&/||/?: merges, and a promoted local read before assignment on one path.
    assert_matches_gcc(
        "int main(){ int a=1,b=2,c=3; a+=b; b*=c; c-=a; a<<=1; b|=5; c%=4; \
         return a*1000 + b*10 + c; }",
    );
    assert_matches_gcc(
        "int main(){ int i=0,s=0; while(i<10){ s+=i*i; ++i; } i=10; do { s-=i--; } while(i); return s; }",
    );
    assert_matches_gcc(
        "int main(){ int x=0; for(int i=0;i<8;i++){ x += (i&1) ? i*2 : -i; if(i==5) x++; } return x; }",
    );
    // A promoted local assigned only inside a conditional, then read after the merge.
    assert_matches_gcc("int main(){ int x=7; int y; if(x>3) y=x*x; else y=0; return y; }");
    // A long accumulator and a promoted pointer walking a local array (`arr` is
    // address-taken so it stays in memory; `p` and `sum` promote).
    assert_matches_gcc(
        "int main(){ int arr[5]={4,8,15,16,23}; long sum=0; \
         for(int *p=arr; p<arr+5; p++) sum += *p; return (int)sum; }",
    );
    // Post- vs pre-increment used for their values (sequenced, so well-defined).
    assert_matches_gcc("int main(){ int i=5; int j=i++; int k=++i; return i*100 + j*10 + k; }");
}

/// Structural guard for the headline §3 SSA-promotion win: a hot loop over non-address-taken
/// scalars must lower to **zero loop-body memory ops** ("~22 → 0" in the design notes). The
/// `c_matches_gcc_*` tests only pin *correctness*, which the everything-in-memory lowering
/// also satisfies — so they would stay green if promotion silently stopped firing. This pins
/// the optimization itself: it fails the moment a promotable loop body goes back to memory.
#[test]
fn c_ssa_promotion_eliminates_loop_body_memory_ops() {
    // Each: a tight loop whose scalars (accumulator, counter) are never address-taken, so
    // every one promotes to a block param — the loop body should touch memory zero times.
    // (A `?:`/`&&`/`||` *inside* the loop is deliberately avoided here: it spills "stranded"
    // sub-expression values to scratch by design — see §7a — which is correct but not zero.)
    let promoted = [
        "int main(){ int s=0; for(int i=0;i<10;i++) s += i*i; return s; }",
        "int main(){ int i=0,s=0; while(i<10){ s+=i; s*=2; ++i; } return s; }",
        "int main(){ long acc=0; for(int i=0;i<20;i++) acc += (long)i*3 - (i>>1); return (int)acc; }",
    ];
    for src in promoted {
        let ir = c_to_ir(src);
        assert_eq!(
            loop_region_mem_ops(&ir),
            0,
            "SSA promotion should leave zero loop-body memory ops for:\n{src}\n--- IR ---\n{ir}"
        );
        // Belt and suspenders: the promoted form must still compute the right answer.
        run_c(src);
    }

    // Non-vacuous control: take the accumulator's address, forcing it to stay in memory. The
    // metric must now *see* the loop's load/store — otherwise the zeros above prove nothing.
    let in_memory = "int main(){ int s=0; int *p=&s; for(int i=0;i<10;i++) *p += i*i; return s; }";
    let ir = c_to_ir(in_memory);
    assert!(
        loop_region_mem_ops(&ir) > 0,
        "an address-taken accumulator must keep loop-body memory ops (else the guard is blind):\n{ir}"
    );
}
