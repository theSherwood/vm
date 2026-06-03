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

use core::ffi::c_void;
use svm_interp::{run_with_host, GuestMem, Host, StreamRole, Trap, Value, WindowMem};
use svm_ir::ValType;
use svm_jit::{compile_and_run_with_host, JitOutcome, TrapKind, EXIT_CODE};
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

/// Bridge the JIT's capability-thunk ABI to the interpreter's `Host` — the host
/// trampoline a real embedder supplies (§9). Mirrors the one in `jit_diff.rs`, so both
/// backends share capability semantics.
///
/// # Safety
/// Honours the `CapThunk` contract: `ctx` is a `*mut Host`, the slot/window pointers are
/// valid for their lengths, and `trap_out` is live.
unsafe extern "C" fn cap_thunk(
    ctx: *mut c_void,
    mem_base: *mut u8,
    mem_size: u64,
    type_id: u32,
    op: u32,
    handle: i32,
    args: *const i64,
    n_args: u64,
    results: *mut i64,
    n_results: u64,
    trap_out: *mut i64,
) {
    let host = &mut *(ctx as *mut Host);
    let arg_slots = std::slice::from_raw_parts(args, n_args as usize);
    let mut empty: [u8; 0] = [];
    let window: &mut [u8] = if mem_base.is_null() {
        &mut empty
    } else {
        std::slice::from_raw_parts_mut(mem_base, mem_size as usize)
    };
    let mut wm = WindowMem::new(window, mem_size);
    let gm: Option<&mut dyn GuestMem> = if mem_base.is_null() {
        None
    } else {
        Some(&mut wm)
    };
    match host.cap_dispatch_slots(type_id, op, handle, arg_slots, gm) {
        Ok(res) => {
            let out = std::slice::from_raw_parts_mut(results, n_results as usize);
            for (o, r) in out.iter_mut().zip(res) {
                *o = r;
            }
            *trap_out = 0;
        }
        Err(Trap::Exit(code)) => *trap_out = EXIT_CODE as i64 | ((code as i64) << 32),
        Err(_) => *trap_out = TrapKind::CapFault as i64,
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

    // `_start(stdout, stdin, exit)` takes the powerbox handles. Grant them identically on
    // both hosts (grants are deterministic, so the handle values match).
    let mut hi = Host::new();
    let mut hj = Host::new();
    let grant = |h: &mut Host| {
        [
            Value::I32(h.grant_stream(StreamRole::Out)),
            Value::I32(h.grant_stream(StreamRole::In)),
            Value::I32(h.grant_exit()),
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
