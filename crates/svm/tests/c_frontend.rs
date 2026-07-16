//! End-to-end C: the vendored chibicc fork (`frontend/chibicc`, `--emit-ir`) compiles C
//! to our text IR, which we verify and run on the reference interpreter. This is the
//! Phase-2 "it works" milestone (`DESIGN.md` §18) — real C through the whole pipeline.
//!
//! Each test runs `main` on **both** the interpreter and the JIT and asserts they agree
//! (results + captured stdout/exit), so every test doubles as a JIT differential test. A
//! second tier (`c_matches_gcc_*`) compiles the *same* C with native `cc` and compares
//! exit code + stdout, validating C semantics against a real compiler.
//!
//! Requires a unix C toolchain (`make` + `cc`) to build the chibicc fork, so this suite is gated to
//! `#![cfg(unix)]` — Windows lacks the toolchain. It runs on **both** Linux (4 KiB pages) and
//! macOS-ARM (16 KiB pages): the frontend now pins its RO-data isolation and heap-growth granularity
//! to the largest common host page (16 KiB, a multiple of 4 KiB), so a guest's read-only segment
//! never shares a host page with writable data and its writes don't trip a spurious RO
//! over-protection fault on a 16 KiB host (Phase 3.5 §16). The frontend is outside the escape-TCB
//! (§2a): whatever IR it emits still goes through the verifier, and the JIT/PAL it exercises is
//! validated cross-platform by `jit_fuzz`/`escape_oracle` + the `svm-jit` PAL conformance test.
#![cfg(unix)]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use core::ffi::c_void;
use svm_interp::{
    run_scheduled, run_with_host, Host, Inspector, IrPc, StreamRole, Trap, Value, VarValue,
};
use svm_jit::{compile_and_run_with_host, JitOutcome, TrapKind};
use svm_run::cap_thunk; // the shared JIT-CapThunk → reference-Host bridge (§9)
use svm_text::parse_module as parse_module_raw;
use svm_verify::verify_module;

/// Parse frontend (chibicc) text IR and resolve its §7 named capability imports under the
/// reference host policy ([`svm_run::resolve_capability_imports`]). The frontend now emits
/// `call.import "<name>"` for capabilities instead of inline `cap.call`, so every harness parses
/// through this — the *resolved* (import-free) module is what verifies and runs. A no-op for
/// hand-written test IR that has no imports; an unresolved name is a frontend bug, so it panics.
fn parse_module(ir: &str) -> Result<svm_ir::Module, svm_text::ParseError> {
    let m = parse_module_raw(ir)?;
    Ok(svm_run::resolve_capability_imports(m)
        .unwrap_or_else(|e| panic!("resolve capability imports: {e}")))
}

fn to_slot(v: Value) -> i64 {
    match v {
        Value::I32(x) => x as i64,
        Value::I64(x) => x,
        Value::F32(x) => x.to_bits() as i64,
        Value::F64(x) => x.to_bits() as i64,
        Value::V128(b) => i64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]),
        Value::Ref(x) => x as i64,
    }
}

/// The fixed **8-handle powerbox** a chibicc `_start` imports — stdout, stdin, exit, memory,
/// addrspace (§14), ioring + blocking (§9/§12), jit (DESIGN.md §22) — granted in that order so the
/// handle values are deterministic (and identical across two hosts). Every entry imports the
/// same set (one `_start` shape); a guest that never touches the ring or the JIT just leaves
/// those handles stashed and unused. `block_for` is the mock Blocking op's duration — `ZERO`
/// for ordinary programs, non-zero for an async demo that wants its I/O to actually block.
fn powerbox(h: &mut Host, win: u64, block_for: std::time::Duration) -> [Value; 8] {
    h.set_region_factory(svm_run::new_shared_region);
    h.set_jit_validator(svm_run::jit_blob_validator);
    let mem_log2 = (win != 0).then(|| win.trailing_zeros() as u8);
    [
        Value::I32(h.grant_stream(StreamRole::Out)),
        Value::I32(h.grant_stream(StreamRole::In)),
        Value::I32(h.grant_exit()),
        Value::I32(h.grant_memory()),
        Value::I32(h.grant_address_space(0, win)),
        Value::I32(h.grant_io_ring()),
        Value::I32(h.grant_blocking(block_for, None)),
        Value::I32(h.grant_jit(mem_log2)),
    ]
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

/// Like [`c_to_ir`] but with `-g`: emit the debug-info section (and, as `-Og`, disable SSA
/// promotion so locals keep stable window slots — DEBUGGING.md §6/W6).
fn c_to_ir_g(src: &str) -> String {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static N: AtomicUsize = AtomicUsize::new(0);
    let id = N.fetch_add(1, Ordering::Relaxed);
    let base = std::env::temp_dir().join(format!("svm_cfeg_{}_{id}", std::process::id()));
    let cfile = base.with_extension("c");
    let irfile = base.with_extension("svm");
    std::fs::write(&cfile, src).unwrap();
    let status = Command::new(chibicc())
        .args([
            "-cc1",
            "--emit-ir",
            "-g",
            "-cc1-input",
            cfile.to_str().unwrap(),
            "-cc1-output",
            irfile.to_str().unwrap(),
            cfile.to_str().unwrap(),
        ])
        .status()
        .expect("run chibicc");
    assert!(status.success(), "chibicc -g failed on:\n{src}");
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
///
/// Driven through the public, frontend-independent embedding API (F1): [`svm_run::instantiate`]
/// resolves + verifies (the resolve is a no-op here — [`parse_module`] already lowered the §7
/// imports), and [`svm_run::Instance::run_diff`] runs `_start` on the tree-walker *and* the JIT under
/// the fixed §3e powerbox and asserts they agree (results, stdout, stderr) — the same grant/compile/run
/// core the CLI (`run_powerbox`) and embedders use, including the concurrent-guest `Mutex<Host>` path
/// and the guest-driven `Jit` capability. A divergence/trap surfaces as an `Err`, re-panicked with the
/// C source + IR for a legible failure.
fn run_c_full(src: &str) -> CRun {
    let ir = c_to_ir(src);
    let m =
        parse_module(&ir).unwrap_or_else(|e| panic!("parse IR failed: {e:?}\n--- IR ---\n{ir}"));
    let inst = svm_run::instantiate(m)
        .unwrap_or_else(|e| panic!("instantiate failed: {e}\n--- IR ---\n{ir}"));
    let run = inst
        .run_diff(&svm_run::RunConfig::default())
        .unwrap_or_else(|e| panic!("interp/JIT differential failed: {e}\n{src}\n--- IR ---\n{ir}"));
    let outcome = match run.outcome {
        svm_run::Outcome::Returned(v) => Outcome::Returned(v),
        svm_run::Outcome::Exited(c) => Outcome::Exited(c),
    };
    CRun {
        outcome,
        stdout: run.stdout,
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

/// Like [`run_c_full`] but **interpreter-only** — for programs using §12 fibers, which the
/// JIT does not yet lower (it bails `Unsupported`, step 4), so the differential `run_c_full`
/// cannot drive them. The frontend → verifier → reference-interpreter path is the full story
/// for fibers today. Returns the outcome and captured stdout.
fn run_c_interp(src: &str) -> CRun {
    let ir = c_to_ir(src);
    let m =
        parse_module(&ir).unwrap_or_else(|e| panic!("parse IR failed: {e:?}\n--- IR ---\n{ir}"));
    verify_module(&m).unwrap_or_else(|e| panic!("verify failed: {e:?}\n--- IR ---\n{ir}"));
    let mut h = Host::new();
    let win = m.memory.map_or(0, |mc| 1u64 << mc.size_log2);
    let args = powerbox(&mut h, win, std::time::Duration::ZERO);
    let mut fuel = 50_000_000u64;
    let outcome = match run_with_host(&m, 0, &args, &mut fuel, &mut h) {
        Ok(v) => Outcome::Returned(v),
        Err(Trap::Exit(c)) => Outcome::Exited(c),
        Err(e) => panic!("interp trapped: {e:?}\n{src}\n{ir}"),
    };
    CRun {
        outcome,
        stdout: h.stdout,
    }
}

/// Run a normally-returning fiber program (interpreter-only) and return its single i32.
fn fiber_i32(src: &str) -> i32 {
    match run_c_interp(src).outcome {
        Outcome::Returned(v) => match v.as_slice() {
            [Value::I32(x)] => *x,
            other => panic!("expected one i32 result, got {other:?}"),
        },
        Outcome::Exited(c) => panic!("expected a normal return, but the program exited({c})"),
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
    let grant = |h: &mut Host| powerbox(h, 1 << 20, std::time::Duration::ZERO);
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

/// The guest can **query the host page size** it is being given — Memory cap op 3, the
/// `__vm_page_size` builtin — so its own allocator can align to the real MMU granularity
/// (4 KiB / 16 KiB / …) and adapt, instead of assuming a fixed size. The value must be a positive
/// power of two ≥ 4 KiB; interp and JIT both report the host page they actually round to, so they
/// must agree (which `run_c` asserts internally).
#[test]
fn c_guest_queries_page_size() {
    let src = r#"
long __vm_page_size(void);
int main() {
  long p = __vm_page_size();
  return (p >= 4096 && (p & (p - 1)) == 0) ? (int)p : -1; /* the page, or -1 if implausible */
}
"#;
    let p = i32_of(src);
    assert!(
        p >= 4096 && (p & (p - 1)) == 0,
        "guest-queried page size is not a sane power of two: {p}"
    );
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
    let grant = |h: &mut Host| powerbox(h, 1 << 20, std::time::Duration::ZERO);
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
fn c_defined_write_shadows_the_builtin() {
    // A guest **definition** of `write` (a body, not a bare `extern`) shadows the powerbox Stream
    // builtin — the frontend hook the POSIX personality libc relies on to own `write`/`read`/`exit`
    // with the real signature (PROCESS.md S15 (b)). Here the guest `write` just runs its own body
    // and returns; the Stream builtin would instead have written to stdout and returned the byte
    // count. `calls == 1` proves the body ran; empty stdout proves the builtin did *not* fire.
    let r = run_c_full(
        "int calls = 0; \
         int write(int fd, char *b, int n) { calls = calls + 1; return 100 + n; } \
         int main() { int r = write(1, 0, 7); return r + calls; }",
    );
    assert_eq!(
        r.outcome,
        Outcome::Returned(vec![Value::I32(108)]),
        "the guest write body ran (107) and bumped calls (1) — not the Stream builtin"
    );
    assert_eq!(r.stdout, b"", "the Stream builtin must not have fired");
}

#[test]
fn c_undefined_write_still_hits_the_builtin() {
    // The negative of the above: a bare `extern write` (no body) keeps the powerbox Stream builtin,
    // so the existing fixed-powerbox programs are unchanged by the shadowing hook.
    let r = run_c_full(
        "int write(int fd, char *buf, int n); \
         int main() { write(1, \"hey\", 3); return 0; }",
    );
    assert_eq!(
        r.stdout, b"hey",
        "extern write still routes to Stream.write"
    );
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
    let grant = |h: &mut Host| powerbox(h, 1 << 20, std::time::Duration::ZERO);
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

/// Narrowing **rvalue casts** to `char`/`short`/`_Bool` must truncate to the target width — the IR
/// carries all three as `i32`, so `gen_convert` has to reduce the value (sign-extend the low byte/
/// halfword, mask for unsigned, `!= 0` for `_Bool`), not lean on a store width. Before the fix
/// `(char)200` kept `200` and these `== expected` checks returned `0` on the VM vs `1` on `cc`
/// (non-vacuous). Each `main` returns `0`/`1`, so the comparison — not an exit code (which truncates
/// mod 256 and would hide the bug) — is what's checked.
#[test]
fn c_matches_gcc_narrowing_casts() {
    assert_matches_gcc("int main(){ return (char)200 == -56; }");
    assert_matches_gcc("int main(){ return (unsigned char)300 == 44; }");
    assert_matches_gcc("int main(){ return (char)(-200) == 56; }");
    assert_matches_gcc("int main(){ return (short)100000 == -31072; }");
    assert_matches_gcc("int main(){ return (unsigned short)0x12345 == 0x2345; }");
    assert_matches_gcc("int main(){ return (_Bool)200 == 1; }");
    assert_matches_gcc("int main(){ return (_Bool)0 == 0; }");
    // `_Bool` tests the *source* value: a long's high bits and a fractional float both count.
    assert_matches_gcc("int main(){ long x = 0x100000000L; return (_Bool)x == 1; }");
    assert_matches_gcc("int main(){ double d = 0.5; return (_Bool)d == 1; }");
    assert_matches_gcc("int main(){ long x = 0xAB; return (char)x == -85; }"); // i64 -> char
                                                                               // 125+126+127 + (-128)+(-127)+(-126) = -3 (each i past 127 wraps signed).
    assert_matches_gcc("int main(){ int s=0; for(int i=125;i<131;i++) s+=(char)i; return s==-3; }");
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

// ---- §12 fibers through real C (interpreter-only; the JIT bails Unsupported) ----

// Prototypes for the three intercepted fiber builtins, shared by the C tests below. A fiber
// body is an ordinary `long f(long)`; the guest hands each fiber its own data stack.
const FIBER_DECLS: &str = "\
long __vm_fiber_new(long (*f)(long), void *stack);\n\
long __vm_fiber_resume(long k, long arg, int *done);\n\
long __vm_fiber_suspend(long value);\n";

#[cfg(unix)]
#[test]
fn c_fiber_generator_yields_then_returns() {
    // `counter` yields start+1 and start+2 via suspend, then returns start+3. `main` drives
    // it: the first resume passes 100 (the body's arg), later resumes pass 0; it sums every
    // yielded value plus the final return. 101 + 102 + 103 = 306.
    let src = format!(
        "{FIBER_DECLS}\
        static char stack0[8192];\n\
        long counter(long start) {{\n\
        \x20 __vm_fiber_suspend(start + 1);\n\
        \x20 __vm_fiber_suspend(start + 2);\n\
        \x20 return start + 3;\n\
        }}\n\
        int main() {{\n\
        \x20 long k = __vm_fiber_new(counter, stack0);\n\
        \x20 int done = 0;\n\
        \x20 long sum = 0;\n\
        \x20 long v = __vm_fiber_resume(k, 100, &done);\n\
        \x20 while (!done) {{ sum += v; v = __vm_fiber_resume(k, 0, &done); }}\n\
        \x20 sum += v;\n\
        \x20 return (int)sum;\n\
        }}\n"
    );
    assert_eq!(fiber_i32(&src), 306);
}

#[cfg(unix)]
#[test]
fn c_fiber_round_trips_resume_arguments() {
    // The value passed to resume is delivered as the *result* of the body's `suspend`, so a
    // fiber can be a two-way channel. `echo` yields 0 once, then returns whatever it was
    // resumed with — proving the resume arg threads back into the suspended body. main feeds
    // 77 on the second resume and returns it.
    let src = format!(
        "{FIBER_DECLS}\
        static char st[4096];\n\
        long echo(long start) {{\n\
        \x20 (void)start;\n\
        \x20 long got = __vm_fiber_suspend(0);\n\
        \x20 return got * 2;\n\
        }}\n\
        int main() {{\n\
        \x20 long k = __vm_fiber_new(echo, st);\n\
        \x20 int done = 0;\n\
        \x20 __vm_fiber_resume(k, 0, &done);\n\
        \x20 long r = __vm_fiber_resume(k, 77, &done);\n\
        \x20 return (int)(done * 1000 + r);\n\
        }}\n"
    );
    // done = 1 (RETURNED) after the second resume; r = 77 * 2 = 154 -> 1154.
    assert_eq!(fiber_i32(&src), 1154);
}

#[cfg(unix)]
#[test]
fn c_two_fibers_are_independent() {
    // Two live fibers on distinct stacks interleave without clobbering each other's locals —
    // the data-stack-per-fiber property (§3d). Each keeps its own running counter across
    // suspends; main ping-pongs between them and sums their yields.
    let src = format!(
        "{FIBER_DECLS}\
        static char sa[4096];\n\
        static char sb[4096];\n\
        long acc(long step) {{\n\
        \x20 long total = 0;\n\
        \x20 for (;;) {{ total += step; __vm_fiber_suspend(total); }}\n\
        \x20 return 0;\n\
        }}\n\
        int main() {{\n\
        \x20 int da = 0, db = 0;\n\
        \x20 long a = __vm_fiber_new(acc, sa);\n\
        \x20 long b = __vm_fiber_new(acc, sb);\n\
        \x20 long s = 0;\n\
        \x20 s += __vm_fiber_resume(a, 10, &da);\n\
        \x20 s += __vm_fiber_resume(b, 3, &db);\n\
        \x20 s += __vm_fiber_resume(a, 0, &da);\n\
        \x20 s += __vm_fiber_resume(b, 0, &db);\n\
        \x20 s += __vm_fiber_resume(a, 0, &da);\n\
        \x20 return (int)s;\n\
        }}\n"
    );
    // a yields 10, 20, 30; b yields 3, 6. Sum = 10+3+20+6+30 = 69.
    assert_eq!(fiber_i32(&src), 69);
}

#[cfg(unix)]
#[test]
fn c_cooperative_threads_round_robin() {
    // Cooperative *multithreading* built entirely in guest C on top of the fiber builtins —
    // no new VM primitive (DESIGN §12: scheduling is runtime policy, "the guest sees a
    // thread, never an OS thread"). Three worker "threads" each sum 1..n, yielding after
    // every add; a round-robin scheduler in `main` interleaves them to completion. This is
    // the cooperative-thread model the pthreads shim will later wrap.
    let src = format!(
        "{FIBER_DECLS}\
        #define NT 3\n\
        static char stacks[NT][4096];\n\
        static long handles[NT];\n\
        static int  finished[NT];\n\
        static int  started[NT];\n\
        static long results[NT];\n\
        static void co_yield(void) {{ __vm_fiber_suspend(0); }}\n\
        long worker(long n) {{\n\
        \x20 long s = 0;\n\
        \x20 for (long i = 1; i <= n; i++) {{ s += i; co_yield(); }}\n\
        \x20 return s;\n\
        }}\n\
        int main() {{\n\
        \x20 long ns[NT]; ns[0]=3; ns[1]=4; ns[2]=5;\n\
        \x20 int remaining = NT;\n\
        \x20 for (int i = 0; i < NT; i++) {{\n\
        \x20   handles[i] = __vm_fiber_new(worker, stacks[i]);\n\
        \x20   finished[i] = 0; started[i] = 0;\n\
        \x20 }}\n\
        \x20 while (remaining > 0) {{\n\
        \x20   for (int i = 0; i < NT; i++) {{\n\
        \x20     if (finished[i]) continue;\n\
        \x20     long arg = started[i] ? 0 : ns[i];\n\
        \x20     started[i] = 1;\n\
        \x20     int done = 0;\n\
        \x20     long v = __vm_fiber_resume(handles[i], arg, &done);\n\
        \x20     if (done) {{ results[i] = v; finished[i] = 1; remaining--; }}\n\
        \x20   }}\n\
        \x20 }}\n\
        \x20 return (int)(results[0] + results[1] + results[2]);\n\
        }}\n"
    );
    // sum(1..3)=6, sum(1..4)=10, sum(1..5)=15  ->  31, regardless of interleaving.
    assert_eq!(fiber_i32(&src), 31);
}

// ---- §12 real threads + atomics from C (the `__vm_thread_*` / `__vm_atomic_*` builtins) ----

/// End-to-end multi-threaded C: four threads each atomically bump a shared counter 500×; the total
/// is 2000 on every interleaving. The thread body's loop counter is SSA-promoted (not address-taken),
/// so the worker uses no data stack and a 0 stack base is fine. Interpreter-only (the JIT doesn't
/// lower thread ops yet, step 4); ThreadSanitizer-clean via the shared `Region`.
const C_ATOMIC_COUNTER: &str = r#"
long counter;
long __vm_atomic_add(void *p, long v);
long __vm_atomic_load(void *p);
int  __vm_thread_spawn(long (*fn)(long), void *stack, long arg);
long __vm_thread_join(int h);

long worker(long iters) {
  for (long i = 0; i < iters; i++)
    __vm_atomic_add(&counter, 1);
  return 0;
}

int main(void) {
  int a = __vm_thread_spawn(worker, (void *)0, 500);
  int b = __vm_thread_spawn(worker, (void *)0, 500);
  int c = __vm_thread_spawn(worker, (void *)0, 500);
  int d = __vm_thread_spawn(worker, (void *)0, 500);
  __vm_thread_join(a);
  __vm_thread_join(b);
  __vm_thread_join(c);
  __vm_thread_join(d);
  return (int)__vm_atomic_load(&counter);
}
"#;

#[test]
fn c_threads_atomic_counter() {
    // The headline — C source → IR → threads → exactly 2000 — on the interpreter's real M:N executor.
    match run_c_interp(C_ATOMIC_COUNTER).outcome {
        Outcome::Returned(v) => assert_eq!(v.as_slice(), [Value::I32(2000)]),
        Outcome::Exited(c) => panic!("unexpected exit({c})"),
    }
}

#[test]
#[cfg(all(unix, target_arch = "x86_64"))]
fn c_threads_atomic_counter_jit() {
    // The same multi-threaded C, end to end on the **JIT** (cooperative green threads over the shared
    // window + hardware atomics), differentially checked against the interpreter by `run_c_full` —
    // real C with threads compiles and runs natively, not just on the interpreter.
    match run_c_full(C_ATOMIC_COUNTER).outcome {
        Outcome::Returned(v) => assert_eq!(v.as_slice(), [Value::I32(2000)]),
        Outcome::Exited(c) => panic!("unexpected exit({c})"),
    }
}

/// A spawned thread does **I/O** (`write` → a stream `cap.call`). This exercises the per-thread
/// powerbox: the worker shares the domain's capabilities, so its `write` reaches the same stdout.
/// Before that fix the interpreter gave each vCPU an *empty* powerbox, so the worker's `cap.call`
/// CapFaulted while the JIT (shared ctx) succeeded — a latent divergence `run_c_full` now pins
/// (interp == JIT, both print the line). Deterministic: `main` joins the worker before doing anything,
/// so only the worker writes.
#[test]
#[cfg(all(unix, target_arch = "x86_64"))]
fn c_thread_shares_powerbox_for_io() {
    let src = format!(
        "{LIBC}\n\
         int  __vm_thread_spawn(long (*fn)(long), void *stack, long arg);\n\
         long __vm_thread_join(int h);\n\
         long worker(long arg) {{ write(1, \"hi from a thread\\n\", 17); return 0; }}\n\
         int main(void) {{ int h = __vm_thread_spawn(worker, (void *)0, 0); __vm_thread_join(h); return 0; }}\n"
    );
    let run = run_c_full(&src);
    assert_eq!(run.stdout, b"hi from a thread\n");
}

#[test]
fn c_threads_deterministic_sweep() {
    // Same compiled C run through the seeded explorer (§18): every interleaving yields 2000, and each
    // is reproducible from its seed. The program makes no cap.calls, so dummy powerbox handles + the
    // explorer's empty host suffice.
    let ir = c_to_ir(C_ATOMIC_COUNTER);
    let m = parse_module(&ir).unwrap_or_else(|e| panic!("parse: {e:?}\n{ir}"));
    verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}\n{ir}"));
    let args = [Value::I32(0); 8]; // 8 dummy powerbox handles (the program makes no cap.calls)
    for seed in 0..100u64 {
        let r = run_scheduled(&m, 0, &args, 50_000_000, seed);
        assert_eq!(r, Ok(vec![Value::I32(2000)]), "explorer seed {seed}");
    }
}

// ---- §12 the C-compatible pthreads layer (`#include <pthread.h>`, D56) -----------------------
// Standard C threading over the VM primitives: `pthread_t` = one vCPU = one OS thread (1:1), with
// mutexes/conds built on the i32 atomics + futex. Differentially checked interp == JIT by `run_c_full`.

/// `pthread_create`/`join` + a futex-backed `pthread_mutex_t`: four threads each take the lock and bump
/// a *plain* shared `int` 500×. The mutex serialises the non-atomic `counter++`, so the total is
/// exactly 2000 (a missing lock would lose updates). Real pthreads code, unmodified.
#[test]
#[cfg(all(unix, target_arch = "x86_64"))]
fn c_pthread_mutex_counter() {
    let src = "#include <pthread.h>\n\
        static int counter = 0;\n\
        static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;\n\
        static void *worker(void *arg) {\n\
        \x20 (void)arg;\n\
        \x20 for (int i = 0; i < 500; i++) {\n\
        \x20   pthread_mutex_lock(&mu);\n\
        \x20   counter++;\n\
        \x20   pthread_mutex_unlock(&mu);\n\
        \x20 }\n\
        \x20 return 0;\n\
        }\n\
        int main(void) {\n\
        \x20 pthread_t t[4];\n\
        \x20 for (int i = 0; i < 4; i++) pthread_create(&t[i], 0, worker, 0);\n\
        \x20 for (int i = 0; i < 4; i++) pthread_join(t[i], 0);\n\
        \x20 return counter;\n\
        }\n";
    match run_c_full(src).outcome {
        Outcome::Returned(v) => assert_eq!(v.as_slice(), [Value::I32(2000)]),
        Outcome::Exited(c) => panic!("unexpected exit({c})"),
    }
}

/// `pthread_join` delivers the thread's return value: `start_routine` returns `arg*2` (via the
/// trampoline + `thread.join`), retrieved through the `void **retval` out-param.
#[test]
#[cfg(all(unix, target_arch = "x86_64"))]
fn c_pthread_join_returns_value() {
    let src = "#include <pthread.h>\n\
        static void *dbl(void *arg) { return (void *)((long)arg * 2); }\n\
        int main(void) {\n\
        \x20 pthread_t t;\n\
        \x20 pthread_create(&t, 0, dbl, (void *)21);\n\
        \x20 void *r;\n\
        \x20 pthread_join(t, &r);\n\
        \x20 return (int)(long)r;\n\
        }\n";
    match run_c_full(src).outcome {
        Outcome::Returned(v) => assert_eq!(v.as_slice(), [Value::I32(42)]),
        Outcome::Exited(c) => panic!("unexpected exit({c})"),
    }
}

/// `pthread_cond_t` handoff: a consumer waits on the cond under the mutex until `ready`, the producer
/// (main) publishes a payload, sets the predicate, and signals. Correct whether the consumer parks
/// first (woken by the signal) or the signal lands first (predicate already true, no wait) — the
/// result is the payload, 42, on every interleaving.
#[test]
#[cfg(all(unix, target_arch = "x86_64"))]
fn c_pthread_cond_handoff() {
    let src = "#include <pthread.h>\n\
        static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;\n\
        static pthread_cond_t cv = PTHREAD_COND_INITIALIZER;\n\
        static int ready = 0;\n\
        static int payload = 0;\n\
        static void *consumer(void *arg) {\n\
        \x20 (void)arg;\n\
        \x20 pthread_mutex_lock(&mu);\n\
        \x20 while (!ready) pthread_cond_wait(&cv, &mu);\n\
        \x20 int p = payload;\n\
        \x20 pthread_mutex_unlock(&mu);\n\
        \x20 return (void *)(long)p;\n\
        }\n\
        int main(void) {\n\
        \x20 pthread_t t;\n\
        \x20 pthread_create(&t, 0, consumer, 0);\n\
        \x20 pthread_mutex_lock(&mu);\n\
        \x20 payload = 42;\n\
        \x20 ready = 1;\n\
        \x20 pthread_cond_signal(&cv);\n\
        \x20 pthread_mutex_unlock(&mu);\n\
        \x20 void *r;\n\
        \x20 pthread_join(t, &r);\n\
        \x20 return (int)(long)r;\n\
        }\n";
    match run_c_full(src).outcome {
        Outcome::Returned(v) => assert_eq!(v.as_slice(), [Value::I32(42)]),
        Outcome::Exited(c) => panic!("unexpected exit({c})"),
    }
}

// §13/§14 the **magic ring buffer in real C**, end to end through the `__vm_region_*` builtins —
// proving the powerbox wiring (the 5th handle: an AddressSpace the guest mints from) and the
// builtin lowering, differentially on both backends. A guest mints a region, maps it at two
// adjacent high window offsets, and a single straddling store wraps tail→head as one contiguous
// access — the whole point of the layout, now reachable from C with no host hand-holding.
const C_RING_REGION: &str = "\
#include <svm.h>\n\
/* Force a large window (≥512 KiB) so two granule mappings fit well above the data/stack. */\n\
static char pad[200 * 1024];\n\
int main(void) {\n\
  pad[0] = 1; pad[200 * 1024 - 1] = 2; /* keep `pad` live so the window grows */\n\
  int r = (int)__vm_region_create(1 << 16); /* mint a 64 KiB region */\n\
  if (r < 0) return -1;\n\
  long g = __vm_region_page_size(r);        /* host map granularity */\n\
  long base = 256 * 1024;                    /* aligned, clear of data/stack */\n\
  if (__vm_region_map(r, base, 0, g, 3) < 0) return -2;       /* mapping 1: [base, base+g) */\n\
  if (__vm_region_map(r, base + g, 0, g, 3) < 0) return -3;   /* mapping 2: [base+g, base+2g) */\n\
  /* one 8-byte store straddling the seam: low half → region tail, high half wraps → region head */\n\
  *(unsigned long *)(base + g - 4) = 0x1122334455667788UL;\n\
  unsigned int head = *(unsigned int *)(base);            /* region head, via mapping 1 */\n\
  unsigned int tail = *(unsigned int *)(base + 2 * g - 4); /* region tail, via mapping 2 */\n\
  unsigned long combined = ((unsigned long)head << 32) | tail;\n\
  return combined == 0x1122334455667788UL ? 1 : 0;\n\
}\n";

#[test]
fn c_ring_buffer_via_minted_region() {
    // Both backends must agree the wrap is byte-exact (the guest minted, mapped, and straddled
    // entirely on its own — the host only installed the region factory + granted the AddressSpace).
    assert_eq!(run_c(C_RING_REGION).as_slice(), [Value::I32(1)]);
}

// The `__vm_region_unmap` builtin (`<svm.h>`, op 1) — the one region builtin the ring-buffer test
// doesn't exercise. Mint a region, map it, write through it, then unmap that window range: the
// builtin must lower (`cap.call 4 1`) and the unmap succeed (return 0), identically on both backends.
const C_REGION_UNMAP: &str = "\
#include <svm.h>\n\
static char pad[200 * 1024];\n\
int main(void) {\n\
  pad[0] = 1; pad[200 * 1024 - 1] = 2;\n\
  int r = (int)__vm_region_create(1 << 16);\n\
  if (r < 0) return -1;\n\
  long g = __vm_region_page_size(r);\n\
  long base = 256 * 1024;\n\
  if (__vm_region_map(r, base, 0, g, 3) < 0) return -2;\n\
  *(unsigned int *)base = 77;                 /* write through the mapping */\n\
  long u = __vm_region_unmap(r, base, g);     /* the builtin under test */\n\
  return u == 0 ? 1 : 0;                       /* unmap must succeed */\n\
}\n";

#[test]
fn c_region_unmap_builtin() {
    assert_eq!(run_c(C_REGION_UNMAP).as_slice(), [Value::I32(1)]);
}

/// §7/§4 — a cap-buffer borrow of a **guest-grown** heap page. The program `malloc`s 128 KiB (past
/// the 64 KiB initial window, so `malloc` grows the heap into the reserved tail via the Memory cap),
/// fills it, and `write()`s **that grown buffer** across the §7 trampoline. The interpreter's `Mem`
/// persists its page map across cap.calls, so it borrows the grown pages fine; the JIT's `cap_thunk`
/// must persist its page map too (else the borrow fail-closes and the write drops). `run_c_full`
/// enforces interp == JIT, so this fails if the JIT can't see the growth — the regression guard for
/// the cap-path page-map persistence.
#[test]
fn c_grown_heap_buffer_is_borrowable() {
    let src = "#include <stdlib.h>\n\
        int write(int fd, char *buf, long n);\n\
        int main(void){\n\
        \x20 long n = 128*1024;\n\
        \x20 char *buf = (char*)malloc(n);\n\
        \x20 if(!buf){ write(1,\"OOM\\n\",4); return 1; }\n\
        \x20 for(long i=0;i<n;i++) buf[i] = (char)('A' + (i%26));\n\
        \x20 write(1, buf, n);\n\
        \x20 return 0;\n\
        }\n";
    let run = run_c_full(src);
    let expected: Vec<u8> = (0..128 * 1024).map(|i| b'A' + (i % 26) as u8).collect();
    assert_eq!(
        run.stdout, expected,
        "grown-heap buffer write must reach stdout"
    );
}

/// The §12 capstone: a **guest-built M:N green-thread scheduler** (`demos/mn_sched`) runs
/// identically on the interpreter (the M:N deterministic oracle) and the JIT (real OS threads).
/// 4 worker threads (`thread.spawn`), each cooperatively round-robining 8 fibers (`cont.*`) that
/// yield and increment one shared atomic — the entire scheduler is *guest code* over the VM's
/// primitives (D56/D57). The grand total (4·8·32 = 1024) is interleaving-invariant, so both
/// backends must print exactly it. This proves the abstractions compose into a real M:N runtime
/// with no scheduler baked into the VM. Interp == JIT is enforced inside `run_c_full`.
#[test]
#[cfg(all(unix, target_arch = "x86_64"))]
fn c_guest_mn_scheduler_demo() {
    let src = include_str!("../../svm-run/demos/mn_sched/mn_sched.c");
    let run = run_c_full(src);
    assert_eq!(
        run.stdout, b"1024\n",
        "the guest M:N scheduler must total 4*8*32 = 1024 on both backends"
    );
}

/// The §12 work-stealing capstone: a guest-built **work-stealing** M:N scheduler over **stackless**
/// tasks (`demos/work_stealing`) runs identically on the interpreter (the M:N oracle) and the JIT.
/// The guest-driven **JIT capstone** (DESIGN.md §22, `demos/jit`): a C bytecode interpreter
/// that JITs itself — it emits serialized SVM IR at runtime (the binary `svm-encode` format,
/// byte-by-byte in guest memory), submits it through the `Jit` capability, and invokes the
/// compiled unit, checking it against its own interpreter on a 49-input grid. Differential:
/// under the reference interpreter the `invoke` is a nested eval over the same window; under
/// the JIT it is native Cranelift code — `run_c_full` enforces identical results and stdout.
#[test]
#[cfg(all(unix, target_arch = "x86_64"))]
fn c_guest_jit_demo() {
    let src = include_str!("../../svm-run/demos/jit/jit_demo.c");
    let run = run_c_full(src);
    let out = String::from_utf8_lossy(&run.stdout);
    assert!(
        out.ends_with(
            "98 inputs agree (invoke + installed call_indirect): \
             guest-emitted, host-verified, Cranelift-compiled\n"
        ),
        "the guest's interpreter, its invoked JIT code, and its installed call_indirect slot \
         must all agree on both backends:\n{out}"
    );
}

/// Guest-side **dynamic linking** in C (DESIGN.md §22, `demos/jit/jit_link.c`): a guest emits two
/// units — a self-contained `service` it installs, and a `client` that references the service **by
/// name** (an unresolved import `F`) — builds a symbol table binding `"F"` to the install slot, and
/// `__vm_jit_compile_linked`s the client against it. The host resolves the import by name and
/// re-verifies, so the client reaches the installed service through the table: `client(5,2) = 127`.
/// This is `vm_dlopen`/`vm_dlsym` done in guest C. `run_c_full` enforces interp == JIT.
#[test]
#[cfg(all(unix, target_arch = "x86_64"))]
fn c_guest_jit_link_demo() {
    let src = include_str!("../../svm-run/demos/jit/jit_link.c");
    let run = run_c_full(src);
    let out = String::from_utf8_lossy(&run.stdout);
    assert!(
        out.ends_with("client(5, 2) = 127  [linked by name: service(5,2)+100]\n"),
        "the guest-linked client must reach the installed service by name on both backends:\n{out}"
    );
}

/// The guest-side **`vm_dlopen` loader** in C (DESIGN.md §22, `demos/jit/jit_dlopen.c`): the
/// ergonomic `vm_dlopen`/`vm_dlsym`/`vm_dlclose` library (`<vm_dl.h>` — a name→slot registry over the
/// `Jit` cap) used to build functions that compose **by name**. The guest loads `add` and `mul`, then
/// `poly = add(mul(a,a), b)` which imports both by name; `poly(5,2)=27`, `poly(3,4)=13`; then
/// `vm_dlclose("poly")` unloads it. The guest-C twin of `dynlink_repl.rs`. `run_c_full` enforces
/// interp == JIT.
#[test]
#[cfg(all(unix, target_arch = "x86_64"))]
fn c_guest_jit_dlopen_demo() {
    let src = include_str!("../../svm-run/demos/jit/jit_dlopen.c");
    let run = run_c_full(src);
    let out = String::from_utf8_lossy(&run.stdout);
    assert!(
        out.contains("poly(5, 2) = 27\n")
            && out.contains("poly(3, 4) = 13\n")
            && out.ends_with("linked by name via vm_dlopen/vm_dlsym/vm_dlclose\n"),
        "the guest vm_dlopen loader must compose symbols by name on both backends:\n{out}"
    );
}

/// **Hot reload** over the guest `vm_dlopen` loader (DESIGN.md §22, `demos/jit/jit_hotreload.c`):
/// redefining a symbol gives it a new slot, but units already linked to the old one keep their
/// binding. The guest loads `f` (a+100), then `g` calling `f` by name, hot-reloads `f` (a+200), then
/// loads `h` calling `f` by name: `g(5)=105` (pinned to the old `f`), `h(5)=205` (sees the new one).
/// Proves the slot model's live-patch behaviour. `run_c_full` enforces interp == JIT.
#[test]
#[cfg(all(unix, target_arch = "x86_64"))]
fn c_guest_jit_hotreload_demo() {
    let src = include_str!("../../svm-run/demos/jit/jit_hotreload.c");
    let run = run_c_full(src);
    let out = String::from_utf8_lossy(&run.stdout);
    assert!(
        out.contains("g(5) = 105") && out.contains("h(5) = 205"),
        "an old caller must keep its binding across a hot reload, on both backends:\n{out}"
    );
}

/// The **threaded** guest-driven JIT capstone (`demos/jit/jit_threads.c`, DESIGN.md §22), run as a
/// full interp≡JIT **differential**: `NWORKERS` guest threads each emit a distinct unit, `Jit.compile`
/// it **concurrently**, and invoke the native code, checking it against a C reference. Because the
/// guest `thread.spawn`s, `run_c_full` drives the JIT side through the serialized `cap_thunk_locked`
/// (a `Mutex<Host>`) — so concurrent `Jit.compile`s don't race — while the interpreter (the M:N
/// oracle) compiles each unit as a nested eval. The `0` mismatch total is interleaving-invariant, so
/// both backends must print it; `run_c_full` enforces identical stdout. This is what makes a
/// genuinely-concurrent C JIT feature differentially testable, not merely demonstrable.
#[test]
#[cfg(all(unix, target_arch = "x86_64"))]
fn c_guest_jit_threads_demo() {
    let src = include_str!("../../svm-run/demos/jit/jit_threads.c");
    let run = run_c_full(src);
    assert_eq!(
        run.stdout, b"0\n",
        "every worker's concurrently-JITed unit must agree with the reference on both backends"
    );
}

/// Drive the **auto-compacting JIT REPL** C demo (`demos/jit/jit_repl.c`) through
/// `svm_run::JitSession`: re-enter the C `_start` (func 0) once per prompt over a persistent window,
/// passing the fixed 8-handle powerbox each time, with the session auto-compacting once the live code
/// crosses `watermark` bytes. Returns `(per-prompt results, captured stdout, final occupancy,
/// compactions run)`. The accumulator/prompt counter live in BSS (no `data` segment), so they carry
/// across prompts as window state; each prompt JITs + invokes + releases a fresh unit.
#[cfg(all(unix, target_arch = "x86_64"))]
fn run_jit_repl_session(
    src: &str,
    watermark: usize,
    n: usize,
) -> (Vec<i64>, Vec<u8>, usize, usize) {
    let ir = c_to_ir(src);
    let m = parse_module(&ir).unwrap_or_else(|e| panic!("parse IR: {e:?}\n{ir}"));
    verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}\n{ir}"));
    let win = m.memory.map_or(0, |mc| 1u64 << mc.size_log2);

    let mut host = Host::new();
    // The fixed 8-handle powerbox a chibicc `_start` imports; the 8th (JIT) grants an invoke-only
    // domain (`table_log2 = 0`, no install table), matching the session's `table_reserve_log2` below.
    let args = powerbox(&mut host, win, std::time::Duration::ZERO);
    let Value::I32(jit_handle) = args[7] else {
        panic!("the 8th powerbox handle is the JIT domain");
    };
    let domain = host.resolve_jit_domain(jit_handle).expect("jit domain");
    let slot_args: Vec<i64> = args.iter().copied().map(to_slot).collect();

    // The session takes ownership of the host (boxed `Mutex<Host>`); recover it for stdout below.
    let mut session = svm_run::JitSession::new(
        &m,
        0,
        svm_ir::DEFAULT_RESERVED_LOG2,
        0,
        domain,
        watermark,
        host,
    )
    .expect("session");

    let mut results = Vec::new();
    for _ in 0..n {
        match session.run_prompt(&slot_args).expect("prompt") {
            JitOutcome::Returned(s) if s.len() == 1 => results.push(s[0]),
            other => panic!("prompt returned {other:?}"),
        }
    }
    let (occ, compactions) = (session.occupancy(), session.compactions());
    let host = session.into_host();
    (results, host.stdout, occ, compactions)
}

/// The **auto-compacting guest-driven JIT REPL** capstone in real C (`demos/jit/jit_repl.c`, DESIGN.md §22
/// §6 #1): a long REPL that `__vm_jit_compile`s a fresh unit every prompt would exhaust the code
/// arena, but `JitSession` recompacts between prompts so occupancy stays bounded — transparently. A
/// 30-prompt session produces byte-identical results **and** stdout transcript whether
/// auto-compaction is off (`watermark = 0`) or on (a byte watermark of ~4 units), and with it on the
/// session compacts while its code-arena occupancy stays near the watermark instead of growing with
/// every prompt. The watermark is derived from a one-prompt byte probe so the test is robust to
/// per-platform code sizes.
#[test]
#[cfg(all(unix, target_arch = "x86_64"))]
fn c_guest_jit_repl_compacts() {
    let src = include_str!("../../svm-run/demos/jit/jit_repl.c");
    let n = 30;
    // Probe one prompt's code-byte cost (a fresh unit + trampoline) to set a per-platform-robust
    // watermark that trips roughly every ~4 prompts.
    let (_, _, unit_bytes, _) = run_jit_repl_session(src, 0, 1);
    assert!(unit_bytes > 0, "a compiled unit must report nonzero bytes");
    let watermark = unit_bytes * 4;

    let (r_off, out_off, occ_off, c_off) = run_jit_repl_session(src, 0, n);
    let (r_on, out_on, occ_on, c_on) = run_jit_repl_session(src, watermark, n);

    assert_eq!(r_off, r_on, "per-prompt results must be identical");
    assert_eq!(
        out_off, out_on,
        "the REPL transcript must be byte-identical with/without compaction"
    );
    // The accumulator advanced: prompt i (0-based) folds in (i+2)*(i+2) + 10; the returned value is
    // the running total, so the last result is the full sum.
    let expected: i64 = (0..n as i64).map(|i| (i + 2) * (i + 2) + 10).sum();
    assert_eq!(*r_on.last().unwrap(), expected);

    assert_eq!(c_off, 0, "watermark 0 disables auto-compaction");
    assert!(c_on > 0, "the byte watermark must trip auto-compaction");
    assert!(
        occ_off > watermark,
        "without compaction the byte occupancy grows past the watermark: {occ_off}"
    );
    assert!(
        occ_on <= watermark + unit_bytes && occ_on < occ_off,
        "auto-compaction must bound byte occupancy near the watermark: {occ_on} (off {occ_off}, wm {watermark})"
    );
}

/// Tasks are state-machine structs (just data), so an idle worker steals one from a busy sibling /
/// the global injector and resumes it on its own thread — cross-thread task migration with **no VM
/// change** (the migratable-fiber primitive, D57, is *not* needed for stackless tasks). The total
/// (16·16 = 256) is interleaving-invariant, so both backends must print it regardless of *which*
/// worker ran each task. Proves work-stealing M:N composes from the primitives. Interp == JIT is
/// enforced inside `run_c_full`.
#[test]
#[cfg(all(unix, target_arch = "x86_64"))]
fn c_guest_work_stealing_demo() {
    let src = include_str!("../../svm-run/demos/work_stealing/work_stealing.c");
    let run = run_c_full(src);
    assert_eq!(
        run.stdout, b"256\n",
        "the guest work-stealing scheduler must total 16*16 = 256 on both backends"
    );
}

/// **Demo 3 (DESIGN.md §23) — work-stealing over *stackful, migratable* fibers (D57 complete).**
/// Tasks are fibers whose handles sit in guest queues (injector + per-worker deques); an idle
/// worker steals a **suspended fiber** and resumes it on its own OS thread — the fiber's whole
/// native stack migrates (on the JIT, a real cross-thread `svm-fiber` switch claimed through the
/// loom-verified `Ownership` word; on the interp, a `Vec<Frame>` hand-off). The task yields from
/// *inside a nested call frame* (`step_in_callee`) — inexpressible for a stackless state machine —
/// and its return value depends on locals carried across every yield/migration, so the second
/// printed total (`121920`) is the stack-integrity check, not just a work count. Both totals are
/// interleaving-invariant; interp == JIT is enforced inside `run_c_full`.
#[test]
#[cfg(all(unix, target_arch = "x86_64"))]
fn c_guest_steal_fibers_demo() {
    let src = include_str!("../../svm-run/demos/steal_fibers/steal_fibers.c");
    let run = run_c_full(src);
    assert_eq!(
        run.stdout, b"256\n121920\n",
        "the stackful work-stealing scheduler must produce both invariant totals on both backends"
    );
}

/// The §3d **thread-safe guest `malloc`** (`include/stdlib.h`): 4 vCPUs each `malloc` 64 blocks and
/// fill them with per-block patterns; main re-checks every byte. If two concurrent allocations had
/// overlapped (the race the old bump allocator allowed), a fill would have clobbered another block and
/// the re-check would find corruption. The demo prints the corrupt-block count — `0` on a correct
/// allocator, on both the interpreter (M:N oracle) and the JIT (real OS threads). The lock-free
/// atomic-bump claim + spinlock-guarded page growth is what makes concurrent allocation safe (so
/// threaded guests no longer have to pre-allocate on the main thread).
#[test]
#[cfg(all(unix, target_arch = "x86_64"))]
fn c_guest_thread_safe_malloc() {
    let src = include_str!("../../svm-run/demos/malloc_threads/malloc_threads.c");
    let run = run_c_full(src);
    assert_eq!(
        run.stdout, b"0\n",
        "concurrent malloc must hand out disjoint blocks (0 corrupt) on both backends"
    );
}

/// The host's deterministic `Blocking.work(i)` result (mirrors `svm_interp::AsyncState::mix`).
#[cfg(all(unix, target_arch = "x86_64"))]
fn async_mix(arg: i64) -> i64 {
    arg.wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407)
}

/// Compile + run an async-ring C demo on **both** backends, returning their captured stdout. The
/// `Blocking` op blocks ~10 ms so the batch is genuinely in flight when a vCPU parks. The interp drives
/// the M:N executor (`run_with_host` → `drive` installs the `Scheduler::notify` wake hook); the JIT
/// uses the async entry + `svm_run::HostAsyncHooks` (its per-run `Domain` futex as the wake hook). Both
/// must return 0; the caller compares their stdout to the order-invariant expected total.
#[cfg(all(unix, target_arch = "x86_64"))]
fn run_async_demo(src: &str) -> (Vec<u8>, Vec<u8>) {
    use std::time::Duration;
    let ir = c_to_ir(src);
    let m = parse_module(&ir).unwrap_or_else(|e| panic!("parse IR: {e:?}\n{ir}"));
    verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}\n{ir}"));
    let win = m.memory.map_or(0, |mc| 1u64 << mc.size_log2);

    let mut hi = Host::new();
    let mut hj = Host::new();
    let args = powerbox(&mut hi, win, Duration::from_millis(10));
    assert_eq!(
        args,
        powerbox(&mut hj, win, Duration::from_millis(10)),
        "grants are deterministic"
    );

    let mut fuel = 500_000_000u64;
    let interp = run_with_host(&m, 0, &args, &mut fuel, &mut hi).expect("interp ran ok");
    assert_eq!(interp, vec![Value::I32(0)], "demo returns 0 (interp)");

    let slots: Vec<i64> = args.iter().copied().map(to_slot).collect();
    let init = vec![0u8; win as usize];
    // SAFETY: `hj` is the live cap-ctx Host for this run and outlives it.
    let hooks = unsafe { svm_run::HostAsyncHooks::new(&mut hj as *mut Host) };
    // `DEFAULT_RESERVED_LOG2` gives the window its large reserved growth tail (§4) — needed so a guest
    // `malloc` (e.g. a 256 KiB pthread stack) can grow the heap past the backed prefix via the Memory
    // cap. (Passing `0` here would leave no tail, so `malloc` returns NULL.)
    let (jit, _jmem) = svm_jit::compile_and_run_capture_reserved_with_host_async(
        &m,
        0,
        &slots,
        &init,
        svm_ir::DEFAULT_RESERVED_LOG2,
        cap_thunk,
        &mut hj as *mut Host as *mut c_void,
        &hooks,
    )
    .expect("jit ran");
    assert!(
        matches!(jit, JitOutcome::Returned(ref s) if s == &[0]),
        "jit demo returns 0: {jit:?}"
    );
    (hi.stdout, hj.stdout)
}

/// §9/§12 increment 3c — the async **event-loop runtime** in real C (`demos/async_io`). One vCPU
/// `submit_async`s a batch of `Blocking` ops onto the host offload pool, then parks on an in-window
/// completion **counter** (`__vm_wait32`) and reaps completions as the pool delivers them — the
/// "submit, park, run another, resume on completion" loop, with the parked vCPU woken by a pool
/// worker's `notify` (an I/O completion is a futex notify, DESIGN §12). The printed total — the sum of
/// the host's deterministic per-op results — is completion-order-invariant, so the interpreter (its
/// `Scheduler::notify` wake hook) and the JIT (its per-run `Domain` futex) must agree. Exercises the
/// new `codegen_ir.c` ring builtins + the 7-handle powerbox end to end.
#[test]
#[cfg(all(unix, target_arch = "x86_64"))]
fn c_guest_async_io_runtime() {
    let (interp, jit) = run_async_demo(include_str!("../../svm-run/demos/async_io/async_io.c"));
    // NTASKS = 8 (see the demo).
    let total: u64 = (0..8).fold(0u64, |a, i| a.wrapping_add(async_mix(i) as u64));
    let expected = format!("{total}\n").into_bytes();
    assert_eq!(
        interp, expected,
        "interp total must be Σ mix(i) for i in 0..8"
    );
    assert_eq!(jit, expected, "jit total must be Σ mix(i) for i in 0..8");
}

/// §9/§12 increment 3c (capstone) — the async **work-stealing M:N runtime** in real C
/// (`demos/async_work_stealing`): `NWORKERS` vCPUs cooperatively drain `NTASKS` I/O-bound tasks, each
/// issuing a blocking op through the ring. A worker never blocks on an I/O — it `submit_async`s a
/// task's op onto the offload pool and moves on, **parking** on the completion counter only when
/// nothing is runnable, woken by a pool worker's `notify`. Work-stealing and I/O overlap: N ops in
/// flight on K pool threads while the vCPUs reap, not block. The total is completion-order- *and*
/// interleaving-invariant, so the interp (M:N oracle) and JIT (real OS threads) must print the same
/// regardless of which worker submitted/reaped each task.
#[test]
#[cfg(all(unix, target_arch = "x86_64"))]
fn c_guest_async_work_stealing() {
    let (interp, jit) = run_async_demo(include_str!(
        "../../svm-run/demos/async_work_stealing/async_work_stealing.c"
    ));
    // NTASKS = 16 (see the demo).
    let total: u64 = (0..16).fold(0u64, |a, i| a.wrapping_add(async_mix(i) as u64));
    let expected = format!("{total}\n").into_bytes();
    assert_eq!(
        interp, expected,
        "interp total must be Σ mix(i) for i in 0..16"
    );
    assert_eq!(jit, expected, "jit total must be Σ mix(i) for i in 0..16");
}

/// §7 slice 3b — a brand-new host capability reached as a plain `extern`, no frontend special-case.
/// chibicc's builtins are `__vm_*`, but the host policy keys are `vm_*`, so `vm_page_size` is *not*
/// a recognized builtin: it flows through the GENERIC undefined-extern → `call.import` path, with
/// the capability handle obtained via `__vm_cap`. The default policy still binds the name to
/// (Memory, page_size), so the generic-import result must equal the hardcoded `__vm_page_size()`
/// builtin — proving the late-binding extern path is correct end-to-end (frontend + host + run).
#[test]
fn generic_extern_capability_import_equals_builtin() {
    let src = r#"
        extern long vm_page_size(int h);   /* undefined extern -> call.import "vm_page_size" */
        long __vm_page_size(void);         /* recognized builtin -> inline Memory.page_size  */
        int __vm_cap(int i);
        int main(void) {
          long viaextern  = vm_page_size(__vm_cap(3)); /* slot 3 = the Memory handle */
          long viabuiltin = __vm_page_size();
          return (viaextern == viabuiltin && viaextern > 0) ? 42 : 0;
        }
    "#;
    assert_eq!(
        i32_of(src),
        42,
        "the generic extern-import path must equal the hardcoded builtin"
    );
}

/// §7 reflection from C: a guest enumerates the capabilities its host granted and introspects each
/// one's interface type_id — `__vm_cap_count` / `__vm_cap_at` (`cap.self.count`/`cap.self.get`).
/// Interp-only (the JIT bails `Unsupported` on `cap.self.*`, like fibers). The c_frontend powerbox
/// grants 8 capabilities with exactly one Exit (type_id 1), so `n*100 + exits == 801`.
#[test]
fn reflection_enumerates_granted_capabilities() {
    let src = r#"
        int __vm_cap_count(void);
        int __vm_cap_at(int i, int *type_id_out);
        int main(void) {
          int n = __vm_cap_count();
          int exits = 0;
          for (int i = 0; i < n; i++) {
            int t;
            __vm_cap_at(i, &t);
            if (t == 1) exits++;   /* iface::EXIT == 1 */
          }
          return n * 100 + exits;  /* 8 capabilities, exactly one Exit -> 801 */
        }
    "#;
    match run_c_interp(src).outcome {
        Outcome::Returned(v) => {
            assert_eq!(
                v,
                vec![Value::I32(801)],
                "8 capabilities granted, exactly one Exit"
            )
        }
        other => panic!("unexpected outcome: {other:?}"),
    }
}

// ---- W4 chibicc debug emission: read C locals by name on real frontend output ----

/// Read an i32 source variable, whether it's a promoted SSA value or a window slot.
fn as_i32_var(v: Option<VarValue>) -> i32 {
    match v {
        Some(VarValue::Value(Value::I32(n))) => n,
        Some(VarValue::Bytes(b)) => i32::from_le_bytes(b.try_into().expect("4 bytes")),
        other => panic!("expected an i32 var, got {other:?}"),
    }
}

#[test]
fn chibicc_g_emits_named_locals_resolved_by_the_inspector() {
    // No `main` ⇒ no `_start`/powerbox; `compute` is function 0. `-g` now keeps SSA promotion and
    // emits a location list (`ssalist`), so a promoted scalar is debuggable *as optimized* — the
    // interpreter resolves its holding SSA value per pc. (`return t + s` gives a final op where
    // both `s` and `t` are already live to inspect.)
    let src = r#"
int compute(int a, int b) {
  int s = a + b;
  int t = s + 100;
  return t + s;
}
"#;
    let ir = c_to_ir_g(src);
    assert!(
        ir.contains("debug.var 0 \"s\" ssalist"),
        "emits a location-list debug.var for the promoted s:\n{ir}"
    );
    let m = parse_module(&ir).expect("parse");

    // Drive `compute(5, 3)` directly: v0 is the data-SP (a window base with frame headroom),
    // then the two i32 params.
    let sp = 32768i64;
    let mut insp = Inspector::attach(
        &m,
        0,
        &[Value::I64(sp), Value::I32(5), Value::I32(3)],
        1_000_000,
    );

    // Break at the last op (the `t + s` add): `s` and `t` are both already live there.
    let last = m.funcs[0].blocks[0].insts.len() - 1;
    insp.set_breakpoint(IrPc {
        module: 0,
        func: 0,
        block: 0,
        inst: last,
    });
    assert!(matches!(
        insp.run_until_stop(),
        svm_interp::Stop::Break { .. }
    ));

    // Read the C locals by their source names: s = a + b = 8, t = s + 100 = 108, params a/b.
    assert_eq!(as_i32_var(insp.read_var(0, "s", 4)), 8);
    assert_eq!(as_i32_var(insp.read_var(0, "t", 4)), 108);
    assert_eq!(as_i32_var(insp.read_var(0, "a", 4)), 5);
    assert_eq!(as_i32_var(insp.read_var(0, "b", 4)), 3);
    assert_eq!(insp.read_var(0, "nonesuch", 4), None);
}

#[test]
fn chibicc_g_emits_function_names() {
    // `-g` emits the §6 function-name table (`debug.fname <func> "<name>"`), so a backtrace / gdb
    // `bt` / kill message reads the C name instead of `fn{N}`.
    let src = r#"
int helper(int x) { return x + 1; }
int compute(int a) { return helper(a) * 2; }
"#;
    let ir = c_to_ir_g(src);
    assert!(
        ir.contains("debug.fname"),
        "emits the function-name table:\n{ir}"
    );
    let m = parse_module(&ir).expect("parse");
    let names: Vec<&str> = m
        .debug_info
        .as_ref()
        .expect("debug info")
        .func_names
        .iter()
        .map(|f| f.name.as_str())
        .collect();
    assert!(
        names.contains(&"helper") && names.contains(&"compute"),
        "func_names carries the C names, got {names:?}"
    );
}

#[test]
fn chibicc_g_emits_module_scoped_globals_read_in_any_frame() {
    // chibicc as a *second* producer of the §6 module-scoped-global primitive (slice 28): a source
    // global lives at a fixed window address and is inspectable by name in every frame.
    let src = r#"
int counter = 7;
struct P { int a; int b; } origin = { 3, 4 };
int bump(int n) { counter = counter + n; return counter + origin.a; }
"#;
    let ir = c_to_ir_g(src);
    assert!(
        ir.contains("debug.var global \"counter\" fixed"),
        "emits a fixed-address global debug.var for `counter`:\n{ir}"
    );
    assert!(
        ir.contains("debug.var global \"origin\" fixed"),
        "emits a fixed-address global for the struct `origin`:\n{ir}"
    );
    let m = parse_module(&ir).expect("parse");

    // `bump` is function 0 (no `main` ⇒ no `_start`). At entry the data segment holds counter = 7;
    // read it by name through the global scope (frame-independent `Fixed` address).
    let sp = 1i64 << 20;
    let mut insp = Inspector::attach(&m, 0, &[Value::I64(sp), Value::I32(10)], 1_000_000);
    insp.set_breakpoint(IrPc {
        module: 0,
        func: 0,
        block: 0,
        inst: 0,
    });
    assert!(matches!(
        insp.run_until_stop(),
        svm_interp::Stop::Break { .. }
    ));
    assert_eq!(
        as_i32_var(insp.read_var(0, "counter", 4)),
        7,
        "global read in bump's frame"
    );
}

#[test]
fn chibicc_g_resolves_shadowed_locals_by_scope() {
    // C shadowing: an inner block redeclares `x`. chibicc emits two `debug.var 0 "x"` lines, each
    // with a `scope <start> <end>` source-line range; reading `x` by name resolves to the one **in
    // scope at the stopped pc** (the inner shadow inside the block, the outer one after it) — not
    // just the first declared (DEBUGGING.md §6 lexical-scope resolution).
    let src = r#"
int f(int n) {
  int x = n + 1;
  {
    int x = n + 100;
    n = n + x;
  }
  return x + n;
}
"#;
    let ir = c_to_ir_g(src);
    // Both shadows emitted under the same name+func, each carrying a distinct lexical scope.
    assert_eq!(
        ir.matches("debug.var 0 \"x\"").count(),
        2,
        "both shadows emitted:\n{ir}"
    );
    assert!(
        ir.contains(" scope "),
        "shadows carry source-line scopes:\n{ir}"
    );
    let m = parse_module(&ir).expect("parse");

    let pc_for_line = |line: u32| {
        let l = m
            .debug_info
            .as_ref()
            .unwrap()
            .locs
            .iter()
            .find(|l| l.line == line)
            .unwrap_or_else(|| panic!("no loc for line {line}"));
        IrPc {
            module: 0,
            func: l.func,
            block: l.block as usize,
            inst: l.inst as usize,
        }
    };
    let sp = 32768i64;
    let read_x_at = |line: u32| {
        let mut insp = Inspector::attach(&m, 0, &[Value::I64(sp), Value::I32(5)], 1_000_000);
        insp.set_breakpoint(pc_for_line(line));
        assert!(matches!(
            insp.run_until_stop(),
            svm_interp::Stop::Break { .. }
        ));
        as_i32_var(insp.read_var(0, "x", 4))
    };

    // Inside the inner block (line 6, `n = n + x`): the inner `x = n + 100 = 105` is in scope.
    assert_eq!(read_x_at(6), 105, "inner shadow resolved inside the block");
    // After the block (line 8, `return x + n`): the outer `x = n + 1 = 6` is back in scope.
    assert_eq!(read_x_at(8), 6, "outer x resolved after the block");
}

#[test]
fn chibicc_g_maps_breakpoints_to_source_lines() {
    // Raw string starts with a newline, so: line 2 = signature, 3 = `int s`, 4 = `int t`,
    // 5 = `return t + s` (its add is the block's last op, so it's a real step point on line 5).
    let src = r#"
int compute(int a, int b) {
  int s = a + b;
  int t = s + 100;
  return t + s;
}
"#;
    let ir = c_to_ir_g(src);
    assert!(ir.contains("debug.loc 0 "), "emits debug.loc rows:\n{ir}");
    assert!(ir.contains("debug.file 0 "), "emits a debug.file");
    let m = parse_module(&ir).expect("parse");

    let sp = 32768i64;
    let mut insp = Inspector::attach(
        &m,
        0,
        &[Value::I64(sp), Value::I32(5), Value::I32(3)],
        1_000_000,
    );

    // The last op of the single block is the return's value load → the `return t` line (5).
    let last = m.funcs[0].blocks[0].insts.len() - 1;
    let bp = IrPc {
        module: 0,
        func: 0,
        block: 0,
        inst: last,
    };
    insp.set_breakpoint(bp);
    assert!(matches!(
        insp.run_until_stop(),
        svm_interp::Stop::Break { .. }
    ));

    let loc = insp.source_loc(bp).expect("source loc at the return");
    assert_eq!(loc.line, 5, "last block op maps to `return t`");
    assert!(
        loc.file.ends_with(".c"),
        "file is the C source: {}",
        loc.file
    );
    // The backtrace frame carries the same source line.
    assert_eq!(insp.backtrace()[0].source.as_ref().map(|s| s.line), Some(5));
    // And `t` is still readable by name (= s + 100 = 108).
    assert_eq!(as_i32_var(insp.read_var(0, "t", 4)), 108);
}

#[test]
fn chibicc_g_emits_structured_types_with_field_offsets() {
    use svm_ir::{Encoding, TypeDef};

    // A struct local carries its layout through the §6 `TypeRef` waist: the type table records
    // `struct Point { int x; int y; }` with field offsets, an array records its element + count,
    // and a pointer records its pointee — the data aggregate inspection (struct/array expansion,
    // `a.b` / `arr[i]`) needs.
    let src = r#"
struct Point { int x; int y; };
int dist(int ax, int ay) {
  struct Point p;
  int row[4];
  struct Point *pp;
  p.x = ax;
  p.y = ay;
  pp = &p;
  row[0] = pp->x;
  return p.x + p.y + row[0];
}
"#;
    let ir = c_to_ir_g(src);
    // Producer side: the structured directives are present in the text.
    assert!(
        ir.contains("debug.type") && ir.contains(" agg \"struct Point\""),
        "emits an aggregate type named by its tag:\n{ir}"
    );
    assert!(
        ir.contains("debug.field") && ir.contains("\"y\" 4"),
        "emits field `y` at offset 4:\n{ir}"
    );

    // ABI side: parse and walk the structured types.
    let m = parse_module(&ir).expect("parse");
    let di = m.debug_info.as_ref().expect("debug info");
    // The test is about the structured *type* table; a var's `loc` (window for the aggregates, a
    // promoted `ssalist` for the pointer) is incidental, so resolve only via its `type_id`.
    let resolve = |name: &str| -> &TypeDef {
        let v = di
            .vars
            .iter()
            .find(|v| v.name == name)
            .unwrap_or_else(|| panic!("var {name}"));
        let tid = v
            .type_id
            .unwrap_or_else(|| panic!("{name} carries a structured type"));
        &di.types[tid as usize]
    };

    // `struct Point p` — an aggregate with x@0, y@4, both 4-byte signed ints, size 8.
    let TypeDef::Aggregate { name, size, fields } = resolve("p") else {
        panic!("p is a struct");
    };
    assert_eq!(name, "struct Point"); // render name carries the C tag
    assert_eq!(*size, 8);
    assert_eq!(fields.len(), 2);
    assert_eq!((fields[0].name.as_str(), fields[0].offset), ("x", 0));
    assert_eq!((fields[1].name.as_str(), fields[1].offset), ("y", 4));
    for f in fields {
        let TypeDef::Base { encoding, size, .. } = &di.types[f.ty as usize] else {
            panic!("field {} is a base type", f.name);
        };
        assert_eq!((*encoding, *size), (Encoding::Signed, 4));
    }

    // `int row[4]` — an array of 4 elements; element resolves to a 4-byte int.
    let TypeDef::Array { elem, count, name } = resolve("row") else {
        panic!("row is an array");
    };
    assert_eq!(name, "int[4]", "composite array render name");
    assert_eq!(*count, 4);
    assert!(matches!(
        &di.types[*elem as usize],
        TypeDef::Base { size: 4, .. }
    ));

    // `struct Point *pp` — a pointer (named `struct Point *`) whose pointee is the same aggregate.
    let TypeDef::Pointer {
        pointee,
        size,
        name,
    } = resolve("pp")
    else {
        panic!("pp is a pointer");
    };
    assert_eq!(name, "struct Point *", "composite pointer render name");
    assert_eq!(*size, 8, "pointer width");
    assert!(
        matches!(&di.types[*pointee as usize], TypeDef::Aggregate { name, .. } if name == "struct Point"),
        "pp points at the struct"
    );
}

#[test]
fn chibicc_g_location_list_tracks_a_loop_accumulator_across_blocks() {
    // The headline of the `ssalist` producer: a promoted accumulator/counter changes value across
    // the loop's blocks (a different block parameter each iteration), and a mid-body write. The
    // emitted location list must resolve each to the right value at the loop-body breakpoint — i.e.
    // chibicc can debug the *optimized* (promoted) build, no `-Og`. Line 5 is `acc = acc + i;`.
    let src = r#"
int run(int n) {
  int acc = 0;
  for (int i = 0; i < n; i = i + 1) {
    acc = acc + i;
  }
  return acc;
}
"#;
    let ir = c_to_ir_g(src);
    let m = parse_module(&ir).expect("parse");
    let di = m.debug_info.as_ref().expect("debug info");

    // Breakpoint at the loop body (line 5), found via the emitted source map — its first IR pc.
    let bl = di
        .locs
        .iter()
        .filter(|l| l.line == 5)
        .min_by_key(|l| (l.block, l.inst))
        .expect("line 5 (loop body) is mapped");
    let bp = IrPc {
        module: 0,
        func: 0,
        block: bl.block as usize,
        inst: bl.inst as usize,
    };
    let mut insp = Inspector::attach(&m, 0, &[Value::I64(32768), Value::I32(3)], 1_000_000);
    insp.set_breakpoint(bp);

    // Before `acc = acc + i` each iteration, (i, acc) = (0,0), (1,0), (2,1) — read by name from the
    // location lists as the promoted values shift across the loop blocks.
    for (expect_i, expect_acc) in [(0, 0), (1, 0), (2, 1)] {
        assert!(matches!(
            insp.run_until_stop(),
            svm_interp::Stop::Break { .. }
        ));
        assert_eq!(
            as_i32_var(insp.read_var(0, "i", 4)),
            expect_i,
            "i per iteration"
        );
        assert_eq!(
            as_i32_var(insp.read_var(0, "acc", 4)),
            expect_acc,
            "acc per iteration"
        );
    }
}
