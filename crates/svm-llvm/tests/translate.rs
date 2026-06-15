//! Translator tests: compile C through *stock clang* to legalized bitcode, translate it to SVM
//! IR, **verify** it (the untrusted-frontend re-check, §2a), and run it on **both** the reference
//! interpreter and the Cranelift JIT — asserting they agree with each other and with the
//! hand-computed result. This is the chibicc-as-oracle differential (LLVM.md §5) plus the §18
//! interp↔JIT differential, applied to the LLVM on-ramp.

use std::path::PathBuf;
use std::process::Command;

use svm_interp::Value;
use svm_ir::ValType;
use svm_jit::JitOutcome;

/// Compile a C snippet to legalized LLVM bitcode with the pinned pipeline (LLVM.md §4):
/// `-O2` runs `mem2reg`/SROA (the §3a two-stack split for free); `-fno-*-vectorize` keeps SIMD
/// out of the MVP. Returns `None` (skip, don't fail) when `clang` is unavailable.
fn compile_to_bc(name: &str, src: &str) -> Option<PathBuf> {
    let dir = std::env::temp_dir();
    let c = dir.join(format!("svm_llvm_{}_{}.c", std::process::id(), name));
    let bc = dir.join(format!("svm_llvm_{}_{}.bc", std::process::id(), name));
    std::fs::write(&c, src).expect("write C source");
    let status = Command::new("clang")
        .args([
            "-O2",
            "-emit-llvm",
            "-c",
            "-fno-vectorize",
            "-fno-slp-vectorize",
        ])
        .arg(&c)
        .arg("-o")
        .arg(&bc)
        .status();
    match status {
        Ok(s) if s.success() => Some(bc),
        _ => {
            eprintln!("note: skipping {name} (clang unavailable)");
            None
        }
    }
}

fn to_slot(v: &Value) -> i64 {
    match v {
        Value::I32(x) => *x as i64,
        Value::I64(x) => *x,
        Value::F32(x) => x.to_bits() as i64,
        Value::F64(x) => x.to_bits() as i64,
        other => panic!("unsupported arg {other:?}"),
    }
}

fn from_slot(t: ValType, s: i64) -> Value {
    match t {
        ValType::I32 => Value::I32(s as i32),
        ValType::I64 => Value::I64(s),
        ValType::F32 => Value::F32(f32::from_bits(s as u32)),
        ValType::F64 => Value::F64(f64::from_bits(s as u64)),
        other => panic!("unsupported result type {other:?}"),
    }
}

/// Translate `src` (one defined function, the unit under test at index 0), verify, and run on
/// **both** backends with `args`; assert they agree and equal `expect`. Returns silently if clang
/// is unavailable.
fn check(name: &str, src: &str, args: &[Value], expect: &[Value]) {
    let Some(bc) = compile_to_bc(name, src) else {
        return;
    };
    let t = svm_llvm::translate_bc_path(&bc).expect("translate bitcode");
    let module = t.module;
    svm_verify::verify_module(&module).expect("verify translated IR");
    let results = module.funcs[0].results.clone();

    // The IR signature prepends the data-SP (§3d): the entry takes `(sp, c-args…)`. Pass the
    // translator-computed initial data-stack base (just above the globals), then the C arguments.
    let mut full: Vec<Value> = vec![Value::I64(t.entry_sp as i64)];
    full.extend_from_slice(args);

    let mut fuel = 10_000_000u64;
    let interp = svm_interp::run(&module, 0, &full, &mut fuel).expect("interp run");
    assert_eq!(interp, expect, "{name}: interp result");

    let slots: Vec<i64> = full.iter().map(to_slot).collect();
    let jit = match svm_jit::compile_and_run(&module, 0, &slots).expect("jit run") {
        JitOutcome::Returned(s) => s
            .iter()
            .zip(&results)
            .map(|(&v, &t)| from_slot(t, v))
            .collect::<Vec<_>>(),
        other => panic!("{name}: unexpected JIT outcome {other:?}"),
    };
    assert_eq!(jit, expect, "{name}: JIT result (interp said {interp:?})");
}

/// Translate + verify + run on both backends, asserting both **trap** (neither returns a value).
/// Used for the data-stack guard: a deep recursion with a real frame must fault past the window's
/// mapped region, not corrupt globals or return garbage.
fn check_traps(name: &str, src: &str, args: &[Value]) {
    let Some(bc) = compile_to_bc(name, src) else {
        return;
    };
    let t = svm_llvm::translate_bc_path(&bc).expect("translate bitcode");
    let module = t.module;
    svm_verify::verify_module(&module).expect("verify translated IR");
    let mut full: Vec<Value> = vec![Value::I64(t.entry_sp as i64)];
    full.extend_from_slice(args);

    let mut fuel = 1_000_000_000u64;
    let interp = svm_interp::run(&module, 0, &full, &mut fuel);
    assert!(
        interp.is_err(),
        "{name}: interp should trap, got {interp:?}"
    );

    let slots: Vec<i64> = full.iter().map(to_slot).collect();
    match svm_jit::compile_and_run(&module, 0, &slots).expect("jit run") {
        JitOutcome::Trapped(_) => {}
        other => panic!("{name}: JIT should trap, got {other:?}"),
    }
}

#[test]
fn data_stack_guard_traps_on_overflow() {
    // A non-tail recursion (the `+ buf[k]` after the call keeps the result live) with a large
    // 32 KiB frame — a *runtime* index `k` stops clang from shrinking the `volatile` array. A
    // shallow call returns; a deep one overflows the data stack past the window's mapped region
    // and faults (§5) — the guard catches it, no corruption. deep(n) sums 0..n (buf[k] == n).
    let src = "int deep(int n){ volatile int buf[8192]; int k = n & 8191; buf[k] = n; \
               if (n <= 0) return 0; return deep(n - 1) + buf[k]; }";
    check("deep_3", src, &[Value::I32(3)], &[Value::I32(6)]); // 4 frames ≪ window
    check_traps("deep_overflow", src, &[Value::I32(2000)]); // ~64 frames fills the ~2 MiB window
}

#[test]
fn returns_constant() {
    check(
        "ret_const",
        "int main(void){ return 42; }",
        &[],
        &[Value::I32(42)],
    );
}

#[test]
fn integer_add_with_params() {
    check(
        "add",
        "int f(int a, int b){ return a + b; }",
        &[Value::I32(40), Value::I32(2)],
        &[Value::I32(42)],
    );
}

#[test]
fn i64_arithmetic() {
    check(
        "mul64",
        "long g(long a, long b){ return a * b - 1; }",
        &[Value::I64(7), Value::I64(6)],
        &[Value::I64(41)],
    );
}

#[test]
fn icmp_select_zext() {
    // -O2 if-converts this to icmp + zext i1 + select (a single block) — exercises the
    // comparison, the i1→i32 zero-extend, and branchless select.
    let src = "int classify(int x){ if (x < 0) return -1; if (x == 0) return 0; return 1; }";
    check("classify_neg", src, &[Value::I32(-5)], &[Value::I32(-1)]);
    check("classify_zero", src, &[Value::I32(0)], &[Value::I32(0)]);
    check("classify_pos", src, &[Value::I32(7)], &[Value::I32(1)]);
}

#[test]
fn loop_with_phi_popcount() {
    // A data-dependent loop -O2 cannot close-form: a multi-block CFG with φ-nodes for the loop
    // variables — the SSA→block-argument conversion (the slice-A headline).
    let src = "int popcount(unsigned x){ int c = 0; while (x) { c += x & 1; x >>= 1; } return c; }";
    check("pop_ff", src, &[Value::I32(0xFF)], &[Value::I32(8)]);
    check("pop_0", src, &[Value::I32(0)], &[Value::I32(0)]);
    check("pop_mix", src, &[Value::I32(0x10101)], &[Value::I32(3)]);
}

#[test]
fn loop_with_branchy_body() {
    // Collatz: a loop whose body branches (if/else) — back-edges, a join with φ, and nested
    // control flow, all over i32.
    let src = "int collatz(int n){ int c = 0; while (n != 1) { if (n & 1) n = 3*n + 1; else n = n/2; c++; } return c; }";
    check("collatz_6", src, &[Value::I32(6)], &[Value::I32(8)]);
    check("collatz_1", src, &[Value::I32(1)], &[Value::I32(0)]);
    check("collatz_27", src, &[Value::I32(27)], &[Value::I32(111)]);
}

#[test]
fn shifts_and_divrem() {
    let src = "int mix(int a, int b){ return (a << 3) | (a >> 1) ^ (a / b) + (a % b); }";
    // a=29,b=5: (29<<3)=232, (29>>1)=14, 29/5=5, 29%5=4 -> 5+4=9; 14^9=7; 232|7=239
    check(
        "mix",
        src,
        &[Value::I32(29), Value::I32(5)],
        &[Value::I32(239)],
    );
}

#[test]
fn stack_array_sum() {
    // An address-taken stack array indexed by a loop variable — `-O2` keeps it in memory
    // (`alloca [N x i32]`, GEP, store/load), exercising the §3d data-stack frame. n ≤ 8 (array
    // bound). sum of i*i for i in 0..n.
    let src = "int sumsq(int n){ int a[8]; for(int i=0;i<n;i++) a[i]=i*i; int s=0; for(int i=0;i<n;i++) s+=a[i]; return s; }";
    check("sumsq_5", src, &[Value::I32(5)], &[Value::I32(30)]); // 0+1+4+9+16
    check("sumsq_8", src, &[Value::I32(8)], &[Value::I32(140)]); // +25+36+49
    check("sumsq_0", src, &[Value::I32(0)], &[Value::I32(0)]);
}

#[test]
fn stack_array_reverse() {
    // Write then read in reverse — distinct store and load address arithmetic over the frame.
    let src = "int revsum(int n){ int a[8]; for(int i=0;i<n;i++) a[i]=i+1; int s=0; for(int i=n-1;i>=0;i--) s=s*10+a[i]; return s; }";
    check("rev_4", src, &[Value::I32(4)], &[Value::I32(4321)]);
    check("rev_3", src, &[Value::I32(3)], &[Value::I32(321)]);
}

#[test]
fn recursive_call_fib() {
    // Self-recursion: the call survives `-O2` (can't fully inline), exercising direct `call` by
    // index, the threaded data-SP, and result/argument marshalling. fib(0..) = 0,1,1,2,3,5,8,...
    let src = "int fib(int n){ if (n < 2) return n; return fib(n-1) + fib(n-2); }";
    check("fib_10", src, &[Value::I32(10)], &[Value::I32(55)]);
    check("fib_1", src, &[Value::I32(1)], &[Value::I32(1)]);
    check("fib_0", src, &[Value::I32(0)], &[Value::I32(0)]);
}

#[test]
fn cross_function_call() {
    // A call to a *different* function (kept distinct by `noinline`), so the entry (`g`, index 0)
    // calls `add1` (index 1) — exercises the name→index resolution and a non-recursive call.
    let src = "int add1(int x); int g(int x){ return add1(x) + add1(x + 10); } \
               __attribute__((noinline)) int add1(int x){ return x + 1; }";
    check("g_5", src, &[Value::I32(5)], &[Value::I32(22)]); // (5+1)+(15+1)=22
    check("g_0", src, &[Value::I32(0)], &[Value::I32(12)]); // 1 + 11
}

#[test]
fn switch_dense() {
    // A dense `switch` -O2 keeps as a jump table → `br_table`.
    let src = "int sw(int x){ switch (x) { case 0: return 100; case 1: return 200; \
               case 2: return 300; case 3: return 400; default: return -1; } }";
    check("sw_0", src, &[Value::I32(0)], &[Value::I32(100)]);
    check("sw_2", src, &[Value::I32(2)], &[Value::I32(300)]);
    check("sw_3", src, &[Value::I32(3)], &[Value::I32(400)]);
    check("sw_def", src, &[Value::I32(9)], &[Value::I32(-1)]);
    check("sw_neg", src, &[Value::I32(-5)], &[Value::I32(-1)]);
}

#[test]
fn mutual_recursion_even_odd() {
    // `-O2` lowers this mutual recursion into a `switch`-driven parity loop (the case that
    // motivated switch support). even(n) = 1 if n even else 0.
    let src = "int odd(int); \
               int even(int n){ return n == 0 ? 1 : odd(n - 1); } \
               int odd(int n){ return n == 0 ? 0 : even(n - 1); }";
    check("even_10", src, &[Value::I32(10)], &[Value::I32(1)]);
    check("even_7", src, &[Value::I32(7)], &[Value::I32(0)]);
    check("even_0", src, &[Value::I32(0)], &[Value::I32(1)]);
}

#[test]
fn global_const_table() {
    // A `static const` lookup table → an `internal constant [4 x i32]` global; the read is a
    // GEP on the global's window address + a load. Exercises read-only `data` segments (D40).
    let src = "int tbl(int i){ static const int t[4] = {10,20,30,40}; return t[i & 3]; }";
    check("tbl_0", src, &[Value::I32(0)], &[Value::I32(10)]);
    check("tbl_2", src, &[Value::I32(2)], &[Value::I32(30)]);
    check("tbl_5", src, &[Value::I32(5)], &[Value::I32(20)]); // 5 & 3 == 1
}

#[test]
fn global_mutable_counter() {
    // A mutable initialized global → a writable `data` segment; `++g` is load + add + store.
    let src = "int g = 7; int bump(void){ return ++g; }";
    check("bump", src, &[], &[Value::I32(8)]);
}

#[test]
fn global_string_indexed() {
    // A string literal → a `[N x i8]` constant global; a runtime-indexed read is GEP + narrow
    // (`i8`) load, sign-extended to the `char` return.
    let src = "int nth(int i){ return \"Xyz!\"[i & 3]; }";
    check("nth_0", src, &[Value::I32(0)], &[Value::I32('X' as i32)]);
    check("nth_1", src, &[Value::I32(1)], &[Value::I32('y' as i32)]);
    check("nth_3", src, &[Value::I32(3)], &[Value::I32('!' as i32)]);
}

#[test]
fn switch_with_gaps_via_global_table() {
    // `-O2` compiles a gapped switch into a global lookup table + a range-check + GEP/load — so
    // this now works via global-variable support (it was the case that revealed the need).
    let src = "int sg(int x){ switch (x) { case 2: return 20; case 5: return 50; \
               case 8: return 80; default: return 0; } }";
    check("sg_2", src, &[Value::I32(2)], &[Value::I32(20)]);
    check("sg_5", src, &[Value::I32(5)], &[Value::I32(50)]);
    check("sg_4", src, &[Value::I32(4)], &[Value::I32(0)]); // a gap → default
    check("sg_8", src, &[Value::I32(8)], &[Value::I32(80)]);
}

#[test]
fn float_arithmetic_and_fmuladd() {
    // a*b + a/b - b — `-O2` contracts `a*b + (a/b)` into `llvm.fmuladd`, which we lower unfused.
    let src = "double fa(double a, double b){ return a*b + a/b - b; }";
    check(
        "fa",
        src,
        &[Value::F64(6.0), Value::F64(2.0)],
        &[Value::F64(13.0)],
    );
}

#[test]
fn float_compare() {
    let src = "int cmp(double a, double b){ return a < b ? 1 : (a == b ? 0 : -1); }";
    check(
        "cmp_lt",
        src,
        &[Value::F64(1.0), Value::F64(2.0)],
        &[Value::I32(1)],
    );
    check(
        "cmp_eq",
        src,
        &[Value::F64(2.0), Value::F64(2.0)],
        &[Value::I32(0)],
    );
    check(
        "cmp_gt",
        src,
        &[Value::F64(3.0), Value::F64(2.0)],
        &[Value::I32(-1)],
    );
}

#[test]
fn float_int_conversions() {
    check(
        "i2d",
        "double i2d(int n){ return (double)n + 0.5; }",
        &[Value::I32(3)],
        &[Value::F64(3.5)],
    );
    check(
        "d2i",
        "int d2i(double x){ return (int)(x * 2.0); }",
        &[Value::F64(3.5)],
        &[Value::I32(7)],
    );
}

#[test]
fn float_promote_demote() {
    // f32 → f64 (fpext), arithmetic, then f64 → f32 (fptrunc).
    let src = "float fp(float a, double b){ return (float)(a + b); }";
    check(
        "fp",
        src,
        &[Value::F32(1.5), Value::F64(2.25)],
        &[Value::F32(3.75)],
    );
}

#[test]
fn float_intrinsics_abs_floor() {
    // `fabs`/`floor` lower to `llvm.fabs`/`llvm.floor` (errno-free, so real intrinsics at -O2,
    // unlike `sqrt` which stays a libc call pending the libc-binding slice).
    check(
        "ab",
        "double ab(double x){ return __builtin_fabs(x); }",
        &[Value::F64(-3.5)],
        &[Value::F64(3.5)],
    );
    check(
        "fl",
        "double fl(double x){ return __builtin_floor(x); }",
        &[Value::F64(3.7)],
        &[Value::F64(3.0)],
    );
    check(
        "fl_neg",
        "double fl(double x){ return __builtin_floor(x); }",
        &[Value::F64(-2.1)],
        &[Value::F64(-3.0)],
    );
}

#[test]
fn unsupported_is_fail_closed() {
    // A 128-bit integer is outside the subset — it must be a clean `Unsupported`, never a silent
    // mis-translation (LLVM.md §2/§8, the fail-closed chokepoint).
    let Some(bc) = compile_to_bc("i128", "__int128 big(__int128 a){ return a + 1; }") else {
        return;
    };
    match svm_llvm::translate_bc_path(&bc) {
        Err(svm_llvm::Error::Unsupported(_)) => {}
        other => panic!("expected Unsupported, got {other:?}"),
    }
}
