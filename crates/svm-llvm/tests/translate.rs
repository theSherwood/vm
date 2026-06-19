//! Translator tests: compile C through *stock clang* to legalized bitcode, translate it to SVM
//! IR, **verify** it (the untrusted-frontend re-check, §2a), and run it on **both** the reference
//! interpreter and the Cranelift JIT — asserting they agree with each other and with the
//! hand-computed result. This is the chibicc-as-oracle differential (LLVM.md §5) plus the §18
//! interp↔JIT differential, applied to the LLVM on-ramp.

use std::path::{Path, PathBuf};
use std::process::Command;

use svm_interp::Value;
use svm_ir::ValType;
use svm_jit::JitOutcome;

/// Compile a C snippet to legalized LLVM bitcode with the pinned pipeline (LLVM.md §4): `-O2` runs
/// `mem2reg`/SROA (the §3a two-stack split for free) **and auto-vectorization** — the on-ramp now
/// ingests the full SIMD output (slices AN–AT: i32x4 → legalization → conversions/rotate/shuffle/
/// `<N x i1>` masks). Returns `None` (skip, don't fail) when `clang` is unavailable.
fn compile_to_bc(name: &str, src: &str) -> Option<PathBuf> {
    let dir = std::env::temp_dir();
    let c = dir.join(format!("svm_llvm_{}_{}.c", std::process::id(), name));
    let bc = dir.join(format!("svm_llvm_{}_{}.bc", std::process::id(), name));
    std::fs::write(&c, src).expect("write C source");
    let status = Command::new("clang")
        .args(["-O2", "-emit-llvm", "-c"])
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

/// Compile a C snippet to legalized bitcode **with debug info** (`-g`). Uses `-Og` (optimize for
/// debugging): mem2reg/SROA still run — so scalars arrive promoted, the legalized shape the on-ramp
/// needs — while the per-statement line table is preserved (`-O2` collapses a tiny function's lines
/// onto one). So the §6 source-line ingest can be exercised against real, multi-line clang debug
/// metadata. `None` (skip) if clang is unavailable.
fn compile_to_bc_g(name: &str, src: &str) -> Option<PathBuf> {
    compile_g(name, src, "-Og")
}

/// Compile at `-O0 -g`: every C local stays an `alloca` + `llvm.dbg.declare`, the shape the §6
/// **variable** ingest reads (a `dbg.declare` → a `Window` frame slot). `None` (skip) if clang is
/// unavailable.
fn compile_to_bc_o0g(name: &str, src: &str) -> Option<PathBuf> {
    compile_g(name, src, "-O0")
}

fn compile_g(name: &str, src: &str, opt: &str) -> Option<PathBuf> {
    let dir = std::env::temp_dir();
    let c = dir.join(format!("svm_llvm_g_{}_{}.c", std::process::id(), name));
    let bc = dir.join(format!("svm_llvm_g_{}_{}.bc", std::process::id(), name));
    std::fs::write(&c, src).expect("write C source");
    let status = Command::new("clang")
        .args([opt, "-g", "-emit-llvm", "-c"])
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

/// Compile `src` (which defines `int run(int)` first and an `int main(){ return run(SEED); }`)
/// with native `cc`, run it, and assert the SVM translation of `run(SEED)` matches the native exit
/// code on **both** backends. The native compiler is the strongest oracle (the chibicc Tier-2
/// pattern); `run` returns a byte so the full result survives the 8-bit Unix exit code.
fn check_vs_native(name: &str, src: &str, seed: i32) {
    let Some(bc) = compile_to_bc(name, src) else {
        return;
    };
    let exe = std::env::temp_dir().join(format!("svm_llvm_native_{}_{}", std::process::id(), name));
    let c = std::env::temp_dir().join(format!("svm_llvm_{}_{}.c", std::process::id(), name));
    // `-lm` so demos that call libm (`sqrt`/`floor`/…) link natively; harmless for the rest.
    match Command::new("cc")
        .arg(&c)
        .arg("-lm")
        .arg("-o")
        .arg(&exe)
        .status()
    {
        Ok(s) if s.success() => {}
        _ => {
            eprintln!("note: skipping {name} (cc unavailable)");
            return;
        }
    }
    let native = Command::new(&exe)
        .status()
        .expect("run native")
        .code()
        .unwrap() as u8;

    let t = svm_llvm::translate_bc_path(&bc).expect("translate bitcode");
    let module = t.module;
    svm_verify::verify_module(&module).expect("verify translated IR");
    let full = vec![Value::I64(t.entry_sp as i64), Value::I32(seed)];
    let mut fuel = 100_000_000u64;
    let interp = svm_interp::run(&module, 0, &full, &mut fuel).expect("interp run");
    let slots: Vec<i64> = full.iter().map(to_slot).collect();
    let jit = match svm_jit::compile_and_run(&module, 0, &slots).expect("jit run") {
        JitOutcome::Returned(s) => Value::I32(s[0] as i32),
        other => panic!("{name}: unexpected JIT outcome {other:?}"),
    };
    assert_eq!(interp, vec![jit], "{name}: interp vs JIT");
    let svm = match jit {
        Value::I32(x) => x as u8,
        _ => panic!("expected i32"),
    };
    assert_eq!(svm, native, "{name}: svm={svm} vs native cc={native}");
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
fn byval_small_struct_arg() {
    // A small struct passed by value — clang coerces it to an `i64` register. `run` packs {a,b}
    // and calls `sumP(i64)`; the callee unpacks the fields.
    let src = "struct P { int x; int y; }; int sumP(struct P); \
               int run(int a, int b){ struct P p = {a, b}; return sumP(p); } \
               __attribute__((noinline)) int sumP(struct P p){ return p.x + p.y; }";
    check(
        "bvarg",
        src,
        &[Value::I32(3), Value::I32(4)],
        &[Value::I32(7)],
    );
}

#[test]
fn byval_small_struct_return() {
    // A small struct returned by value — coerced to an `i64` return. `run` calls `mkP` and reads
    // the returned fields.
    let src = "struct P { int x; int y; }; struct P mkP(int, int); \
               int run(int a, int b){ struct P p = mkP(a, b); return p.x * 10 + p.y; } \
               __attribute__((noinline)) struct P mkP(int a, int b){ struct P p = {a, b}; return p; }";
    check(
        "bvret",
        src,
        &[Value::I32(3), Value::I32(4)],
        &[Value::I32(34)],
    );
}

#[test]
fn byval_two_eightbyte_struct() {
    // A 12-byte struct coerced to *two* registers `(i64, i32)` (two eightbytes).
    let src = "struct Q { int x; int y; int z; }; int sumQ(struct Q); \
               int run(int a, int b, int c){ struct Q q = {a, b, c}; return sumQ(q); } \
               __attribute__((noinline)) int sumQ(struct Q q){ return q.x + q.y + q.z; }";
    check(
        "bvq",
        src,
        &[Value::I32(1), Value::I32(2), Value::I32(3)],
        &[Value::I32(6)],
    );
}

#[test]
fn byval_sse_struct() {
    // A `{double, double}` struct — two SSE eightbytes, coerced to `(double, double)`.
    let src = "struct DD { double a; double b; }; double useDD(struct DD); \
               double run(double x, double y){ struct DD d = {x, y}; return useDD(d); } \
               __attribute__((noinline)) double useDD(struct DD d){ return d.a * d.b; }";
    check(
        "bvdd",
        src,
        &[Value::F64(3.0), Value::F64(4.0)],
        &[Value::F64(12.0)],
    );
}

#[test]
fn byval_and_sret_large_struct() {
    // A large struct returned via `sret` (`mkBig`) and passed via `byval` (`sumBig`) — both are
    // hidden caller-allocated pointers in the IR.
    let src = "struct Big { int a[8]; }; long sumBig(struct Big); struct Big mkBig(int); \
               long run(int v, int i){ struct Big b = mkBig(v); return sumBig(b) + b.a[i & 7]; } \
               __attribute__((noinline)) struct Big mkBig(int v){ struct Big b; for(int i=0;i<8;i++) b.a[i]=v+i; return b; } \
               __attribute__((noinline)) long sumBig(struct Big b){ long s=0; for(int i=0;i<8;i++) s+=b.a[i]; return s; }";
    check(
        "bvbig",
        src,
        &[Value::I32(10), Value::I32(0)],
        &[Value::I64(118)],
    ); // sum 10..17 + 10
}

#[test]
fn int_minmax_bit_intrinsics() {
    // `a > b ? a : b` → `llvm.smax`; the bit builtins → `llvm.ctlz`/`ctpop`; `abs` → `llvm.abs`.
    check(
        "imax",
        "int imax(int a, int b){ return a > b ? a : b; }",
        &[Value::I32(3), Value::I32(7)],
        &[Value::I32(7)],
    );
    check(
        "clz",
        "int clz(unsigned x){ return __builtin_clz(x); }",
        &[Value::I32(1)],
        &[Value::I32(31)],
    );
    check(
        "pc",
        "int pc(unsigned x){ return __builtin_popcount(x); }",
        &[Value::I32(0xFF)],
        &[Value::I32(8)],
    );
    check(
        "absn",
        "int ab(int x){ return x < 0 ? -x : x; }",
        &[Value::I32(-5)],
        &[Value::I32(5)],
    );
}

#[test]
fn libm_math_calls() {
    // `sqrt`/`fmin` stay external libm calls at -O2 (errno) — we recognize the named function and
    // lower it to the SVM float op inline.
    check(
        "sq",
        "double sq(double x){ return __builtin_sqrt(x); }",
        &[Value::F64(16.0)],
        &[Value::F64(4.0)],
    );
    check(
        "mn",
        "double mn(double a, double b){ return __builtin_fmin(a, b); }",
        &[Value::F64(3.0), Value::F64(5.0)],
        &[Value::F64(3.0)],
    );
}

#[test]
fn function_pointer_table_global() {
    // A global `static fp tbl[2] = {inc, dec}` — a relocation: each element serializes to the
    // function's funcref index. `viafp` indexes the table at runtime and calls indirectly.
    let src = "int inc(int); int dec(int); typedef int(*fp)(int); \
               static fp tbl[2] = {inc, dec}; \
               int viafp(int sel, int x){ return tbl[sel & 1](x); } \
               __attribute__((noinline)) int inc(int x){ return x + 1; } \
               __attribute__((noinline)) int dec(int x){ return x - 1; }";
    check(
        "fpt_inc",
        src,
        &[Value::I32(0), Value::I32(10)],
        &[Value::I32(11)],
    );
    check(
        "fpt_dec",
        src,
        &[Value::I32(1), Value::I32(10)],
        &[Value::I32(9)],
    );
}

#[test]
fn struct_with_string_pointer_global() {
    // A global struct holding a string pointer — the pointer field is a relocation (`@.str`
    // address). The runtime reads the pointed-to char.
    let src = "struct S { const char *name; int v; }; \
               static const struct S g = { \"hi\", 7 }; \
               int f(int i){ return g.name[i] + g.v; }";
    check(
        "sws_h",
        src,
        &[Value::I32(0)],
        &[Value::I32('h' as i32 + 7)],
    );
    check(
        "sws_i",
        src,
        &[Value::I32(1)],
        &[Value::I32('i' as i32 + 7)],
    );
}

#[test]
fn memcpy_struct_copy() {
    // `struct Big q = G` → `llvm.memcpy` (32 bytes) from a const global into a stack struct; we
    // lower it to chunked load/stores. A runtime field read keeps the alloca + copy.
    let src = "struct Big { int a[8]; }; \
               static const struct Big G = { {1,2,3,4,5,6,7,8} }; \
               int pick(int i){ struct Big q = G; q.a[0] += 100; return q.a[i & 7]; }";
    check("pick_0", src, &[Value::I32(0)], &[Value::I32(101)]); // modified field
    check("pick_3", src, &[Value::I32(3)], &[Value::I32(4)]);
    check("pick_7", src, &[Value::I32(7)], &[Value::I32(8)]);
}

#[test]
fn memset_fill() {
    // `__builtin_memset(b, 0xAB, 16)` → `llvm.memset`; the fill byte is replicated across the
    // chunked stores. A signed-char read sign-extends 0xAB to -85.
    let src = "int hset(int i){ char b[16]; __builtin_memset(b, 0xAB, 16); return b[i & 15]; }";
    check("hset_0", src, &[Value::I32(0)], &[Value::I32(-85)]);
    check("hset_9", src, &[Value::I32(9)], &[Value::I32(-85)]);
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
fn indirect_call_via_function_pointer() {
    // `pick` (noinline) returns a function pointer; `run` calls it indirectly — `-O2` keeps it a
    // real `call_indirect`. Exercises taking a function's address (funcref), threading it as an
    // i64 pointer through a `select`/return, and the §3c masked + type-checked indirect dispatch.
    let src = "int inc(int); int dec(int); typedef int(*fp)(int); fp pick(int); \
               int run(int sel, int x){ return pick(sel)(x); } \
               __attribute__((noinline)) fp pick(int sel){ return sel ? inc : dec; } \
               __attribute__((noinline)) int inc(int x){ return x + 1; } \
               __attribute__((noinline)) int dec(int x){ return x - 1; }";
    check(
        "run_inc",
        src,
        &[Value::I32(1), Value::I32(10)],
        &[Value::I32(11)],
    );
    check(
        "run_dec",
        src,
        &[Value::I32(0), Value::I32(10)],
        &[Value::I32(9)],
    );
}

#[test]
fn global_struct_fields() {
    // A global struct {i32, i32, i64} read field-by-field — struct GEP (constant field offsets) +
    // a read-only struct `data` segment with the {1,2,3} initializer laid out with field padding.
    let src = "struct Point { int x; int y; long tag; }; \
               const struct Point g = {1, 2, 3}; \
               long sum(void){ return g.x + g.y + g.tag; }";
    check("gstruct", src, &[], &[Value::I64(6)]);
}

#[test]
fn array_of_structs() {
    // `arr[i].field` — an array-of-struct GEP: a variable array index (stride = struct size) then
    // a constant struct-field offset.
    let src = "struct P { int x; int y; }; \
               static const struct P arr[3] = {{1,2},{3,4},{5,6}}; \
               int get(int i){ return arr[i].x + arr[i].y; }";
    check("aos_1", src, &[Value::I32(1)], &[Value::I32(7)]); // 3+4
    check("aos_2", src, &[Value::I32(2)], &[Value::I32(11)]); // 5+6
}

#[test]
fn stack_struct() {
    // A `volatile` struct local stays on the data stack (`alloca` of the struct) — exercises a
    // struct-sized frame slot plus field store/load via struct GEP.
    let src = "struct Point { int x; int y; long tag; }; \
               long f(int a){ volatile struct Point p; p.x = a; p.y = a * 2; p.tag = a; \
               return p.x + p.y + p.tag; }";
    check("sstruct", src, &[Value::I32(5)], &[Value::I64(20)]); // 5+10+5
}

#[test]
fn kitchen_sink_vs_native() {
    // A large self-contained program exercising the whole translator at once — structs by value
    // (Vec3 → byval), a function-pointer table (`ops`, a relocation + indirect call), floats +
    // libm (`sqrt`/`fabs`), recursion (`fib`), loops + an array copy (→ memcpy), a const global
    // array, `switch`, and the int min/max + bit intrinsics — folded to a byte and checked against
    // native `cc`. `run` is defined first (func 0); `seed` is runtime so clang can't constant-fold.
    let src = "\
struct Vec3 { double x; double y; double z; }; \
double dot(struct Vec3, struct Vec3); int fib(int); int add(int,int); int mul(int,int); \
typedef int (*op)(int,int); static op ops[2] = { add, mul }; int apply(int,int,int); \
static const int squares[8] = { 0,1,4,9,16,25,36,49 }; \
int run(int seed){ \
    unsigned h = 2166136261u ^ (unsigned)seed; \
    struct Vec3 a = { 1.5, 2.0, (double)(3 + seed) }, b = { 4.0, 0.5, 2.0 }; \
    double d = dot(a, b); \
    h ^= (unsigned)(d * 8.0); h *= 16777619u; \
    double s = __builtin_sqrt(d * d) + __builtin_fabs((double)(-seed)); \
    h ^= (unsigned)s; h *= 16777619u; \
    h ^= (unsigned)fib(12 + (seed & 1)); \
    h ^= (unsigned)apply(0, seed, 4); h ^= (unsigned)apply(1, seed, 3); \
    int arr[16]; for (int i=0;i<16;i++) arr[i] = i*i + seed; \
    int arr2[16]; for (int i=0;i<16;i++) arr2[i] = arr[i]; \
    int sum = 0; for (int i=0;i<16;i++) sum += arr2[i]; \
    h ^= (unsigned)sum; \
    for (int i=0;i<8;i++){ int v; \
        switch ((i + seed) & 3){ \
            case 0: v = __builtin_popcount((unsigned)(i+1)); break; \
            case 1: v = (i > 4 ? i : 4); break; \
            case 2: v = __builtin_clz((unsigned)(i+1)); break; \
            default: v = squares[i] - i; break; } \
        h += (unsigned)v; } \
    return (int)((h ^ (h>>8) ^ (h>>16) ^ (h>>24)) & 0xFF); \
} \
int main(void){ return run(7); } \
double dot(struct Vec3 a, struct Vec3 b){ return a.x*b.x + a.y*b.y + a.z*b.z; } \
int fib(int n){ if (n < 2) return n; return fib(n-1) + fib(n-2); } \
int add(int a, int b){ return a + b; } \
int mul(int a, int b){ return a * b; } \
int apply(int sel, int a, int b){ return ops[sel & 1](a, b); }";
    check_vs_native("kitchen", src, 7);
}

/// The shared core of the powerbox differential: given the legalized `bc` and the C source file it
/// came from, build the source natively with `cc`, run both, and assert the SVM translation's stdout
/// **and** exit code match native. Exercises the whole Lane C on-ramp end-to-end (the synthesized
/// `_start`, the handle stash, libc → `Stream`/`Exit`). Skips silently if `cc` is unavailable.
fn powerbox_diff(name: &str, bc: &std::path::Path, c_src: &std::path::Path, stdin: &[u8]) {
    // Native oracle: build with `cc`, run, capture stdout + exit code.
    let exe = std::env::temp_dir().join(format!("svm_llvm_pb_{}_{}", std::process::id(), name));
    // `-lm` so demos that call libm (`sqrt`/`floor`/…) link natively; harmless for the rest.
    match Command::new("cc")
        .arg(c_src)
        .arg("-lm")
        .arg("-o")
        .arg(&exe)
        .status()
    {
        Ok(s) if s.success() => {}
        _ => {
            eprintln!("note: skipping {name} (cc unavailable)");
            return;
        }
    }
    let native = {
        use std::io::Write;
        let mut child = Command::new(&exe)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .spawn()
            .expect("spawn native");
        child.stdin.take().unwrap().write_all(stdin).ok();
        child.wait_with_output().expect("run native")
    };
    let native_code = native.status.code().unwrap_or(-1) as u8;

    // The on-ramp: translate → resolve §7 imports to concrete capabilities → verify → run.
    let t = svm_llvm::translate_bc_path(bc).expect("translate bitcode");
    assert!(
        svm_run::is_powerbox_entry(&t.module),
        "{name}: a libc program must produce a powerbox entry"
    );
    let module = svm_run::resolve_capability_imports(t.module).expect("resolve capability imports");
    svm_verify::verify_module(&module).expect("verify translated IR");
    let run = svm_run::run_powerbox(&module, stdin).expect("powerbox run");

    assert_eq!(
        run.stdout, native.stdout,
        "{name}: svm stdout {:?} vs native {:?}",
        run.stdout, native.stdout
    );
    let svm_code = match run.outcome {
        svm_run::Outcome::Exited(c) => c as u8,
        svm_run::Outcome::Returned(ref v) => match v.first() {
            Some(svm_interp::Value::I32(x)) => *x as u8,
            _ => 0,
        },
    };
    assert_eq!(
        svm_code, native_code,
        "{name}: svm exit {svm_code} vs native {native_code}"
    );
}

/// Compile a **powerbox program** (real I/O via libc) from an inline source string and run the
/// differential ([`powerbox_diff`]).
fn check_powerbox_vs_native(name: &str, src: &str, stdin: &[u8]) {
    let Some(bc) = compile_to_bc(name, src) else {
        return;
    };
    let c = std::env::temp_dir().join(format!("svm_llvm_{}_{}.c", std::process::id(), name));
    powerbox_diff(name, &bc, &c, stdin);
}

/// The argv differential: build/run the native binary with a **controlled `argv`** (`args[0]` as the
/// process name, via `arg0`, so it matches the SVM blob) and a cleared+seeded environment, then run
/// the SVM translation with the same vectors through [`svm_run::run_powerbox_with_args`], asserting
/// stdout + exit code match. This is the only way to compare a `main(int, char**)` program: native
/// `argv[0]` is otherwise the temp path, which the guest can't (and shouldn't) reproduce.
fn check_powerbox_vs_native_args(name: &str, src: &str, args: &[&str], env: &[&str]) {
    use std::os::unix::process::CommandExt;
    let Some(bc) = compile_to_bc(name, src) else {
        return;
    };
    let c = std::env::temp_dir().join(format!("svm_llvm_{}_{}.c", std::process::id(), name));
    std::fs::write(&c, src).expect("write c source");
    let exe = std::env::temp_dir().join(format!("svm_llvm_pba_{}_{}", std::process::id(), name));
    match Command::new("cc").arg(&c).arg("-o").arg(&exe).status() {
        Ok(s) if s.success() => {}
        _ => {
            eprintln!("note: skipping {name} (cc unavailable)");
            return;
        }
    }
    let mut cmd = Command::new(&exe);
    cmd.arg0(args[0]).args(&args[1..]).env_clear();
    for e in env {
        let (k, v) = e.split_once('=').expect("env entry KEY=VALUE");
        cmd.env(k, v);
    }
    let native = cmd.output().expect("run native");
    let native_code = native.status.code().unwrap_or(-1) as u8;

    let t = svm_llvm::translate_bc_path(&bc).expect("translate bitcode");
    let module = svm_run::resolve_capability_imports(t.module).expect("resolve capability imports");
    svm_verify::verify_module(&module).expect("verify translated IR");
    let argv: Vec<&[u8]> = args.iter().map(|s| s.as_bytes()).collect();
    let envv: Vec<&[u8]> = env.iter().map(|s| s.as_bytes()).collect();
    let run =
        svm_run::run_powerbox_with_args(&module, b"", &argv, &envv).expect("powerbox run (args)");

    assert_eq!(
        run.stdout, native.stdout,
        "{name}: svm stdout {:?} vs native {:?}",
        run.stdout, native.stdout
    );
    let svm_code = match run.outcome {
        svm_run::Outcome::Exited(c) => c as u8,
        svm_run::Outcome::Returned(ref v) => match v.first() {
            Some(svm_interp::Value::I32(x)) => *x as u8,
            _ => 0,
        },
    };
    assert_eq!(
        svm_code, native_code,
        "{name}: svm exit {svm_code} vs native {native_code}"
    );
}

/// Run the powerbox differential on a **real corpus demo** (`crates/svm-run/demos/<rel>`) — a
/// whole-program, self-contained C file (its own `memset`, `write`-only output). This is the D54
/// "matches native clang" exit criterion applied to an actual library. The file is compiled *in
/// place* so its same-directory `#include "…"`s resolve.
fn check_demo_vs_native(name: &str, rel: &str, stdin: &[u8]) {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../svm-run/demos")
        .join(rel);
    let bc = std::env::temp_dir().join(format!("svm_llvm_demo_{}_{}.bc", std::process::id(), name));
    let status = Command::new("clang")
        .args(["-O2", "-emit-llvm", "-c"])
        .arg(&path)
        .arg("-o")
        .arg(&bc)
        .status();
    match status {
        Ok(s) if s.success() => {}
        _ => {
            eprintln!("note: skipping {name} (clang unavailable)");
            return;
        }
    }
    powerbox_diff(name, &bc, &path, stdin);
}

#[test]
fn powerbox_hello_write() {
    // The headline Lane C demo: write a string to stdout, then return an exit code. The synthesized
    // `_start` grants the handles, `write(1, …)` lowers to the `Stream` capability, `main`'s return
    // is the exit code — all matching the native build byte-for-byte.
    let src = "#include <unistd.h>\n\
               int main(void){ write(1, \"hello, on-ramp!\\n\", 16); return 3; }";
    check_powerbox_vs_native("pb_hello", src, b"");
}

#[test]
fn powerbox_exit_capability() {
    // `exit(code)` lowers to the `Exit` capability (terminal): the program writes, then exits with a
    // non-zero code — the SVM `Outcome::Exited` must match the native process exit code.
    let src = "#include <unistd.h>\n#include <stdlib.h>\n\
               int main(void){ write(1, \"bye\\n\", 4); exit(7); }";
    check_powerbox_vs_native("pb_exit", src, b"");
}

#[test]
fn powerbox_echo_stdin() {
    // A stdin → stdout round-trip through the `Stream` capability (read on the stdin handle, write on
    // stdout), driven by a loop — exercises `read`, `write`, the handle stash, and a real data frame.
    let src = "#include <unistd.h>\n\
               int main(void){ char buf[64]; long n; \
               while ((n = read(0, buf, sizeof buf)) > 0) write(1, buf, n); return 0; }";
    check_powerbox_vs_native("pb_echo", src, b"ping pong\n");
}

#[test]
fn powerbox_computed_output() {
    // Compose the on-ramp's existing machinery with I/O: build a string in a stack buffer (a real
    // data frame + stores), then write it out — the byte-exact stdout must match native.
    let src = "#include <unistd.h>\n\
               int main(void){ char buf[16]; for (int i = 0; i < 10; i++) buf[i] = '0' + i; \
               buf[10] = '\\n'; write(1, buf, 11); return 0; }";
    check_powerbox_vs_native("pb_computed", src, b"");
}

#[test]
fn stdio_puts() {
    // `puts(s)` writes the string + a newline. clang keeps it as a `puts` call (the on-ramp supplies
    // the literal's length + the newline) — stdout must match native byte-for-byte.
    let src = "#include <stdio.h>\nint main(void){ puts(\"hello, stdio\"); return 0; }";
    check_powerbox_vs_native("pb_puts", src, b"");
}

#[test]
fn stdio_printf_constant_string() {
    // clang -O2 rewrites `printf("literal\n")` → `puts("literal")` — so a format-free printf works
    // through the same path (no varargs). Two lines exercise repeated calls.
    let src = "#include <stdio.h>\n\
               int main(void){ printf(\"first line\\n\"); printf(\"second line\\n\"); return 0; }";
    check_powerbox_vs_native("pb_printf_str", src, b"");
}

#[test]
fn stdio_putchar_loop() {
    // `putchar` (clang lowers it to `putc(c, stdout)`) writing each char of a computed range — a
    // single byte staged through the stash scratch per call.
    let src = "#include <stdio.h>\n\
               int main(void){ for (int c = 'A'; c <= 'F'; c++) putchar(c); putchar('\\n'); return 0; }";
    check_powerbox_vs_native("pb_putchar", src, b"");
}

#[test]
fn stdio_fwrite_and_fputs() {
    // `fwrite(buf, 1, n, stdout)` writes a byte slice; `fputs(s, stdout)` (clang lowers it to
    // `fwrite`) writes a string with no newline. Mix both, plus a trailing newline via putchar.
    let src = "#include <stdio.h>\n#include <string.h>\n\
               int main(void){ const char* a = \"abc\"; fwrite(a, 1, 3, stdout); \
               fputs(\"-def\", stdout); putchar('\\n'); return 0; }";
    check_powerbox_vs_native("pb_fwrite", src, b"");
}

#[test]
fn stdio_mixed_then_exit() {
    // Compose stdio output with the `exit` capability: print via puts/printf, then exit(non-zero) —
    // both the stdout bytes and the exit code must match the native build.
    let src = "#include <stdio.h>\n#include <stdlib.h>\n\
               int main(void){ puts(\"goodbye\"); printf(\"done\\n\"); exit(42); }";
    check_powerbox_vs_native("pb_mixed_exit", src, b"");
}

#[test]
fn funnel_shift_rotate() {
    // SHA-style rotates: clang -O2 turns `(x<<n)|(x>>(w-n))` into `llvm.fshl`/`fshr` (the operands
    // identical → a rotate). Lowered to `rotl`/`rotr`. Checked on i32 and i64 against hand values.
    let src = "unsigned rotr32(unsigned x, unsigned n){ return (x >> n) | (x << (32 - n)); } \
               unsigned long rotl64(unsigned long x, unsigned n){ return (x << n) | (x >> (64 - n)); }";
    // rotr32(0x12345678, 8) = 0x78123456
    check(
        "rotr32",
        src,
        &[Value::I32(0x12345678), Value::I32(8)],
        &[Value::I32(0x78123456u32 as i32)],
    );
}

#[test]
fn variable_length_memset_loop() {
    // A runtime-length zero-fill: clang's loop-idiom recognizer emits `llvm.memset.p0.i64` with a
    // non-constant length, which lowers to a call to the synthesized `__svm_memset` loop helper.
    // `run` zeroes `n` bytes of a stack buffer (seeded non-zero), then sums them → 0.
    let src = "int run(int n){ unsigned char buf[300]; \
               for (int i = 0; i < 300; i++) buf[i] = (unsigned char)(i + 1); \
               for (int i = 0; i < n; i++) buf[i] = 0; \
               int s = 0; for (int i = 0; i < 300; i++) s += buf[i]; return s; }";
    check_vs_native(
        "var_memset",
        &format!("{src} int main(){{ return run(300); }}"),
        300,
    );
}

#[test]
fn demo_sha256_vs_native() {
    // The first real corpus library end-to-end: B-Con's SHA-256 hashing "", "abc", and the
    // pangram, printing each digest as hex via `write` — byte-identical to the native `clang` build.
    // Exercises funnel-shift rotates, the variable-length `memset` loop helper, multi-function calls,
    // the data stack, a const global table, and the `Stream` capability all at once (the D54 goal).
    check_demo_vs_native("sha256", "sha256/sha_demo.c", b"");
}

#[test]
fn demo_xxhash_vs_native() {
    // xxHash (XXH32/64) over the same inputs — 32- and 64-bit funnel-shift rotates + wide integer
    // mixing. Already covered by slices A–P; byte-identical to native `clang`.
    check_demo_vs_native("xxhash", "xxhash/xxh_demo.c", b"");
}

#[test]
fn demo_perlin_vs_native() {
    // stb_perlin: float-heavy noise (`fmuladd`/`fabs` intrinsics, int↔float, a const gradient table).
    // The float coverage (slice F) + `llvm.abs` (slice M) carry it — matching native `clang`.
    check_demo_vs_native("perlin", "perlin/perlin_demo.c", b"");
}

#[test]
fn demo_regex_vs_native() {
    // kokke/tiny-regex-c: a backtracking matcher over a table of (pattern, text) cases. Exercises
    // `ptrtoint`/`freeze`, a constexpr GEP (interior string pointer), writable function-static arrays
    // sharing the globals region with read-only string literals (the page-isolation fix), and deep
    // recursive control flow — byte-identical to native `clang`.
    check_demo_vs_native("regex", "regex/regex_demo.c", b"");
}

#[test]
fn demo_jsmn_vs_native() {
    // jsmn: a zero-allocation JSON parser. Parses an embedded document into a fixed token array and
    // prints each token's type/size/text. Exercises `llvm.load.relative` — clang lowers the
    // type→name `switch` into a relative lookup table (`&str − &table` offsets) — plus struct-array
    // indexing and interior string pointers. Byte-identical to native `clang`.
    check_demo_vs_native("jsmn", "jsmn/jsmn_demo.c", b"");
}

#[test]
fn demo_heapgrow_vs_native() {
    // The §1a headline: a guest **grows its own heap past the initial window** — allocating eight
    // 128 KiB blocks (~1 MiB, ~16× the initial mapped window) through `malloc`, which commits
    // reserved-tail pages on demand via the `Memory` capability (`vm_map`). Fills/sums/frees each and
    // prints running totals — byte-identical to the native `cc` build (which uses the real `malloc`).
    check_demo_vs_native("heapgrow", "heapgrow/heapgrow.c", b"");
}

#[test]
fn heap_malloc_calloc_free() {
    // The allocator directly: a `malloc` large enough to force `vm_map` growth past the initial
    // window (filled/summed), a `free` (no-op), then a `calloc` that must read back as zero (freshly
    // committed pages are zeroed and the bump heap never reuses). Exit code = (s + z) & 0xff vs native.
    let src = "#include <stdlib.h>\n\
               int run(void){ int *a = (int*)malloc(300000 * sizeof(int)); \
               for (int i = 0; i < 300000; i++) a[i] = (i * 3 + 1) & 255; \
               long s = 0; for (int i = 0; i < 300000; i++) s += a[i]; free(a); \
               int *b = (int*)calloc(1000, sizeof(int)); \
               long z = 0; for (int i = 0; i < 1000; i++) z += b[i]; \
               return (int)((s + z) & 0xff); } \
               int main(void){ return run(); }";
    check_powerbox_vs_native("heap_alloc", src, b"");
}

#[test]
fn ro_and_writable_global_page_isolation() {
    // A read-only global (string literal) next to a writable one (a mutable array) must not share a
    // protected page: a write to the writable global would otherwise fault on the read-only page
    // (D40 is page-granular). `run` writes the array, reads the constant, returns their combination.
    let src = "static char buf[8]; static const char msg[] = \"hi\"; \
               int run(int n){ for (int i = 0; i < 8; i++) buf[i] = (char)(n + i); \
               return buf[3] + msg[0] + msg[1]; }";
    // buf[3] = n+3; msg = "hi" → 'h'(104) + 'i'(105). run(10): 13 + 104 + 105 = 222.
    check_vs_native(
        "ro_rw_page",
        &format!("{src} int main(){{ return run(10); }}"),
        10,
    );
}

#[test]
fn demo_clay_vs_native() {
    // Clay UI layout: 2D points/dimensions as `<2 x float>`/`<2 x i32>` vectors (loads/stores/`fadd`/
    // phi/extractelement) plus `{i64,ptr}` array returns — the on-ramp scalarizes each 2-lane vector
    // to a packed `i64`. Lays out a small UI and prints the render commands, byte-identical to native.
    // The eighth and final corpus demo (the D54 exit criterion).
    check_demo_vs_native("clay", "clay/clay_demo.c", b"");
}

#[test]
fn demo_calc_vs_native() {
    // The chibicc `calc` demo (a recursive-descent arithmetic calculator) through the LLVM on-ramp.
    // Exercises a **global array of string pointers** + a **global struct array holding function
    // pointers** (both relocations, slice K), **indirect calls** through that dispatch table (slice G
    // → `call_indirect`), and **recursion** (expr → term → factor). It drives itself from a global
    // expression table and writes `"<expr> = <result>"` rows — byte-identical to native `cc`. The
    // first of the two non-corpus chibicc demos the on-ramp now covers (LLVM is the main frontend).
    check_demo_vs_native("calc", "calc.c", b"");
}

#[test]
fn demo_rational_vs_native() {
    // The chibicc `rational` demo (exact-rational arithmetic) through the LLVM on-ramp. Where `calc`
    // stresses the function-pointer table, this hammers the **by-value aggregate ABI** (D39 / slice
    // J): every op takes two `struct Rat` *by value* and returns one *by value* (the hidden-`sret`
    // path), composed with recursion (Euclid's `gcd`) and an **indirect call that both passes and
    // returns a struct by value** through a global dispatch table — sret + a function-pointer
    // relocation + a struct-valued `call_indirect`, all at once. Byte-identical to native `cc`.
    check_demo_vs_native("rational", "rational.c", b"");
}

/// Build a **guest-concurrency** corpus demo (`crates/svm-run/demos/<rel>`) that pulls in chibicc's
/// bundled guest libc — `<pthread.h>` (a 1:1 threading layer over the `__vm_thread_spawn`/`join` +
/// futex + atomic builtins) and `<stdlib.h>` (`malloc`). clang compiles the demo with the chibicc
/// include dir on the path, so `pthread_create`/`pthread_mutex_*`/etc. resolve to that guest shim,
/// which the on-ramp lowers to the §12 primitives (`thread.spawn`, `i32.atomic.wait`/`notify`, the
/// `iN.atomic.*` ops). The same source built with native `cc` would instead use the platform pthreads
/// — but these demos call `__vm_*` builtins / guest fibers with no native symbol, so they have no
/// native oracle; the assertion is the **interleaving-invariant total** (the chibicc `c_guest_*`
/// contract, now via the LLVM frontend). `None` (skip) if clang is unavailable.
#[cfg(all(unix, target_arch = "x86_64"))]
fn compile_demo_libc_to_bc(name: &str, rel: &str) -> Option<PathBuf> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../svm-run/demos")
        .join(rel);
    let inc = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../frontend/chibicc/include");
    let bc = std::env::temp_dir().join(format!("svm_llvm_cc_{}_{}.bc", std::process::id(), name));
    let status = Command::new("clang")
        .args(["-O2", "-emit-llvm", "-c"])
        .arg("-I")
        .arg(&inc)
        .arg(&path)
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

/// Compile an inline C **source string** with chibicc's guest-libc include dir on the path (the
/// `<pthread.h>`/`<svm.h>` shims) → bitcode. The text variant of [`compile_demo_libc_to_bc`], for a
/// demo that must be *patched* before compiling (the guest-JIT blob descriptor). `None` if clang is
/// unavailable.
#[cfg(all(unix, target_arch = "x86_64"))]
fn compile_libc_src_to_bc(name: &str, src: &str) -> Option<PathBuf> {
    let inc = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../frontend/chibicc/include");
    let c = std::env::temp_dir().join(format!("svm_llvm_cc_{}_{}.c", std::process::id(), name));
    let bc = std::env::temp_dir().join(format!("svm_llvm_cc_{}_{}.bc", std::process::id(), name));
    std::fs::write(&c, src).expect("write C source");
    let status = Command::new("clang")
        .args(["-O2", "-emit-llvm", "-c"])
        .arg("-I")
        .arg(&inc)
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

/// Run a guest-built concurrency scheduler demo (threads / stackful fibers / futex over the `__vm_*`
/// primitives + the guest pthread shim) through the on-ramp on the **real powerbox** and assert its
/// stdout. The printed total is interleaving-invariant — deterministic regardless of which worker ran
/// each unit — so it is the same fixed value the chibicc `c_guest_*` tests check (the LLVM frontend
/// must reach the same answer). A generous deadline guards against a hang (a livelocked scheduler is a
/// failure, not an infinite test). Skips silently if clang is unavailable.
#[cfg(all(unix, target_arch = "x86_64"))]
fn check_guest_concurrency_demo(name: &str, rel: &str, expect: &[u8]) {
    let Some(bc) = compile_demo_libc_to_bc(name, rel) else {
        return;
    };
    let t = svm_llvm::translate_bc_path(&bc).expect("translate bitcode");
    assert!(
        svm_run::is_powerbox_entry(&t.module),
        "{name}: a threads/libc program must produce a powerbox entry"
    );
    let module = svm_run::resolve_capability_imports(t.module).expect("resolve capability imports");
    svm_verify::verify_module(&module).expect("verify translated IR");
    let run =
        svm_run::run_powerbox_with_deadline(&module, b"", Some(std::time::Duration::from_secs(60)))
            .expect("powerbox run");
    assert_eq!(
        run.stdout, expect,
        "{name}: stdout {:?} vs expected {:?}",
        run.stdout, expect
    );
    let code = match run.outcome {
        svm_run::Outcome::Exited(c) => c as u8,
        svm_run::Outcome::Returned(ref v) => match v.first() {
            Some(svm_interp::Value::I32(x)) => *x as u8,
            _ => 0,
        },
    };
    assert_eq!(code, 0, "{name}: exit/return code {code} (expected 0)");
}

/// The chibicc `work_stealing` demo through the LLVM on-ramp: a guest-built **work-stealing M:N
/// scheduler** over *stackless* tasks (a global injector + per-worker deques + stealing, the tokio
/// shape). Four vCPUs (`thread.spawn`) drain 16 tasks of 16 steps each, coordinating only through
/// `pthread_mutex` (the futex) + C11 atomics — no fibers, no scheduler in the VM (D56). The grand
/// total `NTASKS * STEPS = 256` is interleaving-invariant. Mirrors `c_frontend::c_guest_work_stealing`.
#[test]
#[cfg(all(unix, target_arch = "x86_64"))]
fn demo_work_stealing_vs_chibicc() {
    check_guest_concurrency_demo("work_stealing", "work_stealing/work_stealing.c", b"256\n");
}

/// The chibicc `mn_sched` demo through the LLVM on-ramp: a guest-built **sharded M:N green-thread
/// scheduler** — `NWORKERS` OS threads (`thread.spawn`), each running a cooperative round-robin over
/// `TASKS_PER_WORKER` **stackful fibers** (`__vm_fiber_new`/`resume`/`suspend` → `cont.*`), pinned
/// per worker (fibers are thread-affine, D57). Coordinates through one shared atomic counter. The
/// total `NWORKERS * TASKS_PER_WORKER * STEPS = 1024` is interleaving-invariant. Mirrors
/// `c_frontend::c_guest_mn_scheduler`.
#[test]
#[cfg(all(unix, target_arch = "x86_64"))]
fn demo_mn_sched_vs_chibicc() {
    check_guest_concurrency_demo("mn_sched", "mn_sched/mn_sched.c", b"1024\n");
}

/// The chibicc `steal_fibers` demo through the LLVM on-ramp: a work-stealing scheduler over
/// **stackful, migratable fibers** (D57) — suspended fibers are stolen across real OS threads and
/// resumed inside nested call frames. Prints both interleaving-invariant totals: `256` work units and
/// `121920`, the sum of returns whose values depend on locals carried across **every migration** (the
/// stack-integrity check that a stackless state machine cannot express). Mirrors
/// `c_frontend::c_guest_steal_fibers`.
#[test]
#[cfg(all(unix, target_arch = "x86_64"))]
fn demo_steal_fibers_vs_chibicc() {
    check_guest_concurrency_demo(
        "steal_fibers",
        "steal_fibers/steal_fibers.c",
        b"256\n121920\n",
    );
}

/// The chibicc `malloc_threads` demo through the LLVM on-ramp: concurrent `malloc` from `NWORKERS`
/// vCPUs, exercising the **thread-safe** guest allocator. Each worker `malloc`s 64 disjoint blocks
/// and fills every byte with a `(worker, block, offset)`-unique pattern; after join, main re-checks
/// every byte — a clobber from an overlapping allocation (the race a non-thread-safe bump allocator
/// would allow) would show as a corrupt block. Prints `0` (no corruption). Mirrors
/// `c_frontend::c_guest_malloc_threads`.
#[test]
#[cfg(all(unix, target_arch = "x86_64"))]
fn demo_malloc_threads_vs_chibicc() {
    check_guest_concurrency_demo("malloc_threads", "malloc_threads/malloc_threads.c", b"0\n");
}

#[test]
fn demo_hexdump_vs_native() {
    // A `hexdump -C`-style tool: read stdin in 16-byte rows, print `%08lx  HH×16  |ascii|` via the
    // guest-side `printf` format engine (parsed at translate time → `__svm_utoa` + width/zero-pad →
    // `Stream.write`). Exercises `%08lx`/`%02x` (hex + width + zero-pad + `l` modifier) and literals;
    // clang lowers the `%c` and `"…\n"` to `putchar`/`puts`. Byte-identical to native.
    let input = b"Hello, hexdump!\nThe quick brown fox.\x00\x01\xff\xfe rest";
    check_demo_vs_native("hexdump", "hexdump/hexdump.c", input);
}

#[test]
fn demo_crc32_vs_native() {
    // CRC-32 over stdin (shift/xor) + a big-endian u32 reader (`__builtin_bswap32` on host-endian
    // loads) — drives `llvm.bswap` (an inline byte reversal). Prints the CRC and the be32 sum.
    // Byte-identical to native.
    let input = b"The quick brown fox jumps over the lazy dog.\nbig-endian!\x00\x01\x02\x03";
    check_demo_vs_native("crc32", "crc32/crc32.c", input);
}

#[test]
fn demo_raytrace_vs_native() {
    // A tiny ASCII sphere raytracer: diffuse lighting + sinusoidal bands + an exponential rim, with
    // a `libm` bundled as *guest code* (`g_sin`/`g_exp` poly approximations). `sqrt`/`floor` lower to
    // SVM float ops; the transcendentals run in the guest, so native `cc` compiles the same source
    // and every value is bit-identical. Byte-identical to native.
    check_demo_vs_native("raytrace", "raytrace/raytrace.c", b"");
}

#[test]
fn guest_libm_transcendental() {
    // A guest `exp` (range-reduced Taylor) + the IEEE `sqrt` op: compute an RMS over a damped wave.
    // The transcendental is guest code (native compiles the same), `sqrt` is the shared IEEE op, so
    // the int-quantized result matches native exactly.
    let src = "double sqrt(double); double floor(double); \
               static double g_exp(double x){ const double LN2=0.69314718055994530942; \
                 double kf=floor(x/LN2+0.5); int k=(int)kf; double r=x-kf*LN2; \
                 double er=1.0+r*(1.0+r*(0.5+r*(1.0/6+r*(1.0/24+r/120)))); double p=1.0; \
                 if(k>=0) for(int i=0;i<k;i++) p*=2.0; else for(int i=0;i<-k;i++) p*=0.5; \
                 return er*p; } \
               int run(int n){ double acc=0; \
                 for(int i=0;i<n;i++){ double x=(double)i*0.1; double w=g_exp(-x*0.3); acc+=w*w; } \
                 double rms=sqrt(acc/(double)n); return (int)(rms*1000.0); } \
               int main(void){ return run(40); }";
    check_vs_native("guest_libm", src, 40);
}

#[test]
fn demo_lineedit_vs_native() {
    // A tiny line editor: read a line, wrap it in `[...]` (a right shift, `dst > src` → backward
    // copy), then delete the middle char (a left shift, `dst < src` → forward copy). The runtime
    // length keeps clang from folding the `memmove`s inline, so both route to the synthesized
    // direction-aware `__svm_memmove`. Byte-identical to native. Empty + non-empty inputs.
    check_demo_vs_native("lineedit", "lineedit/lineedit.c", b"hello world\n");
    check_demo_vs_native("lineedit_short", "lineedit/lineedit.c", b"ab\n");
}

#[test]
fn memmove_overlap_runtime() {
    // Variable-length `memmove` with overlap in both directions, driven through the guest helper.
    // A right shift by 1 (dst > src, backward) then a left shift by 1 (dst < src, forward) over an
    // 8-byte window; the surviving bytes must match native (which uses the libc `memmove`).
    let src = "void *memmove(void *, const void *, unsigned long); \
               int run(int n){ char b[16]; for (int i=0;i<8;i++) b[i]='a'+i; \
               unsigned long m=(unsigned long)n; \
               memmove(b+1, b, m);            /* shift right: dst>src */ \
               memmove(b, b+1, m);            /* shift left:  dst<src */ \
               int s=0; for (int i=0;i<8;i++) s+=b[i]; return s & 0x7f; } \
               int main(void){ return run(7); }";
    check_vs_native("memmove_overlap", src, 7);
}

#[test]
fn bswap_intrinsic() {
    // `__builtin_bswap32`/`bswap64` → inline byte reversal, checked vs native.
    let src = "int run(int n){ unsigned x = 0x11223344u + (unsigned)n; \
               unsigned long y = 0xaabbccdd00112233UL; \
               unsigned s = __builtin_bswap32(x); unsigned long t = __builtin_bswap64(y); \
               return (int)((s ^ (unsigned)t ^ (unsigned)(t >> 32)) & 0xff); } \
               int main(void){ return run(5); }";
    check_vs_native("bswap", src, 5);
}

#[test]
fn demo_mat4_vs_native() {
    // A 4×4 matrix × vec4 affine transform using `<4 x float>` (vector_size(16)) — `matvec`
    // broadcasts each component and accumulates the columns (`llvm.fmuladd.v4f32`), printing the
    // int-truncated results. Drives 128-bit SIMD: `v128.load`/`store`, `f32x4` mul/add, extract/
    // replace lane, and the splat `shufflevector`. Byte-identical to native.
    check_demo_vs_native("mat4", "mat4/mat4.c", b"");
}

#[test]
fn vec4_float_scale() {
    // A `<4 x float>` passed/returned by value (a `v128` call/ret) and scaled by a broadcast scalar
    // (`splat` + `f32x4.mul`), then lane-summed. scale({1,2,3,4}, 3) = {3,6,9,12} → 30.
    let src =
        "typedef float float4 __attribute__((vector_size(16))); float4 scale(float4, float); \
               int run(int n){ float4 v = {1.0f, 2.0f, 3.0f, 4.0f}; float4 r = scale(v, (float)n); \
               return (int)(r[0] + r[1] + r[2] + r[3]); } \
               __attribute__((noinline)) float4 scale(float4 v, float s){ return v * s; } \
               int main(void){ return run(3); }";
    check_vs_native("vec4_scale", src, 3);
}

#[test]
fn demo_sortvec_vs_native() {
    // A growable int vector + insertion sort: 50 pseudo-random signed ints into a `realloc`-doubling
    // buffer (from `realloc(NULL,…)` ≡ malloc), sorted, printed 10/line via `printf("%d%c")`. Drives
    // `realloc` (the header-bearing bump allocator: malloc + copy old contents) and signed `%d`.
    check_demo_vs_native("sortvec", "sortvec/sortvec.c", b"");
}

#[test]
fn printf_signed_formats() {
    // Signed `%d` (incl. negatives) with plain and space-padded fields, mixed with `%u` — checked vs
    // native. (Zero-padded `%d` is intentionally fail-closed, so it is not exercised here.)
    let src = "#include <stdio.h>\n\
               int main(void){ \
                 printf(\"%d %d %d %6d\\n\", 0, -7, 12345, -42); \
                 printf(\"u=%u neg=%d\\n\", 4000000000u, -1); \
                 return 0; }";
    check_powerbox_vs_native("printf_d", src, b"");
}

#[test]
fn tail_call_mutual_recursion() {
    // `musttail` tail calls lower to the `return_call` terminator, so an unbounded tail/mutual
    // recursion runs in **constant native-stack space**. `even`/`odd` tail-call each other 2,000,000
    // deep — without tail-call lowering the JIT would recurse that far and blow the native stack;
    // with `return_call` the frame is replaced, not nested. `noinline` stops clang from collapsing
    // the pair, `musttail` forces a real tail call (not a rewritten loop). Both functions have no
    // allocas (`frame_size == 0`), so the linear-memory data stack is constant too. vs native (which
    // also TCOs `musttail`), byte-identical.
    let src = "#include <stdio.h>\n\
               __attribute__((noinline)) static int odd(int n);\n\
               __attribute__((noinline)) static int even(int n) {\n\
                   if (n == 0) return 1;\n\
                   __attribute__((musttail)) return odd(n - 1);\n\
               }\n\
               __attribute__((noinline)) static int odd(int n) {\n\
                   if (n == 0) return 0;\n\
                   __attribute__((musttail)) return even(n - 1);\n\
               }\n\
               int main(void) {\n\
                   printf(\"%d %d\\n\", even(2000000), odd(1999999));\n\
                   return 0;\n\
               }";
    check_powerbox_vs_native("tail_mutual", src, b"");
}

// Narrow (i8/i16) atomics — emulated via the 32-bit CAS-loop helpers. rmw add (with wrap-around),
// or/and/xor, exchange, load, store, and cmpxchg (success + failure) on a byte and a halfword.
const ATOMICS_NARROW_SRC: &str = "#include <stdio.h>\n#include <stdatomic.h>\n\
    int main(void){\n\
        atomic_uchar c = 0;\n\
        atomic_fetch_add(&c, 200); atomic_fetch_add(&c, 100);\n\
        atomic_fetch_or(&c, 0x80); atomic_fetch_and(&c, 0xF0); atomic_fetch_xor(&c, 0x0F);\n\
        unsigned char oc = atomic_exchange(&c, 50);\n\
        unsigned char lc = atomic_load(&c);\n\
        atomic_store(&c, 99);\n\
        unsigned char e1 = 99; int ok1 = atomic_compare_exchange_strong(&c, &e1, 123);\n\
        unsigned char e2 = 7;  int ok2 = atomic_compare_exchange_strong(&c, &e2, 0);\n\
        atomic_ushort s = 1000; atomic_fetch_add(&s, 70000);\n\
        printf(\"%d %d %d %d %d %d\\n\", oc, lc, ok1, ok2, atomic_load(&c), atomic_load(&s));\n\
        return 0;\n\
    }";

#[test]
fn atomics_narrow() {
    check_powerbox_vs_native("atomics_narrow", ATOMICS_NARROW_SRC, b"");
}

#[test]
fn atomics_narrow_lowers_and_runs() {
    // Local validation (no native cc). c: 0→200→44(wrap)→172→160→175; oc=175, store 99, cas(99→123)
    // ok, cas(7) fail, c=123. s: 1000+70000 ≡ 5464 (mod 2^16).
    let Some(bc) = compile_to_bc("atomics_n", ATOMICS_NARROW_SRC) else {
        return;
    };
    let t = svm_llvm::translate_bc_path(&bc).expect("translate bitcode");
    // The narrow CAS-loop helpers were synthesized (each contains an AtomicCmpxchg).
    let cas_loops = t
        .module
        .funcs
        .iter()
        .filter(|f| {
            f.blocks.iter().any(|b| {
                b.insts
                    .iter()
                    .any(|i| matches!(i, svm_ir::Inst::AtomicCmpxchg { .. }))
            })
        })
        .count();
    assert!(
        cas_loops >= 2,
        "expected the rmw + cas narrow helpers, got {cas_loops}"
    );
    let module = svm_run::resolve_capability_imports(t.module).expect("resolve capability imports");
    svm_verify::verify_module(&module).expect("verify translated IR");
    let run = svm_run::run_powerbox(&module, b"").expect("powerbox run");
    assert_eq!(run.stdout, b"175 50 1 0 123 5464\n");
}

// C11 `<stdatomic.h>` exercising every native atomic instruction the on-ramp now lowers: rmw
// add/sub/and/or/xor and exchange, load, store, and compare-exchange (success + failure) on `i32`
// and `i64`. Single-threaded, so seq-cst is trivially observable.
const ATOMICS_WIDE_SRC: &str = "#include <stdio.h>\n#include <stdatomic.h>\n\
    int main(void){\n\
        atomic_int a = 0;\n\
        atomic_fetch_add(&a, 5); atomic_fetch_sub(&a, 2);\n\
        atomic_fetch_or(&a, 8); atomic_fetch_and(&a, 0xFF); atomic_fetch_xor(&a, 1);\n\
        int old = atomic_exchange(&a, 100);\n\
        int loaded = atomic_load(&a);\n\
        atomic_store(&a, 42);\n\
        int e1 = 42; int ok1 = atomic_compare_exchange_strong(&a, &e1, 99);\n\
        int e2 = 7;  int ok2 = atomic_compare_exchange_strong(&a, &e2, 0);\n\
        atomic_long b = 1000000000000L; atomic_fetch_add(&b, 1);\n\
        printf(\"%d %d %d %d %d %ld\\n\", old, loaded, ok1, ok2, atomic_load(&a), atomic_load(&b));\n\
        return 0;\n\
    }";

#[test]
fn atomics_wide() {
    check_powerbox_vs_native("atomics_wide", ATOMICS_WIDE_SRC, b"");
}

#[test]
fn atomics_wide_lowers_and_runs() {
    // Local validation (no native cc): the native atomics lower, verify, and run to the
    // hand-computed result. a: 0→5→3→11→11→10; old=10, store 42, cas(42→99) ok, cas(7) fail, a=99;
    // b: 1e12 + 1.
    let Some(bc) = compile_to_bc("atomics_w", ATOMICS_WIDE_SRC) else {
        return;
    };
    let t = svm_llvm::translate_bc_path(&bc).expect("translate bitcode");
    let has_atomic = t.module.funcs.iter().flat_map(|f| &f.blocks).any(|b| {
        b.insts.iter().any(|i| {
            matches!(
                i,
                svm_ir::Inst::AtomicRmw { .. }
                    | svm_ir::Inst::AtomicCmpxchg { .. }
                    | svm_ir::Inst::AtomicLoad { .. }
                    | svm_ir::Inst::AtomicStore { .. }
            )
        })
    });
    assert!(
        has_atomic,
        "expected native atomic ops in the lowered module"
    );
    let module = svm_run::resolve_capability_imports(t.module).expect("resolve capability imports");
    svm_verify::verify_module(&module).expect("verify translated IR");
    let run = svm_run::run_powerbox(&module, b"").expect("powerbox run");
    assert_eq!(run.stdout, b"10 100 1 0 99 1000000000001\n");
}

#[test]
fn tail_call_lowers_and_runs() {
    // Local validation (no native `cc` needed): the `musttail` mutual recursion translates with
    // `return_call` terminators, verifies, and runs to completion at 2,000,000 depth — which only
    // works because the frame is replaced, not nested (a plain-`call` lowering would recurse 2M deep
    // and overflow the native stack). `even(2e6)=1`, `odd(1999999)=even(1999998)=1` ⇒ "1 1\n".
    let src = "#include <stdio.h>\n\
               __attribute__((noinline)) static int odd(int n);\n\
               __attribute__((noinline)) static int even(int n) {\n\
                   if (n == 0) return 1;\n\
                   __attribute__((musttail)) return odd(n - 1);\n\
               }\n\
               __attribute__((noinline)) static int odd(int n) {\n\
                   if (n == 0) return 0;\n\
                   __attribute__((musttail)) return even(n - 1);\n\
               }\n\
               int main(void) {\n\
                   printf(\"%d %d\\n\", even(2000000), odd(1999999));\n\
                   return 0;\n\
               }";
    let Some(bc) = compile_to_bc("tail_lower", src) else {
        return; // clang unavailable
    };
    let t = svm_llvm::translate_bc_path(&bc).expect("translate bitcode");
    let n_tail = t
        .module
        .funcs
        .iter()
        .flat_map(|f| &f.blocks)
        .filter(|b| {
            matches!(
                b.term,
                svm_ir::Terminator::ReturnCall { .. }
                    | svm_ir::Terminator::ReturnCallIndirect { .. }
            )
        })
        .count();
    assert!(
        n_tail >= 2,
        "expected ≥2 return_call terminators (even/odd tail calls), got {n_tail}"
    );
    let module = svm_run::resolve_capability_imports(t.module).expect("resolve capability imports");
    svm_verify::verify_module(&module).expect("verify translated IR"); // checks return_call shape
    let run = svm_run::run_powerbox(&module, b"").expect("powerbox run"); // 2M-deep, constant stack
    assert_eq!(run.stdout, b"1 1\n", "mutual tail recursion result");
}

#[test]
fn printf_float_scientific() {
    // `%e`/`%E` via the bignum Dragon4 formatter (`__svm_dtoa_sci`) — exact, correctly-rounded across
    // the whole double range (no magnitude cap). Covers default/explicit precision, the exponent sign
    // and ≥2-digit padding, very large/small magnitudes (1e300, 1e-300 — impossible for the 128-bit
    // `%f` path), uppercase `%E`, sign flags, width + justification, a carry-on-round, and inf/nan.
    // Byte-for-byte stdout vs native.
    let src = "#include <stdio.h>\n#include <math.h>\n\
               int main(void){ \
                 printf(\"%e\\n\", 3.14); \
                 printf(\"%.2e %.0e\\n\", 12345.678, 9.6); \
                 printf(\"%e %e\\n\", 1e300, 1e-300); \
                 printf(\"%E\\n\", 6.022e23); \
                 printf(\"%.3e\\n\", 9.9999); \
                 printf(\"%+.1e % .1e\\n\", 1.5, 2.5); \
                 printf(\"[%14.3e][%-14.3e]\\n\", 2.5, 2.5); \
                 printf(\"%e %e\\n\", 0.0, -0.0); \
                 volatile double inf = INFINITY, nan = NAN; \
                 printf(\"%e %e %E\\n\", inf, -inf, nan); \
                 return 0; }";
    check_powerbox_vs_native("printf_e", src, b"");
}

#[test]
fn printf_float_general() {
    // `%g`/`%G` via the bignum formatter (`__svm_dtoa_gen`): rounds to P significant digits, picks
    // `%e` vs `%f` by exponent (`E < -4 || E >= P`), and strips trailing zeros. Covers both layout
    // branches, the e/f boundary with a carry (999999.9 → 1e+06), trailing-zero stripping (100000,
    // 42.0), tiny/huge magnitudes, `%G`, precision 0 (⇒ 1 sig digit), sign + width, and inf/nan.
    // Byte-for-byte vs native.
    let src = "#include <stdio.h>\n#include <math.h>\n\
               int main(void){ \
                 printf(\"%g %g\\n\", 3.14159, 100000.0); \
                 printf(\"%g %g\\n\", 1000000.0, 0.0001); \
                 printf(\"%g %g\\n\", 0.00001, 1.0/3.0); \
                 printf(\"%.10g\\n\", 1.0/3.0); \
                 printf(\"%g %g\\n\", 999999.9, 42.0); \
                 printf(\"%g %g\\n\", 1e300, 1e-300); \
                 printf(\"%G %.0g\\n\", 6.022e23, 1234.0); \
                 printf(\"[%12g][%-12g]\\n\", -2.5, -2.5); \
                 printf(\"%g %g\\n\", 0.0, -0.0); \
                 volatile double inf = INFINITY, nan = NAN; \
                 printf(\"%g %g %G\\n\", inf, -inf, nan); \
                 return 0; }";
    check_powerbox_vs_native("printf_g", src, b"");
}

#[test]
fn printf_float_fixed_bignum() {
    // `%f` now goes through the exact bignum formatter (`__svm_dtoa_fix_big`), so large magnitudes
    // that overflowed the old 128-bit path — and used to trap — format correctly, as does `%F` and
    // higher precision. The integer parts here have 30–300+ digits. Byte-for-byte vs native.
    let src = "#include <stdio.h>\n\
               int main(void){ \
                 printf(\"%.2f\\n\", 1e30); \
                 printf(\"%.0f\\n\", 1e60); \
                 printf(\"%.1f\\n\", 1.23456789e40); \
                 printf(\"%f\\n\", 1e300); \
                 printf(\"%.20f\\n\", 0.1); \
                 printf(\"%F\\n\", 12345.678); \
                 printf(\"%.40f\\n\", 1.0/3.0); \
                 printf(\"[%50.2f]\\n\", 1e20); \
                 return 0; }";
    check_powerbox_vs_native("printf_f_big", src, b"");
}

#[test]
fn printf_float_fixed() {
    // `%f` via the synthesized exact-decimal helper (`__svm_dtoa_fixed`, fixed 128-bit integer
    // arithmetic — no host float formatting). Covers: default precision (6), explicit precision,
    // round-half-to-even ties (2.5→2, 3.5→4), a non-exactly-representable decimal (0.1), field width
    // with right/left justification, a negative value, the `+`/space sign flags, zero, and a value
    // with a multi-digit integer part. Byte-for-byte stdout compared to the native clang build.
    let src = "#include <stdio.h>\n\
               int main(void){ \
                 printf(\"%f\\n\", 3.14); \
                 printf(\"%.2f\\n\", 3.14159); \
                 printf(\"%.0f %.0f\\n\", 2.5, 3.5); \
                 printf(\"%.3f\\n\", 0.1); \
                 printf(\"[%8.2f][%-8.2f]\\n\", 3.5, 3.5); \
                 printf(\"%f\\n\", -2.75); \
                 printf(\"%+.1f % .1f\\n\", 1.5, 1.5); \
                 printf(\"%.2f %.2f\\n\", 0.0, 100.0); \
                 printf(\"%.3f\\n\", 12345.6789); \
                 printf(\"%.10f\\n\", 0.5); \
                 printf(\"%.1f\\n\", 9007199254740992.0); \
                 printf(\"%.0f %.0f %.0f\\n\", 0.5, 1.5, 4.5); \
                 printf(\"%.4f\\n\", 2.0 / 3.0); \
                 printf(\"%.2f\\n\", -0.0); \
                 printf(\"%.2f\\n\", 1e30); \
                 printf(\"%.1f\\n\", 18014398509481984.0); \
                 return 0; }";
    check_powerbox_vs_native("printf_f", src, b"");
}

#[test]
fn printf_float_nonfinite() {
    // `%f` of the non-finite doubles: `__svm_dtoa_fixed` writes "inf"/"nan" (lowercase, with the sign
    // byte / `+`/space flags) just like glibc, rather than trapping. `volatile` keeps clang from
    // constant-folding the printf away. stdout compared byte-for-byte to native.
    let src = "#include <stdio.h>\n#include <math.h>\n\
               int main(void){ \
                 volatile double inf = INFINITY, nan = NAN, big = 1e308; \
                 printf(\"%f %f\\n\", inf, -inf); \
                 printf(\"%+f % f\\n\", inf, inf); \
                 printf(\"%f\\n\", nan); \
                 printf(\"[%8.2f][%-8.2f]\\n\", inf, inf); \
                 printf(\"%f\\n\", big * 10.0); \
                 return 0; }";
    check_powerbox_vs_native("printf_f_nonfinite", src, b"");
}

#[test]
fn realloc_grow_preserves() {
    // `realloc` must preserve the old contents across a grow (the header gives the copy length). Push
    // 20 ints into a doubling buffer, then sum — exit code compared to native.
    let src = "#include <stdlib.h>\n\
               int run(int seed){ int *a = (int*)malloc(2 * sizeof(int)); int cap = 2, n = 0; \
               for (int i = 0; i < 20; i++) { if (n == cap) { cap *= 2; a = (int*)realloc(a, (unsigned long)cap * sizeof(int)); } a[n++] = i * seed; } \
               long s = 0; for (int i = 0; i < n; i++) s += a[i]; return (int)(s & 0xff); } \
               int main(void){ return run(3); }";
    check_powerbox_vs_native("realloc_grow", src, b"");
}

#[test]
fn printf_unsigned_formats() {
    // The `printf` engine directly: unsigned decimal/hex with field width + zero/space padding, a
    // 64-bit (`%lx`) arg, `%c`, and `%%` — all in one mixed format (so clang keeps it as `printf`,
    // not a `putchar`/`puts` special-case). stdout + exit compared to the native build.
    let src = "#include <stdio.h>\n\
               int main(void){ \
                 printf(\"u=%u x=%x p=%05x w=%8x c=%c pct=%%\\n\", 42u, 255u, 7u, 0xabcu, 'Z'); \
                 printf(\"%lx %02x\\n\", 0xdeadbeefcafeUL, 5u); \
                 return 0; }";
    check_powerbox_vs_native("printf_u", src, b"");
}

#[test]
fn main_argc_argv() {
    // `int main(int argc, char** argv)`: the synthesized argv-parsing `_start` reads the §3e args
    // buffer, builds `argv[]` (with the `argv[argc] == NULL` terminator), and passes `argc`/`argv`
    // to `main` — checked byte-for-byte vs native with a controlled `argv`. Covers iterating argv,
    // a NULL-terminator walk (`while (argv[i])`), and `argc` flowing to the exit code.
    let src = "#include <stdio.h>\n\
               int main(int argc, char** argv){ \
                 printf(\"argc=%d\\n\", argc); \
                 for (int i = 0; argv[i]; i++) printf(\"argv[%d]=%s\\n\", i, argv[i]); \
                 return argc; }";
    // The common case (program name only), then a multi-arg vector.
    check_powerbox_vs_native_args("argv1", src, &["prog"], &[]);
    check_powerbox_vs_native_args("argvN", src, &["myprog", "hello", "world", "42"], &[]);
}

#[test]
fn main_argc_argv_envp() {
    // `int main(int argc, char** argv, char** envp)`: the synthesized `_start` parses the §3e blob's
    // `envc` env strings (packed right after the argv strings) into a second NULL-terminated `char**`
    // parked just above `argv[]`, and passes it as the third parameter — checked byte-for-byte vs
    // native with a controlled, `env_clear`ed environment. Covers walking `envp` to its NULL
    // terminator and `argv`/`envp` coexisting at the entry stack base.
    let src = "#include <stdio.h>\n\
               int main(int argc, char** argv, char** envp){ \
                 printf(\"argc=%d\\n\", argc); \
                 for (int i = 0; argv[i]; i++) printf(\"argv[%d]=%s\\n\", i, argv[i]); \
                 int n = 0; for (char** e = envp; *e; e++) { printf(\"env[%d]=%s\\n\", n++, *e); } \
                 return n; }";
    // Empty env (just the NULL terminator), then a multi-entry environment. The entries are passed in
    // sorted-by-key order because `std::process::Command` stores its env in a `BTreeMap`, so native's
    // child `environ` is key-sorted; the SVM blob preserves the order we hand it, so we match by
    // pre-sorting (`EMPTY` < `FOO` < `PATH`).
    check_powerbox_vs_native_args("envp0", src, &["prog"], &[]);
    check_powerbox_vs_native_args(
        "envpN",
        src,
        &["prog", "a"],
        &["EMPTY=", "FOO=bar", "PATH=/x:/y"],
    );
}

#[test]
fn getenv_lookup() {
    // `getenv` on a `main(void)` program (so it exercises the synthesized `__svm_getenv` decoupled
    // from any argv parsing): it scans the §3e blob's env strings for `KEY=`, returning the value
    // pointer or NULL. Covers a hit, a miss, and the prefix guard (`"F"` must not match `"FOO=..."`),
    // checked byte-for-byte vs native `getenv` with a controlled, `env_clear`ed environment.
    let src = "#include <stdio.h>\n#include <stdlib.h>\n\
               int main(void){ \
                 char* a = getenv(\"FOO\"); char* b = getenv(\"PATH\"); \
                 char* c = getenv(\"MISSING\"); char* d = getenv(\"F\"); \
                 printf(\"FOO=%s\\n\", a ? a : \"(null)\"); \
                 printf(\"PATH=%s\\n\", b ? b : \"(null)\"); \
                 printf(\"MISSING=%s\\n\", c ? c : \"(null)\"); \
                 printf(\"F=%s\\n\", d ? d : \"(null)\"); \
                 return 0; }";
    check_powerbox_vs_native_args("getenv1", src, &["prog"], &["FOO=bar", "PATH=/x:/y"]);
    // No environment at all: every lookup must be NULL (the blob reads envc == 0).
    check_powerbox_vs_native_args("getenv0", src, &["prog"], &[]);
}

#[test]
fn printf_precision_formats() {
    // Integer min-digit precision (`%.Nd`/`%.Nx`, incl. `%.0` of zero → no digits, and precision
    // overriding the `0` flag → space field padding) and string truncating precision (`%.Ns`),
    // checked byte-for-byte vs native `printf`. `s` is a non-literal pointer (runtime strlen).
    let src = "#include <stdio.h>\n\
               int main(void){ \
                 const char* s = \"hello world\"; \
                 printf(\"|%.4d|%.4d|%.0d|%8.4d|%-8.4d|\\n\", 42, -42, 0, 42, 42); \
                 printf(\"|%.4x|%#.4x|%08.4d|\\n\", 0xabu, 0xabu, 42); \
                 printf(\"|%.5s|%.20s|%8.3s|\\n\", s, s, s); \
                 return 0; }";
    check_powerbox_vs_native("printf_prec", src, b"");
}

#[test]
fn printf_flag_formats() {
    // The full flag matrix on integer conversions, checked byte-for-byte vs native `printf`:
    //   `-` left-justify, `+`/space forced sign, `0` zero-pad (incl. the previously-fail-closed
    //   zero-padded signed), and `#` (the `0x` hex prefix, suppressed for zero).
    let src = "#include <stdio.h>\n\
               int main(void){ \
                 printf(\"|%-6d|%-6d|\\n\", 42, -42); \
                 printf(\"|%+d|%+d|% d|% d|\\n\", 42, -42, 42, -42); \
                 printf(\"|%05d|%05d|%+05d|\\n\", 42, -42, 42); \
                 printf(\"|%#x|%#x|%#08x|\\n\", 255u, 0u, 0xabcu); \
                 printf(\"|%-8x|%08x|\\n\", 0xbeefu, 0xbeefu); \
                 return 0; }";
    check_powerbox_vs_native("printf_flags", src, b"");
}

#[test]
fn printf_string_formats() {
    // `%s` (runtime `strlen`) plain and right-justified in a field width, mixed with `%d`/`%c` so
    // clang keeps it a real varargs `printf` (not a `puts` rewrite). A pointer that is *not* a string
    // literal (the `b+1` tail) exercises the runtime strlen, not a constant length. stdout + exit vs
    // native.
    let src = "#include <stdio.h>\n\
               int main(void){ \
                 const char* a = \"hi\"; const char* b = \"world\"; \
                 printf(\"[%s] [%8s] n=%d\\n\", a, b, 3); \
                 printf(\"%s|%s|%c\\n\", b + 1, a, '!'); \
                 return 0; }";
    check_powerbox_vs_native("printf_s", src, b"");
}

#[test]
fn vec2_float_struct() {
    // A `{float,float}` struct passed/returned by value — clang coerces it to `<2 x float>` and does
    // `extractelement`/`insertelement`/lane-wise `fadd`. Scalarized to a packed i64. addv({1.5,2.5},
    // {7,0.5}) = {8.5,3.0} → 8.5*10 + 3 = 88.
    let src = "struct V2 { float x, y; }; struct V2 addv(struct V2 a, struct V2 b); \
               int run(int n){ struct V2 a = {1.5f, 2.5f}; struct V2 b = {(float)n, 0.5f}; \
               struct V2 c = addv(a, b); return (int)(c.x * 10.0f + c.y); } \
               struct V2 addv(struct V2 a, struct V2 b){ struct V2 r = {a.x + b.x, a.y + b.y}; return r; } \
               int main(void){ return run(7); }";
    check_vs_native("vec2f", src, 7);
}

#[test]
fn demo_tinfl_vs_native() {
    // miniz's tinfl DEFLATE/zlib inflate engine: a deeply nested coroutine-macro state machine with
    // Huffman fast/slow lookup tables (`mz_int16`) and a 32 KiB LZ77 dictionary. Inflates an embedded
    // zlib stream and writes it out — byte-identical to native. (Regression for the narrow-signed
    // `icmp` fix: the slow Huffman walk tests a sign-extended `i16` table entry `< 0`.)
    check_demo_vs_native("tinfl", "tinfl/tinfl_demo.c", b"");
}

#[test]
fn narrow_signed_compare() {
    // §3b narrow-int hazard: a *signed* `i16`/`i8` value loaded zero-extended must be sign-extended
    // before a signed `icmp` (else `< 0` is always false). Sum the negative entries of a signed-short
    // table — wrong (0) without the fix. Compared to native `cc`.
    let src = "int run(int n){ static const short t[6] = {-1, -100, 5, -32768, 32767, -7}; \
               int s = 0; for (int k = 0; k < 6; k++) if (t[k] < 0) s += t[k]; \
               return (s ^ n) & 0xff; } \
               int main(void){ return run(0); }";
    check_vs_native("narrow_signed_cmp", src, 0);
}

#[test]
fn multi_value_struct_return() {
    // A small by-value struct returned in two registers — clang coerces it to a `{ i64, i64 }`
    // return (`insertvalue`/`ret`) the caller destructures with `extractvalue`. Exercises the §3a
    // multi-result path: `mk` returns two values, `run` reads both. mk(7,14) → 7+14 = 21.
    // `run` is defined first (the unit at index 0 the harness invokes); `mk` is declared then defined
    // after, so it stays a real out-of-line multi-result call.
    let src = "struct Pair { long a; long b; }; \
               struct Pair mk(long a, long b); \
               int run(int x){ struct Pair p = mk(x, (long)x * 2); return (int)(p.a + p.b); } \
               __attribute__((noinline)) struct Pair mk(long a, long b){ struct Pair p = {a, b}; return p; } \
               int main(void){ return run(7); }";
    check_vs_native("multival", src, 7);
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

// ============================================================================================
// `<svm.h>` low-level builtins (P0+P1+Memory): the SVM capability/concurrency/GC surface the
// LLVM on-ramp lowers to the matching SVM IR ops or `Memory` capability calls. These mirror the
// chibicc oracle (`frontend/chibicc/codegen_ir.c`), so a guest language emitting LLVM bitcode
// reaches fibers, threads, atomics, the futex, conservative GC roots, direct window memory
// management, and capability reflection — the JACL GC + scheduler primitives.
// ============================================================================================

/// Translate `src` and return its verified module + entry-SP, or `None` if clang is unavailable.
fn translate_verified(name: &str, src: &str) -> Option<(svm_ir::Module, u64)> {
    let bc = compile_to_bc(name, src)?;
    let t = svm_llvm::translate_bc_path(&bc).expect("translate bitcode");
    svm_verify::verify_module(&t.module).expect("verify translated IR");
    Some((t.module, t.entry_sp))
}

/// Run the **first defined function** (index 0 — the unit under test, since no powerbox is
/// synthesized for a pure-compute program) on the reference interpreter, the universal oracle for
/// fibers/threads/atomics/GC (the JIT bails `Unsupported` on fibers/`cap.self`, like the chibicc
/// tests). Returns the result values, or `None` if clang is unavailable.
fn run_interp(name: &str, src: &str, args: &[Value]) -> Option<Vec<Value>> {
    let (m, entry_sp) = translate_verified(name, src)?;
    let mut full = vec![Value::I64(entry_sp as i64)];
    full.extend_from_slice(args);
    let mut fuel = 200_000_000u64;
    Some(svm_interp::run(&m, 0, &full, &mut fuel).expect("interp run"))
}

/// Does any function's body contain an instruction matching `pred`? (Structural lowering check.)
fn module_has_inst(m: &svm_ir::Module, pred: impl Fn(&svm_ir::Inst) -> bool) -> bool {
    m.funcs
        .iter()
        .flat_map(|f| f.blocks.iter())
        .flat_map(|b| b.insts.iter())
        .any(pred)
}

// ---- §3e/§4 Memory capability: `__vm_map` / `__vm_page_size` (end-to-end on the JIT) ----------

/// The guest **manages its own window** directly from C through `<svm.h>`: `__vm_page_size()` (Memory
/// op 3) and `__vm_map(off,len,prot)` (op 0) grow the reserved tail, then a store/load round-trips
/// through the freshly committed page. Run through the real powerbox (JIT) — which grants the
/// `Memory` handle for the 4-param `_start` the on-ramp now synthesizes when a Memory builtin is used
/// — proving the handle grant, the `CallImport` resolution, and the lowering all compose. The byte it
/// writes is the success marker (`'Y'`), asserted against the captured stdout (`__vm_*` symbols don't
/// exist in a native build, so this lane has no native oracle — the marker is the contract).
#[test]
fn vm_memory_map_and_page_size() {
    let src = r#"
long __vm_page_size(void);
long __vm_map(long off, long len, int prot);
long write(int fd, const void *buf, long n);
int main(void) {
  long ps = __vm_page_size();
  volatile long vbase = 268435456;                /* 256 MiB — deep in the reserved tail */
  long base = vbase;                              /* volatile defeats clang's constant inttoptr fold */
  if (__vm_map(base, 4096, 3) != 0) { char e='E'; write(1,&e,1); return 1; }
  int *p = (int *)base;
  p[0] = 43981;                                   /* 0xABCD */
  p[1] = p[0] + 1;
  char ok = (ps >= 4096 && (ps & (ps - 1)) == 0 && p[1] == 43982) ? 'Y' : 'N';
  write(1, &ok, 1);
  return 0;
}
"#;
    let Some(bc) = compile_to_bc("vm_mem", src) else {
        return;
    };
    let m = svm_llvm::translate_bc_path(&bc)
        .expect("translate bitcode")
        .module;
    // Structural: the Memory ops became `CallImport`s (resolved to the Memory cap at load). The raw
    // module carries unresolved imports, so it is resolved *before* `verify` (a `CallImport` reaching
    // the verifier is a fail-closed error — resolution is mandatory).
    let import_names: Vec<&str> = m.imports.iter().map(|i| i.name.as_str()).collect();
    assert!(
        import_names.contains(&"vm_map") && import_names.contains(&"vm_page_size"),
        "expected vm_map + vm_page_size imports, got {import_names:?}"
    );
    assert!(
        svm_run::is_powerbox_entry(&m),
        "a Memory-builtin program must produce a powerbox entry (granting the Memory handle)"
    );
    let module = svm_run::resolve_capability_imports(m).expect("resolve capability imports");
    svm_verify::verify_module(&module).expect("verify resolved IR");
    let run = svm_run::run_powerbox(&module, b"").expect("powerbox run");
    assert_eq!(
        run.stdout, b"Y",
        "memory round-trip marker (got {:?})",
        run.stdout
    );
}

// ---- §12 fibers: `__vm_fiber_new` / `resume` / `suspend` → `cont.new` / `resume` / `suspend` ---

/// A fiber generator yields twice then returns, driven from C exactly like the chibicc oracle
/// (`c_fiber_generator_yields_then_returns`). The entry is the first defined function (index 0); the
/// fiber body `counter` is reached as a funcref through `cont.new`. Interpreter-only (fibers are an
/// interp op). 101 + 102 + 103 = 306.
#[test]
#[cfg(unix)]
fn vm_fibers_generator() {
    let src = r#"
long __vm_fiber_new(long (*f)(long), void *stack);
long __vm_fiber_resume(long k, long arg, int *done);
long __vm_fiber_suspend(long value);
long counter(long start);
static char stack0[8192];
int driver(void) {
  long k = __vm_fiber_new(counter, stack0); // i64 fiber handle (16-bit slot + 48-bit generation)
  int done = 0;
  long sum = 0;
  long v = __vm_fiber_resume(k, 100, &done);
  while (!done) { sum += v; v = __vm_fiber_resume(k, 0, &done); }
  sum += v;
  return (int)sum;
}
long counter(long start) {
  __vm_fiber_suspend(start + 1);
  __vm_fiber_suspend(start + 2);
  return start + 3;
}
"#;
    if let Some(r) = run_interp("vm_fibers", src, &[]) {
        assert_eq!(r, vec![Value::I32(306)]);
    }
}

// ---- §12 atomics: `__vm_atomic_*` → the `iN.atomic.*` ops (single-threaded value semantics) ----

/// The atomic ops have plain value semantics when single-threaded, so they are checkable on **both**
/// backends (the JIT lowers them to hardware atomics). Exercises 64- and 32-bit `add`/`load`/`store`
/// and the 32-bit `cmpxchg`: x starts 5, +3 → 8, store 100, +1 → 101; a CAS 101→200 succeeds; final
/// load = 200. Returns 200.
#[test]
fn vm_atomics_single_threaded() {
    let src = r#"
long __vm_atomic_add(void *p, long v);
long __vm_atomic_load(void *p);
void __vm_atomic_store(void *p, long v);
int  __vm_atomic_add32(void *p, int v);
int  __vm_atomic_cas32(void *p, int expected, int desired);
int  __vm_atomic_load32(void *p);
int f(void) {
  long x = 5;
  __vm_atomic_add(&x, 3);                 /* 8 */
  __vm_atomic_store(&x, 100);             /* 100 */
  if (__vm_atomic_load(&x) != 100) return -1;
  int y = (int)x;
  __vm_atomic_add32(&y, 1);               /* 101 */
  int old = __vm_atomic_cas32(&y, 101, 200);  /* old=101, y=200 */
  if (old != 101) return -2;
  return __vm_atomic_load32(&y);          /* 200 */
}
"#;
    check("vm_atomics", src, &[], &[Value::I32(200)]);
}

// ---- §12 futex: `__vm_wait32` / `__vm_notify` → `i32.atomic.wait` / `atomic.notify` ------------

/// The futex primitives, exercised single-threaded so they return deterministically without
/// blocking: `__vm_wait32(p, expected, …)` returns `1` ("not-equal") when `*p != expected`, and
/// `__vm_notify(p, n)` returns `0` when no vCPU is parked. Interpreter-only (the JIT futex is
/// platform/scheduler-gated). Returns 1*10 + 0 = 10.
#[test]
#[cfg(unix)]
fn vm_futex_wait_notify() {
    let src = r#"
int __vm_wait32(void *p, int expected, long timeout_ns);
int __vm_notify(void *p, int count);
int f(void) {
  int word = 7;
  int waited = __vm_wait32(&word, 9, 0);  /* 7 != 9 → returns 1, no block */
  int woke = __vm_notify(&word, 1);       /* no waiters → 0 */
  return waited * 10 + woke;
}
"#;
    if let Some(r) = run_interp("vm_futex", src, &[]) {
        assert_eq!(r, vec![Value::I32(10)]);
    }
}

// ---- §12 threads + atomics: `__vm_thread_spawn` / `join` → `thread.spawn` / `thread.join` -------

/// Four worker vCPUs (`thread.spawn`, with a static funcidx) each atomically bump a shared global
/// 500 times; the entry joins them and reads the total — `thread.join` and the cross-thread atomic
/// `fetch-add` end to end on the interpreter's M:N executor (the chibicc `c_threads_atomic_counter`
/// headline, via the LLVM on-ramp). 4 × 500 = 2000.
#[test]
#[cfg(unix)]
fn vm_threads_atomic_counter() {
    let src = r#"
long __vm_atomic_add(void *p, long v);
long __vm_atomic_load(void *p);
int  __vm_thread_spawn(long (*fn)(long), void *stack, long arg);
long __vm_thread_join(int h);
long counter = 0;
long worker(long arg);
int driver(void) {
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
long worker(long arg) {
  for (long i = 0; i < arg; i++) __vm_atomic_add(&counter, 1);
  return 0;
}
"#;
    if let Some(r) = run_interp("vm_threads", src, &[]) {
        assert_eq!(r, vec![Value::I32(2000)]);
    }
}

// ---- §GC conservative roots: `__vm_gc_roots` → `gc.roots` --------------------------------------

/// Conservative root enumeration scans the calling computation's live frames for candidate window
/// pointers in `[lo, hi)`, writing them into a buffer and returning the total found. The precise set
/// is a backend-specific over-approximation (GC.md §3.2), so this asserts the *contract*: the op
/// lowers to `Inst::GcRoots`, runs without trapping, and returns a count within `[0, cap]` (here it
/// scans the whole window, so any live in-range root the frame holds is counted). Interpreter-only.
#[test]
#[cfg(unix)]
fn vm_gc_roots_smoke() {
    let src = r#"
long __vm_gc_roots(long heap_lo, long heap_hi, long mask, void *buf, long cap);
static long out[64];
int f(void) {
  /* a live, in-range candidate pointer (into `out` itself) the conservative scan may see */
  volatile long *root = out;
  /* mask = ~0 is the untagged (identity) case: every scanned word is tested as-is. */
  long n = __vm_gc_roots(0, 1 << 20, ~0L, out, 64);
  (void)root;
  /* `n` is the total candidates found (may exceed cap=64; a tiny frame stays well under 1024) —
     the assertion is that the op ran, returned a sane non-negative count, and didn't trap. */
  return (n >= 0 && n <= 1024) ? 1 : 0;
}
"#;
    let Some((m, _)) = translate_verified("vm_gc", src) else {
        return;
    };
    assert!(
        module_has_inst(&m, |i| matches!(i, svm_ir::Inst::GcRoots { .. })),
        "expected a gc.roots instruction"
    );
    if let Some(r) = run_interp("vm_gc", src, &[]) {
        assert_eq!(r, vec![Value::I32(1)], "gc.roots count out of [0, cap]");
    }
}

// ---- §12 per-vCPU TLS register: `__vm_vcpu_tls_get` / `__vm_vcpu_tls_set` -----------------------

/// The per-vCPU TLS register lowers from the `<svm.h>` builtins and round-trips a written value:
/// `__vm_vcpu_tls_set(99)` then `__vm_vcpu_tls_get()` returns 99 (the root vCPU is seeded to 0, then
/// overwritten). Asserts the ops lowered and ran on the interpreter.
#[test]
#[cfg(unix)]
fn vm_vcpu_tls_round_trip() {
    let src = r#"
long __vm_vcpu_tls_get(void);
void __vm_vcpu_tls_set(long x);
int f(void) {
  __vm_vcpu_tls_set(99);
  return (int)__vm_vcpu_tls_get();   /* the value we just set */
}
"#;
    let Some((m, _)) = translate_verified("vm_vcpu_tls", src) else {
        return;
    };
    assert!(
        module_has_inst(&m, |i| matches!(i, svm_ir::Inst::VcpuTlsGet))
            && module_has_inst(&m, |i| matches!(i, svm_ir::Inst::VcpuTlsSet { .. })),
        "expected vcpu.tls.get + vcpu.tls.set instructions"
    );
    if let Some(r) = run_interp("vm_vcpu_tls", src, &[]) {
        assert_eq!(r, vec![Value::I32(99)], "vcpu.tls round-trip");
    }
}

// ---- Separate-artifact on-ramp: the export map + the `svm-llvm-translate` CLI ------------------

/// A hand-written caller that resolves `twice` **by name**, used to prove the export map links:
/// `svm_ir::link` rewrites the `call.import` to a direct cross-unit call. Every translated function
/// takes the §3d data-stack pointer `sp` as its first parameter, so `twice` is `(i64 sp, i64 x) ->
/// (i64)`; `twice` has no `alloca`s, so any `sp` works — the caller passes `0`. (`v1` is the unused
/// capability-handle operand `call.import` carries; resolving to a direct call drops it.)
#[cfg(unix)]
const TWICE_CALLER: &str = "\
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const 0
  v2 = i64.const 0
  v3 = call.import \"twice\" (i64, i64) -> (i64) v1 (v2, v0)
  return v3
}
";

/// Link a translated runtime (its module + export map) against [`TWICE_CALLER`] and assert the
/// cross-unit `twice(21) == 42` call resolves on the interpreter — the shared core of both the
/// in-process ([`exports_feed_a_link_unit`]) and CLI ([`translate_cli_emits_module_and_syms`]) paths.
#[cfg(unix)]
fn assert_links_and_runs(runtime_module: svm_ir::Module, exports: Vec<(String, u32)>) {
    use svm_ir::{link, LinkUnit};
    assert!(
        exports.iter().any(|(n, _)| n == "twice"),
        "the runtime must export `twice`; got {exports:?}"
    );
    let runtime = LinkUnit {
        module: runtime_module,
        exports,
        ..Default::default()
    };
    let app = LinkUnit {
        module: svm_text::parse_module(TWICE_CALLER).expect("parse caller"),
        ..Default::default()
    };
    let linked = link(&[runtime, app]).expect("link runtime + caller");
    svm_verify::verify_module(&linked).expect("verify linked program");
    // The caller is the last function (concatenated after the runtime's functions). twice(21) = 42.
    let entry = (linked.funcs.len() - 1) as u32;
    let mut fuel = 10_000_000u64;
    let r = svm_interp::run(&linked, entry, &[Value::I64(21)], &mut fuel).expect("run linked");
    assert_eq!(
        r,
        vec![Value::I64(42)],
        "cross-unit call to translated `twice`"
    );
}

/// **Ask 1 — the export map.** `Translated::exports` pairs each defined function with its final
/// `module.funcs` index (`base + i`), so a separately-compiled program can `call.import` a runtime
/// function and have `svm_ir::link` resolve it. Translate a two-function library and link a caller
/// against it purely by name.
#[test]
#[cfg(unix)]
fn exports_feed_a_link_unit() {
    let src = r#"
long twice(long x) { return x + x; }
long inc(long x) { return x + 1; }
"#;
    let Some(bc) = compile_to_bc("exports", src) else {
        return;
    };
    let t = svm_llvm::translate_bc_path(&bc).expect("translate bitcode");
    // Every export indexes a real function, and both library functions are present.
    for (name, idx) in &t.exports {
        assert!(
            (*idx as usize) < t.module.funcs.len(),
            "export {name} → {idx} is out of range"
        );
    }
    assert!(
        t.exports.iter().any(|(n, _)| n == "inc"),
        "inc must be exported; got {:?}",
        t.exports
    );
    assert_links_and_runs(t.module, t.exports);
}

/// **Ask 2 — the `svm-llvm-translate` CLI.** Translate a library `.bc` to a `.svm` module plus a
/// `.syms` export sidecar, then re-read both, rebuild the export map from the sidecar, and link a
/// caller against the module — the scriptable, separate-artifact analogue of `exports_feed_a_link_unit`.
/// Also smoke-tests the binary (`.svmb`) output path (it must `decode_module`).
#[test]
#[cfg(unix)]
fn translate_cli_emits_module_and_syms() {
    let src = r#"
long twice(long x) { return x + x; }
"#;
    let Some(bc) = compile_to_bc("cli", src) else {
        return;
    };
    let dir = std::env::temp_dir();
    let pid = std::process::id();
    let out = dir.join(format!("svm_llvm_cli_{pid}.svm"));
    let syms = dir.join(format!("svm_llvm_cli_{pid}.syms"));
    let outb = dir.join(format!("svm_llvm_cli_{pid}.svmb"));

    let status = Command::new(env!("CARGO_BIN_EXE_svm-llvm-translate"))
        .arg(&bc)
        .arg("-o")
        .arg(&out)
        .arg("--emit-syms")
        .arg(&syms)
        .status()
        .expect("run svm-llvm-translate");
    assert!(status.success(), "CLI exited non-zero");

    // Binary path: `-o *.svmb` selects `encode_module`; the bytes must round-trip through the decoder.
    let statusb = Command::new(env!("CARGO_BIN_EXE_svm-llvm-translate"))
        .arg(&bc)
        .arg("-o")
        .arg(&outb)
        .status()
        .expect("run svm-llvm-translate (binary)");
    assert!(statusb.success(), "CLI (binary) exited non-zero");
    svm_encode::decode_module(&std::fs::read(&outb).expect("read .svmb"))
        .expect("decode binary module");

    // Re-read the text module + parse the `name idx` sidecar into an export map.
    let module = svm_text::parse_module(&std::fs::read_to_string(&out).expect("read .svm"))
        .expect("parse emitted module");
    let syms_txt = std::fs::read_to_string(&syms).expect("read .syms");
    let exports: Vec<(String, u32)> = syms_txt
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let mut it = l.split_whitespace();
            let name = it.next().expect("sidecar name").to_string();
            let idx = it
                .next()
                .expect("sidecar idx")
                .parse()
                .expect("sidecar idx int");
            (name, idx)
        })
        .collect();

    assert_links_and_runs(module, exports);

    let _ = std::fs::remove_file(&out);
    let _ = std::fs::remove_file(&syms);
    let _ = std::fs::remove_file(&outb);
}

// ---- §7 capability reflection: `__vm_cap_count` / `__vm_cap_at` → `cap.self.count` / `.get` -----

/// Capability **reflection**: a domain discovers what its host granted it. Run on the interpreter
/// under a full 8-handle powerbox host (the grants live in the host-owned table `cap.self.*` reads,
/// independent of the call's params), so `__vm_cap_count()` returns 8 and `__vm_cap_at(0, &t)` yields
/// a valid (non-negative) interface type_id. Returns count*10 + (t >= 0) = 81. Interpreter-only (the
/// JIT bails `Unsupported` on `cap.self`, like fibers).
#[test]
#[cfg(unix)]
fn vm_cap_reflection() {
    use svm_interp::{Host, StreamRole};
    let src = r#"
int __vm_cap_count(void);
int __vm_cap_at(int i, int *type_id_out);
int f(void) {
  int n = __vm_cap_count();
  int t = -1;
  __vm_cap_at(0, &t);
  return n * 10 + (t >= 0 ? 1 : 0);
}
"#;
    let Some((m, entry_sp)) = translate_verified("vm_cap", src) else {
        return;
    };
    // Structural: the reflection ops are present.
    assert!(
        module_has_inst(&m, |i| matches!(i, svm_ir::Inst::CapSelfCount))
            && module_has_inst(&m, |i| matches!(i, svm_ir::Inst::CapSelfGet { .. })),
        "expected cap.self.count + cap.self.get"
    );
    // Run on the interpreter with a granted powerbox: 8 capabilities held → count 8.
    let mut h = Host::new();
    h.set_region_factory(svm_run::new_shared_region);
    h.set_jit_validator(svm_run::jit_blob_validator);
    let win = m.memory.map_or(0, |mc| 1u64 << mc.size_log2);
    let _handles = [
        h.grant_stream(StreamRole::Out),
        h.grant_stream(StreamRole::In),
        h.grant_exit(),
        h.grant_memory(),
        h.grant_address_space(0, win),
        h.grant_io_ring(),
        h.grant_blocking(std::time::Duration::ZERO, None),
        h.grant_jit(None),
    ];
    let mut fuel = 50_000_000u64;
    let r = svm_interp::run_with_host(&m, 0, &[Value::I64(entry_sp as i64)], &mut fuel, &mut h)
        .expect("interp run");
    assert_eq!(
        r,
        vec![Value::I32(81)],
        "cap reflection: 8 caps, type_id >= 0"
    );
}

// ---- §9/§12 async I/O ring: `__vm_io_submit_async` / `__vm_io_reap` / `__vm_blocking_handle` ------

/// Grant a powerbox on an interpreter `Host`, returning the first `n` handles (the prefix `_start`
/// declares) as call args. Mirrors `svm/tests/c_frontend.rs::powerbox`; `block` is the `Blocking`
/// op's dwell so an async batch is genuinely in flight when a vCPU parks.
#[cfg(unix)]
fn grant_powerbox(
    h: &mut svm_interp::Host,
    win: u64,
    n: usize,
    block: std::time::Duration,
) -> Vec<Value> {
    use svm_interp::StreamRole;
    h.set_region_factory(svm_run::new_shared_region);
    h.set_jit_validator(svm_run::jit_blob_validator);
    let mem_log2 = (win != 0).then(|| win.trailing_zeros() as u8);
    let all = [
        h.grant_stream(StreamRole::Out),
        h.grant_stream(StreamRole::In),
        h.grant_exit(),
        h.grant_memory(),
        h.grant_address_space(0, win),
        h.grant_io_ring(),
        h.grant_blocking(block, None),
        h.grant_jit(mem_log2),
    ];
    all[..n].iter().map(|&x| Value::I32(x)).collect()
}

/// The host's deterministic `Blocking.work(i)` result (mirrors `svm_interp::AsyncState::mix`, the
/// value the chibicc async tests check).
fn async_mix(arg: i64) -> i64 {
    arg.wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407)
}

/// The async **event-loop runtime** in real C (`demos/async_io`), driven through the LLVM on-ramp:
/// one vCPU `submit_async`s a batch of `Blocking` ops onto the host offload pool, parks on an in-window
/// completion counter (`__vm_wait32`), and reaps completions as a pool worker `notify`s it — exercising
/// `__vm_io_submit_async`/`__vm_io_reap`/`__vm_blocking_handle` and the **7-handle powerbox**
/// (`synth_start` now grants through `Blocking`) end to end. Interpreter-only (it is the M:N executor
/// and offload-pool oracle; the JIT async path needs the separate `HostAsyncHooks` harness). The
/// printed total is completion-order-invariant: Σ mix(i) for i in 0..8.
#[test]
#[cfg(unix)]
fn vm_async_io_runtime() {
    let src = include_str!("../../svm-run/demos/async_io/async_io.c");
    let Some(bc) = compile_to_bc("async_io", src) else {
        return;
    };
    let m = svm_llvm::translate_bc_path(&bc)
        .expect("translate bitcode")
        .module;
    // The async-ring imports are registered and the entry grants through Blocking (7 handles).
    let import_names: Vec<&str> = m.imports.iter().map(|i| i.name.as_str()).collect();
    assert!(
        import_names.contains(&"vm_io_submit_async") && import_names.contains(&"vm_io_reap"),
        "expected async-ring imports, got {import_names:?}"
    );
    let module = svm_run::resolve_capability_imports(m).expect("resolve capability imports");
    svm_verify::verify_module(&module).expect("verify resolved IR");
    let nparams = module.funcs[0].params.len();
    assert_eq!(
        nparams, 7,
        "async program must declare the 7-handle powerbox entry"
    );

    let win = module.memory.map_or(0, |mc| 1u64 << mc.size_log2);
    let mut h = svm_interp::Host::new();
    let args = grant_powerbox(&mut h, win, nparams, std::time::Duration::from_millis(10));
    let mut fuel = 500_000_000u64;
    let out =
        svm_interp::run_with_host(&module, 0, &args, &mut fuel, &mut h).expect("interp async run");
    assert_eq!(out, vec![Value::I32(0)], "async demo returns 0");
    // NTASKS = 8 (see the demo); the total is Σ of the host's deterministic per-op results.
    let total: u64 = (0..8).fold(0u64, |a, i| a.wrapping_add(async_mix(i) as u64));
    assert_eq!(
        h.stdout,
        format!("{total}\n").into_bytes(),
        "async total Σ mix(0..8)"
    );
}

/// The chibicc `async_work_stealing` demo through the LLVM on-ramp: the async **work-stealing M:N
/// runtime** (the union of the stackless work-stealing scheduler and the async submit/complete ring).
/// `NWORKERS` vCPUs cooperatively drain `NTASKS = 16` I/O-bound tasks, each issuing a `Blocking` op
/// through the ring; a worker **never blocks on an I/O** — it `submit_async`s and moves on, parking on
/// the completion counter only when nothing is runnable, woken by a pool worker's `notify`. Combines
/// the guest pthread shim (`thread.spawn` + futex), the async ring (`__vm_io_*` → 7-handle powerbox),
/// and the offload pool. Interpreter-only (the M:N executor + offload-pool oracle; the JIT async path
/// needs the separate `HostAsyncHooks` harness), exactly like `vm_async_io_runtime`. The total is
/// completion-order- *and* interleaving-invariant: Σ mix(i) for i in 0..16. Mirrors
/// `c_frontend::c_guest_async_work_stealing`.
#[test]
#[cfg(all(unix, target_arch = "x86_64"))]
fn vm_async_work_stealing_runtime() {
    let Some(bc) = compile_demo_libc_to_bc(
        "async_work_stealing",
        "async_work_stealing/async_work_stealing.c",
    ) else {
        return;
    };
    let m = svm_llvm::translate_bc_path(&bc)
        .expect("translate bitcode")
        .module;
    let import_names: Vec<&str> = m.imports.iter().map(|i| i.name.as_str()).collect();
    assert!(
        import_names.contains(&"vm_io_submit_async") && import_names.contains(&"vm_io_reap"),
        "expected async-ring imports, got {import_names:?}"
    );
    let module = svm_run::resolve_capability_imports(m).expect("resolve capability imports");
    svm_verify::verify_module(&module).expect("verify resolved IR");
    let nparams = module.funcs[0].params.len();
    assert_eq!(
        nparams, 7,
        "async program must declare the 7-handle powerbox entry"
    );

    let win = module.memory.map_or(0, |mc| 1u64 << mc.size_log2);
    let mut h = svm_interp::Host::new();
    let args = grant_powerbox(&mut h, win, nparams, std::time::Duration::from_millis(10));
    let mut fuel = 1_000_000_000u64;
    let out =
        svm_interp::run_with_host(&module, 0, &args, &mut fuel, &mut h).expect("interp async run");
    assert_eq!(out, vec![Value::I32(0)], "async demo returns 0");
    // NTASKS = 16 (see the demo); the total is Σ of the host's deterministic per-op results.
    let total: u64 = (0..16).fold(0u64, |a, i| a.wrapping_add(async_mix(i) as u64));
    assert_eq!(
        h.stdout,
        format!("{total}\n").into_bytes(),
        "async total Σ mix(0..16)"
    );
}

/// `__vm_cap(i)` reaches the **tail** powerbox handles (`i ≥ 4`) now that `synth_start` stashes them:
/// `__vm_cap(6)` (the Blocking slot) must equal `__vm_blocking_handle()` — both read stash slot 24.
/// Proves the relocated 8-handle layout is wired through both the generic `__vm_cap` reader and the
/// named builtin. Run on the interpreter under the 7-handle powerbox.
#[test]
#[cfg(unix)]
fn vm_cap_index_reaches_tail_handles() {
    let src = r#"
int __vm_blocking_handle(void);
int __vm_cap(int i);
int main(void) { return (__vm_cap(6) == __vm_blocking_handle()) ? 7 : 0; }
"#;
    let Some((m, _)) = translate_verified("vm_cap_tail", src) else {
        return;
    };
    assert_eq!(
        m.funcs[0].params.len(),
        7,
        "a Blocking-handle program declares the 7-handle entry"
    );
    let win = m.memory.map_or(0, |mc| 1u64 << mc.size_log2);
    let mut h = svm_interp::Host::new();
    let args = grant_powerbox(&mut h, win, 7, std::time::Duration::ZERO);
    let mut fuel = 50_000_000u64;
    let r = svm_interp::run_with_host(&m, 0, &args, &mut fuel, &mut h).expect("interp run");
    assert_eq!(
        r,
        vec![Value::I32(7)],
        "__vm_cap(6) == __vm_blocking_handle()"
    );
}

// ---- §22 guest-driven JIT: `__vm_jit_compile` / `invoke2` / `release` / `install` / `uninstall` ---

/// Structural: every guest-driven-JIT builtin lowers to a `CallImport` on the `Jit` import, and a
/// program using them is granted the full 8-handle powerbox (the `Jit` handle is the last `VM_CAP_*`
/// index, so `synth_start` grants the whole prefix). Verifies the import table + entry arity without
/// the (heavier) end-to-end blob dance below.
#[test]
fn vm_jit_builtins_lower_and_grant_full_powerbox() {
    let src = r#"
long __vm_jit_compile(void *blob, long len);
long __vm_jit_invoke2(long code, long a, long b);
long __vm_jit_install(long code);
long __vm_jit_uninstall(long slot);
long __vm_jit_release(long code);
static char blob[64];
int main(void) {
  long code = __vm_jit_compile(blob, 64);
  long r = __vm_jit_invoke2(code, 2, 3);
  long slot = __vm_jit_install(code);
  __vm_jit_uninstall(slot);
  __vm_jit_release(code);
  return (int)(r + slot);
}
"#;
    let Some(bc) = compile_to_bc("vm_jit_struct", src) else {
        return;
    };
    let m = svm_llvm::translate_bc_path(&bc)
        .expect("translate bitcode")
        .module;
    let imports: Vec<&str> = m.imports.iter().map(|i| i.name.as_str()).collect();
    for want in [
        "vm_jit_compile",
        "vm_jit_invoke2",
        "vm_jit_install",
        "vm_jit_uninstall",
        "vm_jit_release",
    ] {
        assert!(
            imports.contains(&want),
            "missing JIT import {want}: {imports:?}"
        );
    }
    assert!(
        svm_run::is_powerbox_entry(&m),
        "a JIT-builtin program must produce a powerbox entry"
    );
    assert_eq!(
        m.funcs[0].params.len(),
        8,
        "the Jit handle is the last VM_CAP_* index → the full 8-handle powerbox"
    );
    // Resolves + re-verifies (every `CallImport` binds to the `Jit` capability).
    let module = svm_run::resolve_capability_imports(m).expect("resolve capability imports");
    svm_verify::verify_module(&module).expect("verify resolved IR");
}

/// End-to-end: the **guest that JITs itself** (`demos/jit/jit_demo.c`) through the LLVM on-ramp. The
/// guest emits serialized SVM IR byte-by-byte into its window, `__vm_jit_compile`s it, and checks
/// `__vm_jit_invoke2` (raw unit) **and** an `__vm_jit_install`ed unit reached via a C function pointer
/// (`call_indirect`) against its own bytecode interpreter on a 49-input grid — guest-emitted,
/// host-verified, Cranelift-compiled, on the real JIT powerbox.
///
/// The blob's memory descriptor must declare the **same** `size_log2` as the parent (the validator's
/// exact-match precondition). The demo hardcodes chibicc's `16`; svm-llvm sizes the parent window
/// differently, so we translate once to learn its `size_log2`, patch the demo's descriptor to match,
/// then translate+run the patched source (no magic constant — it tracks svm-llvm's sizing).
#[test]
#[cfg(all(unix, target_arch = "x86_64"))]
fn vm_jit_guest_self_jit_demo() {
    // Replace the chibicc-only `#include <svm.h>` with the `__vm_jit_*` prototypes the demo uses (it
    // declares `write` itself) — so clang resolves them without chibicc's include dir on the path
    // (which would shadow the system `<stdlib.h>`/`<stdint.h>` for the other tests).
    let jit_decls = "\
long __vm_jit_compile(void *blob, long len);\n\
long __vm_jit_invoke2(long code, long a, long b);\n\
long __vm_jit_release(long code);\n\
long __vm_jit_install(long code);\n\
long __vm_jit_uninstall(long slot);\n";
    let src0 =
        include_str!("../../svm-run/demos/jit/jit_demo.c").replace("#include <svm.h>", jit_decls);
    let Some(bc0) = compile_to_bc("jit_probe", &src0) else {
        return;
    };
    // Probe svm-llvm's parent window size (translation does not check the memory-match — that is a
    // runtime precondition of `__vm_jit_compile`), then patch the blob's descriptor to it.
    let s = svm_llvm::translate_bc_path(&bc0)
        .expect("translate probe")
        .module
        .memory
        .expect("powerbox program declares a window")
        .size_log2;
    let src = src0.replace("eb(buf, 16);", &format!("eb(buf, {s});"));
    assert_ne!(
        src, src0,
        "expected to patch the blob memory descriptor `eb(buf, 16);`"
    );

    let Some(bc) = compile_to_bc("jit_demo", &src) else {
        return;
    };
    let m = svm_llvm::translate_bc_path(&bc)
        .expect("translate bitcode")
        .module;
    let module = svm_run::resolve_capability_imports(m).expect("resolve capability imports");
    svm_verify::verify_module(&module).expect("verify resolved IR");
    let run = svm_run::run_powerbox(&module, b"").expect("powerbox run");
    let out = String::from_utf8_lossy(&run.stdout);
    assert!(
        out.contains("98 inputs agree (invoke + installed call_indirect)"),
        "guest self-JIT (invoke + installed call_indirect) must agree on every input:\n{out}"
    );
}

/// End-to-end: the **threaded** guest-driven JIT (`demos/jit/jit_threads.c`, DESIGN §22) through the
/// LLVM on-ramp — the threaded sibling of `vm_jit_guest_self_jit_demo`. `NWORKERS` pthreads each build
/// serialized SVM IR for a *distinct* unit at runtime and `__vm_jit_compile` it — so several
/// `Jit.compile`s are in flight at once — then `__vm_jit_invoke2` the freshly-native code and check it
/// against a C reference on a grid of inputs. Combines the guest pthread shim (`thread.spawn`) with the
/// `Jit` capability + the **8-handle powerbox**; the host serializes the concurrent compiles through
/// the per-domain `Mutex<Host>` (engaged automatically for a `thread.spawn`ing guest) while execution
/// stays parallel. Prints `0` — no worker's concurrently-JITed unit disagreed.
///
/// Like the single-threaded demo, the blob's memory descriptor must match the parent window's
/// `size_log2` (the validator's exact-match precondition); the demo hardcodes chibicc's `16`, so we
/// probe svm-llvm's window and patch `eb(&e, 16);` to it before the real translate+run.
#[test]
#[cfg(all(unix, target_arch = "x86_64"))]
fn vm_jit_threads_demo() {
    let src0 = include_str!("../../svm-run/demos/jit/jit_threads.c");
    let Some(bc0) = compile_libc_src_to_bc("jit_threads_probe", src0) else {
        return;
    };
    // Probe svm-llvm's parent window size, then patch the blob descriptor to it (no magic constant —
    // it tracks svm-llvm's sizing, exactly as `vm_jit_guest_self_jit_demo` does).
    let s = svm_llvm::translate_bc_path(&bc0)
        .expect("translate probe")
        .module
        .memory
        .expect("powerbox program declares a window")
        .size_log2;
    let src = src0.replace("eb(&e, 16);", &format!("eb(&e, {s});"));
    assert_ne!(
        src, src0,
        "expected to patch the blob memory descriptor `eb(&e, 16);`"
    );

    let Some(bc) = compile_libc_src_to_bc("jit_threads", &src) else {
        return;
    };
    let t = svm_llvm::translate_bc_path(&bc).expect("translate bitcode");
    assert_eq!(
        t.module.funcs[0].params.len(),
        8,
        "a Jit-using program declares the full 8-handle powerbox entry"
    );
    let module = svm_run::resolve_capability_imports(t.module).expect("resolve capability imports");
    svm_verify::verify_module(&module).expect("verify resolved IR");
    let run =
        svm_run::run_powerbox_with_deadline(&module, b"", Some(std::time::Duration::from_secs(60)))
            .expect("powerbox run");
    assert_eq!(
        run.stdout, b"0\n",
        "every worker's concurrently-JITed unit must agree with the reference (got {:?})",
        String::from_utf8_lossy(&run.stdout)
    );
}

// ---- §13/§14 SharedRegion: `__vm_region_create` / `map` / `unmap` / `page_size` -----------------

/// The **magic ring buffer** in C through the LLVM on-ramp: a guest mints a `SharedRegion` from its
/// `AddressSpace` handle, maps it at two adjacent window offsets, and a single 8-byte store straddling
/// the seam wraps tail→head as one contiguous access — the whole point of the §13/§14 layout, with no
/// host hand-holding (the host only installed the region factory + granted AddressSpace). Exercises
/// `create` (on the stashed AddressSpace handle, slot 4) and `map`/`unmap`/`page_size` (on the returned
/// region handle), and the **5-handle powerbox** `synth_start` now grants. Run on the JIT powerbox
/// (real shared-memory aliasing); the success marker `'Y'` is asserted against stdout.
#[test]
#[cfg(unix)]
fn vm_region_magic_ring_buffer() {
    let src = r#"
long __vm_region_create(long len);
long __vm_region_map(int r, long win_off, long region_off, long len, int prot);
long __vm_region_unmap(int r, long win_off, long len);
long __vm_region_page_size(int r);
long write(int fd, const void *buf, long n);
int main(void) {
  int r = (int)__vm_region_create(1 << 16);          /* mint a 64 KiB region */
  if (r < 0) { char e='E'; write(1,&e,1); return 1; }
  long g = __vm_region_page_size(r);                 /* host map granularity */
  volatile long vbase = 268435456;                   /* 256 MiB — reserved tail, clear of data/stack */
  long base = vbase;                                 /* volatile defeats constant inttoptr fold */
  if (__vm_region_map(r, base, 0, g, 3) < 0)      { char e='1'; write(1,&e,1); return 2; }
  if (__vm_region_map(r, base + g, 0, g, 3) < 0)  { char e='2'; write(1,&e,1); return 3; }
  /* one 8-byte store straddling the seam: low half -> region tail, high half wraps -> region head */
  *(unsigned long *)(base + g - 4) = 0x1122334455667788UL;
  unsigned int head = *(unsigned int *)(base);             /* region head, via mapping 1 */
  unsigned int tail = *(unsigned int *)(base + 2 * g - 4); /* region tail, via mapping 2 */
  unsigned long combined = ((unsigned long)head << 32) | tail;
  long u = __vm_region_unmap(r, base, g);                  /* exercise unmap too (must return 0) */
  char ok = (combined == 0x1122334455667788UL && u == 0) ? 'Y' : 'N';
  write(1, &ok, 1);
  return 0;
}
"#;
    let Some(bc) = compile_to_bc("vm_region", src) else {
        return;
    };
    let m = svm_llvm::translate_bc_path(&bc)
        .expect("translate bitcode")
        .module;
    // Structural: the SharedRegion ops became their `CallImport`s, and the entry grants through
    // AddressSpace (5 handles: stdout, stdin, exit, memory, addrspace).
    let imports: Vec<&str> = m.imports.iter().map(|i| i.name.as_str()).collect();
    for want in [
        "vm_region_create",
        "vm_region_map",
        "vm_region_unmap",
        "vm_region_page_size",
    ] {
        assert!(
            imports.contains(&want),
            "missing region import {want}: {imports:?}"
        );
    }
    assert_eq!(
        m.funcs[0].params.len(),
        5,
        "a region-minting program grants through the AddressSpace handle (5-handle entry)"
    );
    let module = svm_run::resolve_capability_imports(m).expect("resolve capability imports");
    svm_verify::verify_module(&module).expect("verify resolved IR");
    let run = svm_run::run_powerbox(&module, b"").expect("powerbox run");
    assert_eq!(
        run.stdout, b"Y",
        "magic ring buffer: seam-straddling store must wrap tail->head (got {:?})",
        run.stdout
    );
}

// ============================================================================================
// Milestone 2 — beyond chibicc's C subset: the D54 **breadth proof**. The on-ramp consumes any
// LLVM frontend's bitcode, so a freestanding C++ TU (`-fno-exceptions -fno-rtti`) — classes,
// virtual dispatch (vtables → `call_indirect`), templates, mangled names — must run byte-identical
// to native `clang++`, with no translator change beyond what the C corpus already proved. (Rust is
// gated on a toolchain re-pin: `rustc`'s bundled LLVM is newer than our pinned 18.)
// ============================================================================================

/// Compile a freestanding C++ snippet to legalized LLVM-18 bitcode: `-fno-exceptions -fno-rtti`
/// keeps EH/RTTI out (the §18 stance), `-O2` runs mem2reg/SROA and auto-vectorization (the on-ramp
/// ingests the SIMD output). Returns `None` (skip) if `clang++` is unavailable.
fn compile_cpp_to_bc(name: &str, src: &str) -> Option<PathBuf> {
    let dir = std::env::temp_dir();
    let cc = dir.join(format!("svm_llvm_{}_{}.cpp", std::process::id(), name));
    let bc = dir.join(format!("svm_llvm_cpp_{}_{}.bc", std::process::id(), name));
    std::fs::write(&cc, src).expect("write C++ source");
    let status = Command::new("clang++")
        .args(["-O2", "-emit-llvm", "-c", "-fno-exceptions", "-fno-rtti"])
        .arg(&cc)
        .arg("-o")
        .arg(&bc)
        .status();
    match status {
        Ok(s) if s.success() => Some(bc),
        _ => {
            eprintln!("note: skipping {name} (clang++ unavailable)");
            None
        }
    }
}

/// The C++ breadth differential: build the TU with native `clang++` and through the on-ramp, and
/// assert identical stdout + exit. The program is a powerbox program (`extern "C" int main`, output
/// via `extern "C" write`), exactly like the C corpus demos.
fn check_cpp_vs_native(name: &str, src: &str, stdin: &[u8]) {
    let Some(bc) = compile_cpp_to_bc(name, src) else {
        return;
    };
    let cc = std::env::temp_dir().join(format!("svm_llvm_{}_{}.cpp", std::process::id(), name));
    let exe = std::env::temp_dir().join(format!("svm_llvm_cppnat_{}_{}", std::process::id(), name));
    match Command::new("clang++")
        .arg(&cc)
        .arg("-o")
        .arg(&exe)
        .status()
    {
        Ok(s) if s.success() => {}
        _ => {
            eprintln!("note: skipping {name} (clang++ link unavailable)");
            return;
        }
    }
    let native = {
        use std::io::Write;
        let mut child = Command::new(&exe)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .spawn()
            .expect("spawn native");
        child.stdin.take().unwrap().write_all(stdin).ok();
        child.wait_with_output().expect("run native")
    };
    let native_code = native.status.code().unwrap_or(-1) as u8;

    let t = svm_llvm::translate_bc_path(&bc).expect("translate C++ bitcode");
    assert!(
        svm_run::is_powerbox_entry(&t.module),
        "{name}: a libc-using C++ program must produce a powerbox entry"
    );
    let module = svm_run::resolve_capability_imports(t.module).expect("resolve capability imports");
    svm_verify::verify_module(&module).expect("verify translated IR");
    let run = svm_run::run_powerbox(&module, stdin).expect("powerbox run");
    assert_eq!(
        run.stdout, native.stdout,
        "{name}: svm stdout {:?} vs native {:?}",
        run.stdout, native.stdout
    );
    let svm_code = match run.outcome {
        svm_run::Outcome::Exited(c) => c as u8,
        svm_run::Outcome::Returned(ref v) => match v.first() {
            Some(svm_interp::Value::I32(x)) => *x as u8,
            _ => 0,
        },
    };
    assert_eq!(
        svm_code, native_code,
        "{name}: svm exit {svm_code} vs native {native_code}"
    );
}

/// **C++ first light** — classes + virtual dispatch through the on-ramp. Two shapes derive a common
/// polymorphic base; a loop sums their areas through a base pointer (a virtual call per element →
/// a vtable load + `call_indirect`), and the total is printed. Exercises vtables (function-pointer
/// global initializers, slice K), the `this` pointer, mangled names, and `call_indirect` (slice G) —
/// byte-identical to native `clang++`.
#[test]
fn cpp_virtual_dispatch_first_light() {
    let src = r#"
extern "C" long write(int fd, const void *buf, long n);

struct Shape {
  virtual int area() const { return 0; }
};
struct Square : Shape {
  int s;
  Square(int s) : s(s) {}
  int area() const override { return s * s; }
};
struct Rect : Shape {
  int w, h;
  Rect(int w, int h) : w(w), h(h) {}
  int area() const override { return w * h; }
};

static int sum_areas(Shape **shapes, int n) {
  int t = 0;
  for (int i = 0; i < n; i++) t += shapes[i]->area();
  return t;
}

extern "C" int main() {
  Square sq(5);
  Rect r(3, 4);
  Shape *shapes[2] = { &sq, &r };
  int t = sum_areas(shapes, 2); // 25 + 12 = 37

  char buf[16];
  int n = 0;
  char tmp[16];
  int k = 0;
  if (t == 0) buf[n++] = '0';
  while (t > 0) { tmp[k++] = (char)('0' + (t % 10)); t /= 10; }
  while (k > 0) buf[n++] = tmp[--k];
  buf[n++] = '\n';
  write(1, buf, n);
  return 0;
}
"#;
    check_cpp_vs_native("cpp_vdispatch", src, b"");
}

/// C++ breadth, deeper: heap `new`/`delete`, **virtual destructors**, and templates. A polymorphic
/// hierarchy is heap-allocated through `operator new` (defined over the guest `malloc`), summed via
/// virtual dispatch, then `delete`d through a base pointer — exercising the *deleting destructor* the
/// vtable carries (a virtual-call chain into `operator delete` → `free`). A function template
/// monomorphizes to an ordinary function. Byte-identical to native `clang++`.
#[test]
fn cpp_new_delete_virtual_dtor_templates() {
    let src = r#"
extern "C" long write(int fd, const void *buf, long n);
extern "C" void *malloc(unsigned long n);
extern "C" void free(void *p);

void *operator new(unsigned long n) { return malloc(n); }
void operator delete(void *p) noexcept { free(p); }
void operator delete(void *p, unsigned long) noexcept { free(p); }

template <typename T> static T add(T a, T b) { return a + b; }

struct Animal {
  int legs;
  Animal(int l) : legs(l) {}
  virtual int sound() const { return 0; }
  virtual ~Animal() {}
};
struct Dog : Animal {
  Dog() : Animal(4) {}
  int sound() const override { return 7; }
};
struct Bird : Animal {
  Bird() : Animal(2) {}
  int sound() const override { return 3; }
};

extern "C" int main() {
  Animal *zoo[2];
  zoo[0] = new Dog();
  zoo[1] = new Bird();
  int total = 0;
  for (int i = 0; i < 2; i++)
    total = add(total, zoo[i]->legs + zoo[i]->sound()); // (4+7) + (2+3) = 16
  for (int i = 0; i < 2; i++)
    delete zoo[i]; // virtual dtor via base ptr -> deleting dtor -> operator delete -> free

  char buf[16];
  int n = 0;
  char tmp[16];
  int k = 0;
  if (total == 0) buf[n++] = '0';
  while (total > 0) { tmp[k++] = (char)('0' + (total % 10)); total /= 10; }
  while (k > 0) buf[n++] = tmp[--k];
  buf[n++] = '\n';
  write(1, buf, n);
  return 0;
}
"#;
    check_cpp_vs_native("cpp_new_delete", src, b"");
}

/// C++ **static initialization** — `@llvm.global_ctors`. A global object with a side-effecting
/// constructor must run **before** `main` (the C++ [basic.start] order), exactly as native: the
/// on-ramp's `_start` now calls the global constructors (priority order) before `main`. Here a global
/// `Banner` ctor writes "init\n" and `main` writes "main\n" — the on-ramp must emit both, in order,
/// byte-identical to native `clang++` (a bug that drops static init would print only "main\n").
#[test]
fn cpp_global_constructor_runs_before_main() {
    let src = r#"
extern "C" long write(int fd, const void *buf, long n);
struct Banner {
  Banner() { write(1, "init\n", 5); }
};
static Banner g_banner;
extern "C" int main() {
  write(1, "main\n", 5);
  return 0;
}
"#;
    check_cpp_vs_native("cpp_static_init", src, b"");
}

// ============================================================================================
// Milestone 2 — Rust through the on-ramp (the D54 breadth headline). `rustc` bundles its own LLVM,
// so the bitcode version must match our pin (LLVM 18): the container's default `rustc` ships LLVM
// 21 (rejected by the pinned `llvm-ir`), but a **pinned LLVM-18 Rust toolchain** (1.81, LLVM 18.1)
// emits bitcode the existing reader accepts — no re-pin needed. A `no_std`/`panic=abort` crate has
// no EH/unwinding, so it lowers like C. (CI must `rustup toolchain install 1.81.0`, as it installs
// `llvm-18-dev` for the bitcode lane.)
// ============================================================================================

/// Compile a `no_std`/`panic=abort` Rust source to legalized **LLVM-18** bitcode via the pinned
/// Rust 1.81 toolchain (its LLVM 18.1 matches our `llvm-ir` pin). `-O` runs mem2reg/SROA. Returns
/// `None` (skip) if the toolchain is unavailable.
fn compile_rust_to_bc(name: &str, src: &str) -> Option<PathBuf> {
    let dir = std::env::temp_dir();
    let rs = dir.join(format!("svm_llvm_{}_{}.rs", std::process::id(), name));
    let bc = dir.join(format!("svm_llvm_rust_{}_{}.bc", std::process::id(), name));
    std::fs::write(&rs, src).expect("write Rust source");
    let status = Command::new("rustc")
        .args([
            "+1.81.0",
            "--emit=llvm-bc",
            "--crate-type=lib",
            "-C",
            "panic=abort",
            "-C",
            "opt-level=2",
            "-C",
            "overflow-checks=off",
            // `-O2` auto-vectorization stays **enabled**: the on-ramp now ingests the full SIMD
            // output (slices AN–AT — legalization + conversions/rotate/shuffle/`<N x i1>` masks), so a
            // `&[i32]` reduction becoming `<N x i32>` + a horizontal reduce is fine. (Determinism is
            // preserved by the fixed-128 chunk legalization, not by suppressing vectorization.)
        ])
        .arg(&rs)
        .arg("-o")
        .arg(&bc)
        .status();
    match status {
        Ok(s) if s.success() => Some(bc),
        _ => {
            eprintln!("note: skipping {name} (rustc +1.81.0 unavailable — `rustup toolchain install 1.81.0`)");
            None
        }
    }
}

/// The body of `compute` (shared by the `no_std` bitcode lib and the native std oracle): a sum of
/// squares. `clang`/`rustc -O2` closes this loop into a **polynomial with `i33` intermediates** (to
/// hold `n·(n-1)·(2n-1)` before a magic-constant divide), so it exercises the on-ramp's non-power-of-
/// two integer support — `i33` held in an `i64`, kept canonical by masking after the de-normalizing
/// ops. `wrapping_*` matches `-C overflow-checks=off` (LLVM's `nsw`/`nuw` are wrap for us, §3b).
const RUST_COMPUTE_BODY: &str = "{
    let mut acc: i32 = 0;
    let mut i: i32 = 0;
    while i < n { acc = acc.wrapping_add(i.wrapping_mul(i)); i = i.wrapping_add(1); }
    acc
}";

/// Run `compute(n)` through the on-ramp (a `no_std`/`panic=abort` lib → LLVM-18 bitcode → translate →
/// both backends), asserting interp == JIT, and return the result. `None` if the toolchain is absent.
fn rust_compute_onramp(n: i32) -> Option<i32> {
    let src = format!(
        "#![no_std]\n#![no_main]\n\
         #[panic_handler] fn ph(_: &core::panic::PanicInfo) -> ! {{ loop {{}} }}\n\
         #[no_mangle] pub extern \"C\" fn compute(n: i32) -> i32 {RUST_COMPUTE_BODY}\n"
    );
    let bc = compile_rust_to_bc("rs_compute", &src)?;
    let t = svm_llvm::translate_bc_path(&bc).expect("translate Rust bitcode");
    let module = t.module;
    svm_verify::verify_module(&module).expect("verify translated Rust IR");
    // A `no_std` lib has no `main`/powerbox; the panic handler (`rust_begin_unwind`, a `loop {}`) is
    // also defined (at index 0), so locate `compute` by its IR signature `(i64 sp, i32) -> i32`.
    let idx = module
        .funcs
        .iter()
        .position(|f| f.params == [ValType::I64, ValType::I32] && f.results == [ValType::I32])
        .expect("compute present") as u32;
    let full = vec![Value::I64(t.entry_sp as i64), Value::I32(n)];
    let mut fuel = 100_000_000u64;
    let interp = match svm_interp::run(&module, idx, &full, &mut fuel)
        .expect("interp run")
        .as_slice()
    {
        [Value::I32(x)] => *x,
        other => panic!("compute: expected one i32, got {other:?}"),
    };
    let slots = vec![t.entry_sp as i64, n as i64];
    let jit = match svm_jit::compile_and_run(&module, idx, &slots).expect("jit run") {
        JitOutcome::Returned(s) => s[0] as i32,
        other => panic!("unexpected JIT outcome {other:?}"),
    };
    assert_eq!(interp, jit, "compute({n}): interp {interp} vs JIT {jit}");
    Some(interp)
}

/// The native oracle: the **same** `compute` body compiled by `rustc 1.81` into a std binary that
/// prints `compute(n)`, run natively. (A `no_std` lib can't be run directly; std `compute` is the
/// identical function, so it is the ground truth — incl. the `i33` overflow/wrap path.)
fn rust_compute_native(n: i32) -> Option<i32> {
    let dir = std::env::temp_dir();
    let rs = dir.join(format!("svm_llvm_{}_rsnat.rs", std::process::id()));
    let exe = dir.join(format!("svm_llvm_{}_rsnat", std::process::id()));
    std::fs::write(
        &rs,
        format!(
            "fn compute(n: i32) -> i32 {RUST_COMPUTE_BODY}\n\
             fn main() {{ println!(\"{{}}\", compute({n})); }}\n"
        ),
    )
    .expect("write Rust source");
    match Command::new("rustc")
        .args(["+1.81.0", "-C", "opt-level=2", "-C", "overflow-checks=off"])
        .arg(&rs)
        .arg("-o")
        .arg(&exe)
        .status()
    {
        Ok(s) if s.success() => {}
        _ => return None,
    }
    let out = Command::new(&exe).output().expect("run native rust");
    Some(
        String::from_utf8_lossy(&out.stdout)
            .trim()
            .parse()
            .expect("parse native compute"),
    )
}

/// **Rust through the on-ramp — the D54 breadth headline.** A `no_std`/`panic=abort` Rust crate
/// (a different frontend, its own ABI/lowering) translates and runs with no translator change beyond
/// the C corpus, given a version-matched toolchain (Rust 1.81 / LLVM 18). The chosen function forces
/// LLVM's `i33` closed-form (the non-power-of-two integer support added here), and the values include
/// `n` large enough that the `i33` intermediate **overflows 33 bits and wraps** — caught only by the
/// native differential (interp == JIT alone would agree even if the masking were wrong). Every value
/// matches native `rustc`.
#[test]
fn rust_no_std_matches_native() {
    for n in [5i32, 1000, 46341, 200000, -7] {
        let (Some(svm), Some(native)) = (rust_compute_onramp(n), rust_compute_native(n)) else {
            return; // toolchain unavailable — skip
        };
        assert_eq!(
            svm, native,
            "compute({n}): on-ramp {svm} vs native rustc {native} (i33 wrap mismatch?)"
        );
    }
}

// ---- §6 / D-DBG-7: the debug-info waist (LLVM as the third producer) -------------------------

/// The LLVM on-ramp populates the §6 frontend-neutral debug-info waist's **source-line half** from
/// each instruction's `!DILocation` — the third independent frontend (after chibicc and the wasm
/// DWARF producer) to feed the *same* neutral core, the cross-check that the waist isn't coupled to
/// any one frontend (DEBUGGING.md §6). (The variable/type half is covered by the `_o0_`/`_og_`
/// tests; here `n` rides in as an argument var, asserted minimally.)
#[test]
fn llvm_dilocation_maps_into_the_debug_info_waist() {
    // A chain of dependent statements over a runtime input: each keeps its own source line and
    // lowers to a real (non-terminator) arithmetic op, so several distinct lines reach the IR pcs
    // (clang can't constant-fold the chain away, and nothing collapses onto one line).
    let src = "\
int chain(int n) {
  int a = n + 1;
  int b = a * 3;
  int c = b - 2;
  int d = c * c;
  return d + a;
}
";
    let Some(bc) = compile_to_bc_g("dilocation", src) else {
        return; // toolchain unavailable — skip
    };
    let t = svm_llvm::translate_bc_path(&bc).expect("translate bitcode");
    // The debug section is strippable / untrusted-for-escape — it must not affect verification.
    svm_verify::verify_module(&t.module).expect("verify");

    let di = t
        .module
        .debug_info
        .as_ref()
        .expect("debug info populated from !DILocation");

    // The file table names the C source (clang records its compile path).
    assert!(
        di.files.iter().any(|f| f.ends_with(".c")),
        "file table names the C source: {:?}",
        di.files
    );
    // Source lines are mapped onto IR pcs.
    assert!(!di.locs.is_empty(), "some source locations were mapped");
    // The parameter `n` is ingested via `dbg.value` (the variable half — exercised in depth by the
    // `_o0_`/`_og_` tests).
    assert!(
        di.vars.iter().any(|v| v.name == "n"),
        "the parameter is ingested"
    );

    // Every loc resolves to an in-range IR `(func, block, inst)` — the cross-check that the
    // LLVM-instruction → SVM-op mapping is self-consistent.
    for l in &di.locs {
        assert!((l.file as usize) < di.files.len(), "loc file in range");
        let f = l.func as usize;
        assert!(f < t.module.funcs.len(), "loc func {f} in range");
        let b = l.block as usize;
        assert!(b < t.module.funcs[f].blocks.len(), "loc block in range");
        assert!(
            (l.inst as usize) < t.module.funcs[f].blocks[b].insts.len(),
            "loc inst in range"
        );
        assert!(l.line >= 1, "a real source line");
    }

    // The body spans several source lines (the multiply/add at line 4, the return at line 5) — not
    // everything collapsed onto the function's opening line.
    let lines: std::collections::BTreeSet<u32> = di.locs.iter().map(|l| l.line).collect();
    assert!(
        lines.len() >= 3,
        "the statement chain maps several distinct source lines: {lines:?}"
    );

    // §6 function names: the `DISubprogram` source name is ingested into `func_names` (mapped to its
    // IR function index), so an LLVM-frontend backtrace reads `chain` instead of `fn{N}`.
    let chain = di
        .func_names
        .iter()
        .find(|fnm| fnm.name == "chain")
        .expect("the chain() function name is ingested");
    assert!(
        (chain.func as usize) < t.module.funcs.len(),
        "func_names index in range"
    );
}

/// A non-`-g` build carries **no** debug section — the waist is absent (zero cost), byte-identical
/// to before this producer existed.
#[test]
fn llvm_without_g_has_no_debug_info() {
    let Some((m, _)) = translate_verified("no_g", "int id(int x) { return x; }") else {
        return;
    };
    assert!(m.debug_info.is_none(), "no -g ⇒ no debug section");
}

/// The §6 **variable/type half** for LLVM: a direct `llvm-sys` walk of the `-O0 -g` DI metadata
/// recovers each source local's name + structured type and correlates it to the IR by alloca
/// ordinal, landing it in the waist as a `Window` var — the LLVM analog of the wasm DWARF
/// aggregate/pointer/array ingest (DEBUGGING.md slice 25). Mirrors the wasm
/// `wasm_dwarf_ingests_aggregate_pointer_and_array_types` test over the same struct/array/pointer
/// shapes, the cross-frontend cross-check that the structured-type waist is genuinely neutral.
#[test]
fn llvm_o0_ingests_aggregate_pointer_and_array_variables() {
    use svm_ir::{TypeDef, VarLoc};

    // `pp = &p` forces the struct to stay in memory (a real dbg.declare/alloca, not a dbg.value).
    let src = "\
struct Point { int x; int y; };
int dist(int n) {
  struct Point p;
  int row[3];
  struct Point *pp = &p;
  p.x = n; p.y = n + 1;
  row[0] = n;
  return p.x + p.y + row[0] + pp->x;
}
";
    let Some(bc) = compile_to_bc_o0g("llvm_vars", src) else {
        return; // toolchain unavailable — skip
    };
    let t = svm_llvm::translate_bc_path(&bc).expect("translate bitcode");
    svm_verify::verify_module(&t.module).expect("verify"); // debug info is escape-irrelevant
    let di = t.module.debug_info.as_ref().expect("debug info");

    let var = |n: &str| {
        di.vars.iter().find(|v| v.name == n).unwrap_or_else(|| {
            panic!(
                "var {n}: {:?}",
                di.vars.iter().map(|v| &v.name).collect::<Vec<_>>()
            )
        })
    };
    let var_type = |n: &str| &di.types[var(n).type_id.expect("typed") as usize];

    // Every local is a `Window` frame slot (the alloca's data-stack offset); offsets are distinct.
    let mut offs = Vec::new();
    for n in ["p", "row", "pp"] {
        let VarLoc::Window { off } = var(n).loc else {
            panic!("{n} is a Window var, got {:?}", var(n).loc);
        };
        assert!(off >= 0 && (off as u64) < t.module.funcs[0].blocks.len() as u64 * 4096);
        offs.push(off);
    }
    offs.sort();
    offs.dedup();
    assert_eq!(offs.len(), 3, "p/row/pp occupy distinct frame slots");

    // `struct Point p` — aggregate x@0, y@4, both 4-byte ints, size 8.
    let TypeDef::Aggregate { name, size, fields } = var_type("p") else {
        panic!("p is a struct, got {:?}", var_type("p"));
    };
    assert_eq!(name, "struct Point");
    assert_eq!(*size, 8);
    assert_eq!(
        fields
            .iter()
            .map(|f| (f.name.as_str(), f.offset))
            .collect::<Vec<_>>(),
        vec![("x", 0), ("y", 4)]
    );
    assert!(matches!(
        &di.types[fields[0].ty as usize],
        TypeDef::Base { size: 4, .. }
    ));

    // `int row[3]` — array of 3 ints.
    let TypeDef::Array { elem, count, .. } = var_type("row") else {
        panic!("row is an array, got {:?}", var_type("row"));
    };
    assert_eq!(*count, 3);
    assert!(matches!(
        &di.types[*elem as usize],
        TypeDef::Base { size: 4, .. }
    ));

    // `struct Point *pp` — pointer whose pointee is the same aggregate as `p`.
    let TypeDef::Pointer { pointee, name, .. } = var_type("pp") else {
        panic!("pp is a pointer, got {:?}", var_type("pp"));
    };
    assert_eq!(name, "struct Point *");
    assert!(matches!(
        &di.types[*pointee as usize],
        TypeDef::Aggregate { name, .. } if name == "struct Point"
    ));
}

/// Runtime proof that the alloca-ordinal correlation lands on the **right** frame slots: stop the
/// interpreter just before `dist` returns and read each source variable back through the §6 waist
/// (`Window` reads at the resolved data-stack offset). Locks that the `dbg.declare` address →
/// alloca ordinal → frame offset chain is correct, not merely structurally plausible.
#[test]
fn llvm_o0_variables_read_at_runtime() {
    use svm_interp::{Inspector, IrPc, Stop, VarValue};

    let src = "\
struct Point { int x; int y; };
int dist(int n) {
  struct Point p;
  int row[3];
  struct Point *pp = &p;
  p.x = n; p.y = n + 1;
  row[0] = n;
  return p.x + p.y + row[0] + pp->x;
}
";
    let Some(bc) = compile_to_bc_o0g("llvm_vars_rt", src) else {
        return; // toolchain unavailable — skip
    };
    let t = svm_llvm::translate_bc_path(&bc).expect("translate bitcode");
    svm_verify::verify_module(&t.module).expect("verify");

    // Break at the last instruction of the last block — `dist` is branch-free at -O0, so by here
    // every store (p.x=n, p.y=n+1, row[0]=n) has executed.
    let lb = t.module.funcs[0].blocks.len() - 1;
    let li = t.module.funcs[0].blocks[lb].insts.len() - 1;
    let n = 5i32;
    let args = [Value::I64(t.entry_sp as i64), Value::I32(n)];
    let mut insp = Inspector::attach(&t.module, 0, &args, 50_000_000);
    insp.set_breakpoint(IrPc {
        module: 0,
        func: 0,
        block: lb,
        inst: li,
    });
    assert!(
        matches!(insp.run_until_stop(), Stop::Break { .. }),
        "stopped at the return"
    );

    let i32_at = |vv: Option<VarValue>, off: usize| -> i32 {
        match vv {
            Some(VarValue::Bytes(b)) => i32::from_le_bytes(b[off..off + 4].try_into().unwrap()),
            other => panic!("expected window bytes, got {other:?}"),
        }
    };
    // `struct Point p` reads x = n, y = n + 1 (8 bytes: x then y).
    let p = insp.read_var(0, "p", 8);
    assert_eq!(i32_at(p.clone(), 0), n, "p.x");
    assert_eq!(i32_at(p, 4), n + 1, "p.y");
    // `int row[3]` — element 0 was set to n.
    assert_eq!(i32_at(insp.read_var(0, "row", 4), 0), n, "row[0]");
}

/// The §6 variable half at **`-O2`/`-Og`**: `llvm.dbg.value` binds a source variable to an SSA
/// value rather than memory (mem2reg/SROA promoted it). The `di` reader recovers `dbg.value`
/// bindings to a function **argument** and the translator emits a `VarLoc::SsaList` over the
/// argument's live range (its block-local index per block) — the LLVM frontend exercising the same
/// location-list machinery chibicc and wasm use, the case where LLVM's debug intrinsics surviving
/// optimization make the parameter inspectable for free (DEBUGGING.md slice 26).
#[test]
fn llvm_og_ingests_argument_via_dbg_value_ssalist() {
    use svm_interp::{Inspector, IrPc, Stop, VarValue};
    use svm_ir::{Encoding, TypeDef, VarLoc};

    // A loop keeps `n` live across multiple blocks, so the SsaList spans more than the entry block.
    let src = "\
int scaled(int n) {
  int total = 0;
  for (int k = 0; k < 4; k++)
    total += n;
  return total;
}
";
    let Some(bc) = compile_to_bc_g("og_arg", src) else {
        return; // toolchain unavailable — skip
    };
    let t = svm_llvm::translate_bc_path(&bc).expect("translate bitcode");
    svm_verify::verify_module(&t.module).expect("verify");
    let di = t.module.debug_info.as_ref().expect("debug info");

    // `n` (the argument) is ingested as a typed SSA-located var with at least one location-list
    // entry (no `Window` slot — it was promoted to a register).
    let n_var = di.vars.iter().find(|v| v.name == "n").unwrap_or_else(|| {
        panic!(
            "n ingested: {:?}",
            di.vars.iter().map(|v| &v.name).collect::<Vec<_>>()
        )
    });
    let VarLoc::SsaList(locs) = &n_var.loc else {
        panic!("n is an SsaList var, got {:?}", n_var.loc);
    };
    assert!(!locs.is_empty(), "n has location-list entries");
    for l in locs {
        assert!(
            (l.block as usize) < t.module.funcs[0].blocks.len(),
            "entry block in range"
        );
    }
    assert!(matches!(
        &di.types[n_var.type_id.expect("typed") as usize],
        TypeDef::Base {
            encoding: Encoding::Signed,
            size: 4,
            ..
        }
    ));

    // Runtime: stop in the entry block and read `n` back through the SsaList → the argument value.
    let n = 7i32;
    let args = [Value::I64(t.entry_sp as i64), Value::I32(n)];
    let mut insp = Inspector::attach(&t.module, 0, &args, 50_000_000);
    // The first entry-block instruction is a step point with the argument already live.
    insp.set_breakpoint(IrPc {
        module: 0,
        func: 0,
        block: 0,
        inst: 0,
    });
    assert!(
        matches!(insp.run_until_stop(), Stop::Break { .. }),
        "stopped in entry"
    );
    assert_eq!(
        insp.read_var(0, "n", 4),
        Some(VarValue::Value(Value::I32(n))),
        "n reads the arg"
    );
}

/// The §6 **module-scoped global** half (DEBUGGING.md slice 28): a source-level global variable is
/// ingested from its `!dbg` `DIGlobalVariableExpression` as a `GLOBAL_SCOPE` `VarLoc::Fixed` var at
/// the global's window address (correlated by symbol name to the globals layout) — visible in every
/// frame, with its structured type. Reads back its data-segment value at runtime.
#[test]
fn llvm_ingests_source_globals_as_fixed_vars() {
    use svm_interp::{Inspector, IrPc, Stop, VarValue};
    use svm_ir::{TypeDef, VarLoc};

    let src = "\
int counter = 7;
struct P { int a; int b; } origin = { 3, 4 };
int bump(int n) { counter = counter + n; return counter + origin.a; }
";
    let Some(bc) = compile_to_bc_o0g("globals", src) else {
        return; // toolchain unavailable — skip
    };
    let t = svm_llvm::translate_bc_path(&bc).expect("translate bitcode");
    svm_verify::verify_module(&t.module).expect("verify");
    let di = t.module.debug_info.as_ref().expect("debug info");

    let g = |n: &str| {
        di.vars
            .iter()
            .find(|v| v.name == n && v.func == svm_ir::GLOBAL_SCOPE)
            .unwrap_or_else(|| panic!("global {n} ingested"))
    };
    // `counter` is a fixed-address int global; `origin` a fixed-address struct (expandable).
    let counter = g("counter");
    let VarLoc::Fixed { addr: counter_addr } = counter.loc else {
        panic!("counter is Fixed, got {:?}", counter.loc);
    };
    assert!(matches!(
        &di.types[counter.type_id.expect("typed") as usize],
        TypeDef::Base { size: 4, .. }
    ));
    let origin = g("origin");
    assert!(matches!(origin.loc, VarLoc::Fixed { .. }));
    assert!(matches!(
        &di.types[origin.type_id.expect("typed") as usize],
        TypeDef::Aggregate { name, .. } if name == "struct P"
    ));
    assert_ne!(counter_addr, 0, "a real window address");

    // Runtime: at `bump`'s entry the data segment holds counter = 7; read it through the global.
    let args = [Value::I64(t.entry_sp as i64), Value::I32(10)];
    let mut insp = Inspector::attach(&t.module, 0, &args, 50_000_000);
    insp.set_breakpoint(IrPc {
        module: 0,
        func: 0,
        block: 0,
        inst: 0,
    });
    assert!(
        matches!(insp.run_until_stop(), Stop::Break { .. }),
        "stopped at entry"
    );
    let read_i32 = |insp: &Inspector| match insp.read_var(0, "counter", 4) {
        Some(VarValue::Bytes(b)) => i32::from_le_bytes(b[..4].try_into().unwrap()),
        other => panic!("expected window bytes, got {other:?}"),
    };
    assert_eq!(
        read_i32(&insp),
        7,
        "counter's initial value, read globally at entry"
    );
}

// --- Rust breadth, deeper: `core`-using programs (enums/`match`, slices, iterators, `Option`) ----

/// Translate a `no_std` Rust crate whose `items` define `#[no_mangle] pub extern "C" fn
/// run(n: i32) -> i32` (+ any types/helpers), run `run(n)` on both backends (interp == JIT), and
/// return the result. `None` if the toolchain is unavailable.
fn rust_run_onramp(name: &str, items: &str, n: i32) -> Option<i32> {
    let src = format!(
        "#![no_std]\n#![no_main]\n\
         #[panic_handler] fn ph(_: &core::panic::PanicInfo) -> ! {{ loop {{}} }}\n{items}\n"
    );
    let bc = compile_rust_to_bc(name, &src)?;
    let t = svm_llvm::translate_bc_path(&bc).expect("translate Rust bitcode");
    let module = t.module;
    svm_verify::verify_module(&module).expect("verify translated Rust IR");
    let idx = module
        .funcs
        .iter()
        .position(|f| f.params == [ValType::I64, ValType::I32] && f.results == [ValType::I32])
        .expect("run present") as u32;
    let mut fuel = 200_000_000u64;
    let interp = match svm_interp::run(
        &module,
        idx,
        &[Value::I64(t.entry_sp as i64), Value::I32(n)],
        &mut fuel,
    )
    .expect("interp run")
    .as_slice()
    {
        [Value::I32(x)] => *x,
        other => panic!("run: expected one i32, got {other:?}"),
    };
    let jit = match svm_jit::compile_and_run(&module, idx, &[t.entry_sp as i64, n as i64])
        .expect("jit run")
    {
        JitOutcome::Returned(s) => s[0] as i32,
        other => panic!("unexpected JIT outcome {other:?}"),
    };
    assert_eq!(interp, jit, "run({n}): interp {interp} vs JIT {jit}");
    Some(interp)
}

/// Native oracle: the same `items` (with `run`) compiled by `rustc 1.81` into a std binary printing
/// `run(n)`, run natively.
fn rust_run_native(name: &str, items: &str, n: i32) -> Option<i32> {
    let dir = std::env::temp_dir();
    // Per-test unique paths (`name`) — tests run in parallel, so a shared path would race.
    let rs = dir.join(format!("svm_llvm_{}_{}_rsrun.rs", std::process::id(), name));
    let exe = dir.join(format!("svm_llvm_{}_{}_rsrun", std::process::id(), name));
    std::fs::write(
        &rs,
        format!("{items}\nfn main() {{ println!(\"{{}}\", run({n})); }}\n"),
    )
    .expect("write Rust source");
    match Command::new("rustc")
        .args([
            "+1.81.0",
            "--edition",
            "2021",
            "-C",
            "opt-level=2",
            "-C",
            "overflow-checks=off",
        ])
        .arg(&rs)
        .arg("-o")
        .arg(&exe)
        .status()
    {
        Ok(s) if s.success() => {}
        _ => return None,
    }
    let out = Command::new(&exe).output().expect("run native rust");
    Some(
        String::from_utf8_lossy(&out.stdout)
            .trim()
            .parse()
            .expect("parse native run"),
    )
}

/// Drive `items` (which define `run(i32)->i32`) through both lanes for each `n`, asserting the on-ramp
/// matches native `rustc`.
fn check_rust_run_vs_native(name: &str, items: &str, ns: &[i32]) {
    for &n in ns {
        let (Some(svm), Some(native)) = (
            rust_run_onramp(name, items, n),
            rust_run_native(name, items, n),
        ) else {
            return; // toolchain unavailable — skip
        };
        assert_eq!(
            svm, native,
            "{name}: run({n}) on-ramp {svm} vs native {native}"
        );
    }
}

/// **Real Rust** — idiomatic `core` (a `#[repr(u8)]` enum dispatched by `match`, fixed arrays +
/// slice iteration, `Option` + `match`, an iterator `find` with a closure) through the on-ramp,
/// byte-identical to native `rustc`. A bytecode-style fold over a data array, then an `Option`-typed
/// search — the shapes a real `no_std` Rust program is built from. Exercises enum discriminants →
/// `switch`/`br_table`, slice indexing, niche-optimized `Option<i32>`, and `-O2` iterator lowering.
#[test]
fn rust_core_enum_slice_option() {
    let items = r#"
#[no_mangle]
pub extern "C" fn run(n: i32) -> i32 {
    #[derive(Clone, Copy)]
    #[repr(u8)]
    enum Op { Add, Mul, Xor, Sub }
    fn apply(op: Op, a: i32, x: i32) -> i32 {
        match op {
            Op::Add => a.wrapping_add(x),
            Op::Mul => a.wrapping_mul(x),
            Op::Xor => a ^ x,
            Op::Sub => a.wrapping_sub(x),
        }
    }
    let prog = [Op::Add, Op::Mul, Op::Xor, Op::Sub, Op::Add];
    let data = [3i32, 1, 4, 1, 5, 9, 2, 6];
    let mut acc = n;
    let mut k = 0usize;
    for &x in data.iter() {
        acc = apply(prog[k % prog.len()], acc, x);
        k += 1;
    }
    acc = acc.wrapping_add(match data.iter().copied().find(|&v| v > n) {
        Some(v) => v,
        None => -1,
    });
    acc
}
"#;
    check_rust_run_vs_native("rs_core", items, &[0, 1, 5, 100, -3, 1000]);
}

/// **Real Rust** — by-value `struct`s, an array-of-struct with slice iteration + field access, a
/// by-value struct argument, signed `/`+`%`, and an iterator `map().max()` yielding `Option<i32>`.
/// The aggregate/by-value-struct shapes (clang's small-struct coercion, slice J) and `-O2` iterator
/// lowering, from the Rust frontend — byte-identical to native `rustc`.
#[test]
fn rust_core_structs_and_iterators() {
    let items = r#"
#[no_mangle]
pub extern "C" fn run(n: i32) -> i32 {
    #[derive(Clone, Copy)]
    struct Pt { x: i32, y: i32 }
    fn norm(p: Pt) -> i32 { p.x.wrapping_mul(p.x).wrapping_add(p.y.wrapping_mul(p.y)) }
    let pts = [
        Pt { x: 1, y: 2 },
        Pt { x: n, y: 3 },
        Pt { x: 4, y: n / 2 },
        Pt { x: n % 5, y: 7 },
    ];
    let mut s = 0i32;
    for p in pts.iter() {
        s = s.wrapping_add(norm(*p));
    }
    let best = pts.iter().map(|p| norm(*p)).max();
    s.wrapping_add(match best { Some(m) => m, None => 0 })
}
"#;
    check_rust_run_vs_native("rs_structs", items, &[0, 3, 10, -8, 100]);
}

/// **Real Rust panic paths** — a runtime division emits non-elidable div-by-zero + overflow checks
/// whose panic branches call `core::panicking::panic_const_*` (external libcore). Under `panic=abort`
/// the on-ramp lowers those to a trap (drop + the trailing `unreachable`), so the program *translates*
/// — the gap that blocks essentially all real Rust. The divisor here is `(n & 7) + 1` through
/// `black_box` (always ≥ 1, but opaque so the panic paths stay in the IR), so the non-panic path runs
/// and `n / d` matches native `rustc` exactly. Without the fix, translation fails on the undefined
/// `panic_const_div_by_zero` reference.
#[test]
fn rust_panic_path_div_traps_and_runs() {
    let items = r#"
#[no_mangle]
pub extern "C" fn run(n: i32) -> i32 {
    let tmp = (n & 7) + 1;                      // [1, 8] — never zero…
    let d = unsafe { core::ptr::read_volatile(&tmp) }; // …but opaque (a volatile load), so the
                                                // div-by-zero + overflow panic checks stay in the IR
    (n / d).wrapping_add(n % d)
}
"#;
    check_rust_run_vs_native("rs_panic", items, &[0, 1, 7, 100, -50, 1234]);
}

/// **Rust trait objects** — `&dyn Trait` dynamic dispatch through the on-ramp. Two types implement a
/// trait; an array of `&dyn Shape` (each a `{data, vtable}` fat pointer) is iterated and the method is
/// called dynamically — a vtable load + `call_indirect` per element, the Rust analog of the C++ vtable
/// path (slice AG). Exercises Rust vtable globals (function-pointer initializers, slice K), fat-pointer
/// aggregates, and dynamic dispatch — byte-identical to native `rustc`.
#[test]
fn rust_trait_object_dispatch() {
    let items = r#"
trait Shape {
    fn area(&self) -> i32;
}
struct Sq(i32);
struct Rect(i32, i32);
impl Shape for Sq {
    fn area(&self) -> i32 { self.0.wrapping_mul(self.0) }
}
impl Shape for Rect {
    fn area(&self) -> i32 { self.0.wrapping_mul(self.1) }
}

#[no_mangle]
pub extern "C" fn run(n: i32) -> i32 {
    let sq = Sq(n);
    let rect = Rect(n, 3);
    let shapes: [&dyn Shape; 2] = [&sq, &rect];
    let mut total = 0i32;
    for s in shapes.iter() {
        total = total.wrapping_add(s.area()); // dynamic dispatch via the vtable
    }
    total
}
"#;
    check_rust_run_vs_native("rs_traits", items, &[0, 2, 7, -4, 100]);
}

/// **Rust slices as arguments** — `&[i32]` (a `{ptr, len}` fat pointer) passed across a real
/// (`#[inline(never)]`) call boundary, plus a sub-slice (`&data[1..4]`). Exercises the slice-arg ABI
/// and bounds-checked range indexing (provably in-bounds → elided), vs native `rustc`.
#[test]
fn rust_slice_argument() {
    let items = r#"
#[inline(never)]
fn sum(s: &[i32]) -> i32 {
    let mut t = 0i32;
    for &x in s { t = t.wrapping_add(x); }
    t
}
#[no_mangle]
pub extern "C" fn run(n: i32) -> i32 {
    let data = [n, n.wrapping_add(1), n.wrapping_add(2), 7, 5, 6];
    sum(&data).wrapping_add(sum(&data[1..4]))
}
"#;
    check_rust_run_vs_native("rs_slice", items, &[0, 10, -3]);
}

/// **Rust `Option::unwrap`** — the unwrap panic path (`core::panicking::panic` / `unwrap_failed`) is
/// in the IR; under `panic=abort` it lowers to a trap (slice AI's recognizer). The value is always
/// `Some` at runtime, so the non-panic path runs and matches native `rustc`.
#[test]
fn rust_option_unwrap() {
    let items = r#"
#[no_mangle]
pub extern "C" fn run(n: i32) -> i32 {
    let v: Option<i32> = if (n & 1) == 0 { Some(n.wrapping_mul(3)) } else { Some(n.wrapping_sub(1)) };
    v.unwrap().wrapping_add(7) // always Some at runtime; the None panic path traps
}
"#;
    check_rust_run_vs_native("rs_unwrap", items, &[0, 1, 8, -5]);
}

/// Run a `no_std` + `alloc` Rust crate (whose `items` define `fn compute() -> i32`) **through the
/// powerbox**: the on-ramp synthesizes `#[no_mangle] extern "C" fn main` calling `compute`, so it gets
/// a powerbox `_start` that grants the `Memory` handle and seeds the heap (the `vm_map`-growing bump
/// allocator the program's `#[global_allocator]` reaches via `malloc`). Returns `compute()`'s value as
/// the program's `u8` exit/return code, run on the JIT. `None` if the toolchain is unavailable.
fn rust_alloc_onramp(name: &str, items: &str) -> Option<u8> {
    let src = format!(
        "#![no_std]\n#![no_main]\n\
         #[panic_handler] fn ph(_: &core::panic::PanicInfo) -> ! {{ loop {{}} }}\n\
         {items}\n\
         #[no_mangle] pub extern \"C\" fn main() -> i32 {{ compute() }}\n"
    );
    let bc = compile_rust_to_bc(name, &src)?;
    let t = svm_llvm::translate_bc_path(&bc).expect("translate Rust bitcode");
    assert!(
        svm_run::is_powerbox_entry(&t.module),
        "{name}: an alloc program must produce a powerbox entry (Memory granted)"
    );
    let module = svm_run::resolve_capability_imports(t.module).expect("resolve capability imports");
    svm_verify::verify_module(&module).expect("verify translated Rust IR");
    let run = svm_run::run_powerbox(&module, b"").expect("powerbox run");
    Some(match run.outcome {
        svm_run::Outcome::Exited(c) => c as u8,
        svm_run::Outcome::Returned(ref v) => match v.first() {
            Some(svm_interp::Value::I32(x)) => *x as u8,
            _ => 0,
        },
    })
}

/// Native oracle: the same `items` (with `compute`) built as a std binary that `process::exit`s with
/// `compute()`, run natively; its `u8` exit code is the ground truth.
fn rust_alloc_native(name: &str, items: &str) -> Option<u8> {
    let dir = std::env::temp_dir();
    let rs = dir.join(format!("svm_llvm_{}_{}_alloc.rs", std::process::id(), name));
    let exe = dir.join(format!("svm_llvm_{}_{}_alloc", std::process::id(), name));
    std::fs::write(
        &rs,
        format!("{items}\nfn main() {{ std::process::exit(compute()); }}\n"),
    )
    .expect("write Rust source");
    match Command::new("rustc")
        .args([
            "+1.81.0",
            "--edition",
            "2021",
            "-C",
            "opt-level=2",
            "-C",
            "overflow-checks=off",
        ])
        .arg(&rs)
        .arg("-o")
        .arg(&exe)
        .status()
    {
        Ok(s) if s.success() => {}
        _ => return None,
    }
    Some(
        Command::new(&exe)
            .status()
            .expect("run native")
            .code()
            .unwrap_or(-1) as u8,
    )
}

/// **Rust `alloc` / heap — `Vec` via a guest `#[global_allocator]`.** The headline for *real* Rust: a
/// `no_std` + `alloc` crate whose global allocator routes to the guest `malloc`/`free`. Run through the
/// powerbox (the on-ramp synthesizes `main` → `_start`, granting the `Memory` handle and the
/// `vm_map`-growing bump allocator), `Vec::push` grows the heap (alloc + `memcpy` + free) and the sum
/// is returned as the exit code — heap data structures from Rust, byte-identical to native `rustc`.
#[test]
fn rust_alloc_vec_via_global_allocator() {
    let items = r#"
extern crate alloc;
use alloc::vec::Vec;
use core::alloc::{GlobalAlloc, Layout};

extern "C" {
    fn malloc(size: usize) -> *mut u8;
    fn free(ptr: *mut u8);
}
struct Guest;
unsafe impl GlobalAlloc for Guest {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 { malloc(layout.size()) }
    unsafe fn dealloc(&self, ptr: *mut u8, _layout: Layout) { free(ptr); }
}
#[global_allocator]
static GA: Guest = Guest;

fn compute() -> i32 {
    let mut v: alloc::vec::Vec<i32> = Vec::new();
    let mut i = 0i32;
    while i < 64 {
        v.push(i.wrapping_mul(i)); // grows the heap several times
        i = i.wrapping_add(1);
    }
    let mut s: i32 = 0;
    for &x in v.iter() {
        s = s.wrapping_add(x);
    }
    s.rem_euclid(251) // a deterministic value in [0, 250] — fits the u8 exit code
}
"#;
    let (Some(svm), Some(native)) = (
        rust_alloc_onramp("rs_alloc", items),
        rust_alloc_native("rs_alloc", items),
    ) else {
        return;
    };
    assert_eq!(
        svm, native,
        "rust alloc/Vec: on-ramp exit {svm} vs native {native}"
    );
    // Pin the value so the differential can't pass vacuously: Σ i² for i in 0..64 = 85344; % 251 = 4.
    assert_eq!(svm, 4, "rust alloc/Vec: expected 4, got {svm}");
}

/// **Rust `Box` + `String` — a mini expression evaluator (the heap capstone).** A recursive-descent
/// parser over a byte slice builds a `Box`ed recursive AST (`enum Expr { Num, Add(Box,Box), … }` — the
/// canonical use of `Box`), `eval` walks it recursively, and `render` serializes it back into a
/// `String` (heap text via the guest allocator). The result + the rendered length is returned — `Box`,
/// `String`, recursive enums, slice parsing, and the panic paths (a malformed parse would trap) all at
/// once, byte-identical to native `rustc`. A tiny interpreter, right at home next to the guest-JIT demo.
#[test]
fn rust_box_string_expr_evaluator() {
    let items = r#"
extern crate alloc;
use alloc::boxed::Box;
use alloc::string::String;
use core::alloc::{GlobalAlloc, Layout};

extern "C" {
    fn malloc(size: usize) -> *mut u8;
    fn free(ptr: *mut u8);
}
struct Guest;
unsafe impl GlobalAlloc for Guest {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 { malloc(l.size()) }
    unsafe fn dealloc(&self, p: *mut u8, _l: Layout) { free(p); }
}
#[global_allocator]
static GA: Guest = Guest;

enum Expr {
    Num(i64),
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
}

fn eval(e: &Expr) -> i64 {
    match e {
        Expr::Num(n) => *n,
        Expr::Add(a, b) => eval(a).wrapping_add(eval(b)),
        Expr::Sub(a, b) => eval(a).wrapping_sub(eval(b)),
        Expr::Mul(a, b) => eval(a).wrapping_mul(eval(b)),
    }
}

fn render(e: &Expr, out: &mut String) {
    match e {
        Expr::Num(n) => {
            let mut v = *n;
            if v == 0 { out.push('0'); }
            let mut tmp = [0u8; 20];
            let mut k = 0;
            while v > 0 { tmp[k] = b'0' + (v % 10) as u8; v /= 10; k += 1; }
            while k > 0 { k -= 1; out.push(tmp[k] as char); }
        }
        Expr::Add(a, b) => { out.push('('); render(a, out); out.push('+'); render(b, out); out.push(')'); }
        Expr::Sub(a, b) => { out.push('('); render(a, out); out.push('-'); render(b, out); out.push(')'); }
        Expr::Mul(a, b) => { out.push('('); render(a, out); out.push('*'); render(b, out); out.push(')'); }
    }
}

struct Parser<'a> { s: &'a [u8], pos: usize }
impl<'a> Parser<'a> {
    fn peek(&self) -> u8 { if self.pos < self.s.len() { self.s[self.pos] } else { 0 } }
    fn bump(&mut self) -> u8 { let c = self.peek(); self.pos += 1; c }
    fn number(&mut self) -> Box<Expr> {
        let mut n: i64 = 0;
        while self.peek().is_ascii_digit() {
            n = n.wrapping_mul(10).wrapping_add((self.bump() - b'0') as i64);
        }
        Box::new(Expr::Num(n))
    }
    fn factor(&mut self) -> Box<Expr> {
        if self.peek() == b'(' {
            self.bump();
            let e = self.expr();
            self.bump(); // ')'
            e
        } else {
            self.number()
        }
    }
    fn term(&mut self) -> Box<Expr> {
        let mut left = self.factor();
        while self.peek() == b'*' {
            self.bump();
            let right = self.factor();
            left = Box::new(Expr::Mul(left, right));
        }
        left
    }
    fn expr(&mut self) -> Box<Expr> {
        let mut left = self.term();
        loop {
            match self.peek() {
                b'+' => { self.bump(); let r = self.term(); left = Box::new(Expr::Add(left, r)); }
                b'-' => { self.bump(); let r = self.term(); left = Box::new(Expr::Sub(left, r)); }
                _ => break,
            }
        }
        left
    }
}

fn compute() -> i32 {
    let input = b"2+3*4-(5-1)*2+10";          // = 2 + 12 - 8 + 10 = 16
    let mut p = Parser { s: input, pos: 0 };
    let ast = p.expr();
    let result = eval(&ast);
    let mut s = String::new();
    render(&ast, &mut s);                       // a fully-parenthesized rendering
    result.wrapping_add(s.len() as i64).rem_euclid(251) as i32
}
"#;
    let (Some(svm), Some(native)) = (
        rust_alloc_onramp("rs_expr", items),
        rust_alloc_native("rs_expr", items),
    ) else {
        return;
    };
    assert_eq!(
        svm, native,
        "expr evaluator: on-ramp {svm} vs native {native}"
    );
    // Pin it (non-vacuous): eval = 16, render = `(((2+(3*4))-((5-1)*2))+10)` (26 chars); 42 % 251 = 42.
    assert_eq!(svm, 42, "expr evaluator: expected 42, got {svm}");
}

// ============================================================================================
// SIMD — focused tests pinning specific `-O2` **auto-vectorized** shapes (§17/D58 `v128`). The
// C/C++/Rust breadth lanes now vectorize too (slices AN–AT), so the real corpus demos exercise SIMD
// end to end; these tests additionally pin each shape/op-class (conversions, rotate, shuffle, masks)
// to a known value for non-vacuity.
// ============================================================================================

/// Like [`compile_to_bc`] (auto-vectorization is enabled in both now), kept as the explicit SIMD
/// harness so a reduction loop's `<4 x i32>` + `llvm.vector.reduce.*` is pinned to a known value.
fn compile_to_bc_vectorized(name: &str, src: &str) -> Option<PathBuf> {
    let dir = std::env::temp_dir();
    let c = dir.join(format!("svm_llvm_{}_{}.c", std::process::id(), name));
    let bc = dir.join(format!("svm_llvm_simd_{}_{}.bc", std::process::id(), name));
    std::fs::write(&c, src).expect("write C source");
    let status = Command::new("clang")
        .args(["-O2", "-emit-llvm", "-c"])
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

/// Like [`compile_to_bc_vectorized`] but targeting **AVX2** (`-mavx2`), so the auto-vectorizer emits
/// wider-than-128-bit vectors (`<8 x i32>`, and `<16 x i32>` under interleave) — the exact shapes
/// the I2 legalization pass splits into `v128` chunks. The bitcode only *names* AVX vectors; the SVM
/// JIT still lowers each chunk to SSE2/NEON, so no AVX2 hardware is needed to run the result.
fn compile_to_bc_avx(name: &str, src: &str) -> Option<PathBuf> {
    let dir = std::env::temp_dir();
    let c = dir.join(format!("svm_llvm_{}_{}.c", std::process::id(), name));
    let bc = dir.join(format!("svm_llvm_simd_{}_{}.bc", std::process::id(), name));
    std::fs::write(&c, src).expect("write C source");
    let status = Command::new("clang")
        .args(["-O2", "-mavx2", "-emit-llvm", "-c"])
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

/// Run `run(seed)` (function 0) from **vectorized** bitcode on both backends and assert it equals a
/// native `cc` build's exit code (the on-ramp's SIMD lowering vs the scalar native result — they
/// compute the same value).
fn check_vectorized_vs_native(name: &str, src: &str, seed: i32) {
    let Some(bc) = compile_to_bc_vectorized(name, src) else {
        return;
    };
    check_simd_bc_vs_native(name, &bc, seed);
}

/// Like [`check_vectorized_vs_native`] but ingests **AVX2** auto-vectorized bitcode (wider-than-128
/// shapes), which the I2 legalization pass splits into `v128` chunks. The native oracle is a plain
/// scalar `cc` build of the same loop (gcc needs no `-mavx2`), so SVM-chunked == native-scalar.
fn check_avx_vs_native(name: &str, src: &str, seed: i32) {
    let Some(bc) = compile_to_bc_avx(name, src) else {
        return;
    };
    check_simd_bc_vs_native(name, &bc, seed);
}

/// Shared body: build the source's `.c` natively with `cc`, run it, then translate `bc`, verify, and
/// run `run(seed)` (function 0) on both backends — asserting interp == JIT == native exit code.
fn check_simd_bc_vs_native(name: &str, bc: &Path, seed: i32) {
    let exe =
        std::env::temp_dir().join(format!("svm_llvm_simdnat_{}_{}", std::process::id(), name));
    let c = std::env::temp_dir().join(format!("svm_llvm_{}_{}.c", std::process::id(), name));
    match Command::new("cc").arg(&c).arg("-o").arg(&exe).status() {
        Ok(s) if s.success() => {}
        _ => return,
    }
    let native = Command::new(&exe)
        .status()
        .expect("run native")
        .code()
        .unwrap() as u8;

    let t = svm_llvm::translate_bc_path(bc).expect("translate vectorized bitcode");
    let module = t.module;
    svm_verify::verify_module(&module).expect("verify translated IR");
    let full = vec![Value::I64(t.entry_sp as i64), Value::I32(seed)];
    let mut fuel = 100_000_000u64;
    let interp = svm_interp::run(&module, 0, &full, &mut fuel).expect("interp run");
    let slots: Vec<i64> = full.iter().map(to_slot).collect();
    let jit = match svm_jit::compile_and_run(&module, 0, &slots).expect("jit run") {
        JitOutcome::Returned(s) => Value::I32(s[0] as i32),
        other => panic!("{name}: unexpected JIT outcome {other:?}"),
    };
    assert_eq!(interp, vec![jit], "{name}: interp vs JIT");
    let svm = match jit {
        Value::I32(x) => x as u8,
        _ => panic!("expected i32"),
    };
    assert_eq!(svm, native, "{name}: svm={svm} vs native cc={native}");
}

/// **SIMD first light — an auto-vectorized integer reduction.** A `noinline` `sum` over an opaque
/// pointer vectorizes at `-O2` to an `<4 x i32>` accumulator + `llvm.vector.reduce.add.v4i32`; the
/// on-ramp ingests the vector lane add (`VIntBin i32x4`) and unrolls the reduce. `run(7)` fills a
/// global with `7 + i²` (i in 0..20) and sums it = 140 + 2470 = 2610 (exit `2610 & 0xff = 50`), vs
/// native — proving the on-ramp consumes real `-O2` vectorized output.
#[test]
fn simd_int_reduction_first_light() {
    let src = "int sum(const int *a, int n);\n\
        static int data[20];\n\
        int run(int seed) {\n\
        \x20 for (int i = 0; i < 20; i++) data[i] = seed + i * i;\n\
        \x20 return sum(data, 20);\n\
        }\n\
        __attribute__((noinline)) int sum(const int *a, int n) {\n\
        \x20 int s = 0;\n\
        \x20 for (int i = 0; i < n; i++) s += a[i];\n\
        \x20 return s;\n\
        }\n\
        int main(void) { return run(7); }\n";
    check_vectorized_vs_native("simd_reduce", src, 7);
}

/// SIMD — an auto-vectorized **max reduction** (`llvm.vector.reduce.smax.v4i32`, + any `<4 x i32>`
/// lane `smax`). `run(seed)` fills a global with a wave of values and takes the max, vs native —
/// exercising the min/max reduce fold (`cmp`+`select`).
#[test]
fn simd_int_max_reduction() {
    let src = "int amax(const int *a, int n);\n\
        static int data[24];\n\
        int run(int seed) {\n\
        \x20 for (int i = 0; i < 24; i++) data[i] = ((i * 7 + seed) & 31) - 13;\n\
        \x20 return amax(data, 24) + 20;\n\
        }\n\
        __attribute__((noinline)) int amax(const int *a, int n) {\n\
        \x20 int m = a[0];\n\
        \x20 for (int i = 1; i < n; i++) if (a[i] > m) m = a[i];\n\
        \x20 return m;\n\
        }\n\
        int main(void) { return run(3); }\n";
    check_vectorized_vs_native("simd_max", src, 3);
}

/// **Capstone — ingesting *real* `-O2 -mavx2` auto-vectorized output.** The motivating I2 case: the
/// same reduction loop, vectorized for AVX2, emits a wider-than-128 `<8 x i32>` accumulator +
/// `llvm.vector.reduce.add.v8i32` (and `<16 x i32>` under interleave). The legalization pass splits
/// each into `v128` chunks, so the on-ramp now ingests it (previously a fail-closed `Unsupported`),
/// running byte-identical to the native scalar oracle on both backends.
#[test]
fn simd_autovec_avx2_reduction() {
    let src = "int sum(const int *a, int n);\n\
        static int data[64];\n\
        int run(int seed) {\n\
        \x20 for (int i = 0; i < 64; i++) data[i] = seed + i * i;\n\
        \x20 return sum(data, 64);\n\
        }\n\
        __attribute__((noinline)) int sum(const int *a, int n) {\n\
        \x20 int s = 0;\n\
        \x20 for (int i = 0; i < n; i++) s += a[i];\n\
        \x20 return s;\n\
        }\n\
        int main(void) { return run(7); }\n";
    check_avx_vs_native("simd_avx2_reduce", src, 7);
}

/// `-O2 -mavx2` auto-vectorized **elementwise** kernel (`c[i] = a[i]*b[i] + a[i]`) — wide `<8 x i32>`
/// lane multiply/add across `v128` chunks (no horizontal reduce), vs the native scalar oracle.
#[test]
fn simd_autovec_avx2_elementwise() {
    let src = "void mul(const int *a, const int *b, int *c, int n);\n\
        static int A[64], B[64], C[64];\n\
        int run(int seed) {\n\
        \x20 for (int i = 0; i < 64; i++) { A[i] = seed + i; B[i] = i * 2 + 1; }\n\
        \x20 mul(A, B, C, 64);\n\
        \x20 int s = 0;\n\
        \x20 for (int i = 0; i < 64; i++) s += C[i];\n\
        \x20 return s;\n\
        }\n\
        __attribute__((noinline)) void mul(const int *a, const int *b, int *c, int n) {\n\
        \x20 for (int i = 0; i < n; i++) c[i] = a[i] * b[i] + a[i];\n\
        }\n\
        int main(void) { return run(4); }\n";
    check_avx_vs_native("simd_avx2_elem", src, 4);
}

// ============================================================================================
// SIMD — the **other 128-bit lane shapes** (`i8x16`/`i16x8`/`i64x2`/`f64x2`), beyond the original
// `i32x4`/`f32x4`. These use explicit `vector_size(16)` types compiled with vectorization *off*
// (`compile_to_bc` / `check_vs_native`), so the on-ramp sees exactly the declared 128-bit shape —
// no auto-vectorizer widening. A `noinline` helper takes opaque pointers so clang must emit real
// `<N x T>` loads/ops, not scalarize them. Each `vec128_shape` op (load/store, `VIntBin`,
// `VFloatBin`, `ExtractLane`) is exercised against the native oracle on both backends.
// ============================================================================================

/// `<2 x i64>` lane multiply + add + per-lane extract (`i64x2` `VIntBin` Mul/Add, `ExtractLane`).
/// `run(7)`: a={7,9}, b={3,5}, c=a*b+b={24,50}; c[0]+c[1]=74.
#[test]
fn simd_i64x2_mul_add_extract() {
    let src = "long long vdot(const long long *A, const long long *B);\n\
        static long long A[2], B[2];\n\
        int run(int seed) {\n\
        \x20 A[0] = seed; A[1] = seed + 2; B[0] = 3; B[1] = 5;\n\
        \x20 return (int)(vdot(A, B) & 0xff);\n\
        }\n\
        typedef long long i64x2 __attribute__((vector_size(16)));\n\
        __attribute__((noinline)) long long vdot(const long long *A, const long long *B) {\n\
        \x20 i64x2 a = *(const i64x2 *)A;\n\
        \x20 i64x2 b = *(const i64x2 *)B;\n\
        \x20 i64x2 c = a * b + b;\n\
        \x20 return c[0] + c[1];\n\
        }\n\
        int main(void) { return run(7); }\n";
    check_vs_native("simd_i64x2", src, 7);
}

/// `<16 x i8>` lane add through a `v128` load → add → store (`i8x16` load/`VIntBin` Add/store).
/// Two distinct input arrays so `a + b` stays a real lane add (a self-add would strength-reduce to
/// a vector shift). The tail-sum reads the stored bytes back from memory and folds them (auto-
/// vectorized like the rest of the suite — the on-ramp ingests its `zext`/reduce lowering too).
#[test]
fn simd_i8x16_add_load_store() {
    let src = "void vadd(const unsigned char *P, const unsigned char *Q, unsigned char *O);\n\
        static unsigned char D[16], F[16], E[16];\n\
        int run(int seed) {\n\
        \x20 for (int i = 0; i < 16; i++) { D[i] = (unsigned char)(seed + i); F[i] = (unsigned char)(3 * i + 1); }\n\
        \x20 vadd(D, F, E);\n\
        \x20 int s = 0;\n\
        \x20 for (int i = 0; i < 16; i++) s += E[i];\n\
        \x20 return s & 0xff;\n\
        }\n\
        typedef unsigned char u8x16 __attribute__((vector_size(16)));\n\
        __attribute__((noinline)) void vadd(const unsigned char *P, const unsigned char *Q, unsigned char *O) {\n\
        \x20 u8x16 a = *(const u8x16 *)P;\n\
        \x20 u8x16 b = *(const u8x16 *)Q;\n\
        \x20 *(u8x16 *)O = a + b;\n\
        }\n\
        int main(void) { return run(5); }\n";
    check_vs_native("simd_i8x16", src, 5);
}

/// `<8 x i16>` lane multiply through a `v128` load → mul → store (`i16x8` `VIntBin` Mul). Wraps
/// mod 2^16 per lane; the scalar read-back truncates to the lane width.
#[test]
fn simd_i16x8_mul_load_store() {
    let src = "void vmul(const unsigned short *P, const unsigned short *Q, unsigned short *O);\n\
        static unsigned short D[8], F[8], E[8];\n\
        int run(int seed) {\n\
        \x20 for (int i = 0; i < 8; i++) { D[i] = (unsigned short)(seed + i); F[i] = (unsigned short)(i + 1); }\n\
        \x20 vmul(D, F, E);\n\
        \x20 int s = 0;\n\
        \x20 for (int i = 0; i < 8; i++) s += E[i];\n\
        \x20 return s & 0xff;\n\
        }\n\
        typedef unsigned short u16x8 __attribute__((vector_size(16)));\n\
        __attribute__((noinline)) void vmul(const unsigned short *P, const unsigned short *Q, unsigned short *O) {\n\
        \x20 u16x8 a = *(const u16x8 *)P;\n\
        \x20 u16x8 b = *(const u16x8 *)Q;\n\
        \x20 *(u16x8 *)O = a * b;\n\
        }\n\
        int main(void) { return run(2); }\n";
    check_vs_native("simd_i16x8", src, 2);
}

/// `<2 x double>` lane multiply + add (`f64x2` `VFloatBin` Mul/Add). Finite values only, so the
/// per-lane-NaN differential caveat (§17) doesn't apply; the derived integer result is exact.
#[test]
fn simd_f64x2_mul_add() {
    let src = "double vfma(const double *A, const double *B);\n\
        static double A[2], B[2];\n\
        int run(int seed) {\n\
        \x20 A[0] = seed; A[1] = seed + 1; B[0] = 2.0; B[1] = 3.0;\n\
        \x20 return (int)vfma(A, B);\n\
        }\n\
        typedef double f64x2 __attribute__((vector_size(16)));\n\
        __attribute__((noinline)) double vfma(const double *A, const double *B) {\n\
        \x20 f64x2 a = *(const f64x2 *)A;\n\
        \x20 f64x2 b = *(const f64x2 *)B;\n\
        \x20 f64x2 c = a * b + b;\n\
        \x20 return c[0] + c[1];\n\
        }\n\
        int main(void) { return run(4); }\n";
    check_vs_native("simd_f64x2", src, 4);
}

// ============================================================================================
// SIMD — **wider-than-128-bit** vectors legalized to fixed-128 `v128` chunks + a scalar tail (I2
// fix-sketch step 1). These use explicit `vector_size` types (compiled with vectorization off) so
// the on-ramp sees an exact wide shape, and `noinline` helpers with opaque pointers force real wide
// loads/ops — all *within a single block* (no control flow), exercising the in-block splitter. The
// `<8 x i32>` / `<4 x i64>` cases split into 2 clean `v128` chunks (no tail); the `<8 x i8>` case is
// sub-128 and fully scalarized into the tail (0 chunks, 8 lane scalars).
// ============================================================================================

/// `<8 x i32>` (256-bit) elementwise add → split into **2 `i32x4` chunks** (load/`VIntBin`/store).
#[test]
fn simd_i32x8_add_store() {
    let src = "void vadd8(const int *P, const int *Q, int *O);\n\
        static int D[8], F[8], E[8];\n\
        int run(int seed) {\n\
        \x20 for (int i = 0; i < 8; i++) { D[i] = seed + i; F[i] = 2 * i + 1; }\n\
        \x20 vadd8(D, F, E);\n\
        \x20 int s = 0;\n\
        \x20 for (int i = 0; i < 8; i++) s += E[i];\n\
        \x20 return s & 0xff;\n\
        }\n\
        typedef int v8si __attribute__((vector_size(32)));\n\
        __attribute__((noinline)) void vadd8(const int *P, const int *Q, int *O) {\n\
        \x20 v8si a = *(const v8si *)P, b = *(const v8si *)Q;\n\
        \x20 *(v8si *)O = a + b;\n\
        }\n\
        int main(void) { return run(5); }\n";
    check_vs_native("simd_i32x8", src, 5);
}

/// `<8 x i32>` horizontal `llvm.vector.reduce.add.v8i32` (via `__builtin_reduce_add`) over 2 chunks
/// — the wide-reduce fold (extract every lane of both chunks, sum). `__builtin_reduce_*` is a clang
/// builtin (`cc`/gcc lacks it), so this uses `check` with a computed expected value (interp == JIT)
/// rather than the native oracle. `run(3)`: Σ(3 + i²) for i in 0..8 = 24 + 140 = 164.
#[test]
fn simd_i32x8_reduce_add() {
    let src = "int vred8(const int *P);\n\
        static int D[8];\n\
        int run(int seed) {\n\
        \x20 for (int i = 0; i < 8; i++) D[i] = seed + i * i;\n\
        \x20 return vred8(D);\n\
        }\n\
        typedef int v8si __attribute__((vector_size(32)));\n\
        __attribute__((noinline)) int vred8(const int *P) {\n\
        \x20 v8si a = *(const v8si *)P;\n\
        \x20 return __builtin_reduce_add(a);\n\
        }\n\
        int main(void) { return run(3); }\n";
    check("simd_i32x8_red", src, &[Value::I32(3)], &[Value::I32(164)]);
}

/// `<4 x i64>` (256-bit) lane multiply + horizontal `reduce.add.v4i64` over **2 `i64x2` chunks**.
/// `run(2)`: Σ (2+i)·(i+1) for i in 0..4 = 2 + 6 + 12 + 20 = 40.
#[test]
fn simd_i64x4_mul_reduce() {
    let src = "long long vred4(const long long *P, const long long *Q);\n\
        static long long D[4], F[4];\n\
        int run(int seed) {\n\
        \x20 for (int i = 0; i < 4; i++) { D[i] = seed + i; F[i] = i + 1; }\n\
        \x20 return (int)vred4(D, F);\n\
        }\n\
        typedef long long v4di __attribute__((vector_size(32)));\n\
        __attribute__((noinline)) long long vred4(const long long *P, const long long *Q) {\n\
        \x20 v4di a = *(const v4di *)P, b = *(const v4di *)Q;\n\
        \x20 v4di c = a * b;\n\
        \x20 return __builtin_reduce_add(c);\n\
        }\n\
        int main(void) { return run(2); }\n";
    check("simd_i64x4", src, &[Value::I32(2)], &[Value::I32(40)]);
}

/// `<8 x i8>` (64-bit, sub-128) elementwise add — **fully scalarized into the tail** (0 chunks, 8
/// lane scalars: a 16-byte `v128.load` would overrun its 8-byte image, so each lane is a byte op).
#[test]
fn simd_i8x8_add_tail() {
    let src = "void vadd8b(const unsigned char *P, const unsigned char *Q, unsigned char *O);\n\
        static unsigned char D[8], F[8], E[8];\n\
        int run(int seed) {\n\
        \x20 for (int i = 0; i < 8; i++) { D[i] = (unsigned char)(seed + i); F[i] = (unsigned char)(3 * i + 1); }\n\
        \x20 vadd8b(D, F, E);\n\
        \x20 int s = 0;\n\
        \x20 for (int i = 0; i < 8; i++) s += E[i];\n\
        \x20 return s & 0xff;\n\
        }\n\
        typedef unsigned char v8qi __attribute__((vector_size(8)));\n\
        __attribute__((noinline)) void vadd8b(const unsigned char *P, const unsigned char *Q, unsigned char *O) {\n\
        \x20 v8qi a = *(const v8qi *)P, b = *(const v8qi *)Q;\n\
        \x20 *(v8qi *)O = a + b;\n\
        }\n\
        int main(void) { return run(5); }\n";
    check_vs_native("simd_i8x8", src, 5);
}

/// **Cross-block wide vector.** A reduction loop carries an `<8 x i32>` accumulator across the loop
/// backedge as a *wide phi* — the case the legalization fan-out exists for: one wide LLVM value
/// becomes `K` block params (2 `i32x4` chunks here) supplied as `K` branch args on every edge into
/// the loop header. `n` is opaque (a `noinline` param), so the loop is a real backedge, not unrolled.
/// `run(3)`: Σ(3 + i) for i in 0..16 = 48 + 120 = 168.
#[test]
fn simd_i32x8_loop_accumulator() {
    let src = "int vsum(const int *P, int n);\n\
        static int D[16];\n\
        int run(int seed) {\n\
        \x20 for (int i = 0; i < 16; i++) D[i] = seed + i;\n\
        \x20 return vsum(D, 16);\n\
        }\n\
        typedef int v8si __attribute__((vector_size(32)));\n\
        __attribute__((noinline)) int vsum(const int *P, int n) {\n\
        \x20 v8si acc = {0, 0, 0, 0, 0, 0, 0, 0};\n\
        \x20 for (int i = 0; i < n; i += 8) {\n\
        \x20   v8si a = *(const v8si *)(P + i);\n\
        \x20   acc += a;\n\
        \x20 }\n\
        \x20 return __builtin_reduce_add(acc);\n\
        }\n\
        int main(void) { return run(3); }\n";
    check("simd_i32x8_loop", src, &[Value::I32(3)], &[Value::I32(168)]);
}

// ── Auto-vectorized lane-wise integer conversions (`zext`/`sext`/`trunc`) ──────────────────────
// svm-ir has no vector-convert op, so the on-ramp scalarizes a `<N x iA> → <N x iB>` widen/narrow:
// explode the source to lane scalars, convert each in its `i32`/`i64` container, repack into the
// destination representation (packed-`i64` `<2 x i32>`, a single `v128`, or legalized wide chunks +
// tail). These `check_vectorized_vs_native` tests ingest **real `clang -O2`** output that emits the
// conversion (verified to vectorize), covering every source↔dest representation pairing: a wide-tail
// source widening to a `v128` (`zext <4 x i8> → <4 x i32>`), a `v128` narrowing to a wide tail
// (`trunc <8 x i32> → <8 x i8>`), a wide source narrowing to a `v128` (`trunc <2 x i64> → <2 x i32>`),
// and a packed-`i64` `<2 x i32>` widening to a `v128` (`sext <2 x i32> → <2 x i64>`).

/// `zext <4 x i8> → <4 x i32>` (a `u8 → int` widening store loop: wide-tail source widened to a
/// `v128`), plus the seeding store's `trunc <16 x i32> → <16 x i8>`. A fixed-index read-back (no
/// horizontal reduction) keeps the conversion the only vector op under test.
#[test]
fn simd_conv_zext_u8_to_i32() {
    let src = "static unsigned char b[128]; static int out[128]; \
               int run(int seed){ for(int i=0;i<128;i++) b[i]=(unsigned char)(seed+i); \
               for(int i=0;i<128;i++) out[i]=b[i]; \
               return (out[0]+out[63]+out[127]) & 0xff; } \
               int main(void){ return run(7); }";
    check_vectorized_vs_native("simd_conv_zext", src, 7);
}

/// `sext <2 x i32> → <2 x i64>` — a sign-extending `int → long long` store loop (the packed-`i64`
/// `<2 x i32>` representation widened to an `i64x2` `v128`), the exact shape `heapgrow` emits.
#[test]
fn simd_conv_sext_i32_to_i64() {
    let src = "int run(int seed){ int in[64]; long long out[64]; \
               for(int i=0;i<64;i++) in[i]=seed-i; \
               for(int i=0;i<64;i++) out[i]=(long long)in[i]; \
               long long s=0; for(int i=0;i<64;i++) s+=out[i]; return (int)(s & 0xff); } \
               int main(void){ return run(9); }";
    check_vectorized_vs_native("simd_conv_sext", src, 9);
}

/// `trunc <2 x i64> → <2 x i32>` — a narrowing `long long → int` store loop (a wide `i64x2` source
/// narrowed to a `v128`), the shape `revsum`/`heapgrow` emit.
#[test]
fn simd_conv_trunc_i64_to_i32() {
    let src = "int run(int seed){ long long in[64]; int out[64]; \
               for(int i=0;i<64;i++) in[i]=(long long)seed*1000+i; \
               for(int i=0;i<64;i++) out[i]=(int)in[i]; \
               int s=0; for(int i=0;i<64;i++) s+=out[i]; return s & 0xff; } \
               int main(void){ return run(4); }";
    check_vectorized_vs_native("simd_conv_trunc64", src, 4);
}

/// `trunc <8 x i16> → <8 x i8>` and `trunc <8 x i32> → <8 x i16>` (a `u16 → u8` narrowing store
/// loop: `v128`/wide sources narrowed to a wide all-tail / `v128` destination). Fixed-index read-back
/// keeps the conversions the only vector ops under test.
#[test]
fn simd_conv_trunc_to_u8() {
    let src = "static unsigned short in[128]; static unsigned char out[128]; \
               int run(int seed){ for(int i=0;i<128;i++) in[i]=(unsigned short)(seed*7+i); \
               for(int i=0;i<128;i++) out[i]=(unsigned char)in[i]; \
               return (out[0]+out[63]+out[127]) & 0xff; } \
               int main(void){ return run(5); }";
    check_vectorized_vs_native("simd_conv_trunc8", src, 5);
}

/// **Auto-vectorized vector rotate (`llvm.fshl.v4i32`).** A `(x<<13)|(x>>19)` rotate loop, which
/// `clang -O2` recognizes as a funnel shift and vectorizes to `llvm.fshl.v4i32`. svm-ir's `VShift`
/// takes only a scalar amount, so the on-ramp scalarizes the rotate idiom (`a == b`) lane-wise — each
/// lane a scalar `Rotl`/`Rotr` (mask-mod-width, no shift-by-width edge) — then repacks the `v128`.
/// This is the shape xxHash's `XXH32_round` emits. Fixed-index read-back avoids a vector reduction.
#[test]
fn simd_vector_rotate_fshl() {
    let src = "static unsigned int a[64]; static unsigned int out[64]; \
               int run(int seed){ for(int i=0;i<64;i++) a[i]=(unsigned)(seed*2654435761u + i); \
               for(int i=0;i<64;i++){ unsigned x=a[i]; out[i]=(x<<13)|(x>>19); } \
               return (int)((out[0]^out[31]^out[63]) & 0xff); } \
               int main(void){ return run(7); }";
    check_vectorized_vs_native("simd_rotate", src, 7);
}

/// **Wide non-splat shuffle (`shufflevector <8 x i8>` byte-reverse `<7,6,…,0>`).** A sub-128 vector
/// (8 bytes → 0 chunks, 8 scalar tail lanes) permuted by a general constant mask — the legalized
/// analog of the single-`v128` `Inst::Shuffle` path. The on-ramp explodes both operands' lanes,
/// gathers per the mask (each result lane picks from the `a ++ b` concat), and repacks. This is the
/// shape the `async_io` demo's byte-reversal emits; here forced via `__builtin_shufflevector`.
#[test]
fn simd_wide_shuffle_reverse() {
    let src = "typedef unsigned char v8qi __attribute__((vector_size(8))); \
               void rev8(const unsigned char *P, unsigned char *O); \
               static unsigned char a[8], out[8]; \
               int run(int seed){ for(int i=0;i<8;i++) a[i]=(unsigned char)(seed+i*3); \
               rev8(a, out); \
               return (out[0]*1 + out[3]*5 + out[7]*7) & 0xff; } \
               __attribute__((noinline)) void rev8(const unsigned char *P, unsigned char *O){ \
               v8qi v = *(const v8qi*)P; \
               v8qi r = __builtin_shufflevector(v, v, 7,6,5,4,3,2,1,0); \
               *(v8qi*)O = r; } \
               int main(void){ return run(4); }";
    check_vectorized_vs_native("simd_wide_shuffle", src, 4);
}

/// **`<N x i1>` boolean mask — vector `icmp` + `select`.** A `(a[i]==b[i]) ? a[i] : b[i]+1` loop,
/// which `clang -O2` vectorizes to `icmp eq <4 x i32>` (producing a `<4 x i1>` mask) feeding
/// `select <4 x i1>, …`. svm-ir has no first-class `<N x i1>`, so the on-ramp holds the mask lane-wise
/// (per-lane scalar compare) and lowers the `select` as a per-lane scalar `select` over the exploded
/// data, repacked. The same machinery (plus mask `extractelement`) carries the real `crc32` demo.
#[test]
fn simd_mask_icmp_select() {
    let src = "static int a[64], b[64], out[64]; \
               int run(int seed){ for(int i=0;i<64;i++){ a[i]=seed+i; b[i]=seed*2-i; } \
               for(int i=0;i<64;i++) out[i] = (a[i]==b[i]) ? a[i] : (b[i]+1); \
               return (out[0]+out[15]+out[31]+out[63]) & 0xff; } \
               int main(void){ return run(8); }";
    check_vectorized_vs_native("simd_mask_select", src, 8);
}

/// **Rust capstone — a `jsmn`-style JSON tokenizer (a real `no_std` program).** The analog of the C
/// corpus's `jsmn` demo, in Rust: scan a JSON document (`&[u8]`) into a heap `Vec` of typed tokens
/// (`enum Kind { Obj, Arr, Str, Prim }` + span), handling strings (with `\`-escapes), whitespace, and
/// bare primitives, then fold a deterministic digest over the tokens. Exercises enums, `Vec<struct>`
/// (heap, growth via the guest allocator), `&[u8]` scanning, `match` on bytes, and an enum-to-int
/// cast — a recognizable real Rust library end to end, byte-identical to native `rustc`.
#[test]
fn rust_json_tokenizer_capstone() {
    let items = r##"
extern crate alloc;
use alloc::vec::Vec;
use core::alloc::{GlobalAlloc, Layout};

extern "C" {
    fn malloc(size: usize) -> *mut u8;
    fn free(ptr: *mut u8);
}
struct Guest;
unsafe impl GlobalAlloc for Guest {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 { malloc(l.size()) }
    unsafe fn dealloc(&self, p: *mut u8, _l: Layout) { free(p); }
}
#[global_allocator]
static GA: Guest = Guest;

#[derive(Clone, Copy)]
enum Kind { Obj, Arr, Str, Prim }

#[derive(Clone, Copy)]
struct Tok { kind: Kind, start: usize, end: usize }

fn is_ws(c: u8) -> bool { matches!(c, b' ' | b'\t' | b'\n' | b'\r') }

fn tokenize(js: &[u8]) -> Vec<Tok> {
    let mut toks: Vec<Tok> = Vec::new();
    let mut i = 0usize;
    while i < js.len() {
        let c = js[i];
        if c == b'{' {
            toks.push(Tok { kind: Kind::Obj, start: i, end: i });
            i += 1;
        } else if c == b'[' {
            toks.push(Tok { kind: Kind::Arr, start: i, end: i });
            i += 1;
        } else if c == b'"' {
            let start = i + 1;
            i += 1;
            while i < js.len() && js[i] != b'"' {
                if js[i] == b'\\' { i += 1; } // skip the escaped char
                i += 1;
            }
            toks.push(Tok { kind: Kind::Str, start, end: i });
            i += 1; // closing quote
        } else if is_ws(c) || c == b':' || c == b',' || c == b'}' || c == b']' {
            i += 1;
        } else {
            let start = i;
            while i < js.len()
                && !is_ws(js[i])
                && js[i] != b','
                && js[i] != b'}'
                && js[i] != b']'
            {
                i += 1;
            }
            toks.push(Tok { kind: Kind::Prim, start, end: i });
        }
    }
    toks
}

fn compute() -> i32 {
    let doc = br#"{ "name": "v\"m", "tags": ["a", "b", "c"], "n": 42, "ok": true, "x": null }"#;
    let toks = tokenize(doc);
    let mut acc: i64 = toks.len() as i64;
    for t in toks.iter() {
        acc = acc
            .wrapping_mul(31)
            .wrapping_add(t.kind as i64)
            .wrapping_add(t.end.wrapping_sub(t.start) as i64);
    }
    acc.rem_euclid(251) as i32
}
"##;
    let (Some(svm), Some(native)) = (
        rust_alloc_onramp("rs_json", items),
        rust_alloc_native("rs_json", items),
    ) else {
        return;
    };
    assert_eq!(
        svm, native,
        "json tokenizer: on-ramp {svm} vs native {native}"
    );
    // Pin it (non-vacuous): 14 tokens over the doc, folded digest % 251 = 135.
    assert_eq!(svm, 135, "json tokenizer: expected digest 135, got {svm}");
}

/// §6 lexical-scope ingest from LLVM DI: an inner-block redeclaration (`DILexicalBlock`) is scoped
/// to its block (decl line → the block's last instruction line, since `DILexicalBlock` has no end
/// line), so reading `x` resolves to the in-scope shadow at the stopped pc — the LLVM producer
/// driving the same shadowing resolution as chibicc/wasm.
#[test]
fn llvm_o0_resolves_shadowed_locals_by_lexical_scope() {
    use svm_interp::{Inspector, IrPc, Stop, VarValue};
    use svm_ir::VarLoc;

    let src = "\
int f(int n) {
  int x = n + 1;
  {
    int x = n + 100;
    n = n + x;
  }
  return x + n;
}
";
    let Some(bc) = compile_to_bc_o0g("shadow", src) else {
        return;
    };
    let t = svm_llvm::translate_bc_path(&bc).expect("translate");
    svm_verify::verify_module(&t.module).expect("verify");
    let di = t.module.debug_info.as_ref().expect("debug info");

    // Two `x` vars; exactly one carries a lexical scope (the inner block), the other function-wide.
    let xs: Vec<_> = di.vars.iter().filter(|v| v.name == "x").collect();
    assert_eq!(xs.len(), 2, "both shadows ingested");
    for x in &xs {
        assert!(
            matches!(x.loc, VarLoc::Window { .. }),
            "x is a -O0 Window var"
        );
    }
    let scoped: Vec<_> = xs.iter().filter_map(|v| v.scope).collect();
    assert_eq!(
        scoped.len(),
        1,
        "one inner-block scope; the outer is function-wide"
    );
    let (start, _end) = scoped[0];
    assert_eq!(start, 4, "inner x scope starts at its declaration line");

    // Runtime: read `x` resolves the right shadow at the stopped pc.
    let pc_for_line = |line: u32| {
        let l = di.locs.iter().find(|l| l.line == line)?;
        Some(IrPc {
            module: 0,
            func: l.func,
            block: l.block as usize,
            inst: l.inst as usize,
        })
    };
    let read_x_at = |line: u32| {
        let mut insp = Inspector::attach(
            &t.module,
            0,
            &[Value::I64(t.entry_sp as i64), Value::I32(5)],
            5_000_000,
        );
        insp.set_breakpoint(pc_for_line(line).unwrap_or_else(|| panic!("no pc for line {line}")));
        assert!(
            matches!(insp.run_until_stop(), Stop::Break { .. }),
            "stop at line {line}"
        );
        match insp.read_var(0, "x", 4) {
            Some(VarValue::Bytes(b)) => i32::from_le_bytes(b[..4].try_into().unwrap()),
            other => panic!("expected window bytes for x, got {other:?}"),
        }
    };
    // Line 5 = `n = n + x` (inside the block): inner x = n + 100 = 105.
    assert_eq!(read_x_at(5), 105, "inner shadow resolved inside the block");
    // Line 7 = `return x + n` (after the block): outer x = n + 1 = 6.
    assert_eq!(read_x_at(7), 6, "outer x resolved after the block");
}
