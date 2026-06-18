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
        .args([
            "-O2",
            "-emit-llvm",
            "-c",
            "-fno-vectorize",
            "-fno-slp-vectorize",
        ])
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
int  __vm_fiber_new(long (*f)(long), void *stack);
long __vm_fiber_resume(int k, long arg, int *done);
long __vm_fiber_suspend(long value);
long counter(long start);
static char stack0[8192];
int driver(void) {
  int k = __vm_fiber_new(counter, stack0);
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
long __vm_gc_roots(long heap_lo, long heap_hi, void *buf, long cap);
static long out[64];
int f(void) {
  /* a live, in-range candidate pointer (into `out` itself) the conservative scan may see */
  volatile long *root = out;
  long n = __vm_gc_roots(0, 1 << 20, out, 64);
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
/// keeps EH/RTTI out (the §18 stance), `-O2` runs mem2reg/SROA, `-fno-*-vectorize` keeps SIMD out.
/// Returns `None` (skip) if `clang++` is unavailable.
fn compile_cpp_to_bc(name: &str, src: &str) -> Option<PathBuf> {
    let dir = std::env::temp_dir();
    let cc = dir.join(format!("svm_llvm_{}_{}.cpp", std::process::id(), name));
    let bc = dir.join(format!("svm_llvm_cpp_{}_{}.bc", std::process::id(), name));
    std::fs::write(&cc, src).expect("write C++ source");
    let status = Command::new("clang++")
        .args([
            "-O2",
            "-emit-llvm",
            "-c",
            "-fno-exceptions",
            "-fno-rtti",
            "-fno-vectorize",
            "-fno-slp-vectorize",
        ])
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
            "--edition",
            "2021",
            "--emit=llvm-bc",
            "--crate-type=lib",
            "-C",
            "panic=abort",
            "-C",
            "opt-level=2",
            "-C",
            "overflow-checks=off",
            // Keep SIMD out of the MVP (matching the C/C++ lanes' `-fno-*-vectorize`): `-O2`
            // auto-vectorizes reductions/loops into `<N x iM>` + horizontal reduces, which is a
            // separate §17/D58 concern. The native oracle keeps vectorizing — an integer reduction is
            // associative, so scalar (on-ramp) and vectorized (native) agree.
            "-C",
            "llvm-args=-vectorize-loops=false",
            "-C",
            "llvm-args=-vectorize-slp=false",
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
// SIMD — ingesting `-O2` **auto-vectorized** output (§17/D58 `v128`). The C/C++/Rust breadth lanes
// disable vectorization (`-fno-*-vectorize`); these tests *enable* it, so the on-ramp consumes the
// `<4 x i32>` lane ops + horizontal `llvm.vector.reduce.*` a real `-O2` reduction loop emits.
// ============================================================================================

/// Like [`compile_to_bc`] but with **auto-vectorization enabled** (no `-fno-*-vectorize`), so a
/// reduction loop lowers to `<4 x i32>` SIMD + `llvm.vector.reduce.*`.
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

/// Run `run(seed)` (function 0) from **vectorized** bitcode on both backends and assert it equals a
/// native `cc` build's exit code (the on-ramp's SIMD lowering vs the scalar native result — they
/// compute the same value).
fn check_vectorized_vs_native(name: &str, src: &str, seed: i32) {
    let Some(bc) = compile_to_bc_vectorized(name, src) else {
        return;
    };
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

    let t = svm_llvm::translate_bc_path(&bc).expect("translate vectorized bitcode");
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
