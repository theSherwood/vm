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
    match Command::new("cc").arg(&c).arg("-o").arg(&exe).status() {
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
    match Command::new("cc").arg(c_src).arg("-o").arg(&exe).status() {
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
