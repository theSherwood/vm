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

/// `svm_jit::compile` (compile once) → reuse `CompiledModule::run` returns the same result as the
/// one-shot `compile_and_run`, across multiple inputs. Guards the compile-once API the cross-engine
/// bench relies on for honest JIT timing (its loop must carry no per-call Cranelift codegen).
#[test]
fn jit_compile_once_run_many() {
    let Some(bc) = compile_to_bc("compile_once", "int run(int x){ return x * x + 1; }") else {
        return;
    };
    let t = svm_llvm::translate_bc_path(&bc).expect("translate");
    svm_verify::verify_module(&t.module).expect("verify");
    let e = t
        .exports
        .iter()
        .find(|(n, _)| n == "run")
        .map(|x| x.1)
        .expect("run export");
    let sp = t.entry_sp as i64;
    let mut cm = svm_jit::compile(&t.module, e).expect("compile once");
    let val = |o: JitOutcome| match o {
        JitOutcome::Returned(v) => v[0] as i32,
        other => panic!("unexpected outcome {other:?}"),
    };
    for x in [3i64, 10, -4] {
        let once = val(cm.run(&[sp, x], None, None, None).expect("run").0);
        let one_shot = val(svm_jit::compile_and_run(&t.module, e, &[sp, x]).expect("one-shot"));
        assert_eq!(once, one_shot, "compile-once vs one-shot at x={x}");
        assert_eq!(once, (x * x + 1) as i32, "result at x={x}");
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

/// A **struct `phi`** — a small by-value struct (`{i64, i64}`) carried across a loop backedge,
/// zero-initialized on the entry edge and rebuilt by `insertvalue` on the backedge, read by
/// `extractvalue`. The aggregate side-table is block-local, so this exercises the **cross-block
/// aggregate threading** (per-field block-param fan-out + `branch_args` field materialization) that
/// makes Embench `wikisort`'s `MakeRange` result translate. Authored as IR because clang's SROA
/// scalarizes such a carried struct into per-field `i64` φs in C, so a struct φ can't be coaxed from a
/// small C kernel. `run(n)` accumulates `a=Σi`, `b=Σ2i` over `i∈[0,n)`, returns `a+b = 3·n·(n−1)/2`.
#[test]
fn struct_phi_cross_block() {
    let ir = "\
target datalayout = \"e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128\"\n\
target triple = \"x86_64-pc-linux-gnu\"\n\
define i64 @run(i64 %n) {\n\
entry:\n\
  br label %loop\n\
loop:\n\
  %i = phi i64 [ 0, %entry ], [ %inext, %body ]\n\
  %acc = phi { i64, i64 } [ zeroinitializer, %entry ], [ %accnext, %body ]\n\
  %cmp = icmp slt i64 %i, %n\n\
  br i1 %cmp, label %body, label %exit\n\
body:\n\
  %a = extractvalue { i64, i64 } %acc, 0\n\
  %b = extractvalue { i64, i64 } %acc, 1\n\
  %anew = add i64 %a, %i\n\
  %ti = mul i64 %i, 2\n\
  %bnew = add i64 %b, %ti\n\
  %t = insertvalue { i64, i64 } undef, i64 %anew, 0\n\
  %accnext = insertvalue { i64, i64 } %t, i64 %bnew, 1\n\
  %inext = add i64 %i, 1\n\
  br label %loop\n\
exit:\n\
  %ra = extractvalue { i64, i64 } %acc, 0\n\
  %rb = extractvalue { i64, i64 } %acc, 1\n\
  %r = add i64 %ra, %rb\n\
  ret i64 %r\n\
}\n";
    let dir = std::env::temp_dir();
    let ll = dir.join(format!("svm_structphi_{}.ll", std::process::id()));
    let bc = dir.join(format!("svm_structphi_{}.bc", std::process::id()));
    std::fs::write(&ll, ir).expect("write IR");
    // Assemble the textual IR with clang (no extra tool dependency beyond the one tests already need).
    match Command::new("clang")
        .args(["-x", "ir", "-c", "-emit-llvm"])
        .arg(&ll)
        .arg("-o")
        .arg(&bc)
        .status()
    {
        Ok(s) if s.success() => {}
        _ => {
            eprintln!("note: skipping struct_phi_cross_block (clang unavailable)");
            return;
        }
    }
    let t = svm_llvm::translate_bc_path(&bc).expect("translate struct-φ IR");
    svm_verify::verify_module(&t.module).expect("verify");
    let run = t
        .exports
        .iter()
        .find(|(n, _)| n == "run")
        .map(|x| x.1)
        .expect("run export");
    for n in [5i64, 7] {
        let expect = 3 * n * (n - 1) / 2;
        let mut fuel = 10_000_000u64;
        let interp = match svm_interp::run(
            &t.module,
            run,
            &[Value::I64(t.entry_sp as i64), Value::I64(n)],
            &mut fuel,
        )
        .expect("interp")[0]
        {
            Value::I64(x) => x,
            other => panic!("unexpected {other:?}"),
        };
        let jit =
            match svm_jit::compile_and_run(&t.module, run, &[t.entry_sp as i64, n]).expect("jit") {
                JitOutcome::Returned(v) => v[0],
                o => panic!("jit outcome {o:?}"),
            };
        assert_eq!(interp, jit, "struct-φ n={n}: interp vs jit");
        assert_eq!(interp, expect, "struct-φ n={n}: result");
    }
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
fn switch_sparse_compare_chain() {
    // Far-apart `i64` cases (span ≫ MAX_SWITCH_SPAN) — clang keeps a real `switch` instruction, which
    // the on-ramp lowers to an **equality compare chain** of synthetic blocks (a dense `br_table`
    // would be astronomically large). Rust's niche-optimized enum discriminants produce exactly these.
    // Every case, the default, and several between-case misses must agree on interp and JIT.
    let src = "long sw(long x){ switch(x){ \
               case 0: return 11; \
               case 1000000L: return 22; \
               case 1000000000000L: return 33; \
               case -2000000000000L: return 44; \
               default: return 99; } }";
    check("ssw_0", src, &[Value::I64(0)], &[Value::I64(11)]);
    check("ssw_1m", src, &[Value::I64(1_000_000)], &[Value::I64(22)]);
    check(
        "ssw_1t",
        src,
        &[Value::I64(1_000_000_000_000)],
        &[Value::I64(33)],
    );
    check(
        "ssw_neg",
        src,
        &[Value::I64(-2_000_000_000_000)],
        &[Value::I64(44)],
    );
    check("ssw_def1", src, &[Value::I64(5)], &[Value::I64(99)]);
    check("ssw_def2", src, &[Value::I64(-1)], &[Value::I64(99)]);
    check(
        "ssw_def3",
        src,
        &[Value::I64(999_999_999_999)],
        &[Value::I64(99)],
    );
    check(
        "ssw_defmin",
        src,
        &[Value::I64(i64::MIN)],
        &[Value::I64(99)],
    );
    check(
        "ssw_defmax",
        src,
        &[Value::I64(i64::MAX)],
        &[Value::I64(99)],
    );
}

#[test]
fn switch_sparse_threads_live_ins_and_phi() {
    // The hard case for the compare chain: the case bodies read `a`/`b` (live-ins that must be threaded
    // through the synthetic chain blocks to the case targets), and the cases converge on a common
    // successor whose `r` is a φ (so the chain's targets carry a branch argument). Exercises the full
    // arg/live-in/φ threading, not just `(SP, operand)`.
    let src = "long sw2(long x, long a, long b){ long r; switch(x){ \
               case 0: r = a + 1; break; \
               case 1000000000000L: r = b * 2; break; \
               case -3000000000000L: r = a - b; break; \
               default: r = a + b; break; } return r * 3; }";
    // (x, a, b) with a=10, b=4
    check(
        "sw2_c0",
        src,
        &[Value::I64(0), Value::I64(10), Value::I64(4)],
        &[Value::I64(33)],
    ); // (10+1)*3
    check(
        "sw2_c1",
        src,
        &[Value::I64(1_000_000_000_000), Value::I64(10), Value::I64(4)],
        &[Value::I64(24)], // (4*2)*3
    );
    check(
        "sw2_c2",
        src,
        &[
            Value::I64(-3_000_000_000_000),
            Value::I64(10),
            Value::I64(4),
        ],
        &[Value::I64(18)], // (10-4)*3
    );
    check(
        "sw2_def",
        src,
        &[Value::I64(7), Value::I64(10), Value::I64(4)],
        &[Value::I64(42)], // (10+4)*3
    );
}

#[test]
fn switch_sparse_long_chain() {
    // A six-case sparse switch → a five-block compare chain: stresses the synthetic-block indexing
    // (each chain block branches to the next, the last to the default).
    let src = "long swN(long x){ switch(x){ \
               case 0: return 1; \
               case 100000L: return 2; \
               case 200000000L: return 3; \
               case 300000000000L: return 4; \
               case -400000000000L: return 5; \
               case -500000L: return 6; \
               default: return 0; } }";
    check("swN_a", src, &[Value::I64(0)], &[Value::I64(1)]);
    check("swN_b", src, &[Value::I64(100_000)], &[Value::I64(2)]);
    check("swN_c", src, &[Value::I64(200_000_000)], &[Value::I64(3)]);
    check(
        "swN_d",
        src,
        &[Value::I64(300_000_000_000)],
        &[Value::I64(4)],
    );
    check(
        "swN_e",
        src,
        &[Value::I64(-400_000_000_000)],
        &[Value::I64(5)],
    );
    check("swN_f", src, &[Value::I64(-500_000)], &[Value::I64(6)]);
    check("swN_def", src, &[Value::I64(12345)], &[Value::I64(0)]);
}

#[test]
fn switch_sparse_i32() {
    // The `i32` (width ≤ 32) compare-chain path: far-apart 32-bit cases. Confirms the chain compares
    // and materializes constants at `i32`, and a negative case round-trips.
    let src = "int sw32(int x){ switch(x){ \
               case 0: return 7; \
               case 100000: return 8; \
               case -200000: return 9; \
               case 50000000: return 10; \
               default: return -1; } }";
    check("sw32_0", src, &[Value::I32(0)], &[Value::I32(7)]);
    check("sw32_p", src, &[Value::I32(100_000)], &[Value::I32(8)]);
    check("sw32_n", src, &[Value::I32(-200_000)], &[Value::I32(9)]);
    check(
        "sw32_big",
        src,
        &[Value::I32(50_000_000)],
        &[Value::I32(10)],
    );
    check("sw32_def", src, &[Value::I32(1)], &[Value::I32(-1)]);
    check(
        "sw32_defmin",
        src,
        &[Value::I32(i32::MIN)],
        &[Value::I32(-1)],
    );
}

#[test]
fn memcmp_synthesized_helper() {
    // `memcmp(a, b, n)` with a **runtime** length (so clang emits a real `memcmp` call rather than
    // folding it) → the synthesized `__svm_memcmp` counted-loop helper. `a = [1..8]`, `b` equal except
    // `b[3] = x`, then the sign of `memcmp(a, b, n)` is returned. Covers equal, first-mismatch both
    // directions, a short prefix that stops before the mismatch, and `n == 0`. Differential interp+JIT.
    let src = "int f(int n, int x){ char a[8]; char b[8]; \
               for (int i=0;i<8;i++){ a[i]=(char)(i+1); b[i]=(char)(i+1); } \
               b[3]=(char)x; \
               int r = __builtin_memcmp(a, b, (unsigned long)(unsigned)n); \
               return (r>0)-(r<0); }";
    check(
        "mc_eq3",
        src,
        &[Value::I32(3), Value::I32(99)],
        &[Value::I32(0)],
    ); // first 3 match
    check(
        "mc_eq8",
        src,
        &[Value::I32(8), Value::I32(4)],
        &[Value::I32(0)],
    ); // b[3]==a[3]==4
    check(
        "mc_lt",
        src,
        &[Value::I32(8), Value::I32(99)],
        &[Value::I32(-1)],
    ); // a[3]=4 < 99
    check(
        "mc_gt",
        src,
        &[Value::I32(8), Value::I32(0)],
        &[Value::I32(1)],
    ); // a[3]=4 > 0
    check(
        "mc_short",
        src,
        &[Value::I32(4), Value::I32(99)],
        &[Value::I32(-1)],
    ); // mismatch in range
    check(
        "mc_zero",
        src,
        &[Value::I32(0), Value::I32(99)],
        &[Value::I32(0)],
    ); // n==0 → equal
}

#[test]
fn fcmp_unordered_ordered() {
    // The NaN-test float predicates `fcmp uno`/`ord` have no single svm-ir op; the on-ramp expands
    // them (`uno` = `a!=a | b!=b`, `ord` = `a==a & b==b`). `__builtin_isunordered` emits `uno`, its
    // negation emits `ord` (verified). Runtime args incl. NaN prevent folding. Differential interp+JIT.
    let src = "int f(int which, double a, double b){ switch(which){ \
               case 0: return __builtin_isunordered(a, b); \
               case 1: return !__builtin_isunordered(a, b); \
               default: return 0; } }";
    let nan = f64::NAN;
    // uno (which==0): 1 iff either operand is NaN.
    check(
        "uno_a",
        src,
        &[Value::I32(0), Value::F64(nan), Value::F64(1.0)],
        &[Value::I32(1)],
    );
    check(
        "uno_b",
        src,
        &[Value::I32(0), Value::F64(1.0), Value::F64(nan)],
        &[Value::I32(1)],
    );
    check(
        "uno_both",
        src,
        &[Value::I32(0), Value::F64(nan), Value::F64(nan)],
        &[Value::I32(1)],
    );
    check(
        "uno_none",
        src,
        &[Value::I32(0), Value::F64(1.0), Value::F64(2.0)],
        &[Value::I32(0)],
    );
    // ord (which==1): 1 iff neither operand is NaN.
    check(
        "ord_nan",
        src,
        &[Value::I32(1), Value::F64(nan), Value::F64(1.0)],
        &[Value::I32(0)],
    );
    check(
        "ord_none",
        src,
        &[Value::I32(1), Value::F64(1.0), Value::F64(2.0)],
        &[Value::I32(1)],
    );
}

#[test]
fn bitreverse_intrinsic() {
    // A bit-reversal loop `-O2` folds into `llvm.bitreverse.i32`; lowered inline via the log-N
    // swap network. Checked against native `cc`.
    let src = "unsigned br(unsigned x);\n\
               int run(int s){\n\
                 unsigned acc = 0;\n\
                 for (int i = 0; i < 6; i++) acc = acc*7 + br((unsigned)(s + i) * 2654435761u);\n\
                 return (int)(acc & 0x7fffffff);\n\
               }\n\
               unsigned br(unsigned x){\n\
                 unsigned r = 0; for (int i = 0; i < 32; i++){ r = (r << 1) | (x & 1u); x >>= 1; } return r;\n\
               }";
    check_vs_native("bitreverse", src, 5);
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
    check_demo_vs_native_flags(name, rel, stdin, &[]);
}

/// Like [`check_demo_vs_native`] but threads `extra` clang flags into the **bitcode** compile only
/// — the native `cc` oracle (in [`powerbox_diff`]) is unchanged. Used to pass
/// `-fno-vectorize -fno-slp-vectorize` on demos whose hot code clang would auto-SIMD-vectorize:
/// the vector lane (§17/D58) is outside the scalar on-ramp's scope, and exact integer code gives
/// the identical bytes scalar-vs-vectorized, so the on-ramp consumes scalar bitcode while the
/// oracle keeps vectorizing — the same split the Rust lane uses (`rust_*` helper, LLVM.md).
fn check_demo_vs_native_flags(name: &str, rel: &str, stdin: &[u8], extra: &[&str]) {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../svm-run/demos")
        .join(rel);
    let bc = std::env::temp_dir().join(format!("svm_llvm_demo_{}_{}.bc", std::process::id(), name));
    let status = Command::new("clang")
        .args(["-O2", "-emit-llvm", "-c"])
        .args(extra)
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
fn funnel_shift_general_const() {
    // A **non-rotate** funnel shift with a constant amount — clang's canonical form for a double-word
    // shift `(hi << k) | (lo >> (64 - k))` (the `fshl.i64(hi, lo, k)` Embench `aha-mont64`'s `modul64`
    // emits). Distinct operands, so this is the general path: `(a << s) | (b >>u (w - s))`. Three
    // amounts (1, 5, 63 — `(lo>>1)|(hi<<63)` canonicalizes to `fshl(.,.,63)`) and the full 64-bit result
    // is folded down into the exit byte, so an error in *any* bit flips it. Bit-exact vs native `cc`.
    let src = "int run(int seed) {\n\
        \x20 unsigned long hi = (unsigned long) seed * 0x9E3779B97F4A7C15UL + 1;\n\
        \x20 unsigned long lo = (unsigned long) seed * 0xC2B2AE3D27D4EB4FUL + 7;\n\
        \x20 unsigned long a = (hi << 1)  | (lo >> 63);\n\
        \x20 unsigned long b = (hi << 5)  | (lo >> 59);\n\
        \x20 unsigned long c = (lo >> 1)  | (hi << 63);\n\
        \x20 unsigned long r = a ^ (b * 3) ^ c;\n\
        \x20 r ^= r >> 32; r ^= r >> 16; r ^= r >> 8;\n\
        \x20 return (int)(r & 0xff);\n\
        }\n\
        int main(void) { return run(5); }\n";
    check_vs_native("funnel_general", src, 5);
}

#[test]
fn strlen_builtin() {
    // A direct `strlen` call (not via `printf %s`) routes to the synthesized `__svm_strlen` NUL-scan
    // helper — even in a `run`-only module with no `main` (the helper reads memory, needs no powerbox).
    // Two calls (a base pointer and a `buf + k` offset) over a runtime-length string; bit-exact vs the
    // native `cc` oracle on both backends. Found needed by Embench `slre`.
    let src = "#include <string.h>\n\
        int run(int seed) {\n\
        \x20 const char *s = \"the quick brown fox jumps over the lazy dog\";\n\
        \x20 const char *t = \"embench strlen test vector\";\n\
        \x20 unsigned long total = strlen(s + (seed % 5)) + strlen(t + (seed % 3));\n\
        \x20 return (int)(total & 0x7fffffff);\n\
        }\n\
        int main(void) { return run(7); }\n";
    check_vs_native("strlen_builtin", src, 7);
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

/// A **threaded (computed-`goto`) bytecode interpreter** — the canonical `indirectbr`/`blockaddress`
/// idiom (`static void *tbl[] = {&&l0,…}; goto *tbl[op];`), the dispatch shape every real bytecode VM
/// (SQLite's VDBE, Lua, QuickJS) is built on. clang `-O2` lowers `&&label` to `blockaddress` constants
/// in the dispatch-table global and `goto *p` to an `indirectbr`. The on-ramp recovers the (otherwise
/// `llvm-ir`-erased) blockaddress targets via `llvm-sys` ([`svm_llvm::blockaddr`]) — baking each as its
/// block index into the table global — and lowers the `indirectbr` to a `br_table` over those indices.
/// The program is **derived from `n` at runtime** so no dispatch target constant-folds (which would let
/// clang thread a `blockaddress` through a φ — an operand-position use, the deferred follow-up); every
/// blockaddress stays in the table global. Verified byte-for-byte vs native on both backends.
const COMPUTED_GOTO_SRC: &str = r#"
int run(int n) {
  static const void *const tbl[] = {&&op_halt, &&op_dbl, &&op_inc, &&op_xor};
  unsigned char prog[16];
  for (int i = 0; i < 15; i++) prog[i] = (unsigned char)(((n + i) * 2654435761u) % 4);
  prog[15] = 0; /* guaranteed halt */
  int pc = 0, acc = n, steps = 0;
  goto *tbl[prog[pc]];
op_dbl:  acc = acc * 2 + 1; pc++; if (++steps > 64) goto op_halt; goto *tbl[prog[pc]];
op_inc:  acc += 3;         pc++; if (++steps > 64) goto op_halt; goto *tbl[prog[pc]];
op_xor:  acc ^= 0x5a;      pc++; if (++steps > 64) goto op_halt; goto *tbl[prog[pc]];
op_halt: return acc & 0xff;
}
int main(void) { return run(7); }
"#;

#[test]
fn computed_goto_threaded_interpreter() {
    check_vs_native("computed_goto", COMPUTED_GOTO_SRC, 7);
}

/// Structural companion to [`computed_goto_threaded_interpreter`]: prove the computed-`goto` path is
/// actually exercised (not optimized away) — clang emitted `blockaddress`es into the dispatch global,
/// the `llvm-sys` recovery found them, and the `indirectbr` lowered to a `br_table`.
#[test]
fn computed_goto_lowers_indirectbr_to_br_table() {
    let Some(bc) = compile_to_bc("computed_goto_struct", COMPUTED_GOTO_SRC) else {
        return;
    };
    // The recovery found the dispatch table's blockaddress labels (one global, ≥ 2 entries).
    let ba = svm_llvm::blockaddr::read_block_addrs(bc.to_str().unwrap())
        .expect("blockaddress recovery should find the dispatch table");
    assert!(
        ba.per_global.values().any(|labels| labels.len() >= 2),
        "expected a dispatch-table global with multiple blockaddress labels, got {:?}",
        ba.per_global
    );
    // The `indirectbr` lowered to a `br_table` terminator.
    let t = svm_llvm::translate_bc_path(&bc).expect("translate bitcode");
    svm_verify::verify_module(&t.module).expect("verify");
    let has_br_table = t
        .module
        .funcs
        .iter()
        .flat_map(|f| f.blocks.iter())
        .any(|b| matches!(b.term, svm_ir::Terminator::BrTable { .. }));
    assert!(has_br_table, "indirectbr should lower to a br_table");
}

/// Computed `goto` where clang's `-O2` **jump-threading threads a `blockaddress` through a φ** (slice
/// AW) — an *operand-position* blockaddress, not a global-table entry. This is the shape a real
/// interpreter produces (and the AV follow-up): the program has a constant first dispatch
/// (`prog[0] == 2`), so clang knows the entry target and threads `blockaddress(@run, …)` into a φ that
/// feeds the `indirectbr`. The on-ramp recovers it via `llvm-sys` (`blockaddr::phi`, keyed by φ
/// position) and materializes the block-index constant. Byte-identical to native on both backends.
const COMPUTED_GOTO_PHI_SRC: &str = r#"
int run(int n) {
  static const void *const tbl[] = {&&op_halt, &&op_dbl, &&op_inc, &&op_loop};
  static const unsigned char prog[] = {2, 1, 2, 3, 0}; /* inc,dbl,inc,loop,halt — constant first op */
  int pc = 0, acc = n, iters = 0;
  goto *tbl[prog[pc]];
op_dbl:  acc *= 2; pc++; goto *tbl[prog[pc]];
op_inc:  acc += 1; pc++; goto *tbl[prog[pc]];
op_loop: if (++iters < 3) pc = 0; else pc++; goto *tbl[prog[pc]];
op_halt: return acc & 0xff;
}
int main(void) { return run(7); }
"#;

#[test]
fn computed_goto_phi_threaded_blockaddress() {
    check_vs_native("computed_goto_phi", COMPUTED_GOTO_PHI_SRC, 7);
}

/// Structural companion: confirm clang actually threaded a `blockaddress` through a φ (so the
/// operand-position recovery path — not just the global-table path — is exercised).
#[test]
fn computed_goto_phi_recovery_finds_operand_blockaddress() {
    let Some(bc) = compile_to_bc("computed_goto_phi_struct", COMPUTED_GOTO_PHI_SRC) else {
        return;
    };
    let ba = svm_llvm::blockaddr::read_block_addrs(bc.to_str().unwrap())
        .expect("recovery should find blockaddresses");
    assert!(
        !ba.phi.is_empty(),
        "expected a φ-threaded (operand-position) blockaddress, got phi map {:?}",
        ba.phi
    );
    // And it still translates + verifies (the operand-position label resolved, no fail-closed).
    let t = svm_llvm::translate_bc_path(&bc).expect("translate bitcode");
    svm_verify::verify_module(&t.module).expect("verify");
}

/// `<setjmp.h>` non-local jump (`setjmp`/`longjmp` → the `SetJmp`/`LongJmp` core ops). Compiles `run`
/// plus `int main(){return run(SEED);}` natively (real libc `setjmp`/`longjmp`) and on the on-ramp,
/// asserting **all three engines** — tree-walker, bytecode, and JIT — match the native exit code. The
/// JIT runs `setjmp`/`longjmp` via libc `_setjmp`/`_longjmp` inline from JITted code (LLVM.md §"JIT
/// `longjmp`"); on a target without that runtime it declines cleanly and the interpreters cover it.
/// `run` returns a byte so the result survives the Unix exit code.
fn check_setjmp_vs_native(name: &str, src: &str, seed: i32) {
    let Some(bc) = compile_to_bc(name, src) else {
        return;
    };
    let exe = std::env::temp_dir().join(format!("svm_llvm_sj_{}_{}", std::process::id(), name));
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
    let svm = match interp.first() {
        Some(Value::I32(x)) => *x as u8,
        other => panic!("{name}: expected i32 result, got {other:?}"),
    };
    assert_eq!(
        svm, native,
        "{name}: tree-walker={svm} vs native cc={native}"
    );

    // The **bytecode** engine implements setjmp/longjmp too (interpreter-grade) and must agree — it
    // runs the module (does not decline) and matches the tree-walker + native.
    let mut bfuel = 100_000_000u64;
    let bc_out = svm_interp::bytecode::compile_and_run(&module, 0, &full, &mut bfuel)
        .expect("bytecode engine should run setjmp/longjmp (not decline)")
        .expect("bytecode run");
    let bsvm = match bc_out.first() {
        Some(Value::I32(x)) => *x as u8,
        other => panic!("{name}: bytecode expected i32 result, got {other:?}"),
    };
    assert_eq!(
        bsvm, native,
        "{name}: bytecode={bsvm} vs native cc={native}"
    );

    // The JIT runs setjmp/longjmp natively (libc `_setjmp`/`_longjmp` inline from JITted code, with a
    // host-side `jmp_buf` table — LLVM.md §"JIT `longjmp`", Option B) on the targets where its runtime
    // exists (`setjmp_rt` = unix among `fiber_rt`). It must agree with the interpreters + native. On a
    // target without the runtime it declines cleanly (the interpreters above already proved
    // correctness); either way it must never miscompile.
    let slots: Vec<i64> = full.iter().map(to_slot).collect();
    match svm_jit::compile_and_run(&module, 0, &slots) {
        Ok(JitOutcome::Returned(s)) => {
            let jsvm = s[0] as i32 as u8;
            assert_eq!(jsvm, native, "{name}: JIT={jsvm} vs native cc={native}");
        }
        Ok(other) => panic!("{name}: unexpected JIT outcome {other:?} on a valid setjmp program"),
        Err(svm_jit::JitError::Unsupported(_)) => {
            eprintln!(
                "note: {name} JIT declined setjmp/longjmp (no native-stack runtime on this target)"
            );
        }
        Err(e) => panic!("{name}: JIT errored on setjmp/longjmp: {e:?}"),
    }
}

#[test]
fn setjmp_longjmp_round_trip() {
    // `setjmp` returns 0 on the direct call; `deep` `longjmp`s back with `n*7+1`, so `setjmp` "returns
    // twice" and `run` yields that value. The longjmp unwinds across `deep`'s frame to the `setjmp`
    // frame, restoring its data-SP and value state. Byte-identical to native libc.
    let src = "#include <setjmp.h>\n\
               static jmp_buf env;\n\
               static void deep(int x){ longjmp(env, x*7+1); }\n\
               int run(int n){ int r = setjmp(env); if (r==0){ deep(n); return -1; } return r & 0xff; }\n\
               int main(void){ return run(5); }";
    check_setjmp_vs_native("setjmp_basic", src, 5);
}

#[test]
fn setjmp_longjmp_loop_and_deep_nesting() {
    // A retry loop: each `longjmp` (from several frames deep) re-enters the `setjmp`, incrementing a
    // counter carried in memory (a `volatile`/`static`, which survives the jump per C), until it
    // reaches the limit — exercising repeated re-entry (the checkpoint is overwritten on each
    // `setjmp`) and a multi-frame unwind. Byte-identical to native.
    let src = "#include <setjmp.h>\n\
               static jmp_buf env;\n\
               static int counter;\n\
               static void c(int d, int n){ if (d > 0) { c(d-1, n); return; } longjmp(env, n); }\n\
               int run(int n){ counter = 0; int r = setjmp(env); \
                 if (r != 0) counter += r; \
                 if (counter < n) c(3, counter + 1); \
                 return counter & 0xff; }\n\
               int main(void){ return run(20); }";
    check_setjmp_vs_native("setjmp_loop", src, 20);
}

#[test]
fn setjmp_value_live_across() {
    // The returns-twice hazard (LLVM.md §"JIT `longjmp`"): a `volatile` automatic is live across the
    // `setjmp` — modified before the `longjmp` and read after the re-entry. Per C a `volatile` auto is
    // preserved across `longjmp`; clang spills it to the stack at `-O2`, so it rides in guest window
    // memory and survives the native `_longjmp`. Result = (100 + n + 7) & 0xff. Byte-identical to native.
    let src = "#include <setjmp.h>\n\
               static jmp_buf env;\n\
               static void boom(int x){ longjmp(env, x); }\n\
               int run(int n){ volatile int acc = 100; int r = setjmp(env); \
                 if (r == 0){ acc += n; boom(7); return -1; } \
                 return (acc + r) & 0xff; }\n\
               int main(void){ return run(20); }";
    check_setjmp_vs_native("setjmp_value_live", src, 20);
}

#[test]
fn setjmp_nested_buffers() {
    // Two distinct `jmp_buf`s (two host table slots): an inner `setjmp` then a `longjmp` to the
    // **outer** buffer, skipping the inner — exercises keying by buffer address and a longjmp that
    // crosses a frame holding a *different* live checkpoint. `setjmp(outer)` re-enters with 9 → 42.
    let src = "#include <setjmp.h>\n\
               static jmp_buf outer, inner;\n\
               static void deep(void){ longjmp(outer, 9); }\n\
               int run(int n){ if (setjmp(outer) != 0) return (40 + n) & 0xff; \
                 if (setjmp(inner) == 0){ deep(); return -1; } return -2; }\n\
               int main(void){ return run(2); }";
    check_setjmp_vs_native("setjmp_nested_bufs", src, 2);
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
fn demo_monocypher_vs_native() {
    // Monocypher 4.0.2 (public domain): modern crypto — BLAKE2b, ChaCha20, Poly1305, and an X25519
    // ECDH known-answer test. The crypto / 64-bit-carry shakedown: AEAD bit-mixing plus the curve's
    // 25.5-bit-limb field arithmetic (`i32 × i32 → i64` products with carry propagation) stress the
    // 64-bit shift/rotate/multiply paths hard. Outputs are hex (no float formatting); the X25519
    // section also self-validates (both ECDH sides must agree, exit code = mismatch count).
    // Byte-identical to native `clang`. Compiled with auto-vectorization off for the on-ramp
    // (the crypto hot loops clang would SIMD-vectorize are the §17 vector lane, not the scalar
    // arithmetic this demo targets); the native oracle keeps vectorizing — exact integer crypto
    // agrees scalar-vs-vectorized. Mirrors the Rust lane.
    check_demo_vs_native_flags(
        "monocypher",
        "monocypher/monocypher_demo.c",
        b"",
        &["-fno-vectorize", "-fno-slp-vectorize"],
    );
}

#[test]
fn demo_stb_image_vs_native() {
    // Sean Barrett's stb_image (public domain), PNG-only: decode an embedded 24×24 RGBA PNG and
    // write the raw decoded pixels. A real-parser shakedown — stb's built-in zlib inflate
    // (Huffman + LZ77), the PNG row unfilters (None/Sub/Up/Average/Paeth — the test image cycles
    // all five, hitting the narrow `unsigned char` predictor arithmetic, the slice-U class), the
    // chunk/CRC walk, and heap traffic through the on-ramp's synthesized malloc/realloc/free. The
    // native build decodes the same bytes → byte-exact oracle. Vectorization off for the on-ramp
    // (the inflate/unfilter loops clang would SIMD-vectorize are the §17 lane); exact integer
    // decoding agrees scalar-vs-vectorized.
    check_demo_vs_native_flags(
        "stb_image",
        "stb_image/stb_image_demo.c",
        b"",
        &["-fno-vectorize", "-fno-slp-vectorize"],
    );
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
    // i128 *division* is outside the subset (I14 tier 3 lowers i128 add/sub/mul/shift/bitwise, but not
    // div/rem — `udiv i128` / the `__udivti3` libcall) — it must be a clean `Unsupported`, never a
    // silent mis-translation (LLVM.md §2/§8, the fail-closed chokepoint).
    let Some(bc) = compile_to_bc(
        "i128div",
        "unsigned __int128 dv(unsigned __int128 a, unsigned __int128 b){ return a / b; }",
    ) else {
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
        run.stdout,
        b"0\n",
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
    compile_cpp_to_bc_flags(name, src, &["-fno-exceptions", "-fno-rtti"])
}

/// Like [`compile_cpp_to_bc`] but with caller-chosen flags — used by the EH tests, which keep
/// exceptions on (drop `-fno-exceptions`) so `invoke`/`landingpad`/`__cxa_*` reach the on-ramp.
fn compile_cpp_to_bc_flags(name: &str, src: &str, extra: &[&str]) -> Option<PathBuf> {
    let dir = std::env::temp_dir();
    let cc = dir.join(format!("svm_llvm_{}_{}.cpp", std::process::id(), name));
    let bc = dir.join(format!("svm_llvm_cpp_{}_{}.bc", std::process::id(), name));
    std::fs::write(&cc, src).expect("write C++ source");
    let status = Command::new("clang++")
        .args(["-O2", "-emit-llvm", "-c"])
        .args(extra)
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
    check_cpp_bc_vs_native(name, &bc, stdin);
}

/// The C++ **exception-handling** differential: same as [`check_cpp_vs_native`] but compiled with
/// exceptions *on* (only `-fno-rtti`), so `invoke`/`landingpad`/`resume` + the `__cxa_*` runtime
/// reach the on-ramp. The native oracle links the default (exceptions-on) `clang++`.
fn check_cpp_eh_vs_native(name: &str, src: &str, stdin: &[u8]) {
    let Some(bc) = compile_cpp_to_bc_flags(name, src, &["-fno-rtti"]) else {
        return;
    };
    check_cpp_bc_vs_native(name, &bc, stdin);
}

fn check_cpp_bc_vs_native(name: &str, bc: &std::path::Path, stdin: &[u8]) {
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

    let t = svm_llvm::translate_bc_path(bc).expect("translate C++ bitcode");
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

/// **C++ exceptions first light** — `throw`/`try`/`catch` through the on-ramp, built on the
/// setjmp/longjmp stack-transfer core. A `noinline` thrower raises an `int` (caught by `catch(int)`)
/// or a `const char*` (caught by `catch(...)`); the caller `invoke`s it and routes via the
/// landingpad selector. Exercises `invoke`/`landingpad`/`resume`, `llvm.eh.typeid.for`, and the
/// synthesized `__cxa_allocate_exception`/`__cxa_throw`/`__cxa_begin_catch`/`__cxa_end_catch`
/// runtime — byte-identical to native `clang++` on all three engines.
#[test]
fn cpp_eh_throw_catch_int() {
    let src = r#"
extern "C" int write(int fd, const char *buf, long n);

static void puti(int v) {
  char b[16]; int i = 16;
  int neg = v < 0; unsigned u = neg ? -(unsigned)v : (unsigned)v;
  b[--i] = '\n';
  do { b[--i] = '0' + u % 10; u /= 10; } while (u);
  if (neg) b[--i] = '-';
  write(1, b + i, 16 - i);
}

__attribute__((noinline)) static int thrower(int x) {
  if (x < 0) throw x;
  if (x == 5) throw "five";
  return x * 2;
}

static int classify(int x) {
  try { return thrower(x); }
  catch (int e) { return e - 1000; }
  catch (...)   { return 999; }
}

int main() {
  for (int i = -2; i <= 6; i++) puti(classify(i));
  return 0;
}
"#;
    check_cpp_eh_vs_native("cpp_eh_throw_catch_int", src, b"");
}

/// **C++ exceptions — rethrow + cleanup unwind** — the paths the [`cpp_eh_throw_catch_int`] demo
/// does *not* reach. `middle` catches an `int` and re-raises it with a bare `throw;`
/// (`__cxa_rethrow`), which clang wraps in an `invoke` to a **cleanup-only landingpad** (calling
/// `__cxa_end_catch` then `_Unwind_Resume` on the way out) — exercising `resume`/`_Unwind_Resume`
/// and the preservation of the in-flight `cur_exn`/`cur_sel` across the rethrow. `relabel` instead
/// throws a *fresh* exception from inside its catch (the first is fully caught, so the shared
/// single-slot object/selector model is allowed to be overwritten). Both reach an outer handler;
/// byte-identical to native `clang++` on all three engines pins the model.
#[test]
fn cpp_eh_rethrow_nested() {
    let src = r#"
extern "C" int write(int fd, const char *buf, long n);

static void puti(int v) {
  char b[16]; int i = 16;
  int neg = v < 0; unsigned u = neg ? -(unsigned)v : (unsigned)v;
  b[--i] = '\n';
  do { b[--i] = '0' + u % 10; u /= 10; } while (u);
  if (neg) b[--i] = '-';
  write(1, b + i, 16 - i);
}

__attribute__((noinline)) static int inner(int x) {
  if (x < 0) throw x;            // raise an int
  return x;
}

// Catch the int and re-raise the *same* exception with a bare `throw;`. clang lowers the rethrow
// as an invoke whose unwind edge is a cleanup-only landingpad (__cxa_end_catch + _Unwind_Resume).
__attribute__((noinline)) static int middle(int x) {
  try { return inner(x); }
  catch (int) { throw; }
}

// Catch the int, then throw a *fresh* exception from inside the handler (clobbers the single slot,
// which is fine: the original is fully caught before the new one is raised).
__attribute__((noinline)) static int relabel(int x) {
  try { return inner(x); }
  catch (int e) { throw e * 100; }
}

static int outer(int x) {
  try { return middle(x); }
  catch (int e) { return e - 7; }   // catches the rethrown int
}

static int outer2(int x) {
  try { return relabel(x); }
  catch (int e) { return e + 1; }   // catches the fresh int
}

int main() {
  for (int i = -2; i <= 2; i++) puti(outer(i));
  for (int i = -2; i <= 2; i++) puti(outer2(i));
  return 0;
}
"#;
    check_cpp_eh_vs_native("cpp_eh_rethrow_nested", src, b"");
}

/// **C++ exceptions — multiple catch clauses** — a single `try` with several typed handlers plus a
/// catch-all: `catch (A&) / catch (B&) / catch (int) / catch (...)`. A `noinline` thrower raises one
/// of four things by argument; the landingpad's selector is matched against each clause's
/// `llvm.eh.typeid.for` in turn, falling through to `catch (...)`. Exercises the multi-clause
/// selector-dispatch chain (the demo only had one typed clause) and class-typed exception objects
/// caught by reference. Byte-identical to native `clang++` across all three engines.
#[test]
fn cpp_eh_multi_catch() {
    let src = r#"
extern "C" int write(int fd, const char *buf, long n);

static void puti(int v) {
  char b[16]; int i = 16;
  int neg = v < 0; unsigned u = neg ? -(unsigned)v : (unsigned)v;
  b[--i] = '\n';
  do { b[--i] = '0' + u % 10; u /= 10; } while (u);
  if (neg) b[--i] = '-';
  write(1, b + i, 16 - i);
}

struct A { int tag; };
struct B { int tag; };

__attribute__((noinline)) static int thrower(int x) {
  if (x == 0) throw A{11};
  if (x == 1) throw B{22};
  if (x == 2) throw 7;
  if (x == 3) throw "str";   // const char* — only the catch-all matches
  return x;
}

static int classify(int x) {
  try { return thrower(x); }
  catch (A &a)  { return 1000 + a.tag; }
  catch (B &b)  { return 2000 + b.tag; }
  catch (int e) { return 3000 + e; }
  catch (...)   { return 9999; }
}

int main() {
  for (int i = 0; i <= 4; i++) puti(classify(i));
  return 0;
}
"#;
    check_cpp_eh_vs_native("cpp_eh_multi_catch", src, b"");
}

/// **C++ exceptions — propagation through a cleanup frame** — a throw unwinds through an intermediate
/// function that installs no handler but owns a local with a non-trivial destructor (`Guard`). clang
/// gives that frame a **cleanup-only** landingpad (run the destructor, then `_Unwind_Resume` to keep
/// unwinding); the exception is caught two frames up. Pins that cleanup landingpads fire during
/// propagation and that the handler-stack depth is restored correctly across the resumed unwind —
/// the `log` accumulator observes each `Guard` destructor exactly once. Byte-identical to native.
#[test]
fn cpp_eh_unwind_cleanup() {
    let src = r#"
extern "C" int write(int fd, const char *buf, long n);

static void puti(int v) {
  char b[16]; int i = 16;
  int neg = v < 0; unsigned u = neg ? -(unsigned)v : (unsigned)v;
  b[--i] = '\n';
  do { b[--i] = '0' + u % 10; u /= 10; } while (u);
  if (neg) b[--i] = '-';
  write(1, b + i, 16 - i);
}

struct Guard {
  int *log; int id;
  Guard(int *l, int i) : log(l), id(i) {}
  ~Guard() { *log += id; }    // observable on unwind
};

__attribute__((noinline)) static int inner(int x) {
  if (x < 0) throw x;
  return x;
}

// No catch here: the throw propagates, but `g`'s destructor must run on the way out
// (a cleanup-only landingpad → _Unwind_Resume).
__attribute__((noinline)) static int middle(int x, int *log) {
  Guard g(log, 10);
  return inner(x);
}

// A second cleanup frame stacked on top, to pin nested cleanups during one unwind.
__attribute__((noinline)) static int middle2(int x, int *log) {
  Guard g(log, 3);
  return middle(x, log);
}

static int outer(int x, int *log) {
  try { return middle2(x, log); }
  catch (int e) { return e; }
}

int main() {
  int log = 0;
  int r = outer(-5, &log);   // throws; unwinds through middle (+10) and middle2 (+3); caught
  puti(r);                   // -5
  puti(log);                 // 13
  int log2 = 0;
  int ok = outer(4, &log2);  // no throw: no destructors-on-unwind, normal return runs them once
  puti(ok);                  // 4
  puti(log2);                // 13 (both Guards destroyed on normal scope exit)
  return 0;
}
"#;
    check_cpp_eh_vs_native("cpp_eh_unwind_cleanup", src, b"");
}

/// **C++ exceptions — catch-by-value + destructor** — the path that forces `__cxa_end_catch` to stop
/// being a no-op. The thrown type `E` has an observable copy-constructor (`+1`) and destructor
/// (`+100`). `catch (E e)` copy-constructs a local from the exception object (`+1`), and on handler
/// exit *two* destructors run: the local copy's (clang-emitted, `+100`) and the exception object's
/// own (`__cxa_end_catch` → the synthesized `__svm_eh_destroy`, which runs the destructor funcref
/// `__cxa_throw` registered, `+100`). `catch (E &e)` binds by reference (no copy, no local), so only
/// the exception-object destructor runs (`+100`). Byte-identical to native `clang++` pins that the
/// registered destructor fires exactly once per object across all three engines.
#[test]
fn cpp_eh_catch_by_value_dtor() {
    let src = r#"
extern "C" int write(int fd, const char *buf, long n);

static void puti(int v) {
  char b[16]; int i = 16;
  int neg = v < 0; unsigned u = neg ? -(unsigned)v : (unsigned)v;
  b[--i] = '\n';
  do { b[--i] = '0' + u % 10; u /= 10; } while (u);
  if (neg) b[--i] = '-';
  write(1, b + i, 16 - i);
}

struct E {
  int *log; int v;
  E(int *l, int val) : log(l), v(val) {}
  E(const E &o) : log(o.log), v(o.v) { *log += 1; }   // copy-ctor: observable
  ~E() { *log += 100; }                                // dtor: observable
};

__attribute__((noinline)) static void thrower(int *log, int val) {
  throw E(log, val);
}

static int by_value(int *log, int val) {
  try { thrower(log, val); }
  catch (E e) { return e.v; }   // copy in, local dtor + exception-object dtor (end_catch) out
  return -1;
}

static int by_ref(int *log, int val) {
  try { thrower(log, val); }
  catch (E &e) { return e.v; }  // no copy; only the exception-object dtor (end_catch) runs
  return -1;
}

int main() {
  int a = 0;
  int r1 = by_value(&a, 42);
  puti(r1);   // 42
  puti(a);    // copy(+1) + local-dtor(+100) + end_catch-dtor(+100) = 201

  int b = 0;
  int r2 = by_ref(&b, 7);
  puti(r2);   // 7
  puti(b);    // end_catch-dtor(+100) = 100
  return 0;
}
"#;
    check_cpp_eh_vs_native("cpp_eh_catch_by_value_dtor", src, b"");
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

// ============================================================================================
// Milestone 1 (PEVAL.md) — `core + alloc` through the Rust on-ramp. The existing Rust lane proves
// `core` (a pure compute fn). This proves the next layer: a heap-allocating `no_std` program whose
// `#[global_allocator]` is backed by the guest `malloc`/`free` (the same `vm_map`-growing bump
// allocator the C/C++ heap tests use). `Vec`/`Box` from `alloc` lower to `__rust_alloc` →
// (our `#[global_allocator]`) → `extern "C" malloc`, so the on-ramp synthesizes the allocator and
// the program grows its own heap. This is the prerequisite for running `svm-peval` (all
// `Vec`/`BTreeMap`) as an svm-IR guest. Differential: stdout matches the *same* program built as a
// native `std` Rust binary.
// ============================================================================================

/// Build a native `std` Rust binary from `src`, run it (feeding `stdin`), return its stdout. `None`
/// (skip) if `rustc +1.81.0` is unavailable — the on-ramp lane is pinned to that toolchain, so the
/// oracle uses it too (no behavioural difference for these programs, but keeps one toolchain).
fn rust_native_stdout(name: &str, src: &str, stdin: &[u8]) -> Option<Vec<u8>> {
    use std::io::Write;
    let dir = std::env::temp_dir();
    let rs = dir.join(format!("svm_llvm_{}_{}_nat.rs", std::process::id(), name));
    let exe = dir.join(format!("svm_llvm_{}_{}_nat", std::process::id(), name));
    std::fs::write(&rs, src).expect("write native Rust source");
    match Command::new("rustc")
        .args(["+1.81.0", "-C", "opt-level=2"])
        .arg(&rs)
        .arg("-o")
        .arg(&exe)
        .status()
    {
        Ok(s) if s.success() => {}
        _ => return None,
    }
    let mut child = Command::new(&exe)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .expect("spawn native rust");
    child.stdin.take().unwrap().write_all(stdin).ok();
    Some(child.wait_with_output().expect("run native rust").stdout)
}

/// Translate a `no_std`/`alloc` powerbox Rust program through the on-ramp and run it, returning its
/// stdout. Mirrors [`powerbox_diff`]'s SVM half but for a Rust frontend: the program must produce a
/// powerbox entry (it uses `malloc` + `write`), so we resolve §7 imports to capabilities, verify, and
/// run through `run_powerbox`. `None` (skip) if `rustc +1.81.0` is unavailable.
fn rust_powerbox_stdout(name: &str, src: &str, stdin: &[u8]) -> Option<Vec<u8>> {
    let bc = compile_rust_to_bc(name, src)?;
    let t = svm_llvm::translate_bc_path(&bc).expect("translate Rust heap bitcode");
    assert!(
        svm_run::is_powerbox_entry(&t.module),
        "{name}: a heap-allocating Rust program must produce a powerbox entry"
    );
    let module = svm_run::resolve_capability_imports(t.module).expect("resolve capability imports");
    svm_verify::verify_module(&module).expect("verify translated Rust IR");
    Some(
        svm_run::run_powerbox(&module, stdin)
            .expect("powerbox run")
            .stdout,
    )
}

/// **`core + alloc` through the Rust on-ramp (PEVAL.md Milestone 1).** A `no_std` Rust program with a
/// `#[global_allocator]` over the guest `malloc`/`free` builds a `Vec` that grows past its initial
/// capacity (many `RawVec` reallocs → `malloc`/`free` churn → `vm_map` heap growth), boxes a value,
/// and prints a heap-derived sum. The whole `alloc` stack (`RawVec`, the global-allocator shims
/// `__rust_alloc`/`__rust_dealloc`/`__rust_realloc`, `Box`) lowers through the on-ramp with no change
/// beyond the C heap path, and the output is byte-identical to the same program as a native `std`
/// binary. This is the layer `svm-peval` needs (it is all `Vec`/maps).
#[test]
fn rust_core_alloc_heap_matches_native() {
    // The on-ramp guest: `no_std` + `alloc`, allocator backed by the guest `malloc`/`free`. Builds a
    // growing `Vec<u64>` of squares (forces reallocs), a `Box`, sums on the heap, prints the decimal.
    let onramp = r#"
#![no_std]
#![no_main]
extern crate alloc;
use alloc::boxed::Box;
use alloc::vec::Vec;
use core::alloc::{GlobalAlloc, Layout};

extern "C" {
    fn malloc(n: usize) -> *mut u8;
    fn free(p: *mut u8);
    fn write(fd: i32, buf: *const u8, n: isize) -> isize;
}

struct Guest;
unsafe impl GlobalAlloc for Guest {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 { malloc(l.size()) }
    unsafe fn dealloc(&self, p: *mut u8, _l: Layout) { free(p) }
}
#[global_allocator]
static A: Guest = Guest;

#[panic_handler]
fn ph(_: &core::panic::PanicInfo) -> ! { loop {} }

fn putdec(mut x: u64) {
    let mut buf = [0u8; 24];
    let mut i = 24usize;
    if x == 0 { i -= 1; buf[i] = b'0'; }
    while x > 0 { i -= 1; buf[i] = b'0' + (x % 10) as u8; x /= 10; }
    unsafe { write(1, buf.as_ptr().add(i), (24 - i) as isize); }
    unsafe { write(1, b"\n".as_ptr(), 1); }
}

#[no_mangle]
pub extern "C" fn main() -> i32 {
    let mut v: Vec<u64> = Vec::new();
    for i in 0..1000u64 { v.push(i * i); }   // grows -> realloc -> malloc/free churn
    let mut sum: u64 = 0;
    for &x in &v { sum = sum.wrapping_add(x); }
    let boxed = Box::new(sum.wrapping_mul(2));
    putdec(*boxed);
    0
}
"#;
    // The native `std` oracle: the *same* computation, printed with `println!`.
    let native = r#"
fn main() {
    let mut v: Vec<u64> = Vec::new();
    for i in 0..1000u64 { v.push(i * i); }
    let mut sum: u64 = 0;
    for &x in &v { sum = sum.wrapping_add(x); }
    let boxed = Box::new(sum.wrapping_mul(2));
    println!("{}", *boxed);
}
"#;
    let (Some(svm), Some(nat)) = (
        rust_powerbox_stdout("rs_heap", onramp, b""),
        rust_native_stdout("rs_heap", native, b""),
    ) else {
        return; // toolchain unavailable — skip
    };
    assert_eq!(
        svm,
        nat,
        "heap Rust: on-ramp stdout {:?} vs native {:?}",
        String::from_utf8_lossy(&svm),
        String::from_utf8_lossy(&nat)
    );
}

/// **`BTreeMap` through the on-ramp (`core::slice::index` panic shims).** A `no_std` program builds a
/// `BTreeMap<u64,u64>` past one node's capacity (so it splits — exercising node slicing / element
/// shifts), then sums `k + v` over an ordered iteration, byte-identical to the native `std` build.
/// `BTreeMap`'s node code is littered with slice range accesses whose bounds-panic helpers
/// (`slice_{start,end}_index_len_fail`) are external `-> !` lang items; the on-ramp shims the whole
/// `slice…_fail` family to `trap` (`is_rust_abort_call`). Before that, this hit
/// `Unsupported("call to external/undefined function …slice_end_index_len_fail…")`. This is the
/// collection `svm-peval` uses for its memo tables.
#[test]
fn rust_btreemap_matches_native() {
    let logic = "let mut mp: alloc::collections::BTreeMap<u64,u64> = alloc::collections::BTreeMap::new();\n\
        for i in 0..40u64 { mp.insert(i.wrapping_mul(7) % 101, i.wrapping_mul(3)); }\n\
        let mut s: u64 = 0; for (k, v) in &mp { s = s.wrapping_add(*k).wrapping_add(*v); }\n\
        putdec(s); putdec(mp.len() as u64);\n\
        putdec(*mp.get(&0).unwrap_or(&999));";
    let onramp = format!(
        "#![no_std]\n#![no_main]\nextern crate alloc;\n\
         use core::alloc::{{GlobalAlloc, Layout}};\n\
         extern \"C\" {{ fn malloc(n: usize)->*mut u8; fn free(p:*mut u8); fn write(fd:i32,buf:*const u8,n:isize)->isize; }}\n\
         struct G; unsafe impl GlobalAlloc for G {{ unsafe fn alloc(&self,l:Layout)->*mut u8{{malloc(l.size())}} unsafe fn dealloc(&self,p:*mut u8,_:Layout){{free(p)}} }}\n\
         #[global_allocator] static A: G = G;\n\
         #[panic_handler] fn ph(_:&core::panic::PanicInfo)->!{{loop{{}}}}\n\
         fn putdec(x:u64){{let mut m=x;let mut b=[0u8;24];let mut i=24usize;if m==0{{i-=1;b[i]=b'0';}}while m>0{{i-=1;b[i]=b'0'+(m%10)as u8;m/=10;}}unsafe{{write(1,b.as_ptr().add(i),(24-i)as isize);write(1,b\"\\n\".as_ptr(),1);}}}}\n\
         #[no_mangle] pub extern \"C\" fn main()->i32{{ {logic} 0 }}\n"
    );
    let native = format!(
        "use std::collections::BTreeMap as _Unused; mod alloc {{ pub use std::collections; }}\n\
         fn putdec(x:u64){{ println!(\"{{}}\", x); }}\n\
         fn main(){{ {logic} }}\n"
    );
    let (Some(svm), Some(nat)) = (
        rust_powerbox_stdout("rs_btree", &onramp, b""),
        rust_native_stdout("rs_btree", &native, b""),
    ) else {
        return; // toolchain unavailable — skip
    };
    assert_eq!(
        svm,
        nat,
        "BTreeMap on-ramp stdout {:?} vs native {:?}",
        String::from_utf8_lossy(&svm),
        String::from_utf8_lossy(&nat)
    );
}

/// **ZST struct field → element-stride/offset layout (PEVAL.md Milestone 3 corruption).** A `no_std`
/// program builds a `Vec<Inner>` where `Inner { data: Vec<u64>, tag: u64 }` *contains a `Vec`*, then
/// indexes the outer vector (`v[i].tag`, `v[i].data.len()`, `v[i].data[0]`) and sums. A `Vec`/`RawVec`
/// carries the zero-sized `alloc::alloc::Global` allocator marker (`type {}`), so the element stride of
/// `Vec<Inner>` and the offset of `Inner.tag`/`Inner.data.len()` both depend on the on-ramp sizing an
/// **empty struct as 0 bytes**. A previous `struct_layout` clamped it to 1, inflating every `RawVec` by
/// a byte (24-byte `Vec`s padded to 32, `len` shifted 16→24) and desyncing every field offset from
/// LLVM's GEPs — so an indexed `v[i].data.len()` read garbage (it returned the *outer* `Vec::len()`).
/// This is precisely the bug that made the in-sandbox `svm-peval` (all nested `Vec`/`BTreeMap`) read a
/// corrupted length and trap; a flat `Vec<u64>` (existing tests) never exercised a ZST-bearing element.
/// Differential: byte-identical to the same program as a native `std` binary.
#[test]
fn rust_zst_struct_field_layout_matches_native() {
    let logic = "\
        struct Inner { data: alloc::vec::Vec<u64>, tag: u64 }\n\
        let mut outer: alloc::vec::Vec<Inner> = alloc::vec::Vec::new();\n\
        for i in 0..6u64 {\n\
            let mut d: alloc::vec::Vec<u64> = alloc::vec::Vec::new();\n\
            d.push(i.wrapping_mul(10)); d.push(i.wrapping_mul(10).wrapping_add(1));\n\
            outer.push(Inner { data: d, tag: i.wrapping_mul(100) });\n\
        }\n\
        let mut s: u64 = 0;\n\
        for i in 0..outer.len() {\n\
            s = s.wrapping_add(outer[i].tag);\n\
            s = s.wrapping_add(outer[i].data.len() as u64);\n\
            s = s.wrapping_add(outer[i].data[0]);\n\
        }\n\
        putdec(s); putdec(outer.len() as u64);";
    let onramp = format!(
        "#![no_std]\n#![no_main]\nextern crate alloc;\n\
         use core::alloc::{{GlobalAlloc, Layout}};\n\
         extern \"C\" {{ fn malloc(n: usize)->*mut u8; fn free(p:*mut u8); fn write(fd:i32,buf:*const u8,n:isize)->isize; }}\n\
         struct G; unsafe impl GlobalAlloc for G {{ unsafe fn alloc(&self,l:Layout)->*mut u8{{malloc(l.size())}} unsafe fn dealloc(&self,p:*mut u8,_:Layout){{free(p)}} }}\n\
         #[global_allocator] static A: G = G;\n\
         #[panic_handler] fn ph(_:&core::panic::PanicInfo)->!{{loop{{}}}}\n\
         fn putdec(x:u64){{let mut m=x;let mut b=[0u8;24];let mut i=24usize;if m==0{{i-=1;b[i]=b'0';}}while m>0{{i-=1;b[i]=b'0'+(m%10)as u8;m/=10;}}unsafe{{write(1,b.as_ptr().add(i),(24-i)as isize);write(1,b\"\\n\".as_ptr(),1);}}}}\n\
         #[no_mangle] pub extern \"C\" fn main()->i32{{ {logic} 0 }}\n"
    );
    let native = format!(
        "mod alloc {{ pub use std::vec; }}\n\
         fn putdec(x:u64){{ println!(\"{{}}\", x); }}\n\
         fn main(){{ {logic} }}\n"
    );
    let (Some(svm), Some(nat)) = (
        rust_powerbox_stdout("rs_zst", &onramp, b""),
        rust_native_stdout("rs_zst", &native, b""),
    ) else {
        return; // toolchain unavailable — skip
    };
    assert_eq!(
        svm,
        nat,
        "ZST-struct layout on-ramp stdout {:?} vs native {:?}",
        String::from_utf8_lossy(&svm),
        String::from_utf8_lossy(&nat)
    );
}

/// The statements both the on-ramp `no_std` program and the native `std` oracle run — each prints one
/// value with `putdec`. Exercises the saturating-arithmetic intrinsics (`llvm.{u,s}{add,sub}.sat`, on
/// i32/i64, both the clamped and the non-clamped path) and the saturating float→int casts
/// (`llvm.fpto{si,ui}.sat`, f32/f64 → i32/i64, incl. ±overflow and NaN). Each side defines `putdec`
/// differently (manual `write` vs `println!`) but the call sequence is identical, so the stdout must
/// match byte-for-byte.
const RUST_SAT_BODY: &str = "
    putdec((10u64).saturating_sub(25) as i64);   // usub.sat -> 0
    putdec((100u64).saturating_sub(40) as i64);  // usub.sat -> 60
    putdec(u64::MAX.saturating_add(5) as i64);   // uadd.sat -> u64::MAX (-1 as i64)
    putdec((7u64).saturating_add(8) as i64);     // uadd.sat -> 15 (no clamp)
    putdec(i64::MIN.saturating_sub(1));          // ssub.sat -> i64::MIN
    putdec(i64::MAX.saturating_add(1));          // sadd.sat -> i64::MAX
    putdec((5i64).saturating_add(3));            // sadd.sat -> 8 (no clamp)
    putdec((-5i64).saturating_sub(3));           // ssub.sat -> -8 (no clamp)
    putdec((7u32).saturating_sub(100) as i64);   // usub.sat.i32 -> 0
    putdec(u32::MAX.saturating_add(2) as i64);   // uadd.sat.i32 -> u32::MAX
    putdec(i32::MIN.saturating_sub(1) as i64);   // ssub.sat.i32 -> i32::MIN
    putdec(i32::MAX.saturating_add(1) as i64);   // sadd.sat.i32 -> i32::MAX
    putdec((1e30f64) as i64);                    // fptosi.sat.i64.f64 -> i64::MAX
    putdec((-1e30f64) as i64);                   // -> i64::MIN
    putdec((f64::NAN) as i64);                   // -> 0
    putdec((3.99f64) as i64);                    // -> 3
    putdec((1e30f64) as u64 as i64);             // fptoui.sat.i64.f64 -> u64::MAX (-1)
    putdec((-7.0f64) as u64 as i64);             // -> 0 (clamped low)
    putdec((1e20f32) as i32 as i64);             // fptosi.sat.i32.f32 -> i32::MAX
    putdec((-1e20f32) as i32 as i64);            // -> i32::MIN
    putdec((1e20f32) as u32 as i64);             // fptoui.sat.i32.f32 -> u32::MAX
    putdec((2.5f32) as i32 as i64);              // -> 2
";

/// **Saturating arithmetic + saturating float→int casts through the on-ramp.** These are the LLVM
/// intrinsics Rust emits for `saturating_add`/`saturating_sub` and float `as` integer casts — the gaps
/// the specializer hit. The on-ramp lowers them inline (clamp via `select`; `FToISat`), and every
/// result is byte-identical to the same program built natively, across the clamp/no-clamp and
/// over/underflow/NaN cases on both `i32` and `i64`.
#[test]
fn rust_saturating_and_fp_sat_casts_match_native() {
    let onramp = format!(
        "#![no_std]\n#![no_main]\n\
         extern \"C\" {{ fn write(fd: i32, buf: *const u8, n: isize) -> isize; }}\n\
         #[panic_handler] fn ph(_: &core::panic::PanicInfo) -> ! {{ loop {{}} }}\n\
         fn putdec(x: i64) {{\n\
             let neg = x < 0;\n\
             let mut mag: u64 = if neg {{ (x as u64).wrapping_neg() }} else {{ x as u64 }};\n\
             let mut buf = [0u8; 24];\n\
             let mut i = 24usize;\n\
             if mag == 0 {{ i -= 1; buf[i] = b'0'; }}\n\
             while mag > 0 {{ i -= 1; buf[i] = b'0' + (mag % 10) as u8; mag /= 10; }}\n\
             if neg {{ i -= 1; buf[i] = b'-'; }}\n\
             unsafe {{ write(1, buf.as_ptr().add(i), (24 - i) as isize); }}\n\
             unsafe {{ write(1, b\"\\n\".as_ptr(), 1); }}\n\
         }}\n\
         #[no_mangle] pub extern \"C\" fn main() -> i32 {{ {RUST_SAT_BODY} 0 }}\n"
    );
    let native = format!(
        "fn putdec(x: i64) {{ println!(\"{{}}\", x); }}\nfn main() {{ {RUST_SAT_BODY} }}\n"
    );
    let (Some(svm), Some(nat)) = (
        rust_powerbox_stdout("rs_sat", &onramp, b""),
        rust_native_stdout("rs_sat", &native, b""),
    ) else {
        return; // toolchain unavailable — skip
    };
    assert_eq!(
        svm,
        nat,
        "saturating/fp-sat Rust: on-ramp stdout {:?} vs native {:?}",
        String::from_utf8_lossy(&svm),
        String::from_utf8_lossy(&nat)
    );
}

#[test]
fn bitint56_load_store_roundtrips() {
    // A non-power-of-two integer (`_BitInt(56)` = `i56`) round-trips through memory: the on-ramp
    // legalizes `load i56` (read the enclosing i64, mask to 56 bits), `store i56` (byte-exact, so it
    // never clobbers an adjacent field), and the `i56 → i64` zero/sign-extend (in i64). A `volatile`
    // local forces the real store+load. Differential on interp + JIT (`check`).
    // Unsigned: store, load (mask), add — no sign extension.
    let u = "unsigned long f(void){ volatile unsigned _BitInt(56) g = 0x1234567890ABULL; \
             return (unsigned long)g + 1; }";
    check("u56", u, &[], &[Value::I64(20_015_998_341_292)]);
    // Signed negative: store, load, then sign-extend the 56-bit value to i64.
    let s = "long f(void){ volatile _BitInt(56) g = -100; return (long)g; }";
    check("s56", s, &[], &[Value::I64(-100)]);
    // Signed positive (top niche bit clear): sign-extend must keep it positive.
    let p = "long f(void){ volatile _BitInt(56) g = 0x1234567890AB; return (long)g; }";
    check("p56", p, &[], &[Value::I64(20_015_998_341_291)]);
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

/// `-O2 -mavx2` auto-vectorized **wide lane shifts** (`shl`/`lshr`/`ashr` on a `<8 x i32>` by a
/// constant splat) — ISSUES.md I11. The legalization pass splits the `<8 x i32>` shift into two
/// `v128` `VShift` chunks (the `wide_int_shift` arm in `lower_wide`); before that the on-ramp
/// fail-closed on `lshr <8 x i32>` even though the v128 case worked. Mixes a logical and an arithmetic shift so
/// both `ShrU`/`Shl` and `ShrS` are exercised. Byte-identical to the native scalar oracle.
#[test]
fn simd_autovec_avx2_wide_shifts() {
    let src = "void sh(const int *a, int *c, int n);\n\
        static int A[64], C[64];\n\
        int run(int seed) {\n\
        \x20 for (int i = 0; i < 64; i++) A[i] = (seed + i * 7) * 1103515245 + 12345;\n\
        \x20 sh(A, C, 64);\n\
        \x20 int s = 0;\n\
        \x20 for (int i = 0; i < 64; i++) s += C[i];\n\
        \x20 return s & 0xff;\n\
        }\n\
        __attribute__((noinline)) void sh(const int *a, int *c, int n) {\n\
        \x20 for (int i = 0; i < n; i++) c[i] = ((unsigned) a[i] >> 5) ^ (a[i] << 3) ^ (a[i] >> 2);\n\
        }\n\
        int main(void) { return run(9); }\n";
    check_avx_vs_native("simd_avx2_wide_shifts", src, 9);
}

/// ISSUES.md I13 (root fix, isolated) — `<2 x i32>` lane arithmetic carried as a packed `i64` must be
/// lane-wise for **every** cross-lane-unsafe op, not just `mul`: `add`/`sub` carry across the 32-bit lane
/// boundary and `shl`/`lshr`/`ashr` shift bits between lanes. This uses an explicit `vector_size(8)`
/// `<2 x i32>` so clang emits the ops directly (`add`, `mul`, `shl`), with large lane values chosen so a
/// packed-`i64` op would visibly corrupt the high lane. Bit-exact vs the native `cc` oracle on both
/// backends (and interp == JIT).
#[test]
fn simd_vec2_i32_lane_arith_add_shift_i13() {
    let src = "typedef int v2 __attribute__((vector_size(8)));\n\
        int run(int seed) {\n\
        \x20 v2 a = {seed * 7 + 100000, seed + 30000};\n\
        \x20 v2 b = {seed + 3, seed * 5 + 70000};\n\
        \x20 v2 c = (a + b) * b;\n\
        \x20 v2 d = c << 2;\n\
        \x20 v2 e = d - a;\n\
        \x20 return e[0] + e[1];\n\
        }\n\
        int main(void) { return run(4); }\n";
    check_vs_native("i13_vec2_addshift", src, 4);
}

/// ISSUES.md I13 (root fix) — Embench `edn`'s `fir_no_red_ld` ("no-redundant-load" FIR) carries a
/// `<2 x i16>` across the loop and auto-vectorizes the deinterleaved widening multiply to **`<2 x i32>`
/// lane arithmetic**. A 2-lane 32-bit vector is held as a *packed* `i64` (lane 0 low, lane 1 high), and
/// the lane `mul` was lowered as a single `i64` multiply on that packed image — which cross-contaminates
/// the lanes (the low product's carry and the lane0×lane1 cross term corrupt lane 1). That was a silent
/// miscompile (previously fail-closed by a φ guard). The fix lowers `<2 x i32>` integer arithmetic
/// lane-wise. This pins the kernel **bit-exact (full 64-bit checksum)** vs the native `cc` oracle on
/// both backends — and forces the `mul` lowering to be per-lane `i32`, never a packed `i64.mul`.
#[test]
fn simd_vec2_i32_carried_widening_mul_i13() {
    // `run(long n)` runs the FIR `n` times over a seeded buffer and folds a weighted 64-bit checksum —
    // wide enough that a corrupted high lane changes the result well beyond a low-byte coincidence.
    let kernel = "void fir_no_red_ld(const short x[], const short h[], long y[]);\n\
        void fir_no_red_ld(const short x[], const short h[], long y[]) {\n\
        \x20 long i, j, sum0, sum1; short x0, x1, h0, h1;\n\
        \x20 for (j = 0; j < 100; j += 2) { sum0 = 0; sum1 = 0; x0 = x[j];\n\
        \x20   for (i = 0; i < 32; i += 2) {\n\
        \x20     x1 = x[j+i+1]; h0 = h[i]; sum0 += x0*h0; sum1 += x1*h0;\n\
        \x20     x0 = x[j+i+2]; h1 = h[i+1]; sum0 += x1*h1; sum1 += x0*h1; }\n\
        \x20   y[j] = sum0 >> 15; y[j+1] = sum1 >> 15; }\n\
        }\n\
        long run(long n) {\n\
        \x20 long acc = 0;\n\
        \x20 for (long t = 0; t < n; t++) {\n\
        \x20   short X[132], H[32]; long Y[100];\n\
        \x20   for (int i=0;i<132;i++) X[i]=(short)((i*7+t*3+1)%97 - 48);\n\
        \x20   for (int i=0;i<32;i++) H[i]=(short)((i*5+t+1)%31 - 15);\n\
        \x20   for (int i=0;i<100;i++) Y[i]=0;\n\
        \x20   fir_no_red_ld(X,H,Y);\n\
        \x20   for (int i=0;i<100;i++) acc += Y[i]*(i+1);\n\
        \x20 }\n\
        \x20 return acc;\n\
        }\n";
    let main = "long run(long);\n#include <stdio.h>\n\
        int main(void){ printf(\"%ld %ld\\n\", run(1), run(7)); return 0; }\n";
    let Some(bc) = compile_to_bc("i13_vec2_fir", kernel) else {
        return; // clang unavailable
    };
    // Native oracle: full 64-bit results for n=1 and n=7 via stdout.
    let dir = std::env::temp_dir();
    let csrc = dir.join(format!("svm_llvm_i13_{}.c", std::process::id()));
    let exe = dir.join(format!("svm_llvm_i13_{}", std::process::id()));
    std::fs::write(&csrc, format!("{kernel}{main}")).expect("write C");
    match Command::new("cc").arg(&csrc).arg("-o").arg(&exe).status() {
        Ok(s) if s.success() => {}
        _ => {
            eprintln!("note: skipping i13 (cc unavailable)");
            return;
        }
    }
    let out = Command::new(&exe).output().expect("run native");
    let s = String::from_utf8_lossy(&out.stdout);
    let nat: Vec<i64> = s
        .split_whitespace()
        .map(|w| w.parse().expect("native i64"))
        .collect();
    assert_eq!(nat.len(), 2, "native printed two checksums");

    let t =
        svm_llvm::translate_bc_path(&bc).expect("translate (I13 root fix: no longer fail-closed)");
    let module = &t.module;
    svm_verify::verify_module(module).expect("verify");
    let e = t
        .exports
        .iter()
        .find(|(n, _)| n == "run")
        .map(|x| x.1)
        .expect("run export");
    for (k, &expect) in [1i64, 7].iter().zip(&nat) {
        let mut fuel = 200_000_000u64;
        let interp = svm_interp::run(
            module,
            e,
            &[Value::I64(t.entry_sp as i64), Value::I64(*k)],
            &mut fuel,
        )
        .expect("interp")[0];
        let jit = match svm_jit::compile_and_run(module, e, &[t.entry_sp as i64, *k]).expect("jit")
        {
            JitOutcome::Returned(v) => v[0],
            o => panic!("jit outcome {o:?}"),
        };
        let interp = match interp {
            Value::I64(x) => x,
            other => panic!("unexpected {other:?}"),
        };
        assert_eq!(interp, jit, "I13 n={k}: interp vs jit");
        assert_eq!(interp, expect, "I13 n={k}: svm vs native cc");
    }
}

/// `-O2 -mavx2` auto-vectorized **fixed-point DSP** kernel (Embench `edn`'s `vec_mpy` shape):
/// `y[i] += (short)((scaler * x[i]) >> 15)`. The `short` widening multiply produces a wide `<8 x i32>`
/// intermediate that is then **shifted** (`>>15`) and truncated back to `<8 x i16>` — the I11 shape the
/// wide legalizer rejected before it dispatched shifts through the chunk path. Verified against the
/// native scalar oracle on both backends.
#[test]
fn simd_autovec_avx2_fixed_point_shift() {
    let src = "void vmpy(short *y, const short *x, short scaler, int n);\n\
        static short Y[64], X[64];\n\
        int run(int seed) {\n\
        \x20 for (int i = 0; i < 64; i++) { X[i] = (short)((seed + i) * 200); Y[i] = (short)(i * 3); }\n\
        \x20 vmpy(Y, X, (short)(seed * 300 + 100), 64);\n\
        \x20 int s = 0;\n\
        \x20 for (int i = 0; i < 64; i++) s += Y[i];\n\
        \x20 return s;\n\
        }\n\
        __attribute__((noinline)) void vmpy(short *y, const short *x, short scaler, int n) {\n\
        \x20 for (int i = 0; i < n; i++) y[i] += (short)((scaler * x[i]) >> 15);\n\
        }\n\
        int main(void) { return run(4); }\n";
    check_avx_vs_native("simd_avx2_fixshift", src, 4);
}

// ============================================================================================
// SIMD — the **other 128-bit lane shapes** (`i8x16`/`i16x8`/`i64x2`/`f64x2`), beyond the original
// `i32x4`/`f32x4`. These use explicit `vector_size(16)` types compiled with vectorization *off*
// (`compile_to_bc` / `check_vs_native`), so the on-ramp sees exactly the declared 128-bit shape —
// no auto-vectorizer widening. A `noinline` helper takes opaque pointers so clang must emit real
// `<N x T>` loads/ops, not scalarize them. Each `vec128_shape` op (load/store, `VIntBin`,
// `VFloatBin`, `ExtractLane`) is exercised against the native oracle on both backends.
// ============================================================================================

/// `<2 x ptr>` — an SLP-vectorized pointer-pair copy (`load <2 x ptr>` → `store`, e.g. Embench
/// `sglib-combined`'s linked-list/struct shuffles). A pointer lane is an `i64` window offset, so the
/// on-ramp packs `<2 x ptr>` exactly like `<2 x i64>` (an `i64x2` v128) and the 16-byte move is a
/// `V128Load`/`V128Store`. Compares pointer **identity** (portable: absolute addresses differ between
/// native and svm, but "the copy preserved both pointers" does not).
#[test]
fn simd_ptr2_copy() {
    let src = "struct N { int *a; int *b; };\n\
        void cp(struct N *d, struct N *s);\n\
        int run(int seed){\n\
        \x20 int arr[4]; struct N s, d;\n\
        \x20 s.a = &arr[seed & 3]; s.b = &arr[(seed + 1) & 3];\n\
        \x20 cp(&d, &s);\n\
        \x20 return (d.a == s.a && d.b == s.b) ? 7 : 0;\n\
        }\n\
        __attribute__((noinline)) void cp(struct N *d, struct N *s){ d->a = s->a; d->b = s->b; }\n\
        int main(void){ return run(2); }\n";
    check_vs_native("simd_ptr2_copy", src, 2);
}

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

/// `<16 x i8>` lane multiply through a `v128` load → mul → store (`i8x16` `VIntBin` Mul). x86 has no
/// per-byte multiply, so the JIT emulates it (widen → `i16x8` multiply → low-byte pack); this pins
/// that lowering against the native oracle on both backends. Wraps mod 2^8 per lane.
#[test]
fn simd_i8x16_mul_load_store() {
    let src = "void vmul(const unsigned char *P, const unsigned char *Q, unsigned char *O);\n\
        static unsigned char D[16], F[16], E[16];\n\
        int run(int seed) {\n\
        \x20 for (int i = 0; i < 16; i++) { D[i] = (unsigned char)(seed + i); F[i] = (unsigned char)(i + 1); }\n\
        \x20 vmul(D, F, E);\n\
        \x20 int s = 0;\n\
        \x20 for (int i = 0; i < 16; i++) s += E[i];\n\
        \x20 return s & 0xff;\n\
        }\n\
        typedef unsigned char u8x16 __attribute__((vector_size(16)));\n\
        __attribute__((noinline)) void vmul(const unsigned char *P, const unsigned char *Q, unsigned char *O) {\n\
        \x20 u8x16 a = *(const u8x16 *)P;\n\
        \x20 u8x16 b = *(const u8x16 *)Q;\n\
        \x20 *(u8x16 *)O = a * b;\n\
        }\n\
        int main(void) { return run(2); }\n";
    check_vs_native("simd_i8x16_mul", src, 2);
}

/// `<4 x i32>` lane shifts by a **constant splat** (`shl`/`lshr`/`ashr` → `VShift`). The on-ramp
/// ingests a vector shift whose amount is a constant-splat vector (the shape `clang -O2` emits for
/// `v >> k`); a signed lane covers the arithmetic (`ashr`) path. Read-back folds the lanes.
#[test]
fn simd_i32x4_const_shifts() {
    let src = "void vsh(const int *P, int *O);\n\
        static int D[4], E[4];\n\
        int run(int seed) {\n\
        \x20 for (int i = 0; i < 4; i++) D[i] = (seed << 8) - i * 12345;\n\
        \x20 vsh(D, E);\n\
        \x20 int s = 0;\n\
        \x20 for (int i = 0; i < 4; i++) s += E[i];\n\
        \x20 return s & 0xff;\n\
        }\n\
        typedef int i32x4 __attribute__((vector_size(16)));\n\
        typedef unsigned u32x4 __attribute__((vector_size(16)));\n\
        __attribute__((noinline)) void vsh(const int *P, int *O) {\n\
        \x20 i32x4 a = *(const i32x4 *)P;\n\
        \x20 i32x4 sl = a << 3;\n\
        \x20 i32x4 sa = a >> 2;\n\
        \x20 u32x4 su = (u32x4)a >> 5u;\n\
        \x20 *(i32x4 *)O = sl + sa + (i32x4)su;\n\
        }\n\
        int main(void) { return run(7); }\n";
    check_vs_native("simd_i32x4_shift", src, 7);
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

/// I14 tier 2 — the i128 **widening-multiply** idiom (`(unsigned __int128)a * b >> 64`, a 64×64→128
/// `mulhi`) now translates and runs on both engines, matching a `u128` oracle. This is the
/// `aha-mont64 mulul64` core that previously fail-closed with `Unsupported("i128")`.
#[test]
fn i128_widening_mul_hi() {
    let src = "unsigned long mulhi(unsigned long a, unsigned long b) {\n\
        \x20 return (unsigned long)(((unsigned __int128)a * b) >> 64);\n\
        }\n";
    for (a, b) in [
        (0x1234_5678_9abc_def0u64, 0xfedc_ba98_7654_3210u64),
        (u64::MAX, u64::MAX),
        (3, 5),
        (1u64 << 63, 6),
        (0xdead_beef_0000_0001, 0x0000_0001_cafe_babe),
    ] {
        let hi = (((a as u128) * (b as u128)) >> 64) as u64;
        check(
            "i128_mulhi",
            src,
            &[Value::I64(a as i64), Value::I64(b as i64)],
            &[Value::I64(hi as i64)],
        );
    }
}

/// The full `mulul64` shape: one `unsigned __int128` product feeding **both** a low-half `trunc` and a
/// high-half `lshr 64`+`trunc` — the same symbolic product consumed twice. The halves are combined
/// through shifts (which don't distribute over `trunc`, so clang keeps them as the two separate i64
/// truncs the real `mulul64` emits, rather than folding into an `xor i128`).
#[test]
fn i128_widening_mul_lo_and_hi() {
    let src = "unsigned long mont(unsigned long a, unsigned long b) {\n\
        \x20 unsigned __int128 p = (unsigned __int128)a * b;\n\
        \x20 unsigned long lo = (unsigned long)p;\n\
        \x20 unsigned long hi = (unsigned long)(p >> 64);\n\
        \x20 return (hi ^ (lo >> 17)) + (lo ^ (hi >> 13));\n\
        }\n";
    for (a, b) in [
        (0x1234_5678_9abc_def0u64, 0xfedc_ba98_7654_3210u64),
        (u64::MAX, 7),
        (0x8000_0000_0000_0000, 0x8000_0000_0000_0000),
    ] {
        let p = (a as u128) * (b as u128);
        let (lo, hi) = (p as u64, (p >> 64) as u64);
        let want = (hi ^ (lo >> 17)).wrapping_add(lo ^ (hi >> 13));
        check(
            "i128_mont",
            src,
            &[Value::I64(a as i64), Value::I64(b as i64)],
            &[Value::I64(want as i64)],
        );
    }
}

// ---- I14 tier 3: general i128 arithmetic (every i128 a materialized (lo, hi) pair) ----------------

/// i128 `add`/`sub` with full carry/borrow across the word boundary. Builds two 128-bit values from
/// four u64 halves, then folds both words of `x+y` and `x-y` into one i64 vs a `u128` oracle.
#[test]
fn i128_add_sub_carry() {
    let src = "unsigned long f(unsigned long ah, unsigned long al, unsigned long bh, unsigned long bl) {\n\
        \x20 unsigned __int128 x = ((unsigned __int128)ah << 64) | al;\n\
        \x20 unsigned __int128 y = ((unsigned __int128)bh << 64) | bl;\n\
        \x20 unsigned __int128 s = x + y;\n\
        \x20 unsigned __int128 d = x - y;\n\
        \x20 return ((unsigned long)s ^ (unsigned long)(s >> 64)) + ((unsigned long)d ^ (unsigned long)(d >> 64));\n\
        }\n";
    for (ah, al, bh, bl) in [
        (1u64, 2u64, 3u64, 4u64),
        (0, u64::MAX, 0, 1), // carry out of the low word
        (5, 0, 9, u64::MAX), // borrow into the high word
        (0xdead_beef, 0xffff_ffff_ffff_ffff, 0xcafe, 0x1),
    ] {
        let x = ((ah as u128) << 64) | al as u128;
        let y = ((bh as u128) << 64) | bl as u128;
        let s = x.wrapping_add(y);
        let d = x.wrapping_sub(y);
        let want = ((s as u64) ^ ((s >> 64) as u64)).wrapping_add((d as u64) ^ ((d >> 64) as u64));
        check(
            "i128_addsub",
            src,
            &[
                Value::I64(ah as i64),
                Value::I64(al as i64),
                Value::I64(bh as i64),
                Value::I64(bl as i64),
            ],
            &[Value::I64(want as i64)],
        );
    }
}

/// Full 128×128→128 `mul` (the schoolbook expansion, both operands genuinely 128-bit) + `and`/`or`/
/// `xor`, vs a `u128` oracle.
#[test]
fn i128_mul_and_bitwise() {
    let src = "unsigned long f(unsigned long ah, unsigned long al, unsigned long bh, unsigned long bl) {\n\
        \x20 unsigned __int128 x = ((unsigned __int128)ah << 64) | al;\n\
        \x20 unsigned __int128 y = ((unsigned __int128)bh << 64) | bl;\n\
        \x20 unsigned __int128 p = x * y;\n\
        \x20 unsigned __int128 m = (x & y) ^ (x | y);\n\
        \x20 return ((unsigned long)p ^ (unsigned long)(p >> 64)) + ((unsigned long)m ^ (unsigned long)(m >> 64));\n\
        }\n";
    for (ah, al, bh, bl) in [
        (0, 3, 0, 5),
        (1, 2, 3, 4),
        (
            0xffff_ffff,
            0xffff_ffff_ffff_ffff,
            0x1234,
            0x5678_9abc_def0_1234,
        ),
        (u64::MAX, u64::MAX, u64::MAX, u64::MAX),
    ] {
        let x = ((ah as u128) << 64) | al as u128;
        let y = ((bh as u128) << 64) | bl as u128;
        let p = x.wrapping_mul(y);
        let m = (x & y) ^ (x | y);
        let want = ((p as u64) ^ ((p >> 64) as u64)).wrapping_add((m as u64) ^ ((m >> 64) as u64));
        check(
            "i128_mul",
            src,
            &[
                Value::I64(ah as i64),
                Value::I64(al as i64),
                Value::I64(bh as i64),
                Value::I64(bl as i64),
            ],
            &[Value::I64(want as i64)],
        );
    }
}

/// Double-word **variable** shifts: `shl` / logical `>>` / arithmetic `>>` by a runtime amount across
/// the full `[0, 128)` range (including 0, `<64`, `==64`, `>64`) — the cross-word carry + `n>=64` word
/// move that `aha-mont64`'s `modul64` needs. Vs a `u128`/`i128` oracle.
#[test]
fn i128_variable_shifts() {
    let src = "unsigned long f(unsigned long h, unsigned long l, unsigned long s) {\n\
        \x20 unsigned __int128 x = ((unsigned __int128)h << 64) | l;\n\
        \x20 unsigned n = (unsigned)s & 127;\n\
        \x20 unsigned __int128 a = x << n;\n\
        \x20 unsigned __int128 b = x >> n;\n\
        \x20 __int128 c = ((__int128)x) >> n;\n\
        \x20 unsigned __int128 r = a ^ b ^ (unsigned __int128)c;\n\
        \x20 return (unsigned long)r ^ (unsigned long)(r >> 64);\n\
        }\n";
    let h = 0xfedc_ba98_7654_3210u64;
    let l = 0x0123_4567_89ab_cdefu64;
    for s in [0u64, 1, 17, 63, 64, 65, 100, 127] {
        let x = ((h as u128) << 64) | l as u128;
        let n = s as u32 & 127;
        let a = x << n;
        let b = x >> n;
        let c = ((x as i128) >> n) as u128;
        let r = a ^ b ^ c;
        let want = (r as u64) ^ ((r >> 64) as u64);
        check(
            "i128_shifts",
            src,
            &[
                Value::I64(h as i64),
                Value::I64(l as i64),
                Value::I64(s as i64),
            ],
            &[Value::I64(want as i64)],
        );
    }
}

/// i128 **parameter and return** (clang's `{i64,i64}` ABI split): `__int128 a + 1` reconstructs the
/// value from its two i64 halves and re-splits the result. Confirms the param/return path + the
/// carry across the word boundary compute correctly (not just translate).
#[test]
fn i128_param_and_return() {
    let src = "__int128 big(__int128 a){ return a + 1; }\n";
    for (lo, hi) in [
        (0u64, 0u64),
        (u64::MAX, 0x1234),
        (41, 0),
        (u64::MAX, u64::MAX),
    ] {
        let a = ((hi as u128) << 64) | lo as u128;
        let r = a.wrapping_add(1);
        check(
            "i128_big",
            src,
            &[Value::I64(lo as i64), Value::I64(hi as i64)],
            &[
                Value::I64(r as u64 as i64),
                Value::I64((r >> 64) as u64 as i64),
            ],
        );
    }
}

/// i128 comparisons across **all predicates** (signed + unsigned ordering, eq/ne), each compared to a
/// native `i128`/`u128` oracle. Packs the ten results into an int.
#[test]
fn i128_compares_all_predicates() {
    let src = "int cmp(unsigned long ah, unsigned long al, unsigned long bh, unsigned long bl) {\n\
        \x20 unsigned __int128 x = ((unsigned __int128)ah << 64) | al;\n\
        \x20 unsigned __int128 y = ((unsigned __int128)bh << 64) | bl;\n\
        \x20 __int128 sx = (__int128)x, sy = (__int128)y;\n\
        \x20 int r = 0;\n\
        \x20 r |= (x <  y) << 0; r |= (x <= y) << 1; r |= (x >  y) << 2; r |= (x >= y) << 3;\n\
        \x20 r |= (sx <  sy) << 4; r |= (sx <= sy) << 5; r |= (sx >  sy) << 6; r |= (sx >= sy) << 7;\n\
        \x20 r |= (x == y) << 8; r |= (x != y) << 9;\n\
        \x20 return r;\n\
        }\n";
    let cases = [
        (1u64, 2u64, 1u64, 2u64),         // equal
        (0, 1, 0, 2),                     // low differs
        (1, 0, 2, 0),                     // high differs
        (u64::MAX, 5, 0, 9),              // x huge (neg as signed), y small positive
        (0x8000_0000_0000_0000, 0, 0, 1), // signedness boundary in the high word
    ];
    for (ah, al, bh, bl) in cases {
        let x = ((ah as u128) << 64) | al as u128;
        let y = ((bh as u128) << 64) | bl as u128;
        let (sx, sy) = (x as i128, y as i128);
        let mut r = 0i32;
        r |= (x < y) as i32;
        r |= ((x <= y) as i32) << 1;
        r |= ((x > y) as i32) << 2;
        r |= ((x >= y) as i32) << 3;
        r |= ((sx < sy) as i32) << 4;
        r |= ((sx <= sy) as i32) << 5;
        r |= ((sx > sy) as i32) << 6;
        r |= ((sx >= sy) as i32) << 7;
        r |= ((x == y) as i32) << 8;
        r |= ((x != y) as i32) << 9;
        check(
            "i128_cmp",
            src,
            &[
                Value::I64(ah as i64),
                Value::I64(al as i64),
                Value::I64(bh as i64),
                Value::I64(bl as i64),
            ],
            &[Value::I32(r)],
        );
    }
}
